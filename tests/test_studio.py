import base64,json,subprocess,sys,time,zipfile
from pathlib import Path
import httpx,pytest
from akshara_forge.studio import generation as g
from akshara_forge.studio.evidence import snapshot


def test_evidence_never_invents_missing_scores(tmp_path):
    d=snapshot(tmp_path)
    assert d['math']['before'] is None
    assert d['coding']['after'] is None
    assert d['engineering']['before'] is None


def test_generated_bundle_and_runtime(tmp_path,monkeypatch):
    monkeypatch.setenv('AKSHARA_GENERATOR_PROVIDER','openai');monkeypatch.setenv('OPENAI_API_KEY','test')
    monkeypatch.delenv('AKSHARA_INFERENCE_CONFIG',raising=False)
    counter=[0]
    def provider(prompt,folder,*,system=g.SYSTEM):
        if system!=g.SYSTEM:
            request=json.loads(prompt)
            return {'problems':[{'id':r['id'],'unambiguous':True,'reason':'Add one','source_quote':'Addition','reference_answer':int(r['prompt'].split()[1])+1,'attacks':[{'answer':a,'expected_accept':False,'reason':'Wrong value or type'} for a in [-1,-2,'wrong']]} for r in request['problems']]}
        rows=[]
        for _ in range(10):
            counter[0]+=1
            rows.append({'prompt':f'Compute {counter[0]} + 1','source_quote':'Addition','solution_outline':'Add one.','reference_answer':counter[0]+1,'verification':'numeric','tolerance':0})
        return {'title':'Addition','problems':rows}
    monkeypatch.setattr(g,'call_model',provider)
    jobs=g.GenerationJobs(tmp_path);s=jobs.start({'filename':'spec.txt','data':base64.b64encode(b'Addition').decode(),'count':20})
    for _ in range(100):
        s=jobs.read(s['id'])
        if s['status'] in ('ready','failed'):break
        time.sleep(.01)
    assert s['status']=='ready',s
    p=tmp_path/s['id']/'environment'
    rows=json.loads((p/'private/problems.json').read_text());assert sum(r['split']=='heldout' for r in rows)==4
    public=json.loads((p/'tasks.json').read_text());assert 'reference_answer' not in public[0]
    assert (p/'learner/tasks.json').exists()
    audit=json.loads((p/'private/adversarial-audit.json').read_text());assert all(a['passed'] for a in audit)
    assert s['manifest']['adversarial_cases']>=80
    result=subprocess.run([sys.executable,str(p/'environment.py')],input='{"op":"reset","id":"problem-001"}\n{"op":"step","answer":2}\n{"op":"step","answer":2}\n',text=True,capture_output=True,check=True)
    result=[json.loads(l) for l in result.stdout.splitlines()]
    assert result[1]=={'reward':1,'done':True};assert 'error' in result[2]
    assert zipfile.is_zipfile(tmp_path/s['id']/'environment.zip')


def test_bad_source_evidence_rejected():
    with pytest.raises(ValueError,match='quotation'):
        g.validate({'problems':[{'prompt':'Q','source_quote':'invented','solution_outline':'S','verification':'numeric','reference_answer':2}]},'actual source',1)


def test_vultr_identity_no_silent_fallback(tmp_path,monkeypatch):
    monkeypatch.delenv('AKSHARA_INFERENCE_CONFIG',raising=False)
    monkeypatch.setenv('AKSHARA_GENERATOR_PROVIDER','vultr');monkeypatch.setenv('AKSHARA_GENERATOR_MODEL','chosen');monkeypatch.setenv('VULTR_INFERENCE_API_KEY','test')
    def transport(req):
        if req.method=='GET':return httpx.Response(200,json={'data':[{'id':'chosen','hugging_face_id':'org/chosen'}]})
        return httpx.Response(200,json={'model':'different','choices':[{'finish_reason':'stop','message':{'content':'{}'}}]})
    with httpx.Client(transport=httpx.MockTransport(transport)) as c:
        with pytest.raises(ValueError,match='identity'):g.call_model('source',tmp_path,c)


def test_optional_reasoning_is_disabled_for_structured_generation(tmp_path,monkeypatch):
    monkeypatch.delenv('AKSHARA_INFERENCE_CONFIG',raising=False)
    monkeypatch.setenv('AKSHARA_GENERATOR_PROVIDER','vultr')
    monkeypatch.setenv('AKSHARA_GENERATOR_MODEL','chosen')
    monkeypatch.setenv('VULTR_INFERENCE_API_KEY','test')
    def transport(req):
        if req.method=='GET':return httpx.Response(200,json={'data':[{'id':'chosen','reasoning':{'mandatory':False}}]})
        payload=json.loads(req.content)
        assert payload['reasoning']=={'enabled':False}
        return httpx.Response(200,json={'model':'chosen','choices':[{'finish_reason':'stop','message':{'content':'{"problems":[]}'}}]})
    with httpx.Client(transport=httpx.MockTransport(transport)) as c:
        assert g.call_model('source',tmp_path,c)=={'problems':[]}


def test_final_complete_packet_preserves_strict_json_values():
    assert g.decode_packet('Draft notes {not JSON}\n```json\n{"problems":[{"reference_answer":2}]}\n```')=={'problems':[{'reference_answer':2}]}
    with pytest.raises(ValueError,match='complete environment'):
        g.decode_packet('{"problems":[{"reference_answer":2}')
