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
    def provider(prompt,folder,*,system=g.SYSTEM,json_prefix=False):
        assert json_prefix is True
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
    assert len(json.loads((p/'learner/tasks.json').read_text()))==16
    assert len(json.loads((p/'evaluation/tasks.json').read_text()))==4
    assert s['manifest']['runtime_audit']['cross_split_access_blocked']
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


def test_vultr_json_continuation_reassembles_only_new_output(tmp_path,monkeypatch):
    monkeypatch.delenv('AKSHARA_INFERENCE_CONFIG',raising=False)
    monkeypatch.setenv('AKSHARA_GENERATOR_PROVIDER','vultr')
    monkeypatch.setenv('AKSHARA_GENERATOR_MODEL','chosen')
    monkeypatch.setenv('VULTR_INFERENCE_API_KEY','test')
    def transport(req):
        if req.method=='GET':return httpx.Response(200,json={'data':[{'id':'chosen'}]})
        payload=json.loads(req.content)
        assert payload['continue_final_message'] is True
        assert payload['temperature']==0.2
        assert payload['messages'][-1]=={'role':'assistant','content':'{"problems":['}
        return httpx.Response(200,json={'model':'chosen','choices':[{'finish_reason':'stop','message':{'content':'{"id":"one"}]}'} }]})
    with httpx.Client(transport=httpx.MockTransport(transport)) as c:
        assert g.call_model('source',tmp_path,c,json_prefix=True)=={'problems':[{'id':'one'}]}


def test_bridge_gallery_pairs_by_identity_and_keeps_frozen_order(tmp_path):
    from akshara_forge.studio.evidence import engineering_samples
    dataset=tmp_path/'runs/engineering-interactive/dataset-v1'
    dataset.mkdir(parents=True)
    rows=[{'id':f'case-{i}','task':{},'target':{},'initial':[]} for i in range(3)]
    (dataset/'heldout.json').write_text(json.dumps(rows))
    run=tmp_path/'run'
    for phase,ids in [('before',[2,0,1]),('after',[1,0])]:
        dest=run/phase;dest.mkdir(parents=True)
        (dest/'traces.jsonl').write_text('\n'.join(json.dumps({'id':f'case-{i}','state':{'observation':{'case':i,'phase':phase}}}) for i in ids))
    samples=engineering_samples(tmp_path,run)
    assert [s['id'] for s in samples]==['case-0','case-1','case-2']
    assert samples[0]['before']['case']==samples[0]['after']['case']==0
    assert samples[2]['after'] is None
    with (run/'after/traces.jsonl').open('a') as f:
        f.write('\n'+json.dumps({'id':'unknown','state':{'observation':{}}}))
    with pytest.raises(ValueError,match='identity mismatch'):
        engineering_samples(tmp_path,run)


def test_source_alignment_preserves_symbols_and_requires_unique_span():
    from akshara_forge.studio.source import bind_quote
    source='The rule is x = 1\n + y.';row={'source_quote':'The rule is x = 1 + y.'}
    bind_quote(row,source)
    assert row['source_quote']==source
    assert row['source_quote_submitted']=='The rule is x = 1 + y.'
    assert source[row['source_quote_alignment']['start_char']:row['source_quote_alignment']['end_char']]==row['source_quote']
    with pytest.raises(ValueError):bind_quote({'source_quote':'The rule is x = 1 - y.'},source)
    with pytest.raises(ValueError,match='unique'):bind_quote({'source_quote':'a b'},'a\n b; a\t b')


def test_rejected_upload_batch_is_preserved_and_repaired_before_release(tmp_path,monkeypatch):
    monkeypatch.setenv('AKSHARA_GENERATOR_PROVIDER','openai');monkeypatch.setenv('OPENAI_API_KEY','test')
    monkeypatch.delenv('AKSHARA_INFERENCE_CONFIG',raising=False)
    calls=[];drafts=[0]
    def provider(prompt,folder,*,system=g.SYSTEM,json_prefix=False):
        request=json.loads(prompt);calls.append((folder,request))
        if system!=g.SYSTEM:
            return {'problems':[{'id':r['id'],'unambiguous':True,'reason':'Add one','source_quote':'Addition','reference_answer':int(r['prompt'].split()[1])+1,'attacks':[{'answer':a,'expected_accept':False,'reason':'Wrong value'} for a in [-1,-2,-3]]} for r in request['problems']]}
        drafts[0]+=1
        return {'problems':[{'prompt':f'Compute {drafts[0]*10+i} + 1','source_quote':'invented' if drafts[0]==1 else 'Addition','solution_outline':'Add one.','reference_answer':drafts[0]*10+i+1,'verification':'numeric','tolerance':0} for i in range(10)]}
    monkeypatch.setattr(g,'call_model',provider)
    jobs=g.GenerationJobs(tmp_path);s=jobs.start({'filename':'source.txt','data':base64.b64encode(b'Addition').decode(),'count':20})
    for _ in range(200):
        s=jobs.read(s['id'])
        if s['status'] in ('ready','failed'):break
        time.sleep(.01)
    assert s['status']=='ready',s
    root=tmp_path/s['id']
    assert (root/'calls/00/attempt-1/rejection.json').exists()
    assert any(req.get('revision_feedback',{}).get('error') for _,req in calls if req.get('revision_feedback'))
    feedback=next(req['revision_feedback'] for _,req in calls if req.get('revision_feedback'))
    assert feedback['rejected_packet']['problems'][0]['source_quote']=='invented'
    rows=json.loads((root/'environment/private/problems.json').read_text())
    assert len(rows)==20 and all(r['source_quote']=='Addition' for r in rows)
    assert drafts[0]==3  # One rejected draft, then exactly two accepted batches.


def test_source_ids_resolve_to_exact_preserved_passages():
    from akshara_forge.studio.source import passages,bind_reference
    source='A theorem with symbols x ≥ 2.\n'*120
    parts=passages(source)
    assert ''.join(p['text'] for p in parts)==source
    row={'source_id':parts[1]['id']}
    bind_reference(row,source)
    span=row['source_quote_alignment']
    assert row['source_quote']==source[span['start_char']:span['end_char']]
    assert row['source_quote']==parts[1]['text']
    with pytest.raises(ValueError,match='Unknown source_id'):
        bind_reference({'source_id':'invented'},source)


def test_structured_problem_states_response_shape_without_answer_values():
    packet={'problems':[{'prompt':'Compute the sum and path.','source_quote':'Addition',
      'solution_outline':'Calculate.','verification':'exact_json','reference_answer':{'sum':437,'path':['A','B','C']}}]}
    row=g.validate(packet,'Addition',1)[0]
    assert '"sum"' in row['prompt'] and '"path"' in row['prompt']
    assert '437' not in row['prompt'] and '"A"' not in row['prompt']
    assert 'minItems' not in row['prompt'] and 'maxItems' not in row['prompt']
    assert row['answer_schema']['properties']['sum']=={'type':'integer'}


def test_response_shape_does_not_reveal_null_or_empty_answers():
    from akshara_forge.studio.source import answer_schema
    assert answer_schema(None)=={}
    assert answer_schema({})=={'type':'object'}
    assert answer_schema([])=={'type':'array','items':{}}


def test_malformed_provider_json_retries_identical_blind_prompt_and_retains_receipts(tmp_path,monkeypatch):
    monkeypatch.delenv('AKSHARA_INFERENCE_CONFIG',raising=False)
    monkeypatch.setenv('AKSHARA_GENERATOR_PROVIDER','vultr')
    monkeypatch.setenv('AKSHARA_GENERATOR_MODEL','chosen')
    monkeypatch.setenv('VULTR_INFERENCE_API_KEY','test')
    requests=[]
    def transport(req):
        if req.method=='GET':return httpx.Response(200,json={'data':[{'id':'chosen'}]})
        requests.append(json.loads(req.content))
        content='{"answer":"117,"reason":"broken"}]}' if len(requests)==1 else '{"answer":117}]}'
        return httpx.Response(200,json={'model':'chosen','choices':[{'finish_reason':'stop','message':{'content':content}}]})
    with httpx.Client(transport=httpx.MockTransport(transport)) as c:
        assert g.call_model('Only the question',tmp_path,c,system='Blind reviewer',json_prefix=True)=={'problems':[{'answer':117}]}
    assert requests[0]==requests[1]
    assert (tmp_path/'parse-error.json').exists()
    assert (tmp_path/'response.json').exists() and (tmp_path/'json-retry/response.json').exists()


def test_upload_keeps_validated_items_and_only_replaces_rejected_ones(tmp_path,monkeypatch):
    monkeypatch.setenv('AKSHARA_GENERATOR_PROVIDER','openai');monkeypatch.setenv('OPENAI_API_KEY','test')
    monkeypatch.delenv('AKSHARA_INFERENCE_CONFIG',raising=False)
    counts=[];drafts=[0]
    def provider(prompt,folder,*,system=g.SYSTEM,json_prefix=False):
        request=json.loads(prompt)
        if system!=g.SYSTEM:
            return {'problems':[{'id':r['id'],'unambiguous':True,'reason':'Add one','source_quote':'Addition',
                'reference_answer':int(r['prompt'].split()[1])+1,
                'attacks':[{'answer':a,'expected_accept':False,'reason':'Wrong'} for a in [-1,-2,-3]]} for r in request['problems']]}
        drafts[0]+=1;n=int(request['request'].split()[1]);counts.append(n)
        return {'problems':[{'prompt':f'Compute {drafts[0]*100+i} + 1','source_quote':'Addition','solution_outline':'Add one.',
            'reference_answer':0 if drafts[0]==1 and i==3 else drafts[0]*100+i+1,
            'verification':'numeric','tolerance':0} for i in range(n)]}
    monkeypatch.setattr(g,'call_model',provider)
    jobs=g.GenerationJobs(tmp_path);s=jobs.start({'filename':'source.txt','data':base64.b64encode(b'Addition').decode(),'count':20})
    for _ in range(200):
        s=jobs.read(s['id'])
        if s['status'] in ('ready','failed'):break
        time.sleep(.01)
    assert s['status']=='ready',s
    assert counts==[10,1,10]
    root=tmp_path/s['id'];rows=json.loads((root/'environment/private/problems.json').read_text())
    assert [r['id'] for r in rows]==[f'problem-{i:03}' for i in range(1,21)]
    assert rows[0]['reference_answer']==101 and rows[3]['reference_answer']==201
    assert sum(r['split']=='train' for r in rows)==16
    original=json.loads((root/'calls/00/attempt-1/audit.json').read_text())
    assert not original['passed']
    published=json.loads((root/'environment/private/adversarial-audit.json').read_text())
    assert all(r['passed'] for a in published for r in a['problems'])
    assert published[0]['problems'][0]['review_evidence']=='private/reviews/calls/00/attempt-1/audit.json'
    assert published[0]['problems'][3]['review_evidence']=='private/reviews/calls/00/attempt-2/audit.json'
    progress=json.loads((root/'verified-progress.json').read_text())
    assert progress['completed']==20 and progress['problems']==rows

    for audit in published:
        for item in audit['problems']:
            assert (root/'environment'/item['review_evidence']).is_file()


@pytest.mark.parametrize('answer,mode',[(3,'numeric'),({'cost':3},'exact_json')])
def test_revalidating_revised_question_does_not_duplicate_output_instructions(answer,mode):
    packet={'problems':[{'prompt':'Compute the cost','source_quote':'Addition','solution_outline':'Add one',
        'reference_answer':answer,'verification':mode,'tolerance':0}]}
    g.validate(packet,'Addition',1);first=packet['problems'][0]['prompt']
    g.validate(packet,'Addition',1)
    assert packet['problems'][0]['prompt']==first
    assert first.count('Return only')==1


def test_coding_gallery_pairs_complete_baseline_with_partial_final_evaluation(tmp_path):
    from akshara_forge.studio.evidence import coding_samples
    for phase in ('before','after'):(tmp_path/phase).mkdir()
    before=[{'id':f'case-{i}','completion':str(i)} for i in range(3)]
    (tmp_path/'before/episodes.jsonl').write_text('\n'.join(map(json.dumps,before)))
    (tmp_path/'after/episodes.jsonl').write_text(json.dumps(before[0]))
    samples=coding_samples(tmp_path)
    assert [r['id'] for r in samples]==['case-0','case-1','case-2']
    assert samples[0]['after']['id']=='case-0' and samples[1]['after'] is None
    (tmp_path/'after/episodes.jsonl').write_text(json.dumps(before[1]))
    with pytest.raises(ValueError,match='pairing'):coding_samples(tmp_path)
    (tmp_path/'after/episodes.jsonl').write_text('\n'.join([json.dumps(before[0])]*2))
    with pytest.raises(ValueError,match='Duplicate'):coding_samples(tmp_path)
