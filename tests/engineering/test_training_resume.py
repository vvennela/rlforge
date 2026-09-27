import json
import httpx
import pytest
from akshara_forge.engineering import train_interactive as t


def test_resume_retains_only_checkpoint_prefix_and_original_baseline(tmp_path):
    parent=tmp_path/'parent';parent.mkdir();checkpoint=parent/'checkpoint-2';checkpoint.mkdir()
    for name in ('adapter_model.safetensors','optimizer.pt'):(checkpoint/name).write_bytes(b'complete')
    (parent/'initial-adapter.pt').write_bytes(b'initial')
    protocol={'steps':50,'manifest':{'sha256':{'train':'frozen'}},'source_hashes':{'trainer':'old'}}
    (parent/'protocol.json').write_text(json.dumps(protocol))
    (parent/'before').mkdir();baseline={'complete':True,'records':[{'id':'heldout-1','success':False}]}
    (parent/'before/summary.json').write_text(json.dumps(baseline))
    (parent/'optimizer-events.jsonl').write_text(''.join(json.dumps({'step':n,'reward_contrast_updates':n})+'\n' for n in range(1,5)))
    (parent/'training-traces.jsonl').write_text('\n'.join(json.dumps({'attempt':n}) for n in range(8))+'\n')
    output=tmp_path/'resumed';output.mkdir()
    before,effective=t.resume_evidence(checkpoint,output,{**protocol,'source_hashes':{'trainer':'recovery'}},2)
    assert before==baseline['records'] and effective==2
    assert len((output/'optimizer-events.jsonl').read_text().splitlines())==2
    assert len((output/'training-traces.jsonl').read_text().splitlines())==4
    assert len((parent/'optimizer-events.jsonl').read_text().splitlines())==4
    with pytest.raises(ValueError,match='manifest'):
        t.resume_evidence(checkpoint,tmp_path/'bad',{**protocol,'manifest':{}},2)


def test_stateless_reward_retries_server_failure_but_not_auth_failure(tmp_path,monkeypatch):
    token=tmp_path/'token';token.write_text('test-token')
    monkeypatch.setenv('AKSHARA_ENGINEERING_TOKEN_FILE',str(token));monkeypatch.setenv('AKSHARA_ENGINEERING_REWARD_URL','http://reward')
    monkeypatch.setattr(t.time,'sleep',lambda _:None)
    calls=[];statuses=[500,200]
    real_client=httpx.Client
    def transport(req):
        calls.append(json.loads(req.content));status=statuses.pop(0)
        return httpx.Response(status,json={'ok':True})
    monkeypatch.setattr(t.httpx,'Client',lambda **kw:real_client(transport=httpx.MockTransport(transport)))
    assert t.remote({'id':'train-1'},['action'],'train')=={'ok':True}
    assert len(calls)==2 and calls[0]==calls[1]
    statuses[:]=[403]
    with pytest.raises(httpx.HTTPStatusError):t.remote({'id':'train-1'},[],'train')
    assert len(calls)==3
