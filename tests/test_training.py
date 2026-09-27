import pytest
import json
from akshara_forge.training import make_reward,ordered_training_rows
from akshara_forge.io import digest_file


def test_reward_integration_and_unresolved_proof_stops(tmp_path):
    problems={'a':{'reference_answer':4,'verification':{'type':'numeric'}},
              'b':{'reference_answer':'proof','verification':{'type':'judge'}}}
    reward=make_reward(problems,tmp_path)
    assert reward(['{"answer":4}','{"answer":5}'],['a','a'])==[1,0]
    with pytest.raises(RuntimeError,match='require Astra'):reward(['{"answer":"proof"}'],['b'])
    reward=make_reward(problems,tmp_path,teacher=lambda *a:{'reward':None})
    with pytest.raises(RuntimeError,match='Unresolved'):reward(['{"answer":"proof"}'],['b'])


def test_teacher_reward_and_audit_trace(tmp_path):
    p={'b':{'reference_answer':'proof','verification':{'type':'judge'}}}
    reward=make_reward(p,tmp_path,teacher=lambda *a:{'reward':1,'verification':'teacher_judgment'})
    assert reward([[{'role':'assistant','content':'{"steps":["deduction"],"answer":"proof"}'}]],['b'])==[1]
    assert 'deduction' in (tmp_path/'reward-traces.jsonl').read_text()


def test_curriculum_freezes_training_order_and_rejects_stale_data(tmp_path):
    source=tmp_path/'train.jsonl';source.write_text('frozen training data')
    schedule=tmp_path/'curriculum.json'
    schedule.write_text(json.dumps({'split':'train','source_sha256':digest_file(source),'ordered_ids':['b','a']}))
    rows=[{'problem_id':'a'},{'problem_id':'b'}];problems={'a':{},'b':{}}
    assert [r['problem_id'] for r in ordered_training_rows(rows,problems,schedule,source)]==['b','a']
    source.write_text('changed')
    with pytest.raises(ValueError,match='frozen training'):ordered_training_rows(rows,problems,schedule,source)
