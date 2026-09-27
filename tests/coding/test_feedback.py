import json
from unittest.mock import patch
from akshara_forge.coding.feedback import feedback
from akshara_forge.coding.service import compare_results


def test_feedback_never_exposes_private_test_inputs_or_answers():
    row={'tests':[{'input':{'secret':'private graph'},'expected':{'path':['SECRET']}}],
         'development_tests':[{'input':{'graph':{'A':[['B',1]]}},'expected':{'path':['A','B']}}]}
    with patch('akshara_forge.coding.feedback.execute',return_value={'results':[{'value':{'path':None}}]}) as run:
        result=feedback(row,'def solve(p): return {"path":None}')
    assert run.call_args.args[1]==[row['development_tests'][0]['input']]
    assert result['feedback']['failed_examples'][0]['expected']=={'path':['A','B']}
    assert 'SECRET' not in json.dumps(result) and 'private graph' not in json.dumps(result)


def test_error_feedback_preserves_zero_score_and_execution_error():
    row={'development_tests':[{'input':{},'expected':{'path':[]}}]}
    with patch('akshara_forge.coding.feedback.execute',return_value={'error':'SyntaxError'}):
        result=feedback(row,'invalid code')
    assert result['feedback']['error']=='SyntaxError'
    assert result['development_score']==0 and not result['development_success']


def test_partial_field_credit_is_not_full_success():
    row={'tests':[{'expected':{'path':['A'],'cost':0}}]}
    result=compare_results(row,{'results':[{'value':{'path':['A'],'cost':7}}]},'components')
    assert result['reward']==.25 and result['field_score']==.5
    assert not result['success'] and result['passed']==0


def test_reward_envelope_and_extra_fields_cannot_earn_credit():
    row={'tests':[{'expected':{'path':['A'],'cost':0}}]}
    for value in [{'reward':1,'success':True},{'path':['A'],'cost':0,'reward':1}]:
        result=compare_results(row,{'results':[{'value':value}]},'components')
        assert result['reward']==0 and not result['success']
