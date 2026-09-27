import math
import pytest
from akshara_forge.environment import MathEnvironment, grade_answer, parse_response, public_problem


def task(kind='numeric'):
    return {'problem_id':'p', 'paper_id':'s', 'kind':'numeric','prompt':'2+2?',
            'reference_answer':4, 'rubric':['secret'], 'verification':{'type':kind,'tolerance':1e-6}}


def test_private_evaluator_data_never_enters_observation():
    env = MathEnvironment([task()])
    obs = env.reset('p')
    assert obs['observation'] == {'problem_id':'p','paper_id':'s','kind':'numeric','prompt':'2+2?'}
    assert env.step({'steps':['2+2=4'], 'answer':4})['reward'] == 1
    with pytest.raises(RuntimeError): env.step({'answer':4})


def test_ungraded_proof_is_neither_zero_nor_pass():
    result = grade_answer(task('judge'), {'answer':'sure'})
    assert result['status'] == 'needs_judge'
    assert result['reward'] is None and result['correct'] is None


@pytest.mark.parametrize('answer',[True,float('nan'),float('inf'),'NaN',{},[4]])
def test_numeric_verifier_rejects_invalid_answers(answer):
    assert not grade_answer(task(), {'answer':answer})['correct']


def test_absolute_tolerance_and_nested_shape():
    p=task(); p['reference_answer']={'x':[1e9,4]}
    assert not grade_answer(p, {'answer':{'x':[1e9+1,4]}})['correct']
    assert grade_answer(p, {'answer':{'x':[1e9,4+1e-7]}})['correct']
    assert not grade_answer(p, {'answer':{'x':[1e9,4],'extra':0}})['correct']


def test_parser_rejects_nan_and_accepts_json_fences():
    assert parse_response('{"answer":NaN}')['parse_error']
    assert parse_response('```json\n{"steps":["sum"],"answer":4}\n```')['answer']==4


def test_no_efficiency_reward_yet():
    with pytest.raises(ValueError): MathEnvironment([task()],step_penalty=.1)
