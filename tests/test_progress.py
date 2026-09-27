import pytest
from akshara_forge.progress import score_progress, validate_contract
from akshara_forge.training import make_reward


CONTRACT={'milestones':[{'id':'a','description':'derive the derivative','prerequisites':[]},
                        {'id':'b','description':'establish the minimizer','prerequisites':['a']}],
          'outcome_weight':0.7,'progress_weight':0.3}


def judgment(steps, satisfied=(True,True)):
    return {'verdict':'incorrect','steps':[{'index':i,'verdict':s} for i,s in enumerate(steps)],
            'milestones':[{'id':m,'satisfied':ok,'supporting_steps':[i],'feedback':''}
                          for i,(m,ok) in enumerate(zip(['a','b'],satisfied))]}


def test_partial_progress_and_no_padding_bonus():
    action={'steps':['derivative','bad conclusion']}
    j=judgment(['valid','invalid'])
    result=score_progress(CONTRACT,action,j,{'reward':0})
    assert result['reward']==0.15
    assert result['credited_milestones']==['a']
    # Twenty extra true but irrelevant statements cannot change the fixed denominator.
    action['steps']+=['given']*20
    j['steps'] += [{'index':i,'verdict':'valid'} for i in range(2,22)]
    assert score_progress(CONTRACT,action,j,{'reward':0})['reward']==0.15


def test_invalid_dependency_cannot_earn_downstream_credit():
    result=score_progress(CONTRACT,{'steps':['wrong','dependent']},judgment(['invalid','valid']),{'reward':0})
    assert result['reward']==0


def test_full_solution_and_unresolved_problem():
    j=judgment(['valid','valid']);j['verdict']='correct'
    assert score_progress(CONTRACT,{'steps':['a','b']},j,{'reward':1})['reward']==1
    j['verdict']='invalid_problem'
    assert score_progress(CONTRACT,{'steps':['a','b']},j,{'reward':1})['reward'] is None


def test_citations_and_contract_cycles_rejected():
    j=judgment(['valid','valid']);j['milestones'][0]['supporting_steps']=[99]
    with pytest.raises(ValueError,match='nonexistent'):
        score_progress(CONTRACT,{'steps':['a','b']},j,{'reward':0})
    with pytest.raises(ValueError,match='prerequisites'):
        validate_contract({'milestones':[{'id':'a','description':'x','prerequisites':['a']}]})


def test_training_consumes_process_scalar(tmp_path):
    p={'a':{'reference_answer':4,'verification':{'type':'numeric'}}}
    reward=make_reward(p,tmp_path,process_teacher=lambda *args:{'reward':0.15,'outcome_reward':0})
    assert reward(['{"steps":["partial"],"answer":5}'],['a'])==[0.15]
