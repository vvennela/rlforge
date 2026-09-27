import json
import math
import pytest
from akshara_forge.studio.adversarial import review
from akshara_forge.studio.verifier import accepts
from akshara_forge.coding.adversarial import independent, validate_input, mutant_solvers
from akshara_forge.coding.tasks import cases, oracle


def row():
    return {'id':'q1','prompt':'Compute 2+2','source_quote':'Addition','solution_outline':'secret author reasoning',
            'reference_answer':4,'verification':'numeric','tolerance':0}


def reviewer(answer=4,attack=5):
    def call(prompt,folder,*,system):
        inputs=json.loads(prompt)
        assert set(inputs['problems'][0])=={'id','prompt'}
        assert 'secret author reasoning' not in prompt
        return {'problems':[{'id':'q1','unambiguous':True,'reason':'2+2=4','source_quote':'Addition',
            'reference_answer':answer,'attacks':[{'answer':a,'expected_accept':False,'reason':'Incorrect'} for a in [attack,-1,'wrong']]}]}
    return call


def test_blind_review_hides_author_answers_and_preserves_probes(tmp_path):
    audit=review([row()],'Addition',tmp_path,reviewer())
    assert audit['passed'] and audit['problems'][0]['nonfinite_rejected']
    assert len(audit['problems'][0]['verifier_tests'])>=8


@pytest.mark.parametrize('provider',[reviewer(answer=3),reviewer(attack=4)])
def test_bad_reference_or_false_attack_prevents_release(tmp_path,provider):
    with pytest.raises(ValueError,match='rejected'):review([row()],'Addition',tmp_path,provider)
    assert not json.loads((tmp_path/'audit.json').read_text())['passed']


def test_numeric_reward_rejects_coercion_and_large_integer_rounding():
    r=row();r['reference_answer']=1
    for a in [True,'1',{'reward':1},None,float('nan'),float('inf')]:assert not accepts(r,a)
    r['reference_answer']=2**60
    assert accepts(r,2**60) and not accepts(r,2**60+1)
    r.update(reference_answer=.5,tolerance=1e-6)
    assert accepts(r,.500001) and not accepts(r,.500002)


def test_json_reward_rejects_extra_keys_nonfinite_and_boolean_coercion():
    r={'verification':'exact_json','reference_answer':{'value':1}}
    assert accepts(r,{'value':1})
    for a in [{'value':True},{'value':1,'reward':1},{'value':float('nan')},'{"value":1}']:
        assert not accepts(r,a)


def test_bad_glm_inputs_never_reach_reference_execution():
    for p in [
        {'graph':{'a':[['b',0]]},'start':'a','goal':'b','heuristic':{}},
        {'graph':{'a':[['b',1]]},'start':'a','goal':'b','heuristic':{'a':2}},
        {'graph':{},'start':'a','goal':'a','heuristic':{'a':1}}]:
        with pytest.raises(ValueError):validate_input(p)


def test_independent_stack_solver_and_mutation_gate():
    mutants=mutant_solvers();killed=set()
    for seed in range(4):
        for p in cases(seed):
            validate_input(p)
            c={'order':'given','root':True,'cutoffs':True,'goal':True,'initial':'heuristic','fields':['path','cost','bounds','visits','counts']}
            expected=oracle(p,c)
            assert independent(p,c)==expected
            for name,fn in mutants.items():
                if name in killed:continue
                try:value=fn(p,c)
                except (ValueError,RecursionError):value=None
                if value!=expected:killed.add(name)
    assert killed==set(mutants)
