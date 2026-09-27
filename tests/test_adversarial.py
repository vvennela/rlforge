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


@pytest.mark.parametrize('forged',[{'reward':1,'success':True},[],{'results':'pass'},
    {'results':[{'value':True}]},{'results':[{'value':1,'reward':1}]},
    {'results':[{'value':1},{'value':1}]},{'results':[{'value':float('nan')}]}])
def test_untrusted_program_cannot_supply_its_own_score(monkeypatch,forged):
    from akshara_forge.coding import service
    monkeypatch.setattr(service,'execute',lambda *a,**kw:forged)
    assert service.grade({'tests':[{'input':{},'expected':1}]},'code')['reward']==0


def test_adversarial_dataset_freezes_only_after_reference_and_mutation_gates(tmp_path,monkeypatch):
    from akshara_forge.coding import adversarial as a
    from akshara_forge.coding.tasks import generate
    source=tmp_path/'source.json';source.write_text('{"cards":[]}');original=tmp_path/'original';generate(original,source)
    counter=[0]
    def provider(prompt,folder,**kwargs):
        counter[0]+=1;items=[]
        for p in cases(counter[0])[:2]:
            def label(n):return str(counter[0])+'_'+n
            p={'graph':{label(n):[[label(c),w] for c,w in es] for n,es in p['graph'].items()},'heuristic':{label(n):v for n,v in p['heuristic'].items()},'start':label(p['start']),'goal':label(p['goal'])}
            items.append({'input':p,'failure_mode':'boundary and tie handling','reason':'Distinguishes cutoff and traversal mistakes'})
        return {'problems':items}
    monkeypatch.setattr(a,'call_model',provider)
    target=tmp_path/'adversarial';a.build(original,target)
    manifest=json.loads((target/'manifest.json').read_text())
    assert manifest['tests_per_task']==24 and counter[0]==8
    audit=json.loads((target/'adversarial-audit.json').read_text())
    assert not audit['survivors'] and all(c['mutants_killed'] for cs in audit['glm_case_coverage'].values() for c in cs)
    with pytest.raises(ValueError,match='Frozen'):a.build(original,target,resume=True)


def test_redundant_review_array_preserves_verification(tmp_path):
    def nested(answer):
        def call(prompt,folder,**kwargs):
            return {'problems':[reviewer(answer)(prompt,folder,**kwargs)['problems']]}
        return call
    assert review([row()],'Addition',tmp_path/'good',nested(4))['passed']
    with pytest.raises(ValueError,match='rejected'):
        review([row()],'Addition',tmp_path/'bad',nested(3))


def test_malformed_reviewer_retries_without_author_answers(tmp_path):
    calls=[]
    def provider(prompt,folder,**kwargs):
        calls.append(json.loads(prompt))
        assert 'secret author reasoning' not in prompt
        if len(calls)==1:return {'problems':['invalid']}
        if len(calls)==2:return {'problems':[{'id':[],'attacks':[]}]}
        return reviewer()(prompt,folder,**kwargs)
    assert review([row()],'Addition',tmp_path,provider)['passed']
    assert len(calls)==3
    assert all(c['problems']==[{'id':'q1','prompt':'Compute 2+2'}] for c in calls)
    assert (tmp_path/'blind-review/schema-error.json').exists()
    assert (tmp_path/'blind-review/schema-retry-1/schema-error.json').exists()


def test_advanced_task_cannot_pass_foundation_label(tmp_path):
    r=row();r['curriculum']={'level':1,'requirements':'One operation with scaffold'}
    def provider(prompt,folder,**kw):
        request=json.loads(prompt);assert request['difficulty_contracts']['q1']['level']==1
        packet=reviewer()(prompt,folder,**kw)
        packet['problems'][0].update(difficulty_appropriate=False,difficulty_reason='No required scaffold')
        return packet
    with pytest.raises(ValueError,match='rejected'):review([r],'Addition',tmp_path,provider)
    assert 'difficulty contract' in (tmp_path/'audit.json').read_text()


@pytest.mark.parametrize('level,operations,scaffold,concepts',[(1,5,True,['sum']),(3,4,True,['sum']),(4,5,False,['sum'])])
def test_difficulty_gate_checks_evidence_even_if_reviewer_says_appropriate(tmp_path,level,operations,scaffold,concepts):
    r=row();r['curriculum']={'level':level}
    def provider(prompt,folder,**kw):
        packet=reviewer()(prompt,folder,**kw)
        packet['problems'][0].update(difficulty_appropriate=True,difficulty_reason='Claimed fit',atomic_operations=operations,scaffold_present=scaffold,concepts_used=concepts)
        return packet
    with pytest.raises(ValueError,match='rejected'):review([r],'Addition',tmp_path,provider)
