import copy,os
import pytest
from akshara_forge.engineering.environment import BrickEnvironment,cells,evaluate,reference
from akshara_forge.engineering.task_library import LIBRARY,targets
from akshara_forge.engineering.worker import apply

@pytest.mark.parametrize('key',list(LIBRARY))
def test_reference_is_feasible_and_independent_of_ids(key):
 task=LIBRARY[key];bricks=reference(task);result=evaluate(bricks,task)
 assert result['success'] and result['scalar_reward']==1
 assert len(bricks)<=task['limits']['bricks']
 for i,b in enumerate(bricks):b['id']=f'new_{i}'
 assert evaluate(list(reversed(bricks)),task)=={**result}  # geometry, not semantic labels

@pytest.mark.parametrize('key',list(LIBRARY))
def test_missing_component_and_overlap_are_not_accepted(key):
 task=LIBRARY[key];bricks=reference(task)
 missing=next(c['cells'] for c in targets(task) if c['id']==task['fault_component'])
 damaged=[b for b in bricks if not cells(b)&missing]
 result=evaluate(damaged,task)
 assert not result['success']
 part=next(c for c in result['components'] if c['id']==task['fault_component'])
 assert part['scores']['overlap']==0 and not part['complete']
 overlapped=bricks+[{**bricks[0],'id':'extra'}]
 result=evaluate(overlapped,task)
 assert not result['success'] and not result['checks']['collision_free']

@pytest.mark.parametrize('key',list(LIBRARY))
def test_displacement_cannot_hide_in_correct_sizes(key):
 task=LIBRARY[key];bricks=reference(task)
 for b in bricks:b['x']+=2
 result=evaluate(bricks,task)
 assert not result['success'] and result['reward_vector']['shape_fidelity']<1
 assert any(c['scores']['position']<1 for c in result['components'])

def test_evidence_is_not_invented_for_turbine():
 car=evaluate(reference(LIBRARY['car']),LIBRARY['car'])
 assert car['dimension_details']['wheelbase']['source_inches']==80
 turbine=evaluate(reference(LIBRARY['turbine']),LIBRARY['turbine'])
 assert turbine['dimension_details']=={}
 assert 'authored' in LIBRARY['turbine']['provenance']

def test_task_piece_limit_rejects_transaction_atomically():
 b=reference(LIBRARY['car']);original=copy.deepcopy(b)
 with pytest.raises(ValueError):apply([], [{'tool':'place','brick':p} for p in b],max_bricks=256)
 assert b==original

@pytest.mark.skipif(os.getenv('AKSHARA_TEST_DOCKER')!='1',reason='explicit Docker integration test')
@pytest.mark.parametrize('key',list(LIBRARY))
def test_new_tasks_in_actual_sandbox(tmp_path,key):
 task=LIBRARY[key]
 with BrickEnvironment(tmp_path,task=task) as env:
  obs=env.step([{'tool':'place','brick':b} for b in reference(task)],actor='scripted_fixture')
  assert obs['evaluation']['success'] and obs['done']
