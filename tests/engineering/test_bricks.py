import copy,json,os,subprocess
from pathlib import Path
import pytest
from akshara_forge.engineering.environment import BRIDGE,BrickEnvironment,evaluate,reference
from akshara_forge.engineering.worker import apply


def test_feasible_reference_and_source_tolerances():
 r=evaluate(reference());assert r['success'];assert r['scalar_reward']==1
 assert r['dimension_details']['width']['actual_studs']==6
 assert r['dimension_details']['width']['target_studs']==5.5


def test_empty_has_no_success_or_complete_supports():
 r=evaluate([]);assert not r['success'];assert not r['checks']['two_complete_piers'];assert r['scalar_reward']==0


def test_overlap_cannot_get_success_even_with_perfect_outline():
 b=reference();b.append({**b[-1],'id':'duplicate'});r=evaluate(b)
 assert r['collision_cells']>0;assert not r['success'];assert r['scalar_reward']<=.25


def test_sparse_bounding_box_does_not_pass_geometry():
 b=[{'id':'a','part':'plate_2x2','x':0,'y':0,'z':7,'rotation':0},{'id':'b','part':'plate_2x2','x':36,'y':4,'z':7,'rotation':0}]
 r=evaluate(b);assert r['reward_vector']['length']==1;assert r['reward_vector']['deck_coverage']<.1;assert not r['success']


def test_wrong_pier_geometry_then_repair():
 b=reference()
 for part in b:
  if part['id'] in ['b2','b3','b5']:part['x']-=4
 before=evaluate(b);assert before['checks']['one_connected_assembly'];assert not before['success'];assert before['reward_vector']['support_span']<1
 b=apply(b,[{'tool':'move','id':'b2','x':26,'y':0,'z':0,'rotation':0},{'tool':'move','id':'b3','x':26,'y':0,'z':3,'rotation':0},{'tool':'move','id':'b5','x':26,'y':0,'z':6,'rotation':0}])
 assert evaluate(b)['success']


def test_side_contact_not_a_stud_connection():
 b=[{'id':'a','part':'brick_2x2','x':0,'y':0,'z':0,'rotation':0},{'id':'b','part':'brick_2x2','x':2,'y':0,'z':0,'rotation':0}]
 assert not evaluate(b)['checks']['one_connected_assembly']


def test_unneeded_geometry_cannot_pass():
 b=reference();b.append({'id':'extra','part':'plate_2x2','x':0,'y':0,'z':9,'rotation':0})
 r=evaluate(b);assert r['checks']['one_connected_assembly'];assert r['reward_vector']['shape_fidelity']<1;assert not r['success']


@pytest.mark.parametrize('action',[
 {'tool':'exec','command':'anything'},
 {'tool':'place','brick':{'id':'a','part':'brick_2x2','x':0.5,'y':0,'z':0,'rotation':0}},
 {'tool':'place','brick':{'id':'a','part':'brick_2x2','x':-1,'y':0,'z':0,'rotation':0}},
 {'tool':'place','brick':{'id':'a','part':'brick_2x2','x':0,'y':0,'z':0,'rotation':45}},
 {'tool':'place','brick':{'id':'a','part':'brick_2x2','x':0,'y':0,'z':0,'rotation':0,'role':'support'}},
])
def test_invalid_actions_are_atomic(action):
 b=reference();original=copy.deepcopy(b)
 with pytest.raises((ValueError,KeyError,TypeError)):apply(b,[{'tool':'remove','id':'b0'},action])
 assert b==original


@pytest.mark.skipif(os.getenv('AKSHARA_TEST_DOCKER')!='1',reason='explicit Docker integration test')
def test_docker_boundary_and_episode_trace(tmp_path):
 with BrickEnvironment(tmp_path) as env:
  obs=env.step([{'tool':'place','brick':b} for b in reference()]);assert obs['evaluation']['success'];assert obs['done']
  config=json.loads(subprocess.check_output(['docker','inspect',env.name]))[0]
  assert config['HostConfig']['NetworkMode']=='none'
  assert config['HostConfig']['ReadonlyRootfs']
  assert config['Config']['User']=='65534:65534'
  assert len(config['Mounts'])==1 and config['Mounts'][0]['Destination']=='/worker.py'
  assert config['Mounts'][0]['RW'] is False
  with pytest.raises(ValueError):env.step([])
  log=[json.loads(x) for x in env.log.read_text().splitlines()]
  assert log[0]['sandbox']=='docker-network-none-read-only';assert len(log)==2

@pytest.mark.skipif(os.getenv('AKSHARA_TEST_DOCKER')!='1',reason='explicit Docker integration test')
def test_tool_errors_remain_in_agent_observation(tmp_path):
 with BrickEnvironment(tmp_path) as env:
  obs=env.step([{'tool':'remove','id':'missing'}])
  assert obs['last_tool_error'] and obs['steps']==1 and obs['bricks']==[]
  assert env.observe()['last_tool_error']==obs['last_tool_error']


def test_missing_posts_cannot_hide_in_good_overall_proportions():
 b=[p for p in reference() if p['z']<9];r=evaluate(b)
 assert r['reward_vector']['length']==1 and r['reward_vector']['width']==1
 assert not r['success'] and not r['checks']['all_components_match']
 poles=[c for c in r['components'] if c['kind']=='upright']
 assert len(poles)==4 and all(c['scores']['presence']==0 for c in poles)


def test_component_matching_ignores_part_ids_and_list_order():
 b=reference()
 for i,p in enumerate(b):p['id']=f'renamed_{1000-i}'
 r=evaluate(list(reversed(b)));assert r['success']
 assert len(r['components'])==7 and all(c['complete'] for c in r['components'])


def test_short_pole_is_distinct_from_missing_pole():
 b=reference();b=[p for p in b if not (p['x']==26 and p['y']==0 and p['z']>=18)]
 r=evaluate(b);pole=next(c for c in r['components'] if c['id']=='upright_2_near')
 assert pole['scores']['presence']==1
 assert pole['scores']['size']<1 and pole['scores']['proportions']<1
 assert not r['dimension_details']['upright_2_near_top_height']['pass']
 assert not r['success']


def test_vertical_dimensions_preserve_evidence_type():
 r=evaluate(reference())
 assert r['dimension_details']['deck_top_height']['source_inches']==41.5
 assert r['dimension_details']['upright_2_far_top_height']['source_inches']==106.5
 assert r['dimension_details']['upright_1_near_top_height']['evidence_type']=='calibrated estimate'
 assert r['unmodeled_source_features'] and 'historic' in r['coverage_claim']
