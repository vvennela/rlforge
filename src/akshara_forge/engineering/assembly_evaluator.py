"""Fixed-region occupancy rewards for curated multi-part assembly tasks."""
from .components import bounds,compare_masks
from .task_library import targets


def evaluate_assembly(bricks,task):
 from .environment import assembly_geometry,fingerprint
 occupied,collisions,connected,connections=assembly_geometry(bricks,task['limits']['bricks'])
 expected=targets(task);target=set().union(*(t['cells'] for t in expected))
 components=compare_masks(expected,occupied)
 scores={'shape_fidelity':len(occupied&target)/len(occupied|target)}
 details={}
 if task['family']=='car':
  actual=bounds(occupied) if occupied else None
  measured={k:actual['size'][i]*(.4 if i==2 else 1) if actual else None for i,k in enumerate(['length','width','height'])}
  wheels={c['id']:c['actual_bounds'] for c in components if c['id'].startswith('wheel_')}
  measured['wheelbase']=sum(wheels['wheel_rear_'+s]['center'][0]-wheels['wheel_front_'+s]['center'][0] for s in ['near','far'])/2 if all(wheels.values()) else None
  for name,inches in task['source_dimensions_inches'].items():
   want=inches/task['inches_per_stud'];got=measured[name];tol=task['dimension_tolerance_studs'];error=abs(want-got) if got is not None else None
   scores[name]=max(0.,1-max(0.,error-tol)/want) if error is not None else 0.
   details[name]={'target_studs':want,'actual_studs':got,'source_inches':inches,'evidence_type':'printed dimension','error_studs':error,'tolerance_studs':tol,'pass':error is not None and error<=tol}
 for c in components:
  for k,v in c['scores'].items():scores[c['id']+'/'+k]=v
 checks={'collision_free':collisions==0,'one_connected_assembly':connected,'all_components_match':all(c['complete'] for c in components)}
 valid=all(checks.values());success=valid and scores['shape_fidelity']==1 and all(d['pass'] for d in details.values())
 issues=[]
 if collisions:issues.append(f'{collisions} overlapping occupied cells')
 if not connected:issues.append('Assembly is empty or has disconnected components')
 return {'reward_vector':scores,'scalar_reward':sum(scores.values())/len(scores)*(1 if valid else .25),'success':success,'dimension_details':details,'checks':checks,'issues':issues,'brick_count':len(bricks),'collision_cells':collisions,'connection_count':connections,'spec_hash':fingerprint(task),'components':components,'unmodeled_source_features':task['unmodeled_source_features'],'coverage_claim':task['coverage_claim'],'verifier':'fixed-region-assembly-v1','limitations':'Known target-region occupancy and vertical stud connections only; no physical stability, insertion-order, functional mechanism or automatic semantic recognition certification.'}
