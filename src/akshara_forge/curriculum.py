"""Versioned difficulty contracts shared by generated learning environments."""
import copy

VERSION = 'easy-to-hard-v1'
STAGES = [
    {'level':1,'name':'foundation','prerequisites':[],
     'requirements':'One concept and one operation on small explicit inputs. Supply a worked rule or a nearly complete scaffold. Math: direct substitution or one algebraic step. Coding: repair one local bug in a supplied function. Engineering: place one missing piece at an explicit target. No combined edge cases.'},
    {'level':2,'name':'guided','prerequisites':[1],
     'requirements':'Two or three operations using the foundation skill. Supply a scaffold or intermediate subgoals. Math: a short derivation with explicit values. Coding: fill one bounded-search component on small trees. Engineering: stack two or three pieces in a single upright. Introduce one boundary case at a time.'},
    {'level':3,'name':'application','prerequisites':[1,2],
     'requirements':'Apply the method independently with small finite inputs and limited output requirements. Math: a new worked instance without the solution scaffold. Coding: implement the basic algorithm returning path and cost before instrumentation. Engineering: complete a longer upright with mixed piece sizes. Combine at most two previously practiced constraints.'},
    {'level':4,'name':'composition','prerequisites':[1,2,3],
     'requirements':'Combine mastered components with edge cases and richer outputs. Math: a multi-step composition of source methods. Coding: add cycles, tie handling and exact instrumentation to the algorithm. Engineering: correct misplaced pieces while completing the target. Every task must remain finite, explicit and independently verifiable.'},
]

def stage(level):
    if type(level) is not int or not 1<=level<=4:raise ValueError('Curriculum level must be 1–4')
    return copy.deepcopy(STAGES[level-1])

def slot(index,count):
    """Equal stage quotas; 80/20 split within every stage, never hardest-only holdout."""
    if count not in (20,100) or not 0<=index<count:raise ValueError('Invalid curriculum position')
    size=count//4;level=index//size+1;within=index%size
    return {**stage(level),'split':'train' if within<size*4//5 else 'heldout'}

def manifest(count):
    return {'version':VERSION,'order':'foundation → guided → application → composition',
            'split_policy':'80/20 within each difficulty level; heldout answers never select progression or checkpoints',
            'stages':[{**s,'train':count//5,'heldout':count//20} for s in STAGES],
            'difficulty_basis':'Explicit task constraints and reviewer check; levels are design targets, not measured model proficiency.'}

def schedule(rows,steps):
    """A bounded run traverses every available stage, instead of stopping in level 1."""
    if steps<1:raise ValueError('Positive update count required')
    if any(r.get('split')!='train' for r in rows):raise ValueError('Only training tasks may enter a curriculum')
    levels=sorted({r['curriculum']['level'] for r in rows})
    if steps<len(levels):raise ValueError('At least one update per curriculum level required')
    result=[]
    for j,level in enumerate(levels):
        group=sorted((r for r in rows if r['curriculum']['level']==level),key=lambda r:r['id'])
        quota=steps//len(levels)+(j<steps%len(levels))
        result.extend(group[i%len(group)] for i in range(quota))
    return result
