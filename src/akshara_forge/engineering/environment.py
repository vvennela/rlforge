"""Trusted constraint evaluator and Docker workspace controller.

Units: x/y in studs; z in plate heights. All parts are simplified rectangular
stud-grid solids. This does not certify clutch strength or assembly order.
"""
from __future__ import annotations
import copy, hashlib, json, math, select, subprocess, time, uuid
from pathlib import Path
from .worker import PALETTE, validate

BRIDGE={
 'id':'ak0443-bridge-v0','title':'Old Bridge · Anaktuvuk Pass',
 'source_url':'https://www.loc.gov/item/ak0443/',
 'source_sha256':'7773dcab25ba8d65d9d27619da86c48dc1bc61df20aec66eba98fccefef0cb57',
 'provenance':'Manually reviewed drawing dimensions; task compilation is curated, not automatic OCR-to-environment.',
 'source_dimensions_inches':{'left_span':139.5,'central_span':183,'right_span':138.5,'width':66},
 'inches_per_stud':12,'dimension_tolerance_studs':0.5,
 'grid':{'length':38,'width':6,'deck_z':6,'deck_layers':2,'pier_x':[11,26],'pier_thickness':2},
 'abstraction':'Two rectangular piers and a continuous two-layer deck. Source ramps, log profiles, cross-braces, terrain and real structural behavior are excluded. Deck height is a toy construction choice, not extracted from the drawing.',
 'limits':{'transactions':16,'bricks':256},
 'model_interface':'reset → observe → step(actions) → reward_vector, scalar_reward, done',
}

def fingerprint(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def cells(b):
 dx,dy,dz=validate(b)
 return {(x,y,z) for x in range(b['x'],b['x']+dx) for y in range(b['y'],b['y']+dy) for z in range(b['z'],b['z']+dz)}

def reference(task=BRIDGE):
 """Deterministic feasible fixture for verifier QA. Never a model rollout."""
 g=task['grid'];L,W,Z=g['length'],g['width'],g['deck_z'];bricks=[]
 def add(part,x,y,z,r=0):bricks.append(dict(id=f'b{len(bricks)}',part=part,x=x,y=y,z=z,rotation=r))
 for x in g['pier_x']:
  for z in [0,3]:add('brick_2x6',x,0,z)
 for x in range(0,L,2):
  for y in range(0,W,2):add('plate_2x2',x,y,Z)
 # Stagger the upper layer across both lower-layer seam directions.
 for x in range(1,L-2,2):add('plate_2x4',x,1,Z+1)
 for x in [0,L-1]:add('plate_1x4',x,1,Z+1)
 for y in [0,W-1]:
  for x in range(0,L,2):add('plate_1x2',x,y,Z+1,90)
 return bricks

def evaluate(bricks,task=BRIDGE):
 g=task['grid'];L,W,Z=g['length'],g['width'],g['deck_z'];issues=[];voxels=[]
 if not isinstance(bricks,list) or len(bricks)>256:raise ValueError('Invalid assembly')
 ids=set()
 for b in bricks:
  if b['id'] in ids:raise ValueError('Duplicate id')
  ids.add(b['id']);voxels.append(cells(b))
 occupied=set().union(*voxels) if voxels else set()
 collision_cells=sum(map(len,voxels))-len(occupied)
 # Connections exist only across horizontal interfaces with overlapping stud cells.
 # Touching vertical side walls does not connect adjacent bricks.
 graph=[set() for _ in bricks]
 for i,a in enumerate(bricks):
  ax,ay,az=validate(a)
  for j in range(i):
   b=bricks[j];bx,by,bz=validate(b)
   interface=a['z']+az==b['z'] or b['z']+bz==a['z']
   if interface and min(a['x']+ax,b['x']+bx)>max(a['x'],b['x']) and min(a['y']+ay,b['y']+by)>max(a['y'],b['y']):graph[i].add(j);graph[j].add(i)
 reached=set();todo=[0] if bricks else []
 while todo:
  i=todo.pop()
  if i in reached:continue
  reached.add(i);todo.extend(graph[i]-reached)
 connected=bool(bricks) and len(reached)==len(bricks)
 target_deck={(x,y,z) for x in range(L) for y in range(W) for z in range(Z,Z+2)}
 target_piers={(x,y,z) for start in g['pier_x'] for x in range(start,start+2) for y in range(W) for z in range(Z)}
 target=target_deck|target_piers
 deck={(x,y,z) for x,y,z in occupied if Z<=z<Z+2}
 lengths=[max(p[k] for p in deck)-min(p[k] for p in deck)+1 if deck else None for k in [0,1]]
 # Infer support locations from lower geometry, not agent-assigned semantic labels.
 xs=sorted({x for x,y,z in occupied if z<Z});groups=[]
 for x in xs:
  if not groups or x>groups[-1][-1]+1:groups.append([x])
  else:groups[-1].append(x)
 centers=[(min(a)+max(a)+1)/2 for a in groups]
 two_piers=len(centers)==2 and bool(target_piers) and target_piers<=occupied
 support_span=centers[1]-centers[0] if len(centers)==2 else None
 support_offset=centers[0] if len(centers)==2 else None
 source=task['source_dimensions_inches'];scale=task['inches_per_stud'];tol=task['dimension_tolerance_studs']
 desired={'length':sum(source[k] for k in ['left_span','central_span','right_span'])/scale,'width':source['width']/scale,'support_span':source['central_span']/scale,'support_offset':source['left_span']/scale}
 measured={'length':lengths[0],'width':lengths[1],'support_span':support_span,'support_offset':support_offset}
 scores={};detail={}
 for key,want in desired.items():
  got=measured[key];error=abs(got-want) if got is not None else None
  scores[key]=max(0.,1-max(0.,error-tol)/want) if error is not None else 0.
  detail[key]={'target_studs':want,'actual_studs':got,'error_studs':error,'tolerance_studs':tol,'pass':error is not None and error<=tol}
 scores['deck_coverage']=len(occupied&target_deck)/len(target_deck)
 scores['support_coverage']=len(occupied&target_piers)/len(target_piers)
 scores['shape_fidelity']=len(occupied&target)/len(occupied|target) if occupied else 0.
 checks={'collision_free':collision_cells==0,'one_connected_assembly':connected,'two_complete_piers':two_piers}
 valid=all(checks.values());geometry=sum(scores.values())/len(scores)
 # Incomplete builds still receive bounded shaping; acceptance requires all checks.
 scalar=geometry*(1. if valid else .25)
 success=valid and all(d['pass'] for d in detail.values()) and all(scores[k]==1 for k in ['deck_coverage','support_coverage','shape_fidelity'])
 if collision_cells:issues.append(f'{collision_cells} overlapping occupied cells')
 if not connected:issues.append('Assembly is empty or has disconnected components')
 if not two_piers:issues.append('Two complete piers required at the specified positions')
 return {'reward_vector':scores,'scalar_reward':scalar,'success':success,'dimension_details':detail,'checks':checks,'issues':issues,'brick_count':len(bricks),'collision_cells':collision_cells,'connection_count':sum(map(len,graph))//2,'spec_hash':fingerprint(task),'verifier':'brick-grid-v0','limitations':'Grid connection and geometry checks only; no strength, friction, clutch, physical stability or insertion-order certification.'}

class BrickEnvironment:
 def __init__(self,trace_dir:Path,task=None):
  self.task=copy.deepcopy(task or BRIDGE);self.trace_dir=Path(trace_dir);self.trace_dir.mkdir(parents=True,exist_ok=True)
  self.name='akshara-bricks-'+uuid.uuid4().hex[:12];self.bricks=[];self.steps=0;self.done=False;self.proc=None;self.last_tool_error=None
  self.episode=uuid.uuid4().hex;self.log=self.trace_dir/f'{self.episode}.jsonl'
  worker=Path(__file__).with_name('worker.py').resolve()
  cmd=['docker','run','--rm','-i','--name',self.name,'--network=none','--read-only','--cap-drop=ALL','--security-opt=no-new-privileges','--pids-limit=16','--memory=128m','--cpus=0.5','--user=65534:65534','--mount',f'type=bind,source={worker},target=/worker.py,readonly','python:3.12-slim','python','-I','-B','-u','/worker.py']
  self.proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,bufsize=1)
  self._record({'event':'reset','task':self.task,'worker_sha256':hashlib.sha256(worker.read_bytes()).hexdigest(),'sandbox':'docker-network-none-read-only','verifier_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
 def _record(self,data):
  with self.log.open('a') as f:f.write(json.dumps({'time':time.time(),'episode':self.episode,**data})+'\n')
 def observe(self):return {'task':copy.deepcopy(self.task),'palette':copy.deepcopy(PALETTE),'bricks':copy.deepcopy(self.bricks),'evaluation':evaluate(self.bricks,self.task),'steps':self.steps,'done':self.done,'last_tool_error':self.last_tool_error,'episode':self.episode,'sandbox':'Docker: no network, read-only, unprivileged; verifier on host'}
 def step(self,actions,*,actor="controller"):
  if self.done:raise ValueError('Episode ended; reset required')
  raw=json.dumps({'actions':actions},allow_nan=False)
  if len(raw)>100000:raise ValueError('Action request too large')
  self.proc.stdin.write(raw+'\n');self.proc.stdin.flush()
  if not select.select([self.proc.stdout],[],[],20)[0]:self.close();raise TimeoutError('Sandbox response timed out')
  line=self.proc.stdout.readline(100001)
  if not line or len(line)>100000:self.close();raise RuntimeError('Sandbox terminated or returned oversized output')
  result=json.loads(line)
  self.last_tool_error=result.get('error')
  if result.get('ok'):evaluate(result['bricks'],self.task);self.bricks=result['bricks']
  self.steps+=1;self.done=self.steps>=self.task['limits']['transactions']
  obs=self.observe();obs['tool_error']=result.get('error');obs['done']=self.done or obs['evaluation']['success'];self.done=obs['done']
  self._record({'event':'step','actor':actor,'actions':actions,'observation':obs})
  return obs
 def close(self):
  if self.proc:
   subprocess.run(['docker','rm','-f',self.name],capture_output=True,timeout=15)
   self.proc.communicate(timeout=10);self.proc=None
 def __enter__(self):return self
 def __exit__(self,*args):self.close()
