"""Frozen bridge-repair pilot and a trusted, sandbox-backed reward service.

Synthetic geometric variants of the curated bridge abstraction, not newly
extracted drawings. The learner only places pieces; it cannot edit the verifier.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import hmac
import json
import random
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .environment import BRIDGE, BrickEnvironment, cells, evaluate, fingerprint, reference
from .worker import PALETTE, apply

SYSTEM = '''Repair the LEGO-style bridge with one JSON tool transaction. Return ONLY {"actions":[{"tool":"place","brick":{"id":"new1","part":"brick_1x1","x":0,"y":0,"z":9,"rotation":0}}]}. Use unique new IDs. Coordinates and rotation must be integers. x/y are studs; z is plate heights. The part brick_1x1 measures 1x1x3; plate_1x1 measures 1x1x1. Other parts are allowed but must fit the required shape exactly. Rotations are 0 or 90. Existing pieces are locked. Fill each missing upright from its current top up to its required top (exclusive), without overlaps, floating pieces, or extra material. At most 16 place actions. Do not explain your answer.'''


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def make_case(index, spec, split):
    length, left, right, top1, top2, side, station, depth = spec
    task = copy.deepcopy(BRIDGE)
    task.update(id=f'synthetic-bridge-{index:03}', provenance='Synthetic parametric bridge variant; not source-extracted dimensions.',
                source_dimensions_inches={'left_span':12*(left+1), 'central_span':12*(right-left), 'right_span':12*(length-right-1), 'width':72})
    task['grid'].update(length=length, pier_x=[left,right], upright_tops=[top1,top2])
    task['view_bounds']=[length,6,max(top1,top2)]
    task['vertical_dimensions'].update(deck_top_inches=43.2, short_post_top_inches_estimated=top1*4.8,
                                        short_post_estimate_tolerance_inches=0, tall_post_top_inches=top2*4.8)
    task['vertical_evidence']='Synthetic target heights, not measurements from the source drawing.'
    full = reference(task)
    x=[left,right][station]; top=[top1,top2][station]
    # Every cut lies on a real boundary in the deterministic reference assembly.
    candidates=sorted({b['z'] for b in full if b['x']==x and b['y']==side and b['z']>=9})
    cut=candidates[max(0,len(candidates)-depth)]
    missing=[b for b in full if b['x']==x and b['y']==side and b['z']>=cut]
    initial=[b for b in full if b not in missing]
    description={'units':{'x_y':'studs','z':'plate heights'},'bridge_length':length,'bridge_width':6,
                 'deck_top':9,'repair_uprights':[{'x':x,'y':side,'cross_section':[1,1],
                     'current_top':cut,'required_top':top}],
                 'existing_geometry':'The rest of the bridge is complete, connected, and locked. Place only the missing upper section.',
                 'allowed_parts':PALETTE}
    prompt=[{'role':'system','content':SYSTEM},{'role':'user','content':json.dumps(description,separators=(',',':'))}]
    return {'id':task['id'],'split':split,'task':task,'initial':initial,'missing':missing,
            'prompt':prompt,'geometry_signature':fingerprint(spec),'difficulty':len(missing)}


def build_dataset(output):
    output=Path(output)
    if output.exists(): raise ValueError('Dataset path already exists; do not overwrite frozen data')
    rng=random.Random(20260927); specs=set()
    while len(specs)<100:
        length=rng.choice(range(24,47,2)); left=rng.randint(5,10); right=length-rng.randint(7,11)
        specs.add((length,left,right,rng.randint(13,22),rng.randint(16,25),rng.choice([0,5]),rng.randrange(2),rng.randint(1,4)))
    specs=sorted(specs); rng.shuffle(specs)
    rows=[make_case(i,s,'train' if i<80 else 'heldout') for i,s in enumerate(specs)]
    # Prevent identical observable repair tasks from crossing the split.
    prompts=[fingerprint(r['prompt']) for r in rows]
    if len(set(prompts))!=len(rows): raise ValueError('Duplicate prompt; revise deterministic generation before training')
    for split in ['train','heldout']:
        write(output/f'{split}.json',[r for r in rows if r['split']==split])
    write(output/'manifest.json',{'seed':20260927,'train':80,'heldout':20,'kind':'synthetic_bridge_upright_repair',
          'reward':'0.8 added-geometry IoU with missing target + 0.2 full success; multiply by 0.1 if collisions or disconnected',
          'scope':'One-turn repair of a missing upper upright, not complete construction or automatic drawing extraction.',
          'sha256':{s:hashlib.sha256((output/f'{s}.json').read_bytes()).hexdigest() for s in ['train','heldout']}})


def parse(text):
    obj=json.loads(text)
    if not isinstance(obj,dict) or set(obj)!={'actions'}: raise ValueError('Return only actions')
    actions=obj['actions']
    if not isinstance(actions,list) or len(actions)>16: raise ValueError('At most 16 actions')
    if any(not isinstance(a,dict) or a.get('tool')!='place' for a in actions): raise ValueError('Only place actions permitted')
    return actions


def score(row, final):
    before=set().union(*(cells(b) for b in row['initial']))
    desired=set().union(*(cells(b) for b in row['missing']))
    occupied=set().union(*(cells(b) for b in final)) if final else set()
    added=occupied-before
    iou=len(added & desired)/len(added | desired)
    evaluation=evaluate(final,row['task'])
    valid=evaluation['collision_cells']==0 and evaluation['checks']['one_connected_assembly']
    reward=(0.8*iou+0.2*float(evaluation['success']))*(1 if valid else 0.1)
    return {'reward':reward,'success':evaluation['success'],'gap_iou':iou,'valid_geometry':valid,
            'reward_vector':evaluation['reward_vector'],'geometry_evaluation':evaluation}


def sandbox_grade(row,text,traces):
    try:
        actions=parse(text)
        # Validate structural tool input before spending a container launch.
        apply(row['initial'],actions,max_bricks=256)
    except (ValueError,KeyError,TypeError) as exc:
        return {'reward':0.,'success':False,'gap_iou':0.,'error':str(exc),'invalid_action':True}
    with BrickEnvironment(traces,task=row['task']) as env:
        env.step([{'tool':'place','brick':b} for b in row['initial']],actor='frozen_initial_state')
        obs=env.step(actions,actor='qwen_policy')
        if obs.get('tool_error'): return {'reward':0.,'success':False,'gap_iou':0.,'error':obs['tool_error']}
        return {**score(row,obs['bricks']),'episode':env.episode}


def serve(dataset,output,token_file,port):
    token=Path(token_file).read_text().strip(); output=Path(output); output.mkdir(parents=True,exist_ok=True)
    rows={r['id']:r for s in ['train','heldout'] for r in json.loads((Path(dataset)/f'{s}.json').read_text())}
    lock=threading.Lock(); slots=threading.Semaphore(2)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def do_POST(self):
            if self.path!='/grade' or not hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+token):
                self.send_error(403); return
            try:
                n=int(self.headers.get('Content-Length','0'))
                if not 0<n<=100000: raise ValueError('Invalid request length')
                req=json.loads(self.rfile.read(n)); row=rows[req['id']]
                if req['phase'] not in ['train','before','after','preflight']: raise ValueError('Unknown phase')
                if req['phase']=='train' and row['split']!='train': raise ValueError('Held-out task cannot supply training reward')
                if req['phase'] in ['before','after'] and row['split']!='heldout': raise ValueError('Evaluation must use heldout')
                with slots: result=sandbox_grade(row,req['completion'],output/'episodes')
                record={**req,'result':result}
                with lock:
                    with (output/'reward-traces.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
                raw=json.dumps(result).encode(); self.send_response(200); self.end_headers(); self.wfile.write(raw)
            except Exception as exc:
                raw=json.dumps({'service_error':str(exc)}).encode(); self.send_response(500); self.end_headers(); self.wfile.write(raw)
    write(output/'ready.json',{'port':port,'tasks':len(rows),'dataset_manifest':json.loads((Path(dataset)/'manifest.json').read_text())})
    ThreadingHTTPServer(('127.0.0.1',port),Handler).serve_forever()


def main():
    p=argparse.ArgumentParser(); p.add_argument('command',choices=['generate','serve']); p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path); p.add_argument('--token-file',type=Path); p.add_argument('--port',type=int,default=8767)
    a=p.parse_args()
    if a.command=='generate': build_dataset(a.dataset)
    else: serve(a.dataset,a.output,a.token_file,a.port)


if __name__=='__main__': main()
