"""Eight-turn, sandbox-executed bridge repair with observable vector feedback.

Stateless replay makes a timed-out request safe to retry: no action is duplicated
in a persistent workspace. Immutable bridge pieces cannot be moved or removed.
"""
from __future__ import annotations
import argparse, copy, hashlib, hmac, json, random, threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from .environment import BrickEnvironment, cells, evaluate, fingerprint
from .pilot import make_case, write
from .worker import apply, PALETTE

TURNS=8
SYSTEM='''You are repairing a LEGO-style bridge in an interactive sandbox. You have up to 8 turns. After each turn you receive the actual structure and measured feedback. Build the entire requested upright; one brick may not be enough. Correct your own mistakes by moving or removing editable pieces. The rest of the bridge is locked.
Return JSON only, with an actions array containing 1 to 4 actions. Action schemas:
place: {"tool":"place","brick":{"id":STRING,"part":STRING,"x":INTEGER,"y":INTEGER,"z":INTEGER,"rotation":0}}
move: {"tool":"move","id":STRING,"x":INTEGER,"y":INTEGER,"z":INTEGER,"rotation":0}
remove: {"tool":"remove","id":STRING}
Use unique IDs for new pieces. Available parts: brick_1x1 is 1x1 studs and 3 plate-heights tall; plate_1x1 is 1x1 studs and 1 plate-height tall. x and y are studs; z is plate heights. A piece at z occupies z through z+height, excluding the upper boundary. Read the CURRENT observation after every turn. Fill all missing target cells, avoid extra material and overlaps, and keep the build connected. No explanatory text.'''


def generate(path,old_dataset):
    path=Path(path)
    if path.exists(): raise ValueError('Never overwrite a frozen dataset')
    excluded={r['geometry_signature'] for s in ['train','heldout'] for r in json.loads((Path(old_dataset)/f'{s}.json').read_text())}
    rng=random.Random(20260928);rows=[];seen=set()
    while len(rows)<100:
        L=rng.choice(range(24,47,2)); spec=(L,rng.randint(5,10),L-rng.randint(7,11),rng.randint(16,24),rng.randint(17,26),rng.choice([0,5]),rng.randrange(2),rng.choice([2,3,4]))
        sig=fingerprint(spec)
        if sig in excluded or sig in seen: continue
        i=len(rows);row=make_case(i,spec,'train' if i<80 else 'heldout')
        gap=set().union(*(cells(b) for b in row['missing']))
        if len(gap)<4: continue
        target={'x':min(c[0] for c in gap),'y':min(c[1] for c in gap),'bottom':min(c[2] for c in gap),'top':max(c[2] for c in gap)+1,'cross_section':[1,1]}
        key=json.dumps(target,sort_keys=True)
        if key in seen: continue
        row['id']='interactive-'+row['id'];row['task']['id']=row['id']; row.pop('prompt')
        row['target']=target; row['editable_initial']=[]
        row['mode']='correct_misplaced' if i%4==0 else 'build_missing'
        if row['mode']=='correct_misplaced':
            row['editable_initial']=[{'id':'repair_seed','part':'brick_1x1','x':target['x']+1,'y':target['y'],'z':target['bottom'],'rotation':0}]
        rows.append(row);seen.update([sig,key])
    for split in ['train','heldout']:write(path/f'{split}.json',[r for r in rows if r['split']==split])
    write(path/'manifest.json',{'seed':20260928,'train':80,'heldout':20,'turns':TURNS,'kind':'interactive_single_upright_repair',
         'scope':'Synthetic bridge repairs; includes misplaced editable starter bricks. Not whole-bridge generation.',
         'sha256':{s:hashlib.sha256((path/f'{s}.json').read_bytes()).hexdigest() for s in ['train','heldout']}})


def actions(text,locked):
    value=json.loads(text)
    if not isinstance(value,dict) or set(value)!={'actions'}: raise ValueError('Return a JSON actions array')
    result=value['actions']
    if not isinstance(result,list) or not 1<=len(result)<=4: raise ValueError('Use 1 to 4 actions per turn')
    for a in result:
        if not isinstance(a,dict): raise ValueError('Action must be an object')
        if a.get('id') in locked or a.get('brick',{}).get('id') in locked: raise ValueError('Existing bridge pieces are locked')
        if a.get('tool')=='place' and a.get('brick',{}).get('part') not in ['brick_1x1','plate_1x1']: raise ValueError('Use the two supplied upright parts')
    return result


def measure(row,bricks):
    fixed={b['id'] for b in row['initial']}; editable=[b for b in bricks if b['id'] not in fixed]
    target=set().union(*(cells(b) for b in row['missing']))
    occupied=set().union(*(cells(b) for b in editable)) if editable else set()
    full=evaluate(bricks,row['task']);t=row['target']
    correct=occupied&target; extra=occupied-target
    coverage=len(correct)/len(target); precision=len(correct)/len(occupied) if occupied else 0.
    def alignment(axis,want):
        return max(0.,1-sum(abs(c[axis]-want) for c in occupied)/len(occupied)/8) if occupied else 0.
    top=max((c[2]+1 for c in occupied if c[0]==t['x'] and c[1]==t['y']),default=t['bottom'])
    height=max(0.,1-abs(top-t['top'])/(t['top']-t['bottom'])) if occupied else 0.
    connected=full['checks']['one_connected_assembly']; collision_free=full['collision_cells']==0
    v={'coverage':coverage,'material_precision':precision,'x_alignment':alignment(0,t['x']),
       'y_alignment':alignment(1,t['y']),'height_match':height,'connected':float(connected),'collision_free':float(collision_free)}
    potential=.45*coverage+.15*precision+.1*v['x_alignment']+.1*v['y_alignment']+.1*height+.1*float(full['success'])
    if not connected or not collision_free:potential*=.1
    return {'reward_vector':v,'potential':potential,'success':full['success'],'editable_bricks':editable,
            'missing_cells':len(target-correct),'extra_cells':len(extra),'collision_cells':full['collision_cells'],
            'current_top_at_target':top,'required_top':t['top'],'issues':full['issues'],
            'gap_iou':len(correct)/len(occupied|target)}


def observation(row,m,turn,error=None):
    return {'target':row['target'],'turns_used':turn,'turns_remaining':TURNS-turn,
            'fixed_bridge':'All other bridge geometry is locked and complete. The target starts on an existing upright.',
            **m,'last_action_error':error,'done':m['success'] or turn>=TURNS}


def replay(row,history,traces,sandbox=True):
    if not isinstance(history,list) or len(history)>TURNS or any(not isinstance(x,str) for x in history): raise ValueError('At most eight text turns')
    fixed={b['id'] for b in row['initial']}; bricks=copy.deepcopy(row['initial']+row['editable_initial'])
    env=None
    try:
        if sandbox:
            env=BrickEnvironment(traces,task=row['task'])
            env.step([{'tool':'place','brick':b} for b in bricks],actor='frozen_initial_state')
        initial=measure(row,bricks); previous=initial; transitions=[]
        for i,text in enumerate(history,1):
            if previous['success']:raise ValueError('Episode already succeeded')
            error=None
            try:
                act=actions(text,fixed); candidate=apply(bricks,act,max_bricks=256)
            except (ValueError,KeyError,TypeError,AttributeError) as exc:
                error=str(exc);act=[];candidate=bricks
            if env and not error:
                obs=env.step(act,actor='qwen_interactive')
                if obs.get('tool_error'): raise RuntimeError('Validated action failed inside sandbox: '+obs['tool_error'])
                candidate=obs['bricks']
            bricks=candidate;m=measure(row,bricks)
            reward=m['potential']-previous['potential']+.5*float(m['success'])
            transitions.append({'turn':i,'actions':act,'error':error,'reward':reward,'before':previous['potential'],'after':m['potential'],
                                'before_extra_cells':previous['extra_cells'],'after_extra_cells':m['extra_cells']})
            previous=m
        return {'observation':observation(row,previous,len(history),transitions[-1]['error'] if transitions else None),
                'step_reward':transitions[-1]['reward'] if transitions else 0.,
                'return':previous['potential']-initial['potential']+.5*float(previous['success']),
                'transitions':transitions,'sandbox_episode':env.episode if env else None}
    finally:
        if env:env.close()


def serve(dataset,output,token_file,port):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);token=Path(token_file).read_text().strip()
    rows={r['id']:r for split in ['train','heldout'] for r in json.loads((Path(dataset)/f'{split}.json').read_text())}
    lock=threading.Lock();slots=threading.Semaphore(2)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            if self.path!='/step' or not hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+token):self.send_error(403);return
            try:
                n=int(self.headers.get('Content-Length','0'))
                if not 0<n<100000:raise ValueError('Invalid request length')
                req=json.loads(self.rfile.read(n));row=rows[req['id']];phase=req['phase']
                if phase not in ['train','quick','before','after','preflight']:raise ValueError('Unknown phase')
                if phase in ['train','quick'] and row['split']!='train':raise ValueError('Held-out leakage blocked')
                if phase in ['before','after'] and row['split']!='heldout':raise ValueError('Evaluation requires heldout')
                with slots:result=replay(row,req['history'],output/'episodes')
                with lock:
                    with (output/'requests.jsonl').open('a') as f:f.write(json.dumps({**req,'result':result})+'\n')
                self.send_response(200);self.end_headers();self.wfile.write(json.dumps(result).encode())
            except Exception as exc:self.send_response(500);self.end_headers();self.wfile.write(json.dumps({'error':str(exc)}).encode())
    write(output/'ready.json',{'port':port,'manifest':json.loads((Path(dataset)/'manifest.json').read_text())})
    ThreadingHTTPServer(('127.0.0.1',port),Handler).serve_forever()


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['generate','serve']);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--old-dataset',type=Path);p.add_argument('--output',type=Path);p.add_argument('--token-file',type=Path);p.add_argument('--port',type=int,default=8769)
    a=p.parse_args()
    if a.command=='generate':generate(a.dataset,a.old_dataset)
    else:serve(a.dataset,a.output,a.token_file,a.port)


if __name__=='__main__':main()
