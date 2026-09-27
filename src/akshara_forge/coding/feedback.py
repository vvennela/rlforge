"""Separate development feedback from sealed scoring tests for coding repair."""
import argparse,copy,hashlib,hmac,itertools,json,random,threading,time
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from .tasks import SYSTEM,prompt,cases,oracle
from .service import execute,code_from,compare_results
from ..engineering.pilot import write

REPAIR_INSTRUCTION='You have three attempts. After each attempt you receive execution feedback on development examples. Use it to fix the implementation. Return the full replacement Python program defining solve(problem), not a patch. Final correctness is scored on separate private inputs; do not hardcode the examples.'

def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'))
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def generate(source,dest):
    from .adversarial import independent,mutant_solvers
    if dest.exists():raise ValueError('Use a fresh immutable dataset directory')
    training=json.loads((source/'train.json').read_text());oldheld=json.loads((source/'heldout.json').read_text())
    used={canonical(r['configuration']) for r in training+oldheld}
    candidates=[{'order':o,'root':r,'cutoffs':c,'goal':g,'initial':i,'fields':f} for o,r,c,g,i,f in itertools.product(['given','ascending','descending'],[False,True],[False,True],[False,True],['zero','heuristic'],[['path','cost','bounds','visits'],['path','bounds','counts'],['cost','bounds','visits','counts']])]
    candidates=[c for c in candidates if canonical(c) not in used];random.Random(20260930).shuffle(candidates)
    attacks=[t['input'] for t in training[0]['tests'][-8:]]
    held=[]
    for i,c in enumerate(candidates[:20]):
        inputs=cases(720000+i)+attacks
        held.append({'id':f'feedback-heldout-{i:03}','split':'heldout','configuration':c,'tests':[{'input':x,'expected':oracle(x,c)} for x in inputs]})
    mutations=mutant_solvers();killed={s:set() for s in ['train','heldout']}
    for split,rows in [('train',training),('heldout',held)]:
        for i,row in enumerate(rows):
            c=row['configuration'];row['prompt']=[{'role':'system','content':SYSTEM+' '+REPAIR_INSTRUCTION},{'role':'user','content':prompt(c)}]
            seen={canonical(t['input']) for t in row['tests']};dev=[];seed=810000+i+(1000 if split=='heldout' else 0)
            while len(dev)<6:
                for x in cases(seed)[4:]:
                    if canonical(x) in seen:continue
                    dev.append({'input':x,'expected':oracle(x,c)});seen.add(canonical(x))
                    if len(dev)==6:break
                seed+=10000
            row['development_tests']=dev
            for t in row['tests']+dev:
                if oracle(t['input'],c)!=t['expected'] or independent(t['input'],c)!=t['expected']:raise ValueError('Reference disagreement')
            # The strict suite must still reject each known wrong algorithm.
            for name,fn in mutations.items():
                if name in killed[split]:continue
                for t in row['tests']:
                    try:wrong=fn(t['input'],c)!=t['expected']
                    except Exception:wrong=True
                    if wrong:killed[split].add(name);break
    assert len(held)==20 and all(len(k)==8 for k in killed.values())
    dest.mkdir(parents=True)
    for split,rows in [('train',training),('heldout',held)]:write(dest/f'{split}.json',rows)
    write(dest/'manifest.json',{'kind':'three_turn_coding_repair','train':80,'heldout':20,'development_tests':6,'private_tests':24,'turns':3,'seed':20260930,'parent_sha256':digest(source/'manifest.json'),'sha256':{s:digest(dest/f'{s}.json') for s in ['train','heldout']},'fresh_holdout':'Twenty configurations excluded from both earlier splits. New seeded private inputs plus the training-source adversarial graph patterns.','feedback_policy':'Development inputs/expected/actual only. Private test outputs never enter model context.','reference_agreement':True,'mutations_rejected':{k:sorted(v) for k,v in killed.items()}})

def feedback(row,completion):
    dev={'tests':row['development_tests']}
    try:out=execute(code_from(completion),[t['input'] for t in dev['tests']])
    except ValueError as e:out={'error':str(e)}
    graded=compare_results(dev,out,'components');results=out.get('results',[]) if isinstance(out,dict) else []
    mismatches=[]
    for i,(test,passed) in enumerate(zip(dev['tests'],graded['checks'])):
        if passed:continue
        item={'input':test['input'],'expected':test['expected'],'actual':results[i] if isinstance(results,list) and i<len(results) else None}
        if len(canonical(item))>5000:
            item={'input':test['input'],'expected_fields':list(test['expected']),'field_matches':graded['field_checks'][i],'note':'Large trace omitted; field matches shown.'}
        mismatches.append(item)
        if len(mismatches)==2:break
    return {'development_score':graded['reward'],'development_success':graded['success'],'feedback':{'suite':'development examples; separate from final private tests','passed':graded['passed'],'total':graded['total'],'field_accuracy':graded['field_score'],'error':graded['error'],'failed_examples':mismatches,'instruction':'Fix the errors and return the full replacement solve(problem) program. Implement the specification for arbitrary inputs.'}}

def serve(dataset,output,token_file,port):
    token=token_file.read_text().strip();output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((dataset/'manifest.json').read_text())
    for s,d in manifest['sha256'].items():
        if digest(dataset/f'{s}.json')!=d:raise ValueError('Dataset changed')
    rows={r['id']:r for s in ['train','heldout'] for r in json.loads((dataset/f'{s}.json').read_text())}
    slots=threading.Semaphore(2);lock=threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            if self.path not in ['/feedback','/score'] or not hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+token):self.send_error(403);return
            req=None
            try:
                n=int(self.headers.get('Content-Length','0'))
                if not 0<n<100000:raise ValueError('Invalid size')
                req=json.loads(self.rfile.read(n));row=rows[req['id']];phase=req['phase']
                if phase not in ['train','before','after','preflight']:raise ValueError('Unknown phase')
                if phase=='train' and row['split']!='train':raise ValueError('Heldout leakage blocked')
                if phase in ['before','after'] and row['split']!='heldout':raise ValueError('Wrong split')
                with slots:
                    if self.path=='/feedback':result=feedback(row,req['completion'])
                    else:
                        try:out=execute(code_from(req['completion']),[t['input'] for t in row['tests']])
                        except ValueError as e:out={'error':str(e)}
                        result=compare_results(row,out,'components' if phase=='train' else 'tests')
                result['manifest_sha256']=digest(dataset/'manifest.json')
                with lock:
                    with (output/'requests.jsonl').open('a') as f:f.write(json.dumps({'time':time.time(),'endpoint':self.path,**req,'result':result})+'\n')
                raw=json.dumps(result).encode();self.send_response(200);self.end_headers();self.wfile.write(raw)
            except Exception as e:
                with lock:
                    with (output/'errors.jsonl').open('a') as f:f.write(json.dumps({'request':req,'error':str(e)})+'\n')
                self.send_response(400);self.end_headers();self.wfile.write(json.dumps({'error':str(e)}).encode())
    write(output/'ready.json',{'manifest_sha256':digest(dataset/'manifest.json'),'port':port})
    ThreadingHTTPServer(('127.0.0.1',port),Handler).serve_forever()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['generate','serve']);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--source',type=Path);p.add_argument('--output',type=Path);p.add_argument('--token-file',type=Path);p.add_argument('--port',type=int,default=8775);p.add_argument('--legacy-flat',action='store_true',help='Reproduce the old full-algorithm-only dataset');a=p.parse_args()
    if a.command=='generate':
        if a.legacy_flat:generate(a.source,a.dataset)
        else:
            from .curriculum import generate as generate_curriculum
            generate_curriculum(a.dataset,a.source)
    else:serve(a.dataset,a.output,a.token_file,a.port)
