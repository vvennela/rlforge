"""Private tests stay on the Vultr controller; candidate code runs in gVisor."""
import argparse, hashlib, hmac, json, re, select, subprocess, threading, time, traceback, uuid
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer

def code_from(text):
    blocks=re.findall(r'```(?:python|py)?\s*\n(.*?)```',text,re.S)
    code=blocks[-1] if blocks else text.strip()
    if len(code)>24000:raise ValueError('Code too long')
    return code

def execute(code,inputs,timeout=30):
    name='forge-code-'+uuid.uuid4().hex[:12];worker=Path(__file__).with_name('worker.py').resolve()
    command=['docker','run','--rm','-i','--name',name,'--runtime=runsc','--network=none','--read-only','--cap-drop=ALL','--security-opt=no-new-privileges','--pids-limit=32','--memory=256m','--cpus=.5','--user=65534:65534','--mount',f'type=bind,source={worker},target=/worker.py,readonly','python:3.12-slim','python','-I','-B','/worker.py']
    proc=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
    try:
        proc.stdin.write(json.dumps({'code':code,'inputs':inputs}).encode()+b'\n');proc.stdin.close()
        start=time.monotonic();data=b''
        while time.monotonic()-start<timeout:
            if select.select([proc.stdout],[],[],.2)[0]:
                chunk=__import__('os').read(proc.stdout.fileno(),8192)
                if not chunk:break
                data+=chunk
                if len(data)>65536:return {'error':'OutputLimit'}
            if proc.poll() is not None:break
        else:return {'error':'Timeout'}
        try:return json.loads(data)
        except (ValueError,UnicodeDecodeError):return {'error':'InvalidOutput'}
    finally:
        subprocess.run(['docker','rm','-f',name],capture_output=True,timeout=20)
        if proc.poll() is None:proc.kill()
        proc.wait(timeout=5)

def grade(row,completion,training_reward='tests'):
    if training_reward not in ('tests','components'):raise ValueError('Unknown reward mode')
    try:out=execute(code_from(completion),[t['input'] for t in row['tests']])
    except (ValueError,SyntaxError) as exc:out={'error':type(exc).__name__}
    if not isinstance(out,dict):out={'error':'InvalidEnvelope'}
    results=out.get('results',[])
    if not isinstance(results,list) or len(results)!=len(row['tests']):
        out={'error':out.get('error','InvalidResultCount')};results=[]
    checks=[];field_checks=[]
    for i,test in enumerate(row['tests']):
        got=results[i] if i<len(results) else {}
        try:
            ok=isinstance(got,dict) and set(got)=={'value'} and json.dumps(got['value'],sort_keys=True,allow_nan=False)==json.dumps(test['expected'],sort_keys=True,allow_nan=False)
        except (ValueError,TypeError,OverflowError):ok=False
        checks.append(ok)
        expected=test['expected'];value=got.get('value') if isinstance(got,dict) and set(got)=={'value'} else None
        if not isinstance(expected,dict):
            field_checks.append({'value':ok});continue
        schema_ok=isinstance(value,dict) and set(value)==set(expected)
        fields={}
        for key,want in expected.items():
            try:fields[key]=schema_ok and json.dumps(value[key],sort_keys=True,allow_nan=False)==json.dumps(want,sort_keys=True,allow_nan=False)
            except (ValueError,TypeError,OverflowError):fields[key]=False
        field_checks.append(fields)
    field_total=sum(len(c) for c in field_checks)
    field_score=sum(sum(c.values()) for c in field_checks)/field_total
    test_score=sum(checks)/len(checks)
    reward=test_score if training_reward=='tests' else .5*test_score+.5*field_score
    return {'reward':reward,'reward_mode':training_reward,'field_score':field_score,'field_checks':field_checks,'test_score':test_score,'success':all(checks),'passed':sum(checks),'total':len(checks),'checks':checks,'error':out.get('error'),'sandbox':'gVisor / no network / read-only / unprivileged','wall_timeout_seconds':30}

def serve(dataset,output,token_file,port,training_reward='tests'):
    token=token_file.read_text().strip();output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((dataset/'manifest.json').read_text())
    manifest_hash=hashlib.sha256((dataset/'manifest.json').read_bytes()).hexdigest()
    for split,digest in manifest['sha256'].items():
        if hashlib.sha256((dataset/f'{split}.json').read_bytes()).hexdigest()!=digest:raise ValueError('Dataset hash changed')
    rows={r['id']:r for split in ('train','heldout') for r in json.loads((dataset/f'{split}.json').read_text())};slots=threading.Semaphore(2);lock=threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            if self.path!='/grade' or not hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+token):self.send_error(403);return
            req=None
            try:
                n=int(self.headers.get('Content-Length','0'))
                if not 0<n<100000:raise ValueError('Invalid size')
                req=json.loads(self.rfile.read(n));row=rows[req['id']];phase=req['phase']
                if phase not in ('train','before','after','preflight'):raise ValueError('Unknown phase')
                if phase=='train' and row['split']!='train':raise ValueError('Heldout leakage blocked')
                if phase in ('before','after') and row['split']!='heldout':raise ValueError('Wrong evaluation split')
                with slots:result=grade(row,req['completion'],training_reward if phase=='train' else 'tests')
                result['dataset_manifest_sha256']=manifest_hash
                with lock:
                    with (output/'requests.jsonl').open('a') as f:f.write(json.dumps({'time':time.time(),**req,'result':result})+'\n')
                raw=json.dumps(result).encode();self.send_response(200);self.end_headers();self.wfile.write(raw)
            except Exception as exc:
                with lock:
                    with (output/'errors.jsonl').open('a') as f:
                        f.write(json.dumps({'time':time.time(),'request':req,'error':str(exc),'traceback':traceback.format_exc()})+'\n')
                self.send_response(400);self.end_headers();self.wfile.write(json.dumps({'error':str(exc)[:300]}).encode())
    ThreadingHTTPServer(('127.0.0.1',port),Handler).serve_forever()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--token-file',type=Path,required=True);p.add_argument('--port',type=int,default=8771);p.add_argument('--training-reward',choices=['tests','components'],default='tests');a=p.parse_args();serve(a.dataset,a.output,a.token_file,a.port,a.training_reward)
