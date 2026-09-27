"""Loopback reward bridge for a trusted GPU trainer through an SSH reverse tunnel.

Only frozen training IDs are accepted. Codex credentials remain on the local
controller; the remote machine receives a separate short-lived bridge token.
"""
from __future__ import annotations
import argparse
import hmac
import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from threading import BoundedSemaphore
from .io import write_json
from .progress import fingerprint,prepare_contract,judge_progress


def remote_reward(problem: dict, action: dict, output: Path) -> dict:
    url=os.environ['AKSHARA_REWARD_URL'].rstrip('/')+'/grade'
    token=Path(os.environ['AKSHARA_REWARD_TOKEN_FILE']).read_text().strip()
    packet={'problem_id':problem['problem_id'],'problem_sha256':fingerprint(problem),'action':action}
    req=urllib.request.Request(url,data=json.dumps(packet).encode(),headers={
        'Content-Type':'application/json','Authorization':'Bearer '+token})
    with urllib.request.urlopen(req,timeout=900) as response:result=json.load(response)
    write_json(output/'result.json',result)
    return result


def serve(root: Path,paper: str,output: Path,token_file: Path,port: int=8766):
    from .training import training_rows
    _,problems=training_rows(root,paper)
    def prepare(item):
        key,problem=item
        c=prepare_contract(problem,output/'contracts'/key)
        print('contract',key,flush=True)
        return key,c
    with ThreadPoolExecutor(max_workers=2) as pool:contracts=dict(pool.map(prepare,problems.items()))
    token=token_file.read_text().strip()
    if len(token)<32:raise ValueError('Bridge token must have at least 32 random characters')
    slots=BoundedSemaphore(4)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            if not hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+token):
                self.send_error(401);return
            if self.path!='/grade':self.send_error(404);return
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=131072:raise ValueError('Invalid payload length')
                data=json.loads(self.rfile.read(length));p=problems[data['problem_id']]
                if data['problem_sha256']!=fingerprint(p):raise ValueError('Frozen dataset mismatch')
                action=data['action']
                if not isinstance(action,dict) or 'answer' not in action:raise ValueError('Invalid action')
                steps=action.get('steps',[])
                if not isinstance(steps,list) or len(steps)>256 or not all(isinstance(s,str) for s in steps):
                    raise ValueError('Invalid written steps')
                with slots:
                    result=judge_progress(p,action,output/'judgments'/fingerprint(data),contracts[p['problem_id']])
                body=json.dumps(result,allow_nan=False).encode()
                self.send_response(200);self.send_header('Content-Type','application/json')
                self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
            except Exception as exc:
                write_json(output/'last-error.json',{'error':str(exc)})
                self.send_error(422,'Reward evaluation failed; inspect controller logs')
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    write_json(output/'ready.json',{'paper':paper,'training_problems':len(problems),'port':port,
                                   'contract_hashes':{k:fingerprint(v) for k,v in contracts.items()}})
    print('READY loopback reward service',port,flush=True)
    server.serve_forever()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--paper',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--token-file',type=Path,required=True)
    p.add_argument('--port',type=int,default=8766)
    a=p.parse_args();serve(a.root,a.paper,a.output,a.token_file,a.port)


if __name__=='__main__':main()
