"""Loopback-only competition prototype; no cloud deployment implied."""
import argparse,json,threading,time
from pathlib import Path
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from .environment import BRIDGE,BrickEnvironment,reference,cells
from .task_library import LIBRARY,targets
from .agent import repair,configuration
from ..studio.routes import Studio

def main():
 p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8787);p.add_argument('--runs',type=Path,required=True);p.add_argument('--drawing',type=Path,required=True);a=p.parse_args()
 if not a.drawing.is_file():p.error('Drawing file missing')
 studio=Studio(a.runs)
 tasks={'bridge':BRIDGE,**LIBRARY};intake=a.drawing.resolve().parent.parent
 lock=threading.RLock();env=BrickEnvironment(a.runs/'episodes');state={'mode':'empty workspace','busy':False,'error':None,'agent':configuration()}
 class Handler(BaseHTTPRequestHandler):
  def log_message(self,*args):pass
  def send(self,data,status=200,ctype='application/json'):
   raw=data if isinstance(data,bytes) else json.dumps(data).encode();self.send_response(status);self.send_header('Content-Type',ctype);self.send_header('Content-Length',str(len(raw)));self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(raw)
  def allowed(self):return self.headers.get('Host') in [f'127.0.0.1:{a.port}',f'localhost:{a.port}']
  def do_GET(self):
   if not self.allowed():return self.send({'error':'Invalid host'},403)
   if studio.get(self):return
   if self.path=='/workbench':return self.send(Path(__file__).with_name('demo.html').read_bytes(),ctype='text/html; charset=utf-8')
   if self.path=='/renderer.js':return self.send(Path(__file__).with_name('renderer.js').read_bytes(),ctype='text/javascript; charset=utf-8')
   if self.path=='/viewer.js':return self.send(Path(__file__).with_name('viewer.js').read_bytes(),ctype='text/javascript; charset=utf-8')
   if self.path.startswith('/drawing'):
    key=self.path.partition('?task=')[2] or 'bridge'
    if key not in tasks:return self.send({'error':'Unknown task'},404)
    drawing=a.drawing if key=='bridge' else intake/tasks[key]['source_id']/'preview.jpg'
    return self.send(drawing.read_bytes(),ctype='image/jpeg')
   if self.path=='/api/state':
    with lock:return self.send({**env.observe(),**state})
   self.send({'error':'Not found'},404)
  def do_POST(self):
   nonlocal env
   if not self.allowed() or self.headers.get('Origin') not in [None,f'http://127.0.0.1:{a.port}',f'http://localhost:{a.port}']:return self.send({'error':'Invalid origin'},403)
   if self.headers.get('Content-Type')!='application/json':return self.send({'error':'JSON required'},415)
   try:
    size=int(self.headers.get('Content-Length','0'))
    if not 0<size<=(12*1024*1024 if self.path=='/api/generate' else 100000):raise ValueError('Invalid request size')
    req=json.loads(self.rfile.read(size))
    if studio.post(self,req):return
    with lock:
     if state['busy']:return self.send({'error':'Agent running; wait for it to finish'},409)
     if self.path=='/api/reset':
      mode=req.get('mode','empty');key=req.get('task','bridge')
      if key not in tasks:raise ValueError('Unknown task')
      if mode not in ['empty','damaged','reference','missing_uprights']:raise ValueError('Unknown fixture')
      env.close();env=BrickEnvironment(a.runs/'episodes',task=tasks[key]);state.update(mode=mode+' — scripted fixture' if mode!='empty' else 'empty workspace',error=None)
      if mode!='empty':
       bricks=reference(env.task)
       if key!='bridge' and mode in ['missing_uprights','damaged']:
        missing=next(c['cells'] for c in targets(env.task) if c['id']==env.task['fault_component'])
        bricks=[b for b in bricks if not cells(b)&missing]
       if key=='bridge' and mode=='missing_uprights':bricks=[b for b in bricks if b['z']<env.task['grid']['deck_z']+2]
       if key=='bridge' and mode=='damaged':
        for b in bricks:
         if b['id'] in ['b2','b3','b5']:b['x']-=4
       env.step([{'tool':'place','brick':b} for b in bricks],actor='scripted_fixture')
     elif self.path=='/api/step':env.step(req['actions'],actor='manual_ui');state['mode']='manual tool actions'
     elif self.path=='/api/agent':
      agent=configuration()
      if not agent['configured']:raise ValueError('Configure the Vultr Serverless Inference API key before running an agent')
      if env.done:raise ValueError('Load the damaged fixture or reset before running the agent')
      state.update(busy=True,mode=agent['model']+' · inference-time repair',error=None)
      def run():
       try:repair(env,a.runs/'agents'/env.episode)
       except Exception as exc:state['error']=str(exc)[:300]
       finally:state['busy']=False
      threading.Thread(target=run,daemon=True).start()
     else:return self.send({'error':'Not found'},404)
     self.send({**env.observe(),**state})
   except Exception as exc:self.send({'error':str(exc)[:300]},400)
 server=ThreadingHTTPServer(('127.0.0.1',a.port),Handler)
 print(f'Engineering environment: http://127.0.0.1:{a.port}',flush=True)
 try:server.serve_forever()
 finally:server.server_close();env.close()

if __name__=='__main__':main()
