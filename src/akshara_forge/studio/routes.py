import json,os
import httpx
from pathlib import Path
from urllib.parse import urlsplit
from .generation import GenerationJobs,config
from ..engineering.environment import BRIDGE,reference
from ..engineering.worker import PALETTE

class Studio:
    def __init__(self,runs):
        self.assets=Path(__file__).parent
        self.runs=Path(runs)
        os.environ.setdefault('AKSHARA_INFERENCE_CONFIG',str(self.runs/'inference.json'))
        self.jobs=GenerationJobs(self.runs/'generation')
    def get(self,h):
        path=urlsplit(h.path).path
        if path=='/':h.send((self.assets/'index.html').read_bytes(),ctype='text/html; charset=utf-8');return True
        if path.startswith('/studio/'):
            target=(self.assets/path.removeprefix('/studio/')).resolve()
            if not target.is_relative_to(self.assets.resolve()) or not target.is_file() or target.suffix not in ('.css','.js','.woff2','.woff','.ttf'):h.send({'error':'Not found'},404);return True
            types={'.css':'text/css','.js':'text/javascript','.woff2':'font/woff2','.woff':'font/woff','.ttf':'font/ttf'}
            h.send(target.read_bytes(),ctype=types.get(target.suffix,'application/octet-stream'));return True
        if path=='/api/models':
            try:
                r=httpx.get('https://api.vultrinference.com/v1/models',timeout=15);r.raise_for_status()
                h.send({'models':[{'id':m['id'],'name':m.get('name',m['id'])} for m in r.json()['data'] if any(o['type']=='text' for o in m.get('output_modalities',[]))]})
            except Exception:h.send({'error':'Could not load Vultr model catalog'},502)
            return True
        if path=='/api/studio':
            file=Path(os.getenv('AKSHARA_CASE_STUDIES',str(self.runs/'case-studies.json')))
            data=json.loads(file.read_text()) if file.exists() else {'model':'Qwen 2.5 · 7B parameters'}
            h.send({'evidence':data,'generation':config()});return True
        if path=='/api/bridge-sample':
            h.send({'task':BRIDGE,'bricks':reference(),'palette':PALETTE,'label':'Reference assembly'});return True
        if path.startswith('/api/generation/'):
            parts=path.strip('/').split('/');id=parts[2]
            try:
                status=self.jobs.read(id)
                if len(parts)==4 and parts[3]=='download':
                    if status['status']!='ready':h.send({'error':'Environment is not ready'},409)
                    else:h.send((self.jobs.root/id/'environment.zip').read_bytes(),ctype='application/zip')
                else:h.send(status)
            except (ValueError,FileNotFoundError):h.send({'error':'Job not found'},404)
            return True
        return False
    def post(self,h,req):
        if h.path=='/api/provider':
            try:
                if self.jobs.busy:raise ValueError('Wait for current generation to finish before changing providers.')
                key=req.get('key','');model=req.get('model','')
                if not isinstance(key,str) or not 16<=len(key)<=512:raise ValueError('Enter your Vultr Serverless Inference API key.')
                r=httpx.get('https://api.vultrinference.com/v1/models',timeout=15);r.raise_for_status()
                if model not in [m['id'] for m in r.json()['data'] if any(o['type']=='text' for o in m.get('output_modalities',[]))]:raise ValueError('Choose a model from the Vultr catalog.')
                p=Path(os.environ['AKSHARA_INFERENCE_CONFIG']);p.parent.mkdir(parents=True,exist_ok=True)
                fd=os.open(str(p),os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
                with os.fdopen(fd,'w') as f:json.dump({'provider':'vultr','model':model,'key':key},f)
                os.chmod(p,0o600);h.send(config())
            except ValueError as exc:h.send({'error':str(exc)},400)
            except Exception:h.send({'error':'Provider connection could not be saved.'},502)
            return True
        if h.path=='/api/generate':
            try:h.send(self.jobs.start(req),202)
            except (ValueError,TypeError) as e:h.send({'error':str(e)},400)
            return True
        return False
