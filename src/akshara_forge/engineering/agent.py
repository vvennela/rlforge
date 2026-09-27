"""Ollama tool agent: inference-time repair, never claimed as RL weight training."""
import json,time,hashlib
from pathlib import Path
import httpx

SYSTEM='''You control a brick assembly environment. Return JSON only: {"actions":[...]}. Available actions: {"tool":"move","id":"example_brick","x":4,"y":0,"z":0,"rotation":0}, {"tool":"remove","id":"example_brick"}, or {"tool":"place","brick":{"id":"new1","part":"plate_2x2","x":0,"y":0,"z":6,"rotation":0}}. Coordinates x,y are studs; z is plate heights. Use only the supplied palette, integer coordinates and rotations 0/90. The goal is all geometry checks passing without collisions. Inspect the actual bricks and reward diagnostics. Make minimal corrections. Do not output explanatory prose. At most 32 actions per response.'''

def repair(env,output:Path,model='qwen2.5:7b-instruct',rounds=3):
 output=Path(output);output.mkdir(parents=True,exist_ok=True)
 if not 1<=rounds<=8:raise ValueError('1..8 rounds')
 receipt={'agent_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'model':model,'kind':'inference_time_repair','weights_updated':False,'rounds_limit':rounds,'episode':env.episode,'temperature':0,'num_predict':1024,'num_ctx':8192,'initial_score':env.observe()['evaluation']}
 with httpx.Client(timeout=180) as c:
  receipt['model_inventory']=c.get('http://127.0.0.1:11434/api/tags').json()
  (output/'receipt.json').write_text(json.dumps(receipt,indent=2))
  for n in range(rounds):
   obs=env.observe()
   if obs['done'] or obs['evaluation']['success']:break
   messages=[{'role':'system','content':SYSTEM},{'role':'user','content':json.dumps(obs)}]
   request={'model':model,'stream':False,'format':'json','messages':messages,'options':{'temperature':0,'num_predict':1024,'num_ctx':8192}}
   (output/f'{n:02}-request.json').write_text(json.dumps(request,indent=2))
   response=c.post('http://127.0.0.1:11434/api/chat',json=request);response.raise_for_status();raw=response.json()
   (output/f'{n:02}-response.json').write_text(json.dumps(raw,indent=2))
   try:
    parsed=json.loads(raw['message']['content']);actions=parsed['actions']
    if not isinstance(actions,list) or len(actions)>32:raise ValueError('At most 32 actions')
    result=env.step(actions,actor="model:"+model)
   except (ValueError,KeyError) as exc:
    (output/f'{n:02}-error.json').write_text(json.dumps({'error':str(exc)}));continue
   (output/f'{n:02}-result.json').write_text(json.dumps(result,indent=2))
 final=env.observe()['evaluation'];(output/'summary.json').write_text(json.dumps({'kind':'inference_time_repair','weights_updated':False,'initial':receipt['initial_score'],'final':final},indent=2))
 return final
