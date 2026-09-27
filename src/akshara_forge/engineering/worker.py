"""Untrusted assembly workspace. No verifier, source archive, or host mounts."""
import json
import sys

PALETTE = {
 'brick_1x2':[1,2,3], 'brick_1x4':[1,4,3], 'brick_2x4':[2,4,3], 'brick_2x8':[2,8,3],
 'brick_1x1': [1,1,3], 'plate_1x1': [1,1,1], 'plate_2x6': [2,6,1],
 'brick_2x2': [2,2,3], 'brick_2x6': [2,6,3],
 'plate_1x2': [1,2,1], 'plate_1x4': [1,4,1],
 'plate_2x2': [2,2,1], 'plate_2x4': [2,4,1], 'plate_2x8': [2,8,1],
}

def validate(b):
 if not isinstance(b,dict) or set(b)!={'id','part','x','y','z','rotation'}: raise ValueError('Brick requires id, part, x, y, z, rotation only')
 if not isinstance(b['id'],str) or not b['id'].isascii() or not b['id'].replace('_','').isalnum() or len(b['id'])>40: raise ValueError('Invalid brick id')
 if b['part'] not in PALETTE: raise ValueError('Unknown part')
 if any(type(b[k]) is not int for k in ['x','y','z','rotation']): raise ValueError('Coordinates and rotation must be integers')
 if b['rotation'] not in [0,90]: raise ValueError('Rotation must be 0 or 90')
 dx,dy,dz=PALETTE[b['part']]
 if b['rotation']==90:dx,dy=dy,dx
 if not (0<=b['x']<=64-dx and 0<=b['y']<=24-dy and 0<=b['z']<=64-dz): raise ValueError('Outside 64 x 24 stud, 64 plate-height workspace')
 return dx,dy,dz

def apply(state,actions,max_bricks=512):
 if type(max_bricks) is not int or not 1<=max_bricks<=512:raise ValueError("Invalid brick limit")
 if not isinstance(actions,list) or len(actions)>512: raise ValueError('At most 512 actions per transaction')
 result={b['id']:dict(b) for b in state}
 for a in actions:
  if not isinstance(a,dict):raise ValueError('Action must be an object')
  op=a.get('tool')
  if op=='place':
   if set(a)!={'tool','brick'}:raise ValueError('place requires brick')
   b=a['brick'];validate(b)
   if b['id'] in result:raise ValueError('Duplicate brick id')
   result[b['id']]=dict(b)
  elif op=='move':
   if set(a)!={'tool','id','x','y','z','rotation'}:raise ValueError('move requires id, x, y, z, rotation')
   b=dict(result[a['id']]);b.update({k:a[k] for k in ['x','y','z','rotation']});validate(b);result[a['id']]=b
  elif op=='remove':
   if set(a)!={'tool','id'}:raise ValueError('remove requires id')
   del result[a['id']]
  else:raise ValueError('Unknown tool; use place, move, remove')
  if len(result)>max_bricks:raise ValueError(f'{max_bricks} brick limit')
 return list(result.values())

if __name__=='__main__':
 state=[]
 for line in sys.stdin:
  try:
   if len(line)>100000:raise ValueError('Request too large')
   req=json.loads(line);new=apply(state,req['actions'],req.get('max_bricks',512));state=new
   print(json.dumps({'ok':True,'bricks':state}),flush=True)
  except Exception as exc:print(json.dumps({'ok':False,'error':str(exc)[:250],'bricks':state}),flush=True)
