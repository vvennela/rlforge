"""Reviewed teaching abstractions, not automatic or complete CAD reconstructions."""
from functools import lru_cache
import json
from .worker import PALETTE


def box(x,y,z,w,d,h):return {'box':[x,y,z,w,d,h]}
def cylinder(cx,cy,z,r,h,inner=0):return {'cylinder':[cx,cy,z,r,h,inner]}
def wheel(cx,y,cz,r,depth):return {'wheel':[cx,y,cz,r,depth]}

def shape_cells(shape):
 if 'box' in shape:
  x,y,z,w,d,h=shape['box'];return {(a,b,c) for a in range(x,x+w) for b in range(y,y+d) for c in range(z,z+h)}
 if 'cylinder' in shape:
  cx,cy,z,r,h,inner=shape['cylinder']
  return {(x,y,k) for x in range(int(cx-r),int(cx+r)) for y in range(int(cy-r),int(cy+r)) for k in range(z,z+h) if inner**2<=(x+.5-cx)**2+(y+.5-cy)**2<r*r}
 cx,y,cz,r,depth=shape['wheel']
 return {(x,b,z) for x in range(int(cx-r),int(cx+r)) for b in range(y,y+depth) for z in range(int((cz-r)/.4),int((cz+r)/.4)) if (x+.5-cx)**2+((z+.5)*.4-cz)**2<r*r}


def component(name,shapes,color,evidence):return {'id':name,'shapes':shapes,'color':color,'evidence':evidence}

CAR={
 'id':'al1361-car-layout-v1','family':'car','title':'Ford G.P. Pygmy · vehicle layout',
 'source_id':'al1361','source_url':'https://www.loc.gov/item/al1361/',
 'source_sha256':'4caab780571fb3ab554bb91a5b1c0c0f5e6d53a1ffb35b566448be1108c5c92d',
 'provenance':'Printed external dimensions and wheelbase from the archived drawing; brick topology and internal component sizes are authored proxies.',
 'geometry_kind':'component_assembly','view_bounds':[33,14,38],
 'inches_per_stud':4,'dimension_tolerance_studs':.5,
 'source_dimensions_inches':{'length':133.0625,'width':56,'height':60,'wheelbase':80},
 'abstraction':'Static brick vehicle with four voxel tire proxies, chassis, floor, hood, windshield, two seats and rear bed. Wheels connect through the floor/fenders; no working axle mechanism.',
 'unmodeled_source_features':['Steering and drivetrain','Real wheel/axle parts and rolling','Spare wheel and small fittings','Exact curved body panels'],
 'coverage_claim':'External source proportions plus a curated static assembly, not a complete CAD reconstruction.',
 'limits':{'transactions':16,'bricks':512},
 'components':[
  *[component(f'wheel_{axle}_{side}',[wheel(x,y,3,3,2)],[63,73,79],'Axle locations from 6 ft 8 in wheelbase; tire size and voxel profile are authored.') for axle,x in [('front',7),('rear',27)] for side,y in [('near',0),('far',12)]]],
 'fault_component':'wheel_rear_far',
}
CAR['components'] += [
 component('chassis',[box(0,2,9,33,10,6)],[100,105,83],'Undercarriage composition; dimensions are authored.'),
 component('floor_fenders',[box(0,0,15,33,14,2)],[172,173,108],'Authored continuous floor/fender plate connects static wheel proxies.'),
 component('hood',[box(0,2,17,12,10,9)],[170,181,112],'Front hood in plan and elevation; simplified cuboid.'),
 component('windshield',[box(13,2,17,1,10,21)],[91,159,180],'Upright windshield; total height constrained by printed 5 ft datum.'),
 component('seat_near',[box(17,3,17,4,3,5),box(20,3,22,1,3,6)],[141,101,69],'Two seats visible in plan; seating sizes authored.'),
 component('seat_far',[box(17,8,17,4,3,5),box(20,8,22,1,3,6)],[141,101,69],'Two seats visible in plan; seating sizes authored.'),
 component('rear_bed',[box(24,2,17,9,10,3),box(24,2,20,9,1,6),box(24,11,20,9,1,6),box(32,3,20,1,8,6)],[148,159,100],'Rear body compartment from plan; detailed panel geometry not modeled.'),
]
TURBINE={
 'id':'al1187-turbine-layout-v1','family':'machinery','title':'Wilson turbine / generator · composition study',
 'source_id':'al1187','source_url':'https://www.loc.gov/item/al1187/',
 'source_sha256':'8c1326bb0f25245ec55a5084ecd2584fcf61fb23a823bc15ed7f315ee100a970',
 'provenance':'Source supplies component identities and vertical/coaxial arrangement. This fixture uses authored dimensions; it is not a source-dimension reconstruction.',
 'geometry_kind':'component_assembly','view_bounds':[20,20,35],
 'abstraction':'Static cutaway-inspired arrangement: lower runner, shaft, rotor inside stator, upper bearing proxy and exciter. A teaching model of composition.',
 'unmodeled_source_features':['Hydraulic flow and power generation','Calibrated source dimensions','Actual runner blades and winding geometry','Governor, oil systems and surrounding civil structure'],
 'coverage_claim':'Composition-only task with authored sizes; no claim of geometric fidelity to the full machine.',
 'limits':{'transactions':16,'bricks':512},
 'components':[
 component('runner',[cylinder(10,10,0,5,6)],[69,131,143],'Francis turbine/runner beneath generator; circular proxy, no blades.'),
 component('shaft',[box(9,9,6,2,2,13)],[178,158,102],'Central shaft connects lower turbine to upper generator.'),
 component('rotor',[cylinder(10,10,19,2.5,6)],[177,114,64],'Rotor inside the stator, coaxial with the shaft.'),
 component('stator',[cylinder(10,10,16,7,9,2.5)],[91,129,107],'Outer generator body around rotor; annular proxy.'),
 component('bearing_support',[cylinder(10,10,25,7,1)],[125,142,135],'Authored top plate stands in for upper bracket/bearing support.'),
 component('exciter',[cylinder(10,10,26,3,9)],[184,150,91],'Exciter above main generator on the same axis.'),
 ],'fault_component':'exciter',
}
LIBRARY={'car':CAR,'turbine':TURBINE}


def targets(task):
 return [{**c,'kind':c['id'],'cells':set().union(*(shape_cells(s) for s in c['shapes']))} for c in task['components']]


def tile_component(cells,thin_rows=False):
 """Pack a reviewed voxel mask with the same legal parts available to agents."""
 remaining=set(cells);pieces=[]
 choices=[]
 for name,dim in PALETTE.items():
  for r in [0,90]:
   dx,dy,dz=dim if r==0 else [dim[1],dim[0],dim[2]]
   choices.append((dx*dy*dz,name,dx,dy,dz,r))
 choices.sort(reverse=True)
 while remaining:
  z=min(p[2] for p in remaining)
  reverse=(z//3)%2
  x,y,z=min(remaining,key=lambda p:(p[2],-p[1] if reverse else p[1],-p[0] if reverse else p[0]))
  for _,name,dx,dy,dz,r in sorted(choices,key=lambda c:(c[0],c[2] if z%4==0 else c[3]),reverse=True):
   if dz==3 and z%4!=1:continue
   plane_y=[p[1] for p in (remaining if thin_rows else cells) if p[2]==z]
   edge_y=max(plane_y) if reverse else min(plane_y)
   if z%4==0 and y==edge_y and dy!=1:continue
   a0=x-dx+1 if reverse else x;b0=y-dy+1 if reverse else y
   candidate={(a,b,c) for a in range(a0,a0+dx) for b in range(b0,b0+dy) for c in range(z,z+dz)}
   if candidate<=remaining:
    pieces.append({'part':name,'x':a0,'y':b0,'z':z,'rotation':r});remaining-=candidate;break
  else:raise ValueError('Mask cannot be tiled with the palette')
 return pieces


@lru_cache(maxsize=8)
def reference_cached(serialized_task):
 task=json.loads(serialized_task);result=[]
 ts=targets(task);mask=set().union(*(c['cells'] for c in ts))
 for i,b in enumerate(tile_component(mask,thin_rows=task['family']=='machinery')):
  owner=next((c['id'] for c in ts if (b['x'],b['y'],b['z']) in c['cells']),'assembly')
  result.append({**b,'id':f'{owner}_{i}'})
 return result


def reference(task):
 return [dict(b) for b in reference_cached(json.dumps(task,sort_keys=True))]
