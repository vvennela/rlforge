"""Match geometric components without trusting brick IDs or semantic labels."""
import itertools
import math


def groups(occupied):
    remaining=set(occupied);result=[]
    while remaining:
        first=remaining.pop();group={first};pending=[first]
        while pending:
            x,y,z=pending.pop()
            for p in [(x-1,y,z),(x+1,y,z),(x,y-1,z),(x,y+1,z),(x,y,z-1),(x,y,z+1)]:
                if p in remaining:remaining.remove(p);group.add(p);pending.append(p)
        result.append(group)
    return sorted(result,key=lambda g:(-len(g),min(g)))


def bounds(points):
    low=[min(p[i] for p in points) for i in range(3)]
    high=[max(p[i] for p in points)+1 for i in range(3)]
    return {'min':low,'max':high,'size':[b-a for a,b in zip(low,high)],'center':[(a+b)/2 for a,b in zip(low,high)]}


def compare_components(expected,occupied,deck_z):
    # First segment by construction zone, then connected occupied geometry.
    zones={
        'deck':{p for p in occupied if deck_z<=p[2]<deck_z+2},
        'pier':{p for p in occupied if p[2]<deck_z},
        'upright':{p for p in occupied if p[2]>=deck_z+2},
    }
    result=[]
    for kind,zone in zones.items():
        targets=[t for t in expected if t['kind']==kind]
        candidates=groups(zone)[:8]  # Extra fragments still penalize whole-shape IoU.
        candidates += [set()] * max(0,len(targets)-len(candidates))
        def cost(target,candidate):
            if not candidate:return 1e6
            return math.dist(bounds(target['cells'])['center'],bounds(candidate)['center'])
        assignment=min(itertools.permutations(range(len(candidates)),len(targets)),
                       key=lambda perm:sum(cost(t,candidates[j]) for t,j in zip(targets,perm))) if targets else []
        for target,j in zip(targets,assignment):
            result.append(score_component(target,candidates[j]))
    return result


def score_component(target,got):
    want=target['cells'];tb=bounds(want);actual=bounds(got) if got else None
    dims={};size_score=position_score=proportion_score=0.
    if got:
        # z converts plate heights to studs before comparing physical proportions.
        tsize=[tb['size'][0],tb['size'][1],tb['size'][2]*.4]
        asize=[actual['size'][0],actual['size'][1],actual['size'][2]*.4]
        size_score=sum(max(0.,1-abs(a-b)/b) for a,b in zip(asize,tsize))/3
        delta=[abs(a-b)*(.4 if i==2 else 1) for i,(a,b) in enumerate(zip(actual['center'],tb['center']))]
        position_score=max(0.,1-math.sqrt(sum(d*d for d in delta))/max(1.,math.sqrt(sum(d*d for d in tsize))))
        # Aspect ratios remove uniform scale; size_score retains absolute scale.
        ratios=[(asize[i]/asize[0])/(tsize[i]/tsize[0]) for i in [1,2]]
        proportion_score=sum(max(0.,1-abs(r-1)) for r in ratios)/2
    for i,label in enumerate(['x_studs','y_studs','z_plates']):
        dims[label]={'target':tb['size'][i],'actual':actual['size'][i] if actual else None}
    scores={'presence':float(bool(got)),'position':position_score,'size':size_score,
            'proportions':proportion_score,'overlap':len(want&got)/len(want|got)}
    return ({'id':target['id'],'kind':target['kind'],'dimensions':dims,'target_bounds':tb,'actual_bounds':actual,
                   'scores':scores,'complete':want==got,
                   'evidence':target['evidence']})


def compare_masks(expected,occupied):
    """Compare labeled target regions to geometry, never to agent-provided IDs.

    Exact target cells get their fixed region label. Off-target cells are assigned
    to the nearest target bounding box; global IoU also penalizes every extra cell.
    This evaluates occupancy in known regions, not learned semantic recognition.
    """
    regions=[set() for _ in expected]
    owners={p:i for i,t in enumerate(expected) for p in t['cells']}
    boxes=[bounds(t['cells']) for t in expected]
    for p in occupied:
        i=owners.get(p)
        if i is None:
            def distance(b):
                return sum((max(b['min'][k]-(p[k]+.5),0,p[k]+.5-b['max'][k])*(.4 if k==2 else 1))**2 for k in range(3))
            i=min(range(len(boxes)),key=lambda j:distance(boxes[j]))
        regions[i].add(p)
    return [score_component(t,g) for t,g in zip(expected,regions)]
