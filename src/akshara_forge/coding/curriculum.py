"""Executable IDA* curriculum: local repair, bounded DFS, search, instrumentation.

Every stage has independently seeded development/private inputs. The heldout split
measures new instances of the practiced skills; it is not an unseen-algorithm test.
"""
import argparse, copy, hashlib, json, random
from pathlib import Path
from ..curriculum import slot, manifest as curriculum_manifest
from .tasks import SYSTEM, cases, oracle, prompt
from .feedback import REPAIR_INSTRUCTION, canonical, digest
from .adversarial import independent
from ..engineering.pilot import write

CONFIG={'order':'given','root':True,'cutoffs':True,'goal':True,'initial':'heuristic','fields':['path','cost']}


def foundation(kind,p):
    if kind=='score':return {'value':p['g']+p['h']}
    if kind=='cutoff':return {'value':p['g']+p['h']>p['bound']}
    if kind=='next_bound':return {'value':min((v for v in p['values'] if v>p['bound']),default=None)}
    if kind=='neighbors':return {'value':[n for n in p['neighbors'] if n not in p['path']]}
    if kind=='path_cost':return {'value':sum(p['weights'])}
    raise ValueError(kind)


def foundation_second(kind,p):
    if kind=='score':value=sum([p['h'],p['g']])
    elif kind=='cutoff':value=p['h']>p['bound']-p['g']
    elif kind=='next_bound':
        candidates=sorted(p['values']);value=next((v for v in candidates if p['bound']<v),None)
    elif kind=='neighbors':
        value=[]
        for n in p['neighbors']:
            if all(n!=v for v in p['path']):value.append(n)
    else:
        value=0
        for w in p['weights']:value+=w
    return {'value':value}


def bounded(p):
    """Recursive reference for a single cost-limited DFS on a tree."""
    def visit(node,g,path):
        if g>p['bound']:return None
        if node==p['goal']:return path
        for child,w in p['graph'].get(node,[]):
            answer=visit(child,g+w,path+[child])
            if answer is not None:return answer
        return None
    return {'path':visit(p['start'],0,[p['start']])}


def bounded_second(p):
    stack=[(p['start'],0,[p['start']])]
    while stack:
        node,g,path=stack.pop()
        if g>p['bound']:continue
        if node==p['goal']:return {'path':path}
        stack.extend((c,g+w,path+[c]) for c,w in reversed(p['graph'].get(node,[])))
    return {'path':None}


def tree_inputs(seed,n,max_nodes=6):
    rng=random.Random(seed);out=[]
    for i in range(n):
        nodes=[f'n{seed}_{j}' for j in range(rng.randint(1,max_nodes))];graph={x:[] for x in nodes}
        for j in range(1,len(nodes)):graph[nodes[rng.randrange(j)]].append([nodes[j],rng.randint(1,5)])
        out.append({'graph':graph,'start':nodes[0],'goal':nodes[-1] if i%4 else 'absent','heuristic':{},'bound':rng.randint(0,15)})
    return out


def foundation_inputs(kind,seed,n):
    rng=random.Random(seed);out=[]
    for i in range(n):
        if kind in ('score','cutoff'):
            p={'g':rng.randrange(20),'h':rng.randrange(20),'bound':rng.randrange(40)}
            if i%3==0:p['bound']=p['g']+p['h']
        elif kind=='next_bound':p={'values':[rng.randrange(30) for _ in range(i%7)],'bound':rng.randrange(30)}
        elif kind=='neighbors':
            labels=[f'n{seed}_{j}' for j in range(5)];p={'neighbors':rng.sample(labels,rng.randrange(6)),'path':rng.sample(labels,rng.randrange(6))}
        else:p={'weights':[rng.randint(1,9) for _ in range(i%8)]}
        out.append(p)
    return out


def foundation_prompt(kind,variant):
    rules={
        'score':('Return {"value": g+h}, the cost-so-far plus heuristic.','p["g"] + p["h"]',['p["g"] - p["h"]','p["g"]','p["h"]','p["g"] * p["h"]','p["g"] + p["h"] + 1']),
        'cutoff':('Return {"value": boolean}: prune exactly when g+h is strictly greater than bound. Equality is allowed.','p["g"] + p["h"] > p["bound"]',['p["g"] + p["h"] >= p["bound"]','p["g"] > p["bound"]','p["h"] > p["bound"]','p["g"] + p["h"] < p["bound"]','False']),
        'next_bound':('Return {"value": the smallest value strictly exceeding bound}, or None if none exceeds it.','min((v for v in p["values"] if v > p["bound"]), default=None)',['max(p["values"], default=None)','min(p["values"], default=None)','None','min((v for v in p["values"] if v >= p["bound"]), default=None)','min((v for v in p["values"] if v < p["bound"]), default=None)']),
        'neighbors':('Return {"value": list of neighbors not already in path}, preserving supplied order.','[n for n in p["neighbors"] if n not in p["path"]]',['p["neighbors"]','[]','[n for n in p["neighbors"] if n in p["path"]]','sorted(n for n in p["neighbors"] if n not in p["path"])','list(reversed([n for n in p["neighbors"] if n not in p["path"]]))']),
        'path_cost':('Return {"value": sum of the edge weights}; the empty path has cost 0.','sum(p["weights"])',['len(p["weights"])','sum(p["weights"]) + 1','max(p["weights"], default=0)','0','sum(p["weights"][:-1])']),
    }
    rule,correct,bugs=rules[kind]
    return 'Repair the single incorrect return expression in this IDA* helper. '+rule+'\nReturn the complete corrected function.\n```python\ndef solve(p):\n    return {"value": '+bugs[variant]+'}\n```',correct


def generate(dest,source):
    dest=Path(dest)
    if dest.exists():raise ValueError('Use a fresh immutable dataset directory')
    rows=[];identities=set();mutants={};kinds=['score','cutoff','next_bound','neighbors','path_cost']
    for index in range(100):
        assigned=slot(index,100);split=assigned.pop('split');level=assigned['level'];j=index%25
        seed=920000+index*1000
        if level==1:
            kind=kinds[j%5];description,correct=foundation_prompt(kind,j//5)
            maker=lambda seed,n:foundation_inputs(kind,seed,n)
            ref=lambda p:foundation(kind,p);second=lambda p:foundation_second(kind,p)
            config={'skill':kind,'bug':j//5}
        elif level==2:
            # Each task uses a different explicit bound; search remains one bounded pass.
            bound=j
            description=f'''Implement solve(p) for ONE depth-first search on an acyclic rooted tree, with fixed cost bound {bound}. Return {{"path": first goal path or None}}. Children are visited in supplied order. Prune when accumulated edge cost exceeds {bound}, BEFORE checking the goal. start==goal is valid. Missing adjacency lists have no children. Positive weights; no cycles. Complete the scaffold below; return the full program.
def solve(p):
    def dfs(node, cost, path):
        # Fill: cutoff, goal, then children. Return a path or None.
        pass
    return {{"path": dfs(p["start"], 0, [p["start"]])}}'''
            def maker(seed,n):
                result=tree_inputs(seed,n)
                for p in result:p['bound']=bound
                return result
            ref=bounded;second=bounded_second;config={'skill':'bounded_dfs','bound':bound}
        else:
            config=copy.deepcopy(CONFIG)
            config.update(order=['given','ascending','descending'][j%3],initial=['heuristic','zero'][j%2])
            if level==4:config.update(root=bool(j%2),cutoffs=bool((j//2)%2),goal=bool((j//4)%2),fields=['path','cost','bounds','visits','counts'])
            description=prompt(config)
            if level==3:
                description=description.split('For each iteration record')[0]+' Return exactly {"path": successful node path or None, "cost": summed edge costs or None}. Inputs are acyclic trees with zero heuristics. No trace logging is required.'
                maker=lambda seed,n:tree_inputs(seed,n,8)
            else:
                def maker(seed,n):
                    # Rename nodes to prevent fixed examples crossing development/private sets.
                    result=cases(seed)
                    for p in result:
                        label=lambda x:f'{seed}_{x}'
                        p['graph']={label(a):[[label(b),w] for b,w in es] for a,es in p['graph'].items()}
                        p['start']=label(p['start']);p['goal']=label(p['goal']);p['heuristic']={label(a):v for a,v in p['heuristic'].items()}
                    return result[:n]
            ref=lambda p:oracle(p,config);second=lambda p:independent(p,config)
        seen=set();sets={}
        for key,n,delta in [('development_tests',6,0),('tests',24,100)]:
            tests=[];round=0
            while len(tests)<n:
                for p in maker(seed+delta+round, max(n,16)):
                    identity=canonical(p)
                    if identity in seen or identity in identities:continue
                    seen.add(identity);identities.add(identity);expected=ref(p)
                    if second(p)!=expected:raise ValueError('Independent reference disagreement')
                    tests.append({'input':p,'expected':expected})
                    if len(tests)==n:break
                round+=1
                if round>100:raise ValueError('Cannot generate distinct cases')
            sets[key]=tests
        if level==1:
            buggy={};exec(description.split('```python\n')[1].split('```')[0],buggy)
            if all(buggy['solve'](t['input'])==t['expected'] for t in sets['tests']):raise ValueError('The planted bug survived private tests')
        # These negative programs must fail at least one private test at each stage.
        wrong=[{},None,{'value':0},{'path':None,'cost':None}]
        killed=[any(t['expected']!=v for t in sets['tests']) for v in wrong]
        if not all(killed):raise ValueError('Degenerate constant survived private tests')
        mutants[str(index)]=killed
        rows.append({'id':f'curriculum-code-{index:03}','split':split,'curriculum':assigned,'configuration':config,
                     'prompt':[{'role':'system','content':SYSTEM+' '+REPAIR_INSTRUCTION},{'role':'user','content':description}],**sets})
    # Inputs are withheld even for tasks sharing an algorithm/scaffold family.
    train_inputs={canonical(t['input']) for r in rows if r['split']=='train' for t in r['tests']+r['development_tests']}
    held_inputs={canonical(t['input']) for r in rows if r['split']=='heldout' for t in r['tests']+r['development_tests']}
    if train_inputs&held_inputs:raise ValueError('Input leakage across splits')
    for split in ('train','heldout'):write(dest/f'{split}.json',[r for r in rows if r['split']==split])
    write(dest/'manifest.json',{'kind':'coding_easy_to_hard','train':80,'heldout':20,'development_tests':6,'private_tests':24,'turns':3,
        'curriculum':curriculum_manifest(100),'source_manifest_sha256':digest(Path(source)/'manifest.json' if Path(source).is_dir() else Path(source)),
        'scope':'Held-out input sets and task variants within practiced skill families; not unseen-algorithm generalization.',
        'cross_split_input_overlap':len(train_inputs&held_inputs),'reference_agreement':True,'constant_mutations_rejected':all(all(v) for v in mutants.values()),
        'sha256':{s:digest(dest/f'{s}.json') for s in ('train','heldout')}})


def reference_program(row):
    import inspect
    level=row['curriculum']['level'];c=row['configuration']
    if level==1:
        _,expression=foundation_prompt(c['skill'],c['bug'])
        return 'def solve(p):\n    return {"value": '+expression+'}\n'
    if level==2:return inspect.getsource(bounded)+'\ndef solve(p): return bounded(p)\n'
    return inspect.getsource(oracle)+'\ndef solve(p): return oracle(p, '+repr(c)+')\n'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--source',type=Path,required=True);a=p.parse_args();generate(a.output,a.source)
