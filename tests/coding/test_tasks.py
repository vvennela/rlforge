import heapq,json
from akshara_forge.coding.tasks import oracle,cases,generate
from akshara_forge.coding.service import code_from


def test_exact_cutoff_trace_and_tie_order():
    p={'graph':{'A':[['C',1],['B',1]],'B':[['G',2]],'C':[['G',2]]},'start':'A','goal':'G','heuristic':{}}
    c={'order':'given','root':True,'cutoffs':True,'goal':True,'initial':'zero','fields':['path','cost','bounds','visits']}
    assert oracle(p,c)=={'path':['A','C','G'],'cost':3,'bounds':[0,1,3],'visits':[['A','C','B'],['A','C','G','B','G'],['A','C','G']]}
    c.update(root=False,cutoffs=False,goal=False)
    assert oracle(p,c)['visits']==[[],['C','B'],['C']]
    c['order']='ascending'
    assert oracle(p,c)['path']==['A','B','G']


def test_oracle_costs_match_independent_dijkstra():
    for seed in range(20):
        for p in cases(seed):
            q=[(0,p['start'])];seen={};answer=None
            while q:
                cost,node=heapq.heappop(q)
                if node in seen:continue
                seen[node]=cost
                if node==p['goal']:answer=cost;break
                for n,w in p['graph'].get(node,[]):heapq.heappush(q,(cost+w,n))
            config={'order':'descending','root':False,'cutoffs':True,'goal':False,'initial':'heuristic','fields':['cost']}
            assert oracle(p,config)['cost']==answer


def test_frozen_split_has_unique_specifications(tmp_path):
    source=tmp_path/'source.json';source.write_text(json.dumps({'cards':[]}));dest=tmp_path/'data';generate(dest,source)
    a=json.loads((dest/'train.json').read_text());b=json.loads((dest/'heldout.json').read_text())
    assert len(a)==80 and len(b)==20
    assert len({json.dumps(r['configuration'],sort_keys=True) for r in a+b})==100
    assert all(len(r['tests'])==16 for r in a+b)
    assert not ({r['id'] for r in a}&{r['id'] for r in b})


def test_code_fence_extraction():
    assert code_from('```python\ndef solve(p): return None\n```')=='def solve(p): return None\n'
