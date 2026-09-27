"""Freeze compositional IDA* implementation specifications and private test cases."""
import argparse, hashlib, itertools, json, random
from pathlib import Path

SYSTEM='Write Python 3 code only, defining solve(problem). No explanation. Standard library imports are allowed. The function must work for arbitrary inputs satisfying the specification. Do not print; return the requested JSON-serializable result.'

def oracle(problem, config):
    graph=problem['graph'];goal=problem['goal'];start=problem['start'];h=problem['heuristic']
    bound=h.get(start,0) if config['initial']=='heuristic' else 0
    bounds=[];visits=[];answer=None;cost=None
    while True:
        bounds.append(bound);trace=[];best=float('inf')
        def search(path,g):
            nonlocal best,answer,cost
            node=path[-1];f=g+h.get(node,0);cut=f>bound
            if (config['root'] or len(path)>1) and (config['cutoffs'] or not cut) and (config['goal'] or node!=goal):trace.append(node)
            if cut:best=min(best,f);return False
            if node==goal:answer=list(path);cost=g;return True
            edges=graph.get(node,[])
            if config['order']!='given':edges=sorted(edges,key=lambda e:e[0],reverse=config['order']=='descending')
            for child,weight in edges:
                if child not in path and search(path+[child],g+weight):return True
            return False
        found=search([start],0);visits.append(trace)
        if found or best==float('inf'):break
        bound=best
    fields={'path':answer,'cost':cost,'bounds':bounds,'visits':visits,'counts':[len(v) for v in visits]}
    return {k:fields[k] for k in config['fields']}

def prompt(config):
    order={'given':'the order supplied in each adjacency list','ascending':'ascending lexicographic node-label order','descending':'descending lexicographic node-label order'}[config['order']]
    return f'''Implement solve(problem), an instrumented iterative-deepening A* search.
Input is {{"graph": {{node: [[neighbor,positive_integer_cost],...]}}, "start": node, "goal": node, "heuristic": {{node: nonnegative_integer}}}}. Node labels are strings. Missing graph entries have no edges; missing heuristic entries mean 0. The heuristic is admissible. No repeated node is allowed in the current path, but there is no global visited set.
Initialize the cost bound to {'heuristic[start] (default 0)' if config['initial']=='heuristic' else '0'}. Each iteration runs depth-first search from start with g=0 and records its bound. At each entered node compute f=g+h(node). If f>bound, prune and contribute f to the minimum exceeded bound. Check for goal ONLY after that cutoff. On success stop at the first goal reached. Otherwise increase bound to the minimum exceeded f; if none exists, stop as unreachable. Visit children in {order}. Skip children already in the current path BEFORE calling recursion; skipped children never enter the trace.
For each iteration record an ordered trace of entered node labels. {'Include' if config['root'] else 'Exclude'} the iteration root. {'Include' if config['cutoffs'] else 'Exclude'} nodes pruned by f>bound. {'Include' if config['goal'] else 'Exclude'} every entry whose label equals the goal. These filters apply jointly, only to logging, never to search. Repeated entries via different paths remain in the trace. Keep traces from all iterations, including the final one.
Return exactly these keys: {json.dumps(config['fields'])}. Available meanings: path = successful node path or None; cost = its summed edge costs or None; bounds = attempted cost bounds; visits = list of iteration traces; counts = lengths of those traces. Return no extra keys. Handle start==goal, cycles, ties, missing adjacency lists and unreachable goals. All inputs are small enough for exhaustive path search. The algorithm is derived from Korf's iterative-deepening search; logging and return conventions are exercise specifications.'''

def cases(seed):
    r=random.Random(seed)
    fixed=[{'graph':{},'start':'A','goal':'A','heuristic':{}},
      {'graph':{'A':[['C',1],['B',1]],'B':[['G',2]],'C':[['G',2]]},'start':'A','goal':'G','heuristic':{}},
      {'graph':{'A':[['B',2]],'B':[['A',1]]},'start':'A','goal':'G','heuristic':{}},
      {'graph':{'A':[['G',5],['B',1]],'B':[['C',1]],'C':[['G',1]]},'start':'A','goal':'G','heuristic':{'A':3,'B':2,'C':1,'G':0}}]
    for i in range(12):
        nodes=[chr(65+j) for j in range(r.randint(3,7))];graph={a:[] for a in nodes}
        for a in nodes:
            for b in nodes:
                if a!=b and r.random()<.29:graph[a].append([b,r.randint(1,6)])
            r.shuffle(graph[a])
        # Exact reverse distances certify admissibility; sampled smaller estimates vary pruning.
        dist={n:float('inf') for n in nodes};dist[nodes[-1]]=0
        for _ in nodes:
            for a,edges in graph.items():
                for b,w in edges:dist[a]=min(dist[a],w+dist[b])
        h={n:r.randint(0,int(d)) if d!=float('inf') else 0 for n,d in dist.items()}
        fixed.append({'graph':graph,'start':nodes[0],'goal':nodes[-1],'heuristic':h})
    return fixed

def generate(dest,source):
    if dest.exists():raise ValueError('Frozen dataset already exists')
    source_data=json.loads(source.read_text());rng=random.Random(20260929)
    configs=[{'order':o,'root':r,'cutoffs':c,'goal':g,'initial':i,'fields':f} for o,r,c,g,i,f in itertools.product(['given','ascending','descending'],[False,True],[False,True],[False,True],['zero','heuristic'],[['path','cost','bounds','visits'],['path','bounds','counts'],['cost','bounds','visits','counts']])]
    rng.shuffle(configs);rows=[]
    for n,config in enumerate(configs[:100]):
        inputs=cases(982000+n);row={'id':f'korf-code-{n:03}','split':'train' if n<80 else 'heldout','configuration':config,'prompt':[{'role':'system','content':SYSTEM},{'role':'user','content':prompt(config)}], 'tests':[{'input':x,'expected':oracle(x,config)} for x in inputs]};rows.append(row)
    dest.mkdir(parents=True)
    for split in ('train','heldout'):
        (dest/f'{split}.json').write_text(json.dumps([r for r in rows if r['split']==split],indent=2))
    manifest={'kind':'executable_python_idastar','train':80,'heldout':20,'tests_per_task':16,'seed':20260929,'scope':'Generalization across unseen combinations of IDA* instrumentation conventions and fresh graph inputs','source_paper':'columbia-1006','source_card_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'source_cards':[{k:c[k] for k in ('card_id','page','statement')} for c in source_data['cards']],'split_policy':'Unique specification configurations; 80 training, 20 frozen heldout; no task or checkpoint selection from heldout scores','sha256':{s:hashlib.sha256((dest/f'{s}.json').read_bytes()).hexdigest() for s in ('train','heldout')}}
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--source',type=Path,required=True);a=p.parse_args();generate(a.output,a.source)
