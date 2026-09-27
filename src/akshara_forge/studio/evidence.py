"""Build presentation data directly from recorded runs; absent scores stay null."""
import argparse,json,time
from pathlib import Path

def read(p):
    return json.loads(p.read_text()) if p.exists() else None

def snapshot(root):
    runs=root/'runs/aws/results/runs'
    before=read(runs/'bf16-before-normalized/graded-summary.json')
    after=read(runs/'bf16-after-epoch-1-normalized/graded-summary.json')
    launch=read(root/'runs/engineering-interactive/launch.json') or {}
    er=root/'runs/engineering-interactive/results'/launch.get('long_run','long-003')
    eb=read(er/'before/summary.json');ea=read(er/'after/summary.json');done=read(er/'completion.json')
    events=er/'optimizer-events.jsonl'
    records=[]
    if events.exists():
        for line in events.read_text().splitlines():
            try:records.append(json.loads(line))
            except json.JSONDecodeError:pass
    def score(s,math=False):
        if not s:return None
        return {'correct':s['correct'] if math else s['successes'],'completed':s['episodes'] if math else s['completed'],'total':20,'complete':s.get('complete',s.get('graded')==20)}
    return {'updated_at':time.time(),'model':'Qwen 2.5 · 7B parameters','math':{'before':score(before,True),'after':score(after,True),'steps':80,'status':'Evaluation complete' if after else 'Evaluation scheduled','source':'Optimization · augmented Lagrangian methods','protocol':'20 held-out problems · 10 exact-answer checks + 10 Astra-graded proofs','run':'curriculum-epoch-1'},
      'coding':{'before':None,'after':None,'status':'Dataset ready','source':'Korf · iterative-deepening search','protocol':'20 held-out algorithm problems · exact structured-answer checks','run':None},
      'engineering':{'before':score(eb),'after':score(ea),'status':'Evaluation complete' if done else 'Training' if records else 'Baseline evaluation','steps':len(records),'target_steps':launch.get('long_steps',50),'reward_updates':records[-1]['reward_contrast_updates'] if records else 0,'source':'Bridge assembly · eight-turn upright repair','protocol':'20 held-out assemblies · identical tools and eight-turn budgets','run':launch.get('long_run','long-003')}}

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.parent.mkdir(parents=True,exist_ok=True);temp=a.output.with_suffix('.tmp');temp.write_text(json.dumps(snapshot(a.root),indent=2));temp.replace(a.output)
if __name__=='__main__':main()
