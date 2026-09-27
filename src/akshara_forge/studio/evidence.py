"""Build presentation data directly from recorded runs; absent scores stay null."""
import argparse,json,time
from pathlib import Path

def read(p):
    return json.loads(p.read_text()) if p.exists() else None

def first_record(path):
    if not path.exists():return None
    for line in path.read_text().splitlines():
        try:return json.loads(line)
        except json.JSONDecodeError:continue
    return None

def engineering_samples(root,run):
    dataset=root/'runs/engineering-interactive/dataset-v1/heldout.json'
    if not dataset.exists():return []
    rows=json.loads(dataset.read_text())
    def indexed(path):
        if not path.exists():return {}
        records=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        by_id={r['id']:r for r in records}
        if len(by_id)!=len(records) or set(by_id)-{r['id'] for r in rows}:
            raise ValueError('Held-out trace identity mismatch')
        return by_id
    before=indexed(run/'before/traces.jsonl');after=indexed(run/'after/traces.jsonl')
    if set(after)-set(before):raise ValueError('Sample pairing mismatch')
    from ..engineering.worker import PALETTE
    return [{'id':row['id'],'task':row['task'],'target':row['target'],'fixed':row['initial'],
             'initial_editable':row.get('editable_initial',[]),'palette':PALETTE,
             'before':before[row['id']]['state']['observation'],
             'after':after[row['id']]['state']['observation'] if row['id'] in after else None}
            for row in rows if row['id'] in before]

def engineering_sample(root,run):
    samples=engineering_samples(root,run)
    return samples[0] if samples else None

def coding_sample(run):
    before=first_record(run/'before/episodes.jsonl');after=first_record(run/'after/episodes.jsonl')
    if not before:return None
    if after and before['id']!=after['id']:raise ValueError('Sample pairing mismatch')
    return {'id':before['id'],'before':before,'after':after}

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
    coding_launch=read(root/'runs/coding-training/launch.json') or {}
    cr=root/'runs/coding-training/results'/coding_launch.get('run','code-001')
    cb=read(cr/'before/summary.json');ca=read(cr/'after/summary.json');cd=read(cr/'completion.json')
    ce=cr/'optimizer-events.jsonl';coding_records=[]
    if ce.exists():
        for line in ce.read_text().splitlines():
            try:coding_records.append(json.loads(line))
            except json.JSONDecodeError:pass
    coding_steps=len(coding_records)
    def score(s,math=False):
        if not s:return None
        return {'correct':s['correct'] if math else s['successes'],'completed':s['episodes'] if math else s['completed'],'total':20,'complete':s.get('complete',s.get('graded')==20),'mean_reward':s.get('mean_reward'),'mean_gap_iou':s.get('mean_gap_iou')}
    samples=engineering_samples(root,er)
    return {'updated_at':time.time(),'model':'Qwen 2.5 · 7B parameters','math':{'before':score(before,True),'after':score(after,True),'steps':80,'status':'Evaluation complete' if after else 'Evaluation scheduled','source':'Optimization · augmented Lagrangian methods','protocol':'20 held-out problems · 10 exact-answer checks + 10 Astra-graded proofs','run':'curriculum-epoch-1'},
      'coding':{'sample':coding_sample(cr),'before':score(cb),'after':score(ca),'status':'Evaluation complete' if cd else 'Final evaluation' if ca or coding_steps>=coding_launch.get('steps',80) else 'Training' if coding_steps else 'Baseline evaluation' if cb else 'Queued after engineering' if coding_launch else 'Dataset ready','source':'Korf · executable IDA* implementations','tests_per_task':coding_launch.get('tests_per_task',16),'protocol':f"20 held-out Python specifications · {coding_launch.get('tests_per_task',16)} sandboxed tests each · same greedy decoding before/after",'run':coding_launch.get('run'),'steps':coding_steps,'target_steps':coding_launch.get('steps',80),'weight_update':{k:coding_records[-1].get(k) for k in ('changed_parameters','delta_l2','nonzero_gradient_elements')} if coding_records else None},
      'engineering':{'sample':samples[0] if samples else None,'samples':samples,'before':score(eb),'after':score(ea),'status':'Evaluation complete' if done else 'Final evaluation' if ea or len(records)>=launch.get('long_steps',50) else 'Training' if records else 'Baseline evaluation','steps':len(records),'target_steps':launch.get('long_steps',50),'reward_updates':records[-1]['reward_contrast_updates'] if records else 0,'weight_update':{k:records[-1].get(k) for k in ('changed_parameters','delta_l2','nonzero_gradient_elements')} if records else None,'source':'Bridge assembly · eight-turn upright repair','protocol':'20 held-out assemblies · identical tools and eight-turn budgets','run':launch.get('long_run','long-003')}}

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.parent.mkdir(parents=True,exist_ok=True);temp=a.output.with_suffix('.tmp');temp.write_text(json.dumps(snapshot(a.root),indent=2));temp.replace(a.output)
if __name__=='__main__':main()
