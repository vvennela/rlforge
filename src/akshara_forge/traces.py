"""Offline viewer for explicit learner work and structured feedback, not hidden reasoning."""
from __future__ import annotations
import json
from pathlib import Path
from .io import read_jsonl


def build(root: Path) -> Path:
    runs=[(root/'runs/qwen-base-heldout',root),(root/'runs/qwen-train-smoke',root)]
    runs += [(p/'runs/qwen-demo',p) for p in (root/'runs/auto-environments').glob('*') if p.is_dir()]
    runs += [(p,root/'datasets/pilot-v2') for p in (root/'runs/aws/results/runs').glob('bf16-*') if p.is_dir()]
    items=[]
    for run,dataset in runs:
        source=run/'graded-episodes.jsonl'
        if not source.exists():source=run/'episodes.jsonl'
        if not source.exists():continue
        problems={p['problem_id']:p for file in (dataset/'data/problems').glob('*/all.jsonl') for p in read_jsonl(file)}
        if not problems:
            problems={p['problem_id']:p for file in (dataset/'data/problems').glob('*/test.jsonl') for p in read_jsonl(file)}
        for r in read_jsonl(source):
            items.append({'run':str(run.relative_to(root)), 'problem_id':r['problem_id'],
                          'prompt':problems.get(r['problem_id'],{}).get('prompt',''),
                          'action':r.get('action',{}),'status':r['status'],'correct':r.get('correct'),
                          'reward':r.get('reward'),'seconds':r.get('seconds'),'tokens':r.get('eval_count'),
                          'judgment':r.get('teacher_verdict'),'feedback':r.get('feedback'),
                          'raw_response':r.get('response'),'kind':'evaluation'})
    for path in (root/'runs/aws/results/checkpoints').glob('*/reward-traces.jsonl'):
        problems={p['problem_id']:p for file in (root/'datasets/pilot-v2/data/problems').glob('*/train.jsonl') for p in read_jsonl(file)}
        for r in read_jsonl(path):
            g=r['grade']
            items.append({'run':str(path.parent.relative_to(root)), 'problem_id':r['problem_id'],
                          'prompt':problems.get(r['problem_id'],{}).get('prompt',''),
                          'action':r['submission'],'status':g['status'],'reward':g.get('reward'),
                          'judgment':g.get('judgment'),'milestones':g.get('milestone_grades'),
                          'training_context':r.get('training_context'),'kind':'training',
                          'outcome_reward':g.get('outcome_reward'),'progress_coverage':g.get('progress_coverage')})
    data=json.dumps(items,ensure_ascii=False).replace('<','\\u003c').replace('\u2028','\\u2028').replace('\u2029','\\u2029')
    html='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AksharaForge · Rollout traces</title>
<style>body{margin:0;background:#111b23;color:#e0e9ed;font:16px/1.5 system-ui}header{padding:24px 4vw;border-bottom:1px solid #345}h1{color:#91e2d2;margin:0}p{max-width:1000px}a{color:#91e2d2}main{display:grid;grid-template-columns:32% 1fr;gap:22px;padding:20px 4vw}button,select,input{background:#1b303b;color:inherit;border:1px solid #456;border-radius:5px;padding:10px;font:inherit}button{display:block;text-align:left;width:100%;margin:6px 0;cursor:pointer}#list{max-height:74vh;overflow:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#1b2b35;padding:16px;border-radius:8px}small{color:#a9bdc9}article{min-width:0}li{margin-bottom:14px}.tools{display:flex;gap:10px;flex-wrap:wrap;margin-top:18px}@media(max-width:800px){main{display:block}#list{max-height:30vh}}</style>
<header><h1>Rollout traces</h1><p>Inspect Qwen's submitted mathematical work, the examiner's feedback, and training rewards. These records are explicit solutions and structured grading evidence. A high process reward does not by itself establish evaluation uplift.</p><a href="ARTIFACTS.html">Source artifacts</a> · <a href="RUN_STATUS.md">Run status</a><div class="tools"><select id="run" aria-label="Run"><option value="">All runs</option></select><input id="search" aria-label="Search problems" placeholder="Search a question or problem ID"><span id="count"></span></div></header><main><div id="list"></div><article id="detail"><h2>Select a rollout</h2></article></main>
<script type="application/json" id="records">PAYLOAD</script><script>
const rows=JSON.parse(document.getElementById('records').textContent),$=id=>document.getElementById(id);
function add(p,t,s){let e=document.createElement(t);e.textContent=s;p.append(e);return e}
for(let r of [...new Set(rows.map(x=>x.run))]){let o=add($('run'),'option',r);o.value=r}
function show(r){let d=$('detail');d.replaceChildren();add(d,'h2',r.problem_id);add(d,'small',r.run);add(d,'p','Status: '+r.status+' · Reward: '+(r.reward??'ungraded'));if(r.training_context)add(d,'p','Optimizer step: '+r.training_context.global_step+' · Outcome: '+r.outcome_reward+' · Milestone coverage: '+r.progress_coverage);add(d,'h3','Question');add(d,'pre',r.prompt);add(d,'h3','Qwen’s written steps');let ul=add(d,'ol','');let steps=r.action.steps||[];if(!steps.length)add(d,'p','No structured written steps were supplied.');for(let [i,s] of steps.entries()){let li=add(ul,'li',s),g=r.judgment?.steps?.find(x=>x.index===i);if(g)add(li,'p',g.verdict+': '+g.feedback)}add(d,'h3','Final answer');add(d,'pre',JSON.stringify(r.action.answer,null,2));if(r.judgment){add(d,'h3','Examiner verdict');add(d,'p',r.judgment.verdict+': '+r.judgment.feedback)}else if(r.feedback)add(d,'p',r.feedback);if(r.milestones){add(d,'h3','Fixed milestone credit');for(let m of r.milestones)add(d,'p',m.id+' · '+(m.credited?'credited':'not credited')+' · '+m.feedback)}if(r.raw_response){let x=add(d,'details','');add(x,'summary','Raw model response');add(x,'pre',r.raw_response)}}
function render(){let q=$('search').value.toLowerCase(),rs=rows.filter(r=>(!$('run').value||r.run===$('run').value)&&(!q||(r.prompt+' '+r.problem_id).toLowerCase().includes(q)));$('count').textContent=rs.length+' rollouts';$('list').replaceChildren();for(let r of rs){let b=add($('list'),'button',r.problem_id);add(b,'br','');add(b,'small',r.kind+' · '+r.status+' · reward '+(r.reward??'pending'));b.onclick=()=>show(r)}}
$('run').onchange=render;$('search').oninput=render;render();</script></html>'''.replace('PAYLOAD',data)
    path=root/'TRACES.html';path.write_text(html)
    return path


if __name__=='__main__':print(build(Path.cwd()))
