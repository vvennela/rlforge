"""Offline artifact browser. Text is inserted with textContent, never interpreted as HTML."""
from __future__ import annotations
import json
from pathlib import Path
from .io import read_jsonl


def build_browser(root: Path) -> Path:
    config=json.loads((root/'configs/pilot.json').read_text())
    records=[]
    semantic={}
    for name in ['semantic-candidates.jsonl','semantic-prose.jsonl']:
        semantic_path=root/'data/relevance'/name
        if semantic_path.exists():semantic.update({r['artifact_id']:r for r in read_jsonl(semantic_path)})
    for paper in config['papers']:
        relevance={}
        for name in ['index.jsonl','semantic-index.jsonl']:
            p=root/'data/relevance'/paper/name
            if p.exists():relevance.update({r['artifact_id']:r for r in read_jsonl(p)})
        relevance.update(semantic)
        for a in read_jsonl(root/'data/artifacts'/paper/'index.jsonl'):
            r=relevance.get(a['artifact_id'],{})
            a={**a,'relevance_record':r,'level':r.get('relevance','pending'),
               'source_image':str(Path(a['source_image']).relative_to(root)),
               'text_path':f"data/artifacts/{paper}/"+a['text_path']}
            a.pop('source_strips',None)
            records.append(a)
    payload=json.dumps(records,ensure_ascii=False).replace('<','\\u003c').replace('\u2028','\\u2028').replace('\u2029','\\u2029')
    html='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AksharaForge · Artifact library</title><style>
:root{font:16px/1.5 system-ui;color:#dfe8ef;background:#101a20}body{margin:0}header{padding:28px 4vw;border-bottom:1px solid #345}h1{margin:0;color:#90ddd3}p{max-width:900px;color:#bac8d2}.toolbar{display:flex;gap:12px;flex-wrap:wrap;padding:18px 4vw}input,select{font:inherit;color:inherit;background:#1b2b34;border:1px solid #456;border-radius:5px;padding:9px}input{min-width:270px}main{display:grid;grid-template-columns:minmax(260px,35%) 1fr;gap:20px;padding:0 4vw 30px}#list{max-height:70vh;overflow:auto}button{width:100%;text-align:left;background:#192a33;color:inherit;border:1px solid #304753;border-radius:5px;padding:12px;margin-bottom:8px;cursor:pointer}button:hover{border-color:#90ddd3}small{color:#9fb6c5}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#152630;padding:18px;border-radius:8px}a{color:#90ddd3}#detail{min-width:0}#detail h2{font-size:21px}.tag{color:#90ddd3}details{margin:12px 0}@media(max-width:780px){main{display:block}#list{max-height:35vh}header{padding:20px}}
</style><header><h1>AksharaForge</h1><p>Source artifacts → relevance → mathematical exercises. Every excerpt retains its original OCR and page evidence. Relevance is a selection judgment; it does not certify mathematical correctness.</p><nav><a href="QUICKSTART.md">Run the pipeline</a> · <a href="data/relevance/priority-shortlist.md">Relevance notes</a> · <a href="RUN_STATUS.md">Experiment status</a></nav></header>
<div class="toolbar"><input id="search" placeholder="Search mathematical content…" aria-label="Search"><select id="paper" aria-label="Paper"><option value="">All papers</option></select><select id="kind" aria-label="Artifact type"><option value="">All artifact types</option></select><select id="level" aria-label="Relevance"><option value="">All relevance levels</option><option>high</option><option>medium</option><option>low</option><option>uncertain</option><option>pending</option></select><span id="count"></span></div><main><div id="list"></div><article id="detail"><h2>Select an artifact</h2><p>Inspect exact source text, relevance rationale and provenance.</p></article></main><script id="records" type="application/json">PAYLOAD</script><script>
const rows=JSON.parse(document.getElementById('records').textContent),$=id=>document.getElementById(id);
for(const field of ['paper','kind'])for(const v of [...new Set(rows.map(r=>r[field==='paper'?'paper_id':'kind']))].sort()){let o=document.createElement('option');o.value=v;o.textContent=v;$(field).append(o)}
function add(parent,tag,text){let e=document.createElement(tag);e.textContent=text;parent.append(e);return e}
function show(r){let d=$('detail');d.replaceChildren();add(d,'h2',r.paper_id+' · page '+r.page+' · '+r.kind);add(d,'div','Relevance: '+r.level).className='tag';let a=add(d,'a','Open original page image');a.href=r.source_image;a.target='_blank';add(d,'span',' · ');a=add(d,'a','Open separate text artifact');a.href=r.text_path;a.target='_blank';add(d,'pre',r.content);add(d,'h3','Relevance assessment');let review=r.relevance_record;add(d,'p',review.rationale||'Assessment pending.');add(d,'p','Verifier: '+(review.verifier_feasibility||'Not assessed.'));if(review.exercise_types){let ul=add(d,'ul','');for(let idea of review.exercise_types)add(ul,'li',idea)}add(d,'p',review.source_uncertainty||'Source accuracy remains unverified.');add(d,'small',review.teacher_model?'Assessed by '+review.teacher_model:'Preliminary rule-based triage');let det=add(d,'details','');add(det,'summary','Source hashes and exact OCR offsets');add(det,'pre',JSON.stringify({artifact_id:r.artifact_id,content_sha256:r.content_sha256,source_sha256:r.source_sha256,ocr_sha256:r.ocr_sha256,page_body_sha256:r.page_body_sha256,field:r.field,start:r.start,end:r.end,status:r.status},null,2))}
function render(){let q=$('search').value.toLowerCase(),matches=rows.filter(r=>(!$('paper').value||r.paper_id===$('paper').value)&&(!$('kind').value||r.kind===$('kind').value)&&(!$('level').value||r.level===$('level').value)&&(!q||r.content.toLowerCase().includes(q)));$('count').textContent=matches.length+' artifacts';$('list').replaceChildren();for(let r of matches.slice(0,200)){let b=add($('list'),'button','');add(b,'strong',r.paper_id+' · p.'+r.page+' · '+r.kind);add(b,'br','');add(b,'small',r.level+' · '+r.content.slice(0,130));b.onclick=()=>show(r)}if(matches.length>200)add($('list'),'p','Showing first 200. Filter to narrow the results.')}
for(let id of ['search','paper','kind','level'])$(id).addEventListener('input',render);render();
</script></html>'''.replace('PAYLOAD',payload)
    path=root/'ARTIFACTS.html';path.write_text(html)
    return path
