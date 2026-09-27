"""Compare frozen evaluations only after checking paired protocols."""
from __future__ import annotations
import json
from pathlib import Path
from .io import read_jsonl


def compare(before: Path, after: Path) -> dict:
    configs=[json.loads((p/'config.json').read_text()) for p in [before,after]]
    matched=['papers','split','limit','attempts','seed','max_output_tokens','temperature','sandbox','base_url','step_penalty','problem_hash','completion_parser_sha256']
    if any(configs[0].get(k) != configs[1].get(k) for k in matched):
        raise ValueError('Evaluation protocols or dataset hashes differ')
    if configs[0]['split'] != 'test':raise ValueError('Uplift requires held-out evaluation')
    records=[]
    for path in [before,after]:
        source=path/'graded-episodes.jsonl'
        if not source.exists():source=path/'episodes.jsonl'
        records.append({(r['problem_id'],r['attempt']):r for r in read_jsonl(source)})
    if records[0].keys()!=records[1].keys():raise ValueError('Evaluation episode sets differ')
    if configs[0].get('limit') is not None:
        raise ValueError('Limited smoke runs are not complete held-out uplift evaluations')
    expected = len(configs[0]['papers']) * 20 * configs[0]['attempts']
    if len(records[0]) != expected:
        raise ValueError('Held-out evaluation is incomplete')
    for path in [before,after]:
        if not (path/'model-receipt.json').exists():
            raise ValueError('Both runs require model identity/precision receipts')
    receipts=[json.loads((p/'model-receipt.json').read_text()) for p in [before,after]]
    backends=[r.get('backend','ollama') for r in receipts]
    if backends[0]!=backends[1]:raise ValueError('Inference backend changed')
    if backends[0]=='huggingface':
        fields=['evaluator_sha256','system_prompt_sha256','torch','transformers','base_model','base_commit']
        precision=[r.get('precision') for r in receipts]
    else:
        fields=['worker_sha256','ollama_version']
        precision=[r.get('model',{}).get('details',{}).get('quantization_level') for r in receipts]
    for field in fields:
        if receipts[0].get(field)!=receipts[1].get(field):raise ValueError('Inference implementation changed')
    if not all(precision) or precision[0]!=precision[1]:raise ValueError('Inference precision missing or different')
    pairs=[]; unresolved=[]
    for key,a in records[0].items():
        b=records[1][key]
        if a['status']!='graded' or b['status']!='graded':unresolved.append(key);continue
        if a.get('verification')!=b.get('verification'):raise ValueError('Grading method changed')
        pairs.append({'problem_id':key[0],'paper_id':a['paper_id'],'verification':a.get('verification'),
                      'before':int(a['correct']),'after':int(b['correct']),
                      'delta':int(b['correct'])-int(a['correct'])})
    return {'paired_graded':len(pairs),'unresolved_pairs':len(unresolved),
            'accuracy_delta':sum(p['delta'] for p in pairs)/len(pairs) if pairs else None,
            'improved':sum(p['delta']==1 for p in pairs),'regressed':sum(p['delta']==-1 for p in pairs),
            'pairs':pairs,'caveat':'Protocol match is necessary but not sufficient: verify model lineage and matched quantization/precision; judge grades are probabilistic.'}
