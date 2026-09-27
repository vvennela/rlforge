"""Apply identical conservative answer extraction to saved before/after outputs.

Original rollouts remain unchanged; no model is called and no answer is repaired.
The derived evaluation declares its parser hash and exact source-file hash.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from . import completion
from .baseline import summarize
from .environment import grade_answer,load_problems
from .io import digest_file,read_jsonl,write_json,write_jsonl


def normalize(root: Path,source: Path,output: Path):
    if output.exists():raise ValueError('Derived evaluation output must be new')
    config=json.loads((source/'config.json').read_text())
    problems={p['problem_id']:p for p in load_problems(root,config['papers'],config['split'])}
    records=read_jsonl(source/'episodes.jsonl')
    if len(records)!=20*len(config['papers'])*config['attempts']:
        raise ValueError('Wait until the complete held-out rollout is saved')
    parser_hash=digest_file(Path(completion.__file__))
    config['completion_parser_sha256']=parser_hash
    write_json(output/'config.json',config)
    receipt=json.loads((source/'model-receipt.json').read_text())
    receipt['completion_parser_sha256']=parser_hash
    write_json(output/'model-receipt.json',receipt)
    for r in records:
        action=completion.parse_completion(r['response'])
        r['action']=action;r.update(grade_answer(problems[r['problem_id']],action))
    write_jsonl(output/'episodes.jsonl',records)
    write_json(output/'summary.json',summarize(records))
    write_json(output/'derivation.json',{'source':str(source),'source_episodes_sha256':digest_file(source/'episodes.jsonl'),
               'parser_sha256':parser_hash,'new_inference':False,'answer_values_repaired':False,
               'note':'Complete trailing JSON is extracted verbatim; remaining written work is preserved as paragraphs. Incomplete final JSON is not repaired.'})
    return summarize(records)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(normalize(a.root,a.source,a.output),indent=2))
