"""Recompute a saved MATH-500 result without loading model weights."""
import argparse
import importlib.metadata
import json
from pathlib import Path

from .io import digest_file, read_jsonl
from .math500 import INSTRUCTION, SYSTEM, completed_prefix, load_benchmark, score, summary


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(dataset, run, plan_path, adapter=None, allow_partial=False):
    plan = json.loads(plan_path.read_text())
    protocol = json.loads((run / 'protocol.json').read_text())
    identity = json.loads((run / 'identity.json').read_text())
    runtime = json.loads((run / 'runtime.json').read_text())
    require(digest_file(dataset) == plan['dataset_sha256'] == protocol['dataset_sha256'],
            'Dataset fingerprint differs from frozen plan')
    for key, plan_key in [('benchmark', 'dataset'), ('dataset_revision', 'revision'),
                          ('questions', 'count'), ('evaluator_sha256', 'evaluator_sha256'),
                          ('precision', 'precision'), ('do_sample', 'do_sample'),
                          ('batch_size', 'batch_size'), ('max_new_tokens', 'max_new_tokens')]:
        require(protocol[key] == plan[plan_key], 'Frozen setting differs: ' + key)
    require(Path(protocol['base_model']).name == plan['model_revision'], 'Base model revision differs')
    require(protocol['attention'] == 'sdpa' and protocol['num_beams'] == 1
            and protocol['seed'] == 20260926, 'Frozen decoding settings differ')
    require(protocol['system'] == SYSTEM and protocol['instruction'] == INSTRUCTION,
            'Frozen prompts differ')
    require(digest_file(Path(__file__).with_name('math500.py')) == protocol['evaluator_sha256'],
            'Local evaluator differs from the frozen evaluator')
    for package in ('math-verify', 'sympy', 'latex2sympy2-extended', 'antlr4-python3-runtime'):
        require(importlib.metadata.version(package) == protocol['versions'][package],
                'Local scoring dependency differs: ' + package)
    if adapter:
        for filename, key in [('adapter_model.safetensors', 'adapter_sha256'),
                              ('adapter_config.json', 'adapter_config_sha256')]:
            require(digest_file(adapter / filename) == identity[key], 'Adapter fingerprint differs')
    else:
        require(identity['adapter_sha256'] is None, 'Trained result requires its adapter for audit')
    rows = load_benchmark(dataset)
    records = read_jsonl(run / 'episodes.jsonl')
    completed_prefix(rows, records)
    require(bool(records), 'No evaluated questions')
    require(allow_partial or len(records) == 500, 'Full 500-question result required')
    for question, record in zip(rows, records):
        require(all(record[key] == question[key] for key in ('subject', 'level')), 'Question metadata differs')
        require(type(record['correct']) is bool, 'Invalid correctness flag')
        require(type(record['generated_tokens']) is int and 0 <= record['generated_tokens'] <= 4096,
                'Invalid token count')
        require(record['done_reason'] in ('stop', 'length'), 'Invalid termination reason')
        rescored = score(question['answer'], record['response'])
        require(all(record[key] == value for key, value in rescored.items()),
                'Stored score differs from recomputed score: ' + question['unique_id'])
    result = summary(records)
    require(result == json.loads((run / 'summary.json').read_text()), 'Saved summary differs')
    complete_path = run / 'COMPLETE.json'
    if len(records) == 500:
        require(complete_path.exists() and json.loads(complete_path.read_text()) == result,
                'Completion receipt missing or inconsistent')
    else:
        require(not complete_path.exists(), 'Partial run has a completion receipt')
    names = ['protocol.json', 'identity.json', 'runtime.json', 'episodes.jsonl', 'summary.json']
    if complete_path.exists():
        names.append('COMPLETE.json')
    return {'verified': True, 'complete': result['complete'], 'summary': result,
            'runtime': runtime, 'identity': identity, 'paired_gain_claimed': False,
            'evidence_sha256': {name: digest_file(run / name) for name in names}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'run', 'plan', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    result = audit(args.dataset, args.run, args.plan, args.adapter, args.allow_partial)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({key: result[key] for key in ('verified', 'complete', 'summary')}, indent=2))


if __name__ == '__main__':
    main()
