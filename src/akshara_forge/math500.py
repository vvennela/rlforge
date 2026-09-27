"""Frozen MATH-500 evaluation, with no teacher or paper-training dependencies."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

from .io import digest_file, read_jsonl, write_json

DATASET_REVISION = '6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be'
SYSTEM = 'You are a helpful mathematics assistant.'
INSTRUCTION = r'Solve the following problem. Show your work and put your final answer in \boxed{}.'


def load_benchmark(path: Path) -> list[dict]:
    rows = read_jsonl(path)
    if len(rows) != 500 or len({r['unique_id'] for r in rows}) != 500:
        raise ValueError('MATH-500 requires exactly 500 unique problems')
    for r in rows:
        if not all(isinstance(r[k], str) and r[k] for k in ('problem', 'answer', 'unique_id')):
            raise ValueError('Invalid benchmark record')
    return rows


def messages(problem: dict) -> list[dict]:
    # Reference answers and solutions are never included in model input.
    return [{'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': INSTRUCTION + '\n\n' + problem['problem']}]


def score(answer: str, response: str) -> dict:
    from math_verify import parse, verify, LatexExtractionConfig
    gold = parse(r'\boxed{' + answer + '}', extraction_config=[LatexExtractionConfig()])
    if not gold:
        raise ValueError('Reference answer extraction failed')
    prediction = parse(response)
    return {'correct': bool(verify(gold, prediction)) if prediction else False,
            'prediction_extracted': bool(prediction),
            'gold_parsed': [str(x) for x in gold],
            'prediction_parsed': [str(x) for x in prediction]}


def summary(records: list[dict]) -> dict:
    n = len(records)
    groups = {}
    for field in ('subject', 'level'):
        groups[field] = {}
        for row in records:
            key = str(row[field])
            item = groups[field].setdefault(key, {'completed': 0, 'correct': 0})
            item['completed'] += 1
            item['correct'] += int(row['correct'])
    return {'completed': n, 'expected': 500, 'complete': n == 500,
            'correct': sum(r['correct'] for r in records),
            'accuracy': sum(r['correct'] for r in records) / n if n else None,
            'extraction_failures': sum(not r['prediction_extracted'] for r in records),
            'truncated': sum(r['done_reason'] == 'length' for r in records),
            'generated_tokens': sum(r['generated_tokens'] for r in records),
            'breakdown': groups,
            'metric': 'single greedy completion; Math-Verify final-answer correctness',
            'target': 'more than 10 percentage points absolute accuracy improvement'}


def freeze(path: Path, value: dict) -> None:
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'Frozen evaluation configuration differs: {path.name}')
    else:
        write_json(path, value)


def completed_prefix(rows: list[dict], records: list[dict]) -> int:
    ids = [r['unique_id'] for r in records]
    if ids != [r['unique_id'] for r in rows[:len(records)]]:
        raise ValueError('Saved episodes are not a unique ordered benchmark prefix')
    return len(records)


def evaluate(dataset: Path, model_name: str, output: Path, adapter: Path | None = None,
             batch_size: int = 8, max_tokens: int = 4096) -> dict:
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

    if batch_size < 1 or max_tokens < 1:
        raise ValueError('Batch size and token budget must be positive')
    rows = load_benchmark(dataset)
    # Validate every reference before spending GPU time on this protocol.
    for row in rows:
        if not score(row['answer'], r'\boxed{' + row['answer'] + '}')['correct']:
            raise ValueError('Reference self-check failed: ' + row['unique_id'])
    protocol = {'benchmark': 'HuggingFaceH4/MATH-500', 'dataset_revision': DATASET_REVISION,
                'dataset_sha256': digest_file(dataset), 'questions': 500,
                'base_model': model_name, 'precision': 'bfloat16', 'attention': 'sdpa',
                'batch_size': batch_size, 'max_new_tokens': max_tokens,
                'do_sample': False, 'num_beams': 1, 'seed': 20260926,
                'system': SYSTEM, 'instruction': INSTRUCTION,
                'evaluator_sha256': digest_file(Path(__file__)),
                'versions': {p: importlib.metadata.version(p) for p in
                             ('torch', 'transformers', 'math-verify', 'sympy',
                              'latex2sympy2-extended', 'antlr4-python3-runtime')}}
    freeze(output/'protocol.json', protocol)
    identity = {'adapter_sha256': digest_file(adapter/'adapter_model.safetensors') if adapter else None,
                'adapter_config_sha256': digest_file(adapter/'adapter_config.json') if adapter else None,
                'model_config_sha256': digest_file(Path(model_name)/'config.json')
                    if (Path(model_name)/'config.json').exists() else None}
    freeze(output/'identity.json', identity)
    records = read_jsonl(output/'episodes.jsonl') if (output/'episodes.jsonl').exists() else []
    completed_prefix(rows, records)
    if len(records) == 500:
        result = summary(records)
        write_json(output/'summary.json', result)
        write_json(output/'COMPLETE.json', result)
        return result
    set_seed(protocol['seed'])
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16,
              attn_implementation='sdpa', trust_remote_code=False).to('cuda')
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    model.eval()
    freeze(output/'runtime.json', {'base_commit': getattr(model.config, '_commit_hash', None),
           'chat_template_sha256': hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
           'gpu': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda})
    # Preserve batch membership when resuming after a partial batch write.
    start_index = (len(records) // batch_size) * batch_size
    for start in range(start_index, len(rows), batch_size):
        batch = rows[start:start+batch_size]
        prompts = [tokenizer.apply_chat_template(messages(r), tokenize=False,
                   add_generation_prompt=True) for r in batch]
        inputs = tokenizer(prompts, padding=True, return_tensors='pt').to('cuda')
        begin = time.monotonic()
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False,
                    num_beams=1, pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id)
        elapsed = time.monotonic() - begin
        for offset, row in enumerate(batch):
            if start + offset < len(records):
                continue
            ids = generated[offset, inputs['input_ids'].shape[1]:].tolist()
            stopped = tokenizer.eos_token_id in ids
            if stopped:
                ids = ids[:ids.index(tokenizer.eos_token_id)]
            response = tokenizer.decode(ids, skip_special_tokens=True)
            record = {'unique_id': row['unique_id'], 'subject': row['subject'], 'level': row['level'],
                      'response': response, 'generated_tokens': len(ids),
                      'done_reason': 'stop' if stopped else 'length', 'batch_seconds': elapsed,
                      'timestamp': dt.datetime.now(dt.timezone.utc).isoformat(),
                      **score(row['answer'], response)}
            records.append(record)
        # Atomic batch snapshots leave a valid prefix even if the host stops.
        snapshot = output/'episodes.jsonl.tmp'
        with snapshot.open('w') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n')
        snapshot.replace(output/'episodes.jsonl')
        result = summary(records)
        write_json(output/'summary.json', result)
        print(json.dumps({k: result[k] for k in ('completed', 'correct', 'accuracy', 'truncated')}), flush=True)
    write_json(output/'COMPLETE.json', summary(records))
    return summary(records)


def compare(before: Path, after: Path) -> dict:
    for filename in ('protocol.json', 'runtime.json'):
        if json.loads((before/filename).read_text()) != json.loads((after/filename).read_text()):
            raise ValueError('Benchmark protocol/runtime mismatch')
    a, b = (read_jsonl(p/'episodes.jsonl') for p in (before, after))
    if len(a) != 500 or len(b) != 500:
        raise ValueError('Both full 500-question evaluations must finish')
    if len({r['unique_id'] for r in a}) != 500 or [r['unique_id'] for r in a] != [r['unique_id'] for r in b]:
        raise ValueError('Benchmark question sets differ')
    improved = sum(not x['correct'] and y['correct'] for x, y in zip(a, b))
    regressed = sum(x['correct'] and not y['correct'] for x, y in zip(a, b))
    delta = (improved-regressed)/500
    return {'before': summary(a), 'after': summary(b), 'accuracy_delta': delta,
            'percentage_point_gain': 100*delta, 'improved': improved, 'regressed': regressed,
            'target_exceeded': improved-regressed > 50,
            'interpretation': 'Paired final-answer benchmark result, not verification of written proofs.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-tokens', type=int, default=4096)
    args = parser.parse_args()
    evaluate(args.dataset, args.model, args.output, args.adapter, args.batch_size, args.max_tokens)


if __name__ == '__main__':
    main()
