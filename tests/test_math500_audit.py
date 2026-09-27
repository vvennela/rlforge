import importlib.metadata
import json
from pathlib import Path

import pytest

from akshara_forge import math500, math500_audit
from akshara_forge.io import digest_file


@pytest.fixture
def saved_run(tmp_path, monkeypatch):
    pytest.importorskip('math_verify')
    dataset = tmp_path / 'dataset.jsonl'
    rows = [dict(unique_id=str(i), problem='Compute 1+1.', answer='2', subject='Algebra', level=1)
            for i in range(500)]
    dataset.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    run = tmp_path / 'run'
    run.mkdir()
    plan = dict(dataset='HuggingFaceH4/MATH-500', revision='frozen', count=500,
                dataset_sha256=digest_file(dataset), evaluator_sha256=digest_file(Path(math500.__file__)),
                model_revision='pinned', precision='bfloat16', do_sample=False,
                batch_size=8, max_new_tokens=4096)
    plan_path = tmp_path / 'plan.json'
    plan_path.write_text(json.dumps(plan))
    protocol = {k: plan[k] for k in ('dataset_sha256', 'evaluator_sha256', 'precision',
                                    'do_sample', 'batch_size', 'max_new_tokens')}
    protocol.update(benchmark=plan['dataset'], dataset_revision=plan['revision'], questions=500,
                    base_model='/models/pinned', attention='sdpa', num_beams=1, seed=20260926,
                    system=math500.SYSTEM, instruction=math500.INSTRUCTION,
                    versions={p: importlib.metadata.version(p) for p in
                              ('math-verify', 'sympy', 'latex2sympy2-extended', 'antlr4-python3-runtime')})
    (run / 'protocol.json').write_text(json.dumps(protocol))
    (run / 'identity.json').write_text(json.dumps({'adapter_sha256': None}))
    (run / 'runtime.json').write_text(json.dumps({'gpu': 'fixture'}))
    records = [dict(unique_id=str(i), subject='Algebra', level=1, response=r'\boxed{2}',
                    correct=True, prediction_extracted=True, gold_parsed=['2'], prediction_parsed=['2'],
                    generated_tokens=5, done_reason='stop') for i in range(500)]
    # Exercise record/receipt validation separately from symbolic parsing (tested in test_math500).
    monkeypatch.setattr(math500_audit, 'score', lambda a, b: dict(correct=True,
        prediction_extracted=True, gold_parsed=['2'], prediction_parsed=['2']))

    def save(count=500):
        (run / 'episodes.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records[:count]))
        result = math500.summary(records[:count])
        (run / 'summary.json').write_text(json.dumps(result))
        if count == 500:
            (run / 'COMPLETE.json').write_text(json.dumps(result))
    save()
    return dataset, run, plan_path, records, save


def test_complete_saved_result(saved_run):
    dataset, run, plan, _, _ = saved_run
    result = math500_audit.audit(dataset, run, plan)
    assert result['verified'] and result['complete']
    assert result['summary']['correct'] == 500
    assert not result['paired_gain_claimed']


def test_partial_requires_explicit_label(saved_run):
    dataset, run, plan, _, save = saved_run
    (run / 'COMPLETE.json').unlink()
    save(8)
    with pytest.raises(ValueError, match='Full 500'):
        math500_audit.audit(dataset, run, plan)
    assert not math500_audit.audit(dataset, run, plan, allow_partial=True)['complete']


@pytest.mark.parametrize('tamper', ['score', 'order', 'summary', 'completion', 'prompt', 'dataset'])
def test_rejects_tampered_result(saved_run, tamper):
    dataset, run, plan, records, save = saved_run
    if tamper == 'score':
        records[0]['correct'] = False
        save()
    elif tamper == 'order':
        records[0]['unique_id'] = '1'
        save()
    elif tamper == 'summary':
        (run / 'summary.json').write_text('{}')
    elif tamper == 'completion':
        (run / 'COMPLETE.json').unlink()
    elif tamper == 'prompt':
        p = run / 'protocol.json'
        value = json.loads(p.read_text())
        value['instruction'] = 'A changed prompt'
        p.write_text(json.dumps(value))
    else:
        dataset.write_text(dataset.read_text() + '\n')
    with pytest.raises(ValueError):
        math500_audit.audit(dataset, run, plan)
