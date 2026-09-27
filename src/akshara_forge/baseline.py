from __future__ import annotations

import datetime as dt
import hashlib
import json
import time
import urllib.request
from collections import Counter
from pathlib import Path

from .environment import MathEnvironment, load_problems, parse_response, public_problem
from .io import digest_file, read_jsonl, write_json
from .sandbox import run_learner


def summarize(records: list[dict]) -> dict:
    successes = [r for r in records if r.get("status") == "graded"]
    correct = sum(bool(r.get("correct")) for r in successes)
    by_paper = {}
    for paper in sorted({r['paper_id'] for r in records}):
        by_paper[paper] = {}
        for label, verification in [('finite_answers', 'programmatic_answer'), ('judged_proofs', 'teacher_judgment')]:
            subset = [r for r in successes if r['paper_id'] == paper and r.get('verification') == verification]
            wins = sum(bool(r.get('correct')) for r in subset)
            by_paper[paper][label] = {'graded':len(subset), 'correct':wins,
                                     'accuracy':wins/len(subset) if subset else None}
    return {"episodes": len(records), "status_counts": dict(Counter(r.get("status", "unknown") for r in records)),
            "graded": len(successes), "correct": correct,
            "graded_accuracy": correct / len(successes) if successes else None,
            "programmatic_graded": sum(r.get("verification") == "programmatic_answer" for r in successes),
            "teacher_graded": sum(r.get("verification") == "teacher_judgment" for r in successes),
            "generated_tokens": sum(r.get("eval_count") or 0 for r in records),
            "truncated_generations": sum(r.get('done_reason') == 'length' for r in records),
            "parse_failures": sum(bool(r.get('action', {}).get('parse_error')) for r in records),
            "written_steps_missing": sum(not r.get('action', {}).get('steps') for r in records),
            "by_paper": by_paper,
            "note": "Pending proofs and transport failures are not silently scored as incorrect. Programmatic grades certify final answers, not written proofs."}


def run_baseline(root: Path, *, papers: list[str], split: str = "train", model: str = "qwen2.5:7b-instruct",
                 output: Path, limit: int | None = None, attempts: int = 1, seed: int = 20260926,
                 max_output_tokens: int = 768, temperature: float = 0.2, sandbox: str = "auto",
                 base_url: str = "http://127.0.0.1:11434") -> dict:
    problems = load_problems(root, papers, split)
    # Round-robin papers so a smoke run exercises all domains.
    groups = [[p for p in problems if p["paper_id"] == paper] for paper in papers]
    problems = [g[i] for i in range(max(map(len, groups), default=0)) for g in groups if i < len(g)]
    if limit is not None:
        problems = problems[:limit]
    if not problems:
        raise ValueError("No problems selected")
    if attempts < 1 or max_output_tokens < 1:
        raise ValueError("attempts and output tokens must be positive")
    output.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(base_url.rstrip('/') + '/api/tags', timeout=10) as response:
        tags = json.load(response)
    identity = next((m for m in tags['models'] if m['name'] == model or m['name'] == model + ':latest'), None)
    if identity is None:
        raise ValueError(f'Model {model!r} is not installed in Ollama')
    with urllib.request.urlopen(base_url.rstrip('/') + '/api/version', timeout=10) as response:
        version = json.load(response)['version']
    receipt = {'model':identity, 'ollama_version':version,
               'worker_sha256':digest_file(Path(__file__).with_name('worker.py'))}
    receipt_path = output / 'model-receipt.json'
    if receipt_path.exists():
        saved = json.loads(receipt_path.read_text())
        if (saved['model']['digest'], saved['worker_sha256'], saved['ollama_version']) != (identity['digest'], receipt['worker_sha256'], version):
            raise ValueError('Model weights or inference implementation changed; use a new run directory')
    else:
        write_json(receipt_path, receipt)
    config = {"model": model, "papers": papers, "split": split, "limit": limit, "attempts": attempts,
              "seed": seed, "max_output_tokens": max_output_tokens, "temperature": temperature,
              "sandbox": sandbox, "base_url": base_url, "step_penalty": 0.0,
              "problem_hash": hashlib.sha256(json.dumps(problems, sort_keys=True).encode()).hexdigest()}
    config_path = output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Run directory has a different frozen configuration; choose a new output directory")
    write_json(config_path, config)
    records_path = output / "episodes.jsonl"
    prior = read_jsonl(records_path) if records_path.exists() else []
    completed = {(r["problem_id"], r["attempt"]) for r in prior}
    env = MathEnvironment(problems)
    for problem in problems:
        for attempt in range(attempts):
            if (problem["problem_id"], attempt) in completed:
                continue
            env.reset(problem["problem_id"])
            start = time.monotonic()
            record = {"problem_id": problem["problem_id"], "paper_id": problem["paper_id"],
                      "kind": problem["kind"], "split": problem["split"], "attempt": attempt,
                      "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(), "model": model}
            try:
                result = run_learner({"problem": public_problem(problem), "model": model, "base_url": base_url,
                                      "seed": seed + attempt, "max_output_tokens": max_output_tokens,
                                      "temperature": temperature}, backend=sandbox)
                action = parse_response(result["response"])
                grade = env.step(action)
                record.update(result)
                record.update(grade)
                record["action"] = action
            except Exception as exc:
                record.update({"status": "runtime_error", "error": str(exc), "reward": None, "correct": None})
            record["wall_seconds"] = round(time.monotonic() - start, 3)
            with records_path.open("a") as f:
                f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                f.flush()
            prior.append(record)
            write_json(output / "summary.json", summarize(prior))
            print(f"{problem['problem_id']} attempt={attempt} status={record['status']} reward={record.get('reward')} {record['wall_seconds']}s", flush=True)
    summary = summarize(prior)
    write_json(output / "summary.json", summary)
    return summary
