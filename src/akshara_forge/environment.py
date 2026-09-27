"""Model-independent question/answer environment; graders stay outside the learner."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable

from .io import read_jsonl


def load_problems(root: Path, papers: list[str], split: str = "train") -> list[dict]:
    if split not in {"train", "test", "all"}:
        raise ValueError("split must be train, test or all")
    records = []
    for paper in papers:
        if not paper or Path(paper).name != paper:
            raise ValueError("invalid paper ID")
        records.extend(read_jsonl(root / "data" / "problems" / paper / f"{split}.jsonl"))
    return records


def public_problem(problem: dict) -> dict:
    # Deliberate allow-list: no reference, rubric, private verifier or source cards.
    return {k: problem[k] for k in ("problem_id", "paper_id", "kind", "prompt") if k in problem}


def parse_response(text: str) -> dict:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            stripped = "\n".join(lines[1:-1])
    try:
        value = json.loads(stripped, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    except (json.JSONDecodeError, ValueError):
        return {"steps": [], "answer": text, "parse_error": True}
    if isinstance(value, dict) and "answer" in value:
        steps = value.get("steps", [])
        if not isinstance(steps, list) or not all(isinstance(s, str) for s in steps):
            return {"steps": [], "answer": value["answer"], "parse_error": True}
        return {"steps": steps, "answer": value["answer"], "parse_error": False}
    return {"steps": [], "answer": value, "parse_error": False}


def _equal(actual: Any, expected: Any, tolerance: float) -> bool:
    if isinstance(expected, bool):
        return type(actual) is bool and actual == expected
    if isinstance(expected, (int, float)):
        if isinstance(actual, bool) or not isinstance(actual, (int, float, str)):
            return False
        try:
            a, b = float(actual), float(expected)
        except (ValueError, OverflowError):
            return False
        return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= tolerance
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            _equal(a, b, tolerance) for a, b in zip(actual, expected))
    if isinstance(expected, dict):
        return isinstance(actual, dict) and actual.keys() == expected.keys() and all(
            _equal(actual[k], value, tolerance) for k, value in expected.items())
    return type(actual) is type(expected) and actual == expected


def grade_answer(problem: dict, action: dict) -> dict:
    verification = problem["verification"]
    kind = verification["type"]
    if kind == "judge":
        return {"status": "needs_judge", "reward": None, "correct": None,
                "verification": "ungraded_proof", "feedback": "A strong teacher must evaluate this proof."}
    if kind not in {"numeric", "json_exact"}:
        raise ValueError(f"Unsupported verifier: {kind}")
    expected = verification.get("expected", problem["reference_answer"])
    tolerance = float(verification.get("tolerance", 1e-6)) if kind == "numeric" else 0.0
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Invalid verification tolerance")
    correct = _equal(action.get("answer"), expected, tolerance)
    return {"status": "graded", "reward": float(correct), "correct": correct,
            "verification": "programmatic_answer", "feedback": "Answer accepted." if correct else "Answer does not match the reference.",
            "proof_verified": False}


class MathEnvironment:
    """Standalone reset/step contract. A single answer is one episode in v0.1.

    No model code or model-authored shell command executes in this process.
    A registered teacher can grade proof tasks; missing grades never become zero.
    """
    def __init__(self, problems: list[dict], judge: Callable | None = None, step_penalty: float = 0.0):
        if step_penalty != 0:
            raise ValueError("Efficiency rewards are intentionally disabled for the pilot")
        self.problems = {p["problem_id"]: p for p in problems}
        if len(self.problems) != len(problems):
            raise ValueError("Duplicate problem IDs")
        self.judge = judge
        self.current = None
        self.done = True

    def reset(self, problem_id: str) -> dict:
        self.current = self.problems[problem_id]
        self.done = False
        return {"observation": public_problem(self.current), "max_submissions": 1,
                "response_format": {"steps": ["Your written mathematical steps"], "answer": "Final answer"}}

    def step(self, action: dict) -> dict:
        if self.done or self.current is None:
            raise RuntimeError("Call reset before submitting an answer")
        if not isinstance(action, dict) or "answer" not in action:
            raise ValueError("Action must contain answer")
        self.done = True
        result = grade_answer(self.current, action)
        if result["status"] == "needs_judge" and self.judge:
            result = self.judge(self.current, action)
        return {"terminated": True, "truncated": False, "steps_submitted": len(action.get("steps", [])), **result}
