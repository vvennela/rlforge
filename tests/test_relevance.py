import json
import tempfile
import unittest
from pathlib import Path

from akshara_forge.relevance import build_full_coverage, classify_all, classify_artifact


ROOT = Path(__file__).resolve().parents[1]


class RelevanceTriageTests(unittest.TestCase):
    def test_core_statement_is_high_and_bound_to_sources(self):
        row = {
            "artifact_id": "columbia-1006-p0008-statement_candidate-demo",
            "paper_id": "columbia-1006", "page": 8, "kind": "statement_candidate",
            "content": "Lemma 6.2. For any admissible cost function construct a monotone admissible function.",
            "content_sha256": "c" * 64, "source_sha256": "s" * 64,
            "ocr_sha256": "o" * 64, "page_body_sha256": "p" * 64,
            "source_image": "/source/page-0008.png",
        }
        result = classify_artifact(row)
        self.assertEqual(result["relevance"], "high")
        self.assertEqual(result["content_sha256"], row["content_sha256"])
        self.assertEqual(result["source_sha256"], row["source_sha256"])
        self.assertIn("finite graphs", result["verifier_feasibility"].lower())
        self.assertIn("general proof", result["verifier_feasibility"].lower())

    def test_unknown_paper_uses_general_mathematics_guidance(self):
        row = {
            "artifact_id": "new-paper-p0001-equation-x", "paper_id": "new-paper", "page": 1,
            "kind": "equation", "content": "$$ x^2 = 4 $$", "content_sha256": "c" * 64,
            "source_sha256": "s" * 64, "ocr_sha256": "o" * 64, "page_body_sha256": "p" * 64,
        }
        result = classify_artifact(row)
        self.assertIn("finite numerical example", result["exercise_types"][1])
        self.assertIn("independent mathematical review", result["verifier_feasibility"])

    def test_full_coverage_join_marks_model_and_rule_assessors(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            artifacts = base / "artifacts"
            output = base / "relevance"
            (artifacts / "columbia-1006").mkdir(parents=True)
            (output / "columbia-1006").mkdir(parents=True)
            source_rows = [
                {"artifact_id": "model-row", "content_sha256": "a", "source_sha256": "s", "ocr_sha256": "o", "page_body_sha256": "p"},
                {"artifact_id": "rule-row", "content_sha256": "b", "source_sha256": "s", "ocr_sha256": "o", "page_body_sha256": "p"},
            ]
            (artifacts / "columbia-1006" / "index.jsonl").write_text("".join(json.dumps(x) + "\n" for x in source_rows))
            rule_rows = [
                {**source_rows[0], "relevance": "medium", "rationale": "rule evidence", "paper_id": "columbia-1006"},
                {**source_rows[1], "relevance": "low", "rationale": "rule evidence", "paper_id": "columbia-1006"},
            ]
            (output / "columbia-1006" / "index.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rule_rows))
            model_row = {**source_rows[0], "relevance": "high", "rationale": "model evidence", "exercise_types": [], "verifier_feasibility": "finite", "prerequisites": [], "source_uncertainty": "OCR", "decision_method": "Luna", "teacher_model": "gpt-6-luna"}
            (output / "semantic-candidates.jsonl").write_text(json.dumps(model_row) + "\n")
            counts = build_full_coverage(artifacts, output)
            joined = [json.loads(x) for x in (output / "full-coverage-index.jsonl").read_text().splitlines()]
            self.assertEqual(counts["high"], 1)
            self.assertEqual(counts["low"], 1)
            self.assertEqual(joined[0]["assessor"], "gpt-6-luna")
            self.assertEqual(joined[0]["heuristic_relevance"], "medium")
            self.assertEqual(joined[1]["assessor"], "rule_based_preliminary_triage")

    @unittest.skipUnless((ROOT / "data/artifacts").exists(), "Optional local paper-artifact integration fixture")
    def test_all_extracted_artifacts_receive_one_source_bound_decision(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            counts = classify_all(ROOT / "data" / "artifacts", ROOT / "canonical-v2", out)
            self._assert_coverage(out, counts)

    def _assert_coverage(self, out, counts):
        self.assertEqual(sum(counts.values()), 1188)
        total = 0
        for paper, count in counts.items():
            source_rows = [json.loads(x) for x in (ROOT / "data" / "artifacts" / paper / "index.jsonl").read_text().splitlines()]
            judged = [json.loads(x) for x in (out / paper / "index.jsonl").read_text().splitlines()]
            self.assertEqual(len(judged), count)
            self.assertEqual([r["artifact_id"] for r in judged], [r["artifact_id"] for r in source_rows])
            for src, dst in zip(source_rows, judged):
                self.assertEqual(dst["content_sha256"], src["content_sha256"])
                self.assertEqual(dst["source_sha256"], src["source_sha256"])
                self.assertIn(dst["relevance"], {"high", "medium", "low", "uncertain"})
                self.assertTrue(dst["rationale"])
            total += len(judged)
        self.assertEqual(total, 1188)


if __name__ == "__main__":
    unittest.main()
