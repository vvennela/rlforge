# AksharaForge

Turn scientific papers into source-grounded mathematics exercises and standalone reinforcement-learning environments.

The pipeline preserves complete OCR evidence, extracts equations, statements and visual artifacts, assesses their relevance, generates exercises, and independently reviews the proposed solutions before exporting an environment. The learner submits written steps and an answer; the controller grades its work and can use those rewards to update Qwen through GRPO and LoRA.

## Quick start

Requires Python 3.11+, Tesseract for scanned pages, and an authenticated Codex CLI with access to the configured teacher models (`gpt-6-luna` and `gpt-6-astra`). Learner inference uses Ollama. The trainer requires a separate CUDA environment; Ollama does not update weights.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock
pip install --no-deps -e .

forge from-pdf /path/to/paper.pdf --count 5 --output work/paper

ollama pull qwen2.5:7b-instruct
forge --root work/paper/environment baseline --split train --output work/paper/qwen
forge --root work/paper/environment grade work/paper/qwen
```

The output directory must be new. Failed runs retain their evidence for inspection. Teacher calls use the signed-in Codex account; review the selected paper before sending its contents to that service. Sandboxed learner inference uses macOS Seatbelt or Docker.

## Outputs

- `canonical-v2/`: complete page text, page images, OCR strips, hashes and provenance.
- `data/artifacts/` and `data/visual-artifacts/`: separately indexed mathematical and visual evidence.
- `data/relevance/`: heuristic triage and model relevance assessments, distinguished in their records.
- `environment/generation/` and `environment/reviews/`: generated exercises and independent review records.
- `environment/data/problems/`: accepted problems with source references and controller-side solutions.
- `environment/standalone/`: a standard-library-only reset/step environment bundle.
- `ARTIFACTS.html`: a local evidence browser.

Reference solutions and grading rubrics belong to the controller. Give learners only the public observations returned by the environment, not the full exported dataset.

## Training and evaluation

`training.py` provides a Qwen GRPO/LoRA trainer, fixed curriculum ordering, rollout traces, checkpointing and measured adapter changes. `reward_service.py` allows a remote GPU trainer to use a local grading controller. Process rewards combine final correctness with coverage of predefined mathematical milestones; they do not reward answer length or formatting.

`hf_evaluate.py`, `normalize_run.py` and `compare.py` support matched before/after evaluation with recorded model, parser and inference settings. Their current pilot protocol expects 80 training problems and 20 held-out problems per paper. Keep complete task families separate and freeze the evaluation set before training.

The automatic `from-pdf` compiler currently requests 1–20 exercises and exports accepted exercises as training/development data. It does **not** automatically construct the 100-problem benchmark required by the pilot trainer. It creates a mathematical question-answer environment, not an arbitrary engineering simulator. Model-judged proofs are not formally verified, and successful environment generation alone does not establish model uplift.

Inspect module-specific options with:

```bash
python -m akshara_forge.training --help
python -m akshara_forge.reward_service --help
python -m akshara_forge.hf_evaluate --help
```

## Development

```bash
pytest -q
```

Tests cover source integrity, reference isolation, sandbox behavior, grading, curriculum contracts, output parsing and evaluation compatibility. The optional archive integration fixture is skipped when local paper artifacts are absent.

This repository contains pipeline code and tests. Papers, generated datasets, credentials, model weights, private run traces and cloud connection settings are excluded.
