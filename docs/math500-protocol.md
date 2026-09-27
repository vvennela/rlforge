# External MATH-500 evaluation

The original Qwen2.5-7B-Instruct baseline scored **396/500 (79.2%)** on an
NVIDIA L40S. Its 500 saved responses were independently rescored with the pinned
verifier and reproduced the recorded result exactly.

The fixed, 80-step **math-trained adapter scored 395/500 (79.0%)** on an NVIDIA
A10G. All 500 questions completed; four responses reached the token limit and
none failed answer extraction. Rescoring every saved response reproduced this result. It is separate from the coding and LEGO adapters. Its SHA-256 is
`77783a675d4abb84568de75b7bb5c2f3c2863e14a6af13aca1933fbdd39bdc61`, matching both the
archived final math adapter and checkpoint 80. These experiments use curated pilot
training tasks; upload-generated environments are a separate pipeline output.

## Frozen settings

- All 500 questions, dataset revision `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`.
- Dataset SHA-256 `35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132`.
- Base model revision `a09a35458c702b33eeacc393d103063234e8bc28`.
- One greedy completion per question, BF16, SDPA, batch size eight, 4,096 new tokens.
- Seed 20260926; fixed system and question instructions in `math500.py`.
- Math-Verify 0.9.0 final-answer equivalence; no teacher grading of benchmark traces.
- Questions, answers, and scores never guide training, generation, or checkpoint selection.

The two GPU runtimes differ, so the website reports their scores separately.
It does not claim a matched before/after gain. The strict comparison function
rejects a protocol or runtime mismatch. An interrupted run reports the number
actually evaluated and remains partial.

## Reproduce the saved-result audit

```sh
PYTHONPATH=src python -m akshara_forge.math500_audit \
  --dataset /path/to/math500/dataset/test.jsonl \
  --run /path/to/math500/after \
  --plan /path/to/frozen-math500-plan.json \
  --adapter /path/to/math500/adapter \
  --output /path/to/math500/after-audit.json
```

For the original baseline, omit `--adapter`. The audit checks dataset and adapter
fingerprints, frozen settings and prompts, question order, verifier versions,
recomputed per-question scores, summary arithmetic, and the completion receipt.
It defaults to requiring all 500 questions. Use `--allow-partial` only to audit an
explicitly labeled snapshot of an incomplete run; that flag never marks it complete.

## Final campaign result

| Model | GPU | Correct | Accuracy | Questions evaluated |
|---|---|---:|---:|---:|
| Original Qwen 2.5 7B | L40S | 396 | 79.2% | 500 |
| Math-trained Qwen 2.5 7B | A10G | 395 | 79.0% | 500 |

The campaign did not demonstrate the requested external benchmark improvement.
The paper-specific holdout remains a separate result: 2/20 → 5/20. The runtime
mismatch prevents a matched causal claim about the one-question difference here.

The [baseline audit](results/math500-before-audit.json) and
[trained-model audit](results/math500-after-audit.json) include recomputed scores,
runtime provenance, adapter identities, and evidence fingerprints. These public
reports contain no benchmark prompts or reference answers.
