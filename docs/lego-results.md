# LEGO bridge repair result

**Qwen 2.5 7B improved from 3/20 to 7/20 successful repairs: +20 percentage points.**

Mean target geometry overlap increased from 36.74% to 71.09% (+34.35 percentage points). Four failures became successful repairs; no previously successful case regressed.

The final adapter follows 50 optimizer batches, with two sampled eight-turn trajectories per batch. Forty-two batches had reward contrast; 2,523,134 adapter parameters changed. Training resumed the preserved step-10 adapter, optimizer, and RNG state. The baseline uses the original pinned base model.

All twenty frozen held-out cases use identical greedy decoding, tools, feedback, eight-turn limits, and 256-token turn budgets before and after. This benchmark measures missing or misplaced bridge-upright repairs; the surrounding bridge is fixed. The final planned checkpoint was evaluated once, without selecting it from held-out scores.

The website exposes every case in frozen dataset order, the trained repair output, and both models’ recorded actions and rewards. Its left-hand viewer shows the original deck-and-supports prototype, explicitly distinguished from the selected repair task.

| Case | Base success | Trained success | Base overlap | Trained overlap |
|---|---:|---:|---:|---:|
| 080 | No | No | 54.5% | 54.5% |
| 081 | No | No | 23.1% | 23.1% |
| 082 | Yes | Yes | 100.0% | 100.0% |
| 083 | Yes | Yes | 100.0% | 100.0% |
| 084 | No | No | 37.5% | 37.5% |
| 085 | No | No | 30.0% | 60.0% |
| 086 | No | No | 0.0% | 66.7% |
| 087 | Yes | Yes | 100.0% | 100.0% |
| 088 | No | No | 77.8% | 77.8% |
| 089 | No | Yes | 0.0% | 100.0% |
| 090 | No | No | 0.0% | 0.0% |
| 091 | No | No | 0.0% | 37.5% |
| 092 | No | No | 76.9% | 76.9% |
| 093 | No | No | 75.0% | 75.0% |
| 094 | No | No | 0.0% | 77.8% |
| 095 | No | Yes | 0.0% | 100.0% |
| 096 | No | No | 60.0% | 60.0% |
| 097 | No | Yes | 0.0% | 100.0% |
| 098 | No | No | 0.0% | 75.0% |
| 099 | No | Yes | 0.0% | 100.0% |

## Evidence fingerprints

Run: `long-004`. Source dataset: `dataset-v1` (80 train, 20 held out). Full trajectories, checkpoints, and optimizer evidence are retained in the run archive.

```json
{
  "protocol.json": "97c4707262e9757fc20acd203a2bb9bd44e479d038dab66975612768907e23de",
  "completion.json": "b5d94edc4e4f41ec8915e37440cdba2285f50003680194a9c914f9c0616472e4",
  "before/summary.json": "191274af8d09fbd459600660e546f2b49e9e894f0dae3628e77351fec732c8df",
  "after/summary.json": "7782632277b3b2e6f810eeb719bd0c689954251ad3c1779c083b62d633b5ab78",
  "before/traces.jsonl": "4d1e5c76b32cf164f46a0faa52fdd281e05e7135729cdb89043fb2fc3a8be23e",
  "after/traces.jsonl": "583fcf24e38091f611d8af260a171e430ed0c6e5ef931766bf562e6de781b8b9",
  "adapter/adapter_model.safetensors": "b717c6d6519c9cb47f43198d88b1b1dc0fb99a19c5c280417ebee689e0a9907d"
}
```

## Independent weight audit

A separate CPU comparison of all 112 saved LoRA tensors against the initial
adapter confirmed **2,523,134 changed parameters** and delta L2
**0.16287758055618165**, exactly matching the training receipt. The adapter hash
matched the archived file above. This check loads only adapter tensors, not the
base model; use `python -m akshara_forge.adapter_audit` to reproduce it.
