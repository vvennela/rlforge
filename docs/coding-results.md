# Coding result

**Qwen 2.5 7B completed 80 training batches. Held-out program correctness remained 0/20 → 0/20.**

Mean exact-test pass rate also stayed at 0%. Mean exact-field accuracy fell from
2.36% to 0.42%. There was no held-out coding uplift from this run.

Each of the twenty programs faced 24 hidden tests in the Vultr gVisor sandbox.
The before and after evaluations used the same frozen specification order,
greedy decoding, 1536-token budget, and 30-second execution limit. No evaluation
completion reached the token limit. The final planned adapter was evaluated once;
held-out scores did not select checkpoints or revise the training protocol.

## What training changed

The 320 sampled training programs produced 55 positive rewards and no fully
correct programs. Thirty-seven of eighty batches had reward contrast. Training
errors included 97 timeouts, 18 syntax errors, five invalid outputs, and three
value errors; the other 197 programs executed and were graded for correctness.
This shows that the chosen tasks and reward produced little successful behavior
to reinforce. More updates alone are not evidence of improvement.

The final adapter changed **2,523,028 parameters**, with delta L2
**0.27353035899268224**. An independent CPU comparison of all 112 saved tensors
confirmed both values. Saved weights and execution outcomes are separate checks:
the first establishes a weight update; the second measures task performance.

Before evaluation, six programs timed out; afterward, three timed out and two
had syntax errors. The remaining programs executed but failed exact correctness.
The verifier was checked against independent reference solvers, correct sandbox
programs, and eight incorrect implementation mutations before this experiment.

## Verified evidence

Run `code-002` starts from the pinned original Qwen2.5-7B-Instruct base. The
[experiment protocol](coding-experiment.md) records the training-only reward
revision and its preceding run. Recovery from the interrupted step 26 resumed
checkpoint 25 with its optimizer and RNG state; the trainer and settings stayed
unchanged.

The result audit matched all **320 training and 40 evaluation attempts** to
Vultr receipts, recomputed reward arithmetic and scores, and verified the frozen
80/20 dataset. All twenty before/after programs remain inspectable in the website.

```json
{
  "protocol.json": "4d63f006ddfe27174454c861a1aba5f8683593c281ea2c5f784a74f722bfb46f",
  "completion.json": "1c707ce99722f7e7f777f35173b0dc9e7d73ead725e819f566e97913573ec3d5",
  "optimizer-events.jsonl": "f54d828804a33672d24ba73d0f6d1abfe48aa984cbef41dda94d89ec23d08f08",
  "rollouts.jsonl": "4d5e2575f4a003bf3309b3655ab378ff4ec6a9323e226639794bd665be93d379",
  "before/summary.json": "58bc7e967b0d6f957141ace134197f338e523b1df97158c027f63bfed54cd3d2",
  "after/summary.json": "d46969d309867f4ca191ece5482491c659497b3d0fa717741ff1ccac5d14d01b",
  "before/episodes.jsonl": "9c6fb0a69554bba1c82ee4ab3858caf04636ab43d80ce014177b030a14d4e5e6",
  "after/episodes.jsonl": "bce0bdbfd4747a7fabae41524dae41fa5b4b19954e594ead7b9da3dc3468de77",
  "adapter/adapter_model.safetensors": "15e3c2d3b4899f5b52cc0dc87f4a49586dd55385fe6c444958f72f41dee12f96"
}
```

## Next experiment

Use training-only diagnostics to build a progression from correct bounded search
to full IDA* trace specifications. Validate that the starting model receives useful
reward contrast before allocating a longer run. Freeze a new unseen evaluation
suite before that experiment and report it separately from this result.
