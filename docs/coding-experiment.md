# Executable coding experiment

Qwen2.5-7B-Instruct starts from the original pinned base, independently of the
math and LEGO adapters. It writes Python `solve(problem)` implementations of
Korf-derived IDA* specifications. The source card and dataset hashes are frozen
before the baseline. The original curated paper questions remain separate.

The dataset contains 80 training specifications and 20 held-out specifications.
Each is a unique combination of traversal order, initial bound, trace filters,
and output fields. Each program faces 24 tests: four common edge-case patterns, twelve
independently seeded graph inputs, and eight GLM-generated adversarial inputs. Test graphs include cycles,
unreachable goals, weighted edges, admissible heuristics, ties, and start=goal.
The independent Dijkstra tests certify the reference costs; hand-worked tests
certify cutoff ordering and trace semantics.

Candidate programs run in gVisor containers on Vultr: no network, read-only,
unprivileged, bounded CPU/memory/output/runtime. The frozen wall-clock budget is
30 seconds per program, including container startup; correct reference programs
exposed startup jitter at the original 12-second preflight limit. The trainer
checks the reward server’s dataset fingerprint and execution budget. Only inputs enter the container.
Expected answers and comparisons stay on the controller. Training requests for
held-out identifiers are rejected. The GPU receives pass fractions and records
programs, per-test pass vectors, gradients, and adapter deltas.

The fixed run uses 80 optimizer batches, four sampled programs per batch,
1536 generated tokens, temperature 0.8, and a new rank-8 LoRA adapter. Learning
rate is 2e-5; group-relative REINFORCE uses a leave-one-out reward baseline and
0.01 sampled KL penalty. Only generated code tokens receive gradients. All 80
training specifications appear once. No teacher solutions are training labels.

The baseline and final adapter use the same 20 held-out specifications, greedy
decoding and identical execution checks. Report full-program correctness out of
20 and mean per-test pass fraction, plus paired improvement/regression counts.
Only the final fixed-step adapter is evaluated. No held-out checkpoint selection.
This measures executable algorithm implementation under unseen specification
combinations; it is not a general-purpose coding benchmark.

Campaign deadline: 2026-09-27 15:05 UTC. LEGO training and its final evaluation run
first; coding follows on the same A10G. Optional MATH-500 uses the fixed math
adapter, with the existing frozen baseline protocol, only if time remains.

## Adversarial suite revision

Before the coding baseline, `coding.adversarial` builds a new immutable dataset
from the original 80/20 split. GLM supplies eight additional graph inputs per
split, with intended failure modes; it does not supply expected outputs. Two
independently implemented IDA* solvers must agree on every returned field, and
Dijkstra independently checks path costs. Input validation rejects inadmissible
heuristics, invalid edge costs, and oversized graphs.

A mutation gate rejects suites that let any of eight deliberately incorrect
implementations survive in either split: inclusive cutoff, ignored heuristic,
goal before cutoff, reversed traversal, wrong next bound, unit edge costs,
global visited set, and discarded prior traces. Each GLM case must distinguish
at least one mutation. Raw model responses, failure reasons, and mutation
witnesses stay with the dataset. Truncated or invalid responses are not accepted.
The resulting protocol has 24 private tests per program and is frozen before
any before/after coding scores are collected. No test is selected using Qwen's
held-out performance.

## Recorded reward revision: code-002

The strict-reward run `code-001` was preserved after eight optimizer batches.
Seven groups had four zero rewards; one group contained a 1/24 test pass and
produced the first update (917,504 changed adapter parameters). This was a sparse
training signal. A diagnostic on two training programs found 4/96 and 1/96
exactly correct output fields despite both failing every complete test.

`code-002` starts afresh from the same pinned base and runs the full 80 batches.
Its training reward is 0.5 × exact-test pass fraction + 0.5 × exact-field match
fraction. A field is credited only when the returned object has exactly the
required keys. Extra keys, missing keys, wrong types, nonfinite values, and
forged envelopes do not receive credit. Reward one still requires every field
of every test to match. Field credit is partial correctness, not program success.

The 24-test suite, dataset split, runtime limit, generation settings, and strict
held-out success criterion remain unchanged. Both baseline and final evaluation
record exact-field accuracy as an additional metric while the primary score
still requires all tests to pass. A fresh baseline is recorded for this run.
The reward revision uses training-attempt evidence; held-out cases remain outside
training, task generation, and checkpoint selection.
