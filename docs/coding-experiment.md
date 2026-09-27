# Executable coding experiment

Qwen2.5-7B-Instruct starts from the original pinned base, independently of the
math and LEGO adapters. It writes Python `solve(problem)` implementations of
Korf-derived IDA* specifications. The source card and dataset hashes are frozen
before the baseline. The original curated paper questions remain separate.

The dataset contains 80 training specifications and 20 held-out specifications.
Each is a unique combination of traversal order, initial bound, trace filters,
and output fields. Each program faces 16 tests: four common edge-case patterns
and twelve independently seeded graph inputs. Test graphs include cycles,
unreachable goals, weighted edges, admissible heuristics, ties, and start=goal.
The independent Dijkstra tests certify the reference costs; hand-worked tests
certify cutoff ordering and trace semantics.

Candidate programs run in gVisor containers on Vultr: no network, read-only,
unprivileged, bounded CPU/memory/output/runtime. Only inputs enter the container.
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
