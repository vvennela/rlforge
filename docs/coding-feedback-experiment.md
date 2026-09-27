# Execution-feedback coding experiment

This follow-up separates two effects: repairing a program after execution feedback,
and improving that repair policy through weight updates. It starts from the pinned
original Qwen2.5-7B-Instruct, independently of the previous coding adapter.

Each attempt writes a complete `solve(problem)` implementation. The Vultr gVisor
sandbox executes six development examples, returning up to two failing inputs,
expected outputs, actual outputs, and execution errors. Qwen receives that feedback
alongside its previous code and can submit a full replacement, up to three attempts.
Development success ends an episode early. Private tests never determine when to
stop or which candidate to select: the last permitted candidate is scored.

The private suite remains 24 executable tests per specification. Its inputs and
answers never enter repair feedback. Eighty original training configurations are
retained. Twenty new held-out configurations are selected from the combinations
excluded from both previous splits, with fresh seeded private inputs and retained
adversarial graph patterns. Development inputs are disjoint from each task's private
inputs. Two independent solvers agree on all answers, Dijkstra checks reference
costs, and eight incorrect algorithm variants are rejected in both splits.

Before and after evaluation record both first-attempt correctness and final repair
correctness on the same twenty specifications. Both use greedy decoding, 1,536
generated tokens per attempt, the same feedback, and the same three-attempt limit.
These results form a new experiment; they do not replace the previous 0/20 result.

Training samples two trajectories per specification. The final private training
reward is half exact-test fraction plus half exact-field fraction. Group-relative
REINFORCE updates only generated assistant tokens, including repair attempts;
feedback tokens are context and receive no supervised loss. The new query/value
LoRA adapter has rank eight and alpha sixteen. Learning rate is 1e-5 and KL weight
0.01. Three-output training configurations precede four-output configurations.

The run allows at most sixteen updates. Training stops starting new batches at
17:40 UTC on September 27, reserving the final evaluation window before the
18:10 UTC GPU stop. The final completed update is evaluated, regardless of score.
Saved traces, optimizer events, checkpoints, and sandbox receipts remain separate
from the published case studies. No result is published automatically.
