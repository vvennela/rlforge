# Eight-turn engineering experiment

The learner observes, places or edits pieces, receives deterministic measurements,
and acts again for at most eight turns. Each response can contain one to four
placement, movement, or removal operations. Immutable reference bridge geometry
cannot be edited; all learner-created pieces and a misplaced starter piece can
be corrected. Invalid responses consume a turn without changing the structure.

This remains a narrow synthetic upright-repair experiment, not complete bridge,
car, or engine construction and not automatic drawing extraction. The two allowed
parts are 1x1 bricks (three plate-heights) and 1x1 plates (one plate-height). Every
case needs at least four cells filled, so one allowed piece cannot solve it.

There are 80 frozen training cases and 20 fresh held-out cases (seed 20260928).
One quarter start with a misplaced editable brick. Geometry signatures from the
previous one-response pilot are excluded. The training order progresses through
all 80 cases by difficulty, then repeats the hardest 20. Held-out cases are never
sampled by the optimizer or used by the quick smoke test.

## Feedback and reward

Every turn returns the actual editable pieces, current and required height,
missing and extra cells, collisions, errors, and a vector of coverage, material
precision, x alignment, y alignment, height match, connectivity, and collision
freedom. The scalar potential is:

`0.45 coverage + 0.15 precision + 0.1 x + 0.1 y + 0.1 height + 0.1 success`.

Collisions or disconnected geometry multiply that potential by 0.1. Per-turn
reward is the potential difference, with a 0.5 terminal success bonus. Undiscounted
episode return telescopes to final minus initial potential plus that bonus.
Removing and replacing the same piece cannot farm cumulative progress reward.
Dimensions are input specifications; reference repair actions are not shown to
the learner. The verifier measures the resulting structure rather than trusting
semantic labels or requested rewards.

The stateless reward API replays the bounded action history in a fresh gVisor
sandbox. A retry cannot duplicate a previous placement in a persistent workspace.
The controller validates permissions and keeps the verifier outside the sandbox.
Every valid operation executes in the restricted worker. Network, filesystem,
memory, process and user restrictions match the deployed engineering sandbox.
Only the trusted controller has access to Docker.

## Weight updates

The custom trainer implements on-policy REINFORCE with a leave-one-out baseline
over two complete trajectories from the same task. Both trajectories are sampled
with temperature 1, top-p 1 and no top-k truncation before any update. A trajectory's
advantage is its return minus the other trajectory's return. Action-token log
probabilities are summed; verifier feedback is context and is never a label.
The loss is normalized by the fixed group size times the eight-turn limit, not
by variable trajectory length. A coefficient-0.02 sampled KL regularizer keeps
the adapter near the original base model. Gradient norm is clipped at 1.

This is a new trainer and protocol, not a continuation or matched ablation of the
old single-response GRPO pilot. Qwen2.5-7B-Instruct uses the original pinned base
revision in BF16, a rank-8 query/value LoRA adapter, and learning rate 1e-5. The
quick and long processes each initialize from that original base, independently.
They do not inherit the earlier math adapter or failed repair adapter.

The quick stage uses four training cases, four paired training batches, and before
and after checks on those same training cases. It must show multiple placements
and a move/removal that improves geometry or reduces excess material before the
long run launches. Those checks establish behavior, not generalization. Scripted
sandbox preflights are labeled separately and cannot satisfy the Qwen behavior
gate.

The long stage performs 100 paired batches and evaluates the final fixed-step
adapter on all 20 held-out cases. Base and trained evaluations both get eight
turns, identical feedback, greedy decoding, and 256 generated tokens per turn.
Record actual reward-contrast batches, gradient norms and weight deltas, since
identical trajectory returns provide no task-policy gradient even when an
optimizer step is called. Success, geometric overlap, placements, corrections,
and turns are reported separately. No efficiency penalty is active.

## Operation

Vultr hosts the reward service and sandbox execution. A forwarding-only SSH key
allows a systemd-managed tunnel directly from AWS to the Vultr reward port. The
laptop is not in the reward path. No inbound application port is exposed.

The current run receipts and credentials live outside Git in
`../runs/engineering-interactive`. They identify the AWS instance, runtime cap,
SSH config, deployment units, dataset hashes, monitor and artifact paths. The
monitor copies both AWS results and Vultr traces, then stops the idle GPU after
successful completion; a hard shutdown is the fallback. Earlier experimental
artifacts and the MATH-500 baseline remain separate and unchanged.
