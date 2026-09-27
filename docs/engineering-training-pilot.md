# Engineering weight-update pilot

This experiment tests whether Qwen2.5-7B-Instruct can learn a narrow LEGO-style
repair task from deterministic geometric rewards. It does not train on images,
compile new drawings, construct complete models, or establish general engineering
ability. The original paper-math adapter is preserved and is not the starting
point for this experiment.

The frozen dataset contains 80 training cases and 20 held-out cases generated
with seed 20260927. Each case varies bridge length, pier positions, upright
heights, side, and the missing upper section. Dimensions are synthetic variations
of the curated bridge abstraction, not new measurements from its source drawing.
The model sees the target coordinates, existing top, required top, and palette.
The rest of the bridge is locked. It must emit up to 16 actual placement actions.
Evaluation measures interpolation within this repair family, not out-of-family
generalization. Dataset hashes and unique observable prompts freeze the split.

The reward service executes valid submissions in individual gVisor containers on
Vultr. The trusted host verifies the resulting geometry. Containers have no
network, read-only filesystems, no capabilities, an unprivileged user, a 512 MB
memory bound, half a CPU, and a 128-process ceiling. The previous 16-process bound
prevented gVisor startup on the deployment host; 128 passed integration tests.
Only the worker is mounted into each sandbox. Neither reference repairs nor the
verifier are available inside it. The trusted controller alone accesses Docker.

Rewards retain the full component/dimension vector in the traces. For this
optimizer, the scalar is `0.8 * added_geometry_IoU + 0.2 * full_success`. Extra
material lowers IoU. Collisions or a disconnected assembly multiply the reward by
0.1. Invalid actions score zero; infrastructure failures stop the job rather than
silently becoming training penalties. Reference solutions are verifier tests,
never learner demonstrations.

The AWS L40S executes the pinned original Qwen weights in BF16. GRPO trains a
rank-8 LoRA adapter on query/value projections for 40 optimizer updates, with four
sampled completions per prompt, temperature 0.8, learning rate 2e-5, and KL weight
0.02. The 80 cases form the training pool; 40 updates do not imply a complete
80-case pass. This is a first gradient and learning-signal experiment. No
efficiency penalty is active. Before and after evaluation use the same 20 cases,
greedy decoding and 768-token limit. The final fixed-step adapter is evaluated;
held-out results do not select checkpoints or alter the reward.

Artifacts include all learner completions, sandbox actions and observations,
per-component rewards, clipped gradient norms, nonzero gradient counts, initial
adapter tensors, and numerical changes to trained parameters. Changed weights
prove updates occurred; only the paired held-out result measures improvement.
One seed and 20 cases provide preliminary evidence, not a reliable broad uplift
claim.

The reward API binds to loopback, requires a bearer token stored outside Git, and
is reached over two SSH tunnels via the local controller. This prototype depends
on those tunnels remaining alive. No inbound application port is exposed. The
GPU has a two-hour shutdown safeguard; checkpoints and traces should be copied
before stopping it earlier on completion. The Vultr service remains the source of
sandbox execution evidence.

Generate with `python -m akshara_forge.engineering.pilot generate --dataset PATH`.
Serve with the module's `serve` command and run training with
`python -m akshara_forge.engineering.train_pilot --help`. Credentials, machine IDs,
the frozen dataset, and run receipts live outside the repository under
`../runs/engineering-training`.
