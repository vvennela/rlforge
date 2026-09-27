# Engineering environment v0

AksharaForge now has a working, model-independent brick geometry environment. The initial task is a **curated abstraction** of the archived Old Bridge drawing (Library of Congress `ak0443`). This is not automatic drawing interpretation, a complete historical reconstruction, a validated engineering simulation or trained-model uplift.

## Run locally

Requires Python dependencies from this repository, Docker running, and the source preview from the separately extracted design catalog. Optional Qwen repair uses the existing local Ollama `qwen2.5:7b-instruct` model.

```sh
PYTHONPATH=src python -m akshara_forge.engineering.server \
  --runs ../runs/engineering \
  --drawing ../design-pilot/intake/ak0443/preview.jpg
```

Open http://127.0.0.1:8787. The server intentionally binds only to loopback. It has not been deployed to Vultr.

1. Load the faulty fixture: the right pier is four studs too far left.
2. Inspect each reward component and the rendered assembly.
3. Run Qwen repair for an actual local model attempt, or use the manual console.
4. Verified reference loads a scripted fixture that exercises the verifier. It is explicitly not a model output.

Each reset creates a fresh unprivileged Docker container with no network, a read-only filesystem, CPU/memory/process limits and only the action worker mounted. The trusted host retains the target and scores. Model inference is outside this container; the constrained assembly workspace is inside it. This is tool-action isolation, not unrestricted agent shell execution. No silent unsandboxed fallback exists.

## API and tools

`BrickEnvironment(trace_dir)` exposes `observe()`, `step(actions)` and `close()`; constructing a new instance resets the episode. Observations contain task constraints, palette, current bricks, per-dimension diagnostics, reward vector, scalar reward and terminal status. Models can use the same interface irrespective of provider; only the included example agent is tied to Ollama.

Actions are transactional JSON lists of `place`, `move`, and `remove`. The workspace enforces integer coordinates, an explicit rectangular-part palette, 0/90 degree rotations, 256 bricks, and a 64-by-24 stud, 24 plate-height boundary. Invalid transactions preserve the previous assembly and consume a transaction. Success or 16 transactions ends the episode.

## Source dimensions and abstraction

Source: https://www.loc.gov/item/ak0443/ and sheet https://www.loc.gov/resource/hhh.ak0443.sheet/?sp=1

Original TIFF SHA-256: `7773dcab25ba8d65d9d27619da86c48dc1bc61df20aec66eba98fccefef0cb57`.

Manual transcription from the plan: left span 11′7½″, center span 15′3″, right span 11′6½″, width 5′6″. At 12 source inches per stud these imply length 38.4167 studs, width 5.5, pier spacing 15.25, first pier center 11.625. The prototype accepts half-stud dimensional tolerance; its reference occupies 38 by 6 studs with pier centers at 12 and 27.

The source width dimension covers its indicated plan feature, not a claim of constant deck width throughout the actual bridge. The prototype uses a uniform deck width deliberately. The six-plate underside height, two-layer deck, rectangular piers, grid occupancy template and parts palette are authored construction choices. Terrain, ramps, cross-braces and log geometry are excluded. The shape template enforces this curated abstraction; generating that template from documents is still future work.

## Rewards

The vector contains length, width, pier spacing, first-pier offset, deck coverage, support coverage, and target occupancy intersection-over-union. Support locations are inferred from the lower occupied geometry rather than labels supplied by the agent. Sparse corner blocks cannot satisfy coverage. Collisions and unconnected geometry cannot pass acceptance.

Dimension scores equal one inside tolerance and decline with normalized excess error. Scalar reward is the unweighted mean of seven components, multiplied by 0.25 while validity checks fail. This provides shaping for partial construction. Success requires all validity checks, dimension tolerances and complete target occupancy. Brick efficiency is not rewarded yet.

Connection checks model stud-grid overlap across horizontal interfaces. Side contact does not count. They do not certify real part clutch geometry, strength, friction, stability, build sequence or manufacturability. Rectangular shapes are brick-like proxies, not a complete LDraw parts implementation.

## Evidence and training boundary

Episode JSONL records task hashes, verifier code hash, actions and observations. Qwen run directories retain model inventory/digest, prompts, raw responses, results and initial/final scores. Qwen repair uses greedy local inference and at most three rounds; it does not update weights. No engineering uplift claim is justified yet.

First actual Qwen smoke: 119-brick faulty fixture, initial reward 0.2157; after three rounds, 0.2024 and unsuccessful. The model added colliding plates and then attempted duplicate IDs. Two rejected transactions preserved the prior assembly. This is one observed failure on one task, not a benchmark of the model. The deterministic reference reaches reward 1.0; that is a verifier feasibility test.

After retaining tool errors in observations, a second three-round Qwen attempt remained at 0.2157 and failed. A subsequent manual two-action correction reached 1.0; that manual result is not attributed to Qwen. Both model attempts and their summaries remain saved. New traces explicitly label model, fixture and manual actions.

Next: broaden the reviewed task schema, freeze held-out drawings and attempt budgets, benchmark models, then connect a trainer. Evaluate both construction from an empty workspace and repair separately. Do not use a single repair task as evidence of general engineering skill. The existing math training is a separate experiment.

## Validation

```sh
AKSHARA_TEST_DOCKER=1 PYTHONPATH=src python -m pytest tests/engineering -q
```

Tests cover reference feasibility, overlap, empty submissions, sparse bounding-box gaming, wrong pier placement, disconnected side contact, extra geometry, invalid and atomic actions, and the real Docker boundary. The browser demo is manually checked separately.
