# RLForge

**Turn technical knowledge into environments where AI agents learn by doing.**

RLForge (formerly AksharaForge) connects document extraction, task generation, sandboxed execution, reward verification, and reinforcement learning. Upload a paper or specification; the pipeline preserves its source evidence and builds exercises with a standalone evaluation runtime. Vultr hosts the web application, sandboxes, and GLM-powered generation; an AWS GPU trains Qwen 2.5 7B.

## What we tested

- **Mathematics:** After 80 training updates, Qwen improved from **2/20 to 5/20** on the same held-out paper-derived questions: **+15 percentage points**. Ten questions use exact-answer checks; ten use teacher-graded proofs.
- **Engineering:** An eight-turn LEGO bridge environment returns feedback on geometry, collisions, and connectivity. The baseline solved **3/20** held-out repairs; the 50-batch training run is underway.
- **Coding:** Qwen writes IDA* search implementations for execution in isolated containers. We are adding GLM-generated adversarial cases, independent reference checks, and mutation testing before the paired evaluation.

Training traces, checkpoints, and evaluation protocols make each result inspectable. Held-out tasks stay separate from training; final scores compare identical evaluation conditions.

## What comes next

Automate the full path from engineering drawings, algorithms, and scientific papers to reusable learning environments. Expand the task catalog, strengthen independent verifiers, and measure transfer to unseen problems and external benchmarks.

See the [pipeline guide](docs/pipeline-guide.md), [web studio](docs/studio.md), and [coding experiment](docs/coding-experiment.md) for setup and protocols.
