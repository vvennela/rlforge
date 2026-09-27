# RLForge

**Turn technical knowledge into environments where AI agents learn by doing.**

RLForge connects document extraction, task generation, sandboxed execution, reward verification, and reinforcement learning. Upload a paper or specification; the pipeline preserves its source evidence and builds exercises with a standalone evaluation runtime. Vultr hosts the web application, sandboxes, and GLM-powered generation; an AWS GPU trains Qwen 2.5 7B.

## What we tested

- **Mathematics:** After 80 training updates, Qwen improved from **2/20 to 5/20** on the same held-out paper-derived questions: **+15 percentage points**. Ten questions use exact-answer checks; ten use teacher-graded proofs.
- **Engineering:** An eight-turn LEGO bridge environment returns feedback on geometry, collisions, and connectivity. After 50 batches, successful repairs rose **3/20 → 7/20 (+20pp)**; mean geometry overlap rose **36.7% → 71.1%**.
- **Coding:** Qwen writes IDA* search implementations for execution in isolated containers. Its frozen suite includes GLM-generated adversarial cases, independent reference checks, and mutation testing. The baseline completed at **0/20**; 80 training batches and paired evaluation are underway.

Training traces, checkpoints, and evaluation protocols make each result inspectable. Held-out tasks stay separate from training; final scores compare identical evaluation conditions.

## What comes next

Automate the full path from engineering drawings, algorithms, and scientific papers to reusable learning environments. Expand the task catalog, strengthen independent verifiers, and measure transfer to unseen problems and external benchmarks.

See the [pipeline guide](docs/pipeline-guide.md), [web studio](docs/studio.md), and [coding experiment](docs/coding-experiment.md), and [LEGO results](docs/lego-results.md) for setup and protocols.
