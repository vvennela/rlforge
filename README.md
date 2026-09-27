# RLForge

**Turn technical knowledge into environments where AI agents learn by doing.**

RLForge connects document extraction, task generation, sandboxed execution, reward verification, and reinforcement learning. Upload a paper or specification; the pipeline preserves its source evidence and builds exercises with a standalone evaluation runtime. Vultr hosts the web application, sandboxes, and GLM-powered generation; an AWS GPU trains Qwen 2.5 7B.

## What we tested

- **Mathematics:** After 80 updates, paper holdout accuracy rose **2/20 → 5/20 (+15pp)**, with ten exact-answer checks and ten teacher-graded proofs. MATH-500 scored **396/500 original; 395/500 math-trained**, on separate GPU runtimes.
- **Engineering:** An eight-turn LEGO bridge environment returns feedback on geometry, collisions, and connectivity. After 50 batches, successful repairs rose **3/20 → 7/20 (+20pp)**; mean geometry overlap rose **36.7% → 71.1%**.
- **Coding:** Qwen writes IDA* search implementations for execution in isolated containers. Its frozen suite includes GLM-generated adversarial cases, independent reference checks, and mutation testing. After 80 batches, full correctness stayed **0/20 → 0/20**; exact-field accuracy fell **2.36% → 0.42%**. This run produced no coding uplift.

Training traces, checkpoints, and evaluation protocols make each result inspectable. Held-out tasks stay separate from training; final scores compare identical evaluation conditions.

## What comes next

Automate the full path from engineering drawings, algorithms, and scientific papers to reusable learning environments. Expand the task catalog, strengthen independent verifiers, and measure transfer to unseen problems and external benchmarks.

See the [pipeline guide](docs/pipeline-guide.md), [web studio](docs/studio.md), and [coding results](docs/coding-results.md), and [LEGO results](docs/lego-results.md) for setup and protocols.
