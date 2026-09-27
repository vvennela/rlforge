# RLForge live demo

## Three-minute walkthrough

**0:00–0:25 — The product.** Open Upload. “RLForge turns technical documents into
learning environments: tasks, sandbox tools, feedback, and measurable evaluation.”
Upload a short source document. Show OCR/source artifacts and generation progress.

**0:25–1:05 — The environment.** Open Engineering. Rotate the three bridge views:
starting structure, base Qwen, and trained Qwen. Explain the eight-turn tool budget
and geometry/connectivity rewards. Use the displayed completed holdout scores;
partial results remain labeled as such. The visual example is the fixed first
held-out case, selected before final evaluation.

**1:05–1:45 — Learning and verification.** Show Math’s 2/20 → 5/20 result after
80 updates on Qwen 2.5 7B. Explain its ten exact-answer questions and ten
teacher-graded proofs. Show Coding’s recorded status: 24 hidden tests per program,
including adversarial GLM cases, independent reference solvers, and eight mutation
checks. Show final coding scores only once both evaluations complete.

**1:45–2:25 — Real execution.** Open the engineering workspace and run a small
bridge repair through Vultr Serverless Inference. Show the resulting geometry and
reward feedback. Return to the uploaded environment and download its runtime and
source evidence when generation completes.

**2:25–3:00 — The direction.** “The product is a reusable pipeline for constructing
learning environments from technical knowledge.” Show the repository and explain
how new domains add tools and independent verifiers.

## Q&A facts

- Vultr hosts the app, OCR, GLM inference, sandboxes, and reward services. AWS
  supplies the training GPU. NetBird carries public demo traffic to the private app.
- The reported math and bridge experiments use curated pilot tasks. The upload
  pipeline generates new source-linked environments with separate audit artifacts.
- Expected coding outputs stay outside candidate containers. Network access is
  disabled; the controller computes rewards from hidden tests.
- Gradients, adapter changes, sampled attempts, checkpoints, and fixed evaluation
  conditions are recorded. Parameter changes establish training happened;
  matched held-out scores establish behavioral improvement.
- The original MATH-500 baseline used an L40S. An A10G result is labeled separately
  unless a matched baseline is available. Benchmark questions never guide training.
