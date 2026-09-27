# RLForge live demo

## Three-minute walkthrough

**0:00–0:25 — The product.** Open Upload. “RLForge turns technical documents into
learning environments: tasks, sandbox tools, feedback, and measurable evaluation.”
Choose the 20-problem practice set and upload a short source document for the
live walkthrough. Show OCR/source artifacts and generation progress. The 100-problem
option remains available; the trained case studies use their frozen 80/20 datasets.

**0:25–1:05 — The environment.** Open Engineering. Rotate the two bridge views:
the original deck-and-supports structure and a trained single-post repair example.
These are distinct task states; the recorded baseline remains available in the action traces. Explain the eight-turn tool budget
and geometry/connectivity rewards. Show 3/20 → 7/20 successful repairs (+20pp)
and 36.7% → 71.1% mean target overlap after fifty batches. Use the displayed completed holdout scores;
partial results remain labeled as such. The gallery opens on the first newly solved case, labeled “Newly solved example.”
For this run that is Case 10 (089): the base model leaves the upright missing,
while the trained model completes it. This is one of four new successes. The
headline score includes all twenty cases, and the selector retains dataset order
so unchanged repairs and failures remain equally accessible.

**1:05–1:45 — Learning and verification.** Show Math’s 2/20 → 5/20 result after
80 updates on Qwen 2.5 7B. Explain its ten exact-answer questions and ten
teacher-graded proofs. Show Coding’s recorded status: 24 hidden tests per program,
including adversarial GLM cases, independent reference solvers, and eight mutation
checks. The completed 80-batch coding run stayed at 0/20; exact-field accuracy fell
from 2.36% to 0.42%. Show the recorded programs and explain that the framework
measures unsuccessful training runs as well as gains.

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
- MATH-500 completed at 396/500 (79.2%) for original Qwen on L40S and 395/500
  (79.0%) for the math-trained adapter on A10G. Report them separately; this
  experiment did not demonstrate external benchmark improvement. Benchmark
  questions never guide training.

## Completed upload example

[Open the Korf scanned-paper environment](https://aksharaforge-yhm6.netbird.64-177-45-215.sslip.io/?environment=9c298aac6f3747fc83e7f043d9fd8ce4#upload)
when presenting the finished output: 13 OCR pages, 39 strips, 20 reviewed problems,
and a portable runtime checked against 270 answer probes. Uploading a new paper
shows the generation process; the saved example supplies an immediate completed
artifact for the three-minute presentation.
