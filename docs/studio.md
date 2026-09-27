# AksharaForge environment studio

The root page is the document-upload studio. `/workbench` retains the engineering sandbox workspace. The studio uses vendored KaTeX 0.16.22 (MIT license in `studio/vendor/LICENSE`) and the existing depth-buffered bridge renderer.

Generation runs on the web backend. Default provider: Vultr Serverless Inference; default model: `glm-5.3`. The backend checks the live model catalog and returned model identity. Set `VULTR_INFERENCE_API_KEY` and optionally `AKSHARA_GENERATOR_MODEL`, or use the connection form. The form saves a mode-0600 file inside the run directory. The key is never returned to the browser or written into generation artifacts. OpenAI Responses support is available through `AKSHARA_GENERATOR_PROVIDER=openai`, `OPENAI_API_KEY`, and `AKSHARA_GENERATOR_MODEL=gpt-6-astra`.

Upload PDF, TXT, or Markdown (8 MB maximum). PDF uploads run the existing Akshara canonicalization stack: native text extraction plus Tesseract on scanned/sparse pages, lossless 200-DPI page images, overlapping strips, raw OCR TSV, and source hashes. Source-bound text and visual candidates are extracted into separate directories. OCR progress and a first-page preview appear in the upload panel. The complete source-evidence ZIP can be downloaded before connecting inference and is included in completed environment bundles. Install `tesseract-ocr` and `tesseract-ocr-eng` on the Vultr CPU. Every page is retained; the existing 200-page upload limit rejects oversized documents rather than truncating them. Every generation saves request/response artifacts. Generated tasks must contain exact source quotations, supported answer types, and unique prompts. Every batch receives a blind GLM solution cross-check (author answers withheld), at least three model-written wrong-answer probes per task, controller type/boundary/spoofing probes, and an execution test of the exported runtime. Disagreements or accepted wrong-answer probes block release and retain audit artifacts. Same-model agreement is not a formal mathematical proof; domain-specific executable checks provide stronger guarantees.

The downloadable ZIP contains a standard-library JSON-lines `reset`/`step` runtime, public task prompts, private reference answers, source text, and a manifest. Training mode refuses held-out task IDs. Start the controller with `--split heldout` for evaluation. Mount only learner/ (training prompts) in the training sandbox and evaluation/ (held-out prompts) during evaluation; keep private answers, review traces, and adversarial probes on the trusted verifier host. This is a one-answer environment; no generated code is automatically executed on the controller. It is separate from the ongoing eight-turn engineering experiment.

Case-study evidence is read from `case-studies.json` in the run directory (override with `AKSHARA_CASE_STUDIES`). Generate it with:

```sh
PYTHONPATH=src python -m akshara_forge.studio.evidence --root /path/to/AksharaForge --output /path/to/runs/case-studies.json
```

Missing scores remain null. Partial evaluation counts retain their completed denominator and are labeled partial. Deltas appear only after both 20-case evaluations complete. Coding has a completed 0/20 baseline; the 80-batch run and final paired evaluation supply its next result. The bridge on the landing page is labeled reference geometry; the current training experiment repairs a single upright.

The server binds to loopback and validates Host/Origin. The deployed demo uses a lifecycle-bound NetBird reverse-proxy route with public access, as configured for the presentation; private provider configuration stays blocked at the gateway. It is not exposed on a public application port.

## Live pipeline verification

A 20-problem source specification completed generation and two blind GLM review
passes on Vultr. The downloaded runtime passed all 265 answer probes locally.
The split-enforced runtime also rejected cross-split reset requests in both modes.
Raw generation/review responses and the original download remain preserved in
private run artifacts; no training score is inferred from these compiler checks.

## Generation recovery and provenance

The controller supplies exact source passages with stable IDs. The model selects an
ID; the controller attaches the original text, character span, and source hash.
Structured questions receive an explicit output schema before blind review, without
including reference values.

Malformed model JSON triggers at most two identical-request retries, each with its
own preserved response. Content checks remain strict. Within each ten-question
batch, fully reviewed questions are retained while rejected questions are replaced
over at most five attempts. Accepted questions keep their complete review and
adversarial probes; original batch audits are included under `private/reviews/`.
A package is released only when every requested question passes and its exported
runtime passes the answer probes and train/held-out isolation checks.

## Scanned-paper demonstration

The deployed Korf paper example completed OCR on all 13 scanned pages, preserving
39 strips and 26 extracted artifacts. GLM generated 20 problems with a 16/4 split;
eight blind review calls supplied the accepted questions and adversarial probes.
The downloaded runtime passed 270 answer probes and both split-isolation checks.
Independent arithmetic and finite-tree calculations matched all 20 numerical
references. This numerical audit does not certify every explanatory source claim.

[Open the completed paper environment](https://aksharaforge-yhm6.netbird.64-177-45-215.sslip.io/?environment=9c298aac6f3747fc83e7f043d9fd8ce4#upload).
The demo route expires with its NetBird session. Its downloaded bundle SHA-256 is
`c385439c5123730515bed3038c623a2ea2019a4e7b75619617eab090ea4dd219`.
The bundle includes source evidence, accepted reviews, and the original batch
audits, including rejected drafts. The local run archive preserves every raw call.
