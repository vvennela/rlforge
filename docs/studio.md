# AksharaForge environment studio

The root page is the document-upload studio. `/workbench` retains the engineering sandbox workspace. The studio uses vendored KaTeX 0.16.22 (MIT license in `studio/vendor/LICENSE`) and the existing depth-buffered bridge renderer.

Generation runs on the web backend. Default provider: Vultr Serverless Inference; default model: `glm-5.3`. The backend checks the live model catalog and returned model identity. Set `VULTR_INFERENCE_API_KEY` and optionally `AKSHARA_GENERATOR_MODEL`, or use the connection form. The form saves a mode-0600 file inside the run directory. The key is never returned to the browser or written into generation artifacts. OpenAI Responses support is available through `AKSHARA_GENERATOR_PROVIDER=openai`, `OPENAI_API_KEY`, and `AKSHARA_GENERATOR_MODEL=gpt-6-astra`.

Upload PDF, TXT, or Markdown (8 MB maximum). Native PDF extraction preserves page text and the original file. Image-only pages require an Akshara OCR text import; the uploader does not silently substitute or invent OCR. Every generation saves request/response artifacts. Generated tasks must contain exact source quotations, supported answer types, and unique prompts. Independent mathematical/reference validation remains required before training. Source/schema checks alone do not certify the answers.

The downloadable ZIP contains a standard-library JSON-lines `reset`/`step` runtime, public task prompts, private reference answers, source text, and a manifest. Keep private answers on the trusted verifier host, outside learner containers. This is a one-answer environment; no generated code is automatically executed on the controller. It is separate from the ongoing eight-turn engineering experiment.

Case-study evidence is read from `case-studies.json` in the run directory (override with `AKSHARA_CASE_STUDIES`). Generate it with:

```sh
PYTHONPATH=src python -m akshara_forge.studio.evidence --root /path/to/AksharaForge --output /path/to/runs/case-studies.json
```

Missing scores remain null. Partial evaluation counts retain their completed denominator and are labeled partial. Deltas appear only after both 20-case evaluations complete. Coding currently has prepared tasks and no recorded before/after training result. The bridge on the landing page is labeled reference geometry; the current training experiment repairs a single upright.

The server binds to loopback and validates Host/Origin. Access the deployed Vultr instance through an SSH tunnel or an authenticated proxy with explicitly configured routing. It is not exposed on a public application port.
