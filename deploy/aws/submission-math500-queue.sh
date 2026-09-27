#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/AksharaForge-interactive
# This optional run cannot displace the required coding evaluation.
# Use the remaining window when at least one hour remains; the 15:05 UTC
# instance shutdown still applies, and incomplete benchmark scores stay partial.
while [ ! -f coding/runs/code-002/completion.json ] || systemctl is-active --quiet aksharaforge-coding-queue.service; do
    # A recoverable coding interruption must not discard the benchmark queue.
    if [ "$(date -u +%s)" -gt "$(date -u -d '2026-09-27 14:05:00' +%s)" ]; then
        /usr/bin/python3 -c 'import json;json.dump({"status":"not_started","reason":"Coding did not finish with 60 minutes remaining before the authorized deadline"},open("math500/queue-status.json","w"))'
        exit 0
    fi
    sleep 30
done
if [ "$(date -u +%s)" -gt "$(date -u -d '2026-09-27 14:05:00' +%s)" ]; then
    /usr/bin/python3 -c 'import json;json.dump({"status":"not_started","reason":"Less than 60 minutes remain before the authorized deadline"},open("math500/queue-status.json","w"))'
    exit 0
fi
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then exit 1; fi
export PYTHONPATH="$PWD/math500/libs:$PWD/src"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
model_path=/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
/opt/pytorch/bin/python -u -m akshara_forge.math500 --dataset math500/dataset/test.jsonl --model "$model_path" --adapter math500/adapter --output math500/after --batch-size 8 --max-tokens 4096 > math500/after.log 2>&1
/opt/pytorch/bin/python - <<'PY'
import json
from pathlib import Path
from akshara_forge.math500 import compare
try:
    comparison=compare(Path('math500/before'),Path('math500/after'))
except ValueError as exc:
    comparison={'status':'unpaired_result','reason':str(exc),
        'before_runtime':json.loads(Path('math500/before/runtime.json').read_text()),
        'after_runtime':json.loads(Path('math500/after/runtime.json').read_text()),
        'after':json.loads(Path('math500/after/summary.json').read_text()),
        'note':'Do not claim matched benchmark uplift; a baseline on the same runtime is required.'}
Path('math500/comparison.json').write_text(json.dumps(comparison,indent=2))
PY
