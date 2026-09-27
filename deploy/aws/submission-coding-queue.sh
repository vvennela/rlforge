#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/AksharaForge-interactive
export PYTHONPATH="$PWD/src"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export AKSHARA_CODING_TOKEN_FILE="$PWD/coding/reward-token"
export AKSHARA_CODING_REWARD_URL=http://127.0.0.1:8772
mkdir -p coding/runs
trap 'code=$?; /usr/bin/python3 -c "import json,time;json.dump(dict(exit_code=$code,finished_at=time.time()),open(\"coding/queue-exit.json\",\"w\"))"' EXIT
while systemctl is-active --quiet aksharaforge-interactive.service; do sleep 20; done
/usr/bin/python3 - <<'PY'
import json,hashlib
from pathlib import Path
preflight=json.loads(Path('coding/preflight-v3.json').read_text())
assert preflight['passed'] and preflight['heldout_train_blocked']
fingerprint=hashlib.sha256(Path('coding/dataset-v3/manifest.json').read_bytes()).hexdigest()
assert all(c['result']['dataset_manifest_sha256']==fingerprint for c in preflight['checks'])
assert all(c['result']['wall_timeout_seconds']==30 for c in preflight['checks'])
root=Path('runs/long-004')
d=json.loads((root/'completion.json').read_text())
assert d['optimizer_steps']==50 and d['total']==20
assert json.loads((root/'after/summary.json').read_text())['complete']
PY
# A queued job never competes with another model for GPU memory.
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
    echo 'GPU is occupied; inspect before starting coding.' >&2
    exit 1
fi
model_path=/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
/opt/pytorch/bin/python -u -m akshara_forge.coding.train --dataset coding/dataset-v3 --output coding/runs/code-001 --model "$model_path" --steps 80 > coding/runs/code-001.log 2>&1
