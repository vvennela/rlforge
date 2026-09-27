from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def mac_profile() -> str:
    # Allow OS/Python runtime reads; never allow project data or the user's home.
    paths = {"/System", "/usr", "/bin", "/sbin", "/Library", "/Applications", "/opt", "/private/var/db", "/dev", "/etc",
             str(Path(sys.base_prefix).resolve()), str(Path(sys.prefix).resolve()),
             str(Path(__file__).resolve().parent)}
    read_rules = "\n".join(f"(allow file-read* (subpath {json.dumps(p)}))" for p in sorted(paths))
    ancestors = set()
    for p in (Path(sys.prefix), Path(__file__).resolve().parent):
        ancestors.update(str(x) for x in p.parents)
    literal_rules = "\n".join(f"(allow file-read* (literal {json.dumps(p)}))" for p in sorted(ancestors))
    return f'''(version 1)
(deny default)
(allow process-exec process-fork sysctl-read mach-lookup signal)
(allow file-read-metadata)
{read_rules}
{literal_rules}
(allow network-outbound (remote ip "localhost:11434"))
(allow file-write* (literal "/dev/null"))
'''


def run_learner(request: dict, *, backend: str = "auto", timeout: int = 330) -> dict:
    if backend == "auto":
        backend = "macos" if platform.system() == "Darwin" else "docker"
    worker = Path(__file__).with_name("worker.py").resolve()
    if backend == "macos":
        if platform.system() != "Darwin" or not Path("/usr/bin/sandbox-exec").exists():
            raise RuntimeError("macOS sandbox unavailable; choose docker")
        if request.get("base_url", "http://127.0.0.1:11434") not in {
            "http://127.0.0.1:11434", "http://localhost:11434"}:
            raise ValueError("macOS learner sandbox permits only local Ollama on port 11434")
        command = ["/usr/bin/sandbox-exec", "-p", mac_profile(), sys.executable, "-I", "-B", str(worker)]
    elif backend == "docker":
        if not shutil.which("docker"):
            raise RuntimeError("Docker is required on this platform")
        request = {**request, "base_url": "http://host.docker.internal:11434"}
        command = ["docker", "run", "--rm", "-i", "--read-only", "--cap-drop=ALL",
                   "--security-opt=no-new-privileges", "--pids-limit=32", "--memory=256m", "--cpus=1",
                   "--user=65534:65534", "--add-host=host.docker.internal:host-gateway",
                   "--mount", f"type=bind,source={worker},target=/worker.py,readonly",
                   "python:3.12-slim", "python", "-I", "-B", "/worker.py"]
    else:
        raise ValueError("Sandbox must be auto, macos or docker; no unsandboxed learner fallback")
    clean_env = {k: v for k, v in os.environ.items() if k in {"PATH", "LANG", "LC_ALL", "TMPDIR", "SYSTEMROOT"}}
    result = subprocess.run(command, input=json.dumps(request), text=True, capture_output=True,
                            timeout=timeout, env=clean_env, cwd="/tmp")
    if result.returncode:
        raise RuntimeError(f"Learner sandbox failed ({result.returncode}): {result.stderr[-1500:]}")
    return {**json.loads(result.stdout), "sandbox": backend}
