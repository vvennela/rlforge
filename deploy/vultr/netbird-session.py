#!/usr/bin/env python3
"""Keep a public NetBird URL alive only for this systemd demo session."""
import json
import os
from pathlib import Path
import re
import signal
import subprocess

session = Path('/run/aksharaforge-demo/session.json')
child = subprocess.Popen(['/usr/bin/netbird', 'expose', '8789', '--with-name-prefix',
                          'aksharaforge'],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
def stop(*_):
    child.terminate()
signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
try:
    for line in child.stdout:
        match = re.search(r'URL:\s+(https://[a-zA-Z0-9.-]+)(?:\s|$)', line)
        if match:
            temp = session.with_suffix('.tmp')
            temp.write_text(json.dumps({'origin':match[1], 'role':'public-demo'}))
            temp.chmod(0o644)
            os.replace(temp, session)
            print('Demo URL: '+match[1], flush=True)
        # Do not echo arbitrary CLI diagnostics that might include credentials.
    raise SystemExit(child.wait())
finally:
    session.unlink(missing_ok=True)
    if child.poll() is None:
        child.terminate()
        child.wait(timeout=10)
