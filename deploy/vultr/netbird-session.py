#!/usr/bin/env python3
"""Keep a gated NetBird URL alive only for this systemd demo session."""
import json
import os
from pathlib import Path
import re
import signal
import subprocess

session = Path('/run/aksharaforge-demo/session.json')
password = Path('/etc/aksharaforge-netbird-demo.password').read_text().strip()
if len(password) < 16:
    raise SystemExit('Demo password must have at least 16 characters')
child = subprocess.Popen(['/usr/bin/netbird', 'expose', '8789', '--with-name-prefix',
                          'aksharaforge', '--with-password', password],
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
            temp.write_text(json.dumps({'origin':match[1], 'role':'hackathon-judge'}))
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
