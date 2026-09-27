#!/usr/bin/env python3
"""Manage a public NetBird reverse proxy for the lifetime of a demo unit."""
import json
import os
from pathlib import Path
import signal
import sys
import threading
import urllib.error
import urllib.request

SESSION = Path('/run/aksharaforge-demo/session.json')
STATE = Path('/opt/aksharaforge-netbird/demo-service.json')
DOMAIN = 'aksharaforge-yhm6.netbird.64-177-45-215.sslip.io'
TOKEN_FILE = Path('/opt/aksharaforge-netbird/bootstrap.json')

def api(method, path, body=None):
    token = json.loads(TOKEN_FILE.read_text())['personal_access_token']
    req = urllib.request.Request('http://127.0.0.1:19080/api/' + path,
        data=json.dumps(body).encode() if body is not None else None,
        method=method, headers={'Authorization': 'Token ' + token,
                               'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=10) as response:
        content = response.read()
        return json.loads(content) if content else None

def check_health(service_id):
    """A failed observation must not revoke a still-valid public route."""
    try:
        current = api('GET', 'reverse-proxies/services/' + service_id)
    except urllib.error.HTTPError as error:
        if error.code not in (408, 429) and error.code < 500:
            raise
        print('NetBird health check deferred: HTTP ' + str(error.code), flush=True)
        return False
    except (TimeoutError, ConnectionError, urllib.error.URLError) as error:
        print('NetBird health check deferred: ' + type(error).__name__, flush=True)
        return False
    if not current.get('enabled'):
        raise RuntimeError('NetBird demo service was disabled')
    return True

def cleanup():
    SESSION.unlink(missing_ok=True)
    if STATE.exists():
        service_id = json.loads(STATE.read_text())['id']
        try:
            api('DELETE', 'reverse-proxies/services/' + service_id)
        except urllib.error.HTTPError as error:
            if error.code != 404:
                raise
        STATE.unlink(missing_ok=True)

def main():
    cleanup()
    if '--cleanup' in sys.argv:
        return
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    peers = api('GET', 'peers')
    peer = next(p for p in peers if p['ip'] == '100.81.229.176')
    service = api('POST', 'reverse-proxies/services', {
        'name': 'AksharaForge public demo', 'domain': DOMAIN,
        'enabled': True, 'pass_host_header': True, 'auth': {},
        'targets': [{'target_id': peer['id'], 'target_type': 'peer',
                     'host': peer['ip'], 'protocol': 'http', 'port': 8789,
                     'path': '/', 'enabled': True}]})
    STATE.write_text(json.dumps({'id': service['id'], 'domain': DOMAIN}))
    STATE.chmod(0o600)
    try:
        SESSION.parent.mkdir(parents=True, exist_ok=True)
        temporary = SESSION.with_suffix('.tmp')
        temporary.write_text(json.dumps({'origin': 'https://' + DOMAIN, 'role': 'public-demo'}))
        temporary.chmod(0o644)
        os.replace(temporary, SESSION)
        print('Demo URL: https://' + DOMAIN, flush=True)
        # REST-managed service avoids the CLI expose stream's renewal timeout.
        # systemd RuntimeMaxSec and ExecStopPost enforce the session lifetime.
        while not stop.wait(30):
            check_health(service['id'])
    finally:
        cleanup()

if __name__ == '__main__':
    main()
