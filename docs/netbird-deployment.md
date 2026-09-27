# NetBird demo deployment

The app runs on Vultr at 127.0.0.1:8787. The restricted judge frontend runs at
127.0.0.1:8789 and permits studio assets, uploads, job polling, and downloads.
Provider configuration and engineering control endpoints remain operator-only.
The existing SSH tunnel continues to provide operator access.

## Infrastructure

NetBird client 0.79.0 and Docker Compose 2.40.3 are installed on 64.177.45.215.
The upstream quickstart installer is staged in /opt/aksharaforge-netbird.
The management/proxy stack has NOT been started: a domain is required.

Use the official self-hosted quickstart with built-in Traefik and the NetBird
proxy enabled. Configure an A record for the chosen management hostname and a
wildcard CNAME below it, both DNS-only. The gateway needs TCP 80/443 and UDP 3478;
application ports 8787/8789 remain loopback-only. Co-locating the gateway and app
on this VM does not mean this VM has zero inbound gateway ports. For a physically
separate zero-inbound application VM, place the gateway on a separate Vultr VM.

Bootstrap the owner before allowing public management access. Enroll this peer
using a one-use setup key, enable peer expose only for the demo peer group, and
keep the management API token private. Never expose an unclaimed setup page.

## Demo lifecycle

Store a randomly generated password (at least 16 characters) in
/etc/aksharaforge-netbird-demo.password, root-only mode 0600. This credential
represents the hackathon judge role; it does not grant provider administration.
Install the two service units and start aksharaforge-netbird-demo.service once
NetBird is connected. The wrapper records the exact generated HTTPS origin in
/run/aksharaforge-demo/session.json; the frontend rejects requests until that is
available. It checks Host and same-origin POSTs before forwarding to the app.

The expose process binds the URL lifetime to the demo session and application.
Stopping the app, stopping the demo unit, or the 24-hour limit ends exposure.
NetBird removes the service on graceful stop; failed renewal expires it after
90 seconds. Starting a new session creates a new URL. These units are not enabled
at boot and do not change any AWS training time or cost limits.

## Acceptance checks before claiming live

- Valid HTTPS certificate and unauthenticated password gate from the public URL.
- Judge login can view all three case studies and OCR an uploaded scanned PDF.
- Public POST /api/provider is rejected; local operator flow still works.
- Direct public connections to application ports fail.
- Stopping the demo removes the URL; restarting creates a fresh session.

Sources:
- https://docs.netbird.io/selfhosted/selfhosted-quickstart
- https://docs.netbird.io/manage/reverse-proxy/expose-from-cli
