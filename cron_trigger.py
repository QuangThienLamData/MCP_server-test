"""Called by a Render Cron Job. POSTs the web service's /internal/crawl endpoint so the
crawl runs INSIDE the web service (sharing its SQLite + Pinecone), instead of in this
short-lived cron container. Stdlib only — no dependencies to install.

Usage:  python cron_trigger.py [target]        target = all | rag | news | reviews
Env:    WEB_SERVICE_URL  (e.g. https://mcp-server-test-2soh.onrender.com)
        CRON_SECRET      (must match the web service's CRON_SECRET)
"""
import os
import sys
import urllib.request

target = sys.argv[1] if len(sys.argv) > 1 else "all"
base = (os.environ.get("WEB_SERVICE_URL") or os.environ.get("RENDER_EXTERNAL_URL") or "").rstrip("/")
secret = os.environ.get("CRON_SECRET", "")

if not base or not secret:
    print("ERROR: set WEB_SERVICE_URL and CRON_SECRET env vars.", flush=True)
    sys.exit(1)

url = f"{base}/internal/crawl?target={target}"
req = urllib.request.Request(url, method="POST", headers={"X-Cron-Secret": secret})
try:
    # timeout is generous: on Render free tier the first request also wakes the web service.
    with urllib.request.urlopen(req, timeout=120) as r:
        print(f"{r.status} {r.read().decode()}", flush=True)
except Exception as e:
    print(f"Trigger failed: {e}", flush=True)
    sys.exit(1)
