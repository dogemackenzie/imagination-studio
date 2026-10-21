#!/usr/bin/env python3
"""GET /api/health inside this container. The standard library only.

No disk walk, GPU query, update fetch, or model load.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request


def main() -> int:
    port = os.environ.get("PORT", "7870").strip() or "7870"
    bind = os.environ.get("STUDIO_BIND", "127.0.0.1").strip() or "127.0.0.1"
    host = "127.0.0.1" if bind in {"0.0.0.0", "::", "::0"} else bind
    url = f"http://{host}:{port}/api/health"
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            body = response.read(4096)
            code = response.status
    except (OSError, TimeoutError, ValueError) as error:
        print(f"health check failed: {error}", file=sys.stderr)
        return 1
    try:
        data = json.loads(body)
    except ValueError:
        print("health check failed: not json", file=sys.stderr)
        return 1
    if code != 200 or data.get("ok") is not True:
        print(f"health check failed: {code} {body[:200]!r}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
