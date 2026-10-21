#!/usr/bin/env python3
"""Read the shared ComfyUI queue before an integration test. GET /queue only.

The allowed origins are the shared server's loopback address and docker0. Any other
origin is refused. This command does not submit a prompt, and it does not start, stop,
interrupt, unload, or clear that server. A non-empty queue, an unreadable queue, or a
missing owner process is not clear.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ORIGINS = ("http://127.0.0.1:8188", "http://172.17.0.1:8188")


def counts(payload: dict) -> dict:
    if not isinstance(payload, dict):
        return {"running": None, "pending": None}
    running, pending = payload.get("queue_running"), payload.get("queue_pending")
    return {
        "running": len(running) if isinstance(running, list) else None,
        "pending": len(pending) if isinstance(pending, list) else None,
    }


def assess(payload: dict, owner_alive: bool | None = None) -> dict:
    """Whether an integration test may submit. ``owner_alive`` None skips the pid check."""

    found = counts(payload)
    running, pending = found["running"], found["pending"]
    reason = ""
    clear = True
    if running is None or pending is None:
        clear, reason = False, "queue_unreadable"
    elif running or pending:
        clear, reason = False, "queue_not_empty"
    elif owner_alive is False:
        clear, reason = False, "owner_not_running"
    return {"clear": clear, "running": running, "pending": pending, "reason": reason, "submit": False}


def owner_running(pid: int) -> bool:
    """True when that pid's command line is the shared ComfyUI process. This does not signal it."""

    try:
        text = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode("utf-8", "replace")
    except OSError:
        return False
    return "ComfyUI" in text or "main.py" in text


def fetch(origin: str, timeout: float = 3.0) -> dict:
    if origin not in ORIGINS:
        raise ValueError("protected_endpoint")
    request = urllib.request.Request(origin + "/queue", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read(65536)
    except (OSError, urllib.error.URLError, TimeoutError) as error:
        raise OSError(str(error)) from error
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("queue_unreadable") from error
    if not isinstance(payload, dict):
        raise ValueError("queue_unreadable")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GET the shared ComfyUI queue. This does not submit a prompt.")
    parser.add_argument("--origin", default=ORIGINS[0], choices=ORIGINS)
    parser.add_argument("--owner-pid", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=3.0)
    args = parser.parse_args(argv)
    try:
        payload = fetch(args.origin, args.timeout)
    except (OSError, ValueError) as error:
        print(json.dumps({"clear": False, "reason": "queue_unreadable", "error": str(error), "submit": False}))
        return 1
    alive = owner_running(args.owner_pid) if args.owner_pid else None
    report = assess(payload, alive)
    print(json.dumps(report))
    return 0 if report["clear"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
