"""A record of jobs. This file is not the queue.

The live queue is the in-memory ``jobs`` dict in the server process. Mounting the
data directory does not restore that queue and does not run a job again. A record
left ``queued`` or ``running`` was an interruption: the next start marks it
``interrupted``. A graceful stop marks it ``cancelled``. Neither status is success.
"""
from __future__ import annotations

import fcntl
import json
import os
import threading
import time
from pathlib import Path

NOTE = "This file is a record of jobs. It is not the queue. Starting the studio does not run these jobs again."
CANCELLED = "The studio shut down and cancelled this job. It was not a success and it was not run again."
INTERRUPTED = "The studio stopped before this job finished. It was not a success and it was not run again."
ACTIVE = {"queued", "running"}
KEEP = 200

_lock = threading.Lock()


def empty() -> dict:
    return {"queue": False, "note": NOTE, "jobs": []}


def path() -> Path:
    named = os.environ.get("STUDIO_JOB_HISTORY", "").strip()
    if named:
        return Path(named).expanduser()
    from studio_home import studio_home
    return studio_home() / "job-history.json"


def _row(job: dict) -> dict:
    return {
        "id": str(job.get("id") or ""),
        "kind": str(job.get("kind") or ""),
        "status": str(job.get("status") or ""),
        "title": str(job.get("title") or "")[:120],
        "error": str(job.get("error") or "")[:400],
        "created": job.get("created"),
        "finished": job.get("finished"),
    }


def _update(mutator) -> dict:
    """Read, change, and replace the record. The replacement is one rename."""

    file = path()
    file.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        fd = os.open(file, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            raw = b""
            while True:
                block = os.read(fd, 1024 * 1024)
                if not block:
                    break
                raw += block
                if len(raw) > 8_000_000:
                    break
            try:
                data = json.loads(raw.decode("utf-8")) if raw.strip() else empty()
            except (UnicodeError, json.JSONDecodeError):
                aside = file.with_name(file.name + ".unreadable")
                aside.write_bytes(raw[:100_000])
                data = empty()
            if not isinstance(data, dict) or not isinstance(data.get("jobs"), list):
                data = empty()
            mutator(data)
            data["queue"] = False
            data["note"] = NOTE
            data["jobs"] = [row for row in data["jobs"] if isinstance(row, dict) and row.get("id")][-KEEP:]
            text = json.dumps(data, indent=1) + "\n"
            part = file.with_name(file.name + ".part")
            part.write_text(text)
            os.replace(part, file)
            return data
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)


def remember(job: dict) -> None:
    """Store one minimal record. Parameters, logs, and prompts stay out of it."""

    row = _row(job)
    if not row["id"]:
        return

    def mutate(data: dict) -> None:
        data["jobs"] = [one for one in data["jobs"] if one.get("id") != row["id"]]
        data["jobs"].append(row)

    _update(mutate)


def recover() -> list[dict]:
    """Mark queued and running records interrupted. Return those rows. Do not requeue them."""

    file = path()
    if not file.is_file():
        return []
    changed: list[dict] = []

    def mutate(data: dict) -> None:
        for row in data["jobs"]:
            if row.get("status") in ACTIVE:
                row["status"] = "interrupted"
                row["error"] = INTERRUPTED
                row["finished"] = row.get("finished") or time.time()
                changed.append(dict(row))

    _update(mutate)
    return changed


def load() -> dict:
    file = path()
    try:
        data = json.loads(file.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError):
        return empty()
    if not isinstance(data, dict) or not isinstance(data.get("jobs"), list):
        return empty()
    data["queue"] = False
    data["note"] = NOTE
    return data
