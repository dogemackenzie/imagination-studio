"""Admission and client ownership for a shared ComfyUI. Never manages the server.

The backend serializes GPU workflows. Our file lock serializes Studio clients only;
it cannot reserve the backend against another application's submissions.
"""
from __future__ import annotations

import contextlib
try:
    import fcntl
except ImportError:  # imported by the cross-platform app; these still routes are Linux only
    fcntl = None
import json
import math
import os
from pathlib import Path

HOST_MARGIN_MB = 512
VRAM_MARGIN_MIB = 256


def budgets(sample: dict, profile: dict | None, weight_mib: int) -> dict:
    """Use the observed cold-load deltas, never weight-file size as a RAM estimate."""
    profile = profile or {}
    render = profile.get("render") or {}
    pairs = (("host_available_mb", "host_available_mb_min"), ("vram_free_mib", "vram_free_mib_min"))
    values = [(profile.get(before), render.get(after)) for before, after in pairs]
    valid = all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x >= 0
                for pair in values for x in pair)
    valid = valid and all(before >= after for before, after in values)
    if not valid:
        return {**sample, "profile": False, "need_host_mb": None, "need_vram_mib": weight_mib,
                "measurement_missing": True}
    host, vram = (math.ceil(before - after) for before, after in values)
    return {**sample, "profile": True, "need_host_mb": host + HOST_MARGIN_MB,
            "need_vram_mib": max(weight_mib, vram) + VRAM_MARGIN_MIB}



def shared_vram(sample: dict, free_bytes) -> dict:
    """Use the lower available reading; never count another process's allocation as free."""
    if not isinstance(free_bytes, int) or isinstance(free_bytes, bool) or free_bytes < 0:
        return {**sample, "vram_known": False, "vram_free_mib": None}
    free = free_bytes // (1024 * 1024)
    observed = sample.get("vram_free_mib")
    if sample.get("vram_known") and isinstance(observed, (int, float)) and math.isfinite(observed):
        free = min(free, observed)
    return {**sample, "vram_known": True, "vram_free_mib": free}


def decide(sample: dict) -> dict:
    import rtx_admit
    if sample.get("measurement_missing"):
        return {"ok": False, "code": "unmeasured_configuration",
                "reason": "No complete cold-load RAM and VRAM measurement is recorded for this recipe."}
    found = rtx_admit.decide(sample)
    if found["code"] == "insufficient_memory":
        found["reason"] = (f"This still needs {sample['need_host_mb']} MB of measured RAM headroom including margin, "
                           f"plus the {rtx_admit.host_floor_mb(sample.get('host_total_mb'))} MB reserve; "
                           f"{sample.get('host_available_mb')} MB is available. The shared backend is left unchanged.")
    elif found["code"] == "insufficient_vram":
        found["reason"] = (f"This still needs {sample['need_vram_mib']} MiB of GPU headroom including margin, "
                           f"plus the {rtx_admit.vram_reserve_mib()} MiB reserve; {sample.get('vram_free_mib')} MiB is free.")
    return found


def state_path() -> Path:
    return Path(os.environ.get("STUDIO_HOME", Path.home() / "ImaginationStudio")) / "shared-prompt.json"


def _save(value: dict) -> None:
    path = state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix('.part')
    part.write_text(json.dumps(value) + '\n')
    part.replace(path)


@contextlib.contextmanager
def session(exchange, origin: str):
    """A cancelled/timed-out client never loses track of a possibly running prompt."""
    if fcntl is None:
        raise RuntimeError("The shared Linux still client is unavailable on this platform.")
    path = state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another Studio client is using the shared backend.") from None
        if path.exists():
            try:
                pending = json.loads(path.read_text())
            except (OSError, ValueError):
                raise RuntimeError("The shared-prompt record is unreadable; review it before another submission.") from None
            prompt_id = pending.get('prompt_id')
            if pending.get('origin') != origin or not prompt_id:
                raise RuntimeError("A previous shared submission has an uncertain outcome. Review the backend and Studio's shared-prompt record before retrying.")
            history = exchange('GET', '/history/' + prompt_id, timeout=10)
            status = (history.get(prompt_id) or {}).get('status') or {}
            if not (status.get('completed') or status.get('status_str') in ('success', 'error')):
                raise RuntimeError("A previous Studio prompt may still be running on shared ComfyUI. Wait for it to finish; Studio will not interrupt it.")
            path.unlink()
        yield


def submit(exchange, origin: str, payload: dict) -> dict:
    # Save before the POST: a lost response is not permission to submit a duplicate.
    _save({'origin': origin, 'state': 'submitting'})
    reply = exchange('POST', '/prompt', payload, timeout=60)
    prompt_id = str(reply.get('prompt_id') or '')
    if prompt_id:
        _save({'origin': origin, 'state': 'submitted', 'prompt_id': prompt_id})
    return reply


def finished() -> None:
    state_path().unlink(missing_ok=True)
