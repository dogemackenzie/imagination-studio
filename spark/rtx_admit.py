#!/usr/bin/env python3
"""RTX admission beside the Spark guard. Spark tables in memory_guard.py stay as they were.

Three budgets are read separately: host MemAvailable, cgroup v2 headroom, and free VRAM.
Host RAM is not added to VRAM. A missing reading is not free capacity. Another program's GPU
memory is not reclaimed. The host floor stays at least 8000 MB.

Nothing here starts a renderer, submits a prompt, or signals a process it does not own.
An empty measured profile keeps ordinary image jobs refused until a profile is recorded.
"""
from __future__ import annotations

import json
import os
import signal
import sys
import time
import urllib.parse
from pathlib import Path

import memory_guard

# The same floor as capabilities.FLOOR_MB and scripts/doctor.py. An environment value cannot lower it.
HOST_FLOOR_MB = 8000
# Kept for the app itself. Measured app-only use was about 70 MiB. This is not a model size.
CONTAINER_RESERVE_MB = 256
# Left unused on the GPU after other programs' allocations. Not a measured model need.
VRAM_RESERVE_MIB = 512
# cgroup v2 "full" avg10, in percent. At or above this the cgroup is already stalled.
PRESSURE_FULL_AVG10 = 5.0
VRAM_CACHE_S = 1.0
PROTECTED_PORTS = {8188}
# The measured RTX route also refuses the shared host port and the other studio's published renderer.
MEASURED_FORBIDDEN_PORTS = {8188, 8190}

REASONS = {
    "insufficient_memory": "Available host memory is under the 8000 MB floor, so a render cannot start.",
    "insufficient_cgroup": "Container memory headroom is below what a render must leave free, so a render cannot start.",
    "insufficient_vram": "Free GPU memory is below the reserve. Memory held by another program is not reclaimed.",
    "occupied_port": "The renderer port is in use by a process this Studio does not own.",
    "protected_endpoint": "The host renderer at this address is protected. Studio does not start, stop, or submit work there.",
    "foreign_lock": "A renderer lock belongs to another process. It was left in place.",
    "missing_telemetry": "A memory or GPU reading failed. A missing reading is not treated as free capacity.",
    "busy_renderer": "The renderer is busy with another job.",
    "unmeasured_configuration": (
        "This machine has no measured render route. An RTX card is not a production render mode, "
        "and a managed RTX image does not enable one."
    ),
    "not_owned": "This Studio does not own the renderer at this address. A server answering the port is not ownership.",
}


def host_floor_mb(total_mb: int | None = None) -> int:
    """At least 8000 MB. A larger Spark floor is kept. A smaller requested floor is ignored."""

    return max(HOST_FLOOR_MB, memory_guard.floor_mb(total_mb))


def _int_env(name: str, default: int, least: int) -> int:
    raw = os.environ.get(name, "").strip()
    if raw.isdigit():
        return max(least, int(raw))
    return default


def container_reserve_mb() -> int:
    return _int_env("RTX_CONTAINER_RESERVE_MB", CONTAINER_RESERVE_MB, 64)


def vram_reserve_mib() -> int:
    return _int_env("RTX_VRAM_RESERVE_MIB", VRAM_RESERVE_MIB, 64)


def owner_path() -> Path:
    named = os.environ.get("STUDIO_RENDER_OWNER", "").strip()
    if named:
        return Path(named)
    home = os.environ.get("STUDIO_HOME", "").strip()
    if home:
        return Path(home) / "renderer.owner"
    state = os.environ.get("STUDIO_COMFY_STATE", "").strip() or "/tmp"
    return Path(state) / "renderer.owner"


def slot_path() -> Path:
    return owner_path().with_name("renderer.slot")


def origin_of(url: str) -> str:
    parsed = urllib.parse.urlsplit(url.strip())
    host = (parsed.hostname or "").lower()
    port = parsed.port
    scheme = parsed.scheme or "http"
    if port is None:
        port = 443 if scheme == "https" else 80
    return f"{scheme}://{host}:{port}"


def port_of(url: str) -> int | None:
    try:
        return urllib.parse.urlsplit(origin_of(url)).port
    except ValueError:
        return None


def protected(url: str) -> bool:
    """The shared host ComfyUI port, plus any address named in STUDIO_PROTECTED_URLS."""

    port = port_of(url)
    if port in PROTECTED_PORTS:
        return True
    origin = origin_of(url)
    extra = os.environ.get("STUDIO_PROTECTED_URLS", "")
    named = {origin_of(part) for part in extra.split(",") if part.strip()}
    return origin in named


def pressure_full_avg10(text: str) -> float | None:
    for line in text.splitlines():
        if line.startswith("full "):
            for part in line.split():
                if part.startswith("avg10="):
                    try:
                        return float(part.split("=", 1)[1])
                    except ValueError:
                        return None
    return None


def read_cgroup(leaf: Path, root: Path, reserve_mb: int | None = None) -> dict:
    """Headroom across this cgroup and its parents. An unreadable limit is unknown, not unlimited.

    ``memory.max`` of ``max`` is a known absence of a cap at that level. A numeric cap uses that
    level's own ``memory.current``. The tightest level wins. ``reserve_mb`` is subtracted there.

    The cgroup v2 mount root often has no ``memory.max`` (only ``memory.pressure``). That missing
    file ends the walk. A chain of ``max`` up to that root is a known absence of a cap, and the
    host floor still applies. A missing ``memory.max`` on a directory below the root stays unknown.
    """

    reserve = container_reserve_mb() if reserve_mb is None else reserve_mb
    try:
        leaf_r, root_r = leaf.resolve(), root.resolve()
    except OSError:
        return {"known": False, "cap": False, "headroom_mb": None, "pressure_full_avg10": None}
    if leaf_r != root_r and root_r not in leaf_r.parents:
        return {"known": False, "cap": False, "headroom_mb": None, "pressure_full_avg10": None}
    levels: list[tuple[int, int, float]] = []
    current = leaf_r
    while True:
        at_root = current == root_r
        limit_path = current / "memory.max"
        if not limit_path.is_file():
            if at_root:
                break
            return {"known": False, "cap": False, "headroom_mb": None, "pressure_full_avg10": None}
        try:
            raw = limit_path.read_text().strip()
        except OSError:
            return {"known": False, "cap": False, "headroom_mb": None, "pressure_full_avg10": None}
        if raw != "max":
            used_path = current / "memory.current"
            pressure_path = current / "memory.pressure"
            try:
                limit_b = int(raw)
                used_b = int(used_path.read_text().strip())
                pressure = pressure_full_avg10(pressure_path.read_text())
            except (OSError, ValueError):
                return {"known": False, "cap": False, "headroom_mb": None, "pressure_full_avg10": None}
            if pressure is None:
                return {"known": False, "cap": False, "headroom_mb": None, "pressure_full_avg10": None}
            levels.append((limit_b, used_b, pressure))
        if at_root:
            break
        current = current.parent
    if not levels:
        return {"known": True, "cap": False, "headroom_mb": None, "pressure_full_avg10": 0.0}
    headroom = min((limit_b - used_b) // (1024 * 1024) - reserve for limit_b, used_b, _pressure in levels)
    pressure = max(item[2] for item in levels)
    return {"known": True, "cap": True, "headroom_mb": headroom, "pressure_full_avg10": pressure}


def _refuse(code: str, reason: str | None = None) -> dict:
    text = reason or REASONS[code]
    return {"ok": False, "fits": False, "code": code, "reason": text, "reasons": [{"code": code, "reason": text}]}


def decide(sample: dict) -> dict:
    """Whether one local GPU job may start. ``profile`` stays false until a measured RTX route exists.

    ``sample`` carries the three budgets already read by the caller. None or ``known: false`` is a
    failed probe. Free VRAM is what was free, not total minus a guess about other programs.
    """

    if sample.get("foreign_lock"):
        return _refuse("foreign_lock")
    if sample.get("occupied_port") or sample.get("protected"):
        code = "protected_endpoint" if sample.get("protected") else "occupied_port"
        return _refuse(code)
    host = sample.get("host_available_mb")
    cgroup = sample.get("cgroup") or {}
    vram_known = bool(sample.get("vram_known"))
    if host is None or not cgroup.get("known") or not vram_known:
        return _refuse("missing_telemetry")
    floor = host_floor_mb(sample.get("host_total_mb"))
    need_host = sample.get("need_host_mb")
    if host < floor or (need_host is not None and host - floor < need_host):
        return _refuse("insufficient_memory")
    if cgroup.get("cap"):
        headroom = cgroup.get("headroom_mb")
        pressure = cgroup.get("pressure_full_avg10")
        need = sample.get("need_host_mb")
        if headroom is None or pressure is None:
            return _refuse("missing_telemetry")
        if pressure >= PRESSURE_FULL_AVG10 or headroom < 0 or (need is not None and headroom < need):
            return _refuse("insufficient_cgroup")
    free = sample.get("vram_free_mib")
    reserve = sample.get("vram_reserve_mib")
    if reserve is None:
        reserve = vram_reserve_mib()
    need_vram = sample.get("need_vram_mib")
    if free is None or free < reserve or (need_vram is not None and free - reserve < need_vram):
        return _refuse("insufficient_vram")
    if sample.get("busy"):
        return _refuse("busy_renderer")
    if not sample.get("profile"):
        return _refuse("unmeasured_configuration")
    return {"ok": True, "fits": True, "code": "", "reason": "", "reasons": [], "floor_mb": floor}


class VramCache:
    """One successful GPU reading is reused for at most ``ttl_s``. A later failure is returned as failure."""

    def __init__(self, probe, ttl_s: float = VRAM_CACHE_S, now=time.monotonic):
        self.probe = probe
        self.ttl_s = ttl_s
        self.now = now
        self.at: float | None = None
        self.value = None

    def read(self):
        moment = self.now()
        if self.at is not None and self.value is not None and moment - self.at < self.ttl_s:
            return self.value
        self.value = self.probe()
        self.at = moment
        return self.value


def budget_breach(sample: dict) -> str:
    """Why a running owned renderer must stop. An unmeasured profile is not a breach by itself."""

    found = decide({**sample, "profile": True, "foreign_lock": False, "occupied_port": False, "protected": False, "busy": False})
    if found["ok"]:
        return ""
    if found["code"] in ("insufficient_memory", "insufficient_cgroup", "insufficient_vram", "missing_telemetry"):
        return found["reason"]
    return ""


def plain_watch(owned_pid: int, samples: list[dict], stop) -> dict:
    """The plain loop: the same budgets, without the Spark meter. ``stop`` is called only with ``owned_pid``."""

    for sample in samples:
        why = budget_breach(sample)
        if why:
            stop(owned_pid)
            return {"stopped": owned_pid, "reason": why}
    return {"stopped": None, "reason": ""}


def watch_owned(owned_pid: int, probe, stop, ticks: int = 1) -> dict:
    """The Python guard for an owned renderer. ``probe`` returns one sample per tick."""

    last = {"stopped": None, "reason": ""}
    for _ in range(ticks):
        last = plain_watch(owned_pid, [probe()], stop)
        if last["stopped"] is not None:
            return last
    return last


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload) + "\n")
    os.replace(temp, path)


def _read_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def read_identity(pid: int) -> dict:
    """starttime and pid-namespace id from /proc. ``alive`` is false when the process is gone."""

    stat = Path(f"/proc/{pid}/stat")
    ns = Path(f"/proc/{pid}/ns/pid")
    try:
        text = stat.read_text()
        starttime = int(text.rsplit(")", 1)[1].split()[19])
        namespace = os.readlink(ns)
        state = text.rsplit(")", 1)[1].split()[0]
    except (OSError, IndexError, ValueError):
        return {"alive": False, "starttime": None, "ns": None}
    return {"alive": state != "Z", "starttime": starttime, "ns": namespace}


def claim(endpoint: str, pid: int, identity: dict | None = None) -> dict:
    """Record that this Studio started ``pid`` at ``endpoint``. A protected address is never recorded."""

    if protected(endpoint):
        return _refuse("protected_endpoint")
    who = identity if identity is not None else read_identity(pid)
    if not who.get("alive"):
        return _refuse("not_owned", "The renderer process is not running, so it was not claimed.")
    payload = {"endpoint": origin_of(endpoint), "pid": pid, "starttime": who.get("starttime"), "ns": who.get("ns")}
    _write_json(owner_path(), payload)
    return {"ok": True, "fits": True, "code": "", "reason": "", "owner": payload}


def allow(endpoint: str, identity_of=read_identity) -> dict:
    """Whether Studio may cancel, unload, or delete a queue on ``endpoint``.

    A listening port is not enough. The owner file must name this endpoint and the same live
    process that was claimed. A reused pid or a different pid namespace is a foreign lock and
    is not signaled.
    """

    if protected(endpoint):
        return _refuse("protected_endpoint")
    record = _read_json(owner_path())
    if not record or record.get("endpoint") != origin_of(endpoint):
        return _refuse("not_owned")
    try:
        pid = int(record.get("pid"))
    except (TypeError, ValueError):
        return _refuse("not_owned")
    who = identity_of(pid)
    if not who.get("alive"):
        return _refuse("not_owned")
    if who.get("starttime") != record.get("starttime") or who.get("ns") != record.get("ns"):
        return _refuse("foreign_lock")
    return {"ok": True, "fits": True, "code": "", "reason": "", "pid": pid}


def recover() -> dict:
    """Drop an owner file whose pid is gone. A live pid with a different identity is left untouched."""

    path = owner_path()
    record = _read_json(path)
    if not record:
        return {"ok": True, "cleared": False, "code": ""}
    try:
        pid = int(record.get("pid"))
    except (TypeError, ValueError):
        path.unlink(missing_ok=True)
        return {"ok": True, "cleared": True, "code": ""}
    who = read_identity(pid)
    if not who.get("alive"):
        path.unlink(missing_ok=True)
        return {"ok": True, "cleared": True, "code": ""}
    if who.get("starttime") != record.get("starttime") or who.get("ns") != record.get("ns"):
        return _refuse("foreign_lock")
    return {"ok": True, "cleared": False, "code": ""}


def hold_slot(pid: int) -> None:
    _write_json(slot_path(), {"pid": pid})


def clear_slot(pid: int | None = None) -> None:
    path = slot_path()
    record = _read_json(path)
    if pid is not None and record and record.get("pid") != pid:
        return
    path.unlink(missing_ok=True)


def slot_busy(identity_of=read_identity) -> bool:
    record = _read_json(slot_path())
    if not record:
        return False
    try:
        pid = int(record.get("pid"))
    except (TypeError, ValueError):
        return False
    return bool(identity_of(pid).get("alive"))


def stop_owned(pid: int) -> None:
    os.kill(pid, signal.SIGKILL)


def block_foreign_render(spark: bool | None = None) -> str | None:
    """A reason to stop before any renderer URL is opened. A Spark keeps its own guard."""

    if spark is None:
        flag = os.environ.get("RTX_ADMIT_SPARK", "").strip()
        if flag == "1":
            spark = True
        elif flag == "0":
            spark = False
        elif sys.platform == "darwin":
            spark = False
        else:
            app = Path(__file__).resolve().parent.parent / "app"
            if str(app) not in sys.path:
                sys.path.insert(0, str(app))
            import hardware
            spark = hardware.is_spark()
    if spark or sys.platform == "darwin":
        return None
    found = decide(collect())
    if found["ok"]:
        return None
    # A measured SDXL Turbo route may pass the empty-profile gate. Every other refusal still stands,
    # and the shared renderer ports stay closed.
    if found["code"] == "unmeasured_configuration" and _sdxl_measured_open():
        return measured_url_block()
    return found["reason"]


def _sdxl_measured_open() -> bool:
    if os.environ.get("RTX_MEASURED_ROUTE", "").strip() != "sdxl-turbo":
        return False
    if os.environ.get("RTX_SDXL_CALIBRATE", "").strip() == "1":
        return True
    import sdxl_turbo
    return sdxl_turbo.offered()


def measured_url_block() -> str | None:
    """A reason to refuse the measured route's renderer URL. None when the URL is an owned loopback port."""

    url = os.environ.get("SPARK_COMFY_URL") or (
        "http://127.0.0.1:" + os.environ.get("SPARK_COMFY_PORT", "8190"))
    if protected(url) or port_of(url) in MEASURED_FORBIDDEN_PORTS:
        return REASONS["protected_endpoint"]
    return None


def _host_sample() -> tuple[int | None, int | None]:
    info = memory_guard.meminfo()
    return info.get("MemAvailable"), info.get("MemTotal")


def _self_cgroup() -> dict:
    try:
        line = Path("/proc/self/cgroup").read_text().splitlines()[-1]
        path = line.split(":", 2)[-1].strip() or "/"
    except (OSError, IndexError):
        return {"known": False, "cap": False, "headroom_mb": None, "pressure_full_avg10": None}
    root = Path("/sys/fs/cgroup")
    leaf = root / path.lstrip("/")
    if not leaf.is_dir():
        return {"known": False, "cap": False, "headroom_mb": None, "pressure_full_avg10": None}
    return read_cgroup(leaf, root)


def _vram_probe() -> int | None:
    import subprocess
    try:
        done = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    line = (done.stdout or "").strip().splitlines()
    if done.returncode != 0 or not line:
        return None
    try:
        return int(line[0].strip())
    except ValueError:
        return None


def collect() -> dict:
    """One admission sample from this machine. The measured RTX profile is empty, so a job stays refused."""

    host, total = _host_sample()
    cgroup = _self_cgroup()
    free = _vram_probe()
    return {
        "host_available_mb": host,
        "host_total_mb": total,
        "cgroup": cgroup,
        "vram_free_mib": free,
        "vram_known": free is not None,
        "need_host_mb": None,
        "need_vram_mib": None,
        "profile": False,
        "occupied_port": False,
        "protected": False,
        "foreign_lock": False,
        "busy": slot_busy(),
    }


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Print RTX admission. This does not start a render.")
    parser.add_argument("command", choices=("admit",))
    parser.parse_args(argv)
    found = decide(collect())
    print(json.dumps({k: found[k] for k in ("ok", "code", "reason")}))
    return 0 if found["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
