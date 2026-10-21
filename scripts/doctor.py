#!/usr/bin/env python3
"""Read-only baseline for this machine and the shared ComfyUI service.

    python3 scripts/doctor.py
    python3 scripts/doctor.py --json
    python3 scripts/doctor.py --json --gpu-container --select-model checkpoints/sd_xl_turbo_1.0_fp16.safetensors

The shared ComfyUI also serves another application. This command only reads: GET requests to the
loopback inventory routes, file sizes, and (when asked) one model hash and one disposable nvidia-smi
container. It does not submit prompts, upload files, or start, stop, or reconfigure that service.
Hashes run only for --select-model. Rendering stays disabled in the report.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import io
import json
import os
import platform
import re
import shutil
import socket
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

DOCTOR_VERSION = 1
# Same rule as spark/memory_guard.py floor_mb. Reported here so a later RTX route can see it.
# Do not lower it to make a job fit.
HOST_FLOOR_MB = 8000
MAX_HASH_BYTES = 20 * 1024 ** 3
MODEL_SUFFIXES = {".safetensors", ".ckpt", ".pt", ".pth", ".gguf", ".bin"}
SKIP_DIRS = {"__pycache__", ".git", "input", "output", "temp", "output_images"}
CANDIDATES = ("checkpoints/sd_xl_turbo_1.0_fp16.safetensors",)
NODE_NAMES = (
    "CheckpointLoaderSimple",
    "CLIPTextEncode",
    "KSampler",
    "VAEDecode",
    "SaveImage",
    "EmptyLatentImage",
)
WATCH_PORTS = (
    (8188, "shared ComfyUI; leave this listener alone"),
    (8190, "occupied listener; identify the publisher and leave it alone"),
    (7870, "proposed Studio app"),
    (8191, "proposed managed renderer"),
)
TOOL_NAMES = ("python3", "ffmpeg", "ffprobe", "nvcc", "uv", "docker", "git", "node", "nvidia-smi", "nvidia-ctk")
LOCAL_IMAGE_PREFERENCE = ("ubuntu:24.04", "ubuntu:22.04", "ubuntu:latest", "debian:bookworm-slim", "debian:stable-slim")
FALLBACK_PULL = "ubuntu:24.04"
CUDA_IMAGE = re.compile(r"^nvidia/cuda:[A-Za-z0-9._-]+$")
GPU_UUID = re.compile(r"^GPU-[0-9a-fA-F-]+$")
COMMIT = re.compile(r"^[0-9a-f]{40}$")
SAFETY = {
    "comfy_methods": ["GET"],
    "comfy_paths": ["/system_stats", "/queue (counts only)", "/object_info/<selected node> (presence only)"],
    "gpu_container": "only with --gpu-container; nvidia-smi; no project or model mounts",
    "rendering_enabled": False,
}


class ProbeError(ValueError):
    """The command line asked for something the probe must not do."""


class RedirectRefused(Exception):
    """The ComfyUI endpoint tried to send the probe somewhere else."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RedirectRefused()


@dataclass(frozen=True)
class Options:
    json_output: bool
    comfy_url: str
    comfy_dir: Path | None
    models_dir: Path | None
    select_model: str | None
    timeout: float
    hash_timeout: float
    pull_timeout: float
    container_timeout: float
    gpu_container: bool
    skip_comfy: bool
    skip_docker: bool
    skip_gpu: bool
    skip_models: bool
    repo: Path


def run_cmd(argv: list[str], timeout: float) -> dict:
    try:
        completed = subprocess.run(
            argv,
            timeout=timeout,
            check=False,
            capture_output=True,
            text=True,
            shell=False,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "code": None, "stdout": "", "stderr": "", "error": "timeout"}
    except OSError:
        return {"ok": False, "code": None, "stdout": "", "stderr": "", "error": "os_error"}
    return {
        "ok": completed.returncode == 0,
        "code": completed.returncode,
        # Callers parse this. Truncation used to hide the nvidia runtime in a large docker info document.
        "stdout": completed.stdout[-1_000_000:],
        "stderr": completed.stderr[-2000:],
        "error": None if completed.returncode == 0 else "nonzero",
    }


def reported_floor(total_mb: int) -> dict:
    asked = os.environ.get("SPARK_MEMORY_FLOOR_MB", "").strip()
    if asked.isdigit() and int(asked) > 0:
        return {"floor_mb": int(asked), "source": "SPARK_MEMORY_FLOOR_MB"}
    return {"floor_mb": max(HOST_FLOOR_MB, total_mb // 10), "source": "max(8000, MemTotal/10)"}


def meminfo() -> dict[str, int]:
    try:
        text = Path("/proc/meminfo").read_text(errors="replace")
    except OSError:
        return {}
    found = {}
    for line in text.splitlines():
        name, _, rest = line.partition(":")
        parts = rest.split()
        if parts and parts[0].isdigit():
            found[name] = int(parts[0]) // 1024
    return found


def host_info() -> dict:
    release = {}
    try:
        for line in Path("/etc/os-release").read_text(errors="replace").splitlines():
            key, _, value = line.partition("=")
            if key:
                release[key] = value.strip().strip('"')
    except OSError:
        pass
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "kernel": platform.release(),
        "pretty": release.get("PRETTY_NAME"),
        "version_id": release.get("VERSION_ID"),
        "python": platform.python_version(),
    }


def cpu_info() -> dict:
    model, flags, cores = None, "", 0
    try:
        text = Path("/proc/cpuinfo").read_text(errors="replace")
    except OSError:
        text = ""
    for line in text.splitlines():
        if line.startswith("model name") and model is None:
            model = line.split(":", 1)[1].strip()
        elif line.startswith("processor"):
            cores += 1
        elif line.startswith("flags") and not flags:
            flags = line.split(":", 1)[1]
    present = set(flags.split())
    wanted = ("sse4_2", "avx", "avx2", "avx512f")
    return {"model": model, "logical_cpus": cores, "flags": {name: name in present for name in wanted}}


def disk_points(paths: list[tuple[str, Path]]) -> list[dict]:
    rows = []
    for label, path in paths:
        if not path.exists():
            rows.append({"label": label, "path": str(path), "present": False})
            continue
        usage = shutil.disk_usage(path)
        rows.append({
            "label": label,
            "path": str(path),
            "present": True,
            "total_bytes": usage.total,
            "free_bytes": usage.free,
        })
    return rows


def resources(extra_disks: list[tuple[str, Path]]) -> dict:
    info = meminfo()
    total = info.get("MemTotal", 0)
    available = info.get("MemAvailable", 0)
    floor = reported_floor(total)
    home = Path.home()
    return {
        "memory_unit": "MiB (kB from /proc/meminfo, divided by 1024)",
        "mem_total_mb": total,
        "mem_available_mb": available,
        "mem_free_mb": info.get("MemFree"),
        "swap_total_mb": info.get("SwapTotal"),
        "swap_free_mb": info.get("SwapFree"),
        "floor_mb": floor["floor_mb"],
        "floor_source": floor["source"],
        "available_below_floor": available < floor["floor_mb"],
        "disks": disk_points([("home", home), *extra_disks]),
    }


def _cgroup_file(path: Path) -> str | None:
    try:
        return path.read_text(errors="replace").strip()
    except OSError:
        return None


def cgroup_info() -> dict:
    try:
        text = Path("/proc/self/cgroup").read_text(errors="replace")
    except OSError:
        return {"version": None}
    relative = None
    for line in text.splitlines():
        parts = line.split(":", 2)
        if len(parts) == 3 and parts[0] == "0":
            relative = parts[2].lstrip("/")
    if relative is None:
        return {"version": 1, "note": "no cgroup v2 line on this process"}
    root = Path("/sys/fs/cgroup")
    current = root / relative
    chain = []
    for _ in range(8):
        if not current.is_relative_to(root):
            break
        entry = {"path": str(current)}
        for name in ("memory.max", "memory.high", "memory.current", "cpu.max"):
            value = _cgroup_file(current / name)
            if value is not None:
                entry[name] = value
                if value.isdigit():
                    entry[name + "_bytes"] = int(value)
        chain.append(entry)
        if current == root:
            break
        current = current.parent
    return {"version": 2, "self": chain[0] if chain else None, "parents": chain[1:]}


def tool_presence() -> dict:
    found = {}
    for name in TOOL_NAMES:
        path = shutil.which(name)
        found[name] = {"present": path is not None, "path": path}
    return found


def parse_gpu_csv(text: str) -> list[dict]:
    rows = []
    for record in csv.reader(io.StringIO(text)):
        if len(record) < 7:
            continue
        uuid, name, total, used, free, driver, cap = [part.strip() for part in record[:7]]
        rows.append({
            "uuid": uuid,
            "name": name,
            "memory_total_mib": int(total) if total.isdigit() else None,
            "memory_used_mib": int(used) if used.isdigit() else None,
            "memory_free_mib": int(free) if free.isdigit() else None,
            "driver_version": driver,
            "compute_cap": cap,
        })
    return rows


def parse_compute_apps(text: str) -> list[dict]:
    apps = []
    for record in csv.reader(io.StringIO(text)):
        if len(record) < 3:
            continue
        pid, name, used = [part.strip() for part in record[:3]]
        if not pid.isdigit():
            continue
        apps.append({"pid": int(pid), "name": name, "memory_mib": int(used) if used.isdigit() else None})
    return apps


def read_gpu(timeout: float) -> dict:
    query = run_cmd(
        ["nvidia-smi", "--query-gpu=uuid,name,memory.total,memory.used,memory.free,driver_version,compute_cap",
         "--format=csv,noheader,nounits"],
        timeout,
    )
    if not query["ok"]:
        return {"ok": False, "error": query["error"], "gpus": [], "compute_apps": []}
    apps = run_cmd(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"],
        timeout,
    )
    return {
        "ok": True,
        "gpus": parse_gpu_csv(query["stdout"]),
        "compute_apps": parse_compute_apps(apps["stdout"]) if apps["ok"] else [],
        "compute_apps_error": None if apps["ok"] else apps["error"],
    }


def select_gpu(gpus: list[dict]) -> dict | None:
    for gpu in gpus:
        if "3060" in (gpu.get("name") or ""):
            return gpu
    return gpus[0] if gpus else None


def parse_published_ports(text: str) -> list[dict]:
    """Keep docker ps rows that publish one of the ports this baseline cares about."""

    watched = {port for port, _purpose in WATCH_PORTS}
    rows = []
    for line in text.splitlines():
        name, separator, rest = line.partition("\t")
        image, separator2, ports = rest.partition("\t")
        if not separator or not separator2:
            continue
        mentioned = [port for port in sorted(watched) if f":{port}->" in ports or f":{port}/" in ports]
        if mentioned:
            rows.append({"name": name, "image": image[:120], "ports": ports[:240], "watched_ports": mentioned})
    return rows


def read_published(timeout: float) -> dict:
    result = run_cmd(["docker", "ps", "--format", "{{.Names}}\t{{.Image}}\t{{.Ports}}"], timeout)
    if not result["ok"]:
        return {"ok": False, "error": result["error"], "matches": []}
    return {"ok": True, "matches": parse_published_ports(result["stdout"])}


def read_docker(timeout: float) -> dict:
    version = run_cmd(["docker", "version", "--format", "{{.Server.Version}}"], timeout)
    compose = run_cmd(["docker", "compose", "version", "--short"], timeout)
    info = run_cmd(["docker", "info", "--format", "{{range $k, $v := .Runtimes}}{{println $k}}{{end}}"], timeout)
    runtimes = sorted({line.strip() for line in info["stdout"].splitlines() if line.strip()}) if info["ok"] else []
    nvidia = "nvidia" in runtimes
    return {
        "ok": version["ok"],
        "server_version": version["stdout"].strip() if version["ok"] else None,
        "compose_version": compose["stdout"].strip() if compose["ok"] else None,
        "runtimes": runtimes,
        "nvidia_runtime": nvidia,
        "error": None if version["ok"] else (version["error"] or "docker_unavailable"),
    }


def parse_proc_address(field: str, ipv6: bool) -> tuple[str, int]:
    ip_hex, _, port_hex = field.partition(":")
    port = int(port_hex, 16)
    raw = bytes.fromhex(ip_hex)
    if not ipv6:
        return ".".join(str(byte) for byte in reversed(raw)), port
    parts = [raw[index:index + 4][::-1] for index in range(0, 16, 4)]
    return socket.inet_ntop(socket.AF_INET6, b"".join(parts)), port


def parse_proc_net(text: str, ipv6: bool) -> list[dict]:
    found = []
    for line in text.splitlines()[1:]:
        columns = line.split()
        if len(columns) < 10 or columns[3] != "0A":
            continue
        try:
            address, port = parse_proc_address(columns[1], ipv6)
        except ValueError:
            continue
        uid = int(columns[7]) if columns[7].isdigit() else None
        found.append({"address": address, "port": port, "inode": columns[9], "uid": uid})
    return found


def process_brief(pid: str) -> dict:
    comm, cmdline = "", ""
    try:
        comm = (Path("/proc") / pid / "comm").read_text(errors="replace").strip()
    except OSError:
        pass
    try:
        raw = (Path("/proc") / pid / "cmdline").read_bytes()[:240]
        cmdline = raw.replace(b"\x00", b" ").decode("utf-8", "replace").strip()
    except OSError:
        pass
    return {"pid": int(pid), "comm": comm, "cmdline": cmdline}


def socket_owners(inodes: set[str], deadline: float) -> dict[str, list[dict]]:
    owners: dict[str, list[dict]] = {inode: [] for inode in inodes}
    if not inodes:
        return owners
    for entry in Path("/proc").iterdir():
        if time.monotonic() > deadline:
            break
        if not entry.name.isdigit():
            continue
        fd_dir = entry / "fd"
        try:
            descriptors = list(fd_dir.iterdir())
        except OSError:
            continue
        brief = None
        for descriptor in descriptors:
            try:
                target = os.readlink(descriptor)
            except OSError:
                continue
            if not (target.startswith("socket:[") and target.endswith("]")):
                continue
            inode = target[len("socket:["):-1]
            if inode not in owners:
                continue
            if brief is None:
                brief = process_brief(entry.name)
            if all(item["pid"] != brief["pid"] for item in owners[inode]):
                owners[inode].append(brief)
    return owners


def read_listeners(timeout: float) -> list[dict]:
    found = []
    for name, ipv6 in (("/proc/net/tcp", False), ("/proc/net/tcp6", True)):
        try:
            found.extend(parse_proc_net(Path(name).read_text(errors="replace"), ipv6))
        except OSError:
            continue
    interesting = {port for port, _purpose in WATCH_PORTS}
    needed = [row for row in found if row["port"] in interesting]
    owners = socket_owners({row["inode"] for row in needed}, time.monotonic() + timeout)
    for row in needed:
        row["owners"] = owners.get(row["inode"], [])
    return needed


def summarize_listeners(found: list[dict]) -> list[dict]:
    by_port: dict[int, list[dict]] = {}
    for row in found:
        by_port.setdefault(row["port"], []).append(row)
    summary = []
    for port, purpose in WATCH_PORTS:
        rows = by_port.get(port, [])
        owners = []
        seen = set()
        for row in rows:
            for owner in row.get("owners") or []:
                if owner["pid"] in seen:
                    continue
                seen.add(owner["pid"])
                owners.append(owner)
        summary.append({
            "port": port,
            "purpose": purpose,
            "listening": bool(rows),
            "addresses": sorted({row["address"] for row in rows}),
            "uids": sorted({row["uid"] for row in rows if row.get("uid") is not None}),
            "owners": owners,
        })
    return summary


def check_loopback(url: str) -> str:
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != "http" or parts.hostname != "127.0.0.1":
        raise ProbeError("ComfyUI probe is limited to http://127.0.0.1:<port>")
    if parts.username or parts.password or parts.query or parts.fragment or parts.path not in ("", "/"):
        raise ProbeError("ComfyUI base URL cannot carry credentials, a query, or a path")
    if parts.port is None:
        raise ProbeError("ComfyUI probe needs an explicit port")
    return f"http://127.0.0.1:{parts.port}"


def allowed_comfy_path(path: str) -> bool:
    if path in ("/system_stats", "/queue"):
        return True
    prefix = "/object_info/"
    if not path.startswith(prefix):
        return False
    name = path[len(prefix):]
    return name in NODE_NAMES and "/" not in name


def comfy_get(base: str, path: str, timeout: float, max_bytes: int) -> dict:
    """GET one inventory route. Redirects and any other path are refused, and the body is capped."""

    base = check_loopback(base)
    if not allowed_comfy_path(path):
        raise ProbeError("refusing ComfyUI path")
    request = urllib.request.Request(base + path, method="GET")
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        response = opener.open(request, timeout=timeout)
    except RedirectRefused:
        return {"ok": False, "status": None, "error": "redirect_refused"}
    except urllib.error.HTTPError as exc:
        return {"ok": False, "status": exc.code, "error": "http_error"}
    except (urllib.error.URLError, TimeoutError, socket.timeout):
        return {"ok": False, "status": None, "error": "unreachable"}
    try:
        chunks = []
        total = 0
        while True:
            block = response.read(65536)
            if not block:
                break
            total += len(block)
            if total > max_bytes:
                return {"ok": False, "status": getattr(response, "status", None), "error": "response_too_large"}
            chunks.append(block)
    finally:
        response.close()
    try:
        payload = json.loads(b"".join(chunks).decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        return {"ok": False, "status": getattr(response, "status", None), "error": "not_json"}
    return {"ok": True, "status": getattr(response, "status", 200), "json": payload}


def system_summary(payload: dict) -> dict:
    system = payload.get("system") if isinstance(payload, dict) else None
    system = system if isinstance(system, dict) else {}
    kept = ("os", "comfyui_version", "python_version", "pytorch_version", "ram_total", "ram_free")
    devices = []
    for device in payload.get("devices") or []:
        if not isinstance(device, dict):
            continue
        devices.append({key: device.get(key) for key in ("name", "type", "index", "vram_total", "vram_free")})
    return {"system": {key: system.get(key) for key in kept}, "devices": devices}


def queue_counts(payload: dict) -> dict:
    if not isinstance(payload, dict):
        return {"running": None, "pending": None}
    running, pending = payload.get("queue_running"), payload.get("queue_pending")
    return {
        "running": len(running) if isinstance(running, list) else None,
        "pending": len(pending) if isinstance(pending, list) else None,
    }


def node_presence(result: dict) -> dict:
    if result.get("error") == "response_too_large":
        return {"present": None, "schema_stored": False, "error": "response_too_large"}
    if not result.get("ok"):
        return {"present": False, "schema_stored": False, "error": result.get("error"), "status": result.get("status")}
    payload = result.get("json")
    return {"present": isinstance(payload, dict) and bool(payload), "schema_stored": False}


def systemd_show(unit: str, timeout: float) -> dict:
    result = run_cmd(
        ["systemctl", "--user", "show", unit, "-p", "Id", "-p", "ActiveState", "-p", "SubState",
         "-p", "MainPID", "-p", "FragmentPath", "-p", "MemoryCurrent"],
        timeout,
    )
    if not result["ok"] and not result["stdout"]:
        return {"ok": False, "error": result["error"]}
    shown = {}
    for line in result["stdout"].splitlines():
        key, _, value = line.partition("=")
        if key:
            shown[key] = value
    return {
        "ok": shown.get("Id") == unit or result["ok"],
        "active": shown.get("ActiveState"),
        "sub": shown.get("SubState"),
        "main_pid": int(shown["MainPID"]) if shown.get("MainPID", "").isdigit() else None,
        "fragment": shown.get("FragmentPath") or None,
        "memory_current": shown.get("MemoryCurrent") or None,
    }


def git_head(root: Path, timeout: float) -> str | None:
    result = run_cmd(["git", "-C", str(root), "rev-parse", "HEAD"], timeout)
    commit = result["stdout"].strip()
    return commit if result["ok"] and COMMIT.fullmatch(commit) else None


def pinned_comfy_commit(repo: Path) -> str | None:
    try:
        text = (repo / "oneshot-setup-spark.sh").read_text(errors="replace")
    except OSError:
        return None
    match = re.search(r"^COMFY_COMMIT=([0-9a-f]{40})\b", text, re.M)
    return match.group(1) if match else None


def repo_info(repo: Path, timeout: float) -> dict:
    return {"root": str(repo), "comfy_pin": pinned_comfy_commit(repo), "studio_head": git_head(repo, timeout)}


def custom_node_names(root: Path) -> list[str]:
    path = root / "custom_nodes"
    if not path.is_dir():
        return []
    names = []
    try:
        entries = sorted(path.iterdir(), key=lambda item: item.name)
    except OSError:
        return []
    for entry in entries:
        if entry.name.startswith(".") or entry.name == "__pycache__":
            continue
        names.append(entry.name)
        if len(names) >= 100:
            break
    return names


def inspect_comfy(base: str, comfy_dir: Path | None, timeout: float) -> dict:
    report: dict = {"url": check_loopback(base), "dir": str(comfy_dir) if comfy_dir else None}
    report["service"] = systemd_show("comfyui.service", timeout)
    system = comfy_get(base, "/system_stats", timeout, 262144)
    report["reachable"] = bool(system.get("ok"))
    report["system"] = system_summary(system["json"]) if system.get("ok") else {
        "ok": False, "error": system.get("error"), "status": system.get("status")}
    queue = comfy_get(base, "/queue", timeout, 65536)
    if queue.get("ok"):
        report["queue"] = queue_counts(queue["json"])
    else:
        report["queue"] = {"running": None, "pending": None, "error": queue.get("error")}
    report["nodes"] = {}
    for name in NODE_NAMES:
        report["nodes"][name] = node_presence(comfy_get(base, "/object_info/" + name, timeout, 524288))
    if comfy_dir and comfy_dir.is_dir():
        report["git_head"] = git_head(comfy_dir, timeout)
        report["custom_nodes"] = custom_node_names(comfy_dir)
    return report


def list_models(root: Path, limit: int, deadline: float) -> dict:
    largest: list[tuple[int, str]] = []
    count = 0
    total = 0
    truncated = False
    errors: list[str] = []

    def onerror(exc: OSError) -> None:
        if len(errors) < 5:
            errors.append(type(exc).__name__)

    for dirpath, dirnames, filenames in os.walk(root, followlinks=False, onerror=onerror):
        dirnames[:] = [name for name in dirnames if name not in SKIP_DIRS and not name.startswith(".")]
        for name in filenames:
            if time.monotonic() > deadline:
                truncated = True
                break
            if Path(name).suffix.lower() not in MODEL_SUFFIXES:
                continue
            path = Path(dirpath) / name
            try:
                size = path.stat().st_size
            except OSError:
                continue
            count += 1
            total += size
            relative = str(path.relative_to(root))
            heapq.heappush(largest, (size, relative))
            if len(largest) > limit:
                heapq.heappop(largest)
        if truncated:
            break
    files = [{"path": path, "bytes": size} for size, path in sorted(largest, reverse=True)]
    return {
        "present": True,
        "root": str(root),
        "count": count,
        "total_bytes": total,
        "largest": files,
        "time_truncated": truncated,
        "errors": errors,
        "checksum_policy": "not_computed except --select-model",
    }


def resolve_model(models_dir: Path, selection: str) -> Path:
    root = models_dir.resolve()
    candidate = Path(selection)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ProbeError("selected model is not available") from exc
    if not resolved.is_relative_to(root):
        raise ProbeError("selected model is outside the models directory")
    if not stat.S_ISREG(resolved.stat().st_mode):
        raise ProbeError("selected model is not a regular file")
    return resolved


def hash_file(path: Path, timeout: float) -> dict:
    size = path.stat().st_size
    if size > MAX_HASH_BYTES:
        return {"algorithm": "sha256", "ok": False, "error": "file_too_large", "bytes": size}
    digest = hashlib.sha256()
    deadline = time.monotonic() + timeout
    read = 0
    try:
        with path.open("rb") as handle:
            while True:
                if time.monotonic() > deadline:
                    return {"algorithm": "sha256", "ok": False, "error": "timeout", "bytes_read": read}
                block = handle.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
                read += len(block)
    except OSError:
        return {"algorithm": "sha256", "ok": False, "error": "unreadable", "bytes_read": read}
    return {"algorithm": "sha256", "ok": True, "hex": digest.hexdigest(), "bytes_read": read}


def candidate_rows(root: Path) -> list[dict]:
    rows = []
    for relative in CANDIDATES:
        path = root / relative
        row = {"path": relative, "present": path.is_file(), "checksum": "not_computed"}
        if row["present"]:
            try:
                row["bytes"] = path.stat().st_size
            except OSError:
                row["bytes"] = None
        rows.append(row)
    return rows


def inspect_models(root: Path | None, selection: str | None, hash_timeout: float, list_timeout: float) -> dict:
    if root is None or not root.is_dir():
        return {"present": False, "root": None if root is None else str(root)}
    report = list_models(root, 40, time.monotonic() + list_timeout)
    report["candidates"] = candidate_rows(root)
    if selection:
        try:
            resolved = resolve_model(root, selection)
        except ProbeError as exc:
            report["selected"] = {"path": selection, "checksum": {"ok": False, "error": str(exc)}}
        else:
            print(f"doctor: hashing {resolved.relative_to(root)} (read-only)", file=sys.stderr)
            report["selected"] = {
                "path": str(resolved.relative_to(root)),
                "bytes": resolved.stat().st_size,
                "checksum": hash_file(resolved, hash_timeout),
            }
    return report


def choose_image(names: set[str]) -> tuple[str, bool]:
    for name in LOCAL_IMAGE_PREFERENCE:
        if name in names:
            return name, False
    bases = sorted(
        name for name in names
        if CUDA_IMAGE.fullmatch(name) and "base" in name and "devel" not in name
    )
    if bases:
        return bases[0], False
    return FALLBACK_PULL, True


def gpu_passthrough(device: str, timeout: float, pull_timeout: float, run_timeout: float) -> dict:
    if not GPU_UUID.fullmatch(device):
        return {"attempted": False, "ok": False, "error": "unexpected gpu uuid"}
    images = run_cmd(["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"], timeout)
    if not images["ok"]:
        return {"attempted": True, "ok": False, "error": "docker_images_failed", "detail": images["stderr"][-300:]}
    names = {line.strip() for line in images["stdout"].splitlines() if line.strip() and "<none>" not in line}
    image, need_pull = choose_image(names)
    pulled = False
    if need_pull:
        print(f"doctor: pulling {image} for the GPU visibility probe", file=sys.stderr)
        pull = run_cmd(["docker", "pull", image], pull_timeout)
        pulled = bool(pull["ok"])
        if not pull["ok"]:
            return {"attempted": True, "ok": False, "image": image, "pulled": False,
                    "error": "pull_failed", "detail": pull["stderr"][-300:]}
    argv = [
        "docker", "run", "--rm", "--network", "none", "--read-only",
        "--tmpfs", "/tmp:rw,nosuid,size=64m", "--memory", "512m", "--pids-limit", "128",
        "--security-opt", "no-new-privileges", "--gpus", f"device={device}",
        "--entrypoint", "nvidia-smi", image,
        "--query-gpu=uuid,name,memory.total", "--format=csv,noheader,nounits",
    ]
    result = run_cmd(argv, run_timeout)
    return {
        "attempted": True,
        "ok": result["ok"],
        "image": image,
        "pulled": pulled,
        "device": device,
        "argv": argv,
        "stdout": result["stdout"].strip(),
        "error": None if result["ok"] else (result["error"] or "nonzero"),
        "detail": None if result["ok"] else result["stderr"][-500:],
    }


def default_comfy_dir() -> Path | None:
    env = os.environ.get("COMFYUI_DIR", "").strip()
    if env:
        path = Path(env)
        return path if path.is_dir() else None
    candidate = Path.home() / "ComfyUI"
    return candidate if candidate.is_dir() else None


def protected_paths(comfy_dir: Path | None, fragment: str | None) -> list[Path]:
    if comfy_dir is None:
        paths = []
    else:
        paths = [comfy_dir / name for name in ("main.py", "extra_model_paths.yaml", "custom_nodes", "models")]
    if fragment:
        paths.append(Path(fragment))
    return paths


def fingerprints(paths: list[Path]) -> dict:
    found = {}
    for path in paths:
        key = str(path)
        try:
            info = os.lstat(path)
        except OSError:
            found[key] = None
            continue
        found[key] = {"mtime_ns": info.st_mtime_ns, "size": info.st_size, "mode": info.st_mode}
    return found


def requirements(report: dict) -> dict:
    """What an app-only container still needs. Local rendering stays off."""

    tools = report.get("tools") or {}
    blockers = []
    notes = []
    if not (tools.get("python3") or {}).get("present"):
        blockers.append("python3 is missing")
    if not (tools.get("docker") or {}).get("present"):
        blockers.append("docker is missing from PATH")
    elif not report.get("docker", {}).get("skipped") and not (report.get("docker") or {}).get("ok"):
        blockers.append("the Docker daemon did not answer")
    ports = {row["port"]: row for row in report.get("ports") or []}
    studio = ports.get(7870)
    if studio and studio.get("listening"):
        blockers.append("proposed Studio port 7870 is already in use")
    elif studio:
        notes.append("port 7870 is free")
    if tools and not (tools.get("ffmpeg") or {}).get("present"):
        notes.append("host ffmpeg is absent; the app image is expected to provide it")
    if tools and not (tools.get("ffprobe") or {}).get("present"):
        notes.append("host ffprobe is absent; the app image is expected to provide it")
    render = ["local rendering is not enabled by this diagnostic"]
    resources_report = report.get("resources") or {}
    if resources_report.get("available_below_floor"):
        render.append(
            f"MemAvailable {resources_report.get('mem_available_mb')} MB is below the "
            f"{resources_report.get('floor_mb')} MB floor"
        )
    comfy = report.get("comfy") or {}
    if comfy.get("reachable"):
        render.append("shared ComfyUI answered on loopback; do not submit jobs to it or change its service")
    pin = (report.get("repo") or {}).get("comfy_pin")
    observed = comfy.get("git_head")
    if pin and observed and pin != observed:
        render.append("shared ComfyUI HEAD differs from the repository pin " + pin[:12])
    renderer = ports.get(8191)
    if renderer and renderer.get("listening"):
        render.append("proposed renderer port 8191 is already in use")
    gpu = report.get("gpu") or {}
    if not gpu.get("skipped") and not gpu.get("gpus"):
        render.append("nvidia-smi reported no GPU")
    container = report.get("gpu_container") or {}
    if container.get("attempted") and not container.get("ok"):
        render.append("GPU passthrough probe failed: " + str(container.get("error") or "unknown"))
    if not container.get("attempted"):
        notes.append("GPU container probe was not requested")
    for row in report.get("disks_low") or []:
        notes.append(row)
    return {
        "app_only_blockers": blockers,
        "app_only_notes": notes,
        "local_render_blockers": render,
        "local_render_enabled": False,
    }


def render_text(report: dict) -> str:
    lines = [
        "Studio doctor (read-only)",
        "Shared ComfyUI is only read: system stats, queue counts, and selected node presence.",
        "",
    ]
    req = report["requirements"]
    lines.append("App-only blockers: " + (", ".join(req["app_only_blockers"]) or "none"))
    for note in req["app_only_notes"]:
        lines.append(f"note: {note}")
    lines.append("Local render: disabled")
    for item in req["local_render_blockers"]:
        lines.append(f"- {item}")
    host = report["host"]
    cpu = report["cpu"]
    lines.append("")
    lines.append(f"Host: {host.get('pretty') or host.get('system')} {host.get('machine')} kernel {host.get('kernel')}")
    lines.append(f"CPU: {cpu.get('model') or 'unknown'} ({cpu.get('logical_cpus')} logical)")
    resources_report = report["resources"]
    lines.append(
        f"Memory: {resources_report.get('mem_available_mb')} MB available of "
        f"{resources_report.get('mem_total_mb')} MB; floor {resources_report.get('floor_mb')} MB "
        f"({resources_report.get('floor_source')})"
    )
    for disk in resources_report.get("disks") or []:
        if disk.get("present"):
            free = disk["free_bytes"] / (1024 ** 3)
            lines.append(f"Disk {disk['label']}: {free:.1f} GiB free at {disk['path']}")
    gpu = report.get("gpu") or {}
    for card in gpu.get("gpus") or []:
        lines.append(
            f"GPU: {card.get('name')} {card.get('uuid')} "
            f"{card.get('memory_free_mib')}/{card.get('memory_total_mib')} MiB free, driver {card.get('driver_version')}"
        )
    apps = gpu.get("compute_apps") or []
    if apps:
        lines.append("GPU compute apps: " + ", ".join(f"{item['name']}({item['pid']})" for item in apps))
    docker_report = report.get("docker") or {}
    if not docker_report.get("skipped"):
        lines.append(
            f"Docker: server {docker_report.get('server_version')} compose {docker_report.get('compose_version')} "
            f"nvidia runtime {'yes' if docker_report.get('nvidia_runtime') else 'no'}"
        )
    for match in (report.get("published_ports") or {}).get("matches") or []:
        lines.append(
            f"Published: {match['name']} ({match['image']}) ports {match['ports']} "
            f"watched {match['watched_ports']}"
        )
    lines.append("Ports:")
    for row in report.get("ports") or []:
        owner = ", ".join(f"{item['comm']}:{item['pid']}" for item in row.get("owners") or [])
        if not owner:
            uids = ", ".join(str(uid) for uid in row.get("uids") or [])
            owner = f"uid {uids}, process not readable" if uids else "no readable owner"
        state = "listening" if row.get("listening") else "free"
        lines.append(f"  {row['port']} {state} {', '.join(row.get('addresses') or [])} {owner} — {row['purpose']}")
    comfy = report.get("comfy") or {}
    if not comfy.get("skipped"):
        system = (comfy.get("system") or {}).get("system") or {}
        lines.append(
            f"ComfyUI: reachable={comfy.get('reachable')} version={system.get('comfyui_version')} "
            f"HEAD={comfy.get('git_head')} pin={(report.get('repo') or {}).get('comfy_pin')}"
        )
        queue = comfy.get("queue") or {}
        lines.append(f"ComfyUI queue: running={queue.get('running')} pending={queue.get('pending')}")
        present = [name for name, item in (comfy.get("nodes") or {}).items() if item.get("present")]
        lines.append("ComfyUI nodes present: " + (", ".join(present) or "none confirmed"))
    models = report.get("models") or {}
    if models.get("present"):
        lines.append(f"Models: {models.get('count')} files under {models.get('root')}")
        for candidate in models.get("candidates") or []:
            lines.append(f"  candidate {candidate['path']}: present={candidate['present']} bytes={candidate.get('bytes')}")
        selected = models.get("selected")
        if selected:
            checksum = selected.get("checksum") or {}
            lines.append(f"  selected {selected['path']}: sha256={checksum.get('hex') or checksum.get('error')}")
    container = report.get("gpu_container") or {}
    if container.get("attempted"):
        lines.append(
            f"GPU container: ok={container.get('ok')} image={container.get('image')} "
            f"pulled={container.get('pulled')} error={container.get('error')}"
        )
        if container.get("stdout"):
            lines.append("  nvidia-smi: " + container["stdout"].replace("\n", " | "))
    protected = report.get("protected_files") or {}
    if "unchanged" in protected:
        lines.append("Protected ComfyUI paths unchanged during this run: " + ("yes" if protected["unchanged"] else "no"))
    lines.append("")
    return "\n".join(lines)


def collect(options: Options) -> dict:
    disks = [("repo", options.repo)]
    if options.models_dir is not None:
        disks.append(("models", options.models_dir))
    report = {
        "doctor": DOCTOR_VERSION,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "safety": SAFETY,
        "host": host_info(),
        "cpu": cpu_info(),
        "resources": resources(disks),
        "cgroup": cgroup_info(),
        "tools": tool_presence(),
        "ports": summarize_listeners(read_listeners(options.timeout)),
        "repo": repo_info(options.repo, options.timeout),
        "gpu": {"skipped": True} if options.skip_gpu else read_gpu(options.timeout),
        "docker": {"skipped": True} if options.skip_docker else read_docker(options.timeout),
        "published_ports": {"skipped": True} if options.skip_docker else read_published(options.timeout),
    }
    before_paths = [] if options.skip_comfy else protected_paths(options.comfy_dir, None)
    before = fingerprints(before_paths)
    comfy: dict = {"skipped": True}
    if not options.skip_comfy:
        comfy = inspect_comfy(options.comfy_url, options.comfy_dir, options.timeout)
        fragment = (comfy.get("service") or {}).get("fragment")
        if fragment:
            unit = Path(fragment)
            if unit not in before_paths:
                before_paths.append(unit)
                before.update(fingerprints([unit]))
    report["comfy"] = comfy
    if options.skip_models:
        report["models"] = {"skipped": True}
    else:
        report["models"] = inspect_models(
            options.models_dir, options.select_model, options.hash_timeout, max(options.timeout, 20))
    if options.gpu_container:
        chosen = None if options.skip_gpu else select_gpu(report["gpu"].get("gpus") or [])
        device = (chosen or {}).get("uuid")
        report["gpu_container"] = (
            gpu_passthrough(device, options.timeout, options.pull_timeout, options.container_timeout)
            if device else {"attempted": False, "ok": False, "error": "no gpu uuid"}
        )
        if not options.skip_comfy:
            again = comfy_get(options.comfy_url, "/system_stats", options.timeout, 262144)
            report["comfy_after_gpu_probe"] = (
                {"reachable": True, "version": system_summary(again["json"])["system"].get("comfyui_version")}
                if again.get("ok") else {"reachable": False, "error": again.get("error")}
            )
    else:
        report["gpu_container"] = {"attempted": False}
    after = fingerprints(before_paths)
    report["protected_files"] = {"skipped": True} if not before_paths else {
        "paths": [str(path) for path in before_paths],
        "unchanged": before == after,
    }
    report["requirements"] = requirements(report)
    return report


def _positive(name: str, value: float, limit: float) -> float:
    if not 0 < value <= limit:
        raise ProbeError(f"{name} must be between 0 and {limit} seconds")
    return value


def parse_args(argv: list[str] | None, repo: Path | None = None) -> Options:
    parser = argparse.ArgumentParser(description="Read-only Studio and shared-ComfyUI baseline")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    parser.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    parser.add_argument("--comfy-dir", default=None)
    parser.add_argument("--models-dir", default=None)
    parser.add_argument("--select-model", default=None, help="hash this one file inside the models directory")
    parser.add_argument("--timeout", type=float, default=15)
    parser.add_argument("--hash-timeout", type=float, default=180)
    parser.add_argument("--pull-timeout", type=float, default=180)
    parser.add_argument("--container-timeout", type=float, default=60)
    parser.add_argument("--gpu-container", action="store_true", help="run nvidia-smi in a disposable container")
    parser.add_argument("--skip-comfy", action="store_true")
    parser.add_argument("--skip-docker", action="store_true")
    parser.add_argument("--skip-gpu", action="store_true")
    parser.add_argument("--skip-models", action="store_true")
    args = parser.parse_args(argv)
    if args.skip_gpu and args.gpu_container:
        raise ProbeError("--gpu-container needs the GPU query")
    if args.skip_models and args.select_model:
        raise ProbeError("--select-model cannot be combined with --skip-models")
    if args.comfy_dir:
        comfy_dir = Path(args.comfy_dir)
    elif args.skip_comfy and args.skip_models and not args.select_model:
        comfy_dir = None
    else:
        comfy_dir = default_comfy_dir()
    if args.models_dir:
        models_dir = Path(args.models_dir)
    elif comfy_dir is not None and not args.skip_models:
        models_dir = comfy_dir / "models"
    else:
        models_dir = None
    return Options(
        json_output=args.json,
        comfy_url=check_loopback(args.comfy_url) if not args.skip_comfy else args.comfy_url,
        comfy_dir=comfy_dir,
        models_dir=models_dir,
        select_model=args.select_model,
        timeout=_positive("timeout", args.timeout, 120),
        hash_timeout=_positive("hash-timeout", args.hash_timeout, 900),
        pull_timeout=_positive("pull-timeout", args.pull_timeout, 900),
        container_timeout=_positive("container-timeout", args.container_timeout, 300),
        gpu_container=args.gpu_container,
        skip_comfy=args.skip_comfy,
        skip_docker=args.skip_docker,
        skip_gpu=args.skip_gpu,
        skip_models=args.skip_models,
        repo=repo or Path(__file__).resolve().parent.parent,
    )


def main(argv: list[str] | None = None) -> int:
    try:
        options = parse_args(argv)
        report = collect(options)
    except ProbeError as exc:
        print(f"doctor: {exc}", file=sys.stderr)
        return 2
    if options.json_output:
        json.dump(report, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        sys.stdout.write(render_text(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
