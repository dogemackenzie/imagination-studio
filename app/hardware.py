"""What this machine is, as plain observations. Standard library only.

A DGX Spark is a positive check: the GPU name or the DMI product text contains ``GB10`` or
``DGX Spark``. Linux, aarch64, or an NVIDIA card by itself is not a Spark. A Jetson is NVIDIA
and can be aarch64, and an RTX card is a different family. Nothing here imports the web server
or a model framework, so a script can ask before it renders.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_cache: dict | None = None


def family_of(gpu_name: str, product: str = "") -> str:
    """``nvidia-spark``, ``nvidia-rtx``, ``apple``, ``none`` or ``other``.

    Spark is only a GB10 or a DGX Spark named by the GPU or the product text. An RTX name is
    ``nvidia-rtx``. A Jetson, or any other named GPU, is ``other``. No name is ``none``.
    """

    blob = f"{gpu_name or ''} {product or ''}".casefold()
    if "gb10" in blob or "dgx spark" in blob:
        return "nvidia-spark"
    name = (gpu_name or "").casefold()
    if "rtx" in name or "geforce" in name or "quadro" in name:
        return "nvidia-rtx"
    if "apple" in name:
        return "apple"
    if not name.strip():
        return "none"
    return "other"


def _text(path: str) -> str:
    try:
        return Path(path).read_text(errors="replace").strip()
    except OSError:
        return ""


def _memory_mb() -> dict:
    total = available = None
    try:
        lines = Path("/proc/meminfo").read_text(errors="replace").splitlines()
    except OSError:
        return {"total": None, "available": None}
    for line in lines:
        parts = line.split()
        if len(parts) < 2:
            continue
        if parts[0] == "MemTotal:":
            total = int(parts[1]) // 1024
        elif parts[0] == "MemAvailable:":
            available = int(parts[1]) // 1024
    return {"total": total, "available": available}


def _cpu() -> str:
    for line in _text("/proc/cpuinfo").splitlines():
        if line.lower().startswith("model name"):
            return line.split(":", 1)[1].strip()
    return ""


def _gpu_line() -> dict:
    """One nvidia-smi row, parsed from the right so a name may contain commas. Empty when it cannot be asked."""

    try:
        done = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,uuid,memory.total,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {"name": "", "uuid": "", "memory_mib": None, "driver": ""}
    line = (done.stdout or "").strip().splitlines()
    if done.returncode != 0 or not line:
        return {"name": "", "uuid": "", "memory_mib": None, "driver": ""}
    parts = [part.strip() for part in line[0].split(",")]
    if len(parts) < 4:
        return {"name": "", "uuid": "", "memory_mib": None, "driver": ""}
    driver, memory, uuid = parts[-1], parts[-2], parts[-3]
    name = ", ".join(parts[:-3]).strip()
    try:
        memory_mib = int(float(memory))
    except ValueError:
        memory_mib = None
    return {"name": name, "uuid": uuid, "memory_mib": memory_mib, "driver": driver}


def observe() -> dict:
    """Cached observations. ``/api/state`` and the scripts share the one nvidia-smi call."""

    global _cache
    if _cache is not None:
        return _cache
    gpu = _gpu_line()
    product = _text("/sys/class/dmi/id/product_name")
    family = family_of(gpu["name"], product)
    if family == "none" and sys.platform == "darwin":
        family = "apple"
    system = "linux" if sys.platform.startswith("linux") else "darwin" if sys.platform == "darwin" else sys.platform
    try:
        arch = __import__("platform").machine()
    except Exception:  # noqa: BLE001 - the arch is a label, not a gate by itself
        arch = ""
    _cache = {
        "os": system,
        "arch": arch,
        "cpu": _cpu(),
        "product": product,
        "gpu": {**gpu, "family": family},
        "memory_mb": _memory_mb(),
    }
    return _cache


def clear() -> None:
    """Drop the cache. The next ``observe`` asks the machine again."""

    global _cache
    _cache = None


def use_observation(obs: dict | None) -> None:
    """Pin a fake observation for a test. ``None`` clears it. This does not change ``server.SPARK``."""

    global _cache
    _cache = obs


def is_spark() -> bool:
    """True only when the GPU or the product is a DGX Spark. An RTX card is not."""

    return observe()["gpu"]["family"] == "nvidia-spark"
