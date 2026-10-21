"""Which routes this process may run. Standard library plus hardware.py. No web server.

A mode names a machine. It cannot invent a measured route: ``managed-rtx`` and ``app-only``
never render, and ``STUDIO_MODE=spark`` is ignored unless this process is actually a Spark
(the flag passed in, or ``hardware.is_spark()``). ``controller`` may queue one video clip for a
worker and does not render it here. ``use`` is how a test simulates a whole machine. Export and
soundtrack stay available everywhere: they assemble files that already exist.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import hardware

# The same 8000 MB floor as spark/memory_guard.py floor_mb and scripts/doctor.py HOST_FLOOR_MB.
# Duplicated on purpose: this module must not import the guard or the doctor.
FLOOR_MB = 8000

MODES = ("app-only", "managed-rtx", "mac", "spark", "controller")
# Picture, video, edit, library actions, sound, voice and music. Aliases map onto these.
GENERATIVE = (
    "scout", "video", "studio", "long", "edit",
    "turnaround", "upscale", "mask", "remove",
    "sfx", "foley", "voice", "music",
)
PRESERVED = ("export", "soundtrack")
ALIASES = {"image": "scout", "picture": "scout", "clip": "video", "song": "music"}

REASON_TEXT = {
    "unavailable_hardware": "No GPU is available for a local render on this machine.",
    "missing_runtime": "The runtime this route needs is not installed.",
    "missing_model": "The model this route needs is not installed.",
    "incompatible_nodes": "The installed nodes cannot run this route.",
    "busy_renderer": "The renderer is busy with another job.",
    "insufficient_memory": "Available memory is under the 8000 MB floor, so a render cannot start.",
    "unmeasured_configuration": (
        "This machine has no measured render route. An RTX card is not a production render mode, "
        "and a managed RTX image does not enable one."
    ),
}

_forced: str | None = None


def use(mode: str | None) -> None:
    """Force a mode for this process. ``None`` follows the machine and ``STUDIO_MODE`` again."""

    global _forced
    if mode is not None and mode not in MODES:
        raise ValueError(mode)
    _forced = mode


def _env_ok(asked: str, spark: bool | None) -> bool:
    if asked == "spark":
        return bool(spark) or hardware.is_spark()
    if asked == "mac":
        return sys.platform == "darwin"
    return asked in ("app-only", "managed-rtx", "controller")


def _physical(spark: bool | None) -> str:
    # A true flag is the test seam for Spark behaviour, even when the GPU is not a Spark.
    if spark:
        return "spark"
    if spark is None and hardware.is_spark():
        return "spark"
    if sys.platform == "darwin":
        return "mac"
    return "app-only"


def mode(spark: bool | None = None) -> str:
    """The mode in force. ``spark`` is the caller's SPARK flag (True, False, or unknown)."""

    if _forced:
        return _forced
    asked = os.environ.get("STUDIO_MODE", "").strip().lower()
    if asked in MODES and _env_ok(asked, spark):
        return asked
    return _physical(spark)


def renders(spark: bool | None = None) -> bool:
    """True only for a Mac or a Spark. ``managed-rtx`` stays false even when ``spark`` is true."""

    return mode(spark) in ("mac", "spark")


def _item(code: str) -> dict:
    return {"code": code, "reason": REASON_TEXT[code]}


def reasons(spark: bool | None = None) -> list[dict]:
    """Why a generative route is closed. Empty when this mode renders.

    A named GPU leads with ``unmeasured_configuration``. No GPU leads with ``unavailable_hardware``.
    ``insufficient_memory`` follows when available memory is under ``FLOOR_MB``. Free memory later
    does not open a route that was never measured.
    """

    if renders(spark):
        return []
    obs = hardware.observe()
    gpu = (obs.get("gpu") or {})
    name = gpu.get("name") or ""
    family = gpu.get("family") or hardware.family_of(name, obs.get("product") or "")
    out = []
    if not name.strip() or family == "none":
        out.append(_item("unavailable_hardware"))
    out.append(_item("unmeasured_configuration"))
    available = (obs.get("memory_mb") or {}).get("available")
    if available is not None and available < FLOOR_MB:
        out.append(_item("insufficient_memory"))
    return out


def refusal(kind: str, spark: bool | None = None) -> dict | None:
    """A machine-readable refusal, or None when this kind may run.

    Unknown kinds are None so the caller can keep its own "unknown kind" error. Export and
    soundtrack are None on every machine.
    """

    kind = ALIASES.get(kind, kind)
    if kind in PRESERVED or kind not in GENERATIVE:
        return None
    # A controller queues one video clip. The worker admits it. Nothing else opens.
    if kind == "video" and mode(spark) == "controller":
        return None
    items = reasons(spark)
    if not items:
        return None
    return {"error": items[0]["reason"], "code": items[0]["code"], "reasons": items}


def install_requirements() -> str:
    """What a weight install needs. Not an image-only fit check, and not a promise that RTX renders."""

    return (
        "Picture and video weights install only on a measured machine. "
        "A Mac uses bash oneshot-setup.sh (about 175 GB of weights and 64 GB of memory). "
        "A DGX Spark uses bash oneshot-setup-spark.sh (about 99 GB). "
        "Qwen-Image-2.1 is under the Qwen Research License (non-commercial). "
        "MiniMax H3 is under the MiniMax H3 Community License, with territory limits. "
        "The weights are not redistributed. "
        "A managed RTX image does not create a render route."
    )


def public(spark: bool | None = None) -> dict:
    """The capabilities block of ``/api/state``."""

    obs = hardware.observe()
    routes = {}
    for kind in GENERATIVE:
        detail = refusal(kind, spark)
        if detail is None:
            routes[kind] = {"enabled": True}
        else:
            routes[kind] = {"enabled": False, "code": detail["code"], "reason": detail["error"]}
    for kind in PRESERVED:
        routes[kind] = {"enabled": True}
    memory = dict(obs.get("memory_mb") or {})
    memory["floor"] = FLOOR_MB
    return {
        "mode": mode(spark),
        "local_render": renders(spark),
        "os": obs.get("os"),
        "arch": obs.get("arch"),
        "gpu": obs.get("gpu") or {},
        "memory_mb": memory,
        "routes": routes,
        "reasons": reasons(spark),
    }


def script_exit(kind: str) -> int:
    """0 when this machine may run ``kind``. Otherwise the lead reason on stderr and 1.

    A real Spark (``hardware.is_spark``) and a Mac pass. An app-only Linux machine stops
    before a script hands the job to spark/.
    """

    if kind == "video" and mode(hardware.is_spark()) == "controller":
        sys.stderr.write("This machine does not render H3. A clip is queued for a worker from the app.\n")
        return 1
    if kind in ("sdxl-turbo", "z-image"):
        spark = Path(__file__).resolve().parent.parent / "spark"
        if str(spark) not in sys.path:
            sys.path.insert(0, str(spark))
        if kind == "sdxl-turbo":
            import sdxl_turbo
            reason = sdxl_turbo.script_block()
        else:
            import z_image_turbo
            reason = z_image_turbo.script_block()
        if reason is None:
            return 0
        sys.stderr.write(reason + "\n")
        return 1
    detail = refusal(kind, hardware.is_spark())
    if detail is None:
        return 0
    sys.stderr.write(detail["error"] + "\n")
    return 1
