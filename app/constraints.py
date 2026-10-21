"""What this process can see, and the limits it keeps.

A GPU reading is not permission to render. ``capabilities.local_render`` stays false on this
Linux app unless the mode is a Mac or a DGX Spark. The only picture job this process sends
from an unmeasured machine is SDXL Turbo, and only when ``image_models()`` already lists it.

The shared ComfyUI is read with GET, and only at the two allowed origins (loopback and
docker0, port 8188). Nothing here posts a prompt, and nothing starts, stops, interrupts,
unloads, or clears that server. A filename on that server is not a licence and not a route.
FLUX.2 Klein is not FLUX3-K. A Qwen image VAE is not Qwen-Image.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request

import capabilities
import hardware

ORIGINS = ("http://127.0.0.1:8188", "http://172.17.0.1:8188")
MODEL_FOLDERS = ("checkpoints", "diffusion_models", "vae")
TIMEOUT = 2.0
MAX_BYTES = 1_000_000
STATS_FRESH = 5.0
FILES_FRESH = 600.0
PORT = 8188

NOTE = (
    "The studio uses what this machine can run. "
    "A GPU reading does not open a route that was never measured."
)
CONTAINER_NOTE = " This app process does not hold the GPU. A measured still is sent to the shared ComfyUI, which does."

_lock = threading.Lock()
_stats: dict = {"at": 0.0, "value": None, "asked": False}
_files: dict = {"at": 0.0, "value": None, "asked": False}


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


def clear() -> None:
    """Drop the cached readings. The next ask talks to the machine again."""

    with _lock:
        _stats.update(at=0.0, value=None, asked=False)
        _files.update(at=0.0, value=None, asked=False)


def reading_origin() -> str | None:
    """The shared ComfyUI, when this process was told its address. Any other address is ignored."""

    raw = os.environ.get("SDXL_COMFY_URL", "").strip().rstrip("/")
    if raw in ORIGINS:
        return raw
    return None


def allowed_path(path: str) -> bool:
    if path == "/system_stats":
        return True
    prefix = "/models/"
    if not path.startswith(prefix):
        return False
    folder = path[len(prefix):]
    return folder in MODEL_FOLDERS and "/" not in folder


def fetch_json(origin: str, path: str, timeout: float):
    """GET one allowed path. A redirect, a huge body, or a bad path returns None and posts nothing."""

    if origin not in ORIGINS or not allowed_path(path) or timeout is None or timeout <= 0:
        return None
    request = urllib.request.Request(origin + path, method="GET")
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        response = opener.open(request, timeout=timeout)
    except (OSError, urllib.error.URLError, TimeoutError):
        return None
    try:
        chunks = []
        total = 0
        while True:
            block = response.read(65536)
            if not block:
                break
            total += len(block)
            if total > MAX_BYTES:
                return None
            chunks.append(block)
    finally:
        response.close()
    try:
        return json.loads(b"".join(chunks).decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        return None


def device_name(raw: str) -> str:
    """ComfyUI writes ``cuda:0 NVIDIA GeForce RTX 3060 : cudaMallocAsync``. The card's own name is the middle."""

    text = (raw or "").strip()
    if " : " in text:
        text = text.rsplit(" : ", 1)[0].strip()
    head, sep, rest = text.partition(" ")
    if head.startswith("cuda:") and sep:
        text = rest.strip()
    return text[:80]


def _mib(value) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        return None
    return int(value) // (1024 * 1024)


def parse_device(stats: dict | None) -> dict | None:
    """Name and VRAM from ``/system_stats``. Load, temperature, and power are not in that answer."""

    if not isinstance(stats, dict):
        return None
    devices = [item for item in stats.get("devices") or [] if isinstance(item, dict)]
    device = next((item for item in devices if item.get("type") == "cuda" and device_name(str(item.get("name") or ""))), None)
    if device is None:
        return None
    return {
        "name": device_name(str(device.get("name") or "")),
        "util_pct": None,
        "temp_c": None,
        "power_w": None,
        "clock_mhz": None,
        "vram_total_mb": _mib(device.get("vram_total")),
        "vram_free_mb": _mib(device.get("vram_free")),
        "source": "comfyui",
    }


def system_stats(force: bool = False) -> dict | None:
    origin = reading_origin()
    if not origin:
        return None
    now = time.monotonic()
    with _lock:
        if not force and _stats["asked"] and now - _stats["at"] < STATS_FRESH:
            return _stats["value"]
    payload = fetch_json(origin, "/system_stats", timeout=TIMEOUT)
    value = payload if isinstance(payload, dict) else None
    with _lock:
        _stats.update(at=time.monotonic(), value=value, asked=True)
    return value


def gpu_reading() -> dict | None:
    """The shared server's card, when this process has no nvidia-smi reading of its own."""

    if reading_origin() is None:
        return None
    return parse_device(system_stats())


def smi_gpu() -> dict | None:
    """nvidia-smi's own line, when this process can run it. A container without it gets None."""

    import telemetry  # the CSV parser lives with the status bar; imported here to avoid a cycle

    if not shutil.which("nvidia-smi"):
        return None
    try:
        done = subprocess.run(
            ["nvidia-smi", f"--query-gpu={telemetry.GPU_FIELDS}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    lines = (done.stdout or "").strip().splitlines()
    if done.returncode != 0 or not lines:
        return None
    return telemetry.parse_gpu(lines[0])


def visible_gpu() -> dict | None:
    """The richest reading this process can ask for. The shared server fills in when nvidia-smi is absent."""

    found = smi_gpu()
    if found:
        return found
    return gpu_reading()


def shared_renderer() -> dict | None:
    """The shared ComfyUI, for the Machine room. None when this process was not given its address."""

    if reading_origin() is None:
        return None
    gpu = parse_device(system_stats())
    if gpu is None:
        return {"up": False, "port": PORT}
    return {"up": True, "port": PORT, "name": gpu["name"],
            "vram_total_mb": gpu["vram_total_mb"], "vram_free_mb": gpu["vram_free_mb"]}


def model_files(force: bool = False) -> dict | None:
    """Filenames the shared server lists. None when it was not asked, or it did not answer."""

    origin = reading_origin()
    if not origin:
        return None
    now = time.monotonic()
    with _lock:
        fresh = FILES_FRESH if _files["value"] is not None else STATS_FRESH
        if not force and _files["asked"] and now - _files["at"] < fresh:
            return _files["value"]
    found: dict[str, list[str]] = {}
    for folder in MODEL_FOLDERS:
        payload = fetch_json(origin, "/models/" + folder, timeout=TIMEOUT)
        if not isinstance(payload, list):
            with _lock:
                _files.update(at=time.monotonic(), value=None, asked=True)
            return None
        found[folder] = [item for item in payload if isinstance(item, str)][:400]
    with _lock:
        _files.update(at=time.monotonic(), value=found, asked=True)
    return found


def match_file(folder: str, name: str) -> tuple[str, str, str] | None:
    """(id, label, route) for a filename. ``route`` is empty when the studio does not send that file a job."""

    low = (name or "").casefold()
    if folder == "checkpoints":
        if low == "sd_xl_turbo_1.0_fp16.safetensors":
            return ("sdxl-turbo", "SDXL Turbo", "sdxl-turbo")
        if low.startswith("sd_xl_base"):
            return ("sdxl-base", "SDXL base", "")
        if low.startswith("sd_xl_refiner"):
            return ("sdxl-refiner", "SDXL refiner", "")
        if low.startswith("stable-audio"):
            return ("stable-audio", "Stable Audio Open", "")
        if low.startswith("yue"):
            return ("yue2", "YuE2", "")
        if low.startswith("hunyuan_3d") or low.startswith("hunyuan3d"):
            return ("hunyuan3d", "Hunyuan3D", "")
        if low.startswith("sdpose"):
            return ("sdpose", "SDPose", "")
        if low.startswith("sam"):
            return ("sam", "SAM", "")
    elif folder == "diffusion_models":
        if "flux-2-klein" in low or "flux2-klein" in low or low.startswith("flux-2"):
            return ("flux2-klein", "FLUX.2 Klein", "")
        if "flux3" in low and "klein" in low:
            return ("flux3k", "FLUX3-K", "flux3k")
        if low.startswith("wan"):
            return ("wan", "Wan", "")
        if low.startswith("z_image") or low.startswith("z-image"):
            return ("z-image", "Z-Image", "")
        if low.startswith("seedvr"):
            return ("seedvr2", "SeedVR2", "")
        if low.startswith("acestep") or low.startswith("ace_step"):
            return ("acestep", "ACE-Step", "")
        if low.startswith("anima"):
            return ("anima", "Anima", "")
        if low.startswith("mage_flow") or low.startswith("mage-flow"):
            return ("mage-flow", "Mage Flow", "")
        if low.startswith("pixeldit"):
            return ("pixeldit", "PixelDiT", "")
        if low.startswith("rt_detr") or low.startswith("rt-detr"):
            return ("rt-detr", "RT-DETR", "")
        if "qwen" in low and "image" in low:
            return ("qwen-image", "Qwen-Image", "qwen")
        if "minimax" in low or "fl2v" in low:
            return ("minimax-h3", "MiniMax H3", "h3")
    elif folder == "vae" and "qwen" in low and "vae" in low:
        return ("qwen-vae", "Qwen image VAE", "")
    return None


def classify(files: dict | None) -> tuple[list[dict], dict]:
    """Families in those filename lists, plus a count of names that matched nothing."""

    families: list[dict] = []
    seen: set[str] = set()
    flags = {"qwen_vae": False, "other": 0, "sdxl_file": False, "qwen_file": False, "flux3_file": False, "h3_file": False}
    if not files:
        return families, flags
    for folder, names in files.items():
        if folder not in MODEL_FOLDERS or not isinstance(names, list):
            continue
        for name in names:
            if not isinstance(name, str):
                continue
            found = match_file(folder, name)
            if found is None:
                if folder in ("checkpoints", "diffusion_models"):
                    flags["other"] += 1
                continue
            ident, label, route = found
            if ident == "qwen-vae":
                flags["qwen_vae"] = True
                continue
            if ident == "sdxl-turbo":
                flags["sdxl_file"] = True
            elif route == "qwen":
                flags["qwen_file"] = True
            elif route == "flux3k":
                flags["flux3_file"] = True
            elif route == "h3":
                flags["h3_file"] = True
            if ident in seen:
                continue
            seen.add(ident)
            families.append({"id": ident, "label": label, "route": route, "file": name})
    return families, flags


def _gb(mb: int | None) -> str:
    if mb is None:
        return "unknown"
    return f"{mb / 1024:.1f} GB"


def _row(ident: str, label: str, state: str, detail: str) -> dict:
    return {"id": ident, "label": label, "state": state, "detail": detail}


def report(spark: bool, picture_models: list[str] | None, fresh: bool = False) -> dict:
    """Hardware this process can see, the routes it will run, and files that stay on the server."""

    if fresh:
        clear()
    pictures = set(picture_models or [])
    public = capabilities.public(spark)
    local = bool(public["local_render"])
    files = model_files(force=fresh)
    families, flags = classify(files)
    gpu = visible_gpu()
    memory = hardware._memory_mb()
    available, total = memory.get("available"), memory.get("total")
    hardware_rows = []
    if gpu:
        if gpu.get("source") == "comfyui" and gpu.get("vram_total_mb"):
            detail = (f"{gpu['name']}, {_gb(gpu.get('vram_free_mb'))} free of {_gb(gpu.get('vram_total_mb'))}. "
                      "Read from the shared ComfyUI.")
        else:
            bits = [gpu.get("name") or "GPU"]
            if gpu.get("util_pct") is not None:
                bits.append(f"load {gpu['util_pct']:g}%")
            if gpu.get("temp_c") is not None:
                bits.append(f"{gpu['temp_c']:g} °C")
            if gpu.get("power_w") is not None:
                bits.append(f"{gpu['power_w']:g} W")
            detail = ", ".join(bits) + "."
        hardware_rows.append(_row("gpu", "GPU", "seen", detail))
    else:
        hardware_rows.append(_row("gpu", "GPU", "missing", "This process has no GPU reading."))
    if available is not None and available < capabilities.FLOOR_MB:
        mem_state = "tight"
    elif available is not None:
        mem_state = "ok"
    else:
        mem_state = "missing"
    hardware_rows.append(_row(
        "memory", "Memory", mem_state,
        f"{_gb(available)} free of {_gb(total)}. A measured still needs {capabilities.FLOOR_MB} MB free on the host."))
    cores = os.cpu_count()
    hardware_rows.append(_row("cpu", "Processor", "seen", f"{cores} cores" if cores else "Core count was not reported."))
    if reading_origin():
        shared = shared_renderer()
        if shared and shared.get("up"):
            hardware_rows.append(_row("comfy", "Shared ComfyUI", "seen", f"Answering on port {PORT}."))
        else:
            hardware_rows.append(_row(
                "comfy", "Shared ComfyUI", "down",
                f"Port {PORT} did not answer. Studio does not start that server."))

    import sdxl_turbo  # the measured still's size and licence; no prompt is built here

    if "sdxl-turbo" in pictures:
        sdxl_state = "runs"
        sdxl_detail = (
            f"{sdxl_turbo.WIDTH}×{sdxl_turbo.HEIGHT}, one step, through the shared ComfyUI. "
            f"A still is sent when that queue is empty and the host has {capabilities.FLOOR_MB} MB free. "
            f"{sdxl_turbo.LICENSE}."
        )
        if reading_origin() and not (shared_renderer() or {}).get("up"):
            sdxl_detail += " The shared ComfyUI is not answering right now. Studio does not start it."
    elif flags["sdxl_file"]:
        sdxl_state = "closed"
        sdxl_detail = "The checkpoint is on the shared ComfyUI. The measured profile is not enabled, so a still is not sent."
    else:
        sdxl_state = "missing"
        sdxl_detail = "The measured SDXL Turbo checkpoint is not available to this process."

    # A failed list is not a catalogue of absences. The server may hold a file this process did not see.
    unknown = reading_origin() is not None and files is None

    def picture_row(ident: str, label: str, running: bool, on_server: bool, missing: str, present: str) -> dict:
        if running:
            return _row(ident, label, "runs", "Installed for this machine.")
        if on_server:
            return _row(ident, label, "present", present)
        if unknown:
            return _row(ident, label, "unknown",
                        "The shared ComfyUI did not list its files. Studio does not treat that as this model being absent.")
        return _row(ident, label, "missing", missing)

    qwen_detail_missing = "Qwen-Image-2.1 weights are not on this machine."
    if flags["qwen_vae"] and "qwen" not in pictures and not flags["qwen_file"]:
        qwen_detail_missing += " A Qwen image VAE is on the shared ComfyUI. The picture weights are a separate file."
    software = [
        _row("sdxl-turbo", "SDXL Turbo", sdxl_state, sdxl_detail),
        picture_row("qwen", "Qwen-Image-2.1", bool(pictures & {"qwen", "qwen-full"}), flags["qwen_file"],
                    qwen_detail_missing, "Qwen-Image weights are on the shared ComfyUI. They wait for a measured route."),
        picture_row("flux3k", "FLUX3-K", any(name.startswith("flux3k") for name in pictures), flags["flux3_file"],
                    "FLUX 3 Image Klein runs where its own weights are installed.",
                    "FLUX3-K weights are on the shared ComfyUI. They wait for a measured route."),
        picture_row("h3", "MiniMax H3", local, flags["h3_file"],
                    "Video runs on a Mac or a DGX Spark, or on a worker a controller sends it to.",
                    "MiniMax H3 weights are on the shared ComfyUI. This process does not render video locally."),
    ]
    for kind, label, detail in (
        ("export", "Export", "Assembles files that already exist."),
        ("soundtrack", "Soundtrack", "Lays a finished song under a clip."),
    ):
        route = (public["routes"] or {}).get(kind) or {}
        software.append(_row(kind, label, "runs" if route.get("enabled", True) else "closed", detail))

    skip = {"sdxl-turbo", "qwen-image", "flux3k", "minimax-h3", "qwen-vae"}
    present = []
    for item in families:
        if item["id"] in skip:
            continue
        extra = ""
        if item["id"] == "flux2-klein":
            extra = " It is a different model from FLUX3-K."
        present.append(_row(item["id"], item["label"], "present", f"On the shared ComfyUI ({item['file']}).{extra}"))
    if flags["other"]:
        present.append(_row("other", "Other files", "present",
                            f"{flags['other']} other files are on the shared ComfyUI."))
    present.sort(key=lambda item: item["label"])
    note = NOTE + (CONTAINER_NOTE if os.environ.get("STUDIO_CONTAINER", "").strip() == "1" else "")
    return {
        "local_render": local,
        "note": note,
        "hardware": hardware_rows,
        "software": software,
        "present": present,
        "present_note": "These files are on the shared ComfyUI. A job is sent only for a measured route above." if present else "",
        "listed": {"qwen_vae": flags["qwen_vae"], "files": files is not None, "known": [item["id"] for item in families]},
    }
