#!/usr/bin/env python3
"""SDXL Turbo on this RTX machine, as one prompt to the shared ComfyUI. No standalone inference.

The checkpoint is the existing fp16 file. Nothing here downloads, moves, or hashes it. Diffusers
``guidance_scale=0.0`` is not ComfyUI's ``cfg``. ComfyUI's sampler uses ``uncond + cfg * (cond -
uncond)`` and skips the unconditional pass when cfg is 1, so the recipe's cfg is 1.0. cfg 0 would
keep only the empty negative.

The shared server is already running at http://127.0.0.1:8188. A still is a POST to ``/prompt``
when its queue is empty. This module does not start, stop, interrupt, unload, or clear that
server, and it does not launch a second one. Spark's own ``call()`` still refuses port 8188.

The model stays unoffered until a profile with ``enabled`` true records a finished render.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import rtx_admit

ID = "sdxl-turbo"
LABEL = "SDXL Turbo"
NOTE = ("1 step at 512×512, fp16, batch size one. Stability AI Non-Commercial Research Community License. "
        "A still picture only: no edit, LoRA, reference, or video.")
CHECKPOINT = "sd_xl_turbo_1.0_fp16.safetensors"
# Phase 0 recorded this size and sha256. The size is checked again. The file is not re-read under the floor.
BYTES = 6_938_081_905
SHA256 = "e869ac7d6942cb327d68d5ed83a40447aadf20e0c3358d98b2cc9e270db0da26"
WIDTH = 512
HEIGHT = 512
STEPS = 1
CFG = 1.0
SAMPLER = "euler_ancestral"
SCHEDULER = "sgm_uniform"
BATCH = 1
MAX_COUNT = 4
WEIGHT_MIB = (BYTES + 1024 * 1024 - 1) // (1024 * 1024)
PINNED_COMFY = "b9b290e43f1eb379946dfeb013faf9bbefa1a618"
SHARED_URL = "http://127.0.0.1:8188"
SHARED_PORT = 8188
WAIT_S = 600


def profile_path() -> Path:
    named = os.environ.get("SDXL_PROFILE", "").strip()
    if named:
        return Path(named).expanduser()
    return Path(__file__).resolve().parent.parent / "docs" / "benchmarks" / "rtx3060-images.json"


def load_profile() -> dict | None:
    path = profile_path()
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def checkpoint_path() -> Path | None:
    named = os.environ.get("SDXL_CHECKPOINT", "").strip()
    candidates = [Path(named).expanduser()] if named else []
    home = Path.home()
    candidates.append(home / "ComfyUI" / "models" / "checkpoints" / CHECKPOINT)
    prefix = Path(os.environ.get("PREFIX", home / ".local/opt/tensorfold-studio"))
    candidates.append(prefix / "ComfyUI" / "models" / "checkpoints" / CHECKPOINT)
    for path in candidates:
        if path.is_file():
            return path
    return None


def max_count() -> int:
    recipe = (load_profile() or {}).get("recipe") or {}
    try:
        count = int(recipe.get("max_count", 1))
    except (TypeError, ValueError):
        return 1
    return max(1, min(count, MAX_COUNT))


def offered() -> bool:
    """True only for a finished measured profile whose checkpoint is still the same size."""

    profile = load_profile()
    if not profile or profile.get("enabled") is not True or profile.get("model") != ID:
        return False
    recipe = profile.get("recipe") or {}
    if (recipe.get("width"), recipe.get("height"), recipe.get("steps"), recipe.get("cfg"),
            recipe.get("sampler"), recipe.get("scheduler"), recipe.get("batch_size")) != (
            WIDTH, HEIGHT, STEPS, CFG, SAMPLER, SCHEDULER, BATCH):
        return False
    path = checkpoint_path()
    recorded = (profile.get("checkpoint") or {}).get("bytes")
    if path is None or recorded != BYTES:
        return False
    try:
        return path.stat().st_size == BYTES
    except OSError:
        return False


def public_limits() -> dict:
    return {"width": WIDTH, "height": HEIGHT, "steps": STEPS, "cfg": CFG, "max_count": max_count(),
            "batch_size": BATCH, "sampler": SAMPLER, "scheduler": SCHEDULER}


def script_block() -> str | None:
    """Why ``require-route.sh sdxl-turbo`` must stop. None when the measured profile is offered."""

    if offered():
        return None
    return ("SDXL Turbo has no measured profile on this machine. "
            "An RTX card is not a production render mode until that profile exists.")


def _refuse(code: str, reason: str) -> dict:
    return {"ok": False, "error": reason, "code": code, "reasons": [{"code": code, "reason": reason}]}


def submission(kind: str, params: dict) -> dict | None:
    """None when this job is not SDXL Turbo. Otherwise an allow or a refusal for the queue."""

    model = str(params.get("image_model") or "")
    if model != ID:
        return None
    if kind in ("edit", "video", "studio", "long", "turnaround", "upscale", "mask", "remove"):
        return _refuse("unmeasured_configuration",
                       "SDXL Turbo makes a still picture. It does not edit, animate, or take a LoRA or a reference.")
    if kind != "scout":
        return _refuse("unmeasured_configuration", "SDXL Turbo is a still-picture scout, not this job.")
    if params.get("loras") or params.get("references") or params.get("reference_videos") or params.get("reference_audios"):
        return _refuse("unmeasured_configuration",
                       "SDXL Turbo does not take a LoRA, a reference picture, a reference video, or reference audio.")
    if params.get("negative") or params.get("base") or params.get("upsample") is False:
        return _refuse("unmeasured_configuration",
                       "SDXL Turbo keeps the prompt as written. It has no negative prompt, base model, or caption upsampler.")
    if not offered():
        return _refuse("unmeasured_configuration", script_block() or "")
    try:
        count = int(params.get("count", 1))
    except (TypeError, ValueError):
        return _refuse("unmeasured_configuration", "SDXL Turbo count is a whole number of pictures.")
    if not 1 <= count <= max_count():
        return _refuse("unmeasured_configuration",
                       f"SDXL Turbo scouts are 1 to {max_count()} pictures, one at a time.")
    if params.get("width") or params.get("height"):
        try:
            asked = (int(params.get("width")), int(params.get("height")))
        except (TypeError, ValueError):
            return _refuse("unmeasured_configuration", f"SDXL Turbo is measured at {WIDTH}x{HEIGHT} only.")
        if asked != (WIDTH, HEIGHT):
            return _refuse("unmeasured_configuration",
                           f"SDXL Turbo is measured at {WIDTH}x{HEIGHT} only. That size is not a video preset.")
    return {"ok": True}


def canvas(params: dict) -> tuple[int, int]:
    """The measured still size. A video scout preset does not choose it."""

    if params.get("width") or params.get("height"):
        try:
            asked = (int(params.get("width")), int(params.get("height")))
        except (TypeError, ValueError) as error:
            raise ValueError(f"SDXL Turbo is measured at {WIDTH}x{HEIGHT} only.") from error
        if asked != (WIDTH, HEIGHT):
            raise ValueError(f"SDXL Turbo is measured at {WIDTH}x{HEIGHT} only.")
    return WIDTH, HEIGHT


def workflow(prompt: str, seed: int, width: int = WIDTH, height: int = HEIGHT, prefix: str = "studio/sdxl") -> dict:
    """ComfyUI's standard checkpoint graph. The prompt is the positive text, unchanged."""

    if width != WIDTH or height != HEIGHT or width % 16 or height % 16:
        raise ValueError(f"SDXL Turbo is measured at {WIDTH}x{HEIGHT} only.")
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CHECKPOINT}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": BATCH}},
        "5": {"class_type": "KSampler", "inputs": {
            "seed": int(seed), "steps": STEPS, "cfg": CFG, "sampler_name": SAMPLER, "scheduler": SCHEDULER,
            "denoise": 1.0, "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0]}},
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": prefix}},
    }


def comfy_root() -> Path:
    return Path(os.environ.get("SDXL_COMFY_ROOT", Path.home() / "ComfyUI")).expanduser()


def listener_pids(port: int = SHARED_PORT) -> set[int] | None:
    """Pids listening on 127.0.0.1:port. None when the table cannot be read. This does not signal them."""

    try:
        text = Path("/proc/net/tcp").read_text(errors="replace")
    except OSError:
        return None
    inodes = set()
    want = f"{port:04X}"
    for line in text.splitlines()[1:]:
        parts = line.split()
        if len(parts) < 10:
            continue
        local, state, inode = parts[1], parts[3], parts[9]
        addr, _, phex = local.partition(":")
        if state == "0A" and addr.upper() == "0100007F" and phex.upper() == want:
            inodes.add(inode)
    if not inodes:
        return set()
    found = set()
    try:
        entries = list(Path("/proc").iterdir())
    except OSError:
        return None
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            descriptors = list((entry / "fd").iterdir())
        except OSError:
            continue
        for descriptor in descriptors:
            try:
                target = os.readlink(descriptor)
            except OSError:
                continue
            if target.startswith("socket:[") and target.endswith("]") and target[8:-1] in inodes:
                found.add(int(entry.name))
                break
    return found


def exchange(method: str, path: str, payload: dict | None = None, timeout: float = 30):
    """One read, or a POST to /prompt. Anything else is refused before a socket opens."""

    if method == "GET":
        allowed = path in ("/queue", "/system_stats") or path.startswith("/history/") or path.startswith("/view?")
        if payload is not None or not allowed:
            raise RuntimeError("that shared ComfyUI request is not a still")
    elif method == "POST" and path == "/prompt" and payload is not None:
        allowed = True
    else:
        raise RuntimeError("Studio does not start, stop, interrupt, unload, or clear the shared ComfyUI")
    request = urllib.request.Request(
        SHARED_URL + path, data=None if payload is None else json.dumps(payload).encode(),
        method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as reply:
            body = reply.read()
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")[:500]
        raise RuntimeError(f"the shared ComfyUI answered {error.code}: {detail}") from None
    except OSError as error:
        raise RuntimeError(f"the shared ComfyUI at {SHARED_URL} did not answer: {error}") from None
    if path.startswith("/view?"):
        return body
    return json.loads(body) if body else {}


def queue_busy(body: dict) -> bool:
    return bool((body or {}).get("queue_running") or (body or {}).get("queue_pending"))


def _saved_image(record: dict) -> dict:
    found = [item for output in (record.get("outputs") or {}).values() for items in output.values()
             for item in (items if isinstance(items, list) else [])
             if isinstance(item, dict) and item.get("filename") and item.get("type", "output") == "output"]
    if not found:
        raise RuntimeError("the shared ComfyUI finished without an image")
    return found[0]


def gpu_processes() -> list[dict] | None:
    """Compute apps on the GPU, or None when the probe fails. This does not signal them."""

    try:
        done = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if done.returncode != 0:
        return None
    rows = []
    for line in (done.stdout or "").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3 or not parts[0].isdigit():
            continue
        vram = int(parts[2]) if parts[2].isdigit() else None
        rows.append({"pid": int(parts[0]), "name": parts[1], "vram_mib": vram})
    return rows


def assess() -> dict:
    """Budgets, the checkpoint size, and whether another process holds the GPU. No render."""

    path = checkpoint_path()
    size = None
    if path is not None:
        try:
            size = path.stat().st_size
        except OSError:
            size = None
    sample = rtx_admit.collect()
    sample["need_vram_mib"] = WEIGHT_MIB
    sample["profile"] = True
    found = rtx_admit.decide(sample)
    processes = gpu_processes()
    listeners = listener_pids()
    shared = set(listeners or [])
    foreign = [row for row in (processes or []) if row["pid"] not in shared and row["pid"] != os.getpid()]
    reasons = []
    if size != BYTES:
        reasons.append({"code": "missing_model", "reason": "The SDXL Turbo checkpoint is not the measured file."})
    if not found["ok"]:
        reasons.append({"code": found["code"], "reason": found["reason"]})
    if listeners is None:
        reasons.append({"code": "missing_telemetry",
                        "reason": "The shared ComfyUI listener could not be read. A missing reading is not a free server."})
    elif not listeners:
        reasons.append({"code": "renderer_down",
                        "reason": "The shared ComfyUI is not listening on 127.0.0.1:8188. Studio does not start it."})
    if processes is None:
        reasons.append({"code": "missing_telemetry",
                        "reason": "The GPU process list could not be read. A missing reading is not a free GPU."})
    elif foreign:
        reasons.append({"code": "busy_renderer",
                        "reason": "Another process besides the shared ComfyUI holds the GPU. Studio does not stop it."})
    return {
        "ok": not reasons,
        "model": ID,
        "enabled": False,
        "reasons": reasons,
        "code": reasons[0]["code"] if reasons else "",
        "reason": reasons[0]["reason"] if reasons else "",
        "checkpoint": {"file": CHECKPOINT, "bytes": size, "expected_bytes": BYTES, "sha256": SHA256,
                       "sha256_source": "phase 0; this run checked the file size and did not read the file",
                       "license": "Stability AI Non-Commercial Research Community License",
                       "path": str(path) if path else ""},
        "recipe": public_limits(),
        "host_available_mb": sample.get("host_available_mb"),
        "host_total_mb": sample.get("host_total_mb"),
        "floor_mb": rtx_admit.host_floor_mb(sample.get("host_total_mb")),
        "vram_free_mib": sample.get("vram_free_mib"),
        "vram_need_mib": WEIGHT_MIB,
        "vram_reserve_mib": sample.get("vram_reserve_mib") or rtx_admit.vram_reserve_mib(),
        "cgroup": sample.get("cgroup"),
        "gpu_processes": processes,
        "shared_pids": sorted(shared),
        "shared_url": SHARED_URL,
        "comfy_pinned": PINNED_COMFY,
        "attention": "pytorch",
        "precision": "fp16",
        "render": None,
    }


def _png_size(data: bytes) -> tuple[int, int]:
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
        raise RuntimeError("the shared ComfyUI did not return a PNG")
    return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")


def _sample_floor() -> dict:
    """One more read of the budgets while a still is in the shared queue. This does not stop it."""

    try:
        sample = rtx_admit.collect()
    except Exception:
        return {}
    return {"host_available_mb": sample.get("host_available_mb"), "vram_free_mib": sample.get("vram_free_mib")}


def render_images(args, text: str, render=None, fetch=None, url: str | None = None) -> list[dict]:
    """Submit stills to the shared ComfyUI. ``render``, ``fetch`` and ``url`` are ignored.

    Spark's renderer posts interrupts and follows SPARK_COMFY_URL. This route does neither.
    """

    del render, fetch, url
    if args.base:
        sys.exit("SDXL Turbo has no base-model switch.")
    if (args.negative or "").strip():
        sys.exit("SDXL Turbo keeps the prompt as written and takes no negative prompt.")
    if args.steps not in (None, STEPS):
        sys.exit(f"SDXL Turbo is measured at {STEPS} step.")
    if args.width != WIDTH or args.height != HEIGHT:
        sys.exit(f"SDXL Turbo is measured at {WIDTH}x{HEIGHT} only.")
    if not offered() and os.environ.get("RTX_SDXL_CALIBRATE") != "1":
        sys.exit(script_block())
    report = assess()
    if not report["ok"]:
        sys.exit(report["reason"])
    seeds = [int(part) for part in args.seeds.split(",") if part.strip()] if args.seeds else [args.seed]
    if not 1 <= len(seeds) <= max_count():
        sys.exit(f"SDXL Turbo scouts are 1 to {max_count()} pictures, one at a time.")
    target = Path(args.output)
    made = []
    for seed in seeds:
        queue = exchange("GET", "/queue", timeout=10)
        if queue_busy(queue):
            sys.exit("The shared ComfyUI queue already has work. Studio leaves that queue alone.")
        started = time.time()
        peaks = _sample_floor()
        posted = exchange("POST", "/prompt", {
            "prompt": workflow(text, seed, args.width, args.height),
            "client_id": "studio-sdxl"}, timeout=60)
        prompt_id = str(posted.get("prompt_id") or "")
        if not prompt_id:
            sys.exit("The shared ComfyUI did not accept the still.")
        record = None
        deadline = time.time() + WAIT_S
        while time.time() < deadline:
            sample = _sample_floor()
            for key, value in sample.items():
                if value is None:
                    continue
                current = peaks.get(key)
                peaks[key] = value if current is None else min(current, value)
            history = exchange("GET", f"/history/{prompt_id}", timeout=30)
            if prompt_id in history and history[prompt_id].get("status", {}).get("completed") is not None:
                record = history[prompt_id]
                break
            time.sleep(1)
        if record is None:
            sys.exit("The still did not finish within the wait. Studio did not interrupt the shared ComfyUI.")
        if record.get("status", {}).get("status_str") != "success":
            sys.exit("The shared ComfyUI did not finish the still. Studio did not interrupt it.")
        item = _saved_image(record)
        query = urllib.parse.urlencode({
            "filename": item["filename"], "subfolder": item.get("subfolder", ""),
            "type": item.get("type", "output")})
        data = exchange("GET", "/view?" + query, timeout=60)
        wide, tall = _png_size(data)
        if (wide, tall) != (args.width, args.height):
            sys.exit(f"The still came back as {wide}x{tall}. SDXL Turbo is measured at {args.width}x{args.height}.")
        out = target.with_name(f"{target.stem}_s{seed}{target.suffix}") if args.seeds else target
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        row = {"output": str(out), "engine": "cuda", "model": ID, "width": wide, "height": tall,
               "seed": seed, "steps": STEPS, "cfg": CFG, "sampler": SAMPLER, "scheduler": SCHEDULER,
               "prompt_id": prompt_id, "bytes": len(data), "total_s": round(time.time() - started, 1),
               "url": SHARED_URL, "host_available_mb_min": peaks.get("host_available_mb"),
               "vram_free_mib_min": peaks.get("vram_free_mib")}
        print("[tensorfold] " + json.dumps(row), flush=True)
        made.append(row)
    return made


def _runtime_facts() -> dict:
    """Read-only names for the report. This does not start ComfyUI or read the checkpoint bytes."""

    facts = {"comfy_installed": "", "attention_flag": False, "fp16_unet_flag": False,
             "gpu_name": "", "gpu_uuid": "", "gpu_memory_mib": None, "driver": ""}
    root = comfy_root()
    try:
        done = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=2)
        if done.returncode == 0:
            facts["comfy_installed"] = (done.stdout or "").strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    cli = root / "comfy" / "cli_args.py"
    try:
        text = cli.read_text(errors="replace") if cli.is_file() else ""
    except OSError:
        text = ""
    facts["attention_flag"] = "--use-pytorch-cross-attention" in text
    facts["fp16_unet_flag"] = "--fp16-unet" in text
    try:
        done = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,uuid,memory.total,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        done = None
    if done is not None and done.returncode == 0 and (done.stdout or "").strip():
        parts = [part.strip() for part in done.stdout.splitlines()[0].split(",")]
        if len(parts) >= 4:
            facts["gpu_name"] = parts[0]
            facts["gpu_uuid"] = parts[1]
            facts["gpu_memory_mib"] = int(parts[2]) if parts[2].isdigit() else None
            facts["driver"] = parts[3]
    return facts


def _limit_note(report: dict) -> str:
    """Which budgets failed. A clear budget still does not mean a render ran."""

    limits = []
    available = report.get("host_available_mb")
    floor = report.get("floor_mb")
    if available is not None and floor is not None and available < floor:
        limits.append(f"host available {available} MB is under the {floor} MB floor")
    free = report.get("vram_free_mib")
    need = report.get("vram_need_mib")
    reserve = report.get("vram_reserve_mib")
    if free is not None and need is not None and reserve is not None and free - reserve < need:
        limits.append(f"free VRAM {free} MiB cannot hold the {need} MiB checkpoint plus the {reserve} MiB reserve")
    cleared = []
    shared = set(report.get("shared_pids") or [])
    foreign = [row for row in (report.get("gpu_processes") or []) if row.get("pid") not in shared and row.get("pid") != os.getpid()]
    if report.get("gpu_processes") is None:
        limits.append("the GPU process list could not be read")
    elif not shared:
        limits.append("the shared ComfyUI is not listening on 127.0.0.1:8188 and was not started")
    elif foreign:
        limits.append("another process besides the shared ComfyUI holds the GPU and was left running")
    else:
        cleared.append("the shared ComfyUI on 127.0.0.1:8188 is the renderer and was left running")
    if available is not None and floor is not None and available >= floor:
        cleared.append(f"host available {available} MB is at or above the {floor} MB floor")
    if free is not None and need is not None and reserve is not None and free - reserve >= need:
        cleared.append(f"free VRAM {free} MiB covers the {need} MiB checkpoint plus the {reserve} MiB reserve")
    if not limits:
        limits.append("no budget failed")
    said = ". ".join(part[:1].upper() + part[1:] for part in limits + cleared)
    return ("No render ran. The profile stays disabled. " + said + ". "
            "A still is one prompt to that server. Studio does not start, stop, interrupt, or unload it.")


def _write_report(path: str, report: dict) -> None:
    if not path:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Submit one SDXL Turbo still to the shared ComfyUI, or only check the budgets.")
    parser.add_argument("command", choices=("calibrate", "render"))
    parser.add_argument("--record", default="", help="write the JSON report here")
    parser.add_argument("--prompt", default="a red apple on a wooden table, soft daylight")
    parser.add_argument("--output", default="")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)
    report = assess()
    report.update(_runtime_facts())
    report["comfy_spark_attention"] = "--use-ck-attention"
    if args.command == "calibrate":
        report["render"] = None
        report["enabled"] = False
        report["note"] = _limit_note(report)
        _write_report(args.record, report)
        print(json.dumps({"ok": report["ok"], "enabled": False, "code": report["code"], "reason": report["reason"],
                          "host_available_mb": report["host_available_mb"], "floor_mb": report["floor_mb"],
                          "vram_free_mib": report["vram_free_mib"], "vram_need_mib": report["vram_need_mib"],
                          "note": report["note"]}))
        return 0 if report["ok"] else 1
    if not report["ok"]:
        report["render"] = None
        report["enabled"] = False
        report["note"] = _limit_note(report)
        _write_report(args.record, report)
        print(json.dumps({"ok": False, "enabled": False, "code": report["code"], "reason": report["reason"]}))
        return 1
    output = args.output or str(Path.home() / "ImaginationStudio" / "sdxl-turbo" / f"still_s{args.seed}.png")
    request = type("Request", (), {
        "base": False, "negative": "", "steps": None, "width": WIDTH, "height": HEIGHT,
        "seeds": "", "seed": args.seed, "output": output,
    })()
    os.environ["RTX_SDXL_CALIBRATE"] = "1"
    try:
        made = render_images(request, args.prompt)
    except SystemExit as error:
        report["render"] = None
        report["enabled"] = False
        report["ok"] = False
        report["code"] = "render_failed"
        report["reason"] = str(error)
        report["note"] = "No finished still was recorded. The profile stays disabled. " + str(error)
        _write_report(args.record, report)
        print(json.dumps({"ok": False, "enabled": False, "reason": str(error)}))
        return 1
    report["render"] = made[0]
    report["enabled"] = True
    report["ok"] = True
    report["code"] = ""
    report["reason"] = ""
    low = made[0].get("host_available_mb_min")
    floor = rtx_admit.HOST_FLOOR_MB
    dip = ""
    if low is not None and low < floor:
        dip = (f" During the load, host available fell to {low} MB, under the {floor} MB floor. "
               "The floor was not lowered. A later still still has to clear it before it is submitted.")
    report["note"] = ("One still finished on the shared ComfyUI at 127.0.0.1:8188. "
                      "Studio did not start, stop, interrupt, or unload that server." + dip)
    _write_report(args.record, report)
    print(json.dumps({"ok": True, "enabled": True, "output": made[0]["output"], "total_s": made[0]["total_s"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
