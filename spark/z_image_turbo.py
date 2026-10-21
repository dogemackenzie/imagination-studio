#!/usr/bin/env python3
"""Z-Image Turbo on this RTX machine, as one prompt to the shared ComfyUI. No standalone inference.

The graph is the installed int8 template, not the bf16 blueprint. UNET
``z_image_turbo_int8_convrot.safetensors``, CLIP ``qwen_3_4b_fp8_mixed.safetensors``
as ``lumina2``, VAE ``ae.safetensors``, an empty SD3 latent at 1024×1024, AuraFlow
shift 3, and a KSampler of 8 steps at cfg 1 with ``res_multistep`` / ``simple``.
The negative is that conditioning zeroed out. The prompt is the positive text, unchanged.

ComfyUI loads the text encoder and the VAE and offloads them. The admission need is the
diffusion file, the weight the 8 steps hold. The three files together are larger than
free VRAM on the measured 12 GB card, and a sum would refuse the still the template
already runs. The still records the real dip. The 8000 MB host floor is not lowered.

The shared server is already running. A still is a POST to ``/prompt`` when its queue
is empty. This module does not start, stop, interrupt, unload, or clear that server.
The model stays unoffered until a profile with ``enabled`` true records a finished render.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.parse
from pathlib import Path

import rtx_admit
import sdxl_turbo

ID = "z-image"
LABEL = "Z-Image Turbo"
LICENSE = "Apache License 2.0"
NOTE = ("8 steps at 1024×1024, int8, batch size one. Apache License 2.0. "
        "A still picture only: no edit, LoRA, reference, or video. "
        "Z-Image Turbo is a different model from Qwen-Image-2.1.")
UNET = "z_image_turbo_int8_convrot.safetensors"
CLIP = "qwen_3_4b_fp8_mixed.safetensors"
VAE = "ae.safetensors"
UNET_BYTES = 6_201_001_296
CLIP_BYTES = 5_631_994_051
VAE_BYTES = 335_304_388
WIDTH = 1024
HEIGHT = 1024
STEPS = 8
CFG = 1.0
SHIFT = 3.0
SAMPLER = "res_multistep"
SCHEDULER = "simple"
CLIP_TYPE = "lumina2"
WEIGHT_DTYPE = "default"
BATCH = 1
MAX_COUNT = 1
WAIT_S = 600


def _mib(size: int) -> int:
    return (size + 1024 * 1024 - 1) // (1024 * 1024)


UNET_MIB = _mib(UNET_BYTES)
CLIP_MIB = _mib(CLIP_BYTES)
VAE_MIB = _mib(VAE_BYTES)
# What the sampler holds. Not the sum of the three files.
WEIGHT_MIB = UNET_MIB
FILES = (
    ("unet", "ZIMAGE_UNET", "diffusion_models", UNET, UNET_BYTES),
    ("clip", "ZIMAGE_CLIP", "text_encoders", CLIP, CLIP_BYTES),
    ("vae", "ZIMAGE_VAE", "vae", VAE, VAE_BYTES),
)


def profile_path() -> Path:
    named = os.environ.get("ZIMAGE_PROFILE", "").strip()
    if named:
        return Path(named).expanduser()
    return Path(__file__).resolve().parent.parent / "docs" / "benchmarks" / "rtx3060-z-image-turbo.json"


def load_profile() -> dict | None:
    path = profile_path()
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _weight_path(env_name: str, folder: str, filename: str) -> Path | None:
    named = os.environ.get(env_name, "").strip()
    candidates = [Path(named).expanduser()] if named else []
    home = Path.home()
    candidates.append(home / "ComfyUI" / "models" / folder / filename)
    prefix = Path(os.environ.get("PREFIX", home / ".local/opt/tensorfold-studio"))
    candidates.append(prefix / "ComfyUI" / "models" / folder / filename)
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0] if named else None


def weight_files() -> dict[str, tuple[Path | None, int | None]]:
    """A stat of each measured file. This does not read or hash the bytes."""

    found = {}
    for key, env_name, folder, filename, _expected in FILES:
        path = _weight_path(env_name, folder, filename)
        size = None
        if path is not None:
            try:
                size = path.stat().st_size
            except OSError:
                size = None
        found[key] = (path, size)
    return found


def max_count() -> int:
    recipe = (load_profile() or {}).get("recipe") or {}
    try:
        count = int(recipe.get("max_count", 1))
    except (TypeError, ValueError):
        return 1
    return max(1, min(count, MAX_COUNT))


def _recipe_ok(recipe: dict) -> bool:
    return (recipe.get("width"), recipe.get("height"), recipe.get("steps"), recipe.get("cfg"),
            recipe.get("shift"), recipe.get("sampler"), recipe.get("scheduler"),
            recipe.get("batch_size"), recipe.get("clip_type"), recipe.get("weight_dtype")) == (
            WIDTH, HEIGHT, STEPS, CFG, SHIFT, SAMPLER, SCHEDULER, BATCH, CLIP_TYPE, WEIGHT_DTYPE)


def classify(profile: dict | None, sizes: dict[str, int | None]) -> dict:
    """Separate files that exist from a model this process may run. This does not submit a prompt."""

    profile = profile or {}
    weights = profile.get("weights") if isinstance(profile.get("weights"), dict) else {}
    recipe = profile.get("recipe") if isinstance(profile.get("recipe"), dict) else {}
    render = profile.get("render") if isinstance(profile.get("render"), dict) else {}
    expected = {key: size for key, _env, _folder, _name, size in FILES}
    compatible = True
    present = True
    for key, size in expected.items():
        block = weights.get(key) if isinstance(weights.get(key), dict) else {}
        got = sizes.get(key)
        if got is None:
            present = False
        if got != size or block.get("bytes") != size:
            compatible = False
    licensed = all(isinstance(weights.get(key), dict) and weights[key].get("license") == LICENSE for key, *_ in FILES)
    benchmarked = bool(render.get("output") and profile.get("gpu_name") and profile.get("model") == ID)
    runnable = bool(profile.get("enabled") is True and profile.get("model") == ID and _recipe_ok(recipe) and compatible)
    return {
        "id": ID,
        "present": present,
        "compatible": compatible,
        "licensed": licensed,
        "license": LICENSE if licensed else "",
        "benchmarked": benchmarked,
        "runnable": runnable,
        "enabled": profile.get("enabled") is True,
        "note": ("The int8 files are runnable when the measured profile is enabled and each file is the recorded size. "
                 "The licence is Apache License 2.0. Admission, including the 8000 MB floor and an empty queue, "
                 "is checked again when a still is submitted."),
    }


def offered() -> bool:
    """True only for a finished measured profile whose three files are still the recorded sizes."""

    sizes = {key: size for key, (_path, size) in weight_files().items()}
    if any(size is None for size in sizes.values()):
        return False
    return classify(load_profile(), sizes)["runnable"]


def public_limits() -> dict:
    return {"width": WIDTH, "height": HEIGHT, "steps": STEPS, "cfg": CFG, "shift": SHIFT,
            "max_count": max_count(), "batch_size": BATCH, "sampler": SAMPLER, "scheduler": SCHEDULER,
            "clip_type": CLIP_TYPE, "weight_dtype": WEIGHT_DTYPE}


def script_block() -> str | None:
    """Why ``require-route.sh z-image`` must stop. None when the measured profile is offered."""

    if offered():
        return None
    return ("Z-Image Turbo has no measured profile on this machine. "
            "An RTX card is not a production render mode until that profile exists.")


def _refuse(code: str, reason: str) -> dict:
    return {"ok": False, "error": reason, "code": code, "reasons": [{"code": code, "reason": reason}]}


def submission(kind: str, params: dict) -> dict | None:
    """None when this job is not Z-Image Turbo. Otherwise an allow or a refusal for the queue."""

    model = str(params.get("image_model") or "")
    if model != ID:
        return None
    if kind in ("edit", "video", "studio", "long", "turnaround", "upscale", "mask", "remove"):
        return _refuse("unmeasured_configuration",
                       "Z-Image Turbo makes a still picture. It does not edit, animate, or take a LoRA or a reference.")
    if kind != "scout":
        return _refuse("unmeasured_configuration", "Z-Image Turbo is a still-picture scout, not this job.")
    if params.get("loras") or params.get("references") or params.get("reference_videos") or params.get("reference_audios"):
        return _refuse("unmeasured_configuration",
                       "Z-Image Turbo does not take a LoRA, a reference picture, a reference video, or reference audio.")
    if params.get("negative") or params.get("base") or params.get("upsample") is False:
        return _refuse("unmeasured_configuration",
                       "Z-Image Turbo keeps the prompt as written. It has no negative prompt, base model, or caption upsampler.")
    if not offered():
        return _refuse("unmeasured_configuration", script_block() or "")
    try:
        count = int(params.get("count", 1))
    except (TypeError, ValueError):
        return _refuse("unmeasured_configuration", "Z-Image Turbo count is a whole number of pictures.")
    if not 1 <= count <= max_count():
        return _refuse("unmeasured_configuration",
                       f"Z-Image Turbo scouts are 1 to {max_count()} pictures, one at a time.")
    if params.get("width") or params.get("height"):
        try:
            asked = (int(params.get("width")), int(params.get("height")))
        except (TypeError, ValueError):
            return _refuse("unmeasured_configuration", f"Z-Image Turbo is measured at {WIDTH}x{HEIGHT} only.")
        if asked != (WIDTH, HEIGHT):
            return _refuse("unmeasured_configuration",
                           f"Z-Image Turbo is measured at {WIDTH}x{HEIGHT} only. That size is not a video preset.")
    return {"ok": True}


def canvas(params: dict) -> tuple[int, int]:
    """The measured still size. A video scout preset does not choose it."""

    if params.get("width") or params.get("height"):
        try:
            asked = (int(params.get("width")), int(params.get("height")))
        except (TypeError, ValueError) as error:
            raise ValueError(f"Z-Image Turbo is measured at {WIDTH}x{HEIGHT} only.") from error
        if asked != (WIDTH, HEIGHT):
            raise ValueError(f"Z-Image Turbo is measured at {WIDTH}x{HEIGHT} only.")
    return WIDTH, HEIGHT


def workflow(prompt: str, seed: int, width: int = WIDTH, height: int = HEIGHT, prefix: str = "studio/z-image") -> dict:
    """The installed int8 template, as an API prompt. The prompt is the positive text, unchanged."""

    if width != WIDTH or height != HEIGHT or width % 16 or height % 16:
        raise ValueError(f"Z-Image Turbo is measured at {WIDTH}x{HEIGHT} only.")
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": UNET, "weight_dtype": WEIGHT_DTYPE}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": CLIP, "type": CLIP_TYPE, "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "5": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}},
        "6": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0], "shift": SHIFT}},
        "7": {"class_type": "EmptySD3LatentImage", "inputs": {"width": width, "height": height, "batch_size": BATCH}},
        "8": {"class_type": "KSampler", "inputs": {
            "seed": int(seed), "steps": STEPS, "cfg": CFG, "sampler_name": SAMPLER, "scheduler": SCHEDULER,
            "denoise": 1.0, "model": ["6", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["7", 0]}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
        "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": prefix}},
    }


def _weight_report(files: dict[str, tuple[Path | None, int | None]]) -> dict:
    report = {}
    for key, _env, _folder, filename, expected in FILES:
        path, size = files[key]
        report[key] = {"file": filename, "bytes": size, "expected_bytes": expected, "license": LICENSE,
                       "path": str(path) if path else ""}
    return report


def _budgets(sample: dict) -> dict:
    sample["need_vram_mib"] = WEIGHT_MIB
    sample["profile"] = True
    return sample


def _reasons_for(files: dict, found: dict | None) -> list[dict]:
    reasons = []
    for key, (_path, size) in files.items():
        expected = dict((item[0], item[4]) for item in FILES)[key]
        if size != expected:
            reasons.append({"code": "missing_model", "reason": f"The Z-Image Turbo {key} file is not the measured file."})
            break
    if found is not None and not found["ok"]:
        reasons.append({"code": found["code"], "reason": found["reason"]})
    return reasons


def _report(files: dict, sample: dict, reasons: list[dict], processes, shared, origin: str) -> dict:
    return {
        "ok": not reasons,
        "model": ID,
        "enabled": False,
        "reasons": reasons,
        "code": reasons[0]["code"] if reasons else "",
        "reason": reasons[0]["reason"] if reasons else "",
        "weights": _weight_report(files),
        "recipe": public_limits(),
        "host_available_mb": sample.get("host_available_mb"),
        "host_total_mb": sample.get("host_total_mb"),
        "floor_mb": rtx_admit.host_floor_mb(sample.get("host_total_mb")),
        "vram_free_mib": sample.get("vram_free_mib"),
        "vram_need_mib": WEIGHT_MIB,
        "vram_files_mib": {"unet": UNET_MIB, "clip": CLIP_MIB, "vae": VAE_MIB},
        "vram_reserve_mib": sample.get("vram_reserve_mib") or rtx_admit.vram_reserve_mib(),
        "cgroup": sample.get("cgroup"),
        "gpu_processes": processes,
        "shared_pids": sorted(shared),
        "shared_url": origin,
        "attention": "pytorch",
        "precision": "int8",
        "render": None,
    }


def _assess_shared() -> dict:
    """Budgets for a still that runs in the shared ComfyUI, not in this container."""

    files = weight_files()
    sample = _budgets(rtx_admit.collect())
    sample["cgroup"] = {"known": True, "cap": False, "headroom_mb": None, "pressure_full_avg10": 0.0}
    try:
        origin = sdxl_turbo.shared_origin()
    except RuntimeError as error:
        origin = ""
        reasons = _reasons_for(files, None)
        reasons.append({"code": "protected_endpoint", "reason": str(error)})
        return _report(files, sample, reasons, None, [], origin)
    stats = None
    try:
        stats = sdxl_turbo.exchange("GET", "/system_stats", timeout=5)
    except RuntimeError:
        stats = None
    reasons = _reasons_for(files, None)
    if not isinstance(stats, dict):
        reasons.append({"code": "renderer_down", "reason": "The shared ComfyUI did not answer. Studio does not start it."})
    else:
        device = next((item for item in (stats.get("devices") or []) if item.get("type") == "cuda"), None)
        free = device.get("vram_free") if isinstance(device, dict) else None
        if isinstance(free, int):
            sample["vram_free_mib"] = free // (1024 * 1024)
            sample["vram_known"] = True
        else:
            sample["vram_known"] = False
            sample["vram_free_mib"] = None
        found = rtx_admit.decide(sample)
        if not found["ok"]:
            reasons.append({"code": found["code"], "reason": found["reason"]})
    return _report(files, sample, reasons, None, [], origin)


def assess() -> dict:
    """Budgets and the three file sizes. No render."""

    if sdxl_turbo.in_container():
        return _assess_shared()
    files = weight_files()
    sample = _budgets(rtx_admit.collect())
    found = rtx_admit.decide(sample)
    processes = sdxl_turbo.gpu_processes()
    listeners = sdxl_turbo.listener_pids()
    shared = set(listeners or [])
    foreign = [row for row in (processes or []) if row["pid"] not in shared and row["pid"] != os.getpid()]
    reasons = _reasons_for(files, found)
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
    return _report(files, sample, reasons, processes, shared, sdxl_turbo.SHARED_URL)


def render_images(args, text: str, render=None, fetch=None, url: str | None = None) -> list[dict]:
    """Submit one still to the shared ComfyUI. ``render``, ``fetch`` and ``url`` are ignored."""

    del render, fetch, url
    if args.base:
        sys.exit("Z-Image Turbo has no base-model switch.")
    if (args.negative or "").strip():
        sys.exit("Z-Image Turbo keeps the prompt as written and takes no negative prompt.")
    if args.steps not in (None, STEPS):
        sys.exit(f"Z-Image Turbo is measured at {STEPS} steps.")
    if args.width != WIDTH or args.height != HEIGHT:
        sys.exit(f"Z-Image Turbo is measured at {WIDTH}x{HEIGHT} only.")
    if not offered() and os.environ.get("RTX_ZIMAGE_CALIBRATE") != "1":
        sys.exit(script_block())
    report = assess()
    if not report["ok"]:
        sys.exit(report["reason"])
    seeds = [int(part) for part in args.seeds.split(",") if part.strip()] if args.seeds else [args.seed]
    if not 1 <= len(seeds) <= max_count():
        sys.exit(f"Z-Image Turbo scouts are 1 to {max_count()} pictures, one at a time.")
    target = Path(args.output)
    made = []
    for seed in seeds:
        queue = sdxl_turbo.exchange("GET", "/queue", timeout=10)
        if sdxl_turbo.queue_busy(queue):
            sys.exit("The shared ComfyUI queue already has work. Studio leaves that queue alone.")
        started = time.time()
        peaks = sdxl_turbo._sample_floor()
        posted = sdxl_turbo.exchange("POST", "/prompt", {
            "prompt": workflow(text, seed, args.width, args.height),
            "client_id": "studio-z-image"}, timeout=60)
        prompt_id = str(posted.get("prompt_id") or "")
        if not prompt_id:
            sys.exit("The shared ComfyUI did not accept the still.")
        record = None
        deadline = time.time() + WAIT_S
        while time.time() < deadline:
            sample = sdxl_turbo._sample_floor()
            for key, value in sample.items():
                if value is None:
                    continue
                current = peaks.get(key)
                peaks[key] = value if current is None else min(current, value)
            history = sdxl_turbo.exchange("GET", f"/history/{prompt_id}", timeout=30)
            if prompt_id in history and history[prompt_id].get("status", {}).get("completed") is not None:
                record = history[prompt_id]
                break
            time.sleep(1)
        if record is None:
            sys.exit("The still did not finish within the wait. Studio did not interrupt the shared ComfyUI.")
        if record.get("status", {}).get("status_str") != "success":
            sys.exit("The shared ComfyUI did not finish the still. Studio did not interrupt it.")
        item = sdxl_turbo._saved_image(record)
        query = urllib.parse.urlencode({
            "filename": item["filename"], "subfolder": item.get("subfolder", ""),
            "type": item.get("type", "output")})
        data = sdxl_turbo.exchange("GET", "/view?" + query, timeout=60)
        wide, tall = sdxl_turbo._png_size(data)
        if (wide, tall) != (args.width, args.height):
            sys.exit(f"The still came back as {wide}x{tall}. Z-Image Turbo is measured at {args.width}x{args.height}.")
        out = target.with_name(f"{target.stem}_s{seed}{target.suffix}") if args.seeds else target
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        row = {"output": str(out), "engine": "cuda", "model": ID, "width": wide, "height": tall,
               "seed": seed, "steps": STEPS, "cfg": CFG, "shift": SHIFT, "sampler": SAMPLER,
               "scheduler": SCHEDULER, "prompt_id": prompt_id, "bytes": len(data),
               "total_s": round(time.time() - started, 1), "url": sdxl_turbo.shared_origin(),
               "host_available_mb_min": peaks.get("host_available_mb"),
               "vram_free_mib_min": peaks.get("vram_free_mib")}
        print("[tensorfold] " + json.dumps(row), flush=True)
        made.append(row)
    return made


def _limit_note(report: dict) -> str:
    limits = []
    available = report.get("host_available_mb")
    floor = report.get("floor_mb")
    if available is not None and floor is not None and available < floor:
        limits.append(f"host available {available} MB is under the {floor} MB floor")
    free = report.get("vram_free_mib")
    need = report.get("vram_need_mib")
    reserve = report.get("vram_reserve_mib")
    if free is not None and need is not None and reserve is not None and free - reserve < need:
        limits.append(f"free VRAM {free} MiB cannot hold the {need} MiB diffusion file plus the {reserve} MiB reserve")
    shared = set(report.get("shared_pids") or [])
    foreign = [row for row in (report.get("gpu_processes") or []) if row.get("pid") not in shared and row.get("pid") != os.getpid()]
    cleared = []
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
        cleared.append(f"free VRAM {free} MiB covers the {need} MiB diffusion file plus the {reserve} MiB reserve")
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
    parser = argparse.ArgumentParser(description="Submit one Z-Image Turbo still to the shared ComfyUI, or only check the budgets.")
    parser.add_argument("command", choices=("calibrate", "render"))
    parser.add_argument("--record", default="", help="write the JSON report here")
    parser.add_argument("--prompt", default="a red apple on a wooden table, soft daylight")
    parser.add_argument("--output", default="")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)
    report = assess()
    report.update(sdxl_turbo._runtime_facts())
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
    output = args.output or str(Path.home() / "ImaginationStudio" / "z-image" / f"still_s{args.seed}.png")
    request = type("Request", (), {
        "base": False, "negative": "", "steps": None, "width": WIDTH, "height": HEIGHT,
        "seeds": "", "seed": args.seed, "output": output,
    })()
    os.environ["RTX_ZIMAGE_CALIBRATE"] = "1"
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
        dip += (f" During the load, host available fell to {low} MB, under the {floor} MB floor. "
                "The floor was not lowered. A later still still has to clear it before it is submitted.")
    vram_low = made[0].get("vram_free_mib_min")
    reserve = report.get("vram_reserve_mib") or rtx_admit.vram_reserve_mib()
    if vram_low is not None and vram_low < reserve:
        dip += (f" Free VRAM fell to {vram_low} MiB, under the {reserve} MiB reserve. "
                "The reserve was not lowered.")
    report["note"] = ("One still finished on the shared ComfyUI at 127.0.0.1:8188. "
                      "Studio did not start, stop, interrupt, or unload that server. "
                      "The admission need is the diffusion file. The text encoder and the VAE are loaded and offloaded."
                      + dip)
    _write_report(args.record, report)
    print(json.dumps({"ok": True, "enabled": True, "output": made[0]["output"], "total_s": made[0]["total_s"],
                      "host_available_mb_min": made[0].get("host_available_mb_min"),
                      "vram_free_mib_min": made[0].get("vram_free_mib_min")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
