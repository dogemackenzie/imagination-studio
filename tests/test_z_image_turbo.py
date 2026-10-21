"""Z-Image Turbo on this RTX machine (spark/z_image_turbo.py).

Tests never open a socket to the shared ComfyUI. A measured profile in a test is three tiny
stand-in files. The real weights are not read. An allowed scout is built and not queued.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "spark"))
sys.path.insert(0, str(ROOT / "app"))
import capabilities  # noqa: E402
import constraints  # noqa: E402
import nodes  # noqa: E402
import z_image_turbo as Z  # noqa: E402
import server  # noqa: E402
from fastapi import HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(server.app)
PROMPT = "a red apple on a wooden table, soft daylight"


def budgets(**over):
    sample = {
        "host_available_mb": 16000, "host_total_mb": 16000,
        "cgroup": {"known": True, "cap": False, "headroom_mb": None, "pressure_full_avg10": 0.0},
        "vram_free_mib": 12000, "vram_known": True, "need_host_mb": None, "need_vram_mib": None,
        "profile": False, "occupied_port": False, "protected": False, "foreign_lock": False,
        "busy": False, "vram_reserve_mib": 512,
    }
    sample.update(over)
    return sample


@pytest.fixture
def measured(tmp_path, monkeypatch):
    """A profile the queue would accept. Each file is four bytes, not the real weight."""

    tiny = tuple((key, env, folder, name, 4) for key, env, folder, name, _size in Z.FILES)
    monkeypatch.setattr(Z, "FILES", tiny)
    for key, env, _folder, name, _size in tiny:
        path = tmp_path / name
        path.write_bytes(b"test")
        monkeypatch.setenv(env, str(path))
    recipe = Z.public_limits()
    profile = {
        "enabled": True, "model": "z-image", "gpu_name": "NVIDIA GeForce RTX 3060",
        "recipe": recipe, "render": {"output": "/home/spark/ImaginationStudio/z-image/still_s1.png"},
        "weights": {key: {"bytes": 4, "license": Z.LICENSE} for key, *_rest in tiny},
    }
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    monkeypatch.setenv("ZIMAGE_PROFILE", str(path))
    return path


def test_the_graph_is_the_int8_template():
    graph = Z.workflow(PROMPT, 1)
    assert graph["1"]["inputs"] == {"unet_name": Z.UNET, "weight_dtype": "default"}
    assert graph["2"]["inputs"]["clip_name"] == Z.CLIP and graph["2"]["inputs"]["type"] == "lumina2"
    assert graph["3"]["inputs"] == {"vae_name": Z.VAE}
    assert graph["4"]["inputs"]["text"] == PROMPT and graph["4"]["inputs"]["clip"] == ["2", 0]
    assert graph["5"]["class_type"] == "ConditioningZeroOut" and graph["5"]["inputs"]["conditioning"] == ["4", 0]
    assert graph["6"]["inputs"] == {"model": ["1", 0], "shift": 3.0}
    assert graph["7"]["inputs"] == {"width": 1024, "height": 1024, "batch_size": 1}
    sampler = graph["8"]["inputs"]
    assert sampler["steps"] == 8 and sampler["cfg"] == 1.0
    assert sampler["sampler_name"] == "res_multistep" and sampler["scheduler"] == "simple"
    assert sampler["negative"] == ["5", 0] and sampler["positive"] == ["4", 0]
    assert "bf16" not in json.dumps(graph)
    assert sum(1 for node in graph.values() if node["class_type"] == "CLIPTextEncode") == 1


def test_the_recorded_profile_keeps_the_floor_and_a_stand_in_path():
    data = json.loads((ROOT / "docs/benchmarks/rtx3060-z-image-turbo.json").read_text())
    text = json.dumps(data)
    assert data["enabled"] is True and data["model"] == "z-image" and data["floor_mb"] == 8000
    assert data["vram_need_mib"] == Z.UNET_MIB and data["vram_need_mib"] < data["vram_files_mib"]["unet"] + data["vram_files_mib"]["clip"]
    assert data["render"]["prompt_id"] == "8eee4989-b017-419e-9ca2-494bea956040"
    assert data["render"]["width"] == 1024 and data["render"]["bytes"] == 1178389
    assert data["render"]["host_available_mb_min"] == 2079 and data["render"]["vram_free_mib_min"] == 351
    assert data["weights"]["unet"]["bytes"] == Z.UNET_BYTES
    assert data["weights"]["clip"]["file"] == Z.CLIP and data["weights"]["vae"]["file"] == Z.VAE
    assert data["weights"]["unet"]["path"].startswith("/home/spark/")
    assert "192.168." not in text
    assert "not lowered" in data["note"] and "Apache License 2.0" in data["weights"]["unet"]["license"]


def test_admission_need_is_the_diffusion_file(measured, monkeypatch):
    assert Z.WEIGHT_MIB == Z.UNET_MIB
    assert Z.UNET_MIB + Z.CLIP_MIB + Z.VAE_MIB > Z.WEIGHT_MIB + 512
    monkeypatch.setattr(Z.sdxl_turbo, "listener_pids", lambda port=8188: {1})
    monkeypatch.setattr(Z.sdxl_turbo, "gpu_processes", lambda: [{"pid": 1, "name": "python", "vram_mib": 1}])
    monkeypatch.setattr(Z.rtx_admit, "collect", lambda: budgets(vram_free_mib=Z.WEIGHT_MIB + 512))
    assert Z.assess()["ok"] is True and Z.assess()["vram_need_mib"] == Z.UNET_MIB
    monkeypatch.setattr(Z.rtx_admit, "collect", lambda: budgets(vram_free_mib=Z.WEIGHT_MIB + 511))
    low = Z.assess()
    assert low["ok"] is False and low["code"] == "insufficient_vram" and low["enabled"] is False
    monkeypatch.setattr(Z.rtx_admit, "collect", lambda: budgets(host_available_mb=7999, vram_free_mib=12000))
    assert Z.assess()["code"] == "insufficient_memory"


def test_a_missing_profile_is_not_offered():
    assert Z.offered() is False
    assert "measured profile" in (Z.script_block() or "")


def test_an_offered_still_is_1024_and_does_not_open_video(measured, named_rtx):
    assert Z.offered() is True
    models = server.image_models()
    assert models[-1] == "z-image" and "qwen" not in models and "flux3k-fast" not in models
    assert capabilities.renders(False) is False
    state = client.get("/api/state").json()
    card = next(item for item in state["image_models"] if item["id"] == "z-image")
    assert card["width"] == 1024 and card["height"] == 1024 and card["steps"] == 8 and card["cfg"] == 1.0
    assert state["capabilities"]["local_render"] is False
    assert state["capabilities"]["routes"]["video"]["enabled"] is False
    assert state["capabilities"]["routes"]["scout"]["enabled"] is False
    edit = client.get("/api/edit/state").json()
    assert all(model["id"] != "z-image" for model in edit["models"])
    assert "z-image" in nodes.IMAGE_MODELS and "z-image" not in nodes.EDIT_MODELS
    before = set(server.jobs)
    for params in (
        {"prompt": PROMPT, "image_model": "z-image", "width": 512, "height": 512},
        {"prompt": PROMPT, "image_model": "z-image", "count": 2},
        {"prompt": PROMPT, "image_model": "z-image", "negative": "blurry"},
        {"prompt": PROMPT, "image_model": "z-image", "loras": ["x"]},
        {"prompt": PROMPT, "image_model": "z-image", "references": ["a.png"]},
    ):
        answer = client.post("/api/jobs", json={"kind": "scout", "params": params})
        assert answer.status_code == 400 and answer.json()["code"] == "unmeasured_configuration"
    video = client.post("/api/jobs", json={"kind": "video", "params": {"prompt": PROMPT, "image_model": "z-image", "preset": "small", "seconds": 5}})
    assert video.status_code == 400 and "still picture" in video.json()["error"]
    edit_job = client.post("/api/jobs", json={"kind": "edit", "params": {"prompt": PROMPT, "image_model": "z-image"}})
    assert edit_job.status_code == 400 and "still picture" in edit_job.json()["error"]
    assert set(server.jobs) == before
    command, env, outputs = server.build("scout", {
        "prompt": PROMPT, "count": 1, "seed": 3, "preset": "2k", "image_model": "z-image"}, "j1")
    assert env["WIDTH"] == 1024 and env["HEIGHT"] == 1024 and env["IMAGE_MODEL"] == "z-image"
    assert env["STUDIO_PYTHON"] == sys.executable
    assert command[1].endswith("scout.sh") and outputs[0].endswith("scout_s3.png")
    assert Z.submission("scout", {"prompt": PROMPT, "image_model": "z-image", "preset": "2k", "count": 1}) == {"ok": True}
    assert Z.submission("video", {"prompt": PROMPT, "image_model": "sdxl-turbo"}) is None
    with pytest.raises(HTTPException) as caught:
        server.build("scout", {"prompt": PROMPT, "image_model": "z-image", "width": 768, "height": 768}, "j2")
    assert "1024x1024" in str(caught.value.detail)


def test_a_measured_row_runs_and_qwen_stays_missing(monkeypatch):
    monkeypatch.setattr(constraints, "model_files", lambda force=False: {
        "checkpoints": ["sd_xl_turbo_1.0_fp16.safetensors"],
        "diffusion_models": ["z_image_turbo_int8_convrot.safetensors", "flux-2-klein-4b-fp8.safetensors", "wan2.1_t2v_1.3B_fp16.safetensors"],
        "vae": ["ae.safetensors", "qwen_image_vae.safetensors"],
        "text_encoders": ["qwen_3_4b_fp8_mixed.safetensors"],
    })
    monkeypatch.setattr(constraints, "system_stats", lambda force=False: {})
    body = constraints.report(False, ["sdxl-turbo", "z-image"])
    assert body["local_render"] is False
    zrow = next(row for row in body["software"] if row["id"] == "z-image")
    assert zrow["state"] == "runs" and "1024" in zrow["detail"] and "Apache License 2.0" in zrow["detail"]
    qwen = next(row for row in body["software"] if row["id"] == "qwen")
    assert qwen["state"] == "missing" and "stands in" not in qwen["detail"] and "measured still" in qwen["detail"]
    klein = next(row for row in body["software"] if row["id"] == "flux2-klein")
    wan = next(row for row in body["software"] if row["id"] == "wan")
    assert klein["state"] == "stands" and wan["state"] == "stands"
    assert "z-image" not in {row["id"] for row in body["present"]}
    assert constraints.match_file("diffusion_models", "z_image_turbo_int8_convrot.safetensors")[2] == ""


def test_the_scripts_stop_before_a_render(tmp_path):
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "IMAGE_MODEL": "z-image",
           "STUDIO_PYTHON": sys.executable, "PREFIX": str(tmp_path / "prefix")}
    image = subprocess.run(["bash", str(ROOT / "scripts/image.sh"), PROMPT, str(tmp_path / "out.png")],
                           capture_output=True, text=True, timeout=30, env=env)
    assert image.returncode == 1 and "measured" in image.stderr.lower()
    assert not (tmp_path / "out.png").exists()
    scout = subprocess.run(["bash", str(ROOT / "scripts/scout.sh"), PROMPT, "1", str(tmp_path / "scouts")],
                           capture_output=True, text=True, timeout=30, env=env)
    assert scout.returncode == 1 and "measured" in scout.stderr.lower()


def test_the_home_composer_names_z_image_and_leaves_video_closed():
    if not shutil.which("node"):
        pytest.skip("node is not installed")
    harness = r"""
const vm = require("vm"), fs = require("fs");
globalThis.el = (tag, attrs = {}, ...kids) => ({ tag, attrs, kids: kids.flat() });
globalThis.mark = (id, label) => label;
function text(node) {
  if (node == null || node === false) return "";
  if (typeof node !== "object") return String(node);
  if (Array.isArray(node)) return node.map(text).join(" ");
  return text(node.kids);
}
globalThis.text = text;
globalThis.routeBlock = kind => ({ scout: "closed scout", video: "closed video", voice: "closed voice", music: "closed music", long: "closed long" }[kind] || "closed");
globalThis.S = { image_models: [{ id: "z-image", label: "Z-Image Turbo", max_count: 1 }], capabilities: { local_render: false, routes: {} } };
globalThis.prefs = () => ({ presets: { image_model: "z-image", mode: "picture", recipe: "quick", seconds: 5 } });
vm.runInThisContext(fs.readFileSync(process.argv[2], "utf8") + `
;globalThis.__out = { open: hcBlock("picture"), video: hcBlock("clip"), badge: globalThis.text(pictureBadge()) };
`);
console.log(JSON.stringify(globalThis.__out));
"""
    import tempfile
    handle = tempfile.NamedTemporaryFile("w", suffix=".js", delete=False)
    handle.write(harness)
    handle.close()
    try:
        done = subprocess.run(["node", handle.name, str(ROOT / "app/static/home.js")],
                              capture_output=True, text=True, timeout=30)
    finally:
        Path(handle.name).unlink(missing_ok=True)
    assert done.returncode == 0, done.stderr
    out = json.loads(done.stdout.strip().splitlines()[-1])
    assert out["open"] is None and out["video"] == "closed video" and out["badge"] == "Z-Image Turbo"


def test_the_picture_form_opens_a_measured_z_image_still_at_1024():
    if not shutil.which("node"):
        pytest.skip("node is not installed")
    page = (ROOT / "app/static/index.html").read_text()
    start, end = page.index("// measured-still:start"), page.index("// measured-still:end")
    harness = r"""
const vm = require("vm"), fs = require("fs");
globalThis.el = (tag, attrs = {}, ...kids) => ({ tag, attrs, kids: kids.flat() });
globalThis.mark = (id, label) => ({ mark: id, label });
globalThis.choice = (obj, key, options) => { globalThis.picked = options; return { options }; };
globalThis.render = () => {};
globalThis.routeBlock = kind => kind === "scout" ? "closed scout" : "closed";
globalThis.S = { image_models: [
  { id: "sdxl-turbo", label: "SDXL Turbo", width: 512, height: 512, steps: 1, max_count: 1 },
  { id: "z-image", label: "Z-Image Turbo", width: 1024, height: 1024, steps: 8, max_count: 1 },
  { id: "flux3k-fast", label: "FLUX3-K Fast", width: 1024, height: 1024, steps: 16, max_count: 4 }
]};
globalThis.C = { imageModel: "z-image", imageSize: "1344x768", count: 4 };
vm.runInThisContext(fs.readFileSync(process.argv[2], "utf8"));
const z = pictureStillFacts();
C.imageModel = "sdxl-turbo"; C.count = 4; C.imageSize = "1344x768";
const sdxl = pictureStillFacts();
C.imageModel = "qwen"; C.count = 4; C.imageSize = "1344x768";
const qwen = pictureStillFacts();
C.imageModel = "z-image";
imageModelPick();
console.log(JSON.stringify({ z, sdxl, qwen, titles: { z: picked["z-image"][0], sdxl: picked["sdxl-turbo"][0], flux: picked["flux3k-fast"][0] } }));
"""
    import tempfile
    page_js = tempfile.NamedTemporaryFile("w", suffix=".js", delete=False)
    page_js.write(page[start:end])
    page_js.close()
    handle = tempfile.NamedTemporaryFile("w", suffix=".js", delete=False)
    handle.write(harness)
    handle.close()
    try:
        done = subprocess.run(["node", handle.name, page_js.name], capture_output=True, text=True, timeout=30)
    finally:
        Path(handle.name).unlink(missing_ok=True)
        Path(page_js.name).unlink(missing_ok=True)
    assert done.returncode == 0, done.stderr
    out = json.loads(done.stdout.strip().splitlines()[-1])
    assert out["z"]["open"] is True and out["z"]["size"] == "1024x1024" and out["z"]["max"] == 1
    assert "1024 × 1024" in out["z"]["note"] and "8 steps" in out["z"]["note"] and "Z-Image Turbo" in out["z"]["note"]
    assert out["z"]["choices"]["1024x1024"][1] == "1024 × 1024, 8 steps"
    assert out["sdxl"]["open"] is True and out["sdxl"]["size"] == "512x512" and "one step" in out["sdxl"]["note"]
    assert out["qwen"]["open"] is False and out["qwen"]["reason"] == "closed scout" and out["qwen"]["size"] == "1344x768"
    assert out["titles"]["z"] == "Z-Image Turbo" and out["titles"]["sdxl"] == "SDXL Turbo"
    assert out["titles"]["flux"]["mark"] == "flux"
