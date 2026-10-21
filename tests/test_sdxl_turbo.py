"""SDXL Turbo on this RTX machine (spark/sdxl_turbo.py).

Tests never open a socket to the shared ComfyUI. A measured profile in a test is a tiny stand-in
file. The real checkpoint is not read. An allowed scout is built and not queued.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "spark"))
sys.path.insert(0, str(ROOT / "app"))
import capabilities  # noqa: E402
import memory_guard  # noqa: E402
import nodes  # noqa: E402
import rtx_admit as A  # noqa: E402
import sdxl_turbo as S  # noqa: E402
import server  # noqa: E402
from fastapi import HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(server.app)
PROMPT = "a red door, exactly as written"


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


def recipe(**over):
    base = {"width": 512, "height": 512, "steps": 1, "cfg": 1.0, "sampler": "euler_ancestral",
            "scheduler": "sgm_uniform", "batch_size": 1, "max_count": 1}
    base.update(over)
    return base


@pytest.fixture
def measured(tmp_path, monkeypatch):
    """A profile the queue would accept. The file is four bytes, not the real checkpoint."""

    weight = tmp_path / "sd_xl_turbo_1.0_fp16.safetensors"
    weight.write_bytes(b"test")
    path = tmp_path / "profile.json"
    path.write_text(json.dumps({"enabled": True, "model": "sdxl-turbo", "recipe": recipe(),
                                 "checkpoint": {"bytes": 4}}))
    monkeypatch.setenv("SDXL_PROFILE", str(path))
    monkeypatch.setenv("SDXL_CHECKPOINT", str(weight))
    monkeypatch.setattr(S, "BYTES", 4)
    return path


class Args:
    def __init__(self, **kw):
        self.base = False
        self.negative = ""
        self.steps = None
        self.width = 512
        self.height = 512
        self.seeds = ""
        self.seed = 4
        self.output = "/tmp/studio-sdxl-not-written.png"
        self.prompt = PROMPT
        self.prompt_file = ""
        for key, value in kw.items():
            setattr(self, key, value)


def test_the_workflow_keeps_the_prompt_and_uses_cfg_one():
    graph = S.workflow(PROMPT, 7)
    kinds = {node["class_type"] for node in graph.values()}
    assert kinds == {"CheckpointLoaderSimple", "CLIPTextEncode", "EmptyLatentImage", "KSampler", "VAEDecode", "SaveImage"}
    assert graph["1"]["inputs"]["ckpt_name"] == S.CHECKPOINT
    assert graph["2"]["inputs"]["text"] == PROMPT
    assert graph["3"]["inputs"]["text"] == ""
    sampler = graph["5"]["inputs"]
    assert (sampler["steps"], sampler["cfg"], sampler["sampler_name"], sampler["scheduler"], sampler["seed"]) == (
        1, 1.0, "euler_ancestral", "sgm_uniform", 7)
    assert graph["4"]["inputs"] == {"width": 512, "height": 512, "batch_size": 1}
    assert "TextGenerate" not in json.dumps(graph) and "Lora" not in json.dumps(graph)
    with pytest.raises(ValueError):
        S.workflow(PROMPT, 1, 768, 768)


def test_the_shared_address_is_only_port_8188(monkeypatch):
    for bad in ("http://127.0.0.1:8190", "http://192.168.50.22:8188", "http://172.17.0.1:8190", "http://0.0.0.0:8188"):
        monkeypatch.setenv("SDXL_COMFY_URL", bad)
        with pytest.raises(RuntimeError):
            S.shared_origin()
    monkeypatch.setenv("SDXL_COMFY_URL", "http://172.17.0.1:8188")
    assert S.shared_origin() == "http://172.17.0.1:8188"
    monkeypatch.delenv("SDXL_COMFY_URL", raising=False)
    assert S.shared_origin() == "http://127.0.0.1:8188"


def test_a_container_reads_vram_from_the_shared_server(tmp_path, monkeypatch):
    weight = tmp_path / "w.safetensors"
    weight.write_bytes(b"x")
    monkeypatch.setenv("STUDIO_CONTAINER", "1")
    monkeypatch.setenv("SDXL_COMFY_URL", "http://172.17.0.1:8188")
    monkeypatch.setattr(S, "BYTES", 1)
    monkeypatch.setattr(S, "checkpoint_path", lambda: weight)
    monkeypatch.setattr(S.rtx_admit, "collect", lambda: budgets())

    def stats(free_mib):
        def exchange(method, path, payload=None, timeout=30):
            assert method == "GET" and path == "/system_stats"
            return {"devices": [{"type": "cuda", "vram_free": free_mib * 1024 * 1024}]}
        return exchange

    monkeypatch.setattr(S, "exchange", stats(12000))
    report = S.assess()
    assert report["ok"] is True and report["enabled"] is False
    assert report["vram_free_mib"] == 12000 and report["shared_url"] == "http://172.17.0.1:8188"
    assert report["cgroup"]["cap"] is False
    monkeypatch.setattr(S, "exchange", stats(100))
    assert S.assess()["code"] == "insufficient_vram"

    def down(method, path, payload=None, timeout=30):
        raise RuntimeError("did not answer")

    monkeypatch.setattr(S, "exchange", down)
    assert S.assess()["code"] == "renderer_down"


def test_a_still_only_posts_a_prompt_to_the_shared_server(monkeypatch):
    source = (ROOT / "spark/sdxl_turbo.py").read_text()
    assert "main.py --listen" not in source and "comfy.sh" not in source
    def boom(*args, **kwargs):
        raise AssertionError("a lifecycle call opened a socket")
    monkeypatch.setattr(S.urllib.request, "urlopen", boom)
    for method, path, payload in (("POST", "/interrupt", {}), ("POST", "/free", {}), ("DELETE", "/queue", None),
                                  ("POST", "/prompt", None), ("GET", "/interrupt", None)):
        with pytest.raises(RuntimeError):
            S.exchange(method, path, payload)


def test_the_model_stays_unoffered_without_a_matching_profile(tmp_path, monkeypatch):
    monkeypatch.setenv("SDXL_PROFILE", str(tmp_path / "missing.json"))
    monkeypatch.setenv("SDXL_CHECKPOINT", str(tmp_path / "missing.safetensors"))
    assert S.offered() is False
    assert S.script_block() and "measured profile" in S.script_block()
    weight = tmp_path / "w.safetensors"
    weight.write_bytes(b"test")
    monkeypatch.setenv("SDXL_CHECKPOINT", str(weight))
    monkeypatch.setattr(S, "BYTES", 4)
    path = tmp_path / "off.json"
    path.write_text(json.dumps({"enabled": False, "model": "sdxl-turbo", "recipe": recipe(),
                                 "checkpoint": {"bytes": 4}}))
    monkeypatch.setenv("SDXL_PROFILE", str(path))
    assert S.offered() is False
    path.write_text(json.dumps({"enabled": True, "model": "sdxl-turbo", "recipe": recipe(cfg=0.0),
                                 "checkpoint": {"bytes": 4}}))
    assert S.offered() is False


def test_a_clear_budget_does_not_enable_the_model(tmp_path, monkeypatch):
    weight = tmp_path / "w.safetensors"
    weight.write_bytes(b"x")
    monkeypatch.setattr(S, "BYTES", 1)
    monkeypatch.setattr(S, "checkpoint_path", lambda: weight)
    monkeypatch.setattr(S, "listener_pids", lambda: {4242})
    monkeypatch.setattr(S, "gpu_processes", lambda: [{"pid": 4242, "name": "python", "vram_mib": 250}])
    monkeypatch.setattr(S.rtx_admit, "collect", lambda: budgets())
    report = S.assess()
    assert report["ok"] is True and report["enabled"] is False and report["render"] is None
    assert report["shared_pids"] == [4242] and report["shared_url"] == "http://127.0.0.1:8188"
    monkeypatch.setattr(S, "gpu_processes", lambda: [
        {"pid": 4242, "name": "python", "vram_mib": 250},
        {"pid": os.getpid() + 1, "name": "python", "vram_mib": 9}])
    busy = S.assess()
    assert busy["ok"] is False and busy["enabled"] is False and busy["code"] == "busy_renderer"
    monkeypatch.setattr(S, "listener_pids", lambda: set())
    assert S.assess()["code"] == "renderer_down"
    monkeypatch.setattr(S, "listener_pids", lambda: {4242})
    monkeypatch.setattr(S, "gpu_processes", lambda: None)
    assert S.assess()["code"] == "missing_telemetry"
    monkeypatch.setattr(S, "listener_pids", lambda: {4242})
    monkeypatch.setattr(S, "gpu_processes", lambda: [{"pid": 4242, "name": "python", "vram_mib": 250}])
    monkeypatch.setattr(S.rtx_admit, "collect", lambda: budgets(host_available_mb=1000))
    low = S.assess()
    assert low["code"] == "insufficient_memory" and low["enabled"] is False
    assert memory_guard.ESTIMATES == {"int8": (54_500, 215), "int8-lora": (54_500, 215), "bf16": (75_000, 47)}


def test_the_queue_refuses_sdxl_until_a_profile_exists(tmp_path, monkeypatch):
    # The committed profile may be enabled. This test pins a missing one so it cannot enqueue a real still.
    monkeypatch.setenv("SDXL_PROFILE", str(tmp_path / "missing.json"))
    before = set(server.jobs)
    scout = client.post("/api/jobs", json={"kind": "scout", "params": {
        "prompt": PROMPT, "image_model": "sdxl-turbo", "count": 1, "width": 512, "height": 512}})
    assert scout.status_code == 400
    body = scout.json()
    assert body["code"] == "unmeasured_configuration" and "measured profile" in body["error"]
    plain = client.post("/api/jobs", json={"kind": "scout", "params": {"prompt": PROMPT, "image_model": "qwen"}})
    assert plain.status_code == 400 and plain.json()["error"] == capabilities.REASON_TEXT["unmeasured_configuration"]
    assert "Qwen" not in plain.json()["error"]
    video = client.post("/api/jobs", json={"kind": "video", "params": {
        "prompt": PROMPT, "image_model": "sdxl-turbo", "preset": "small", "seconds": 5}})
    assert video.status_code == 400 and "still picture" in video.json()["error"]
    edit = client.post("/api/jobs", json={"kind": "edit", "params": {"prompt": PROMPT, "image_model": "sdxl-turbo"}})
    assert edit.status_code == 400 and "still picture" in edit.json()["error"]
    generic = client.post("/api/jobs", json={"kind": "video", "params": {"prompt": PROMPT, "preset": "small", "seconds": 5}})
    assert generic.json()["error"] == capabilities.REASON_TEXT["unmeasured_configuration"]
    assert set(server.jobs) == before
    with pytest.raises(HTTPException) as caught:
        server.build("scout", {"prompt": PROMPT, "image_model": "sdxl-turbo", "count": 4}, "j-direct")
    assert "measured profile" in str(caught.value.detail)


def test_an_offered_still_is_512_and_does_not_open_video(measured):
    assert S.offered() is True
    assert server.image_models() == ["sdxl-turbo"]
    assert "qwen" not in server.image_models()
    had = server.SPARK
    try:
        server.SPARK = True
        assert "sdxl-turbo" not in server.image_models()
    finally:
        server.SPARK = had
    codes = [item["code"] for item in capabilities.reasons(False)]
    assert codes[0] == "unmeasured_configuration"
    assert capabilities.renders(False) is False
    state = client.get("/api/state").json()
    card = state["image_models"][0]
    assert card["id"] == "sdxl-turbo" and card["width"] == 512 and card["height"] == 512 and card["steps"] == 1
    assert state["capabilities"]["local_render"] is False
    assert state["capabilities"]["routes"]["scout"]["enabled"] is False
    assert state["capabilities"]["routes"]["video"]["enabled"] is False
    edit = client.get("/api/edit/state").json()
    assert all(model["id"] != "sdxl-turbo" for model in edit["models"])
    assert "sdxl-turbo" in nodes.IMAGE_MODELS and "sdxl-turbo" not in nodes.EDIT_MODELS
    before = set(server.jobs)
    for params in (
        {"prompt": PROMPT, "image_model": "sdxl-turbo", "width": 1280, "height": 736},
        {"prompt": PROMPT, "image_model": "sdxl-turbo", "count": 3, "width": 512, "height": 512},
        {"prompt": PROMPT, "image_model": "sdxl-turbo", "loras": ["x"]},
        {"prompt": PROMPT, "image_model": "sdxl-turbo", "negative": "blurry"},
        {"prompt": PROMPT, "image_model": "sdxl-turbo", "preset": "2k", "references": ["a.png"]},
    ):
        answer = client.post("/api/jobs", json={"kind": "scout", "params": params})
        assert answer.status_code == 400, params
        assert answer.json()["code"] == "unmeasured_configuration"
    assert set(server.jobs) == before
    command, env, outputs = server.build("scout", {
        "prompt": PROMPT, "count": 1, "seed": 3, "preset": "2k", "image_model": "sdxl-turbo"}, "j1")
    assert env["WIDTH"] == 512 and env["HEIGHT"] == 512 and env["IMAGE_MODEL"] == "sdxl-turbo"
    assert env["STUDIO_PYTHON"] == sys.executable
    assert command[1].endswith("scout.sh") and outputs[0].endswith("scout_s3.png")
    allowed = S.submission("scout", {"prompt": PROMPT, "image_model": "sdxl-turbo", "preset": "2k", "count": 1})
    assert allowed == {"ok": True}
    with pytest.raises(HTTPException) as caught:
        server.build("scout", {"prompt": PROMPT, "image_model": "sdxl-turbo", "width": 768, "height": 768}, "j2")
    assert "512x512" in str(caught.value.detail)


def _png(width: int, height: int) -> bytes:
    data = bytearray(b"\x89PNG\r\n\x1a\n" + b"\0" * 16)
    data[16:20] = width.to_bytes(4, "big")
    data[20:24] = height.to_bytes(4, "big")
    return bytes(data)


def test_render_images_posts_one_prompt_and_leaves_a_busy_queue_alone(tmp_path, monkeypatch):
    monkeypatch.setenv("RTX_SDXL_CALIBRATE", "1")
    monkeypatch.setattr(S, "assess", lambda: {"ok": True, "reason": ""})
    calls = []

    def exchange(method, path, payload=None, timeout=30):
        calls.append((method, path))
        if path == "/queue":
            return {"queue_running": [], "queue_pending": []}
        if path == "/prompt":
            assert method == "POST" and payload["prompt"]["2"]["inputs"]["text"] == PROMPT
            assert payload["prompt"]["5"]["inputs"]["seed"] == 4
            return {"prompt_id": "p1"}
        if path == "/history/p1":
            return {"p1": {"status": {"completed": True, "status_str": "success"},
                           "outputs": {"7": {"images": [{"filename": "studio-sdxl_00001_.png", "subfolder": "", "type": "output"}]}}}}
        if path.startswith("/view?"):
            return _png(512, 512)
        raise AssertionError(path)

    monkeypatch.setattr(S, "exchange", exchange)
    out = tmp_path / "out.png"
    made = S.render_images(Args(seed=4, output=str(out)), PROMPT, None, None, "http://127.0.0.1:8190")
    assert out.read_bytes().startswith(b"\x89PNG") and made[0]["seed"] == 4 and made[0]["url"] == S.SHARED_URL
    assert calls == [("GET", "/queue"), ("POST", "/prompt"), ("GET", "/history/p1"), ("GET", "/view?filename=studio-sdxl_00001_.png&subfolder=&type=output")]
    assert "/interrupt" not in [path for _, path in calls] and "/free" not in [path for _, path in calls]

    def busy(method, path, payload=None, timeout=30):
        calls.append((method, path))
        return {"queue_running": [[1, "other"]], "queue_pending": []}

    monkeypatch.setattr(S, "exchange", busy)
    with pytest.raises(SystemExit) as caught:
        S.render_images(Args(output=str(out)), PROMPT, None, None, "http://127.0.0.1:8188")
    assert "queue" in str(caught.value).lower()
    assert calls[-1] == ("GET", "/queue")


def test_the_measured_route_does_not_open_a_shared_port(monkeypatch):
    monkeypatch.setenv("RTX_ADMIT_SPARK", "0")
    monkeypatch.setattr(A, "collect", lambda: budgets())
    monkeypatch.delenv("RTX_MEASURED_ROUTE", raising=False)
    monkeypatch.delenv("RTX_SDXL_CALIBRATE", raising=False)
    monkeypatch.delenv("SPARK_COMFY_URL", raising=False)
    # Another test pins SPARK_COMFY_PORT at import. The default renderer port is still 8190.
    monkeypatch.setenv("SPARK_COMFY_PORT", "8190")
    assert "measured render route" in A.block_foreign_render()
    monkeypatch.setenv("RTX_MEASURED_ROUTE", "sdxl-turbo")
    monkeypatch.setenv("RTX_SDXL_CALIBRATE", "1")
    assert A.block_foreign_render() == A.REASONS["protected_endpoint"]
    monkeypatch.setenv("SPARK_COMFY_URL", "http://127.0.0.1:8191")
    assert A.block_foreign_render() is None
    for url in ("http://127.0.0.1:8188", "http://192.168.50.22:8190", "http://127.0.0.1:8190"):
        monkeypatch.setenv("SPARK_COMFY_URL", url)
        assert A.block_foreign_render() == A.REASONS["protected_endpoint"]
    monkeypatch.setenv("SPARK_COMFY_URL", "http://127.0.0.1:8191")
    monkeypatch.setattr(A, "collect", lambda: budgets(host_available_mb=1000))
    assert "8000" in A.block_foreign_render()
    monkeypatch.delenv("RTX_MEASURED_ROUTE")
    monkeypatch.setattr(A, "collect", lambda: budgets())
    assert "measured render route" in A.block_foreign_render()


def test_the_scripts_stop_before_a_render(tmp_path):
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "IMAGE_MODEL": "sdxl-turbo",
           "STUDIO_PYTHON": sys.executable, "PREFIX": str(tmp_path / "prefix")}
    image = subprocess.run(["bash", str(ROOT / "scripts/image.sh"), PROMPT, str(tmp_path / "out.png")],
                           capture_output=True, text=True, timeout=30, env=env)
    assert image.returncode == 1 and "measured" in image.stderr.lower()
    assert not (tmp_path / "out.png").exists()
    scout = subprocess.run(["bash", str(ROOT / "scripts/scout.sh"), PROMPT, "1", str(tmp_path / "scouts")],
                           capture_output=True, text=True, timeout=30, env=env)
    assert scout.returncode == 1 and "measured" in scout.stderr.lower()
    assert not (tmp_path / "scouts").exists()
    spark = subprocess.run(["bash", str(ROOT / "spark/image.sh"), PROMPT, str(tmp_path / "spark.png")],
                           capture_output=True, text=True, timeout=30, env=env)
    assert spark.returncode == 1 and "measured profile" in spark.stderr
    assert not (tmp_path / "spark.png").exists()
    script = (ROOT / "spark/image.sh").read_text()
    start = script.index('if [ "${IMAGE_MODEL:-qwen}" = "sdxl-turbo" ]; then')
    block = script[start:script.index("\nfi\n", start)]
    assert "comfy.sh" not in block and "spark_generate.py" in block
    assert "shared ComfyUI" in (ROOT / "scripts/image.sh").read_text()


def test_the_home_composer_names_sdxl_and_leaves_video_closed():
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
globalThis.S = { image_models: [], capabilities: { local_render: false, routes: {} } };
globalThis.prefs = () => ({ presets: { image_model: "sdxl-turbo", mode: "picture", recipe: "quick", seconds: 5 } });
vm.runInThisContext(fs.readFileSync(process.argv[2], "utf8") + `
;globalThis.__out = {};
__out.closed = hcBlock("picture");
__out.video = hcBlock("clip");
S.image_models = [{ id: "sdxl-turbo", label: "SDXL Turbo", max_count: 1 }];
__out.open = hcBlock("picture");
__out.stillVideo = hcBlock("clip");
__out.badge = globalThis.text(pictureBadge());
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
    assert out["closed"] == "closed scout" and out["video"] == "closed video"
    assert out["open"] is None and out["stillVideo"] == "closed video" and out["badge"] == "SDXL Turbo"
