"""What this process can see (app/constraints.py) and the Machine tab's reading of it.

A GPU name from the shared ComfyUI is a reading. It does not open a render route, and it is
not written into hardware.observe(). No test here posts a prompt or opens a socket.
"""
import ast
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
import server  # noqa: E402
import capabilities  # noqa: E402
import constraints  # noqa: E402
import hardware  # noqa: E402
import telemetry  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(server.app)
_fetch = constraints.fetch_json

STATS = {
    "devices": [{
        "name": "cuda:0 NVIDIA GeForce RTX 3060 : cudaMallocAsync",
        "type": "cuda", "index": 0,
        "vram_total": 12489916416, "vram_free": 12371492864,
        "torch_vram_total": 0, "torch_vram_free": 0,
    }],
}
FILES = {
    "checkpoints": [
        "sd_xl_turbo_1.0_fp16.safetensors",
        "sd_xl_base_1.0.safetensors",
        "sd_xl_refiner_1.0.safetensors",
        "stable-audio-open-1.0.safetensors",
        "yue2.safetensors",
        "hunyuan_3d.safetensors",
        "sdpose.safetensors",
        "sam3.1.safetensors",
        "notes.txt",
    ],
    "diffusion_models": [
        "flux-2-klein-4b.safetensors",
        "wan2.1_t2v.safetensors",
        "z_image.safetensors",
        "seedvr2.safetensors",
        "acestep.safetensors",
        "anima.safetensors",
        "mage_flow.safetensors",
        "pixeldit.safetensors",
        "rt_detr.safetensors",
        "h3_transformer.safetensors",
    ],
    "vae": ["qwen_image_vae.safetensors", "ae.safetensors"],
}


def _no_socket(*_args, **_kwargs):
    raise AssertionError("constraints opened a socket")


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    """No inherited address, no nvidia-smi, and no socket. A test that needs one opts in."""

    capabilities.use(None)
    constraints.clear()
    monkeypatch.delenv("SDXL_COMFY_URL", raising=False)
    monkeypatch.delenv("STUDIO_MODE", raising=False)
    monkeypatch.delenv("STUDIO_CONTAINER", raising=False)
    monkeypatch.setattr(constraints, "smi_gpu", lambda: None)
    monkeypatch.setattr(constraints, "fetch_json", _no_socket)
    monkeypatch.setattr(hardware, "_memory_mb", lambda: {"total": 32000, "available": 16000})
    yield
    capabilities.use(None)
    constraints.clear()


def _row(body, section, ident):
    return next(row for row in body[section] if row["id"] == ident)


def test_the_card_name_is_the_middle_of_comfys_device_line():
    assert constraints.device_name("cuda:0 NVIDIA GeForce RTX 3060 : cudaMallocAsync") == "NVIDIA GeForce RTX 3060"
    assert constraints.device_name("NVIDIA GeForce RTX 3060") == "NVIDIA GeForce RTX 3060"
    assert constraints.device_name("") == ""


def test_vram_comes_from_the_device_and_the_sensors_stay_empty():
    one = constraints.parse_device(STATS)
    assert one["name"] == "NVIDIA GeForce RTX 3060"
    assert one["source"] == "comfyui"
    assert one["util_pct"] is None and one["temp_c"] is None and one["power_w"] is None and one["clock_mhz"] is None
    assert one["vram_total_mb"] == 12489916416 // (1024 * 1024)
    assert one["vram_free_mb"] == 12371492864 // (1024 * 1024)
    assert one["vram_total_mb"] != 0  # torch_vram_* is 0 and is not the card
    assert "argv" not in one
    assert constraints.parse_device({"devices": [{"name": "cpu", "type": "cpu"}]}) is None
    assert constraints.parse_device(None) is None


def test_only_the_two_origins_and_three_folders_are_read(monkeypatch):
    assert constraints.reading_origin() is None
    for raw, ok in (
        ("http://127.0.0.1:8188", "http://127.0.0.1:8188"),
        ("http://172.17.0.1:8188/", "http://172.17.0.1:8188"),
        ("http://172.21.0.1:8188", None),
        ("http://127.0.0.1:8188/system_stats", None),
        ("https://127.0.0.1:8188", None),
    ):
        monkeypatch.setenv("SDXL_COMFY_URL", raw)
        assert constraints.reading_origin() == ok
    monkeypatch.delenv("SDXL_COMFY_URL", raising=False)
    assert constraints.allowed_path("/system_stats")
    for folder in ("checkpoints", "diffusion_models", "vae"):
        assert constraints.allowed_path("/models/" + folder)
    for path in ("/prompt", "/object_info", "/models/loras", "/models/checkpoints/x", "/models/../checkpoints"):
        assert not constraints.allowed_path(path)
    assert constraints._NoRedirect().redirect_request(None, None, 302, "", {}, "http://127.0.0.1:8188/prompt") is None


def test_a_disallowed_address_never_opens_a_socket(monkeypatch):
    monkeypatch.setattr(constraints, "fetch_json", _fetch)
    monkeypatch.setattr(constraints.urllib.request, "build_opener", _no_socket)
    assert constraints.fetch_json("http://172.21.0.1:8188", "/system_stats", 2) is None
    assert constraints.fetch_json("http://127.0.0.1:8188", "/prompt", 2) is None
    assert constraints.fetch_json("http://127.0.0.1:8188", "/models/loras", 2) is None
    assert constraints.fetch_json("http://127.0.0.1:8188", "/system_stats", 0) is None
    assert constraints.fetch_json("http://127.0.0.1:8188", "/system_stats", None) is None
    assert constraints.gpu_reading() is None
    assert constraints.shared_renderer() is None
    assert constraints.model_files() is None


def test_a_filename_is_not_a_route():
    assert constraints.match_file("diffusion_models", "flux-2-klein-4b.safetensors")[:2] == ("flux2-klein", "FLUX.2 Klein")
    assert constraints.match_file("diffusion_models", "flux-2-klein-4b.safetensors")[2] == ""
    assert constraints.match_file("diffusion_models", "flux3-image-klein.safetensors")[2] == "flux3k"
    assert constraints.match_file("vae", "qwen_image_vae.safetensors")[0] == "qwen-vae"
    assert constraints.match_file("diffusion_models", "h3_transformer.safetensors") is None
    assert constraints.match_file("diffusion_models", "qwen_image_2.1.safetensors")[2] == "qwen"
    assert constraints.match_file("checkpoints", "sd_xl_turbo_1.0_fp16.safetensors")[2] == "sdxl-turbo"
    families, flags = constraints.classify(FILES)
    known = {item["id"] for item in families}
    assert "flux2-klein" in known and "qwen-vae" not in known and "minimax-h3" not in known
    assert flags["qwen_vae"] is True and flags["qwen_file"] is False and flags["flux3_file"] is False
    assert flags["sdxl_file"] is True and flags["h3_file"] is False
    assert flags["other"] == 2  # notes.txt and h3_transformer; a bare h3 name is not MiniMax H3. The other vae is not counted.


def test_a_seen_card_does_not_open_a_local_render(monkeypatch):
    monkeypatch.setenv("SDXL_COMFY_URL", "http://127.0.0.1:8188")
    monkeypatch.setenv("STUDIO_MODE", "spark")  # ignored: this process is not a Spark
    monkeypatch.setattr(constraints, "system_stats", lambda force=False: STATS)
    monkeypatch.setattr(constraints, "model_files", lambda force=False: FILES)
    before = hardware.observe()
    body = constraints.report(False, ["sdxl-turbo"])
    assert body["local_render"] is False
    assert hardware.observe()["gpu"] == before["gpu"]
    assert "hardware.observe" not in (ROOT / "app/constraints.py").read_text()
    assert capabilities.FLOOR_MB == 8000
    assert "No GPU is available" in capabilities.REASON_TEXT["unavailable_hardware"]
    assert "No GPU is available" not in body["note"]
    assert "does not open a route" in body["note"]
    assert "172.17.0.1" not in str(body) and "http://" not in str(body)
    gpu = _row(body, "hardware", "gpu")
    assert gpu["state"] == "seen" and "RTX 3060" in gpu["detail"] and "Read from the shared ComfyUI" in gpu["detail"]
    assert "8000" in _row(body, "hardware", "memory")["detail"]
    sdxl = _row(body, "software", "sdxl-turbo")
    assert sdxl["state"] == "runs" and "512" in sdxl["detail"] and "8000" in sdxl["detail"]
    assert "Stability AI Non-Commercial Research Community License" in sdxl["detail"]
    assert _row(body, "software", "qwen")["state"] == "missing"
    assert "separate file" in _row(body, "software", "qwen")["detail"]
    assert _row(body, "software", "flux3k")["state"] == "missing"
    assert _row(body, "software", "h3")["state"] == "missing"
    assert _row(body, "software", "export")["state"] == "runs"
    assert _row(body, "software", "soundtrack")["state"] == "runs"
    present = {row["id"]: row for row in body["present"]}
    assert "flux2-klein" in present and "different model" in present["flux2-klein"]["detail"]
    assert "sdxl-turbo" not in present and "qwen-vae" not in present and "minimax-h3" not in present
    assert present["sdxl-base"]["state"] == "present"
    assert "qwen-image" not in body["listed"]["known"]


def test_a_failed_file_list_is_not_a_list_of_absences(monkeypatch):
    monkeypatch.setenv("SDXL_COMFY_URL", "http://172.17.0.1:8188")
    monkeypatch.setattr(constraints, "system_stats", lambda force=False: {})
    monkeypatch.setattr(constraints, "model_files", lambda force=False: None)
    body = constraints.report(False, ["sdxl-turbo"])
    assert body["local_render"] is False
    assert _row(body, "hardware", "comfy")["state"] == "down"
    assert "does not start" in _row(body, "hardware", "comfy")["detail"]
    sdxl = _row(body, "software", "sdxl-turbo")
    assert sdxl["state"] == "runs" and "does not start" in sdxl["detail"]
    for ident in ("qwen", "flux3k", "h3"):
        row = _row(body, "software", ident)
        assert row["state"] == "unknown"
        assert "not on this machine" not in row["detail"]
        assert "being absent" in row["detail"]
    assert body["present"] == []


def test_nvidia_smi_wins_when_this_process_can_run_it(monkeypatch):
    monkeypatch.setattr(constraints, "smi_gpu", lambda: {
        "name": "NVIDIA GeForce RTX 3060", "util_pct": 4.0, "temp_c": 41.0, "power_w": 20.0, "clock_mhz": 1800,
    })
    monkeypatch.setattr(constraints, "gpu_reading", lambda: {"name": "not this", "source": "comfyui", "vram_total_mb": 1, "vram_free_mb": 1})
    found = constraints.visible_gpu()
    assert found["util_pct"] == 4.0 and "source" not in found and "vram_total_mb" not in found
    body = constraints.report(False, [])
    detail = _row(body, "hardware", "gpu")["detail"]
    assert "load 4%" in detail and "41 °C" in detail and "Read from the shared ComfyUI" not in detail
    assert _row(body, "software", "sdxl-turbo")["state"] == "missing"


def test_a_mac_or_a_spark_still_renders_and_a_container_says_it_does_not_hold_the_card(monkeypatch):
    body = constraints.report(True, ["qwen"])
    assert body["local_render"] is True
    assert _row(body, "software", "qwen")["state"] == "runs"
    assert _row(body, "software", "h3")["state"] == "runs"
    monkeypatch.setenv("STUDIO_CONTAINER", "1")
    boxed = constraints.report(False, [])
    assert boxed["local_render"] is False
    assert "does not hold the GPU" in boxed["note"]
    assert "shared ComfyUI" in boxed["note"]


def test_the_endpoint_is_a_get_and_it_queues_nothing(monkeypatch, tmp_path):
    monkeypatch.setenv("SDXL_PROFILE", str(tmp_path / "missing.json"))
    monkeypatch.setenv("SDXL_CHECKPOINT", str(tmp_path / "missing.safetensors"))
    monkeypatch.setenv("SDXL_COMFY_URL", "http://127.0.0.1:8188")
    monkeypatch.setattr(constraints, "system_stats", lambda force=False: STATS)
    monkeypatch.setattr(constraints, "model_files", lambda force=False: FILES)
    before = set(server.jobs)
    answer = client.get("/api/constraints?fresh=1")
    assert answer.status_code == 200
    body = answer.json()
    assert set(server.jobs) == before
    assert body["local_render"] is False
    assert _row(body, "software", "sdxl-turbo")["state"] == "closed"  # the file is listed; the profile is not enabled
    route = next(one for one in server.app.routes if getattr(one, "path", "") == "/api/constraints")
    assert route.methods == {"GET"}


def test_telemetry_uses_the_shared_card_when_nvidia_smi_is_absent(monkeypatch):
    monkeypatch.setattr(telemetry.shutil, "which", lambda name, *_args, **_kwargs: None)
    monkeypatch.setenv("SDXL_COMFY_URL", "http://127.0.0.1:8188")
    monkeypatch.setattr(constraints, "system_stats", lambda force=False: STATS)
    monkeypatch.setattr(telemetry, "GPU_FRESH", telemetry.FRESH)
    monkeypatch.setattr(telemetry, "GPU_BUSY", telemetry.FRESH)
    time.sleep(telemetry.FRESH + 0.05)
    now = client.get("/api/telemetry").json()
    assert now["gpu"]["name"] == "NVIDIA GeForce RTX 3060"
    assert now["gpu"]["source"] == "comfyui"
    assert now["gpu"]["util_pct"] is None and now["gpu"]["temp_c"] is None and now["gpu"]["power_w"] is None
    assert now["gpu"]["vram_total_mb"] == 12489916416 // (1024 * 1024)
    assert "argv" not in now["gpu"]
    shared = now["renderer"]["shared"]
    assert shared == {
        "up": True, "port": 8188, "name": "NVIDIA GeForce RTX 3060",
        "vram_total_mb": now["gpu"]["vram_total_mb"], "vram_free_mb": now["gpu"]["vram_free_mb"],
    }
    # The studio renderer keeps its own port. The shared row is 8188, even when another test
    # has pointed SPARK_COMFY_PORT somewhere else after the app started.
    assert now["renderer"]["comfy"]["port"] != 8188
    assert now["renderer"]["shared"]["port"] == 8188


def test_the_snapshot_does_not_ask_the_shared_server_unless_its_address_is_set():
    now = client.get("/api/telemetry").json()
    assert "shared" not in now["renderer"]


def test_nothing_here_posts_or_runs_a_tool_without_a_limit():
    tree = ast.parse((ROOT / "app/constraints.py").read_text())
    requests, tools = [], []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = ast.unparse(node.func)
        if name.endswith("Request"):
            requests.append(node)
        if name.startswith("subprocess."):
            tools.append(node)
    assert requests and all(
        any(keyword.arg == "method" and isinstance(keyword.value, ast.Constant) and keyword.value.value == "GET"
            for keyword in node.keywords)
        for node in requests
    )
    assert tools and all(any(keyword.arg == "timeout" for keyword in node.keywords) for node in tools)
    text = (ROOT / "app/constraints.py").read_text()
    assert "/prompt" not in text and "/object_info" not in text
    assert "method=\"POST\"" not in text and "method='POST'" not in text
