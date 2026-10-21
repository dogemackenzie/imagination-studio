"""scripts/doctor.py stays a read-only inventory: python3 -m pytest tests/test_doctor.py.

No probe contacts the shared ComfyUI, Docker, or the GPU. The command-line smoke test skips those too.
"""
import ast
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("studio_doctor", ROOT / "scripts" / "doctor.py")
doctor = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules["studio_doctor"] = doctor
SPEC.loader.exec_module(doctor)


def test_every_subprocess_has_a_timeout_and_no_shell():
    tree = ast.parse((ROOT / "scripts" / "doctor.py").read_text())
    missing = []
    shells = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = ast.unparse(node.func)
        if name not in {"subprocess.run", "subprocess.check_output", "subprocess.Popen"}:
            continue
        if not any(keyword.arg == "timeout" for keyword in node.keywords):
            missing.append(name)
        for keyword in node.keywords:
            if keyword.arg == "shell" and isinstance(keyword.value, ast.Constant) and keyword.value.value is True:
                shells.append(name)
    assert missing == []
    assert shells == []


def test_http_method_is_get_only():
    tree = ast.parse((ROOT / "scripts" / "doctor.py").read_text())
    methods = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg == "method" and isinstance(keyword.value, ast.Constant):
                methods.append(keyword.value.value)
    assert methods == ["GET"]


def test_allowed_comfy_paths_are_the_inventory_routes():
    assert doctor.allowed_comfy_path("/system_stats")
    assert doctor.allowed_comfy_path("/queue")
    for name in doctor.NODE_NAMES:
        assert doctor.allowed_comfy_path("/object_info/" + name)
    for blocked in (
        "/prompt", "/interrupt", "/free", "/history", "/upload/image", "/queue/delete",
        "/object_info/NotANode", "/object_info/CheckpointLoaderSimple/extra", "/system_stats/../prompt",
    ):
        assert doctor.allowed_comfy_path(blocked) is False


def test_comfy_url_stays_on_loopback():
    assert doctor.check_loopback("http://127.0.0.1:8188") == "http://127.0.0.1:8188"
    assert doctor.check_loopback("http://127.0.0.1:8188/") == "http://127.0.0.1:8188"
    for blocked in (
        "http://172.17.0.1:8188",
        "http://localhost:8188",
        "https://127.0.0.1:8188",
        "http://127.0.0.1:8188/system_stats",
        "http://user:secret@127.0.0.1:8188",
        "http://127.0.0.1:8188?status=1",
    ):
        with pytest.raises(doctor.ProbeError):
            doctor.check_loopback(blocked)


def test_redirects_are_refused():
    with pytest.raises(doctor.RedirectRefused):
        doctor._NoRedirect().redirect_request(None, None, 302, "found", {}, "http://example.com")


def test_queue_counts_drop_prompt_text():
    payload = {
        "queue_running": [[0, "id", {"prompt": "secret story about a person"}, {}]],
        "queue_pending": [{"prompt": "another secret"}],
    }
    counts = doctor.queue_counts(payload)
    assert counts == {"running": 1, "pending": 1}
    assert "secret" not in json.dumps(counts)


def test_system_summary_keeps_versions_and_drops_argv():
    summary = doctor.system_summary({
        "system": {"comfyui_version": "0.40.0", "argv": ["--token", "secret"], "os": "linux", "python_version": "3.12.3"},
        "devices": [{"name": "RTX", "type": "cuda", "index": 0, "vram_total": 1, "vram_free": 1, "extra": "secret"}],
    })
    assert summary["system"]["comfyui_version"] == "0.40.0"
    assert "argv" not in summary["system"]
    assert "extra" not in summary["devices"][0]
    assert "secret" not in json.dumps(summary)


def test_parse_proc_tcp_loopback_ports():
    sample = (
        "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode\n"
        "   0: 0100007F:1FFC 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1000        0 12345 "
        "1 0000000000000000 100 0 0 10 0\n"
        "   1: 0100007F:1F90 00000000:0000 01 00000000:00000000 00:00000000 00000000  1000        0 99 "
        "1 0000000000000000 100 0 0 10 0\n"
    )
    found = doctor.parse_proc_net(sample, ipv6=False)
    assert found == [{"address": "127.0.0.1", "port": 8188, "inode": "12345", "uid": 1000}]
    address, port = doctor.parse_proc_address("00000000000000000000000001000000:1F90", ipv6=True)
    assert address == "::1"
    assert port == 8080


def test_listener_summary_reports_a_free_studio_port():
    found = [{
        "address": "127.0.0.1", "port": 8188, "inode": "1",
        "owners": [{"pid": 9, "comm": "python", "cmdline": "python main.py"}],
    }]
    summary = {row["port"]: row for row in doctor.summarize_listeners(found)}
    assert summary[8188]["listening"] is True
    assert summary[8188]["owners"][0]["comm"] == "python"
    assert summary[7870]["listening"] is False
    assert summary[8191]["listening"] is False


def test_selected_model_must_stay_inside_the_models_directory(tmp_path):
    models = tmp_path / "models"
    good = models / "checkpoints" / "a.safetensors"
    good.parent.mkdir(parents=True)
    good.write_bytes(b"abc")
    outside = tmp_path / "secret.safetensors"
    outside.write_bytes(b"nope")
    assert doctor.resolve_model(models, "checkpoints/a.safetensors") == good.resolve()
    with pytest.raises(doctor.ProbeError):
        doctor.resolve_model(models, str(outside))
    with pytest.raises(doctor.ProbeError):
        doctor.resolve_model(models, "../secret.safetensors")
    digest = doctor.hash_file(good, timeout=5)
    assert digest["ok"] is True
    assert digest["hex"] == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_a_missing_selection_is_recorded_without_raising(tmp_path):
    report = doctor.inspect_models(tmp_path, "checkpoints/missing.safetensors", hash_timeout=5, list_timeout=5)
    assert report["present"] is True
    assert report["selected"]["checksum"]["ok"] is False


def test_floor_matches_the_spark_guard_rule(monkeypatch):
    monkeypatch.delenv("SPARK_MEMORY_FLOOR_MB", raising=False)
    assert doctor.reported_floor(16000)["floor_mb"] == 8000
    assert doctor.reported_floor(200000)["floor_mb"] == 20000
    monkeypatch.setenv("SPARK_MEMORY_FLOOR_MB", "9000")
    assert doctor.reported_floor(16000)["floor_mb"] == 9000


def test_image_choice_prefers_a_small_local_image_and_otherwise_cuda_base():
    assert doctor.choose_image({"studio:0.1.0", "ubuntu:24.04"}) == ("ubuntu:24.04", False)
    chosen, pull = doctor.choose_image({
        "studio:0.1.0",
        "nvidia/cuda:12.6.3-devel-ubuntu24.04",
        "nvidia/cuda:12.6.3-base-ubuntu24.04",
    })
    assert chosen == "nvidia/cuda:12.6.3-base-ubuntu24.04"
    assert pull is False
    assert doctor.choose_image({"studio:0.1.0"}) == ("ubuntu:24.04", True)


def test_gpu_container_is_disposable_and_offline(monkeypatch):
    seen = []

    def fake(argv, timeout):
        seen.append(argv)
        if argv[1] == "images":
            return {"ok": True, "code": 0, "stdout": "ubuntu:24.04\n", "stderr": "", "error": None}
        if argv[1] == "run":
            return {"ok": True, "code": 0, "stdout": "GPU-abc, NVIDIA GeForce RTX 3060, 12288\n", "stderr": "", "error": None}
        raise AssertionError(argv)

    monkeypatch.setattr(doctor, "run_cmd", fake)
    result = doctor.gpu_passthrough("GPU-abc-def", timeout=5, pull_timeout=5, run_timeout=5)
    run = next(argv for argv in seen if argv[1] == "run")
    assert run[:3] == ["docker", "run", "--rm"]
    assert run[run.index("--network") + 1] == "none"
    assert "--read-only" in run
    assert "--privileged" not in run
    assert "-v" not in run
    assert "--mount" not in run
    assert "--pid" not in run
    joined = " ".join(run)
    assert "8188" not in joined and "ComfyUI" not in joined
    assert "device=GPU-abc-def" in run
    assert result["ok"] is True
    assert result["pulled"] is False


def test_gpu_container_pulls_only_the_small_fallback(monkeypatch):
    seen = []

    def fake(argv, timeout):
        seen.append(argv)
        if argv[1] == "images":
            return {"ok": True, "code": 0, "stdout": "studio:0.1.0\n", "stderr": "", "error": None}
        if argv[1] == "pull":
            return {"ok": True, "code": 0, "stdout": "pulled\n", "stderr": "", "error": None}
        if argv[1] == "run":
            return {"ok": True, "code": 0, "stdout": "ok\n", "stderr": "", "error": None}
        raise AssertionError(argv)

    monkeypatch.setattr(doctor, "run_cmd", fake)
    result = doctor.gpu_passthrough("GPU-abc", timeout=5, pull_timeout=5, run_timeout=5)
    assert seen[1] == ["docker", "pull", "ubuntu:24.04"]
    assert result["pulled"] is True


def test_published_ports_keep_only_watched_listeners():
    text = "\n".join([
        "eager_curie\tmcr.microsoft.com/playwright/mcp\t",
        "studio-studio-1\t03fd36151c01\t192.168.50.22:8190->8190/tcp, 192.168.50.22:8443->8443/tcp",
    ])
    matches = doctor.parse_published_ports(text)
    assert matches == [{
        "name": "studio-studio-1",
        "image": "03fd36151c01",
        "ports": "192.168.50.22:8190->8190/tcp, 192.168.50.22:8443->8443/tcp",
        "watched_ports": [8190],
    }]


def test_docker_runtime_names_include_nvidia(monkeypatch):
    def fake(argv, timeout):
        if argv[1] == "version":
            return {"ok": True, "code": 0, "stdout": "29.9.0\n", "stderr": "", "error": None}
        if argv[1] == "compose":
            return {"ok": True, "code": 0, "stdout": "5.6.0\n", "stderr": "", "error": None}
        if argv[1] == "info":
            assert "json" not in argv[-1]
            return {"ok": True, "code": 0, "stdout": "runc\nnvidia\n", "stderr": "", "error": None}
        raise AssertionError(argv)

    monkeypatch.setattr(doctor, "run_cmd", fake)
    report = doctor.read_docker(5)
    assert report["nvidia_runtime"] is True
    assert report["runtimes"] == ["nvidia", "runc"]


def test_requirements_never_enable_local_rendering():
    report = {
        "tools": {"python3": {"present": True}, "docker": {"present": True}, "ffmpeg": {"present": True},
                  "ffprobe": {"present": True}},
        "docker": {"ok": True, "skipped": False},
        "ports": [{"port": 7870, "listening": False}, {"port": 8191, "listening": False}],
        "resources": {"available_below_floor": False, "mem_available_mb": 32000, "floor_mb": 8000},
        "comfy": {"reachable": False, "skipped": True},
        "repo": {"comfy_pin": "abc"},
        "gpu": {"skipped": True, "gpus": [{"name": "RTX"}]},
        "gpu_container": {"attempted": True, "ok": True},
    }
    ready = doctor.requirements(report)
    assert ready["local_render_enabled"] is False
    assert ready["app_only_blockers"] == []


def test_cli_json_skips_comfy_docker_gpu_and_models():
    result = subprocess.run(
        [sys.executable, "scripts/doctor.py", "--json", "--skip-comfy", "--skip-docker", "--skip-gpu", "--skip-models"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["comfy"]["skipped"] is True
    assert report["docker"]["skipped"] is True
    assert report["gpu"]["skipped"] is True
    assert report["models"]["skipped"] is True
    assert report["gpu_container"]["attempted"] is False
    assert report["requirements"]["local_render_enabled"] is False
    assert report["host"]["machine"]
    assert "secret" not in result.stdout
