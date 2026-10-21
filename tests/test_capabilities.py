"""Hardware and capability boundaries. A test that describes an RTX card pins that card. The live machine
is read only for what it actually reports. An RTX host is app-only: the GPU is named, and no generative
route is enabled.
"""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
import capabilities  # noqa: E402
import hardware  # noqa: E402
import server  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(server.app)


@pytest.fixture(autouse=True)
def reset_mode():
    hardware.clear()
    capabilities.use(None)
    yield
    hardware.clear()
    capabilities.use(None)


def test_this_machine_is_not_a_spark():
    obs = hardware.observe()
    assert obs["arch"] == "x86_64"
    assert hardware.is_spark() is False
    name = (obs["gpu"]["name"] or "").strip()
    if "RTX 3060" in name:
        assert obs["gpu"]["family"] == "nvidia-rtx"
    elif not name:
        assert obs["gpu"]["family"] == "none"
    else:
        assert obs["gpu"]["family"] == hardware.family_of(name, obs.get("product") or "")
    assert hardware.family_of("NVIDIA GB10") == "nvidia-spark"
    assert hardware.family_of("Something", "NVIDIA DGX Spark") == "nvidia-spark"
    assert hardware.family_of("NVIDIA GeForce RTX 3060") == "nvidia-rtx"
    assert hardware.family_of("NVIDIA Jetson AGX Orin") != "nvidia-spark"
    assert hardware.family_of("NVIDIA Jetson AGX Orin") == "other"
    assert set(capabilities.REASON_TEXT) == {
        "unavailable_hardware", "missing_runtime", "missing_model", "incompatible_nodes",
        "busy_renderer", "insufficient_memory", "unmeasured_configuration",
    }


def test_state_is_app_only_and_names_the_gpu(named_rtx):
    state = client.get("/api/state").json()
    assert state["platform"] == "linux"
    assert "RTX 3060" in state["platform_label"]
    assert "DGX SPARK" not in state["platform_label"]
    assert "APPLE SILICON" not in state["platform_label"]
    assert state["engine"] == "none"
    ids = [item["id"] if isinstance(item, dict) else item for item in state["image_models"]]
    assert "qwen" not in ids and set(ids) <= {"sdxl-turbo", "z-image"}
    assert state["turbo"] is False and state["x2"] is False
    caps = state["capabilities"]
    assert caps["mode"] == "app-only" and caps["local_render"] is False
    assert caps["gpu"]["family"] == "nvidia-rtx"
    assert caps["memory_mb"]["floor"] == 8000
    for kind in ("scout", "video", "studio", "long", "edit", "turnaround", "upscale", "mask", "remove",
                 "sfx", "foley", "voice", "music"):
        assert caps["routes"][kind]["enabled"] is False
        assert caps["routes"][kind]["code"] == "unmeasured_configuration"
    assert caps["routes"]["export"]["enabled"] is True
    assert caps["routes"]["soundtrack"]["enabled"] is True
    codes = [item["code"] for item in caps["reasons"]]
    assert codes[0] == "unmeasured_configuration"
    available = caps["memory_mb"]["available"]
    if available is not None and available < 8000:
        assert "insufficient_memory" in codes
    else:
        assert "insufficient_memory" not in codes
    assert state["voice"]["credit"] == "Built with Fish Audio" and state["voice"]["available"] is False
    assert state["voice"]["live"] is False and "defaults" in state["voice"]
    assert state["music"]["credit"] == "MiniMax-Music3" and state["music"]["available"] is False
    assert state["music"]["stream"] is False and "Chorus" in state["music"]["sections"]
    assert state["sparks"]["supported"] is server.SPARK


def test_a_direct_generative_post_is_refused_and_not_queued(named_rtx):
    before = set(server.jobs)
    video = client.post("/api/jobs", json={"kind": "video", "params": {"prompt": "a harbour at dawn", "preset": "small",
                                                                       "quality": "fh8", "seconds": 5}})
    assert video.status_code == 400
    body = video.json()
    assert isinstance(body["error"], str) and body["code"] == "unmeasured_configuration"
    assert "Qwen" not in body["error"] and "oneshot" not in body["error"]
    voice = client.post("/api/voice/takes", json={"text": "Hello from the studio."})
    assert voice.status_code == 400 and isinstance(voice.json()["error"], str)
    assert voice.json()["code"] == "unmeasured_configuration"
    song = client.post("/api/music/takes", json={"caption": "a warm song", "lyrics": "[Chorus]\nHello", "seconds": 30})
    assert song.status_code == 400 and song.json()["code"] == "unmeasured_configuration"
    # An export is not a generative route. A bad clip list is refused as an export, before it is queued,
    # on a machine with ffmpeg and on one without.
    exported = client.post("/api/jobs", json={"kind": "export", "params": {"clips": ["not-a-clip"]}})
    assert exported.status_code == 400
    assert exported.json().get("code") != "unmeasured_configuration"
    assert "clips" in exported.json()["error"]
    assert set(server.jobs) == before


def test_modes_cannot_invent_a_route(monkeypatch):
    capabilities.use("managed-rtx")
    assert capabilities.renders(True) is False
    assert capabilities.mode(True) == "managed-rtx"
    capabilities.use(None)
    monkeypatch.setenv("STUDIO_MODE", "spark")
    assert capabilities.mode(False) == "app-only"
    monkeypatch.delenv("STUDIO_MODE", raising=False)
    capabilities.use("spark")
    assert capabilities.refusal("video", False) is None
    assert capabilities.renders(False) is True
    capabilities.use("mac")
    assert capabilities.mode(False) == "mac" and capabilities.renders(False) is True
    capabilities.use(None)


def test_a_missing_gpu_and_a_free_machine_stay_closed():
    hardware.use_observation({
        "os": "linux", "arch": "x86_64", "cpu": "", "product": "",
        "gpu": {"name": "", "family": "none", "uuid": "", "memory_mib": None, "driver": ""},
        "memory_mb": {"total": 16000, "available": 3000},
    })
    codes = [item["code"] for item in capabilities.reasons(False)]
    assert codes == ["unavailable_hardware", "unmeasured_configuration", "insufficient_memory"]
    hardware.use_observation({
        "os": "linux", "arch": "x86_64", "cpu": "", "product": "Inspiron",
        "gpu": {"name": "NVIDIA GeForce RTX 3060", "family": "nvidia-rtx", "uuid": "GPU-x", "memory_mib": 12288, "driver": "1"},
        "memory_mb": {"total": 16000, "available": 15000},
    })
    codes = [item["code"] for item in capabilities.reasons(False)]
    assert codes == ["unmeasured_configuration"]
    assert capabilities.renders(False) is False
    assert capabilities.refusal("export", False) is None
    assert capabilities.refusal("picture", False)["code"] == "unmeasured_configuration"


def test_a_controller_does_not_open_a_local_render(named_rtx, capsys):
    capabilities.use("controller")
    assert capabilities.mode(False) == "controller"
    assert capabilities.renders(False) is False
    assert capabilities.refusal("video", False) is None
    for kind in ("scout", "studio", "long", "edit", "voice", "music"):
        assert capabilities.refusal(kind, False)["code"] == "unmeasured_configuration"
    assert capabilities.refusal("export", False) is None
    assert server.machine() == ("linux", "CONTROLLER")
    assert capabilities.script_exit("video") == 1
    assert "does not render H3" in capsys.readouterr().err
    assert capabilities.script_exit("export") == 0


def test_scripts_refuse_before_the_spark_handoff():
    # A new process reads this machine. A named GPU says the route is unmeasured. No GPU says none is available.
    items = capabilities.reasons(hardware.is_spark())
    assert items
    reason = items[0]["reason"]
    done = subprocess.run(["bash", str(ROOT / "scripts/require-route.sh"), "video"], capture_output=True, text=True, timeout=30)
    assert done.returncode == 1 and reason in done.stderr
    image = subprocess.run(["bash", str(ROOT / "scripts/image.sh"), "a lighthouse", "/tmp/studio-should-not-render.png"],
                           capture_output=True, text=True, timeout=30)
    assert image.returncode == 1 and reason in image.stderr
    assert "spark/image.sh" not in image.stderr and not Path("/tmp/studio-should-not-render.png").exists()


@pytest.mark.parametrize("host", ["spark", "mac", "rtx", "none"], indirect=True)
def test_host_profiles_keep_each_machines_routes(host):
    if host == "spark":
        assert "small" in server.PRESETS and "2k" not in server.PRESETS
        assert "fh8" in server.QUALITIES and "standard" not in server.QUALITIES
        assert capabilities.renders(server.SPARK) is True
        assert server.machine()[0] == "spark"
    elif host == "mac":
        assert "draft" in server.PRESETS and "2k" in server.PRESETS and "standard" in server.QUALITIES
        assert capabilities.mode(server.SPARK) == "mac"
        assert server.machine()[1].startswith("MAC")
    elif host == "rtx":
        assert capabilities.mode(server.SPARK) == "managed-rtx"
        assert capabilities.renders(server.SPARK) is False
        assert server.machine() == ("linux", "MANAGED RTX (UNMEASURED)")
    else:
        assert capabilities.mode(server.SPARK) == "app-only"
        assert capabilities.renders(server.SPARK) is False
        label = server.machine()[1]
        name = ((hardware.observe().get("gpu") or {}).get("name") or "").strip()
        assert (name.upper() in label) if name else label == "LINUX"
    assert server.SPARK is (host == "spark")
