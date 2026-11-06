"""Every test runs with a studio folder of its own: a test that imports the server without naming STUDIO_HOME must
never use (or move, app/studio_home.py) the person's real ~/ImaginationStudio or ~/TensorFoldStudio.

Import and collection do not start the queue or a model scan. The scan delay is forced off here, before any test
module imports the app: a positive delay walks this machine, including another application's model files. ``begin``
runs once the app has been imported, including when a test imports it midway through the session.
"""
import importlib.util
import os
import sys
import tempfile
import threading
import traceback
from pathlib import Path

import pytest

os.environ.setdefault("STUDIO_HOME", str(Path(tempfile.mkdtemp(prefix="studio-test-")) / "ImaginationStudio"))
os.environ.setdefault("PREFIX", tempfile.mkdtemp(prefix="studio-test-runtime-"))
os.environ.setdefault("STUDIO_WINDOWS_IMAGE_DIR", tempfile.mkdtemp(prefix="studio-test-images-"))
os.environ.setdefault("STUDIO_WINDOWS_SOUND_DIR", tempfile.mkdtemp(prefix="studio-test-sounds-"))
os.environ.setdefault("STUDIO_WINDOWS_FOLEY_DIR", tempfile.mkdtemp(prefix="studio-test-foley-"))
os.environ.setdefault("STUDIO_WINDOWS_FEATURE_DIR", tempfile.mkdtemp(prefix="studio-test-features-"))
os.environ.setdefault("GRAPHICS_DIR", tempfile.mkdtemp(prefix="studio-test-graphics-"))
# the native Windows runtime's own folder (app/windows_cuda.py root(), finish.py prefix()): read off Windows by the
# tests that set sys.platform to win32, so the Linux and Mac runs have one of their own too
os.environ.setdefault("LOCALAPPDATA", tempfile.mkdtemp(prefix="studio-test-localappdata-"))
# Import-time managers and per-test managers must never share the user's leases or RAM ledger.
_gpu_temp = Path(tempfile.mkdtemp(prefix="studio-test-gpu-"))
os.environ["STUDIO_GPU_RUNTIME_DIR"] = str(_gpu_temp / "runtime")
os.environ["STUDIO_GPU_RAM_DIR"] = str(_gpu_temp / "ram")
# These constants are read on the first server import, before individual test modules are collected.
# Never discover the person's existing ComfyUI on its real port.
os.environ.setdefault("SPARK_COMFY_PORT", "1")
# Hardware integration is intentional and opt-in; collecting the normal suite does not probe NVIDIA.
_spec = importlib.util.spec_from_file_location("suite_gpu", Path(__file__).with_name("suite_gpu.py"))
_suite_gpu = importlib.util.module_from_spec(_spec)
sys.modules["suite_gpu"] = _suite_gpu
_spec.loader.exec_module(_suite_gpu)
if os.environ.get("STUDIO_TEST_HARDWARE") == "1":
    _suite_gpu.pin_suite_gpu()
# nor walk the machine for models: the app starts that scan 20 s after it is imported (app/discover.py), and it would
# hold the scan's lock while tests/test_discover.py runs its own scans
os.environ["STUDIO_DISCOVER_DELAY"] = "-1"
# nor start a model server of this machine's: NVIDIA Nemotron 3.5 Bolt as the Studio's engine (app/bolt.py) is a stand-in
# in tests/test_bolt.py, and everywhere else the brains' local server is the one a test names
os.environ.setdefault("BOLT_BACKEND", "external")
# nor run on this machine's native Windows CUDA install (app/windows_cuda.py, an N1X laptop's cuda-tier.json): the suite
# tests the Windows preview tier as CI's Windows runner has it; tests/test_windows_cuda.py drives the native tier itself
os.environ.setdefault("STUDIO_WINDOWS_TIER", "preview")
# nor ask for the studio's access key (app/access.py): tests/test_access.py turns it on for its own requests
os.environ.setdefault("STUDIO_ACCESS_KEY", "off")
# An inherited shared-ComfyUI address must not open a socket during the suite. A test that
# reads it sets the variable itself. The default in sdxl_turbo is not used from telemetry.
os.environ.pop("SDXL_COMFY_URL", None)
# The measured Z-Image profile is enabled in the tree and the weights sit on this host.
# A missing file keeps the suite from offering that still or posting one. A test that
# offers it sets these itself.
os.environ["ZIMAGE_PROFILE"] = "/tmp/studio-test-zimage-missing.json"
os.environ["ZIMAGE_UNET"] = "/tmp/studio-test-zimage-missing-unet.safetensors"
os.environ["ZIMAGE_CLIP"] = "/tmp/studio-test-zimage-missing-clip.safetensors"
os.environ["ZIMAGE_VAE"] = "/tmp/studio-test-zimage-missing-vae.safetensors"

# A named RTX card, with memory above the floor. Tests that describe this card pin it, so a runner
# with no GPU still checks the same refusal. It is not a claim that the runner has this card.
RTX_OBS = {
    "os": "linux", "arch": "x86_64", "cpu": "", "product": "",
    "gpu": {"name": "NVIDIA GeForce RTX 3060", "family": "nvidia-rtx", "uuid": "", "memory_mib": 12288, "driver": ""},
    "memory_mb": {"total": 16000, "available": 15000},
}


@pytest.fixture
def named_rtx():
    """Observe a named RTX card for one test, then read the machine again."""

    import hardware
    hardware.use_observation(RTX_OBS)
    yield RTX_OBS
    hardware.clear()


def _server():
    return sys.modules.get("server")


@pytest.fixture(scope="session", autouse=True)
def studio_background():
    """Start the queue if collection already imported the app, and stop it after the last test."""

    server = _server()
    if server is not None:
        server.begin()
    yield
    server = _server()
    if server is not None:
        server.end()


@pytest.fixture(autouse=True)
def studio_queue_if_loaded():
    """A test that imports the app after session setup, or a lifespan shutdown, still has a queue."""

    server = _server()
    if server is not None:
        server.begin()
    yield


@pytest.fixture
def host(request, monkeypatch):
    """``mac``, ``spark``, ``rtx`` or ``none``.

    Only ``spark`` sets the ``SPARK`` flags. ``mac`` and ``spark`` open that machine's existing routes.
    ``rtx`` is managed-rtx and ``none`` is app-only: neither enables a render route, and neither fakes the GPU.
    """

    import capabilities
    import discover
    import music
    import server
    import telemetry
    import voice

    name = request.param
    if name not in {"mac", "spark", "rtx", "none"}:
        raise ValueError(name)
    spark = name == "spark"
    for mod in (server, voice, music, discover, telemetry):
        monkeypatch.setattr(mod, "SPARK", spark)
    capabilities.use({"mac": "mac", "spark": "spark", "rtx": "managed-rtx", "none": "app-only"}[name])
    try:
        yield name
    finally:
        capabilities.use(None)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import local_gpus

_real_inventory = local_gpus.inventory


def _no_inventory(**_kwargs):
    return {"supported": False, "devices": []}


# Install before collection: modules import server at module scope, before fixtures can run.
local_gpus.inventory = _no_inventory


@pytest.fixture
def gpu_launch_environment():
    """A Manager exports launch pins directly; restore those after all test monkeypatches unwind."""
    names = ("STUDIO_LOCAL_GPU_KEY", "STUDIO_LOCAL_GPU_UUID", "STUDIO_LOCAL_GPU_CHILD", "STUDIO_LOCAL_GPU_FD",
             "STUDIO_GPU_JOB_KEY", "STUDIO_GPU_MODEL_RAM", "STUDIO_GPU_MODEL_ROUTE", "STUDIO_GPU",
             "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER")
    saved = {name: os.environ.get(name) for name in names}
    yield
    for name, value in saved.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


@pytest.fixture(autouse=True)
def isolated_gpu_runtime(gpu_launch_environment, tmp_path, monkeypatch):
    monkeypatch.setenv("STUDIO_GPU_RUNTIME_DIR", str(tmp_path / "gpu-runtime"))
    monkeypatch.setenv("STUDIO_GPU_RAM_DIR", str(tmp_path / "gpu-ram"))
    monkeypatch.setattr(local_gpus, "inventory", _no_inventory)
    if os.environ.get("STUDIO_TEST_HARDWARE") != "1":
        import windows_runtime
        monkeypatch.setattr(windows_runtime, "physical_memory_mb", lambda: 65536)
        import windows_setup_fit
        monkeypatch.setattr(windows_setup_fit, "inventory_hardware", lambda: {
            "physical_total_mb": 65536, "host_total_mb": 32460,
            "gpu_total_mb": 31232, "n1x": True})
    server = sys.modules.get("server")
    if server is not None and os.environ.get("STUDIO_TEST_HARDWARE") != "1":
        monkeypatch.setattr(server, "GPU_NAME", ["GB10"])


@pytest.fixture
def real_inventory(isolated_gpu_runtime, monkeypatch):
    """Opt into inventory implementation; callers must supply their own fake subprocess."""
    monkeypatch.setattr(local_gpus, "inventory", _real_inventory)
    monkeypatch.setattr(local_gpus, "_CACHE", {"at": float("-inf"), "value": None})
    return _real_inventory

@pytest.fixture(autouse=True)
def fresh_imported_hub():
    """Each test reuses an inactive imported app, not a process that was permanently shut down.

    Actual shutdown tests use isolated Hubs/processes. A TestClient lifespan must not
    leave this shared test singleton quiesced for unrelated tests or module fixtures.
    """
    def reopen():
        hub = getattr(sys.modules.get("server"), "hub", None)
        if hub is not None and not hub.started and not hub.local.active:
            hub.closing = hub.local.closing = False
    reopen()
    yield
    reopen()



@pytest.fixture
def spark_memory(monkeypatch):
    """A virtual 128 GB Spark for contract tests, independent of real models or the CI runner's RAM."""
    if not sys.platform.startswith("linux"):
        return
    import memory_guard
    monkeypatch.setattr(memory_guard, "meminfo", lambda: {"MemTotal": 124546, "MemAvailable": 118700})
    monkeypatch.setattr(memory_guard, "device", lambda **_: None)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """On a failure, attach the threads still running so a hang leaves something to read."""

    outcome = yield
    report = outcome.get_result()
    if not report.failed:
        return
    names = {}
    lines = []
    for thread in threading.enumerate():
        names[thread.ident] = thread.name
        lines.append(f"{thread.name} daemon={thread.daemon} ident={thread.ident}")
    chunks = ["\n".join(lines)]
    for ident, frame in sys._current_frames().items():
        stack = "".join(traceback.format_stack(frame, limit=12))
        chunks.append(f"{names.get(ident, ident)}\n{stack}")
    report.sections.append(("background threads", "\n".join(chunks)[:8000]))
