"""Every test runs with a studio folder of its own: a test that imports the server without naming STUDIO_HOME must
never use (or move, app/studio_home.py) the person's real ~/ImaginationStudio or ~/TensorFoldStudio.

Import and collection do not start the queue or a model scan. The scan delay is forced off here, before any test
module imports the app: a positive delay walks this machine, including another application's model files. ``begin``
runs once the app has been imported, including when a test imports it midway through the session.
"""
import os
import sys
import tempfile
import threading
import traceback
from pathlib import Path

import pytest

os.environ.setdefault("STUDIO_HOME", str(Path(tempfile.mkdtemp(prefix="studio-test-")) / "ImaginationStudio"))
os.environ["STUDIO_DISCOVER_DELAY"] = "-1"
# NVIDIA Nemotron 3.5 Bolt as the Studio's engine (app/bolt.py) is a stand-in in tests/test_bolt.py.
# Everywhere else the brains' local server is the one a test names.
os.environ.setdefault("BOLT_BACKEND", "external")
# The access key (app/access.py) stays off unless tests/test_access.py turns it on for its own requests.
os.environ.setdefault("STUDIO_ACCESS_KEY", "off")

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


@pytest.fixture
def spark_memory():
    """For a test that queues a job on Linux: the app takes Linux for a DGX Spark (128 GB) and its memory guard
    (spark/memory_guard.py) reads this machine's own /proc/meminfo, so a smaller Linux machine, such as a CI runner,
    refuses the job before the test's point. Skipped there; it runs on a Mac and on a Spark."""

    if not sys.platform.startswith("linux"):
        return
    try:
        total_kb = int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemTotal:")))
    except (OSError, StopIteration, ValueError, IndexError):
        return
    if total_kb < 100 * 1024 * 1024:
        pytest.skip(f"this Linux machine has {total_kb / 1024 / 1024:.0f} GB: the app takes it for a DGX Spark and its "
                    "memory guard refuses the job before the test's point (runs on a Mac and on a Spark)")


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
