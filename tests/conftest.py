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

    Production still has one ``SPARK`` flag. Only ``spark`` sets it. ``rtx`` names this machine's later profile and
    does not enable a render route, and ``none`` is a machine with no GPU. Admission tests stay on the real guard.
    """

    import server

    name = request.param
    if name not in {"mac", "spark", "rtx", "none"}:
        raise ValueError(name)
    monkeypatch.setattr(server, "SPARK", name == "spark")
    return name


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
