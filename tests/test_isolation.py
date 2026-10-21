"""Collection and the app lifecycle stay off this machine's model store and off its memory guard.

The Spark tests used to replace ``server.memory_of`` while their module was imported, which disabled the guard for
every test collected after them. Only the spark profile sets the SPARK flag. mac and spark open that machine's
existing routes; rtx and none do not.
"""
import importlib.util
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
import server  # noqa: E402


@pytest.mark.parametrize("host", ["mac", "spark", "rtx", "none"], indirect=True)
def test_a_host_profile_only_marks_a_spark(host):
    assert server.SPARK is (host == "spark")


def test_a_frame_build_without_room_does_not_ask_the_memory_guard(monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("memory_of was called")

    monkeypatch.setattr(server, "memory_of", boom)
    preset = "small" if "small" in server.PRESETS else next(iter(server.PRESETS))
    quality = "fh8" if "fh8" in server.QUALITIES else next(iter(server.QUALITIES))
    _, env, _ = server.build("video", {"prompt": "a harbour", "preset": preset, "quality": quality, "frames": 209}, "iso", room=False)
    assert env["FRAMES"] == 209


def test_importing_the_spark_tests_leaves_the_memory_guard_in_place():
    loaded = [module for module in sys.modules.values()
              if str(getattr(module, "__file__", "") or "").endswith("test_sparks.py")]
    if not loaded:
        path = Path(__file__).resolve().parent / "test_sparks.py"
        spec = importlib.util.spec_from_file_location("test_sparks_loaded_for_isolation", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    assert server.memory_of.__module__ == "server"
    assert server.memory_of.__name__ == "memory_of"


def test_the_queue_is_started_by_the_lifecycle():
    assert server._began is True
    assert server._worker_thread is not None and server._worker_thread.is_alive()


def test_a_negative_discover_delay_does_not_scan(monkeypatch):
    import discover

    monkeypatch.setattr(discover, "start_scan", lambda: (_ for _ in ()).throw(AssertionError("scan started")))
    assert os.environ["STUDIO_DISCOVER_DELAY"] == "-1"
    discover.cancel()
    discover.schedule(-1)
    assert discover._scheduled is None
    discover.schedule()  # reads the forced-off delay
    assert discover._scheduled is None
