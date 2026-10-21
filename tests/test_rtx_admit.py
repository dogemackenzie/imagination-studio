"""RTX admission and renderer ownership (spark/rtx_admit.py).

No shared ComfyUI, no model, no network. Process checks stop a sleep this test started.
"""
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "spark"))
sys.path.insert(0, str(ROOT / "app"))
import memory_guard  # noqa: E402
import rtx_admit as A  # noqa: E402


def budgets(**over):
    sample = {
        "host_available_mb": 16000,
        "host_total_mb": 16000,
        "cgroup": {"known": True, "cap": False, "headroom_mb": None, "pressure_full_avg10": 0.0},
        "vram_free_mib": 8000,
        "vram_known": True,
        "need_host_mb": None,
        "need_vram_mib": None,
        "profile": False,
        "occupied_port": False,
        "protected": False,
        "foreign_lock": False,
        "busy": False,
        "vram_reserve_mib": 512,
    }
    sample.update(over)
    return sample


def test_spark_measurements_stay_on_their_own_table():
    assert memory_guard.ESTIMATES == {"int8": (54_500, 215), "int8-lora": (54_500, 215), "bf16": (75_000, 47)}
    assert memory_guard.MEASURED_ROWS["bf16"] == 34_960
    assert memory_guard.MEASURED_ROWS["int8-lora"] == 75_336


def test_the_host_floor_cannot_be_lowered(monkeypatch):
    monkeypatch.setenv("SPARK_MEMORY_FLOOR_MB", "1000")
    assert A.host_floor_mb(16_000) == 8000
    monkeypatch.setenv("SPARK_MEMORY_FLOOR_MB", "30000")
    assert A.host_floor_mb(16_000) == 30000


def test_low_host_memory_is_refused():
    found = A.decide(budgets(host_available_mb=2000))
    assert found["ok"] is False and found["code"] == "insufficient_memory"


def test_low_cgroup_headroom_is_refused():
    cgroup = {"known": True, "cap": True, "headroom_mb": 10, "pressure_full_avg10": 0.0}
    found = A.decide(budgets(cgroup=cgroup, need_host_mb=50, profile=True))
    assert found["code"] == "insufficient_cgroup"
    pressed = {"known": True, "cap": True, "headroom_mb": 4000, "pressure_full_avg10": 9.0}
    assert A.decide(budgets(cgroup=pressed, profile=True))["code"] == "insufficient_cgroup"


def test_low_vram_is_refused_without_adding_host_ram():
    assert A.decide(budgets(vram_free_mib=100))["code"] == "insufficient_vram"
    assert A.decide(budgets(host_available_mb=2000, vram_free_mib=20_000))["code"] == "insufficient_memory"


def test_a_missing_reading_is_not_free_capacity():
    assert A.decide(budgets(vram_known=False, vram_free_mib=None))["code"] == "missing_telemetry"
    assert A.decide(budgets(host_available_mb=None))["code"] == "missing_telemetry"
    unknown = {"known": False, "cap": False, "headroom_mb": None, "pressure_full_avg10": None}
    assert A.decide(budgets(cgroup=unknown))["code"] == "missing_telemetry"


def test_occupied_port_foreign_lock_and_a_busy_slot_refuse():
    assert A.decide(budgets(occupied_port=True))["code"] == "occupied_port"
    assert A.decide(budgets(protected=True))["code"] == "protected_endpoint"
    assert A.decide(budgets(foreign_lock=True))["code"] == "foreign_lock"
    assert A.decide(budgets(busy=True, profile=True))["code"] == "busy_renderer"


def test_a_passing_budget_still_has_no_measured_image_route(monkeypatch):
    monkeypatch.delenv("SPARK_MEMORY_FLOOR_MB", raising=False)
    monkeypatch.delenv("RTX_VRAM_RESERVE_MIB", raising=False)
    found = A.decide(budgets())
    assert found["ok"] is False and found["code"] == "unmeasured_configuration"
    assert A.decide(budgets(profile=True))["ok"] is True


def test_collect_does_not_invent_a_profile(monkeypatch):
    monkeypatch.setattr(A, "_host_sample", lambda: (16_000, 16_000))
    monkeypatch.setattr(A, "_self_cgroup", lambda: budgets()["cgroup"])
    monkeypatch.setattr(A, "_vram_probe", lambda: 8000)
    monkeypatch.setattr(A, "slot_busy", lambda: False)
    found = A.decide(A.collect())
    assert found["code"] == "unmeasured_configuration"


def test_cgroup_parents_and_an_unreadable_limit(tmp_path):
    root = tmp_path / "cg"
    leaf = root / "a" / "b"
    leaf.mkdir(parents=True)
    pressure = "some avg10=0.00 avg60=0.00 avg300=0.00 total=0\nfull avg10=0.00 avg60=0.00 avg300=0.00 total=0\n"
    (root / "memory.max").write_text("max\n")
    parent = root / "a"
    (parent / "memory.max").write_text(str(200 * 1024 * 1024))
    (parent / "memory.current").write_text(str(180 * 1024 * 1024))
    (parent / "memory.pressure").write_text(pressure)
    (leaf / "memory.max").write_text(str(100 * 1024 * 1024))
    (leaf / "memory.current").write_text(str(10 * 1024 * 1024))
    (leaf / "memory.pressure").write_text(pressure)
    found = A.read_cgroup(leaf, root, reserve_mb=0)
    assert found["known"] is True and found["headroom_mb"] == 20
    (leaf / "memory.max").unlink()
    assert A.read_cgroup(leaf, root, reserve_mb=0)["known"] is False


def test_a_root_without_memory_max_is_no_cap_and_a_gap_below_it_is_unknown(tmp_path):
    """This host's /sys/fs/cgroup has no memory.max. Every controller level above it is "max"."""

    root = tmp_path / "sys"
    leaf = root / "user.slice" / "user-1000.slice" / "session.scope"
    leaf.mkdir(parents=True)
    for directory in (leaf, leaf.parent, leaf.parent.parent):
        (directory / "memory.max").write_text("max\n")
    assert not (root / "memory.max").exists()
    found = A.read_cgroup(leaf, root, reserve_mb=0)
    assert found == {"known": True, "cap": False, "headroom_mb": None, "pressure_full_avg10": 0.0}
    pressure = "some avg10=0.00 avg60=0.00 avg300=0.00 total=0\nfull avg10=1.00 avg60=0.00 avg300=0.00 total=0\n"
    parent = leaf.parent
    (parent / "memory.max").write_text(str(200 * 1024 * 1024))
    (parent / "memory.current").write_text(str(180 * 1024 * 1024))
    (parent / "memory.pressure").write_text(pressure)
    capped = A.read_cgroup(leaf, root, reserve_mb=0)
    assert capped["known"] is True and capped["cap"] is True and capped["headroom_mb"] == 20
    (parent / "memory.max").unlink()
    assert A.read_cgroup(leaf, root, reserve_mb=0)["known"] is False


def test_vram_cache_is_bounded():
    calls = {"n": 0}

    def probe():
        calls["n"] += 1
        return 100 if calls["n"] < 3 else None

    clock = {"t": 0.0}
    cache = A.VramCache(probe, ttl_s=1.0, now=lambda: clock["t"])
    assert cache.read() == 100 and cache.read() == 100 and calls["n"] == 1
    clock["t"] = 2.0
    assert cache.read() == 100 and calls["n"] == 2
    clock["t"] = 4.0
    assert cache.read() is None


def test_the_plain_loop_and_the_python_guard_stop_only_the_owned_process():
    owned = subprocess.Popen(["sleep", "30"])
    other = subprocess.Popen(["sleep", "30"])
    stopped = []

    def stop(pid):
        assert pid == owned.pid and pid != other.pid
        stopped.append(pid)
        os.kill(pid, signal.SIGKILL)

    try:
        sample = budgets(host_available_mb=1000, profile=True)
        found = A.plain_watch(owned.pid, [sample], stop)
        assert found["stopped"] == owned.pid and stopped == [owned.pid]
        owned.wait(timeout=2)
        assert other.poll() is None
        other_stop = []
        second = subprocess.Popen(["sleep", "30"])
        try:
            def stop_second(pid):
                assert pid == second.pid
                other_stop.append(pid)
                os.kill(pid, signal.SIGKILL)

            A.watch_owned(second.pid, lambda: budgets(vram_free_mib=1, profile=True), stop_second, ticks=2)
            second.wait(timeout=2)
            assert other_stop == [second.pid] and other.poll() is None
        finally:
            if second.poll() is None:
                second.kill()
    finally:
        if owned.poll() is None:
            owned.kill()
        if other.poll() is None:
            other.kill()


def test_ownership_rejects_protected_and_foreign_endpoints(tmp_path, monkeypatch):
    monkeypatch.setenv("STUDIO_RENDER_OWNER", str(tmp_path / "owner"))
    seen = []

    def post(url):
        held = A.allow(url)
        if not held["ok"]:
            return held
        seen.append(url)
        return held

    assert post("http://127.0.0.1:8188")["code"] == "protected_endpoint"
    assert post("http://10.9.8.7:9000")["code"] == "not_owned"
    monkeypatch.setenv("STUDIO_PROTECTED_URLS", "http://10.9.8.7:9000")
    assert post("http://10.9.8.7:9000")["code"] == "protected_endpoint"
    assert A.claim("http://127.0.0.1:8188", os.getpid())["ok"] is False
    assert A.claim("http://127.0.0.1:4321", os.getpid())["ok"] is True
    assert post("http://127.0.0.1:4321")["ok"] is True
    assert seen == ["http://127.0.0.1:4321"]


def test_a_foreign_lock_is_not_signaled_and_a_dead_one_is_only_unlinked(tmp_path, monkeypatch):
    monkeypatch.setenv("STUDIO_RENDER_OWNER", str(tmp_path / "owner"))
    owned = subprocess.Popen(["sleep", "30"])
    try:
        who = A.read_identity(owned.pid)
        assert who["alive"] is True
        A._write_json(A.owner_path(), {"endpoint": "http://127.0.0.1:4321", "pid": owned.pid,
                                      "starttime": who["starttime"] + 1, "ns": who["ns"]})
        assert A.allow("http://127.0.0.1:4321")["code"] == "foreign_lock"
        assert A.recover()["code"] == "foreign_lock"
        assert owned.poll() is None and A.owner_path().is_file()
    finally:
        owned.kill()
        owned.wait(timeout=2)
    A._write_json(A.owner_path(), {"endpoint": "http://127.0.0.1:4321", "pid": 2**22, "starttime": 1, "ns": "gone"})
    if A.read_identity(2**22)["alive"]:
        pytest.skip("pid 4194304 is alive on this machine")
    assert A.recover()["cleared"] is True
    assert not A.owner_path().exists()


def test_one_live_slot_is_the_only_admitted_job(tmp_path, monkeypatch):
    monkeypatch.delenv("SPARK_MEMORY_FLOOR_MB", raising=False)
    monkeypatch.delenv("RTX_VRAM_RESERVE_MIB", raising=False)
    monkeypatch.setenv("STUDIO_RENDER_OWNER", str(tmp_path / "owner"))
    monkeypatch.setattr(A, "_host_sample", lambda: (16_000, 16_000))
    monkeypatch.setattr(A, "_self_cgroup", lambda: budgets()["cgroup"])
    monkeypatch.setattr(A, "_vram_probe", lambda: 8000)
    A.hold_slot(os.getpid())
    try:
        assert A.decide(A.collect())["code"] == "busy_renderer"
    finally:
        A.clear_slot(os.getpid())
    sample = A.collect()
    sample["profile"] = True
    assert A.decide(sample)["ok"] is True


def test_a_foreign_machine_does_not_open_the_renderer_url(monkeypatch):
    import spark_generate as sg

    monkeypatch.setenv("RTX_ADMIT_SPARK", "0")
    monkeypatch.setattr(A, "collect", lambda: budgets(host_available_mb=1000))
    monkeypatch.setattr(sg, "URL", "http://127.0.0.1:8188")

    def opened(*_args, **_kwargs):
        raise AssertionError("opened a renderer url")

    monkeypatch.setattr(sg.urllib.request, "urlopen", opened)
    with pytest.raises(SystemExit) as caught:
        sg.call("/system_stats")
    assert "8000" in str(caught.value)

    class Body:
        def read(self):
            return b"{}"

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setenv("RTX_ADMIT_SPARK", "1")
    monkeypatch.setattr(sg.urllib.request, "urlopen", lambda *_args, **_kwargs: Body())
    assert sg.call("/system_stats") == {}
