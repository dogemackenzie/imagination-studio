"""Phase 7 operations: job records, one data directory, backup, and the shared-queue preflight.

python3 -m pytest tests/test_operations.py

The preflight tests do not open a socket to the shared ComfyUI. A server subprocess uses a
temporary data directory.
"""
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest
from fastapi import HTTPException

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "spark"))
sys.path.insert(0, str(ROOT / "app"))

import discover  # noqa: E402
import history  # noqa: E402
import housekeeping  # noqa: E402
import opsdata  # noqa: E402
import server  # noqa: E402


def _preflight():
    import importlib.util
    spec = importlib.util.spec_from_file_location("comfy_preflight", ROOT / "scripts" / "comfy_preflight.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _start_server(home: Path, port: int, log_path: Path):
    env = os.environ.copy()
    env.update(STUDIO_HOME=str(home), STUDIO_CONTAINER="1", STUDIO_DISCOVER_DELAY="-1",
               STUDIO_GPU_LOCK=str(home / "gpu.lock.d"), PORT=str(port), PYTHONDONTWRITEBYTECODE="1")
    env.pop("HOST", None)
    env.pop("STUDIO_BIND", None)
    env.pop("STUDIO_UPDATE_SESSION", None)
    env.pop("STUDIO_JOB_HISTORY", None)
    env.pop("STUDIO_DISK_RESERVE_BYTES", None)
    handle = log_path.open("w")
    proc = subprocess.Popen([sys.executable, str(ROOT / "app" / "server.py")], cwd=ROOT, env=env,
                            stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    return proc, handle


def _wait_health(port: int, proc: subprocess.Popen, log_path: Path, seconds: float = 40) -> dict:
    deadline = time.time() + seconds
    while time.time() < deadline:
        if proc.poll() is not None:
            break
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=1) as response:
                body = json.loads(response.read())
            if response.status == 200 and body.get("ok") is True:
                return body
        except (OSError, ValueError):
            time.sleep(0.2)
    raise AssertionError(log_path.read_text()[-2000:])


def test_an_interrupted_record_is_not_replayed(tmp_path, monkeypatch):
    monkeypatch.setenv("STUDIO_JOB_HISTORY", str(tmp_path / "job-history.json"))
    history.remember({"id": "gone", "kind": "scout", "status": "running", "title": "x" * 200,
                      "error": "", "params": {"prompt": "secret"}})
    history.remember({"id": "kept", "kind": "export", "status": "done", "title": "cut", "error": ""})
    changed = history.recover()
    assert [row["id"] for row in changed] == ["gone"]
    assert changed[0]["status"] == "interrupted"
    assert "not a success" in changed[0]["error"]
    assert "not run again" in changed[0]["error"]
    assert "gone" not in server.jobs and "kept" not in server.jobs
    body = history.load()
    assert body["queue"] is False
    assert "not the queue" in body["note"]
    rows = {row["id"]: row for row in body["jobs"]}
    assert rows["gone"]["status"] == "interrupted"
    assert rows["kept"]["status"] == "done"
    assert len(rows["gone"]["title"]) == 120
    assert "prompt" not in (tmp_path / "job-history.json").read_text()
    assert history.recover() == []
    again = {row["id"]: row["status"] for row in history.load()["jobs"]}
    assert again == {"gone": "interrupted", "kept": "done"}


def test_a_shutdown_record_is_cancelled(tmp_path, monkeypatch):
    monkeypatch.setenv("STUDIO_JOB_HISTORY", str(tmp_path / "job-history.json"))
    before = {key: (job.get("status"), job.get("error")) for key, job in server.jobs.items()}
    server.jobs["phase7"] = {"id": "phase7", "kind": "export", "status": "queued", "title": "cut", "error": None}
    try:
        server._cancel_owned_work()
        assert server.jobs["phase7"]["status"] == "cancelled"
        row = next(one for one in history.load()["jobs"] if one["id"] == "phase7")
        assert row["status"] == "cancelled"
        assert row["status"] != "done"
        assert "not a success" in row["error"]
        saved = history.load()
        assert saved["queue"] is False
    finally:
        server.jobs.pop("phase7", None)
        for key, (status, error) in before.items():
            if key in server.jobs:
                server.jobs[key]["status"] = status
                server.jobs[key]["error"] = error


def test_history_endpoint_is_not_the_queue(tmp_path, monkeypatch):
    monkeypatch.setenv("STUDIO_JOB_HISTORY", str(tmp_path / "job-history.json"))
    history.remember({"id": "row", "kind": "export", "status": "failed", "title": "cut", "error": "disk full"})
    from fastapi.testclient import TestClient
    body = TestClient(server.app).get("/api/jobs/history").json()
    assert body["queue"] is False
    assert body["jobs"][0]["status"] == "failed"
    assert body["jobs"][0]["error"] == "disk full"


def test_a_second_process_leaves_the_data_directory(tmp_path):
    home = tmp_path / "studio"
    home.mkdir()
    port = _free_port()
    log_path = tmp_path / "first.log"
    proc, handle = _start_server(home, port, log_path)
    other = None
    other_handle = None
    try:
        _wait_health(port, proc, log_path)
        other_log = tmp_path / "second.log"
        other, other_handle = _start_server(home, _free_port(), other_log)
        code = other.wait(timeout=40)
        assert code == 1
        assert "another Studio already has this data directory" in other_log.read_text()
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=2) as response:
            assert json.loads(response.read())["ok"] is True
    finally:
        if other is not None and other.poll() is None:
            os.killpg(other.pid, signal.SIGKILL)
            other.wait(timeout=5)
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=20)
        handle.close()
        if other_handle is not None:
            other_handle.close()


def test_a_live_gpu_lock_refuses_a_second_studio(tmp_path):
    home = tmp_path / "studio"
    lock = home / "gpu.lock.d"
    lock.mkdir(parents=True)
    holder = subprocess.Popen(["sleep", "60"])
    (lock / "owner").write_text(f"foreign {holder.pid}\n")
    log_path = tmp_path / "server.log"
    proc, handle = _start_server(home, _free_port(), log_path)
    try:
        code = proc.wait(timeout=40)
        assert code == 1
        assert "GPU lock" in log_path.read_text()
        assert holder.poll() is None
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=5)
        if holder.poll() is None:
            holder.kill()
            holder.wait(timeout=5)
        handle.close()


def test_a_full_disk_refuses_a_job(monkeypatch):
    monkeypatch.setenv("STUDIO_DISK_RESERVE_BYTES", str(1 << 62))
    with pytest.raises(HTTPException) as caught:
        server.enqueue("export", "export", {"clips": []})
    assert caught.value.status_code == 507
    assert "unfinished render" in caught.value.detail


def test_sweep_removes_only_old_studio_temp(tmp_path):
    home = tmp_path / "studio"
    temp = home / "comfy" / "temp"
    temp.mkdir(parents=True)
    old = temp / "old.bin"
    fresh = temp / "fresh.bin"
    old.write_text("old")
    fresh.write_text("fresh")
    os.utime(old, (time.time() - 90000, time.time() - 90000))
    outside = tmp_path / "weights.bin"
    outside.write_bytes(b"weight")
    link = temp / "weights.bin"
    link.symlink_to(outside)
    os.utime(link, (time.time() - 90000, time.time() - 90000), follow_symlinks=False)
    keep = home / "renders" / "keep.png"
    keep.parent.mkdir()
    keep.write_text("keep")
    removed = housekeeping.sweep(home)
    assert removed == [str(old)]
    assert fresh.is_file() and outside.is_file() and link.is_symlink() and keep.is_file()


def test_backup_restore_and_rollback_stay_on_the_copy(tmp_path):
    home = tmp_path / "home"
    project = home / "projects" / "one"
    project.mkdir(parents=True)
    (project / "project.json").write_text('{"name": "one"}\n')
    (home / "settings.json").write_text("{}\n")
    (home / "schema.json").write_text('{"version": 1}\n')
    (home / "renders").mkdir()
    (home / "renders" / "keep.txt").write_text("keep")
    (home / "job-history.json").write_text('{"queue": false, "jobs": []}\n')
    outside = tmp_path / "weights.bin"
    outside.write_bytes(b"weight")
    (project / "weight").symlink_to(outside)
    (home / "comfy" / "temp").mkdir(parents=True)
    (home / "comfy" / "temp" / "scratch").write_text("no")
    (home / "studio.lock").write_text("1\n")
    saved = opsdata.backup(home, tmp_path / "backup")
    assert saved["queue"] is False
    assert not (tmp_path / "backup" / "projects" / "one" / "weight").exists()
    assert not (tmp_path / "backup" / "comfy").exists()
    assert not (tmp_path / "backup" / "studio.lock").exists()
    restored = opsdata.restore(tmp_path / "backup", tmp_path / "restored")
    assert restored["replayed"] is False
    assert (tmp_path / "restored" / "renders" / "keep.txt").read_text() == "keep"
    report = opsdata.rehearse(home, tmp_path / "work", "newrev", "14151f6")
    assert report["working_version"] == 2
    assert report["restored_version"] == 1
    assert report["prior_image"] == "14151f6"
    assert report["docker"] is False and report["pruned"] is False and report["replayed"] is False
    assert json.loads((tmp_path / "work" / "restored" / "projects" / "one" / "project.json").read_text())["name"] == "one"
    with pytest.raises(ValueError):
        opsdata.backup(home, home / "inside")


def test_the_data_script_prints_a_backup(tmp_path):
    home = tmp_path / "home"
    (home / "projects").mkdir(parents=True)
    (home / "projects" / "note.txt").write_text("note")
    done = subprocess.run([sys.executable, str(ROOT / "scripts" / "studio-data.py"), "backup",
                           "--home", str(home), "--to", str(tmp_path / "out")],
                          capture_output=True, text=True, timeout=20)
    assert done.returncode == 0, done.stderr
    body = json.loads(done.stdout)
    assert body["queue"] is False
    assert (tmp_path / "out" / "projects" / "note.txt").read_text() == "note"


def test_a_named_file_is_not_a_runnable_route(tmp_path):
    missing = tmp_path / "missing.bin"
    found = tmp_path / "found.bin"
    found.write_bytes(b"abc")
    item = {"expected": missing, "kind": "file", "executable": False, "members": (), "names": ()}
    loose = discover.readiness(item, {"candidates": [{"path": str(found), "verified": False}]})
    assert loose == {"present": True, "compatible": False, "licensed": False, "benchmarked": False, "runnable": False}
    proved = discover.readiness(item, {"candidates": [{"path": str(found), "verified": True}]})
    assert proved["compatible"] is True and proved["runnable"] is False


def test_a_container_walks_only_configured_mounts(tmp_path, monkeypatch):
    monkeypatch.setenv("STUDIO_CONTAINER", "1")
    monkeypatch.setenv("STUDIO_SCAN_ROOTS", "")
    assert discover.roots() == []


def test_sdxl_standing_keeps_the_five_facts_apart():
    import sdxl_turbo as picture
    size = picture.BYTES
    recipe = {"width": 512, "height": 512, "steps": 1, "cfg": 1.0, "sampler": "euler_ancestral",
              "scheduler": "sgm_uniform", "batch_size": 1}
    profile = {"enabled": False, "model": "sdxl-turbo",
               "checkpoint": {"bytes": size, "license": picture.LICENSE},
               "recipe": recipe, "render": {"output": "/tmp/still.png"},
               "gpu_name": "NVIDIA GeForce RTX 3060", "driver": "595.84", "floor_mb": 8000}
    closed = picture.classify(profile, size)
    assert closed["present"] and closed["compatible"] and closed["licensed"] and closed["benchmarked"]
    assert closed["runnable"] is False
    assert closed["hardware"]["floor_mb"] == 8000
    opened = picture.classify({**profile, "enabled": True}, size)
    assert opened["runnable"] is True
    wrong = picture.classify({**profile, "enabled": True}, size - 1)
    assert wrong["present"] is True and wrong["compatible"] is False and wrong["runnable"] is False
    assert "8000" in opened["note"]


def test_preflight_refuses_a_busy_or_foreign_queue():
    preflight = _preflight()
    clear = preflight.assess({"queue_running": [], "queue_pending": []})
    assert clear["clear"] is True and clear["submit"] is False
    busy = preflight.assess({"queue_running": [{}], "queue_pending": []})
    assert busy["clear"] is False and busy["reason"] == "queue_not_empty" and busy["submit"] is False
    gone = preflight.assess({"queue_running": [], "queue_pending": []}, owner_alive=False)
    assert gone["reason"] == "owner_not_running"
    with pytest.raises(ValueError, match="protected_endpoint"):
        preflight.fetch("http://127.0.0.1:8190")
    with pytest.raises(ValueError, match="protected_endpoint"):
        preflight.fetch("http://10.1.1.1:8188")
