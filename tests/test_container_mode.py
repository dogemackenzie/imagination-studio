"""Container mode stays app-only: no home migration, no disk walk, no git or pip update, a cheap health check,
and a shutdown that cancels only this process's work.

python3 -m pytest tests/test_container_mode.py
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
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

import discover  # noqa: E402
import hardware  # noqa: E402
import server  # noqa: E402
import studio_home as studio_home_module  # noqa: E402
import updates  # noqa: E402
import voice  # noqa: E402
from studio_home import studio_home  # noqa: E402


def test_a_container_does_not_rename_the_old_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("STUDIO_HOME", raising=False)
    monkeypatch.setenv("STUDIO_CONTAINER", "1")
    monkeypatch.setattr(studio_home_module, "server_running", lambda: False)
    (tmp_path / "TensorFoldStudio" / "renders").mkdir(parents=True)
    assert studio_home(move=True) == tmp_path / "TensorFoldStudio"
    assert not (tmp_path / "ImaginationStudio").exists()
    assert not (tmp_path / "TensorFoldStudio").is_symlink()


def test_a_container_does_not_walk_mounted_disks(tmp_path, monkeypatch):
    monkeypatch.setenv("STUDIO_CONTAINER", "1")
    monkeypatch.delenv("STUDIO_SCAN_ROOTS", raising=False)
    assert discover.roots() == []
    named = tmp_path / "named-models"
    other = tmp_path / "not-configured"
    named.mkdir()
    other.mkdir()
    monkeypatch.setenv("STUDIO_SCAN_ROOTS", str(named))
    assert [str(path) for path in discover.roots()] == [str(named)]


def test_a_container_does_not_schedule_a_scan_on_startup(monkeypatch):
    calls = []
    monkeypatch.setenv("STUDIO_CONTAINER", "1")
    monkeypatch.delenv("STUDIO_DISCOVER_DELAY", raising=False)
    monkeypatch.setattr(discover, "start_scan", lambda: calls.append("scan"))
    try:
        discover.schedule()
        assert calls == []
        assert discover._scheduled is None
    finally:
        discover.cancel()


def test_health_does_not_scan_query_or_fetch(monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("health did extra work")

    monkeypatch.setattr(hardware, "observe", boom)
    monkeypatch.setattr(hardware, "_gpu_line", boom)
    monkeypatch.setattr(discover, "scan", boom)
    monkeypatch.setattr(discover, "schedule", boom)
    monkeypatch.setattr(updates, "git", boom)
    monkeypatch.setenv("STUDIO_REVISION", "rev-health")
    monkeypatch.delenv("STUDIO_UPSTREAM_REV", raising=False)
    response = TestClient(server.app).get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True and body["version"] and body["revision"] == "rev-health"
    assert body["upstream"] == "" and "gpu" not in body
    monkeypatch.setenv("STUDIO_UPSTREAM_REV", "ab" * 20)
    assert TestClient(server.app).get("/api/health").json()["upstream"] == "ab" * 20
    monkeypatch.setenv("STUDIO_UPSTREAM_REV", "unknown")
    assert TestClient(server.app).get("/api/health").json()["upstream"] == ""


def test_state_shows_the_image_revision(monkeypatch):
    monkeypatch.setenv("STUDIO_REVISION", "abc1234")
    monkeypatch.setenv("STUDIO_CONTAINER", "1")
    body = TestClient(server.app).get("/api/state").json()
    assert body["version"] and body["revision"] == "abc1234" and body["container"] is True


def test_a_container_update_does_not_fetch_or_install(monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("git was called")

    monkeypatch.setenv("STUDIO_CONTAINER", "1")
    monkeypatch.setenv("STUDIO_REVISION", "abc1234")
    monkeypatch.delenv("STUDIO_UPSTREAM_REV", raising=False)
    monkeypatch.setattr(updates, "git", boom)
    monkeypatch.setattr(updates, "inspect", boom)
    monkeypatch.setattr(updates, "github_api", boom)
    saved = dict(server.updater.info)
    try:
        status = server.updater.check()
        assert status["container"] is True and status["managed"] is False
        assert status["available"] is False and status["installed_commit"] == "abc1234"
        assert "scripts/update-studio.sh" in status["message"] and "recreate the container" in status["message"]
        with pytest.raises(updates.UpdateError, match="recreate the container"):
            server.updater.apply()
        with pytest.raises(updates.UpdateError, match="recreate the container"):
            server.updater.set_automatic(True)
        server.updater.tick()
    finally:
        server.updater.info = saved


def test_a_container_compares_the_baked_upstream_commit(monkeypatch):
    """The image names the source commit it contains. A newer main is reported and is not installed here."""

    baked = "a" * 40
    newer = "b" * 40
    asked = []

    def lookup(path):
        asked.append(path)
        return {"sha": newer}

    monkeypatch.setenv("STUDIO_CONTAINER", "1")
    monkeypatch.setenv("STUDIO_UPSTREAM_REV", baked)
    monkeypatch.setattr(updates, "git", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("git was called")))
    monkeypatch.setattr(updates, "github_api", lookup)
    updates.upstream_answers.clear()
    saved = dict(server.updater.info)
    try:
        status = server.updater.check()
        assert status["available"] is True and status["target"] == newer and status["current"] == baked
        assert status["source"] == updates.DEFAULT_UPSTREAM and "An update is on the source." in status["message"]
        assert asked == [updates.DEFAULT_UPSTREAM + "/commits/main"]
        with pytest.raises(updates.UpdateError, match="recreate the container"):
            server.updater.apply()
        updates.upstream_answers.clear()
        monkeypatch.setattr(updates, "github_api", lambda _path: {"sha": baked})
        current = server.updater.check()
        assert current["available"] is False and current["message"] == "Studio is up to date."
        monkeypatch.setattr(updates, "github_api", lambda _path: (_ for _ in ()).throw(OSError("offline")))
        updates.upstream_answers.clear()
        missed = server.updater.check()
        assert missed["available"] is False and missed["message"].startswith("The source could not be read.")
    finally:
        updates.upstream_answers.clear()
        server.updater.info = saved


def test_a_published_container_accepts_the_bridge_peer_for_updates(monkeypatch):
    assert updates.may_manage("172.18.0.1", "127.0.0.1", True) is True
    assert updates.may_manage("172.18.0.1", "127.0.0.1", False) is False
    assert updates.may_manage("172.18.0.1", "evil.example", True) is False
    assert updates.may_manage("192.168.1.2", "127.0.0.1", False) is False
    monkeypatch.setenv("STUDIO_CONTAINER", "1")
    monkeypatch.setenv("STUDIO_REVISION", "abc1234")
    published = TestClient(server.app, client=("172.18.0.1", 9))
    got = published.get("/api/updates")
    assert got.status_code == 200
    assert got.json()["container"] is True
    monkeypatch.delenv("STUDIO_CONTAINER", raising=False)
    native = TestClient(server.app, client=("172.18.0.1", 9))
    assert native.get("/api/updates").status_code == 403
    monkeypatch.setenv("STUDIO_CONTAINER", "1")
    assert published.get("/api/brains").status_code == 200
    assert published.get("/api/agent/tools").status_code == 200
    monkeypatch.delenv("STUDIO_CONTAINER", raising=False)
    assert native.get("/api/brains").status_code == 403


def test_listen_host_refuses_the_lan_switch_in_a_container(monkeypatch):
    monkeypatch.setenv("STUDIO_CONTAINER", "1")
    monkeypatch.setenv("HOST", "0.0.0.0")
    with pytest.raises(SystemExit):
        server.listen_host()
    monkeypatch.delenv("HOST", raising=False)
    monkeypatch.setenv("STUDIO_BIND", "0.0.0.0")
    assert server.listen_host() == "0.0.0.0"
    monkeypatch.delenv("STUDIO_CONTAINER", raising=False)
    monkeypatch.delenv("STUDIO_BIND", raising=False)
    assert server.listen_host() == "127.0.0.1"


def test_a_forbidden_engine_path_is_not_installed(monkeypatch, tmp_path):
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    blocked.chmod(0)
    try:
        monkeypatch.setattr(server, "ZIG_ENGINE", blocked / "zig-engine")
        assert server.zig_ready() is False
        assert server.turbo_ready() is False
    finally:
        blocked.chmod(0o755)


def test_shutdown_refuses_a_new_job():
    server._closing.set()
    try:
        with pytest.raises(HTTPException) as caught:
            server.enqueue("export", "export", {"clips": []})
        assert caught.value.status_code == 503
    finally:
        server._closing.clear()


def test_shutdown_signals_only_the_owned_child(tmp_path, monkeypatch):
    owned = subprocess.Popen(["sleep", "60"], start_new_session=True)
    foreign = subprocess.Popen(["sleep", "60"], start_new_session=True)
    lock = tmp_path / "gpu"
    lock.mkdir()
    (lock / "owner").write_text(f"foreign {foreign.pid}\n")
    monkeypatch.setattr(voice, "LOCK", lock)
    server.jobs["phase3-owned"] = {"id": "phase3-owned", "status": "running", "params": {}}
    server.running.update(process=owned, id="phase3-owned")
    try:
        server.end()
        owned.wait(timeout=5)
        assert foreign.poll() is None
        assert (lock / "owner").read_text().strip().endswith(str(foreign.pid))
        assert server.jobs["phase3-owned"]["status"] == "cancelled"
    finally:
        for proc in (owned, foreign):
            if proc.poll() is None:
                os.kill(proc.pid, signal.SIGKILL)
                proc.wait(timeout=5)
        server.jobs.pop("phase3-owned", None)
        server.running.update(process=None, id=None)
        server.begin()


def test_shutdown_releases_a_lock_this_process_owns(tmp_path, monkeypatch):
    lock = tmp_path / "gpu"
    lock.mkdir()
    (lock / "owner").write_text(f"studio {os.getpid()}\n")
    monkeypatch.setattr(voice, "LOCK", lock)
    try:
        server.end()
        assert not lock.exists()
    finally:
        server.begin()


def test_container_comfy_does_not_copy_or_start(tmp_path):
    prefix = tmp_path / "prefix"
    node = prefix / "ComfyUI" / "custom_nodes" / "tensorfold_h3"
    node.mkdir(parents=True)
    (node / "keep").write_text("native")
    script = (ROOT / "spark" / "comfy.sh").read_text()
    assert 'exec python3 "$HERE/managed_comfy.py" run "$PREFIX/venv/bin/python" main.py --listen' in script
    assert 'nice -n 19 "$PREFIX/venv/bin/python" main.py --listen 127.0.0.1 --port "$PORT"' in script
    assert script.count('if [ -n "${SPARK_GOVERNOR:-}" ]') == 2
    env = os.environ.copy()
    env.update(PREFIX=str(prefix), STUDIO_CONTAINER="1", STUDIO_COMFY_STATE=str(tmp_path / "state"))
    for command in ("run", "start"):
        done = subprocess.run(["bash", str(ROOT / "spark" / "comfy.sh"), command], env=env,
                              capture_output=True, text=True, timeout=10)
        assert done.returncode != 0
        assert "does not start a renderer" in done.stderr
        assert (node / "keep").read_text() == "native"


def test_scout_sheet_uses_the_named_interpreter(tmp_path):
    text = (ROOT / "scripts" / "scout.sh").read_text()
    assert text.index("require-route.sh") < text.index("STUDIO_PYTHON")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "scout.sh").write_text(text)
    (scripts / "require-route.sh").write_text("#!/bin/bash\nexit 0\n")
    (scripts / "image.sh").write_text("#!/bin/bash\nexit 0\n")
    called = tmp_path / "called"
    recorder = tmp_path / "record.sh"
    recorder.write_text("#!/bin/sh\necho \"$2\" > \"$CALLED\"\nexit 0\n")
    for path in (scripts / "scout.sh", scripts / "require-route.sh", scripts / "image.sh", recorder):
        path.chmod(0o755)
    env = os.environ.copy()
    env.update(STUDIO_PYTHON=str(recorder), PREFIX=str(tmp_path / "missing"), CALLED=str(called), SEED="1")
    done = subprocess.run(["bash", str(scripts / "scout.sh"), "a picture", "1", str(tmp_path / "out")],
                          env=env, capture_output=True, text=True, timeout=15)
    assert done.returncode == 0, done.stderr
    assert called.read_text().strip() == str(tmp_path / "out")


def test_the_page_keeps_the_native_update_sentence():
    page = (ROOT / "app" / "static" / "updates.js").read_text()
    native = "Start this installation with scripts/app.sh to install updates. Worker Sparks update through the Sparks tab."
    container = ("This Studio is a container image. On the host, run scripts/update-studio.sh to merge the source, "
                 "run the tests, rebuild the image, and recreate the container.")
    assert native in page and container in page and container == updates.CONTAINER_UPDATE
    assert page.index("d.container") < page.index("d.managed === false")
    html = (ROOT / "app" / "static" / "index.html").read_text()
    assert 'S.revision ? " · " + S.revision' in html


def test_sigterm_stops_the_server(tmp_path):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    home = tmp_path / "studio"
    lock = tmp_path / "gpu.lock.d"
    env = os.environ.copy()
    env.update(STUDIO_HOME=str(home), STUDIO_CONTAINER="1", STUDIO_DISCOVER_DELAY="-1",
               STUDIO_GPU_LOCK=str(lock), PORT=str(port), PYTHONDONTWRITEBYTECODE="1")
    env.pop("HOST", None)
    env.pop("STUDIO_BIND", None)
    env.pop("STUDIO_UPDATE_SESSION", None)
    log_path = tmp_path / "server.log"
    handle = log_path.open("w")
    proc = subprocess.Popen([sys.executable, str(ROOT / "app" / "server.py")], cwd=ROOT, env=env,
                            stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        deadline = time.time() + 30
        up = False
        while time.time() < deadline:
            if proc.poll() is not None:
                break
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=1) as response:
                    body = json.loads(response.read())
                up = response.status == 200 and body.get("ok") is True and body.get("container") is True
                if up:
                    break
            except (OSError, ValueError):
                time.sleep(0.2)
        assert up, log_path.read_text()[-2000:]
        lock.mkdir()
        (lock / "owner").write_text(f"studio {proc.pid}\n")
        started = time.time()
        os.kill(proc.pid, signal.SIGTERM)
        code = proc.wait(timeout=20)
        # Uvicorn shuts the lifespan down, then re-raises SIGTERM. -15 is that exit.
        assert code in (0, -signal.SIGTERM)
        assert time.time() - started < 20
        assert not lock.exists()
        assert "ComfyUI starting" not in log_path.read_text()
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=5)
        handle.close()
