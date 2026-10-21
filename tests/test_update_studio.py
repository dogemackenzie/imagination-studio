"""The host update script merges origin, tests, and rebuilds only after the tests pass.

It never fast-forwards away this checkout's commits, and it does not recreate the container
when the merge or the tests fail.
"""
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "update-studio.sh"


def git(folder, *args, check=True):
    result = subprocess.run(["git", "-C", str(folder), *args], capture_output=True, text=True, timeout=20)
    if check:
        assert result.returncode == 0, result.stderr
    return result


def init(path):
    path.mkdir()
    git(path, "init", "-b", "main")
    git(path, "config", "user.name", "Update tests")
    git(path, "config", "user.email", "tests@example.invalid")


def commit(path, name, text):
    (path / name).write_text(text)
    git(path, "add", name)
    git(path, "commit", "-m", name)


def installation(tmp_path):
    upstream, clone = tmp_path / "upstream", tmp_path / "checkout"
    init(upstream)
    commit(upstream, "VERSION", "1\n")
    git(tmp_path, "clone", str(upstream), str(clone))
    git(clone, "config", "user.name", "Update tests")
    git(clone, "config", "user.email", "tests@example.invalid")
    scripts = clone / "scripts"
    scripts.mkdir()
    shutil.copy(SCRIPT, scripts / "update-studio.sh")
    (scripts / "update-studio.sh").chmod(0o755)
    git(clone, "add", "scripts/update-studio.sh")
    git(clone, "commit", "-m", "script")
    return upstream, clone


def run(clone, tmp_path, pytest_code=0, health=None, push="0"):
    log = tmp_path / "compose.log"
    tester = tmp_path / "pytest.sh"
    tester.write_text(f"#!/bin/sh\necho ran > \"{tmp_path / 'pytest.ran'}\"\nexit {pytest_code}\n")
    tester.chmod(0o755)
    recorder = tmp_path / "compose.sh"
    recorder.write_text(
        "#!/bin/sh\nprintf '%s\\n' \"SOURCE_REV=${SOURCE_REV-}\" \"UPSTREAM_REV=${UPSTREAM_REV-}\" \"$*\" >> \"$LOG\"\n")
    recorder.chmod(0o755)
    env = os.environ.copy()
    env.update(STUDIO_UPDATE_PYTEST=str(tester), STUDIO_UPDATE_COMPOSE=str(recorder),
               STUDIO_UPDATE_PUSH=push, LOG=str(log))
    env.pop("STUDIO_CONTAINER", None)
    if health is None:
        env["STUDIO_UPDATE_HEALTH_JSON"] = "{}"
    else:
        env["STUDIO_UPDATE_HEALTH_JSON"] = json.dumps(health)
    done = subprocess.run(["bash", str(clone / "scripts" / "update-studio.sh")], cwd=clone, env=env,
                          capture_output=True, text=True, timeout=30)
    calls = log.read_text() if log.exists() else ""
    ran = (tmp_path / "pytest.ran").exists()
    return done, calls, ran


def test_a_matching_container_is_left_alone(tmp_path):
    upstream, clone = installation(tmp_path)
    head = git(clone, "rev-parse", "--short=7", "HEAD").stdout.strip()
    full = git(upstream, "rev-parse", "HEAD").stdout.strip()
    done, calls, ran = run(clone, tmp_path, health={"revision": head, "upstream": full, "ok": True})
    assert done.returncode == 0, done.stderr
    assert calls == "" and not ran
    assert "already has this checkout" in done.stdout


def test_a_divergent_source_is_merged_and_rebuilt(tmp_path):
    upstream, clone = installation(tmp_path)
    commit(clone, "ours.txt", "ours\n")
    commit(upstream, "theirs.txt", "theirs\n")
    done, calls, ran = run(clone, tmp_path)
    assert done.returncode == 0, done.stderr
    assert ran
    assert (clone / "ours.txt").is_file() and (clone / "theirs.txt").read_text() == "theirs\n"
    short = git(clone, "rev-parse", "--short=7", "HEAD").stdout.strip()
    full = git(upstream, "rev-parse", "HEAD").stdout.strip()
    assert calls.splitlines() == [
        f"SOURCE_REV={short}", f"UPSTREAM_REV={full}", "build studio",
        f"SOURCE_REV={short}", f"UPSTREAM_REV={full}", "up -d --force-recreate --no-deps studio"]
    assert "--ff-only" not in calls and "origin" not in calls


def test_a_conflict_leaves_the_checkout_and_the_container(tmp_path):
    upstream, clone = installation(tmp_path)
    before = git(clone, "rev-parse", "HEAD").stdout.strip()
    commit(clone, "same.txt", "ours\n")
    commit(upstream, "same.txt", "theirs\n")
    done, calls, ran = run(clone, tmp_path)
    assert done.returncode != 0
    assert "unchanged" in done.stderr
    assert calls == "" and not ran
    assert git(clone, "rev-parse", "HEAD").stdout.strip() != before  # the local commit stays
    assert git(clone, "rev-parse", "HEAD^").stdout.strip() == before
    assert (clone / "same.txt").read_text() == "ours\n"
    assert git(clone, "rev-parse", "--verify", "--quiet", "MERGE_HEAD", check=False).returncode != 0
    assert git(clone, "status", "--porcelain").stdout == ""


def test_a_failed_test_does_not_recreate_the_container(tmp_path):
    upstream, clone = installation(tmp_path)
    commit(upstream, "theirs.txt", "theirs\n")
    done, calls, ran = run(clone, tmp_path, pytest_code=1)
    assert done.returncode == 1
    assert ran and calls == ""
    assert "not recreated" in done.stderr
    assert (clone / "theirs.txt").read_text() == "theirs\n"


def test_local_changes_stop_before_the_fetch(tmp_path):
    upstream, clone = installation(tmp_path)
    (clone / "notes.txt").write_text("local\n")
    old = git(clone, "rev-parse", "origin/main").stdout.strip()
    commit(upstream, "theirs.txt", "theirs\n")
    done, calls, ran = run(clone, tmp_path)
    assert done.returncode != 0 and "Local changes" in done.stderr
    assert calls == "" and not ran
    assert git(clone, "rev-parse", "origin/main").stdout.strip() == old


def test_the_container_refuses_to_rebuild_itself(tmp_path):
    _upstream, clone = installation(tmp_path)
    env = os.environ.copy()
    env["STUDIO_CONTAINER"] = "1"
    done = subprocess.run(["bash", str(clone / "scripts" / "update-studio.sh")], cwd=clone, env=env,
                          capture_output=True, text=True, timeout=20)
    assert done.returncode != 0 and "host" in done.stderr


def test_a_finished_update_can_be_pushed_to_the_fork_only(tmp_path):
    upstream, clone = installation(tmp_path)
    fork = tmp_path / "fork.git"
    git(tmp_path, "init", "--bare", str(fork))
    git(clone, "remote", "add", "fork", str(fork))
    git(clone, "push", "fork", "main")
    commit(upstream, "theirs.txt", "theirs\n")
    done, calls, _ran = run(clone, tmp_path, push="1")
    assert done.returncode == 0, done.stderr
    assert "build studio" in calls
    assert git(fork, "rev-parse", "main").stdout.strip() == git(clone, "rev-parse", "HEAD").stdout.strip()
    text = SCRIPT.read_text()
    assert "gitc push fork main" in text and "push origin" not in text
    assert "0.0.0.0" not in text and "sudo" not in text
