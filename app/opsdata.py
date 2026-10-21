"""Copy this Studio's projects and settings.

The copy is the projects, uploads, renders, scouts, exports, slate, settings, the job
record, and a schema file when one exists. Restoring it does not run a job. The job
record says ``queue`` false. A symlink is left behind so a link out to a model library
is not copied. Docker images, model weights, and another application's files are not
part of the copy and are not deleted.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

INCLUDE_DIRS = ("projects", "uploads", "renders", "scouts", "exports", "slate")
INCLUDE_FILES = ("settings.json", "job-history.json", "schema.json")
NOTE = "This copy is project and configuration data. It is not the queue. Restoring it does not run a job."


def _outside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return True
    return False


def _copy_file(src: Path, dest: Path, root: Path) -> None:
    if src.is_symlink() or not src.is_file() or _outside(src, root):
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def _copy_tree(src: Path, dest: Path, root: Path) -> None:
    if src.is_symlink() or not src.is_dir() or _outside(src, root):
        return
    dest.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dest / item.name
        if item.is_symlink():
            continue
        if item.is_dir():
            _copy_tree(item, target, root)
        elif item.is_file():
            _copy_file(item, target, root)


def _empty_dir(dest: Path) -> None:
    if dest.exists() and any(dest.iterdir()):
        raise ValueError("the destination is not empty")
    dest.mkdir(parents=True, exist_ok=True)


def backup(home: Path, dest: Path) -> dict:
    """Copy project data from ``home`` into an empty ``dest`` outside ``home``."""

    home = home.resolve()
    dest = dest.resolve()
    if dest == home or home in dest.parents or dest in home.parents:
        raise ValueError("the backup directory has to sit outside the studio folder")
    _empty_dir(dest)
    copied: list[str] = []
    for name in INCLUDE_DIRS:
        src = home / name
        if src.is_dir() and not src.is_symlink():
            _copy_tree(src, dest / name, home)
            copied.append(name)
    for name in INCLUDE_FILES:
        src = home / name
        if src.is_file() and not src.is_symlink():
            _copy_file(src, dest / name, home)
            copied.append(name)
    manifest = {
        "queue": False,
        "note": NOTE,
        "copied": copied,
        "image": os.environ.get("STUDIO_REVISION", ""),
        "prior_image": os.environ.get("STUDIO_PRIOR_IMAGE", ""),
    }
    (dest / "backup.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def restore(src: Path, into: Path) -> dict:
    """Copy a backup into an empty directory. This does not start a job."""

    src = src.resolve()
    into = into.resolve()
    if into == src or src in into.parents or into in src.parents:
        raise ValueError("the restore directory has to sit outside the backup")
    _empty_dir(into)
    copied: list[str] = []
    for name in INCLUDE_DIRS:
        folder = src / name
        if folder.is_dir() and not folder.is_symlink():
            _copy_tree(folder, into / name, src)
            copied.append(name)
    for name in INCLUDE_FILES:
        file = src / name
        if file.is_file() and not file.is_symlink():
            _copy_file(file, into / name, src)
            copied.append(name)
    return {"queue": False, "note": NOTE, "copied": copied, "replayed": False}


def rehearse(home: Path, work: Path, image: str, prior_image: str) -> dict:
    """Back up ``home``, bump a schema file on that copy's source, and restore the backup.

    ``work`` receives ``backup`` and ``restored``. The restored tree is the pre-migration
    data. ``image`` and ``prior_image`` are recorded and are not deleted. This function
    does not run Docker.
    """

    home = home.resolve()
    work = work.resolve()
    if work == home or home in work.parents or work in home.parents:
        raise ValueError("the rehearsal directory has to sit outside the studio folder")
    work.mkdir(parents=True, exist_ok=True)
    saved = backup(home, work / "backup")
    schema = home / "schema.json"
    try:
        previous = json.loads(schema.read_text()) if schema.is_file() else {"version": 1}
        version = int(previous.get("version", 1))
    except (OSError, ValueError, TypeError):
        version = 1
    schema.write_text(json.dumps({"version": version + 1, "image": image}, indent=1) + "\n")
    restored = restore(work / "backup", work / "restored")
    restored_schema = work / "restored" / "schema.json"
    try:
        restored_version = json.loads(restored_schema.read_text()).get("version", 1) if restored_schema.is_file() else 1
    except (OSError, ValueError):
        restored_version = 1
    return {
        "queue": False,
        "replayed": False,
        "docker": False,
        "pruned": False,
        "image": image,
        "prior_image": prior_image,
        "backup": saved,
        "restored": restored,
        "working_version": version + 1,
        "restored_version": restored_version,
    }
