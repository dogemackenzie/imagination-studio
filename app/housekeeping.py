"""Disk room for an unfinished render, and Studio-owned temporary files.

``sweep`` deletes regular files under ``STUDIO_HOME/comfy/temp`` that are older than a day.
It does not follow a symlink, and it does not delete a model, a Docker image, a cache
outside that directory, or another application's files.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

OWNED_TEMP = ("comfy/temp",)
OLDER_THAN_S = 24 * 60 * 60


def reserve_bytes() -> int | None:
    """The free-space reserve, in bytes. Unset means this process does not add one."""

    raw = os.environ.get("STUDIO_DISK_RESERVE_BYTES", "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def short_disk(home: Path) -> str:
    """Why a job cannot be queued, or "" when the reserve is unset or the disk has room."""

    reserve = reserve_bytes()
    if reserve is None:
        return ""
    try:
        stats = os.statvfs(home)
        free = stats.f_bavail * stats.f_frsize
    except OSError:
        return "The free space on the studio folder could not be read, so a job was not queued."
    if free >= reserve:
        return ""
    return (f"The studio folder has {free // (1024 * 1024)} MB free. "
            f"A job needs {reserve // (1024 * 1024)} MB left for an unfinished render.")


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return False
    return True


def sweep(home: Path, *, now: float | None = None, older_than_s: float = OLDER_THAN_S) -> list[str]:
    """Delete old regular files in the Studio temp directories. Return the paths removed."""

    now = time.time() if now is None else now
    removed: list[str] = []
    root = home.resolve()
    for relative in OWNED_TEMP:
        folder = (home / relative).resolve()
        if not _inside(folder, root) or not folder.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(folder, followlinks=False):
            here = Path(dirpath)
            if here.is_symlink() or not _inside(here, folder):
                dirnames.clear()
                continue
            kept = []
            for name in dirnames:
                child = here / name
                if child.is_symlink() or not _inside(child, folder):
                    continue
                kept.append(name)
            dirnames[:] = kept
            for name in filenames:
                file = here / name
                if file.is_symlink() or not file.is_file() or not _inside(file, folder):
                    continue
                try:
                    if now - file.stat().st_mtime < older_than_s:
                        continue
                    file.unlink()
                except OSError:
                    continue
                removed.append(str(file))
    return removed
