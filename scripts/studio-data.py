#!/usr/bin/env python3
"""Copy, restore, or rehearse a rollback of this Studio's project data.

    python3 scripts/studio-data.py backup --home DIR --to DIR
    python3 scripts/studio-data.py restore --from DIR --into DIR
    python3 scripts/studio-data.py rehearse --home DIR --work DIR --image TAG --prior-image TAG
    python3 scripts/studio-data.py sweep --home DIR

The copy is projects and settings. Restoring it does not run a job. ``rehearse`` bumps a
schema file only after the backup exists, restores that backup beside it, and records both
image tags. It does not run Docker and it does not delete an image. ``sweep`` removes old
files under that home's ``comfy/temp`` only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

import housekeeping  # noqa: E402
import opsdata  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Back up or restore Studio project data. This does not run a job.")
    commands = parser.add_subparsers(dest="cmd", required=True)

    save = commands.add_parser("backup")
    save.add_argument("--home", type=Path, required=True)
    save.add_argument("--to", type=Path, required=True)

    back = commands.add_parser("restore")
    back.add_argument("--from", dest="src", type=Path, required=True)
    back.add_argument("--into", type=Path, required=True)

    roll = commands.add_parser("rehearse")
    roll.add_argument("--home", type=Path, required=True)
    roll.add_argument("--work", type=Path, required=True)
    roll.add_argument("--image", required=True)
    roll.add_argument("--prior-image", required=True)

    clean = commands.add_parser("sweep")
    clean.add_argument("--home", type=Path, required=True)

    args = parser.parse_args(argv)
    try:
        if args.cmd == "backup":
            report = opsdata.backup(args.home, args.to)
        elif args.cmd == "restore":
            report = opsdata.restore(args.src, args.into)
        elif args.cmd == "rehearse":
            report = opsdata.rehearse(args.home, args.work, args.image, args.prior_image)
        else:
            report = {"removed": housekeeping.sweep(args.home), "queue": False}
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
