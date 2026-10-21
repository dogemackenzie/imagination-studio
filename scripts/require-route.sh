#!/bin/bash
# Stop a generative script on a machine with no measured render route.
# A Mac and a DGX Spark (hardware.is_spark) exit 0. App-only prints the reason and exits 1.
# Usage: bash scripts/require-route.sh KIND
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KIND="${1:-}"
export PYTHONPATH="$HERE/../app${PYTHONPATH:+:$PYTHONPATH}"
exec python3 -c 'import sys, capabilities; raise SystemExit(capabilities.script_exit(sys.argv[1]))' "$KIND"
