#!/bin/sh
# Start the Studio app directly. No git updater, no pip, no renderer, no home-folder migration.
set -eu

if [ "$(id -u)" = 0 ]; then
  mkdir -p /data
  if [ "$(stat -c %u /data)" != "10001" ]; then
    chown -R studio:studio /data
  fi
  exec setpriv --reuid=studio --regid=studio --init-groups --inh-caps=-all -- "$0" "$@"
fi

host="${HOST:-}"
if [ "$host" = "0.0.0.0" ] || [ "$host" = "::" ] || [ "$host" = "::0" ]; then
  echo "container mode does not open the LAN: leave HOST unset" >&2
  exit 1
fi

export STUDIO_CONTAINER=1
export HOME=/home/studio
export STUDIO_HOME="${STUDIO_HOME:-/data}"
export STUDIO_DISCOVER_DELAY="${STUDIO_DISCOVER_DELAY:--1}"
export STUDIO_BIND="${STUDIO_BIND:-0.0.0.0}"
export PORT="${PORT:-7870}"
export STUDIO_COMFY_STATE="${STUDIO_COMFY_STATE:-$STUDIO_HOME/comfy}"
export STUDIO_GPU_LOCK="${STUDIO_GPU_LOCK:-$STUDIO_HOME/studio-gpu.lock.d}"
export STUDIO_STAGED_NODES="${STUDIO_STAGED_NODES:-/opt/studio/staged-nodes}"
unset STUDIO_UPDATE_SESSION || true

mkdir -p "$STUDIO_HOME/projects" "$STUDIO_HOME/uploads" "$STUDIO_HOME/renders" \
  "$STUDIO_HOME/scouts" "$STUDIO_HOME/exports" "$STUDIO_HOME/slate" \
  "$STUDIO_COMFY_STATE/output" "$STUDIO_COMFY_STATE/input" "$STUDIO_COMFY_STATE/temp"

exec /opt/studio/venv/bin/python /opt/studio/app/server.py
