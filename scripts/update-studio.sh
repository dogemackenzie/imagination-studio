#!/bin/bash
# Apply source updates to this Linux app container, on the host.
# Fetches origin, merges origin/main when this checkout has its own commits, runs the tests,
# then rebuilds and recreates only the app container. Pushes fork. Origin is fetch-only.
# The container has no git and no Docker socket, so it cannot do this itself.
# A native Mac or Spark install keeps scripts/app.sh and its fast-forward.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export GIT_TERMINAL_PROMPT=0
export GIT_EDITOR=true

if [[ "${STUDIO_CONTAINER:-}" == "1" ]]; then
  echo "Run this on the host. The container does not rebuild itself." >&2
  exit 1
fi

gitc() {
  git -c core.hooksPath=/dev/null "$@"
}

if ! gitc rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "This folder is not a git checkout." >&2
  exit 1
fi

branch="$(gitc rev-parse --abbrev-ref HEAD)"
if [[ "$branch" != "main" ]]; then
  echo "Updates merge origin/main into main. This checkout is on ${branch}." >&2
  exit 1
fi

if [[ -n "$(gitc status --porcelain --untracked-files=all)" ]]; then
  echo "Local changes are present. Commit or set them aside before updating." >&2
  exit 1
fi

if ! gitc remote get-url origin >/dev/null 2>&1; then
  echo "origin is not configured." >&2
  exit 1
fi

echo "Fetching origin…"
gitc fetch --no-tags --no-recurse-submodules origin

if ! gitc rev-parse --verify --quiet origin/main >/dev/null; then
  echo "origin/main is not available." >&2
  exit 1
fi

upstream="$(gitc rev-parse origin/main)"
before="$(gitc rev-parse HEAD)"

if gitc merge-base --is-ancestor "$upstream" HEAD; then
  echo "origin/main is already in this checkout."
else
  echo "Merging origin/main…"
  if ! gitc merge --no-edit origin/main; then
    gitc merge --abort || true
    echo "The merge did not apply. The checkout is unchanged. The container was not recreated." >&2
    exit 1
  fi
fi

read_health() {
  if [[ -n "${STUDIO_UPDATE_HEALTH_JSON+x}" ]]; then
    printf '%s' "$STUDIO_UPDATE_HEALTH_JSON"
    return 0
  fi
  local url="${STUDIO_UPDATE_HEALTH_URL:-http://127.0.0.1:7870/api/health}"
  curl -fsS --max-time 3 "$url" 2>/dev/null || true
}

json_field() {
  python3 -c 'import json,sys; print(json.loads(sys.argv[1]).get(sys.argv[2]) or "")' "$1" "$2"
}

short="$(gitc rev-parse --short=7 HEAD)"
health="$(read_health)"
running_rev=""
running_up=""
if [[ -n "$health" ]]; then
  running_rev="$(json_field "$health" revision)"
  running_up="$(json_field "$health" upstream)"
fi

same_image=0
if [[ "$(gitc rev-parse HEAD)" == "$before" && "$running_rev" == "$short" && "$running_up" == "$upstream" ]]; then
  same_image=1
fi

push_fork() {
  if [[ "${STUDIO_UPDATE_PUSH:-1}" != "1" ]]; then
    return 0
  fi
  if ! gitc push fork main; then
    echo "The update is on this machine. The push to fork did not finish." >&2
    return 1
  fi
}

if [[ "$same_image" == 1 ]]; then
  echo "The published container already has this checkout."
  push_fork
  exit 0
fi

echo "Running tests…"
set +e
if [[ -n "${STUDIO_UPDATE_PYTEST:-}" ]]; then
  "$STUDIO_UPDATE_PYTEST"
  code=$?
elif [[ -x /tmp/imagination-audit-venv/bin/pytest ]]; then
  /tmp/imagination-audit-venv/bin/pytest tests --timeout=240 --timeout-method=signal -q --tb=line
  code=$?
elif python3 -m pytest --version >/dev/null 2>&1; then
  python3 -m pytest tests --timeout=240 --timeout-method=signal -q --tb=line
  code=$?
else
  echo "pytest is not available." >&2
  exit 1
fi
set -e
if [[ "$code" -ne 0 ]]; then
  echo "Tests failed. The container was not recreated." >&2
  exit "$code"
fi

if [[ -z "${STUDIO_UPDATE_COMPOSE:-}" && ! -f compose.yaml ]]; then
  echo "compose.yaml is not in this checkout. The container was not recreated." >&2
  exit 1
fi

compose() {
  if [[ -n "${STUDIO_UPDATE_COMPOSE:-}" ]]; then
    "$STUDIO_UPDATE_COMPOSE" "$@"
  else
    docker compose "$@"
  fi
}

echo "Building imagination-studio-app:${short}…"
SOURCE_REV="$short" UPSTREAM_REV="$upstream" compose build studio
SOURCE_REV="$short" UPSTREAM_REV="$upstream" compose up -d --force-recreate --no-deps studio

if [[ -z "${STUDIO_UPDATE_COMPOSE:-}" ]]; then
  echo "Waiting for the app…"
  ok=0
  for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30; do
    reported="$(curl -fsS --max-time 2 http://127.0.0.1:7870/api/health 2>/dev/null || true)"
    if [[ -n "$reported" ]] && [[ "$(json_field "$reported" revision)" == "$short" ]]; then
      ok=1
      break
    fi
    sleep 1
  done
  if [[ "$ok" != 1 ]]; then
    echo "The container did not report revision ${short}. The previous image tag is still on this machine." >&2
    exit 1
  fi
fi

push_fork
echo "Published imagination-studio-app:${short}."
