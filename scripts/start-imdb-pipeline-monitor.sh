#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_DIR="${RUN_DIR:-${PROJECT_DIR}/runs/imdb_pipeline_monitor}"
STDOUT_LOG="${RUN_DIR}/stdout.log"
LAUNCHER_PID="${RUN_DIR}/launcher.pid"

mkdir -p "${RUN_DIR}"
cd "${PROJECT_DIR}"

if pgrep -af 'monitor-imdb-pipeline' | grep -v -E 'grep|pgrep' >/dev/null; then
  echo "imdb pipeline monitor already running"
  pgrep -af 'monitor-imdb-pipeline' | grep -v -E 'grep|pgrep'
  exit 0
fi

setsid bash scripts/monitor-imdb-pipeline.sh > "${STDOUT_LOG}" 2>&1 < /dev/null &
echo $! > "${LAUNCHER_PID}"
echo "imdb pipeline monitor started: $(cat "${LAUNCHER_PID}")"
