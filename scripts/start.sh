#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python"
LOG_DIR="${REPO_ROOT}/.runtime/logs"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Missing ${PYTHON}. Run ./scripts/setup.sh first." >&2
  exit 1
fi
if [[ ! -d "${REPO_ROOT}/Frontend/node_modules" ]]; then
  echo "Frontend dependencies are missing. Run ./scripts/setup.sh first." >&2
  exit 1
fi

if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.env"
  set +a
fi

export SEGMENTATION_BACKEND="${SEGMENTATION_BACKEND:-vista3d}"
export VISTA3D_ROOT="${VISTA3D_ROOT:-${REPO_ROOT}/third_party/VISTA/vista3d}"
export VISTA3D_CONFIG="${VISTA3D_CONFIG:-${REPO_ROOT}/Port_B/SegAgent/VISTA3d/configs/infer.yaml}"

if [[ "${SEGMENTATION_BACKEND,,}" == "vista3d" ]] && \
   [[ ! -f "${VISTA3D_ROOT}/scripts/infer.py" ]]; then
  echo "VISTA3D is not installed at ${VISTA3D_ROOT}. Run ./scripts/setup.sh first." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
PIDS=()
NAMES=()

start_service() {
  local name="$1"
  local directory="$2"
  shift 2
  (
    cd "${directory}"
    exec "$@"
  ) >"${LOG_DIR}/${name}.log" 2>&1 &
  PIDS+=("$!")
  NAMES+=("${name}")
}

cleanup() {
  trap - EXIT INT TERM HUP
  if ((${#PIDS[@]})); then
    kill "${PIDS[@]}" 2>/dev/null || true
    wait "${PIDS[@]}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM HUP

start_service imaging-api "${REPO_ROOT}/Port_B" \
  "${PYTHON}" API.py server --port 8765
start_service output-proxy "${REPO_ROOT}/Port_B" \
  "${PYTHON}" file_proxy.py --port 8898
start_service agent-service "${REPO_ROOT}/Port_A" \
  "${PYTHON}" -m core.server
start_service frontend "${REPO_ROOT}" \
  npm --prefix Frontend run dev

echo "VoxelSage is starting with segmentation backend: ${SEGMENTATION_BACKEND}"
echo "Open http://localhost:3000"
echo "Logs: ${LOG_DIR}"
echo "Press Ctrl+C to stop all services."

set +e
wait -n "${PIDS[@]}"
status=$?
set -e

for index in "${!PIDS[@]}"; do
  if ! kill -0 "${PIDS[index]}" 2>/dev/null; then
    echo "Service stopped: ${NAMES[index]} (see ${LOG_DIR}/${NAMES[index]}.log)" >&2
  fi
done
exit "${status}"
