#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
VISTA_REPO="${REPO_ROOT}/third_party/VISTA"
WITH_TOTALSEGMENTATOR=0

usage() {
  cat <<'EOF'
Usage: ./scripts/setup.sh [--with-totalsegmentator]

Installs VoxelSage and its default VISTA3D backend. The optional flag also
installs TotalSegmentator so it can be selected per server or API request.
EOF
}

while (($#)); do
  case "$1" in
    --with-totalsegmentator) WITH_TOTALSEGMENTATOR=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

for command in python3 git npm; do
  if ! command -v "${command}" >/dev/null 2>&1; then
    echo "Missing required command: ${command}" >&2
    exit 1
  fi
done

if ! python3 -c 'import sys; raise SystemExit(not ((3, 10) <= sys.version_info[:2] < (3, 13)))'; then
  echo "VoxelSage requires Python 3.10, 3.11, or 3.12." >&2
  exit 1
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  python3 -m venv "${VENV_DIR}"
fi

"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install \
  -r "${REPO_ROOT}/Port_B/requirements.txt" \
  -r "${REPO_ROOT}/Port_A/requirements.txt"

if [[ ! -d "${VISTA_REPO}/.git" ]]; then
  mkdir -p "${REPO_ROOT}/third_party"
  git clone --depth 1 https://github.com/Project-MONAI/VISTA.git "${VISTA_REPO}"
else
  echo "Using existing VISTA3D source: ${VISTA_REPO}"
fi

if [[ ! -f "${VISTA_REPO}/vista3d/scripts/infer.py" ]]; then
  echo "Invalid VISTA3D checkout: ${VISTA_REPO}/vista3d/scripts/infer.py is missing" >&2
  exit 1
fi

if ((WITH_TOTALSEGMENTATOR)); then
  "${VENV_DIR}/bin/python" -m pip install \
    -r "${REPO_ROOT}/Port_B/requirements-totalsegmentator.txt"
fi
"${VENV_DIR}/bin/python" -m pip check

npm --prefix "${REPO_ROOT}/Frontend" ci
if [[ ! -f "${REPO_ROOT}/Frontend/.env.local" ]]; then
  cp "${REPO_ROOT}/Frontend/.env.example" "${REPO_ROOT}/Frontend/.env.local"
fi
if [[ -t 0 ]]; then
  echo
  "${REPO_ROOT}/scripts/configure.sh"
elif [[ ! -f "${REPO_ROOT}/.env" ]]; then
  cp "${REPO_ROOT}/.env.example" "${REPO_ROOT}/.env"
  echo "No interactive terminal detected. Run ./scripts/configure.sh before startup."
fi

echo
echo "Setup complete. Run ./scripts/start.sh"
echo "VISTA3D weights are downloaded automatically from Hugging Face on first inference."
"${VENV_DIR}/bin/python" "${REPO_ROOT}/scripts/doctor.py" --require-vista-compatible
