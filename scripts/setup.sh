#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
VISTA_REPO="${REPO_ROOT}/third_party/VISTA"
TOOLS_DIR="${REPO_ROOT}/.runtime/tools"
VENV_BACKUP_DIR="${REPO_ROOT}/.runtime/venv-backups"
UV_VERSION="0.12.5"
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

for command in git npm; do
  if ! command -v "${command}" >/dev/null 2>&1; then
    echo "Missing required command: ${command}" >&2
    exit 1
  fi
done

python_is_compatible() {
  "$1" -c \
    'import sys; raise SystemExit(not ((3, 10) <= sys.version_info[:2] < (3, 13)))' \
    >/dev/null 2>&1
}

install_uv() {
  local installer

  if command -v uv >/dev/null 2>&1; then
    UV_COMMAND="$(command -v uv)"
    return
  fi
  if [[ -x "${TOOLS_DIR}/uv" ]]; then
    UV_COMMAND="${TOOLS_DIR}/uv"
    return
  fi

  mkdir -p "${TOOLS_DIR}"
  installer="$(mktemp)"
  echo "Installing uv to prepare a compatible Python environment automatically..."
  if command -v curl >/dev/null 2>&1; then
    if ! curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" -o "${installer}"; then
      rm -f "${installer}"
      echo "Could not download the uv installer." >&2
      return 1
    fi
  elif command -v wget >/dev/null 2>&1; then
    if ! wget -q "https://astral.sh/uv/${UV_VERSION}/install.sh" -O "${installer}"; then
      rm -f "${installer}"
      echo "Could not download the uv installer." >&2
      return 1
    fi
  else
    echo "Automatic Python setup requires curl or wget." >&2
    rm -f "${installer}"
    return 1
  fi
  if ! env UV_UNMANAGED_INSTALL="${TOOLS_DIR}" sh "${installer}"; then
    rm -f "${installer}"
    echo "Could not install uv." >&2
    return 1
  fi
  rm -f "${installer}"
  UV_COMMAND="${TOOLS_DIR}/uv"
  if [[ ! -x "${UV_COMMAND}" ]]; then
    echo "The uv installer completed, but ${UV_COMMAND} was not created." >&2
    return 1
  fi
}

create_compatible_venv() {
  local backup_path=""
  local failed_path

  install_uv
  mkdir -p "${VENV_BACKUP_DIR}"
  if [[ -e "${VENV_DIR}" ]]; then
    backup_path="${VENV_BACKUP_DIR}/venv-incompatible-$(date +%Y%m%d-%H%M%S)-$$"
    mv "${VENV_DIR}" "${backup_path}"
    echo "Saved the incompatible virtual environment as ${backup_path}"
  fi

  if ! "${UV_COMMAND}" venv --python '>=3.10,<3.13' --seed "${VENV_DIR}"; then
    if [[ -e "${VENV_DIR}" ]]; then
      failed_path="${VENV_BACKUP_DIR}/venv-failed-$(date +%Y%m%d-%H%M%S)-$$"
      mv "${VENV_DIR}" "${failed_path}"
      echo "Saved the incomplete virtual environment as ${failed_path}" >&2
    fi
    if [[ -n "${backup_path}" ]]; then
      mv "${backup_path}" "${VENV_DIR}"
      echo "Restored the previous virtual environment." >&2
    fi
    return 1
  fi
}

if [[ -x "${VENV_DIR}/bin/python" ]] && \
   python_is_compatible "${VENV_DIR}/bin/python"; then
  echo "Using existing compatible virtual environment: ${VENV_DIR}"
else
  create_compatible_venv
fi

if ! python_is_compatible "${VENV_DIR}/bin/python"; then
  echo "Could not create a Python 3.10, 3.11, or 3.12 virtual environment." >&2
  exit 1
fi
echo "Python environment ready: $("${VENV_DIR}/bin/python" --version)"

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
  bash "${REPO_ROOT}/scripts/configure.sh"
elif [[ ! -f "${REPO_ROOT}/.env" ]]; then
  cp "${REPO_ROOT}/.env.example" "${REPO_ROOT}/.env"
  echo "No interactive terminal detected. Run bash ./scripts/configure.sh before startup."
fi

echo
echo "Setup complete. Run ./scripts/start.sh"
echo "VISTA3D weights are downloaded automatically from Hugging Face on first inference."
"${VENV_DIR}/bin/python" "${REPO_ROOT}/scripts/doctor.py" --require-vista-compatible
