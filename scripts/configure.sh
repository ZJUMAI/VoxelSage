#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${REPO_ROOT}/.env"
ENV_TEMPLATE="${REPO_ROOT}/.env.example"

if [[ ! -f "${ENV_FILE}" ]]; then
  cp "${ENV_TEMPLATE}" "${ENV_FILE}"
fi

current_value() {
  local name="$1"
  sed -n "s/^${name}=//p" "${ENV_FILE}" | tail -n 1
}

is_placeholder() {
  local value="$1"
  [[ -z "${value}" || "${value}" == *your-* || "${value}" == *example.com* ]]
}

prompt_value() {
  local name="$1"
  local label="$2"
  local example="$3"
  local secret="${4:-0}"
  local current input
  current="$(current_value "${name}")"

  while true; do
    if ! is_placeholder "${current}"; then
      printf '%s (press Enter to keep the current value)\n' "${label}"
    else
      printf '%s (example: %s)\n' "${label}" "${example}"
    fi

    if ((secret)); then
      read -r -s -p "> " input
      printf '\n'
    else
      read -r -p "> " input
    fi

    if [[ -z "${input}" ]] && ! is_placeholder "${current}"; then
      input="${current}"
    fi
    if [[ -n "${input}" && "${input}" != *$'\n'* && "${input}" != *$'\r'* ]]; then
      printf -v "${name}" '%s' "${input}"
      return
    fi
    echo "A non-empty value is required."
  done
}

echo "Configure the OpenAI-compatible LLM endpoint for VoxelSage."
prompt_value DASHSCOPE_API_KEY "DASHSCOPE_API_KEY" "sk-cc8d****c840" 1
prompt_value DASHSCOPE_BASE_URL "DASHSCOPE_BASE_URL" "https://api.deepseek.com"
prompt_value LLM_MODEL_NAME "LLM_MODEL_NAME" "deepseek-v4-flash-vision-exp"

TEMP_FILE="$(mktemp "${ENV_FILE}.tmp.XXXXXX")"
trap 'rm -f "${TEMP_FILE}"' EXIT

awk '{ sub(/\r$/, "") } !/^(DASHSCOPE_API_KEY|DASHSCOPE_BASE_URL|LLM_MODEL_NAME)=/' \
  "${ENV_FILE}" >"${TEMP_FILE}"
{
  printf 'DASHSCOPE_API_KEY=%q\n' "${DASHSCOPE_API_KEY}"
  printf 'DASHSCOPE_BASE_URL=%q\n' "${DASHSCOPE_BASE_URL}"
  printf 'LLM_MODEL_NAME=%q\n' "${LLM_MODEL_NAME}"
  cat "${TEMP_FILE}"
} >"${ENV_FILE}"

trap - EXIT
rm -f "${TEMP_FILE}"
chmod 600 "${ENV_FILE}"
echo "Configuration saved to ${ENV_FILE}."
