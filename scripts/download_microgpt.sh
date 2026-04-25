#!/usr/bin/env bash
#
# Download Karpathy's `microgpt.py` gist for fixture generation and parity checks.

set -euo pipefail

readonly SCRIPT_NAME="$(basename "$0")"

readonly DEFAULT_OUTPUT="microgpt.py"
readonly MICROGPT_RAW_URL="https://gist.githubusercontent.com/karpathy/8627fe009c40f57531cb18360106ce95/raw/microgpt.py"

usage() {
  cat <<EOF
Usage: ${SCRIPT_NAME} [OUTPUT_PATH]

Download Karpathy's microgpt.py gist to OUTPUT_PATH.

Arguments:
  OUTPUT_PATH   Destination file path. Defaults to: ${DEFAULT_OUTPUT}
EOF
}

download() {
  local output_path="$1"

  if ! command -v curl >/dev/null 2>&1; then
    echo "Error: 'curl' is required to download microgpt.py." >&2
    exit 1
  fi

  curl --fail --location --silent --show-error \
    --output "${output_path}" \
    "${MICROGPT_RAW_URL}"
}

main() {
  local output_path="${DEFAULT_OUTPUT}"

  if [[ $# -gt 1 ]]; then
    usage >&2
    exit 1
  fi

  if [[ $# -eq 1 ]]; then
    case "$1" in
      -h|--help)
        usage
        exit 0
        ;;
      *)
        output_path="$1"
        ;;
    esac
  fi

  mkdir -p "$(dirname "${output_path}")"
  download "${output_path}"
  echo "Downloaded microgpt.py to ${output_path}"
}

main "$@"
