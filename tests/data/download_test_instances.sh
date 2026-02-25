#!/usr/bin/env bash
# Download the datasets needed for local test coverage into gitignored folders.
# Usage:
#   ./tests/data/download_test_instances.sh           # small sets (default)
#   ./tests/data/download_test_instances.sh --full    # full Netlib + full MIPLIB

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODE="small"
if [[ "${1:-}" == "--full" ]]; then
    MODE="full"
elif [[ -n "${1:-}" ]]; then
    echo "Usage: $0 [--full]"
    exit 1
fi

if [[ "${MODE}" == "small" ]]; then
    echo "Downloading Netlib small subset..."
    "${SCRIPT_DIR}/download_netlib.sh" --small

    echo "Downloading MIPLIB small subset..."
    "${SCRIPT_DIR}/download_miplib.sh" --small
else
    echo "Downloading full Netlib set..."
    "${SCRIPT_DIR}/download_netlib.sh"

    echo "Downloading full MIPLIB collection..."
    "${SCRIPT_DIR}/download_miplib.sh"
fi

echo "Done downloading test instances."
