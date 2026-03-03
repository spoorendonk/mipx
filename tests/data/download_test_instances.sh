#!/usr/bin/env bash
# Download the datasets needed for local test coverage into gitignored folders.
# Usage:
#   ./tests/data/download_test_instances.sh               # small sets (default)
#   ./tests/data/download_test_instances.sh --full         # full Netlib + full MIPLIB
#   ./tests/data/download_test_instances.sh --mittelman    # Mittelman LP + MIPLIB benchmark set
#   ./tests/data/download_test_instances.sh --all          # all available LP/MIP corpora

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODE="small"
if [[ "${1:-}" == "--full" ]]; then
    MODE="full"
elif [[ "${1:-}" == "--mittelman" ]]; then
    MODE="mittelman"
elif [[ "${1:-}" == "--all" ]]; then
    MODE="all"
elif [[ -n "${1:-}" ]]; then
    echo "Usage: $0 [--full|--mittelman|--all]"
    exit 1
fi

case "${MODE}" in
    small)
        echo "Downloading Netlib small subset..."
        "${SCRIPT_DIR}/download_netlib.sh" --small

        echo "Downloading MIPLIB small subset..."
        "${SCRIPT_DIR}/download_miplib.sh" --small
        ;;
    full)
        echo "Downloading full Netlib set..."
        "${SCRIPT_DIR}/download_netlib.sh"

        echo "Downloading full MIPLIB collection..."
        "${SCRIPT_DIR}/download_miplib.sh"
        ;;
    mittelman)
        echo "Downloading full Netlib set..."
        "${SCRIPT_DIR}/download_netlib.sh"

        echo "Downloading Mittelman LP instances..."
        "${SCRIPT_DIR}/download_mittelman_lp.sh" --full

        echo "Downloading MIPLIB 2017 benchmark set (Mittelman MILP)..."
        "${SCRIPT_DIR}/download_miplib.sh" --mittelman
        ;;
    all)
        echo "Downloading full Netlib set..."
        "${SCRIPT_DIR}/download_netlib.sh" --all

        echo "Downloading full MIPLIB collection..."
        "${SCRIPT_DIR}/download_miplib.sh" --all

        echo "Downloading full Mittelman LP set..."
        "${SCRIPT_DIR}/download_mittelman_lp.sh" --all
        ;;
esac

echo "Done downloading test instances."
