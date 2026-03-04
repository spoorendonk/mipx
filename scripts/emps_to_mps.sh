#!/usr/bin/env bash
# Convert MPC-compressed Netlib LP files to standard MPS format.
#
# Usage:
#   emps_to_mps.sh file1 [file2 ...]   — convert files, writing file.mps
#   cat file | emps_to_mps.sh           — pipeline mode, writes to stdout
#
# The script downloads and compiles the emps decompressor from netlib.org
# on first use, caching the binary in a local directory.

set -euo pipefail

CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/mipx"
EMPS_BIN="${CACHE_DIR}/emps"
EMPS_SRC_URL="https://www.netlib.org/lp/data/emps.c"

ensure_emps() {
    if [[ -x "${EMPS_BIN}" ]]; then
        return
    fi
    mkdir -p "${CACHE_DIR}"
    local src="${CACHE_DIR}/emps.c"
    echo "Downloading emps.c from ${EMPS_SRC_URL}..." >&2
    curl -sS -f -L -o "${src}" "${EMPS_SRC_URL}"
    echo "Compiling emps..." >&2
    cc -o "${EMPS_BIN}" "${src}"
    echo "emps binary cached at ${EMPS_BIN}" >&2
}

ensure_emps

if [[ $# -eq 0 ]]; then
    # Pipeline mode: read stdin, write stdout.
    "${EMPS_BIN}"
else
    for input in "$@"; do
        if [[ ! -f "${input}" ]]; then
            echo "Error: file not found: ${input}" >&2
            exit 1
        fi
        # Output filename: strip any extension and add .mps.
        base="${input%.*}"
        output="${base}.mps"
        echo "Converting ${input} -> ${output}" >&2
        "${EMPS_BIN}" < "${input}" > "${output}"
    done
fi
