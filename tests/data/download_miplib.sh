#!/usr/bin/env bash
# Download MIPLIB 2017 instances from the collection.zip archive.
# Usage: ./tests/data/download_miplib.sh [--small]
#
# --small: Download only a curated small subset (fast, suitable for CI).
# Default: Download the full MIPLIB 2017 benchmark set.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST_DIR="${SCRIPT_DIR}/miplib"

COLLECTION_URL="https://miplib.zib.de/downloads/collection.zip"
SOLU_URL="https://miplib.zib.de/downloads/miplib2017-v23.solu"

# Curated small subset — instances in the MIPLIB 2017 collection, small/fast.
SMALL_SET=(
    air04
    blend2
    dcmulti
    flugpl
    gen
    gt2
    mod010
    noswot
    p0201
    pk1
    ran13x13
    stein9inf
)

# Larger benchmark subset (all MIPLIB 2017 benchmark set).
BENCHMARK_SET=(
    # Empty = extract all from collection.zip
)

usage() {
    echo "Usage: $0 [--small]"
    echo "  --small   Download only a small subset suitable for CI"
    exit 1
}

SMALL_MODE=false
if [[ "${1:-}" == "--small" ]]; then
    SMALL_MODE=true
elif [[ -n "${1:-}" ]]; then
    usage
fi

mkdir -p "${DEST_DIR}"

# Download the .solu file.
SOLU_FILE="${DEST_DIR}/miplib.solu"
if [[ ! -f "${SOLU_FILE}" ]]; then
    echo "Downloading MIPLIB solution file..."
    curl -sSL -o "${SOLU_FILE}" "${SOLU_URL}" 2>/dev/null || true
fi

# Download the collection zip to a cache location.
CACHE_DIR="${SCRIPT_DIR}/.cache"
mkdir -p "${CACHE_DIR}"
ZIP_FILE="${CACHE_DIR}/miplib_collection.zip"

if [[ ! -f "${ZIP_FILE}" ]]; then
    echo "Downloading MIPLIB 2017 collection (~300MB)..."
    curl -sSL -f -o "${ZIP_FILE}" "${COLLECTION_URL}" || {
        echo "Failed to download collection.zip"
        exit 1
    }
fi

if [[ "${SMALL_MODE}" == "true" ]]; then
    echo "Extracting ${#SMALL_SET[@]} instances to ${DEST_DIR}/"
    for name in "${SMALL_SET[@]}"; do
        outfile="${DEST_DIR}/${name}.mps.gz"
        if [[ -f "${outfile}" ]]; then
            continue
        fi
        echo -n "  ${name}..."
        if unzip -j -o "${ZIP_FILE}" "${name}.mps.gz" -d "${DEST_DIR}/" >/dev/null 2>&1; then
            echo " ok"
        else
            echo " not found"
        fi
    done
else
    echo "Extracting all instances to ${DEST_DIR}/"
    unzip -j -o "${ZIP_FILE}" "*.mps.gz" -d "${DEST_DIR}/" 2>/dev/null || true
fi

echo "Done. $(ls "${DEST_DIR}"/*.mps.gz 2>/dev/null | wc -l) instances available."
