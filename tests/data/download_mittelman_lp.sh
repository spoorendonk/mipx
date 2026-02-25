#!/usr/bin/env bash
# Download LP instances used in Hans Mittelman's LPopt benchmark.
# https://plato.asu.edu/ftp/lpopt.html
#
# The LPopt benchmark tests solvers on finding an optimal basic solution.
# Instances come from multiple sources:
#   [1] miplib2010.zib.de          — LP relaxations of MIP instances
#   [2] plato.asu.edu/ftp/lptestset — Mittelman's own collection
#   [3] netlib.org/lp/data          — classic Netlib LP set
#   [4-9] sztaki.hu/~meszaros       — Meszaros LP test sets
#
# Usage:
#   ./tests/data/download_mittelman_lp.sh           # curated subset (~25 instances)
#   ./tests/data/download_mittelman_lp.sh --full    # all publicly known instances
#
# Note: 16 of Mittelman's 65 LP instances are undisclosed. This script
# downloads the ~49 publicly identified instances.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST_DIR="${SCRIPT_DIR}/mittelman_lp"

# --- Instance sources ---
# [2] plato.asu.edu/ftp/lptestset/ — Mittelman's hosted collection
MITTELMAN_BASE="https://plato.asu.edu/ftp/lptestset"
# [4-9] Meszaros collection hosted at CAS
MESZAROS_BASE="https://www.sztaki.hu/~meszaros/public_ftp/lptestset"

# --- Instance lists by source ---
# Instances from Mittelman's lptestset (plato.asu.edu/ftp/lptestset/)
# These are .mps.bz2 files hosted directly by Mittelman.
# Note: pds-* instances live in the pds/ subdirectory (see MITTELMAN_PDS_INSTANCES).
MITTELMAN_INSTANCES=(
    buildingenergy
    cont1
    cont4
    cont11
    datt256
    ex10
    fome13
    graph40-40
    irish-e
    L1_sixm250obs
    Linf_520c
    neos
    neos3
    nug08-3rd
    qap15
    rmine15
    s100
    s250r10
    savsched1
    scpm1
    square41
    support10
)

# Instances available as rail/ subdirectory at Mittelman's site
MITTELMAN_RAIL_INSTANCES=(
    rail02
    rail4284
    rail507
    rail516
    rail582
    rail2586
)

# Instances from Mittelman's pds/ subdirectory
MITTELMAN_PDS_INSTANCES=(
    pds-20
    pds-40
    pds-50
    pds-80
    pds-100
)

# LP relaxations from MIPLIB (used by Mittelman for LP benchmarks)
MIPLIB_LP_INSTANCES=(
    a2864
    fhnw-binschedule0
    fhnw-binschedule1
    neos-3025225-shelon
    neos-5052403-cygnet
    neos-5251015-ogosta
    ns1644855
    ns1687037
    ns1688926
    nug15
)

# Instances from Meszaros collection
MESZAROS_INSTANCES=(
    bdry2
    dlr1
    s82
    dbic1
)

# Curated subset: smaller/medium instances that can solve in < 5 minutes.
# Good for CI and development testing.
SMALL_SET=(
    cont1
    datt256
    ex10
    fome13
    irish-e
    L1_sixm250obs
    Linf_520c
    neos
    neos3
    nug08-3rd
    pds-100
    qap15
    rmine15
    s100
    s250r10
    rail4284
    a2864
    ns1687037
    nug15
    bdry2
)

usage() {
    echo "Usage: $0 [--full]"
    echo "  (default)  Download curated subset (~20 instances, CI-friendly)"
    echo "  --full     Download all publicly known Mittelman LP instances (~49)"
    exit 1
}

FULL_MODE=false
if [[ "${1:-}" == "--full" ]]; then
    FULL_MODE=true
elif [[ -n "${1:-}" ]]; then
    usage
fi

mkdir -p "${DEST_DIR}"

TOTAL=0
FAILED=0

download_instance() {
    local name="$1"
    local url="$2"
    local outfile="${DEST_DIR}/${name}.mps.gz"

    if [[ -f "${outfile}" ]]; then
        return 0
    fi

    echo -n "  ${name}..."
    local tmpfile
    tmpfile=$(mktemp)

    # Try .mps.bz2 first (Mittelman format), then .mps.gz, then .mps
    if curl -sS -f -o "${tmpfile}" "${url}.mps.bz2" 2>/dev/null; then
        bunzip2 -c "${tmpfile}" | gzip -c > "${outfile}"
        rm -f "${tmpfile}"
        echo " ok (bz2)"
        ((TOTAL++)) || true
        return 0
    elif curl -sS -f -o "${tmpfile}" "${url}.mps.gz" 2>/dev/null; then
        mv "${tmpfile}" "${outfile}"
        echo " ok (gz)"
        ((TOTAL++)) || true
        return 0
    elif curl -sS -f -o "${tmpfile}" "${url}.mps" 2>/dev/null; then
        gzip -c "${tmpfile}" > "${outfile}"
        rm -f "${tmpfile}"
        echo " ok (mps)"
        ((TOTAL++)) || true
        return 0
    else
        rm -f "${tmpfile}"
        echo " FAILED"
        ((FAILED++)) || true
        return 1
    fi
}

download_from_mittelman() {
    local name="$1"
    download_instance "${name}" "${MITTELMAN_BASE}/${name}"
}

download_from_mittelman_rail() {
    local name="$1"
    download_instance "${name}" "${MITTELMAN_BASE}/rail/${name}"
}

download_from_mittelman_pds() {
    local name="$1"
    download_instance "${name}" "${MITTELMAN_BASE}/pds/${name}"
}

download_from_miplib() {
    local name="$1"
    local outfile="${DEST_DIR}/${name}.mps.gz"
    if [[ -f "${outfile}" ]]; then
        return 0
    fi
    echo -n "  ${name}..."
    local tmpfile
    tmpfile=$(mktemp)
    # MIPLIB provides .mps.gz directly
    if curl -sS -f -o "${tmpfile}" "https://miplib.zib.de/WebData/instances/${name}.mps.gz" 2>/dev/null; then
        mv "${tmpfile}" "${outfile}"
        echo " ok"
        ((TOTAL++)) || true
        return 0
    else
        rm -f "${tmpfile}"
        echo " FAILED"
        ((FAILED++)) || true
        return 1
    fi
}

download_from_meszaros() {
    local name="$1"
    local outfile="${DEST_DIR}/${name}.mps.gz"
    if [[ -f "${outfile}" ]]; then
        return 0
    fi
    echo -n "  ${name}..."
    local tmpfile
    tmpfile=$(mktemp)

    # Try different Meszaros subdirectories
    for subdir in MISC NEW PROBLEMATIC STOCHLP; do
        if curl -sS -f -o "${tmpfile}" "${MESZAROS_BASE}/${subdir}/${name}.mps.bz2" 2>/dev/null; then
            bunzip2 -c "${tmpfile}" | gzip -c > "${outfile}"
            rm -f "${tmpfile}"
            echo " ok (meszaros/${subdir})"
            ((TOTAL++)) || true
            return 0
        fi
        if curl -sS -f -o "${tmpfile}" "${MESZAROS_BASE}/${subdir}/${name}.mps" 2>/dev/null; then
            gzip -c "${tmpfile}" > "${outfile}"
            rm -f "${tmpfile}"
            echo " ok (meszaros/${subdir})"
            ((TOTAL++)) || true
            return 0
        fi
    done

    rm -f "${tmpfile}"
    echo " FAILED"
    ((FAILED++)) || true
    return 1
}

if [[ "${FULL_MODE}" == "true" ]]; then
    echo "Downloading all publicly known Mittelman LP instances to ${DEST_DIR}/"

    echo "--- Mittelman lptestset ---"
    for name in "${MITTELMAN_INSTANCES[@]}"; do
        download_from_mittelman "${name}"
    done

    echo "--- Mittelman rail instances ---"
    for name in "${MITTELMAN_RAIL_INSTANCES[@]}"; do
        download_from_mittelman_rail "${name}"
    done

    echo "--- Mittelman pds instances ---"
    for name in "${MITTELMAN_PDS_INSTANCES[@]}"; do
        download_from_mittelman_pds "${name}"
    done

    echo "--- MIPLIB LP relaxations ---"
    for name in "${MIPLIB_LP_INSTANCES[@]}"; do
        download_from_miplib "${name}"
    done

    echo "--- Meszaros collection ---"
    for name in "${MESZAROS_INSTANCES[@]}"; do
        download_from_meszaros "${name}"
    done
else
    echo "Downloading Mittelman LP curated subset (${#SMALL_SET[@]} instances) to ${DEST_DIR}/"

    for name in "${SMALL_SET[@]}"; do
        # Try Mittelman first, then MIPLIB, then Meszaros
        if printf '%s\n' "${MITTELMAN_INSTANCES[@]}" | grep -qx "${name}"; then
            download_from_mittelman "${name}"
        elif printf '%s\n' "${MITTELMAN_RAIL_INSTANCES[@]}" | grep -qx "${name}"; then
            download_from_mittelman_rail "${name}"
        elif printf '%s\n' "${MITTELMAN_PDS_INSTANCES[@]}" | grep -qx "${name}"; then
            download_from_mittelman_pds "${name}"
        elif printf '%s\n' "${MIPLIB_LP_INSTANCES[@]}" | grep -qx "${name}"; then
            download_from_miplib "${name}"
        elif printf '%s\n' "${MESZAROS_INSTANCES[@]}" | grep -qx "${name}"; then
            download_from_meszaros "${name}"
        else
            echo "  ${name}... UNKNOWN SOURCE"
            ((FAILED++)) || true
        fi
    done
fi

echo "Done. $(ls "${DEST_DIR}"/*.mps.gz 2>/dev/null | wc -l) instances available."
if [[ ${FAILED} -gt 0 ]]; then
    echo "Warning: ${FAILED} downloads failed."
fi
