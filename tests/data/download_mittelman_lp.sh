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
#   ./tests/data/download_mittelman_lp.sh            # curated subset (~20 instances)
#   ./tests/data/download_mittelman_lp.sh --small    # curated subset (same as default)
#   ./tests/data/download_mittelman_lp.sh --all      # full currently accessible set
#   ./tests/data/download_mittelman_lp.sh --full     # alias for --all
#
# Note: as of 2026-03-01, several historical aliases are no longer available
# on upstream mirrors (for example: dbic1, ns1687037, ns1688926, nug15).

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
# Note: some files are .mps.bz2, others are just .bz2 (no .mps).
# The download_instance function tries both patterns.
MITTELMAN_INSTANCES=(
    a2864
    bdry2
    bharat
    brazil3
    chromaticindex1024-7
    datt256_lp
    dlr1
    dlr2
    Dual2_5000
    ex10
    fhnw-binschedule1
    graph40-40
    irish-electricity
    L1_sixm250obs
    L1_sixm1000obs
    L2CTA3D
    Linf_520c
    neos-3025225
    neos-5052403-cygnet
    neos-5251015
    physiciansched3-3
    Primal2_1000
    qap15
    rmine15
    s82
    s100
    s250r10
    savsched1
    scpm1
    set-cover-model
    square41
    supportcase10
    supportcase19
    thk_48
    thk_63
    tpl-tub-ws1617
    woodlands09
)

# Instances available as rail/ subdirectory at Mittelman's site
MITTELMAN_RAIL_INSTANCES=(
    rail4284
    rail507
    rail516
    rail582
    rail2586
)

# Instances from Mittelman's pds/ subdirectory
MITTELMAN_PDS_INSTANCES=(
    pds-20
    pds-30
    pds-40
    pds-50
    pds-60
    pds-70
    pds-80
    pds-90
    pds-100
)

# Instances from Mittelman's fome/ subdirectory
MITTELMAN_FOME_INSTANCES=(
    fome11
    fome12
    fome13
    fome21
)

# Instances from Mittelman's nug/ subdirectory
MITTELMAN_NUG_INSTANCES=(
    nug08-3rd
    nug20
    nug30
)

# LP relaxations from MIPLIB (not on Mittelman's server)
MIPLIB_LP_INSTANCES=(
    fhnw-binschedule0
    ns1644855
)

# Instances from Meszaros collection currently unavailable from known mirrors.
MESZAROS_INSTANCES=()

# Curated subset: smaller/medium instances that can solve in < 5 minutes.
# Good for CI and development testing.
SMALL_SET=(
    datt256_lp
    ex10
    fome13
    irish-electricity
    L1_sixm250obs
    Linf_520c
    neos-3025225
    nug08-3rd
    pds-100
    qap15
    rmine15
    s100
    s250r10
    rail4284
    a2864
    ns1644855
    fhnw-binschedule0
    bdry2
    s82
    dlr1
)

usage() {
    echo "Usage: $0 [--small|--all|--full]"
    echo "  (default)  Download curated subset (~20 instances, CI-friendly)"
    echo "  --small    Download curated subset (~20 instances, CI-friendly)"
    echo "  --all      Download full currently accessible Mittelman LP set"
    echo "  --full     Alias for --all"
    exit 1
}

FULL_MODE=false
if [[ "${1:-}" == "--small" ]]; then
    FULL_MODE=false
elif [[ "${1:-}" == "--full" ]]; then
    FULL_MODE=true
elif [[ "${1:-}" == "--all" ]]; then
    FULL_MODE=true
elif [[ -n "${1:-}" ]]; then
    usage
fi

mkdir -p "${DEST_DIR}"

TOTAL=0
FAILED=0

is_valid_gzip_file() {
    local file="$1"
    gzip -t "${file}" >/dev/null 2>&1
}

is_probable_mps_text() {
    local file="$1"
    head -n 40 "${file}" | grep -Eq '^[[:space:]]*(\*|NAME|ROWS|COLUMNS|RHS|BOUNDS|RANGES|OBJSENSE|OBJSEN|ENDATA)'
}

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

    # Try .mps.bz2 first (most Mittelman files), then .bz2 (some use this without
    # .mps in filename), then .mps.gz, then .mps
    if curl -sS -f -L -o "${tmpfile}" "${url}.mps.bz2" 2>/dev/null; then
        if bunzip2 -c "${tmpfile}" | gzip -c > "${outfile}"; then
            rm -f "${tmpfile}"
            echo " ok (mps.bz2)"
            ((TOTAL++)) || true
            return 0
        fi
        rm -f "${outfile}"
    elif curl -sS -f -L -o "${tmpfile}" "${url}.bz2" 2>/dev/null; then
        if bunzip2 -c "${tmpfile}" | gzip -c > "${outfile}"; then
            rm -f "${tmpfile}"
            echo " ok (bz2)"
            ((TOTAL++)) || true
            return 0
        fi
        rm -f "${outfile}"
    elif curl -sS -f -L -o "${tmpfile}" "${url}.mps.gz" 2>/dev/null; then
        mv "${tmpfile}" "${outfile}"
        echo " ok (gz)"
        ((TOTAL++)) || true
        return 0
    elif curl -sS -f -L -o "${tmpfile}" "${url}.mps" 2>/dev/null; then
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

download_from_mittelman_fome() {
    local name="$1"
    download_instance "${name}" "${MITTELMAN_BASE}/fome/${name}"
}

download_from_mittelman_nug() {
    local name="$1"
    download_instance "${name}" "${MITTELMAN_BASE}/nug/${name}"
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
    if curl -sS -f -L -o "${tmpfile}" "https://miplib.zib.de/WebData/instances/${name}.mps.gz" 2>/dev/null; then
        if is_valid_gzip_file "${tmpfile}"; then
            mv "${tmpfile}" "${outfile}"
            echo " ok (gz)"
            ((TOTAL++)) || true
            return 0
        fi
        if is_probable_mps_text "${tmpfile}"; then
            gzip -c "${tmpfile}" > "${outfile}"
            rm -f "${tmpfile}"
            echo " ok (mps)"
            ((TOTAL++)) || true
            return 0
        fi
    fi

    rm -f "${tmpfile}"
    echo " FAILED"
    ((FAILED++)) || true
    return 1
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
        if curl -sS -f -L -o "${tmpfile}" "${MESZAROS_BASE}/${subdir}/${name}.mps.bz2" 2>/dev/null; then
            if bunzip2 -c "${tmpfile}" | gzip -c > "${outfile}"; then
                rm -f "${tmpfile}"
                echo " ok (meszaros/${subdir})"
                ((TOTAL++)) || true
                return 0
            fi
            rm -f "${outfile}"
        fi
        if curl -sS -f -L -o "${tmpfile}" "${MESZAROS_BASE}/${subdir}/${name}.mps" 2>/dev/null; then
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
    echo "Downloading full currently accessible Mittelman LP set to ${DEST_DIR}/"

    echo "--- Mittelman lptestset ---"
    for name in "${MITTELMAN_INSTANCES[@]}"; do
        download_from_mittelman "${name}" || true
    done

    echo "--- Mittelman rail instances ---"
    for name in "${MITTELMAN_RAIL_INSTANCES[@]}"; do
        download_from_mittelman_rail "${name}" || true
    done

    echo "--- Mittelman pds instances ---"
    for name in "${MITTELMAN_PDS_INSTANCES[@]}"; do
        download_from_mittelman_pds "${name}" || true
    done

    echo "--- Mittelman fome instances ---"
    for name in "${MITTELMAN_FOME_INSTANCES[@]}"; do
        download_from_mittelman_fome "${name}" || true
    done

    echo "--- Mittelman nug instances ---"
    for name in "${MITTELMAN_NUG_INSTANCES[@]}"; do
        download_from_mittelman_nug "${name}" || true
    done

    echo "--- MIPLIB LP relaxations ---"
    for name in "${MIPLIB_LP_INSTANCES[@]}"; do
        download_from_miplib "${name}" || true
    done

    echo "--- Meszaros collection ---"
    for name in "${MESZAROS_INSTANCES[@]}"; do
        download_from_meszaros "${name}" || true
    done
else
    echo "Downloading Mittelman LP curated subset (${#SMALL_SET[@]} instances) to ${DEST_DIR}/"

    for name in "${SMALL_SET[@]}"; do
        # Try Mittelman first (root, rail, pds, fome, nug), then MIPLIB, then Meszaros
        if printf '%s\n' "${MITTELMAN_INSTANCES[@]}" | grep -qx "${name}"; then
            download_from_mittelman "${name}" || true
        elif printf '%s\n' "${MITTELMAN_RAIL_INSTANCES[@]}" | grep -qx "${name}"; then
            download_from_mittelman_rail "${name}" || true
        elif printf '%s\n' "${MITTELMAN_PDS_INSTANCES[@]}" | grep -qx "${name}"; then
            download_from_mittelman_pds "${name}" || true
        elif printf '%s\n' "${MITTELMAN_FOME_INSTANCES[@]}" | grep -qx "${name}"; then
            download_from_mittelman_fome "${name}" || true
        elif printf '%s\n' "${MITTELMAN_NUG_INSTANCES[@]}" | grep -qx "${name}"; then
            download_from_mittelman_nug "${name}" || true
        elif printf '%s\n' "${MIPLIB_LP_INSTANCES[@]}" | grep -qx "${name}"; then
            download_from_miplib "${name}" || true
        elif printf '%s\n' "${MESZAROS_INSTANCES[@]}" | grep -qx "${name}"; then
            download_from_meszaros "${name}" || true
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
