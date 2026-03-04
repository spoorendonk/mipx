#!/usr/bin/env bash
# Download Netlib LP test set (standard MPS files, gzipped).
# Usage: ./tests/data/download_netlib.sh [--small]
#
# --small: Download only a curated subset (fast, suitable for CI).
# Default: Download the full Netlib set.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEST_DIR="${SCRIPT_DIR}/netlib"
EMPS_SCRIPT="${REPO_ROOT}/scripts/emps_to_mps.sh"

# Standard MPS files from SkyLiu0/NETLIB GitHub repo.
NETLIB_BASE_URLS=(
    "https://raw.githubusercontent.com/SkyLiu0/NETLIB/master/feasible"
    "https://www.netlib.org/lp/data"
)

# Canonical instance names -> upstream filename variants.
declare -A SOURCE_NAME_BY_INSTANCE=(
    [pilot-ja]="pilot.ja"
    [vtp-base]="vtp.base"
)

# Curated small subset — representative, fast to solve.
SMALL_SET=(
    adlittle
    afiro
    blend
    kb2
    sc50a
    sc50b
    sc105
    sc205
    share1b
    share2b
    stocfor1
)

# Full Netlib set.
FULL_SET=(
    25fv47
    80bau3b
    adlittle
    afiro
    agg
    agg2
    agg3
    bandm
    beaconfd
    blend
    bnl1
    bnl2
    bore3d
    brandy
    capri
    cycle
    czprob
    d2q06c
    d6cube
    degen2
    degen3
    e226
    etamacro
    fffff800
    finnis
    fit1d
    fit1p
    fit2d
    fit2p
    forplan
    ganges
    gfrd-pnc
    greenbea
    greenbeb
    grow15
    grow22
    grow7
    israel
    kb2
    lotfi
    maros
    maros-r7
    modszk1
    nesm
    perold
    pilot
    pilot4
    pilot87
    pilot-ja
    pilotnov
    recipe
    sc105
    sc205
    sc50a
    sc50b
    scagr25
    scagr7
    scfxm1
    scfxm2
    scfxm3
    scorpion
    scrs8
    scsd1
    scsd6
    scsd8
    sctap1
    sctap2
    sctap3
    seba
    share1b
    share2b
    shell
    ship04l
    ship04s
    ship08l
    ship08s
    ship12l
    ship12s
    sierra
    stair
    standata
    standmps
    stocfor1
    stocfor2
    stocfor3
    truss
    tuff
    vtp-base
    wood1p
    woodw
)

usage() {
    echo "Usage: $0 [--small|--all]"
    echo "  --small   Download only a small subset suitable for CI"
    echo "  --all     Download full Netlib set (same as default)"
    exit 1
}

INSTANCES=("${FULL_SET[@]}")
if [[ "${1:-}" == "--small" ]]; then
    INSTANCES=("${SMALL_SET[@]}")
elif [[ "${1:-}" == "--all" ]]; then
    INSTANCES=("${FULL_SET[@]}")
elif [[ -n "${1:-}" ]]; then
    usage
fi

mkdir -p "${DEST_DIR}"

# Copy the .solu file from tests/data if present.
if [[ -f "${SCRIPT_DIR}/netlib.solu" && ! -f "${DEST_DIR}/netlib.solu" ]]; then
    cp "${SCRIPT_DIR}/netlib.solu" "${DEST_DIR}/netlib.solu"
fi

echo "Downloading ${#INSTANCES[@]} Netlib instances to ${DEST_DIR}/"

FAILED=0
for name in "${INSTANCES[@]}"; do
    outfile="${DEST_DIR}/${name}.mps.gz"
    if [[ -f "${outfile}" ]]; then
        continue
    fi
    source_name="${SOURCE_NAME_BY_INSTANCE[${name}]:-${name}}"
    echo -n "  ${name}..."
    tmpfile=$(mktemp)
    downloaded=false
    for base_url in "${NETLIB_BASE_URLS[@]}"; do
        for suffix in ".mps" ".MPS" ""; do
            if curl -sS -f -L -o "${tmpfile}" "${base_url}/${source_name}${suffix}" 2>/dev/null; then
                # Detect MPC-compressed files (netlib.org raw format).
                # MPC files do not start with standard MPS section headers.
                first_word=$(head -c 80 "${tmpfile}" | awk 'NR==1{print $1}')
                if [[ "${first_word}" != "NAME" && "${first_word}" != "ROWS" && \
                      "${base_url}" == *"netlib.org"* && -x "${EMPS_SCRIPT}" ]]; then
                    # Pipe through emps to convert MPC -> standard MPS.
                    if "${EMPS_SCRIPT}" < "${tmpfile}" 2>/dev/null | gzip -c > "${outfile}"; then
                        downloaded=true
                        break
                    fi
                else
                    gzip -c "${tmpfile}" > "${outfile}"
                    downloaded=true
                    break
                fi
            fi
        done
        if [[ "${downloaded}" == "true" ]]; then
            break
        fi
    done
    rm -f "${tmpfile}"
    if [[ "${downloaded}" == "true" ]]; then
        echo " ok"
    else
        echo " FAILED"
        ((FAILED++)) || true
    fi
done

echo "Done. $(ls "${DEST_DIR}"/*.mps.gz 2>/dev/null | wc -l) instances available."
if [[ ${FAILED} -gt 0 ]]; then
    echo "Warning: ${FAILED} downloads failed."
fi
