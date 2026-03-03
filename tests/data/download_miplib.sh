#!/usr/bin/env bash
# Download MIPLIB 2017 instances from the collection.zip archive.
# Usage: ./tests/data/download_miplib.sh [--small|--mittelman]
#
# --small:      Download only a curated small subset (fast, suitable for CI).
# --mittelman:  Download the 240 MIPLIB 2017 benchmark instances used by
#               Hans Mittelman (plato.asu.edu/ftp/milp.html).
# Default:      Download the full MIPLIB 2017 collection (~1065 instances).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST_DIR="${SCRIPT_DIR}/miplib"

COLLECTION_URL="https://miplib.zib.de/downloads/collection.zip"
SOLU_URL="https://miplib.zib.de/downloads/miplib2017-v23.solu"
BENCHMARK_TAG_URL="https://miplib.zib.de/tag_benchmark.html"

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

# Fallback benchmark set used when dynamic benchmark discovery fails.
# Source: https://miplib.zib.de/tag_benchmark.html (older snapshot).
# The dynamic list parsed from MIPLIB webpage is preferred.
BENCHMARK_SET=(
    10teams 30n20b8 a1c1s1 aflow30a aflow40b
    air04 air05 app1-2 arki001 ash608gpia-3col
    bab5 beasleyC3 bell5 biella1 binkar10_1
    blend2 bley_xl1 bnatt350 cap6000 cdc7-4-3-2
    chromaticindex1024-7 chromaticindex32-8 cmflsp50-24-8-8
    co-100 cod105 control40-5-2-1 core2536-691
    core4872-1529 cost266-fac-3 cplex2 csched010
    cvs16r89-60L cvs16r106-72L d10200 d20200
    dc1l dano3_3 dano3_4 dano3_5 dano3mip
    dcmulti dg012142 disctom ds-big eil33-2
    eilB101 enlight13 enlight14 enigma ex1010-pi
    ex9 exp-1-500-5-5 f2000 fhnw-binpack4-48
    fhnw-binschedule0 fhnw-binschedule1 fhnw-binschedule2
    fiball fixnet6 flugpl gen gen-ip002
    gen-ip021 gen-ip054 gesa2 gesa2-o gesa3
    gesa3-o glass4 gmu-35-40 gmu-35-50 go19
    gp graph40-20 graph40-40 graphdraw-domain
    graphdraw-gemcutter graphdraw-grafo gt2 haprp
    harp2 iis-100-0-cov iis-bupa-cov iis-pima-cov
    lectsched-4-obj liu m100n500k4r1 macrophage
    map18 map20 markshare1 markshare2 mas74
    mas76 mcsched mc11 methanosarcina mik-250-1-100-1
    mine-166-5 mine-90-10 misc03 misc06 misc07
    mod008 mod010 mod011 momentum1 momentum2
    momentum3 msc98-ip mzzv11 mzzv42z n3div36
    n3seq24 n5-3 n9-3 neos-1109824 neos-1122047
    neos-1140050 neos-1171448 neos-1171692
    neos-1224597 neos-1228986 neos-1311124
    neos-1337307 neos-1396125 neos-1426635
    neos-1426662 neos-1440225 neos-1440460
    neos-1467511 neos-1601936 neos-2657525-crna
    neos-2978193-inde neos-3004026-krka neos-3024952-loue
    neos-3025225-shelon neos-3065745-stann neos-3075395-selune
    neos-3083819-nuhou neos-3116779-oise neos-3135526-osun
    neos-3209462-rhin neos-3216931-puriri
    neos-3381206-awhea neos-3421095-turia
    neos-3426085-ticino neos-3530903-gauja
    neos-3555904-turama neos-3610040-quoich
    neos-3610051-quella neos-3627168-naan
    neos-3660371-siret neos-3754224-navua
    neos-3754480-nidda neos-4338804-snowy
    neos-4387871-tavua neos-4409277-trave
    neos-4647027-thurso neos-4650160-yukon
    neos-4722843-widden neos-4760493-puerua
    neos-4954672-berkel neos-5049753-pigou
    neos-5052403-cygnet neos-5075914-elvire
    neos-5107597-kakapo neos-5140963-mincio
    neos-5188808-nattai neos-5195221-niemur
    neos-5251015-ogosta neos-5261882-treska
    neos11 neos16 neos18 neos6 neos8 neos9
    net12 netdiversion noswot nsr8k ns1456591
    ns1644855 ns1688926 ns1766074 ns1769397
    ns1830653 ns1856153 nw04 opm2-z7-s2 opm2-z12-s14
    p0033 p0201 p0282 p0548 p2756 pg pg5_34
    pigeon-10 pk1 pp08a pp08aCUTS protfold
    pw-myciel4 qiu qnet1 qnet1_o r80x800
    rail01 rail02 ran13x13 ran14x18-disj-8
    reblock115 reblock166 reblock354 reblock420
    reblock67 rentacar rgn roi2alpha3n4 roi5alpha5n8
    rococoB10-011000 rococoC10-001000 rococoC11-011100
    roll3000 rout rvb-sub satellites1-25
    seqsolve set1ch set3-10 set3-15 set3-20
    sp97ar sp97ic stein27 stein45 stein9inf
    supportcase18 supportcase3 swath t1717
    tanglegram1 tanglegram2 timtab1 toll-like
    tr12-30 transportmoment triptim1 triptim2
    uccase12 uccase9 unitcal_7 vpm2 vs13
)

# Mittelman medium subset — instances solvable by HiGHS in < 30min.
# Good for regression testing against HiGHS on the Mittelman MILP instance set.
MITTELMAN_SMALL=(
    air04 air05 bell5 blend2 cap6000 dcmulti
    eilB101 fixnet6 flugpl gen gen-ip002 gesa2
    gesa2-o gesa3 gesa3-o glass4 gt2 harp2
    markshare1 markshare2 mas74 mas76 misc03
    misc06 misc07 mod008 mod010 mod011
    noswot nw04 p0033 p0201 p0282 p0548 p2756
    pk1 pp08a pp08aCUTS qiu qnet1 qnet1_o
    ran13x13 rentacar rgn rout set1ch
    stein27 stein45 stein9inf timtab1 vpm2
)

usage() {
    echo "Usage: $0 [--small|--mittelman|--mittelman-small|--all]"
    echo "  --small             Download only a small subset suitable for CI (12 instances)"
    echo "  --mittelman         Download the MIPLIB 2017 benchmark set (Mittelman MILP)"
    echo "  --mittelman-small   Download ~50 Mittelman-set instances solvable by HiGHS in <30min"
    echo "  --all               Download full MIPLIB 2017 collection (~1065 instances)"
    echo "  (default)           Same as --all"
    exit 1
}

MODE="full"
if [[ "${1:-}" == "--small" ]]; then
    MODE="small"
elif [[ "${1:-}" == "--mittelman" ]]; then
    MODE="mittelman"
elif [[ "${1:-}" == "--mittelman-small" ]]; then
    MODE="mittelman-small"
elif [[ "${1:-}" == "--all" ]]; then
    MODE="full"
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

# Build zip index by instance basename -> zip entry path so named extraction works
# regardless of archive folder layout.
declare -A ZIP_ENTRY_BY_NAME
while IFS= read -r entry; do
    [[ "${entry}" == *.mps.gz ]] || continue
    base="$(basename "${entry}")"
    name="${base%.mps.gz}"
    if [[ -z "${ZIP_ENTRY_BY_NAME[${name}]:-}" ]]; then
        ZIP_ENTRY_BY_NAME["${name}"]="${entry}"
    fi
done < <(unzip -Z1 "${ZIP_FILE}")

extract_named_set() {
    local -n set_ref=$1
    local missing=0
    echo "Extracting ${#set_ref[@]} instances to ${DEST_DIR}/"
    for name in "${set_ref[@]}"; do
        outfile="${DEST_DIR}/${name}.mps.gz"
        if [[ -f "${outfile}" ]]; then
            continue
        fi
        echo -n "  ${name}..."
        local entry="${ZIP_ENTRY_BY_NAME[${name}]:-}"
        if [[ -n "${entry}" ]] && unzip -p "${ZIP_FILE}" "${entry}" > "${outfile}"; then
            echo " ok"
        elif unzip -j -o "${ZIP_FILE}" "${name}.mps.gz" -d "${DEST_DIR}/" >/dev/null 2>&1; then
            # Fallback for legacy flat archives.
            echo " ok"
        else
            echo " not found"
            ((missing++)) || true
        fi
    done
    if [[ ${missing} -gt 0 ]]; then
        echo "Warning: ${missing} requested instances were not found in collection.zip"
    fi
}

load_dynamic_benchmark_set() {
    local tmpfile
    tmpfile=$(mktemp)
    local -n out_ref=$1
    out_ref=()
    if ! curl -sS -f -L -o "${tmpfile}" "${BENCHMARK_TAG_URL}" 2>/dev/null; then
        rm -f "${tmpfile}"
        return 1
    fi

    mapfile -t out_ref < <(
        grep -oE 'instance_details_[A-Za-z0-9._+-]+\.html' "${tmpfile}" \
            | sed 's/^instance_details_//; s/\.html$//' \
            | sort -u
    )
    rm -f "${tmpfile}"
    [[ ${#out_ref[@]} -gt 0 ]]
}

case "${MODE}" in
    small)
        extract_named_set SMALL_SET
        ;;
    mittelman)
        dynamic_set=()
        if load_dynamic_benchmark_set dynamic_set; then
            echo "Using dynamic benchmark list from ${BENCHMARK_TAG_URL} (${#dynamic_set[@]} instances)"
            extract_named_set dynamic_set
        else
            echo "Warning: failed to fetch dynamic benchmark list; using fallback list (${#BENCHMARK_SET[@]} instances)"
            extract_named_set BENCHMARK_SET
        fi
        ;;
    mittelman-small)
        extract_named_set MITTELMAN_SMALL
        ;;
    full)
        echo "Extracting all instances to ${DEST_DIR}/"
        unzip -j -o "${ZIP_FILE}" "*.mps.gz" -d "${DEST_DIR}/" 2>/dev/null || true
        ;;
esac

echo "Done. $(ls "${DEST_DIR}"/*.mps.gz 2>/dev/null | wc -l) instances available."
