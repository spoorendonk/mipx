#!/usr/bin/env bash
# Generate HiGHS (highspy) + mipx baselines for Mittelman benchmark sets.
#
# This produces baselines for both LP (Mittelman LPopt instances) and
# MIP (MIPLIB 2017 benchmark set) matching Mittelman's configuration.
#
# Usage:
#   ./tests/perf/generate_mittelman_baselines.sh [--highs-only] [--mipx-only]

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${ROOT_DIR}/tests/perf/baselines"
MITTELMAN_LP_DIR="${ROOT_DIR}/tests/data/mittelman_lp"
NETLIB_DIR="${ROOT_DIR}/tests/data/netlib"
MIPLIB_DIR="${ROOT_DIR}/tests/data/miplib"
BIN="${ROOT_DIR}/build/mipx-solve"

# Default: generate both HiGHS and mipx baselines
HIGHS=true
MIPX=true
if [[ "${1:-}" == "--highs-only" ]]; then
    MIPX=false
elif [[ "${1:-}" == "--mipx-only" ]]; then
    HIGHS=false
fi

mkdir -p "${OUT_DIR}"

# --- Mittelman-matching parameters ---
# LP: 15000s time limit (we use 600s for baselines — enough for HiGHS)
LP_TIME_LIMIT=600
LP_REPEATS=3

# MIP: 7200s time limit, 8 threads (we use smaller subset for baselines)
MIP_TIME_LIMIT=7200
MIP_THREADS=8
MIP_REPEATS=1
MIP_GAP_TOL=1e-4

# Mittelman-small MIP instance set — instances HiGHS can typically solve in <30min.
MITTELMAN_MIP_INSTANCES="air04,air05,bell5,blend2,dcmulti,flugpl,gen,gesa2,gesa2-o,glass4,gt2,misc03,misc06,misc07,mod008,mod010,noswot,p0033,p0201,p0282,p0548,pk1,pp08a,pp08aCUTS,qiu,qnet1,qnet1_o,ran13x13,rentacar,rgn,set1ch,stein27,stein45,stein9inf,vpm2"

# --- HiGHS baselines ---
if [[ "${HIGHS}" == "true" ]]; then
    echo "=== Generating HiGHS Mittelman LP baseline ==="

    HIGHS_LP_OUT="${OUT_DIR}/highspy_lp_mittelman.csv"
    python3 "${ROOT_DIR}/tests/perf/run_highspy_bench.py" \
        --mode lp \
        --instances-dir "${MITTELMAN_LP_DIR}" \
        --output "${HIGHS_LP_OUT}" \
        --repeats "${LP_REPEATS}" \
        --threads 1 \
        --time-limit "${LP_TIME_LIMIT}" \
        --solver simplex \
        --simplex-strategy 1 \
        --presolve choose 2>/dev/null || echo "Warning: HiGHS LP Mittelman baseline skipped (instances may be missing)"

    # Also run on Netlib instances (included in Mittelman LPopt)
    HIGHS_LP_NETLIB="${OUT_DIR}/highspy_lp_netlib_full.csv"
    if [[ -d "${NETLIB_DIR}" ]] && ls "${NETLIB_DIR}"/*.mps.gz >/dev/null 2>&1; then
        python3 "${ROOT_DIR}/tests/perf/run_highspy_bench.py" \
            --mode lp \
            --instances-dir "${NETLIB_DIR}" \
            --output "${HIGHS_LP_NETLIB}" \
            --repeats "${LP_REPEATS}" \
            --threads 1 \
            --time-limit "${LP_TIME_LIMIT}" \
            --solver simplex \
            --simplex-strategy 1 \
            --presolve choose 2>/dev/null || echo "Warning: HiGHS LP Netlib baseline skipped"
    fi

    echo "=== Generating HiGHS Mittelman MIP baseline ==="

    HIGHS_MIP_OUT="${OUT_DIR}/highspy_mip_mittelman.csv"
    python3 "${ROOT_DIR}/tests/perf/run_highspy_bench.py" \
        --mode mip \
        --instances-dir "${MIPLIB_DIR}" \
        --instances "${MITTELMAN_MIP_INSTANCES}" \
        --output "${HIGHS_MIP_OUT}" \
        --repeats "${MIP_REPEATS}" \
        --threads "${MIP_THREADS}" \
        --time-limit "${MIP_TIME_LIMIT}" \
        --mip-rel-gap "${MIP_GAP_TOL}" \
        --presolve choose \
        --solver choose 2>/dev/null || echo "Warning: HiGHS MIP Mittelman baseline skipped (instances may be missing)"

    # Write metadata
    HIGHS_META="${OUT_DIR}/highspy_mittelman_meta.json"
    python3 - <<'PY' "${HIGHS_META}"
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
import highspy

meta = {
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "python": sys.version.split()[0],
    "highspy_version": highspy.Highs().version(),
    "platform": platform.platform(),
    "benchmark_config": {
        "lp_time_limit": 600,
        "lp_repeats": 3,
        "mip_time_limit": 7200,
        "mip_threads": 8,
        "mip_gap_tol": 1e-4,
        "reference": "https://plato.asu.edu/bench.html",
    },
}
try:
    cpu = subprocess.check_output(["bash", "-lc", "lscpu | head -n 20"], text=True)
    meta["cpu_info"] = cpu
except Exception:
    pass

with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)
PY

    echo "Wrote HiGHS Mittelman baselines:"
    [[ -f "${HIGHS_LP_OUT}" ]] && echo "  ${HIGHS_LP_OUT}"
    [[ -f "${HIGHS_LP_NETLIB}" ]] && echo "  ${HIGHS_LP_NETLIB}"
    [[ -f "${HIGHS_MIP_OUT}" ]] && echo "  ${HIGHS_MIP_OUT}"
    echo "  ${HIGHS_META}"
fi

# --- mipx baselines ---
if [[ "${MIPX}" == "true" ]]; then
    if [[ ! -x "${BIN}" ]]; then
        echo "mipx binary not found: ${BIN}" >&2
        echo "Build first: cmake --build build -j\$(nproc)" >&2
        exit 1
    fi

    echo "=== Generating mipx Mittelman LP baseline ==="

    MIPX_LP_OUT="${OUT_DIR}/mipx_lp_mittelman.csv"
    "${ROOT_DIR}/tests/perf/run_mittelman_lp_bench.sh" \
        --binary "${BIN}" \
        --mittelman-dir "${MITTELMAN_LP_DIR}" \
        --netlib-dir "${NETLIB_DIR}" \
        --output "${MIPX_LP_OUT}" \
        --repeats "${LP_REPEATS}" \
        --time-limit "${LP_TIME_LIMIT}" \
        --solver-arg --quiet

    echo "=== Generating mipx Mittelman MIP baseline ==="

    MIPX_MIP_OUT="${OUT_DIR}/mipx_mip_mittelman.csv"
    "${ROOT_DIR}/tests/perf/run_mittelman_mip_bench.sh" \
        --binary "${BIN}" \
        --miplib-dir "${MIPLIB_DIR}" \
        --output "${MIPX_MIP_OUT}" \
        --threads "${MIP_THREADS}" \
        --time-limit "${MIP_TIME_LIMIT}" \
        --gap-tol "${MIP_GAP_TOL}" \
        --instances "${MITTELMAN_MIP_INSTANCES}" \
        --repeats "${MIP_REPEATS}" \
        --solver-arg --quiet

    # Write metadata
    MIPX_META="${OUT_DIR}/mipx_mittelman_meta.json"
    python3 - <<'PY' "${MIPX_META}" "${ROOT_DIR}"
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

meta_path = Path(sys.argv[1])
repo = Path(sys.argv[2])

def cmd(text: str) -> str:
    return subprocess.check_output(["bash", "-lc", text], cwd=repo, text=True).strip()

meta = {
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "git_commit": cmd("git rev-parse HEAD"),
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "benchmark_config": {
        "lp_time_limit": 600,
        "lp_repeats": 3,
        "mip_time_limit": 7200,
        "mip_threads": 8,
        "mip_gap_tol": 1e-4,
        "reference": "https://plato.asu.edu/bench.html",
    },
}
try:
    meta["cpu_info"] = cmd("lscpu | head -n 20")
except Exception:
    pass

with meta_path.open("w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)
PY

    echo "Wrote mipx Mittelman baselines:"
    echo "  ${MIPX_LP_OUT}"
    echo "  ${MIPX_MIP_OUT}"
    echo "  ${MIPX_META}"
fi
