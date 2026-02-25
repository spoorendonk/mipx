#!/usr/bin/env bash
# Run LP + MIP performance gates end-to-end with one command.
#
# Usage:
#   ./tests/perf/run_full_gate.sh \
#       --candidate-binary ./build/mipx-solve \
#       --baseline-binary /tmp/mipx_main/build/mipx-solve \
#       --netlib-dir ./tests/data/netlib \
#       --miplib-dir ./tests/data/miplib

set -euo pipefail

CAND_BIN=""
BASE_BIN=""
NETLIB_DIR=""
MIPLIB_DIR=""
OUT_DIR="/tmp/mipx_fullgate"

LP_REPEATS=3
MIP_REPEATS=1
THREADS=1
TIME_LIMIT=30
NODE_LIMIT=100000
GAP_TOL=1e-4
MIP_INSTANCES="p0201,pk1,gt2"

METRIC="work_units"
MAX_REGRESSION_PCT=0.0
LP_MIN_COMMON=5
MIP_MIN_COMMON=3

BASELINE_LP_CSV=""
BASELINE_MIP_CSV=""
EXTRA_ARGS=()
SOLVER_ARG_FLAGS=()

usage() {
    cat <<EOF >&2
Usage: $0 \\
  --candidate-binary <path> \\
  --netlib-dir <dir> \\
  --miplib-dir <dir> \\
  [--baseline-binary <path>] \\
  [--baseline-lp-csv <csv>] \\
  [--baseline-mip-csv <csv>] \\
  [--out-dir <dir>] \\
  [--lp-repeats <n>] [--mip-repeats <n>] \\
  [--threads <n>] [--time-limit <sec>] [--node-limit <n>] [--gap-tol <g>] \\
  [--mip-instances <a,b,c>] \\
  [--metric <name>] [--max-regression-pct <pct>] \\
  [--lp-min-common <n>] [--mip-min-common <n>] \\
  [--solver-arg <arg>]
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --candidate-binary)
            CAND_BIN="$2"
            shift 2
            ;;
        --baseline-binary)
            BASE_BIN="$2"
            shift 2
            ;;
        --netlib-dir)
            NETLIB_DIR="$2"
            shift 2
            ;;
        --miplib-dir)
            MIPLIB_DIR="$2"
            shift 2
            ;;
        --out-dir)
            OUT_DIR="$2"
            shift 2
            ;;
        --lp-repeats)
            LP_REPEATS="$2"
            shift 2
            ;;
        --mip-repeats)
            MIP_REPEATS="$2"
            shift 2
            ;;
        --threads)
            THREADS="$2"
            shift 2
            ;;
        --time-limit)
            TIME_LIMIT="$2"
            shift 2
            ;;
        --node-limit)
            NODE_LIMIT="$2"
            shift 2
            ;;
        --gap-tol)
            GAP_TOL="$2"
            shift 2
            ;;
        --mip-instances)
            MIP_INSTANCES="$2"
            shift 2
            ;;
        --metric)
            METRIC="$2"
            shift 2
            ;;
        --max-regression-pct)
            MAX_REGRESSION_PCT="$2"
            shift 2
            ;;
        --lp-min-common)
            LP_MIN_COMMON="$2"
            shift 2
            ;;
        --mip-min-common)
            MIP_MIN_COMMON="$2"
            shift 2
            ;;
        --baseline-lp-csv)
            BASELINE_LP_CSV="$2"
            shift 2
            ;;
        --baseline-mip-csv)
            BASELINE_MIP_CSV="$2"
            shift 2
            ;;
        --solver-arg)
            EXTRA_ARGS+=("$2")
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

for arg in "${EXTRA_ARGS[@]}"; do
    SOLVER_ARG_FLAGS+=(--solver-arg "${arg}")
done

if [[ -z "${CAND_BIN}" || -z "${NETLIB_DIR}" || -z "${MIPLIB_DIR}" ]]; then
    usage
    exit 1
fi

if [[ ! -x "${CAND_BIN}" ]]; then
    echo "Candidate binary not executable: ${CAND_BIN}" >&2
    exit 1
fi

if [[ ! -d "${NETLIB_DIR}" ]]; then
    echo "Netlib directory not found: ${NETLIB_DIR}" >&2
    exit 1
fi

if [[ ! -d "${MIPLIB_DIR}" ]]; then
    echo "MIPLIB directory not found: ${MIPLIB_DIR}" >&2
    exit 1
fi

if [[ -z "${BASELINE_LP_CSV}" || -z "${BASELINE_MIP_CSV}" ]]; then
    if [[ -z "${BASE_BIN}" || ! -x "${BASE_BIN}" ]]; then
        echo "Need --baseline-binary when baseline CSVs are not fully provided." >&2
        exit 1
    fi
fi

mkdir -p "${OUT_DIR}"

CAND_LP_CSV="${OUT_DIR}/candidate_lp.csv"
CAND_MIP_CSV="${OUT_DIR}/candidate_mip.csv"
BASE_LP_CSV="${BASELINE_LP_CSV:-${OUT_DIR}/baseline_lp.csv}"
BASE_MIP_CSV="${BASELINE_MIP_CSV:-${OUT_DIR}/baseline_mip.csv}"

echo "[fullgate] Running candidate LP bench..."
./tests/perf/run_netlib_lp_bench.sh \
    --binary "${CAND_BIN}" \
    --netlib-dir "${NETLIB_DIR}" \
    --output "${CAND_LP_CSV}" \
    --repeats "${LP_REPEATS}" \
    "${SOLVER_ARG_FLAGS[@]}"

echo "[fullgate] Running candidate MIP bench..."
./tests/perf/run_miplib_mip_bench.sh \
    --binary "${CAND_BIN}" \
    --miplib-dir "${MIPLIB_DIR}" \
    --output "${CAND_MIP_CSV}" \
    --repeats "${MIP_REPEATS}" \
    --threads "${THREADS}" \
    --time-limit "${TIME_LIMIT}" \
    --node-limit "${NODE_LIMIT}" \
    --gap-tol "${GAP_TOL}" \
    --instances "${MIP_INSTANCES}" \
    "${SOLVER_ARG_FLAGS[@]}"

if [[ -z "${BASELINE_LP_CSV}" ]]; then
    echo "[fullgate] Running baseline LP bench..."
    ./tests/perf/run_netlib_lp_bench.sh \
        --binary "${BASE_BIN}" \
        --netlib-dir "${NETLIB_DIR}" \
        --output "${BASE_LP_CSV}" \
        --repeats "${LP_REPEATS}" \
        "${SOLVER_ARG_FLAGS[@]}"
fi

if [[ -z "${BASELINE_MIP_CSV}" ]]; then
    echo "[fullgate] Running baseline MIP bench..."
    ./tests/perf/run_miplib_mip_bench.sh \
        --binary "${BASE_BIN}" \
        --miplib-dir "${MIPLIB_DIR}" \
        --output "${BASE_MIP_CSV}" \
        --repeats "${MIP_REPEATS}" \
        --threads "${THREADS}" \
        --time-limit "${TIME_LIMIT}" \
        --node-limit "${NODE_LIMIT}" \
        --gap-tol "${GAP_TOL}" \
        --instances "${MIP_INSTANCES}" \
        "${SOLVER_ARG_FLAGS[@]}"
fi

echo "[fullgate] Checking LP regression (${METRIC})..."
python3 tests/perf/check_regression.py \
    --baseline "${BASE_LP_CSV}" \
    --candidate "${CAND_LP_CSV}" \
    --metric "${METRIC}" \
    --max-regression-pct "${MAX_REGRESSION_PCT}" \
    --min-common-instances "${LP_MIN_COMMON}"

echo "[fullgate] Checking MIP regression (${METRIC})..."
python3 tests/perf/check_regression.py \
    --baseline "${BASE_MIP_CSV}" \
    --candidate "${CAND_MIP_CSV}" \
    --metric "${METRIC}" \
    --max-regression-pct "${MAX_REGRESSION_PCT}" \
    --min-common-instances "${MIP_MIN_COMMON}"

echo "[fullgate] PASS"
echo "[fullgate] LP baseline: ${BASE_LP_CSV}"
echo "[fullgate] LP candidate: ${CAND_LP_CSV}"
echo "[fullgate] MIP baseline: ${BASE_MIP_CSV}"
echo "[fullgate] MIP candidate: ${CAND_MIP_CSV}"
