#!/usr/bin/env bash
# Run Mittelman-style LP + MIP performance regression gate.
#
# Matches Hans Mittelman's benchmark configuration:
#   LP (LPopt):  single-threaded simplex, 15000s time limit, ~65 instances
#   MIP (MILP):  8 threads, 7200s time limit, MIPLIB 2017 benchmark set
#
# Compares candidate mipx build against either:
#   a) A baseline mipx build (self-regression)
#   b) HiGHS baselines (cross-solver comparison, informational)
#
# Usage:
#   # Self-regression (candidate vs baseline mipx):
#   ./tests/perf/run_mittelman_gate.sh \
#       --candidate-binary ./build/mipx-solve \
#       --baseline-binary /tmp/mipx_main/build/mipx-solve
#
#   # Against stored HiGHS baselines:
#   ./tests/perf/run_mittelman_gate.sh \
#       --candidate-binary ./build/mipx-solve \
#       --baseline-lp-csv tests/perf/baselines/highspy_lp_mittelman.csv \
#       --baseline-mip-csv tests/perf/baselines/highspy_mip_mittelman.csv \
#       --metric time_seconds \
#       --max-regression-pct 100000
#
#   # Against stored mipx baselines (strict work_units gate):
#   ./tests/perf/run_mittelman_gate.sh \
#       --candidate-binary ./build/mipx-solve \
#       --baseline-lp-csv tests/perf/baselines/mipx_lp_mittelman.csv \
#       --baseline-mip-csv tests/perf/baselines/mipx_mip_mittelman.csv

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CAND_BIN=""
BASE_BIN=""
OUT_DIR="/tmp/mipx_mittelman_gate"

# Mittelman LP parameters
MITTELMAN_LP_DIR="${ROOT_DIR}/tests/data/mittelman_lp"
NETLIB_DIR="${ROOT_DIR}/tests/data/netlib"
LP_REPEATS=3
LP_TIME_LIMIT=15000

# Mittelman MIP parameters
MIPLIB_DIR="${ROOT_DIR}/tests/data/miplib"
MIP_REPEATS=1
MIP_THREADS=8
MIP_TIME_LIMIT=7200
MIP_GAP_TOL=1e-4
MIP_INSTANCES=""  # empty = all available

# Gate parameters
METRIC="work_units"
MAX_REGRESSION_PCT=0.0
LP_MIN_COMMON=5
MIP_MIN_COMMON=3

# Pre-computed baselines (optional)
BASELINE_LP_CSV=""
BASELINE_MIP_CSV=""
EXTRA_ARGS=()
SOLVER_ARG_FLAGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --candidate-binary)   CAND_BIN="$2";        shift 2 ;;
        --baseline-binary)    BASE_BIN="$2";         shift 2 ;;
        --out-dir)            OUT_DIR="$2";          shift 2 ;;
        --mittelman-lp-dir)   MITTELMAN_LP_DIR="$2"; shift 2 ;;
        --netlib-dir)         NETLIB_DIR="$2";       shift 2 ;;
        --miplib-dir)         MIPLIB_DIR="$2";       shift 2 ;;
        --lp-repeats)         LP_REPEATS="$2";       shift 2 ;;
        --mip-repeats)        MIP_REPEATS="$2";      shift 2 ;;
        --lp-time-limit)      LP_TIME_LIMIT="$2";    shift 2 ;;
        --mip-time-limit)     MIP_TIME_LIMIT="$2";   shift 2 ;;
        --mip-threads)        MIP_THREADS="$2";      shift 2 ;;
        --mip-instances)      MIP_INSTANCES="$2";    shift 2 ;;
        --metric)             METRIC="$2";           shift 2 ;;
        --max-regression-pct) MAX_REGRESSION_PCT="$2"; shift 2 ;;
        --lp-min-common)      LP_MIN_COMMON="$2";    shift 2 ;;
        --mip-min-common)     MIP_MIN_COMMON="$2";   shift 2 ;;
        --baseline-lp-csv)    BASELINE_LP_CSV="$2";  shift 2 ;;
        --baseline-mip-csv)   BASELINE_MIP_CSV="$2"; shift 2 ;;
        --solver-arg)         EXTRA_ARGS+=("$2");    shift 2 ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

for arg in "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"; do
    SOLVER_ARG_FLAGS+=(--solver-arg "${arg}")
done

if [[ -z "${CAND_BIN}" ]]; then
    echo "Required: --candidate-binary <path>" >&2
    exit 1
fi
if [[ ! -x "${CAND_BIN}" ]]; then
    echo "Candidate binary not executable: ${CAND_BIN}" >&2
    exit 1
fi
if [[ -z "${BASELINE_LP_CSV}" || -z "${BASELINE_MIP_CSV}" ]]; then
    if [[ -z "${BASE_BIN}" || ! -x "${BASE_BIN}" ]]; then
        echo "Need --baseline-binary or both --baseline-lp-csv and --baseline-mip-csv" >&2
        exit 1
    fi
fi

mkdir -p "${OUT_DIR}"

CAND_LP_CSV="${OUT_DIR}/candidate_lp.csv"
CAND_MIP_CSV="${OUT_DIR}/candidate_mip.csv"
BASE_LP_CSV="${BASELINE_LP_CSV:-${OUT_DIR}/baseline_lp.csv}"
BASE_MIP_CSV="${BASELINE_MIP_CSV:-${OUT_DIR}/baseline_mip.csv}"

# --- Candidate LP benchmark ---
echo "[mittelman-gate] Running candidate LP benchmark (Mittelman LPopt)..."
LP_ARGS=(
    --binary "${CAND_BIN}"
    --output "${CAND_LP_CSV}"
    --mittelman-dir "${MITTELMAN_LP_DIR}"
    --netlib-dir "${NETLIB_DIR}"
    --repeats "${LP_REPEATS}"
    --time-limit "${LP_TIME_LIMIT}"
)
if [[ ${#SOLVER_ARG_FLAGS[@]} -gt 0 ]]; then
    LP_ARGS+=("${SOLVER_ARG_FLAGS[@]}")
fi
"${SCRIPT_DIR}/run_mittelman_lp_bench.sh" "${LP_ARGS[@]}"

# --- Candidate MIP benchmark ---
echo "[mittelman-gate] Running candidate MIP benchmark (Mittelman MILP)..."
MIP_ARGS=(
    --binary "${CAND_BIN}"
    --miplib-dir "${MIPLIB_DIR}"
    --output "${CAND_MIP_CSV}"
    --repeats "${MIP_REPEATS}"
    --threads "${MIP_THREADS}"
    --time-limit "${MIP_TIME_LIMIT}"
    --gap-tol "${MIP_GAP_TOL}"
)
if [[ -n "${MIP_INSTANCES}" ]]; then
    MIP_ARGS+=(--instances "${MIP_INSTANCES}")
fi
if [[ ${#SOLVER_ARG_FLAGS[@]} -gt 0 ]]; then
    MIP_ARGS+=("${SOLVER_ARG_FLAGS[@]}")
fi
"${SCRIPT_DIR}/run_mittelman_mip_bench.sh" "${MIP_ARGS[@]}"

# --- Baseline LP benchmark (if not provided) ---
if [[ -z "${BASELINE_LP_CSV}" ]]; then
    echo "[mittelman-gate] Running baseline LP benchmark..."
    LP_BASE_ARGS=(
        --binary "${BASE_BIN}"
        --output "${BASE_LP_CSV}"
        --mittelman-dir "${MITTELMAN_LP_DIR}"
        --netlib-dir "${NETLIB_DIR}"
        --repeats "${LP_REPEATS}"
        --time-limit "${LP_TIME_LIMIT}"
    )
    if [[ ${#SOLVER_ARG_FLAGS[@]} -gt 0 ]]; then
        LP_BASE_ARGS+=("${SOLVER_ARG_FLAGS[@]}")
    fi
    "${SCRIPT_DIR}/run_mittelman_lp_bench.sh" "${LP_BASE_ARGS[@]}"
fi

# --- Baseline MIP benchmark (if not provided) ---
if [[ -z "${BASELINE_MIP_CSV}" ]]; then
    echo "[mittelman-gate] Running baseline MIP benchmark..."
    MIP_BASE_ARGS=(
        --binary "${BASE_BIN}"
        --miplib-dir "${MIPLIB_DIR}"
        --output "${BASE_MIP_CSV}"
        --repeats "${MIP_REPEATS}"
        --threads "${MIP_THREADS}"
        --time-limit "${MIP_TIME_LIMIT}"
        --gap-tol "${MIP_GAP_TOL}"
    )
    if [[ -n "${MIP_INSTANCES}" ]]; then
        MIP_BASE_ARGS+=(--instances "${MIP_INSTANCES}")
    fi
    if [[ ${#SOLVER_ARG_FLAGS[@]} -gt 0 ]]; then
        MIP_BASE_ARGS+=("${SOLVER_ARG_FLAGS[@]}")
    fi
    "${SCRIPT_DIR}/run_mittelman_mip_bench.sh" "${MIP_BASE_ARGS[@]}"
fi

# --- Regression checks ---
GATE_PASSED=true

echo ""
echo "[mittelman-gate] Checking LP regression (${METRIC})..."
if ! python3 "${SCRIPT_DIR}/check_regression.py" \
    --baseline "${BASE_LP_CSV}" \
    --candidate "${CAND_LP_CSV}" \
    --metric "${METRIC}" \
    --max-regression-pct "${MAX_REGRESSION_PCT}" \
    --min-common-instances "${LP_MIN_COMMON}"; then
    GATE_PASSED=false
    echo "[mittelman-gate] LP regression gate FAILED"
fi

echo ""
echo "[mittelman-gate] Checking MIP regression (${METRIC})..."
if ! python3 "${SCRIPT_DIR}/check_regression.py" \
    --baseline "${BASE_MIP_CSV}" \
    --candidate "${CAND_MIP_CSV}" \
    --metric "${METRIC}" \
    --max-regression-pct "${MAX_REGRESSION_PCT}" \
    --min-common-instances "${MIP_MIN_COMMON}"; then
    GATE_PASSED=false
    echo "[mittelman-gate] MIP regression gate FAILED"
fi

echo ""
echo "[mittelman-gate] Results:"
echo "  LP candidate:  ${CAND_LP_CSV}"
echo "  LP baseline:   ${BASE_LP_CSV}"
echo "  MIP candidate: ${CAND_MIP_CSV}"
echo "  MIP baseline:  ${BASE_MIP_CSV}"

if [[ "${GATE_PASSED}" == "true" ]]; then
    echo ""
    echo "[mittelman-gate] PASS"
else
    echo ""
    echo "[mittelman-gate] FAIL"
    exit 1
fi
