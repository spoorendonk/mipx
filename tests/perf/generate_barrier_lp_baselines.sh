#!/usr/bin/env bash
# Generate LP barrier comparison baseline CSVs for mipx/HiGHS/cuOpt.
#
# Outputs:
#   tests/perf/baselines/barrier_lp_compare_netlib.csv
#   tests/perf/baselines/barrier_lp_compare_netlib_forced_gpu.csv
#   tests/perf/baselines/barrier_lp_compare_meta.json

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${ROOT_DIR}/tests/perf/baselines"
BIN="${ROOT_DIR}/build/mipx-solve"
NETLIB_DIR="${ROOT_DIR}/tests/data/netlib"
SCRIPT="${ROOT_DIR}/tests/perf/run_barrier_lp_compare.py"

OUT_NETLIB="${OUT_DIR}/barrier_lp_compare_netlib.csv"
OUT_NETLIB_FORCED="${OUT_DIR}/barrier_lp_compare_netlib_forced_gpu.csv"
OUT_META="${OUT_DIR}/barrier_lp_compare_meta.json"

if [[ ! -x "${BIN}" ]]; then
    echo "mipx binary not found/executable: ${BIN}" >&2
    echo "Build first: cmake --build build -j" >&2
    exit 1
fi

if [[ ! -d "${NETLIB_DIR}" ]]; then
    echo "Netlib dir not found: ${NETLIB_DIR}" >&2
    exit 1
fi

mkdir -p "${OUT_DIR}"

echo "=== Generating Netlib barrier comparison baseline (GPU auto) ==="
python3 "${SCRIPT}" \
  --mipx-binary "${BIN}" \
  --instances-dir "${NETLIB_DIR}" \
  --output "${OUT_NETLIB}" \
  --repeats 3 \
  --threads 1 \
  --time-limit 30 \
  --disable-presolve

echo
echo "=== Generating Netlib barrier comparison baseline (GPU forced) ==="
python3 "${SCRIPT}" \
  --mipx-binary "${BIN}" \
  --instances-dir "${NETLIB_DIR}" \
  --output "${OUT_NETLIB_FORCED}" \
  --repeats 2 \
  --threads 1 \
  --time-limit 30 \
  --disable-presolve \
  --force-mipx-gpu

python3 - <<PY
import datetime
import json
import pathlib
import platform
import subprocess

def run(cmd):
    try:
        out = subprocess.check_output(cmd, text=True).strip()
        return out
    except Exception:
        return ""

meta = {
    "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "python_version": platform.python_version(),
    "platform": platform.platform(),
    "mipx_binary": "${BIN}",
    "highspy_version": "",
    "cuopt_version": "",
    "nvidia_smi": run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
}

try:
    import highspy
    meta["highspy_version"] = highspy.Highs().version()
except Exception:
    pass

cuopt_ver = run(["python3", "-m", "libcuopt._cli_wrapper", "--version"])
if cuopt_ver:
    meta["cuopt_version"] = cuopt_ver

pathlib.Path("${OUT_META}").write_text(json.dumps(meta, indent=2) + "\\n", encoding="utf-8")
print(f"Wrote metadata: ${OUT_META}")
PY

echo
echo "Wrote baseline CSVs:"
echo "  ${OUT_NETLIB}"
echo "  ${OUT_NETLIB_FORCED}"
