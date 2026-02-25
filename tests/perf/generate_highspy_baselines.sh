#!/usr/bin/env bash
# Generate versioned highspy baselines for LP and MIP performance comparisons.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${ROOT_DIR}/tests/perf/baselines"
NETLIB_DIR="${ROOT_DIR}/tests/data/netlib"
MIPLIB_DIR="${ROOT_DIR}/tests/data/miplib"

LP_OUT="${OUT_DIR}/highspy_lp_netlib_small.csv"
MIP_OUT="${OUT_DIR}/highspy_mip_miplib_small.csv"

mkdir -p "${OUT_DIR}"

python3 "${ROOT_DIR}/tests/perf/run_highspy_bench.py" \
  --mode lp \
  --instances-dir "${NETLIB_DIR}" \
  --output "${LP_OUT}" \
  --repeats 3 \
  --threads 1 \
  --solver simplex \
  --simplex-strategy 1 \
  --presolve choose

python3 "${ROOT_DIR}/tests/perf/run_highspy_bench.py" \
  --mode mip \
  --instances-dir "${MIPLIB_DIR}" \
  --instances p0201,gt2,flugpl \
  --output "${MIP_OUT}" \
  --repeats 1 \
  --threads 1 \
  --time-limit 30 \
  --presolve choose \
  --solver choose

META_JSON="${OUT_DIR}/highspy_baseline_meta.json"
python3 - <<'PY' "${META_JSON}"
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
}
try:
    cpu = subprocess.check_output(["bash", "-lc", "lscpu | head -n 20"], text=True)
    meta["cpu_info"] = cpu
except Exception:
    pass

with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)
PY

echo "Wrote baselines:"
echo "  ${LP_OUT}"
echo "  ${MIP_OUT}"
echo "  ${META_JSON}"
