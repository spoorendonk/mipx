#!/usr/bin/env bash
# Generate committed mipx baseline CSVs for strict work_units regression gates.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${ROOT_DIR}/tests/perf/baselines"
NETLIB_DIR="${ROOT_DIR}/tests/data/netlib"
MIPLIB_DIR="${ROOT_DIR}/tests/data/miplib"
BIN="${ROOT_DIR}/build/mipx-solve"

LP_OUT="${OUT_DIR}/mipx_lp_netlib_small.csv"
MIP_OUT="${OUT_DIR}/mipx_mip_miplib_small.csv"
META_JSON="${OUT_DIR}/mipx_baseline_meta.json"

if [[ ! -x "${BIN}" ]]; then
  echo "Binary not found: ${BIN}" >&2
  echo "Build first: cmake --build build -j\$(nproc)" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

./tests/perf/run_netlib_lp_bench.sh \
  --binary "${BIN}" \
  --netlib-dir "${NETLIB_DIR}" \
  --output "${LP_OUT}" \
  --repeats 3 \
  --solver-arg --quiet

./tests/perf/run_miplib_mip_bench.sh \
  --binary "${BIN}" \
  --miplib-dir "${MIPLIB_DIR}" \
  --output "${MIP_OUT}" \
  --repeats 1 \
  --threads 1 \
  --time-limit 30 \
  --instances p0201,pk1,gt2 \
  --solver-arg --quiet

python3 - <<'PY' "${META_JSON}" "${ROOT_DIR}"
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
}

try:
    meta["cpu_info"] = cmd("lscpu | head -n 20")
except Exception:
    pass

with meta_path.open("w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)
PY

echo "Wrote baselines:"
echo "  ${LP_OUT}"
echo "  ${MIP_OUT}"
echo "  ${META_JSON}"
