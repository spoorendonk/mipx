#!/usr/bin/env python3
"""Generate committed mipx baseline CSVs for strict work_units regression gates."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT_DIR / "tests" / "perf" / "baselines"
NETLIB_DIR = ROOT_DIR / "tests" / "data" / "netlib"
MIPLIB_DIR = ROOT_DIR / "tests" / "data" / "miplib"
BIN = ROOT_DIR / "build" / "mipx-solve"

LP_OUT = OUT_DIR / "mipx_lp_netlib_small.csv"
MIP_OUT = OUT_DIR / "mipx_mip_miplib_small.csv"
META_JSON = OUT_DIR / "mipx_baseline_meta.json"

NETLIB_RUNNER = ROOT_DIR / "tests" / "perf" / "run_netlib_lp_bench.sh"
MIPLIB_RUNNER = ROOT_DIR / "tests" / "perf" / "run_miplib_mip_bench.sh"


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(cmd))
    return subprocess.run(cmd, check=check, text=True)


def capture(cmd: list[str], *, cwd: Path | None = None) -> str:
    try:
        return subprocess.check_output(cmd, cwd=cwd, text=True).strip()
    except Exception:
        return ""


def main() -> int:
    if not BIN.is_file() or not os.access(BIN, os.X_OK):
        raise SystemExit(f"Binary not found: {BIN}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    run(
        [
            str(NETLIB_RUNNER),
            "--binary",
            str(BIN),
            "--netlib-dir",
            str(NETLIB_DIR),
            "--output",
            str(LP_OUT),
            "--repeats",
            "3",
            "--solver-arg",
            "--quiet",
        ]
    )

    run(
        [
            str(MIPLIB_RUNNER),
            "--binary",
            str(BIN),
            "--miplib-dir",
            str(MIPLIB_DIR),
            "--output",
            str(MIP_OUT),
            "--repeats",
            "1",
            "--threads",
            "1",
            "--time-limit",
            "30",
            "--instances",
            "p0201,pk1,gt2",
            "--solver-arg",
            "--quiet",
        ]
    )

    meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": capture(["git", "rev-parse", "HEAD"], cwd=ROOT_DIR),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    cpu = capture(["bash", "-lc", "lscpu | head -n 20"])
    if cpu:
        meta["cpu_info"] = cpu

    META_JSON.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print("Wrote baselines:")
    print(f"  {LP_OUT}")
    print(f"  {MIP_OUT}")
    print(f"  {META_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
