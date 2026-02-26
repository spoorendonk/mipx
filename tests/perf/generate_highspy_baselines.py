#!/usr/bin/env python3
"""Generate versioned HiGHS CLI baselines for LP and MIP comparisons."""

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
RUNNER = ROOT_DIR / "tests" / "perf" / "run_highspy_bench.py"

LP_OUT = OUT_DIR / "highspy_lp_netlib_small.csv"
MIP_OUT = OUT_DIR / "highspy_mip_miplib_small.csv"
META_JSON = OUT_DIR / "highspy_baseline_meta.json"


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(cmd))
    return subprocess.run(cmd, check=check, text=True)


def capture(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True).strip()
    except Exception:
        return ""


def resolve_highs_bin() -> str:
    return os.environ.get("HIGHS_BIN") or os.environ.get("HIGHS_BINARY") or "highs"


def main() -> int:
    highs_bin = resolve_highs_bin()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    run(
        [
            sys.executable,
            str(RUNNER),
            "--mode",
            "lp",
            "--highs-binary",
            highs_bin,
            "--instances-dir",
            str(NETLIB_DIR),
            "--output",
            str(LP_OUT),
            "--repeats",
            "3",
            "--threads",
            "1",
            "--solver",
            "simplex",
            "--simplex-strategy",
            "1",
            "--presolve",
            "choose",
        ]
    )

    run(
        [
            sys.executable,
            str(RUNNER),
            "--mode",
            "mip",
            "--highs-binary",
            highs_bin,
            "--instances-dir",
            str(MIPLIB_DIR),
            "--instances",
            "p0201,gt2,flugpl",
            "--output",
            str(MIP_OUT),
            "--repeats",
            "1",
            "--threads",
            "1",
            "--time-limit",
            "30",
            "--presolve",
            "choose",
            "--solver",
            "choose",
        ]
    )

    meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "highs_version": capture([highs_bin, "--version"]),
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
