#!/usr/bin/env python3
"""Generate HiGHS CLI + mipx baselines for Mittelman benchmark sets."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT_DIR / "tests" / "perf" / "baselines"
MITTELMAN_LP_DIR = ROOT_DIR / "tests" / "data" / "mittelman_lp"
NETLIB_DIR = ROOT_DIR / "tests" / "data" / "netlib"
MIPLIB_DIR = ROOT_DIR / "tests" / "data" / "miplib"
BIN = ROOT_DIR / "build" / "mipx-solve"

HIGHS_RUNNER = ROOT_DIR / "tests" / "perf" / "run_highspy_bench.py"
MIPX_LP_RUNNER = ROOT_DIR / "tests" / "perf" / "run_mittelman_lp_bench.sh"
MIPX_MIP_RUNNER = ROOT_DIR / "tests" / "perf" / "run_mittelman_mip_bench.sh"

HIGHS_LP_TIME_LIMIT = 600
MIPX_LP_TIME_LIMIT = 15000
LP_REPEATS = 3
MIP_TIME_LIMIT = 7200
MIP_THREADS = 8
MIP_REPEATS = 1
MIP_GAP_TOL = 1e-4

MITTELMAN_MIP_INSTANCES = (
    "air04,air05,bell5,blend2,dcmulti,flugpl,gen,gesa2,gesa2-o,glass4,gt2,"
    "misc03,misc06,misc07,mod008,mod010,noswot,p0033,p0201,p0282,p0548,pk1,pp08a,"
    "pp08aCUTS,qiu,qnet1,qnet1_o,ran13x13,rentacar,rgn,set1ch,stein27,stein45,stein9inf,vpm2"
)


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(cmd))
    return subprocess.run(cmd, check=check, text=True)


def run_maybe(cmd: list[str], warning: str) -> None:
    proc = run(cmd, check=False)
    if proc.returncode != 0:
        print(f"Warning: {warning}")


def capture(cmd: list[str], *, cwd: Path | None = None) -> str:
    try:
        return subprocess.check_output(cmd, cwd=cwd, text=True).strip()
    except Exception:
        return ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--highs-only", action="store_true")
    g.add_argument("--mipx-only", action="store_true")
    return p.parse_args()


def resolve_highs_bin() -> str:
    return os.environ.get("HIGHS_BIN") or os.environ.get("HIGHS_BINARY") or "highs"


def run_highs_part(highs_bin: str) -> None:
    print("=== Generating HiGHS Mittelman LP baseline ===")

    highs_lp_out = OUT_DIR / "highspy_lp_mittelman.csv"
    run_maybe(
        [
            sys.executable,
            str(HIGHS_RUNNER),
            "--mode",
            "lp",
            "--highs-binary",
            highs_bin,
            "--instances-dir",
            str(MITTELMAN_LP_DIR),
            "--output",
            str(highs_lp_out),
            "--repeats",
            str(LP_REPEATS),
            "--threads",
            "1",
            "--time-limit",
            str(HIGHS_LP_TIME_LIMIT),
            "--solver",
            "simplex",
            "--simplex-strategy",
            "1",
            "--presolve",
            "choose",
        ],
        "HiGHS LP Mittelman baseline skipped (instances may be missing)",
    )

    highs_lp_netlib = OUT_DIR / "highspy_lp_netlib_full.csv"
    if NETLIB_DIR.is_dir() and list(NETLIB_DIR.glob("*.mps.gz")):
        run_maybe(
            [
                sys.executable,
                str(HIGHS_RUNNER),
                "--mode",
                "lp",
                "--highs-binary",
                highs_bin,
                "--instances-dir",
                str(NETLIB_DIR),
                "--output",
                str(highs_lp_netlib),
                "--repeats",
                str(LP_REPEATS),
                "--threads",
                "1",
                "--time-limit",
                str(HIGHS_LP_TIME_LIMIT),
                "--solver",
                "simplex",
                "--simplex-strategy",
                "1",
                "--presolve",
                "choose",
            ],
            "HiGHS LP Netlib baseline skipped",
        )

    print("=== Generating HiGHS Mittelman MIP baseline ===")
    highs_mip_out = OUT_DIR / "highspy_mip_mittelman.csv"
    run_maybe(
        [
            sys.executable,
            str(HIGHS_RUNNER),
            "--mode",
            "mip",
            "--highs-binary",
            highs_bin,
            "--instances-dir",
            str(MIPLIB_DIR),
            "--instances",
            MITTELMAN_MIP_INSTANCES,
            "--output",
            str(highs_mip_out),
            "--repeats",
            str(MIP_REPEATS),
            "--threads",
            str(MIP_THREADS),
            "--time-limit",
            str(MIP_TIME_LIMIT),
            "--mip-rel-gap",
            str(MIP_GAP_TOL),
            "--presolve",
            "choose",
            "--solver",
            "choose",
        ],
        "HiGHS MIP Mittelman baseline skipped (instances may be missing)",
    )

    highs_meta = OUT_DIR / "highspy_mittelman_meta.json"
    meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "highs_version": capture([highs_bin, "--version"]),
        "platform": platform.platform(),
        "benchmark_config": {
            "lp_time_limit": HIGHS_LP_TIME_LIMIT,
            "lp_repeats": LP_REPEATS,
            "mip_time_limit": MIP_TIME_LIMIT,
            "mip_threads": MIP_THREADS,
            "mip_gap_tol": MIP_GAP_TOL,
            "reference": "https://plato.asu.edu/bench.html",
        },
    }
    cpu = capture(["bash", "-lc", "lscpu | head -n 20"])
    if cpu:
        meta["cpu_info"] = cpu
    highs_meta.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print("Wrote HiGHS Mittelman baselines:")
    if highs_lp_out.is_file():
        print(f"  {highs_lp_out}")
    if highs_lp_netlib.is_file():
        print(f"  {highs_lp_netlib}")
    if highs_mip_out.is_file():
        print(f"  {highs_mip_out}")
    print(f"  {highs_meta}")


def run_mipx_part() -> None:
    if not BIN.is_file() or not os.access(BIN, os.X_OK):
        raise SystemExit(f"mipx binary not found: {BIN}")

    print("=== Generating mipx Mittelman LP baseline ===")
    mipx_lp_out = OUT_DIR / "mipx_lp_mittelman.csv"
    run(
        [
            str(MIPX_LP_RUNNER),
            "--binary",
            str(BIN),
            "--mittelman-dir",
            str(MITTELMAN_LP_DIR),
            "--netlib-dir",
            str(NETLIB_DIR),
            "--output",
            str(mipx_lp_out),
            "--repeats",
            str(LP_REPEATS),
            "--time-limit",
            str(MIPX_LP_TIME_LIMIT),
            "--solver-arg",
            "--quiet",
        ]
    )

    print("=== Generating mipx Mittelman MIP baseline ===")
    mipx_mip_out = OUT_DIR / "mipx_mip_mittelman.csv"
    run(
        [
            str(MIPX_MIP_RUNNER),
            "--binary",
            str(BIN),
            "--miplib-dir",
            str(MIPLIB_DIR),
            "--output",
            str(mipx_mip_out),
            "--threads",
            str(MIP_THREADS),
            "--time-limit",
            str(MIP_TIME_LIMIT),
            "--gap-tol",
            str(MIP_GAP_TOL),
            "--instances",
            MITTELMAN_MIP_INSTANCES,
            "--repeats",
            str(MIP_REPEATS),
            "--solver-arg",
            "--quiet",
        ]
    )

    mipx_meta = OUT_DIR / "mipx_mittelman_meta.json"
    meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": capture(["git", "rev-parse", "HEAD"], cwd=ROOT_DIR),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "benchmark_config": {
            "lp_time_limit": MIPX_LP_TIME_LIMIT,
            "lp_repeats": LP_REPEATS,
            "mip_time_limit": MIP_TIME_LIMIT,
            "mip_threads": MIP_THREADS,
            "mip_gap_tol": MIP_GAP_TOL,
            "reference": "https://plato.asu.edu/bench.html",
        },
    }
    cpu = capture(["bash", "-lc", "lscpu | head -n 20"])
    if cpu:
        meta["cpu_info"] = cpu
    mipx_meta.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print("Wrote mipx Mittelman baselines:")
    print(f"  {mipx_lp_out}")
    print(f"  {mipx_mip_out}")
    print(f"  {mipx_meta}")


def main() -> int:
    args = parse_args()
    highs = not args.mipx_only
    mipx = not args.highs_only

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    highs_bin = resolve_highs_bin()

    if highs:
        run_highs_part(highs_bin)
    if mipx:
        run_mipx_part()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
