#!/usr/bin/env python3
"""Run Mittelman-style LP + MIP performance regression gate."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
PERF_DIR = ROOT_DIR / "tests" / "perf"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate-binary", required=True)
    p.add_argument("--baseline-binary", default="")
    p.add_argument("--out-dir", default="/tmp/mipx_mittelman_gate")

    p.add_argument("--mittelman-lp-dir", default=str(ROOT_DIR / "tests" / "data" / "mittelman_lp"))
    p.add_argument("--netlib-dir", default=str(ROOT_DIR / "tests" / "data" / "netlib"))
    p.add_argument("--miplib-dir", default=str(ROOT_DIR / "tests" / "data" / "miplib"))

    p.add_argument("--lp-repeats", type=int, default=3)
    p.add_argument("--mip-repeats", type=int, default=1)
    p.add_argument("--lp-time-limit", type=float, default=15000)
    p.add_argument("--mip-time-limit", type=float, default=7200)
    p.add_argument("--mip-threads", type=int, default=8)
    p.add_argument("--mip-instances", default="")

    p.add_argument("--metric", default="work_units")
    p.add_argument("--max-regression-pct", type=float, default=0.0)
    p.add_argument("--lp-min-common", type=int, default=5)
    p.add_argument("--mip-min-common", type=int, default=3)

    p.add_argument("--baseline-lp-csv", default="")
    p.add_argument("--baseline-mip-csv", default="")
    p.add_argument("--solver-arg", action="append", default=[])
    argv = []
    raw = sys.argv[1:]
    i = 0
    while i < len(raw):
        if raw[i] == "--solver-arg":
            if i + 1 >= len(raw):
                raise SystemExit("--solver-arg requires one argument")
            argv.append(f"--solver-arg={raw[i + 1]}")
            i += 2
            continue
        argv.append(raw[i])
        i += 1
    return p.parse_args(argv)


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(cmd))
    return subprocess.run(cmd, check=check)


def main() -> int:
    args = parse_args()

    cand_bin = Path(args.candidate_binary)
    base_bin = Path(args.baseline_binary) if args.baseline_binary else None

    if not cand_bin.is_file() or not cand_bin.stat().st_mode & 0o111:
        raise SystemExit(f"Candidate binary not executable: {cand_bin}")

    if (not args.baseline_lp_csv or not args.baseline_mip_csv) and (
        base_bin is None or not base_bin.is_file() or not base_bin.stat().st_mode & 0o111
    ):
        raise SystemExit("Need --baseline-binary or both --baseline-lp-csv and --baseline-mip-csv")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cand_lp_csv = out_dir / "candidate_lp.csv"
    cand_mip_csv = out_dir / "candidate_mip.csv"
    base_lp_csv = Path(args.baseline_lp_csv) if args.baseline_lp_csv else out_dir / "baseline_lp.csv"
    base_mip_csv = Path(args.baseline_mip_csv) if args.baseline_mip_csv else out_dir / "baseline_mip.csv"

    run_lp = PERF_DIR / "run_mittelman_lp_bench.py"
    run_mip = PERF_DIR / "run_mittelman_mip_bench.py"
    check = PERF_DIR / "check_regression.py"

    solver_args = []
    for sarg in args.solver_arg:
        solver_args.extend(["--solver-arg", sarg])

    print("[mittelman-gate] Running candidate LP benchmark (Mittelman LPopt)...")
    run(
        [
            sys.executable,
            str(run_lp),
            "--binary",
            str(cand_bin),
            "--output",
            str(cand_lp_csv),
            "--mittelman-dir",
            args.mittelman_lp_dir,
            "--netlib-dir",
            args.netlib_dir,
            "--repeats",
            str(args.lp_repeats),
            "--time-limit",
            f"{args.lp_time_limit:g}",
            *solver_args,
        ]
    )

    print("[mittelman-gate] Running candidate MIP benchmark (Mittelman MILP)...")
    cmd = [
        sys.executable,
        str(run_mip),
        "--binary",
        str(cand_bin),
        "--miplib-dir",
        args.miplib_dir,
        "--output",
        str(cand_mip_csv),
        "--repeats",
        str(args.mip_repeats),
        "--threads",
        str(args.mip_threads),
        "--time-limit",
        f"{args.mip_time_limit:g}",
        "--gap-tol",
        "1e-4",
        *solver_args,
    ]
    if args.mip_instances:
        cmd.extend(["--instances", args.mip_instances])
    run(cmd)

    if not args.baseline_lp_csv:
        print("[mittelman-gate] Running baseline LP benchmark...")
        run(
            [
                sys.executable,
                str(run_lp),
                "--binary",
                str(base_bin),
                "--output",
                str(base_lp_csv),
                "--mittelman-dir",
                args.mittelman_lp_dir,
                "--netlib-dir",
                args.netlib_dir,
                "--repeats",
                str(args.lp_repeats),
                "--time-limit",
                f"{args.lp_time_limit:g}",
                *solver_args,
            ]
        )

    if not args.baseline_mip_csv:
        print("[mittelman-gate] Running baseline MIP benchmark...")
        cmd = [
            sys.executable,
            str(run_mip),
            "--binary",
            str(base_bin),
            "--miplib-dir",
            args.miplib_dir,
            "--output",
            str(base_mip_csv),
            "--repeats",
            str(args.mip_repeats),
            "--threads",
            str(args.mip_threads),
            "--time-limit",
            f"{args.mip_time_limit:g}",
            "--gap-tol",
            "1e-4",
            *solver_args,
        ]
        if args.mip_instances:
            cmd.extend(["--instances", args.mip_instances])
        run(cmd)

    gate_passed = True

    print(f"\n[mittelman-gate] Checking LP regression ({args.metric})...")
    lp_proc = run(
        [
            sys.executable,
            str(check),
            "--baseline",
            str(base_lp_csv),
            "--candidate",
            str(cand_lp_csv),
            "--metric",
            args.metric,
            "--max-regression-pct",
            f"{args.max_regression_pct:g}",
            "--min-common-instances",
            str(args.lp_min_common),
        ],
        check=False,
    )
    if lp_proc.returncode != 0:
        gate_passed = False
        print("[mittelman-gate] LP regression gate FAILED")

    print(f"\n[mittelman-gate] Checking MIP regression ({args.metric})...")
    mip_proc = run(
        [
            sys.executable,
            str(check),
            "--baseline",
            str(base_mip_csv),
            "--candidate",
            str(cand_mip_csv),
            "--metric",
            args.metric,
            "--max-regression-pct",
            f"{args.max_regression_pct:g}",
            "--min-common-instances",
            str(args.mip_min_common),
        ],
        check=False,
    )
    if mip_proc.returncode != 0:
        gate_passed = False
        print("[mittelman-gate] MIP regression gate FAILED")

    print("\n[mittelman-gate] Results:")
    print(f"  LP candidate:  {cand_lp_csv}")
    print(f"  LP baseline:   {base_lp_csv}")
    print(f"  MIP candidate: {cand_mip_csv}")
    print(f"  MIP baseline:  {base_mip_csv}")

    if gate_passed:
        print("\n[mittelman-gate] PASS")
        return 0

    print("\n[mittelman-gate] FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
