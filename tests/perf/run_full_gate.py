#!/usr/bin/env python3
"""Run LP + MIP performance gates end-to-end with one command."""

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
    p.add_argument("--netlib-dir", required=True)
    p.add_argument("--miplib-dir", required=True)
    p.add_argument("--out-dir", default="/tmp/mipx_fullgate")

    p.add_argument("--lp-repeats", type=int, default=3)
    p.add_argument("--mip-repeats", type=int, default=1)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--time-limit", type=float, default=30)
    p.add_argument("--node-limit", type=int, default=100000)
    p.add_argument("--gap-tol", type=float, default=1e-4)
    p.add_argument("--mip-instances", default="p0201,pk1,gt2")

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


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> int:
    args = parse_args()

    cand_bin = Path(args.candidate_binary)
    base_bin = Path(args.baseline_binary) if args.baseline_binary else None
    netlib_dir = Path(args.netlib_dir)
    miplib_dir = Path(args.miplib_dir)

    if not cand_bin.is_file() or not cand_bin.stat().st_mode & 0o111:
        raise SystemExit(f"Candidate binary not executable: {cand_bin}")
    if not netlib_dir.is_dir():
        raise SystemExit(f"Netlib directory not found: {netlib_dir}")
    if not miplib_dir.is_dir():
        raise SystemExit(f"MIPLIB directory not found: {miplib_dir}")

    if (not args.baseline_lp_csv or not args.baseline_mip_csv) and (
        base_bin is None or not base_bin.is_file() or not base_bin.stat().st_mode & 0o111
    ):
        raise SystemExit("Need --baseline-binary when baseline CSVs are not fully provided.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cand_lp_csv = out_dir / "candidate_lp.csv"
    cand_mip_csv = out_dir / "candidate_mip.csv"
    base_lp_csv = Path(args.baseline_lp_csv) if args.baseline_lp_csv else out_dir / "baseline_lp.csv"
    base_mip_csv = Path(args.baseline_mip_csv) if args.baseline_mip_csv else out_dir / "baseline_mip.csv"

    run_netlib = PERF_DIR / "run_netlib_lp_bench.py"
    run_miplib = PERF_DIR / "run_miplib_mip_bench.py"
    check = PERF_DIR / "check_regression.py"

    solver_args = []
    for sarg in args.solver_arg:
        solver_args.extend(["--solver-arg", sarg])

    print("[fullgate] Running candidate LP bench...")
    run(
        [
            sys.executable,
            str(run_netlib),
            "--binary",
            str(cand_bin),
            "--netlib-dir",
            str(netlib_dir),
            "--output",
            str(cand_lp_csv),
            "--repeats",
            str(args.lp_repeats),
            *solver_args,
        ]
    )

    print("[fullgate] Running candidate MIP bench...")
    run(
        [
            sys.executable,
            str(run_miplib),
            "--binary",
            str(cand_bin),
            "--miplib-dir",
            str(miplib_dir),
            "--output",
            str(cand_mip_csv),
            "--repeats",
            str(args.mip_repeats),
            "--threads",
            str(args.threads),
            "--time-limit",
            f"{args.time_limit:g}",
            "--node-limit",
            str(args.node_limit),
            "--gap-tol",
            f"{args.gap_tol:g}",
            "--instances",
            args.mip_instances,
            *solver_args,
        ]
    )

    if not args.baseline_lp_csv:
        print("[fullgate] Running baseline LP bench...")
        run(
            [
                sys.executable,
                str(run_netlib),
                "--binary",
                str(base_bin),
                "--netlib-dir",
                str(netlib_dir),
                "--output",
                str(base_lp_csv),
                "--repeats",
                str(args.lp_repeats),
                *solver_args,
            ]
        )

    if not args.baseline_mip_csv:
        print("[fullgate] Running baseline MIP bench...")
        run(
            [
                sys.executable,
                str(run_miplib),
                "--binary",
                str(base_bin),
                "--miplib-dir",
                str(miplib_dir),
                "--output",
                str(base_mip_csv),
                "--repeats",
                str(args.mip_repeats),
                "--threads",
                str(args.threads),
                "--time-limit",
                f"{args.time_limit:g}",
                "--node-limit",
                str(args.node_limit),
                "--gap-tol",
                f"{args.gap_tol:g}",
                "--instances",
                args.mip_instances,
                *solver_args,
            ]
        )

    print(f"[fullgate] Checking LP regression ({args.metric})...")
    run(
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
        ]
    )

    print(f"[fullgate] Checking MIP regression ({args.metric})...")
    run(
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
        ]
    )

    print("[fullgate] PASS")
    print(f"[fullgate] LP baseline: {base_lp_csv}")
    print(f"[fullgate] LP candidate: {cand_lp_csv}")
    print(f"[fullgate] MIP baseline: {base_mip_csv}")
    print(f"[fullgate] MIP candidate: {cand_mip_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
