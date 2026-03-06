#!/usr/bin/env python3
"""Run deterministic LP+MIP regression gates with algorithmic and wall-clock tracking.

Contract defaults:
- deterministic mode
- fixed seed
- stable search profile

Regression policy:
- strict algorithmic gate on work-like metric (`work_units` by default)
- optional/looser wall-clock gate on `time_seconds`
"""

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

    p.add_argument("--seed", type=int, default=1)
    p.add_argument(
        "--parallel-mode",
        choices=("deterministic", "opportunistic"),
        default="deterministic",
    )
    p.add_argument(
        "--search-profile",
        choices=("stable", "default", "aggressive"),
        default="stable",
    )

    p.add_argument("--algorithmic-metric", default="work_units")
    p.add_argument("--algorithmic-max-regression-pct", type=float, default=0.0)
    p.add_argument("--lp-min-common", type=int, default=5)
    p.add_argument("--mip-min-common", type=int, default=3)

    p.add_argument("--track-wall-clock", action="store_true", default=True)
    p.add_argument("--no-track-wall-clock", action="store_false", dest="track_wall_clock")
    p.add_argument("--wall-metric", default="time_seconds")
    p.add_argument("--wall-max-regression-pct", type=float, default=35.0)
    p.add_argument("--enforce-wall-clock", action="store_true")

    p.add_argument("--run-determinism", action="store_true", default=True)
    p.add_argument("--no-run-determinism", action="store_false", dest="run_determinism")
    p.add_argument("--determinism-runs", type=int, default=5)
    p.add_argument("--determinism-strict-metrics", action="store_true", default=True)

    p.add_argument("--baseline-lp-csv", default="")
    p.add_argument("--baseline-mip-csv", default="")
    p.add_argument("--solver-arg", action="append", default=[])

    # Backward-compatible aliases from previous gate interface.
    p.add_argument("--metric", default=None, help=argparse.SUPPRESS)
    p.add_argument("--max-regression-pct", type=float, default=None, help=argparse.SUPPRESS)

    argv: list[str] = []
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

    args = p.parse_args(argv)
    if args.metric is not None:
        args.algorithmic_metric = args.metric
    if args.max_regression_pct is not None:
        args.algorithmic_max_regression_pct = args.max_regression_pct
    return args


def run(cmd: list[str], check: bool = True) -> int:
    print("+", " ".join(cmd))
    proc = subprocess.run(cmd)
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    return proc.returncode


def flatten_solver_args(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        out.extend(["--solver-arg", value])
    return out


def regression_check(
    *,
    check_script: Path,
    baseline_csv: Path,
    candidate_csv: Path,
    metric: str,
    max_regression_pct: float,
    min_common_instances: int,
    label: str,
    enforce: bool,
) -> bool:
    cmd = [
        sys.executable,
        str(check_script),
        "--baseline",
        str(baseline_csv),
        "--candidate",
        str(candidate_csv),
        "--metric",
        metric,
        "--max-regression-pct",
        f"{max_regression_pct:g}",
        "--min-common-instances",
        str(min_common_instances),
    ]

    print(f"[fullgate] Checking {label} ({metric}, max_reg={max_regression_pct:g}%)...")
    rc = run(cmd, check=False)
    if rc == 0:
        return True

    if enforce:
        print(f"[fullgate] FAIL: {label} regression gate failed")
        return False

    print(f"[fullgate] WARN: {label} regression exceeded threshold (non-fatal)")
    return True


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
    if args.lp_repeats < 1 or args.mip_repeats < 1:
        raise SystemExit("--lp-repeats and --mip-repeats must be >= 1")
    if args.threads < 1:
        raise SystemExit("--threads must be >= 1")

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
    run_det = PERF_DIR / "run_determinism_suite.py"
    check = PERF_DIR / "check_regression.py"

    deterministic_contract_args = [
        "--parallel-mode",
        args.parallel_mode,
        "--seed",
        str(args.seed),
    ]
    if args.search_profile == "stable":
        deterministic_contract_args.append("--search-stable")
    elif args.search_profile == "default":
        deterministic_contract_args.append("--search-default")
    else:
        deterministic_contract_args.append("--search-aggressive")

    # Keep user-provided args last so they can intentionally override defaults
    # for benchmark runs.
    bench_solver_args = flatten_solver_args(deterministic_contract_args + args.solver_arg)
    # Determinism suite already injects deterministic flags internally, so do
    # not pass profile flags here to avoid accidental overrides/conflicts.
    det_solver_args = flatten_solver_args(args.solver_arg)

    if args.run_determinism:
        det_out_dir = out_dir / "determinism_candidate"
        det_cmd = [
            sys.executable,
            str(run_det),
            "--binary",
            str(cand_bin),
            "--miplib-dir",
            str(miplib_dir),
            "--out-dir",
            str(det_out_dir),
            "--instances",
            args.mip_instances,
            "--runs",
            str(args.determinism_runs),
            "--seed",
            str(args.seed),
            "--single-threads",
            "1",
            "--multi-threads",
            str(args.threads),
            "--time-limit",
            f"{args.time_limit:g}",
            "--node-limit",
            str(args.node_limit),
            "--gap-tol",
            f"{args.gap_tol:g}",
            *det_solver_args,
        ]
        if args.determinism_strict_metrics:
            det_cmd.append("--strict-metrics")

        print("[fullgate] Running deterministic reproducibility suite...")
        run(det_cmd)

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
            *bench_solver_args,
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
            *bench_solver_args,
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
                *bench_solver_args,
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
                *bench_solver_args,
            ]
        )

    ok = True
    ok = regression_check(
        check_script=check,
        baseline_csv=base_lp_csv,
        candidate_csv=cand_lp_csv,
        metric=args.algorithmic_metric,
        max_regression_pct=args.algorithmic_max_regression_pct,
        min_common_instances=args.lp_min_common,
        label="LP algorithmic",
        enforce=True,
    ) and ok
    ok = regression_check(
        check_script=check,
        baseline_csv=base_mip_csv,
        candidate_csv=cand_mip_csv,
        metric=args.algorithmic_metric,
        max_regression_pct=args.algorithmic_max_regression_pct,
        min_common_instances=args.mip_min_common,
        label="MIP algorithmic",
        enforce=True,
    ) and ok

    if args.track_wall_clock:
        wall_ok_lp = regression_check(
            check_script=check,
            baseline_csv=base_lp_csv,
            candidate_csv=cand_lp_csv,
            metric=args.wall_metric,
            max_regression_pct=args.wall_max_regression_pct,
            min_common_instances=args.lp_min_common,
            label="LP wall-clock",
            enforce=args.enforce_wall_clock,
        )
        wall_ok_mip = regression_check(
            check_script=check,
            baseline_csv=base_mip_csv,
            candidate_csv=cand_mip_csv,
            metric=args.wall_metric,
            max_regression_pct=args.wall_max_regression_pct,
            min_common_instances=args.mip_min_common,
            label="MIP wall-clock",
            enforce=args.enforce_wall_clock,
        )
        ok = ok and wall_ok_lp and wall_ok_mip

    if not ok:
        print("[fullgate] FAIL")
        return 1

    print("[fullgate] PASS")
    print(f"[fullgate] LP baseline: {base_lp_csv}")
    print(f"[fullgate] LP candidate: {cand_lp_csv}")
    print(f"[fullgate] MIP baseline: {base_mip_csv}")
    print(f"[fullgate] MIP candidate: {cand_mip_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
