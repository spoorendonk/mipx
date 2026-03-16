#!/usr/bin/env python3
"""Run MIP self-regression gate (candidate vs baseline on strict work_units)."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
PERF_DIR = ROOT_DIR / "tests" / "perf"
DEFAULT_MIPLIB_DIR = ROOT_DIR / "tests" / "data" / "miplib"
DEFAULT_CORPUS = PERF_DIR / "mip_regression_corpus.csv"
DEFAULT_BASELINE_CSV = (
    ROOT_DIR
    / "tests"
    / "perf"
    / "baselines"
    / "mipx_mip_regression_small_seed1_t1_stable.csv"
)
CONTRACT_OVERRIDE_PREFIXES = (
    "--parallel-mode",
    "--seed",
    "--search-stable",
    "--search-default",
    "--search-aggressive",
)


def normalize_solver_arg_tokens(argv: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == "--solver-arg":
            if i + 1 >= len(argv):
                raise SystemExit("--solver-arg requires one argument")
            out.append(f"--solver-arg={argv[i + 1]}")
            i += 2
            continue
        out.append(argv[i])
        i += 1
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate-binary", required=True)
    p.add_argument("--baseline-binary", default="")
    p.add_argument("--baseline-csv", default=str(DEFAULT_BASELINE_CSV))
    p.add_argument("--miplib-dir", default=str(DEFAULT_MIPLIB_DIR))
    p.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    p.add_argument("--out-dir", default="/tmp/mipx_mip_gate")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--time-limit", type=float, default=30.0)
    p.add_argument("--node-limit", type=int, default=100000)
    p.add_argument("--gap-tol", type=float, default=1e-4)
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
    p.add_argument("--metric", default="work_units")
    p.add_argument("--max-regression-pct", type=float, default=0.0)
    p.add_argument("--min-common-instances", type=int, default=3)
    p.add_argument("--solver-arg", action="append", default=[])
    return p.parse_args(normalize_solver_arg_tokens(sys.argv[1:]))


def run(cmd: list[str], check: bool = True) -> int:
    print("+", " ".join(cmd))
    proc = subprocess.run(cmd)
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    return proc.returncode


def load_corpus_instances(corpus_csv: Path) -> list[str]:
    if not corpus_csv.is_file():
        raise SystemExit(f"Corpus CSV not found: {corpus_csv}")
    with corpus_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "instance" not in reader.fieldnames:
            raise SystemExit(f"{corpus_csv}: expected CSV header containing 'instance'")
        instances = [row["instance"].strip() for row in reader if row.get("instance", "").strip()]
    if not instances:
        raise SystemExit(f"{corpus_csv}: no instances found")
    return instances


def flatten_solver_args(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        out.extend(["--solver-arg", value])
    return out


def validate_solver_args(solver_args: list[str]) -> None:
    conflicting = []
    for arg in solver_args:
        for key in CONTRACT_OVERRIDE_PREFIXES:
            if arg == key or arg.startswith(f"{key}="):
                conflicting.append(arg)
                break
    if conflicting:
        raise SystemExit(
            "Conflicting --solver-arg values for deterministic MIP gate: "
            f"{conflicting}. Use dedicated gate flags instead."
        )


def benchmark(
    *,
    run_script: Path,
    binary: Path,
    miplib_dir: Path,
    out_csv: Path,
    repeats: int,
    threads: int,
    time_limit: float,
    node_limit: int,
    gap_tol: float,
    instances: list[str],
    solver_args: list[str],
) -> None:
    run(
        [
            sys.executable,
            str(run_script),
            "--binary",
            str(binary),
            "--miplib-dir",
            str(miplib_dir),
            "--output",
            str(out_csv),
            "--repeats",
            str(repeats),
            "--threads",
            str(threads),
            "--time-limit",
            f"{time_limit:g}",
            "--node-limit",
            str(node_limit),
            "--gap-tol",
            f"{gap_tol:g}",
            "--instances",
            ",".join(instances),
            *solver_args,
        ]
    )


def main() -> int:
    args = parse_args()

    candidate_binary = Path(args.candidate_binary)
    baseline_binary = Path(args.baseline_binary) if args.baseline_binary else None
    baseline_csv = Path(args.baseline_csv) if args.baseline_csv else None
    miplib_dir = Path(args.miplib_dir)
    corpus_csv = Path(args.corpus)

    if not candidate_binary.is_file() or not os.access(candidate_binary, os.X_OK):
        raise SystemExit(f"Candidate binary not executable: {candidate_binary}")
    if not miplib_dir.is_dir():
        raise SystemExit(f"MIPLIB directory not found: {miplib_dir}")
    if args.repeats < 1 or args.threads < 1:
        raise SystemExit("--repeats and --threads must be >= 1")
    validate_solver_args(args.solver_arg)

    use_baseline_csv = baseline_csv is not None and baseline_csv.is_file()
    if baseline_binary is not None:
        use_baseline_csv = False
    if not use_baseline_csv:
        if baseline_binary is None:
            raise SystemExit(
                "Need --baseline-binary when --baseline-csv is missing/not found."
            )
        if not baseline_binary.is_file() or not os.access(baseline_binary, os.X_OK):
            raise SystemExit(f"Baseline binary not executable: {baseline_binary}")

    instances = load_corpus_instances(corpus_csv)

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
    bench_solver_args = flatten_solver_args(deterministic_contract_args + args.solver_arg)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate_csv = out_dir / "candidate_mip.csv"
    effective_baseline_csv = baseline_csv if use_baseline_csv else (out_dir / "baseline_mip.csv")

    run_mip = PERF_DIR / "run_miplib_mip_bench.py"
    check = PERF_DIR / "check_regression.py"

    print("[mip-gate] Running candidate benchmark...")
    benchmark(
        run_script=run_mip,
        binary=candidate_binary,
        miplib_dir=miplib_dir,
        out_csv=candidate_csv,
        repeats=args.repeats,
        threads=args.threads,
        time_limit=args.time_limit,
        node_limit=args.node_limit,
        gap_tol=args.gap_tol,
        instances=instances,
        solver_args=bench_solver_args,
    )

    if not use_baseline_csv:
        print("[mip-gate] Running baseline benchmark...")
        benchmark(
            run_script=run_mip,
            binary=baseline_binary,
            miplib_dir=miplib_dir,
            out_csv=effective_baseline_csv,
            repeats=args.repeats,
            threads=args.threads,
            time_limit=args.time_limit,
            node_limit=args.node_limit,
            gap_tol=args.gap_tol,
            instances=instances,
            solver_args=bench_solver_args,
        )

    print(f"[mip-gate] Checking regression ({args.metric})...")
    rc = run(
        [
            sys.executable,
            str(check),
            "--baseline",
            str(effective_baseline_csv),
            "--candidate",
            str(candidate_csv),
            "--metric",
            args.metric,
            "--max-regression-pct",
            f"{args.max_regression_pct:g}",
            "--min-common-instances",
            str(args.min_common_instances),
        ],
        check=False,
    )

    print("[mip-gate] Results:")
    print(f"  Candidate CSV: {candidate_csv}")
    print(f"  Baseline CSV:  {effective_baseline_csv}")
    if rc == 0:
        print("[mip-gate] PASS")
    else:
        print("[mip-gate] FAIL")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
