#!/usr/bin/env python3
"""Generate mipx MIP regression baseline CSV for strict work_units gating."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
PERF_DIR = ROOT_DIR / "tests" / "perf"
DEFAULT_BINARY = ROOT_DIR / "build" / "mipx-solve"
DEFAULT_MIPLIB_DIR = ROOT_DIR / "tests" / "data" / "miplib"
DEFAULT_CORPUS = PERF_DIR / "mip_regression_corpus.csv"
DEFAULT_OUTPUT = (
    ROOT_DIR
    / "tests"
    / "perf"
    / "baselines"
    / "mipx_mip_regression_small_seed1_t1_stable.csv"
)
DEFAULT_META = (
    ROOT_DIR
    / "tests"
    / "perf"
    / "baselines"
    / "mipx_mip_regression_small_seed1_t1_stable_meta.json"
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
    p.add_argument("--binary", default=str(DEFAULT_BINARY))
    p.add_argument("--miplib-dir", default=str(DEFAULT_MIPLIB_DIR))
    p.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    p.add_argument("--meta-json", default=str(DEFAULT_META))
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
    p.add_argument("--solver-arg", action="append", default=[])
    return p.parse_args(normalize_solver_arg_tokens(sys.argv[1:]))


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


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def capture(cmd: list[str], cwd: Path | None = None) -> str:
    try:
        return subprocess.check_output(cmd, cwd=cwd, text=True).strip()
    except Exception:
        return ""


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
            "Conflicting --solver-arg values for deterministic baseline generation: "
            f"{conflicting}. Use dedicated script flags instead."
        )


def main() -> int:
    args = parse_args()

    binary = Path(args.binary)
    miplib_dir = Path(args.miplib_dir)
    corpus = Path(args.corpus)
    output = Path(args.output)
    meta_json = Path(args.meta_json)

    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise SystemExit(f"Binary not executable: {binary}")
    if not miplib_dir.is_dir():
        raise SystemExit(f"MIPLIB directory not found: {miplib_dir}")
    if args.repeats < 1 or args.threads < 1:
        raise SystemExit("--repeats and --threads must be >= 1")
    validate_solver_args(args.solver_arg)

    instances = load_corpus_instances(corpus)

    output.parent.mkdir(parents=True, exist_ok=True)
    meta_json.parent.mkdir(parents=True, exist_ok=True)

    run_mip = PERF_DIR / "run_miplib_mip_bench.py"

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

    run(
        [
            sys.executable,
            str(run_mip),
            "--binary",
            str(binary),
            "--miplib-dir",
            str(miplib_dir),
            "--output",
            str(output),
            "--repeats",
            str(args.repeats),
            "--threads",
            str(args.threads),
            "--time-limit",
            f"{args.time_limit:g}",
            "--node-limit",
            str(args.node_limit),
            "--gap-tol",
            f"{args.gap_tol:g}",
            "--instances",
            ",".join(instances),
            *bench_solver_args,
        ]
    )

    meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": capture(["git", "rev-parse", "HEAD"], cwd=ROOT_DIR),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "binary": str(binary),
        "miplib_dir": str(miplib_dir),
        "corpus_csv": str(corpus),
        "instances": instances,
        "repeats": args.repeats,
        "threads": args.threads,
        "time_limit": args.time_limit,
        "node_limit": args.node_limit,
        "gap_tol": args.gap_tol,
        "parallel_mode": args.parallel_mode,
        "search_profile": args.search_profile,
        "seed": args.seed,
    }
    cpu = capture(["bash", "-lc", "lscpu | head -n 20"])
    if cpu:
        meta["cpu_info"] = cpu
    meta_json.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print("Wrote MIP regression baseline:")
    print(f"  CSV:  {output}")
    print(f"  Meta: {meta_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
