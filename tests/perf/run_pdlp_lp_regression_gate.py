#!/usr/bin/env python3
"""Run PDLP LP self-regression gate (candidate vs baseline mipx binaries)."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
PERF_DIR = ROOT_DIR / "tests" / "perf"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate-binary", required=True)
    p.add_argument("--baseline-binary", required=True)
    p.add_argument("--netlib-dir", default=str(ROOT_DIR / "tests" / "data" / "netlib"))
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--time-limit", type=float, default=60)
    p.add_argument("--metric", default="work_units")
    p.add_argument("--max-regression-pct", type=float, default=0.0)
    p.add_argument("--min-common-instances", type=int, default=5)
    return p.parse_args()


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def filter_solver_csv(in_csv: Path, out_csv: Path, solver: str, metric: str) -> None:
    with in_csv.open("r", encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("solver", "").strip() == solver]

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["instance", metric])
        w.writeheader()
        for row in rows:
            w.writerow({"instance": row.get("instance", ""), metric: row.get(metric, "")})


def main() -> int:
    args = parse_args()

    can_bin = Path(args.candidate_binary)
    base_bin = Path(args.baseline_binary)
    if not can_bin.is_file() or not can_bin.stat().st_mode & 0o111:
        raise SystemExit(f"candidate binary not executable: {can_bin}")
    if not base_bin.is_file() or not base_bin.stat().st_mode & 0o111:
        raise SystemExit(f"baseline binary not executable: {base_bin}")

    compare_script = PERF_DIR / "run_pdlp_lp_compare.py"
    check_script = PERF_DIR / "check_regression.py"

    out_dir = Path("/tmp/mipx_pdlp_gate")
    out_dir.mkdir(parents=True, exist_ok=True)
    can_csv = out_dir / "candidate_compare.csv"
    base_csv = out_dir / "baseline_compare.csv"

    run(
        [
            sys.executable,
            str(compare_script),
            "--mipx-binary",
            str(can_bin),
            "--instances-dir",
            args.netlib_dir,
            "--output",
            str(can_csv),
            "--repeats",
            str(args.repeats),
            "--time-limit",
            f"{args.time_limit:g}",
            "--threads",
            "1",
            "--disable-presolve",
            "--no-highs",
            "--no-cuopt",
        ]
    )

    run(
        [
            sys.executable,
            str(compare_script),
            "--mipx-binary",
            str(base_bin),
            "--instances-dir",
            args.netlib_dir,
            "--output",
            str(base_csv),
            "--repeats",
            str(args.repeats),
            "--time-limit",
            f"{args.time_limit:g}",
            "--threads",
            "1",
            "--disable-presolve",
            "--no-highs",
            "--no-cuopt",
        ]
    )

    for solver in ("mipx_pdlp_cpu", "mipx_pdlp_gpu"):
        base_s = out_dir / f"baseline_{solver}.csv"
        can_s = out_dir / f"candidate_{solver}.csv"
        filter_solver_csv(base_csv, base_s, solver, args.metric)
        filter_solver_csv(can_csv, can_s, solver, args.metric)

        print(f"\n=== Regression gate: {solver} ({args.metric}) ===")
        run(
            [
                sys.executable,
                str(check_script),
                "--baseline",
                str(base_s),
                "--candidate",
                str(can_s),
                "--metric",
                args.metric,
                "--max-regression-pct",
                f"{args.max_regression_pct:g}",
                "--min-common-instances",
                str(args.min_common_instances),
            ]
        )

    print("\nPDLP regression gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
