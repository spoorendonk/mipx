#!/usr/bin/env python3
"""Run barrier LP self-regression gate (candidate vs baseline mipx binaries).

This workflow enforces two independent checks:
1) Algorithmic regressions: strict `work_units` checks (GPU lane by default;
   optional CPU lane when enabled).
2) Wall-clock bands: optional `time_seconds` checks (GPU lane by default;
   optional CPU SIMD/AVX lane when enabled), with separate tolerances for
   machine-noise-aware gating.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
PERF_DIR = ROOT_DIR / "tests" / "perf"
DEFAULT_GPU_STABLE_SET = (
    ROOT_DIR / "tests" / "perf" / "baselines" / "barrier_gpu_stable_netlib_small.txt"
)
DEFAULT_GPU_BASELINE_CSV = (
    ROOT_DIR / "tests" / "perf" / "baselines" / "mipx_barrier_gpu_netlib_small.csv"
)
DEFAULT_CPU_BASELINE_CSV = (
    ROOT_DIR / "tests" / "perf" / "baselines" / "mipx_barrier_cpu_netlib_small.csv"
)


@dataclass(frozen=True)
class GateLane:
    name: str
    solver: str
    metric: str
    max_regression_pct: float
    min_common_instances: int


@dataclass(frozen=True)
class FilterStats:
    total_solver_rows: int
    kept_rows: int
    dropped_non_optimal: int
    dropped_missing_metric: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate-binary", required=True)
    p.add_argument(
        "--baseline-binary",
        default="",
        help=(
            "Optional baseline binary for exploratory candidate-vs-binary comparisons. "
            "If omitted, the committed barrier baseline CSVs are used."
        ),
    )
    p.add_argument(
        "--baseline-gpu-csv",
        default=str(DEFAULT_GPU_BASELINE_CSV),
        help="Committed GPU baseline CSV used by default for the GPU regression lane.",
    )
    p.add_argument(
        "--baseline-cpu-csv",
        default=str(DEFAULT_CPU_BASELINE_CSV),
        help="Committed CPU baseline CSV used when CPU lanes are explicitly enabled.",
    )
    p.add_argument("--instances-dir", default=str(ROOT_DIR / "tests" / "data" / "netlib"))
    p.add_argument("--instances", default="")
    p.add_argument("--max-instances", type=int, default=0)
    p.add_argument(
        "--all-available-instances",
        action="store_true",
        help=(
            "Use all available/selected instances instead of the default curated "
            "stable GPU barrier set when --instances is omitted."
        ),
    )
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--time-limit", type=float, default=60.0)
    p.add_argument(
        "--force-mipx-gpu",
        dest="force_mipx_gpu",
        action="store_true",
        default=True,
        help="Force mipx GPU lane via --gpu-min-rows 0 --gpu-min-nnz 0 (default: on).",
    )
    p.add_argument(
        "--auto-mipx-gpu",
        dest="force_mipx_gpu",
        action="store_false",
        help="Do not force the mipx GPU lane; use default GPU thresholds.",
    )
    p.add_argument(
        "--disable-presolve",
        dest="disable_presolve",
        action="store_true",
        default=True,
        help="Disable presolve to isolate barrier behavior (default: on).",
    )
    p.add_argument(
        "--enable-presolve",
        dest="disable_presolve",
        action="store_false",
        help="Enable presolve during barrier regression runs.",
    )
    p.add_argument(
        "--relax-integrality",
        action="store_true",
        help="Solve LP relaxations when MIP instances are used.",
    )
    p.add_argument("--out-dir", default="/tmp/mipx_barrier_gate")

    # Always-on algorithmic gates.
    p.add_argument(
        "--cpu-work-max-regression-pct",
        type=float,
        default=0.0,
        help="Allowed median work_units regression for CPU barrier lane (default: 0.0).",
    )
    p.add_argument(
        "--gpu-work-max-regression-pct",
        type=float,
        default=0.0,
        help="Allowed median work_units regression for GPU barrier lane (default: 0.0).",
    )
    p.add_argument(
        "--work-min-common-instances",
        type=int,
        default=5,
        help="Minimum common optimal instances required for work_units checks.",
    )
    p.add_argument(
        "--enable-cpu-barrier-lanes",
        action="store_true",
        help=(
            "Enable CPU barrier regression lanes. "
            "Disabled by default while CPU barrier backends are under development."
        ),
    )

    # Opt-in wall-clock band checks.
    p.add_argument(
        "--enable-wall-clock-bands",
        action="store_true",
        help="Enable opt-in wall-clock regression checks (separate CPU/GPU bands).",
    )
    p.add_argument(
        "--simd-wall-clock-max-regression-pct",
        type=float,
        default=10.0,
        help="Allowed median wall-clock regression for CPU (SIMD/AVX) lane.",
    )
    p.add_argument(
        "--gpu-wall-clock-max-regression-pct",
        type=float,
        default=15.0,
        help="Allowed median wall-clock regression for GPU lane.",
    )
    p.add_argument(
        "--wall-clock-min-common-instances",
        type=int,
        default=5,
        help="Minimum common optimal instances required for wall-clock checks.",
    )
    return p.parse_args()


def load_default_gpu_stable_instances() -> list[str]:
    if not DEFAULT_GPU_STABLE_SET.is_file():
        raise SystemExit(f"default GPU stable-set file not found: {DEFAULT_GPU_STABLE_SET}")
    return [
        line.strip()
        for line in DEFAULT_GPU_STABLE_SET.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def resolve_instance_filter(args: argparse.Namespace) -> str:
    if args.instances.strip():
        return args.instances.strip()
    if args.max_instances > 0 or args.all_available_instances:
        return ""
    return ",".join(load_default_gpu_stable_instances())


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def ensure_executable(path: Path, what: str) -> None:
    if not path.is_file() or not os.access(path, os.X_OK):
        raise SystemExit(f"{what} not executable: {path}")


def run_barrier_compare(binary: Path, output_csv: Path, args: argparse.Namespace) -> None:
    compare_script = PERF_DIR / "run_barrier_lp_compare.py"
    instances_filter = resolve_instance_filter(args)
    cmd = [
        sys.executable,
        str(compare_script),
        "--mipx-binary",
        str(binary),
        "--instances-dir",
        args.instances_dir,
        "--output",
        str(output_csv),
        "--repeats",
        str(args.repeats),
        "--threads",
        str(args.threads),
        "--time-limit",
        f"{args.time_limit:g}",
        "--no-highs",
        "--no-cuopt",
    ]
    if args.disable_presolve:
        cmd.append("--disable-presolve")
    if args.force_mipx_gpu:
        cmd.append("--force-mipx-gpu")
    if args.relax_integrality:
        cmd.append("--relax-integrality")
    if instances_filter:
        cmd.extend(["--instances", instances_filter])
    if args.max_instances > 0:
        cmd.extend(["--max-instances", str(args.max_instances)])
    run(cmd)


def write_lane_metric_csv(
    input_csv: Path,
    output_csv: Path,
    solver: str | None,
    metric: str,
) -> FilterStats:
    rows: dict[str, float] = {}
    total_solver_rows = 0
    dropped_non_optimal = 0
    dropped_missing_metric = 0

    with input_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        has_solver_column = reader.fieldnames is not None and "solver" in reader.fieldnames
        for row in reader:
            if solver is not None and has_solver_column and row.get("solver", "").strip() != solver:
                continue
            total_solver_rows += 1

            status = row.get("status", "").strip().lower()
            if status != "optimal":
                dropped_non_optimal += 1
                continue

            raw = row.get(metric, "").strip()
            if not raw:
                dropped_missing_metric += 1
                continue

            try:
                value = float(raw)
            except ValueError:
                dropped_missing_metric += 1
                continue

            if value <= 0.0:
                dropped_missing_metric += 1
                continue

            instance = row.get("instance", "").strip()
            if not instance:
                continue
            rows[instance] = value

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["instance", metric])
        writer.writeheader()
        for instance in sorted(rows):
            writer.writerow({"instance": instance, metric: f"{rows[instance]:.12g}"})

    return FilterStats(
        total_solver_rows=total_solver_rows,
        kept_rows=len(rows),
        dropped_non_optimal=dropped_non_optimal,
        dropped_missing_metric=dropped_missing_metric,
    )


def run_lane_gate(
    lane: GateLane,
    baseline_csv: Path,
    baseline_solver: str | None,
    candidate_compare_csv: Path,
    out_dir: Path,
) -> None:
    check_script = PERF_DIR / "check_regression.py"
    baseline_metric_csv = out_dir / f"baseline_{lane.solver}_{lane.metric}.csv"
    candidate_metric_csv = out_dir / f"candidate_{lane.solver}_{lane.metric}.csv"

    base_stats = write_lane_metric_csv(
        baseline_csv, baseline_metric_csv, baseline_solver, lane.metric
    )
    cand_stats = write_lane_metric_csv(
        candidate_compare_csv, candidate_metric_csv, lane.solver, lane.metric
    )

    print(
        f"[barrier-gate] {lane.name}: "
        f"baseline kept={base_stats.kept_rows}/{base_stats.total_solver_rows} "
        f"(non-opt={base_stats.dropped_non_optimal}, missing={base_stats.dropped_missing_metric}), "
        f"candidate kept={cand_stats.kept_rows}/{cand_stats.total_solver_rows} "
        f"(non-opt={cand_stats.dropped_non_optimal}, missing={cand_stats.dropped_missing_metric})"
    )

    run(
        [
            sys.executable,
            str(check_script),
            "--baseline",
            str(baseline_metric_csv),
            "--candidate",
            str(candidate_metric_csv),
            "--metric",
            lane.metric,
            "--max-regression-pct",
            f"{lane.max_regression_pct:g}",
            "--min-common-instances",
            str(lane.min_common_instances),
        ]
    )


def main() -> int:
    args = parse_args()

    candidate_binary = Path(args.candidate_binary)
    ensure_executable(candidate_binary, "candidate binary")
    baseline_binary = Path(args.baseline_binary) if args.baseline_binary else None
    if baseline_binary is not None:
        ensure_executable(baseline_binary, "baseline binary")

    baseline_gpu_csv = Path(args.baseline_gpu_csv)
    if not baseline_gpu_csv.is_file():
        raise SystemExit(f"baseline GPU CSV not found: {baseline_gpu_csv}")
    baseline_cpu_csv = Path(args.baseline_cpu_csv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    effective_instances = resolve_instance_filter(args)
    if effective_instances:
        print(f"[barrier-gate] Effective instance set: {effective_instances}")
    else:
        print("[barrier-gate] Effective instance set: all available (subject to --max-instances)")

    candidate_compare_csv = out_dir / "candidate_barrier_compare.csv"

    print("[barrier-gate] Running candidate barrier compare...")
    run_barrier_compare(candidate_binary, candidate_compare_csv, args)

    baseline_compare_csv: Path | None = None
    if baseline_binary is not None:
        baseline_compare_csv = out_dir / "baseline_barrier_compare.csv"
        print("[barrier-gate] Running baseline barrier compare...")
        run_barrier_compare(baseline_binary, baseline_compare_csv, args)
    else:
        print(f"[barrier-gate] Using committed GPU baseline CSV: {baseline_gpu_csv}")

    work_lanes = [
        GateLane(
            name="Algorithmic gate (GPU barrier, work_units)",
            solver="mipx_barrier_gpu",
            metric="work_units",
            max_regression_pct=args.gpu_work_max_regression_pct,
            min_common_instances=args.work_min_common_instances,
        ),
    ]
    if args.enable_cpu_barrier_lanes:
        if not baseline_cpu_csv.is_file():
            raise SystemExit(f"baseline CPU CSV not found: {baseline_cpu_csv}")
        work_lanes.insert(
            0,
            GateLane(
                name="Algorithmic gate (CPU barrier, work_units)",
                solver="mipx_barrier_cpu",
                metric="work_units",
                max_regression_pct=args.cpu_work_max_regression_pct,
                min_common_instances=args.work_min_common_instances,
            ),
        )

    for lane in work_lanes:
        print(f"\n=== {lane.name} ===")
        if lane.solver == "mipx_barrier_gpu":
            lane_baseline_csv = baseline_compare_csv or baseline_gpu_csv
        else:
            lane_baseline_csv = baseline_compare_csv or baseline_cpu_csv
        baseline_solver = lane.solver if baseline_compare_csv is not None else None
        run_lane_gate(
            lane, lane_baseline_csv, baseline_solver, candidate_compare_csv, out_dir
        )

    if args.enable_wall_clock_bands:
        time_lanes = [
            GateLane(
                name="Wall-clock band (GPU barrier lane)",
                solver="mipx_barrier_gpu",
                metric="time_seconds",
                max_regression_pct=args.gpu_wall_clock_max_regression_pct,
                min_common_instances=args.wall_clock_min_common_instances,
            ),
        ]
        if args.enable_cpu_barrier_lanes:
            time_lanes.insert(
                0,
                GateLane(
                    name="Wall-clock band (CPU barrier SIMD/AVX lane)",
                    solver="mipx_barrier_cpu",
                    metric="time_seconds",
                    max_regression_pct=args.simd_wall_clock_max_regression_pct,
                    min_common_instances=args.wall_clock_min_common_instances,
                ),
            )
        for lane in time_lanes:
            print(f"\n=== {lane.name} ===")
            if lane.solver == "mipx_barrier_gpu":
                lane_baseline_csv = baseline_compare_csv or baseline_gpu_csv
            else:
                lane_baseline_csv = baseline_compare_csv or baseline_cpu_csv
            baseline_solver = lane.solver if baseline_compare_csv is not None else None
            run_lane_gate(
                lane, lane_baseline_csv, baseline_solver, candidate_compare_csv, out_dir
            )
    else:
        print(
            "\n[barrier-gate] Wall-clock bands skipped "
            "(use --enable-wall-clock-bands to opt in)."
        )

    print("\n[barrier-gate] PASS")
    if baseline_compare_csv is not None:
        print(f"[barrier-gate] Baseline compare CSV: {baseline_compare_csv}")
    else:
        print(f"[barrier-gate] Baseline GPU CSV: {baseline_gpu_csv}")
    print(f"[barrier-gate] Candidate compare CSV: {candidate_compare_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
