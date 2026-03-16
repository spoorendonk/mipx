#!/usr/bin/env python3
"""Run PDLP LP self-regression gates (candidate vs baseline mipx binaries).

Default gate:
- strict algorithmic regression on `work_units` (CPU + forced-GPU lanes)

Optional wall-clock gate (`--wall-clock-gate`):
- CPU single-thread (`SIMD/AVX` proxy): `time_seconds`
- CPU multi-thread (`--wall-mt-threads`): `time_seconds`
- forced-GPU: `time_seconds`
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
PERF_DIR = ROOT_DIR / "tests" / "perf"
COMPARE_SCRIPT = PERF_DIR / "run_pdlp_lp_compare.py"
CHECK_SCRIPT = PERF_DIR / "check_regression.py"


@dataclass(frozen=True)
class GateScenario:
    name: str
    solver: str
    metric: str
    repeats: int
    threads: int
    time_limit: float
    force_gpu: bool
    max_regression_pct: float
    min_common_instances: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate-binary", required=True)
    p.add_argument("--baseline-binary", required=True)
    p.add_argument("--netlib-dir", default=str(ROOT_DIR / "tests" / "data" / "netlib"))
    p.add_argument(
        "--instances",
        default="",
        help="Comma-separated instance names (without .mps/.mps.gz). Empty means all.",
    )
    p.add_argument(
        "--max-instances",
        type=int,
        default=0,
        help="Limit number of instances after filtering (0 means no limit).",
    )
    p.add_argument(
        "--tmp-dir",
        default="/tmp/mipx_pdlp_gate",
        help="Directory for intermediate comparison CSVs.",
    )

    # Algorithmic gate (backward-compatible with legacy arguments).
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--time-limit", type=float, default=60)
    p.add_argument("--metric", default="work_units")
    p.add_argument("--max-regression-pct", type=float, default=0.0)
    p.add_argument("--min-common-instances", type=int, default=5)

    # Optional wall-clock gates.
    p.add_argument(
        "--wall-clock-gate",
        action="store_true",
        help="Enable wall-clock regression checks for SIMD/AVX, multi-thread, and GPU paths.",
    )
    p.add_argument("--wall-repeats", type=int, default=5)
    p.add_argument("--wall-time-limit", type=float, default=60)
    p.add_argument("--wall-min-common-instances", type=int, default=5)
    p.add_argument(
        "--wall-cpu-max-regression-pct",
        type=float,
        default=10.0,
        help="Allowed median time regression for CPU single-thread scenario.",
    )
    p.add_argument(
        "--wall-mt-max-regression-pct",
        type=float,
        default=10.0,
        help="Allowed median time regression for CPU multi-thread scenario.",
    )
    p.add_argument(
        "--wall-gpu-max-regression-pct",
        type=float,
        default=15.0,
        help="Allowed median time regression for GPU scenario.",
    )
    p.add_argument(
        "--wall-mt-threads",
        type=int,
        default=8,
        help="Requested thread count for the multi-thread wall-clock scenario.",
    )
    p.add_argument(
        "--wall-require-multithreading",
        action="store_true",
        help="Fail instead of skipping if PDLP LP path does not honor >1 thread.",
    )
    p.add_argument(
        "--wall-require-gpu-backend",
        action="store_true",
        help="Fail instead of skipping if forced-GPU probe does not report GPU backend.",
    )
    return p.parse_args()


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_capture(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(cmd))
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


def filter_solver_csv(in_csv: Path, out_csv: Path, solver: str, metric: str) -> None:
    with in_csv.open("r", encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("solver", "").strip() == solver]

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["instance", metric])
        w.writeheader()
        for row in rows:
            w.writerow({"instance": row.get("instance", ""), metric: row.get(metric, "")})


def tiny_lp_mps(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "NAME          TINY",
                "ROWS",
                " N  COST",
                " L  C1",
                "COLUMNS",
                "    X1        COST      1",
                "    X1        C1        1",
                "RHS",
                "    RHS1      C1        1",
                "BOUNDS",
                " LO BND1      X1        0",
                "ENDATA",
                "",
            ]
        ),
        encoding="utf-8",
    )


def probe_lp_thread_cap(binary: Path, requested_threads: int) -> int | None:
    with tempfile.TemporaryDirectory(prefix="mipx_pdlp_probe_") as td:
        mps = Path(td) / "tiny.mps"
        tiny_lp_mps(mps)
        proc = run_capture(
            [
                str(binary),
                str(mps),
                "--pdlp",
                "--quiet",
                "--no-presolve",
                "--no-gpu",
                "--threads",
                str(max(1, requested_threads)),
            ]
        )
        out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        m = re.search(r"using up to\s+(\d+)\s+thread", out, re.IGNORECASE)
        if m:
            return int(m.group(1))
        return None


def probe_gpu_backend(binary: Path) -> str | None:
    with tempfile.TemporaryDirectory(prefix="mipx_pdlp_probe_") as td:
        mps = Path(td) / "tiny.mps"
        tiny_lp_mps(mps)
        proc = run_capture(
            [
                str(binary),
                str(mps),
                "--pdlp",
                "--no-presolve",
                "--gpu",
                "--gpu-min-rows",
                "0",
                "--gpu-min-nnz",
                "0",
            ]
        )
        out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        m = re.search(r"PDLP backend:\s*(GPU|CPU)", out)
        if m:
            return m.group(1)
        return None


def run_compare(
    binary: Path,
    netlib_dir: Path,
    out_csv: Path,
    scenario: GateScenario,
    instances: str,
    max_instances: int,
) -> None:
    cmd = [
        sys.executable,
        str(COMPARE_SCRIPT),
        "--mipx-binary",
        str(binary),
        "--instances-dir",
        str(netlib_dir),
        "--output",
        str(out_csv),
        "--repeats",
        str(scenario.repeats),
        "--threads",
        str(max(1, scenario.threads)),
        "--time-limit",
        f"{scenario.time_limit:g}",
        "--disable-presolve",
        "--no-highs",
        "--no-cuopt",
    ]
    if scenario.force_gpu:
        cmd.append("--force-mipx-gpu")
    if instances:
        cmd.extend(["--instances", instances])
    if max_instances > 0:
        cmd.extend(["--max-instances", str(max_instances)])
    run(cmd)


def run_scenario(
    scenario: GateScenario,
    candidate_binary: Path,
    baseline_binary: Path,
    netlib_dir: Path,
    out_dir: Path,
    instances: str,
    max_instances: int,
) -> None:
    print(f"\n=== PDLP gate: {scenario.name} ({scenario.metric}) ===")
    cand_compare = out_dir / f"{scenario.name}_candidate_compare.csv"
    base_compare = out_dir / f"{scenario.name}_baseline_compare.csv"

    run_compare(candidate_binary, netlib_dir, cand_compare, scenario, instances, max_instances)
    run_compare(baseline_binary, netlib_dir, base_compare, scenario, instances, max_instances)

    cand_metric = out_dir / f"{scenario.name}_candidate_metric.csv"
    base_metric = out_dir / f"{scenario.name}_baseline_metric.csv"
    filter_solver_csv(cand_compare, cand_metric, scenario.solver, scenario.metric)
    filter_solver_csv(base_compare, base_metric, scenario.solver, scenario.metric)

    run(
        [
            sys.executable,
            str(CHECK_SCRIPT),
            "--baseline",
            str(base_metric),
            "--candidate",
            str(cand_metric),
            "--metric",
            scenario.metric,
            "--max-regression-pct",
            f"{scenario.max_regression_pct:g}",
            "--min-common-instances",
            str(scenario.min_common_instances),
        ]
    )


def is_executable(path: Path) -> bool:
    return path.is_file() and bool(path.stat().st_mode & 0o111)


def main() -> int:
    args = parse_args()

    candidate_binary = Path(args.candidate_binary)
    baseline_binary = Path(args.baseline_binary)
    netlib_dir = Path(args.netlib_dir)
    out_dir = Path(args.tmp_dir)

    if not is_executable(candidate_binary):
        raise SystemExit(f"candidate binary not executable: {candidate_binary}")
    if not is_executable(baseline_binary):
        raise SystemExit(f"baseline binary not executable: {baseline_binary}")
    if not netlib_dir.is_dir():
        raise SystemExit(f"netlib directory not found: {netlib_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios: list[GateScenario] = [
        GateScenario(
            name="algo_cpu",
            solver="mipx_pdlp_cpu",
            metric=args.metric,
            repeats=args.repeats,
            threads=1,
            time_limit=args.time_limit,
            force_gpu=False,
            max_regression_pct=args.max_regression_pct,
            min_common_instances=args.min_common_instances,
        ),
        GateScenario(
            name="algo_gpu_forced",
            solver="mipx_pdlp_gpu",
            metric=args.metric,
            repeats=args.repeats,
            threads=1,
            time_limit=args.time_limit,
            force_gpu=True,
            max_regression_pct=args.max_regression_pct,
            min_common_instances=args.min_common_instances,
        ),
    ]

    for scenario in scenarios:
        run_scenario(
            scenario,
            candidate_binary,
            baseline_binary,
            netlib_dir,
            out_dir,
            args.instances,
            args.max_instances,
        )

    if args.wall_clock_gate:
        wall_scenarios: list[GateScenario] = [
            GateScenario(
                name="wall_cpu_simd",
                solver="mipx_pdlp_cpu",
                metric="time_seconds",
                repeats=args.wall_repeats,
                threads=1,
                time_limit=args.wall_time_limit,
                force_gpu=False,
                max_regression_pct=args.wall_cpu_max_regression_pct,
                min_common_instances=args.wall_min_common_instances,
            ),
            GateScenario(
                name="wall_gpu_forced",
                solver="mipx_pdlp_gpu",
                metric="time_seconds",
                repeats=args.wall_repeats,
                threads=1,
                time_limit=args.wall_time_limit,
                force_gpu=True,
                max_regression_pct=args.wall_gpu_max_regression_pct,
                min_common_instances=args.wall_min_common_instances,
            ),
        ]

        if args.wall_mt_threads > 1:
            can_threads = probe_lp_thread_cap(candidate_binary, args.wall_mt_threads)
            base_threads = probe_lp_thread_cap(baseline_binary, args.wall_mt_threads)
            if (
                can_threads is not None
                and base_threads is not None
                and can_threads <= 1
                and base_threads <= 1
            ):
                msg = (
                    "Skipping multi-thread PDLP wall-clock gate: "
                    f"candidate uses up to {can_threads} thread, "
                    f"baseline uses up to {base_threads} thread."
                )
                print(f"\nWARNING: {msg}")
                if args.wall_require_multithreading:
                    raise SystemExit("wall multithreading is required but not available")
            else:
                wall_scenarios.append(
                    GateScenario(
                        name="wall_cpu_multithread",
                        solver="mipx_pdlp_cpu",
                        metric="time_seconds",
                        repeats=args.wall_repeats,
                        threads=args.wall_mt_threads,
                        time_limit=args.wall_time_limit,
                        force_gpu=False,
                        max_regression_pct=args.wall_mt_max_regression_pct,
                        min_common_instances=args.wall_min_common_instances,
                    )
                )

        can_gpu_backend = probe_gpu_backend(candidate_binary)
        base_gpu_backend = probe_gpu_backend(baseline_binary)
        if can_gpu_backend != "GPU" or base_gpu_backend != "GPU":
            msg = (
                "Skipping forced-GPU PDLP wall-clock gate: "
                f"candidate backend probe={can_gpu_backend or 'unknown'}, "
                f"baseline backend probe={base_gpu_backend or 'unknown'}."
            )
            print(f"\nWARNING: {msg}")
            if args.wall_require_gpu_backend:
                raise SystemExit("wall GPU backend is required but unavailable")
            wall_scenarios = [s for s in wall_scenarios if s.name != "wall_gpu_forced"]

        for scenario in wall_scenarios:
            run_scenario(
                scenario,
                candidate_binary,
                baseline_binary,
                netlib_dir,
                out_dir,
                args.instances,
                args.max_instances,
            )

    print("\nPDLP regression gates passed.")
    print(f"Artifacts: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
