#!/usr/bin/env python3
"""Run a benchmark matrix over solver x time x threads x mode.

Artifacts:
- matrix_detail.csv: one row per instance/config
- matrix_summary.csv: one row per config with medians/counts
- matrix_summary.md: human-readable ranking table
"""

from __future__ import annotations

import argparse
import csv
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
PERF_DIR = ROOT_DIR / "tests" / "perf"
RUN_MIPLIB = PERF_DIR / "run_miplib_mip_bench.py"
RUN_HIGHS = PERF_DIR / "run_highspy_bench.py"


def parse_csv_tokens(raw: str) -> list[str]:
    return [tok.strip() for tok in raw.split(",") if tok.strip()]


def parse_int_list(raw: str, name: str) -> list[int]:
    vals: list[int] = []
    for tok in parse_csv_tokens(raw):
        try:
            v = int(tok)
        except ValueError as exc:
            raise SystemExit(f"Invalid integer in --{name}: {tok}") from exc
        if v < 1:
            raise SystemExit(f"--{name} values must be >= 1")
        vals.append(v)
    if not vals:
        raise SystemExit(f"--{name} cannot be empty")
    return vals


def parse_float_list(raw: str, name: str) -> list[float]:
    vals: list[float] = []
    for tok in parse_csv_tokens(raw):
        try:
            v = float(tok)
        except ValueError as exc:
            raise SystemExit(f"Invalid float in --{name}: {tok}") from exc
        if v <= 0:
            raise SystemExit(f"--{name} values must be > 0")
        vals.append(v)
    if not vals:
        raise SystemExit(f"--{name} cannot be empty")
    return vals


def parse_float(value: str) -> float | None:
    text = value.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def median_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def format_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def collect_instances(miplib_dir: Path, instance_filter: str, max_instances: int) -> list[str]:
    available = sorted(p.name.removesuffix(".mps.gz") for p in miplib_dir.glob("*.mps.gz"))
    if not available:
        raise SystemExit(f"No .mps.gz instances found in {miplib_dir}")

    if instance_filter:
        requested = parse_csv_tokens(instance_filter)
        selected = [name for name in requested if name in set(available)]
        missing = [name for name in requested if name not in set(available)]
        for name in missing:
            print(f"Warning: requested instance not found: {name}", file=sys.stderr)
    else:
        selected = available

    if max_instances > 0:
        selected = selected[:max_instances]

    if not selected:
        raise SystemExit("No instances selected for matrix run.")
    return selected


def combo_tag(solver: str, mode: str, threads: int, time_limit: float) -> str:
    tl = f"{time_limit:g}".replace(".", "p")
    return f"{solver}_{mode}_t{threads}_tl{tl}"


def parse_mipx_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_highs_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def summarize_rows(rows: list[dict[str, str]]) -> dict[str, str]:
    status_counter = Counter(row["status"] for row in rows)
    ok_rows = [row for row in rows if row["status"] not in {"solve_error", "parse_error"}]

    times = [v for row in ok_rows if (v := parse_float(row["time_seconds"])) is not None]
    works = [v for row in ok_rows if (v := parse_float(row["work_units"])) is not None]
    nodes = [v for row in ok_rows if (v := parse_float(row["nodes"])) is not None]
    iters = [v for row in ok_rows if (v := parse_float(row["lp_iterations"])) is not None]

    status_mix = "; ".join(f"{k}:{status_counter[k]}" for k in sorted(status_counter))

    return {
        "instances_total": str(len(rows)),
        "instances_ok": str(len(ok_rows)),
        "median_time_seconds": format_float(median_or_none(times)),
        "median_work_units": format_float(median_or_none(works)),
        "median_nodes": format_float(median_or_none(nodes)),
        "median_lp_iterations": format_float(median_or_none(iters)),
        "status_mix": status_mix,
    }


def markdown_rank_value(row: dict[str, str], metric: str) -> tuple[float, bool]:
    metric_col = "median_work_units" if metric == "work_units" else "median_time_seconds"
    value = parse_float(row.get(metric_col, ""))
    if value is None:
        return (float("inf"), False)
    return (value, True)


def write_markdown(summary_rows: list[dict[str, str]], metric: str, out_path: Path) -> None:
    ranked = sorted(
        summary_rows,
        key=lambda row: (
            markdown_rank_value(row, metric)[0],
            parse_float(row.get("median_time_seconds", "")) or float("inf"),
            row["solver"],
            row["mode"],
            int(row["threads"]),
            float(row["time_limit"]),
        ),
    )

    lines = [
        "# Benchmark Matrix Summary",
        "",
        f"Ranking metric: `{metric}` (lower is better)",
        "",
        "| solver | mode | threads | time_limit | ok/total | median_work_units | median_time_seconds | median_nodes | median_lp_iterations |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in ranked:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["solver"],
                    row["mode"],
                    row["threads"],
                    row["time_limit"],
                    f"{row['instances_ok']}/{row['instances_total']}",
                    row["median_work_units"] or "-",
                    row["median_time_seconds"] or "-",
                    row["median_nodes"] or "-",
                    row["median_lp_iterations"] or "-",
                ]
            )
            + " |"
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mipx-binary", required=True)
    parser.add_argument("--miplib-dir", required=True)
    parser.add_argument("--out-dir", default="/tmp/mipx_matrix")
    parser.add_argument("--solvers", default="mipx")
    parser.add_argument("--modes", default="deterministic,opportunistic")
    parser.add_argument("--threads", default="1,4")
    parser.add_argument("--time-limits", default="30,120")
    parser.add_argument("--instances", default="p0201,gt2,flugpl")
    parser.add_argument("--max-instances", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--node-limit", type=int, default=100000)
    parser.add_argument("--gap-tol", type=float, default=1e-4)
    parser.add_argument("--metric", choices=("work_units", "time_seconds"), default="work_units")
    parser.add_argument("--highs-binary", default="")
    parser.add_argument("--solver-arg", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    mipx_binary = Path(args.mipx_binary)
    miplib_dir = Path(args.miplib_dir)
    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if not mipx_binary.is_file() or not mipx_binary.stat().st_mode & 0o111:
        raise SystemExit(f"mipx binary not executable: {mipx_binary}")
    if not miplib_dir.is_dir():
        raise SystemExit(f"MIPLIB dir not found: {miplib_dir}")

    solvers = parse_csv_tokens(args.solvers)
    if not solvers:
        raise SystemExit("--solvers cannot be empty")
    for solver in solvers:
        if solver not in {"mipx", "highs"}:
            raise SystemExit(f"Unsupported solver in --solvers: {solver}")

    if "highs" in solvers and not args.highs_binary:
        raise SystemExit("--highs-binary is required when --solvers includes highs")

    modes = parse_csv_tokens(args.modes)
    if not modes:
        raise SystemExit("--modes cannot be empty")
    for mode in modes:
        if mode not in {"deterministic", "opportunistic"}:
            raise SystemExit(f"Unsupported mode in --modes: {mode}")

    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")
    if args.node_limit < 1:
        raise SystemExit("--node-limit must be >= 1")
    if args.gap_tol <= 0:
        raise SystemExit("--gap-tol must be > 0")

    threads_list = parse_int_list(args.threads, "threads")
    time_limits = parse_float_list(args.time_limits, "time-limits")
    selected_instances = collect_instances(miplib_dir, args.instances, args.max_instances)
    instances_csv = ",".join(selected_instances)

    detail_rows: list[dict[str, str]] = []
    summary_rows: list[dict[str, str]] = []

    for solver in solvers:
        solver_modes = modes if solver == "mipx" else ["default"]

        for mode in solver_modes:
            for threads in threads_list:
                for time_limit in time_limits:
                    tag = combo_tag(solver, mode, threads, time_limit)
                    raw_csv = raw_dir / f"{tag}.csv"

                    if solver == "mipx":
                        mode_args = [
                            "--parallel-mode",
                            "deterministic" if mode == "deterministic" else "opportunistic",
                        ]
                        cmd = [
                            sys.executable,
                            str(RUN_MIPLIB),
                            "--binary",
                            str(mipx_binary),
                            "--miplib-dir",
                            str(miplib_dir),
                            "--output",
                            str(raw_csv),
                            "--repeats",
                            str(args.repeats),
                            "--threads",
                            str(threads),
                            "--time-limit",
                            f"{time_limit:g}",
                            "--node-limit",
                            str(args.node_limit),
                            "--gap-tol",
                            f"{args.gap_tol:g}",
                            "--instances",
                            instances_csv,
                            "--solver-arg",
                            "--seed",
                            "--solver-arg",
                            str(args.seed),
                        ]
                        for mode_arg in mode_args:
                            cmd.extend(["--solver-arg", mode_arg])
                        for extra in args.solver_arg:
                            cmd.extend(["--solver-arg", extra])
                        run(cmd)

                        raw_rows = parse_mipx_csv(raw_csv)
                        combo_rows: list[dict[str, str]] = []
                        for row in raw_rows:
                            mapped = {
                                "instance": row.get("instance", ""),
                                "solver": solver,
                                "mode": mode,
                                "threads": str(threads),
                                "time_limit": f"{time_limit:g}",
                                "repeats": str(args.repeats),
                                "status": row.get("status", "unknown"),
                                "time_seconds": row.get("time_seconds", ""),
                                "work_units": row.get("work_units", ""),
                                "nodes": row.get("nodes", ""),
                                "lp_iterations": row.get("lp_iterations", ""),
                                "objective": "",
                                "source_csv": str(raw_csv),
                            }
                            combo_rows.append(mapped)

                    else:
                        cmd = [
                            sys.executable,
                            str(RUN_HIGHS),
                            "--mode",
                            "mip",
                            "--instances-dir",
                            str(miplib_dir),
                            "--output",
                            str(raw_csv),
                            "--repeats",
                            str(args.repeats),
                            "--threads",
                            str(threads),
                            "--time-limit",
                            f"{time_limit:g}",
                            "--instances",
                            instances_csv,
                            "--highs-binary",
                            args.highs_binary,
                        ]
                        run(cmd)

                        raw_rows = parse_highs_csv(raw_csv)
                        combo_rows = []
                        for row in raw_rows:
                            mapped = {
                                "instance": row.get("instance", ""),
                                "solver": solver,
                                "mode": mode,
                                "threads": str(threads),
                                "time_limit": f"{time_limit:g}",
                                "repeats": str(args.repeats),
                                "status": row.get("status", "unknown"),
                                "time_seconds": row.get("time_seconds", ""),
                                "work_units": "",
                                "nodes": row.get("nodes", ""),
                                "lp_iterations": row.get("simplex_iterations", ""),
                                "objective": row.get("objective", ""),
                                "source_csv": str(raw_csv),
                            }
                            combo_rows.append(mapped)

                    summary = summarize_rows(combo_rows)
                    summary_rows.append(
                        {
                            "solver": solver,
                            "mode": mode,
                            "threads": str(threads),
                            "time_limit": f"{time_limit:g}",
                            "repeats": str(args.repeats),
                            **summary,
                        }
                    )
                    detail_rows.extend(combo_rows)

    detail_csv = out_dir / "matrix_detail.csv"
    summary_csv = out_dir / "matrix_summary.csv"
    summary_md = out_dir / "matrix_summary.md"

    detail_header = [
        "instance",
        "solver",
        "mode",
        "threads",
        "time_limit",
        "repeats",
        "status",
        "time_seconds",
        "work_units",
        "nodes",
        "lp_iterations",
        "objective",
        "source_csv",
    ]
    with detail_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=detail_header)
        writer.writeheader()
        writer.writerows(detail_rows)

    summary_header = [
        "solver",
        "mode",
        "threads",
        "time_limit",
        "repeats",
        "instances_total",
        "instances_ok",
        "median_time_seconds",
        "median_work_units",
        "median_nodes",
        "median_lp_iterations",
        "status_mix",
    ]
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_header)
        writer.writeheader()
        writer.writerows(summary_rows)

    write_markdown(summary_rows, args.metric, summary_md)

    print(f"Wrote matrix detail CSV: {detail_csv}")
    print(f"Wrote matrix summary CSV: {summary_csv}")
    print(f"Wrote matrix summary Markdown: {summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
