#!/usr/bin/env python3
"""Run a bounded MIPLIB-vs-HiGHS gap summary on a shared Mittelman subset."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
PERF_DIR = ROOT_DIR / "tests" / "perf"
DEFAULT_INSTANCES = (
    "air04,air05,blend2,flugpl,gt2,p0201,supportcase16,stein45inf"
)
SOLVED_STATUSES = {"optimal", "gap_limit"}


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
    p.add_argument("--binary", default=str(ROOT_DIR / "build" / "mipx-solve"))
    p.add_argument("--highs-binary", default="")
    p.add_argument("--miplib-dir", default=str(ROOT_DIR / "tests" / "data" / "miplib"))
    p.add_argument("--out-dir", default="/tmp/mipx_mip_gap_summary")
    p.add_argument("--instances", default=DEFAULT_INSTANCES)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--time-limit", type=float, default=60.0)
    p.add_argument("--gap-tol", type=float, default=1e-4)
    p.add_argument("--solver-arg", action="append", default=[])
    return p.parse_args(normalize_solver_arg_tokens(sys.argv[1:]))


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def load_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit(f"{path}: missing CSV header")
        rows: dict[str, dict[str, str]] = {}
        for row in reader:
            name = row.get("instance", "").strip()
            if name:
                rows[name] = row
        return rows


def parse_float(raw: str) -> float | None:
    text = raw.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def geomean(values: list[float]) -> float | None:
    if not values or any(v <= 0.0 or not math.isfinite(v) for v in values):
        return None
    return float(statistics.geometric_mean(values))


def count_statuses(rows: dict[str, dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows.values():
        status = (row.get("status") or "unknown").strip().lower() or "unknown"
        counts[status] = counts.get(status, 0) + 1
    return counts


def is_solved_status(status: str) -> bool:
    return status in SOLVED_STATUSES


def compare_rows(
    mipx_rows: dict[str, dict[str, str]],
    highs_rows: dict[str, dict[str, str]],
) -> dict[str, object]:
    common = sorted(set(mipx_rows) & set(highs_rows))
    both_optimal: list[str] = []
    mipx_only_optimal: list[str] = []
    highs_only_optimal: list[str] = []
    neither_optimal: list[str] = []
    time_ratios: list[float] = []
    time_rows: list[dict[str, object]] = []

    for name in common:
        mipx_status = (mipx_rows[name].get("status") or "").strip().lower()
        highs_status = (highs_rows[name].get("status") or "").strip().lower()

        if is_solved_status(mipx_status) and is_solved_status(highs_status):
            both_optimal.append(name)
            mipx_time = parse_float(mipx_rows[name].get("time_seconds", ""))
            highs_time = parse_float(highs_rows[name].get("time_seconds", ""))
            if mipx_time is not None and highs_time is not None and highs_time > 0.0:
                ratio = mipx_time / highs_time
                time_ratios.append(ratio)
                time_rows.append(
                    {
                        "instance": name,
                        "mipx_time_seconds": mipx_time,
                        "highs_time_seconds": highs_time,
                        "ratio": ratio,
                    }
                )
        elif is_solved_status(mipx_status):
            mipx_only_optimal.append(name)
        elif is_solved_status(highs_status):
            highs_only_optimal.append(name)
        else:
            neither_optimal.append(name)

    time_rows.sort(key=lambda row: row["ratio"], reverse=True)
    geomean_ratio = geomean(time_ratios)
    median_ratio = float(statistics.median(time_ratios)) if time_ratios else None
    return {
        "common_instances": common,
        "both_optimal": both_optimal,
        "mipx_only_optimal": mipx_only_optimal,
        "highs_only_optimal": highs_only_optimal,
        "neither_optimal": neither_optimal,
        "time_geomean_ratio": geomean_ratio,
        "time_median_ratio": median_ratio,
        "slowest_losses": time_rows[:5],
        "best_wins": list(reversed(time_rows[-5:])),
    }


def render_markdown(
    *,
    instances: list[str],
    mipx_rows: dict[str, dict[str, str]],
    highs_rows: dict[str, dict[str, str]],
    comparison: dict[str, object],
    args: argparse.Namespace,
) -> str:
    lines: list[str] = []
    lines.append("# Bounded MIP Gap Summary")
    lines.append("")
    lines.append("## Run")
    lines.append(f"- instances: {', '.join(instances)}")
    lines.append(f"- time limit: {args.time_limit:g}s")
    lines.append(f"- threads: {args.threads}")
    lines.append(f"- gap tolerance: {args.gap_tol:g}")
    lines.append("")
    lines.append("## Solve Status")
    mipx_counts = count_statuses(mipx_rows)
    highs_counts = count_statuses(highs_rows)
    mipx_solved = sum(count for status, count in mipx_counts.items() if is_solved_status(status))
    highs_solved = sum(count for status, count in highs_counts.items() if is_solved_status(status))
    lines.append(f"- mipx solved: {mipx_solved}/{len(instances)}")
    lines.append(f"- HiGHS solved: {highs_solved}/{len(instances)}")
    lines.append(f"- both solved: {len(comparison['both_optimal'])}")
    lines.append(f"- mipx-only solved: {len(comparison['mipx_only_optimal'])}")
    lines.append(f"- HiGHS-only solved: {len(comparison['highs_only_optimal'])}")
    lines.append("")
    lines.append("## Time Ratio")
    geomean_ratio = comparison["time_geomean_ratio"]
    median_ratio = comparison["time_median_ratio"]
    if geomean_ratio is None:
        lines.append("- no common solved runs with valid timing")
    else:
        lines.append(f"- geomean mipx/HiGHS time ratio on common solved runs: {geomean_ratio:.3f}x")
        lines.append(f"- median mipx/HiGHS time ratio on common solved runs: {median_ratio:.3f}x")
    lines.append("")
    lines.append("## Per-Instance")
    lines.append("| Instance | mipx status | HiGHS status | mipx time (s) | HiGHS time (s) | Ratio |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name in instances:
        mipx_row = mipx_rows.get(name, {})
        highs_row = highs_rows.get(name, {})
        mipx_status = mipx_row.get("status", "missing")
        highs_status = highs_row.get("status", "missing")
        mipx_time = parse_float(mipx_row.get("time_seconds", "")) if mipx_row else None
        highs_time = parse_float(highs_row.get("time_seconds", "")) if highs_row else None
        ratio = ""
        if mipx_time is not None and highs_time is not None and highs_time > 0.0:
            ratio = f"{mipx_time / highs_time:.3f}x"
        lines.append(
            f"| {name} | {mipx_status} | {highs_status} | "
            f"{'' if mipx_time is None else f'{mipx_time:.3f}'} | "
            f"{'' if highs_time is None else f'{highs_time:.3f}'} | {ratio} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    instances = [name.strip() for name in args.instances.split(",") if name.strip()]
    if not instances:
        raise SystemExit("no instances selected")

    mipx_csv = out_dir / "mipx_mip_gap.csv"
    highs_csv = out_dir / "highs_mip_gap.csv"
    summary_json = out_dir / "summary.json"
    summary_md = out_dir / "summary.md"

    solver_args: list[str] = []
    for sarg in args.solver_arg:
        solver_args.extend(["--solver-arg", sarg])

    run(
        [
            sys.executable,
            str(PERF_DIR / "run_mittelman_mip_bench.py"),
            "--binary",
            args.binary,
            "--miplib-dir",
            args.miplib_dir,
            "--output",
            str(mipx_csv),
            "--threads",
            str(args.threads),
            "--time-limit",
            f"{args.time_limit:g}",
            "--gap-tol",
            f"{args.gap_tol:g}",
            "--instances",
            ",".join(instances),
            "--repeats",
            "1",
            *solver_args,
        ]
    )

    run(
        [
            sys.executable,
            str(PERF_DIR / "run_highs_bench.py"),
            "--mode",
            "mip",
            "--highs-binary",
            args.highs_binary,
            "--instances-dir",
            args.miplib_dir,
            "--instances",
            ",".join(instances),
            "--output",
            str(highs_csv),
            "--repeats",
            "1",
            "--threads",
            str(args.threads),
            "--time-limit",
            f"{args.time_limit:g}",
            "--mip-rel-gap",
            f"{args.gap_tol:g}",
            "--presolve",
            "choose",
            "--solver",
            "choose",
        ]
    )

    mipx_rows = load_rows(mipx_csv)
    highs_rows = load_rows(highs_csv)
    comparison = compare_rows(mipx_rows, highs_rows)
    payload = {
        "instances": instances,
        "mipx_status_counts": count_statuses(mipx_rows),
        "highs_status_counts": count_statuses(highs_rows),
        "comparison": comparison,
    }

    summary_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    summary_md.write_text(
        render_markdown(
            instances=instances,
            mipx_rows=mipx_rows,
            highs_rows=highs_rows,
            comparison=comparison,
            args=args,
        ),
        encoding="utf-8",
    )
    print(summary_md.read_text(encoding="utf-8"), end="")
    print(f"\nWrote {summary_json}")
    print(f"Wrote {summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
