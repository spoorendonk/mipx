#!/usr/bin/env python3
"""Run deterministic reproducibility checks for single and multi-thread MIP solves.

Artifacts:
- determinism_detail.csv: one row per run
- determinism_summary.csv: one row per (instance, profile)
- determinism_summary.md: pass/fail report

Exit code is non-zero if any profile/instance is unstable.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


STATUS_RE = re.compile(r"^Status:\s*(.+?)\s*$", re.MULTILINE)
FLOAT_LINE_PATTERNS = {
    "objective": re.compile(r"^Objective:\s*([\-+0-9.eE]+)\s*$", re.MULTILINE),
    "nodes": re.compile(r"^Nodes:\s*([\-+0-9.eE]+)\s*$", re.MULTILINE),
    "lp_iterations": re.compile(r"^LP iterations:\s*([\-+0-9.eE]+)\s*$", re.MULTILINE),
    "work_units": re.compile(r"^Work units:\s*([\-+0-9.eE]+)\s*$", re.MULTILINE),
    "time_seconds": re.compile(r"^Time:\s*([\-+0-9.eE]+)s\s*$", re.MULTILINE),
}


@dataclass
class RunSample:
    instance: str
    profile: str
    threads: int
    run_index: int
    status: str
    objective: float | None
    nodes: float | None
    lp_iterations: float | None
    work_units: float | None
    time_seconds: float | None
    error: str


def parse_csv_tokens(raw: str) -> list[str]:
    return [tok.strip() for tok in raw.split(",") if tok.strip()]


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
        raise SystemExit("No instances selected for determinism suite.")
    return selected


def parse_float(pattern: re.Pattern[str], text: str) -> float | None:
    match = pattern.search(text)
    if not match:
        return None
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def parse_status(text: str) -> str:
    match = STATUS_RE.search(text)
    if not match:
        return "unknown"
    return match.group(1).strip().lower().replace(" ", "_")


def run_once(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    return proc.returncode, output


def approx_equal(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


def max_abs_delta(values: list[float]) -> float:
    if not values:
        return 0.0
    base = values[0]
    return max(abs(v - base) for v in values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--miplib-dir", required=True)
    parser.add_argument("--out-dir", default="/tmp/mipx_determinism")
    parser.add_argument("--instances", default="p0201,gt2,flugpl")
    parser.add_argument("--max-instances", type=int, default=0)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--single-threads", type=int, default=1)
    parser.add_argument("--multi-threads", type=int, default=4)
    parser.add_argument("--time-limit", type=float, default=30.0)
    parser.add_argument("--node-limit", type=int, default=100000)
    parser.add_argument("--gap-tol", type=float, default=1e-4)
    parser.add_argument("--value-tol", type=float, default=1e-9)
    parser.add_argument(
        "--strict-metrics",
        action="store_true",
        help="Also require node/lp-iteration/work-unit equality (default checks status/objective).",
    )
    parser.add_argument("--solver-arg", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    binary = Path(args.binary)
    miplib_dir = Path(args.miplib_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not binary.is_file() or not binary.stat().st_mode & 0o111:
        raise SystemExit(f"Binary not executable: {binary}")
    if not miplib_dir.is_dir():
        raise SystemExit(f"MIPLIB dir not found: {miplib_dir}")
    if args.runs < 2:
        raise SystemExit("--runs must be >= 2")
    if args.single_threads < 1 or args.multi_threads < 1:
        raise SystemExit("thread counts must be >= 1")
    if args.time_limit <= 0:
        raise SystemExit("--time-limit must be > 0")
    if args.node_limit < 1:
        raise SystemExit("--node-limit must be >= 1")
    if args.gap_tol <= 0:
        raise SystemExit("--gap-tol must be > 0")
    if args.value_tol < 0:
        raise SystemExit("--value-tol must be >= 0")

    selected_instances = collect_instances(miplib_dir, args.instances, args.max_instances)

    profiles: list[tuple[str, int]] = [("deterministic_t1", args.single_threads)]
    if args.multi_threads != args.single_threads:
        profiles.append(("deterministic_tn", args.multi_threads))

    detail: list[RunSample] = []

    for instance_name in selected_instances:
        instance_path = miplib_dir / f"{instance_name}.mps.gz"
        if not instance_path.is_file():
            raise SystemExit(f"Missing instance file: {instance_path}")

        for profile_name, threads in profiles:
            for run_index in range(1, args.runs + 1):
                cmd = [
                    str(binary),
                    str(instance_path),
                    "--threads",
                    str(threads),
                    "--time-limit",
                    f"{args.time_limit:g}",
                    "--node-limit",
                    str(args.node_limit),
                    "--gap-tol",
                    f"{args.gap_tol:g}",
                    "--heur-deterministic",
                    "--search-stable",
                    "--no-cuts",
                    "--seed",
                    str(args.seed),
                ]
                for sarg in args.solver_arg:
                    cmd.append(sarg)

                print("+", " ".join(cmd))
                code, output = run_once(cmd)

                status = parse_status(output)
                objective = parse_float(FLOAT_LINE_PATTERNS["objective"], output)
                nodes = parse_float(FLOAT_LINE_PATTERNS["nodes"], output)
                lp_iterations = parse_float(FLOAT_LINE_PATTERNS["lp_iterations"], output)
                work_units = parse_float(FLOAT_LINE_PATTERNS["work_units"], output)
                time_seconds = parse_float(FLOAT_LINE_PATTERNS["time_seconds"], output)

                error = ""
                if code != 0:
                    error = "solve_error"
                elif (
                    objective is None
                    or nodes is None
                    or lp_iterations is None
                    or work_units is None
                    or time_seconds is None
                ):
                    error = "parse_error"

                detail.append(
                    RunSample(
                        instance=instance_name,
                        profile=profile_name,
                        threads=threads,
                        run_index=run_index,
                        status=status,
                        objective=objective,
                        nodes=nodes,
                        lp_iterations=lp_iterations,
                        work_units=work_units,
                        time_seconds=time_seconds,
                        error=error,
                    )
                )

    detail_csv = out_dir / "determinism_detail.csv"
    with detail_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "instance",
                "profile",
                "threads",
                "run",
                "status",
                "objective",
                "nodes",
                "lp_iterations",
                "work_units",
                "time_seconds",
                "error",
            ]
        )
        for row in detail:
            writer.writerow(
                [
                    row.instance,
                    row.profile,
                    row.threads,
                    row.run_index,
                    row.status,
                    "" if row.objective is None else f"{row.objective:.12g}",
                    "" if row.nodes is None else f"{row.nodes:.12g}",
                    "" if row.lp_iterations is None else f"{row.lp_iterations:.12g}",
                    "" if row.work_units is None else f"{row.work_units:.12g}",
                    "" if row.time_seconds is None else f"{row.time_seconds:.12g}",
                    row.error,
                ]
            )

    grouped: dict[tuple[str, str], list[RunSample]] = {}
    for row in detail:
        grouped.setdefault((row.instance, row.profile), []).append(row)

    summary_rows: list[dict[str, str]] = []
    unstable_count = 0

    for (instance_name, profile_name), rows in sorted(grouped.items()):
        threads = rows[0].threads
        errors = [r.error for r in rows if r.error]

        stable = True
        notes = ""
        status_value = rows[0].status

        if errors:
            stable = False
            notes = ",".join(sorted(set(errors)))
        else:
            statuses = {r.status for r in rows}
            if len(statuses) != 1:
                stable = False
                notes = "status_mismatch"
            else:
                objectives = [r.objective for r in rows if r.objective is not None]
                nodes = [r.nodes for r in rows if r.nodes is not None]
                lp_iters = [r.lp_iterations for r in rows if r.lp_iterations is not None]
                work_units = [r.work_units for r in rows if r.work_units is not None]

                if not (
                    len(objectives) == len(rows)
                    and len(nodes) == len(rows)
                    and len(lp_iters) == len(rows)
                    and len(work_units) == len(rows)
                ):
                    stable = False
                    notes = "missing_metrics"
                else:
                    if max_abs_delta(objectives) > args.value_tol:
                        stable = False
                        notes = "objective_delta"
                    work_delta = max_abs_delta(work_units)
                    node_delta = max_abs_delta(nodes)
                    iter_delta = max_abs_delta(lp_iters)

                    if work_delta > args.value_tol:
                        if args.strict_metrics:
                            stable = False
                        notes = "work_units_delta" if not notes else notes + "+work_units_delta"
                    if node_delta > args.value_tol:
                        if args.strict_metrics:
                            stable = False
                        notes = "nodes_delta" if not notes else notes + "+nodes_delta"
                    if iter_delta > args.value_tol:
                        if args.strict_metrics:
                            stable = False
                        notes = "lp_iterations_delta" if not notes else notes + "+lp_iterations_delta"

        if not stable:
            unstable_count += 1

        obj_values = [r.objective for r in rows if r.objective is not None]
        work_values = [r.work_units for r in rows if r.work_units is not None]
        node_values = [r.nodes for r in rows if r.nodes is not None]
        it_values = [r.lp_iterations for r in rows if r.lp_iterations is not None]

        summary_rows.append(
            {
                "instance": instance_name,
                "profile": profile_name,
                "threads": str(threads),
                "runs": str(len(rows)),
                "stable": "yes" if stable else "no",
                "status": status_value,
                "objective": "" if not obj_values else f"{obj_values[0]:.12g}",
                "nodes": "" if not node_values else f"{node_values[0]:.12g}",
                "lp_iterations": "" if not it_values else f"{it_values[0]:.12g}",
                "work_units": "" if not work_values else f"{work_values[0]:.12g}",
                "max_abs_obj_delta": "" if not obj_values else f"{max_abs_delta(obj_values):.12g}",
                "max_abs_work_delta": "" if not work_values else f"{max_abs_delta(work_values):.12g}",
                "notes": notes,
            }
        )

    summary_csv = out_dir / "determinism_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "instance",
            "profile",
            "threads",
            "runs",
            "stable",
            "status",
            "objective",
            "nodes",
            "lp_iterations",
            "work_units",
            "max_abs_obj_delta",
            "max_abs_work_delta",
            "notes",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    summary_md = out_dir / "determinism_summary.md"
    lines = [
        "# Determinism Suite Summary",
        "",
        f"Runs per profile: {args.runs}",
        f"Tolerance: {args.value_tol:g}",
        "",
        f"Overall result: {'PASS' if unstable_count == 0 else 'FAIL'}",
        "",
        "| instance | profile | threads | stable | status | max_abs_obj_delta | max_abs_work_delta | notes |",
        "|---|---|---:|---|---|---:|---:|---|",
    ]
    for row in summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["instance"],
                    row["profile"],
                    row["threads"],
                    row["stable"],
                    row["status"],
                    row["max_abs_obj_delta"] or "-",
                    row["max_abs_work_delta"] or "-",
                    row["notes"] or "-",
                ]
            )
            + " |"
        )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote determinism detail CSV: {detail_csv}")
    print(f"Wrote determinism summary CSV: {summary_csv}")
    print(f"Wrote determinism summary Markdown: {summary_md}")

    if unstable_count:
        print(f"Determinism suite FAILED: {unstable_count} unstable profile/instance entries")
        return 1

    print("Determinism suite PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
