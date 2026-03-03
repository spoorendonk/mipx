#!/usr/bin/env python3
"""Compare mipx vs HiGHS with presolve on/off and emit detailed CSV + summary.

This script is designed as a correctness-first gate for presolve work:
- status/objective agreement (when both solve to optimality)
- runtime/iteration impact from presolve on/off
- reduction-intensity diagnostics (rows/cols/bounds)
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import shutil
import statistics
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


def instance_name(path: Path) -> str:
    name = path.name
    if name.endswith(".mps.gz"):
        return name[: -len(".mps.gz")]
    if name.endswith(".mps"):
        return name[: -len(".mps")]
    return path.stem


def parse_csv_tokens(raw: str) -> list[str]:
    return [tok.strip() for tok in raw.split(",") if tok.strip()]


def normalize_status(raw: str) -> str:
    status = re.sub(r"[^a-z0-9]+", "_", raw.strip().lower()).strip("_")
    return {
        "time_limit_reached": "time_limit",
        "iteration_limit_reached": "iteration_limit",
        "node_limit_reached": "node_limit",
        "solve_error": "error",
    }.get(status, status or "unknown")


def parse_first_float(pattern: str, text: str) -> float | None:
    m = re.search(pattern, text, re.MULTILINE)
    if not m:
        return None
    try:
        value = float(m.group(1))
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def parse_mipx_status(text: str) -> str:
    matches = re.findall(r"^Status:\s*(.+?)\s*$", text, re.MULTILINE)
    if not matches:
        return "unknown"
    return normalize_status(matches[-1])


def parse_highs_status(text: str) -> str:
    m = re.search(r"^\s*Model status\s*:\s*(.+?)\s*$", text, re.MULTILINE)
    if m:
        return normalize_status(m.group(1))
    matches = re.findall(r"^\s*Status\s+(.+?)\s*$", text, re.MULTILINE)
    if matches:
        return normalize_status(matches[-1])
    if re.search(r"\binfeasible\b", text, re.IGNORECASE):
        return "infeasible"
    if re.search(r"\bunbounded\b", text, re.IGNORECASE):
        return "unbounded"
    return "unknown"


@dataclass
class SolveResult:
    status: str
    objective: float | None
    time_seconds: float | None
    iterations: float | None
    nodes: float | None
    work_units: float | None
    presolve_rows_removed: int | None
    presolve_cols_removed: int | None
    presolve_bounds_tightened: int | None
    presolve_rounds: int | None
    return_code: int


def run_command(cmd: list[str], timeout_s: float) -> tuple[int, str, float, bool]:
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        elapsed = time.perf_counter() - t0
        out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        return proc.returncode, out, elapsed, False
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - t0
        out = ((exc.stdout or "") if isinstance(exc.stdout, str) else "")
        if isinstance(exc.stderr, str) and exc.stderr:
            out += "\n" + exc.stderr
        return 124, out, elapsed, True


def parse_mipx_presolve_stats(text: str) -> tuple[int | None, int | None, int | None, int | None]:
    # Example:
    # Presolve: 73 vars removed, 158 rows removed, 657 bounds tightened, 4 rounds (3 changed), ...
    m = re.search(
        r"Presolve:\s+(\d+)\s+vars removed,\s+(\d+)\s+rows removed,\s+"
        r"(\d+)\s+bounds tightened,\s+(\d+)\s+rounds",
        text,
    )
    if not m:
        return None, None, None, None
    return int(m.group(2)), int(m.group(1)), int(m.group(3)), int(m.group(4))


def parse_highs_presolve_stats(text: str) -> tuple[int | None, int | None]:
    # Example:
    # Presolve reductions: rows 107(-26); columns 177(-24); nonzeros 1449(-474)
    m = re.search(
        r"Presolve reductions:\s+rows\s+\d+\((-?\d+)\);\s+columns\s+\d+\((-?\d+)\);",
        text,
    )
    if not m:
        return None, None
    # HiGHS prints negative deltas in parentheses for removed counts.
    rows_removed = abs(int(m.group(1)))
    cols_removed = abs(int(m.group(2)))
    return rows_removed, cols_removed


def solve_mipx(
    binary: str,
    model: Path,
    mode: str,
    presolve_on: bool,
    threads: int,
    time_limit: float,
    gap_tol: float | None,
    extra_args: list[str],
) -> SolveResult:
    cmd = [
        binary,
        str(model),
        "--threads",
        str(max(1, threads)),
        "--time-limit",
        f"{time_limit:g}",
        "--presolve" if presolve_on else "--no-presolve",
    ]
    if mode == "mip" and gap_tol is not None:
        cmd.extend(["--gap-tol", f"{max(0.0, gap_tol):g}"])
    if mode == "lp":
        cmd.append("--dual")
    cmd.extend(extra_args)

    timeout_s = max(5.0, time_limit * 1.5) if time_limit > 0 else 300.0
    rc, out, elapsed, timed_out = run_command(cmd, timeout_s=timeout_s)
    if timed_out:
        return SolveResult(
            status="time_limit",
            objective=None,
            time_seconds=elapsed,
            iterations=None,
            nodes=None,
            work_units=None,
            presolve_rows_removed=None,
            presolve_cols_removed=None,
            presolve_bounds_tightened=None,
            presolve_rounds=None,
            return_code=rc,
        )

    objective = parse_first_float(r"^Objective:\s*([\-+0-9.eE]+)\s*$", out)
    time_seconds = parse_first_float(r"^Time:\s*([\-+0-9.eE]+)s\s*$", out)
    iterations = parse_first_float(
        r"^(?:Iterations|LP iterations):\s*([\-+0-9.eE]+)\s*$", out
    )
    nodes = parse_first_float(r"^Nodes:\s*([\-+0-9.eE]+)\s*$", out)
    work_units = parse_first_float(r"^Work units:\s*([\-+0-9.eE]+)\s*$", out)
    status = parse_mipx_status(out)
    if rc != 0 and status == "unknown":
        status = "error"

    rows_removed, cols_removed, bounds_tightened, rounds = parse_mipx_presolve_stats(out)

    return SolveResult(
        status=status,
        objective=objective,
        time_seconds=time_seconds if time_seconds is not None else elapsed,
        iterations=iterations,
        nodes=nodes,
        work_units=work_units,
        presolve_rows_removed=rows_removed,
        presolve_cols_removed=cols_removed,
        presolve_bounds_tightened=bounds_tightened,
        presolve_rounds=rounds,
        return_code=rc,
    )


def write_highs_options_file(threads: int) -> str:
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".options", delete=False, encoding="utf-8")
    try:
        tmp.write(f"threads = {max(1, threads)}\n")
    finally:
        tmp.close()
    return tmp.name


def solve_highs(
    highs_binary: str,
    model: Path,
    mode: str,
    presolve_on: bool,
    threads: int,
    time_limit: float,
    extra_args: list[str],
) -> SolveResult:
    options_file = write_highs_options_file(threads)
    try:
        solver = "simplex" if mode == "lp" else "choose"
        cmd = [
            highs_binary,
            str(model),
            "--options_file",
            options_file,
            "--solver",
            solver,
            "--presolve",
            "choose" if presolve_on else "off",
            "--parallel",
            "on" if threads > 1 else "off",
        ]
        if time_limit > 0:
            cmd.extend(["--time_limit", f"{time_limit:g}"])
        cmd.extend(extra_args)

        timeout_s = max(5.0, time_limit * 1.5) if time_limit > 0 else 300.0
        rc, out, elapsed, timed_out = run_command(cmd, timeout_s=timeout_s)
        if timed_out:
            return SolveResult(
                status="time_limit",
                objective=None,
                time_seconds=elapsed,
                iterations=None,
                nodes=None,
                work_units=None,
                presolve_rows_removed=None,
                presolve_cols_removed=None,
                presolve_bounds_tightened=None,
                presolve_rounds=None,
                return_code=rc,
            )

        status = parse_highs_status(out)
        if mode == "mip":
            objective = parse_first_float(r"^\s*Primal bound\s+([\-+0-9.eE]+)\s*$", out)
            iterations = parse_first_float(r"^\s*LP iterations\s+([\-+0-9.eE]+)\s*$", out)
            nodes = parse_first_float(r"^\s*Nodes\s+([\-+0-9.eE]+)\s*$", out)
        else:
            objective = parse_first_float(
                r"^\s*Objective value\s*:\s*([\-+0-9.eE]+)\s*$", out
            )
            iterations = parse_first_float(
                r"^\s*Simplex\s+iterations:\s*([\-+0-9.eE]+)\s*$", out
            )
            nodes = None

        time_seconds = parse_first_float(r"^\s*HiGHS run time\s*:\s*([\-+0-9.eE]+)\s*$", out)
        if time_seconds is None:
            time_seconds = parse_first_float(r"^\s*Timing\s+([\-+0-9.eE]+)\s*$", out)

        rows_removed, cols_removed = parse_highs_presolve_stats(out)
        if rc != 0 and status == "unknown":
            status = "error"

        return SolveResult(
            status=status,
            objective=objective,
            time_seconds=time_seconds if time_seconds is not None else elapsed,
            iterations=iterations,
            nodes=nodes,
            work_units=None,
            presolve_rows_removed=rows_removed,
            presolve_cols_removed=cols_removed,
            presolve_bounds_tightened=None,
            presolve_rounds=None,
            return_code=rc,
        )
    finally:
        try:
            os.unlink(options_file)
        except OSError:
            pass


def collect_instances(instances_dir: Path, instance_filter: str, max_instances: int) -> list[Path]:
    all_instances = sorted(instances_dir.glob("*.mps.gz")) + sorted(instances_dir.glob("*.mps"))
    if not all_instances:
        raise SystemExit(f"No .mps/.mps.gz instances found in {instances_dir}")
    if instance_filter:
        wanted = set(parse_csv_tokens(instance_filter))
        selected = [p for p in all_instances if instance_name(p) in wanted]
    else:
        selected = all_instances
    if max_instances > 0:
        selected = selected[:max_instances]
    if not selected:
        raise SystemExit("No instances selected")
    return selected


def median_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def format_float(v: float | None) -> str:
    if v is None or not math.isfinite(v):
        return ""
    return f"{v:.6f}"


def relative_error(a: float, b: float) -> float:
    return abs(a - b) / max(1.0, abs(b))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("lp", "mip"), required=True)
    parser.add_argument("--instances-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", default="")
    parser.add_argument("--instances", default="")
    parser.add_argument("--max-instances", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--time-limit", type=float, default=30.0)
    parser.add_argument("--mipx-gap-tol", type=float, default=None)
    parser.add_argument("--objective-rel-tol", type=float, default=1e-7)
    parser.add_argument("--mipx-binary", default="./build/mipx-solve")
    parser.add_argument("--highs-binary", default="")
    parser.add_argument("--mipx-arg", action="append", default=[])
    parser.add_argument("--highs-arg", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    instances_dir = Path(args.instances_dir)
    if not instances_dir.is_dir():
        raise SystemExit(f"instances dir not found: {instances_dir}")
    if args.threads < 1:
        raise SystemExit("--threads must be >= 1")
    if args.time_limit <= 0:
        raise SystemExit("--time-limit must be > 0")
    if args.mipx_gap_tol is not None and args.mipx_gap_tol < 0:
        raise SystemExit("--mipx-gap-tol must be >= 0")

    mipx_binary = Path(args.mipx_binary)
    if not mipx_binary.is_file() or not os.access(mipx_binary, os.X_OK):
        raise SystemExit(f"mipx binary not executable: {mipx_binary}")

    highs_binary = args.highs_binary or os.environ.get("HIGHS_BINARY") or shutil.which("highs")
    if not highs_binary:
        raise SystemExit("HiGHS binary not found. Set --highs-binary or HIGHS_BINARY.")

    instances = collect_instances(instances_dir, args.instances, args.max_instances)
    mipx_gap_tol = (
        max(0.0, args.mipx_gap_tol)
        if args.mipx_gap_tol is not None
        else (0.0 if args.mode == "mip" else None)
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    pair_index: dict[tuple[str, str], dict[str, SolveResult]] = {}

    for model in instances:
        name = instance_name(model)
        for presolve_on in (True, False):
            presolve_label = "on" if presolve_on else "off"
            mipx_res = solve_mipx(
                binary=str(mipx_binary),
                model=model,
                mode=args.mode,
                presolve_on=presolve_on,
                threads=args.threads,
                time_limit=args.time_limit,
                gap_tol=mipx_gap_tol,
                extra_args=args.mipx_arg,
            )
            highs_res = solve_highs(
                highs_binary=highs_binary,
                model=model,
                mode=args.mode,
                presolve_on=presolve_on,
                threads=args.threads,
                time_limit=args.time_limit,
                extra_args=args.highs_arg,
            )

            pair_index[(name, presolve_label)] = {
                "mipx": mipx_res,
                "highs": highs_res,
            }

            for solver_name, res in (("mipx", mipx_res), ("highs", highs_res)):
                rows.append(
                    {
                        "instance": name,
                        "mode": args.mode,
                        "solver": solver_name,
                        "presolve": presolve_label,
                        "status": res.status,
                        "objective": format_float(res.objective),
                        "time_seconds": format_float(res.time_seconds),
                        "iterations": format_float(res.iterations),
                        "nodes": format_float(res.nodes),
                        "work_units": format_float(res.work_units),
                        "presolve_rows_removed": (
                            str(res.presolve_rows_removed)
                            if res.presolve_rows_removed is not None
                            else ""
                        ),
                        "presolve_cols_removed": (
                            str(res.presolve_cols_removed)
                            if res.presolve_cols_removed is not None
                            else ""
                        ),
                        "presolve_bounds_tightened": (
                            str(res.presolve_bounds_tightened)
                            if res.presolve_bounds_tightened is not None
                            else ""
                        ),
                        "presolve_rounds": (
                            str(res.presolve_rounds)
                            if res.presolve_rounds is not None
                            else ""
                        ),
                        "return_code": str(res.return_code),
                    }
                )

    fieldnames = [
        "instance",
        "mode",
        "solver",
        "presolve",
        "status",
        "objective",
        "time_seconds",
        "iterations",
        "nodes",
        "work_units",
        "presolve_rows_removed",
        "presolve_cols_removed",
        "presolve_bounds_tightened",
        "presolve_rounds",
        "return_code",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    status_mismatches = 0
    objective_mismatches = 0
    comparable_status = 0
    comparable_optimal = 0
    for key, vals in pair_index.items():
        _ = key
        mr = vals["mipx"]
        hr = vals["highs"]
        if mr.status not in {"error"} and hr.status not in {"error"}:
            comparable_status += 1
            if mr.status != hr.status:
                status_mismatches += 1
        if mr.status == "optimal" and hr.status == "optimal":
            comparable_optimal += 1
            if mr.objective is None or hr.objective is None:
                objective_mismatches += 1
            elif relative_error(mr.objective, hr.objective) > args.objective_rel_tol:
                objective_mismatches += 1

    def collect_times(solver: str, presolve: str) -> list[float]:
        out: list[float] = []
        for r in rows:
            if r["solver"] != solver or r["presolve"] != presolve:
                continue
            if not r["time_seconds"]:
                continue
            out.append(float(r["time_seconds"]))
        return out

    mipx_on = median_or_none(collect_times("mipx", "on"))
    mipx_off = median_or_none(collect_times("mipx", "off"))
    highs_on = median_or_none(collect_times("highs", "on"))
    highs_off = median_or_none(collect_times("highs", "off"))

    summary_lines = [
        f"mode={args.mode}",
        f"instances={len(instances)}",
        f"comparables_status={comparable_status}",
        f"status_mismatches={status_mismatches}",
        f"comparables_optimal={comparable_optimal}",
        f"objective_mismatches={objective_mismatches}",
        f"median_time_mipx_on={format_float(mipx_on)}",
        f"median_time_mipx_off={format_float(mipx_off)}",
        f"median_time_highs_on={format_float(highs_on)}",
        f"median_time_highs_off={format_float(highs_off)}",
    ]

    if args.summary:
        summary_path = Path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
        print(f"Wrote summary: {summary_path}")
    else:
        print("\n".join(summary_lines))

    print(f"Wrote detailed CSV: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
