#!/usr/bin/env python3
"""Run HiGHS CLI benchmarks and emit median CSV results.

Examples:
  LP:
    python3 tests/perf/run_highs_bench.py \
      --mode lp \
      --instances-dir tests/data/netlib \
      --output /tmp/highs_lp.csv \
      --repeats 3 \
      --threads 1

  MIP:
    python3 tests/perf/run_highs_bench.py \
      --mode mip \
      --instances-dir tests/data/miplib \
      --instances p0201,pk1,gt2 \
      --output /tmp/highs_mip.csv \
      --repeats 1 \
      --threads 1 \
      --time-limit 30
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
from typing import Iterable


def instance_name(path: Path) -> str:
    name = path.name
    if name.endswith(".mps.gz"):
        return name[: -len(".mps.gz")]
    if name.endswith(".mps"):
        return name[: -len(".mps")]
    return path.stem


def median(values: Iterable[float]) -> float:
    return float(statistics.median(list(values)))


def normalize_status(raw: str) -> str:
    status = re.sub(r"[^a-z0-9]+", "_", raw.strip().lower()).strip("_")
    status = {
        "time_limit_reached": "time_limit",
        "iteration_limit_reached": "iteration_limit",
    }.get(status, status)
    return status or "unknown"


def parse_first_float(pattern: str, text: str) -> float | None:
    m = re.search(pattern, text, re.MULTILINE)
    if not m:
        return None
    try:
        val = float(m.group(1))
    except ValueError:
        return None
    return val if math.isfinite(val) else None


def parse_status(text: str) -> str:
    m = re.search(r"^\s*Model status\s*:\s*(.+?)\s*$", text, re.MULTILINE)
    if m:
        return normalize_status(m.group(1))
    matches = re.findall(r"^\s*Status\s+(.+?)\s*$", text, re.MULTILINE)
    if matches:
        return normalize_status(matches[-1])
    if "Optimal solution found" in text:
        return "optimal"
    if re.search(r"\binfeasible\b", text, re.IGNORECASE):
        return "infeasible"
    if re.search(r"\bunbounded\b", text, re.IGNORECASE):
        return "unbounded"
    return "unknown"


def parse_runtime(text: str) -> float | None:
    runtime = parse_first_float(r"^\s*HiGHS run time\s*:\s*([\-+0-9.eE]+)\s*$", text)
    if runtime is not None:
        return runtime
    return parse_first_float(r"^\s*Timing\s+([\-+0-9.eE]+)\s*$", text)


def parse_iterations(mode: str, text: str) -> float | None:
    if mode == "mip":
        return parse_first_float(r"^\s*LP iterations\s+([\-+0-9.eE]+)\s*$", text)
    for pat in (
        r"^\s*Simplex\s+iterations:\s*([\-+0-9.eE]+)\s*$",
        r"^\s*IPM\s+iterations:\s*([\-+0-9.eE]+)\s*$",
        r"^\s*PDLP\s+iterations:\s*([\-+0-9.eE]+)\s*$",
    ):
        val = parse_first_float(pat, text)
        if val is not None:
            return val
    return None


def parse_nodes(text: str) -> float | None:
    return parse_first_float(r"^\s*Nodes\s+([\-+0-9.eE]+)\s*$", text)


def parse_objective(mode: str, text: str) -> float | None:
    if mode == "mip":
        val = parse_first_float(r"^\s*Primal bound\s+([\-+0-9.eE]+)\s*$", text)
        if val is not None:
            return val
    return parse_first_float(r"^\s*Objective value\s*:\s*([\-+0-9.eE]+)\s*$", text)


def resolve_highs_binary(explicit: str) -> str | None:
    if explicit:
        return explicit
    env = os.environ.get("HIGHS_BINARY")
    if env:
        return env
    return shutil.which("highs")


def make_options_file(mode: str, threads: int, mip_rel_gap: float, simplex_strategy: int | None) -> str:
    lines = [f"threads = {threads}"]
    if mode == "mip" and mip_rel_gap >= 0:
        lines.append(f"mip_rel_gap = {mip_rel_gap:g}")
    if mode == "lp" and simplex_strategy is not None:
        lines.append(f"simplex_strategy = {simplex_strategy}")

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".options", delete=False, encoding="utf-8")
    try:
        tmp.write("\n".join(lines) + "\n")
    finally:
        tmp.close()
    return tmp.name


@dataclass
class SolveResult:
    status: str
    time_seconds: float
    simplex_iterations: float | None
    nodes: float | None
    objective: float | None
    ok: bool


def run_highs_once(
    highs_binary: str,
    model_path: Path,
    mode: str,
    solver: str,
    presolve: str,
    time_limit: float,
    threads: int,
    options_file: str,
) -> SolveResult:
    cmd = [
        highs_binary,
        str(model_path),
        "--options_file",
        options_file,
        "--solver",
        solver,
        "--presolve",
        presolve,
        "--parallel",
        "off" if threads == 1 else "on",
    ]
    if time_limit > 0:
        cmd.extend(["--time_limit", f"{time_limit:g}"])

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")

    status = parse_status(output)
    runtime = parse_runtime(output)
    if runtime is None:
        runtime = elapsed

    return SolveResult(
        status=status,
        time_seconds=runtime,
        simplex_iterations=parse_iterations(mode, output),
        nodes=parse_nodes(output) if mode == "mip" else None,
        objective=parse_objective(mode, output),
        ok=proc.returncode == 0,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("lp", "mip"), required=True)
    parser.add_argument("--instances-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--time-limit", type=float, default=-1.0)
    parser.add_argument("--mip-rel-gap", type=float, default=1e-4)
    parser.add_argument("--presolve", default="choose")
    parser.add_argument("--solver", default="")
    parser.add_argument("--simplex-strategy", type=int, default=1)
    parser.add_argument("--max-instances", type=int, default=0)
    parser.add_argument("--instances", default="")
    parser.add_argument("--highs-binary", default="")
    return parser.parse_args()


def collect_instances(
    instances_dir: Path,
    instance_filter: str,
    max_instances: int,
) -> list[Path]:
    all_instances = sorted(instances_dir.glob("*.mps.gz")) + sorted(instances_dir.glob("*.mps"))
    if not all_instances:
        raise ValueError(f"no .mps/.mps.gz instances found in {instances_dir}")

    if instance_filter:
        names = {name.strip() for name in instance_filter.split(",") if name.strip()}
        selected = [p for p in all_instances if instance_name(p) in names]
    else:
        selected = all_instances

    if max_instances > 0:
        selected = selected[:max_instances]
    if not selected:
        raise ValueError("no instances selected")
    return selected


def main() -> int:
    args = parse_args()
    if not args.instances_dir.is_dir():
        raise SystemExit(f"instances dir not found: {args.instances_dir}")
    if args.threads <= 0:
        raise SystemExit("--threads must be >= 1")

    highs_binary = resolve_highs_binary(args.highs_binary)
    if not highs_binary:
        raise SystemExit("HiGHS binary not found. Set --highs-binary or HIGHS_BINARY.")
    if not shutil.which(highs_binary) and not Path(highs_binary).is_file():
        raise SystemExit(f"HiGHS binary not found: {highs_binary}")

    repeats = args.repeats
    if repeats <= 0:
        repeats = 3 if args.mode == "lp" else 1

    solver = args.solver
    if not solver:
        solver = "simplex" if args.mode == "lp" else "choose"

    time_limit = args.time_limit
    if time_limit <= 0:
        time_limit = 60.0 if args.mode == "lp" else 30.0

    instances = collect_instances(args.instances_dir, args.instances, args.max_instances)

    options_file = make_options_file(
        mode=args.mode,
        threads=args.threads,
        mip_rel_gap=args.mip_rel_gap,
        simplex_strategy=args.simplex_strategy,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "lp":
        header = ["instance", "time_seconds", "simplex_iterations", "status", "objective"]
    else:
        header = [
            "instance",
            "time_seconds",
            "simplex_iterations",
            "nodes",
            "status",
            "objective",
        ]

    try:
        with args.output.open("w", encoding="utf-8", newline="") as out_file:
            writer = csv.writer(out_file)
            writer.writerow(header)

            for model_path in instances:
                name = instance_name(model_path)
                times: list[float] = []
                simplex_iters: list[float] = []
                nodes: list[float] = []
                objectives: list[float] = []
                statuses: list[str] = []
                failed = False

                for _ in range(repeats):
                    res = run_highs_once(
                        highs_binary=highs_binary,
                        model_path=model_path,
                        mode=args.mode,
                        solver=solver,
                        presolve=args.presolve,
                        time_limit=time_limit,
                        threads=args.threads,
                        options_file=options_file,
                    )
                    if not res.ok:
                        failed = True
                        break

                    times.append(res.time_seconds)
                    if res.simplex_iterations is not None:
                        simplex_iters.append(float(res.simplex_iterations))
                    if args.mode == "mip" and res.nodes is not None:
                        nodes.append(float(res.nodes))
                    if res.objective is not None:
                        objectives.append(float(res.objective))
                    statuses.append(res.status)

                if failed or not times:
                    if args.mode == "lp":
                        writer.writerow([name, "", "", "solve_error", ""])
                    else:
                        writer.writerow([name, "", "", "", "solve_error", ""])
                    continue

                status = statuses[0] if len(set(statuses)) == 1 else "mixed_status"
                med_time = median(times)
                med_simplex = median(simplex_iters) if simplex_iters else float("nan")
                med_obj = median(objectives) if objectives else float("nan")

                if args.mode == "lp":
                    writer.writerow(
                        [
                            name,
                            f"{med_time:.6f}",
                            f"{med_simplex:.0f}" if simplex_iters else "",
                            status,
                            f"{med_obj:.12g}" if objectives else "",
                        ]
                    )
                else:
                    med_nodes = median(nodes) if nodes else float("nan")
                    writer.writerow(
                        [
                            name,
                            f"{med_time:.6f}",
                            f"{med_simplex:.0f}" if simplex_iters else "",
                            f"{med_nodes:.0f}" if nodes else "",
                            status,
                            f"{med_obj:.12g}" if objectives else "",
                        ]
                    )
    finally:
        try:
            os.unlink(options_file)
        except OSError:
            pass

    print(f"Wrote benchmark CSV: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
