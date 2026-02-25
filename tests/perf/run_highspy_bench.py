#!/usr/bin/env python3
"""Run HiGHS (via highspy) benchmarks and emit median CSV results.

Examples:
  LP:
    python3 tests/perf/run_highspy_bench.py \
      --mode lp \
      --instances-dir tests/data/netlib \
      --output /tmp/highspy_lp.csv \
      --repeats 3 \
      --threads 1

  MIP:
    python3 tests/perf/run_highspy_bench.py \
      --mode mip \
      --instances-dir tests/data/miplib \
      --instances p0201,pk1,gt2 \
      --output /tmp/highspy_mip.csv \
      --repeats 1 \
      --threads 1 \
      --time-limit 30
"""

from __future__ import annotations

import argparse
import csv
import gzip
import os
import statistics
import tempfile
import time
from pathlib import Path
from typing import Iterable

import highspy


def instance_name(path: Path) -> str:
    name = path.name
    if name.endswith(".mps.gz"):
        return name[: -len(".mps.gz")]
    if name.endswith(".mps"):
        return name[: -len(".mps")]
    return path.stem


def median(values: Iterable[float]) -> float:
    return float(statistics.median(list(values)))


def status_string(h: highspy.Highs) -> str:
    raw = h.modelStatusToString(h.getModelStatus())
    return raw.strip().lower().replace(" ", "_")


def materialize_model(path: Path) -> tuple[Path, Path | None]:
    if path.suffix != ".gz":
        return path, None
    with gzip.open(path, "rb") as src:
        data = src.read()
    tmp = tempfile.NamedTemporaryFile(suffix=".mps", delete=False)
    try:
        tmp.write(data)
        tmp.flush()
    finally:
        tmp.close()
    tmp_path = Path(tmp.name)
    return tmp_path, tmp_path


def configure_highs(
    h: highspy.Highs,
    mode: str,
    threads: int,
    time_limit: float,
    mip_rel_gap: float,
    presolve: str,
    solver: str,
    simplex_strategy: int | None,
) -> None:
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", threads)
    if time_limit > 0:
        h.setOptionValue("time_limit", time_limit)
    if mip_rel_gap >= 0:
        h.setOptionValue("mip_rel_gap", mip_rel_gap)
    h.setOptionValue("presolve", presolve)
    h.setOptionValue("solver", solver)
    if mode == "lp" and simplex_strategy is not None:
        h.setOptionValue("simplex_strategy", simplex_strategy)


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
    return parser.parse_args()


def collect_instances(
    instances_dir: Path,
    instance_filter: str,
    max_instances: int,
) -> list[Path]:
    all_instances = sorted(instances_dir.glob("*.mps.gz")) + sorted(
        instances_dir.glob("*.mps")
    )
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

    with args.output.open("w", encoding="utf-8", newline="") as out_file:
        writer = csv.writer(out_file)
        writer.writerow(header)

        for model_path in instances:
            name = instance_name(model_path)
            run_path, cleanup_path = materialize_model(model_path)
            times: list[float] = []
            simplex_iters: list[float] = []
            nodes: list[float] = []
            objectives: list[float] = []
            statuses: list[str] = []
            failed = False

            try:
                for _ in range(repeats):
                    h = highspy.Highs()
                    configure_highs(
                        h,
                        mode=args.mode,
                        threads=args.threads,
                        time_limit=time_limit,
                        mip_rel_gap=args.mip_rel_gap,
                        presolve=args.presolve,
                        solver=solver,
                        simplex_strategy=args.simplex_strategy,
                    )
                    read_status = h.readModel(str(run_path))
                    if read_status != highspy.HighsStatus.kOk:
                        failed = True
                        break

                    t0 = time.perf_counter()
                    run_status = h.run()
                    elapsed = time.perf_counter() - t0
                    if run_status != highspy.HighsStatus.kOk:
                        failed = True
                        break

                    info = h.getInfo()
                    times.append(elapsed)
                    simplex_iters.append(float(info.simplex_iteration_count))
                    if args.mode == "mip":
                        nodes.append(float(info.mip_node_count))
                    objectives.append(float(info.objective_function_value))
                    statuses.append(status_string(h))
            finally:
                if cleanup_path is not None:
                    try:
                        os.unlink(cleanup_path)
                    except OSError:
                        pass

            if failed or not times:
                if args.mode == "lp":
                    writer.writerow([name, "", "", "solve_error", ""])
                else:
                    writer.writerow([name, "", "", "", "solve_error", ""])
                continue

            status = statuses[0] if len(set(statuses)) == 1 else "mixed_status"
            med_time = median(times)
            med_simplex = median(simplex_iters)
            med_obj = median(objectives)

            if args.mode == "lp":
                writer.writerow(
                    [
                        name,
                        f"{med_time:.6f}",
                        f"{med_simplex:.0f}",
                        status,
                        f"{med_obj:.12g}",
                    ]
                )
            else:
                med_nodes = median(nodes)
                writer.writerow(
                    [
                        name,
                        f"{med_time:.6f}",
                        f"{med_simplex:.0f}",
                        f"{med_nodes:.0f}",
                        status,
                        f"{med_obj:.12g}",
                    ]
                )

    print(f"Wrote benchmark CSV: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
