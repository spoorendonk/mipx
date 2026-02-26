#!/usr/bin/env python3
"""Compare LP PDLP performance across mipx, HiGHS, and cuOpt.

Example:
  python3 tests/perf/run_pdlp_lp_compare.py \
    --mipx-binary ./build/mipx-solve \
    --instances-dir tests/data/netlib \
    --output /tmp/pdlp_lp_compare.csv \
    --repeats 3 \
    --threads 1 \
    --time-limit 60
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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


def instance_name(path: Path) -> str:
    name = path.name
    if name.endswith(".mps.gz"):
        return name[: -len(".mps.gz")]
    if name.endswith(".mps"):
        return name[: -len(".mps")]
    return path.stem


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def geomean(values: list[float]) -> float:
    positives = [v for v in values if v > 0.0]
    if not positives:
        return float("nan")
    return float(math.exp(sum(math.log(v) for v in positives) / len(positives)))


def parse_first_float(pattern: str, text: str) -> float | None:
    m = re.search(pattern, text, re.MULTILINE)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def parse_last_float(pattern: str, text: str) -> float | None:
    ms = re.findall(pattern, text, re.MULTILINE)
    if not ms:
        return None
    try:
        return float(ms[-1])
    except ValueError:
        return None


def parse_status(text: str) -> str:
    m = re.search(r"^Status:\s*(.+?)(?=\s{2,}\w+:|$)", text, re.MULTILINE)
    if m:
        return m.group(1).strip().lower().replace(" ", "_")
    if "Optimal solution found" in text:
        return "optimal"
    if re.search(r"\binfeasible\b", text, re.IGNORECASE):
        return "infeasible"
    if re.search(r"\bunbounded\b", text, re.IGNORECASE):
        return "unbounded"
    return "unknown"


def normalize_status(raw: str) -> str:
    status = re.sub(r"[^a-z0-9]+", "_", raw.strip().lower()).strip("_")
    status = {
        "time_limit_reached": "time_limit",
        "iteration_limit_reached": "iteration_limit",
    }.get(status, status)
    return status or "unknown"


def parse_highs_status(text: str) -> str:
    m = re.search(r"^\s*Model status\s*:\s*(.+?)\s*$", text, re.MULTILINE)
    if m:
        return normalize_status(m.group(1))
    matches = re.findall(r"^\s*Status\s+(.+?)\s*$", text, re.MULTILINE)
    if matches:
        return normalize_status(matches[-1])
    return "unknown"


def parse_highs_runtime(text: str) -> float | None:
    runtime = parse_first_float(r"^\s*HiGHS run time\s*:\s*([\-+0-9.eE]+)\s*$", text)
    if runtime is not None:
        return runtime
    return parse_first_float(r"^\s*Timing\s+([\-+0-9.eE]+)\s*$", text)


def locate_cuopt_cli(explicit: str) -> str | None:
    if explicit:
        return explicit
    env = os.environ.get("CUOPT_CLI")
    if env:
        return env
    cli = shutil.which("cuopt_cli")
    if cli:
        return cli
    try:
        import libcuopt._cli_wrapper as cli_wrapper

        return str(Path(cli_wrapper.__file__).resolve().parent / "bin" / "cuopt_cli")
    except Exception:
        return None


def locate_highs_binary(explicit: str) -> str | None:
    if explicit:
        return explicit
    env = os.environ.get("HIGHS_BINARY")
    if env:
        return env
    return shutil.which("highs")


def write_highs_options_file(threads: int, relax_integrality: bool) -> str:
    import tempfile

    lines = [f"threads = {threads}"]
    if relax_integrality:
        lines.append("solve_relaxation = true")

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".options", delete=False, encoding="utf-8")
    try:
        tmp.write("\n".join(lines) + "\n")
    finally:
        tmp.close()
    return tmp.name


@dataclass
class SolverResult:
    status: str
    time_seconds: float
    iterations: float | None = None
    objective: float | None = None
    work_units: float | None = None
    error: str | None = None


def run_cmd(cmd: list[str]) -> tuple[int, str, float]:
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    return proc.returncode, output, elapsed


def run_mipx_pdlp(
    binary: str,
    model: Path,
    use_gpu: bool,
    disable_presolve: bool,
    force_gpu: bool,
    time_limit: float,
    relax_integrality: bool,
) -> SolverResult:
    cmd = [binary, str(model), "--pdlp", "--quiet"]
    if use_gpu:
        cmd.append("--gpu")
        if force_gpu:
            cmd.extend(["--gpu-min-rows", "0", "--gpu-min-nnz", "0"])
    else:
        cmd.append("--no-gpu")
    if disable_presolve:
        cmd.append("--no-presolve")
    if relax_integrality:
        cmd.append("--relax-integrality")
    if time_limit > 0:
        cmd.extend(["--time-limit", f"{time_limit:g}"])

    code, out, elapsed = run_cmd(cmd)
    if code != 0:
        return SolverResult(status="solve_error", time_seconds=elapsed, error=out.strip())

    status = parse_status(out)
    objective = parse_first_float(r"^Objective:\s*([\-+0-9.eE]+)\s*$", out)
    iterations = parse_first_float(r"^Iterations:\s*([\-+0-9.eE]+)\s*$", out)
    work = parse_first_float(r"^Work units:\s*([\-+0-9.eE]+)\s*$", out)
    solve_time = parse_first_float(r"^Time:\s*([\-+0-9.eE]+)s\s*$", out)

    return SolverResult(
        status=status,
        time_seconds=solve_time if solve_time is not None else elapsed,
        iterations=iterations,
        objective=objective,
        work_units=work,
    )


def run_highs(
    highs_binary: str,
    model: Path,
    threads: int,
    time_limit: float,
    disable_presolve: bool,
    relax_integrality: bool,
    prefer_pdlp: bool,
) -> SolverResult:
    requested_solver = "pdlp" if prefer_pdlp else "ipx"
    options_file = write_highs_options_file(threads, relax_integrality)
    try:
        cmd = [
            highs_binary,
            str(model),
            "--options_file",
            options_file,
            "--solver",
            requested_solver,
            "--presolve",
            "off" if disable_presolve else "choose",
            "--parallel",
            "off" if threads == 1 else "on",
        ]
        if time_limit > 0:
            cmd.extend(["--time_limit", f"{time_limit:g}"])

        code, out, elapsed = run_cmd(cmd)
        solver_note = ""
        if code != 0 and requested_solver == "pdlp":
            fallback = cmd.copy()
            solver_idx = fallback.index("--solver") + 1
            fallback[solver_idx] = "ipx"
            code, out, elapsed = run_cmd(fallback)
            solver_note = "highs_pdlp_unavailable_fallback_ipx"
        if code != 0:
            return SolverResult(status="solve_error", time_seconds=elapsed, error=out.strip())

        status = parse_highs_status(out)
        iters = parse_first_float(r"^\s*PDLP\s+iterations:\s*([\-+0-9.eE]+)\s*$", out)
        if iters is None:
            iters = parse_first_float(r"^\s*IPM\s+iterations:\s*([\-+0-9.eE]+)\s*$", out)
        if iters is None:
            iters = parse_first_float(r"^\s*Simplex\s+iterations:\s*([\-+0-9.eE]+)\s*$", out)
        objective = parse_first_float(r"^\s*Objective value\s*:\s*([\-+0-9.eE]+)\s*$", out)
        highs_time = parse_highs_runtime(out)
        return SolverResult(
            status=status,
            time_seconds=highs_time if highs_time is not None else elapsed,
            iterations=iters,
            objective=objective,
            error=solver_note or None,
        )
    finally:
        try:
            os.unlink(options_file)
        except OSError:
            pass


def run_cuopt_pdlp(
    cli_path: str,
    model: Path,
    time_limit: float,
    disable_presolve: bool,
    num_gpus: int,
    relax_integrality: bool,
) -> SolverResult:
    cmd = [
        cli_path,
        str(model),
        "--method",
        "1",  # cuOpt PDLP
        "--log-to-console",
        "true",
        "--num-gpus",
        str(max(1, num_gpus)),
    ]
    if disable_presolve:
        cmd.extend(["--presolve", "0"])
    if relax_integrality:
        cmd.append("--relaxation")
    if time_limit > 0:
        cmd.extend(["--time-limit", f"{time_limit:g}"])

    code, out, elapsed = run_cmd(cmd)
    if code != 0:
        return SolverResult(status="solve_error", time_seconds=elapsed, error=out.strip())

    status = parse_status(out)
    objective = parse_last_float(r"\bObjective[: ]+([\-+0-9.eE]+)\b", out)
    iterations = parse_last_float(r"\bIterations:\s*([0-9]+)\b", out)
    if iterations is None:
        iterations = parse_last_float(
            r"(?:Optimal solution found in|PDLP converged in)\s+([0-9]+)\s+iterations", out
        )
    pdlp_time = parse_last_float(r"PDLP finished in\s+([\-+0-9.eE]+)\s+seconds", out)
    if pdlp_time is None:
        pdlp_time = parse_last_float(r"Solve finished in\s+([\-+0-9.eE]+)\s+seconds", out)
    if pdlp_time is None:
        pdlp_time = parse_last_float(r"\bTime:\s*([\-+0-9.eE]+)s\b", out)

    return SolverResult(
        status=status,
        time_seconds=pdlp_time if pdlp_time is not None else elapsed,
        iterations=iterations,
        objective=objective,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mipx-binary", default="./build/mipx-solve")
    p.add_argument("--instances-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--time-limit", type=float, default=60.0)
    p.add_argument("--instances", default="")
    p.add_argument("--max-instances", type=int, default=0)
    p.add_argument("--disable-presolve", action="store_true")
    p.add_argument("--relax-integrality", action="store_true")
    p.add_argument("--force-mipx-gpu", action="store_true")
    p.add_argument("--no-highs", action="store_true")
    p.add_argument("--highs-binary", default="")
    p.add_argument("--highs-ipx", action="store_true", help="Use HiGHS IPX instead of PDLP.")
    p.add_argument("--no-cuopt", action="store_true")
    p.add_argument("--cuopt-cli", default="")
    p.add_argument("--cuopt-num-gpus", type=int, default=1)
    return p.parse_args()


def collect_instances(instances_dir: Path, filt: str, max_instances: int) -> list[Path]:
    models = sorted(instances_dir.glob("*.mps.gz")) + sorted(instances_dir.glob("*.mps"))
    if not models:
        raise ValueError(f"no .mps/.mps.gz files found in {instances_dir}")
    if filt:
        keep = {x.strip() for x in filt.split(",") if x.strip()}
        models = [m for m in models if instance_name(m) in keep]
    if max_instances > 0:
        models = models[:max_instances]
    if not models:
        raise ValueError("no instances selected")
    return models


def summarize_and_write(
    output: Path,
    per_solver_rows: list[dict[str, str]],
    objective_tol: float = 1e-4,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "instance",
                "solver",
                "time_seconds",
                "iterations",
                "status",
                "objective",
                "work_units",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(per_solver_rows)

    by_solver: dict[str, list[dict[str, str]]] = {}
    by_instance: dict[str, list[dict[str, str]]] = {}
    for row in per_solver_rows:
        by_solver.setdefault(row["solver"], []).append(row)
        by_instance.setdefault(row["instance"], []).append(row)

    print("\nPer-solver summary")
    print("solver,rows,optimal,median_time,geomean_time,median_iterations")
    for solver in sorted(by_solver):
        rows = by_solver[solver]
        times = [float(r["time_seconds"]) for r in rows if r["time_seconds"]]
        opt_rows = [r for r in rows if r["status"] == "optimal"]
        opt_times = [float(r["time_seconds"]) for r in opt_rows if r["time_seconds"]]
        opt_iters = [float(r["iterations"]) for r in opt_rows if r["iterations"]]
        med_time = median(times) if times else float("nan")
        gmean_time = geomean(opt_times) if opt_times else float("nan")
        med_iter = median(opt_iters) if opt_iters else float("nan")
        print(
            f"{solver},{len(rows)},{len(opt_rows)},"
            f"{med_time:.6f},{gmean_time:.6f},{med_iter:.2f}"
        )

    print("\nObjective agreement check (optimal-only)")
    disagreements = 0
    for inst, rows in sorted(by_instance.items()):
        objs = []
        for r in rows:
            if r["status"] == "optimal" and r["objective"]:
                objs.append((r["solver"], float(r["objective"])))
        if len(objs) < 2:
            continue
        vals = [v for _, v in objs]
        span = max(vals) - min(vals)
        if abs(span) > objective_tol:
            disagreements += 1
            details = ", ".join(f"{s}={v:.9g}" for s, v in objs)
            print(f"WARNING {inst}: span={span:.3e} :: {details}")
    if disagreements == 0:
        print("No objective disagreements above tolerance.")


def main() -> int:
    args = parse_args()
    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")
    if args.threads < 1:
        raise SystemExit("--threads must be >= 1")
    if not Path(args.mipx_binary).is_file():
        raise SystemExit(f"mipx binary not found: {args.mipx_binary}")
    if not os.access(args.mipx_binary, os.X_OK):
        raise SystemExit(f"mipx binary not executable: {args.mipx_binary}")

    instances = collect_instances(args.instances_dir, args.instances, args.max_instances)
    highs_binary = None if args.no_highs else locate_highs_binary(args.highs_binary)
    if not args.no_highs and not highs_binary:
        raise SystemExit("HiGHS binary not found. Set --highs-binary or HIGHS_BINARY.")
    if highs_binary and not shutil.which(highs_binary) and not Path(highs_binary).is_file():
        raise SystemExit(f"HiGHS binary not found: {highs_binary}")

    cuopt_cli = None if args.no_cuopt else locate_cuopt_cli(args.cuopt_cli)
    if cuopt_cli and not os.path.isfile(cuopt_cli):
        cuopt_cli = shutil.which(cuopt_cli) or None

    solvers: list[tuple[str, Callable[[Path], SolverResult]]] = []
    solvers.append(
        (
            "mipx_pdlp_cpu",
            lambda p: run_mipx_pdlp(
                args.mipx_binary,
                p,
                use_gpu=False,
                disable_presolve=args.disable_presolve,
                force_gpu=False,
                time_limit=args.time_limit,
                relax_integrality=args.relax_integrality,
            ),
        )
    )
    solvers.append(
        (
            "mipx_pdlp_gpu",
            lambda p: run_mipx_pdlp(
                args.mipx_binary,
                p,
                use_gpu=True,
                disable_presolve=args.disable_presolve,
                force_gpu=args.force_mipx_gpu,
                time_limit=args.time_limit,
                relax_integrality=args.relax_integrality,
            ),
        )
    )
    if not args.no_highs:
        highs_solver = "highs_ipx" if args.highs_ipx else "highs_pdlp"
        solvers.append(
            (
                highs_solver,
                lambda p: run_highs(
                    highs_binary,
                    p,
                    threads=args.threads,
                    time_limit=args.time_limit,
                    disable_presolve=args.disable_presolve,
                    relax_integrality=args.relax_integrality,
                    prefer_pdlp=not args.highs_ipx,
                ),
            )
        )
    if cuopt_cli:
        solvers.append(
            (
                "cuopt_pdlp",
                lambda p: run_cuopt_pdlp(
                    cuopt_cli,
                    p,
                    time_limit=args.time_limit,
                    disable_presolve=args.disable_presolve,
                    num_gpus=args.cuopt_num_gpus,
                    relax_integrality=args.relax_integrality,
                ),
            )
        )

    rows: list[dict[str, str]] = []
    for model in instances:
        name = instance_name(model)
        for solver_name, runner in solvers:
            run_results: list[SolverResult] = []
            for _ in range(args.repeats):
                run_results.append(runner(model))

            statuses = {r.status for r in run_results}
            status = run_results[0].status if len(statuses) == 1 else "mixed_status"
            times = [r.time_seconds for r in run_results if r.time_seconds >= 0.0]
            iters = [r.iterations for r in run_results if r.iterations is not None]
            objs = [r.objective for r in run_results if r.objective is not None]
            works = [r.work_units for r in run_results if r.work_units is not None]
            errors = [r.error for r in run_results if r.error]

            row = {
                "instance": name,
                "solver": solver_name,
                "time_seconds": f"{median(times):.6f}" if times else "",
                "iterations": f"{median([float(x) for x in iters]):.0f}" if iters else "",
                "status": status,
                "objective": f"{median([float(x) for x in objs]):.12g}" if objs else "",
                "work_units": f"{median([float(x) for x in works]):.6f}" if works else "",
                "error": " | ".join(errors[:1]) if errors else "",
            }
            rows.append(row)
            print(
                f"{name:16s} {solver_name:14s} status={status:14s} "
                f"time={row['time_seconds'] or '-':>9s} "
                f"iter={row['iterations'] or '-':>6s}"
            )

    summarize_and_write(args.output, rows)
    print(f"\nWrote comparison CSV: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
