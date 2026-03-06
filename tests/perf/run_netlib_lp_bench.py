#!/usr/bin/env python3
"""Run mipx-solve over Netlib LP files and emit median benchmark CSV."""

from __future__ import annotations

import argparse
import csv
import re
import statistics
import subprocess
import sys
from pathlib import Path


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
    p.add_argument("--binary", required=True)
    p.add_argument("--netlib-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument(
        "--instances",
        default="",
        help="Optional comma-separated instance names (without .mps/.mps.gz).",
    )
    p.add_argument("--solver-arg", action="append", default=[])
    return p.parse_args(normalize_solver_arg_tokens(sys.argv[1:]))


def median(vals: list[float]) -> float:
    return float(statistics.median(vals))


def parse_first_float(pattern: str, text: str) -> float | None:
    m = re.search(pattern, text, re.MULTILINE)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def parse_status(text: str) -> str:
    m = re.search(r"^Status:\s*(.+?)\s*$", text, re.MULTILINE)
    if not m:
        return "unknown"
    return m.group(1).strip().lower().replace(" ", "_")


def parse_work_units(text: str) -> float | None:
    for pat in (r"^Work units:\s*([\-+0-9.eE]+)\s*$", r"^Work:\s*([\-+0-9.eE]+)\s*$"):
        v = parse_first_float(pat, text)
        if v is not None:
            return v
    return None


def parse_solve_time(text: str) -> float | None:
    return parse_first_float(r"^Time:\s*([\-+0-9.eE]+)s\s*$", text)


def parse_objective(text: str) -> float | None:
    return parse_first_float(r"^Objective:\s*([\-+0-9.eE]+)\s*$", text)


def parse_iterations(text: str) -> float | None:
    for pat in (r"^Iterations:\s*([\-+0-9.eE]+)\s*$", r"^LP iterations:\s*([\-+0-9.eE]+)\s*$"):
        v = parse_first_float(pat, text)
        if v is not None:
            return v
    return None


def run_cmd(cmd: list[str]) -> tuple[int, str, float]:
    import time

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    return proc.returncode, out, elapsed


def main() -> int:
    args = parse_args()
    bin_path = Path(args.binary)
    netlib_dir = Path(args.netlib_dir)
    out_path = Path(args.output)

    if not bin_path.is_file() or not bin_path.stat().st_mode & 0o111:
        raise SystemExit(f"Binary not executable: {bin_path}")

    # Prefer .mps.gz, then .mps if gzip is not available.
    instance_map: dict[str, Path] = {}
    for inst in sorted(netlib_dir.glob("*.mps.gz")):
        instance_map[inst.name.removesuffix(".mps.gz")] = inst
    for inst in sorted(netlib_dir.glob("*.mps")):
        name = inst.name.removesuffix(".mps")
        instance_map.setdefault(name, inst)

    if not instance_map:
        raise SystemExit(f"No .mps/.mps.gz instances found in {netlib_dir}")

    requested = [x.strip() for x in args.instances.split(",") if x.strip()]
    selected_names = requested if requested else sorted(instance_map)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["instance", "time_seconds", "work_units", "status", "objective", "iterations"])

        for name in selected_names:
            inst = instance_map.get(name)
            if inst is None:
                w.writerow([name, "", "", "missing_instance", "", ""])
                print(f"  {name}: missing_instance")
                continue

            times: list[float] = []
            works: list[float] = []
            objectives: list[float] = []
            iterations: list[float] = []
            status = "ok"

            for _ in range(args.repeats):
                code, out, elapsed = run_cmd([str(bin_path), str(inst), *args.solver_arg])
                if code != 0:
                    status = "solve_error"
                    break

                t = parse_solve_time(out)
                if t is None:
                    t = elapsed
                wu = parse_work_units(out)
                if wu is None:
                    status = "parse_error"
                    break
                obj = parse_objective(out)
                it = parse_iterations(out)

                run_status = parse_status(out)
                if status == "ok":
                    status = run_status
                elif status != run_status:
                    status = "mixed_status"

                times.append(t)
                works.append(wu)
                if obj is not None:
                    objectives.append(obj)
                if it is not None:
                    iterations.append(it)

            if status in {"solve_error", "parse_error"}:
                w.writerow([name, "", "", status, "", ""])
                continue

            obj_out = f"{median(objectives):.12g}" if objectives else ""
            it_out = f"{median(iterations):.12g}" if iterations else ""
            w.writerow([name, f"{median(times):.6f}", f"{median(works):.6f}", status, obj_out, it_out])

    print(f"Wrote benchmark CSV: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
