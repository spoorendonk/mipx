#!/usr/bin/env python3
"""Run mipx-solve LP benchmark matching Mittelman's LPopt configuration."""

from __future__ import annotations

import argparse
import csv
import re
import statistics
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]


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
    p.add_argument("--mittelman-dir", default=str(ROOT_DIR / "tests" / "data" / "mittelman_lp"))
    p.add_argument("--miplib-dir", default=str(ROOT_DIR / "tests" / "data" / "miplib"))
    p.add_argument("--netlib-dir", default=str(ROOT_DIR / "tests" / "data" / "netlib"))
    p.add_argument("--output", required=True)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--time-limit", type=float, default=15000)
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


def run_cmd(cmd: list[str], timeout_s: float) -> tuple[str, str, float]:
    import time

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        elapsed = time.perf_counter() - t0
        out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        if proc.returncode != 0:
            return "solve_error", out, elapsed
        return "ok", out, elapsed
    except subprocess.TimeoutExpired as e:
        elapsed = time.perf_counter() - t0
        out = ((e.stdout or "") if isinstance(e.stdout, str) else "") + (
            "\n" + (e.stderr or "") if isinstance(e.stderr, str) and e.stderr else ""
        )
        return "time_limit", out, elapsed


def main() -> int:
    args = parse_args()
    bin_path = Path(args.binary)
    mittelman_dir = Path(args.mittelman_dir)
    miplib_dir = Path(args.miplib_dir)
    netlib_dir = Path(args.netlib_dir)
    out_path = Path(args.output)

    if not bin_path.is_file() or not bin_path.stat().st_mode & 0o111:
        raise SystemExit(f"Binary not executable: {bin_path}")

    # Dedupe by name, preferring the LPopt-specific directory first, then MIPLIB
    # LP-relaxation anchors, then Netlib fallback names.
    instance_map: dict[str, Path] = {}
    if mittelman_dir.is_dir():
        for f in sorted(mittelman_dir.glob("*.mps.gz")):
            instance_map[f.name.removesuffix(".mps.gz")] = f
    if miplib_dir.is_dir():
        for f in sorted(miplib_dir.glob("*.mps.gz")):
            key = f.name.removesuffix(".mps.gz")
            instance_map.setdefault(key, f)
    if netlib_dir.is_dir():
        for f in sorted(netlib_dir.glob("*.mps.gz")):
            key = f.name.removesuffix(".mps.gz")
            instance_map.setdefault(key, f)

    if not instance_map:
        raise SystemExit(f"No .mps.gz instances found in {mittelman_dir} or {netlib_dir}")

    requested = [x.strip() for x in args.instances.split(",") if x.strip()]
    selected_names = requested if requested else sorted(instance_map)
    print(f"[mittelman-lp] Selected {len(selected_names)} LP instances")

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
                code, out, elapsed = run_cmd([str(bin_path), str(inst), *args.solver_arg], args.time_limit)
                if code in {"time_limit", "solve_error"}:
                    status = code
                    break

                t = parse_solve_time(out)
                if t is None or (t <= 0.0 and elapsed > 0.0):
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

            if status in {"solve_error", "parse_error", "time_limit"}:
                w.writerow([name, "", "", status, "", ""])
                print(f"  {name}: {status}")
                continue

            med_t = median(times)
            med_w = median(works)
            obj_out = f"{median(objectives):.12g}" if objectives else ""
            it_out = f"{median(iterations):.12g}" if iterations else ""
            w.writerow([name, f"{med_t:.6f}", f"{med_w:.6f}", status, obj_out, it_out])
            print(f"  {name}: {med_t:.6f}s, {med_w:.6f} wu, {status}")

    print(f"Wrote Mittelman LP benchmark CSV: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
