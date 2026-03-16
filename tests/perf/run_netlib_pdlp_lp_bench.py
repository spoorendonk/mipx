#!/usr/bin/env python3
"""Run mipx-solve PDLP over Netlib .mps.gz files and emit median benchmark CSV."""

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
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--time-limit", type=float, default=60.0)
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--force-gpu", action="store_true")
    p.add_argument("--disable-presolve", action="store_true")
    p.add_argument("--instances", default="")
    p.add_argument("--max-instances", type=int, default=0)
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


def run_cmd(cmd: list[str]) -> tuple[int, str, float]:
    import time

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    return proc.returncode, out, elapsed


def instance_name(path: Path) -> str:
    name = path.name
    if name.endswith(".mps.gz"):
        return name[: -len(".mps.gz")]
    if name.endswith(".mps"):
        return name[: -len(".mps")]
    return path.stem


def main() -> int:
    args = parse_args()
    bin_path = Path(args.binary)
    netlib_dir = Path(args.netlib_dir)
    out_path = Path(args.output)

    if not bin_path.is_file() or not bin_path.stat().st_mode & 0o111:
        raise SystemExit(f"Binary not executable: {bin_path}")
    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")
    if args.threads < 1:
        raise SystemExit("--threads must be >= 1")

    instances = sorted(netlib_dir.glob("*.mps.gz")) + sorted(netlib_dir.glob("*.mps"))
    if not instances:
        raise SystemExit(f"No .mps/.mps.gz instances found in {netlib_dir}")
    if args.instances:
        keep = {x.strip() for x in args.instances.split(",") if x.strip()}
        instances = [m for m in instances if instance_name(m) in keep]
    if args.max_instances > 0:
        instances = instances[: args.max_instances]
    if not instances:
        raise SystemExit("No instances selected")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["instance", "time_seconds", "work_units", "status"])

        for inst in instances:
            name = instance_name(inst)
            times: list[float] = []
            works: list[float] = []
            status = "ok"

            for _ in range(args.repeats):
                cmd = [
                    str(bin_path),
                    str(inst),
                    "--pdlp",
                    "--threads",
                    str(args.threads),
                ]
                if args.time_limit > 0:
                    cmd.extend(["--time-limit", f"{args.time_limit:g}"])
                if args.disable_presolve:
                    cmd.append("--no-presolve")
                if args.gpu:
                    cmd.append("--gpu")
                    if args.force_gpu:
                        cmd.extend(["--gpu-min-rows", "0", "--gpu-min-nnz", "0"])
                else:
                    cmd.append("--no-gpu")
                cmd.extend(args.solver_arg)

                code, out, elapsed = run_cmd(cmd)
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

                run_status = parse_status(out)
                if status == "ok":
                    status = run_status
                elif status != run_status:
                    status = "mixed_status"

                times.append(t)
                works.append(wu)

            if status in {"solve_error", "parse_error"}:
                w.writerow([name, "", "", status])
                continue

            w.writerow([name, f"{median(times):.6f}", f"{median(works):.6f}", status])

    print(f"Wrote benchmark CSV: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
