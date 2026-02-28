#!/usr/bin/env python3
"""Benchmark mipx dual simplex vs HiGHS CLI on Netlib LP instances."""

from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import time

# Known optimal values for correctness checking.
KNOWN_OPTIMA = {
    "afiro": -4.6475314286e+02,
    "adlittle": 2.2549496316e+05,
    "blend": -3.0812149846e+01,
    "sc50a": -6.4575077059e+01,
    "sc50b": -7.0000000000e+01,
    "kb2": -1.7499001299e+03,
    "sc105": -5.2202061212e+01,
    "sc205": -5.2202061212e+01,
    "share1b": -7.6589318579e+04,
    "share2b": -4.1573224074e+02,
    "stocfor1": -4.1131976219e+04,
}

MIPX_BINARY = os.path.join(os.path.dirname(__file__), "..", "build", "mipx-solve")
NETLIB_DIR = os.path.join(os.path.dirname(__file__), "data", "netlib")
TIMEOUT = 60  # seconds (default; override via --timeout)


def locate_highs_binary() -> str | None:
    explicit = os.environ.get("HIGHS_BINARY")
    if explicit:
        return explicit
    return shutil.which("highs")


def instance_name(path: str) -> str:
    """Extract instance name from path like .../afiro.mps.gz."""
    base = os.path.basename(path)
    return base.replace(".mps.gz", "").replace(".mps", "")


def parse_first_float(pattern: str, text: str) -> float | None:
    m = re.search(pattern, text, re.MULTILINE)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def normalize_status(raw: str) -> str:
    status = re.sub(r"[^a-z0-9]+", "_", raw.strip().lower()).strip("_")
    return {
        "time_limit_reached": "time_limit",
        "iteration_limit_reached": "iteration_limit",
    }.get(status, status or "unknown")


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


def solve_mipx(filepath: str, timeout_s: float, disable_presolve: bool) -> dict:
    """Solve with mipx-solve, return dict with obj, iters, time, status."""
    result = {"obj": None, "iters": None, "time": None, "status": "error"}
    try:
        start = time.perf_counter()
        cmd = [MIPX_BINARY, filepath, "--dual", "--quiet"]
        if disable_presolve:
            cmd.append("--no-presolve")
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        elapsed = time.perf_counter() - start
        output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        result["time"] = elapsed

        for line in output.splitlines():
            m = re.match(r"Objective:\s*(.+)", line)
            if m:
                result["obj"] = float(m.group(1))
            m = re.match(r"Iterations:\s*(\d+)", line)
            if m:
                result["iters"] = int(m.group(1))
            m = re.match(r"Status:\s*(.+)", line)
            if m:
                result["status"] = m.group(1).strip().lower()
            m = re.match(r"Time:\s*([\-+0-9.eE]+)s", line)
            if m:
                result["time"] = float(m.group(1))

        if proc.returncode != 0 and result["status"] in {"error", "", None}:
            result["status"] = "error"

    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["time"] = timeout_s
    except FileNotFoundError:
        result["status"] = "binary not found"

    return result


def solve_highs(filepath: str, highs_binary: str, timeout_s: float, disable_presolve: bool) -> dict:
    """Solve with HiGHS CLI, return dict with obj, iters, time, status."""
    result = {"obj": None, "iters": None, "time": None, "status": "error"}
    cmd = [
        highs_binary,
        filepath,
        "--solver",
        "simplex",
        "--presolve",
        "off" if disable_presolve else "choose",
        "--parallel",
        "off",
        "--time_limit",
        str(timeout_s),
    ]
    try:
        start = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        elapsed = time.perf_counter() - start
        output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        result["status"] = parse_highs_status(output)
        result["obj"] = parse_first_float(r"^\s*Objective value\s*:\s*([\-+0-9.eE]+)\s*$", output)
        result["iters"] = parse_first_float(
            r"^\s*Simplex\s+iterations:\s*([\-+0-9.eE]+)\s*$", output
        )
        if result["iters"] is None:
            result["iters"] = parse_first_float(r"^\s*LP iterations\s+([\-+0-9.eE]+)\s*$", output)
        result["time"] = parse_first_float(r"^\s*HiGHS run time\s*:\s*([\-+0-9.eE]+)\s*$", output)
        if result["time"] is None:
            result["time"] = elapsed
        if proc.returncode != 0 and result["status"] == "unknown":
            result["status"] = "error"

    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["time"] = timeout_s
    except Exception as e:
        result["status"] = f"error: {e}"

    return result


def check_obj(name: str, obj, label: str) -> str:
    """Check objective against known optimal. Return checkmark or cross."""
    if obj is None or name not in KNOWN_OPTIMA:
        return ""
    expected = KNOWN_OPTIMA[name]
    denom = max(1.0, abs(expected))
    rel_err = abs(obj - expected) / denom
    if rel_err < 1e-4:
        return "ok"
    return f"err={rel_err:.2e}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--netlib-dir", default=NETLIB_DIR)
    p.add_argument("--instances", default="")
    p.add_argument("--timeout", type=float, default=TIMEOUT)
    p.add_argument("--max-instances", type=int, default=0)
    p.add_argument("--no-presolve", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    highs_binary = locate_highs_binary()
    if not highs_binary:
        print("HiGHS CLI not found. Set HIGHS_BINARY or add highs to PATH.")
        sys.exit(1)

    if not os.path.isdir(args.netlib_dir):
        print(f"Netlib directory not found: {args.netlib_dir}")
        sys.exit(1)

    files = sorted(glob.glob(os.path.join(args.netlib_dir, "*.mps.gz")))
    if not files:
        files = sorted(glob.glob(os.path.join(args.netlib_dir, "*.mps")))
    if not files:
        print("No .mps/.mps.gz files found in", args.netlib_dir)
        sys.exit(1)

    if args.instances:
        selected = {x.strip() for x in args.instances.split(",") if x.strip()}
        files = [f for f in files if instance_name(f) in selected]
    if args.max_instances > 0:
        files = files[: args.max_instances]
    if not files:
        print("No instances selected after filtering")
        sys.exit(1)

    if not os.path.isfile(MIPX_BINARY):
        print(f"mipx-solve binary not found at {MIPX_BINARY}")
        print("Build first: cmake --build build -j$(nproc)")
        sys.exit(1)

    print(f"Benchmarking {len(files)} Netlib instances\n")

    # Header
    header = (
        f"| {'Instance':15s} | {'mipx obj':>16s} | {'mipx it':>8s} | "
        f"{'mipx t(s)':>9s} | {'chk':>5s} | {'HiGHS obj':>16s} | "
        f"{'HiGHS it':>8s} | {'HiGHS t(s)':>9s} | {'chk':>5s} | "
        f"{'speedup':>8s} |"
    )
    sep = "|" + "|".join("-" * len(c) for c in header.split("|")[1:-1]) + "|"
    print(header)
    print(sep)

    for filepath in files:
        name = instance_name(filepath)

        mipx_res = solve_mipx(filepath, args.timeout, args.no_presolve)
        highs_res = solve_highs(filepath, highs_binary, args.timeout, args.no_presolve)

        mipx_obj_s = (
            f"{mipx_res['obj']:.8e}" if mipx_res["obj"] is not None else mipx_res["status"]
        )
        mipx_it_s = str(mipx_res["iters"]) if mipx_res["iters"] is not None else "-"
        mipx_t_s = f"{mipx_res['time']:.4f}" if mipx_res["time"] is not None else "-"
        mipx_chk = check_obj(name, mipx_res["obj"], "mipx")

        highs_obj_s = (
            f"{highs_res['obj']:.8e}" if highs_res["obj"] is not None else highs_res["status"]
        )
        highs_it_s = str(int(highs_res["iters"])) if highs_res["iters"] is not None else "-"
        highs_t_s = f"{highs_res['time']:.4f}" if highs_res["time"] is not None else "-"
        highs_chk = check_obj(name, highs_res["obj"], "HiGHS")

        speedup = ""
        if mipx_res["time"] and highs_res["time"] and mipx_res["time"] > 0:
            speedup = f"{highs_res['time'] / mipx_res['time']:.2f}x"

        print(
            f"| {name:15s} | {mipx_obj_s:>16s} | {mipx_it_s:>8s} | "
            f"{mipx_t_s:>9s} | {mipx_chk:>5s} | {highs_obj_s:>16s} | "
            f"{highs_it_s:>8s} | {highs_t_s:>9s} | {highs_chk:>5s} | "
            f"{speedup:>8s} |"
        )

    print()


if __name__ == "__main__":
    main()
