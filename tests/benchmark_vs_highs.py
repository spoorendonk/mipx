#!/usr/bin/env python3
"""Benchmark mipx dual simplex vs HiGHS on Netlib LP instances."""

import subprocess
import time
import sys
import os
import glob
import re

try:
    import highspy
except ImportError:
    print("highspy not installed. Install with: pip install highspy")
    sys.exit(1)

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
TIMEOUT = 60  # seconds


def instance_name(path: str) -> str:
    """Extract instance name from path like .../afiro.mps.gz."""
    base = os.path.basename(path)
    return base.replace(".mps.gz", "").replace(".mps", "")


def solve_mipx(filepath: str) -> dict:
    """Solve with mipx-solve, return dict with obj, iters, time, status."""
    result = {"obj": None, "iters": None, "time": None, "status": "error"}
    try:
        start = time.perf_counter()
        proc = subprocess.run(
            [MIPX_BINARY, filepath],
            capture_output=True, text=True, timeout=TIMEOUT,
        )
        elapsed = time.perf_counter() - start
        result["time"] = elapsed

        if proc.returncode != 0:
            result["status"] = "error"
            return result

        for line in proc.stdout.splitlines():
            m = re.match(r"Objective:\s*(.+)", line)
            if m:
                result["obj"] = float(m.group(1))
            m = re.match(r"Iterations:\s*(\d+)", line)
            if m:
                result["iters"] = int(m.group(1))
            m = re.match(r"Status:\s*(.+)", line)
            if m:
                result["status"] = m.group(1).strip().lower()

    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["time"] = TIMEOUT
    except FileNotFoundError:
        result["status"] = "binary not found"

    return result


def solve_highs(filepath: str) -> dict:
    """Solve with HiGHS, return dict with obj, iters, time, status."""
    result = {"obj": None, "iters": None, "time": None, "status": "error"}
    try:
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)

        start = time.perf_counter()
        h.readModel(filepath)
        h.run()
        elapsed = time.perf_counter() - start
        result["time"] = elapsed

        status = h.getInfoValue("primal_solution_status")[1]
        result["obj"] = h.getInfoValue("objective_function_value")[1]
        result["iters"] = int(h.getInfoValue("simplex_iteration_count")[1])

        model_status = h.getModelStatus()
        if model_status == highspy.HighsModelStatus.kOptimal:
            result["status"] = "optimal"
        elif model_status == highspy.HighsModelStatus.kInfeasible:
            result["status"] = "infeasible"
        elif model_status == highspy.HighsModelStatus.kUnbounded:
            result["status"] = "unbounded"
        else:
            result["status"] = str(model_status)

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


def main():
    if not os.path.isdir(NETLIB_DIR):
        print(f"Netlib directory not found: {NETLIB_DIR}")
        sys.exit(1)

    files = sorted(glob.glob(os.path.join(NETLIB_DIR, "*.mps.gz")))
    if not files:
        print("No .mps.gz files found in", NETLIB_DIR)
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

        mipx_res = solve_mipx(filepath)
        highs_res = solve_highs(filepath)

        mipx_obj_s = f"{mipx_res['obj']:.8e}" if mipx_res["obj"] is not None else mipx_res["status"]
        mipx_it_s = str(mipx_res["iters"]) if mipx_res["iters"] is not None else "-"
        mipx_t_s = f"{mipx_res['time']:.4f}" if mipx_res["time"] is not None else "-"
        mipx_chk = check_obj(name, mipx_res["obj"], "mipx")

        highs_obj_s = f"{highs_res['obj']:.8e}" if highs_res["obj"] is not None else highs_res["status"]
        highs_it_s = str(highs_res["iters"]) if highs_res["iters"] is not None else "-"
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
