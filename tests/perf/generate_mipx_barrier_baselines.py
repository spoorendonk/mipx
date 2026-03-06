#!/usr/bin/env python3
"""Generate committed mipx barrier baseline CSVs for regression gates."""

from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT_DIR / "tests" / "perf" / "baselines"
NETLIB_DIR = ROOT_DIR / "tests" / "data" / "netlib"
BIN = ROOT_DIR / "build" / "mipx-solve"

COMPARE_SCRIPT = ROOT_DIR / "tests" / "perf" / "run_barrier_lp_compare.py"
COMPARE_OUT = OUT_DIR / "_mipx_barrier_compare_netlib_tmp.csv"

CPU_OUT = OUT_DIR / "mipx_barrier_cpu_netlib_small.csv"
GPU_OUT = OUT_DIR / "mipx_barrier_gpu_netlib_small.csv"
META_JSON = OUT_DIR / "mipx_barrier_baseline_meta.json"


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(cmd))
    return subprocess.run(cmd, check=check, text=True)


def capture(cmd: list[str], *, cwd: Path | None = None) -> str:
    try:
        return subprocess.check_output(cmd, cwd=cwd, text=True).strip()
    except Exception:
        return ""


def split_compare_csv(compare_csv: Path) -> tuple[int, int]:
    with compare_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    out_fields = ["instance", "time_seconds", "work_units", "status"]
    cpu_rows = [r for r in rows if r.get("solver", "").strip() == "mipx_barrier_cpu"]
    gpu_rows = [r for r in rows if r.get("solver", "").strip() == "mipx_barrier_gpu"]

    if not cpu_rows:
        raise SystemExit("No mipx_barrier_cpu rows found in compare CSV")
    if not gpu_rows:
        raise SystemExit("No mipx_barrier_gpu rows found in compare CSV")

    def write_out(path: Path, lane_rows: list[dict[str, str]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=out_fields)
            writer.writeheader()
            for row in lane_rows:
                writer.writerow(
                    {
                        "instance": row.get("instance", ""),
                        "time_seconds": row.get("time_seconds", ""),
                        "work_units": row.get("work_units", ""),
                        "status": row.get("status", ""),
                    }
                )

    write_out(CPU_OUT, cpu_rows)
    write_out(GPU_OUT, gpu_rows)
    return len(cpu_rows), len(gpu_rows)


def main() -> int:
    if not BIN.is_file() or not os.access(BIN, os.X_OK):
        raise SystemExit(f"Binary not found/executable: {BIN}")
    if not NETLIB_DIR.is_dir():
        raise SystemExit(f"Netlib dir not found: {NETLIB_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    run(
        [
            sys.executable,
            str(COMPARE_SCRIPT),
            "--mipx-binary",
            str(BIN),
            "--instances-dir",
            str(NETLIB_DIR),
            "--output",
            str(COMPARE_OUT),
            "--repeats",
            "3",
            "--threads",
            "1",
            "--time-limit",
            "60",
            "--disable-presolve",
            "--force-mipx-gpu",
            "--no-highs",
            "--no-cuopt",
        ]
    )

    cpu_count, gpu_count = split_compare_csv(COMPARE_OUT)
    try:
        COMPARE_OUT.unlink()
    except OSError:
        pass

    meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": capture(["git", "rev-parse", "HEAD"], cwd=ROOT_DIR),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "mipx_binary": str(BIN),
        "netlib_dir": str(NETLIB_DIR),
        "generator": "generate_mipx_barrier_baselines.py",
        "compare_args": {
            "repeats": 3,
            "threads": 1,
            "time_limit_seconds": 60,
            "disable_presolve": True,
            "force_mipx_gpu": True,
            "no_highs": True,
            "no_cuopt": True,
        },
        "rows": {
            "mipx_barrier_cpu": cpu_count,
            "mipx_barrier_gpu": gpu_count,
        },
    }
    cpu = capture(["bash", "-lc", "lscpu | head -n 20"])
    if cpu:
        meta["cpu_info"] = cpu
    nvidia = capture(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ]
    )
    if nvidia:
        meta["nvidia_smi"] = nvidia

    META_JSON.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print("Wrote baselines:")
    print(f"  {CPU_OUT}")
    print(f"  {GPU_OUT}")
    print(f"  {META_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
