#!/usr/bin/env python3
"""Generate mipx dual-simplex LP baselines for Netlib anchors + Mittelman curated."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
PERF_DIR = ROOT_DIR / "tests" / "perf"
DEFAULT_OUT_DIR = PERF_DIR / "baselines"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", default=str(ROOT_DIR / "build" / "mipx-solve"))
    p.add_argument("--netlib-dir", default=str(ROOT_DIR / "tests" / "data" / "netlib"))
    p.add_argument("--mittelman-dir", default=str(ROOT_DIR / "tests" / "data" / "mittelman_lp"))
    p.add_argument("--miplib-dir", default=str(ROOT_DIR / "tests" / "data" / "miplib"))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--netlib-corpus", default=str(PERF_DIR / "netlib_dual_corpus.csv"))
    p.add_argument("--mittelman-corpus", default=str(PERF_DIR / "mittelman_dual_corpus.csv"))
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--netlib-time-limit", type=float, default=60.0)
    p.add_argument("--mittelman-time-limit", type=float, default=60.0)
    p.add_argument("--solver-arg", action="append", default=[])
    return p.parse_args()


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def capture(cmd: list[str], *, cwd: Path | None = None) -> str:
    try:
        return subprocess.check_output(cmd, cwd=cwd, text=True).strip()
    except Exception:
        return ""


def load_instances(path: Path) -> list[str]:
    if not path.is_file():
        raise SystemExit(f"corpus CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    out: list[str] = []
    for row in rows:
        name = row.get("instance", "").strip()
        if name:
            out.append(name)
    if not out:
        raise SystemExit(f"no instances found in corpus: {path}")
    return out


def main() -> int:
    args = parse_args()

    binary = Path(args.binary)
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise SystemExit(f"mipx binary not executable: {binary}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    netlib_instances = load_instances(Path(args.netlib_corpus))
    mittelman_instances = load_instances(Path(args.mittelman_corpus))

    run_netlib = PERF_DIR / "run_netlib_lp_bench.py"
    run_mittelman = PERF_DIR / "run_mittelman_lp_bench.py"

    netlib_out = out_dir / "mipx_dual_lp_netlib_anchors.csv"
    mittelman_out = out_dir / "mipx_dual_lp_mittelman_curated.csv"
    meta_out = out_dir / "mipx_dual_lp_baseline_meta.json"

    solver_args = ["--dual", "--no-presolve", "--relax-integrality", "--quiet", *args.solver_arg]
    solver_arg_tokens: list[str] = []
    for sarg in solver_args:
        solver_arg_tokens.extend(["--solver-arg", sarg])

    run(
        [
            sys.executable,
            str(run_netlib),
            "--binary",
            str(binary),
            "--netlib-dir",
            args.netlib_dir,
            "--output",
            str(netlib_out),
            "--repeats",
            str(args.repeats),
            "--time-limit",
            f"{args.netlib_time_limit:g}",
            "--instances",
            ",".join(netlib_instances),
            *solver_arg_tokens,
        ]
    )
    run(
        [
            sys.executable,
            str(run_mittelman),
            "--binary",
            str(binary),
            "--mittelman-dir",
            args.mittelman_dir,
            "--miplib-dir",
            args.miplib_dir,
            "--netlib-dir",
            args.netlib_dir,
            "--output",
            str(mittelman_out),
            "--repeats",
            str(args.repeats),
            "--time-limit",
            f"{args.mittelman_time_limit:g}",
            "--instances",
            ",".join(mittelman_instances),
            *solver_arg_tokens,
        ]
    )

    meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": capture(["git", "rev-parse", "HEAD"], cwd=ROOT_DIR),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "binary": str(binary),
        "config": {
            "repeats": args.repeats,
            "netlib_time_limit": args.netlib_time_limit,
            "mittelman_time_limit": args.mittelman_time_limit,
            "solver_args": solver_args,
            "netlib_corpus": args.netlib_corpus,
            "mittelman_corpus": args.mittelman_corpus,
        },
    }
    cpu = capture(["bash", "-lc", "lscpu | head -n 20"])
    if cpu:
        meta["cpu_info"] = cpu
    meta_out.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print("Wrote dual LP baselines:")
    print(f"  {netlib_out}")
    print(f"  {mittelman_out}")
    print(f"  {meta_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
