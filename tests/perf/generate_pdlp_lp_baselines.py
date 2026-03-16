#!/usr/bin/env python3
"""Generate LP PDLP comparison baseline CSVs for mipx/cuPDLPx/HiGHS/cuOpt."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT_DIR / "tests" / "perf" / "baselines"
BIN = ROOT_DIR / "build" / "mipx-solve"
NETLIB_DIR = ROOT_DIR / "tests" / "data" / "netlib"
SCRIPT = ROOT_DIR / "tests" / "perf" / "run_pdlp_lp_compare.py"

OUT_NETLIB = OUT_DIR / "pdlp_lp_compare_netlib.csv"
OUT_NETLIB_FORCED = OUT_DIR / "pdlp_lp_compare_netlib_forced_gpu.csv"
OUT_META = OUT_DIR / "pdlp_lp_compare_meta.json"


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(cmd))
    return subprocess.run(cmd, check=check, text=True)


def capture(cmd: list[str]) -> str:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, check=False)
        txt = (p.stdout or "").strip()
        if txt:
            return txt
        return (p.stderr or "").strip()
    except Exception:
        return ""


def resolve_highs_bin() -> str:
    return os.environ.get("HIGHS_BIN") or os.environ.get("HIGHS_BINARY") or "highs"


def resolve_cupdlpx_bin() -> str:
    return os.environ.get("MIPX_CUPDLPX_BINARY") or os.environ.get("CUPDLPX_BINARY") or "cupdlpx"


def main() -> int:
    highs_bin = resolve_highs_bin()
    cupdlpx_bin = resolve_cupdlpx_bin()

    if not BIN.is_file() or not os.access(BIN, os.X_OK):
        raise SystemExit(f"mipx binary not found/executable: {BIN}")
    if not NETLIB_DIR.is_dir():
        raise SystemExit(f"Netlib directory not found: {NETLIB_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    common_cmd = [
        sys.executable,
        str(SCRIPT),
        "--mipx-binary",
        str(BIN),
        "--highs-binary",
        highs_bin,
        "--instances-dir",
        str(NETLIB_DIR),
        "--repeats",
        "3",
        "--threads",
        "1",
        "--time-limit",
        "120",
        "--disable-presolve",
    ]
    if shutil.which(cupdlpx_bin) or Path(cupdlpx_bin).is_file():
        common_cmd.extend(["--cupdlpx-binary", cupdlpx_bin])
    else:
        common_cmd.append("--no-cupdlpx")

    print("=== Generating Netlib PDLP comparison baseline (GPU auto) ===")
    run(common_cmd + ["--output", str(OUT_NETLIB)])

    print("=== Generating Netlib PDLP comparison baseline (GPU forced) ===")
    run(common_cmd + ["--output", str(OUT_NETLIB_FORCED), "--force-mipx-gpu"])

    meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mipx_binary": str(BIN),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cuopt_version": "",
        "cupdlpx_binary": cupdlpx_bin,
        "cupdlpx_version": "",
        "nvidia_smi": capture(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
        "highs_version": capture([highs_bin, "--version"]),
    }

    cuopt_ver = capture([sys.executable, "-m", "libcuopt._cli_wrapper", "--version"])
    if cuopt_ver:
        meta["cuopt_version"] = cuopt_ver
    cupdlpx_ver = capture([cupdlpx_bin, "--help"])
    if cupdlpx_ver:
        meta["cupdlpx_version"] = cupdlpx_ver.splitlines()[0]

    OUT_META.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print("Wrote:")
    print(f"  {OUT_NETLIB}")
    print(f"  {OUT_NETLIB_FORCED}")
    print(f"  {OUT_META}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
