import csv
import os
import stat
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
COMPARE_SCRIPT = REPO_ROOT / "tests" / "perf" / "run_pdlp_lp_compare.py"


def make_executable(path: Path) -> None:
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR)


def test_compare_script_includes_cupdlpx_rows(tmp_path: Path) -> None:
    instances_dir = tmp_path / "netlib"
    instances_dir.mkdir()
    (instances_dir / "toy.mps.gz").write_bytes(b"")

    mipx = tmp_path / "fake_mipx.sh"
    mipx.write_text(
        """#!/usr/bin/env bash
cat <<'OUT'
Status: Optimal
Objective: -8
Iterations: 34
Work units: 56
Time: 0.02s
OUT
""",
        encoding="utf-8",
    )
    make_executable(mipx)

    cupdlpx = tmp_path / "fake_cupdlpx.sh"
    cupdlpx.write_text(
        """#!/usr/bin/env bash
cat <<'OUT'
Solution Summary
  Status                 : OPTIMAL
  Solve time             : 0.0187 sec
  Iterations             : 600
  Primal objective       : -8
OUT
""",
        encoding="utf-8",
    )
    make_executable(cupdlpx)

    out_csv = tmp_path / "compare.csv"
    cmd = [
        sys.executable,
        str(COMPARE_SCRIPT),
        "--mipx-binary",
        str(mipx),
        "--cupdlpx-binary",
        str(cupdlpx),
        "--instances-dir",
        str(instances_dir),
        "--output",
        str(out_csv),
        "--repeats",
        "1",
        "--threads",
        "1",
        "--time-limit",
        "1",
        "--no-highs",
        "--no-cuopt",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=os.environ.copy())
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr

    rows = list(csv.DictReader(out_csv.open(encoding="utf-8")))
    assert {row["solver"] for row in rows} == {"mipx_pdlp_cpu", "mipx_pdlp_gpu", "cupdlpx"}

    cupdlpx_row = next(row for row in rows if row["solver"] == "cupdlpx")
    assert cupdlpx_row["status"] == "optimal"
    assert cupdlpx_row["objective"] == "-8"
    assert cupdlpx_row["iterations"] == "600"
    assert cupdlpx_row["time_seconds"] == "0.018700"


def test_compare_script_enforces_external_time_limit(tmp_path: Path) -> None:
    instances_dir = tmp_path / "netlib"
    instances_dir.mkdir()
    (instances_dir / "toy.mps.gz").write_bytes(b"")

    sleeper = tmp_path / "fake_timeout_solver.sh"
    sleeper.write_text(
        """#!/usr/bin/env bash
sleep 2
cat <<'OUT'
Status: Optimal
Objective: -8
Iterations: 34
Work units: 56
Time: 2s
OUT
""",
        encoding="utf-8",
    )
    make_executable(sleeper)

    out_csv = tmp_path / "compare_timeout.csv"
    cmd = [
        sys.executable,
        str(COMPARE_SCRIPT),
        "--mipx-binary",
        str(sleeper),
        "--instances-dir",
        str(instances_dir),
        "--output",
        str(out_csv),
        "--repeats",
        "1",
        "--threads",
        "1",
        "--time-limit",
        "0.1",
        "--no-cupdlpx",
        "--no-highs",
        "--no-cuopt",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=os.environ.copy())
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr

    rows = list(csv.DictReader(out_csv.open(encoding="utf-8")))
    assert rows
    assert {row["status"] for row in rows} == {"time_limit"}
