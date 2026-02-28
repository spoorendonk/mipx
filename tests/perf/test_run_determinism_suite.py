import csv
import os
import stat
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SUITE_SCRIPT = REPO_ROOT / "tests" / "perf" / "run_determinism_suite.py"


def make_executable(path: Path) -> None:
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR)


def test_missing_time_line_is_ignored_in_deterministic_checks(tmp_path: Path) -> None:
    miplib_dir = tmp_path / "miplib"
    miplib_dir.mkdir()
    (miplib_dir / "toy.mps.gz").write_bytes(b"")

    fake_solver = tmp_path / "fake_solver.sh"
    fake_solver.write_text(
        """#!/usr/bin/env bash
cat <<'OUT'
Status: Optimal
Objective: -8
Nodes: 12
LP iterations: 34
Work units: 56
OUT
""",
        encoding="utf-8",
    )
    make_executable(fake_solver)

    out_dir = tmp_path / "out"
    cmd = [
        sys.executable,
        str(SUITE_SCRIPT),
        "--binary",
        str(fake_solver),
        "--miplib-dir",
        str(miplib_dir),
        "--out-dir",
        str(out_dir),
        "--instances",
        "toy",
        "--runs",
        "2",
        "--single-threads",
        "1",
        "--multi-threads",
        "1",
        "--time-limit",
        "1",
        "--node-limit",
        "1",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=os.environ.copy())
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr

    detail_rows = list(csv.DictReader((out_dir / "determinism_detail.csv").open(encoding="utf-8")))
    assert detail_rows
    assert all(row["error"] == "" for row in detail_rows)
    assert all(row["time_seconds"] == "" for row in detail_rows)

    summary_rows = list(csv.DictReader((out_dir / "determinism_summary.csv").open(encoding="utf-8")))
    assert len(summary_rows) == 1
    assert summary_rows[0]["stable"] == "yes"


def test_time_line_variation_does_not_break_deterministic_stability(tmp_path: Path) -> None:
    miplib_dir = tmp_path / "miplib"
    miplib_dir.mkdir()
    (miplib_dir / "toy.mps.gz").write_bytes(b"")

    counter_file = tmp_path / "time_counter.txt"
    fake_solver = tmp_path / "fake_solver_varying_time.sh"
    fake_solver.write_text(
        f"""#!/usr/bin/env bash
count=0
if [[ -f "{counter_file}" ]]; then
  count=$(cat "{counter_file}")
fi
next=$((count + 1))
echo "${{next}}" > "{counter_file}"
cat <<OUT
Status: Optimal
Objective: -8
Nodes: 12
LP iterations: 34
Work units: 56
Time: ${{next}}s
OUT
""",
        encoding="utf-8",
    )
    make_executable(fake_solver)

    out_dir = tmp_path / "out"
    cmd = [
        sys.executable,
        str(SUITE_SCRIPT),
        "--binary",
        str(fake_solver),
        "--miplib-dir",
        str(miplib_dir),
        "--out-dir",
        str(out_dir),
        "--instances",
        "toy",
        "--runs",
        "3",
        "--single-threads",
        "1",
        "--multi-threads",
        "1",
        "--time-limit",
        "1",
        "--node-limit",
        "1",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=os.environ.copy())
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr

    summary_rows = list(csv.DictReader((out_dir / "determinism_summary.csv").open(encoding="utf-8")))
    assert len(summary_rows) == 1
    assert summary_rows[0]["stable"] == "yes"
