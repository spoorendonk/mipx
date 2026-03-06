import csv
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "tests" / "perf" / "run_dual_perf_gate.py"


def write_lp_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["instance", "time_seconds", "work_units", "status", "objective", "iterations"],
        )
        writer.writeheader()
        writer.writerows(rows)


def test_dual_perf_gate_csv_only_mode(tmp_path: Path) -> None:
    base_netlib = tmp_path / "base_netlib.csv"
    cand_netlib = tmp_path / "cand_netlib.csv"
    base_mittel = tmp_path / "base_mittel.csv"
    cand_mittel = tmp_path / "cand_mittel.csv"
    out_dir = tmp_path / "out"

    write_lp_csv(
        base_netlib,
        [
            {
                "instance": "afiro",
                "time_seconds": "1.0",
                "work_units": "100",
                "status": "optimal",
                "objective": "-464.75314286",
                "iterations": "10",
            },
            {
                "instance": "adlittle",
                "time_seconds": "2.0",
                "work_units": "200",
                "status": "optimal",
                "objective": "225494.96316",
                "iterations": "20",
            },
        ],
    )
    write_lp_csv(
        cand_netlib,
        [
            {
                "instance": "afiro",
                "time_seconds": "1.1",
                "work_units": "100",
                "status": "optimal",
                "objective": "-464.75314286",
                "iterations": "10",
            },
            {
                "instance": "adlittle",
                "time_seconds": "2.1",
                "work_units": "198",
                "status": "optimal",
                "objective": "225494.96316",
                "iterations": "20",
            },
        ],
    )
    write_lp_csv(
        base_mittel,
        [
            {
                "instance": "ex10",
                "time_seconds": "4.0",
                "work_units": "500",
                "status": "optimal",
                "objective": "0",
                "iterations": "100",
            },
            {
                "instance": "dlr1",
                "time_seconds": "3.0",
                "work_units": "450",
                "status": "optimal",
                "objective": "0",
                "iterations": "120",
            },
        ],
    )
    write_lp_csv(
        cand_mittel,
        [
            {
                "instance": "ex10",
                "time_seconds": "4.1",
                "work_units": "505",
                "status": "optimal",
                "objective": "0",
                "iterations": "101",
            },
            {
                "instance": "dlr1",
                "time_seconds": "3.2",
                "work_units": "440",
                "status": "optimal",
                "objective": "0",
                "iterations": "119",
            },
        ],
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--candidate-binary",
            "/bin/true",
            "--candidate-netlib-csv",
            str(cand_netlib),
            "--candidate-mittelman-csv",
            str(cand_mittel),
            "--baseline-netlib-csv",
            str(base_netlib),
            "--baseline-mittelman-csv",
            str(base_mittel),
            "--out-dir",
            str(out_dir),
            "--netlib-min-common",
            "2",
            "--mittelman-min-common",
            "2",
            "--max-work-regression-pct",
            "10",
            "--max-work-instance-regression-pct",
            "10",
            "--time-regression-mode",
            "warn",
            "--max-time-regression-pct",
            "20",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr

    summary_md = out_dir / "dual_perf_summary.md"
    assert summary_md.is_file()
    text = summary_md.read_text(encoding="utf-8")
    assert "Dual Simplex Performance Gate" in text
    assert "Netlib Anchors" in text
    assert "Mittelman Curated" in text
