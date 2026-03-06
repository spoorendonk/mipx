import csv
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CHECK_SCRIPT = REPO_ROOT / "tests" / "perf" / "check_regression.py"


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["instance", "status", "work_units", "time_seconds"],
        )
        writer.writeheader()
        writer.writerows(rows)


def run_check(args: list[str]) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, str(CHECK_SCRIPT), *args]
    return subprocess.run(cmd, capture_output=True, text=True)


def test_primary_instance_cap_failure(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.csv"
    candidate = tmp_path / "candidate.csv"
    write_csv(
        baseline,
        [
            {"instance": "a", "status": "optimal", "work_units": "100", "time_seconds": "10"},
            {"instance": "b", "status": "optimal", "work_units": "100", "time_seconds": "10"},
        ],
    )
    write_csv(
        candidate,
        [
            {"instance": "a", "status": "optimal", "work_units": "140", "time_seconds": "10"},
            {"instance": "b", "status": "optimal", "work_units": "95", "time_seconds": "10"},
        ],
    )

    proc = run_check(
        [
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
            "--metric",
            "work_units",
            "--max-regression-pct",
            "30",
            "--max-instance-regression-pct",
            "20",
            "--min-common-instances",
            "2",
            "--status-column",
            "status",
            "--required-statuses",
            "optimal",
            "--require-status-match",
        ]
    )
    assert proc.returncode == 1, proc.stdout + "\n" + proc.stderr
    assert "Per-instance cap exceeded" in proc.stdout


def test_secondary_warn_mode_does_not_fail_gate(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.csv"
    candidate = tmp_path / "candidate.csv"
    summary_json = tmp_path / "summary.json"
    write_csv(
        baseline,
        [
            {"instance": "a", "status": "optimal", "work_units": "100", "time_seconds": "10"},
            {"instance": "b", "status": "optimal", "work_units": "100", "time_seconds": "10"},
        ],
    )
    write_csv(
        candidate,
        [
            {"instance": "a", "status": "optimal", "work_units": "100", "time_seconds": "14"},
            {"instance": "b", "status": "optimal", "work_units": "100", "time_seconds": "11"},
        ],
    )

    proc = run_check(
        [
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
            "--metric",
            "work_units",
            "--max-regression-pct",
            "0",
            "--min-common-instances",
            "2",
            "--status-column",
            "status",
            "--required-statuses",
            "optimal",
            "--require-status-match",
            "--secondary-metric",
            "time_seconds",
            "--max-secondary-regression-pct",
            "5",
            "--secondary-mode",
            "warn",
            "--summary-json",
            str(summary_json),
        ]
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    assert "WARN:" in proc.stdout
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    assert payload["pass"] is True
    assert payload["secondary"] is not None


def test_status_mismatch_fails_when_required(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.csv"
    candidate = tmp_path / "candidate.csv"
    write_csv(
        baseline,
        [
            {"instance": "a", "status": "optimal", "work_units": "100", "time_seconds": "10"},
            {"instance": "b", "status": "optimal", "work_units": "100", "time_seconds": "10"},
        ],
    )
    write_csv(
        candidate,
        [
            {"instance": "a", "status": "infeasible", "work_units": "100", "time_seconds": "10"},
            {"instance": "b", "status": "optimal", "work_units": "100", "time_seconds": "10"},
        ],
    )

    proc = run_check(
        [
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
            "--metric",
            "work_units",
            "--max-regression-pct",
            "0",
            "--min-common-instances",
            "1",
            "--status-column",
            "status",
            "--require-status-match",
        ]
    )
    assert proc.returncode == 1, proc.stdout + "\n" + proc.stderr
    assert "status mismatch" in proc.stdout.lower()


def test_secondary_fail_mode_enforces_time_regression(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.csv"
    candidate = tmp_path / "candidate.csv"
    write_csv(
        baseline,
        [
            {"instance": "a", "status": "optimal", "work_units": "100", "time_seconds": "10"},
            {"instance": "b", "status": "optimal", "work_units": "100", "time_seconds": "10"},
        ],
    )
    write_csv(
        candidate,
        [
            {"instance": "a", "status": "optimal", "work_units": "100", "time_seconds": "20"},
            {"instance": "b", "status": "optimal", "work_units": "100", "time_seconds": "20"},
        ],
    )

    proc = run_check(
        [
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
            "--metric",
            "work_units",
            "--max-regression-pct",
            "0",
            "--min-common-instances",
            "2",
            "--status-column",
            "status",
            "--required-statuses",
            "optimal",
            "--require-status-match",
            "--secondary-metric",
            "time_seconds",
            "--max-secondary-regression-pct",
            "5",
            "--secondary-mode",
            "fail",
        ]
    )
    assert proc.returncode == 1, proc.stdout + "\n" + proc.stderr
    assert "Secondary metric" in proc.stdout
