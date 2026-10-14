"""Tests for run_instance_gate.py: classification, list handling, exit codes."""

from __future__ import annotations

import gzip
import importlib.util
import stat
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType

SCRIPT = Path(__file__).resolve().parent / "run_instance_gate.py"


def load_gate() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_instance_gate", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_instance_gate"] = module
    spec.loader.exec_module(module)
    return module


gate = load_gate()


@pytest.mark.parametrize(
    ("status", "objective", "ref", "expected"),
    [
        ("optimal", 100.0, ("opt", 100.0), "match_1e-6"),
        ("optimal", 100.001, ("opt", 100.0), "match_1e-4"),
        ("optimal", 100.1, ("opt", 100.0), "disagree"),
        ("gap_limit", 100.0, ("opt", 100.0), "match_1e-6"),
        ("optimal", None, ("opt", 100.0), "disagree"),
        ("infeasible", None, ("opt", 100.0), "disagree"),
        ("unbounded", None, ("opt", 100.0), "disagree"),
        ("optimal", 0.0, ("inf", None), "disagree"),
        ("infeasible", None, ("inf", None), "match_1e-6"),
        ("unbounded", None, ("unbd", None), "match_1e-6"),
        ("time_limit", 100.0, ("opt", 100.0), "dnf"),
        ("iteration_limit", None, ("opt", 100.0), "dnf"),
        ("error", None, ("opt", 100.0), "dnf"),
        ("crash", None, ("opt", 100.0), "dnf"),
        ("timeout", None, ("inf", None), "dnf"),
        # Near zero the error is absolute, not relative to a tiny reference.
        ("optimal", 1e-7, ("opt", 0.0), "match_1e-6"),
        # A MIP stopped by a limit with an incumbent worse than optimal: dnf.
        ("time_limit", 120.0, ("opt", 100.0), "dnf"),
        ("node_limit", 100.0, ("opt", 100.0), "dnf"),
        # ...but one better than the proven optimum is an infeasible point.
        ("time_limit", 90.0, ("opt", 100.0), "disagree"),
        ("node_limit", 90.0, ("opt", 100.0), "disagree"),
        # Any incumbent on an infeasible instance is wrong.
        ("time_limit", 5.0, ("inf", None), "disagree"),
        # No incumbent found.
        ("time_limit", float("inf"), ("opt", 100.0), "dnf"),
    ],
)
def test_classify(
    status: str, objective: float | None, ref: tuple[str, float | None], expected: str
) -> None:
    outcome = gate.Outcome(status, objective)
    assert gate.classify(outcome, gate.Reference(*ref)) == expected


def test_classify_limited_incumbent_respects_maximize() -> None:
    ref = gate.Reference("opt", 100.0)
    assert gate.classify(gate.Outcome("time_limit", 110.0), ref, True) == "disagree"
    assert gate.classify(gate.Outcome("time_limit", 90.0), ref, True) == "dnf"


@pytest.mark.parametrize(
    ("header", "maximize"),
    [
        ("NAME x\nROWS\n N obj\n", False),
        ("NAME x\nOBJSENSE\n    MAX\nROWS\n", True),
        ("NAME x\nOBJSENSE\n    MIN\nROWS\n", False),
        ("NAME x\nOBJSENSE MAXIMIZE\nROWS\n", True),
    ],
)
def test_read_maximize(tmp_path: Path, header: str, maximize: bool) -> None:
    path = tmp_path / "m.mps.gz"
    with gzip.open(path, "wt") as f:
        f.write(header)
    assert gate.read_maximize(path) is maximize


@pytest.mark.parametrize(
    ("expected", "observed", "tag"),
    [
        ("match_1e-6", "match_1e-6", "ok"),
        ("match_1e-6", "match_1e-4", "fail"),
        ("match_1e-4", "dnf", "fail"),
        ("dnf", "dnf", "ok"),
        ("dnf", "match_1e-4", "better"),
        ("match_1e-4", "match_1e-6", "better"),
        # A wrong answer fails even where the list already expects one.
        ("disagree", "disagree", "fail"),
        ("dnf", "disagree", "fail"),
        (None, "disagree", "fail"),
        (None, "dnf", "unlisted"),
    ],
)
def test_verdict(expected: str | None, observed: str, tag: str) -> None:
    assert gate.verdict(expected, observed) == tag


def test_parse_outcome_uses_the_summary_status_line() -> None:
    text = "Status: Infeasible\nStatus: Infeasible\nObjective: none\n"
    assert gate.parse_outcome(text) == gate.Outcome("infeasible", None)
    text = "Optimal: 1.0\nStatus: Optimal\nObjective: 1.5000000000e+00\n"
    assert gate.parse_outcome(text) == gate.Outcome("optimal", 1.5)
    assert gate.parse_outcome("Status: Time limit\n").status == "time_limit"
    assert gate.parse_outcome("segfault").status == "crash"


def test_read_solu_keeps_only_usable_references(tmp_path: Path) -> None:
    solu = tmp_path / "x.solu"
    solu.write_text("=opt= a 1.5\n=inf= b\n=best= c 3\n=unkn= d\n=unbd= e\n")
    refs = gate.read_solu(solu)
    assert refs == {
        "a": gate.Reference("opt", 1.5),
        "b": gate.Reference("inf"),
        "e": gate.Reference("unbd"),
    }


def fake_solver(tmp_path: Path, answers: dict[str, str]) -> Path:
    """A stand-in mipx-solve printing a canned answer per instance name."""
    lines = ["#!/bin/sh", 'case "$(basename "$1")" in']
    for name, answer in answers.items():
        lines.append(f"  {name}.mps) printf '{answer}' ;;")
    lines += ["  *) exit 3 ;;", "esac"]
    path = tmp_path / "fake-solve"
    path.write_text("\n".join(lines) + "\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def setup_suite(tmp_path: Path, expected_rows: str, header: bool = True) -> tuple[Path, Path]:
    data = tmp_path / "data"
    data.mkdir()
    for name in ("good", "slow", "wrong"):
        (data / f"{name}.mps").write_text("")
    (data / "miplib.solu").write_text("=opt= good 10\n=opt= slow 5\n=opt= wrong 7\n")
    expected = tmp_path / "expected.csv"
    head = "# comment\ninstance,backend,expected\n" if header else ""
    expected.write_text(head + expected_rows)
    return data, expected


def run_gate(tmp_path: Path, data: Path, expected: Path, *extra: str) -> int:
    solver = fake_solver(
        tmp_path,
        {
            "good": "Status: Optimal\\nObjective: 10\\n",
            "slow": "Status: Time limit\\nObjective: 6\\n",
            "wrong": "Status: Optimal\\nObjective: 8\\n",
        },
    )
    argv = ["mip", "--binary", str(solver), "--data-dir", str(data)]
    return int(gate.main([*argv, "--expected-file", str(expected), *extra]))


def test_gate_passes_when_results_match_the_list(tmp_path: Path) -> None:
    data, expected = setup_suite(tmp_path, "good,mip,match_1e-6\nslow,mip,dnf\n")
    assert run_gate(tmp_path, data, expected) == 0


def test_gate_fails_on_a_wrong_answer_even_if_listed(tmp_path: Path) -> None:
    data, expected = setup_suite(tmp_path, "good,mip,match_1e-6\nwrong,mip,disagree\n")
    assert run_gate(tmp_path, data, expected) == 1


def test_gate_fails_on_a_regression(tmp_path: Path) -> None:
    data, expected = setup_suite(tmp_path, "slow,mip,match_1e-4\n")
    assert run_gate(tmp_path, data, expected) == 1


def test_gate_fails_when_a_listed_instance_is_missing(tmp_path: Path) -> None:
    data, expected = setup_suite(tmp_path, "good,mip,match_1e-6\ngone,mip,match_1e-6\n")
    assert run_gate(tmp_path, data, expected) == 1


def test_gate_fails_when_a_listed_instance_has_no_reference(tmp_path: Path) -> None:
    data, expected = setup_suite(tmp_path, "good,mip,match_1e-6\n")
    (data / "miplib.solu").unlink()
    assert run_gate(tmp_path, data, expected) == 1


def test_gate_skips_loudly_without_instances(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    code = gate.main(["lp", "--data-dir", str(empty), "--binary", "/nonexistent"])
    assert code == gate.SKIP_EXIT
    assert "SKIPPED" in capsys.readouterr().err


def test_write_expected_regenerates_and_keeps_other_rows(tmp_path: Path) -> None:
    data, expected = setup_suite(tmp_path, "instance,backend,expected,time_limit\n", header=False)
    expected.write_text(
        "instance,backend,expected,time_limit\ngood,mip,dnf,5\nother,lpx,match_1e-6,\n"
    )
    # --all adds the unlisted instances; the wrong answer still fails the run.
    assert run_gate(tmp_path, data, expected, "--all", "--write-expected") == 1
    rows = expected.read_text().splitlines()
    assert rows[0].startswith("#")
    body = [r for r in rows if not r.startswith("#")]
    assert body[0] == "instance,backend,expected,time_limit"
    # The per-row limit survives the rewrite.
    assert set(body[1:]) == {
        "good,mip,match_1e-6,5",
        "other,lpx,match_1e-6,",
        "slow,mip,dnf,",
        "wrong,mip,disagree,",
    }
