"""Tests for run_instance_gate.py: classification, list handling, exit codes."""

from __future__ import annotations

import argparse
import importlib.util
import stat
import subprocess
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
        # A MIP stopped by a limit is dnf whatever it reports as objective:
        # with no incumbent the CLI prints 0, which is not a wrong answer.
        ("time_limit", 0.0, ("opt", 100.0), "dnf"),
        ("node_limit", 0.0, ("inf", None), "dnf"),
    ],
)
def test_classify(
    status: str, objective: float | None, ref: tuple[str, float | None], expected: str
) -> None:
    outcome = gate.Outcome(status, objective)
    assert gate.classify(outcome, gate.Reference(*ref)) == expected


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


@pytest.mark.parametrize(
    ("backend", "ne"), [("barrier-dense", "dense"), ("barrier-sparse", "sparse")]
)
def test_barrier_backends_run_the_cpu_barrier_on_one_ne_backend(backend: str, ne: str) -> None:
    args = gate.solver_args(backend, 10.0)
    assert args[args.index("--barrier-ne") + 1] == ne
    assert "--barrier" in args
    assert "--no-gpu" in args
    assert f"--{backend}" not in args


def test_barrier_backends_are_selectable_only_on_the_barrier_suite() -> None:
    def pick(suite: str, backends: str) -> list[str]:
        args = argparse.Namespace(suite=suite, backends=backends)
        result: list[str] = gate.selected_backends(args, gate.SUITES[suite])
        return result

    assert pick("barrier", "barrier-dense") == ["barrier-dense"]
    assert pick("barrier", "") == list(gate.SUITES["barrier"].backends)
    with pytest.raises(SystemExit, match="barrier-dense"):
        pick("lp", "barrier-dense")


def test_cg_gate_runs_both_barrier_backends_without_cli_presolve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "grid1.mps").touch()
    (tmp_path / "cg_masters.solu").write_text("=opt= grid1 10\n")
    expected = tmp_path / "expected.csv"
    expected.write_text(
        "instance,backend,expected\ngrid1,barrier-dense,match_1e-6\n"
        "grid1,barrier-sparse,match_1e-6\n"
    )
    factors: list[str] = []

    def solve(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert "--no-presolve" in cmd
        factors.append(cmd[cmd.index("--barrier-ne") + 1])
        return subprocess.CompletedProcess(cmd, 0, "Status: Optimal\nObjective: 10\n", "")

    monkeypatch.setattr(gate.subprocess, "run", solve)
    argv = [
        "barrier-cg",
        "--binary",
        sys.executable,
        "--data-dir",
        str(tmp_path),
        "--expected-file",
        str(expected),
    ]
    assert gate.main(argv) == 0
    assert factors == ["dense", "sparse"]


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


def test_gate_fails_when_nothing_is_gated(tmp_path: Path) -> None:
    data, expected = setup_suite(tmp_path, "")
    assert run_gate(tmp_path, data, expected) == 1
    assert run_gate(tmp_path, data, tmp_path / "no-such-list.csv") == 1


def test_gate_skips_loudly_without_instances(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    code = gate.main(["lp", "--data-dir", str(empty), "--binary", "/nonexistent"])
    assert code == gate.SKIP_EXIT
    assert "SKIPPED" in capsys.readouterr().err


def test_gate_skip_names_the_listed_instances_it_did_not_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _, expected = setup_suite(tmp_path, "good,mip,match_1e-6\nslow,mip,dnf\n")
    empty = tmp_path / "empty"
    empty.mkdir()
    argv = ["mip", "--data-dir", str(empty), "--binary", "/nonexistent"]
    code = gate.main([*argv, "--expected-file", str(expected)])
    assert code == gate.SKIP_EXIT
    assert "not run: good, slow." in capsys.readouterr().err

    (tmp_path / "lp").mkdir()
    _, lp_list = setup_suite(tmp_path / "lp", "good,dual,match_1e-6\nslow,pdlp,dnf\n")
    argv = ["lp", "--data-dir", str(empty), "--binary", "/nonexistent", "--backends", "dual"]
    assert gate.main([*argv, "--expected-file", str(lp_list)]) == gate.SKIP_EXIT
    assert "not run: good." in capsys.readouterr().err


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
