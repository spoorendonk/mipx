from collections.abc import Callable
from pathlib import Path

import pytest

mipx = pytest.importorskip("mipx")


def tiny_mps_path() -> Path:
    return Path(__file__).resolve().parents[2] / "tests" / "data" / "tiny.mps"


def test_read_mps_has_expected_dimensions() -> None:
    model = mipx.read_mps(str(tiny_mps_path()))
    assert model.num_cols == 3
    assert model.num_rows == 3
    assert model.has_integers()


def test_solver_load_solve_flow() -> None:
    model = mipx.read_mps(str(tiny_mps_path()))
    solver = mipx.MipSolver()
    solver.set_verbose(False)
    solver.set_node_limit(256)
    solver.set_heuristic_mode(mipx.HeuristicRuntimeMode.Deterministic)
    solver.load(model)

    result = solver.solve()
    assert result.status == mipx.Status.Infeasible
    assert result.nodes >= 0


def test_solve_mps_helper_runs() -> None:
    result = mipx.solve_mps(
        str(tiny_mps_path()),
        node_limit=256,
        verbose=False,
        heuristic_mode=mipx.HeuristicRuntimeMode.Deterministic,
    )
    assert result.status == mipx.Status.Infeasible
    assert result.work_units >= 0.0


def data_path(name: str) -> Path:
    return Path(__file__).resolve().parents[2] / "tests" / "data" / name


def test_solve_file_reads_lp_format_by_extension() -> None:
    # Issue #214: this used to be parsed as MPS and reported Optimal 0.0.
    result = mipx.solve_file(str(data_path("lp_format.lp")), verbose=False)
    assert result.status == mipx.Status.Optimal
    assert result.objective == pytest.approx(3.0)


def test_solve_mps_alias_reads_lp_format_too() -> None:
    result = mipx.solve_mps(str(data_path("lp_format.lp")), verbose=False)
    assert result.status == mipx.Status.Optimal
    assert result.objective == pytest.approx(3.0)


def test_read_model_detects_and_forces_format() -> None:
    model = mipx.read_model(str(data_path("lp_format.lp")))
    assert (model.num_cols, model.num_rows) == (2, 1)
    assert model.sense == mipx.Sense.Maximize

    forced = mipx.read_model(str(data_path("lp_format_upper.txt")), format=mipx.ModelFormat.Lp)
    assert (forced.num_cols, forced.num_rows) == (2, 1)


def test_read_mps_unchanged_for_valid_mps() -> None:
    model = mipx.read_model(str(tiny_mps_path()), format=mipx.ModelFormat.Mps)
    assert (model.num_cols, model.num_rows) == (3, 3)
    result = mipx.solve_file(str(tiny_mps_path()), node_limit=256, verbose=False)
    assert result.status == mipx.Status.Infeasible


@pytest.mark.parametrize(
    "call",
    [
        lambda: mipx.read_mps(str(data_path("lp_format.lp"))),
        lambda: mipx.read_model(str(data_path("lp_format.lp")), format=mipx.ModelFormat.Mps),
        lambda: mipx.read_model(str(data_path("malformed.mps"))),
        lambda: mipx.read_lp(str(data_path("tiny.mps"))),
        lambda: mipx.solve_mps(str(data_path("malformed.mps"))),
    ],
)
def test_wrong_format_input_raises_model_format_error(call: Callable[[], object]) -> None:
    with pytest.raises(mipx.ModelFormatError) as excinfo:
        call()
    # The C++ message, which names the file, survives the translation.
    assert "Cannot read '" in str(excinfo.value)
    assert issubclass(mipx.ModelFormatError, ValueError)


def test_malformed_lp_bounds_line_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad_bounds.lp"
    path.write_text("Minimize\n obj: x\nSubject To\n c1: x >= 0\nBounds\n x =~ 3\nEnd\n")
    with pytest.raises(mipx.ModelFormatError, match="line 6"):
        mipx.read_lp(str(path))


def test_missing_file_stays_runtime_error() -> None:
    with pytest.raises(RuntimeError):
        mipx.read_model(str(data_path("does_not_exist.mps")))
