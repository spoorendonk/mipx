from collections.abc import Callable
from pathlib import Path

# A plain import, not pytest.importorskip: a missing mipx skipped the module,
# pytest then exited 5 (no tests collected), and the gate's test block passes
# exit 5, so a tree whose bindings never got installed passed it (#255).
import mipx
import pytest


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


def test_sos2_adjacent_intervals_can_be_selected_before_load() -> None:
    path = str(data_path("mps_sos.mps"))
    model = mipx.read_mps(path)
    solver = mipx.MipSolver()
    solver.set_verbose(False)
    solver.load(model)
    legacy = solver.solve()
    assert legacy.status == mipx.Status.Optimal
    assert legacy.objective == pytest.approx(4.0)

    solver.set_sos2_adjacent_intervals(True)
    solver.load(model)
    for result in (
        solver.solve(),
        mipx.solve_file(path, sos2_adjacent_intervals=True),
        mipx.solve_mps(path, sos2_adjacent_intervals=True),
    ):
        assert result.status == mipx.Status.Optimal
        assert result.objective == pytest.approx(4.0)
        assert len(result.solution) == len(legacy.solution) - 1


def test_native_sos2_can_be_selected_before_load() -> None:
    path = str(data_path("mps_sos.mps"))
    model = mipx.read_mps(path)
    solver = mipx.MipSolver()
    solver.set_verbose(False)
    solver.set_sos2_native_branching(True)
    solver.load(model)
    for result in (
        solver.solve(),
        mipx.solve_file(path, sos2_native_branching=True),
        mipx.solve_mps(path, sos2_native_branching=True),
        mipx.solve_file(path, sos2_native_branching=True, sos2_propagation=True),
    ):
        assert result.status == mipx.Status.Optimal
        assert result.objective == pytest.approx(4.0)
        assert len(result.solution) == model.num_cols + 3  # SOS1 selectors
    solver.set_sos2_propagation(True)
    assert solver.solve().objective == pytest.approx(4.0)


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


def test_root_feasibility_jump_pass_limit() -> None:
    solver = mipx.MipSolver()
    solver.set_verbose(False)
    solver.load(mipx.read_mps(str(data_path("cli_features.mps"))))
    for passes in (0, 1, 4):
        solver.set_root_feasibility_jump_restarts(passes)
        result = solver.solve()
        assert result.status == mipx.Status.Optimal
        assert result.objective == pytest.approx(-22.0)
    for passes in (-1, 5):
        with pytest.raises(ValueError, match="Root feasibility-jump passes"):
            solver.set_root_feasibility_jump_restarts(passes)


def test_root_cut_propagation_can_be_enabled(tmp_path: Path) -> None:
    path = tmp_path / "root_cut.lp"
    path.write_text(
        "Minimize\n obj: x\nSubject To\n r1: x + y >= 1\n r2: x - y >= 0\n"
        "Binary\n x y\nEnd\n"
    )
    solver = mipx.MipSolver()
    solver.set_verbose(False)
    solver.set_presolve(False)
    solver.set_root_cut_propagation_enabled(True)
    solver.load(mipx.read_lp(str(path)))
    result = solver.solve()
    assert result.status == mipx.Status.Optimal
    assert result.objective == pytest.approx(1.0)


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
