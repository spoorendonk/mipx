from pathlib import Path

import mipx


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
