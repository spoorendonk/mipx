import pytest
from run_instance_gate import Reference
from run_mip_conflict_bench import (
    TERMINAL_WORK,
    check_reference,
    compare_anchor,
    parse_rapid_metrics,
)


@pytest.mark.parametrize(
    ("status", "sense", "objective", "root", "bound", "reference", "expected"),
    [
        ("Optimal", "min", "10", "8", "10", Reference("opt", 10), "match"),
        ("Node limit", "max", "8", "12", "11", Reference("opt", 10), "consistent limit"),
        ("Time limit", "min", None, None, "-inf", Reference("opt", 10), "consistent limit"),
        (
            "Node limit",
            "min",
            "12",
            "8",
            "11",
            Reference("opt", 10),
            "dual bound excludes reference optimum",
        ),
        (
            "Node limit",
            "max",
            "8",
            "9",
            "11",
            Reference("opt", 10),
            "dual bound excludes reference optimum",
        ),
        (
            "Time limit",
            "max",
            "11",
            "12",
            "12",
            Reference("opt", 10),
            "incumbent improves on reference optimum",
        ),
        ("Infeasible", "min", None, "8", "inf", Reference("inf"), "match"),
        (
            "Node limit",
            "min",
            "12",
            "8",
            "9",
            Reference("inf"),
            "incumbent on reference-infeasible instance",
        ),
    ],
)
def test_reference_check(
    status: str,
    sense: str,
    objective: str | None,
    root: str | None,
    bound: str,
    reference: Reference,
    expected: str,
) -> None:
    assert (
        check_reference(
            {
                "status": status,
                "sense": sense,
                "objective": objective,
                "root_bound": root,
                "final_bound": bound,
            },
            reference,
        )
        == expected
    )


def test_rapid_telemetry_keeps_first_incumbent_and_restart_spacing() -> None:
    log = """Rapid bench attempt: start
Rapid bench phase: name=rapid_learning seconds=0.001
Rapid bench incumbent: source=rounding seconds=0.1 value=22
Rapid bench bound: node=128 lp_iters=162 seconds=0.12 value=19
Rapid bench restart: node=128 conflict=1 spacing=127 required=96 open=50
Root rapid learning: bounds=8 tree_cutoffs=4
Rapid bench summary: restarts=1 dropped=0 seconds=0.14
"""
    parsed = parse_rapid_metrics(log, 1, ["--root-rapid-learning"])
    assert parsed["solver_seconds"] == 0.14
    assert parsed["rapid_attempts"] == [
        {
            "bounds": [{"node": "128", "lp_iters": "162", "seconds": "0.12", "value": "19"}],
            "restarts": [{"node": 128, "conflict": 1, "spacing": 127, "required": 96, "open": 50}],
            "phases_seconds": {"rapid_learning": 0.001},
            "first_incumbent": {"source": "rounding", "seconds": "0.1", "value": "22"},
        }
    ]
    with pytest.raises(ValueError, match="restart trace"):
        parse_rapid_metrics(log, 0, [])
    with pytest.raises(ValueError, match="restart spacing"):
        parse_rapid_metrics(log.replace("required=96", "required=192"), 1, [])


def test_anchor_distinguishes_outcome_mismatches_from_wall_dependent_work() -> None:
    anchor: dict[str, object] = dict.fromkeys(TERMINAL_WORK, 0)
    anchor.update(
        check="match",
        status="Optimal",
        objective="8",
        final_bound="8",
        root_bound="7",
        sense="min",
        input_continuous_cols=0,
        learning_and_allowance={},
        rapid_attempts=[],
    )
    assert compare_anchor(anchor, anchor.copy(), "baseline")
    changed = {**anchor, "nodes": 3}
    assert not compare_anchor(anchor, changed, "baseline")
    assert compare_anchor(anchor, changed, "root-allowance")
    assert changed["anchor_check"] == "wall-dependent counters differ: nodes"
    assert not compare_anchor(anchor, {**changed, "final_bound": "9"}, "root-allowance")
