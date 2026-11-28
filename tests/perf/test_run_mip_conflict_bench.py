import pytest
from run_instance_gate import Reference
from run_mip_conflict_bench import check_reference


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
