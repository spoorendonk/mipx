#!/usr/bin/env python3
"""Fail if candidate benchmark results regress beyond configured tolerances.

Expected CSV columns:
  - instance
  - primary metric column (default: work_units)

Optional gate features:
  - aggregate mode: median or geometric mean
  - per-instance regression cap
  - status-aware filtering and status-match enforcement
  - secondary metric with warn/fail policy
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path


def parse_comma_set(raw: str) -> set[str]:
    return {x.strip().lower() for x in raw.split(",") if x.strip()}


def parse_float(raw: str) -> float | None:
    text = raw.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def load_rows(path: Path, required_columns: set[str]) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing CSV header")

        missing = required_columns - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{path}: missing columns: {sorted(missing)}")

        rows: dict[str, dict[str, str]] = {}
        for row in reader:
            name = row.get("instance", "").strip()
            if not name:
                continue
            rows[name] = row
        return rows


def aggregate_ratio(values: list[float], mode: str) -> float:
    if mode == "median":
        return float(statistics.median(values))
    if mode == "geomean":
        if any(v <= 0 for v in values):
            raise ValueError("geomean requires strictly positive ratio values")
        return float(statistics.geometric_mean(values))
    raise ValueError(f"unsupported aggregate mode: {mode}")


def format_percent(value: float) -> str:
    return f"{value:.2f}%"


def evaluate_metric(
    *,
    baseline: dict[str, dict[str, str]],
    candidate: dict[str, dict[str, str]],
    metric: str,
    aggregate_mode: str,
    max_regression_pct: float,
    max_instance_regression_pct: float,
    min_common_instances: int,
    status_column: str,
    required_statuses: set[str],
    require_status_match: bool,
) -> dict[str, object]:
    common = sorted(set(baseline) & set(candidate))
    if len(common) < min_common_instances:
        return {
            "ok": False,
            "fatal_error": (
                f"only {len(common)} common instances found; "
                f"need at least {min_common_instances}"
            ),
        }

    status_mismatches: list[str] = []
    skipped_status: list[str] = []
    skipped_metric: list[str] = []
    ratios: list[float] = []
    pair_ratios: list[tuple[str, float]] = []

    for name in common:
        b_row = baseline[name]
        c_row = candidate[name]

        b_status = b_row.get(status_column, "").strip().lower()
        c_status = c_row.get(status_column, "").strip().lower()

        if require_status_match and b_status and c_status and b_status != c_status:
            status_mismatches.append(name)
            continue

        if required_statuses and (b_status not in required_statuses or c_status not in required_statuses):
            skipped_status.append(name)
            continue

        b_val = parse_float(b_row.get(metric, ""))
        c_val = parse_float(c_row.get(metric, ""))
        if b_val is None or c_val is None:
            skipped_metric.append(name)
            continue

        if b_val < 0 or c_val < 0:
            skipped_metric.append(name)
            continue

        if b_val == 0:
            ratio = 1.0 if c_val == 0 else float("inf")
        else:
            ratio = c_val / b_val

        ratios.append(ratio)
        pair_ratios.append((name, ratio))

    if not ratios:
        return {
            "ok": False,
            "fatal_error": "no comparable rows after status/metric filtering",
            "common_instances": len(common),
            "status_mismatches": status_mismatches,
            "skipped_status": skipped_status,
            "skipped_metric": skipped_metric,
        }

    if len(ratios) < min_common_instances:
        return {
            "ok": False,
            "fatal_error": (
                f"only {len(ratios)} comparable instances after filtering; "
                f"need at least {min_common_instances}"
            ),
            "common_instances": len(common),
            "status_mismatches": status_mismatches,
            "skipped_status": skipped_status,
            "skipped_metric": skipped_metric,
        }

    agg_ratio = aggregate_ratio(ratios, aggregate_mode)
    agg_reg_pct = (agg_ratio - 1.0) * 100.0

    regressions = sorted(((n, r) for n, r in pair_ratios if r > 1.0), key=lambda x: x[1], reverse=True)
    improvements = sorted(((n, r) for n, r in pair_ratios if r < 1.0), key=lambda x: x[1])

    reasons: list[str] = []
    warnings: list[str] = []

    if require_status_match and status_mismatches:
        reasons.append(f"{len(status_mismatches)} status mismatch(es)")

    if agg_reg_pct > max_regression_pct:
        reasons.append(
            f"aggregate regression {format_percent(agg_reg_pct)} exceeds "
            f"allowed {format_percent(max_regression_pct)}"
        )

    instance_limit_ratio = None
    instance_offenders: list[tuple[str, float]] = []
    if max_instance_regression_pct >= 0:
        instance_limit_ratio = 1.0 + max_instance_regression_pct / 100.0
        instance_offenders = [(n, r) for n, r in regressions if r > instance_limit_ratio]
        if instance_offenders:
            reasons.append(
                f"{len(instance_offenders)} instance(s) exceed per-instance cap "
                f"{format_percent(max_instance_regression_pct)}"
            )

    top_regressions = [
        {"instance": n, "ratio": r, "regression_pct": (r - 1.0) * 100.0} for n, r in regressions[:5]
    ]
    top_improvements = [
        {"instance": n, "ratio": r, "improvement_pct": (1.0 - r) * 100.0} for n, r in improvements[:5]
    ]

    return {
        "ok": not reasons,
        "reasons": reasons,
        "warnings": warnings,
        "metric": metric,
        "aggregate_mode": aggregate_mode,
        "common_instances": len(common),
        "compared_instances": len(ratios),
        "aggregate_ratio": agg_ratio,
        "aggregate_regression_pct": agg_reg_pct,
        "allowed_regression_pct": max_regression_pct,
        "max_instance_regression_pct": max_instance_regression_pct,
        "status_mismatches": status_mismatches,
        "skipped_status": skipped_status,
        "skipped_metric": skipped_metric,
        "top_regressions": top_regressions,
        "top_improvements": top_improvements,
        "instance_offenders": [
            {"instance": n, "ratio": r, "regression_pct": (r - 1.0) * 100.0}
            for n, r in instance_offenders[:10]
        ],
        "instance_limit_ratio": instance_limit_ratio,
    }


def print_metric_report(result: dict[str, object], *, header: str) -> None:
    print(header)

    if fatal := result.get("fatal_error"):
        print(f"ERROR: {fatal}")
        return

    print(f"Common instances: {result['common_instances']}")
    print(f"Compared instances: {result['compared_instances']}")
    print(f"Aggregate mode: {result['aggregate_mode']}")
    print(f"Aggregate ratio (candidate/baseline): {result['aggregate_ratio']:.4f}")
    print(f"Aggregate regression: {result['aggregate_regression_pct']:.2f}%")
    print(f"Allowed aggregate regression: {result['allowed_regression_pct']:.2f}%")

    top_reg = result.get("top_regressions", [])
    if top_reg:
        worst = top_reg[0]
        print(
            "Worst single-instance regression: "
            f"{worst['instance']} ({worst['regression_pct']:.2f}%)"
        )

    status_mismatches = result.get("status_mismatches", [])
    if status_mismatches:
        print(f"Status mismatches: {len(status_mismatches)}")

    skipped_status = result.get("skipped_status", [])
    if skipped_status:
        print(f"Skipped by status filter: {len(skipped_status)}")

    skipped_metric = result.get("skipped_metric", [])
    if skipped_metric:
        print(f"Skipped by metric parse/validity: {len(skipped_metric)}")

    instance_offenders = result.get("instance_offenders", [])
    if instance_offenders:
        max_instance_pct = result.get("max_instance_regression_pct", -1.0)
        print(
            "Per-instance cap exceeded: "
            f"{len(instance_offenders)} instance(s) above {max_instance_pct:.2f}%"
        )

    for warning in result.get("warnings", []):
        print(f"WARN: {warning}")
    for reason in result.get("reasons", []):
        print(f"FAIL: {reason}")

    print("PASS" if result.get("ok", False) else "FAIL")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--metric", default="work_units")
    parser.add_argument(
        "--max-regression-pct",
        type=float,
        default=0.0,
        help="Allowed aggregate regression (percent) for the primary metric.",
    )
    parser.add_argument(
        "--max-instance-regression-pct",
        type=float,
        default=-1.0,
        help="Per-instance cap (percent). Negative disables this check.",
    )
    parser.add_argument(
        "--aggregate",
        choices=("median", "geomean"),
        default="median",
        help="Aggregate ratio mode for regression checks.",
    )
    parser.add_argument(
        "--min-common-instances",
        type=int,
        default=5,
        help="Fail if fewer than this many common/comparable instances are found.",
    )
    parser.add_argument("--status-column", default="status")
    parser.add_argument(
        "--required-statuses",
        default="",
        help="Comma-separated statuses required in both baseline/candidate rows to compare.",
    )
    parser.add_argument(
        "--require-status-match",
        action="store_true",
        help="Fail if baseline/candidate statuses differ on common rows.",
    )
    parser.add_argument(
        "--secondary-metric",
        default="",
        help="Optional secondary metric to track with warn/fail policy.",
    )
    parser.add_argument(
        "--max-secondary-regression-pct",
        type=float,
        default=0.0,
        help="Allowed aggregate regression (percent) for secondary metric.",
    )
    parser.add_argument(
        "--max-secondary-instance-regression-pct",
        type=float,
        default=-1.0,
        help="Per-instance cap (percent) for secondary metric; negative disables.",
    )
    parser.add_argument(
        "--secondary-mode",
        choices=("off", "warn", "fail"),
        default="off",
        help="Secondary metric policy. off=ignore, warn=report, fail=gate.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional output path for structured comparison summary JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    required_statuses = parse_comma_set(args.required_statuses)

    required_columns = {"instance", args.metric}
    if args.secondary_metric:
        required_columns.add(args.secondary_metric)
    if required_statuses or args.require_status_match:
        required_columns.add(args.status_column)

    try:
        baseline = load_rows(args.baseline, required_columns)
        candidate = load_rows(args.candidate, required_columns)
    except ValueError as e:
        print(f"ERROR: {e}")
        return 2

    primary = evaluate_metric(
        baseline=baseline,
        candidate=candidate,
        metric=args.metric,
        aggregate_mode=args.aggregate,
        max_regression_pct=args.max_regression_pct,
        max_instance_regression_pct=args.max_instance_regression_pct,
        min_common_instances=args.min_common_instances,
        status_column=args.status_column,
        required_statuses=required_statuses,
        require_status_match=args.require_status_match,
    )
    print_metric_report(primary, header=f"Primary metric: {args.metric}")

    fatal_error = primary.get("fatal_error")
    overall_ok = bool(primary.get("ok", False))
    warnings: list[str] = []
    secondary_result: dict[str, object] | None = None

    if fatal_error:
        overall_ok = False

    if args.secondary_metric and args.secondary_mode != "off":
        secondary_result = evaluate_metric(
            baseline=baseline,
            candidate=candidate,
            metric=args.secondary_metric,
            aggregate_mode=args.aggregate,
            max_regression_pct=args.max_secondary_regression_pct,
            max_instance_regression_pct=args.max_secondary_instance_regression_pct,
            min_common_instances=args.min_common_instances,
            status_column=args.status_column,
            required_statuses=required_statuses,
            require_status_match=args.require_status_match,
        )
        print_metric_report(secondary_result, header=f"Secondary metric: {args.secondary_metric}")

        if secondary_result.get("fatal_error"):
            if args.secondary_mode == "fail":
                overall_ok = False
            else:
                warnings.append(f"secondary metric error: {secondary_result['fatal_error']}")
        elif not secondary_result.get("ok", False):
            if args.secondary_mode == "fail":
                overall_ok = False
            else:
                warnings.extend(
                    f"secondary metric: {reason}" for reason in secondary_result.get("reasons", [])
                )

    for warning in warnings:
        print(f"WARN: {warning}")

    summary_payload = {
        "pass": overall_ok,
        "primary": primary,
        "secondary": secondary_result,
        "secondary_mode": args.secondary_mode,
        "warnings": warnings,
    }
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote summary JSON: {args.summary_json}")

    if overall_ok:
        print("PASS: regression gate satisfied")
        return 0

    print("FAIL: regression gate not satisfied")
    return 1


if __name__ == "__main__":
    sys.exit(main())
