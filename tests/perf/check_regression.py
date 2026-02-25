#!/usr/bin/env python3
"""Fail if candidate benchmark results regress beyond a tolerance.

Expected CSV columns:
  - instance
  - <metric> (default: time_seconds)
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path


def load_metric(path: Path, metric: str) -> dict[str, float]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing CSV header")
        required = {"instance", metric}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{path}: missing columns: {sorted(missing)}")

        values: dict[str, float] = {}
        for row in reader:
            name = row["instance"].strip()
            if not name:
                continue
            raw = row[metric].strip()
            if not raw:
                continue
            values[name] = float(raw)
        return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--metric", default="time_seconds")
    parser.add_argument(
        "--max-regression-pct",
        type=float,
        default=2.0,
        help="Allowed median regression in percent (default: 2.0).",
    )
    parser.add_argument(
        "--min-common-instances",
        type=int,
        default=5,
        help="Fail if fewer than this many common instances are found.",
    )
    args = parser.parse_args()

    baseline = load_metric(args.baseline, args.metric)
    candidate = load_metric(args.candidate, args.metric)

    common = sorted(set(baseline) & set(candidate))
    if len(common) < args.min_common_instances:
        print(
            f"ERROR: only {len(common)} common instances found; "
            f"need at least {args.min_common_instances}"
        )
        return 2

    ratios = []
    regressions = []
    for name in common:
        b = baseline[name]
        c = candidate[name]
        if b <= 0:
            continue
        ratio = c / b
        ratios.append(ratio)
        if ratio > 1.0:
            regressions.append((name, ratio))

    if not ratios:
        print("ERROR: no valid positive baseline values to compare")
        return 2

    median_ratio = statistics.median(ratios)
    median_reg_pct = (median_ratio - 1.0) * 100.0
    allowed = args.max_regression_pct

    print(f"Compared metric: {args.metric}")
    print(f"Common instances: {len(common)}")
    print(f"Median ratio (candidate/baseline): {median_ratio:.4f}")
    print(f"Median regression: {median_reg_pct:.2f}%")
    print(f"Allowed regression: {allowed:.2f}%")

    if regressions:
        worst_name, worst_ratio = max(regressions, key=lambda x: x[1])
        print(
            f"Worst single-instance regression: {worst_name} "
            f"({(worst_ratio - 1.0) * 100.0:.2f}%)"
        )

    if median_reg_pct > allowed:
        print("FAIL: regression gate not satisfied")
        return 1

    print("PASS: regression gate satisfied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
