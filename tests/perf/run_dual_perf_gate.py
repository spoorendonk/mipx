#!/usr/bin/env python3
"""Run dual-simplex LP regression gate (Netlib anchors + Mittelman curated)."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
PERF_DIR = ROOT_DIR / "tests" / "perf"


def normalize_solver_arg_tokens(argv: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == "--solver-arg":
            if i + 1 >= len(argv):
                raise SystemExit("--solver-arg requires one argument")
            out.append(f"--solver-arg={argv[i + 1]}")
            i += 2
            continue
        out.append(argv[i])
        i += 1
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate-binary", required=True)
    p.add_argument("--baseline-binary", default="")
    p.add_argument("--out-dir", default="/tmp/mipx_dual_perf_gate")

    p.add_argument("--netlib-dir", default=str(ROOT_DIR / "tests" / "data" / "netlib"))
    p.add_argument("--mittelman-dir", default=str(ROOT_DIR / "tests" / "data" / "mittelman_lp"))
    p.add_argument("--netlib-corpus", default=str(PERF_DIR / "netlib_dual_corpus.csv"))
    p.add_argument("--mittelman-corpus", default=str(PERF_DIR / "mittelman_dual_corpus.csv"))

    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--time-limit", type=float, default=900.0)
    p.add_argument("--aggregate", choices=("median", "geomean"), default="median")
    p.add_argument("--netlib-min-common", type=int, default=5)
    p.add_argument("--mittelman-min-common", type=int, default=5)

    p.add_argument("--max-work-regression-pct", type=float, default=0.0)
    p.add_argument("--max-work-instance-regression-pct", type=float, default=20.0)
    p.add_argument("--time-regression-mode", choices=("off", "warn", "fail"), default="warn")
    p.add_argument("--max-time-regression-pct", type=float, default=10.0)
    p.add_argument("--max-time-instance-regression-pct", type=float, default=-1.0)

    p.add_argument("--candidate-netlib-csv", default="")
    p.add_argument("--candidate-mittelman-csv", default="")
    p.add_argument("--baseline-netlib-csv", default="")
    p.add_argument("--baseline-mittelman-csv", default="")

    p.add_argument("--correctness-mode", choices=("off", "warn", "fail"), default="fail")
    p.add_argument("--correctness-corpus", default=str(PERF_DIR / "netlib_dual_corpus.csv"))
    p.add_argument("--correctness-time-limit", type=float, default=60.0)
    p.add_argument("--correctness-objective-rel-tol", type=float, default=1e-6)
    p.add_argument("--correctness-repeats", type=int, default=1)
    p.add_argument("--highs-binary", default="")

    p.add_argument("--summary-md", default="")
    p.add_argument("--solver-arg", action="append", default=[])
    return p.parse_args(normalize_solver_arg_tokens(sys.argv[1:]))


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(cmd))
    return subprocess.run(cmd, check=check, text=True)


def load_corpus_instances(path: Path) -> list[str]:
    if not path.is_file():
        raise SystemExit(f"corpus CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    instances: list[str] = []
    for row in rows:
        name = row.get("instance", "").strip()
        if name:
            instances.append(name)
    if not instances:
        raise SystemExit(f"corpus CSV has no instances: {path}")
    return instances


def ensure_executable(path: Path, label: str) -> None:
    if not path.is_file() or not os.access(path, os.X_OK):
        raise SystemExit(f"{label} not executable: {path}")


def solver_args_tokens(extra_args: list[str]) -> list[str]:
    # Dual perf gate always runs deterministic dual-simplex policy with presolve off.
    fixed = ["--dual", "--no-presolve", "--quiet"]
    merged = [*fixed, *extra_args]
    out: list[str] = []
    for arg in merged:
        out.extend(["--solver-arg", arg])
    return out


def run_lp_bench(
    *,
    runner: Path,
    binary: Path,
    output: Path,
    instances: list[str],
    netlib_dir: Path,
    mittelman_dir: Path,
    repeats: int,
    time_limit: float,
    extra_solver_args: list[str],
    is_mittelman: bool,
) -> None:
    cmd = [
        sys.executable,
        str(runner),
        "--binary",
        str(binary),
        "--output",
        str(output),
        "--repeats",
        str(repeats),
        "--instances",
        ",".join(instances),
        *solver_args_tokens(extra_solver_args),
    ]
    if is_mittelman:
        cmd.extend(
            [
                "--mittelman-dir",
                str(mittelman_dir),
                "--netlib-dir",
                str(netlib_dir),
                "--time-limit",
                f"{time_limit:g}",
            ]
        )
    else:
        cmd.extend(
            [
                "--netlib-dir",
                str(netlib_dir),
                "--time-limit",
                f"{time_limit:g}",
            ]
        )
    run(cmd)


def run_check(
    *,
    check_script: Path,
    baseline_csv: Path,
    candidate_csv: Path,
    min_common: int,
    aggregate: str,
    max_work_regression_pct: float,
    max_work_instance_regression_pct: float,
    time_regression_mode: str,
    max_time_regression_pct: float,
    max_time_instance_regression_pct: float,
    summary_json: Path,
) -> int:
    cmd = [
        sys.executable,
        str(check_script),
        "--baseline",
        str(baseline_csv),
        "--candidate",
        str(candidate_csv),
        "--metric",
        "work_units",
        "--max-regression-pct",
        f"{max_work_regression_pct:g}",
        "--max-instance-regression-pct",
        f"{max_work_instance_regression_pct:g}",
        "--aggregate",
        aggregate,
        "--min-common-instances",
        str(min_common),
        "--status-column",
        "status",
        "--required-statuses",
        "optimal",
        "--require-status-match",
        "--secondary-metric",
        "time_seconds",
        "--max-secondary-regression-pct",
        f"{max_time_regression_pct:g}",
        "--max-secondary-instance-regression-pct",
        f"{max_time_instance_regression_pct:g}",
        "--secondary-mode",
        time_regression_mode,
        "--summary-json",
        str(summary_json),
    ]
    proc = run(cmd, check=False)
    return proc.returncode


def run_correctness_precheck(
    *,
    candidate_binary: Path,
    netlib_dir: Path,
    correctness_script: Path,
    correctness_corpus: Path,
    out_dir: Path,
    instances: list[str],
    time_limit: float,
    objective_rel_tol: float,
    repeats: int,
    highs_binary: str,
) -> tuple[int, Path, Path]:
    csv_out = out_dir / "candidate_correctness.csv"
    summary_out = out_dir / "candidate_correctness_summary.md"
    cmd = [
        sys.executable,
        str(correctness_script),
        "--mipx-binary",
        str(candidate_binary),
        "--netlib-dir",
        str(netlib_dir),
        "--corpus",
        str(correctness_corpus),
        "--output",
        str(csv_out),
        "--summary",
        str(summary_out),
        "--instances",
        ",".join(instances),
        "--time-limit",
        f"{time_limit:g}",
        "--threads",
        "1",
        "--objective-rel-tol",
        f"{objective_rel_tol:g}",
        "--disable-presolve",
        "--repeats",
        str(repeats),
    ]
    if highs_binary:
        cmd.extend(["--highs-binary", highs_binary])
    proc = run(cmd, check=False)
    return proc.returncode, csv_out, summary_out


def render_top_table(items: list[dict[str, object]], *, kind: str) -> list[str]:
    if not items:
        return ["No rows."]
    lines = ["| Instance | Delta |", "|---|---:|"]
    for row in items:
        name = str(row.get("instance", ""))
        if kind == "regression":
            delta = float(row.get("regression_pct", 0.0))
            lines.append(f"| `{name}` | +{delta:.2f}% |")
        else:
            delta = float(row.get("improvement_pct", 0.0))
            lines.append(f"| `{name}` | -{delta:.2f}% |")
    return lines


def render_gate_section(
    *,
    title: str,
    summary: dict[str, object],
) -> list[str]:
    primary = summary.get("primary") or {}
    secondary = summary.get("secondary") or {}
    status = "PASS" if summary.get("pass", False) else "FAIL"

    lines = [f"## {title}", f"- Gate: **{status}**"]

    if isinstance(primary, dict) and not primary.get("fatal_error"):
        lines.extend(
            [
                f"- Common rows: {int(primary.get('common_instances', 0))}",
                f"- Compared rows: {int(primary.get('compared_instances', 0))}",
                (
                    "- Work units aggregate regression: "
                    f"{float(primary.get('aggregate_regression_pct', 0.0)):.2f}%"
                ),
                (
                    "- Work units allowed aggregate regression: "
                    f"{float(primary.get('allowed_regression_pct', 0.0)):.2f}%"
                ),
            ]
        )
    else:
        lines.append(f"- Primary error: {primary.get('fatal_error', 'unknown')}")

    if isinstance(secondary, dict) and secondary:
        if secondary.get("fatal_error"):
            lines.append(f"- Time metric error: {secondary['fatal_error']}")
        else:
            lines.append(
                "- Time aggregate regression: "
                f"{float(secondary.get('aggregate_regression_pct', 0.0)):.2f}%"
            )

    warnings = summary.get("warnings", [])
    if isinstance(warnings, list) and warnings:
        lines.append("- Warnings:")
        for warning in warnings[:10]:
            lines.append(f"  - {warning}")

    reasons: list[str] = []
    if isinstance(primary, dict):
        reasons.extend(str(x) for x in primary.get("reasons", []))
    if isinstance(secondary, dict):
        reasons.extend(str(x) for x in secondary.get("reasons", []))
    if reasons:
        lines.append("- Reasons:")
        for reason in reasons[:10]:
            lines.append(f"  - {reason}")

    if isinstance(primary, dict):
        lines.append("### Worst Work-Unit Regressions")
        lines.extend(render_top_table(primary.get("top_regressions", []), kind="regression"))
        lines.append("### Top Work-Unit Improvements")
        lines.extend(render_top_table(primary.get("top_improvements", []), kind="improvement"))

        skipped = []
        skipped.extend(primary.get("status_mismatches", []))
        skipped.extend(primary.get("skipped_status", []))
        skipped.extend(primary.get("skipped_metric", []))
        if skipped:
            lines.append("### Skipped/Mismatch Rows")
            for name in sorted({str(x) for x in skipped})[:20]:
                lines.append(f"- `{name}`")

    lines.append("")
    return lines


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    netlib_dir = Path(args.netlib_dir)
    mittelman_dir = Path(args.mittelman_dir)
    netlib_corpus = Path(args.netlib_corpus)
    mittelman_corpus = Path(args.mittelman_corpus)

    netlib_instances = load_corpus_instances(netlib_corpus)
    mittelman_instances = load_corpus_instances(mittelman_corpus)

    run_netlib = PERF_DIR / "run_netlib_lp_bench.py"
    run_mittelman = PERF_DIR / "run_mittelman_lp_bench.py"
    check_script = PERF_DIR / "check_regression.py"
    correctness_script = PERF_DIR / "run_dual_correctness_investigation.py"

    cand_bin = Path(args.candidate_binary)
    base_bin = Path(args.baseline_binary) if args.baseline_binary else None
    correctness_corpus = Path(args.correctness_corpus)

    cand_netlib_csv = Path(args.candidate_netlib_csv) if args.candidate_netlib_csv else out_dir / "candidate_netlib.csv"
    cand_mittelman_csv = (
        Path(args.candidate_mittelman_csv) if args.candidate_mittelman_csv else out_dir / "candidate_mittelman.csv"
    )
    base_netlib_csv = Path(args.baseline_netlib_csv) if args.baseline_netlib_csv else out_dir / "baseline_netlib.csv"
    base_mittelman_csv = (
        Path(args.baseline_mittelman_csv) if args.baseline_mittelman_csv else out_dir / "baseline_mittelman.csv"
    )

    need_candidate_netlib = not args.candidate_netlib_csv
    need_candidate_mittelman = not args.candidate_mittelman_csv
    if need_candidate_netlib or need_candidate_mittelman:
        ensure_executable(cand_bin, "candidate binary")

    correctness_status = "not_run"
    correctness_csv: Path | None = None
    correctness_summary: Path | None = None
    if need_candidate_netlib and args.correctness_mode != "off":
        print("[dual-gate] Running candidate correctness precheck...")
        corr_rc, corr_csv, corr_summary = run_correctness_precheck(
            candidate_binary=cand_bin,
            netlib_dir=netlib_dir,
            correctness_script=correctness_script,
            correctness_corpus=correctness_corpus,
            out_dir=out_dir,
            instances=netlib_instances,
            time_limit=args.correctness_time_limit,
            objective_rel_tol=args.correctness_objective_rel_tol,
            repeats=args.correctness_repeats,
            highs_binary=args.highs_binary,
        )
        correctness_csv = corr_csv
        correctness_summary = corr_summary
        if corr_rc == 0:
            correctness_status = "pass"
        elif args.correctness_mode == "warn":
            correctness_status = "warn"
            print("[dual-gate] WARN: candidate correctness precheck failed; continuing (--correctness-mode warn)")
        else:
            correctness_status = "fail"
            print("[dual-gate] FAIL: candidate correctness precheck failed")
            print(f"[dual-gate] Correctness CSV: {corr_csv}")
            print(f"[dual-gate] Correctness summary: {corr_summary}")
            return 1
    elif args.correctness_mode == "off":
        correctness_status = "off"

    if need_candidate_netlib:
        run_lp_bench(
            runner=run_netlib,
            binary=cand_bin,
            output=cand_netlib_csv,
            instances=netlib_instances,
            netlib_dir=netlib_dir,
            mittelman_dir=mittelman_dir,
            repeats=args.repeats,
            time_limit=args.time_limit,
            extra_solver_args=args.solver_arg,
            is_mittelman=False,
        )

    if need_candidate_mittelman:
        run_lp_bench(
            runner=run_mittelman,
            binary=cand_bin,
            output=cand_mittelman_csv,
            instances=mittelman_instances,
            netlib_dir=netlib_dir,
            mittelman_dir=mittelman_dir,
            repeats=args.repeats,
            time_limit=args.time_limit,
            extra_solver_args=args.solver_arg,
            is_mittelman=True,
        )

    need_baseline_netlib = not args.baseline_netlib_csv
    need_baseline_mittelman = not args.baseline_mittelman_csv
    if need_baseline_netlib or need_baseline_mittelman:
        if base_bin is None:
            raise SystemExit(
                "Need --baseline-binary when baseline CSVs are not fully provided "
                "(--baseline-netlib-csv and --baseline-mittelman-csv)."
            )
        ensure_executable(base_bin, "baseline binary")

    if need_baseline_netlib:
        run_lp_bench(
            runner=run_netlib,
            binary=base_bin,
            output=base_netlib_csv,
            instances=netlib_instances,
            netlib_dir=netlib_dir,
            mittelman_dir=mittelman_dir,
            repeats=args.repeats,
            time_limit=args.time_limit,
            extra_solver_args=args.solver_arg,
            is_mittelman=False,
        )

    if need_baseline_mittelman:
        run_lp_bench(
            runner=run_mittelman,
            binary=base_bin,
            output=base_mittelman_csv,
            instances=mittelman_instances,
            netlib_dir=netlib_dir,
            mittelman_dir=mittelman_dir,
            repeats=args.repeats,
            time_limit=args.time_limit,
            extra_solver_args=args.solver_arg,
            is_mittelman=True,
        )

    netlib_summary_json = out_dir / "netlib_regression_summary.json"
    mittelman_summary_json = out_dir / "mittelman_regression_summary.json"

    print("[dual-gate] Checking Netlib anchor regression...")
    netlib_rc = run_check(
        check_script=check_script,
        baseline_csv=base_netlib_csv,
        candidate_csv=cand_netlib_csv,
        min_common=args.netlib_min_common,
        aggregate=args.aggregate,
        max_work_regression_pct=args.max_work_regression_pct,
        max_work_instance_regression_pct=args.max_work_instance_regression_pct,
        time_regression_mode=args.time_regression_mode,
        max_time_regression_pct=args.max_time_regression_pct,
        max_time_instance_regression_pct=args.max_time_instance_regression_pct,
        summary_json=netlib_summary_json,
    )

    print("[dual-gate] Checking Mittelman curated regression...")
    mittelman_rc = run_check(
        check_script=check_script,
        baseline_csv=base_mittelman_csv,
        candidate_csv=cand_mittelman_csv,
        min_common=args.mittelman_min_common,
        aggregate=args.aggregate,
        max_work_regression_pct=args.max_work_regression_pct,
        max_work_instance_regression_pct=args.max_work_instance_regression_pct,
        time_regression_mode=args.time_regression_mode,
        max_time_regression_pct=args.max_time_regression_pct,
        max_time_instance_regression_pct=args.max_time_instance_regression_pct,
        summary_json=mittelman_summary_json,
    )

    netlib_summary = json.loads(netlib_summary_json.read_text(encoding="utf-8"))
    mittelman_summary = json.loads(mittelman_summary_json.read_text(encoding="utf-8"))

    summary_md = Path(args.summary_md) if args.summary_md else out_dir / "dual_perf_summary.md"
    md_lines = [
        "# Dual Simplex Performance Gate",
        "",
        "## Configuration",
        f"- Candidate netlib CSV: `{cand_netlib_csv}`",
        f"- Candidate mittelman CSV: `{cand_mittelman_csv}`",
        f"- Baseline netlib CSV: `{base_netlib_csv}`",
        f"- Baseline mittelman CSV: `{base_mittelman_csv}`",
        f"- Aggregate mode: `{args.aggregate}`",
        f"- Primary gate metric: `work_units` (allowed aggregate regression: {args.max_work_regression_pct:.2f}%)",
        f"- Primary per-instance cap: {args.max_work_instance_regression_pct:.2f}%",
        (
            "- Secondary metric: `time_seconds` "
            f"({args.time_regression_mode}, allowed aggregate regression: {args.max_time_regression_pct:.2f}%)"
        ),
        f"- Correctness precheck mode: `{args.correctness_mode}`",
        f"- Correctness precheck status: `{correctness_status}`",
        (
            f"- Correctness summary: `{correctness_summary}`"
            if correctness_summary is not None
            else "- Correctness summary: n/a"
        ),
        (
            f"- Correctness CSV: `{correctness_csv}`"
            if correctness_csv is not None
            else "- Correctness CSV: n/a"
        ),
        "",
    ]
    md_lines.extend(render_gate_section(title="Netlib Anchors", summary=netlib_summary))
    md_lines.extend(render_gate_section(title="Mittelman Curated", summary=mittelman_summary))
    summary_md.write_text("\n".join(md_lines).rstrip() + "\n", encoding="utf-8")

    print(f"[dual-gate] Wrote summary markdown: {summary_md}")
    print(f"[dual-gate] Netlib summary JSON: {netlib_summary_json}")
    print(f"[dual-gate] Mittelman summary JSON: {mittelman_summary_json}")

    gate_passed = netlib_rc == 0 and mittelman_rc == 0
    print("[dual-gate] PASS" if gate_passed else "[dual-gate] FAIL")
    return 0 if gate_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
