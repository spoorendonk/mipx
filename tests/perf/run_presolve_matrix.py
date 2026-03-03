#!/usr/bin/env python3
"""Run a presolve consistency/performance matrix against HiGHS.

This driver orchestrates repeated calls to `run_presolve_compare.py` and
produces:
- `presolve_matrix_detail.csv`: one row per experiment repeat
- `presolve_matrix_summary.csv`: one aggregated row per experiment
- `presolve_matrix_summary.md`: human-readable summary table

Profiles:
- `ci-smoke`: lightweight, intended for optional CI smoke validation
- `internal`: broader matrix for local optimization work
"""

from __future__ import annotations

import argparse
import csv
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
PERF_DIR = ROOT_DIR / "tests" / "perf"
RUN_COMPARE = PERF_DIR / "run_presolve_compare.py"


def parse_csv_tokens(raw: str) -> list[str]:
    return [tok.strip() for tok in raw.split(",") if tok.strip()]


def parse_int_csv(raw: str, flag: str) -> list[int]:
    out: list[int] = []
    for tok in parse_csv_tokens(raw):
        try:
            val = int(tok)
        except ValueError as exc:
            raise SystemExit(f"Invalid integer in --{flag}: {tok}") from exc
        if val < 1:
            raise SystemExit(f"--{flag} values must be >= 1")
        out.append(val)
    if not out:
        raise SystemExit(f"--{flag} cannot be empty")
    return out


def parse_float(value: str) -> float | None:
    text = value.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_int(value: str) -> int | None:
    text = value.strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def median_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def format_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_summary_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def select_instances(instances_dir: Path, requested_csv: str) -> list[str]:
    available = {p.name[: -len(".mps.gz")] for p in instances_dir.glob("*.mps.gz")}
    available.update({p.stem for p in instances_dir.glob("*.mps")})
    requested = parse_csv_tokens(requested_csv)
    selected = [name for name in requested if name in available]
    missing = [name for name in requested if name not in available]
    for name in missing:
        print(
            f"[presolve-matrix] Warning: instance not found in {instances_dir}: {name}",
            file=sys.stderr,
        )
    if not selected:
        raise SystemExit(f"No requested instances found in {instances_dir}")
    return selected


def collect_repeat_kpis(csv_path: Path) -> dict[str, float | None]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    def collect(
        solver: str, presolve: str, field: str, as_int: bool = False
    ) -> list[float]:
        vals: list[float] = []
        for row in rows:
            if row.get("solver") != solver or row.get("presolve") != presolve:
                continue
            raw = row.get(field, "").strip()
            if not raw:
                continue
            try:
                val = int(raw) if as_int else float(raw)
            except ValueError:
                continue
            vals.append(float(val))
        return vals

    return {
        "mipx_on_rows_removed_med": median_or_none(
            collect("mipx", "on", "presolve_rows_removed", as_int=True)
        ),
        "mipx_on_cols_removed_med": median_or_none(
            collect("mipx", "on", "presolve_cols_removed", as_int=True)
        ),
        "mipx_on_bounds_tightened_med": median_or_none(
            collect("mipx", "on", "presolve_bounds_tightened", as_int=True)
        ),
        "mipx_on_rounds_med": median_or_none(
            collect("mipx", "on", "presolve_rounds", as_int=True)
        ),
        "highs_on_rows_removed_med": median_or_none(
            collect("highs", "on", "presolve_rows_removed", as_int=True)
        ),
        "highs_on_cols_removed_med": median_or_none(
            collect("highs", "on", "presolve_cols_removed", as_int=True)
        ),
    }


@dataclass(frozen=True)
class Experiment:
    name: str
    mode: str
    threads: int
    time_limit: float
    instances_dir: Path
    instances: list[str]
    repeats: int
    lp_solver: str
    cuts: str
    mipx_args: list[str]
    highs_args: list[str]


def profile_defaults(profile: str) -> dict[str, object]:
    if profile == "ci-smoke":
        return {
            "lp_solvers": ["dual"],
            "lp_threads": [1],
            "mip_threads": [1],
            "mip_cuts": ["off"],
            "lp_instances": ["afiro", "blend", "kb2", "sc105"],
            "mip_instances": ["p0201", "flugpl"],
            "lp_time_limit": 15.0,
            "mip_time_limit": 20.0,
            "single_thread_repeats": 1,
            "threaded_repeats": 1,
        }
    if profile == "internal":
        return {
            "lp_solvers": ["dual", "barrier", "pdlp"],
            "lp_threads": [1],
            "mip_threads": [1, 8],
            "mip_cuts": ["off", "on"],
            "lp_instances": ["afiro", "blend", "kb2", "sc105", "ship12l", "sierra"],
            "mip_instances": ["flugpl", "dcmulti", "p0201", "mod010", "timtab1"],
            "lp_time_limit": 30.0,
            "mip_time_limit": 30.0,
            "single_thread_repeats": 1,
            "threaded_repeats": 3,
        }
    raise SystemExit(f"Unknown profile: {profile}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", choices=("ci-smoke", "internal"), default="ci-smoke")
    p.add_argument("--mipx-binary", default=str(ROOT_DIR / "build" / "mipx-solve"))
    p.add_argument("--highs-binary", default="")
    p.add_argument("--netlib-dir", default=str(ROOT_DIR / "tests" / "data" / "netlib"))
    p.add_argument("--miplib-dir", default=str(ROOT_DIR / "tests" / "data" / "miplib"))
    p.add_argument("--out-dir", default="/tmp/mipx_presolve_matrix")

    p.add_argument("--lp-solvers", default="")
    p.add_argument("--lp-threads", default="")
    p.add_argument("--mip-threads", default="")
    p.add_argument("--mip-cuts", default="")
    p.add_argument("--lp-instances", default="")
    p.add_argument("--mip-instances", default="")
    p.add_argument("--lp-time-limit", type=float, default=0.0)
    p.add_argument("--mip-time-limit", type=float, default=0.0)
    p.add_argument("--single-thread-repeats", type=int, default=0)
    p.add_argument("--threaded-repeats", type=int, default=0)
    p.add_argument("--objective-rel-tol", type=float, default=1e-7)
    p.add_argument("--max-experiments", type=int, default=0)

    p.add_argument("--fail-on-mismatch", dest="fail_on_mismatch", action="store_true")
    p.add_argument("--no-fail-on-mismatch", dest="fail_on_mismatch", action="store_false")
    p.set_defaults(fail_on_mismatch=True)

    p.add_argument("--mipx-arg", action="append", default=[])
    p.add_argument("--highs-arg", action="append", default=[])

    argv: list[str] = []
    raw = sys.argv[1:]
    i = 0
    while i < len(raw):
        if raw[i] in {"--mipx-arg", "--highs-arg"}:
            if i + 1 >= len(raw):
                raise SystemExit(f"{raw[i]} requires one argument")
            argv.append(f"{raw[i]}={raw[i + 1]}")
            i += 2
            continue
        argv.append(raw[i])
        i += 1
    return p.parse_args(argv)


def lp_solver_flag(lp_solver: str) -> list[str]:
    if lp_solver == "dual":
        return []
    if lp_solver == "barrier":
        return ["--barrier"]
    if lp_solver == "pdlp":
        return ["--pdlp"]
    raise SystemExit(f"Unsupported LP solver: {lp_solver}")


def build_experiments(
    *,
    netlib_dir: Path,
    miplib_dir: Path,
    lp_solvers: list[str],
    lp_threads: list[int],
    mip_threads: list[int],
    mip_cuts: list[str],
    lp_instances: list[str],
    mip_instances: list[str],
    lp_time_limit: float,
    mip_time_limit: float,
    single_thread_repeats: int,
    threaded_repeats: int,
    global_mipx_args: list[str],
    global_highs_args: list[str],
) -> list[Experiment]:
    exps: list[Experiment] = []

    for solver in lp_solvers:
        for threads in lp_threads:
            repeats = threaded_repeats if threads > 1 else single_thread_repeats
            exps.append(
                Experiment(
                    name=f"lp_{solver}_t{threads}",
                    mode="lp",
                    threads=threads,
                    time_limit=lp_time_limit,
                    instances_dir=netlib_dir,
                    instances=lp_instances,
                    repeats=repeats,
                    lp_solver=solver,
                    cuts="na",
                    mipx_args=[*global_mipx_args, *lp_solver_flag(solver)],
                    highs_args=[*global_highs_args],
                )
            )

    for threads in mip_threads:
        for cuts in mip_cuts:
            if cuts not in {"on", "off"}:
                raise SystemExit(f"Unsupported cut mode in --mip-cuts: {cuts}")
            repeats = threaded_repeats if threads > 1 else single_thread_repeats
            cut_flag = "--cuts" if cuts == "on" else "--no-cuts"
            exps.append(
                Experiment(
                    name=f"mip_cuts-{cuts}_t{threads}",
                    mode="mip",
                    threads=threads,
                    time_limit=mip_time_limit,
                    instances_dir=miplib_dir,
                    instances=mip_instances,
                    repeats=repeats,
                    lp_solver="na",
                    cuts=cuts,
                    mipx_args=[*global_mipx_args, cut_flag],
                    highs_args=[*global_highs_args],
                )
            )

    return exps


def to_float_values(rows: list[dict[str, str]], field: str) -> list[float]:
    out: list[float] = []
    for row in rows:
        v = parse_float(row.get(field, ""))
        if v is not None:
            out.append(v)
    return out


def to_int_values(rows: list[dict[str, str]], field: str) -> list[int]:
    out: list[int] = []
    for row in rows:
        v = parse_int(row.get(field, ""))
        if v is not None:
            out.append(v)
    return out


def write_markdown(summary_rows: list[dict[str, str]], out_path: Path, profile: str) -> None:
    lines = [
        "# Presolve Matrix Summary",
        "",
        f"Profile: `{profile}`",
        "",
        "| experiment | mode | threads | cuts | repeats | mism(status/obj) | mipx_on/off s | highs_on/off s | mipx reductions (rows/cols/bounds/rounds) | verdict |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---|",
    ]

    for row in summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["experiment"],
                    row["mode"],
                    row["threads"],
                    row["cuts"],
                    row["repeats"],
                    f"{row['status_mismatches_max']}/{row['objective_mismatches_max']}",
                    f"{row['median_time_mipx_on_med'] or '-'} / {row['median_time_mipx_off_med'] or '-'}",
                    f"{row['median_time_highs_on_med'] or '-'} / {row['median_time_highs_off_med'] or '-'}",
                    (
                        f"{row['mipx_on_rows_removed_med'] or '-'} / "
                        f"{row['mipx_on_cols_removed_med'] or '-'} / "
                        f"{row['mipx_on_bounds_tightened_med'] or '-'} / "
                        f"{row['mipx_on_rounds_med'] or '-'}"
                    ),
                    row["verdict"],
                ]
            )
            + " |"
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    defaults = profile_defaults(args.profile)

    netlib_dir = Path(args.netlib_dir)
    miplib_dir = Path(args.miplib_dir)
    if not netlib_dir.is_dir():
        raise SystemExit(f"Netlib dir not found: {netlib_dir}")
    if not miplib_dir.is_dir():
        raise SystemExit(f"MIPLIB dir not found: {miplib_dir}")

    mipx_binary = Path(args.mipx_binary)
    if not mipx_binary.is_file() or not mipx_binary.stat().st_mode & 0o111:
        raise SystemExit(f"mipx binary not executable: {mipx_binary}")

    highs_binary = args.highs_binary or ""

    lp_solvers = parse_csv_tokens(args.lp_solvers) or list(defaults["lp_solvers"])
    lp_threads = (
        parse_int_csv(args.lp_threads, "lp-threads")
        if args.lp_threads
        else list(defaults["lp_threads"])
    )
    mip_threads = (
        parse_int_csv(args.mip_threads, "mip-threads")
        if args.mip_threads
        else list(defaults["mip_threads"])
    )
    mip_cuts = parse_csv_tokens(args.mip_cuts) or list(defaults["mip_cuts"])

    requested_lp = args.lp_instances or ",".join(defaults["lp_instances"])
    requested_mip = args.mip_instances or ",".join(defaults["mip_instances"])
    lp_instances = select_instances(netlib_dir, requested_lp)
    mip_instances = select_instances(miplib_dir, requested_mip)

    lp_time_limit = args.lp_time_limit if args.lp_time_limit > 0 else float(defaults["lp_time_limit"])
    mip_time_limit = (
        args.mip_time_limit if args.mip_time_limit > 0 else float(defaults["mip_time_limit"])
    )
    single_thread_repeats = (
        args.single_thread_repeats
        if args.single_thread_repeats > 0
        else int(defaults["single_thread_repeats"])
    )
    threaded_repeats = (
        args.threaded_repeats
        if args.threaded_repeats > 0
        else int(defaults["threaded_repeats"])
    )
    if single_thread_repeats < 1 or threaded_repeats < 1:
        raise SystemExit("Repeat counts must be >= 1")

    experiments = build_experiments(
        netlib_dir=netlib_dir,
        miplib_dir=miplib_dir,
        lp_solvers=lp_solvers,
        lp_threads=lp_threads,
        mip_threads=mip_threads,
        mip_cuts=mip_cuts,
        lp_instances=lp_instances,
        mip_instances=mip_instances,
        lp_time_limit=lp_time_limit,
        mip_time_limit=mip_time_limit,
        single_thread_repeats=single_thread_repeats,
        threaded_repeats=threaded_repeats,
        global_mipx_args=args.mipx_arg,
        global_highs_args=args.highs_arg,
    )
    if args.max_experiments > 0:
        experiments = experiments[: args.max_experiments]

    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    detail_rows: list[dict[str, str]] = []
    summary_rows: list[dict[str, str]] = []

    any_mismatch = False

    for exp in experiments:
        exp_detail: list[dict[str, str]] = []
        for rep in range(1, exp.repeats + 1):
            stem = f"{exp.name}.r{rep}"
            output_csv = raw_dir / f"{stem}.csv"
            output_summary = raw_dir / f"{stem}.summary"

            cmd = [
                sys.executable,
                str(RUN_COMPARE),
                "--mode",
                exp.mode,
                "--instances-dir",
                str(exp.instances_dir),
                "--instances",
                ",".join(exp.instances),
                "--threads",
                str(exp.threads),
                "--time-limit",
                f"{exp.time_limit:g}",
                "--objective-rel-tol",
                f"{args.objective_rel_tol:g}",
                "--mipx-binary",
                str(mipx_binary),
                "--output",
                str(output_csv),
                "--summary",
                str(output_summary),
            ]
            if highs_binary:
                cmd.extend(["--highs-binary", highs_binary])
            for m_arg in exp.mipx_args:
                cmd.append(f"--mipx-arg={m_arg}")
            for h_arg in exp.highs_args:
                cmd.append(f"--highs-arg={h_arg}")

            run(cmd)

            summary = parse_summary_file(output_summary)
            kpi = collect_repeat_kpis(output_csv)

            row = {
                "profile": args.profile,
                "experiment": exp.name,
                "repeat": str(rep),
                "mode": exp.mode,
                "threads": str(exp.threads),
                "cuts": exp.cuts,
                "lp_solver": exp.lp_solver,
                "instances": ",".join(exp.instances),
                "instances_count": str(len(exp.instances)),
                "status_mismatches": summary.get("status_mismatches", ""),
                "objective_mismatches": summary.get("objective_mismatches", ""),
                "comparables_status": summary.get("comparables_status", ""),
                "comparables_optimal": summary.get("comparables_optimal", ""),
                "median_time_mipx_on": summary.get("median_time_mipx_on", ""),
                "median_time_mipx_off": summary.get("median_time_mipx_off", ""),
                "median_time_highs_on": summary.get("median_time_highs_on", ""),
                "median_time_highs_off": summary.get("median_time_highs_off", ""),
                "mipx_on_rows_removed_med": format_float(kpi["mipx_on_rows_removed_med"]),
                "mipx_on_cols_removed_med": format_float(kpi["mipx_on_cols_removed_med"]),
                "mipx_on_bounds_tightened_med": format_float(
                    kpi["mipx_on_bounds_tightened_med"]
                ),
                "mipx_on_rounds_med": format_float(kpi["mipx_on_rounds_med"]),
                "highs_on_rows_removed_med": format_float(kpi["highs_on_rows_removed_med"]),
                "highs_on_cols_removed_med": format_float(kpi["highs_on_cols_removed_med"]),
                "raw_csv": str(output_csv),
                "raw_summary": str(output_summary),
            }

            detail_rows.append(row)
            exp_detail.append(row)

        status_mismatches = to_int_values(exp_detail, "status_mismatches")
        objective_mismatches = to_int_values(exp_detail, "objective_mismatches")
        comparables_status = to_int_values(exp_detail, "comparables_status")
        comparables_optimal = to_int_values(exp_detail, "comparables_optimal")
        mipx_on_times = to_float_values(exp_detail, "median_time_mipx_on")
        mipx_off_times = to_float_values(exp_detail, "median_time_mipx_off")
        highs_on_times = to_float_values(exp_detail, "median_time_highs_on")
        highs_off_times = to_float_values(exp_detail, "median_time_highs_off")

        mipx_on_rows = to_float_values(exp_detail, "mipx_on_rows_removed_med")
        mipx_on_cols = to_float_values(exp_detail, "mipx_on_cols_removed_med")
        mipx_on_bounds = to_float_values(exp_detail, "mipx_on_bounds_tightened_med")
        mipx_on_rounds = to_float_values(exp_detail, "mipx_on_rounds_med")
        highs_on_rows = to_float_values(exp_detail, "highs_on_rows_removed_med")
        highs_on_cols = to_float_values(exp_detail, "highs_on_cols_removed_med")

        max_status_mismatch = max(status_mismatches) if status_mismatches else 0
        max_objective_mismatch = max(objective_mismatches) if objective_mismatches else 0
        if max_status_mismatch > 0 or max_objective_mismatch > 0:
            any_mismatch = True

        summary_row = {
            "profile": args.profile,
            "experiment": exp.name,
            "mode": exp.mode,
            "threads": str(exp.threads),
            "cuts": exp.cuts,
            "lp_solver": exp.lp_solver,
            "repeats": str(exp.repeats),
            "instances_count": str(len(exp.instances)),
            "status_mismatches_max": str(max_status_mismatch),
            "objective_mismatches_max": str(max_objective_mismatch),
            "comparables_status_min": str(min(comparables_status) if comparables_status else 0),
            "comparables_optimal_min": str(min(comparables_optimal) if comparables_optimal else 0),
            "median_time_mipx_on_med": format_float(median_or_none(mipx_on_times)),
            "median_time_mipx_on_min": format_float(min(mipx_on_times) if mipx_on_times else None),
            "median_time_mipx_on_max": format_float(max(mipx_on_times) if mipx_on_times else None),
            "median_time_mipx_off_med": format_float(median_or_none(mipx_off_times)),
            "median_time_highs_on_med": format_float(median_or_none(highs_on_times)),
            "median_time_highs_off_med": format_float(median_or_none(highs_off_times)),
            "mipx_on_rows_removed_med": format_float(median_or_none(mipx_on_rows)),
            "mipx_on_cols_removed_med": format_float(median_or_none(mipx_on_cols)),
            "mipx_on_bounds_tightened_med": format_float(median_or_none(mipx_on_bounds)),
            "mipx_on_rounds_med": format_float(median_or_none(mipx_on_rounds)),
            "highs_on_rows_removed_med": format_float(median_or_none(highs_on_rows)),
            "highs_on_cols_removed_med": format_float(median_or_none(highs_on_cols)),
            "verdict": "PASS"
            if max_status_mismatch == 0 and max_objective_mismatch == 0
            else "FAIL",
        }
        summary_rows.append(summary_row)

    detail_csv = out_dir / "presolve_matrix_detail.csv"
    summary_csv = out_dir / "presolve_matrix_summary.csv"
    summary_md = out_dir / "presolve_matrix_summary.md"

    detail_fields = [
        "profile",
        "experiment",
        "repeat",
        "mode",
        "threads",
        "cuts",
        "lp_solver",
        "instances",
        "instances_count",
        "status_mismatches",
        "objective_mismatches",
        "comparables_status",
        "comparables_optimal",
        "median_time_mipx_on",
        "median_time_mipx_off",
        "median_time_highs_on",
        "median_time_highs_off",
        "mipx_on_rows_removed_med",
        "mipx_on_cols_removed_med",
        "mipx_on_bounds_tightened_med",
        "mipx_on_rounds_med",
        "highs_on_rows_removed_med",
        "highs_on_cols_removed_med",
        "raw_csv",
        "raw_summary",
    ]
    with detail_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=detail_fields)
        writer.writeheader()
        writer.writerows(detail_rows)

    summary_fields = [
        "profile",
        "experiment",
        "mode",
        "threads",
        "cuts",
        "lp_solver",
        "repeats",
        "instances_count",
        "status_mismatches_max",
        "objective_mismatches_max",
        "comparables_status_min",
        "comparables_optimal_min",
        "median_time_mipx_on_med",
        "median_time_mipx_on_min",
        "median_time_mipx_on_max",
        "median_time_mipx_off_med",
        "median_time_highs_on_med",
        "median_time_highs_off_med",
        "mipx_on_rows_removed_med",
        "mipx_on_cols_removed_med",
        "mipx_on_bounds_tightened_med",
        "mipx_on_rounds_med",
        "highs_on_rows_removed_med",
        "highs_on_cols_removed_med",
        "verdict",
    ]
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    write_markdown(summary_rows, summary_md, args.profile)

    print(f"Wrote detail CSV: {detail_csv}")
    print(f"Wrote summary CSV: {summary_csv}")
    print(f"Wrote summary markdown: {summary_md}")

    if args.fail_on_mismatch and any_mismatch:
        print("[presolve-matrix] FAIL: mismatch detected", file=sys.stderr)
        return 1
    print("[presolve-matrix] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
