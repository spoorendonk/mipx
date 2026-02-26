#!/usr/bin/env python3
"""Run a structured MIP parameter sweep and emit CSV + Markdown artifacts."""

from __future__ import annotations

import argparse
import csv
import itertools
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
PERF_DIR = ROOT_DIR / "tests" / "perf"
RUN_MIPLIB = PERF_DIR / "run_miplib_mip_bench.py"


def parse_csv_tokens(raw: str) -> list[str]:
    return [tok.strip() for tok in raw.split(",") if tok.strip()]


def parse_float(text: str) -> float | None:
    value = text.strip()
    if not value:
        return None
    try:
        return float(value)
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


def parse_bool_list(raw: str, name: str) -> list[bool]:
    mapping = {
        "on": True,
        "off": False,
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }
    out: list[bool] = []
    for tok in parse_csv_tokens(raw):
        if tok.lower() not in mapping:
            raise SystemExit(f"Unsupported value in --{name}: {tok} (expected on/off)")
        out.append(mapping[tok.lower()])
    if not out:
        raise SystemExit(f"--{name} cannot be empty")
    return out


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def collect_instances(miplib_dir: Path, instance_filter: str, max_instances: int) -> list[str]:
    available = sorted(p.name.removesuffix(".mps.gz") for p in miplib_dir.glob("*.mps.gz"))
    if not available:
        raise SystemExit(f"No .mps.gz instances found in {miplib_dir}")

    if instance_filter:
        requested = parse_csv_tokens(instance_filter)
        selected = [name for name in requested if name in set(available)]
        missing = [name for name in requested if name not in set(available)]
        for name in missing:
            print(f"Warning: requested instance not found: {name}", file=sys.stderr)
    else:
        selected = available

    if max_instances > 0:
        selected = selected[:max_instances]

    if not selected:
        raise SystemExit("No instances selected for parameter sweep.")
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--miplib-dir", required=True)
    parser.add_argument("--out-dir", default="/tmp/mipx_param_sweep")
    parser.add_argument("--instances", default="p0201,gt2,flugpl")
    parser.add_argument("--max-instances", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--time-limit", type=float, default=30.0)
    parser.add_argument("--node-limit", type=int, default=100000)
    parser.add_argument("--gap-tol", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--search-profiles", default="stable,default,aggressive")
    parser.add_argument("--heur-modes", default="deterministic")
    parser.add_argument("--cuts", default="on,off")
    parser.add_argument("--presolve", default="on,off")
    parser.add_argument("--metric", choices=("work_units", "time_seconds"), default="work_units")
    parser.add_argument("--max-configs", type=int, default=0)
    parser.add_argument("--solver-arg", action="append", default=[])
    return parser.parse_args()


def search_flag(profile: str) -> str:
    profile = profile.lower()
    if profile == "stable":
        return "--search-stable"
    if profile == "default":
        return "--search-default"
    if profile == "aggressive":
        return "--search-aggressive"
    raise SystemExit(f"Unsupported search profile: {profile}")


def heur_flag(mode: str) -> str:
    mode = mode.lower()
    if mode == "deterministic":
        return "--heur-deterministic"
    if mode == "opportunistic":
        return "--heur-opportunistic"
    raise SystemExit(f"Unsupported heuristic mode: {mode}")


def parse_miplib_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def summarize_combo(rows: list[dict[str, str]]) -> dict[str, str]:
    status_counter = Counter(row["status"] for row in rows)
    ok_rows = [row for row in rows if row["status"] not in {"solve_error", "parse_error"}]

    times = [v for row in ok_rows if (v := parse_float(row.get("time_seconds", ""))) is not None]
    works = [v for row in ok_rows if (v := parse_float(row.get("work_units", ""))) is not None]
    nodes = [v for row in ok_rows if (v := parse_float(row.get("nodes", ""))) is not None]
    iters = [v for row in ok_rows if (v := parse_float(row.get("lp_iterations", ""))) is not None]

    return {
        "instances_total": str(len(rows)),
        "instances_ok": str(len(ok_rows)),
        "median_time_seconds": format_float(median_or_none(times)),
        "median_work_units": format_float(median_or_none(works)),
        "median_nodes": format_float(median_or_none(nodes)),
        "median_lp_iterations": format_float(median_or_none(iters)),
        "status_mix": "; ".join(f"{k}:{status_counter[k]}" for k in sorted(status_counter)),
    }


def ranking_key(row: dict[str, str], metric: str) -> tuple[float, float, int]:
    metric_col = "median_work_units" if metric == "work_units" else "median_time_seconds"
    metric_v = parse_float(row.get(metric_col, ""))
    time_v = parse_float(row.get("median_time_seconds", ""))
    ok_count = int(row.get("instances_ok", "0"))
    return (
        metric_v if metric_v is not None else float("inf"),
        time_v if time_v is not None else float("inf"),
        -ok_count,
    )


def write_markdown(summary_rows: list[dict[str, str]], metric: str, path: Path) -> None:
    ranked = sorted(summary_rows, key=lambda row: ranking_key(row, metric))

    lines = [
        "# Parameter Sweep Summary",
        "",
        f"Ranking metric: `{metric}` (lower is better)",
        "",
        "| rank | search | heur | cuts | presolve | ok/total | median_work_units | median_time_seconds | median_nodes | median_lp_iterations |",
        "|---:|---|---|---|---|---:|---:|---:|---:|---:|",
    ]

    for idx, row in enumerate(ranked, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(idx),
                    row["search_profile"],
                    row["heur_mode"],
                    row["cuts"],
                    row["presolve"],
                    f"{row['instances_ok']}/{row['instances_total']}",
                    row["median_work_units"] or "-",
                    row["median_time_seconds"] or "-",
                    row["median_nodes"] or "-",
                    row["median_lp_iterations"] or "-",
                ]
            )
            + " |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()

    binary = Path(args.binary)
    miplib_dir = Path(args.miplib_dir)
    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if not binary.is_file() or not binary.stat().st_mode & 0o111:
        raise SystemExit(f"Binary not executable: {binary}")
    if not miplib_dir.is_dir():
        raise SystemExit(f"MIPLIB dir not found: {miplib_dir}")
    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")
    if args.threads < 1:
        raise SystemExit("--threads must be >= 1")
    if args.time_limit <= 0:
        raise SystemExit("--time-limit must be > 0")
    if args.node_limit < 1:
        raise SystemExit("--node-limit must be >= 1")
    if args.gap_tol <= 0:
        raise SystemExit("--gap-tol must be > 0")

    search_profiles = [tok.lower() for tok in parse_csv_tokens(args.search_profiles)]
    heur_modes = [tok.lower() for tok in parse_csv_tokens(args.heur_modes)]
    cuts_list = parse_bool_list(args.cuts, "cuts")
    presolve_list = parse_bool_list(args.presolve, "presolve")

    selected_instances = collect_instances(miplib_dir, args.instances, args.max_instances)
    instances_csv = ",".join(selected_instances)

    combos = list(itertools.product(search_profiles, heur_modes, cuts_list, presolve_list))
    if args.max_configs > 0:
        combos = combos[: args.max_configs]
    if not combos:
        raise SystemExit("No sweep configs selected.")

    detail_rows: list[dict[str, str]] = []
    summary_rows: list[dict[str, str]] = []

    for idx, (search_profile, heur_mode, cuts_on, presolve_on) in enumerate(combos, start=1):
        sweep_id = f"cfg{idx:03d}"
        raw_csv = raw_dir / f"{sweep_id}.csv"

        solver_args = [
            search_flag(search_profile),
            heur_flag(heur_mode),
            "--cuts" if cuts_on else "--no-cuts",
            "--presolve" if presolve_on else "--no-presolve",
            "--seed",
            str(args.seed),
        ]
        solver_args.extend(args.solver_arg)

        cmd = [
            sys.executable,
            str(RUN_MIPLIB),
            "--binary",
            str(binary),
            "--miplib-dir",
            str(miplib_dir),
            "--output",
            str(raw_csv),
            "--repeats",
            str(args.repeats),
            "--threads",
            str(args.threads),
            "--time-limit",
            f"{args.time_limit:g}",
            "--node-limit",
            str(args.node_limit),
            "--gap-tol",
            f"{args.gap_tol:g}",
            "--instances",
            instances_csv,
        ]
        for sarg in solver_args:
            cmd.extend(["--solver-arg", sarg])
        run(cmd)

        raw_rows = parse_miplib_rows(raw_csv)
        combo_rows: list[dict[str, str]] = []
        for row in raw_rows:
            mapped = {
                "sweep_id": sweep_id,
                "search_profile": search_profile,
                "heur_mode": heur_mode,
                "cuts": "on" if cuts_on else "off",
                "presolve": "on" if presolve_on else "off",
                "threads": str(args.threads),
                "time_limit": f"{args.time_limit:g}",
                "instance": row.get("instance", ""),
                "status": row.get("status", "unknown"),
                "time_seconds": row.get("time_seconds", ""),
                "work_units": row.get("work_units", ""),
                "nodes": row.get("nodes", ""),
                "lp_iterations": row.get("lp_iterations", ""),
                "source_csv": str(raw_csv),
            }
            combo_rows.append(mapped)

        detail_rows.extend(combo_rows)
        summary = summarize_combo(combo_rows)
        summary_rows.append(
            {
                "sweep_id": sweep_id,
                "search_profile": search_profile,
                "heur_mode": heur_mode,
                "cuts": "on" if cuts_on else "off",
                "presolve": "on" if presolve_on else "off",
                "threads": str(args.threads),
                "time_limit": f"{args.time_limit:g}",
                **summary,
            }
        )

    detail_csv = out_dir / "sweep_detail.csv"
    summary_csv = out_dir / "sweep_summary.csv"
    summary_md = out_dir / "sweep_summary.md"

    detail_header = [
        "sweep_id",
        "search_profile",
        "heur_mode",
        "cuts",
        "presolve",
        "threads",
        "time_limit",
        "instance",
        "status",
        "time_seconds",
        "work_units",
        "nodes",
        "lp_iterations",
        "source_csv",
    ]
    with detail_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=detail_header)
        writer.writeheader()
        writer.writerows(detail_rows)

    summary_header = [
        "sweep_id",
        "search_profile",
        "heur_mode",
        "cuts",
        "presolve",
        "threads",
        "time_limit",
        "instances_total",
        "instances_ok",
        "median_time_seconds",
        "median_work_units",
        "median_nodes",
        "median_lp_iterations",
        "status_mix",
    ]
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_header)
        writer.writeheader()
        writer.writerows(summary_rows)

    write_markdown(summary_rows, args.metric, summary_md)

    print(f"Wrote sweep detail CSV: {detail_csv}")
    print(f"Wrote sweep summary CSV: {summary_csv}")
    print(f"Wrote sweep summary Markdown: {summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
