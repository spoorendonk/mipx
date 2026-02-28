#!/usr/bin/env python3
"""Run deterministic dual-simplex correctness checks against .solu + HiGHS simplex.

Outputs one CSV row per instance with mismatch classification to build a
reproducible Step-35 mismatch corpus.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


def instance_name(path: Path) -> str:
    name = path.name
    if name.endswith(".mps.gz"):
        return name[: -len(".mps.gz")]
    if name.endswith(".mps"):
        return name[: -len(".mps")]
    return path.stem


def parse_first_float(pattern: str, text: str) -> float | None:
    m = re.search(pattern, text, re.MULTILINE)
    if not m:
        return None
    try:
        val = float(m.group(1))
    except ValueError:
        return None
    return val if math.isfinite(val) else None


def normalize_status(raw: str) -> str:
    status = re.sub(r"[^a-z0-9]+", "_", raw.strip().lower()).strip("_")
    status = {
        "time_limit_reached": "time_limit",
        "iteration_limit_reached": "iteration_limit",
        "iteration_limit": "iter_limit",
        "node_limit_reached": "node_limit",
        "node_limit": "node_limit",
        "solve_error": "error",
    }.get(status, status)
    return status or "unknown"


def parse_mipx_status(text: str) -> str:
    m = re.search(r"^Status:\s*(.+?)\s*$", text, re.MULTILINE)
    if not m:
        return "unknown"
    return normalize_status(m.group(1))


def parse_highs_status(text: str) -> str:
    m = re.search(r"^\s*Model status\s*:\s*(.+?)\s*$", text, re.MULTILINE)
    if m:
        return normalize_status(m.group(1))
    matches = re.findall(r"^\s*Status\s+(.+?)\s*$", text, re.MULTILINE)
    if matches:
        return normalize_status(matches[-1])
    if re.search(r"\binfeasible\b", text, re.IGNORECASE):
        return "infeasible"
    if re.search(r"\bunbounded\b", text, re.IGNORECASE):
        return "unbounded"
    return "unknown"


def parse_solu(path: Path) -> dict[str, tuple[str, float | None]]:
    out: dict[str, tuple[str, float | None]] = {}
    if not path.is_file():
        return out
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            tag = parts[0]
            name = parts[1]
            if tag == "=inf=":
                out[name] = ("infeasible", None)
                continue
            if tag != "=opt=":
                continue
            if len(parts) < 3:
                continue
            try:
                val = float(parts[2])
            except ValueError:
                continue
            out[name] = ("optimal", val)
    return out


def write_highs_options_file(threads: int) -> str:
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".options", delete=False, encoding="utf-8")
    try:
        tmp.write(f"threads = {threads}\n")
    finally:
        tmp.close()
    return tmp.name


@dataclass
class SolveResult:
    status: str
    objective: float | None
    iterations: float | None
    time_seconds: float
    rc: int
    output: str


def run_cmd(cmd: list[str], timeout_s: float) -> tuple[int, str, float, bool]:
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        elapsed = time.perf_counter() - t0
        out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        return proc.returncode, out, elapsed, False
    except subprocess.TimeoutExpired as e:
        elapsed = time.perf_counter() - t0
        out = ((e.stdout or "") if isinstance(e.stdout, str) else "") + (
            "\n" + (e.stderr or "") if isinstance(e.stderr, str) and e.stderr else ""
        )
        return 124, out, elapsed, True


def run_mipx_dual(binary: str, model: Path, time_limit: float, disable_presolve: bool) -> SolveResult:
    cmd = [binary, str(model), "--dual", "--quiet"]
    if disable_presolve:
        cmd.append("--no-presolve")
    if time_limit > 0:
        cmd.extend(["--time-limit", f"{time_limit:g}"])
    timeout_s = time_limit if time_limit > 0 else 300.0
    rc, out, elapsed, timed_out = run_cmd(cmd, timeout_s=timeout_s)
    if timed_out:
        return SolveResult(
            status="time_limit",
            objective=None,
            iterations=None,
            time_seconds=elapsed,
            rc=rc,
            output=out,
        )
    status = parse_mipx_status(out)
    obj = parse_first_float(r"^Objective:\s*([\-+0-9.eE]+)\s*$", out)
    iters = parse_first_float(r"^Iterations:\s*([\-+0-9.eE]+)\s*$", out)
    t = parse_first_float(r"^Time:\s*([\-+0-9.eE]+)s\s*$", out)
    return SolveResult(
        status=status,
        objective=obj,
        iterations=iters,
        time_seconds=t if t is not None else elapsed,
        rc=rc,
        output=out,
    )


def run_highs_simplex(highs: str, model: Path, time_limit: float, threads: int, disable_presolve: bool) -> SolveResult:
    options_file = write_highs_options_file(threads)
    try:
        cmd = [
            highs,
            str(model),
            "--options_file",
            options_file,
            "--solver",
            "simplex",
            "--presolve",
            "off" if disable_presolve else "choose",
            "--parallel",
            "off" if threads == 1 else "on",
        ]
        if time_limit > 0:
            cmd.extend(["--time_limit", f"{time_limit:g}"])
        timeout_s = time_limit if time_limit > 0 else 300.0
        rc, out, elapsed, timed_out = run_cmd(cmd, timeout_s=timeout_s)
        if timed_out:
            return SolveResult(
                status="time_limit",
                objective=None,
                iterations=None,
                time_seconds=elapsed,
                rc=rc,
                output=out,
            )
        status = parse_highs_status(out)
        obj = parse_first_float(r"^\s*Objective value\s*:\s*([\-+0-9.eE]+)\s*$", out)
        iters = parse_first_float(r"^\s*Simplex\s+iterations:\s*([\-+0-9.eE]+)\s*$", out)
        t = parse_first_float(r"^\s*HiGHS run time\s*:\s*([\-+0-9.eE]+)\s*$", out)
        return SolveResult(
            status=status,
            objective=obj,
            iterations=iters,
            time_seconds=t if t is not None else elapsed,
            rc=rc,
            output=out,
        )
    finally:
        try:
            os.unlink(options_file)
        except OSError:
            pass


def rel_obj_error(a: float, b: float) -> float:
    denom = max(1.0, abs(b))
    return abs(a - b) / denom


def classify(
    expected_status: str,
    expected_objective: float | None,
    mipx: SolveResult,
    highs: SolveResult,
    objective_rel_tol: float,
) -> tuple[str, str]:
    limit_statuses = {"time_limit", "iter_limit", "iteration_limit", "node_limit"}

    if expected_status == "optimal":
        if mipx.status in {"infeasible"}:
            return "false_infeasible", "mipx reported infeasible on expected optimal instance"
        if mipx.status in limit_statuses:
            return "limit_before_optimal", "mipx hit solve limit before proving optimal"
        if mipx.status not in {"optimal"}:
            return "false_unknown_or_error", f"mipx status={mipx.status}"
        if mipx.status == "optimal" and expected_objective is not None and mipx.objective is not None:
            if rel_obj_error(mipx.objective, expected_objective) > objective_rel_tol:
                return "objective_mismatch", "mipx objective differs from .solu beyond tolerance"
        if highs.status == "optimal" and mipx.status == "optimal":
            if highs.objective is not None and mipx.objective is not None:
                if rel_obj_error(mipx.objective, highs.objective) > objective_rel_tol:
                    return "objective_mismatch", "mipx objective differs from HiGHS beyond tolerance"

    if expected_status == "infeasible":
        if mipx.status == "optimal":
            return "status_mismatch", "mipx reported optimal on expected infeasible instance"

    return "ok", ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mipx-binary", default="./build/mipx-solve")
    p.add_argument("--netlib-dir", default="tests/data/netlib")
    p.add_argument("--solu-file", default="tests/data/netlib/netlib.solu")
    p.add_argument("--corpus", default="tests/perf/netlib_dual_corpus.csv")
    p.add_argument("--output", required=True)
    p.add_argument("--summary", default="")
    p.add_argument("--instances", default="")
    p.add_argument("--time-limit", type=float, default=60.0)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--objective-rel-tol", type=float, default=1e-6)
    p.add_argument("--disable-presolve", action="store_true")
    p.add_argument("--highs-binary", default="")
    p.add_argument("--repeats", type=int, default=1)
    return p.parse_args()


def load_corpus(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    args = parse_args()

    mipx_bin = Path(args.mipx_binary)
    if not mipx_bin.is_file() or not os.access(mipx_bin, os.X_OK):
        raise SystemExit(f"mipx binary not executable: {mipx_bin}")

    netlib_dir = Path(args.netlib_dir)
    if not netlib_dir.is_dir():
        raise SystemExit(f"netlib dir not found: {netlib_dir}")

    corpus_path = Path(args.corpus)
    if not corpus_path.is_file():
        raise SystemExit(f"corpus CSV not found: {corpus_path}")

    highs = args.highs_binary or os.environ.get("HIGHS_BINARY") or shutil.which("highs")
    if not highs:
        raise SystemExit("HiGHS binary not found. Set --highs-binary or HIGHS_BINARY.")

    solu = parse_solu(Path(args.solu_file))
    rows = load_corpus(corpus_path)

    inst_filter = {x.strip() for x in args.instances.split(",") if x.strip()}
    if inst_filter:
        rows = [r for r in rows if r.get("instance", "") in inst_filter]

    if not rows:
        raise SystemExit("no instances selected")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    out_rows: list[dict[str, str]] = []

    for row in rows:
        name = row["instance"].strip()
        if not name:
            continue
        model = netlib_dir / f"{name}.mps.gz"
        if not model.is_file():
            model = netlib_dir / f"{name}.mps"
        if not model.is_file():
            out_rows.append(
                {
                    "instance": name,
                    "expected_status": row.get("expected_status", ""),
                    "expected_objective": row.get("expected_objective", ""),
                    "mipx_status": "missing_instance",
                    "mipx_objective": "",
                    "mipx_iterations": "",
                    "mipx_time_seconds": "",
                    "mipx_rc": "",
                    "highs_status": "",
                    "highs_objective": "",
                    "highs_iterations": "",
                    "highs_time_seconds": "",
                    "highs_rc": "",
                    "classification": "missing_instance",
                    "detail": str(model),
                    "mipx_signature": "",
                    "source": row.get("source", ""),
                    "notes": row.get("notes", ""),
                }
            )
            continue

        expected_status = row.get("expected_status", "").strip().lower() or solu.get(name, ("", None))[0]
        expected_obj_str = row.get("expected_objective", "").strip()
        expected_obj: float | None = None
        if expected_obj_str:
            try:
                expected_obj = float(expected_obj_str)
            except ValueError:
                expected_obj = None
        elif name in solu:
            expected_obj = solu[name][1]

        mipx_runs: list[SolveResult] = []
        highs_runs: list[SolveResult] = []
        for _ in range(max(1, args.repeats)):
            mipx_runs.append(
                run_mipx_dual(
                    binary=str(mipx_bin),
                    model=model,
                    time_limit=args.time_limit,
                    disable_presolve=args.disable_presolve,
                )
            )
            highs_runs.append(
                run_highs_simplex(
                    highs=highs,
                    model=model,
                    time_limit=args.time_limit,
                    threads=args.threads,
                    disable_presolve=args.disable_presolve,
                )
            )

        mipx = mipx_runs[0]
        highs_res = highs_runs[0]

        if len(mipx_runs) > 1:
            statuses = {r.status for r in mipx_runs}
            if len(statuses) > 1:
                mipx.status = "mixed_status"
            times = [r.time_seconds for r in mipx_runs if r.time_seconds > 0]
            if times:
                mipx.time_seconds = float(statistics.median(times))

        classification, detail = classify(
            expected_status=expected_status,
            expected_objective=expected_obj,
            mipx=mipx,
            highs=highs_res,
            objective_rel_tol=args.objective_rel_tol,
        )

        sig = ""
        if mipx.rc != 0 or classification in {"false_unknown_or_error", "false_infeasible"}:
            line = ""
            for candidate in mipx.output.splitlines():
                if "singular" in candidate.lower() or "error:" in candidate.lower():
                    line = candidate.strip()
                    break
            if not line:
                line = mipx.output.splitlines()[-1].strip() if mipx.output.splitlines() else ""
            sig = line[:200]

        out_rows.append(
            {
                "instance": name,
                "expected_status": expected_status,
                "expected_objective": f"{expected_obj:.12g}" if expected_obj is not None else "",
                "mipx_status": mipx.status,
                "mipx_objective": f"{mipx.objective:.12g}" if mipx.objective is not None else "",
                "mipx_iterations": f"{mipx.iterations:.0f}" if mipx.iterations is not None else "",
                "mipx_time_seconds": f"{mipx.time_seconds:.6f}",
                "mipx_rc": str(mipx.rc),
                "highs_status": highs_res.status,
                "highs_objective": f"{highs_res.objective:.12g}" if highs_res.objective is not None else "",
                "highs_iterations": f"{highs_res.iterations:.0f}" if highs_res.iterations is not None else "",
                "highs_time_seconds": f"{highs_res.time_seconds:.6f}",
                "highs_rc": str(highs_res.rc),
                "classification": classification,
                "detail": detail,
                "mipx_signature": sig,
                "source": row.get("source", ""),
                "notes": row.get("notes", ""),
            }
        )

    headers = [
        "instance",
        "expected_status",
        "expected_objective",
        "mipx_status",
        "mipx_objective",
        "mipx_iterations",
        "mipx_time_seconds",
        "mipx_rc",
        "highs_status",
        "highs_objective",
        "highs_iterations",
        "highs_time_seconds",
        "highs_rc",
        "classification",
        "detail",
        "mipx_signature",
        "source",
        "notes",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerows(out_rows)

    by_class: dict[str, int] = {}
    for r in out_rows:
        by_class[r["classification"]] = by_class.get(r["classification"], 0) + 1

    summary_lines = [
        "# Dual Simplex Correctness Investigation Summary",
        "",
        f"instances={len(out_rows)}",
        f"output={output_path}",
        "",
        "classification,count",
    ]
    for k in sorted(by_class):
        summary_lines.append(f"{k},{by_class[k]}")

    summary = "\n".join(summary_lines) + "\n"
    if args.summary:
        summary_path = Path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(summary, encoding="utf-8")

    sys.stdout.write(summary)

    hard_fail = sum(by_class.get(k, 0) for k in ("false_infeasible", "false_unknown_or_error", "objective_mismatch"))
    return 1 if hard_fail > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
