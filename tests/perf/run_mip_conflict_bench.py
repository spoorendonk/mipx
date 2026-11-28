#!/usr/bin/env python3
"""Compare isolated serial conflict options using the public MIPLIB references (#478)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shlex
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from run_instance_gate import find_instances, read_solu

if TYPE_CHECKING:
    from run_instance_gate import Reference

CORPUS = (
    "air04",
    "blend2",
    "dcmulti",
    "flugpl",
    "gen",
    "gt2",
    "mod010",
    "noswot",
    "p0201",
    "pk1",
    "ran13x13",
    "stein9inf",
)
PILOT = ("mod010", "gt2", "stein9inf")
ARMS = {
    "baseline": (),
    "watches": ("--conflict-watches",),
    "1uip": ("--bound-conflict-learning",),
    "lp-proofs": ("--conflict-lp-proofs",),
    "reconvergence": ("--reconvergence-cuts",),
    "activity": ("--conflict-activity-branching",),
    "restarts": ("--conflict-restart-threshold", "8"),
}
COUNTERS = (
    "linear_clause_checks",
    "linear_literal_checks",
    "watch_candidates",
    "watch_body_scans",
    "watch_literal_checks",
    "lp_infeasible",
    "cutoff",
    "combined",
    "learned_1uip",
    "reconvergence",
    "legacy_branch_overrides",
    "conflict_restarts",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--build-id", required=True, help="Revision and configuration of BINARY.")
    parser.add_argument("--miplib-dir", type=Path, required=True)
    parser.add_argument("--solu-file", type=Path, help="Defaults to MIPLIB_DIR/miplib.solu.")
    parser.add_argument("--output", type=Path, required=True, help="New directory for raw results.")
    parser.add_argument("--instances", nargs="+", choices=CORPUS, help="Select a corpus subset.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pilot", action="store_true", help="Correctness smoke, 1 s and 200 nodes.")
    mode.add_argument("--time-limit", type=int, help="Hub-scheduled solver limit per job, seconds.")
    args = parser.parse_args()
    if args.time_limit is not None and args.time_limit <= 0:
        parser.error("--time-limit must be positive")
    return args


def parse_fields(line: str) -> dict[str, str]:
    return dict(field.split("=", 1) for field in line.split())


def parse_metrics(log: str) -> dict[str, str | int | float | None]:
    lines = dict(line.split(": ", 1) for line in log.splitlines() if ": " in line)
    bounds = parse_fields(lines["Conflict bench bounds"])
    tree = parse_fields(lines["Conflict bench tree"])
    counters = parse_fields(lines["Conflict bench counters"])
    root = lines.get("Conflict bench root bound")
    solver_seconds = float(lines["Time"].removesuffix("s"))
    if not math.isfinite(solver_seconds) or solver_seconds < 0:
        raise ValueError("invalid solver time")
    return {
        "status": lines["Status"],
        # Keep bounds as strings in JSON so infinities are exact and portable.
        "objective": None if lines["Objective"] == "none" else lines["Objective"],
        "sense": bounds["sense"],
        "input_continuous_cols": int(bounds["input_continuous_cols"]),
        "bound_learning_in_tree": int(tree["bound_learning"]),
        "reconvergence_in_tree": int(tree["reconvergence"]),
        "final_bound": bounds["final"],
        "root_bound": root,
        "nodes": int(lines["Nodes"]),
        "lp_iterations": int(lines["LP iterations"]),
        "solver_seconds": solver_seconds,
        **{name: int(counters[name]) for name in COUNTERS},
    }


def check_reference(metrics: dict[str, str | int | float | None], ref: Reference) -> str:
    status = metrics["status"]
    limited = status in ("Node limit", "Time limit")
    sense = metrics["sense"]
    if sense not in ("min", "max"):
        return "unknown objective sense"
    direction = 1 if sense == "min" else -1
    objective = metrics["objective"]
    incumbent = None if objective is None else direction * float(objective)
    bound = direction * float(str(metrics["final_bound"]))
    root = metrics["root_bound"]
    root_bound = None if root is None else direction * float(root)
    if math.isnan(bound) or (root_bound is not None and not math.isfinite(root_bound)):
        return "invalid bound"
    if incumbent is not None and not math.isfinite(incumbent):
        return "invalid incumbent"
    if ref.value is not None:
        return check_optimum(str(status), incumbent, bound, root_bound, direction * ref.value)
    if ref.kind == "inf" and incumbent is not None:
        return "incumbent on reference-infeasible instance"
    if limited:
        return "consistent limit"
    if (ref.kind, status) in (("inf", "Infeasible"), ("unbd", "Unbounded")):
        return "match"
    return f"verdict conflicts with reference ({ref.kind})"


def check_optimum(
    status: str, incumbent: float | None, bound: float, root_bound: float | None, optimum: float
) -> str:
    tol = 1e-6 * max(1.0, abs(optimum))
    if bound > optimum + tol or (root_bound is not None and root_bound > optimum + tol):
        return "dual bound excludes reference optimum"
    if incumbent is not None and incumbent < optimum - tol:
        return "incumbent improves on reference optimum"
    if status in ("Node limit", "Time limit"):
        return "consistent limit"
    if status == "Optimal" and incumbent is not None:
        if abs(incumbent - optimum) <= tol and abs(bound - optimum) <= tol:
            return "match"
        return "optimal objective or final bound differs from reference"
    return f"verdict conflicts with reference optimum ({status})"


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, allow_nan=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def run_job(
    out: Path, command: list[str], ref: Reference, time_limit: int, *, pilot: bool
) -> dict[str, object]:
    out.mkdir()
    log = out / "solver.log"
    write_json(out / "command.json", {"argv": command, "shell": shlex.join(command)})
    result: dict[str, object] = {"reference": asdict(ref)}
    started = None if pilot else time.monotonic()
    try:
        with log.open("w", encoding="utf-8") as stream:
            proc = subprocess.run(
                command,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=time_limit + 15,
                check=False,
            )
    except subprocess.TimeoutExpired:
        result["check"] = "process watchdog expired"
        return result
    finally:
        if started is not None:
            result["wall_seconds"] = time.monotonic() - started
    result["returncode"] = proc.returncode
    if proc.returncode:
        result["check"] = "solver process failed"
        return result
    try:
        metrics = parse_metrics(log.read_text(encoding="utf-8", errors="replace"))
        result.update(metrics)
        check = check_reference(metrics, ref)
        if pilot and check == "consistent limit" and metrics["root_bound"] is None:
            check = "pilot limit reached before a root bound was recorded"
        result["check"] = check
    except (KeyError, ValueError) as exc:
        result["check"] = f"invalid solver output ({exc})"
    return result


def main() -> int:
    args = parse_args()
    binary = args.binary.resolve(strict=True)
    miplib_dir = args.miplib_dir.resolve()
    solu = (args.solu_file or miplib_dir / "miplib.solu").resolve(strict=True)
    refs = read_solu(solu)
    inputs = find_instances(miplib_dir)
    names = tuple(dict.fromkeys(args.instances or (PILOT if args.pilot else CORPUS)))
    for name in names:
        if name not in inputs or name not in refs:
            raise SystemExit(f"Missing instance or solved MIPLIB reference for {name}")
        value = refs[name].value
        if value is not None and not math.isfinite(value):
            raise SystemExit(f"Nonfinite MIPLIB reference for {name}")
    limit = 1 if args.pilot else args.time_limit
    common = [
        "--dual",
        "--conflicts",
        "--threads",
        "1",
        "--parallel-mode",
        "deterministic",
        "--seed",
        "1",
        "--gap-tol",
        "0",
        "--time-limit",
        str(limit),
        "--print-conflict-bench-stats",
    ]
    if args.pilot:
        common += ["--node-limit", "200"]
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "manifest.json",
        {
            "pilot": args.pilot,
            "build_id": args.build_id,
            "host": platform.uname()._asdict(),
            "python": platform.python_version(),
            "runner_sha256": sha256(Path(__file__)),
            "solver_seconds_limit": limit,
            "process_seconds_limit": limit + 15,
            "solver_sha256": sha256(binary),
            "reference_sha256": sha256(solu),
            "instance_sha256": {name: sha256(inputs[name]) for name in names},
        },
    )
    failures = 0
    with (args.output / "results.jsonl").open("w", encoding="utf-8") as summary:
        for name in names:
            for arm, flags in ARMS.items():
                job = f"{name}-{arm}"
                command = [str(binary), str(inputs[name]), *common, *flags]
                result = run_job(args.output / job, command, refs[name], limit, pilot=args.pilot)
                result.update(instance=name, arm=arm, job=job)
                summary.write(json.dumps(result, allow_nan=False) + "\n")
                summary.flush()
                print(f"{name} {arm}: {result['check']}", flush=True)
                failures += result["check"] not in ("match", "consistent limit")
    return int(failures != 0)


if __name__ == "__main__":
    raise SystemExit(main())
