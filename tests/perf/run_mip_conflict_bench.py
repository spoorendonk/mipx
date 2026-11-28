#!/usr/bin/env python3
"""Compare serial conflict or rapid-learning options using public MIPLIB references."""

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
from typing import TYPE_CHECKING, TypedDict, cast

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
RAPID_CORPUS = ("stein9inf", "mod010", "flugpl", "blend2", "gen", "noswot", "pk1", "ran13x13")
RAPID_PILOT = ("stein9inf", "mod010", "flugpl")
RAPID_SHORT_ANCHORS = ("noswot", "pk1", "ran13x13")
RAPID_ARMS = {
    "baseline": (),
    "restarts": ARMS["restarts"],
    "geometric": (*ARMS["restarts"], "--geometric-restarts"),
    "probing": ("--systematic-probing",),
    "rapid-learning": ("--root-rapid-learning",),
    "lookahead": ("--root-lp-lookahead",),
    "incumbent": (*ARMS["restarts"], "--incumbent-guided-restarts"),
    "diversification": (*ARMS["restarts"], "--restart-branch-diversification"),
    "root-allowance": ("--root-rapid-learning", "--root-learning-allowance", "0.001", "32"),
    "restart-allowance": (*ARMS["restarts"], "--restart-overhead-allowance", "0.0001", "128"),
}
WALL_DEPENDENT_ARMS = {"root-allowance", "restart-allowance"}
RAPID_SECTIONS = {
    "--systematic-probing": "Systematic probing",
    "--root-rapid-learning": "Root rapid learning",
    "--root-lp-lookahead": "Root LP look-ahead",
    "--root-learning-allowance": "Root learning allowance",
    "--restart-overhead-allowance": "Restart overhead allowance",
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
TERMINAL_OUTCOME = ("status", "objective", "final_bound")
TERMINAL_WORK = (
    "nodes",
    "lp_iterations",
    "bound_learning_in_tree",
    "reconvergence_in_tree",
    "rapid_summary",
    *COUNTERS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--build-id", required=True, help="Revision and configuration of BINARY.")
    parser.add_argument("--miplib-dir", type=Path, required=True)
    parser.add_argument("--solu-file", type=Path, help="Defaults to MIPLIB_DIR/miplib.solu.")
    parser.add_argument("--output", type=Path, required=True, help="New directory for raw results.")
    parser.add_argument("--suite", choices=("conflict", "rapid-learning"), default="conflict")
    parser.add_argument("--instances", nargs="+", choices=CORPUS, help="Select a corpus subset.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pilot", action="store_true", help="Short, node-capped correctness smoke.")
    mode.add_argument("--time-limit", type=int, help="Hub-scheduled solver limit per job, seconds.")
    anchors = parser.add_mutually_exclusive_group()
    anchors.add_argument("--anchor", action="store_true", help="Pre-lock rapid-learning checks.")
    anchors.add_argument(
        "--anchor-results", type=Path, help="Verify a timed hold against this run."
    )
    args = parser.parse_args()
    if args.time_limit is not None and args.time_limit <= 0:
        parser.error("--time-limit must be positive")
    if (args.anchor or args.anchor_results) and (
        args.suite != "rapid-learning" or args.time_limit is None
    ):
        parser.error("anchor options require --suite rapid-learning and --time-limit")
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


class RapidAttempt(TypedDict):
    bounds: list[dict[str, str]]
    restarts: list[dict[str, int]]
    phases_seconds: dict[str, float]
    first_incumbent: dict[str, str] | None


def new_rapid_attempt() -> RapidAttempt:
    return {"bounds": [], "restarts": [], "phases_seconds": {}, "first_incumbent": None}


def parse_seconds(value: str) -> float:
    seconds = float(value)
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("invalid time")
    return seconds


def parse_rapid_attempts(lines: list[tuple[str, str]]) -> list[RapidAttempt]:
    attempts: list[RapidAttempt] = []
    for label, value in lines:
        if label == "Rapid bench attempt":
            attempts.append(new_rapid_attempt())
        elif label == "Rapid bench bound":
            attempts[-1]["bounds"].append(parse_fields(value))
        elif label == "Rapid bench restart":
            event = {name: int(field) for name, field in parse_fields(value).items()}
            if event["spacing"] <= 0 or event["spacing"] < event["required"]:
                raise ValueError("invalid restart spacing")
            attempts[-1]["restarts"].append(event)
        elif label == "Rapid bench phase":
            phase = parse_fields(value)
            attempts[-1]["phases_seconds"][phase["name"]] = parse_seconds(phase["seconds"])
        elif label == "Rapid bench incumbent":
            attempts[-1]["first_incumbent"] = parse_fields(value)
    if not attempts:
        raise ValueError("missing rapid-learning bench attempt")
    return attempts


def parse_rapid_metrics(log: str, conflict_restarts: int, command: list[str]) -> dict[str, object]:
    lines = []
    for line in log.splitlines():
        key, separator, value = line.partition(": ")
        if separator:
            lines.append((key, value))
    attempts = parse_rapid_attempts(lines)
    by_label = dict(lines)
    summary = parse_fields(by_label["Rapid bench summary"])
    seconds = parse_seconds(summary.pop("seconds"))
    stats = {name: int(value) for name, value in summary.items()}
    events = attempts[-1]["restarts"]
    if (
        len(events) != stats["restarts"]
        or sum(event["conflict"] for event in events) != conflict_restarts
    ):
        raise ValueError("restart trace does not match counters")
    if stats["dropped"]:
        raise ValueError("restart dropped open nodes")
    return {
        "solver_seconds": seconds,
        "rapid_attempts": attempts,
        "rapid_summary": stats,
        "learning_and_allowance": {
            label: parse_fields(by_label[label])
            for flag, label in RAPID_SECTIONS.items()
            if flag in command
        },
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
    out: Path,
    command: list[str],
    ref: Reference,
    time_limit: int,
    *,
    correctness_only: bool,
    rapid: bool,
) -> dict[str, object]:
    out.mkdir()
    log = out / "solver.log"
    write_json(out / "command.json", {"argv": command, "shell": shlex.join(command)})
    result: dict[str, object] = {"reference": asdict(ref)}
    started = None if correctness_only else time.monotonic()
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
        raw = log.read_text(encoding="utf-8", errors="replace")
        metrics = parse_metrics(raw)
        result.update(metrics)
        if rapid:
            result.update(parse_rapid_metrics(raw, int(str(metrics["conflict_restarts"])), command))
        check = check_reference(metrics, ref)
        if correctness_only and check == "consistent limit" and metrics["root_bound"] is None:
            check = "limit reached before a root bound was recorded"
        result["check"] = check
    except (IndexError, KeyError, ValueError) as exc:
        result["check"] = f"invalid solver output ({exc})"
    return result


def learning_counts(job: dict[str, object], *, terminal: bool) -> dict[str, dict[str, str]]:
    sections = cast("dict[str, dict[str, str]]", job["learning_and_allowance"])
    return {
        name: {
            key: value
            for key, value in fields.items()
            if key != "seconds" and (terminal or not key.startswith("tree_"))
        }
        for name, fields in sections.items()
        if terminal or name != "Restart overhead allowance"
    }


def search_trace(job: dict[str, object]) -> list[dict[str, object]]:
    attempts = cast("list[RapidAttempt]", job["rapid_attempts"])
    return [
        {
            "bounds": [
                {key: value for key, value in event.items() if key != "seconds"}
                for event in attempt["bounds"]
            ],
            "restarts": attempt["restarts"],
            "first_incumbent": {
                key: value
                for key, value in (attempt["first_incumbent"] or {}).items()
                if key != "seconds"
            },
        }
        for attempt in attempts
    ]


def compare_anchor(anchor: dict[str, object], result: dict[str, object], arm: str) -> bool:
    terminal = anchor["check"] == "match"
    outcome: tuple[str, ...] = ("root_bound", "sense", "input_continuous_cols")
    if terminal:
        outcome = (*outcome, *TERMINAL_OUTCOME)
    diff = [key for key in outcome if anchor[key] != result[key]]
    if diff:
        result["anchor_check"] = f"anchor outcome differs: {', '.join(diff)}"
        return False

    work = [key for key in TERMINAL_WORK if terminal and anchor[key] != result[key]]
    if learning_counts(anchor, terminal=terminal) != learning_counts(result, terminal=terminal):
        work.append("learning counters")
    if terminal and search_trace(anchor) != search_trace(result):
        work.append("search trace")
    if not work:
        return True
    detail = ", ".join(work)
    if arm in WALL_DEPENDENT_ARMS:
        result["anchor_check"] = f"wall-dependent counters differ: {detail}"
        return True
    result["anchor_check"] = f"anchor counters differ: {detail}"
    return False


def read_anchor_results(
    directory: Path, manifest: dict[str, object], jobs: list[str]
) -> dict[str, dict[str, object]]:
    saved = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if not saved.get("anchor"):
        raise SystemExit("Comparison directory must contain an anchor run")
    for field in (
        "suite",
        "arms",
        "runner_sha256",
        "solver_sha256",
        "reference_sha256",
        "solver_seconds_limit",
    ):
        if saved[field] != manifest[field]:
            raise SystemExit(f"Anchor manifest differs: {field}")
    instances = cast("dict[str, str]", manifest["instance_sha256"])
    for name, digest in instances.items():
        if saved["instance_sha256"].get(name) != digest:
            raise SystemExit(f"Anchor instance differs or is missing: {name}")
    with (directory / "results.jsonl").open(encoding="utf-8") as stream:
        results = {row["job"]: row for line in stream if (row := json.loads(line))}
    for job in jobs:
        if job not in results or results[job]["check"] not in ("match", "consistent limit"):
            raise SystemExit(f"Missing or failed anchor job: {job}")
    return results


def main() -> int:
    args = parse_args()
    binary = args.binary.resolve(strict=True)
    miplib_dir = args.miplib_dir.resolve()
    solu = (args.solu_file or miplib_dir / "miplib.solu").resolve(strict=True)
    refs = read_solu(solu)
    inputs = find_instances(miplib_dir)
    rapid = args.suite == "rapid-learning"
    arms = RAPID_ARMS if rapid else ARMS
    pilot_corpus = RAPID_PILOT if rapid else PILOT
    full_corpus = RAPID_CORPUS if rapid else CORPUS
    names = tuple(dict.fromkeys(args.instances or (pilot_corpus if args.pilot else full_corpus)))
    for name in names:
        if name not in inputs or name not in refs:
            raise SystemExit(f"Missing instance or solved MIPLIB reference for {name}")
        value = refs[name].value
        if value is not None and not math.isfinite(value):
            raise SystemExit(f"Nonfinite MIPLIB reference for {name}")
    limit = (3 if rapid else 1) if args.pilot else args.time_limit
    node_limit = (1800 if rapid else 200) if args.pilot else None
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
        "--print-rapid-learning-bench-stats" if rapid else "--print-conflict-bench-stats",
    ]
    if args.pilot:
        common += ["--node-limit", str(node_limit)]
    manifest = {
        "pilot": args.pilot,
        "anchor": args.anchor,
        "suite": args.suite,
        "arms": {name: list(flags) for name, flags in arms.items()},
        "instances": names,
        "node_limit": node_limit,
        "build_id": args.build_id,
        "host": platform.uname()._asdict(),
        "python": platform.python_version(),
        "runner_sha256": sha256(Path(__file__)),
        "solver_seconds_limit": limit,
        "process_seconds_limit": limit + 15,
        "solver_sha256": sha256(binary),
        "reference_sha256": sha256(solu),
        "instance_sha256": {name: sha256(inputs[name]) for name in names},
    }
    jobs = [f"{name}-{arm}" for name in names for arm in arms]
    anchors = (
        read_anchor_results(args.anchor_results, manifest, jobs) if args.anchor_results else {}
    )
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "manifest.json", manifest)
    failures = 0
    with (args.output / "results.jsonl").open("w", encoding="utf-8") as summary:
        for name in names:
            node_flags = (
                ["--node-limit", "1800"] if args.anchor and name in RAPID_SHORT_ANCHORS else []
            )
            for arm, flags in arms.items():
                job = f"{name}-{arm}"
                command = [str(binary), str(inputs[name]), *common, *node_flags, *flags]
                result = run_job(
                    args.output / job,
                    command,
                    refs[name],
                    limit,
                    correctness_only=args.pilot or args.anchor,
                    rapid=rapid,
                )
                passed = result["check"] in ("match", "consistent limit")
                if passed and args.anchor and result["status"] == "Time limit":
                    result["check"] = "pre-lock anchor reached the time limit"
                    passed = False
                if passed and job in anchors:
                    passed = compare_anchor(anchors[job], result, arm)
                result.update(instance=name, arm=arm, job=job)
                summary.write(json.dumps(result, allow_nan=False) + "\n")
                summary.flush()
                print(f"{name} {arm}: {result.get('anchor_check', result['check'])}", flush=True)
                failures += not passed
                if not passed and rapid and not args.pilot:
                    return 1
    return int(failures != 0)


if __name__ == "__main__":
    raise SystemExit(main())
