#!/usr/bin/env python3
"""Instance gate: solve Netlib LPs / MIPLIB MIPs and hold each result to its reference.

Every run is classified against the instance's `.solu` reference as one of

  match_1e-6   the reported verdict is right, objective within 1e-6 (relative)
  match_1e-4   the reported verdict is right, objective within 1e-4
  disagree     a wrong answer: an Optimal objective off by more than 1e-4, or a
               verdict (Optimal / Infeasible / Unbounded) that contradicts the
               reference
  dnf          did not finish: time or iteration limit, error, crash, timeout

and compared with the expected class checked in for it. The gate fails when

  - any run disagrees, whatever its expected class: a wrong answer is never
    tolerated by a list entry or a tolerance band; or
  - a run expected to match no longer does (a regression), or a listed
    instance's file is missing.

A run that does better than listed is reported so the list can be updated, and
does not fail the gate. Instances without a usable reference (=best=, =unkn=,
or none) are reported and never gate.

    run_instance_gate.py lp  [--backends dual,barrier,pdlp]   # Netlib, per LP backend
    run_instance_gate.py mip                                   # curated MIPLIB subset

By default only the instances in the expected list run. `--all` runs every
instance in the data directory that has a reference, and `--write-expected`
rewrites the list from the observed classes -- the "regenerate lists" step.

Exit status: 0 pass, 1 fail, 77 skipped because no instance files are present
(download them with tests/data/download_netlib.sh or
tests/data/download_miplib.sh --small). ctest treats 77 as a skip.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import math
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GATE_DIR = Path(__file__).resolve().parent / "instance_gate"

CLASSES = ("match_1e-6", "match_1e-4", "disagree", "dnf")
MATCHES = ("match_1e-6", "match_1e-4")
SKIP_EXIT = 77


@dataclass(frozen=True)
class Reference:
    kind: str  # "opt", "inf" or "unbd"
    value: float | None = None


@dataclass(frozen=True)
class Outcome:
    status: str  # optimal, infeasible, unbounded, gap_limit, time_limit, ..., timeout, crash
    objective: float | None


@dataclass(frozen=True)
class Expectation:
    instance: str
    backend: str
    expected: str
    time_limit: float | None


@dataclass(frozen=True)
class Suite:
    data_dir: Path
    solu: str
    expected: Path
    time_limit: float  # default per-instance limit, seconds
    backends: tuple[str, ...]
    download: str

    def references(self) -> dict[str, Reference]:
        path = self.data_dir / self.solu
        return read_solu(path) if path.is_file() else {}


def read_solu(path: Path) -> dict[str, Reference]:
    """Read a .solu file. Only =opt=, =inf= and =unbd= entries are references."""
    refs: dict[str, Reference] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0] == "=opt=":
            refs[parts[1]] = Reference("opt", float(parts[2]))
        elif len(parts) >= 2 and parts[0] in ("=inf=", "=unbd="):
            refs[parts[1]] = Reference(parts[0].strip("="))
    return refs


def find_instances(data_dir: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for suffix in (".mps.gz", ".mps"):
        for path in sorted(data_dir.glob(f"*{suffix}")):
            found.setdefault(path.name.removesuffix(suffix), path)
    return found


SUITES: dict[str, Suite] = {
    "lp": Suite(
        REPO / "tests" / "data" / "netlib",
        "netlib.solu",
        GATE_DIR / "lp_expected.csv",
        10.0,
        ("dual", "barrier", "pdlp"),
        "tests/data/download_netlib.sh",
    ),
    "mip": Suite(
        REPO / "tests" / "data" / "miplib",
        "miplib.solu",
        GATE_DIR / "mip_expected.csv",
        60.0,
        ("mip",),
        "tests/data/download_miplib.sh --small",
    ),
}


def read_expected(path: Path) -> list[Expectation]:
    if not path.is_file():
        return []
    rows: list[Expectation] = []
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(line for line in f if not line.startswith("#")):
            if row["expected"] not in CLASSES:
                raise SystemExit(f"{path}: unknown class {row['expected']!r} for {row['instance']}")
            limit = row.get("time_limit") or ""
            rows.append(
                Expectation(
                    row["instance"],
                    row["backend"],
                    row["expected"],
                    float(limit) if limit else None,
                )
            )
    return rows


def write_expected(path: Path, rows: list[Expectation], header: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(header)
        writer = csv.writer(f, lineterminator="\n")
        with_limits = any(row.time_limit is not None for row in rows)
        writer.writerow(
            ["instance", "backend", "expected"] + (["time_limit"] if with_limits else [])
        )
        for row in sorted(rows, key=lambda r: (r.instance, r.backend)):
            limit = [] if not with_limits else [f"{row.time_limit:g}" if row.time_limit else ""]
            writer.writerow([row.instance, row.backend, row.expected, *limit])


def parse_outcome(output: str) -> Outcome:
    # The CLI prints a "Status:" line per verdict; the last one is the summary.
    statuses = re.findall(r"^Status:\s*(.+?)\s*$", output, re.MULTILINE)
    status = statuses[-1].lower().replace(" ", "_") if statuses else "crash"
    match = re.search(r"^Objective:\s*([-+0-9.eE]+|inf|-inf)\s*$", output, re.MULTILINE)
    return Outcome(status, float(match.group(1)) if match else None)


def read_maximize(path: Path) -> bool:
    """True when the MPS file's OBJSENSE section (or line) says MAX.

    Only the header is scanned: OBJSENSE has to come before ROWS.
    """
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="latin-1") as f:
        in_objsense = False
        for line in f:
            words = line.split()
            if not words or line.startswith(("*", "NAME")):
                continue
            head = words[0].upper()
            if head == "ROWS":
                return False
            if head == "OBJSENSE":
                if len(words) > 1:
                    return words[1].upper().startswith("MAX")
                in_objsense = True
            elif in_objsense:
                return head.startswith("MAX")
    return False


def limited_incumbent_is_wrong(outcome: Outcome, ref: Reference, maximize: bool) -> bool:
    """A MIP stopped by a limit still reports its incumbent. It is a wrong
    answer when it is better than the proven optimum by more than 1e-4, or
    exists at all for an infeasible instance: either way the solver accepted an
    infeasible point."""
    if outcome.status not in ("time_limit", "node_limit") or outcome.objective is None:
        return False
    if not math.isfinite(outcome.objective):
        return False
    if ref.kind == "inf":
        return True
    if ref.kind != "opt" or ref.value is None:
        return False
    tol = 1e-4 * max(1.0, abs(ref.value))
    if maximize:
        return outcome.objective > ref.value + tol
    return outcome.objective < ref.value - tol


def classify(outcome: Outcome, ref: Reference, maximize: bool = False) -> str:
    claims = {"optimal": "opt", "gap_limit": "opt", "infeasible": "inf", "unbounded": "unbd"}
    claim = claims.get(outcome.status)
    if claim is None:
        return "disagree" if limited_incumbent_is_wrong(outcome, ref, maximize) else "dnf"
    if claim != ref.kind:
        return "disagree"
    if claim != "opt":
        return "match_1e-6"
    assert ref.value is not None
    if outcome.objective is None:
        return "disagree"
    err = abs(outcome.objective - ref.value) / max(1.0, abs(ref.value))
    if err <= 1e-6:
        return "match_1e-6"
    return "match_1e-4" if err <= 1e-4 else "disagree"


def solver_args(backend: str, time_limit: float) -> list[str]:
    args = ["--quiet", "--threads", "1", "--time-limit", f"{time_limit:g}"]
    if backend == "mip":
        # Close the gap, so "Optimal" means proven and the objective can be
        # held to 1e-6: at the default 1e-4 a legitimate gap-limit stop can sit
        # anywhere within 1e-4 of the optimum.
        args += ["--gap-tol", "1e-9"]
    else:
        args.append(f"--{backend}")
    return args


def run_one(binary: Path, instance: Path, backend: str, time_limit: float) -> Outcome:
    cmd = [str(binary), str(instance), *solver_args(backend, time_limit)]
    # The solver's own limit is the real one; the wall-clock guard only
    # catches a hang, so it is generous.
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=2 * time_limit + 30, check=False
        )
    except subprocess.TimeoutExpired:
        return Outcome("timeout", None)
    outcome = parse_outcome(proc.stdout + "\n" + proc.stderr)
    if proc.returncode != 0 and outcome.status == "crash":
        return Outcome("crash", None)
    return outcome


def verdict(expected: str | None, observed: str) -> str:
    """How one result reads against its list entry: ok, fail, better or unlisted."""
    if observed == "disagree":
        return "fail"
    if expected is None:
        return "unlisted"
    rank = {"match_1e-6": 0, "match_1e-4": 1, "dnf": 2, "disagree": 3}
    if rank[observed] > rank[expected] and expected in MATCHES:
        return "fail"
    return "better" if rank[observed] < rank[expected] else "ok"


@dataclass
class Plan:
    runs: list[tuple[str, str, float]]  # instance, backend, time limit
    expected: dict[tuple[str, str], str]  # the selected backends' list entries
    missing: list[tuple[str, str]]  # listed, but no instance file
    unreferenced: list[str]  # --all instances without a usable reference
    no_reference: list[tuple[str, str]] = field(default_factory=list)  # listed, no reference


def selected_backends(args: argparse.Namespace, suite: Suite) -> list[str]:
    chosen = [b for b in args.backends.split(",") if b]
    unknown = sorted(set(chosen) - set(suite.backends))
    if unknown:
        raise SystemExit(f"unknown backend(s) for {args.suite}: {', '.join(unknown)}")
    return chosen or list(suite.backends)


def build_plan(args: argparse.Namespace, suite: Suite, listed: list[Expectation]) -> Plan:
    instances = find_instances(suite.data_dir)
    refs = suite.references()
    backends = selected_backends(args, suite)
    default_limit = args.time_limit if args.time_limit else suite.time_limit

    plan = Plan([], {}, [], [])
    for row in listed:
        if row.backend not in backends:
            continue
        plan.expected[(row.instance, row.backend)] = row.expected
        if row.instance not in instances:
            plan.missing.append((row.instance, row.backend))
        elif row.instance not in refs:
            # Gating nothing must not read as a pass: a .solu that failed to
            # download, or an entry that lost its =opt= value.
            plan.no_reference.append((row.instance, row.backend))
        else:
            plan.runs.append((row.instance, row.backend, row.time_limit or default_limit))
    if args.all:
        for name in sorted(instances):
            if name not in refs:
                plan.unreferenced.append(name)
                continue
            for backend in backends:
                if (name, backend) not in plan.expected:
                    plan.runs.append((name, backend, default_limit))
    return plan


def print_summary(results: dict[tuple[str, str], str], backends: list[str]) -> None:
    print("\nSummary (per backend):")
    print(f"  {'backend':<8} " + " ".join(f"{c:>10}" for c in CLASSES))
    for backend in backends:
        counts = Counter(cls for (_, b), cls in results.items() if b == backend)
        print(f"  {backend:<8} " + " ".join(f"{counts.get(c, 0):>10}" for c in CLASSES))


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("suite", choices=sorted(SUITES))
    p.add_argument("--binary", default=str(REPO / "build" / "mipx-solve"))
    p.add_argument("--backends", default="", help="Comma-separated subset (lp: dual,barrier,pdlp).")
    p.add_argument("--time-limit", type=float, default=0.0, help="Per-instance limit (s).")
    p.add_argument("--data-dir", default="", help="Instance directory (default: the suite's).")
    p.add_argument("--expected-file", default="", help="Expected list (default: the suite's).")
    p.add_argument("--all", action="store_true", help="Also run every unlisted instance.")
    p.add_argument("--write-expected", action="store_true", help="Rewrite the list from this run.")
    return p.parse_args(argv)


def execute(args: argparse.Namespace, suite: Suite, plan: Plan) -> dict[tuple[str, str], str]:
    binary = Path(args.binary)
    instances = find_instances(suite.data_dir)
    refs = suite.references()
    results: dict[tuple[str, str], str] = {}
    for name, backend, limit in plan.runs:
        ref = refs.get(name)
        if ref is None:
            print(f"  {name:<12} {backend:<8} no reference in .solu; not gated")
            continue
        outcome = run_one(binary, instances[name], backend, limit)
        cls = classify(outcome, ref, read_maximize(instances[name]))
        results[(name, backend)] = cls
        tag = verdict(plan.expected.get((name, backend)), cls)
        obj = "" if outcome.objective is None else f" obj={outcome.objective:.10g}"
        print(f"  {name:<12} {backend:<8} {cls:<11} [{tag}] status={outcome.status}{obj}")
    return results


def report(plan: Plan, results: dict[tuple[str, str], str], list_written: bool) -> int:
    failures = [
        key for key, cls in results.items() if verdict(plan.expected.get(key), cls) == "fail"
    ]
    better = [
        key for key, cls in results.items() if verdict(plan.expected.get(key), cls) == "better"
    ]
    for name, backend in plan.missing:
        print(f"MISSING: {name} ({backend}) is listed but its instance file is absent")
    for name, backend in plan.no_reference:
        print(f"MISSING: {name} ({backend}) is listed but has no usable .solu reference")
    if plan.unreferenced:
        print(f"No usable .solu reference, not run: {', '.join(plan.unreferenced)}")
    hint = "list updated" if list_written else "update the list (--write-expected)"
    for name, backend in better:
        print(f"BETTER than listed: {name} ({backend}) -> {results[(name, backend)]}; {hint}")
    for name, backend in failures:
        expected = plan.expected.get((name, backend), "unlisted")
        print(f"FAIL: {name} ({backend}) {results[(name, backend)]}, expected {expected}")
    ok = not failures and not plan.missing and not plan.no_reference
    print("Instance gate: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    suite = SUITES[args.suite]
    if args.data_dir:
        suite = replace(suite, data_dir=Path(args.data_dir))
    if args.expected_file:
        suite = replace(suite, expected=Path(args.expected_file))
    if not find_instances(suite.data_dir):
        print(
            f"SKIPPED: instance gate '{args.suite}': no instances in {suite.data_dir}. "
            f"Download them with {suite.download}.",
            file=sys.stderr,
        )
        return SKIP_EXIT
    if not Path(args.binary).is_file():
        raise SystemExit(f"Solver binary not found: {args.binary} (build first)")

    listed = read_expected(suite.expected)
    plan = build_plan(args, suite, listed)
    backends = selected_backends(args, suite)
    print(f"Instance gate '{args.suite}': {len(plan.runs)} runs, backends {','.join(backends)}")
    results = execute(args, suite, plan)
    print_summary(results, backends)

    if args.write_expected:
        # Every existing entry is kept -- other backends', and listed instances
        # that did not run -- and overwritten only by what this run observed.
        # Per-row time limits are kept too.
        merged = {(row.instance, row.backend): row for row in listed}
        for (name, backend), cls in results.items():
            old = merged.get((name, backend))
            merged[(name, backend)] = Expectation(
                name, backend, cls, old.time_limit if old else None
            )
        script = "tests/perf/run_instance_gate.py"
        header = (
            "# Expected class per instance and backend for run_instance_gate.py.\n"
            f"# Refresh these rows: {script} {args.suite} --write-expected\n"
            f"# Add every instance with a reference: {script} {args.suite} --all --write-expected\n"
        )
        write_expected(suite.expected, list(merged.values()), header)
        print(f"Wrote {suite.expected}")
    return report(plan, results, args.write_expected)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
