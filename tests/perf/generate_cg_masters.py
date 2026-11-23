#!/usr/bin/env python3
"""Generate the column-generation masters of the CPU barrier gate (#331).

Writes, for each of the 15 public CommaLab instances of the cold LP panel
(#272: grid1-9, planar30/50/80/100/150/300), the first master LP of mcf's
column generation as `<name>.mps`, and `cg_masters.solu` with the objective
HiGHS reported for it (simplex, presolve off, as mcf records it), in `.solu`
format for run_instance_gate.py's `barrier-cg` suite.

mcf is https://github.com/tinnet-ai/mcf; its `--cg-set lp_dump=DIR` writes the
master as `lp_0000/master_0000.mps` with the solves in `ops.txt` (format in
tests/perf/README.md, "LP Sequence Replay"). One cg iteration is enough: the
first master is built and solved before any pricing. The masters of #272 came
from mcf 220af3c; at that commit the first masters of grid9 and planar300 come
out byte for byte as the full-run dumps.

A first master holds only the initial columns, one nonzero each, so its
normal equations are diagonal: the suite checks the barrier's answers on
these LPs, not its sparse factorization's fill.

    generate_cg_masters.py --mcf PATH/mcf --instances PATH/data/commalab \\
        [--out tests/data/cg_masters]

`--instances` is mcf's `data/commalab` directory (`grid/gridN.gz`,
`planar/planarN.gz`). Exit status: 0 when every master and the `.solu` were
written; 1 when an mcf run or its dump failed, and then nothing in `--out` is
changed: new masters beside an old `.solu`, or the reverse, would read as a
barrier regression.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "tests" / "data" / "cg_masters"
SOLU = "cg_masters.solu"

INSTANCES = [f"grid{i}" for i in range(1, 10)] + [f"planar{n}" for n in (30, 50, 80, 100, 150, 300)]


def instance_file(instances: Path, name: str) -> Path:
    family = "grid" if name.startswith("grid") else "planar"
    return instances / family / f"{name}.gz"


def read_manifest(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        if sep:
            entries[key.strip()] = value.strip()
    return entries


MPS_SECTIONS = {"NAME", "ROWS", "COLUMNS", "RHS", "RANGES", "BOUNDS", "OBJSENSE", "ENDATA"}


def objective_rows(mps: Path) -> int:
    """Number of N rows in ROWS: the recorder writes a free row as one.

    The recorder starts data lines in column 1 too, so a section is told by
    its keyword, not by indentation.
    """
    count = 0
    in_rows = False
    for line in mps.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] in MPS_SECTIONS:
            in_rows = fields[0] == "ROWS"
        elif in_rows and fields[0] == "N":
            count += 1
    return count


def first_solve(ops: Path) -> tuple[str, float]:
    """Status and objective of the dump's first op, its solve (`solve <seq> <status> <obj> ...`).

    Only a solve before any change op is a solve of the MPS itself.
    """
    for line in ops.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] != "solve" or len(parts) < 4:
            raise ValueError(f"{ops}: first op is not a solve: {line!r}")
        return parts[2], float(parts[3])
    raise ValueError(f"{ops}: no solve line")


def dump_master(mcf: Path, source: Path, dump: Path) -> Path:
    """Run one cg iteration of mcf with lp_dump; return the dump's lp_0000 directory."""
    cmd = [
        str(mcf),
        str(source),
        "--method",
        "cg",
        "--threads",
        "1",
        "--no-sidecar",
        "--cg-set",
        f"lp_dump={dump}",
        "--cg-set",
        "max_iterations=1",
    ]
    # One iteration ends at mcf's iteration limit, which it reports as a
    # failed exact check (exit status not 0): the dump is what is checked.
    # A first master takes seconds; the timeout only ends a hung run.
    try:
        run = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=600)
    except subprocess.TimeoutExpired as err:
        raise RuntimeError(f"mcf ran past {err.timeout} s on {source}") from err
    lp = dump / "lp_0000"
    if not (lp / "master_0000.mps").is_file() or not (lp / "ops.txt").is_file():
        tail = (run.stderr or run.stdout).strip()[-500:]
        raise RuntimeError(
            f"mcf wrote no master for {source} (exit {run.returncode}: {tail}; {' '.join(cmd)})"
        )
    return lp


def checked_master(mcf: Path, source: Path, dump: Path) -> tuple[Path, float]:
    """Dump one instance's first master; return its MPS and reference objective.

    Raises RuntimeError, ValueError or OSError (no mcf at that path, a dump
    file missing) when the master cannot serve as a gate instance.
    """
    if not source.is_file():
        raise RuntimeError(f"no instance file {source}")
    lp = dump_master(mcf, source, dump)
    manifest = read_manifest(lp / "manifest.txt")
    if manifest.get("format") != "mcf-lpseq 1":
        raise RuntimeError(f"unsupported dump format: {manifest.get('format')}")
    # The reference must be the MPS's own objective: refuse what mipx-solve
    # would read differently from what HiGHS solved.
    if manifest.get("sense") != "min" or float(manifest.get("obj_offset", "0")) != 0.0:
        raise RuntimeError(f"sense or offset not min/0: {manifest}")
    mps = lp / "master_0000.mps"
    # The recorder writes a free row as a second N row, which the reader drops.
    if (n_rows := objective_rows(mps)) != 1:
        raise RuntimeError(f"master has {n_rows} N rows, not just the objective")
    status, objective = first_solve(lp / "ops.txt")
    if status != "optimal":
        raise RuntimeError(f"first solve is {status}, not optimal")
    return mps, objective


def generate(mcf: Path, instances: Path, out: Path, workdir: Path) -> tuple[list[str], list[str]]:
    """Dump and check every master; copy them all to out only if none failed.

    Returns (.solu lines, failures); the caller writes the .solu.
    """
    solu: list[str] = []
    failures: list[str] = []
    masters: list[tuple[Path, str]] = []
    for name in INSTANCES:
        try:
            mps, objective = checked_master(mcf, instance_file(instances, name), workdir / name)
        except (RuntimeError, ValueError, OSError) as err:
            failures.append(f"{name}: {err}")
            continue
        masters.append((mps, name))
        solu.append(f"=opt= {name} {objective!r}")
    if failures:
        return solu, failures
    for mps, name in masters:
        try:
            shutil.copyfile(mps, out / f"{name}.mps")
        except OSError as err:
            failures.append(f"{name}: {err}")
    return solu, failures


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--mcf", required=True, type=Path, help="mcf executable")
    p.add_argument("--instances", required=True, type=Path, help="mcf's data/commalab")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".dump_", dir=args.out) as tmp:
        solu, failures = generate(args.mcf, args.instances, args.out, Path(tmp))
    for failure in failures:
        print(f"FAILED: {failure}", file=sys.stderr)
    if failures:
        print(f"{args.out} left as it was", file=sys.stderr)
        return 1
    (args.out / SOLU).write_text("".join(line + "\n" for line in solu), encoding="utf-8")
    print(f"wrote {len(solu)} masters and {SOLU} to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
