#!/usr/bin/env python3
"""Generate the later column-generation masters of the CPU barrier gate (#358).

Writes, for each of the 15 public CommaLab instances of the cold LP panel
(#272: grid1-9, planar30/50/80/100/150/300), the first later master LP with
a path column linking demand and capacity rows as `<name>.mps`, and
`cg_masters.solu` with the corresponding objective
HiGHS reported for it (simplex, presolve off, as mcf records it), in `.solu`
format for run_instance_gate.py's `barrier-cg` suite.

mcf is https://github.com/tinnet-ai/mcf (used at 220af3c). With `lp_dump_every=1`,
its recorder applies changes to its mirror LP and writes `master_NNNN.mps`
immediately before solve NNNN, whose result appears in `ops.txt` (format in
tests/perf/README.md, "LP Sequence Replay").

    generate_cg_masters.py --mcf PATH/mcf --instances PATH/data/commalab \\
        [--out tests/data/cg_masters_later]

`--instances` is mcf's `data/commalab` directory (`grid/gridN.gz`,
`planar/planarN.gz`). Exit status: 0 when every master and the `.solu` were
written; 1 when an mcf run or its dump failed, and then nothing in `--out` is
changed: new masters beside an old `.solu`, or the reverse, would read as a
barrier regression.
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "tests" / "data" / "cg_masters_later"
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


def has_capacity_path(mps: Path) -> bool:
    """Find a path column touching both a demand (G) and a capacity (L) row."""
    row_types: dict[str, str] = {}
    path_sides: dict[str, set[str]] = {}
    section = ""
    for line in mps.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] in MPS_SECTIONS:
            section = fields[0]
        elif section == "ROWS":
            sense, row = fields
            row_types[row] = sense
        elif section == "COLUMNS":
            # The recorder writes one row/value pair on each line.
            col, row, value = fields
            side = row_types.get(row)
            if side is None:
                raise ValueError(f"{mps.name}: column {col} references unknown row {row}")
            if side in {"G", "L"} and float(value) != 0.0:
                path_sides.setdefault(col, set()).add(side)
    # The recorder writes free rows as N, which the MPS reader drops.
    if (n_rows := list(row_types.values()).count("N")) != 1:
        raise ValueError(f"{mps.name}: master has {n_rows} N rows, not just the objective")
    # CG's constraint coefficients are nonnegative, so the shared column
    # contributes a nonzero off-diagonal entry to A D A^T.
    return any(sides == {"G", "L"} for sides in path_sides.values())


def later_master(lp: Path) -> tuple[Path, float]:
    """Choose the earliest coupled later snapshot and its recorded objective."""
    next_seq = 0
    for line in (lp / "ops.txt").read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts or parts[0] != "solve":
            # A snapshot already includes all preceding edits.
            continue
        if len(parts) != 6 or int(parts[1]) != next_seq:
            raise ValueError(f"ops.txt: expected solve {next_seq}, got {line!r}")
        seq = next_seq
        next_seq += 1
        if seq == 0:
            continue
        mps = lp / f"master_{seq:04d}.mps"
        if not has_capacity_path(mps):
            continue
        status, objective = parts[2], float(parts[3])
        if status != "optimal" or not math.isfinite(objective):
            raise ValueError(f"solve {seq}: expected a finite optimal objective, got {line!r}")
        return mps, objective
    raise ValueError("no later master with a path linking demand and capacity rows")


def dump_master(mcf: Path, source: Path, dump: Path) -> Path:
    """Run a few cg iterations of mcf; return the first LP's snapshot directory."""
    cmd = [
        str(mcf),
        str(source),
        "--method",
        "cg",
        "--backend",
        "highs",
        "--highs-method",
        "simplex",
        "--threads",
        "1",
        "--no-sidecar",
        "--cg-set",
        f"lp_dump={dump}",
        "--cg-set",
        "lp_dump_every=1",
        "--cg-set",
        "max_iterations=5",
    ]
    # mcf can report a failed exact check at the iteration limit. The dump's
    # individual LP solve records determine whether it contains a usable master.
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
    """Dump one instance's later master and return its MPS and reference objective.

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
    return later_master(lp)


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
        else:
            print(f"{name}: {mps.name}")
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
