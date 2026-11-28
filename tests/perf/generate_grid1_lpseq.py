#!/usr/bin/env python3
"""Build the nine-solve grid1 replay from the checksum-pinned public archive."""

from __future__ import annotations

import argparse
import hashlib
import io
import tarfile
from pathlib import Path
from typing import TYPE_CHECKING

from generate_cg_masters import DEFAULT_INPUTS, price_paths, read_archives, read_instance

if TYPE_CHECKING:
    from generate_cg_masters import Demand

# Reference objectives for the nine deterministic masters, cross-checked
# with an independent simplex solve of each snapshot.
OBJECTIVES = (2833, 2698277, 2745429, 1463592, 1266620, 1140715, 1112052, 1117442, 1122598)
CHECKSUMS = {
    "manifest.txt": "1e931eeb53f22b9dfa516e9830d741f1c313d6295fccd90f66311064b84b27ba",
    "master_0000.mps": "5ad68b6c3917e6664f5233ade3c6f9c75633367eca839be37a6eb218b74defea",
    "ops.txt": "849be45acb08488c1688cb2c64dedd16691442a67d2c86a26b65b0d923067f9c",
}


def initial_master(demands: list[Demand], paths: list[tuple[int, list[int]]]) -> str:
    count = len(demands)
    with io.StringIO() as out:
        print("NAME grid1\nROWS\n N obj", file=out)
        for i in range(count):
            print(f" G r{i}", file=out)
        print("COLUMNS", file=out)
        for i in range(count):
            print(f" c{i} obj 1\n c{i} r{i} 1", file=out)
        for i, (cost, _) in enumerate(paths[:10]):
            print(f" c{count + i} obj {cost}\n c{count + i} r{i} 1", file=out)
        print("RHS", file=out)
        for i, (_, _, volume) in enumerate(demands):
            print(f" rhs r{i} {volume}", file=out)
        print("ENDATA", file=out)
        return out.getvalue()


def generate_sequence(source: bytes, out: Path) -> None:
    nodes, arcs, demands = read_instance(source)
    # Largest demands first, keeping archive order for ties. The initial ten
    # paths then overload some of the arcs whose capacity rows enter at solve 2.
    demands.sort(key=lambda demand: -demand[2])
    paths = price_paths(nodes, arcs, demands)
    count = len(demands)
    penalty = 2 * max(cost for cost, _ in paths) + 1
    # Start with ten priced paths. Add capacity rows for arcs they use first,
    # so later columns enter a master that already has shared constraints.
    initial_arcs = sorted({arc for _, route in paths[:10] for arc in route})[:10]
    capacity_rows: list[int] = []
    path_columns = list(range(10))

    with io.StringIO() as ops:

        def solve(seq: int) -> None:
            print(f"solve {seq} optimal {OBJECTIVES[seq]} 0 0", file=ops)

        def add_rows(new_arcs: list[int]) -> None:
            if set(new_arcs).intersection(capacity_rows):
                raise ValueError("capacity arc already has a row")
            print(f"add_rows {len(new_arcs)}", file=ops)
            for arc in new_arcs:
                entries = [
                    f"{count + j} 1" for j, i in enumerate(path_columns) if arc in paths[i][1]
                ]
                print(f"-inf {arcs[arc][3]} {len(entries)}", *entries, file=ops)
            capacity_rows.extend(new_arcs)

        def add_paths(new_paths: range) -> None:
            print(f"add_cols {len(new_paths)}", file=ops)
            for i in new_paths:
                cost, route = paths[i]
                entries = [f"{count + j} 1" for j, arc in enumerate(capacity_rows) if arc in route]
                print(f"{cost} 0 inf {1 + len(entries)} {i} 1", *entries, file=ops)
            path_columns.extend(new_paths)

        solve(0)
        print(f"col_costs {count}", *(f"{i} {penalty}" for i in range(count)), file=ops)
        solve(1)
        add_rows(initial_arcs)
        solve(2)
        add_paths(range(10, 31))
        solve(3)
        # These grid1 arcs are shared by many of the priced paths.
        add_rows([40])
        add_paths(range(31, 38))
        solve(4)
        add_paths(range(38, 47))
        solve(5)
        add_paths(range(47, 49))
        solve(6)
        add_rows([56])
        solve(7)
        print(f"del_cols 2 {count} {count + 1}", file=ops)
        print(f"del_rows 2 {count} {count + 1}", file=ops)
        del path_columns[:2]
        del capacity_rows[:2]
        add_paths(range(49, 50))
        solve(8)

        files = {
            "manifest.txt": b"format = lpseq 1\nsense = min\n",
            "master_0000.mps": initial_master(demands, paths).encode("ascii"),
            "ops.txt": ops.getvalue().encode("ascii"),
        }
    for name, data in files.items():
        if hashlib.sha256(data).hexdigest() != CHECKSUMS[name]:
            raise ValueError(f"generated {name} differs from the pinned grid1 fixture")
    out.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (out / name).write_bytes(data)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=Path, default=DEFAULT_INPUTS, help="archive directory")
    parser.add_argument("--out", type=Path, required=True, help="LP sequence directory")
    args = parser.parse_args()
    try:
        archive_data = read_archives(args.instances)["grid.tgz"]
        with tarfile.open(fileobj=io.BytesIO(archive_data), mode="r:gz") as archive:
            source = archive.extractfile("grid1")
            if source is None:
                raise ValueError("grid1 is not a file in grid.tgz")
            with source:
                generate_sequence(source.read(), args.out)
    except (KeyError, OSError, ValueError, tarfile.TarError) as error:
        parser.exit(1, f"Cannot generate grid1 replay: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
