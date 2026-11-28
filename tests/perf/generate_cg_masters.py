#!/usr/bin/env python3
"""Generate the barrier-cg v1 panel from the pinned CommaLAB archives."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import io
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_INPUTS = REPO / "tests" / "data" / "commalab"
CHECKSUMS = DEFAULT_INPUTS.parent / "commalab_archives.sha256"
REFERENCES = Path(__file__).resolve().parent / "instance_gate" / "barrier_cg_v1.solu"
MASTER_CHECKSUMS = REFERENCES.with_suffix(".sha256")
SOLU = "cg_masters.solu"
CASES = {
    "grid.tgz": [f"grid{i}" for i in range(1, 10)],
    "planar.tgz": [f"planar{n}" for n in (30, 50, 80, 100, 150, 300)],
}

Arc = tuple[int, int, int, int]  # tail, head, cost, capacity
Demand = tuple[int, int, int]  # origin, destination, volume
Graph = list[list[tuple[int, int, int]]]  # arc index, head, cost
Tree = tuple[dict[int, int], dict[int, tuple[int, int]]]  # distances, predecessors


def read_instance(data: bytes) -> tuple[int, list[Arc], list[Demand]]:
    nodes, arc_count, demand_count, *values = map(int, data.split())
    arc_end = 4 * arc_count
    if len(values) != arc_end + 3 * demand_count:
        raise ValueError("incorrect arc or commodity count")
    arcs = []
    for i in range(0, arc_end, 4):
        tail, head, cost, capacity = values[i : i + 4]
        arcs.append((tail, head, cost, capacity))
    demands = []
    for i in range(arc_end, len(values), 3):
        origin, destination, volume = values[i : i + 3]
        demands.append((origin, destination, volume))
    return nodes, arcs, demands


def shortest_tree(graph: Graph, origin: int) -> Tree:
    distance = {origin: 0}
    predecessor: dict[int, tuple[int, int]] = {}
    pending = [(0, origin)]
    while pending:
        cost, node = heapq.heappop(pending)
        if cost != distance[node]:
            continue
        # Ties keep the first route: nodes are popped by (distance, node ID)
        # and outgoing arcs are visited in the order of the input file.
        for arc, head, length in graph[node]:
            candidate = cost + length
            if head not in distance or candidate < distance[head]:
                distance[head] = candidate
                predecessor[head] = arc, node
                heapq.heappush(pending, (candidate, head))
    return distance, predecessor


def price_paths(nodes: int, arcs: list[Arc], demands: list[Demand]) -> list[tuple[int, list[int]]]:
    graph: Graph = [[] for _ in range(nodes + 1)]  # CommaLAB node IDs start at 1.
    for index, (tail, head, cost, _) in enumerate(arcs):
        graph[tail].append((index, head, cost))

    trees: dict[int, Tree] = {}
    paths = []
    for origin, destination, _ in demands:
        if origin not in trees:
            trees[origin] = shortest_tree(graph, origin)
        distance, predecessor = trees[origin]
        if destination not in distance:
            raise ValueError(f"node {destination} is unreachable from node {origin}")
        cost = distance[destination]
        route = []
        node = destination
        while node != origin:
            arc, node = predecessor[node]
            route.append(arc)
        route.reverse()
        paths.append((cost, route))
    return paths


def master_mps(source: bytes) -> bytes:
    nodes, arcs, demands = read_instance(source)
    paths = price_paths(nodes, arcs, demands)
    used_arcs = sorted({arc for _, route in paths for arc in route})
    if not used_arcs:
        raise ValueError("later master has no path linking demand and capacity rows")
    # With only artificial columns, the initial master has this demand dual
    # for every commodity and zero capacity duals. One shortest-path pricing
    # sweep adds a negative reduced-cost path for each demand.
    penalty = 2 * max(cost for cost, _ in paths) + 1
    with io.StringIO(newline="\n") as out:
        print("NAME master\nROWS\n N obj", file=out)
        for i in range(len(demands)):
            print(f" G d{i}", file=out)
        for arc in used_arcs:
            print(f" L a{arc}", file=out)
        print("COLUMNS", file=out)
        for i, (cost, route) in enumerate(paths):
            print(f" u{i} obj {penalty}\n u{i} d{i} 1", file=out)
            print(f" p{i} obj {cost}\n p{i} d{i} 1", file=out)
            for arc in route:
                print(f" p{i} a{arc} 1", file=out)
        print("RHS", file=out)
        for i, (_, _, volume) in enumerate(demands):
            print(f" rhs d{i} {volume}", file=out)
        for arc in used_arcs:
            print(f" rhs a{arc} {arcs[arc][3]}", file=out)
        print("ENDATA", file=out)
        return out.getvalue().encode("ascii")


def read_checksums(path: Path) -> dict[str, str]:
    checksums = {}
    for line in path.read_text(encoding="ascii").splitlines():
        checksum, name = line.split()
        checksums[name] = checksum
    return checksums


def read_archives(inputs: Path) -> dict[str, bytes]:
    archives = {}
    for name, checksum in read_checksums(CHECKSUMS).items():
        path = inputs / name
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != checksum:
            raise ValueError(f"incorrect SHA-256 for {path}; rerun tests/data/download_commalab.sh")
        archives[name] = data
    return archives


def generate_panel(inputs: Path, out: Path) -> None:
    archives = read_archives(inputs)
    checksums = read_checksums(MASTER_CHECKSUMS)
    masters = {}
    for name, cases in CASES.items():
        with tarfile.open(fileobj=io.BytesIO(archives[name]), mode="r:gz") as archive:
            for case in cases:
                try:
                    source = archive.extractfile(case)
                except KeyError as error:
                    raise ValueError(f"{case} is missing from {name}") from error
                if source is None:
                    raise ValueError(f"{case} is not a file in {name}")
                with source:
                    model = master_mps(source.read())
                filename = f"{case}.mps"
                if hashlib.sha256(model).hexdigest() != checksums.get(filename):
                    raise ValueError(f"generated {filename} differs from the pinned v1 model")
                masters[filename] = model

    references = REFERENCES.read_bytes()
    out.mkdir(parents=True, exist_ok=True)
    for filename, model in masters.items():
        (out / filename).write_bytes(model)
    (out / SOLU).write_bytes(references)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=Path, default=DEFAULT_INPUTS, help="archive directory")
    parser.add_argument("--out", type=Path, required=True, help="output for MPS and .solu files")
    args = parser.parse_args()
    try:
        generate_panel(args.instances, args.out)
    except (OSError, ValueError, tarfile.TarError) as error:
        parser.exit(1, f"Cannot generate barrier-cg v1 panel: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
