# mipx

A branch-and-cut MIP solver built from scratch in C++23 with Python bindings.

> **Work in progress.** This project is under active development.
> Built entirely with [Claude Code](https://docs.anthropic.com/en/docs/claude-code).

## Performance

On Netlib LP instances, mipx is roughly 2-5x slower than HiGHS on medium problems and 10-40x on the hardest ones. On easy MIPLIB MIPs, it's typically 2-8x slower.

## Features

| Component | Description |
|-----------|-------------|
| **Dual simplex** | Devex pricing, bound-flipping ratio test, Forrest-Tomlin LU, SIMD kernels |
| **Barrier** | Interior-point with CPU and GPU (cuDSS) backends |
| **PDLP** | First-order primal-dual method with CPU and CUDA paths |
| **Branch-and-bound** | Best-first, depth-first, best-estimate node selection, reliability branching with pseudocosts |
| **Cutting planes** | Gomory MIR, cover, clique, zero-half, implied bound, mixing cuts |
| **Presolve** | Variable fixing, singleton elimination, dual fixing, coefficient tightening, bounds propagation |
| **Heuristics** | Rounding, diving, feasibility pump, RINS, RENS, local branching |
| **Parallel search** | TBB-based tree exploration with deterministic and opportunistic modes |
| **Symmetry** | Orbit detection, symmetry-breaking constraints, canonical branching |
| **Exact refinement** | Optional extended-precision LP repair for numerical certification |
| **Concurrent root** | Race dual simplex, barrier, and PDLP at root with cooperative stop |
| **Python API** | nanobind bindings for model I/O and solve |

## Build

Requires C++23 and CMake 3.25+.

```bash
cmake -B build && cmake --build build -j$(nproc)
```

Optional dependencies:

```bash
# TBB for parallel tree search
cmake -B build -DMIPX_USE_TBB=ON && cmake --build build -j$(nproc)

# CUDA for GPU-accelerated barrier and PDLP
cmake -B build -DMIPX_USE_CUDA=ON && cmake --build build -j$(nproc)
```

## Usage

```bash
./build/mipx-solve instance.mps
./build/mipx-solve instance.lp
./build/mipx-solve instance.mps --time-limit 300 --threads 4
```

The reader is chosen from the extension, case-insensitively: `.lp` is read as
CPLEX LP, anything else as MPS, ignoring a `.gz`/`.bz2` suffix
(`instance.mps.gz` is MPS). `--format` overrides that. A file that is not in
the format it is read as fails with an error instead of quietly solving as an
empty or nonsense model.

An RHS entry on the MPS objective row is read as the *negated* objective
constant (`obj_offset = -RHS`), as HiGHS and CPLEX do; the Netlib reference
values in `tests/data/netlib.solu` follow the same convention.

Key options:

| Option | Default | Description |
|--------|---------|-------------|
| `--format auto\|mps\|lp` | auto | Force the model file reader |
| `--time-limit <s>` | 3600 | Wall-clock time limit |
| `--threads <n>` | 1 | Parallel tree-search threads |
| `--gap-tol <g>` | 1e-4 | Relative optimality gap tolerance |
| `--node-limit <n>` | 1000000 | Maximum branch-and-bound nodes |
| `--presolve` / `--no-presolve` | on | Presolve reductions |
| `--cuts` / `--no-cuts` | on | Cutting planes |
| `--barrier` | off | Use barrier for root LP |
| `--pdlp` | off | Use PDLP for root LP |
| `--concurrent-root` | off | Race LP solvers at root |
| `--quiet` | off | Suppress progress output |
| `--probing` / `--no-probing` | off | Root probing in the solver (fixings, implications, variable bounds) |
| `--probing-rounds <n>` / `--probing-probes <n>` / `--probing-time <s>` | 3 / 10000 / 30 | Its budget: rounds, probes per round, seconds |
| `--presolve-probing` / `--no-presolve-probing` | off | Probing pass inside presolve |
| `--presolve-probing-rounds <n>` / `--presolve-probing-probes <n>` / `--presolve-probing-time <s>` | 2 / 5000 / 5 | Its budget, per presolve call |
| `--clique-table` / `--no-clique-table` | off | Root conflict graph and clique table (propagation, clique cuts) |
| `--plunge-depth <n>` / `--no-plunge` | 0 (off) | Maximum consecutive dives after a branch; serial search only |
| `--plunge-bound-quotient <q>` | 0.05 | Decline a dive whose bound is this far (relative) behind the best |

Run `./build/mipx-solve --help` for the full list. The probing, clique-table
and plunging switches are off by default until benchmarks decide otherwise;
leaving a flag out keeps the solver's own default.

## Python API

```bash
python3 -m venv .venv
.venv/bin/pip install .[test]
```

```python
import mipx

result = mipx.solve_file("instance.mps", time_limit=60, verbose=False)
print(result.status, result.objective)

model = mipx.read_model("instance.lp")                           # reader from the extension
model = mipx.read_model("instance.txt", format=mipx.ModelFormat.Lp)  # or forced
```

Every reader checks that the file really is in the format it is read as and
raises `mipx.ModelFormatError` (a `ValueError`) when it is not, rather than
returning an empty model. `solve_mps` is the older name of `solve_file` and
behaves identically: despite the name it reads `.lp` files as LP.

## Tests

```bash
ctest --test-dir build -j$(nproc)
```

Netlib and MIPLIB instances are not in the repo. Download them for benchmark tests:

```bash
./tests/data/download_netlib.sh           # full Netlib LP set
./tests/data/download_miplib.sh --small   # curated MIPLIB subset
```

Tests that require missing instances are skipped automatically.

## Project structure

```
include/mipx/     Public headers
src/
  lp/             Dual simplex, barrier, PDLP, LU factorization, sparse matrix
  mip/            Branch-and-bound, node queue, branching, domain propagation
  cuts/           Cut pool, separators, cut manager
  presolve/       Reductions and postsolve stack
  heuristics/     Primal heuristics and runtime orchestration
  io/             MPS/LP readers, solution file I/O
  cli/            Command-line interface
tests/            Catch2 tests and benchmark instances
python/           nanobind bindings and pytest tests
```

## References

- [HiGHS](https://github.com/ERGO-Code/HiGHS) -- high-performance LP/MIP solver
- [SCIP](https://scipopt.org/) -- constraint integer programming framework
- [Achterberg (2007)](https://doi.org/10.1007/s10107-006-0023-0) -- branch-and-cut thesis
- [Maros (2003)](https://doi.org/10.1007/978-1-4615-0257-9) -- simplex textbook

## License

[MIT](LICENSE)
