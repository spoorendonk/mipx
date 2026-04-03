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
| **Branch-and-bound** | Best-first / depth-first node selection, reliability branching with pseudocosts |
| **Cutting planes** | Gomory MIR, cover, clique, zero-half, implied bound, mixing cuts |
| **Presolve** | Variable fixing, singleton elimination, dual fixing, coefficient tightening, bounds propagation |
| **Heuristics** | Rounding, diving, feasibility pump, RINS, RENS, local branching |
| **Parallel search** | TBB-based tree exploration with deterministic and opportunistic modes |
| **Symmetry** | Orbit detection, symmetry-breaking constraints, canonical branching |
| **Exact refinement** | Optional rational-arithmetic LP repair for numerical certification |
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
./build/mipx-solve instance.mps --time-limit 300 --threads 4
```

Key options:

| Option | Default | Description |
|--------|---------|-------------|
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

Run `./build/mipx-solve --help` for the full list.

## Python API

```bash
python3 -m venv .venv
.venv/bin/pip install .[test]
```

```python
import mipx

result = mipx.solve_mps("instance.mps", time_limit=60, verbose=False)
print(result.status, result.objective)
```

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
