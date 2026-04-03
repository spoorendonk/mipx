# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

@.devkit/standards/nanobind.md

# Project: mipx

A from-scratch branch-and-cut MIP solver in C++23 with Python bindings via nanobind.

## Build & Test

```clean
rm -rf build
```

```build
cmake -B build && cmake --build build -j$(nproc)
```

```test
ctest --test-dir build --output-on-failure -j$(nproc) && { pytest --tb=short -q || test $? -eq 5; }
```

### Running a single C++ test

```bash
ctest --test-dir build -R "test_name_regex" --output-on-failure
```

All C++ tests compile into a single `mipx-tests` binary (Catch2). You can also run it directly:

```bash
./build/tests/mipx-tests "test name or tag"
```

### CMake options

| Option | Default | Purpose |
|---|---|---|
| `MIPX_USE_TBB` | auto-detect | Parallel tree search (`apt install libtbb-dev`) |
| `MIPX_USE_CUDA` | ON | GPU acceleration (barrier, PDLP) |
| `MIPX_BUILD_PYTHON` | OFF | Build nanobind Python extension |
| `MIPX_BUILD_CLI` | ON | Build `mipx-solve` CLI |
| `MIPX_SIMD_ISA` | native | SIMD codegen: `off`, `avx2`, `native` |
| `MIPX_STRICT_WARNINGS` | ON | `-Werror` |

### Test data

Netlib/MIPLIB instances are not in git. Download before running benchmarks:

```bash
./tests/data/download_miplib.sh          # full MIPLIB 2017 set
./tests/data/download_miplib.sh --small  # curated small subset
```

Tests skip automatically when instances are missing.

## Architecture

All code under `mipx::` namespace. Key type aliases: `Real` = `double`, `Int` = `int`, `Index` = `int`.

### Core modules (`src/`)

- **`lp/`** — LP solvers: dual simplex (Devex pricing + BFRT), Forrest-Tomlin LU factorization, interior-point barrier (CPU + GPU/cuDSS), PDLP (CPU + CUDA kernels), exact iterative refinement
- **`mip/`** — Branch-and-bound: node queue, domain propagation, branching strategies
- **`cuts/`** — Cut pool, Gomory MIR separation, cut manager, custom separators
- **`presolve/`** — Reductions and postsolve stack
- **`heuristics/`** — Rounding, diving, RINS, RENS, feasibility pump, local branching, symmetry-aware, budget management
- **`io/`** — MPS/LP readers (mmap + bulk decompress), MPS writer, solution file reader
- **`cli/`** — `mipx-solve` CLI entry point

### Key types

- `SparseMatrix` — CSR-primary storage with lazy CSC transpose built on demand
- `LpProblem` — LP/MIP problem data (objective, bounds, constraints, integrality)
- `LpSolver` — Abstract LP solver interface (dual simplex, barrier, PDLP backends)
- `MipSolver` — Branch-and-cut MIP solver orchestrating all components

### Python bindings (`python/`)

- `python/src/bindings.cpp` — nanobind module exposing core API
- `python/tests/test_api.py` — pytest tests for the Python interface
- Build with `-DMIPX_BUILD_PYTHON=ON`

### Performance benchmarks (`tests/perf/`)

Regression gates and benchmark scripts for Netlib LP, MIPLIB MIP, Mittelman, and dual simplex. Key scripts:

- `run_full_gate.sh` — full performance gate
- `run_dual_perf_gate.sh` — dual simplex regression gate
- `run_mip_regression_gate.sh` — MIP regression gate
- `check_regression.py` — shared regression checking logic

## Project-specific conventions

- CSR-primary sparse storage; lazy CSC built on demand — never store both eagerly.
- Forrest-Tomlin LU updates, not product-form inverse.
- Dual simplex uses Devex pricing + bound flipping ratio test (BFRT).
- Debug builds enable ASan + UBSan automatically.
- No external solver dependencies — everything from scratch.
- CUDA code lives alongside CPU code in the same directories (`.cu` files).
