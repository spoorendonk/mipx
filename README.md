# mip-exact

Exact MIP solver (branch-and-cut) built from scratch in C++23. Dual simplex LP,
cutting planes, presolve, and primal heuristics — no external solver dependencies.

> **Disclaimer:** This project is developed entirely through [Claude Code](https://docs.anthropic.com/en/docs/claude-code).

## Components

| Component | Description |
|-----------|-------------|
| **Dual simplex** | Phase 1+2, steepest-edge pricing, sparse LU with rank-1 updates |
| **Branch-and-bound** | Best-first/DFS/hybrid node selection, reliability branching |
| **Cutting planes** | Gomory MIR, cut pool with aging and parallelism filtering |
| **Presolve** | Singleton, dominated, probing reductions + postsolve stack |
| **Primal heuristics** | Rounding, diving, RINS |
| **Parallel tree search** | Optional TBB-parallel node processing |

## Output

HiGHS-style progress table during solve:

```
mipx v0.1
  Variables   : 1024 (512 binary, 128 integer, 384 continuous)
  Constraints : 768
  Threads     : 8

        Time    Nodes    Open   LP Iter    Incumbent   Best Bound    Gap%
  *      0.1s       1       0      342        85.50        72.30   15.5%
  *      0.4s      28      12     2.1k        83.20        74.10   10.9%
         2.0s     500     180    45.2k        83.20        79.40    4.6%
  *      3.1s     820     210    62.4k        81.00        79.80    1.5%
         5.0s    1.4k      40   104.3k        81.00        80.90    0.1%

  Optimal: 81.00 in 5.2s (1482 nodes, 112k LP iterations)
```

## Build

Requires C++23 and CMake 3.25+. TBB is optional.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

SIMD build defaults:
- `MIPX_SIMD_ISA=native` by default (max local tuning, `-march=native`)
- `MIPX_SIMD_ISA=avx2` for standardized non-AVX512 x86_64 distribution targets
- `MIPX_SIMD_ISA=off` for scalar fallback

```bash
# Default local-machine tuning
cmake -B build -DCMAKE_BUILD_TYPE=Release -DMIPX_SIMD_ISA=native

# Standardized AVX2 target
cmake -B build -DCMAKE_BUILD_TYPE=Release -DMIPX_SIMD_ISA=avx2
```

You can inspect host SIMD support with:

```bash
lscpu | grep -E "Flags|avx2|avx512"
```

Optional TBB support:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DMIPX_USE_TBB=ON
```

## Usage

```bash
# Solve a MIP instance
./build/mipx-solve instance.mps

# With options
./build/mipx-solve instance.mps --time-limit 300 --threads 4 --presolve
```

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--time-limit <seconds>` | 3600 | Wall-clock time limit |
| `--threads <n>` | 1 | Number of parallel tree search threads |
| `--presolve` | on | Enable presolve reductions |
| `--no-presolve` | — | Disable presolve |
| `--cuts` | on | Enable cutting planes |
| `--no-cuts` | — | Disable cutting planes |
| `--gap-tol <g>` | 1e-4 | Relative gap tolerance for optimality |
| `--node-limit <n>` | ∞ | Maximum nodes to explore |
| `--verbose` | off | Verbose output |

## Tests

```bash
ctest --test-dir build -j$(nproc)
```

Netlib and MIPLIB instances are not included in the repo. Download them before running benchmarks:

```bash
./tests/data/download_test_instances.sh   # small Netlib + small MIPLIB (recommended)
./tests/data/download_netlib.sh          # full Netlib LP set
./tests/data/download_netlib.sh --small  # curated small subset (CI)
./tests/data/download_miplib.sh          # MIPLIB 2017 collection
./tests/data/download_miplib.sh --small  # curated small subset (CI)
```

Tests that require missing instances are skipped automatically.

## Benchmarking

Run perf gates on Netlib LP and MIPLIB MIP with deterministic `work_units`:

```bash
# Download datasets (small sets recommended for local gating)
./tests/data/download_test_instances.sh

# Full LP+MIP gate (strict: 0% median regression by default)
./tests/perf/run_full_gate.sh \
  --candidate-binary ./build/mipx-solve \
  --baseline-binary /tmp/mipx_main/build/mipx-solve \
  --netlib-dir ./tests/data/netlib \
  --miplib-dir ./tests/data/miplib \
  --solver-arg --quiet

# Optional: run LP/MIP gates separately

# LP gate input (Netlib)
./tests/perf/run_netlib_lp_bench.sh \
  --binary ./build/mipx-solve \
  --netlib-dir ./tests/data/netlib \
  --output /tmp/netlib_candidate.csv \
  --repeats 3 \
  --solver-arg --quiet

# MIP gate input (MIPLIB)
./tests/perf/run_miplib_mip_bench.sh \
  --binary ./build/mipx-solve \
  --miplib-dir ./tests/data/miplib \
  --output /tmp/miplib_candidate.csv \
  --repeats 1 \
  --threads 1 \
  --time-limit 30 \
  --instances p0201,pk1,gt2 \
  --solver-arg --quiet

# Regression check (default metric is work_units)
python3 tests/perf/check_regression.py \
  --baseline /path/to/baseline.csv \
  --candidate /tmp/netlib_candidate.csv
```

Generate reproducible HiGHS/highspy wall-clock baselines:

```bash
./tests/perf/generate_highspy_baselines.sh
```

Baselines are stored in `tests/perf/baselines/`.

Example comparison against stored highspy LP baseline:

```bash
./tests/perf/run_netlib_lp_bench.sh \
  --binary ./build/mipx-solve \
  --netlib-dir ./tests/data/netlib \
  --output /tmp/netlib_candidate.csv \
  --repeats 3 \
  --solver-arg --quiet

python3 tests/perf/check_regression.py \
  --baseline tests/perf/baselines/highspy_lp_netlib_small.csv \
  --candidate /tmp/netlib_candidate.csv \
  --metric time_seconds \
  --max-regression-pct 100000
```

Use `--metric time_seconds` if you need wall-clock gating instead of
deterministic work-unit gating.
Use `--max-regression-pct` to relax the gate; default is strict `0.0%`.

## Project Structure

```
include/mipx/     Public headers
src/
  lp/             Sparse matrix, LU factorization, dual simplex
  mip/            Branch-and-bound, node queue, branching
  cuts/           Cut pool, Gomory MIR separation
  presolve/       Reductions and postsolve stack
  heuristics/     Rounding, diving, RINS
  io/             MPS/LP readers, solution file I/O
  cli/            Command-line interface
tests/             Catch2 tests and MIPLIB benchmarks
docs/              Documentation and roadmap
```

## Roadmap

See [docs/roadmap.md](docs/roadmap.md) for the full implementation plan.

**Current focus:** LP solver interface → LU factorization → dual simplex.

**Future work:** Barrier/IPM, PDLP+GPU, column generation.

## References

- [HiGHS](https://github.com/ERGO-Code/HiGHS) — high-performance LP/MIP solver
- [SCIP](https://scipopt.org/) — constraint integer programming framework
- [Achterberg (2007)](https://doi.org/10.1007/s10107-006-0023-0) — branch-and-cut thesis
- [Maros (2003)](https://doi.org/10.1007/978-1-4615-0257-9) — simplex textbook

## License

TBD
