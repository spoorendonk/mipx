# mipx

Exact MIP solver (branch-and-cut) built from scratch in C++23. Dual simplex,
barrier, and PDLP LP modes, cutting planes, presolve, and a native heuristic runtime.

> **Disclaimer:** This project is developed entirely through [Claude Code](https://docs.anthropic.com/en/docs/claude-code).

## Components

| Component | Description |
|-----------|-------------|
| **Dual simplex** | Phase 1+2, steepest-edge pricing, sparse LU with rank-1 updates |
| **Barrier / PDLP** | Optional root/LP solve modes with GPU SpMV acceleration + CPU fallback |
| **Branch-and-bound** | Best-first/depth-first node selection, most-fractional/first-fractional branching |
| **Cutting planes** | Gomory MIR, cut pool with aging and parallelism filtering |
| **Presolve** | Singleton, dominated, probing reductions + postsolve stack |
| **Primal heuristics** | Rounding, diving, RINS, RENS, Feasibility Pump, local branching |
| **Heuristic runtime** | Deterministic/opportunistic heuristic execution, restart engine, incumbent sharing pool |
| **Pre-root LP-free stage** | Optional FeasJump/FPR/Local-MIP style incumbent search before root LP |
| **Pre-root LP-light arms** | Optional LP-guided FPR/diving arms behind capability/build gates |
| **Adaptive pre-root portfolio** | Thompson-sampling arm scheduler with deterministic mode and arm-level telemetry |
| **Parallel tree search** | Optional TBB-parallel node processing |

## Build

Requires C++23 and CMake 3.25+. TBB and CUDA are optional.

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

Disable optional LP-light heuristic arms at build time:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DMIPX_ENABLE_LP_LIGHT_HEURISTICS=OFF
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
| `--time-limit <seconds>` | 3600 (MIP) | Wall-clock time limit for MIP solves |
| `--threads <n>` | 1 | Number of parallel tree-search threads (MIP mode) |
| `--presolve` | on | Enable presolve reductions |
| `--no-presolve` | — | Disable presolve |
| `--cuts` | on | Enable cutting planes |
| `--no-cuts` | — | Disable cutting planes |
| `--gap-tol <g>` | 1e-4 | Relative gap tolerance for optimality |
| `--node-limit <n>` | 1000000 | Maximum nodes to explore (MIP mode) |
| `--dual` | on | Use dual simplex for LP/root LP solve |
| `--barrier` | off | Use barrier for LP/root LP solve |
| `--pdlp` | off | Use PDLP for LP/root LP solve |
| `--heur-deterministic` | on | Deterministic heuristic runtime mode |
| `--heur-opportunistic` | off | Opportunistic heuristic runtime mode (throughput-first) |
| `--seed <n>` | 1 | Seed for heuristic runtime restart scheduling |
| `--pre-root-lpfree` | off | Enable LP-free pre-root incumbent stage |
| `--no-pre-root-lpfree` | — | Disable LP-free pre-root stage |
| `--pre-root-work <w>` | 50000 | Work-unit budget for pre-root LP-free stage |
| `--pre-root-rounds <n>` | 24 | Max pre-root LP-free rounds across workers |
| `--pre-root-no-early-stop` | off | Do not stop pre-root stage after first feasible |
| `--pre-root-lplight` | off | Enable LP-light pre-root arms (requires LP-light capability) |
| `--no-pre-root-lplight` | — | Disable LP-light pre-root arms |
| `--pre-root-portfolio` | on | Enable adaptive pre-root arm scheduler (Thompson sampling) |
| `--pre-root-fixed` | off | Use fixed pre-root arm schedule (disable adaptive portfolio) |
| `--gpu` | on | Enable GPU backend for barrier/PDLP when worthwhile |
| `--no-gpu` | — | Force CPU backend for barrier/PDLP |
| `--gpu-min-rows <n>` | 512 | Minimum rows before GPU backend is considered |
| `--gpu-min-nnz <n>` | 10000 | Minimum nonzeros before GPU backend is considered |
| `--relax-integrality` | off | Treat integer variables as continuous (LP relaxation mode) |
| `--verbose` | on | Enable verbose output |
| `--quiet` | off | Suppress verbose progress output |

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
python3 tests/perf/run_full_gate.py \
  --candidate-binary ./build/mipx-solve \
  --baseline-binary /tmp/mipx_main/build/mipx-solve \
  --netlib-dir ./tests/data/netlib \
  --miplib-dir ./tests/data/miplib \
  --solver-arg --quiet

# Optional: run LP/MIP gates separately

# LP gate input (Netlib)
python3 tests/perf/run_netlib_lp_bench.py \
  --binary ./build/mipx-solve \
  --netlib-dir ./tests/data/netlib \
  --output /tmp/netlib_candidate.csv \
  --repeats 3 \
  --solver-arg --quiet

# MIP gate input (MIPLIB)
python3 tests/perf/run_miplib_mip_bench.py \
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

Generate reproducible HiGHS CLI and mipx wall-clock baselines:

```bash
python3 tests/perf/generate_highspy_baselines.py
python3 tests/perf/generate_mipx_baselines.py
```

Baselines are stored in `tests/perf/baselines/`.
Shell wrappers remain available for compatibility:
`./tests/perf/generate_highspy_baselines.sh` and
`./tests/perf/generate_mipx_baselines.sh`.

Step-29 reproducibility/tuning tooling:

```bash
# Determinism checks (single-thread + configured multi-thread deterministic mode)
python3 tests/perf/run_determinism_suite.py \
  --binary ./build/mipx-solve \
  --miplib-dir ./tests/data/miplib \
  --instances p0201,gt2,flugpl \
  --runs 5 \
  --single-threads 1 \
  --multi-threads 4 \
  --solver-arg --quiet

# Benchmark matrix artifacts (solver x time x threads x mode)
python3 tests/perf/run_benchmark_matrix.py \
  --mipx-binary ./build/mipx-solve \
  --miplib-dir ./tests/data/miplib \
  --solvers mipx \
  --modes deterministic,opportunistic \
  --threads 1,4 \
  --time-limits 30,120 \
  --instances p0201,gt2,flugpl \
  --solver-arg --quiet

# Parameter sweep artifacts (CSV + Markdown ranking)
python3 tests/perf/run_param_sweep.py \
  --binary ./build/mipx-solve \
  --miplib-dir ./tests/data/miplib \
  --instances p0201,gt2,flugpl \
  --search-profiles stable,default,aggressive \
  --heur-modes deterministic,opportunistic \
  --cuts on,off \
  --presolve on,off \
  --solver-arg --quiet
```

Example comparison against stored HiGHS LP baseline
(legacy filename prefix `highspy_`):

```bash
python3 tests/perf/run_netlib_lp_bench.py \
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
  heuristics/     Heuristic arms + runtime orchestration
  io/             MPS/LP readers, solution file I/O
  cli/            Command-line interface
tests/             Catch2 tests and MIPLIB benchmarks
docs/              Documentation and roadmap
```

## Roadmap

See [docs/roadmap.md](docs/roadmap.md) for the full implementation plan.

**Current focus:** Python/release pipeline integration (Step 20) and
post-Step-20 quality/performance janitor work.

**Future work:** concurrent root racing, crossover improvements, and column generation.

## References

- [HiGHS](https://github.com/ERGO-Code/HiGHS) — high-performance LP/MIP solver
- [SCIP](https://scipopt.org/) — constraint integer programming framework
- [Achterberg (2007)](https://doi.org/10.1007/s10107-006-0023-0) — branch-and-cut thesis
- [Maros (2003)](https://doi.org/10.1007/978-1-4615-0257-9) — simplex textbook

## License

TBD
