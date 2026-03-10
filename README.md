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
| **Symmetry handling** | Column orbit detection, symmetry-breaking cuts, and orbital bound fixing to enforce canonical order |
| **Exact LP refinement** | Optional root-certificate refinement (`off/auto/on`), long-double checks, and scaled-rational verification |
| **Python API** | Nanobind bindings for model I/O and MIP solve flow (`LpProblem`, `MipSolver`) |
| **Concurrent root racing** | Optional dual/barrier/PDLP root race with cooperative stop and winner telemetry |
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

## Python API

Build and install the Python package with scikit-build-core:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install .[test]
```

Quick smoke test:

```bash
.venv/bin/python -c "import mipx; r = mipx.solve_mps('tests/data/tiny.mps', verbose=False); print(r.status)"
```

The wheel build uses a portable configuration (`abi3`, no `-march=native`) and
is wired to cibuildwheel for Linux x86_64/aarch64, macOS arm64, and Windows x64.

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
| `--concurrent-root` | off | Race dual/barrier/PDLP at root (deterministic or opportunistic policy mode) |
| `--parallel-mode <deterministic|opportunistic>` | deterministic | Parallel scheduling mode (`deterministic` is reproducible; pre-root adaptive portfolio is forced to fixed schedule when `--threads > 1`) |
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
| `--no-symmetry` | off | Disable symmetry detection and canonical branch selection |
| `--exact-refine-off` | on | Disable exact LP refinement checks (default behavior) |
| `--exact-refine-auto` | off | Trigger exact refinement only on numerical warning thresholds |
| `--exact-refine-on` | off | Force exact refinement pipeline at root LP |
| `--exact-rational-check` | off | Enable scaled-rational certificate verification |
| `--exact-no-rational-check` | — | Disable scaled-rational certificate verification |
| `--exact-warning-tol <t>` | 1e-7 | Warning threshold for auto-triggered exact refinement |
| `--exact-cert-tol <t>` | 1e-8 | Certificate tolerance for refinement checks |
| `--exact-max-rounds <n>` | 2 | Maximum exact refinement rounds |
| `--exact-repair-passes <n>` | 2 | Deterministic primal repair passes per refinement round |
| `--exact-rational-scale <s>` | 1e6 | Scaling factor used in rationalized verification |
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

Post-Step-29 janitor correctness/E2E checks (objective validation against
curated `.solu` references, including root-policy variants):

```bash
ctest --test-dir build -R "benchmark.*solve" --output-on-failure
```

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

Dual-simplex-specific LP gate (Netlib anchors + LPopt-style curated LP
relaxations):

```bash
# Full Netlib anchors + small MIPLIB set for LP relaxations
./tests/data/download_netlib.sh
./tests/data/download_miplib.sh --small

python3 tests/perf/run_dual_perf_gate.py \
  --candidate-binary ./build/mipx-solve \
  --highs-binary highs \
  --netlib-dir ./tests/data/netlib \
  --miplib-dir ./tests/data/miplib \
  --baseline-netlib-csv ./tests/perf/baselines/mipx_dual_lp_netlib_anchors.csv \
  --baseline-mittelman-csv ./tests/perf/baselines/mipx_dual_lp_mittelman_curated.csv
```

Generate reproducible HiGHS CLI and mipx wall-clock baselines:

```bash
python3 tests/perf/generate_highs_baselines.py
python3 tests/perf/generate_mipx_baselines.py
```

Baselines are stored in `tests/perf/baselines/`.
Shell wrappers remain available for compatibility:
`./tests/perf/generate_highs_baselines.sh` and
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

# Note: determinism checks use objective/nodes/LP-iterations/work-units.
# Wall-clock Time remains telemetry-only and is excluded from stability checks.

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
  --parallel-modes deterministic,opportunistic \
  --cuts on,off \
  --presolve on,off \
  --solver-arg --quiet

# Presolve consistency/perf matrix (light smoke profile)
python3 tests/perf/run_presolve_matrix.py \
  --profile ci-smoke \
  --mipx-binary ./build/mipx-solve \
  --netlib-dir ./tests/data/netlib \
  --miplib-dir ./tests/data/miplib \
  --out-dir /tmp/mipx_presolve_smoke

# Broader internal presolve profile (optimization loops)
python3 tests/perf/run_presolve_matrix.py \
  --profile internal \
  --mipx-binary ./build/mipx-solve \
  --netlib-dir ./tests/data/netlib \
  --miplib-dir ./tests/data/miplib \
  --out-dir /tmp/mipx_presolve_internal \
  --mipx-arg=--quiet
```

Example comparison against stored HiGHS LP baseline:

```bash
python3 tests/perf/run_netlib_lp_bench.py \
  --binary ./build/mipx-solve \
  --netlib-dir ./tests/data/netlib \
  --output /tmp/netlib_candidate.csv \
  --repeats 3 \
  --solver-arg --quiet

python3 tests/perf/check_regression.py \
  --baseline tests/perf/baselines/highs_lp_netlib_small.csv \
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
python/            Nanobind bindings, Python package, and Python tests
.github/workflows/ CI, wheel build, and tagged PyPI publish workflows
docs/              Documentation and roadmap
```

## Roadmap

See [docs/roadmap.md](docs/roadmap.md) for the full implementation plan.

**Current focus:** symmetry handling (Step 33) and exact-LP refinement planning (Step 34).

**Future work:** concurrent root racing, crossover improvements, and column generation.

## References

- [HiGHS](https://github.com/ERGO-Code/HiGHS) — high-performance LP/MIP solver
- [SCIP](https://scipopt.org/) — constraint integer programming framework
- [Achterberg (2007)](https://doi.org/10.1007/s10107-006-0023-0) — branch-and-cut thesis
- [Maros (2003)](https://doi.org/10.1007/978-1-4615-0257-9) — simplex textbook

## License

TBD
