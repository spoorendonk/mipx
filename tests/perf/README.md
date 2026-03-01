# Performance Regression Gate

Use the benchmark scripts + `check_regression.py` to enforce no-regression policy.

Required CSV columns:
- `instance`
- metric column (default `work_units`)

Full LP+MIP gate example (single command):

```bash
python3 tests/perf/run_full_gate.py \
  --candidate-binary ./build/mipx-solve \
  --baseline-binary /tmp/mipx_main/build/mipx-solve \
  --netlib-dir ./tests/data/netlib \
  --miplib-dir ./tests/data/miplib \
  --solver-arg --quiet
```

## Presolve Matrix (Light CI + Internal)

Run presolve correctness/performance comparisons against HiGHS across a matrix
of LP/MIP solver settings. This uses `run_presolve_compare.py` under the hood
and emits:
- `presolve_matrix_detail.csv`
- `presolve_matrix_summary.csv`
- `presolve_matrix_summary.md`

Lightweight smoke profile (suitable for optional CI checks):

```bash
python3 tests/perf/run_presolve_matrix.py \
  --profile ci-smoke \
  --mipx-binary ./build/mipx-solve \
  --netlib-dir ./tests/data/netlib \
  --miplib-dir ./tests/data/miplib \
  --out-dir /tmp/mipx_presolve_smoke
```

Broader internal profile (for optimization loops):

```bash
python3 tests/perf/run_presolve_matrix.py \
  --profile internal \
  --mipx-binary ./build/mipx-solve \
  --netlib-dir ./tests/data/netlib \
  --miplib-dir ./tests/data/miplib \
  --out-dir /tmp/mipx_presolve_internal \
  --mipx-arg=--quiet
```

Notes:
- MIP comparisons run with exact gap checks by default (`--mipx-gap-tol 0` in
  `run_presolve_compare.py`) so objective mismatches are correctness-significant.
- `ci-smoke` is intentionally small; use `internal` for broader coverage.
- Shell wrapper: `./tests/perf/run_presolve_matrix.sh`.
- Requires HiGHS CLI (`highs`) on `PATH` or `--highs-binary <path>`.

Generate and store HiGHS CLI + mipx wall-clock baselines:

```bash
python3 tests/perf/generate_highs_baselines.py
python3 tests/perf/generate_mipx_baselines.py
python3 tests/perf/generate_barrier_lp_baselines.py
python3 tests/perf/generate_pdlp_lp_baselines.py
```

This writes:
- `tests/perf/baselines/highs_lp_netlib_small.csv`
- `tests/perf/baselines/highs_mip_miplib_small.csv`
- `tests/perf/baselines/mipx_lp_netlib_small.csv`
- `tests/perf/baselines/mipx_mip_miplib_small.csv`
- `tests/perf/baselines/barrier_lp_compare_netlib.csv`
- `tests/perf/baselines/barrier_lp_compare_netlib_forced_gpu.csv`
- `tests/perf/baselines/pdlp_lp_compare_netlib.csv`
- `tests/perf/baselines/pdlp_lp_compare_netlib_forced_gpu.csv`

The MIP HiGHS baseline uses the stable small trio:
`p0201,gt2,flugpl`.
Shell wrapper entrypoints remain available under `tests/perf/generate_*.sh`.

LP example (Netlib, gate on `work_units`):

```bash
python3 tests/perf/run_netlib_lp_bench.py \
  --binary ./build/mipx-solve \
  --netlib-dir ./tests/data/netlib \
  --output /tmp/mipx_candidate.csv \
  --repeats 3 \
  --solver-arg --quiet

python3 tests/perf/check_regression.py \
  --baseline benchmarks/baseline_v1.csv \
  --candidate benchmarks/pr_step2.csv \
  --metric work_units
```

MIP example (MIPLIB, gate on `work_units`):

```bash
python3 tests/perf/run_miplib_mip_bench.py \
  --binary ./build/mipx-solve \
  --miplib-dir ./tests/data/miplib \
  --output /tmp/mipx_mip_candidate.csv \
  --repeats 1 \
  --threads 1 \
  --time-limit 30 \
  --instances p0201,pk1,gt2 \
  --solver-arg --quiet

python3 tests/perf/check_regression.py \
  --baseline benchmarks/miplib_baseline.csv \
  --candidate /tmp/mipx_mip_candidate.csv \
  --metric work_units
```

The regression gate fails if the median candidate metric regresses by more
than the configured percentage. Default gate is strict: `0.0%` allowed median
regression.

## Determinism Suite

Validate deterministic reproducibility at fixed seed in both single-thread
and configured multi-thread mode:

```bash
python3 tests/perf/run_determinism_suite.py \
  --binary ./build/mipx-solve \
  --miplib-dir ./tests/data/miplib \
  --instances p0201,gt2,flugpl \
  --runs 5 \
  --single-threads 1 \
  --multi-threads 4 \
  --solver-arg --quiet
```

Artifacts:
- `determinism_detail.csv`
- `determinism_summary.csv`
- `determinism_summary.md`

The command exits non-zero if any `(instance, profile)` is unstable.
Use `--strict-metrics` to additionally require node/iteration/work-unit
equality across runs.
`time_seconds` remains in detail CSV for telemetry only and is not part of
deterministic stability checks.

## Benchmark Matrix Runner

Run the full `solver x time x threads x mode` matrix and generate artifacts:

```bash
python3 tests/perf/run_benchmark_matrix.py \
  --mipx-binary ./build/mipx-solve \
  --miplib-dir ./tests/data/miplib \
  --solvers mipx \
  --modes deterministic,opportunistic \
  --threads 1,4 \
  --time-limits 30,120 \
  --instances p0201,gt2,flugpl \
  --solver-arg --quiet
```

Optional HiGHS rows:

```bash
python3 tests/perf/run_benchmark_matrix.py \
  --mipx-binary ./build/mipx-solve \
  --miplib-dir ./tests/data/miplib \
  --solvers mipx,highs \
  --highs-binary highs \
  --instances p0201,gt2,flugpl \
  --solver-arg --quiet
```

Artifacts:
- `matrix_detail.csv`
- `matrix_summary.csv`
- `matrix_summary.md`

## Parameter Sweep Runner

Sweep common MIP controls and rank configurations with structured outputs:

```bash
python3 tests/perf/run_param_sweep.py \
  --binary ./build/mipx-solve \
  --miplib-dir ./tests/data/miplib \
  --instances p0201,gt2,flugpl \
  --search-profiles stable,default,aggressive \
  --parallel-modes deterministic,opportunistic \
  --cuts on,off \
  --presolve on,off \
  --threads 1 \
  --time-limit 30 \
  --solver-arg --quiet
```

Artifacts:
- `sweep_detail.csv`
- `sweep_summary.csv`
- `sweep_summary.md`

## Barrier LP Comparison (mipx vs HiGHS IPX vs cuOpt)

For LP barrier-focused comparisons (CPU/GPU and cross-solver):

```bash
python3 tests/perf/run_barrier_lp_compare.py \
  --mipx-binary ./build/mipx-solve \
  --instances-dir tests/data/netlib \
  --output /tmp/barrier_lp_compare.csv \
  --repeats 3 \
  --threads 1 \
  --time-limit 60 \
  --disable-presolve \
  --force-mipx-gpu
```

This emits one CSV row per `(instance, solver)` with:
- `solver in {mipx_barrier_cpu, mipx_barrier_gpu, highs_ipx, cuopt_barrier}`
- `time_seconds, iterations, status, objective, work_units`

Notes:
- `--disable-presolve` isolates barrier-kernel behavior and avoids presolve skew.
- `--force-mipx-gpu` sets `--gpu-min-rows 0 --gpu-min-nnz 0` so the GPU path is always exercised.
- `--relax-integrality` solves LP relaxations for MIP instances (e.g., MIPLIB `.mps.gz`) in all solvers.

To compare mipx against HiGHS CLI, use `--metric time_seconds` and one of
the stored HiGHS baseline CSV files as `--baseline`.
These comparisons are informational (cross-solver wall-clock), not strict
no-regression gates.

## PDLP LP Comparison (mipx vs HiGHS vs cuOpt)

For LP PDLP-focused comparisons (CPU/GPU and cross-solver):

```bash
python3 tests/perf/run_pdlp_lp_compare.py \
  --mipx-binary ./build/mipx-solve \
  --instances-dir tests/data/netlib \
  --output /tmp/pdlp_lp_compare.csv \
  --repeats 3 \
  --threads 1 \
  --time-limit 60 \
  --disable-presolve \
  --force-mipx-gpu
```

This emits one CSV row per `(instance, solver)` with:
- `solver in {mipx_pdlp_cpu, mipx_pdlp_gpu, highs_pdlp|highs_ipx, cuopt_pdlp}`
- `time_seconds, iterations, status, objective, work_units`

Notes:
- `--disable-presolve` isolates PDLP-kernel behavior and avoids presolve skew.
- `--force-mipx-gpu` sets `--gpu-min-rows 0 --gpu-min-nnz 0` so the GPU path is always exercised.
- `--highs-ipx` forces HiGHS IPX instead of attempting HiGHS PDLP.
- `--relax-integrality` solves LP relaxations for MIP instances (e.g., MIPLIB `.mps.gz`) in all solvers.

PDLP self-regression gate (candidate vs baseline mipx binaries):

```bash
python3 tests/perf/run_pdlp_lp_regression_gate.py \
  --candidate-binary ./build/mipx-solve \
  --baseline-binary /tmp/mipx_main/build/mipx-solve \
  --netlib-dir ./tests/data/netlib
```

MIP comparison example (same instance set as HiGHS MIP baseline):

```bash
python3 tests/perf/run_miplib_mip_bench.py \
  --binary ./build/mipx-solve \
  --miplib-dir ./tests/data/miplib \
  --output /tmp/mipx_mip_candidate_highsset.csv \
  --repeats 1 \
  --threads 1 \
  --time-limit 30 \
  --instances p0201,gt2,flugpl \
  --solver-arg --quiet

python3 tests/perf/check_regression.py \
  --baseline tests/perf/baselines/highs_mip_miplib_small.csv \
  --candidate /tmp/mipx_mip_candidate_highsset.csv \
  --metric time_seconds \
  --max-regression-pct 100000 \
  --min-common-instances 3
```

## Dual Correctness Investigation (Step 35)

Deterministic status/objective cross-check for `mipx --dual` against Netlib
`.solu` references plus HiGHS simplex:

```bash
python3 tests/perf/run_dual_correctness_investigation.py \
  --mipx-binary ./build/mipx-solve \
  --netlib-dir tests/data/netlib \
  --solu-file tests/data/netlib/netlib.solu \
  --corpus tests/perf/netlib_dual_corpus.csv \
  --output /tmp/dual_correctness.csv \
  --summary /tmp/dual_correctness_summary.md \
  --time-limit 60 \
  --threads 1 \
  --disable-presolve
```

Output columns include:
- `mipx_status`, `mipx_objective`, `highs_status`, `highs_objective`
- `classification` (`ok`, `false_infeasible`, `false_unknown_or_error`,
  `objective_mismatch`, ...)
- `mipx_signature` (first useful error fingerprint, e.g. singular LU path)

The script exits nonzero when hard mismatches remain.

## Mittelman Benchmarks

Benchmark configuration matching Hans Mittelman's optimization benchmarks
(https://plato.asu.edu/bench.html). These are the industry-standard benchmarks
used to compare LP/MIP solvers.

### Instance Sets

**LP (LPopt):** ~49 publicly known instances from Mittelman's LP benchmark
(65 total, 16 undisclosed). Sources: Netlib, MIPLIB LP relaxations, Meszaros
collection, and Mittelman's own test set.

**MIP (MILP):** The full MIPLIB 2017 benchmark set (240 instances), the same
instances used by Mittelman. A curated ~50-instance subset is available for
faster testing (instances HiGHS can solve in <30 min).

### Download Instances

```bash
# Quick: Mittelman LP curated subset (~20 instances)
./tests/data/download_mittelman_lp.sh

# Full: all publicly known Mittelman LP instances
./tests/data/download_mittelman_lp.sh --full

# MIPLIB 2017 benchmark set (240 instances, used by Mittelman for MILP)
./tests/data/download_miplib.sh --mittelman

# MIPLIB Mittelman-solvable subset (~50 instances, good for HiGHS comparison)
./tests/data/download_miplib.sh --mittelman-small

# All Mittelman sets at once
./tests/data/download_test_instances.sh --mittelman
```

### Mittelman Parameters

| Parameter     | LP (LPopt)  | MIP (MILP)   |
|---------------|-------------|--------------|
| Time limit    | 15000s      | 7200s (2h)  |
| Threads       | 1           | 8            |
| Gap tolerance | N/A         | 1e-4         |
| Instance set  | 65 mixed    | 240 MIPLIB   |

### Running Mittelman Benchmarks

```bash
# LP benchmark (Mittelman LPopt-style):
python3 tests/perf/run_mittelman_lp_bench.py \
  --binary ./build/mipx-solve \
  --output /tmp/mittelman_lp.csv \
  --solver-arg --quiet

# MIP benchmark (Mittelman MILP-style):
python3 tests/perf/run_mittelman_mip_bench.py \
  --binary ./build/mipx-solve \
  --miplib-dir ./tests/data/miplib \
  --output /tmp/mittelman_mip.csv \
  --threads 8 \
  --time-limit 7200 \
  --solver-arg --quiet
```

### Full Mittelman Regression Gate

Self-regression (candidate vs baseline mipx on work_units):

```bash
python3 tests/perf/run_mittelman_gate.py \
  --candidate-binary ./build/mipx-solve \
  --baseline-binary /tmp/mipx_main/build/mipx-solve
```

Compare against HiGHS (informational, time_seconds):

```bash
python3 tests/perf/run_mittelman_gate.py \
  --candidate-binary ./build/mipx-solve \
  --baseline-lp-csv tests/perf/baselines/highs_lp_mittelman.csv \
  --baseline-mip-csv tests/perf/baselines/highs_mip_mittelman.csv \
  --metric time_seconds \
  --max-regression-pct 100000
```

### Generate Mittelman Baselines

```bash
# Both HiGHS and mipx baselines
python3 tests/perf/generate_mittelman_baselines.py

# HiGHS only
python3 tests/perf/generate_mittelman_baselines.py --highs-only

# mipx only
python3 tests/perf/generate_mittelman_baselines.py --mipx-only
```

Shell wrapper equivalent:
`./tests/perf/generate_mittelman_baselines.sh [--highs-only|--mipx-only]`.

All `tests/perf/run_*.sh` scripts are compatibility wrappers that forward to
the Python entrypoints shown above.

This writes:
- `tests/perf/baselines/highs_lp_mittelman.csv`
- `tests/perf/baselines/highs_mip_mittelman.csv`
- `tests/perf/baselines/mipx_lp_mittelman.csv`
- `tests/perf/baselines/mipx_mip_mittelman.csv`
