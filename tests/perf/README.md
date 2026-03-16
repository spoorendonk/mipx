# Performance Regression Gate

Use the benchmark scripts + `check_regression.py` to enforce no-regression policy.

`run_full_gate.py` now follows a dual-metric policy:
- Algorithmic regression: strict gate on `work_units` (default `0.0%` allowed).
- Runtime regression: looser tracking on `time_seconds` (non-fatal by default).

Deterministic contract defaults in `run_full_gate.py`:
- `--parallel-mode deterministic`
- fixed `--seed`
- fixed `--threads`
- `--search-stable`

Recommended LP metadata columns (used by dual-simplex gate/reporting):
- `status`
- `objective`
- `iterations`
- `time_seconds`

Full LP+MIP gate example (single command):

```bash
python3 tests/perf/run_full_gate.py \
  --candidate-binary ./build/mipx-solve \
  --baseline-binary /tmp/mipx_main/build/mipx-solve \
  --netlib-dir ./tests/data/netlib \
  --miplib-dir ./tests/data/miplib \
  --threads 4 \
  --seed 1 \
  --algorithmic-metric work_units \
  --algorithmic-max-regression-pct 0 \
  --wall-metric time_seconds \
  --wall-max-regression-pct 35 \
  --solver-arg --quiet
```

## Dual Simplex Perf Gate (Netlib Anchors + LPopt-Style Curated Corpus)
Use `--enforce-wall-clock` if wall-clock regression should be fatal.

This gate is dual-simplex specific and enforces a **work-units-first**
regression policy:
- Primary hard gate: `work_units` (aggregate + per-instance cap).
- Secondary metric: `time_seconds` (`warn` by default; can be `fail`).

Default solver policy in this gate:
- `--dual`
- `--no-presolve`
- `--relax-integrality`
- `--quiet`

The second corpus keeps the historical `mittelman_dual_corpus.csv` name, but it
is currently curated from LPopt-style MIPLIB LP relaxations so the gate stays
deterministic and comparable in CI.

Correctness precheck (before perf comparison, default `fail` mode):
- Runs `run_dual_correctness_investigation.py` on the Netlib dual corpus.
- Uses `.solu` + HiGHS simplex parity checks.
- Modes: `off`, `warn`, `fail`.

Run candidate vs baseline binaries:

```bash
python3 tests/perf/run_dual_perf_gate.py \
  --candidate-binary ./build/mipx-solve \
  --baseline-binary /tmp/mipx_main/build/mipx-solve \
  --netlib-dir ./tests/data/netlib \
  --mittelman-dir ./tests/data/mittelman_lp \
  --miplib-dir ./tests/data/miplib \
  --max-work-regression-pct 0 \
  --max-work-instance-regression-pct 20 \
  --time-regression-mode warn \
  --max-time-regression-pct 10
```

Artifacts (under `--out-dir`, default `/tmp/mipx_dual_perf_gate`):
- `candidate_netlib.csv`, `baseline_netlib.csv`
- `candidate_mittelman.csv`, `baseline_mittelman.csv`
- `netlib_regression_summary.json`
- `mittelman_regression_summary.json`
- `dual_perf_summary.md`

Dual baseline generation for stored anchor corpora:

```bash
python3 tests/perf/generate_dual_baselines.py \
  --binary ./build/mipx-solve \
  --netlib-dir ./tests/data/netlib \
  --mittelman-dir ./tests/data/mittelman_lp \
  --miplib-dir ./tests/data/miplib \
  --netlib-time-limit 60 \
  --mittelman-time-limit 20
```

Corpora:
- Netlib anchors: `tests/perf/netlib_dual_corpus.csv`
- LPopt-style curated LP anchors: `tests/perf/mittelman_dual_corpus.csv`
  - Current source: MIPLIB LP relaxations downloaded by `./tests/data/download_miplib.sh --small`

SOTA guard policy for dual-simplex work:
- Keep changes aligned with modern HiGHS-style mechanisms.
- Do not accept speedups from numerically weaker legacy shortcuts.
- Treat status/objective mismatches as correctness failures before perf claims.

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
python3 tests/perf/generate_dual_baselines.py
python3 tests/perf/generate_barrier_lp_baselines.py
python3 tests/perf/generate_pdlp_lp_baselines.py
```

This writes:
- `tests/perf/baselines/highs_lp_netlib_small.csv`
- `tests/perf/baselines/highs_mip_miplib_small.csv`
- `tests/perf/baselines/mipx_lp_netlib_small.csv`
- `tests/perf/baselines/mipx_pdlp_lp_netlib_small.csv`
- `tests/perf/baselines/mipx_mip_miplib_small.csv`
- `tests/perf/baselines/mipx_dual_lp_netlib_anchors.csv`
- `tests/perf/baselines/mipx_dual_lp_mittelman_curated.csv`
- `tests/perf/baselines/mipx_dual_lp_baseline_meta.json`
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

PDLP LP example (Netlib CPU PDLP, gate on `work_units`):

```bash
python3 tests/perf/run_netlib_pdlp_lp_bench.py \
  --binary ./build/mipx-solve \
  --netlib-dir ./tests/data/netlib \
  --output /tmp/mipx_pdlp_candidate.csv \
  --repeats 3 \
  --threads 1 \
  --time-limit 60 \
  --disable-presolve \
  --solver-arg --quiet

python3 tests/perf/check_regression.py \
  --baseline tests/perf/baselines/mipx_pdlp_lp_netlib_small.csv \
  --candidate /tmp/mipx_pdlp_candidate.csv \
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

## Dedicated MIP Baseline + Gate

For a focused MIP-only regression loop (similar to dual-focused flows), use the
committed regression corpus at `tests/perf/mip_regression_corpus.csv` and the
dedicated scripts below.

Generate/update the committed MIP baseline CSV:

```bash
python3 tests/perf/generate_mip_regression_baseline.py \
  --binary ./build/mipx-solve \
  --miplib-dir ./tests/data/miplib
```

Default outputs:
- `tests/perf/baselines/mipx_mip_regression_small_seed1_t1_stable.csv`
- `tests/perf/baselines/mipx_mip_regression_small_seed1_t1_stable_meta.json`

Run MIP self-regression gate against committed baseline (strict `work_units`):

```bash
python3 tests/perf/run_mip_regression_gate.py \
  --candidate-binary ./build/mipx-solve
```

Or compare candidate vs a baseline binary instead of committed CSV:

```bash
python3 tests/perf/run_mip_regression_gate.py \
  --candidate-binary ./build/mipx-solve \
  --baseline-binary /tmp/mipx_main/build/mipx-solve
```

When `--baseline-binary` is provided, it takes precedence over `--baseline-csv`.

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

## PDLP LP Comparison (mipx vs cuPDLPx vs HiGHS vs cuOpt)

For LP PDLP-focused comparisons, `cuPDLPx` is the primary external reference.
The tracked compare harness supports CPU/GPU `mipx` rows plus optional
`cuPDLPx`, HiGHS, and cuOpt rows:

```bash
python3 tests/perf/run_pdlp_lp_compare.py \
  --mipx-binary ./build/mipx-solve \
  --cupdlpx-binary /path/to/cupdlpx \
  --instances-dir tests/data/netlib \
  --output /tmp/pdlp_lp_compare.csv \
  --repeats 3 \
  --threads 1 \
  --time-limit 60 \
  --disable-presolve \
  --force-mipx-gpu
```

This emits one CSV row per `(instance, solver)` with:
- `solver in {mipx_pdlp_cpu, mipx_pdlp_gpu, cupdlpx, highs_pdlp|highs_ipx, cuopt_pdlp}`
- `time_seconds, iterations, status, objective, work_units, backend`

Notes:
- `--cupdlpx-binary` or `MIPX_CUPDLPX_BINARY` enables standalone `cuPDLPx` rows.
- If `cuPDLPx` is not configured, the generic compare run skips it cleanly.
- `--time-limit` is enforced as an external wall-clock cap by the harness so cross-solver PDLP runs remain comparable even though `mipx` PDLP CLI time-limit plumbing is still incomplete.
- `backend` records actual `mipx` PDLP execution (`cpu` or `gpu`) when available, so the auto-GPU lane can be separated from threshold-driven CPU fallback.
- `--no-mipx-cpu` or `--no-mipx-gpu` lets you benchmark only the relevant lane.
- `--disable-presolve` isolates PDLP-kernel behavior and avoids presolve skew.
- `--force-mipx-gpu` sets `--gpu-min-rows 0 --gpu-min-nnz 0` so the GPU path is always exercised.
- `--highs-ipx` forces HiGHS IPX instead of attempting HiGHS PDLP.
- `--relax-integrality` is supported for `mipx`, HiGHS, and cuOpt. The standalone `cuPDLPx` lane in this harness is LP-only.

PDLP self-regression gate (candidate vs baseline mipx binaries):

```bash
python3 tests/perf/run_pdlp_lp_regression_gate.py \
  --candidate-binary ./build/mipx-solve \
  --baseline-binary /tmp/mipx_main/build/mipx-solve \
  --netlib-dir ./tests/data/netlib
```

This default run enforces strict PDLP algorithmic regression (`work_units`) on:
- `mipx_pdlp_cpu`
- `mipx_pdlp_gpu` (forced GPU thresholds)

Enable wall-clock gates (SIMD/AVX proxy, multithread probe, GPU):

```bash
python3 tests/perf/run_pdlp_lp_regression_gate.py \
  --candidate-binary ./build/mipx-solve \
  --baseline-binary /tmp/mipx_main/build/mipx-solve \
  --netlib-dir ./tests/data/netlib \
  --wall-clock-gate \
  --wall-mt-threads 8
```

Wall-clock notes:
- CPU single-thread gate uses `time_seconds` as a SIMD/AVX proxy.
- CPU multi-thread gate is auto-skipped if the PDLP LP path reports single-thread execution.
- Forced-GPU gate is auto-skipped if backend probing does not report `PDLP backend: GPU`.

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
  --miplib-dir ./tests/data/miplib \
  --output /tmp/mittelman_lp.csv \
  --solver-arg --dual \
  --solver-arg --no-presolve \
  --solver-arg --relax-integrality \
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
