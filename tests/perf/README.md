# Performance Regression Gate

Use the benchmark scripts + `check_regression.py` to enforce no-regression policy.

Required CSV columns:
- `instance`
- metric column (default `work_units`)

Full LP+MIP gate example (single command):

```bash
./tests/perf/run_full_gate.sh \
  --candidate-binary ./build/mipx-solve \
  --baseline-binary /tmp/mipx_main/build/mipx-solve \
  --netlib-dir ./tests/data/netlib \
  --miplib-dir ./tests/data/miplib \
  --solver-arg --quiet
```

Generate and store HiGHS/highspy wall-clock baselines:

```bash
./tests/perf/generate_highspy_baselines.sh
./tests/perf/generate_mipx_baselines.sh
```

This writes:
- `tests/perf/baselines/highspy_lp_netlib_small.csv`
- `tests/perf/baselines/highspy_mip_miplib_small.csv`
- `tests/perf/baselines/mipx_lp_netlib_small.csv`
- `tests/perf/baselines/mipx_mip_miplib_small.csv`

The MIP highspy baseline uses the stable small trio:
`p0201,gt2,flugpl`.

LP example (Netlib, gate on `work_units`):

```bash
./tests/perf/run_netlib_lp_bench.sh \
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
./tests/perf/run_miplib_mip_bench.sh \
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

To compare mipx against HiGHS/highspy, use `--metric time_seconds` and one of
the stored highspy baseline CSV files as `--baseline`.
These comparisons are informational (cross-solver wall-clock), not strict
no-regression gates.

MIP comparison example (same instance set as highspy MIP baseline):

```bash
./tests/perf/run_miplib_mip_bench.sh \
  --binary ./build/mipx-solve \
  --miplib-dir ./tests/data/miplib \
  --output /tmp/mipx_mip_candidate_highspyset.csv \
  --repeats 1 \
  --threads 1 \
  --time-limit 30 \
  --instances p0201,gt2,flugpl \
  --solver-arg --quiet

python3 tests/perf/check_regression.py \
  --baseline tests/perf/baselines/highspy_mip_miplib_small.csv \
  --candidate /tmp/mipx_mip_candidate_highspyset.csv \
  --metric time_seconds \
  --max-regression-pct 100000 \
  --min-common-instances 3
```

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
./tests/perf/run_mittelman_lp_bench.sh \
  --binary ./build/mipx-solve \
  --output /tmp/mittelman_lp.csv \
  --solver-arg --quiet

# MIP benchmark (Mittelman MILP-style):
./tests/perf/run_mittelman_mip_bench.sh \
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
./tests/perf/run_mittelman_gate.sh \
  --candidate-binary ./build/mipx-solve \
  --baseline-binary /tmp/mipx_main/build/mipx-solve
```

Compare against HiGHS (informational, time_seconds):

```bash
./tests/perf/run_mittelman_gate.sh \
  --candidate-binary ./build/mipx-solve \
  --baseline-lp-csv tests/perf/baselines/highspy_lp_mittelman.csv \
  --baseline-mip-csv tests/perf/baselines/highspy_mip_mittelman.csv \
  --metric time_seconds \
  --max-regression-pct 100000
```

### Generate Mittelman Baselines

```bash
# Both HiGHS and mipx baselines
./tests/perf/generate_mittelman_baselines.sh

# HiGHS only
./tests/perf/generate_mittelman_baselines.sh --highs-only

# mipx only
./tests/perf/generate_mittelman_baselines.sh --mipx-only
```

This writes:
- `tests/perf/baselines/highspy_lp_mittelman.csv`
- `tests/perf/baselines/highspy_mip_mittelman.csv`
- `tests/perf/baselines/mipx_lp_mittelman.csv`
- `tests/perf/baselines/mipx_mip_mittelman.csv`
