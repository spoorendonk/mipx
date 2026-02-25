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
