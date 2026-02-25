# Performance Regression Gate

Use `check_regression.py` to enforce a no-regression policy on benchmark CSVs.

Required CSV columns:
- `instance`
- metric column (default `time_seconds`)

Example:

```bash
python3 tests/perf/check_regression.py \
  --baseline benchmarks/baseline_v1.csv \
  --candidate benchmarks/pr_step2.csv \
  --metric time_seconds \
  --max-regression-pct 2.0
```

This fails if median candidate runtime regresses by more than the configured
percentage.
