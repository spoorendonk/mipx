# highspy Baselines

This directory stores versioned wall-clock baseline CSVs generated with
`highspy` for quick mipx-vs-HiGHS comparisons.

Files:
- `highspy_lp_netlib_small.csv`: Netlib small LP set.
- `highspy_mip_miplib_small.csv`: MIPLIB small trio (`p0201,gt2,flugpl`).
- `highspy_baseline_meta.json`: generation metadata (timestamp, highspy version,
  host CPU/platform).

Regenerate with:

```bash
./tests/perf/generate_highspy_baselines.sh
```

Notes:
- These are machine-specific wall-clock references.
- Use `work_units`-based gates for strict no-regression checks.
