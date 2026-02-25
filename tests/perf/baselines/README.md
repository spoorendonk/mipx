# highspy Baselines

This directory stores versioned wall-clock baseline CSVs generated with
`highspy` for quick mipx-vs-HiGHS comparisons.

Files:
- `highspy_lp_netlib_small.csv`: Netlib small LP set.
- `highspy_mip_miplib_small.csv`: MIPLIB small trio (`p0201,gt2,flugpl`).
- `highspy_baseline_meta.json`: generation metadata (timestamp, highspy version,
  host CPU/platform).
- `mipx_lp_netlib_small.csv`: mipx LP baseline for strict work-unit gating.
- `mipx_mip_miplib_small.csv`: mipx MIP baseline for strict work-unit gating.
- `mipx_baseline_meta.json`: generation metadata for mipx baseline snapshots.

Regenerate with:

```bash
./tests/perf/generate_highspy_baselines.sh
./tests/perf/generate_mipx_baselines.sh
```

Notes:
- These are machine-specific wall-clock references.
- Use `work_units`-based gates for strict no-regression checks.
