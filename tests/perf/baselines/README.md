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
- `barrier_lp_compare_netlib.csv`: LP barrier comparison on Netlib
  (`mipx_barrier_cpu`, `mipx_barrier_gpu`, `highs_ipx`, `cuopt_barrier`).
- `barrier_lp_compare_netlib_forced_gpu.csv`: same as above, but forcing mipx GPU path.
- `barrier_lp_compare_meta.json`: barrier comparison generation metadata
  (tool versions, GPU/driver info).

Regenerate with:

```bash
./tests/perf/generate_highspy_baselines.sh
./tests/perf/generate_mipx_baselines.sh
./tests/perf/generate_barrier_lp_baselines.sh
```

## Mittelman Baselines

Baselines matching Hans Mittelman's benchmark configuration
(https://plato.asu.edu/bench.html):

- `highspy_lp_mittelman.csv`: HiGHS LP baseline on Mittelman LPopt instances.
- `highspy_mip_mittelman.csv`: HiGHS MIP baseline on MIPLIB 2017 benchmark set.
- `highspy_mittelman_meta.json`: generation metadata.
- `mipx_lp_mittelman.csv`: mipx LP baseline on Mittelman LPopt instances.
- `mipx_mip_mittelman.csv`: mipx MIP baseline on MIPLIB 2017 benchmark set.
- `mipx_mittelman_meta.json`: generation metadata.

Regenerate with:

```bash
./tests/perf/generate_mittelman_baselines.sh
```

Notes:
- These are machine-specific wall-clock references.
- Use `work_units`-based gates for strict no-regression checks.
- Mittelman LP params: 15000s time limit, 1 thread (simplex).
- Mittelman MIP params: 7200s time limit, 8 threads, 1e-4 gap tolerance.
