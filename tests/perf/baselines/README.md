# HiGHS and mipx Baselines

This directory stores versioned wall-clock baseline CSVs generated with
HiGHS CLI and mipx for quick comparisons.

Files:
- `highs_lp_netlib_small.csv`: Netlib small LP set.
- `highs_mip_miplib_small.csv`: MIPLIB small trio (`p0201,gt2,flugpl`).
- `highs_baseline_meta.json`: generation metadata (timestamp, HiGHS version,
  host CPU/platform).
- `mipx_lp_netlib_small.csv`: mipx LP baseline for strict work-unit gating.
- `mipx_mip_miplib_small.csv`: mipx MIP baseline for strict work-unit gating.
- `mipx_baseline_meta.json`: generation metadata for mipx baseline snapshots.
- `barrier_lp_compare_netlib.csv`: LP barrier comparison on Netlib
  (`mipx_barrier_cpu`, `mipx_barrier_gpu`, `highs_ipx`, `cuopt_barrier`).
- `barrier_lp_compare_netlib_forced_gpu.csv`: same as above, but forcing mipx GPU path.
- `barrier_lp_compare_meta.json`: barrier comparison generation metadata
  (tool versions, GPU/driver info).
- `pdlp_lp_compare_netlib.csv`: LP PDLP comparison on Netlib
  (`mipx_pdlp_cpu`, `mipx_pdlp_gpu`, `highs_pdlp` or `highs_ipx`, `cuopt_pdlp`).
- `pdlp_lp_compare_netlib_forced_gpu.csv`: same as above, but forcing mipx GPU path.
- `pdlp_lp_compare_meta.json`: PDLP comparison generation metadata
  (tool versions, GPU/driver info).

Regenerate with (canonical Python entrypoints):

```bash
python3 tests/perf/generate_highs_baselines.py
python3 tests/perf/generate_mipx_baselines.py
python3 tests/perf/generate_barrier_lp_baselines.py
python3 tests/perf/generate_pdlp_lp_baselines.py
```

Shell wrappers remain available under `tests/perf/generate_*.sh`.

## Mittelman Baselines

Baselines matching Hans Mittelman's benchmark configuration
(https://plato.asu.edu/bench.html):

- `highs_lp_mittelman.csv`: HiGHS LP baseline on Mittelman LPopt instances.
- `highs_mip_mittelman.csv`: HiGHS MIP baseline on MIPLIB 2017 benchmark set.
- `highs_mittelman_meta.json`: generation metadata.
- `mipx_lp_mittelman.csv`: mipx LP baseline on Mittelman LPopt instances.
- `mipx_mip_mittelman.csv`: mipx MIP baseline on MIPLIB 2017 benchmark set.
- `mipx_mittelman_meta.json`: generation metadata.

Regenerate with:

```bash
python3 tests/perf/generate_mittelman_baselines.py
```

Shell wrapper equivalent:
`./tests/perf/generate_mittelman_baselines.sh`.

Notes:
- These are machine-specific wall-clock references.
- Use `work_units`-based gates for strict no-regression checks.
- Mittelman LP params: 15000s time limit, 1 thread (simplex).
- Mittelman MIP params: 7200s time limit, 8 threads, 1e-4 gap tolerance.
