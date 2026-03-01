# LP Benchmark Notes (HiGHS CLI + cuOpt CLI)

Date: 2026-02-26  
Commit: `8848831`

## Environment

- GPU: NVIDIA GeForce RTX 5070 Ti (driver 580.95.05, 16303 MiB)
- `mipx`: `./build/mipx-solve` (local build)
- HiGHS CLI: `1.13.1` (`highs --version`)
- cuOpt CLI: `26.2.0` (`cuopt_cli --version`)

## Instance Sets

- Small Mittelman LP subset downloaded to `tests/data/mittelman_lp`:
  - `L1_sixm250obs, Linf_520c, datt256, ex10, fome13, irish-e, pds-100, qap15, rail4284, rmine15, s100, s250r10`
- Benchmark focus subsets:
  - Dual/simplex: `datt256, ex10, fome13, qap15, rail4284, rmine15, s100, s250r10`
  - Barrier/PDLP: `ex10, fome13, s100`

## Commands Used

```bash
# mipx dual simplex (8-instance subset)
python3 -u tests/perf/run_mittelman_lp_bench.py \
  --binary ./build/mipx-solve \
  --mittelman-dir /tmp/mittelman_small8 \
  --netlib-dir /tmp/mittelman_small8 \
  --output /tmp/mipx_dual_mittelman_small8.csv \
  --repeats 1 --time-limit 15 \
  --solver-arg --dual --solver-arg --quiet

# mipx barrier cpu+gpu and cuopt barrier (3-instance subset)
python3 -u tests/perf/run_barrier_lp_compare.py \
  --mipx-binary ./build/mipx-solve \
  --instances-dir /tmp/mittelman_small8 \
  --instances ex10,fome13,s100 \
  --output /tmp/barrier_lp_compare_mittelman_small3.csv \
  --repeats 1 --threads 1 --time-limit 20 \
  --force-mipx-gpu --no-highs

# mipx pdlp cpu+gpu and cuopt pdlp (3-instance subset)
python3 -u tests/perf/run_pdlp_lp_compare.py \
  --mipx-binary ./build/mipx-solve \
  --instances-dir /tmp/mittelman_small8 \
  --instances ex10,fome13,s100 \
  --output /tmp/pdlp_lp_compare_mittelman_small3.csv \
  --repeats 1 --threads 1 --time-limit 20 \
  --force-mipx-gpu --no-highs
```

HiGHS numbers below come from direct CLI runs (to preserve `time_limit` status even when return code is nonzero):

- `/tmp/highs_simplex_direct_mittelman_small8.csv`
- `/tmp/highs_ipm_direct_mittelman_small3.csv`
- `/tmp/highs_pdlp_direct_mittelman_small3.csv`

## Results

### Dual Simplex (`mipx --dual`) vs HiGHS `simplex`

| Instance | mipx | HiGHS simplex |
| --- | --- | --- |
| `datt256` | `time_limit` | `time_limit @ 15.01s` |
| `ex10` | `infeasible @ 0.00s` | `time_limit @ 15.01s` |
| `fome13` | `optimal @ 0.00s` | `parser_error @ 0.15s` |
| `qap15` | `time_limit` | `time_limit @ 15.00s` |
| `rail4284` | `optimal @ 0.00s` | `parser_error @ 0.74s` |
| `rmine15` | `time_limit` | `time_limit @ 15.04s` |
| `s100` | `optimal @ 0.03s` | `time_limit @ 15.02s` |
| `s250r10` | `time_limit` | `time_limit @ 15.01s` |

### Barrier (CPU/GPU/cross-solver)

| Instance | mipx barrier CPU | mipx barrier GPU | HiGHS IPM | cuOpt barrier |
| --- | --- | --- | --- | --- |
| `ex10` | `infeasible @ 0.00s` | `infeasible @ 0.21s` | `time_limit @ 20.00s` | `unknown @ 22.20s` |
| `fome13` | `optimal @ 0.00s` | `optimal @ 0.00s` | `parser_error @ 0.16s` | `parser_error @ 0.44s` |
| `s100` | `optimal @ 0.03s` | `optimal @ 0.26s` | `time_limit @ 20.03s` | `optimal @ 5.76s` |

### PDLP (GPU-focused)

| Instance | mipx PDLP CPU | mipx PDLP GPU | HiGHS PDLP | cuOpt PDLP |
| --- | --- | --- | --- | --- |
| `ex10` | `infeasible @ 0.00s` | `infeasible @ 1.20s` | `optimal @ 1.49s` (GPU: `yes`) | `optimal @ 0.416s` |
| `fome13` | `optimal @ 0.00s` | `optimal @ 0.00s` | `parser_error @ 0.16s` | `parser_error @ 0.43s` |
| `s100` | `optimal @ 0.18s` | `optimal @ 1.50s` | `time_limit @ 24.08s` (GPU: `yes`) | `optimal @ 7.833s` |

Direct HiGHS probe confirms GPU PDLP path is active on supported files (`Solving with cuPDLP-C`, CUDA device shown in CLI output).

## Larger-Instance Sanity (Netlib, 60s limit)

To answer larger-instance behavior and correctness, we ran a second pass on larger Netlib LPs:
`greenbea`, `pilot` (full matrix at 60s), plus spot checks on `ship12l`, `sierra`.

Command artifact:

- `/tmp/large_lp_compare_netlib2_t60.csv`

### Large-Run Status Summary (`greenbea`, `pilot`)

| Solver mode | greenbea | pilot |
| --- | --- | --- |
| `mipx_dual` | `unknown` (rc=1) | `unknown` (rc=1) |
| `highs_simplex` | `optimal @ 0.19s` | `optimal @ 0.96s` |
| `mipx_barrier_cpu` | `infeasible @ 0.00s` | `infeasible @ 0.00s` |
| `mipx_barrier_gpu` | `infeasible @ 0.00s` | `infeasible @ 0.00s` |
| `highs_ipm` | `optimal @ 0.20s` | `optimal @ 0.49s` |
| `cuopt_barrier` | `unknown @ 1.86s` | `optimal @ 0.70s` |
| `mipx_pdlp_cpu` | `infeasible @ 0.00s` | `infeasible @ 0.00s` |
| `mipx_pdlp_gpu` | `infeasible @ 0.00s` | `infeasible @ 0.00s` |
| `highs_pdlp_gpu` | `time_limit @ 60.98s` | `time_limit @ 60.51s` |
| `cuopt_pdlp` | `optimal @ 46.66s` | `optimal @ 9.14s` |

### Correctness Check

Reference optimal values from `tests/data/netlib.solu`:

- `greenbea = -7.2555248130e+07`
- `ship12l = 1.4701879193e+06`
- `sierra = 1.5394362184e+07`

Observed:

- HiGHS simplex matches Netlib reference on `greenbea`, `ship12l`, `sierra`.
- `mipx` barrier/PDLP report `Infeasible` on these same instances (and dual exits unknown on `greenbea`/`pilot`), which is inconsistent with known-optimal references.
- `cuopt` often returns near-optimal values but can differ from Netlib/HiGHS objective (e.g., `sierra`, `greenbea`) and needs tighter objective-tolerance validation.

Conclusion from larger runs: current `mipx` LP results are **not yet correctness-safe** on these larger instances, so speed numbers are not trustworthy as performance claims until feasibility/objective agreement is fixed.

## Caveats / Follow-ups

- `fome13` and `rail4284` show parser failures in HiGHS and/or cuOpt CLI on this setup.
- Cross-solver objective agreement is inconsistent on `s100` (`mipx` reports `0`, cuOpt reports around `-0.171`), so objective-validation checks should be added before drawing performance conclusions.
- On larger Netlib instances (`greenbea`, `pilot`, `ship12l`, `sierra`), `mipx` barrier/PDLP return infeasible or unknown while references are optimal; this must be treated as a correctness blocker.
- `tests/perf/run_highs_bench.py` currently treats nonzero exit as `solve_error`, which hides useful `time_limit` results from HiGHS.
- `tests/data/download_mittelman_lp.sh` needs updates for current Mittelman URL/file naming (many files are `.bz2` or moved to subdirectories).

## Versioned `tests/perf` Review

Tracked files (`git ls-files tests/perf`) currently include:

- Benchmark runners: `run_*` Python + `.sh` wrappers for Netlib, MIPLIB, Mittelman, matrix, sweep, determinism, full gate, barrier compare, PDLP compare.
- Baseline generators: `generate_*` Python + `.sh` wrappers.
- Regression checker: `check_regression.py`.
- Docs: `tests/perf/README.md`, `tests/perf/baselines/README.md`.
- Versioned baselines currently present:
  - `highs_lp_netlib_small.csv`
  - `highs_mip_miplib_small.csv`
  - `mipx_lp_netlib_small.csv`
  - `mipx_mip_miplib_small.csv`
  - `barrier_lp_compare_netlib.csv`
  - `barrier_lp_compare_netlib_forced_gpu.csv`
  - metadata JSON files for HiGHS/mipx/barrier

Mismatch observed:

- `tests/perf/README.md` and `tests/perf/baselines/README.md` mention versioned PDLP and Mittelman baseline files, but those files are not currently tracked in git in this branch.
