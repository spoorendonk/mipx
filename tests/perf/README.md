# Instance Gate (correctness on real instances)

`run_instance_gate.py` solves Netlib LPs (per backend: dual, barrier, PDLP) and
the curated MIPLIB subset, and classifies every result against the `.solu`
reference:

| Class | Meaning |
|---|---|
| `match_1e-6` | right verdict, objective within 1e-6 (relative, absolute below 1) |
| `match_1e-4` | right verdict, objective within 1e-4 |
| `disagree` | a wrong answer: Optimal off by more than 1e-4; or a verdict (Optimal / Infeasible / Unbounded) the reference contradicts |
| `dnf` | did not finish: time, node or iteration limit, error, crash |

The expected class for each instance and backend is checked in under
`instance_gate/` (`lp_expected.csv`, `mip_expected.csv`). The gate fails on any
`disagree`, whatever the list says, since a wrong answer is never tolerated. It
also fails when a run expected to match no longer does, or when a listed
instance's file or `.solu` reference is missing. Better than listed is reported,
not failed. A run in which nothing was gated (an empty or missing list) fails
too. MIPs run with `--gap-tol 1e-9`, so Optimal means proven and the objective
can be held to 1e-6. A MIP stopped by a limit is `dnf` whatever its incumbent:
the CLI prints `Objective: none` when it has none (#224), and checking a
limited incumbent is follow-up work on #223.

```bash
./tests/data/download_netlib.sh && ./tests/data/download_miplib.sh --small
python3 tests/perf/run_instance_gate.py lp      # all three LP backends
python3 tests/perf/run_instance_gate.py mip     # curated MIPLIB subset
ctest --test-dir build -L instance-gate --output-on-failure   # both, via ctest
```

Each run prints one line per instance and a per-backend count of the four
classes. Without downloaded instances the script prints `SKIPPED` and exits 77,
which ctest shows as a skip. Options: `--backends dual,pdlp`, `--time-limit S`
(defaults: 10s per LP, 60s per MIP), `--binary PATH`. A row's own
`time_limit` wins over `--time-limit`: to try such an instance at a longer
limit, pass `--expected-file` with a copy of the list without it.

Regenerating the lists:

```bash
python3 tests/perf/run_instance_gate.py lp --write-expected        # refresh listed rows
python3 tests/perf/run_instance_gate.py lp --all --write-expected  # add every instance with a reference
```

`--write-expected` keeps every existing row and overwrites only what this run
observed, at this run's time limit: combine it with `--time-limit` only when
you mean the list to hold at that limit. Review the diff before committing it: a row moving to `dnf` is a
regression, and `disagree` is a bug to file, not a class to accept.

The lists currently cover the instances the unit tests already gate: dual
simplex on nine Netlib LPs, and barrier and PDLP on a four-instance smoke subset.
The barrier matches to 1e-6 on all four since #204, so a barrier that stops
converging on one fails the gate (#256). Those rows were observed on the CPU
barrier: the `lp` suite passes no `--no-gpu`, so a CUDA build runs the GPU
barrier there, which no run has checked against them yet. PDLP also runs forplan, greenbeb and
bnl2, which it once reported Optimal up to 7.7% from the optimum (#205): `dnf`
is expected there, and the gate fails if they disagree again. The MIP list
expects `match_1e-6` for flugpl, gen, gt2 and p0201. That includes gt2 after
current main proved its optimum in under one second on one thread across
seeds 1–8 (#350), superseding the temporary `dnf` expectation from #341.
A gate with no instance files exits 77 and names the listed instances it did
not run; ctest shows only `Skipped` for it unless run with `-V`. Widening the
lists to the full sets is a benchmark-scale run, tracked in #223. The gate
records no timings; timing baselines belong to #213.

The gate's own tests: `uv run --with pytest pytest tests/perf/test_run_instance_gate.py
tests/perf/test_generate_cg_masters.py`.

## CPU Barrier Gate (#296)

The `barrier` suite runs the CPU barrier over every Netlib LP in
`tests/data/netlib`, once per normal-equations backend (#290):
`barrier-dense` (`--barrier --barrier-ne dense --no-gpu`) and `barrier-sparse`
(`--barrier --barrier-ne sparse --no-gpu`), one thread each, classified as
above against `instance_gate/barrier_expected.csv`, with `--time-limit 10`
per instance (the barrier honours it since #275; a time-limit stop reads as
`dnf`), the normal-equations backends' setup included (#347). The dense
backend still holds its m × m normal equations, about 2.2 GB on stocfor3
(m ≈ 16.7k), so it needs that much memory free; the limit bounds the time,
not the memory. It takes minutes, so it has its own ctest label and stays out
of `instance-gate` and the commit and push gates.

The `barrier-cg` suite (#358) runs the same two backends over one later
column-generation master per public CommaLab instance of the cold LP panel
(#272: grid1-9, planar30/50/80/100/150/300). Each master has a path column
that shares a demand row and a capacity row, giving the normal equations
off-diagonal entries. This replaces the diagonal first-master panel (#331)
and keeps the gate at 30 solves. It checks both backends against
`instance_gate/barrier_cg_expected.csv` and the recorded HiGHS objectives in
`tests/data/cg_masters_later/cg_masters.solu`. The suite passes `--no-presolve`
so the CLI cannot solve a master before entering the barrier. The barrier's
own presolve still runs. All 15 sparse factorizations show fill on the pinned
panel, and the two backends together take about 12 s on the test host.

Download [`grid.tgz`](https://commalab.di.unipi.it/files/Data/MMCF/grid.tgz)
and [`planar.tgz`](https://commalab.di.unipi.it/files/Data/MMCF/planar.tgz)
from [CommaLAB's multicommodity flow collection](https://commalab.di.unipi.it/datasets/mmcf/).
The external column-generation tool needs HiGHS support and must read the
gzip-compressed instance files in the original format described by each
archive's `format.doc`. From the directory containing the two archives,
prepare the inputs as follows.

```bash
mkdir -p commalab/{grid,planar}
tar -xzf grid.tgz -C commalab/grid
tar -xzf planar.tgz -C commalab/planar
gzip commalab/grid/grid{1..9} commalab/planar/planar{30,50,80,100,150,300}
```

Run the generator from the mipx checkout.

```bash
python3 tests/perf/generate_cg_masters.py --cg-tool PATH/TOOL --instances PATH/commalab
```

The executable must run HiGHS with presolve disabled and accept an
instance path followed by `--method cg
--backend highs --highs-method simplex --threads 1 --no-sidecar
--cg-set lp_dump=DIR --cg-set lp_dump_every=1 --cg-set max_iterations=5`.
The generator checks for `backend = highs simplex presolve=off` in the
dump's manifest. The tool writes `DIR/lp_0000/` with a full master just
before each solve, with one row/value pair per `COLUMNS` line and
consecutive solve numbers starting at 0. If the dump uses a different
manifest format value with the same version 1 layout, pass `--dump-format`
with that exact value.

The generator picks the first coupled snapshot after solve 0, looks up
its matching `solve` record in `ops.txt`, and writes `<name>.mps` and the
`.solu` in `tests/data/cg_masters_later/`.
Reproducing the reference masters requires the same starting columns and
pricing strategy. A change to the generator's model or pricing can
produce different intermediate LPs for the same inputs.
It requires a finite optimal objective for a minimisation LP with zero
objective offset and no free rows. A failed dump leaves the published
panel untouched. The suite reports `Skipped` until the later masters are
generated. First-master downloads live in a separate directory.

The whole gate (both suites, both backends) takes about 5.7 minutes, nearly
all of it the dense backend on Netlib; there is no faster subset in the commit
and push gates, since it stays out of them.

```bash
python3 tests/perf/run_instance_gate.py barrier                          # both backends
python3 tests/perf/run_instance_gate.py barrier --backends barrier-sparse
python3 tests/perf/run_instance_gate.py barrier --all --write-expected   # regenerate the list
python3 tests/perf/run_instance_gate.py barrier-cg                       # the cg masters
ctest --test-dir build -L barrier-gate --output-on-failure               # both suites
```

## Root heuristic deadline benchmark

This benchmark runs the one-thread seeded set cover from #355 through `MipSolver`,
with the root LP heuristic thresholds raised. Build the target explicitly, then
run it with a time limit in seconds (8 by default).

```bash
cmake --build build --target mipx-root-heuristic-deadline-bench
./build/mipx-root-heuristic-deadline-bench 8 > root-heuristic-deadline.csv
```

The CSV records the solver outcome, bound, work, reported time, and wall time
relative to the limit. On the original one-thread run for #355, the root LP and
cuts used about 7.4 s of the 8 s limit. The solve ran about 0.96 s past the limit
before the root portfolio got the deadline and about 0.07 s past it after the
fix. Use those figures as historical measurements and record fresh samples on
the same host and build when comparing revisions. Root LP and cut times vary
with load, so the stopping stage can differ between runs. The correctness suite
checks deterministic expiry at a root heuristic LP via `HeuristicRuntime`.

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

## Presolve Postsolve Check

`mipx-presolve-bench MODEL --check-postsolve` solves the reduced LP and
checks the reconstructed point's feasibility and objective on the original
LP. To use HiGHS for the solves, configure with `-DMIPX_BENCH_HIGHS=ON` and
run `mipx-presolve-check-highs MODEL --check-postsolve`. This uses one HiGHS
thread with the same mipx postsolve and checks. Supply an LP reference in
`--solu FILE` to skip the original solve when the reduced solve finds an
optimum.

## Presolve Matrix (Smoke + Internal)

Run presolve correctness/performance comparisons against HiGHS across a matrix
of LP/MIP solver settings. This uses `run_presolve_compare.py` under the hood
and emits:
- `presolve_matrix_detail.csv`
- `presolve_matrix_summary.csv`
- `presolve_matrix_summary.md`

Lightweight smoke profile (quick enough to run by hand):

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
python3 tests/perf/generate_mipx_barrier_baselines.py
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
- `tests/perf/baselines/mipx_barrier_cpu_netlib_small.csv`
- `tests/perf/baselines/mipx_barrier_gpu_netlib_small.csv`
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

## LP Sequence Replay (column-generation masters, #267)

`mipx-lpseq-replay` (built with the CLI) replays a recorded LP and its
incremental edits on mipx's LP solvers. It checks every solve against the
recorded status and objective. The [CPU Barrier Gate](#cpu-barrier-gate-296)
uses this format to capture column-generation masters.

```bash
./build/mipx-lpseq-replay tests/data/lpseq/grid1 \
    --arm dual-warm --arm dual-cold --arm barrier --arm barrier-hsd --arm pdlp \
    [--barrier-ne dense|sparse] [--primal-devex] [--csv out.csv]
```

The version 1 format is implemented in [`lpseq_format.h`](lpseq_format.h).
Each sequence directory requires three files.

- `manifest.txt` contains `key = value` lines and requires
  `format = lpseq 1`. If present, `sense` must be `min` and `obj_offset`
  must be zero. The replayers ignore other keys. `backend` identifies the
  recording solver and settings, and `tol_primal` and `tol_dual` give its
  feasibility tolerances. `snapshot_every` gives the optional snapshot
  interval in solves, with 0 indicating no extra snapshots.
  For an existing version 1 recording with a different
  format value, replace that line with `format = lpseq 1` before replay.
- `master_0000.mps` holds the LP at its first solve in free MPS, with rows
  `r<i>` and columns `c<j>` named by zero-based index. Data lines can
  start in column 1. The only `N` row is the objective. Free constraint
  rows would be dropped by the MPS reader and shift subsequent indices.
- `ops.txt` has whitespace-separated records with zero-based indices into
  the current LP. Real numbers are written with `%.17g` precision. Each
  edit starts with an operation and a count `k`.

  | Operation | Data after `k` |
  | --- | --- |
  | `add_cols` | `k` following lines of `cost lower upper nnz` and `nnz` `(row value)` pairs |
  | `add_rows` | `k` following lines of `lower upper nnz` and `nnz` `(column value)` pairs |
  | `del_cols`, `del_rows` | `k` indices on the same line |
  | `col_bounds`, `row_bounds` | `k` `(index lower upper)` triples on the same line |
  | `col_costs` | `k` `(column cost)` pairs on the same line |
  | `coeffs` | `k` `(row column value)` triples on the same line |

  Adds append rows or columns in order. Within one item, each nonzero
  position must appear only once. Deletes compact the remaining indices in
  order and merge repeated delete indices. `coeffs` replaces each listed
  value and requires distinct positions. Bounds can use `-inf` for a
  missing lower bound or `inf` for a missing upper bound.

  Start with the solve for `master_0000.mps`, then record edits before the
  solves they affect. The record `solve <seq> <status> <obj> <iters> <secs>`
  uses status `optimal`, `infeasible`, `unbounded` or `error`. Iteration
  count and seconds describe the recording run.

With a positive `snapshot_every = K`, optional `master_NNNN.mps` files
capture the LP just before each later solve whose number is a multiple of
K, with all preceding edits applied. Replayers use the initial snapshot
and `ops.txt`.

Arms: `dual-warm` keeps one dual simplex and applies each change to it, so its
basis carries over (a `coeffs` change, which `LpSolver` has no call for,
reloads it, flagged in the `reloaded` column); `dual-cold`, `barrier`,
`barrier-hsd` (the homogeneous self-dual loop of #370) and `pdlp` load the
LP afresh for every solve. `--barrier-ne dense|sparse` runs both barrier
arms on that CPU normal-equations factor, as `mipx-solve --barrier-ne X
--no-gpu` does. Without it they take the default: `barrier` the GPU when
there is one, `barrier-hsd` always the CPU (the HSD loop has no GPU path),
on the normal equations or the augmented system as the LP's density picks.
One CSV line per solve and arm:
`seq,arm,status,objective,recorded,rel_err,iterations,seconds,reloaded`.
Exit 0 when every solve matches the recorded status and objective (relative
to max(1, |recorded|), `--rel-tol` 1e-6, `--pdlp-rel-tol` 1e-4), 1 when one
does not, 2 on a format error.

`tests/data/lpseq/grid1` is the sequence of the public CommaLab grid1 (cg on
HiGHS simplex, 9 masters); ctest replays it on `dual-warm`, `dual-cold`,
`barrier` and `pdlp` with primal Devex off and on, and it and the edits
sequence on `barrier-hsd` with each `--barrier-ne` factor.

**The HiGHS side** (#268): configure with `-DMIPX_BENCH_HIGHS=ON` (OFF by
default; fetches HiGHS v1.16.0 with HiPO, needs a BLAS such as
libopenblas-dev) to build `mipx-lpseq-replay-highs`. It reads the ops with
the same parser (`lpseq_format.h`) and writes the same CSV, on HiGHS
through its API, one thread unless `--threads N`:

```bash
./build/mipx-lpseq-replay-highs tests/data/lpseq/grid1 \
    --arm simplex-warm --arm primal-warm --arm simplex-cold --arm ipx-cold --arm hipo-cold \
    [--presolve] [--threads N] [--csv out.csv]
```

`simplex-warm` (dual) and `primal-warm` keep one `Highs` across the changes.
The `-cold` arms solve each LP on a fresh one (IPX and HiPO without
crossover, held to `--ipm-rel-tol` 1e-5). Presolve is off unless `--presolve`.
The mipx library never links HiGHS.

`--threads N` sets HiGHS's `threads` option for every arm; it speeds up HiPO
and `pami-warm`, which is `simplex-warm` on HiGHS's parallel dual simplex
(PAMI, `simplex_strategy` 3). The serial dual of `simplex-warm` stays serial:
in HiGHS 1.16 `parallel=on` does not switch it to PAMI. `--threads` sets the
workers, not PAMI's width, which stays at HiGHS's `simplex_max_concurrency`
(8): `pami-warm` at `--threads 1` is still PAMI, on one worker, not the serial
dual. Time `pami-warm` only on otherwise idle CPUs: sharing them with a build
made a small sequence about 100 times slower.

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

## Barrier Self-Regression Gate

Dedicated barrier workflow (candidate vs committed barrier baselines):

```bash
python3 tests/perf/run_barrier_lp_regression_gate.py \
  --candidate-binary ./build/mipx-solve \
  --instances-dir tests/data/netlib
```

Shell wrapper:

```bash
./tests/perf/run_barrier_lp_regression_gate.sh \
  --candidate-binary ./build/mipx-solve \
  --instances-dir tests/data/netlib
```

Exploratory candidate-vs-binary comparison remains available:

```bash
python3 tests/perf/run_barrier_lp_regression_gate.py \
  --candidate-binary ./build/mipx-solve \
  --baseline-binary /tmp/mipx_main/build/mipx-solve \
  --instances-dir tests/data/netlib
```

Enforced by default:
- algorithmic regression gate on `work_units` for the committed GPU barrier lane
- default instance set is the curated forced-GPU stable subset:
  - `adlittle, afiro, blend, sc50b, share2b, stocfor1`
- default baseline source is `tests/perf/baselines/mipx_barrier_gpu_netlib_small.csv`

Opt-in checks:
- separate wall-clock bands for:
  - GPU barrier lane

Current policy:
- CUDA builds default to GPU barrier execution and do not fall back to CPU at runtime
- non-CUDA builds may still use CPU barrier backends
- CPU barrier regression lanes are not part of the default workflow until CPU baselines are usable

Key flags:
- `--all-available-instances`: use all available selected instances instead of the curated GPU stable set
- `--enable-wall-clock-bands`: enable the optional wall-clock regression checks
- `--simd-wall-clock-max-regression-pct`
- `--gpu-wall-clock-max-regression-pct`
- `--work-min-common-instances`
- `--wall-clock-min-common-instances`

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

### Bounded MIP Gap Summary

For a quick bounded mipx-vs-HiGHS MIP summary on the same Mittelman subset:

```bash
python3 tests/perf/run_mip_gap_summary.py \
  --binary ./build/mipx-solve \
  --threads 8 \
  --time-limit 15 \
  --solver-arg --quiet
```

The default bounded corpus is:
`air04,air05,blend2,dcmulti,flugpl,gen,glass4,gt2,p0201,supportcase16,stein45inf`

This writes matching mipx/HiGHS CSVs plus `summary.md` and `summary.json`
to `/tmp/mipx_mip_gap_summary` by default.

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
