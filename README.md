# mipx

A branch-and-cut MIP solver built from scratch in C++ with Python bindings.

> **Work in progress.** This project is under active development.
> Built entirely with [Claude Code](https://docs.anthropic.com/en/docs/claude-code).

## Performance

On Netlib LP instances, mipx is roughly 2-5x slower than HiGHS on medium problems and 10-40x on the hardest ones. On easy MIPLIB MIPs, it's typically 2-8x slower.

## Features

| Component | Description |
|-----------|-------------|
| **Dual simplex** | Devex pricing, bound-flipping ratio test, Forrest-Tomlin LU, SIMD kernels |
| **Barrier** | Interior-point with CPU and GPU (cuDSS) backends |
| **PDLP** | First-order primal-dual method with CPU and CUDA paths |
| **Branch-and-bound** | Best-first, depth-first, best-estimate node selection, reliability branching with pseudocosts |
| **Cutting planes** | Gomory MIR, cover, clique, zero-half, implied bound, mixing cuts |
| **Presolve** | Variable fixing, singleton elimination, dual fixing, coefficient tightening, bounds propagation |
| **Heuristics** | Rounding, diving, feasibility pump, RINS, RENS, local branching |
| **Parallel search** | TBB-based tree exploration with deterministic and opportunistic modes |
| **Symmetry** | Orbit detection, symmetry-breaking constraints, canonical branching |
| **Exact refinement** | Optional extended-precision LP repair for numerical certification |
| **Concurrent root** | Race dual simplex, barrier, and PDLP at root with cooperative stop |
| **Python API** | nanobind bindings for model I/O and solve |

## Build

Requires C++23 and CMake 3.25+.

```bash
cmake -B build && cmake --build build --parallel
```

Optional dependencies:

```bash
# TBB for parallel tree search
cmake -B build -DMIPX_USE_TBB=ON && cmake --build build --parallel

# CUDA for GPU-accelerated barrier and PDLP
cmake -B build -DMIPX_USE_GPU=ON && cmake --build build --parallel
```

On macOS, the default Apple Clang build works with the Command Line Tools.
CUDA and cuDSS tests require those backends and are unavailable on Apple Silicon.
The sparse barrier uses its own loops when CMake finds only the Accelerate
framework. To use an installed Homebrew OpenBLAS, add its prefix when
configuring a fresh build directory:

```bash
cmake -B build -DCMAKE_PREFIX_PATH="$(brew --prefix openblas)"
```

Build the Python package and install all extras with `uv sync --all-extras`.
For the local correctness suite on macOS, run
`ctest --test-dir build --output-on-failure -LE solver-regression`.

## Usage

```bash
./build/mipx-solve instance.mps
./build/mipx-solve instance.lp
./build/mipx-solve instance.mps --time-limit 300 --threads 4
```

The reader is chosen from the extension, case-insensitively: `.lp` is read as
CPLEX LP, anything else as MPS, ignoring a `.gz`/`.bz2` suffix
(`instance.mps.gz` is MPS). `--format` overrides that. A file that is not in
the format it is read as fails with an error instead of quietly solving as an
empty or nonsense model.

An RHS entry on the MPS objective row is read as the *negated* objective
constant (`obj_offset = -RHS`), as HiGHS and CPLEX do; the Netlib reference
values in `tests/data/netlib.solu` follow the same convention.

Key options:

| Option | Default | Description |
|--------|---------|-------------|
| `--format auto\|mps\|lp` | auto | Force the model file reader |
| `--time-limit <s>` | 3600 | Wall-clock time limit |
| `--threads <n>` | 1 | Parallel tree-search threads |
| `--gap-tol <g>` | 1e-4 | Relative optimality gap tolerance |
| `--node-limit <n>` | 1000000 | Maximum branch-and-bound nodes |
| `--presolve` / `--no-presolve` | on | Presolve reductions |
| `--cuts` / `--no-cuts` | on | Cutting planes |
| `--conflict-watches` / `--no-conflict-watches` | off | Index learned path no-goods by watched bound conditions in the serial tree |
| `--conflict-activity-branching` / `--no-conflict-activity-branching` | off | Bias serial pseudocost ranking by decayed learned-conflict activity |
| `--conflict-restart-threshold <n>` | 0 | Restart serial search when recent conflict activity reaches `n` and the open queue exceeds the restart floor. Use 0 to disable the trigger |
| `--geometric-restarts` / `--no-geometric-restarts` | off | Delay enabled serial restart triggers for 96 nodes from the start of tree search. Double the minimum interval after each restart |
| `--bound-conflict-learning` / `--no-bound-conflict-learning` | off | Learn and propagate 1-UIP bound conflicts in serial search on all-discrete models |
| `--reconvergence-cuts` / `--no-reconvergence-cuts` | off | Add checked binary reconvergence cuts in serial search on all-discrete models with cuts and conflicts enabled |
| `--conflict-ray-shortening` / `--no-conflict-ray-shortening` | off | Shorten serial tree no-goods when a rechecked LP ray proves infeasibility with fewer branch bounds |
| `--pre-lp-propagation` / `--no-pre-lp-propagation` | off | Use bounded row propagation to prune contradictory child domains before simplex |
| `--pre-lp-propagation-bounds` / `--no-pre-lp-propagation-bounds` | off | Apply bounded row deductions before simplex at depth-scheduled tree presolve nodes of all-discrete models in serial or opportunistic search |
| `--root-cut-recheck` | off | Rank uninstalled root cuts by current efficacy and fill the round quota subject to family caps |
| `--root-cut-propagation` | off | Apply bounded row deductions from the root model and installed cuts during root cut rounds |
| `--root-cut-inference` | off | Propagate installed rows and use retained global bounds in implied-bound separation. Also uses cliques with `--clique-table` and binary implications with `--probing` or `--systematic-probing` |
| `--root-inactive-cut-propagation` | off | Use a bounded set of globally valid, uninstalled pool rows, including rows satisfied by the current LP point. Tightenings wake affected active rows |
| `--root-cut-propagation-row-budget N` | 0 | Stop after the cut batch that spends the cumulative visit budget, including visits to inactive cuts, cliques, and implications when enabled. Zero keeps only the per-round limit |
| `--root-flow-cover` | off | Separate singleton signed flow covers from explicit on/off links and continuous balance rows at the root |
| `--sos2-adjacent-intervals` | off | Linearize SOS2 with one binary per adjacent interval and a linear number of rows |
| `--root-feasjump-restarts <n>` | 4 | Cap root feasibility-jump passes at 0 through 4, including the initial pass. Use 0 to skip the heuristic |
| `--barrier` | off | Use barrier for root LP |
| `--barrier-dualize off\|on\|auto` | off | With barrier, force the compact dual with `on` or compare estimated Newton-factor work in both orientations with `auto` |
| `--pdlp` | off | Use PDLP for root LP |
| `--concurrent-root` | off | Race LP solvers at root |
| `--quiet` | off | Suppress progress output |
| `--probing` / `--no-probing` | off | Root probing in the solver (fixings, implications, variable bounds) |
| `--probing-rounds <n>` / `--probing-probes <n>` / `--probing-time <s>` | 3 / 10000 / 30 | Rounds, probes per legacy round, and a time limit checked between pairs in both modes |
| `--systematic-probing` / `--no-systematic-probing` | off | Use a single checked root probing pass in column order. Learn binary fixings and implications and publish checked conflicts in serial search. Takes precedence over `--probing` |
| `--systematic-probing-pairs <n>` / `--systematic-probing-rows <n>` | 128 / 10000 | Cap complete binary probe pairs and row visits per branch |
| `--root-rapid-learning` / `--no-root-rapid-learning` | off | Search binary branches with row propagation before the root LP. Import checked global bounds, serial conflicts and validated incumbents. Stop after two consecutive eight-node blocks yield no new facts |
| `--root-rapid-learning-nodes <n>` / `--root-rapid-learning-rows <n>` | 128 / 20000 | Cap search nodes and total row visits. A zero cap skips the stage |
| `--presolve-probing` / `--no-presolve-probing` | off | Probing pass inside presolve |
| `--presolve-probing-rounds <n>` / `--presolve-probing-probes <n>` / `--presolve-probing-time <s>` | 2 / 5000 / 5 | Its budget, per presolve call |
| `--clique-table` / `--no-clique-table` | off | Root conflict graph and clique table (propagation, clique cuts) |
| `--plunge-depth <n>` / `--no-plunge` | 0 (off) | Maximum consecutive dives after a branch; serial search only |
| `--plunge-bound-quotient <q>` | 0.05 | Decline a dive whose bound is this far (relative) behind the best |
| `--dual-primal-devex` / `--dual-no-primal-devex` | off | LP solves (`--dual`): Devex pricing for the simplex's primal pivots once the basis is primal feasible, as after added columns (`DualSimplexOptions::enable_primal_devex`) |
| `--dual-dense-row-price` | off | LP solves (`--dual`): assemble every pivot row by the dense scan instead of the hyper-sparse row price, a baseline for benchmarking it; the `Row price:` line (not under `--quiet`) reports which kernel ran (`DualSimplexOptions::force_dense_pivot_row`) |
| `--dual-row-price-actual-work` | off | LP solves (`--dual`): charge each pivot-row assembly the work it does (rows visited plus the nonzeros and slack entry of each row it accumulates) instead of the whole matrix, which changes `Work units:` and so the DSE-to-Devex cost fallback; the `DSE fallbacks:` line, printed on every dual LP solve not under `--quiet`, counts those fallbacks by cause (`DualSimplexOptions::row_price_actual_work`) |

Run `./build/mipx-solve --help` for the full list. The probing, clique-table,
plunging and primal-Devex switches are off by default until benchmarks decide otherwise;
leaving a flag out keeps the solver's own default.

Barrier dualization can shrink the normal equations on LPs with many more
rows than columns. Returned solutions and objectives use the input model's
coordinates. The C++ option is `BarrierOptions::dualize`.

## Python API

```bash
python3 -m venv .venv
.venv/bin/pip install .[test]
```

```python
import mipx

result = mipx.solve_file("instance.mps", time_limit=60, verbose=False)
print(result.status, result.objective)

model = mipx.read_model("instance.lp")                           # reader from the extension
model = mipx.read_model("instance.txt", format=mipx.ModelFormat.Lp)  # or forced
```

Every reader checks that the file really is in the format it is read as and
raises `mipx.ModelFormatError` (a `ValueError`) when it is not, rather than
returning an empty model. `solve_mps` is the older name of `solve_file` and
behaves identically: despite the name it reads `.lp` files as LP.

Pass `sos2_adjacent_intervals=True` to `solve_file` or call
`solver.set_sos2_adjacent_intervals(True)` before `solver.load(model)` to use
adjacent interval selectors for SOS2 sets.

## Tests

```bash
ctest --test-dir build --parallel
```

Netlib and MIPLIB instances are not in the repo. Download them for benchmark tests:

```bash
./tests/data/download_netlib.sh           # full Netlib LP set
./tests/data/download_miplib.sh --small   # curated MIPLIB subset
```

Tests that require missing instances are skipped automatically.

## Project structure

```
include/mipx/     Public headers
src/
  lp/             Dual simplex, barrier, PDLP, LU factorization, sparse matrix
  mip/            Branch-and-bound, node queue, branching, domain propagation
  cuts/           Cut pool, separators, cut manager
  presolve/       Reductions and postsolve stack
  heuristics/     Primal heuristics and runtime orchestration
  io/             MPS/LP readers, solution file I/O
  cli/            Command-line interface
tests/            Catch2 tests and benchmark instances
python/           nanobind bindings and pytest tests
```

## References

- [HiGHS](https://github.com/ERGO-Code/HiGHS) -- high-performance LP/MIP solver
- [SCIP](https://scipopt.org/) -- constraint integer programming framework
- [Achterberg (2007)](https://doi.org/10.1007/s10107-006-0023-0) -- branch-and-cut thesis
- [Maros (2003)](https://doi.org/10.1007/978-1-4615-0257-9) -- simplex textbook

## License

[MIT](LICENSE). Code translated from other projects keeps their notices; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
