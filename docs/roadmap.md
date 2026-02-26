# mipx Roadmap

Stepwise implementation plan for an end-to-end MIP solver (branch-and-cut) in C++23.

Each step builds on the previous, produces something testable, and is scoped for 1–2 sessions.

## Table of Contents

- Status: `🟢 done`, `⚪ planned`
- [Dependency Graph](#dependency-graph)
- [🟢 Step 1: Project Skeleton](#step-1)
- [🟢 Step 2: Sparse Matrix](#step-2)
- [🟢 Step 3: LP Problem + File I/O](#step-3)
- [🟢 Step 4: MIPLIB Test Framework](#step-4)
- [🟢 Step 5: LP Solver Interface](#step-5)
- [🟢 Step 6: LU Factorization](#step-6)
- [🟢 Step 7: Dual Simplex](#step-7)
- [🟢 Step 8: Incremental LP Updates](#step-8)
- [🟢 Step 9: Domain Propagation](#step-9)
- [🟢 Step 10: Node Queue + Branching](#step-10)
- [🟢 Step 11: MIP Solver Shell](#step-11)
- [🟢 Step 12: Cutting Planes](#step-12)
- [🟢 Step 13: Presolve](#step-13)
- [🟢 Step 14: Primal Heuristics](#step-14)
- [🟢 Step 15: Parallel Tree Search](#step-15)
- [🟢 Step 16: Heuristic Runtime Subsystem](#step-16)
- [🟢 Step 17: LP-Free Parallel Pre-Root Heuristic Stage](#step-17)
- [⚪ Step 18: LP-Light Heuristics Integration](#step-18)
- [⚪ Step 19: Adaptive Portfolio Orchestrator](#step-19)
- [⚪ Step 20: Python API + Multi-Platform Release Pipeline](#step-20)
- [Janitor Block (pre-Step-21)](#janitor-pre21)
- [🟢 Step 21: Branching Quality Upgrade](#step-21)
- [🟢 Step 22: Core Cut Family Expansion](#step-22)
- [🟢 Step 23: AUTO Cut Policy + Cut Manager](#step-23)
- [🟢 Step 24: In-Tree Cut Management](#step-24)
- [🟢 Step 25: Conflict Analysis + No-Good Learning](#step-25)
- [🟢 Step 26: Search + Restart Controller](#step-26)
- [🟢 Step 27: In-Processing Presolve](#step-27)
- [🟢 Step 28: MIP Feature Coverage Expansion](#step-28)
- [🟢 Step 29: Reproducibility + Tuning + Benchmark Hardening](#step-29)
- [Janitor Block (post-Step-29)](#janitor-post29)
- [Post-Step-29 Expansion Steps (planned)](#post29-expansion)
- [🟢 Step 30: Barrier / Interior-Point LP Backend](#step-30)
- [🟢 Step 31: PDLP + GPU LP Backend](#step-31)
- [⚪ Step 32: Concurrent Root LP Racing (CPU + GPU)](#step-32)
- [⚪ Step 33: Symmetry Handling](#step-33)
- [⚪ Step 34: Exact LP Refinement Mode](#step-34)

## Dependency Graph

```
1 (Skeleton)
├── 2 (Sparse Matrix)
│   └── 3 (LP Problem + I/O)
│       ├── 4 (MIPLIB Test Framework)
│       ├── 5 (LP Solver Interface)
│       │   └── 6 (LU Factorization)  ←── can parallel with 4
│       │       └── 7 (Dual Simplex)  ←── needs 4 for Netlib validation
│       │           └── 8 (Incremental LP)
│       │               ├── 9 (Domain Propagation)  ┐
│       │               └── 10 (Node Queue)          ├── parallel
│       │                   └── 11 (MIP Shell) ←─────┘ merges 9+10
│       │                       ├── 12 (Cuts)         ┐
│       │                       ├── 13 (Presolve)     ├── parallel
│       │                       └── 14 (Heuristics)   ┘
│       │                           └── 15 (Parallel Tree)
│       │                               ├── 16 (Heuristic Runtime / mip-heuristics port base)
│       │                               │   ├── 17 (LP-free pre-root heuristics)
│       │                               │   ├── 18 (LP-light heuristics)
│       │                               │   ├── 19 (Adaptive portfolio)  ← needs 17 + 18
│       │                               │   └── 20 (Python API + release pipeline)
│       │                               │       └── Janitor Block (quality/perf catch-up gate)  ← depends on 17 + 18 + 19 + 20
│       │                               └── 21 (Branching quality, core MIP track)
│       │                                   └── 22 (Core cut family expansion)
│       │                                       └── 23 (AUTO cut policy)
│       │                                           └── 24 (In-tree cut management)
│       │                                               └── 25 (Conflict learning)
│       │                                                   └── 26 (Search + restart control)
│       │                                                       └── 27 (In-processing presolve)  ← also needs 13
│       │                                                           └── 28 (MIP feature coverage)  ← also needs 11
│       │                                                               └── 29 (Repro/tuning hardening)  ← also needs 19 + 20
│       │                                                                   ├── 30 (Barrier / IPM backend ✅) ┐
│       │                                                                   ├── 31 (PDLP + GPU backend ✅)      ├── 32 (Concurrent root LP racing)
│       │                                                                   └── 33 (Symmetry handling)
│       │                                                                       └── 34 (Exact LP refinement mode)
```

**Parallel opportunities:**
- Steps 4 + 6: test framework and LU factorization are independent after Step 3
- Steps 9 + 10: domain propagation and node queue are independent after Step 8
- Steps 12 + 13 + 14: cuts, presolve, and heuristics are independent after Step 11

**Post-15 expansion sequence (MIP-first):**
- `15 -> 16 -> 17 -> 18 -> 19 -> 20` (port `mip-heuristics` first; shutdown path)
- `17 + 18 + 19 + 20 -> Janitor Block`
- `Janitor Block -> 21 -> 22 -> 23 -> 24 -> 25 -> 26 -> 27 -> 28`
- `19 + 20 + 28 -> 29`
- `29 -> 30` and `29 -> 31`; then `30 + 31 -> 32` (advanced LP backend and root-concurrent execution track)
- `29 -> 33 -> 34` (symmetry + exactness track)

---

<a id="step-1"></a>

## Step 1: Project Skeleton ✅

[Back to top](#table-of-contents)

**Goal:** Buildable project with CI-ready test infrastructure.

**Status:** Complete. CMake/Catch2 project skeleton, core types, and optional TBB build path are in place.

**Deliverables:**
- CMake build system (C++23, `-Wall -Wextra -Werror`, sanitizers in debug)
- Directory structure: `src/`, `include/mipx/`, `tests/`, `docs/`
- Catch2 test framework (FetchContent)
- Core type aliases: `Int`, `Real` (double), `Index`, status/sense enums
- Optional TBB dependency (`-DMIPX_USE_TBB=ON`, off by default)

**Test criteria:** `cmake --build . && ctest` passes with a trivial test.

**References:** HiGHS CMakeLists.txt, [baldes](https://github.com/lseman/baldes) build setup.

**Depends on:** nothing
**Unlocks:** 2

---

<a id="step-2"></a>

## Step 2: Sparse Matrix ✅

[Back to top](#table-of-contents)

**Goal:** CSR-primary sparse matrix with lazy CSC view.

**Status:** Complete. `SparseMatrix` CSR/CSC operations, SpMV, and row mutation APIs are implemented and covered by tests.

**Deliverables:**
- `SparseMatrix` class: CSR storage (values, col_indices, row_starts)
- Lazy CSC transpose — built on first column-access, invalidated on mutation
- SpMV: `y = A·x` and `y = Aᵀ·x`
- Row add/remove (efficient for cut management in B&C)
- Column-access iterator for simplex pivot column extraction

**Test criteria:** Round-trip CSR→CSC, SpMV against dense reference, row add/remove preserves structure.

**References:** HiGHS `HighsSparseMatrix`, [baldes](https://github.com/lseman/baldes) sparse utilities.

**Depends on:** 1
**Unlocks:** 3

---

<a id="step-3"></a>

## Step 3: LP Problem + File I/O ✅

[Back to top](#table-of-contents)

**Goal:** Represent LP/MIP problems and read/write standard formats.

**Status:** Complete. LP/MIP model structures and MPS/LP/.solu I/O paths are implemented and validated in tests.

**Deliverables:**
- `LpProblem` struct: objective (min/max), variable bounds, constraint matrix (SparseMatrix), row senses/rhs, integrality markers, names
- MPS reader (fixed + free format, gzip support via zlib)
- MPS writer
- LP format reader (CPLEX-style)
- `.solu` file reader (known optimal values for validation)

**Test criteria:** Round-trip MPS read→write→read matches. Parse a Netlib instance and validate dimensions. Read `.solu` and check value parsing.

**References:** HiGHS `Filereader`, SCIP reader_mps.c.

**Depends on:** 2
**Unlocks:** 4, 5 (parallel)

---

<a id="step-4"></a>

## Step 4: MIPLIB Test Framework ✅ ⚡ parallel with 5→6

[Back to top](#table-of-contents)

**Goal:** Automated benchmark infrastructure against standard LP/MIP test sets.

**Status:** Complete. Netlib/MIPLIB download scripts, benchmark runners, and dataset-backed tests/perf gates are integrated.

**Deliverables:**
- Download script for Netlib LP set and MIPLIB 2017 benchmark/easy subsets
- Benchmark runner: solve instance, compare against `.solu`, report pass/fail/time
- Catch2 test cases parameterized over instance sets
- CI integration: Netlib in CI (small), MIPLIB as manual/nightly

**Test criteria:** Download + parse all Netlib instances without error. Benchmark runner produces summary table.

**References:** mip-heuristics benchmark pattern, MIPLIB 2017 website.

**Depends on:** 3
**Unlocks:** 7 (provides Netlib validation set)

---

<a id="step-5"></a>

## Step 5: LP Solver Interface ✅ ⚡ parallel with 4

[Back to top](#table-of-contents)

**Goal:** Abstract interface that dual simplex (and later barrier/PDLP) will implement.

**Status:** Complete. `LpSolver`/`LpResult` abstraction and interface-backed tests are in place and used by solver components.

**Deliverables:**
- `LpSolver` abstract class:
  - `solve()`, `getStatus()`, `getObjective()`
  - `getPrimalValues()`, `getDualValues()`, `getReducedCosts()`
  - `addRows()`, `removeRows()` (for cutting plane loop)
  - `setBasis()`, `getBasis()` (for warm-starting)
  - `setObjective()`, `setBounds()` (for node subproblems)
- `LpResult` struct with status enum (Optimal, Infeasible, Unbounded, IterLimit, etc.)
- Interface designed so column generation callers can use `addColumns()`/`removeColumns()` later

**Test criteria:** Compiles. Mock implementation passes interface tests.

**References:** HiGHS `Highs` API, flowty-core LP interface.

**Depends on:** 3
**Unlocks:** 6

---

<a id="step-6"></a>

## Step 6: LU Factorization ✅

[Back to top](#table-of-contents)

**Goal:** Sparse LU for basis matrix operations in simplex.

**Status:** Complete. Sparse LU factorization, FTRAN/BTRAN, basis update/refactorization logic, and hot-path optimizations are implemented and tested.

**Deliverables:**
- Sparse Markowitz LU factorization with threshold pivoting
- FTRAN: solve `B·x = b` (forward transformation)
- BTRAN: solve `Bᵀ·y = c` (backward transformation)
- Forrest-Tomlin update for basis changes (not product-form — FT maintains sparsity far better across hundreds of updates)
- Refactorization trigger (growth factor or update count)

**Implementation notes:**
- Markowitz ordering with threshold pivoting is the established approach — no recent alternatives have displaced it
- Forrest-Tomlin (1972) remains the state of the art for basis updates in simplex; Huangfu & Hall explored product-form variants that approach FT performance on some problems but FT is the default in every serious solver
- Hyper-sparsity detection in FTRAN/BTRAN (defer to Step 7 optimization)

**Test criteria:** Factor random sparse matrices, verify `B·(B⁻¹·b) ≈ b`. Rank-1 updates match full refactorization. Numerical stability on ill-conditioned Netlib bases.

**References:** HiGHS `HFactor`, Forrest & Tomlin (1972), Suhl & Suhl (1990), Huangfu & Hall novel update techniques (2015).

**Depends on:** 5
**Unlocks:** 7

---

<a id="step-7"></a>

## Step 7: Dual Simplex ✅

[Back to top](#table-of-contents)

**Goal:** Working LP solver. First real solves.

**Status:** Complete. Solves all 4 Netlib test instances (afiro, sc50a, blend, adlittle) to optimality.

**Deliverables:**
- Dual simplex Phase 1 (cost perturbation for initial dual feasibility) + Phase 2 (bound flipping + primal simplex fallback)
- Harris dual ratio test with anti-cycling (fixed variables excluded from entering selection)
- Equilibration scaling (row/col max-norm)
- Augmented matrix formulation `[A | -I]` with slack variables
- Primal simplex pivot fallback for dual-infeasible cleanup after perturbation removal
- Unbounded detection via primal ratio test
- CLI tool `mipx-solve` with MPS file input
- HiGHS comparison benchmark script (`tests/benchmark_vs_highs.py`)

**Optimization pass (complete):**
- Devex approximate steepest-edge pricing for CHUZR
- Bound Flipping Ratio Test (BFRT) in CHUZC
- Row-wise pivot row computation (CSR access)
- Reduced per-iteration overhead (avoid lazy CSC rebuild, eliminate temp allocations)

**Deferred:**
- Hyper-sparsity exploitation in FTRAN/BTRAN
- PAMI-style multi-iteration parallelism beyond current SIP scan parallelization

**MIP-first optimization wave (sequential execution plan):**
1. GPU workstream scaffold for root policy (`RootLpPolicy`, alternate backend adapter seam) — **done**
2. Node reoptimization fast path (bound-delta apply/restore, avoid full bound reset) — **done**
3. Basis lifecycle hardening for reoptimization (`setBasis` fast path, row-removal remap) — **done**
4. Pricing and scan reduction (partial pricing + periodic full refresh fallback) — **done**
5. LU/FTRAN-BTRAN hot-path memory optimization (no per-row heap allocations in Markowitz updates, reusable solve/update scratch buffers) — **done**
6. Remaining dual-simplex optimizations bundle (adaptive refactorization/stability triggers, SIMD/memory-layout tuning where safe, additional runtime toggles) — **done** (runtime toggles, adaptive refactorization guard, AVX2-gated dense kernels for dual/slack-vector updates, O(1) nonbasic position map, and configurable SIMD build targeting with native default landed)
7. Intra-iteration parallel simplex (SIP-style), gated by model-shape wins and numerical stability — **done** (dual-infeasibility scan, CHUZC candidate scan + optional candidate sort, and CHUZR leaving-row scan have TBB-backed parallel paths behind runtime options, with min-size/min-thread gates plus stall-based serial fallback; default remains off)

**Wave notes:**
- Tree solve remains dual-simplex-first for MIP warm-start and basis continuity.
- GPU concurrent root racing remains deferred until an in-repo PDLP/barrier backend exists.
- MIP-centric instrumentation is active (root/node LP timing split, warm/cold counters) and should be extended as Step 6 lands.

**Performance regression gates (required for all optimization PRs):**
- Correctness gate: no test regressions.
- Performance gate: default `work_units` regression allowance is 0% (explicit override required to relax).
- Combined LP+MIP gate: `tests/perf/run_full_gate.sh` is the default pre-merge workflow.
- Stability gate: no increase in numerical failures / stall rate beyond tolerance.
- Fallback gate: new optimization stays runtime-toggleable until gates are met.

**Test criteria:** Solve all Netlib instances to optimality. Objective matches `.solu` values within tolerance. Competitive iteration counts vs. reference.

**References:** HiGHS `HDual`, Koberstein (2005) dual simplex thesis (best description of Harris + BFRT + cost shifting integration), Maros (2003) textbook, Hall & McKinnon (2005) hyper-sparsity, Huangfu & Hall (2018) parallel dual simplex.

**Depends on:** 4, 6
**Unlocks:** 8

---

<a id="step-8"></a>

## Step 8: Incremental LP Updates ✅

[Back to top](#table-of-contents)

**Goal:** Efficient LP modifications for branch-and-cut use.

**Status:** Complete. All incremental operations implemented and tested (8 test cases).

**Deliverables:**
- `setColBounds()`: update variable bounds with scaling, adjust nonbasic status/value
- `setObjective()`: update objective with sign flip and scaling
- `addRows()`: extend constraint matrix, row bounds, and basis with new slack variables
- `removeRows()`: remove rows, compact data, invalidate basis for cold restart
- Warm-start `solve()`: reuse existing basis (skip `setupInitialBasis()`), re-scale nonbasic primals
- Fixed `setBasis()` to allocate solution vectors before use

**Test criteria:** Add rows, re-solve, verify optimal. Warm-started solve uses fewer iterations than cold start. Bound change + re-solve matches fresh solve.

**References:** HiGHS `HighsLpSolverObject`, SCIP LP interface.

**Depends on:** 7
**Unlocks:** 9, 10 (parallel)

---

<a id="step-9"></a>

## Step 9: Domain Propagation ✅ ⚡ parallel with 10

[Back to top](#table-of-contents)

**Goal:** Infer tighter variable bounds from constraints and integrality.

**Status:** Complete. Domain propagator with checkpoint/restore, propagation queue, and infeasibility detection.

**Deliverables:**
- `DomainPropagator`: analyze `a·x ≤ b` to tighten variable bounds
- Change stack with checkpoint/restore for tree search backtracking
- Propagation queue (constraints to re-examine after a bound change)
- Conflict detection: identify infeasibility from bounds

**Test criteria:** Propagation tightens bounds on textbook examples. Checkpoint + restore round-trips correctly. Infeasibility detected on contradictory bounds.

**References:** SCIP `prop_obbt.c`, `domain.c`; Achterberg (2007) thesis §7.

**Depends on:** 8
**Unlocks:** 11

---

<a id="step-10"></a>

## Step 10: Node Queue + Branching ✅ ⚡ parallel with 9

[Back to top](#table-of-contents)

**Goal:** Tree search infrastructure.

**Status:** Complete. NodeQueue with best-first/depth-first policies, MostFractional and FirstFractional branching, createChildren.

**Deliverables:**
- `BnbNode`: LP bound, parent pointer, branching decision, depth, basis snapshot
- `NodeQueue` with policies: best-first, depth-first
- Branching rules: most fractional, first fractional
- `createChildren()`: create left/right child nodes from branching decision

**Test criteria:** Node queue ordering is correct for each policy. Branching selects correct variable. Pruning removes dominated nodes.

**References:** SCIP `branch_relpscost.c`, Achterberg et al. (2005) branching paper, HiGHS `HighsMipSolver`.

**Depends on:** 8
**Unlocks:** 11

---

<a id="step-11"></a>

## Step 11: MIP Solver Shell ✅

[Back to top](#table-of-contents)

**Goal:** End-to-end MIP solving. The milestone.

**Status:** Complete. Branch-and-bound MIP solver with warm-started LP re-solves, bound pruning, and progress logging.

**Deliverables:**
- `MipSolver` class: load problem → solve root LP → branch-and-bound loop → report solution
- Incumbent tracking: best feasible solution found so far
- Pruning: by bound, by infeasibility, by integrality (all-integer solution)
- Gap tracking: `(incumbent - best_bound) / incumbent`
- Limits: node limit, time limit, gap tolerance
- CLI tool (`mipx-solve`) auto-detects MIP problems and uses MipSolver
- **Bug fix:** Internal LP primals kept in scaled coordinates (no unscaleResults in-place), getters unscale on-the-fly. Fixes warm-start double-scaling issue.

**Test criteria:** Small MIPs solved to optimality (knapsack, simple branching). Infeasibility detection works. Node/time limits respected. MIPLIB gt2 runs without crashing.

**References:** HiGHS `HighsMipSolver`, SCIP `solve.c`, Achterberg (2007).

**Depends on:** 9, 10
**Unlocks:** 12, 13, 14 (parallel)

---

<a id="step-12"></a>

## Step 12: Cutting Planes ✅ ⚡ parallel with 13, 14

[Back to top](#table-of-contents)

**Goal:** Strengthen LP relaxations with valid inequalities.

**Status:** Complete. CutPool with efficacy/parallelism filtering, Gomory MIR separator, MipSolver integration. 10 new tests (147 total).

**Deliverables:**
- `CutPool`: store cuts with efficacy ranking, cosine similarity parallelism filtering, age tracking, purging
- `GomorySeparator`: Gomory MIR cuts from simplex tableau with proper sign adjustment for at-upper nonbasic variables
- `getTableauRow()` in DualSimplexSolver: returns external (unscaled) tableau row for cut generation
- `addRows()` fixed to apply column scaling to new row coefficients
- Cutting plane rounds at root node (configurable max rounds and cuts per round)
- MipSolver integration with enable/disable toggle

**Test criteria:** Root gap closed on standard instances. Fewer nodes needed vs. no-cuts solve. Cut pool doesn't grow unboundedly.

**References:** HiGHS `HighsCutPool`, SCIP `sepa_gomory.c`, Cornuéjols (2007) survey.

**Depends on:** 11
**Unlocks:** 15

---

<a id="step-13"></a>

## Step 13: Presolve ✅ ⚡ parallel with 12, 14

[Back to top](#table-of-contents)

**Goal:** Reduce problem size before solving.

**Status:** Complete. Iterative dirty-workset presolve with postsolve recovery is integrated and validated across unit + MIPLIB parsing/solve tests.

**Deliverables:**
- `Presolver` with reductions: fixed variables, singleton rows/columns, forcing/dominated constraints, coefficient tightening, activity-based tightening, implied equations, dual fixing, duplicate rows
- `PostsolveStack`: record reductions, undo them to recover full solution
- Iterative presolve loop until no more reductions found (dirty row/column workset engine)
- Statistics: vars/cons removed, bounds tightened

**Test criteria:** Presolve + solve + postsolve matches direct solve. Measurable reduction on MIPLIB instances. No incorrect eliminations (validated by postsolve).

**References:** HiGHS `Presolve`, Achterberg et al. (2020) presolve paper.

**Depends on:** 11
**Unlocks:** 15

---

<a id="step-14"></a>

## Step 14: Primal Heuristics ✅ ⚡ parallel with 12, 13

[Back to top](#table-of-contents)

**Goal:** Find feasible solutions faster.

**Status:** Complete. Root/tree heuristic portfolio now includes LP-based bootstrap heuristics plus adaptive budget scheduling, with strict work-units gate compatibility.

**Deliverables:**
- Rounding heuristic: round LP solution, check feasibility
- Diving heuristics: fractional diving, coefficient diving, guided diving
- RINS (Relaxation Induced Neighborhood Search): incumbent-guided agreement fixing, periodic in-tree scheduling, fixed-count diagnostics
- RENS (Relaxation Enforced Neighborhood Search): LP-neighborhood fixing without incumbent
- Feasibility Pump (lightweight): round + guided LP repair loop with cycle perturbation
- Auxiliary-objective heuristic: temporary integrality-guiding LP objective with full LP state restore
- Zero-objective heuristic: objective-free LP feasibility probe with integer repair
- Local Branching heuristic: incumbent-centered binary neighborhoods with multi-radius root portfolio
- Adaptive heuristic budget manager: dynamic root/tree heuristic gating by work-share and recent hit-rate
- Heuristic scheduler: run at root, periodically in tree, after incumbent improvement
- MIP solver integration:
  - Root portfolio: rounding -> aux-objective -> zero-objective -> feasibility pump -> RENS -> RINS -> local branching (radii 8/16/24; gated by root integer infeasibility/size thresholds)
  - Tree portfolio: periodic RINS on promising fractional nodes with incumbent and relative-gap gating, with adaptive spacing
  - LP state hygiene: bounds/objective/iteration-limit/basis restoration around heuristic subsolves
  - Instrumentation: per-heuristic LP iterations/work units and skip reasons for tuning
- Regression hardening:
  - Tight default RINS/RENS/FeasPump budgets and activation thresholds to protect branch-and-bound throughput
  - `tests/perf/run_full_gate.sh` (metric `work_units`, default 0% regression allowance) used as required gate for heuristic tuning

**Test criteria:** Heuristics find incumbent earlier (fewer nodes to first feasible). Solution quality comparable to optimal. Time overhead acceptable.

**References:** SCIP `heur_rounding.c`, `heur_diving.c`, `heur_rins.c`; Berthold (2006).

**Depends on:** 11
**Unlocks:** 15

---

<a id="step-15"></a>

## Step 15: Parallel Tree Search ✅

[Back to top](#table-of-contents)

**Goal:** Exploit multicore via TBB.

**Status:** Complete. Parallel B&B with TBB task_group, mutex-protected shared state, per-thread LP solvers. 6 new tests (153 total with TBB, 148 without).

**Deliverables:**
- `processNode()`: extracted pure function for node processing (apply bounds, warm-start, solve LP, branch)
- `solveSerial()`: original B&B loop refactored as method
- `solveParallel()`: TBB-based parallel B&B — each thread has own `DualSimplexSolver` instance
- Mutex-protected node queue and incumbent with atomic counters
- `setNumThreads(n)`: parameter to control thread count (default 1 = serial)
- CLI `--threads N` flag for `mipx-solve`
- Graceful fallback: compiles and runs single-threaded when TBB unavailable

**Test criteria:** Correct results match serial solver. Speedup on 4+ cores. Graceful fallback without TBB.

**References:** SCIP concurrent solver, HiGHS parallel MIP, cuOpt parallel B&B.

**Depends on:** 12, 13, 14

---

<a id="step-16"></a>

## Step 16: Heuristic Runtime Subsystem (mip-heuristics merge foundation) ✅

[Back to top](#table-of-contents)

**Goal:** Introduce a composable heuristic runtime that supports cross-heuristic cooperation.

**Status:** Complete. A mipx-native runtime foundation now exists with
`HeuristicRuntime`, thread-safe `SolutionPool`, restart strategy selection,
callback hooks, and deterministic/opportunistic + seed controls exposed via CLI/API.
Latest hardening integrated shared runtime budgeting across parallel workers,
canonicalized work-unit accounting for deterministic gating, explicit finish callback
delivery, and regression tests for deterministic/opportunistic runtime paths.

**Deliverables:**
- Two-level heuristic API:
  - standalone solve entrypoint (full solver context)
  - worker entrypoint (shared state, seed, thread id)
- Thread-safe `SolutionPool` for incumbent sharing across heuristic workers
- Restart strategy engine (uniform/tournament crossover, perturbation, polarity/activity/distance-guided restarts)
- Callback interface for heuristics (`on_header`, `on_incumbent`, `on_heartbeat`, `on_finish`)
- Deterministic vs opportunistic mode switch (reproducibility vs max throughput)

**Test criteria:**
- Same-seed determinism in deterministic mode
- Opportunistic mode improves incumbent discovery speed on multi-thread runs
- No regression in existing MIP solve correctness/perf gates

**References:** `mip-heuristics` `Heuristic` + `SolutionPool` + callback design.

**Depends on:** 14, 15
**Unlocks:** 17, 18, 19

---

<a id="step-17"></a>

## Step 17: LP-Free Parallel Pre-Root Heuristic Stage ✅

[Back to top](#table-of-contents)

**Goal:** Add a parallel pre-root primal stage (before/alongside root LP) using LP-free heuristics.

**Status:** Complete. `mipx` now has an opt-in LP-free pre-root stage with
parallel worker execution in opportunistic mode, deterministic single-thread
reproducibility mode, shared incumbent handoff via `SolutionPool`, configurable
work/round budgets with early-stop, and telemetry for calls/work/time-to-first-feasible.

**Deliverables:**
- Integrate LP-free heuristic arms from `../mip-heuristics`:
  - Feasibility Jump
  - Fix-Propagate-Repair (LP-free ranking)
  - Local-MIP neighborhood search
- Configurable pre-root budget and early-stop policy once good incumbent is found
- Shared incumbent handoff into branch-and-bound and root LP cuts/branching
- Pre-root effectiveness telemetry (time-to-first-feasible, incumbent quality at root)

**Test criteria:**
- Earlier feasible incumbents on hard instances vs LP-only root start
- No strict `work_units` regression by default when pre-root stage is disabled
- With pre-root enabled, improved anytime objective profile on selected MIPLIB cases

**References:** Feasibility Jump (MPC 2023), Local-MIP (CP 2024 / AIJ 2025), FPR (MPC 2025).

**Depends on:** 16
**Unlocks:** 19, 20

---

<a id="step-18"></a>

## Step 18: LP-Light Heuristics Integration

[Back to top](#table-of-contents)

**Goal:** Add LP-assisted heuristic arms without forcing core solver dependency changes.

**Deliverables:**
- Optional LP-backend adapter for heuristic-only LP calls (HiGHS first target)
- Scylla-style LP-fractionality-ranked FPR arm
- LP diving arm (fractional/guided/pseudocost scoring)
- Build-time flag and runtime capability detection for optional arms

**Test criteria:**
- No behavior change when optional backend is off
- Additional feasible/improved incumbents when optional arms are enabled
- No strict-gate regression in default configuration

**References:** Scylla (OR 2023), PDLP-ranked fix-and-propagate follow-up work.

**Depends on:** 16
**Unlocks:** 19

---

<a id="step-19"></a>

## Step 19: Adaptive Portfolio Orchestrator

[Back to top](#table-of-contents)

**Goal:** Replace fixed heuristic schedules with data-driven arm selection.

**Deliverables:**
- Thompson Sampling scheduler over available heuristic arms (LP-free + LP-light + existing LP-based arms)
- Warm-up arm rotation and reward model (first feasible / new best / stagnant / fail)
- Adaptive epoch effort scheduler (shrink on stagnation, grow on wins)
- Arm-level observability (selection counts, reward stats, incumbent contribution)

**Test criteria:**
- Portfolio determinism tests in deterministic mode
- Better median time-to-first-feasible and best-at-T metrics on benchmark set
- Strict non-regression gates remain default for merge

**References:** `mip-heuristics` adaptive portfolio, SCIP ALNS bandit literature.

**Depends on:** 16, 17, 18
**Unlocks:** 20, 29

---

<a id="step-20"></a>

## Step 20: Python API + Multi-Platform Release Pipeline

[Back to top](#table-of-contents)

**Goal:** Provide first-class Python distribution and automated releases.

**Deliverables:**
- Nanobind module for core model/solver APIs
- `pyproject.toml` + scikit-build-core packaging with stable ABI wheels
- cibuildwheel matrix for:
  - Linux x86_64
  - Linux aarch64
  - macOS arm64
  - Windows x64
- GitHub Actions workflows:
  - CI build/test (C++)
  - wheel + sdist build
  - tagged release publish to PyPI via trusted publisher (OIDC)
- Python binding tests in CI and wheel smoke tests

**Test criteria:**
- `pip install` wheel works on all target platforms
- Python API parity checks for basic load/solve/result flow
- Release process reproducible from tag-only trigger

**References:** `mip-heuristics` nanobind/scikit-build/cibuildwheel pipeline.

**Depends on:** 16, 17, 19
**Unlocks:** external integration and benchmark ecosystem growth, 29

---

<a id="janitor-pre21"></a>

## Janitor Block (recurring; run before Step 21 and after each major feature wave)

**Purpose:** Keep correctness/performance baselines trustworthy as new capabilities land.
**Dependency:** run after Step 17 + Step 18 + Step 19 + Step 20.

**Janitor 1: Correctness and E2E validation**
- Ensure all relevant unit/integration tests exist for newly added features.
- Add/update end-to-end solve checks against known optimal or best-known solutions (`.solu`) on curated Netlib/MIPLIB sets.
- Require pass on full correctness suite before performance comparisons are accepted.

**Janitor 2: `work_units` KPI coverage audit**
- Verify `work_units` accounting is present for all relevant hot paths and solver stages (root, tree, heuristics, cuts, presolve, branching probes).
- Add missing counters and expose them in logs/benchmark outputs where absent.
- Keep regression gating on `work_units` as default (0% regression unless explicitly overridden).

**Janitor 3: Low-level optimization opportunity pass**
- Run profiling/perf analysis to identify top hot loops and low-hanging fruit.
- Evaluate optimization options in order: data-layout/locality changes -> SIMD (AVX2/AVX-512 where available) -> algorithmic parallelism -> selective kernel rewrites (C/assembly/GPU offload) when justified by measured gains.
- Current state: AVX2-gated kernels are already used in selected dual-simplex dense update paths; broad GPU offload and C/assembly microkernels are not yet standard in `mipx`.
- Any low-level rewrite must preserve numerical behavior and pass strict correctness + `work_units` gates.

**Janitor 4: Documentation synchronization**
- Update `README.md` for user-visible behavior, flags, workflows, and benchmark commands changed in the cycle.
- Update `docs/roadmap.md` completion markers/status text for finished steps or substeps.
- Refresh related docs/changelogs/benchmark notes so implementation status and guidance stay consistent.

**Cadence:** Run this block periodically (for example every 3-5 merged feature PRs, or at least once per release cycle).

---

<a id="step-21"></a>

## Step 21: Branching Quality Upgrade (core MIP track) ✅

[Back to top](#table-of-contents)

**Goal:** Reduce branch-and-bound tree size by upgrading branching decisions.

**Status:** Complete. Reliability branching with pseudocost learning, capped strong-branch probing, root bootstrap, and branching telemetry is integrated.

**Deliverables:**
- Pseudocost storage and updates (up/down gains, reliability counters)
- Reliability branching: strong-branch a limited candidate set until pseudocosts are reliable, then switch to pseudocost scoring
- Root strong-branch bootstrap to seed pseudocosts early
- Candidate prefiltering (fractionality + estimated gain) with capped strong-branch budget per node
- Logging/instrumentation: strong-branch calls, average probe LP iterations/work units, pseudocost hit-rates

**Test criteria:**
- No correctness regressions (`ctest` unchanged pass rate)
- Strict perf gate preserved: `tests/perf/run_full_gate.sh` with default 0% regression allowance on `work_units`
- On MIPLIB small set, demonstrate reduced median explored nodes/work units vs. current most-fractional branching

**References:** SCIP reliability branching, Achterberg (2007), Fischetti & Monaci branching surveys.

**Depends on:** 15
**Unlocks:** 22

---

<a id="step-22"></a>

## Step 22: Core Cut Family Expansion ✅

[Back to top](#table-of-contents)

**Goal:** Implement the high-impact cut families that are default-relevant in commercial solvers.

**Status:** Complete. Added multi-family root separation (Gomory + MIR + cover + implied-bound + clique + zero-half + mixing), family toggles, numerical safety guards, and per-family root telemetry.

**Deliverables:**
- Separator framework expansion beyond Gomory:
  - MIR/CMIR (row and tableau driven variants)
  - Cover cut families (knapsack cover, lifted cover, mixed binary/integer cover, flow-cover where structure matches)
  - Implied-bound cuts (using propagation/implication data)
  - Clique cuts (binary conflict graph)
  - Zero-half and mixing cuts (initial conservative implementation)
- Family-level runtime toggles and common separator API so families can be independently tuned/disabled
- Numerically safe candidate generation rules (scaling checks, coefficient growth guards, reject unstable candidates)
- Root separation telemetry by family (attempted/generated/accepted, efficacy, LP delta, time)

**Test criteria:**
- No correctness regressions (`ctest`)
- Strict `work_units` gate remains non-regressive by default
- On MIPLIB small set, improved root gap closure and fewer median nodes/work units than Gomory-only baseline

**References:** HiGHS separation modules, SCIP default separators, Gurobi/CPLEX/Xpress cut-family defaults (automatic mode).

**Depends on:** 21
**Unlocks:** 23

---

<a id="step-23"></a>

## Step 23: AUTO Cut Policy + Cut Manager ✅

[Back to top](#table-of-contents)

**Goal:** Add commercial-style automatic cut control instead of fixed static cut settings.

**Status:** Complete. Added `off/conservative/aggressive/auto` cut effort modes with AUTO KPI-driven throttling/promotion, per-round/per-node/global work budgets, root-vs-tree policy hooks, and auditable policy telemetry.

**Deliverables:**
- Cut effort modes: `off`, `conservative`, `aggressive`, `auto` (default)
- Dynamic per-family scheduling in `auto` mode using online KPIs:
  - attempted, generated, accepted, rejected
  - efficacy and orthogonality
  - LP bound improvement per cut-family work unit
  - separation time and resulting LP reoptimization overhead
- Root vs tree adaptive policy:
  - Root: broader family activation, capped rounds with diminishing-return stop rules
  - Tree: selective family activation by depth/node type/stagnation state
- Candidate filtering and budget manager:
  - duplicate/near-parallel filtering
  - numerical safety gating
  - per-node/per-round/global work-unit cut budgets
- Auto-demotion/auto-promotion of families based on recent ROI windows
- Logging/reporting for policy decisions so tuning and regressions are auditable

**Test criteria:**
- Deterministic policy behavior in deterministic mode at fixed seed
- On benchmark set, `auto` is non-regressive by default and improves median root gap/work units over static baseline
- Families with negative ROI are automatically throttled in telemetry-backed runs

**References:** Gurobi/CPLEX/Xpress automatic cut controls, SCIP separator frequency/priority model, HiGHS cut management patterns.

**Depends on:** 22
**Unlocks:** 24, 29

---

<a id="step-24"></a>

## Step 24: In-Tree Cut Management ✅

[Back to top](#table-of-contents)

**Goal:** Move from root-only cutting to tree-effective separation without exploding LP time.

**Status:** Complete. Added depth-gated in-tree cut rounds, local-vs-global cut handling, cut lifecycle (aging/activity/purge/revive), cut-overhead skip logic, and root-vs-tree cut telemetry in solver progress output.

**Deliverables:**
- In-tree cut rounds with depth/node-type gating (aggressive near root, conservative deep in tree)
- Local-vs-global cut policy (local cuts stay on subtree, global cuts promoted by efficacy tests)
- Stronger cut lifecycle (aging, activity tracking, purge/revive)
- Node-level cut skip logic when LP resolve overhead dominates recent gains
- Cut telemetry in progress logs (added/active/purged, root-vs-tree impact)

**Test criteria:**
- No correctness regressions (`ctest`)
- Strict `work_units` gate remains non-regressive by default
- On MIPLIB small set, fewer median nodes/work units than root-only cuts with similar LP-time overhead

**References:** SCIP separator scheduling, HiGHS cut lifecycle patterns.

**Depends on:** 21, 23
**Unlocks:** 25, 27

---

<a id="step-25"></a>

## Step 25: Conflict Analysis + No-Good Learning ✅

[Back to top](#table-of-contents)

**Goal:** Learn from infeasible nodes to prevent repeated dead-end exploration.

**Status:** Complete. Added LP/bound-infeasible conflict extraction, minimized no-good literals, conflict pool aging/reuse for subtree pruning, and conflict-participation branching feedback with solver telemetry.

**Deliverables:**
- Conflict extraction from domain propagation and LP-infeasible subproblems
- No-good constraints / bound-disjunction cuts with clause minimization
- Conflict pool with aging and reuse at sibling/cousin nodes
- Branching feedback from conflict participation scores

**Test criteria:**
- Learned conflicts are valid (never cut incumbent-feasible solutions)
- Reduced repeated infeasibility patterns in regression tests
- Non-regressive strict `work_units` gate

**References:** SCIP conflict analysis, Achterberg (2007), CDCL-inspired MIP conflict learning.

**Depends on:** 21, 24
**Unlocks:** 26

---

<a id="step-26"></a>

## Step 26: Search + Restart Controller ✅

[Back to top](#table-of-contents)

**Goal:** Add strategy control to reduce heavy-tail behavior in branch-and-bound.

**Status:** Complete. Added node-selector portfolio switching (`best-bound`, `best-estimate`, `depth-biased`), stagnation-triggered controlled restarts, dynamic strong-branch budget control, sibling branch-variable reuse cache, and runtime search profiles (`stable/default/aggressive`) exposed in API/CLI with tests.

**Deliverables:**
- Node selector portfolio (best-bound, best-estimate, depth-biased dives) with switch rules
- Controlled restarts/restarts-on-stagnation with incumbent carryover
- Strong-branch budget controller and sibling reuse/caching policy
- Runtime strategy profiles (`stable`, `default`, `aggressive`)

**Test criteria:**
- Deterministic profile remains reproducible at fixed seed
- Stagnation cases show improved anytime incumbent behavior
- Strict gate stays non-regressive by default

**References:** SCIP search control, MIP restart studies, reliability branching practice.

**Depends on:** 21, 25
**Unlocks:** 27, 29

---

<a id="step-27"></a>

## Step 27: In-Processing Presolve ✅

[Back to top](#table-of-contents)

**Goal:** Re-apply safe reductions during tree search, not only at root.

**Status:** Complete. Added depth/fractionality-triggered in-tree presolve passes, safe local rollback via node-bound restoration, activity + reduced-cost tightening refresh, and a low-yield benefit model with telemetry and skip controls.

**Deliverables:**
- Triggered in-tree presolve at selected nodes (depth/fractionality/time based)
- Safe rollback/postsolve mapping for local reductions
- Reduced-cost and activity tightening refresh in long-running subtrees
- Presolve benefit model to skip low-yield invocations

**Test criteria:**
- Solution mapping remains correct on all tested instances
- Measurable subtree-size reduction on selected MIPLIB cases
- No strict-gate regression

**References:** In-processing in SCIP/CP-SAT style solvers.

**Depends on:** 13, 24, 26
**Unlocks:** 28, 29

---

<a id="step-28"></a>

## Step 28: MIP Feature Coverage Expansion ✅

[Back to top](#table-of-contents)

**Goal:** Support common production model features beyond baseline MPS integer LP.

**Status:** Complete. Added SOS1/SOS2, indicator, semi-continuous, and semi-integer model constructs with validation + linearization fallback, integrated feature linearization into solver load and MPS writer, and added solve + round-trip tests.

**Deliverables:**
- SOS1/SOS2 support
- Indicator constraints (native handling and linearization fallback)
- Semi-continuous and semi-integer variable types
- Validation and writer support updates for new constructs

**Test criteria:**
- Round-trip parse/write/solve tests for each feature
- Compatibility checks against reference solvers on representative models

**References:** SCIP and Gurobi feature behavior conventions.

**Depends on:** 11, 27
**Unlocks:** 29

---

<a id="step-29"></a>

## Step 29: Reproducibility + Tuning + Benchmark Hardening ✅

[Back to top](#table-of-contents)

**Goal:** Make performance claims auditable and tuning repeatable.

**Status:** Complete. Added dedicated reproducibility/tuning tooling in
`tests/perf`: deterministic single/multi-thread suite (`run_determinism_suite.py`),
full matrix runner with CSV/Markdown artifacts
(`run_benchmark_matrix.py`), and parameter sweep runner with ranked
CSV/Markdown outputs (`run_param_sweep.py`), while preserving strict default
`work_units` regression gates and versioned mipx/HiGHS baseline workflows.

**Deliverables:**
- Determinism test suite (single-thread and configured multi-thread deterministic mode)
- Full benchmark matrix runner (`solver x time x threads x mode`) and summary artifact generation
- Parameter sweep tooling with structured CSV/Markdown outputs
- Baselines stored/versioned for mipx and external references (HiGHS/highspy)

**Test criteria:**
- Repeated runs at fixed seed produce stable metrics in deterministic mode
- Benchmark scripts are CI-runnable on small subsets
- Strict default regression gate stays in place for merge decisions

**References:** `mip-heuristics` phase-9 matrix/sweep workflows; existing `mipx` perf gates.

**Depends on:** 19, 20, 23, 26, 28
**Unlocks:** reliable solver engineering loop

---

<a id="janitor-post29"></a>

## Janitor Block (post-Step-29 recurring maintenance)

Run this after Step 29 and then periodically after major feature batches (including post-Step-29 expansion steps):

1. **Correctness + E2E catch-up:** extend tests for all new features and verify end-to-end objective correctness on known-optimal/known-best benchmark sets.
2. **`work_units` coverage catch-up:** audit that new code paths are measured by `work_units`, and extend logs/gates where metrics are missing.
3. **Performance opportunity catch-up:** profile new code, prioritize low-risk wins, and only then consider deeper rewrites (SIMD/AVX, GPU kernels, or selective C/assembly) with proof from benchmarks and no regression in correctness/stability.
4. **Documentation catch-up:** update `README.md`, mark completed items in `docs/roadmap.md`, and refresh related docs/changelog notes.

---

<a id="post29-expansion"></a>

## Post-Step-29 Expansion Steps (planned)

**Already covered in current roadmap (removed from future backlog):**
- Core default cut families and automatic cut control are now explicit in Steps 22-24.
- Dual-simplex intra-iteration parallel paths (SIP-style) and GPU backend seam scaffolding were already completed in Step 7.

---

<a id="step-30"></a>

## Step 30: Barrier / Interior-Point LP Backend ✅

[Back to top](#table-of-contents)

**Goal:** Provide a robust barrier/IPM backend for large root LP relaxations and root-policy integration.

**Status:** Complete. Barrier solver, root barrier policy integration, GPU-capable backend selection, and barrier LP perf harness are implemented.

**Deliverables:**
- `BarrierSolver` `LpSolver` backend with predictor-corrector style IPM solve loop
- Root policy controls in MIP (`DualDefault`, `BarrierRoot`, concurrent stub policy)
- GPU-capable barrier backend path with thresholds and runtime toggles
- Dual-simplex handoff/sync solve path for tree continuation after barrier-root solves
- Barrier-vs-reference benchmark harness and baselines in `tests/perf`

**Test criteria:**
- Numerical robustness on curated large sparse LPs (stable primal/dual residuals)
- Competitive root LP wall-clock on selected MIPLIB roots
- No regressions in default dual-simplex-first configuration

**References:** Mehrotra (1992), commercial barrier implementations, cuDSS integration patterns.

**Depends on:** 29
**Unlocks:** 32

---

<a id="step-31"></a>

## Step 31: PDLP + GPU LP Backend ✅

[Back to top](#table-of-contents)

**Goal:** Add a first-order LP backend optimized for very large sparse root relaxations.

**Status:** Complete. In-repo `PdlpSolver` backend, LP CLI mode (`--pdlp`), MIP root PDLP policy integration, GPU-capable execution path, and dedicated PDLP tests are implemented.

**Deliverables:**
- PDHG/PDLP backend with CPU baseline and CUDA acceleration path
- Scaling/preconditioning and restart policy suitable for MIP root use
- Early-stop and certification hooks (primal/dual bounds, infeasibility checks)
- Backend integration with root policy selector

**Test criteria:**
- Correctness against reference LP objectives/bounds on benchmark set
- Throughput wins on very large sparse LP roots where first-order methods are favorable
- Default configuration remains non-regressive

**References:** cuPDLP-C/cuPDLP+, Google PDLP literature.

**Depends on:** 29
**Unlocks:** 32

---

<a id="step-32"></a>

## Step 32: Concurrent Root LP Racing (CPU + GPU)

[Back to top](#table-of-contents)

**Goal:** Run complementary LP backends concurrently at root and use the first high-quality result.

**Deliverables:**
- Root race orchestration across dual simplex (CPU), barrier, and PDLP backends
- Shared stop/cancel protocol and deterministic policy mode
- Winner selection policy balancing latency, bound quality, and downstream warm-start value
- Logging and KPIs for backend race outcomes and contribution

**Test criteria:**
- Improved median time-to-root-bound on benchmark set
- Stable deterministic behavior in deterministic mode
- No strict-gate regression when race mode is disabled (default-safe)

**References:** cuOpt concurrent LP strategy, concurrent LP literature.

**Depends on:** 7, 30, 31
**Unlocks:** faster root processing on large instances

---

<a id="step-33"></a>

## Step 33: Symmetry Handling

[Back to top](#table-of-contents)

**Goal:** Reduce redundant branch-and-bound exploration from symmetric solution spaces.

**Deliverables:**
- Symmetry detection pipeline (lightweight graph/group analysis)
- Orbital fixing and symmetry-breaking cuts/constraints
- Integration with branching and propagation to avoid symmetry-breaking conflicts
- Symmetry diagnostics in logs

**Test criteria:**
- Fewer nodes on symmetry-heavy benchmark families
- No incorrect pruning on validation set
- Non-regressive strict `work_units` gate by default

**References:** Orbital fixing and symmetry handling literature in MIP/CP.

**Depends on:** 29
**Unlocks:** 34

---

<a id="step-34"></a>

## Step 34: Exact LP Refinement Mode

[Back to top](#table-of-contents)

**Goal:** Offer optional high-precision/exact LP refinement for numerically difficult instances.

**Deliverables:**
- Iterative refinement pipeline for LP solutions/certificates
- Optional exact/rational verification path for final certificates
- Trigger rules (numerical warning thresholds, user flag) and reporting
- Compatibility with dual-simplex and barrier-produced solutions

**Test criteria:**
- Improved reliability on numerically fragile instances
- Certified objective/bound consistency within configured tolerances
- Default mode unchanged and non-regressive

**References:** Gleixner & Steffy, Eifler et al. iterative refinement work.

**Depends on:** 30, 33
**Unlocks:** high-reliability solve mode
