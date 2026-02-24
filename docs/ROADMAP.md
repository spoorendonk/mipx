# mip-exact Roadmap

Stepwise implementation plan for an end-to-end MIP solver (branch-and-cut) in C++23.

Each step builds on the previous, produces something testable, and is scoped for 1–2 sessions.

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
```

**Parallel opportunities:**
- Steps 4 + 6: test framework and LU factorization are independent after Step 3
- Steps 9 + 10: domain propagation and node queue are independent after Step 8
- Steps 12 + 13 + 14: cuts, presolve, and heuristics are independent after Step 11

---

## Step 1: Project Skeleton

**Goal:** Buildable project with CI-ready test infrastructure.

**Deliverables:**
- CMake build system (C++23, `-Wall -Wextra -Werror`, sanitizers in debug)
- Directory structure: `src/`, `include/mipx/`, `test/`, `benchmark/`, `docs/`
- Catch2 test framework (FetchContent)
- Core type aliases: `Int`, `Real` (double), `Index`, status/sense enums
- Optional TBB dependency (`-DMIPX_USE_TBB=ON`, off by default)

**Test criteria:** `cmake --build . && ctest` passes with a trivial test.

**References:** HiGHS CMakeLists.txt, [baldes](https://github.com/lseman/baldes) build setup.

**Depends on:** nothing
**Unlocks:** 2

---

## Step 2: Sparse Matrix

**Goal:** CSR-primary sparse matrix with lazy CSC view.

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

## Step 3: LP Problem + File I/O

**Goal:** Represent LP/MIP problems and read/write standard formats.

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

## Step 4: MIPLIB Test Framework ⚡ parallel with 5→6

**Goal:** Automated benchmark infrastructure against standard LP/MIP test sets.

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

## Step 5: LP Solver Interface ⚡ parallel with 4

**Goal:** Abstract interface that dual simplex (and later barrier/PDLP) will implement.

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

## Step 6: LU Factorization

**Goal:** Sparse LU for basis matrix operations in simplex.

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

## Step 7: Dual Simplex ✅

**Goal:** Working LP solver. First real solves.

**Status:** Complete. Solves all 4 Netlib test instances (afiro, sc50a, blend, adlittle) to optimality.

**Deliverables:**
- Dual simplex Phase 1 (cost perturbation for initial dual feasibility) + Phase 2 (bound flipping + primal simplex fallback)
- Harris dual ratio test with anti-cycling (fixed variables excluded from entering selection)
- Equilibration scaling (row/col max-norm)
- Augmented matrix formulation `[A | -I]` with slack variables
- Primal simplex pivot fallback for dual-infeasible cleanup after perturbation removal
- Unbounded detection via primal ratio test
- Solver output: iteration count, objective value, primal infeasibility
- CLI tool `mipx-solve` with MPS file input
- HiGHS comparison benchmark script (`tests/benchmark_vs_highs.py`)

**Deferred to optimization pass:**
- Devex pricing (approximate steepest-edge) — currently using Dantzig/Harris
- Bound Flipping Ratio Test (BFRT) — currently single-flip per pivot
- Hyper-sparsity exploitation in FTRAN/BTRAN
- PAMI/SIP parallelism

**Test criteria:** Solve all Netlib instances to optimality. Objective matches `.solu` values within tolerance. Competitive iteration counts vs. reference.

**References:** HiGHS `HDual`, Koberstein (2005) dual simplex thesis (best description of Harris + BFRT + cost shifting integration), Maros (2003) textbook, Hall & McKinnon (2005) hyper-sparsity, Huangfu & Hall (2018) parallel dual simplex.

**Depends on:** 4, 6
**Unlocks:** 8

---

## Step 8: Incremental LP Updates

**Goal:** Efficient LP modifications for branch-and-cut use.

**Deliverables:**
- Add rows (cuts) without full refactorization — extend basis with slacks
- Remove rows (inactive cuts) — shrink basis, maintain feasibility
- Bound changes — update basis status, detect infeasibility cheaply
- Warm-start: save/restore basis for tree node LP resolves
- Objective changes for branching (variable fixing)

**Test criteria:** Add rows, re-solve, verify optimal. Warm-started solve uses fewer iterations than cold start. Bound change + re-solve matches fresh solve.

**References:** HiGHS `HighsLpSolverObject`, SCIP LP interface.

**Depends on:** 7
**Unlocks:** 9, 10 (parallel)

---

## Step 9: Domain Propagation ⚡ parallel with 10

**Goal:** Infer tighter variable bounds from constraints and integrality.

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

## Step 10: Node Queue + Branching ⚡ parallel with 9

**Goal:** Tree search infrastructure.

**Deliverables:**
- `BnbNode`: LP bound, parent pointer, branching decision, depth, basis snapshot
- `NodeQueue` with policies: best-first, depth-first, hybrid (diving + best-first)
- Branching rules: most fractional, strong branching (solve LP for each candidate), reliability branching
- Node selection heuristics: plunge when improving, switch to best-first periodically

**Test criteria:** Node queue ordering is correct for each policy. Strong branching picks intuitively correct variable on small instances. Memory usage reasonable for 10k+ nodes.

**References:** SCIP `branch_relpscost.c`, Achterberg et al. (2005) branching paper, HiGHS `HighsMipSolver`.

**Depends on:** 8
**Unlocks:** 11

---

## Step 11: MIP Solver Shell

**Goal:** End-to-end MIP solving. The milestone.

**Deliverables:**
- `MipSolver` class: read problem → preprocess → solve root LP → branch-and-bound loop → report solution
- Incumbent tracking: best feasible solution found so far
- Pruning: by bound, by infeasibility, by integrality (all-integer solution)
- Gap tracking: `(incumbent - best_bound) / incumbent`
- Limits: node limit, time limit, gap tolerance
- **Solver output:** HiGHS-style MIP progress table — nodes explored, open nodes, LP iters, incumbent, best bound, gap%, time. Solution file output (`.sol` format).

**Test criteria:** Solve MIPLIB easy instances (e.g., `air04`, `bell5`, `blend2`, `dcmulti`, `egout`, `gen`, `lseu`, `misc03`, `mod010`, `p0033`, `p0201`, `stein27`). Optimal value matches `.solu`. Completes within reasonable time and node counts.

**References:** HiGHS `HighsMipSolver`, SCIP `solve.c`, Achterberg (2007).

**Depends on:** 9, 10
**Unlocks:** 12, 13, 14 (parallel)

---

## Step 12: Cutting Planes ⚡ parallel with 13, 14

**Goal:** Strengthen LP relaxations with valid inequalities.

**Deliverables:**
- `CutPool`: store cuts, manage ages, purge inactive cuts
- Gomory mixed-integer rounding (MIR) cuts from optimal simplex tableau
- Separation rounds at root and tree nodes (configurable frequency)
- Cut selection: parallelism filtering, efficacy ranking
- Integration with incremental LP (Step 8): add cuts → re-solve → iterate

**Test criteria:** Root gap closed on standard instances (e.g., `p0033` root gap → near-zero with Gomory cuts). Fewer nodes needed vs. no-cuts solve. Cut pool doesn't grow unboundedly.

**References:** HiGHS `HighsCutPool`, SCIP `sepa_gomory.c`, Cornuéjols (2007) survey.

**Depends on:** 11
**Unlocks:** 15

---

## Step 13: Presolve ⚡ parallel with 12, 14

**Goal:** Reduce problem size before solving.

**Deliverables:**
- `Presolver` with reductions: fixed variables, singleton rows/columns, forcing/dominated constraints, coefficient tightening, probing
- `PostsolveStack`: record reductions, undo them to recover full solution
- Iterative presolve loop until no more reductions found
- Statistics: vars/cons removed, bounds tightened

**Test criteria:** Presolve + solve + postsolve matches direct solve. Measurable reduction on MIPLIB instances. No incorrect eliminations (validated by postsolve).

**References:** HiGHS `Presolve`, Achterberg et al. (2020) presolve paper.

**Depends on:** 11
**Unlocks:** 15

---

## Step 14: Primal Heuristics ⚡ parallel with 12, 13

**Goal:** Find feasible solutions faster.

**Deliverables:**
- Rounding heuristic: round LP solution, check feasibility
- Diving heuristics: fractional diving, coefficient diving, guided diving
- RINS (Relaxation Induced Neighborhood Search): fix variables agreed upon by LP and incumbent, solve sub-MIP
- Heuristic scheduler: run at root, periodically in tree, after incumbent improvement

**Test criteria:** Heuristics find incumbent earlier (fewer nodes to first feasible). Solution quality comparable to optimal. Time overhead acceptable.

**References:** SCIP `heur_rounding.c`, `heur_diving.c`, `heur_rins.c`; Berthold (2006).

**Depends on:** 11
**Unlocks:** 15

---

## Step 15: Parallel Tree Search

**Goal:** Exploit multicore via TBB.

**Deliverables:**
- Thread-safe node queue (TBB concurrent queue or lock-free)
- Parallel node processing: each thread has own LP solver instance
- Shared incumbent with atomic updates
- Deterministic mode: reproducible results regardless of thread count
- Graceful fallback: single-threaded when TBB unavailable

**Test criteria:** Correct results match serial solver. Speedup on 4+ cores. Deterministic mode produces identical solutions across runs.

**References:** SCIP concurrent solver, HiGHS parallel MIP, cuOpt parallel B&B.

**Depends on:** 12, 13, 14

---

## Cross-cutting: Solver Output

Woven into steps as they become relevant:

| Step | Output added |
|------|-------------|
| 7 (Dual simplex) | LP iteration log: iter, objective, primal/dual infeasibility, time |
| 11 (MIP shell) | MIP tree log: nodes, open, LP iters, incumbent, best bound, gap%, time |
| 11 (MIP shell) | Solution file (`.sol` format) |
| 12+ | Cut statistics in log (cuts added, root gap closed) |

Format follows HiGHS style — compact, fixed-width columns, periodic summary lines.

---

## Technical Notes: State of the Art

### Simplex factorization (settled)

Sparse LU factorization for simplex has not changed fundamentally in decades:
- **Markowitz ordering** with threshold pivoting for initial factorization — Suhl & Suhl (1990)
- **Forrest-Tomlin updates** for basis changes — Forrest & Tomlin (1972). Maintains sparsity far better than product-form updates across hundreds of pivots. Every major solver (HiGHS, CPLEX, Gurobi, Xpress, COPT) uses this.
- Huangfu & Hall (2015) explored product-form variants approaching FT performance on some problems, but FT remains the default.

No GPU acceleration applies here — sparse triangular solves are irregular and memory-bound.

### Dual simplex techniques (settled, well-documented)

The three techniques that separate competitive solvers from textbook implementations:
1. **Devex pricing** (approximate steepest-edge) — halves iteration counts vs. Dantzig
2. **Bound Flipping Ratio Test** — Koberstein (2005) gives the definitive integration of Harris + BFRT + cost shifting
3. **Hyper-sparsity** — Hall & McKinnon (2005), order-of-magnitude wall-clock wins on large sparse LPs

Parallelism within simplex (Huangfu & Hall 2018: PAMI/SIP) gives real speedups but is complex — defer.

### Where the action is: barrier and first-order methods on GPU

Recent advances are all on the barrier/PDLP side, not simplex:
- **cuPDLP-C** — GPU first-order LP solver from the COPT team (Huangfu et al.), open-sourced Dec 2023, integrated into COPT 7.1 Feb 2024. Avoids factorization entirely — only needs SpMV. [github.com/COPT-Public/cuPDLP-C](https://github.com/COPT-Public/cuPDLP-C)
- **cuPDLP+** — enhanced version (2025). [arxiv.org/abs/2507.14051](https://arxiv.org/abs/2507.14051)
- **cuDSS** — NVIDIA's GPU sparse direct solver, enabling GPU-accelerated barrier methods in cuOpt. Reports 8x average speedup over open-source CPU solvers.
- **cuOpt concurrent mode** — runs PDLP + barrier on GPU + dual simplex on CPU simultaneously, ranked #1 among open-source solvers (Oct 2025).
- **Iterative refinement** — Eifler, Nicolas-Thouvenin & Gleixner (2024) combined precision boosting with LP iterative refinement for exact rational LP without numerical tolerances.

These validate our roadmap: get dual simplex right on CPU first, add barrier/PDLP+GPU as future work.

---

## Future Work (after end-to-end MIP is working)

- **Barrier / Interior Point Method** — Mehrotra predictor-corrector, sparse Cholesky. GPU-accelerated via cuDSS.
- **PDLP + GPU** — First-order method (PDHG) with CUDA acceleration. Only needs SpMV — ideal for GPU. See cuPDLP-C/cuPDLP+ from COPT team.
- **Concurrent LP** — Run dual simplex (CPU) + barrier (GPU) + PDLP (GPU) simultaneously, return first solution (cuOpt pattern).
- **Simplex parallelism** — PAMI (parallel across multiple iterations) and SIP (single iteration parallelism) from Huangfu & Hall (2018).
- **Column generation** — LP interface already supports `addColumns()`/`removeColumns()`; build pricing loop, branching integration (branch-and-price).
- **Advanced cuts** — Lift-and-project, flow covers, clique cuts.
- **Symmetry handling** — Orbital fixing, isomorphism pruning.
- **Conflict analysis** — CDCL-style learning from infeasible nodes.
- **Exact LP** — Iterative refinement for rational arithmetic solutions (Gleixner & Steffy 2016, Eifler et al. 2024).
