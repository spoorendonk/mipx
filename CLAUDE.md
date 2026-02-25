# mip-exact — Branch-and-Cut MIP Solver

## Git Workflow

- **Never commit directly to main.** Always create a feature branch, push, and open a PR.
- **If user says "commit" while on main**: create a feature branch, commit there, push, and open a PR automatically.
- **Linear history only.** Merge PRs with squash or rebase (no merge commits).
- **No force-push to main.**

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

Optional TBB:
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DMIPX_USE_TBB=ON
```

## Test

```bash
ctest --test-dir build -j$(nproc)
```

Netlib/MIPLIB instances are not in git. Download before running benchmarks:
```bash
./tests/data/download_miplib.sh          # full MIPLIB 2017 benchmark set
./tests/data/download_miplib.sh --small  # curated small subset
```
Tests skip automatically when instances are missing.

## Dependencies

- GCC 14, C++23
- CMake 3.25+
- Catch2: fetched via CMake FetchContent
- zlib: for gzipped MPS files
- `apt install libtbb-dev` (optional, enables parallel tree search)

## Architecture

```
include/mipx/      — public headers
src/
  lp/              — sparse matrix, LU factorization, dual simplex
  mip/             — branch-and-bound, node queue, branching
  cuts/            — cut pool, Gomory MIR separation
  presolve/        — reductions and postsolve stack
  heuristics/      — rounding, diving, RINS
  io/              — MPS/LP readers, solution file I/O
  cli/             — CLI tool (mipx-solve)
tests/              — Catch2 tests and MIPLIB benchmarks
  data/             — benchmark instances (downloaded, not in git)
docs/               — documentation and roadmap
```

## Namespace

All code under `mipx::` namespace.

## Key Types

- `mipx::SparseMatrix` — CSR storage with lazy CSC transpose
- `mipx::LpProblem` — LP/MIP problem data (objective, bounds, constraints, integrality)
- `mipx::LpSolver` — Abstract LP solver interface (dual simplex, later barrier/PDLP)
- `mipx::MipSolver` — Branch-and-cut MIP solver
- `mipx::Real` = `double`, `mipx::Int` = `int`, `mipx::Index` = `int`

## Coding Conventions

- C++23 standard, compiled with `-Wall -Wextra -Werror`
- Sanitizers enabled in debug builds (ASan, UBSan)
- Prefer `std::span`, `std::ranges`, `std::format` where appropriate
- No external solver dependencies — everything from scratch
- CSR-primary sparse storage; lazy CSC built on demand
- Forrest-Tomlin LU updates (not product-form)
- Devex pricing + bound flipping ratio test (BFRT) in dual simplex

## Documentation

- `README.md` — project overview, build, usage
- `docs/roadmap.md` — implementation plan with dependency graph and technical notes
- `CLAUDE.md` — this file

## Agent Coordination

The roadmap (docs/roadmap.md) defines steps 1–15 with explicit dependencies and
parallel opportunities marked with ⚡.

**Before suggesting or starting any step:**
1. Check open branches: `git branch -a`
2. Check open PRs: `gh pr list`
3. Check for running agents on this machine (background tasks, worktrees)
4. Never start a step that another agent has an open branch or PR for
5. Prefer the lowest-numbered unblocked, unclaimed step
6. When multiple steps can run in parallel, suggest launching them concurrently

## Fullgate

When the user says **"fullgate"**, run this sequence in order:

1. **Feature branch** — create one if not already on a feature branch
2. **Create PR** — if no PR exists for the current branch
3. **Sync main** — pull latest main and merge into the current feature branch, resolve conflicts
4. **Tests** — check if new/updated tests are needed and add them
5. **Update docs** — update docs/roadmap.md, README.md as needed
6. **Push & update PR**
7. **Review** — thoroughly review the PR (code quality, correctness, style, tests, performance)
8. **Build** — `cmake --build build -j$(nproc)`
9. **Test** — `ctest --test-dir build -j$(nproc)`
10. **Push & update PR** again with any fixes
11. **Finalize** — if nothing more to do: squash-merge the PR, delete feature branch (local + remote), pull main, switch to main
