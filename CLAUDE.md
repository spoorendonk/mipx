# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Communication Style

Be terse. No preamble. No filler.

## Code Navigation

Prefer narrow queries over full-file reads:

1. **LSP** for symbol questions. `goToDefinition`, `hover`, `documentSymbol`, `workspaceSymbol` answer "where is X / what's its signature" in a few tokens. Use before `Read`.
2. **Grep with `head_limit` (small) + `-n`** to locate lines. Start with `head_limit: 20`; raise only if inconclusive.
3. **Read with `offset`/`limit`** to fetch a slice around the hit. Full-file `Read` is fine for files under ~200 lines or when structure matters.

Know the symbol → LSP. Know a string, not its location → Grep. Full-file Read is the last mile.

This is a preference, not a prohibition. Shelling out to `grep`/`rg` is fine when the built-in tool can't do the job — filtering a pipe (`git log | grep`), or a session where the `Grep` tool isn't available. What matters is bounding the output, not which binary produces it.

## Development Workflow

```
plan (non-trivial) → implement → test → /review → push to main
```

Hooks auto-format and type-check on save — don't fix formatting manually. Run tests locally before considering work done — don't skip the suite even on changes that look trivial. The pre-push hook is the final gate.

## Git Workflow

Trunk-based development with linear history on main. Commit directly to main and push when local gates pass.

Feature branches are optional for larger changes:
- Always branch from main. Run `git checkout main && git pull` first.
- Never branch from another feature branch.
- Keep branches short-lived; rebase or squash merge — no merge commits on main.

After a successful push:
- **Close any gh issue the work resolved**: `gh issue close <num> -c "<one-line note>"`. Do this for every issue covered by the push.
- **Delete the feature branch** if one was used: `git branch -d <branch>` locally, plus `git push origin --delete <branch>` if it was pushed. Don't leave stale branches behind.

## Hooks

Git hooks live in `.githooks/` (`core.hooksPath`) and are the only hooks in this repo. They are vendored here — there is no upstream to escalate to, so a wrong or too-strict hook is fixed here.

`.claude/` is untracked and carries no hooks. Don't add Claude Code hooks to it: agent-side hooks duplicated the git hooks' job while being invisible to anyone not running Claude Code, and the branch-creation one could not be satisfied from a git worktree at all. Enforcement belongs in `.githooks/`, where every contributor and CI runs it.

- **Don't work around a failing hook by weakening it.** Fix the cause, or change the hook deliberately and say why in the commit.
- **Never use `git push --no-verify` or `git commit --no-verify`** unless explicitly asked. A failing hook is a signal.

## Issue Tracking

GitHub Issues is the tracker. Use the `gh` CLI.

- **Default to HTTPS** for GitHub remotes (`https://github.com/...`), not SSH.
- **Read an issue** with `gh issue view <num> --json title,body,labels,state,comments`. Plain `gh issue view <num>` is deprecated for programmatic use.
- Don't propose deferring work via a new gh issue unless it is substantial. Small follow-ups should be either fixed inline or left alone — don't open an issue just because you noticed something.

### Writing Issues

Issues get picked up later in fresh sessions, often by a different agent with no access to the author's machine. Write them to be picked up cold:

- **Self-contained.** Body must carry all needed context: problem, motivation, acceptance criteria, repro steps. Don't assume the reader has the current conversation.
- **No local references.** No local file paths, local repo paths, or machine-specific locations (`/home/user/...`, `~/code/foo/bar.py`, "see my other checkout"). Dead links in a fresh session.
- **Prefer stable external links.** GitHub permalinks, paper URLs, RFCs, official docs.
- **Be vague about local code context.** Describe the concept rather than the path; hint that the agent can search under `..`, `../..`, or `~/code/`.

## Agent Self-Review

**Any agent or agent team that produces code must run `/review` on its own changes before that code can merge to main.** No subagent returns unreviewed work; no orchestrator merges unreviewed work. This applies to every agent team, not just the parallel-issue workflow.

## Parallel Issue Workflow

When the user brings multiple gh issues to work on at once:

1. **Propose parallelism first.** Offer it explicitly and wait for confirmation — don't silently start serial work.
2. **Orchestrator role.** Spawn one subagent per issue (Agent tool with `isolation: "worktree"`). Subagents branch from main, not from the orchestrator's working branch, and work in their own git worktree. Pass each subagent its gh issue number and any plan file path.
3. **Subagents self-review** per the Agent Self-Review rule above. Subagents commit locally in their worktree and **do not push** — worktrees share `.git`, so the orchestrator sees their commits via `git log <branch>` with no network round-trip.
4. **No merging without user OK.** Subagents never merge into main; the orchestrator never merges a subagent's branch without explicit user approval.
5. **Final combined review, then push.** The orchestrator merges all approved branches into local main, runs `/review` over the merged result, and only then runs `git push origin main`. No pushes — of main or feature branches — happen before that final review.

## Commit Messages

Conventional Commits. The commit-msg hook enforces format.

- `type: description` or `type(scope): description`
- Types: `feat`, `fix`, `refactor`, `test`, `docs`, `style`, `perf`, `chore`, `build`, `ci`
- Subject ≤72 chars. Focus on **why**, not what.

## CLAUDE.md Discipline

When Claude gets something wrong, fix CLAUDE.md in the same commit. It's a living document — update it whenever better instructions would have prevented the mistake.

## Complexity

When a complexity warning fires, don't extract methods mechanically. Ask: what are the independent responsibilities here? Split along those boundaries. If the function is genuinely complex because the domain is, add a comment explaining why and suppress the warning.

## Plan Adherence

**Follow the agreed plan.** If you think a plan should change, stop and discuss — don't silently diverge. The same goes outside a written plan: if your current approach isn't working, say so out loud — don't quietly switch strategies. Implement everything specified; don't leave TODO placeholders or stub implementations unless explicitly asked.

## Reference Correctness

When implementing from papers, pseudocode, or open-source references:
- Match the reference algorithm exactly. No early exits, iteration limits, size caps, or "optimization" shortcuts that change behavior.
- Only introduce heuristic approximations when explicitly asked.
- Implement edge cases and special handling — don't simplify them away.
- When in doubt, be faithful to the reference and let tests verify correctness.

## Common Mistakes

- **Don't invent APIs — verify they exist.** Check that functions, flags, and methods actually exist before using them.
- **Don't ignore type errors.** If mypy/clang-tidy flags something, fix the root cause — don't suppress.
- **Don't use deprecated patterns.** Check current docs, not training data.
- **Performance matters.** Most of our code is solvers — profile before micro-optimizing, but don't sacrifice perf for "clean code".

## C++

- Target C++23. Use modern features (`std::expected`, concepts, ranges, `constexpr`).
- Style: Google-based, enforced by `.clang-format` and `.clang-tidy`.
- Use `#pragma once` for include guards.
- Minimize includes in headers. Forward-declare where possible.

## CMake

- `set(CMAKE_EXPORT_COMPILE_COMMANDS ON)` for clang-tidy.
- Use FetchContent for dependencies.
- One `CMakeLists.txt` per directory with source files.

## Testing (GoogleTest)

- Test files: `<module>_test.cpp` in `tests/`.
- Name tests descriptively: `TEST_F(SolverTest, ReturnsOptimalForFeasibleInput)`.
- Terse output: `GTEST_BRIEF=1` prints only failures, `ctest --progress` collapses the running list, `CMAKE_INSTALL_MESSAGE=LAZY` suppresses install chatter. Don't remove these.

## LSP

Install `clangd-lsp@claude-plugins-official` plus `clangd` itself (`apt install clangd` or from LLVM). Devkit ships `.clangd` pointing at `build/compile_commands.json` (produced by `CMAKE_EXPORT_COMPILE_COMMANDS ON`). Prefer `LSP` tool queries (`goToDefinition`, `hover`, `documentSymbol`) over `Read` for symbol questions.

## Python

- Style: enforced by `ruff` (format + lint) and `mypy --strict`, configured in `pyproject.toml`.
- All functions must have full type annotations (mypy strict mode).
- Use built-in generics (`list[int]`, `dict[str, Any]`) and `|` union syntax.

## Testing (pytest)

- Test files: `test_<module>.py` in `tests/`.
- Name tests descriptively: `test_solver_returns_optimal_for_feasible_input`.
- Use `conftest.py` for shared fixtures, `pytest.mark.parametrize` for data-driven tests.
- Terse output: `pyproject.toml` bakes in quiet defaults for pytest (`addopts`), mypy (`pretty = false`), and ruff (`output-format = "concise"`). Override per-invocation (`pytest -v`, `mypy --pretty`) when debugging; don't edit the defaults.

## Dependencies

- Pin with `>=` lower bounds in `pyproject.toml`. Use `uv` or `pip`.

## LSP

Install `pyright-lsp@claude-plugins-official`. Pyright reads `[tool.mypy]` and project layout from `pyproject.toml` — no extra config needed. Prefer `LSP` tool queries (`goToDefinition`, `hover`, `documentSymbol`, `workspaceSymbol`) over `Read` for symbol questions.

## nanobind Bindings

- Place bindings in `bindings/`, separate from core C++ logic. One file per module: `bind_<module>.cpp`.
- C++ `camelCase` methods → Python `snake_case` via nanobind. Use `nb::arg("name")` for Python-friendly parameter names.

## Ownership and Lifetime

- Default: nanobind manages ownership. Use `nb::rv_policy::reference` only when C++ retains ownership and guarantees the object outlives Python references.
- Never return raw pointers without explicit lifetime annotation.
- Prefer returning by value or `std::shared_ptr`. Document ownership on each binding that transfers or shares it.

## Type Conversions

- Use automatic conversions for standard types (`std::string` ↔ `str`, `std::vector` ↔ `list`).
- Use `nb::ndarray` for NumPy interop — specify dtype and shape constraints.

## Testing

- Test bindings from Python using pytest, not from C++. The binding is an implementation detail.
- Ensure round-trip tests: create in Python → pass to C++ → get result back.

# Project: mipx

A from-scratch branch-and-cut MIP solver in C++23 with Python bindings via nanobind.

## Build & Test

```clean
rm -rf build
```

```build
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)
```

```test
ctest --test-dir build --output-on-failure -j$(nproc) && { pytest --tb=short -q || test $? -eq 5; }
```

Keep `CMAKE_BUILD_TYPE` in the build command. `CMakeLists.txt` also defaults it
to Release when unset, so both paths agree; the explicit flag documents the
requirement at the call site. It matters because several benchmark tests carry
wall-clock limits (`MIPLIB: gen ...` allows 20s, against ~1.2s optimized), so
an unoptimized build fails them on timing alone — which reads like a solver
regression rather than a build problem.

If C++ tests suddenly get slower, check `CMAKE_BUILD_TYPE` in
`build/CMakeCache.txt` before suspecting the code: `pip install -e .`
reconfigures this same `build/` directory.

### Running a single C++ test

```bash
ctest --test-dir build -R "test_name_regex" --output-on-failure
```

All C++ tests compile into a single `mipx-tests` binary (Catch2). You can also run it directly:

```bash
./build/tests/mipx-tests "test name or tag"
```

### CMake options

| Option | Default | Purpose |
|---|---|---|
| `MIPX_USE_TBB` | auto-detect | Parallel tree search (`apt install libtbb-dev`) |
| `MIPX_USE_CUDA` | ON | GPU acceleration (barrier, PDLP) |
| `MIPX_BUILD_PYTHON` | auto-detect | Build nanobind Python extension (ON when Python 3.12 dev is found) |
| `MIPX_BUILD_CLI` | ON | Build `mipx-solve` CLI |
| `MIPX_SIMD_ISA` | native | SIMD codegen: `off`, `avx2`, `native` |
| `MIPX_STRICT_WARNINGS` | ON | `-Werror` |

### Test data

Netlib/MIPLIB instances are not in git. Download before running benchmarks:

```bash
./tests/data/download_miplib.sh          # full MIPLIB 2017 set
./tests/data/download_miplib.sh --small  # curated small subset
```

Tests skip automatically when instances are missing.

## Architecture

All code under `mipx::` namespace. Key type aliases: `Real` = `double`, `Int` = `int`, `Index` = `int`.

### Core modules (`src/`)

- **`lp/`** — LP solvers: dual simplex (Devex pricing + BFRT), Forrest-Tomlin LU factorization, interior-point barrier (CPU + GPU/cuDSS), PDLP (CPU + CUDA kernels), exact iterative refinement
- **`mip/`** — Branch-and-bound: node queue, domain propagation, branching strategies
- **`cuts/`** — Cut pool, Gomory MIR separation, cut manager, custom separators
- **`presolve/`** — Reductions and postsolve stack
- **`heuristics/`** — Rounding, diving, RINS, RENS, feasibility pump, local branching, symmetry-aware, budget management
- **`io/`** — MPS/LP readers (mmap + bulk decompress), MPS writer, solution file reader
- **`cli/`** — `mipx-solve` CLI entry point

### Key types

- `SparseMatrix` — CSR-primary storage with lazy CSC transpose built on demand
- `LpProblem` — LP/MIP problem data (objective, bounds, constraints, integrality)
- `LpSolver` — Abstract LP solver interface (dual simplex, barrier, PDLP backends)
- `MipSolver` — Branch-and-cut MIP solver orchestrating all components

### Python bindings (`python/`)

- `python/src/bindings.cpp` — nanobind module exposing core API
- `python/tests/test_api.py` — pytest tests for the Python interface
- Build with `-DMIPX_BUILD_PYTHON=ON`

### Performance benchmarks (`tests/perf/`)

Regression gates and benchmark scripts for Netlib LP, MIPLIB MIP, Mittelman, and dual simplex. Key scripts:

- `run_full_gate.sh` — full performance gate
- `run_dual_perf_gate.sh` — dual simplex regression gate
- `run_mip_regression_gate.sh` — MIP regression gate
- `check_regression.py` — shared regression checking logic

## Project-specific conventions

- CSR-primary sparse storage; lazy CSC built on demand — never store both eagerly.
- Forrest-Tomlin LU updates, not product-form inverse.
- Dual simplex uses Devex pricing + bound flipping ratio test (BFRT).
- Debug builds enable ASan + UBSan automatically.
- No external solver dependencies — everything from scratch.
- CUDA code lives alongside CPU code in the same directories (`.cu` files).
