# Dual Simplex No-Presolve Competitiveness Plan

## Summary
The only focus is dual simplex without presolve.

The goal is to beat HiGHS dual simplex on wall-clock while staying
correctness-safe on the primary Netlib target set.

The dual perf gate exists to prevent regressions, not to define success.
Success is defined by head-to-head no-presolve performance against HiGHS on the
primary Netlib scorecard.

## Current Status
- Infrastructure
  - Dual perf gate is implemented and wired into CI.
  - Stored dual baselines and the LPopt-style curated LP corpus are in place.
- Solver changes already landed on this branch
  - Preserve Devex weights across pure LU reinversion.
  - Retune automatic LU update budget by basis size.
  - Add a sparse dual-simplex work-vector utility and use it for perturbation
    pricing scratch state.
  - Reuse `BTRAN` row support in dual row pricing, track pivot-row alpha
    support explicitly, and clear that support with sparse-state fallback.
  - Fix the row-pricing support path so support stays deduplicated and plain
    `btran(rhs)` does not pay support-extraction overhead.
- Current large-anchor scorecard at `--timeout 10`
  - `greenbea`: `mipx 1.97s`, `12563` iterations vs `HiGHS 0.58s`, `5109`
    iterations
  - `pilot`: `mipx` still `time_limit 10.00s` vs `HiGHS 0.93s`
  - `ship12l`: `mipx 0.05s`, `1287` iterations vs `HiGHS 0.02s`, `1215`
    iterations
  - `sierra`: `mipx 0.03s`, `762` iterations vs `HiGHS 0.01s`, `602`
    iterations
- Current read
  - `greenbea` moved materially in the right direction.
  - `pilot` remains the dominant blocker by a wide margin.
  - The next meaningful gains are more likely to come from LU internals than
    from more dual-loop plumbing.

## Objectives
- Beat HiGHS `simplex --presolve off --parallel off` with
  `mipx --dual --no-presolve`.
- Keep mipx `work_units` non-regressive while improving wall-clock.
- Require correct status/objective on every target instance before accepting any
  performance claim.
- Keep the work strictly inside the dual-simplex path. Barrier, PDLP, presolve,
  and general MIP work are out of scope for this plan.

## Benchmark Priority
- Primary scorecard
  - Netlib core: `afiro`, `adlittle`, `blend`, `sc50a`
  - Large anchors: `greenbea`, `pilot`, `ship12l`, `sierra`
- Fast development gate
  - Stored dual perf gate baselines and curated LP corpus used for short-cycle
    `work_units` regression protection

The curated gate is a development safety net. The primary scoreboard is the
Netlib no-presolve target set, especially the large anchors.

## Algorithm Policy
- HiGHS is the primary implementation and performance reference.
- Broader modern methods are allowed only if they are clearly in the same class
  as current HiGHS-style dual simplex and are not legacy shortcuts.
- Acceptable mechanism families:
  - Devex / DSE-style pricing control
  - Harris / BFRT ratio logic
  - Hypersparse linear algebra
  - Robust reinvert / refactor logic
  - Stable crash / basis handling
  - Pricing / scan reduction
- Do not introduce fake-bound tricks, numerically weaker shortcuts, or other
  old-school legacy tactics unless explicitly discussed first.
- Clp is comparative background only, not a source of default implementation
  direction.

## Optimization Order
Correctness on `greenbea`, `pilot`, `ship12l`, and `sierra` remains a mandatory
gate, but the execution order for architecture work is:

1. Build a proper sparse work-vector layer for dual simplex hot paths.
   - Use a dense array plus touched-index list / sparse-state metadata so hot
     vectors can be cleared and reused without full dense resets.
   - Start by migrating dual-simplex scratch vectors that feed pricing, row
     assembly, and basis solves.
   - Status: started
     - `pricing_reduced_cost` uses sparse touched clearing.
     - A generic `SparseWorkVector` utility exists with unit coverage.
2. Replace the current pivot-row assembly path with a HiGHS-style hyper-sparse
   row-price kernel that can switch to dense when fill-in justifies it.
   - Do not bolt sparse tricks onto the current dense-oriented path and call it
     done.
   - Status: partially done
     - `BTRAN` row support is now reused when building the pivot row.
     - Candidate collection already uses pivot-row alpha support with dense
       fallback.
   - Next step still required
     - Stop rebuilding the full dense alpha row on cases where a real sparse
       row-price kernel can stay sparse deeper into CHUZC / BFRT.
3. Move `CHUZR` toward maintained sparse infeasibility state instead of full
   row rescans each iteration.
   - The goal is a dedicated leaving-row work object, not more ad hoc scan
     tuning.
   - Status: not started
4. Make basis solves and reinversion control more HFactor-like.
   - Stage-specific sparse / hyper-sparse switching for lower and upper
     `FTRAN` / `BTRAN`
   - Density- and history-guided reinversion decisions, not one global
     threshold
   - Status: partially done
     - Reinversion budget tuning landed.
     - The next blocker is no longer generic reinversion policy; it is LU
       runtime inside `factorize()` and the FT update path.
5. Consider a DSE-first path with Devex fallback if iteration-count gaps remain
   after the architecture work above.
   - This is the largest algorithmic step in scope and should follow the
     vector / pricing / NLA refactor rather than precede it.
   - Status: not started

## Next Steps
1. Attack `SparseLU::factorize()` directly.
   - This is still the dominant `pilot` hotspot.
   - Focus on pivot search and active-submatrix maintenance before trying more
     dual-loop tweaks.
2. Reduce `applyFTTranspose()` / `BTRAN` cost with proven stage-specific logic.
   - Keep this HFactor-style and measurement-driven.
   - Do not keep speculative sparse-support paths unless they improve `pilot`
     and preserve the current `greenbea` win.
3. Add explicit regression coverage for the row-pricing support path.
   - Include a unit/regression case that exercises support deduplication under
     cancellation / re-touch so the recent bug cannot return.
4. Move `CHUZR` to maintained sparse infeasibility state once LU-side wins are
   exhausted.
   - That remains the clearest next architectural gap on the dual-simplex side
     outside LU.

No other optimization track belongs in this plan unless explicitly discussed.

## Regression Gate Role
- Keep deterministic dual perf gates on the existing curated LP corpora.
- Use `work_units` as the primary short-cycle regression metric.
- Record wall-clock as a secondary metric, but do not let gate convenience
  redefine the main objective.
- Use the gate to protect progress while optimizing the primary Netlib scorecard.

## Test Plan
- Correctness
  - Existing dual correctness investigation (`.solu` cross-check + HiGHS
    status/objective comparison)
  - Explicit correctness checks on `greenbea`, `pilot`, `ship12l`, and `sierra`
    before claiming speedups
- Performance
  - Canonical no-presolve dual-vs-HiGHS benchmark on the primary Netlib
    target set
  - Fast dual perf gate on the curated LP corpus for `work_units`
    non-regression
- Profiling
  - Profile the worst losing target instances before accepting any optimization
    branch

## Acceptance Criteria
- No correctness regressions on the primary Netlib target set.
- No dual perf gate `work_units` regression.
- Wall-clock median vs HiGHS is below `1.0x` on the primary target set.
- Wall-clock geomean vs HiGHS is below `1.0x` on the primary target set.
- No large-anchor regression is accepted without explicit justification.

## Assumptions
- Single-threaded, deterministic, no-presolve comparisons are the only
  supported comparison mode for this plan.
- Wall-clock vs HiGHS is the primary external win metric.
- mipx `work_units` are the primary internal guardrail metric.
- HiGHS is the first reference, but not the only allowed source of modern
  mechanisms if broader literature stays in the same algorithmic family.
