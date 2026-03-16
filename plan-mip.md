# MIP Performance Plan

## Mission
- Primary goal: beat HiGHS on MIP performance.
- This is a research project. Backward compatibility is not a goal.
- We only prioritize work that improves MIP benchmark outcomes.
- Determinism is a measurement and debugging tool, not the end goal.

## Primary Benchmark Target
- Main external benchmark: MIPLIB Mittelman set against HiGHS.
- Main scorecard:
  - solve rate
  - wall-clock time
  - primal integral / time-to-first-feasible
- Secondary diagnostic metrics:
  - nodes
  - LP iterations
  - `work_units`

## Scope
- Focus only on MIP improvements.
- LP backend work is out of scope unless it directly improves MIP results.
- Cleanup-only work is out of scope unless it improves measurement quality or unlocks benchmark gains.

## Current Foundation
These are already in place and should be treated as baseline infrastructure, not future roadmap items:
- deterministic and opportunistic execution modes
- benchmark and regression tooling
- root and tree heuristics
- reliability branching and pseudocosts
- root and in-tree cut management
- conflict learning and no-good reuse
- search profiles and restart controls
- in-tree presolve hooks
- symmetry handling
- exact refinement hooks
- parallel tree search

## Current Findings
- Initial MIP grinding already found one concrete tree-search lever: serial in-tree presolve.
- Spot benchmarks on local MIPLIB instances:
  - `gt2`: default search timed out at `30s`; enabling tree presolve solved in about `1s`
  - `air04`: tree presolve was roughly neutral at `30s`
  - `p0201`: tree presolve regressed from about `0.16s` to `0.22s`
- Tree cuts alone did not reproduce the `gt2` win.
- Follow-up shape check: the current regression bucket looks like small pure-binary MIPs, not binary models in general.
- Current implication: tree presolve is worth keeping in play, but it needs selective auto-gating for small pure-binary models and eventually a multithread-capable implementation.
- First bounded 8-instance Mittelman-vs-HiGHS summary now exists on
  `air04, air05, blend2, flugpl, gt2, p0201, supportcase16, stein45inf`
  at `15s`, `8` threads:
  - solve rate: `5/8` for both mipx and HiGHS
  - common-solved geomean wall-clock ratio: `0.539x` mipx/HiGHS
  - common-solved median wall-clock ratio: `0.580x` mipx/HiGHS
  - notable wins: `gt2`, `p0201`, `blend2`, `stein45inf`, `flugpl`
  - notable loss: `supportcase16`
- The current threaded-search win came from porting serial search control
  into deterministic parallel search: dynamic queue-policy switching,
  strong-branch budget control, and exact-safe restarts.
- Guardrail learned from this grind: restart logic must stay exact-safe.
  Throwing away queued nodes can create fake "optimal" claims and is out.
- Expanded 11-instance bounded slice
  `air04, air05, blend2, dcmulti, flugpl, gen, glass4, gt2, p0201, supportcase16, stein45inf`
  at `15s`, `8` threads is roughly parity on solve count but exposes a new real
  weakness:
  - solve rate: `7/11` for both mipx and HiGHS
  - common-solved geomean wall-clock ratio: about `0.924x` mipx/HiGHS
  - median wall-clock ratio: about `0.599x` mipx/HiGHS
  - major remaining loss bucket: `gen`
- `gen` loss-bucket update from 2026-03-11:
  - root heuristic gate is blocking the LP-based root portfolio entirely:
    `root_int_inf=22` vs gate `12`, `root_int_vars=113` vs gate `96`
  - widening that gate to `64/192` or `128/256` makes `gen` materially worse;
    the existing root portfolio is not the right primitive for this bucket
  - conclusion: `gen` needs a new cheap large-root heuristic or stronger
    propagation, not a broader version of the current root LP heuristic pack
- `gen` also exposed an important correctness boundary: mixed-model in-tree
  presolve was unsound and is now fenced off. Any future propagation/presolve
  work on mixed MIPs must ship with explicit `.solu` parity coverage.

## Algorithmic Gaps Vs HiGHS
This is the concrete list of solver machinery that HiGHS already has in-tree
and that we either do not have at all, or only have in a much weaker form.

### 1. Real implication and variable-bound infrastructure
- HiGHS has a dedicated implication engine with cached binary implications,
  explicit VUB/VLB storage, probing, implied-bound cut generation, and
  transformed-LP support for those bounds.
- We have lightweight domain propagation, but we do not yet have a persistent
  implication/VUB/VLB graph or implied-bound separation that feeds search,
  cuts, branching, and heuristics together.

### 2. Clique table, clique merging, and clique-driven propagation
- HiGHS has a large dedicated clique table:
  clique extraction from rows and cuts, clique merging/subsumption, objective
  cliques, substitutions, clique partitioning, clique separation, and clique
  propagation/fixing.
- We only have a much lighter clique separator. We do not have a persistent
  clique database that drives propagation, branching, heuristics, and
  substitutions.

### 3. Objective propagation
- HiGHS propagates the incumbent cutoff through the objective function itself,
  including partition/clique-aware objective reasoning.
- We only use the incumbent as a pruning cutoff and for reduced-cost tests.
  We do not have a true objective-propagation engine.

### 4. Conflict analysis integrated with propagation state
- HiGHS conflict analysis is tied into the full domain stack, cut propagation,
  watched literals, reconvergence cuts, and conflict-score feedback into
  pseudocosts.
- Our conflict learning is still much shallower and mostly serial. We do not
  yet have watched-literal conflict propagation, reconvergence conflict cuts,
  or deep propagation explanations across cuts/objective/cliques.

### 5. Stronger root and node propagation loop
- HiGHS repeatedly alternates:
  implied-bound separation, clique separation, propagation, LP resolve,
  reduced-cost fixing, and cutpool propagation.
- Our loop is much simpler. We do not yet have a unified "separate ->
  propagate -> resolve -> propagate again" engine with cutpool and objective
  propagation as first-class participants.

### 6. Richer separator stack
- HiGHS runs several dedicated separator families beyond tableau/Gomory-style
  work, including path aggregation and mod-k separation, plus stronger
  single-row cut generation with CMIR / lifted cover style machinery.
- We have root cut management and some families, but our separator stack is
  materially thinner and less integrated with implication/clique data.

### 7. Inference-aware pseudocost branching
- HiGHS pseudocosts track more than objective gain:
  inference counts, cutoff counts, conflict scores, and degeneracy scaling.
- Our branching is still mainly objective/pseudocost driven with some conflict
  nudging. We do not yet score branch candidates using inference production,
  cutoff production, or degeneracy-aware weighting.

### 8. Deeper search control primitives
- HiGHS search has plunge/backtrack logic, child-selection rules that use
  inference/cost hybrids, a richer open-node data structure, and explicit
  suboptimal-node handling.
- Our recent threaded search-control work helped, but our tree policy is still
  much simpler than that.

### 9. Broader primal-heuristic portfolio
- HiGHS has more root/tree primal machinery than we currently expose:
  feasibility jump, central rounding, randomized rounding, shifting,
  zi-rounding, root reduced-cost heuristic, and adaptive sub-MIP repair/fixing
  rates informed by observed success/infeasibility.
- We already have a good base portfolio, but we are missing several cheap
  incumbent-finders that matter for primal integral.

### 10. Reduced-cost fixing as a tree tool
- HiGHS uses root and local reduced-cost fixing tied into propagation and
  conflict analysis.
- We do some reduced-cost tightening inside tree presolve, but we do not have a
  standalone reduced-cost fixing engine with lurking-bound carryover and domain
  integration.

## Current Focus

### 1. Benchmark-driven gap analysis vs HiGHS
- Use the Mittelman MIP benchmark set as the main prioritization loop.
- For every benchmark run, bucket losses by dominant failure mode:
  - weak root bound
  - slow first feasible solution
  - poor primal integral
  - node explosion
  - heavy tree overhead
  - poor parallel scaling
- Do not start major solver work without a clear benchmark-driven hypothesis.

### 2. Stronger propagation and inference
- Improve pruning on binary-heavy and structured MILPs.
- Priorities:
  - persistent implication graph with cached binary implications
  - VUB/VLB storage and cleanup
  - probing-derived reductions
  - persistent clique table, clique merging, and clique propagation
  - objective propagation from the incumbent cutoff
  - better infeasibility explanations from propagation, cuts, and objective
- Success criterion: fewer nodes and faster pruning on Mittelman instances without invalid deductions.

### 3. Branching quality v2
- Build on the current reliability branching and pseudocost base.
- Priorities:
  - inference-aware candidate scoring
  - conflict-aware branch scoring
  - cutoff/history signals
  - degeneracy-aware scoring
  - stronger candidate filtering before strong branching
  - better telemetry explaining why a branching choice won
- Success criterion: reduced median node count and better bound/incumbent progress on the benchmark target.

### 4. Separator stack and cut-propagation loop
- Build a stronger branch-and-cut core, not just more cut families on paper.
- Priorities:
  - implied-bound separation
  - stronger CMIR / lifted-cover style single-row cuts
  - path / mod-k style aggregated separators where they pay off
  - cutpool propagation and re-propagation after separation
  - tighter cut aging / activation logic tied to propagation usefulness
- Success criterion: better root progress and fewer nodes on structured MIPs without correctness regressions.

### 5. Incumbent-finding and search-control refinement
- Improve early primal progress and solve rate.
- Priorities:
  - feasibility jump or an equivalent fast repair heuristic
  - root reduced-cost heuristic
  - central / randomized / zi / shifting-style rounding variants
  - faster time-to-first-feasible
  - better primal integral
  - adaptive sub-MIP fixing-rate control
  - restart policy tuning
  - incumbent-guided neighborhood scheduling
  - tighter coordination between heuristics, cuts, presolve, and search profiles
- Success criterion: better mixed-score performance even if some internal diagnostics worsen.

### 6. Multi-thread tree intelligence
- Enable smarter tree search under multithreading, not just more workers.
- Priorities:
  - parallel-safe conflict learning and reuse
  - deterministic commit/reduction order in deterministic mode
  - multi-thread tree presolve for all-discrete models first
  - multi-thread tree cut logic
  - better parallel incumbent sharing and tree coordination
- Success criterion: improved benchmark outcomes at higher thread counts without breaking deterministic-mode reproducibility.

## Execution Policy
- Every major MIP task must be justified by expected impact on the Mittelman benchmark target.
- Prefer work that improves solve rate and primal progress over work that only improves internal counters.
- Opportunistic mode is allowed when it wins on benchmark outcomes.
- Deterministic mode remains the default measurement contract for controlled comparisons.

## Evaluation Policy
- Each major MIP change should be evaluated against HiGHS on the Mittelman MIP benchmark set.
- Required reported metrics:
  - solve-rate delta vs HiGHS
  - median wall-clock delta vs HiGHS
  - primal-integral or time-to-first-feasible delta
  - nodes and `work_units` as diagnostics
  - notable wins/losses by instance family
- A change is good if it improves the mixed benchmark outcome, even if node count or `work_units` does not improve.

## Guardrails
- Correctness remains a hard blocker:
  - no invalid cuts
  - no invalid conflicts
  - no unsound pruning
  - no unstable objective/status reporting
- Deterministic reproducibility checks stay in place for deterministic mode.
- Benchmark wins that come from incorrect results do not count.

## Immediate Next Sequence
1. Finish the `gen` loss bucket with exact root-stage telemetry and a candidate list of cheap large-root heuristics.
2. Do not widen the existing root LP heuristic gate by default; that path is now measured and known-bad on `gen`.
3. Build the first real implication/VUB/VLB subsystem and wire it into propagation and cut generation.
4. Add a persistent clique table with clique propagation before adding more tree-presolve complexity.
5. Add reduced-cost fixing and objective propagation as standalone engines, not ad hoc local tightenings.
6. Expand the primal heuristic portfolio with one cheap new winner candidate that is specifically suitable for large mixed roots.
7. Keep or revert direction based on benchmark outcome, not theory alone.
