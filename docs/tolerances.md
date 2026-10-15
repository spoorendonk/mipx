# Tolerance policy

The MIP layer asks a few questions over and over: is this point feasible,
is this value integral, has this domain emptied, can this node beat the
incumbent. Three wrong answers (#234, #235) came from two components asking
the same question with different tolerances. This page names the tolerances,
says what each one means, and records where each one is used (#236).

## The tolerances

| Name | Value | Meaning | Where it is defined |
|---|---|---|---|
| Primal feasibility | 1e-6, absolute | A point is feasible when every row and bound holds to 1e-6. Heuristics, the root check and the LP-free search accept points by this. | `heuristics/common.h` `kFeasTol`, `MipSolver::isFeasibleLp` |
| LP primal tolerance | 1e-7, in the scaled LP | What the dual simplex accepts. A node LP point is accepted as an incumbent on this plus integrality. | `DualSimplexSolver::kPrimalTol` |
| Integrality | 1e-6, absolute | A value is integral when it is within 1e-6 of an integer. | `MipSolver::kIntTol`, `isIntegral` |
| Bound crossing | 1e-8, absolute | A column whose lower bound passes its upper by at most 1e-8 is not empty; keep the swapped (or clamped) interval. | `DomainPropagator::kTol`, `keepCrossingWithinTolerance` |
| Row slack | 1e-8 times the row's magnitude | A row is violated only past 1e-8 times the largest of 1, its finite sides and its finite activity terms. Bounds derived from a row take the same slack. | `DomainPropagator::propagateRow`, `Presolver::rowTol` |
| Integer bound rounding | 1e-6; `1e-8 / \|a\|` when larger | A bound derived for an integer column rounds to the nearest integer within this before it is compared. Through a coefficient `a` the division scales the sums' rounding by `1/\|a\|`. | `DomainPropagator::kIntRoundTol`, `integerRoundTol` in presolve |
| Prune | 1e-6, absolute | A node is pruned when its bound is within 1e-6 of the incumbent. | `ObjectivePropagator::kPruneTol`, the node queue |
| Improvement | 1e-6, absolute | The solution pool keeps a solution only when it improves by more than 1e-6. The tree takes any strict improvement as its incumbent. | `SolutionPool::better` |

## The rule

**A test that prunes, fixes or declares infeasible is never stricter than
the feasibility test it guards.** The search keeps every point that holds
each row to the row slack and each bound to the crossing tolerance. Rounding
error of the row's own arithmetic is not a reason to remove a point.
Acceptance at 1e-6 is looser than that on rows of magnitude below 100. A
point that violates such a row by between the slack and 1e-6 is accepted
when a heuristic finds it, but the search does not promise to keep it.
Concretely:

- Infeasibility of a row is decided at the row slack, not at an absolute
  1e-8: at terms near 1e9 the activity sums round by more than 1e-8, and a
  point lying exactly on the row read as infeasible (#236).
- A bound crossing within 1e-8 is not a conflict anywhere. Node apply,
  symmetry enforcement, tree presolve, orbital fixing and the cutoff bounds
  all read it the same way (#234, #236).
- Integer bounds round at the integer bound rounding tolerance, never at the
  crossing tolerance (#235, and the objective propagator in #236).
- Slack added to a derived bound is safe only while the row stays in the
  model. Presolve, which drops rows, relaxes only its infeasibility claims.
  It keeps its absolute tests for bounds that replace a row (#210, #215),
  and drops an emptied row only when it holds within 1e-6 (`kDropTol`).
  Between that and the slack the row stays, and the LP decides.

## Known divergences left as they are

These answer the same question differently but were not reproduced as wrong
answers, and changing them alters behaviour on every model:

- Crossing tests in the reduced-cost fixer use 1e-9, and FPR and
  propagation completion use 1e-6.
- Pruning is absolute 1e-6 and does not scale with the objective's size.
- The tree accepts any strict improvement as its incumbent, but the pool
  needs 1e-6. The pool's best can therefore lag the tree's incumbent by less
  than 1e-6.
- Tree incumbents are accepted on the scaled LP tolerance. Their rows are
  never re-checked in original units.

## Checking it

The validity audits in `tests/test_validity.cpp` enumerate small models and
check the solver against them. The `Regression #236` cases there are the
constructions that broke the rule: rows with terms near 1e9, and rows with a
coefficient near 1e-9 to 1e-6. Each incumbent the search submits goes through
`MipSolver::setIncumbentObserver` and is checked against the model as given.
