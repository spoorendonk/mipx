#include "mipx/heuristics.h"

#include <cmath>

#include "mipx/branching.h"
#include "mipx/dual_simplex.h"

namespace mipx {

std::optional<HeuristicSolution> RinsHeuristic::run(
    const LpProblem& problem,
    DualSimplexSolver& lp,
    std::span<const Real> primals,
    Real incumbent) {

    const Index n = problem.num_cols;

    // Save original bounds to restore solver state after heuristic run.
    std::vector<Real> orig_lower(problem.col_lower.begin(), problem.col_lower.end());
    std::vector<Real> orig_upper(problem.col_upper.begin(), problem.col_upper.end());

    std::vector<Index> fixed_vars;
    fixed_vars.reserve(n);

    // Fix near-integral integer variables to their rounded LP values.
    for (Index j = 0; j < n; ++j) {
        if (problem.col_type[j] == VarType::Continuous) continue;

        Real rounded = std::round(primals[j]);
        rounded = std::max(rounded, problem.col_lower[j]);
        rounded = std::min(rounded, problem.col_upper[j]);

        if (std::abs(primals[j] - rounded) <= agreement_tol_) {
            lp.setColBounds(j, rounded, rounded);
            fixed_vars.push_back(j);
        }
    }

    if (fixed_vars.empty()) {
        return std::nullopt;
    }

    lp.setIterationLimit(subproblem_iter_limit_);
    auto result = lp.solve();

    std::optional<HeuristicSolution> best;

    if (result.status == Status::Optimal && result.objective < incumbent - 1e-6) {
        auto candidate = lp.getPrimalValues();

        bool integer_feasible = true;
        for (Index j = 0; j < n; ++j) {
            if (problem.col_type[j] == VarType::Continuous) continue;
            if (!isIntegral(candidate[j])) {
                integer_feasible = false;
                break;
            }
        }

        if (integer_feasible) {
            best = HeuristicSolution{std::move(candidate), result.objective};
        }
    }

    // Restore original bounds.
    for (Index j : fixed_vars) {
        lp.setColBounds(j, orig_lower[j], orig_upper[j]);
    }

    // Restore default limit and LP state for caller.
    lp.setIterationLimit(1000000);
    lp.solve();

    return best;
}

}  // namespace mipx
