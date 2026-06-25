#include "common.h"
#include "mipx/dual_simplex.h"
#include "mipx/heuristics.h"

#include <cmath>

namespace mipx {

using namespace heuristic_detail;

std::optional<HeuristicSolution> RoundingHeuristic::run(const LpProblem& problem,
                                                        [[maybe_unused]] DualSimplexSolver& lp,
                                                        std::span<const Real> primals,
                                                        Real incumbent) {
    const Index n = problem.num_cols;
    std::vector<Real> rounded(primals.begin(), primals.end());

    // Round each integer/binary variable to nearest integer.
    for (Index j = 0; j < n; ++j) {
        if (problem.col_type[j] == VarType::Continuous) {
            continue;
        }
        rounded[j] = std::round(primals[j]);
        // Clamp to bounds.
        rounded[j] = std::max(rounded[j], problem.col_lower[j]);
        rounded[j] = std::min(rounded[j], problem.col_upper[j]);
    }

    // Check constraint feasibility: row_lower <= A*x <= row_upper.
    if (!isRowFeasible(problem, rounded)) {
        return std::nullopt;
    }

    // Compute objective.
    Real obj = computeObjective(problem, rounded);

    if (!betterObjective(problem.sense, obj, incumbent)) {
        return std::nullopt;
    }

    return HeuristicSolution{std::move(rounded), obj};
}

}  // namespace mipx
