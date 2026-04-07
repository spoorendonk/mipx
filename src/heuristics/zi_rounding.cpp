#include "mipx/heuristics.h"

#include <cmath>
#include <vector>

#include "mipx/dual_simplex.h"

namespace mipx {

namespace {

constexpr Real kFeasTol = 1e-6;
constexpr Real kObjectiveTol = 1e-6;

bool isIntegerVar(VarType t) {
    return t != VarType::Continuous;
}

bool isRowFeasible(const LpProblem& problem, std::span<const Real> values) {
    for (Index i = 0; i < problem.num_rows; ++i) {
        auto row = problem.matrix.row(i);
        Real activity = 0.0;
        for (Index k = 0; k < row.size(); ++k) {
            activity += row.values[k] * values[row.indices[k]];
        }
        if (activity < problem.row_lower[i] - kFeasTol) return false;
        if (activity > problem.row_upper[i] + kFeasTol) return false;
    }
    return true;
}

}  // namespace

std::optional<HeuristicSolution> ZiRoundingHeuristic::run(
    const LpProblem& problem,
    [[maybe_unused]] DualSimplexSolver& lp,
    std::span<const Real> primals,
    Real incumbent) {

    const Index n = problem.num_cols;
    std::vector<Real> x(primals.begin(), primals.end());

    // ZI-rounding: for each fractional integer variable, round in the
    // direction that does not worsen the objective. If the objective
    // coefficient is zero, round to nearest.
    for (Index j = 0; j < n; ++j) {
        if (!isIntegerVar(problem.col_type[j])) continue;
        if (std::abs(x[j] - std::round(x[j])) <= kFeasTol) {
            x[j] = std::round(x[j]);
            continue;
        }

        Real floor_val = std::floor(x[j]);
        Real ceil_val = std::ceil(x[j]);

        // Clamp to bounds.
        floor_val = std::max(floor_val, problem.col_lower[j]);
        ceil_val = std::min(ceil_val, problem.col_upper[j]);

        Real obj_coeff = problem.obj[j];
        // For minimization: positive obj_coeff means rounding down is better.
        if (obj_coeff > kFeasTol) {
            x[j] = floor_val;
        } else if (obj_coeff < -kFeasTol) {
            x[j] = ceil_val;
        } else {
            // Zero objective coefficient: round to nearest.
            x[j] = std::round(x[j]);
            x[j] = std::max(x[j], problem.col_lower[j]);
            x[j] = std::min(x[j], problem.col_upper[j]);
        }
    }

    if (!isRowFeasible(problem, x)) return std::nullopt;

    Real obj = problem.obj_offset;
    for (Index j = 0; j < n; ++j) {
        obj += problem.obj[j] * x[j];
    }

    if (obj >= incumbent - kObjectiveTol) return std::nullopt;

    return HeuristicSolution{std::move(x), obj};
}

}  // namespace mipx
