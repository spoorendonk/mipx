#include "mipx/heuristics.h"

#include <cmath>

#include "mipx/branching.h"
#include "mipx/dual_simplex.h"

namespace mipx {

std::optional<HeuristicSolution> RoundingHeuristic::run(
    const LpProblem& problem,
    [[maybe_unused]] DualSimplexSolver& lp,
    std::span<const Real> primals,
    Real incumbent) {

    const Index n = problem.num_cols;
    std::vector<Real> rounded(primals.begin(), primals.end());

    // Round each integer/binary variable to nearest integer.
    for (Index j = 0; j < n; ++j) {
        if (problem.col_type[j] == VarType::Continuous) continue;
        rounded[j] = std::round(primals[j]);
        // Clamp to bounds.
        rounded[j] = std::max(rounded[j], problem.col_lower[j]);
        rounded[j] = std::min(rounded[j], problem.col_upper[j]);
    }

    // Check constraint feasibility: row_lower <= A*x <= row_upper.
    const auto& matrix = problem.matrix;
    constexpr Real tol = 1e-6;

    for (Index i = 0; i < problem.num_rows; ++i) {
        auto row = matrix.row(i);
        Real activity = 0.0;
        for (Index k = 0; k < row.size(); ++k) {
            activity += row.values[k] * rounded[row.indices[k]];
        }
        if (activity < problem.row_lower[i] - tol) return std::nullopt;
        if (activity > problem.row_upper[i] + tol) return std::nullopt;
    }

    // Compute objective.
    Real obj = problem.obj_offset;
    for (Index j = 0; j < n; ++j) {
        obj += problem.obj[j] * rounded[j];
    }

    // For maximize problems, the objective is negated internally,
    // but LpProblem stores the original. We compare with incumbent as-is
    // since MipSolver always works in minimization.
    if (obj >= incumbent - 1e-6) return std::nullopt;

    return HeuristicSolution{std::move(rounded), obj};
}

}  // namespace mipx
