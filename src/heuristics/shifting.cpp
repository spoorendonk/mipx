#include "mipx/heuristics.h"

#include <algorithm>
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

std::optional<HeuristicSolution> ShiftingHeuristic::run(
    const LpProblem& problem,
    [[maybe_unused]] DualSimplexSolver& lp,
    std::span<const Real> primals,
    Real incumbent) {

    const Index n = problem.num_cols;
    std::vector<Real> x(primals.begin(), primals.end());

    // Step 1: round all integer variables to nearest integer.
    for (Index j = 0; j < n; ++j) {
        if (!isIntegerVar(problem.col_type[j])) continue;
        x[j] = std::round(primals[j]);
        x[j] = std::max(x[j], problem.col_lower[j]);
        x[j] = std::min(x[j], problem.col_upper[j]);
    }

    // Step 2: iteratively fix violated constraints by shifting variables.
    for (Int pass = 0; pass < max_passes_; ++pass) {
        bool changed = false;

        for (Index i = 0; i < problem.num_rows; ++i) {
            auto row = problem.matrix.row(i);
            Real activity = 0.0;
            for (Index k = 0; k < row.size(); ++k) {
                activity += row.values[k] * x[row.indices[k]];
            }

            Real excess = 0.0;
            bool need_decrease = false;
            if (activity > problem.row_upper[i] + kFeasTol) {
                excess = activity - problem.row_upper[i];
                need_decrease = true;
            } else if (activity < problem.row_lower[i] - kFeasTol) {
                excess = problem.row_lower[i] - activity;
                need_decrease = false;
            } else {
                continue;
            }

            // Find a variable in this row to shift.
            Index best_var = -1;
            Real best_shift = 0.0;
            Real best_cost = kInf;

            for (Index k = 0; k < row.size(); ++k) {
                Index j = row.indices[k];
                Real coeff = row.values[k];
                if (std::abs(coeff) < 1e-12) continue;

                Real shift = 0.0;
                if (need_decrease) {
                    // Need to decrease activity. If coeff > 0, decrease x[j].
                    // If coeff < 0, increase x[j].
                    if (coeff > 0) {
                        Real max_shift = x[j] - problem.col_lower[j];
                        Real needed = excess / coeff;
                        shift = -std::min(max_shift, needed);
                    } else {
                        Real max_shift = problem.col_upper[j] - x[j];
                        Real needed = excess / (-coeff);
                        shift = std::min(max_shift, needed);
                    }
                } else {
                    // Need to increase activity.
                    if (coeff > 0) {
                        Real max_shift = problem.col_upper[j] - x[j];
                        Real needed = excess / coeff;
                        shift = std::min(max_shift, needed);
                    } else {
                        Real max_shift = x[j] - problem.col_lower[j];
                        Real needed = excess / (-coeff);
                        shift = -std::min(max_shift, needed);
                    }
                }

                if (std::abs(shift) < 1e-12) continue;

                // For integer variables, round the shift to an integer.
                Real new_val = x[j] + shift;
                if (isIntegerVar(problem.col_type[j])) {
                    if (shift > 0) {
                        new_val = std::ceil(x[j] + shift - kFeasTol);
                    } else {
                        new_val = std::floor(x[j] + shift + kFeasTol);
                    }
                    new_val = std::max(new_val, problem.col_lower[j]);
                    new_val = std::min(new_val, problem.col_upper[j]);
                    shift = new_val - x[j];
                    if (std::abs(shift) < 1e-12) continue;
                }

                // Cost = objective degradation.
                Real cost = std::abs(problem.obj[j] * shift);
                if (cost < best_cost) {
                    best_cost = cost;
                    best_var = j;
                    best_shift = shift;
                }
            }

            if (best_var >= 0) {
                x[best_var] += best_shift;
                x[best_var] = std::max(x[best_var], problem.col_lower[best_var]);
                x[best_var] = std::min(x[best_var], problem.col_upper[best_var]);
                if (isIntegerVar(problem.col_type[best_var])) {
                    x[best_var] = std::round(x[best_var]);
                }
                changed = true;
            }
        }

        if (!changed) break;
    }

    if (!isRowFeasible(problem, x)) return std::nullopt;

    // Check integrality.
    for (Index j = 0; j < n; ++j) {
        if (!isIntegerVar(problem.col_type[j])) continue;
        if (std::abs(x[j] - std::round(x[j])) > kFeasTol) return std::nullopt;
    }

    Real obj = problem.obj_offset;
    for (Index j = 0; j < n; ++j) {
        obj += problem.obj[j] * x[j];
    }

    if (obj >= incumbent - kObjectiveTol) return std::nullopt;

    return HeuristicSolution{std::move(x), obj};
}

}  // namespace mipx
