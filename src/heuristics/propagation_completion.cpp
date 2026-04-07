#include "mipx/heuristics.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "mipx/branching.h"
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

std::optional<HeuristicSolution> PropagationCompletionHeuristic::run(
    const LpProblem& problem,
    [[maybe_unused]] DualSimplexSolver& lp,
    std::span<const Real> primals,
    Real incumbent) {

    const Index n = problem.num_cols;
    std::vector<Real> x(primals.begin(), primals.end());
    std::vector<Real> lower(problem.col_lower.begin(), problem.col_lower.end());
    std::vector<Real> upper(problem.col_upper.begin(), problem.col_upper.end());

    // Step 1: Fix near-integral integer variables.
    for (Index j = 0; j < n; ++j) {
        if (!isIntegerVar(problem.col_type[j])) continue;
        if (std::abs(primals[j] - std::round(primals[j])) <= kFeasTol) {
            Real val = std::round(primals[j]);
            val = std::max(val, lower[j]);
            val = std::min(val, upper[j]);
            lower[j] = val;
            upper[j] = val;
            x[j] = val;
        }
    }

    // Step 2: Propagate fixings through constraints.
    // For each constraint with only one unfixed integer variable,
    // derive bounds from the constraint and fix the variable if possible.
    for (Int round = 0; round < max_propagation_rounds_; ++round) {
        bool changed = false;

        for (Index i = 0; i < problem.num_rows; ++i) {
            auto row = problem.matrix.row(i);

            // Count unfixed integer variables in this row.
            Index unfixed_var = -1;
            Real unfixed_coeff = 0.0;
            Int unfixed_count = 0;
            Real fixed_activity = 0.0;

            for (Index k = 0; k < row.size(); ++k) {
                Index j = row.indices[k];
                if (isIntegerVar(problem.col_type[j]) &&
                    std::abs(lower[j] - upper[j]) > kFeasTol) {
                    ++unfixed_count;
                    unfixed_var = j;
                    unfixed_coeff = row.values[k];
                    if (unfixed_count > 1) break;
                } else {
                    fixed_activity += row.values[k] * x[j];
                }
            }

            if (unfixed_count != 1) continue;
            if (std::abs(unfixed_coeff) < 1e-12) continue;

            // Derive bounds for the unfixed variable.
            // row_lower <= fixed_activity + coeff * x_j <= row_upper
            Real derived_lower = lower[unfixed_var];
            Real derived_upper = upper[unfixed_var];

            if (unfixed_coeff > 0) {
                if (problem.row_lower[i] > -kInf) {
                    Real lb = (problem.row_lower[i] - fixed_activity) / unfixed_coeff;
                    derived_lower = std::max(derived_lower, lb);
                }
                if (problem.row_upper[i] < kInf) {
                    Real ub = (problem.row_upper[i] - fixed_activity) / unfixed_coeff;
                    derived_upper = std::min(derived_upper, ub);
                }
            } else {
                if (problem.row_upper[i] < kInf) {
                    Real lb = (problem.row_upper[i] - fixed_activity) / unfixed_coeff;
                    derived_lower = std::max(derived_lower, lb);
                }
                if (problem.row_lower[i] > -kInf) {
                    Real ub = (problem.row_lower[i] - fixed_activity) / unfixed_coeff;
                    derived_upper = std::min(derived_upper, ub);
                }
            }

            // Tighten to integer bounds.
            derived_lower = std::ceil(derived_lower - kFeasTol);
            derived_upper = std::floor(derived_upper + kFeasTol);

            if (derived_lower > derived_upper + kFeasTol) {
                // Infeasible propagation; can't fix this variable.
                continue;
            }

            if (derived_lower == derived_upper) {
                // Propagation fixed this variable.
                x[unfixed_var] = derived_lower;
                lower[unfixed_var] = derived_lower;
                upper[unfixed_var] = derived_upper;
                changed = true;
            } else if (derived_lower > lower[unfixed_var] + kFeasTol ||
                       derived_upper < upper[unfixed_var] - kFeasTol) {
                lower[unfixed_var] = std::max(lower[unfixed_var], derived_lower);
                upper[unfixed_var] = std::min(upper[unfixed_var], derived_upper);
                changed = true;
            }
        }

        if (!changed) break;
    }

    // Step 3: Round remaining unfixed integer variables.
    for (Index j = 0; j < n; ++j) {
        if (!isIntegerVar(problem.col_type[j])) continue;
        if (std::abs(lower[j] - upper[j]) <= kFeasTol) continue;  // Already fixed.

        Real rounded = std::round(x[j]);
        rounded = std::max(rounded, lower[j]);
        rounded = std::min(rounded, upper[j]);
        x[j] = rounded;
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
