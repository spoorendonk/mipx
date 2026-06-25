#pragma once

#include "mipx/core.h"
#include "mipx/lp_problem.h"

#include <cmath>
#include <span>

namespace mipx::heuristic_detail {

constexpr Real kObjectiveTol = 1e-6;
constexpr Real kFeasTol = 1e-6;

inline bool isIntegerVar(VarType t) {
    return t != VarType::Continuous;
}

inline bool isIntegral(Real val, Real tol = kFeasTol) {
    return std::abs(val - std::round(val)) <= tol;
}

inline Real fractionality(Real val) {
    return std::abs(val - std::round(val));
}

inline Real computeObjective(const LpProblem& problem, std::span<const Real> values) {
    Real obj = problem.obj_offset;
    for (Index j = 0; j < problem.num_cols; ++j) {
        obj += problem.obj[j] * values[j];
    }
    return obj;
}

inline bool isRowFeasible(const LpProblem& problem, std::span<const Real> values) {
    for (Index i = 0; i < problem.num_rows; ++i) {
        auto row = problem.matrix.row(i);
        Real activity = 0.0;
        for (Index k = 0; k < row.size(); ++k) {
            activity += row.values[k] * values[row.indices[k]];
        }
        if (activity < problem.row_lower[i] - kFeasTol) {
            return false;
        }
        if (activity > problem.row_upper[i] + kFeasTol) {
            return false;
        }
    }
    return true;
}

inline Real clampToBounds(Real val, Real lb, Real ub) {
    return std::max(lb, std::min(val, ub));
}

}  // namespace mipx::heuristic_detail
