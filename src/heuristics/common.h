#pragma once

#include "mipx/core.h"
#include "mipx/lp_problem.h"

#include <cmath>
#include <span>
#include <utility>

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

/// The one test an incumbent passes before it is accepted, from whatever
/// source (#240): every column inside its bounds (a semi-continuous column
/// may also sit at 0) and integral where its type says so, and every row
/// within its sides, all to kFeasTol. A heuristic that copies LP values
/// unchecked cannot then pass an LP point from outside the model.
inline bool isFeasiblePoint(const LpProblem& problem, std::span<const Real> values) {
    if (std::cmp_less(values.size(), problem.num_cols)) {
        return false;
    }
    for (Index j = 0; j < problem.num_cols; ++j) {
        const Real v = values[j];
        if (!std::isfinite(v)) {
            return false;
        }
        const VarType t = problem.col_type[j];
        const bool semi = t == VarType::SemiContinuous || t == VarType::SemiInteger;
        const bool in_bounds =
            v >= problem.col_lower[j] - kFeasTol && v <= problem.col_upper[j] + kFeasTol;
        if (!in_bounds && !(semi && std::abs(v) <= kFeasTol)) {
            return false;
        }
        const bool integral_type =
            t == VarType::Integer || t == VarType::Binary || t == VarType::SemiInteger;
        if (integral_type && !isIntegral(v)) {
            return false;
        }
    }
    return isRowFeasible(problem, values);
}

inline Real clampToBounds(Real val, Real lb, Real ub) {
    return std::max(lb, std::min(val, ub));
}

}  // namespace mipx::heuristic_detail
