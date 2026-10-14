#include "mipx/cut_strengthening.h"

#include "mipx/lp_problem.h"

#include <cmath>

namespace mipx {

namespace {

// Margin taken off every reduction, relative to the magnitude of the terms
// summed into the activity bound. Rounding in those sums is ~1e-16 of it, so
// this leaves ample room; shrinking d never costs validity, only strength.
constexpr Real kMarginRel = 1e-9;
constexpr Real kMarginAbs = 1e-9;
// A reduction smaller than this fraction of the coefficient is noise.
constexpr Real kMinReductionRel = 1e-6;
// Slack when rounding an integer column's bounds to the integers inside them.
constexpr Real kIntBoundTol = 1e-9;

// The bounds the argument below runs on. It splits an integer column into
// x_j = u_j or x_j <= u_j - 1, which needs integral bounds; an integer column
// can carry a fractional one (unpresolved input, reduced-cost fixing), and
// its integer points lie within the rounded-in bounds anyway.
Real lowerOf(const LpProblem& problem, Index j) {
    const Real lb = problem.col_lower[j];
    return problem.col_type[j] == VarType::Continuous ? lb : std::ceil(lb - kIntBoundTol);
}

Real upperOf(const LpProblem& problem, Index j) {
    const Real ub = problem.col_upper[j];
    return problem.col_type[j] == VarType::Continuous ? ub : std::floor(ub + kIntBoundTol);
}

}  // namespace

Int strengthenCut(Cut& cut, const LpProblem& problem) {
    const bool has_upper = cut.upper < kInf;
    const bool has_lower = cut.lower > -kInf;
    if (cut.indices.empty() || cut.indices.size() != cut.values.size() || has_upper == has_lower) {
        return 0;
    }

    // Work in <= form: negate a >= cut, strengthen, negate back.
    const Real sign = has_upper ? 1.0 : -1.0;
    Real rhs = has_upper ? cut.upper : -cut.lower;
    if (!std::isfinite(rhs)) {
        return 0;
    }

    const auto n = static_cast<Index>(cut.indices.size());
    Real max_activity = 0.0;
    Real magnitude = std::abs(rhs);
    for (Index k = 0; k < n; ++k) {
        const Index j = cut.indices[k];
        if (j < 0 || j >= problem.num_cols) {
            return 0;
        }
        const Real a = sign * cut.values[k];
        const Real contrib = (a > 0.0) ? a * upperOf(problem, j) : a * lowerOf(problem, j);
        if (!std::isfinite(contrib)) {
            return 0;
        }
        max_activity += contrib;
        magnitude += std::abs(contrib);
    }
    // A cut the bounds already satisfy cannot be violated, and every reduction
    // would shrink its coefficients towards zero. Otherwise rhs < max_activity,
    // so each d below stays short of its coefficient and no term vanishes.
    if (max_activity <= rhs) {
        return 0;
    }
    const Real margin = kMarginRel * magnitude + kMarginAbs;

    // Each reduction lowers the activity bound and the rhs by the same amount,
    // so the slack b - max_activity the later columns see is unchanged.
    Int changed = 0;
    for (Index k = 0; k < n; ++k) {
        const Index j = cut.indices[k];
        if (problem.col_type[j] == VarType::Continuous) {
            continue;
        }
        const Real lb = lowerOf(problem, j);
        const Real ub = upperOf(problem, j);
        if (ub - lb < 0.5) {
            continue;  // fixed: no x_j strictly inside the bounds to relax
        }

        const Real a = sign * cut.values[k];
        if (a > 0.0) {
            // Largest activity with x_j <= ub - 1, against the rhs.
            const Real rest = max_activity - a * ub;
            const Real d = rhs - a * (ub - 1.0) - rest - margin;
            if (d <= kMinReductionRel * a) {
                continue;
            }
            cut.values[k] = sign * (a - d);
            rhs -= d * ub;
            max_activity -= d * ub;
        } else if (a < 0.0) {
            // Largest activity with x_j >= lb + 1, against the rhs.
            const Real rest = max_activity - a * lb;
            const Real d = rhs - a * (lb + 1.0) - rest - margin;
            if (d <= kMinReductionRel * -a) {
                continue;
            }
            cut.values[k] = sign * (a + d);
            rhs += d * lb;
            max_activity += d * lb;
        } else {
            continue;
        }
        ++changed;
    }
    if (changed == 0) {
        return 0;
    }

    if (has_upper) {
        cut.upper = rhs;
    } else {
        cut.lower = -rhs;
    }

    return changed;
}

}  // namespace mipx
