#include "mipx/cut_strengthening.h"

#include "mipx/lp_problem.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace mipx {

namespace {

constexpr Real kCoeffTol = 1e-10;
constexpr Real kBoundTol = 1e-8;

}  // namespace

bool strengthenCut(Cut& cut, const LpProblem& problem) {
    if (cut.indices.empty())
        return false;

    bool changed = false;

    // Tighten coefficients using variable bound information.
    // For a <= constraint: sum a_j x_j <= b
    // If x_j is integer with bounds [l_j, u_j], and a_j > 0:
    //   We can replace a_j with a_j' = min(a_j, b - sum_{k!=j} a_k * l_k)
    //   if that still dominates.
    // More specifically, for integer variables we can round down coefficients
    // when the fractional part doesn't affect validity.

    if (cut.upper < kInf && cut.lower <= -kInf) {
        // Upper-bounded cut: sum a_j x_j <= b
        for (Index k = 0; k < static_cast<Index>(cut.indices.size()); ++k) {
            const Index j = cut.indices[k];
            if (j < 0 || j >= problem.num_cols)
                continue;
            if (problem.col_type[j] == VarType::Continuous)
                continue;

            const Real a = cut.values[k];
            if (std::abs(a) < kCoeffTol)
                continue;

            // For integer variables, if the coefficient exceeds the RHS
            // minus the minimum contribution of other variables, we can tighten.
            if (a > 0.0 && problem.col_type[j] == VarType::Binary) {
                // Binary variable: coefficient can be tightened to
                // min(a, b - sum_{k!=j, a_k>0} a_k * 0 - sum_{k!=j, a_k<0} a_k * 1)
                // = min(a, b - sum of negative contributions)
                Real other_min = 0.0;
                for (Index m = 0; m < static_cast<Index>(cut.indices.size()); ++m) {
                    if (m == k)
                        continue;
                    const Index jm = cut.indices[m];
                    const Real am = cut.values[m];
                    if (jm < 0 || jm >= problem.num_cols)
                        continue;
                    if (am < 0.0) {
                        const Real ub = problem.col_upper[jm];
                        if (std::isfinite(ub)) {
                            other_min += am * ub;
                        }
                    }
                    // For positive coefficients, the minimum contribution
                    // comes from the lower bound.
                    if (am > 0.0) {
                        const Real lb = problem.col_lower[jm];
                        if (std::isfinite(lb) && lb > 0.0) {
                            other_min += am * lb;
                        }
                    }
                }
                const Real max_allowed = cut.upper - other_min;
                if (max_allowed > kCoeffTol && max_allowed < a - kBoundTol) {
                    cut.values[k] = max_allowed;
                    changed = true;
                }
            }
        }
    } else if (cut.lower > -kInf && cut.upper >= kInf) {
        // Lower-bounded cut: sum a_j x_j >= b
        for (Index k = 0; k < static_cast<Index>(cut.indices.size()); ++k) {
            const Index j = cut.indices[k];
            if (j < 0 || j >= problem.num_cols)
                continue;
            if (problem.col_type[j] == VarType::Continuous)
                continue;

            const Real a = cut.values[k];
            if (std::abs(a) < kCoeffTol)
                continue;

            // For binary variables with positive coefficient in a >= cut:
            // Tighten if coefficient is too large.
            if (a > 0.0 && problem.col_type[j] == VarType::Binary) {
                Real other_max = 0.0;
                for (Index m = 0; m < static_cast<Index>(cut.indices.size()); ++m) {
                    if (m == k)
                        continue;
                    const Index jm = cut.indices[m];
                    const Real am = cut.values[m];
                    if (jm < 0 || jm >= problem.num_cols)
                        continue;
                    if (am > 0.0) {
                        const Real ub = problem.col_upper[jm];
                        if (std::isfinite(ub)) {
                            other_max += am * ub;
                        }
                    }
                    if (am < 0.0) {
                        const Real lb = problem.col_lower[jm];
                        if (std::isfinite(lb)) {
                            other_max += am * lb;
                        }
                    }
                }
                const Real min_needed = cut.lower - other_max;
                if (min_needed > kCoeffTol && min_needed < a - kBoundTol) {
                    // Can tighten coefficient upward (for >= cut, larger coefficients
                    // are tighter). Actually for a >= cut, we want min_needed.
                    // If the variable must contribute at least min_needed when x_j=1,
                    // we can set a_j = max(a_j, min_needed). But that increases the
                    // coefficient, which doesn't help. So skip this direction.
                }
            }
        }
    }

    return changed;
}

bool complementCut(Cut& cut, const LpProblem& problem, std::span<const Real> primals) {
    if (cut.indices.empty())
        return false;
    if (cut.upper >= kInf && cut.lower <= -kInf)
        return false;

    bool changed = false;

    // Substitute bound-shifted variables to improve cut quality.
    // For each variable x_j, if complementing (x_j' = u_j - x_j) would
    // improve the cut violation, do it.
    //
    // This only applies to bounded integer variables.
    // Complementing x_j in sum a_j x_j <= b:
    //   a_j x_j = a_j (u_j - x_j') = -a_j x_j' + a_j u_j
    //   New coefficient: -a_j, new RHS: b - a_j u_j

    for (Index k = 0; k < static_cast<Index>(cut.indices.size()); ++k) {
        const Index j = cut.indices[k];
        if (j < 0 || j >= problem.num_cols)
            continue;
        if (problem.col_type[j] == VarType::Continuous)
            continue;

        const Real a = cut.values[k];
        if (std::abs(a) < kCoeffTol)
            continue;

        const Real lb = problem.col_lower[j];
        const Real ub = problem.col_upper[j];
        if (!std::isfinite(lb) || !std::isfinite(ub))
            continue;

        const Real x = (j < static_cast<Index>(primals.size())) ? primals[j] : lb;

        // Check if complementation would help:
        // Currently the variable contributes a * x to the LHS.
        // After complementation: -a * (u - x) = -a*u + a*x, so LHS change is 0
        // but the RHS changes. The key insight is that complementation can
        // change the rounding in CG/MIR cuts.
        // Here we do a simpler version: if the variable is closer to its
        // upper bound, complement to shift the coefficient sign.
        const Real mid = 0.5 * (lb + ub);
        if (x > mid + kBoundTol) {
            // Complement: x_j = u_j - x_j'
            if (cut.upper < kInf) {
                cut.upper -= a * ub;
            }
            if (cut.lower > -kInf) {
                cut.lower -= a * ub;
            }
            cut.values[k] = -a;
            changed = true;
        }
    }

    return changed;
}

bool makeNumericallySafe(Cut& cut) {
    if (cut.indices.empty())
        return false;

    bool changed = false;

    // Step 1: Remove near-zero coefficients.
    std::vector<Index> new_indices;
    std::vector<Real> new_values;
    new_indices.reserve(cut.indices.size());
    new_values.reserve(cut.values.size());

    for (Index k = 0; k < static_cast<Index>(cut.indices.size()); ++k) {
        const Real v = cut.values[k];
        if (std::abs(v) > 1e-10) {
            new_indices.push_back(cut.indices[k]);
            new_values.push_back(v);
        } else {
            changed = true;
        }
    }

    if (new_indices.empty()) {
        cut.indices.clear();
        cut.values.clear();
        return true;
    }

    // Step 2: Check dynamic range. If max_abs / min_abs > threshold, try to
    // scale the cut to improve conditioning.
    Real max_abs = 0.0;
    Real min_abs = std::numeric_limits<Real>::infinity();
    for (Real v : new_values) {
        const Real av = std::abs(v);
        max_abs = std::max(max_abs, av);
        min_abs = std::min(min_abs, av);
    }

    if (max_abs > 0.0 && min_abs > 0.0 && max_abs / min_abs > 1e6) {
        // Scale the entire cut by 1/max_abs to reduce dynamic range.
        const Real scale = 1.0 / max_abs;
        for (Real& v : new_values)
            v *= scale;
        if (cut.upper < kInf)
            cut.upper *= scale;
        if (cut.lower > -kInf)
            cut.lower *= scale;
        changed = true;
    }

    // Step 3: Round coefficients that are very close to integers (for integer vars).
    // This acts as an "exact arithmetic fallback" for borderline cuts.
    for (Index k = 0; k < static_cast<Index>(new_values.size()); ++k) {
        const Real v = new_values[k];
        const Real rounded = std::round(v);
        if (std::abs(v - rounded) < 1e-9 && std::abs(rounded) > kCoeffTol) {
            if (v != rounded) {
                new_values[k] = rounded;
                changed = true;
            }
        }
    }

    // Step 4: Round RHS if close to integer.
    if (cut.upper < kInf) {
        const Real rounded = std::round(cut.upper);
        if (std::abs(cut.upper - rounded) < 1e-9) {
            if (cut.upper != rounded) {
                cut.upper = rounded;
                changed = true;
            }
        }
    }
    if (cut.lower > -kInf) {
        const Real rounded = std::round(cut.lower);
        if (std::abs(cut.lower - rounded) < 1e-9) {
            if (cut.lower != rounded) {
                cut.lower = rounded;
                changed = true;
            }
        }
    }

    cut.indices = std::move(new_indices);
    cut.values = std::move(new_values);
    return changed;
}

}  // namespace mipx
