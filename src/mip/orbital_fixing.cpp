#include "mipx/orbital_fixing.h"

#include <algorithm>
#include <cmath>
#include <unordered_set>

#include "mipx/schreier_sims.h"

namespace mipx {

namespace {

constexpr Real kTol = 1e-8;

bool isFixed(Real lb, Real ub) {
    return ub - lb < kTol;
}

}  // namespace

OrbitalFixingResult OrbitalFixer::fix(
    std::vector<Real>& col_lower,
    std::vector<Real>& col_upper,
    const std::vector<VarType>& col_type,
    const std::vector<Index>& fixed_vars,
    Index num_cols) const {
    OrbitalFixingResult result;

    if (schreier_sims_ == nullptr || !schreier_sims_->isBuilt()) {
        return result;
    }

    // Collect all currently fixed variables for the stabilizer computation.
    std::vector<Index> all_fixed;
    for (Index j = 0; j < num_cols; ++j) {
        if (isFixed(col_lower[j], col_upper[j])) {
            all_fixed.push_back(j);
        }
    }
    // Also include the explicitly provided fixed_vars.
    std::unordered_set<Index> fixed_set(all_fixed.begin(), all_fixed.end());
    for (Index j : fixed_vars) {
        fixed_set.insert(j);
    }
    all_fixed.assign(fixed_set.begin(), fixed_set.end());

    result.work_units += static_cast<double>(all_fixed.size());

    // For each fixed variable, compute its orbit under the stabilizer
    // of all other fixed variables. All orbit members get the same bounds.
    std::vector<bool> processed(static_cast<std::size_t>(num_cols), false);

    for (Index j : fixed_vars) {
        if (j < 0 || j >= num_cols) continue;
        if (col_type[j] == VarType::Continuous) continue;
        if (processed[j]) continue;

        // Compute orbit of j under stabilizer of other fixed variables.
        std::vector<Index> other_fixed;
        other_fixed.reserve(all_fixed.size());
        for (Index f : all_fixed) {
            if (f != j) other_fixed.push_back(f);
        }

        auto orbit = schreier_sims_->orbitUnderStabilizer(j, other_fixed);
        result.work_units += static_cast<double>(orbit.size());

        Real lb_j = col_lower[j];
        Real ub_j = col_upper[j];

        for (Index k : orbit) {
            if (k < 0 || k >= num_cols) continue;
            if (k == j) continue;
            if (col_type[k] == VarType::Continuous) continue;
            processed[k] = true;

            bool changed = false;

            // Tighten lower bound.
            if (lb_j > col_lower[k] + kTol) {
                if (lb_j > col_upper[k] + kTol) {
                    result.infeasible = true;
                    return result;
                }
                col_lower[k] = lb_j;
                changed = true;
            }

            // Tighten upper bound.
            if (ub_j < col_upper[k] - kTol) {
                if (ub_j < col_lower[k] - kTol) {
                    result.infeasible = true;
                    return result;
                }
                col_upper[k] = ub_j;
                changed = true;
            }

            if (changed) {
                result.tightened_vars.push_back(k);
                ++result.num_fixings;
            }
        }

        processed[j] = true;
    }

    return result;
}

OrbitalFixingResult OrbitalFixer::fixFromCanonical(
    std::vector<Real>& col_lower,
    std::vector<Real>& col_upper,
    const std::vector<Index>& canonical,
    Index num_cols) {
    OrbitalFixingResult result;

    if (canonical.empty()) return result;

    bool changed = true;
    while (changed) {
        changed = false;
        for (Index j = 0; j < num_cols; ++j) {
            Index canon = canonical[j];
            if (canon < 0 || canon == j || canon >= num_cols) continue;

            result.work_units += 1.0;

            // Propagate: canon lower bound >= j lower bound.
            Real new_canon_lower = std::max(col_lower[canon], col_lower[j]);
            if (new_canon_lower > col_upper[canon] + kTol) {
                result.infeasible = true;
                return result;
            }
            if (new_canon_lower > col_lower[canon] + kTol) {
                col_lower[canon] = new_canon_lower;
                changed = true;
                result.tightened_vars.push_back(canon);
                ++result.num_fixings;
            }

            // Propagate: j upper bound <= canon upper bound.
            Real new_var_upper = std::min(col_upper[j], col_upper[canon]);
            if (new_var_upper < col_lower[j] - kTol) {
                result.infeasible = true;
                return result;
            }
            if (new_var_upper < col_upper[j] - kTol) {
                col_upper[j] = new_var_upper;
                changed = true;
                result.tightened_vars.push_back(j);
                ++result.num_fixings;
            }
        }
    }

    return result;
}

}  // namespace mipx
