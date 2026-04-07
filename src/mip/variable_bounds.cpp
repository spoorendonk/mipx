#include "mipx/variable_bounds.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "mipx/lp_problem.h"

namespace mipx {

const std::vector<VariableUpperBound> VariableBoundStore::kEmptyVUB{};
const std::vector<VariableLowerBound> VariableBoundStore::kEmptyVLB{};

void VariableBoundStore::init(Index num_cols) {
    num_cols_ = num_cols;
    vubs_.assign(static_cast<std::size_t>(num_cols), {});
    vlbs_.assign(static_cast<std::size_t>(num_cols), {});
    num_vubs_ = 0;
    num_vlbs_ = 0;
}

void VariableBoundStore::clear() {
    for (auto& v : vubs_) v.clear();
    for (auto& v : vlbs_) v.clear();
    num_vubs_ = 0;
    num_vlbs_ = 0;
}

void VariableBoundStore::addVUB(Index var, Index binary_var, Real coeff,
                                Real constant, Index source_row) {
    if (var < 0 || var >= num_cols_) return;
    if (!std::isfinite(coeff) || !std::isfinite(constant)) return;

    // Check for duplicate or dominated VUBs.
    auto& vubs = vubs_[var];
    for (auto& existing : vubs) {
        if (existing.binary_var == binary_var) {
            // Same binary variable: keep the tighter one.
            // VUB: x <= a*y + b. Tighter means smaller bound values.
            // When y=0: bound = b. When y=1: bound = a + b.
            Real old_at0 = existing.constant;
            Real old_at1 = existing.coeff + existing.constant;
            Real new_at0 = constant;
            Real new_at1 = coeff + constant;
            if (new_at0 <= old_at0 && new_at1 <= old_at1) {
                existing.coeff = coeff;
                existing.constant = constant;
                existing.source_row = source_row;
            }
            return;
        }
    }

    vubs.push_back({binary_var, coeff, constant, source_row});
    ++num_vubs_;
}

void VariableBoundStore::addVLB(Index var, Index binary_var, Real coeff,
                                Real constant, Index source_row) {
    if (var < 0 || var >= num_cols_) return;
    if (!std::isfinite(coeff) || !std::isfinite(constant)) return;

    // Check for duplicate or dominated VLBs.
    auto& vlbs = vlbs_[var];
    for (auto& existing : vlbs) {
        if (existing.binary_var == binary_var) {
            // Same binary variable: keep the tighter one.
            // VLB: x >= a*y + b. Tighter means larger bound values.
            Real old_at0 = existing.constant;
            Real old_at1 = existing.coeff + existing.constant;
            Real new_at0 = constant;
            Real new_at1 = coeff + constant;
            if (new_at0 >= old_at0 && new_at1 >= old_at1) {
                existing.coeff = coeff;
                existing.constant = constant;
                existing.source_row = source_row;
            }
            return;
        }
    }

    vlbs.push_back({binary_var, coeff, constant, source_row});
    ++num_vlbs_;
}

const std::vector<VariableUpperBound>& VariableBoundStore::vubs(Index var) const {
    if (var < 0 || var >= num_cols_) return kEmptyVUB;
    return vubs_[var];
}

const std::vector<VariableLowerBound>& VariableBoundStore::vlbs(Index var) const {
    if (var < 0 || var >= num_cols_) return kEmptyVLB;
    return vlbs_[var];
}

bool VariableBoundStore::hasVUB(Index var) const {
    if (var < 0 || var >= num_cols_) return false;
    return !vubs_[var].empty();
}

bool VariableBoundStore::hasVLB(Index var) const {
    if (var < 0 || var >= num_cols_) return false;
    return !vlbs_[var].empty();
}

Real VariableBoundStore::bestVUB(Index var, const std::vector<Real>& primals) const {
    if (var < 0 || var >= num_cols_) return kInf;
    Real best = kInf;
    for (const auto& vub : vubs_[var]) {
        if (vub.binary_var < 0 ||
            vub.binary_var >= static_cast<Index>(primals.size())) continue;
        Real y = primals[vub.binary_var];
        Real bound = vub.coeff * y + vub.constant;
        best = std::min(best, bound);
    }
    return best;
}

Real VariableBoundStore::bestVLB(Index var, const std::vector<Real>& primals) const {
    if (var < 0 || var >= num_cols_) return -kInf;
    Real best = -kInf;
    for (const auto& vlb : vlbs_[var]) {
        if (vlb.binary_var < 0 ||
            vlb.binary_var >= static_cast<Index>(primals.size())) continue;
        Real y = primals[vlb.binary_var];
        Real bound = vlb.coeff * y + vlb.constant;
        best = std::max(best, bound);
    }
    return best;
}

VariableBoundStore::CoefficientStrengthening VariableBoundStore::strengthenCoefficient(
    Index var, Real coeff, Real rhs,
    const std::vector<Real>& col_lower,
    const std::vector<Real>& col_upper) const {

    CoefficientStrengthening result{coeff, 0.0, false};

    if (var < 0 || var >= num_cols_) return result;
    if (!std::isfinite(coeff) || !std::isfinite(rhs)) return result;
    if (std::abs(coeff) < 1e-10) return result;

    // For positive coefficient a_j > 0 with VUB x_j <= c*y + d:
    // In constraint sum(a_i*x_i) <= rhs, we can strengthen a_j to a_j'
    // where a_j' = a_j - surplus and the rhs is adjusted accordingly.
    if (coeff > 0.0 && !vubs_[var].empty()) {
        for (const auto& vub : vubs_[var]) {
            if (vub.binary_var < 0 ||
                vub.binary_var >= static_cast<Index>(col_upper.size())) continue;

            // The VUB says: x_j <= vub.coeff * y + vub.constant
            // When y=0: x_j <= vub.constant
            // When y=1: x_j <= vub.coeff + vub.constant
            Real ub_at0 = vub.constant;
            Real ub_at1 = vub.coeff + vub.constant;

            if (!std::isfinite(ub_at0) || !std::isfinite(ub_at1)) continue;
            if (ub_at0 <= 0.0) continue;  // Only useful when VUB is binding.

            // The contribution of x_j is at most coeff * ub_at0 when y=0
            // and coeff * ub_at1 when y=1.
            // If coeff * ub_at0 > coeff * ub_at1, the bound at y=0 is dominant
            // and we can potentially tighten.
            Real surplus = coeff * (ub_at0 - ub_at1);
            if (surplus > 1e-8) {
                Real new_coeff = coeff * ub_at1 / ub_at0;
                Real delta = coeff * ub_at0 - new_coeff * ub_at0;
                if (new_coeff < coeff - 1e-8 && new_coeff > 1e-10) {
                    result.new_coeff = new_coeff;
                    result.rhs_delta = -delta;
                    result.strengthened = true;
                    return result;
                }
            }
        }
    }

    // For negative coefficient a_j < 0 with VLB x_j >= c*y + d:
    if (coeff < 0.0 && !vlbs_[var].empty()) {
        for (const auto& vlb : vlbs_[var]) {
            if (vlb.binary_var < 0 ||
                vlb.binary_var >= static_cast<Index>(col_lower.size())) continue;

            Real lb_at0 = vlb.constant;
            Real lb_at1 = vlb.coeff + vlb.constant;

            if (!std::isfinite(lb_at0) || !std::isfinite(lb_at1)) continue;

            // For negative coeff, smaller x_j gives larger contribution.
            // VLB at y=0 is lb_at0, at y=1 is lb_at1.
            Real surplus = coeff * (lb_at0 - lb_at1);  // coeff < 0
            if (surplus > 1e-8) {
                Real new_coeff = coeff * lb_at1 / lb_at0;
                Real delta = coeff * lb_at0 - new_coeff * lb_at0;
                if (new_coeff > coeff + 1e-8 && new_coeff < -1e-10) {
                    result.new_coeff = new_coeff;
                    result.rhs_delta = -delta;
                    result.strengthened = true;
                    return result;
                }
            }
        }
    }

    return result;
}

}  // namespace mipx
