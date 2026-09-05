#include "mipx/variable_bounds.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
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

Real VariableBoundStore::impliedUpper(Index var, Index binary_var,
                                      bool binary_val) const {
    if (var < 0 || var >= num_cols_) return kInf;
    Real best = kInf;
    for (const auto& vub : vubs_[var]) {
        if (vub.binary_var != binary_var) continue;
        const Real bound = binary_val ? vub.coeff + vub.constant : vub.constant;
        if (std::isfinite(bound)) best = std::min(best, bound);
    }
    return best;
}

Real VariableBoundStore::impliedLower(Index var, Index binary_var,
                                      bool binary_val) const {
    if (var < 0 || var >= num_cols_) return -kInf;
    Real best = -kInf;
    for (const auto& vlb : vlbs_[var]) {
        if (vlb.binary_var != binary_var) continue;
        const Real bound = binary_val ? vlb.coeff + vlb.constant : vlb.constant;
        if (std::isfinite(bound)) best = std::max(best, bound);
    }
    return best;
}

VariableBoundStore::CoefficientStrengthening VariableBoundStore::strengthenCoefficient(
    Index binary_var, Real coeff, Real rhs,
    std::span<const Index> row_indices,
    std::span<const Real> row_values,
    const std::vector<Real>& col_lower,
    const std::vector<Real>& col_upper) const {

    constexpr Real kTol = 1e-8;

    CoefficientStrengthening result{coeff, 0.0, false};

    if (binary_var < 0 || binary_var >= num_cols_) return result;
    if (!std::isfinite(coeff) || !std::isfinite(rhs)) return result;
    if (std::abs(coeff) <= kTol) return result;
    if (row_indices.size() != row_values.size()) return result;

    // Branch in which the row has to be slack: y = 0 for a positive
    // coefficient, y = 1 for a negative one (i.e. complementing y).
    const bool branch_val = coeff < 0.0;

    // Maximum activity of the remaining terms in that branch, using the
    // implied bounds contributed by the VUBs/VLBs keyed on `binary_var`.
    Real max_activity = 0.0;
    for (std::size_t k = 0; k < row_indices.size(); ++k) {
        const Index j = row_indices[k];
        if (j == binary_var) continue;
        if (j < 0 || j >= static_cast<Index>(col_lower.size()) ||
            j >= static_cast<Index>(col_upper.size())) {
            return result;
        }
        const Real a = row_values[k];
        if (a == 0.0) continue;

        const Real lo = std::max(col_lower[j], impliedLower(j, binary_var, branch_val));
        const Real hi = std::min(col_upper[j], impliedUpper(j, binary_var, branch_val));
        if (lo > hi + kTol) return result;  // Branch is infeasible; not our job.

        const Real contrib = (a > 0.0) ? a * hi : a * lo;
        if (!std::isfinite(contrib)) return result;
        max_activity += contrib;
    }

    // Right-hand side left to the other terms once y's own contribution in the
    // branch is accounted for (0 when y = 0, `coeff` when y = 1).
    const Real branch_rhs = (coeff > 0.0) ? rhs : rhs - coeff;
    const Real delta = branch_rhs - max_activity;
    const Real magnitude = std::abs(coeff);
    if (!(delta > kTol && delta < magnitude - kTol)) return result;

    if (coeff > 0.0) {
        result.new_coeff = coeff - delta;
        result.rhs_delta = -delta;
    } else {
        result.new_coeff = coeff + delta;
        result.rhs_delta = 0.0;
    }
    result.strengthened = true;
    return result;
}

}  // namespace mipx
