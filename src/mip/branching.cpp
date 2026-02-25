#include "mipx/branching.h"

#include <cmath>

namespace mipx {

Index MostFractionalBranching::select(
    std::span<const Real> primal_values,
    std::span<const VarType> var_types,
    std::span<const Real> col_lower,
    std::span<const Real> col_upper) const {

    Index best = -1;
    Real best_score = -1.0;  // Best = closest to 0.5.

    for (Index j = 0; j < static_cast<Index>(var_types.size()); ++j) {
        if (var_types[j] == VarType::Continuous) continue;
        // Skip fixed variables.
        if (col_lower[j] == col_upper[j]) continue;

        Real val = primal_values[j];
        if (isIntegral(val)) continue;

        Real frac = fractionality(val);
        // Score: closer to 0.5 is better. Use 0.5 - |frac - 0.5|.
        Real score = 0.5 - std::abs(frac - 0.5);
        if (score > best_score) {
            best_score = score;
            best = j;
        }
    }
    return best;
}

Index FirstFractionalBranching::select(
    std::span<const Real> primal_values,
    std::span<const VarType> var_types,
    std::span<const Real> col_lower,
    std::span<const Real> col_upper) const {

    for (Index j = 0; j < static_cast<Index>(var_types.size()); ++j) {
        if (var_types[j] == VarType::Continuous) continue;
        if (col_lower[j] == col_upper[j]) continue;

        if (!isIntegral(primal_values[j])) {
            return j;
        }
    }
    return -1;
}

std::pair<BnbNode, BnbNode> createChildren(const BnbNode& parent,
                                            Index branch_var,
                                            Real branch_val) {
    Real floor_val = std::floor(branch_val);
    Real ceil_val = std::ceil(branch_val);

    BnbNode left;
    left.parent_id = parent.id;
    left.depth = parent.depth + 1;
    left.basis = parent.basis;
    left.branch = {branch_var, floor_val, true};  // x_j <= floor(v)
    left.bound_changes = parent.bound_changes;
    left.bound_changes.push_back(left.branch);

    BnbNode right;
    right.parent_id = parent.id;
    right.depth = parent.depth + 1;
    right.basis = parent.basis;
    right.branch = {branch_var, ceil_val, false};  // x_j >= ceil(v)
    right.bound_changes = parent.bound_changes;
    right.bound_changes.push_back(right.branch);

    return {std::move(left), std::move(right)};
}

}  // namespace mipx
