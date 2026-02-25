#pragma once

#include <cmath>
#include <span>
#include <utility>

#include "mipx/bnb_node.h"
#include "mipx/core.h"

namespace mipx {

/// Check if a value is integer within tolerance.
inline bool isIntegral(Real val, Real tol = 1e-6) {
    return std::abs(val - std::round(val)) <= tol;
}

/// Fractionality: distance to nearest integer.
inline Real fractionality(Real val) {
    return std::abs(val - std::round(val));
}

/// Interface for branching variable selection.
class BranchingRule {
public:
    virtual ~BranchingRule() = default;

    /// Select a fractional integer variable to branch on.
    /// Returns the variable index, or -1 if all integer variables are integral.
    virtual Index select(std::span<const Real> primal_values,
                         std::span<const VarType> var_types,
                         std::span<const Real> col_lower,
                         std::span<const Real> col_upper) const = 0;
};

/// Branch on the variable with fractionality closest to 0.5.
class MostFractionalBranching : public BranchingRule {
public:
    Index select(std::span<const Real> primal_values,
                 std::span<const VarType> var_types,
                 std::span<const Real> col_lower,
                 std::span<const Real> col_upper) const override;
};

/// Branch on the first fractional integer variable found.
class FirstFractionalBranching : public BranchingRule {
public:
    Index select(std::span<const Real> primal_values,
                 std::span<const VarType> var_types,
                 std::span<const Real> col_lower,
                 std::span<const Real> col_upper) const override;
};

/// Create two child nodes by branching on variable branch_var with value branch_val.
/// Left child: x_j <= floor(branch_val), right child: x_j >= ceil(branch_val).
std::pair<BnbNode, BnbNode> createChildren(const BnbNode& parent,
                                            Index branch_var,
                                            Real branch_val);

}  // namespace mipx
