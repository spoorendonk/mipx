#pragma once

#include <span>
#include <vector>

#include "mipx/core.h"

namespace mipx {

/// A variable upper bound (VUB): x_j <= a * y_k + b, where y_k is binary.
/// When y_k = 0: x_j <= b.  When y_k = 1: x_j <= a + b.
struct VariableUpperBound {
    Index binary_var = -1;  // y_k
    Real coeff = 0.0;       // a
    Real constant = 0.0;    // b
    Index source_row = -1;  // Constraint that produced this VUB (-1 if from probing).
};

/// A variable lower bound (VLB): x_j >= a * y_k + b, where y_k is binary.
/// When y_k = 0: x_j >= b.  When y_k = 1: x_j >= a + b.
struct VariableLowerBound {
    Index binary_var = -1;  // y_k
    Real coeff = 0.0;       // a
    Real constant = 0.0;    // b
    Index source_row = -1;  // Constraint that produced this VLB (-1 if from probing).
};

/// Storage for variable-bound constraints indexed per variable.
///
/// For each continuous or general-integer variable, stores the best known
/// VUBs and VLBs involving binary variables. These are used for:
/// - Implied-bound cut generation (tighter coefficients)
/// - LP strengthening (substituting VUBs/VLBs into constraints)
/// - Coefficient strengthening
class VariableBoundStore {
public:
    VariableBoundStore() = default;

    /// Initialize for a problem with num_cols variables.
    void init(Index num_cols);

    /// Clear all stored bounds.
    void clear();

    /// Add a VUB for variable `var`: var <= coeff * binary_var + constant.
    void addVUB(Index var, Index binary_var, Real coeff, Real constant,
                Index source_row = -1);

    /// Add a VLB for variable `var`: var >= coeff * binary_var + constant.
    void addVLB(Index var, Index binary_var, Real coeff, Real constant,
                Index source_row = -1);

    /// Get all VUBs for a variable.
    [[nodiscard]] const std::vector<VariableUpperBound>& vubs(Index var) const;

    /// Get all VLBs for a variable.
    [[nodiscard]] const std::vector<VariableLowerBound>& vlbs(Index var) const;

    /// Check if a variable has any VUBs.
    [[nodiscard]] bool hasVUB(Index var) const;

    /// Check if a variable has any VLBs.
    [[nodiscard]] bool hasVLB(Index var) const;

    /// Get the best (tightest) VUB for a variable given current binary values.
    /// Returns the tightest upper bound achievable.
    [[nodiscard]] Real bestVUB(Index var, const std::vector<Real>& primals) const;

    /// Get the best (tightest) VLB for a variable given current binary values.
    [[nodiscard]] Real bestVLB(Index var, const std::vector<Real>& primals) const;

    /// Total number of VUBs stored.
    [[nodiscard]] Int numVUBs() const { return num_vubs_; }

    /// Total number of VLBs stored.
    [[nodiscard]] Int numVLBs() const { return num_vlbs_; }

    /// Tightest upper bound implied for `var` in the branch where the binary
    /// `binary_var` takes value `binary_val`, taken over the stored VUBs that
    /// reference `binary_var`. Returns +inf when no such VUB exists.
    [[nodiscard]] Real impliedUpper(Index var, Index binary_var, bool binary_val) const;

    /// Tightest lower bound implied for `var` in the branch where the binary
    /// `binary_var` takes value `binary_val`. Returns -inf when none exists.
    [[nodiscard]] Real impliedLower(Index var, Index binary_var, bool binary_val) const;

    /// Result of implication-based coefficient strengthening: the row
    ///     coeff * y + (other terms) <= rhs
    /// may be replaced by
    ///     new_coeff * y + (other terms) <= rhs + rhs_delta.
    struct CoefficientStrengthening {
        Real new_coeff;
        Real rhs_delta;
        bool strengthened = false;
    };

    /// Implication-based coefficient strengthening of a binary variable in a
    /// <= row, using the VUBs/VLBs that reference that binary.
    ///
    /// The row is  coeff * y + sum_{k != y} a_k * x_k <= rhs  with `y` =
    /// `binary_var` binary and (row_indices, row_values) listing every term of
    /// the row, including y's own term. The branch in which the row must be
    /// slack is y = 0 for coeff > 0 and y = 1 for coeff < 0. The maximum
    /// activity M of the other terms in that branch is computed from the
    /// column bounds intersected with the VUB/VLB-implied bounds for that
    /// branch — this is where the implications pay off: a term x_k with
    /// x_k <= c*y + d contributes only d in the y = 0 branch, not its global
    /// upper bound.
    ///
    /// With surplus delta = (rhs - y's contribution in the branch) - M, and
    /// 0 < delta < |coeff|, the row can be replaced by an equivalent one that
    /// is tighter for the LP relaxation:
    ///   coeff > 0:  (coeff - delta) * y + ... <= rhs - delta
    ///   coeff < 0:  (coeff + delta) * y + ... <= rhs
    /// In the branch where y is tight the strengthened row reduces to the
    /// original one; in the slack branch it reduces to "other terms <= M",
    /// which the bounds valid in that branch already imply. So no point that
    /// is feasible for the model is cut off, while the LP relaxation is
    /// strictly tighter.
    ///
    /// The caller must ensure `binary_var` is binary (bounds [0, 1]) and that
    /// `row_values` are the effective coefficients of the row.
    /// Returns {coeff, 0, false} when no strengthening applies.
    [[nodiscard]] CoefficientStrengthening strengthenCoefficient(
        Index binary_var, Real coeff, Real rhs,
        std::span<const Index> row_indices,
        std::span<const Real> row_values,
        const std::vector<Real>& col_lower,
        const std::vector<Real>& col_upper) const;

private:
    Index num_cols_ = 0;
    std::vector<std::vector<VariableUpperBound>> vubs_;
    std::vector<std::vector<VariableLowerBound>> vlbs_;
    Int num_vubs_ = 0;
    Int num_vlbs_ = 0;

    static const std::vector<VariableUpperBound> kEmptyVUB;
    static const std::vector<VariableLowerBound> kEmptyVLB;
};

}  // namespace mipx
