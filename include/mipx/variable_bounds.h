#pragma once

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

    /// Strengthen coefficient of `var` in a <= constraint using VUBs/VLBs.
    /// Given: a_j * x_j in constraint sum(a_i * x_i) <= rhs,
    /// if x_j has VUB x_j <= c * y + d, we can potentially tighten a_j.
    /// Returns {new_coeff, new_rhs_delta} or {original_coeff, 0} if no strengthening.
    struct CoefficientStrengthening {
        Real new_coeff;
        Real rhs_delta;
        bool strengthened = false;
    };
    [[nodiscard]] CoefficientStrengthening strengthenCoefficient(
        Index var, Real coeff, Real rhs,
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
