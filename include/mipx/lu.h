#pragma once

#include <cmath>
#include <span>
#include <vector>

#include "mipx/core.h"

namespace mipx {

class SparseMatrix;

/// Sparse LU factorization with Markowitz pivoting and Forrest-Tomlin updates.
///
/// After factorize(), the basis B is represented as:
///     B = P^T * L * U * Q^T
/// where P, Q are row/column permutations, L is unit lower triangular
/// (stored as eta vectors), and U is upper triangular (stored row-wise).
class SparseLU {
public:
    SparseLU() = default;

    /// Factorize a square basis matrix (given as columns from constraint matrix).
    /// basis_cols[i] = column index in the original matrix for basis position i.
    void factorize(const SparseMatrix& matrix, std::span<const Index> basis_cols);

    /// Solve B*x = rhs in-place.
    void ftran(std::span<Real> rhs) const;

    /// Solve B^T*y = rhs in-place.
    void btran(std::span<Real> rhs) const;

    /// Forrest-Tomlin rank-1 update: replace basis column at position `pivot_pos`
    /// with new column `entering_col` (given in terms of original row indices/values).
    void update(Index pivot_pos, std::span<const Index> indices,
                std::span<const Real> values);

    /// Check if refactorization is needed.
    [[nodiscard]] bool needsRefactorization() const;
    [[nodiscard]] Index numUpdates() const { return num_updates_; }
    [[nodiscard]] Index dimension() const { return dim_; }

private:
    /// Apply L eta vectors forward (for FTRAN).
    void applyL(std::span<Real> x) const;

    /// Apply L^T eta vectors backward (for BTRAN).
    void applyLTranspose(std::span<Real> x) const;

    /// Apply Forrest-Tomlin update etas forward.
    void applyFT(std::span<Real> x) const;

    /// Apply Forrest-Tomlin update etas in reverse.
    void applyFTTranspose(std::span<Real> x) const;

    /// Backward solve with U (upper triangular, row-wise).
    void solveU(std::span<Real> x) const;

    /// Forward solve with U^T.
    void solveUTranspose(std::span<Real> x) const;

    Index dim_ = 0;

    // Permutations: row_perm_[k] = original row for elimination step k.
    std::vector<Index> row_perm_;
    std::vector<Index> col_perm_;
    std::vector<Index> row_perm_inv_;
    std::vector<Index> col_perm_inv_;

    // L stored as sequence of eta vectors.
    // eta_start_[k] .. eta_start_[k+1] gives the range for eta k.
    // eta_index_[p], eta_value_[p] are the (row, multiplier) pairs.
    // The pivot row entry (= 1.0) is not stored.
    std::vector<Index> eta_start_;
    std::vector<Index> eta_index_;
    std::vector<Real> eta_value_;

    // U stored row-wise (in elimination order).
    // u_start_[k] .. u_start_[k+1] gives the range for row k.
    // u_col_[p] is the column index (in elimination order), u_val_[p] the value.
    // Diagonal is stored as the last entry in each row.
    std::vector<Index> u_start_;
    std::vector<Index> u_col_;
    std::vector<Real> u_val_;
    std::vector<Real> u_diag_;  // u_diag_[k] = U(k,k)

    // Forrest-Tomlin update etas.
    // Each update stores an eta vector and the position of the replaced column.
    std::vector<Index> ft_start_;
    std::vector<Index> ft_index_;
    std::vector<Real> ft_value_;
    std::vector<Index> ft_pivot_pos_;  // column position in elimination order
    std::vector<Real> ft_pivot_val_;   // new diagonal value

    Index num_updates_ = 0;

    static constexpr Index kMaxUpdates = 100;
    static constexpr Real kPivotTol = 0.1;
    static constexpr Real kZeroTol = 1e-13;
    static constexpr Real kGrowthLimit = 1e12;

    Real max_u_entry_ = 0.0;  // track growth
};

}  // namespace mipx
