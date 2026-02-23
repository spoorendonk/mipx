#pragma once

#include <algorithm>
#include <cassert>
#include <span>
#include <vector>

#include "mipx/core.h"

namespace mipx {

struct Triplet {
    Index row;
    Index col;
    Real val;
};

struct SparseVectorView {
    std::span<const Index> indices;
    std::span<const Real> values;

    [[nodiscard]] Index size() const { return static_cast<Index>(indices.size()); }
};

class SparseMatrix {
public:
    /// Construct an empty matrix with the given dimensions.
    SparseMatrix(Index rows, Index cols);

    /// Construct from triplets. Sorts by (row, col) and sums duplicates.
    SparseMatrix(Index rows, Index cols, std::vector<Triplet> triplets);

    /// Construct from pre-formed CSR arrays (moved in). col_indices must be
    /// sorted within each row.
    SparseMatrix(Index rows, Index cols, std::vector<Real> values,
                 std::vector<Index> col_indices, std::vector<Index> row_starts);

    [[nodiscard]] Index numRows() const { return rows_; }
    [[nodiscard]] Index numCols() const { return cols_; }
    [[nodiscard]] Index numNonzeros() const {
        return static_cast<Index>(values_.size());
    }

    /// Access row i as a sparse vector view.
    [[nodiscard]] SparseVectorView row(Index i) const;

    /// Access column j as a sparse vector view (triggers lazy CSC build).
    [[nodiscard]] SparseVectorView col(Index j) const;

    /// Look up coefficient (i, j) via binary search. Returns 0 if not present.
    [[nodiscard]] Real coeff(Index i, Index j) const;

    /// y = A * x (CSR-based).
    void multiply(std::span<const Real> x, std::span<Real> y) const;

    /// y = A^T * x (CSR scatter, no CSC needed).
    void multiplyTranspose(std::span<const Real> x, std::span<Real> y) const;

    /// Append a row. Invalidates CSC cache.
    void addRow(std::span<const Index> indices, std::span<const Real> values);

    /// Remove row i by swapping with last row (O(1)). Invalidates CSC cache.
    void removeRow(Index i);

    /// Remove row i preserving order (O(nnz)). Invalidates CSC cache.
    void removeRowStable(Index i);

    /// Raw CSR access.
    [[nodiscard]] std::span<const Real> csr_values() const { return values_; }
    [[nodiscard]] std::span<const Index> csr_col_indices() const {
        return col_indices_;
    }
    [[nodiscard]] std::span<const Index> csr_row_starts() const {
        return row_starts_;
    }

private:
    void buildCSC() const;
    void invalidateCSC() const;

    Index rows_;
    Index cols_;
    std::vector<Real> values_;
    std::vector<Index> col_indices_;
    std::vector<Index> row_starts_;

    // Lazy CSC cache (mutable for const col() access).
    mutable bool csc_valid_ = false;
    mutable std::vector<Real> csc_values_;
    mutable std::vector<Index> csc_row_indices_;
    mutable std::vector<Index> csc_col_starts_;
};

}  // namespace mipx
