#include "mipx/sparse_matrix.h"

#include <algorithm>
#include <cstring>
#include <mutex>

namespace mipx {

SparseMatrix::SparseMatrix(Index rows, Index cols)
    : rows_(rows), cols_(cols), row_starts_(rows + 1, 0) {}

SparseMatrix::SparseMatrix(Index rows, Index cols, std::vector<Triplet> triplets)
    : rows_(rows), cols_(cols) {
    const Index nnz = static_cast<Index>(triplets.size());

    // Counting sort by row — O(N+R) instead of O(N log N).
    // 1. Count entries per row.
    row_starts_.assign(rows + 1, 0);
    for (Index k = 0; k < nnz; ++k) {
        ++row_starts_[triplets[k].row + 1];
    }

    // 2. Prefix sum.
    for (Index i = 0; i < rows; ++i) {
        row_starts_[i + 1] += row_starts_[i];
    }

    // 3. Distribute into row-bucketed arrays.
    std::vector<Triplet> sorted(nnz);
    std::vector<Index> pos(row_starts_.begin(), row_starts_.end());
    for (Index k = 0; k < nnz; ++k) {
        sorted[pos[triplets[k].row]++] = triplets[k];
    }

    // 4. Sort within each row by column + sum duplicates.
    values_.reserve(nnz);
    col_indices_.reserve(nnz);

    // Reset row_starts_ to rebuild with deduplication.
    row_starts_.assign(rows + 1, 0);

    for (Index i = 0; i < rows; ++i) {
        Index rstart = (i == 0) ? 0 : pos[i - 1];
        // pos[i] was advanced past row i's entries; use original prefix sum.
        // Actually, after distribution pos[i] == original row_starts_[i+1].
        // We need original bounds. Recompute from sorted array.
        Index rend = pos[i];
        // Sort this row's entries by column.
        std::sort(sorted.begin() + rstart, sorted.begin() + rend,
                  [](const Triplet& a, const Triplet& b) {
                      return a.col < b.col;
                  });
        // Sum duplicates.
        for (Index k = rstart; k < rend;) {
            Index c = sorted[k].col;
            Real v = 0.0;
            while (k < rend && sorted[k].col == c) {
                v += sorted[k].val;
                ++k;
            }
            if (v != 0.0) {
                values_.push_back(v);
                col_indices_.push_back(c);
                ++row_starts_[i + 1];
            }
        }
    }

    // Final prefix sum.
    for (Index i = 0; i < rows; ++i) {
        row_starts_[i + 1] += row_starts_[i];
    }
}

SparseMatrix::SparseMatrix(Index rows, Index cols, std::vector<Real> values,
                           std::vector<Index> col_indices,
                           std::vector<Index> row_starts)
    : rows_(rows),
      cols_(cols),
      values_(std::move(values)),
      col_indices_(std::move(col_indices)),
      row_starts_(std::move(row_starts)) {
    assert(static_cast<Index>(row_starts_.size()) == rows_ + 1);
    assert(static_cast<Index>(values_.size()) ==
           static_cast<Index>(col_indices_.size()));
}

SparseMatrix::SparseMatrix(const SparseMatrix& other)
    : rows_(other.rows_),
      cols_(other.cols_),
      values_(other.values_),
      col_indices_(other.col_indices_),
      row_starts_(other.row_starts_) {
    // CSC cache is lazily rebuilt per matrix instance.
    csc_valid_.store(false, std::memory_order_relaxed);
}

SparseMatrix& SparseMatrix::operator=(const SparseMatrix& other) {
    if (this == &other) return *this;
    std::lock_guard<std::mutex> lock(csc_mutex_);
    rows_ = other.rows_;
    cols_ = other.cols_;
    values_ = other.values_;
    col_indices_ = other.col_indices_;
    row_starts_ = other.row_starts_;
    csc_valid_.store(false, std::memory_order_relaxed);
    csc_values_.clear();
    csc_row_indices_.clear();
    csc_col_starts_.clear();
    return *this;
}

SparseMatrix::SparseMatrix(SparseMatrix&& other) noexcept
    : rows_(other.rows_),
      cols_(other.cols_),
      values_(std::move(other.values_)),
      col_indices_(std::move(other.col_indices_)),
      row_starts_(std::move(other.row_starts_)),
      csc_values_(std::move(other.csc_values_)),
      csc_row_indices_(std::move(other.csc_row_indices_)),
      csc_col_starts_(std::move(other.csc_col_starts_)) {
    csc_valid_.store(other.csc_valid_.load(std::memory_order_relaxed),
                     std::memory_order_relaxed);
    other.csc_valid_.store(false, std::memory_order_relaxed);
}

SparseMatrix& SparseMatrix::operator=(SparseMatrix&& other) noexcept {
    if (this == &other) return *this;
    std::lock_guard<std::mutex> lock(csc_mutex_);
    rows_ = other.rows_;
    cols_ = other.cols_;
    values_ = std::move(other.values_);
    col_indices_ = std::move(other.col_indices_);
    row_starts_ = std::move(other.row_starts_);
    csc_values_ = std::move(other.csc_values_);
    csc_row_indices_ = std::move(other.csc_row_indices_);
    csc_col_starts_ = std::move(other.csc_col_starts_);
    csc_valid_.store(other.csc_valid_.load(std::memory_order_relaxed),
                     std::memory_order_relaxed);
    other.csc_valid_.store(false, std::memory_order_relaxed);
    return *this;
}

SparseVectorView SparseMatrix::row(Index i) const {
    assert(i >= 0 && i < rows_);
    Index start = row_starts_[i];
    Index end = row_starts_[i + 1];
    return {std::span<const Index>(col_indices_.data() + start, end - start),
            std::span<const Real>(values_.data() + start, end - start)};
}

SparseVectorView SparseMatrix::col(Index j) const {
    assert(j >= 0 && j < cols_);
    if (!csc_valid_.load(std::memory_order_acquire)) {
        std::lock_guard<std::mutex> lock(csc_mutex_);
        if (!csc_valid_.load(std::memory_order_relaxed)) {
            buildCSC();
            csc_valid_.store(true, std::memory_order_release);
        }
    }
    Index start = csc_col_starts_[j];
    Index end = csc_col_starts_[j + 1];
    return {
        std::span<const Index>(csc_row_indices_.data() + start, end - start),
        std::span<const Real>(csc_values_.data() + start, end - start)};
}

Real SparseMatrix::coeff(Index i, Index j) const {
    assert(i >= 0 && i < rows_);
    assert(j >= 0 && j < cols_);
    Index start = row_starts_[i];
    Index end = row_starts_[i + 1];
    auto it = std::lower_bound(col_indices_.begin() + start,
                               col_indices_.begin() + end, j);
    if (it != col_indices_.begin() + end && *it == j) {
        return values_[it - col_indices_.begin()];
    }
    return 0.0;
}

void SparseMatrix::multiply(std::span<const Real> x, std::span<Real> y) const {
    assert(static_cast<Index>(x.size()) == cols_);
    assert(static_cast<Index>(y.size()) == rows_);
    for (Index i = 0; i < rows_; ++i) {
        Real sum = 0.0;
        Index start = row_starts_[i];
        Index end = row_starts_[i + 1];
        for (Index k = start; k < end; ++k) {
            sum += values_[k] * x[col_indices_[k]];
        }
        y[i] = sum;
    }
}

void SparseMatrix::multiplyTranspose(std::span<const Real> x,
                                     std::span<Real> y) const {
    assert(static_cast<Index>(x.size()) == rows_);
    assert(static_cast<Index>(y.size()) == cols_);
    std::fill(y.begin(), y.end(), 0.0);
    for (Index i = 0; i < rows_; ++i) {
        if (x[i] == 0.0) continue;
        Index start = row_starts_[i];
        Index end = row_starts_[i + 1];
        for (Index k = start; k < end; ++k) {
            y[col_indices_[k]] += values_[k] * x[i];
        }
    }
}

void SparseMatrix::addRow(std::span<const Index> indices,
                          std::span<const Real> values) {
    assert(static_cast<Index>(indices.size()) ==
           static_cast<Index>(values.size()));
    invalidateCSC();

    // Build sorted (index, value) pairs.
    std::vector<std::pair<Index, Real>> entries;
    entries.reserve(indices.size());
    for (Index k = 0; k < static_cast<Index>(indices.size()); ++k) {
        entries.emplace_back(indices[k], values[k]);
    }
    std::ranges::sort(entries, [](const auto& a, const auto& b) {
        return a.first < b.first;
    });

    for (const auto& [col, val] : entries) {
        col_indices_.push_back(col);
        values_.push_back(val);
    }
    ++rows_;
    row_starts_.push_back(static_cast<Index>(values_.size()));
}

void SparseMatrix::removeRow(Index i) {
    assert(i >= 0 && i < rows_);
    invalidateCSC();

    Index last = rows_ - 1;
    if (i == last) {
        // Remove last row's entries from flat arrays.
        Index start = row_starts_[last];
        values_.erase(values_.begin() + start, values_.end());
        col_indices_.erase(col_indices_.begin() + start, col_indices_.end());
    } else {
        // Swap row i with last row.
        Index i_start = row_starts_[i];
        Index i_end = row_starts_[i + 1];
        Index i_nnz = i_end - i_start;

        Index last_start = row_starts_[last];
        Index last_end = row_starts_[last + 1];
        Index last_nnz = last_end - last_start;

        // Copy last row's data.
        std::vector<Real> last_vals(values_.begin() + last_start,
                                    values_.begin() + last_end);
        std::vector<Index> last_cols(col_indices_.begin() + last_start,
                                     col_indices_.begin() + last_end);

        // Remove row i's entries and last row's entries from the arrays.
        // Strategy: erase row i entries, then erase last row entries (which
        // have shifted), then insert last row entries at row i's position.

        // Simpler approach: rebuild the affected portion.
        // Remove both rows' entries from the flat arrays.
        // Since last row is always at the end, remove it first, then row i.
        values_.erase(values_.begin() + last_start,
                      values_.begin() + last_end);
        col_indices_.erase(col_indices_.begin() + last_start,
                           col_indices_.begin() + last_end);

        // Now row i is still at the same position (last_start > i_end).
        values_.erase(values_.begin() + i_start, values_.begin() + i_end);
        col_indices_.erase(col_indices_.begin() + i_start,
                           col_indices_.begin() + i_end);

        // Insert last row's data at position i_start.
        values_.insert(values_.begin() + i_start, last_vals.begin(),
                       last_vals.end());
        col_indices_.insert(col_indices_.begin() + i_start, last_cols.begin(),
                            last_cols.end());

        // Rebuild row_starts_ from scratch since the shifts are complex.
        // Adjust row_starts_: row i now has last_nnz entries.
        Index delta = last_nnz - i_nnz;
        row_starts_[i + 1] = row_starts_[i] + last_nnz;
        for (Index r = i + 1; r < last; ++r) {
            row_starts_[r + 1] += delta;
        }
    }

    // Pop the last row.
    --rows_;
    row_starts_.pop_back();
}

void SparseMatrix::removeRowStable(Index i) {
    assert(i >= 0 && i < rows_);
    invalidateCSC();

    Index start = row_starts_[i];
    Index end = row_starts_[i + 1];
    Index nnz = end - start;

    values_.erase(values_.begin() + start, values_.begin() + end);
    col_indices_.erase(col_indices_.begin() + start,
                       col_indices_.begin() + end);

    // Shift row_starts_ down.
    row_starts_.erase(row_starts_.begin() + i);
    for (Index r = i; r < rows_; ++r) {
        row_starts_[r] -= nnz;
    }
    --rows_;
}

void SparseMatrix::buildCSC() const {
    Index nnz = numNonzeros();
    csc_col_starts_.assign(cols_ + 1, 0);
    csc_row_indices_.resize(nnz);
    csc_values_.resize(nnz);

    // Count entries per column.
    for (Index k = 0; k < nnz; ++k) {
        ++csc_col_starts_[col_indices_[k] + 1];
    }

    // Prefix sum.
    for (Index j = 0; j < cols_; ++j) {
        csc_col_starts_[j + 1] += csc_col_starts_[j];
    }

    // Fill CSC arrays.
    std::vector<Index> pos(csc_col_starts_.begin(), csc_col_starts_.end());
    for (Index i = 0; i < rows_; ++i) {
        Index start = row_starts_[i];
        Index end = row_starts_[i + 1];
        for (Index k = start; k < end; ++k) {
            Index j = col_indices_[k];
            Index p = pos[j]++;
            csc_row_indices_[p] = i;
            csc_values_[p] = values_[k];
        }
    }

}

void SparseMatrix::invalidateCSC() const {
    std::lock_guard<std::mutex> lock(csc_mutex_);
    csc_valid_.store(false, std::memory_order_release);
    csc_values_.clear();
    csc_row_indices_.clear();
    csc_col_starts_.clear();
}

}  // namespace mipx
