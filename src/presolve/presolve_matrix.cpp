#include "presolve_matrix.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <limits>
#include <utility>

namespace mipx {

PresolveMatrix::PresolveMatrix(const SparseMatrix& matrix)
    : row_ids_(static_cast<std::size_t>(matrix.numRows())),
      col_ids_(static_cast<std::size_t>(matrix.numCols())),
      row_size_(static_cast<std::size_t>(matrix.numRows()), 0),
      col_size_(static_cast<std::size_t>(matrix.numCols()), 0) {
    const auto values = matrix.csr_values();
    const auto cols = matrix.csr_col_indices();
    const auto starts = matrix.csr_row_starts();
    const std::size_t nnz = values.size();
    entry_row_.reserve(nnz);
    entry_col_.reserve(nnz);
    entry_val_.reserve(nnz);
    entry_live_.assign(nnz, 1);
    for (const Index j : cols) {
        ++col_size_[j];
    }
    for (Index j = 0; j < matrix.numCols(); ++j) {
        col_ids_[j].reserve(static_cast<std::size_t>(col_size_[j]));
    }
    // Rows in order, so each column's ids come out sorted by row.
    for (Index i = 0; i < matrix.numRows(); ++i) {
        row_size_[i] = starts[i + 1] - starts[i];
        row_ids_[i].reserve(static_cast<std::size_t>(row_size_[i]));
        for (Index k = starts[i]; k < starts[i + 1]; ++k) {
            // One entry per position: a repeated column would be two live
            // entries at one (row, col).
            assert(k == starts[i] || cols[k - 1] < cols[k]);
            entry_row_.push_back(i);
            entry_col_.push_back(cols[k]);
            entry_val_.push_back(values[k]);
            row_ids_[i].push_back(k);
            col_ids_[cols[k]].push_back(k);
        }
    }
    num_entries_ = static_cast<Index>(nnz);
}

Index PresolveMatrix::find(Index row, Index col) const {
    // Binary search the shorter list; dead ids keep their keys, so it is
    // sorted, and a dead (row, col) may sit beside the live one.
    if (row_ids_[row].size() <= col_ids_[col].size()) {
        const auto& list = row_ids_[row];
        auto it =
            std::ranges::lower_bound(list, col, {}, [this](Index e) { return entry_col_[e]; });
        for (; it != list.end() && entry_col_[*it] == col; ++it) {
            if (entry_live_[*it] != 0) {
                return *it;
            }
        }
    } else {
        const auto& list = col_ids_[col];
        auto it =
            std::ranges::lower_bound(list, row, {}, [this](Index e) { return entry_row_[e]; });
        for (; it != list.end() && entry_row_[*it] == row; ++it) {
            if (entry_live_[*it] != 0) {
                return *it;
            }
        }
    }
    return -1;
}

Real PresolveMatrix::coeff(Index row, Index col) const {
    const Index e = find(row, col);
    return e >= 0 ? entry_val_[e] : 0.0;
}

void PresolveMatrix::compact(std::vector<Index>& list) const {
    std::erase_if(list, [this](Index e) { return entry_live_[e] == 0; });
}

void PresolveMatrix::set(Index row, Index col, Real value) {
    const Index found = find(row, col);
    if (found >= 0) {
        entry_val_[found] = value;
        return;
    }
    auto& row_list = row_ids_[row];
    auto& col_list = col_ids_[col];
    if (row_list.size() >= 2 * static_cast<std::size_t>(row_size_[row]) + 8) {
        compact(row_list);
    }
    if (col_list.size() >= 2 * static_cast<std::size_t>(col_size_[col]) + 8) {
        compact(col_list);
    }
    assert(entry_row_.size() < static_cast<std::size_t>(std::numeric_limits<Index>::max()));
    const Index e = static_cast<Index>(entry_row_.size());
    entry_row_.push_back(row);
    entry_col_.push_back(col);
    entry_val_.push_back(value);
    entry_live_.push_back(1);
    row_list.insert(
        std::ranges::upper_bound(row_list, col, {}, [this](Index id) { return entry_col_[id]; }),
        e);
    col_list.insert(
        std::ranges::upper_bound(col_list, row, {}, [this](Index id) { return entry_row_[id]; }),
        e);
    ++row_size_[row];
    ++col_size_[col];
    ++num_entries_;
}

void PresolveMatrix::kill(Index entry) {
    --row_size_[entry_row_[entry]];
    --col_size_[entry_col_[entry]];
    --num_entries_;
    entry_live_[entry] = 0;
}

bool PresolveMatrix::erase(Index row, Index col) {
    const Index e = find(row, col);
    if (e < 0) {
        return false;
    }
    kill(e);
    return true;
}

void PresolveMatrix::clearRow(Index row) {
    for (const Index e : row_ids_[row]) {
        if (entry_live_[e] != 0) {
            kill(e);
        }
    }
    row_ids_[row].clear();
}

void PresolveMatrix::clearCol(Index col) {
    for (const Index e : col_ids_[col]) {
        if (entry_live_[e] != 0) {
            kill(e);
        }
    }
    col_ids_[col].clear();
}

SparseMatrix PresolveMatrix::toSparseMatrix() const {
    std::vector<Index> starts(static_cast<std::size_t>(numRows()) + 1, 0);
    std::vector<Index> cols;
    std::vector<Real> values;
    cols.reserve(static_cast<std::size_t>(num_entries_));
    values.reserve(static_cast<std::size_t>(num_entries_));
    for (Index i = 0; i < numRows(); ++i) {
        // Rows are kept sorted by column.
        forEachInRow(i, [&](Index j, Real v) {
            cols.push_back(j);
            values.push_back(v);
        });
        starts[static_cast<std::size_t>(i) + 1] = static_cast<Index>(cols.size());
    }
    return {numRows(), numCols(), std::move(values), std::move(cols), std::move(starts)};
}

}  // namespace mipx
