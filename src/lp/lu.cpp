#include "mipx/lu.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "mipx/sparse_matrix.h"

namespace mipx {

// --------------------------------------------------------------------------
//  Markowitz LU factorization
// --------------------------------------------------------------------------

void SparseLU::factorize(const SparseMatrix& matrix,
                         std::span<const Index> basis_cols) {
    dim_ = static_cast<Index>(basis_cols.size());
    num_updates_ = 0;
    max_u_entry_ = 0.0;

    // Clear Forrest-Tomlin update data.
    ft_start_.clear();
    ft_start_.push_back(0);
    ft_index_.clear();
    ft_value_.clear();
    ft_pivot_pos_.clear();
    ft_pivot_val_.clear();

    // Build dense-ish active submatrix from selected columns.
    // active[i][j] = value at (original_row i, basis_position j).
    // We use a flat vector + linked-list structures for rows/cols.

    // Collect nonzeros into column-oriented lists.
    // For the active submatrix we track: row indices, column indices,
    // values, and counts for Markowitz.

    struct Entry {
        Index row;
        Index col;
        Real val;
        Index next_in_row;  // linked list in same row
        Index next_in_col;  // linked list in same col
        Index prev_in_row;
        Index prev_in_col;
        bool alive;
    };

    std::vector<Entry> entries;
    entries.reserve(dim_ * 4);  // rough estimate

    // Head of linked list for each row/col (-1 = empty).
    std::vector<Index> row_head(dim_, -1);
    std::vector<Index> col_head(dim_, -1);
    std::vector<Index> row_count(dim_, 0);
    std::vector<Index> col_count(dim_, 0);

    // Populate from matrix columns.
    for (Index j = 0; j < dim_; ++j) {
        auto colview = matrix.col(basis_cols[j]);
        for (Index k = 0; k < colview.size(); ++k) {
            Index row = colview.indices[k];
            Real val = colview.values[k];
            if (std::abs(val) < kZeroTol) continue;
            if (row >= dim_) continue;  // skip rows outside basis dimension

            Index eidx = static_cast<Index>(entries.size());
            entries.push_back({row, j, val, row_head[row], col_head[j],
                               -1, -1, true});

            // Update row linked list.
            if (row_head[row] >= 0) {
                entries[row_head[row]].prev_in_row = eidx;
            }
            row_head[row] = eidx;
            ++row_count[row];

            // Update col linked list.
            if (col_head[j] >= 0) {
                entries[col_head[j]].prev_in_col = eidx;
            }
            col_head[j] = eidx;
            ++col_count[j];
        }
    }

    // Permutation arrays.
    row_perm_.resize(dim_);
    col_perm_.resize(dim_);
    row_perm_inv_.resize(dim_);
    col_perm_inv_.resize(dim_);
    std::fill(row_perm_.begin(), row_perm_.end(), -1);
    std::fill(col_perm_.begin(), col_perm_.end(), -1);

    // L eta vectors.
    eta_start_.clear();
    eta_start_.reserve(dim_ + 1);
    eta_start_.push_back(0);
    eta_index_.clear();
    eta_value_.clear();

    // U row-wise storage.
    u_start_.clear();
    u_start_.reserve(dim_ + 1);
    u_start_.push_back(0);
    u_col_.clear();
    u_val_.clear();
    u_diag_.resize(dim_);

    // Track which rows/cols are still active.
    std::vector<bool> row_active(dim_, true);
    std::vector<bool> col_active(dim_, true);

    // Helper: remove entry from row and col lists.
    auto removeEntry = [&](Index eidx) {
        Entry& e = entries[eidx];
        e.alive = false;

        // Remove from row list.
        if (e.prev_in_row >= 0) {
            entries[e.prev_in_row].next_in_row = e.next_in_row;
        } else {
            row_head[e.row] = e.next_in_row;
        }
        if (e.next_in_row >= 0) {
            entries[e.next_in_row].prev_in_row = e.prev_in_row;
        }
        --row_count[e.row];

        // Remove from col list.
        if (e.prev_in_col >= 0) {
            entries[e.prev_in_col].next_in_col = e.next_in_col;
        } else {
            col_head[e.col] = e.next_in_col;
        }
        if (e.next_in_col >= 0) {
            entries[e.next_in_col].prev_in_col = e.prev_in_col;
        }
        --col_count[e.col];
    };

    // Helper: add entry to row and col lists.
    auto addEntry = [&](Index row, Index col, Real val) -> Index {
        Index eidx = static_cast<Index>(entries.size());
        entries.push_back({row, col, val, row_head[row], col_head[col],
                           -1, -1, true});
        if (row_head[row] >= 0) {
            entries[row_head[row]].prev_in_row = eidx;
        }
        row_head[row] = eidx;
        ++row_count[row];

        if (col_head[col] >= 0) {
            entries[col_head[col]].prev_in_col = eidx;
        }
        col_head[col] = eidx;
        ++col_count[col];
        return eidx;
    };

    // Work vector for gathering a column during elimination.
    std::vector<Real> work(dim_, 0.0);
    std::vector<Index> work_indices;
    work_indices.reserve(dim_);

    for (Index step = 0; step < dim_; ++step) {
        // ---- Markowitz pivot selection ----
        // Find pivot minimizing (row_nnz - 1) * (col_nnz - 1)
        // among entries with |a_ij| >= kPivotTol * max|a_*j|.

        Index best_entry = -1;
        long long best_markowitz = static_cast<long long>(dim_) * dim_;
        Real best_abs = 0.0;

        // First check for singleton columns (Markowitz = 0).
        for (Index j = 0; j < dim_; ++j) {
            if (!col_active[j]) continue;
            if (col_count[j] == 1) {
                Index eidx = col_head[j];
                if (eidx >= 0 && entries[eidx].alive) {
                    Real absval = std::abs(entries[eidx].val);
                    if (absval > kZeroTol) {
                        long long m = static_cast<long long>(row_count[entries[eidx].row] - 1) *
                                      static_cast<long long>(col_count[j] - 1);
                        if (m < best_markowitz || (m == best_markowitz && absval > best_abs)) {
                            best_markowitz = m;
                            best_entry = eidx;
                            best_abs = absval;
                        }
                    }
                }
            }
        }

        // Check for singleton rows.
        if (best_markowitz > 0) {
            for (Index i = 0; i < dim_; ++i) {
                if (!row_active[i]) continue;
                if (row_count[i] == 1) {
                    Index eidx = row_head[i];
                    if (eidx >= 0 && entries[eidx].alive) {
                        Real absval = std::abs(entries[eidx].val);
                        if (absval > kZeroTol) {
                            best_markowitz = 0;
                            best_entry = eidx;
                            best_abs = absval;
                            break;
                        }
                    }
                }
            }
        }

        // General search if no singletons found.
        if (best_markowitz > 0) {
            for (Index j = 0; j < dim_; ++j) {
                if (!col_active[j]) continue;

                // Find max in this column for threshold.
                Real col_max = 0.0;
                for (Index eidx = col_head[j]; eidx >= 0;
                     eidx = entries[eidx].next_in_col) {
                    if (entries[eidx].alive) {
                        col_max = std::max(col_max, std::abs(entries[eidx].val));
                    }
                }
                Real threshold = kPivotTol * col_max;

                for (Index eidx = col_head[j]; eidx >= 0;
                     eidx = entries[eidx].next_in_col) {
                    if (!entries[eidx].alive) continue;
                    Real absval = std::abs(entries[eidx].val);
                    if (absval < threshold) continue;

                    Index ri = entries[eidx].row;
                    long long m = static_cast<long long>(row_count[ri] - 1) *
                                  static_cast<long long>(col_count[j] - 1);
                    if (m < best_markowitz ||
                        (m == best_markowitz && absval > best_abs)) {
                        best_markowitz = m;
                        best_entry = eidx;
                        best_abs = absval;
                    }
                }

                // Early exit if we found Markowitz count 1.
                if (best_markowitz <= 1) break;
            }
        }

        if (best_entry < 0 || best_abs < kZeroTol) {
            throw std::runtime_error("SparseLU::factorize: singular basis matrix");
        }

        Index pivot_row = entries[best_entry].row;
        Index pivot_col = entries[best_entry].col;
        Real pivot_val = entries[best_entry].val;

        // Record permutations.
        row_perm_[step] = pivot_row;
        col_perm_[step] = pivot_col;
        row_perm_inv_[pivot_row] = step;
        col_perm_inv_[pivot_col] = step;

        // ---- Extract pivot row into U ----
        // Gather all entries in the pivot row.
        for (Index eidx = row_head[pivot_row]; eidx >= 0;
             eidx = entries[eidx].next_in_row) {
            if (!entries[eidx].alive) continue;
            Index c = entries[eidx].col;
            if (c != pivot_col) {
                u_col_.push_back(c);  // will remap to elimination order later
                u_val_.push_back(entries[eidx].val);
                max_u_entry_ = std::max(max_u_entry_, std::abs(entries[eidx].val));
            }
        }
        u_diag_[step] = pivot_val;
        max_u_entry_ = std::max(max_u_entry_, std::abs(pivot_val));
        u_start_.push_back(static_cast<Index>(u_col_.size()));

        // ---- Compute L eta vector (multipliers) and update active submatrix ----
        // For each row i with a nonzero in pivot_col: multiplier = a(i, pivot_col) / pivot_val
        // Then for each entry in the pivot row: a(i, j) -= multiplier * a(pivot_row, j)

        // Gather pivot row values into work vector.
        work_indices.clear();
        for (Index eidx = row_head[pivot_row]; eidx >= 0;
             eidx = entries[eidx].next_in_row) {
            if (!entries[eidx].alive) continue;
            Index c = entries[eidx].col;
            if (c != pivot_col) {
                work[c] = entries[eidx].val;
                work_indices.push_back(c);
            }
        }

        // Collect rows to update (from pivot column, excluding pivot row).
        std::vector<std::pair<Index, Real>> rows_to_update;
        for (Index eidx = col_head[pivot_col]; eidx >= 0;
             eidx = entries[eidx].next_in_col) {
            if (!entries[eidx].alive) continue;
            if (entries[eidx].row == pivot_row) continue;
            rows_to_update.emplace_back(entries[eidx].row, entries[eidx].val);
        }

        // Store L eta vector.
        for (auto& [ri, aval] : rows_to_update) {
            Real mult = aval / pivot_val;
            eta_index_.push_back(ri);
            eta_value_.push_back(mult);
        }
        eta_start_.push_back(static_cast<Index>(eta_index_.size()));

        // Update rows.
        for (auto& [ri, aval] : rows_to_update) {
            Real mult = aval / pivot_val;

            // Gather existing entries of row ri into a temporary map.
            // We need to update: for each col c in pivot row, a(ri, c) -= mult * a(pivot_row, c).
            // First, build a quick lookup of existing entries in row ri.
            // We use the work vector approach: scan row ri entries for cols in work_indices.

            // For each column in the pivot row, update the corresponding entry in row ri.
            // First, mark which columns row ri already has entries for.
            std::vector<Index> ri_entries;
            for (Index eidx = row_head[ri]; eidx >= 0;
                 eidx = entries[eidx].next_in_row) {
                if (entries[eidx].alive && entries[eidx].col != pivot_col) {
                    ri_entries.push_back(eidx);
                }
            }

            // Build a col->entry_idx map for row ri.
            // Use the work vector temporarily (we'll clean up).
            // Actually, use a separate approach: scan row ri entries
            // and for matching cols, update directly.

            // Sparse-sparse update: for each col c in pivot row...
            // Use a vector marking which cols we found.
            std::vector<Real> ri_vals(dim_, 0.0);
            std::vector<bool> ri_has(dim_, false);
            std::vector<Index> ri_entry_for_col(dim_, -1);

            for (Index eidx : ri_entries) {
                Index c = entries[eidx].col;
                ri_vals[c] = entries[eidx].val;
                ri_has[c] = true;
                ri_entry_for_col[c] = eidx;
            }

            for (Index c : work_indices) {
                Real update_val = mult * work[c];
                if (ri_has[c]) {
                    Real new_val = ri_vals[c] - update_val;
                    if (std::abs(new_val) < kZeroTol) {
                        // Remove entry.
                        removeEntry(ri_entry_for_col[c]);
                    } else {
                        entries[ri_entry_for_col[c]].val = new_val;
                    }
                } else {
                    Real new_val = -update_val;
                    if (std::abs(new_val) >= kZeroTol) {
                        addEntry(ri, c, new_val);
                    }
                }
            }
        }

        // Remove pivot row and pivot column from active submatrix.
        // Remove all entries in pivot row.
        {
            Index eidx = row_head[pivot_row];
            while (eidx >= 0) {
                Index next = entries[eidx].next_in_row;
                if (entries[eidx].alive) {
                    removeEntry(eidx);
                }
                eidx = next;
            }
        }
        // Remove all entries in pivot column.
        {
            Index eidx = col_head[pivot_col];
            while (eidx >= 0) {
                Index next = entries[eidx].next_in_col;
                if (entries[eidx].alive) {
                    removeEntry(eidx);
                }
                eidx = next;
            }
        }

        row_active[pivot_row] = false;
        col_active[pivot_col] = false;

        // Clean up work vector.
        for (Index c : work_indices) {
            work[c] = 0.0;
        }
    }

    // Remap U column indices from original to elimination order.
    for (auto& c : u_col_) {
        c = col_perm_inv_[c];
    }
}

// --------------------------------------------------------------------------
//  FTRAN: solve B*x = b
// --------------------------------------------------------------------------

void SparseLU::applyL(std::span<Real> x) const {
    // Apply L eta vectors in forward order (step 0, 1, ..., dim-1).
    for (Index step = 0; step < dim_; ++step) {
        Index start = eta_start_[step];
        Index end = eta_start_[step + 1];
        Real pivot_x = x[row_perm_[step]];
        if (pivot_x == 0.0) continue;
        for (Index k = start; k < end; ++k) {
            x[eta_index_[k]] -= eta_value_[k] * pivot_x;
        }
    }
}

void SparseLU::applyLTranspose(std::span<Real> x) const {
    // Apply L^T in reverse order.
    for (Index step = dim_ - 1; step >= 0; --step) {
        Index start = eta_start_[step];
        Index end = eta_start_[step + 1];
        Real sum = 0.0;
        for (Index k = start; k < end; ++k) {
            sum += eta_value_[k] * x[eta_index_[k]];
        }
        x[row_perm_[step]] -= sum;
    }
}

void SparseLU::applyFT(std::span<Real> x) const {
    // Product-form update: E^{-1} * x where E has column pivot_pos replaced by d.
    // E^{-1} * x: x'[p] = x[p] / d[p], x'[i] = x[i] - d[i] * x'[p] for i != p.
    for (Index u = 0; u < num_updates_; ++u) {
        Index pos = ft_pivot_pos_[u];
        Index start = ft_start_[u];
        Index end = ft_start_[u + 1];

        x[pos] /= ft_pivot_val_[u];
        Real xp = x[pos];
        for (Index k = start; k < end; ++k) {
            x[ft_index_[k]] -= ft_value_[k] * xp;
        }
    }
}

void SparseLU::applyFTTranspose(std::span<Real> x) const {
    // (E^{-1})^T * x in reverse order.
    // (E^{-1})^T: x'[i] stays for i != p, x'[p] = (x[p] - sum d[i]*x[i]) / d[p].
    for (Index u = num_updates_ - 1; u >= 0; --u) {
        Index pos = ft_pivot_pos_[u];
        Index start = ft_start_[u];
        Index end = ft_start_[u + 1];

        Real sum = 0.0;
        for (Index k = start; k < end; ++k) {
            sum += ft_value_[k] * x[ft_index_[k]];
        }
        x[pos] = (x[pos] - sum) / ft_pivot_val_[u];
    }
}

void SparseLU::solveU(std::span<Real> x) const {
    // Backward substitution: U is upper triangular in elimination order.
    // x is in elimination order.
    for (Index step = dim_ - 1; step >= 0; --step) {
        Real rhs = x[step];
        Index start = u_start_[step];
        Index end = u_start_[step + 1];
        for (Index k = start; k < end; ++k) {
            rhs -= u_val_[k] * x[u_col_[k]];
        }
        x[step] = rhs / u_diag_[step];
    }
}

void SparseLU::solveUTranspose(std::span<Real> x) const {
    // Forward substitution with U^T.
    for (Index step = 0; step < dim_; ++step) {
        x[step] /= u_diag_[step];
        Real val = x[step];
        if (val == 0.0) continue;
        Index start = u_start_[step];
        Index end = u_start_[step + 1];
        for (Index k = start; k < end; ++k) {
            x[u_col_[k]] -= u_val_[k] * val;
        }
    }
}

void SparseLU::ftran(std::span<Real> rhs) const {
    assert(static_cast<Index>(rhs.size()) == dim_);

    // B = P^T * L * U * Q^T, so B^{-1} = Q * U^{-1} * L^{-1} * P.
    // With product-form updates: B'^{-1} = E_n^{-1} * ... * E_1^{-1} * B^{-1}.
    // So x = E_n^{-1} * ... * E_1^{-1} * Q * U^{-1} * L^{-1} * P * b.

    // Step 1: w = P * b.
    std::vector<Real> work(dim_);
    for (Index step = 0; step < dim_; ++step) {
        work[step] = rhs[row_perm_[step]];
    }

    // Step 2: Apply L^{-1} (in elimination order, eta indices in original row space).
    for (Index step = 0; step < dim_; ++step) {
        Index start = eta_start_[step];
        Index end = eta_start_[step + 1];
        Real wk = work[step];
        if (wk == 0.0) continue;
        for (Index k = start; k < end; ++k) {
            work[row_perm_inv_[eta_index_[k]]] -= eta_value_[k] * wk;
        }
    }

    // Step 3: Solve U * y = work (backward substitution).
    solveU(work);

    // Step 4: x = Q * y (map from elimination order to basis positions).
    for (Index step = 0; step < dim_; ++step) {
        rhs[col_perm_[step]] = work[step];
    }

    // Step 5: Apply product-form update etas (in basis-position space).
    applyFT(rhs);
}

void SparseLU::btran(std::span<Real> rhs) const {
    assert(static_cast<Index>(rhs.size()) == dim_);

    // B'^{-T} = B^{-T} * E_1^{-T} * ... * E_n^{-T}.
    // y = B'^{-T} * c = P^T * L^{-T} * U^{-T} * Q^T * E_1^{-T} * ... * E_n^{-T} * c.
    // Wait: B^{-T} = (Q U^{-1} L^{-1} P)^T = P^T L^{-T} U^{-T} Q^T.
    // So y = P^T * L^{-T} * U^{-T} * Q^T * (E_1^{-T} * ... * E_n^{-T} * c).

    // Step 1: Apply FT etas transpose in reverse (in basis-position space).
    applyFTTranspose(rhs);

    // Step 2: w = Q^T * rhs. w[step] = rhs[col_perm[step]].
    std::vector<Real> work(dim_);
    for (Index step = 0; step < dim_; ++step) {
        work[step] = rhs[col_perm_[step]];
    }

    // Step 3: Solve U^T * z = w (forward substitution).
    solveUTranspose(work);

    // Step 4: Apply L^{-T} (in elimination order, reverse).
    for (Index step = dim_ - 1; step >= 0; --step) {
        Index start = eta_start_[step];
        Index end = eta_start_[step + 1];
        Real sum = 0.0;
        for (Index k = start; k < end; ++k) {
            sum += eta_value_[k] * work[row_perm_inv_[eta_index_[k]]];
        }
        work[step] -= sum;
    }

    // Step 5: y = P^T * work. y[row_perm[step]] = work[step].
    // Actually P^T: y[i] = work[row_perm_inv[i]].
    // But we want: for each step, rhs[row_perm[step]] = work[step].
    // That's P^{-1} = P^T since P is a permutation.
    for (Index step = 0; step < dim_; ++step) {
        rhs[row_perm_[step]] = work[step];
    }
}

// --------------------------------------------------------------------------
//  Forrest-Tomlin update
// --------------------------------------------------------------------------

void SparseLU::update(Index pivot_pos,
                      std::span<const Index> indices,
                      std::span<const Real> values) {
    assert(pivot_pos >= 0 && pivot_pos < dim_);

    // Product-form update: compute d = B^{-1} * a_q (FTRAN the new column).
    // Then store eta vector: E such that E * d = e_{pivot_pos}.
    // E^{-1} * x: x[pivot_pos] = (x[pivot_pos] - sum_{i!=p} d[i]*x[i]) / d[p]

    // Build the new column as a dense vector.
    std::vector<Real> d(dim_, 0.0);
    for (Index k = 0; k < static_cast<Index>(indices.size()); ++k) {
        if (indices[k] < dim_) {
            d[indices[k]] = values[k];
        }
    }

    // FTRAN through current factorization (including existing updates).
    ftran(d);

    // d[pivot_pos] is the pivot element.
    Real pivot_val = d[pivot_pos];
    if (std::abs(pivot_val) < kZeroTol) {
        pivot_val = (pivot_val >= 0.0) ? kZeroTol : -kZeroTol;
    }

    // Store eta: non-pivot entries of d.
    for (Index i = 0; i < dim_; ++i) {
        if (i == pivot_pos) continue;
        if (std::abs(d[i]) > kZeroTol) {
            ft_index_.push_back(i);
            ft_value_.push_back(d[i]);
            max_u_entry_ = std::max(max_u_entry_, std::abs(d[i]));
        }
    }
    ft_start_.push_back(static_cast<Index>(ft_index_.size()));
    ft_pivot_pos_.push_back(pivot_pos);
    ft_pivot_val_.push_back(pivot_val);

    max_u_entry_ = std::max(max_u_entry_, std::abs(pivot_val));

    ++num_updates_;
}

bool SparseLU::needsRefactorization() const {
    if (num_updates_ >= kMaxUpdates) return true;
    if (max_u_entry_ > kGrowthLimit) return true;
    return false;
}

}  // namespace mipx
