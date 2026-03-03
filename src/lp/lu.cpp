#include "mipx/lu.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "mipx/sparse_matrix.h"

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace mipx {

namespace {

inline void denseAxpySubtract(std::span<Real> x, const Real* d, Real alpha) {
    const Index n = static_cast<Index>(x.size());
#if defined(__AVX2__)
    const __m256d a = _mm256_set1_pd(alpha);
    Index i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d xv = _mm256_loadu_pd(x.data() + i);
        __m256d dv = _mm256_loadu_pd(d + i);
#if defined(__FMA__)
        xv = _mm256_fnmadd_pd(dv, a, xv);
#else
        xv = _mm256_sub_pd(xv, _mm256_mul_pd(dv, a));
#endif
        _mm256_storeu_pd(x.data() + i, xv);
    }
    for (; i < n; ++i) {
        x[static_cast<std::size_t>(i)] -= d[i] * alpha;
    }
#else
    for (Index i = 0; i < n; ++i) {
        x[static_cast<std::size_t>(i)] -= d[i] * alpha;
    }
#endif
}

inline Real denseDot(const Real* a, std::span<const Real> x) {
    const Index n = static_cast<Index>(x.size());
#if defined(__AVX2__)
    __m256d acc = _mm256_setzero_pd();
    Index i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d av = _mm256_loadu_pd(a + i);
        __m256d xv = _mm256_loadu_pd(x.data() + i);
#if defined(__FMA__)
        acc = _mm256_fmadd_pd(av, xv, acc);
#else
        acc = _mm256_add_pd(acc, _mm256_mul_pd(av, xv));
#endif
    }
    alignas(32) Real lanes[4];
    _mm256_store_pd(lanes, acc);
    Real sum = lanes[0] + lanes[1] + lanes[2] + lanes[3];
    for (; i < n; ++i) {
        sum += a[i] * x[static_cast<std::size_t>(i)];
    }
    return sum;
#else
    Real sum = 0.0;
    for (Index i = 0; i < n; ++i) {
        sum += a[i] * x[static_cast<std::size_t>(i)];
    }
    return sum;
#endif
}

}  // namespace

// --------------------------------------------------------------------------
//  Markowitz LU factorization
// --------------------------------------------------------------------------

void SparseLU::factorize(const SparseMatrix& matrix,
                         std::span<const Index> basis_cols) {
    static constexpr Index kFastPivotMinDim = 500;
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
    ft_pivot_inv_.clear();
    ft_is_dense_.clear();
    ft_dense_offset_.clear();
    ft_dense_value_.clear();
    ft_dense_nnz_ = 0;
    update_touched_.clear();

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
    const Index nnz_hint = std::max<Index>(dim_ * 4, matrix.numNonzeros());
    entries.reserve(static_cast<std::size_t>(std::max<Index>(0, nnz_hint)));

    // Head of linked list for each row/col (-1 = empty).
    std::vector<Index> row_head(dim_, -1);
    std::vector<Index> col_head(dim_, -1);
    std::vector<Index> row_count(dim_, 0);
    std::vector<Index> col_count(dim_, 0);

    std::vector<Index> active_rows(static_cast<std::size_t>(dim_));
    std::vector<Index> active_cols(static_cast<std::size_t>(dim_));
    std::vector<Index> row_pos(static_cast<std::size_t>(dim_));
    std::vector<Index> col_pos(static_cast<std::size_t>(dim_));
    std::iota(active_rows.begin(), active_rows.end(), 0);
    std::iota(active_cols.begin(), active_cols.end(), 0);
    for (Index i = 0; i < dim_; ++i) {
        row_pos[static_cast<std::size_t>(i)] = i;
        col_pos[static_cast<std::size_t>(i)] = i;
    }

    // HFactor-style count buckets: quickly target sparse rows/cols for pivot search.
    std::vector<std::vector<Index>> row_count_bucket(
        static_cast<std::size_t>(dim_ + 1));
    std::vector<std::vector<Index>> col_count_bucket(
        static_cast<std::size_t>(dim_ + 1));
    std::vector<Index> row_bucket_slot(dim_, -1);
    std::vector<Index> col_bucket_slot(dim_, -1);

    auto bucketAdd = [](std::vector<std::vector<Index>>& buckets,
                        std::vector<Index>& slot,
                        Index count,
                        Index id) {
        if (count <= 0) return;
        auto& b = buckets[static_cast<std::size_t>(count)];
        slot[static_cast<std::size_t>(id)] =
            static_cast<Index>(b.size());
        b.push_back(id);
    };

    auto bucketRemove = [](std::vector<std::vector<Index>>& buckets,
                           std::vector<Index>& slot,
                           Index count,
                           Index id) {
        if (count <= 0) return;
        auto& b = buckets[static_cast<std::size_t>(count)];
        const Index pos = slot[static_cast<std::size_t>(id)];
        if (pos < 0) return;
        const Index last = b.back();
        b[static_cast<std::size_t>(pos)] = last;
        slot[static_cast<std::size_t>(last)] = pos;
        b.pop_back();
        slot[static_cast<std::size_t>(id)] = -1;
    };

    auto deactivate = [](std::vector<Index>& active,
                         std::vector<Index>& pos,
                         Index idx) {
        Index p = pos[static_cast<std::size_t>(idx)];
        if (p < 0) return;
        Index last = active.back();
        active[static_cast<std::size_t>(p)] = last;
        pos[static_cast<std::size_t>(last)] = p;
        active.pop_back();
        pos[static_cast<std::size_t>(idx)] = -1;
    };

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

    for (Index i = 0; i < dim_; ++i) {
        bucketAdd(row_count_bucket, row_bucket_slot, row_count[i], i);
    }
    for (Index j = 0; j < dim_; ++j) {
        bucketAdd(col_count_bucket, col_bucket_slot, col_count[j], j);
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
    u_diag_inv_.resize(dim_);

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
        if (row_pos[static_cast<std::size_t>(e.row)] >= 0) {
            const Index old_count = row_count[e.row];
            bucketRemove(row_count_bucket, row_bucket_slot, old_count, e.row);
            --row_count[e.row];
            bucketAdd(row_count_bucket, row_bucket_slot, row_count[e.row], e.row);
        } else {
            --row_count[e.row];
        }

        // Remove from col list.
        if (e.prev_in_col >= 0) {
            entries[e.prev_in_col].next_in_col = e.next_in_col;
        } else {
            col_head[e.col] = e.next_in_col;
        }
        if (e.next_in_col >= 0) {
            entries[e.next_in_col].prev_in_col = e.prev_in_col;
        }
        if (col_pos[static_cast<std::size_t>(e.col)] >= 0) {
            const Index old_count = col_count[e.col];
            bucketRemove(col_count_bucket, col_bucket_slot, old_count, e.col);
            --col_count[e.col];
            bucketAdd(col_count_bucket, col_bucket_slot, col_count[e.col], e.col);
        } else {
            --col_count[e.col];
        }
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
        if (row_pos[static_cast<std::size_t>(row)] >= 0) {
            const Index old_count = row_count[row];
            bucketRemove(row_count_bucket, row_bucket_slot, old_count, row);
            ++row_count[row];
            bucketAdd(row_count_bucket, row_bucket_slot, row_count[row], row);
        } else {
            ++row_count[row];
        }

        if (col_head[col] >= 0) {
            entries[col_head[col]].prev_in_col = eidx;
        }
        col_head[col] = eidx;
        if (col_pos[static_cast<std::size_t>(col)] >= 0) {
            const Index old_count = col_count[col];
            bucketRemove(col_count_bucket, col_bucket_slot, old_count, col);
            ++col_count[col];
            bucketAdd(col_count_bucket, col_bucket_slot, col_count[col], col);
        } else {
            ++col_count[col];
        }
        return eidx;
    };

    // Work vector for gathering a column during elimination.
    std::vector<Real> work(dim_, 0.0);
    std::vector<Index> work_indices;
    work_indices.reserve(dim_);

    // Reusable row update scratch to avoid per-row heap allocations.
    std::vector<Index> row_entry_for_col(dim_, -1);
    std::vector<Index> row_touched_cols;
    row_touched_cols.reserve(dim_);

    std::vector<std::pair<Index, Real>> rows_to_update;
    rows_to_update.reserve(dim_);

    for (Index step = 0; step < dim_; ++step) {
        // ---- Markowitz pivot selection ----
        // Find pivot minimizing (row_nnz - 1) * (col_nnz - 1)
        // among entries with |a_ij| >= kPivotTol * max|a_*j|.

        Index best_entry = -1;
        long long best_markowitz = static_cast<long long>(dim_) * dim_;
        Real best_abs = 0.0;

        // First check singleton columns from count buckets (Markowitz = 0).
        for (Index j : col_count_bucket[1]) {
            if (col_pos[static_cast<std::size_t>(j)] < 0 || col_count[j] != 1) continue;
            const Index eidx = col_head[j];
            if (eidx < 0 || !entries[eidx].alive) continue;
            const Real absval = std::abs(entries[eidx].val);
            if (absval <= kZeroTol) continue;
            const long long m =
                static_cast<long long>(row_count[entries[eidx].row] - 1) *
                static_cast<long long>(col_count[j] - 1);
            if (m < best_markowitz || (m == best_markowitz && absval > best_abs)) {
                best_markowitz = m;
                best_entry = eidx;
                best_abs = absval;
            }
        }

        // Check singleton rows from count buckets.
        if (best_markowitz > 0) {
            for (Index i : row_count_bucket[1]) {
                if (row_pos[static_cast<std::size_t>(i)] < 0 || row_count[i] != 1) continue;
                const Index eidx = row_head[i];
                if (eidx < 0 || !entries[eidx].alive) continue;
                const Real absval = std::abs(entries[eidx].val);
                if (absval > kZeroTol) {
                    best_markowitz = 0;
                    best_entry = eidx;
                    best_abs = absval;
                    break;
                }
            }
        }

        // General search over increasing column-nnz buckets.
        if (best_markowitz > 0) {
            const bool use_fast_large_pivot = dim_ >= kFastPivotMinDim;
            for (Index nnz = 2; nnz <= dim_; ++nnz) {
                const auto& bucket = col_count_bucket[static_cast<std::size_t>(nnz)];
                if (bucket.empty()) continue;

                const long long col_factor = static_cast<long long>(nnz - 1);
                // If even the best possible row-factor (=1) is already worse, stop.
                if (col_factor > best_markowitz) break;

                for (Index j : bucket) {
                    if (col_pos[static_cast<std::size_t>(j)] < 0 || col_count[j] != nnz) continue;
                    if (use_fast_large_pivot) {
                        // Large-dimension fast path: evaluate one strongest candidate per column.
                        Index best_col_entry = -1;
                        Real col_max = 0.0;
                        for (Index eidx = col_head[j]; eidx >= 0;
                             eidx = entries[eidx].next_in_col) {
                            if (!entries[eidx].alive) continue;
                            const Real absval = std::abs(entries[eidx].val);
                            if (absval > col_max) {
                                col_max = absval;
                                best_col_entry = eidx;
                            }
                        }
                        if (best_col_entry >= 0 && col_max > kZeroTol) {
                            const Index ri = entries[best_col_entry].row;
                            const long long m =
                                static_cast<long long>(row_count[ri] - 1) * col_factor;
                            if (m < best_markowitz ||
                                (m == best_markowitz && col_max > best_abs)) {
                                best_markowitz = m;
                                best_entry = best_col_entry;
                                best_abs = col_max;
                            }
                        }
                        continue;
                    }

                    // Two-pass thresholded Markowitz scan for this column.
                    Real col_max = 0.0;
                    for (Index eidx = col_head[j]; eidx >= 0;
                         eidx = entries[eidx].next_in_col) {
                        if (entries[eidx].alive) {
                            col_max = std::max(col_max, std::abs(entries[eidx].val));
                        }
                    }
                    const Real threshold = kPivotTol * col_max;
                    for (Index eidx = col_head[j]; eidx >= 0;
                         eidx = entries[eidx].next_in_col) {
                        if (!entries[eidx].alive) continue;
                        const Real absval = std::abs(entries[eidx].val);
                        if (absval < threshold) continue;

                        const Index ri = entries[eidx].row;
                        const long long row_factor = static_cast<long long>(row_count[ri] - 1);
                        if (row_factor > (best_markowitz / col_factor)) continue;
                        const long long m = row_factor * col_factor;
                        if (m < best_markowitz ||
                            (m == best_markowitz && absval > best_abs)) {
                            best_markowitz = m;
                            best_entry = eidx;
                            best_abs = absval;
                        }
                    }
                }

                // Early exit if we found near-singleton quality pivot.
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
        u_diag_inv_[step] = 1.0 / pivot_val;
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
        rows_to_update.clear();
        for (Index eidx = col_head[pivot_col]; eidx >= 0;
             eidx = entries[eidx].next_in_col) {
            if (!entries[eidx].alive) continue;
            if (entries[eidx].row == pivot_row) continue;
            rows_to_update.emplace_back(entries[eidx].row,
                                        entries[eidx].val / pivot_val);
        }

        // Store L eta vector.
        for (auto& [ri, mult] : rows_to_update) {
            eta_index_.push_back(ri);
            eta_value_.push_back(mult);
        }
        eta_start_.push_back(static_cast<Index>(eta_index_.size()));

        // Update rows.
        for (auto& [ri, mult] : rows_to_update) {

            // Build a temporary col->entry index map for this row, reused across rows.
            row_touched_cols.clear();
            for (Index eidx = row_head[ri]; eidx >= 0;
                 eidx = entries[eidx].next_in_row) {
                if (!entries[eidx].alive || entries[eidx].col == pivot_col) continue;
                Index c = entries[eidx].col;
                row_entry_for_col[c] = eidx;
                row_touched_cols.push_back(c);
            }

            for (Index c : work_indices) {
                Real update_val = mult * work[c];
                Index row_entry = row_entry_for_col[c];
                if (row_entry >= 0) {
                    Real new_val = entries[row_entry].val - update_val;
                    if (std::abs(new_val) < kZeroTol) {
                        // Remove entry.
                        removeEntry(row_entry);
                    } else {
                        entries[row_entry].val = new_val;
                    }
                } else {
                    Real new_val = -update_val;
                    if (std::abs(new_val) >= kZeroTol) {
                        addEntry(ri, c, new_val);
                    }
                }
            }

            for (Index c : row_touched_cols) {
                row_entry_for_col[c] = -1;
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

        if (row_count[pivot_row] > 0) {
            bucketRemove(row_count_bucket, row_bucket_slot,
                         row_count[pivot_row], pivot_row);
        }
        if (col_count[pivot_col] > 0) {
            bucketRemove(col_count_bucket, col_bucket_slot,
                         col_count[pivot_col], pivot_col);
        }
        deactivate(active_rows, row_pos, pivot_row);
        deactivate(active_cols, col_pos, pivot_col);

        // Clean up work vector.
        for (Index c : work_indices) {
            work[c] = 0.0;
        }
    }

    // Remap U column indices from original to elimination order.
    for (auto& c : u_col_) {
        c = col_perm_inv_[c];
    }

    // Build elimination-order eta adjacency for hyper-sparse solves.
    eta_target_.resize(eta_index_.size());
    eta_rev_start_.assign(static_cast<std::size_t>(dim_ + 1), 0);
    for (Index step = 0; step < dim_; ++step) {
        for (Index k = eta_start_[step]; k < eta_start_[step + 1]; ++k) {
            Index target = row_perm_inv_[eta_index_[k]];
            eta_target_[k] = target;
            ++eta_rev_start_[static_cast<std::size_t>(target + 1)];
        }
    }
    for (Index i = 0; i < dim_; ++i) {
        eta_rev_start_[static_cast<std::size_t>(i + 1)] +=
            eta_rev_start_[static_cast<std::size_t>(i)];
    }
    eta_rev_src_.assign(eta_index_.size(), 0);
    std::vector<Index> rev_cursor = eta_rev_start_;
    for (Index step = 0; step < dim_; ++step) {
        for (Index k = eta_start_[step]; k < eta_start_[step + 1]; ++k) {
            Index target = eta_target_[k];
            Index pos = rev_cursor[static_cast<std::size_t>(target)]++;
            eta_rev_src_[static_cast<std::size_t>(pos)] = step;
        }
    }

    // Count work: total nonzeros stored in L and U.
    work_.count(static_cast<uint64_t>(eta_index_.size()) +
                static_cast<uint64_t>(u_col_.size()));
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
        x[pos] *= ft_pivot_inv_[u];
        Real xp = x[pos];
        if (xp == 0.0) continue;
        if (ft_is_dense_[u] != 0) {
            Index dense_off = ft_dense_offset_[u];
            const Real* dense = ft_dense_value_.data() + dense_off;
            denseAxpySubtract(x, dense, xp);
            continue;
        }
        const Index start = ft_start_[u];
        const Index end = ft_start_[u + 1];
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
        Real sum = 0.0;
        if (ft_is_dense_[u] != 0) {
            Index dense_off = ft_dense_offset_[u];
            const Real* dense = ft_dense_value_.data() + dense_off;
            sum = denseDot(dense, x);
        } else {
            const Index start = ft_start_[u];
            const Index end = ft_start_[u + 1];
            for (Index k = start; k < end; ++k) {
                sum += ft_value_[k] * x[ft_index_[k]];
            }
        }
        x[pos] = (x[pos] - sum) * ft_pivot_inv_[u];
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
        x[step] = rhs * u_diag_inv_[step];
    }
}

void SparseLU::solveUTranspose(std::span<Real> x) const {
    // Forward substitution with U^T.
    for (Index step = 0; step < dim_; ++step) {
        x[step] *= u_diag_inv_[step];
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

    // Count work: L nnz + U nnz + FT nnz + permutation overhead.
    work_.count(static_cast<uint64_t>(eta_index_.size()) +
                static_cast<uint64_t>(u_col_.size()) +
                static_cast<uint64_t>(ft_index_.size()) +
                ft_dense_nnz_ +
                static_cast<uint64_t>(dim_) * 2);

    // B = P^T * L * U * Q^T, so B^{-1} = Q * U^{-1} * L^{-1} * P.
    // With product-form updates: B'^{-1} = E_n^{-1} * ... * E_1^{-1} * B^{-1}.
    // So x = E_n^{-1} * ... * E_1^{-1} * Q * U^{-1} * L^{-1} * P * b.

    // Step 1: w = P * b.
    if (static_cast<Index>(solve_work_.size()) < dim_) {
        solve_work_.resize(dim_);
    }
    std::span<Real> work(solve_work_.data(), static_cast<std::size_t>(dim_));
    Index work_nnz = 0;
    for (Index step = 0; step < dim_; ++step) {
        work[step] = rhs[row_perm_[step]];
        if (std::abs(work[step]) > kZeroTol) {
            ++work_nnz;
        }
    }

    // Step 2: Apply L^{-1}. Use a hyper-sparse reach solve when rhs is sparse.
    bool use_hypersparse_l =
        dim_ >= kHyperSparseMinDim &&
        !eta_index_.empty() &&
        work_nnz <= static_cast<Index>(kHyperSparseMaxDensity * static_cast<Real>(dim_));

    if (use_hypersparse_l) {
        if (static_cast<Index>(sparse_epoch_.size()) < dim_) {
            sparse_epoch_.resize(dim_, 0U);
        }
        ++sparse_epoch_id_;
        if (sparse_epoch_id_ == 0U) {
            std::fill(sparse_epoch_.begin(), sparse_epoch_.begin() + dim_, 0U);
            sparse_epoch_id_ = 1U;
        }
        sparse_steps_.clear();
        sparse_steps_.reserve(static_cast<std::size_t>(std::max<Index>(work_nnz, 8)));
        Index min_step = dim_;
        Index max_step = -1;

        for (Index step = 0; step < dim_; ++step) {
            if (std::abs(work[step]) > kZeroTol) {
                sparse_epoch_[static_cast<std::size_t>(step)] = sparse_epoch_id_;
                sparse_steps_.push_back(step);
                min_step = std::min(min_step, step);
                max_step = std::max(max_step, step);
            }
        }

        // Reachability over eta graph: if step is active, all eta targets can be affected.
        for (std::size_t idx = 0; idx < sparse_steps_.size(); ++idx) {
            Index step = sparse_steps_[idx];
            for (Index k = eta_start_[step]; k < eta_start_[step + 1]; ++k) {
                Index target = eta_target_[k];
                if (sparse_epoch_[static_cast<std::size_t>(target)] != sparse_epoch_id_) {
                    sparse_epoch_[static_cast<std::size_t>(target)] = sparse_epoch_id_;
                    sparse_steps_.push_back(target);
                    min_step = std::min(min_step, target);
                    max_step = std::max(max_step, target);
                }
            }
        }

        if (max_step >= min_step) {
            const Index range_len = max_step - min_step + 1;
            const Index reach_nnz = static_cast<Index>(sparse_steps_.size());
            if (range_len <= 4 * reach_nnz) {
                for (Index step = min_step; step <= max_step; ++step) {
                    if (sparse_epoch_[static_cast<std::size_t>(step)] != sparse_epoch_id_) {
                        continue;
                    }
                    Real wk = work[step];
                    if (std::abs(wk) <= kZeroTol) continue;
                    for (Index k = eta_start_[step]; k < eta_start_[step + 1]; ++k) {
                        work[eta_target_[k]] -= eta_value_[k] * wk;
                    }
                }
            } else {
                std::sort(sparse_steps_.begin(), sparse_steps_.end());
                for (Index step : sparse_steps_) {
                    Real wk = work[step];
                    if (std::abs(wk) <= kZeroTol) continue;
                    for (Index k = eta_start_[step]; k < eta_start_[step + 1]; ++k) {
                        work[eta_target_[k]] -= eta_value_[k] * wk;
                    }
                }
            }
        }
    } else {
        for (Index step = 0; step < dim_; ++step) {
            Real wk = work[step];
            if (wk == 0.0) continue;
            for (Index k = eta_start_[step]; k < eta_start_[step + 1]; ++k) {
                work[eta_target_[k]] -= eta_value_[k] * wk;
            }
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

    // Count work: same structure as ftran.
    work_.count(static_cast<uint64_t>(eta_index_.size()) +
                static_cast<uint64_t>(u_col_.size()) +
                static_cast<uint64_t>(ft_index_.size()) +
                ft_dense_nnz_ +
                static_cast<uint64_t>(dim_) * 2);

    // B'^{-T} = B^{-T} * E_1^{-T} * ... * E_n^{-T}.
    // y = B'^{-T} * c = P^T * L^{-T} * U^{-T} * Q^T * E_1^{-T} * ... * E_n^{-T} * c.
    // Wait: B^{-T} = (Q U^{-1} L^{-1} P)^T = P^T L^{-T} U^{-T} Q^T.
    // So y = P^T * L^{-T} * U^{-T} * Q^T * (E_1^{-T} * ... * E_n^{-T} * c).

    // Step 1: Apply FT etas transpose in reverse (in basis-position space).
    applyFTTranspose(rhs);

    // Step 2: w = Q^T * rhs. w[step] = rhs[col_perm[step]].
    if (static_cast<Index>(solve_work_.size()) < dim_) {
        solve_work_.resize(dim_);
    }
    std::span<Real> work(solve_work_.data(), static_cast<std::size_t>(dim_));
    for (Index step = 0; step < dim_; ++step) {
        work[step] = rhs[col_perm_[step]];
    }

    // Step 3: Solve U^T * z = w (forward substitution).
    solveUTranspose(work);

    // Step 4: Apply L^{-T}. Use reverse-reach hyper-sparse solve when possible.
    Index work_nnz = 0;
    for (Index step = 0; step < dim_; ++step) {
        if (std::abs(work[step]) > kZeroTol) {
            ++work_nnz;
        }
    }
    bool use_hypersparse_lt =
        dim_ >= kHyperSparseMinDim &&
        !eta_index_.empty() &&
        work_nnz <= static_cast<Index>(kHyperSparseMaxDensity * static_cast<Real>(dim_));

    if (use_hypersparse_lt) {
        if (static_cast<Index>(sparse_epoch_.size()) < dim_) {
            sparse_epoch_.resize(dim_, 0U);
        }
        ++sparse_epoch_id_;
        if (sparse_epoch_id_ == 0U) {
            std::fill(sparse_epoch_.begin(), sparse_epoch_.begin() + dim_, 0U);
            sparse_epoch_id_ = 1U;
        }
        sparse_steps_.clear();
        sparse_steps_.reserve(static_cast<std::size_t>(std::max<Index>(work_nnz, 8)));
        Index min_step = dim_;
        Index max_step = -1;

        for (Index step = 0; step < dim_; ++step) {
            if (std::abs(work[step]) > kZeroTol) {
                sparse_epoch_[static_cast<std::size_t>(step)] = sparse_epoch_id_;
                sparse_steps_.push_back(step);
                min_step = std::min(min_step, step);
                max_step = std::max(max_step, step);
            }
        }

        // Reverse reachability: any source feeding an active target may become active.
        for (std::size_t idx = 0; idx < sparse_steps_.size(); ++idx) {
            Index target = sparse_steps_[idx];
            Index rs = eta_rev_start_[target];
            Index re = eta_rev_start_[target + 1];
            for (Index p = rs; p < re; ++p) {
                Index src = eta_rev_src_[p];
                if (sparse_epoch_[static_cast<std::size_t>(src)] != sparse_epoch_id_) {
                    sparse_epoch_[static_cast<std::size_t>(src)] = sparse_epoch_id_;
                    sparse_steps_.push_back(src);
                    min_step = std::min(min_step, src);
                    max_step = std::max(max_step, src);
                }
            }
        }

        if (max_step >= min_step) {
            const Index range_len = max_step - min_step + 1;
            const Index reach_nnz = static_cast<Index>(sparse_steps_.size());
            if (range_len <= 4 * reach_nnz) {
                for (Index step = max_step; step >= min_step; --step) {
                    if (sparse_epoch_[static_cast<std::size_t>(step)] != sparse_epoch_id_) {
                        if (step == min_step) break;
                        continue;
                    }
                    Real sum = 0.0;
                    for (Index k = eta_start_[step]; k < eta_start_[step + 1]; ++k) {
                        sum += eta_value_[k] * work[eta_target_[k]];
                    }
                    work[step] -= sum;
                    if (step == min_step) break;
                }
            } else {
                std::sort(sparse_steps_.begin(), sparse_steps_.end());
                for (auto it = sparse_steps_.rbegin(); it != sparse_steps_.rend(); ++it) {
                    const Index step = *it;
                    Real sum = 0.0;
                    for (Index k = eta_start_[step]; k < eta_start_[step + 1]; ++k) {
                        sum += eta_value_[k] * work[eta_target_[k]];
                    }
                    work[step] -= sum;
                }
            }
        }
    } else {
        for (Index step = dim_ - 1; step >= 0; --step) {
            Real sum = 0.0;
            for (Index k = eta_start_[step]; k < eta_start_[step + 1]; ++k) {
                sum += eta_value_[k] * work[eta_target_[k]];
            }
            work[step] -= sum;
        }
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
    if (static_cast<Index>(update_work_.size()) < dim_) {
        update_work_.resize(dim_);
    }
    for (Index idx : update_touched_) {
        update_work_[static_cast<std::size_t>(idx)] = 0.0;
    }
    std::span<Real> d(update_work_.data(), static_cast<std::size_t>(dim_));
    for (Index k = 0; k < static_cast<Index>(indices.size()); ++k) {
        if (indices[k] >= 0 && indices[k] < dim_) {
            d[indices[k]] = values[k];
        }
    }

    // FTRAN through current factorization (including existing updates).
    ftran(d);

    updateFromFtranColumn(pivot_pos, d);
}

void SparseLU::updateFromFtranColumn(
    Index pivot_pos, std::span<const Real> transformed_col) {
    assert(pivot_pos >= 0 && pivot_pos < dim_);
    assert(static_cast<Index>(transformed_col.size()) == dim_);

    // d[pivot_pos] is the pivot element.
    Real pivot_val = transformed_col[pivot_pos];
    if (std::abs(pivot_val) < kZeroTol) {
        pivot_val = (pivot_val >= 0.0) ? kZeroTol : -kZeroTol;
    }
    const Real drop_tol = ft_drop_tol_;
    const bool track_update_touched =
        !update_work_.empty() && transformed_col.data() == update_work_.data();
    if (track_update_touched) {
        update_touched_.clear();
    }

    Index update_nnz = 0;
    for (Index i = 0; i < dim_; ++i) {
        const Real v = transformed_col[i];
        if (track_update_touched && v != 0.0) {
            update_touched_.push_back(i);
        }
        if (i == pivot_pos) continue;
        if (std::abs(v) > drop_tol) {
            ++update_nnz;
        }
    }

    const bool use_dense_store =
        dim_ >= kFtDenseMinDim &&
        update_nnz >= static_cast<Index>(kFtDenseThreshold * static_cast<Real>(dim_));

    if (use_dense_store) {
        Index dense_off = static_cast<Index>(ft_dense_value_.size());
        ft_dense_offset_.push_back(dense_off);
        ft_is_dense_.push_back(1);
        ft_dense_value_.resize(static_cast<std::size_t>(dense_off + dim_), 0.0);
        for (Index i = 0; i < dim_; ++i) {
            if (i == pivot_pos) continue;
            const Real v =
                (std::abs(transformed_col[i]) > drop_tol) ? transformed_col[i] : 0.0;
            ft_dense_value_[static_cast<std::size_t>(dense_off + i)] = v;
            if (v != 0.0) {
                max_u_entry_ = std::max(max_u_entry_, std::abs(v));
            }
        }
        ft_dense_nnz_ += static_cast<uint64_t>(update_nnz);
    } else {
        ft_dense_offset_.push_back(-1);
        ft_is_dense_.push_back(0);
        // Store eta: non-pivot entries of d.
        for (Index i = 0; i < dim_; ++i) {
            if (i == pivot_pos) continue;
            if (std::abs(transformed_col[i]) > drop_tol) {
                ft_index_.push_back(i);
                ft_value_.push_back(transformed_col[i]);
                max_u_entry_ = std::max(max_u_entry_, std::abs(transformed_col[i]));
            }
        }
    }

    ft_start_.push_back(static_cast<Index>(ft_index_.size()));
    ft_pivot_pos_.push_back(pivot_pos);
    ft_pivot_val_.push_back(pivot_val);
    ft_pivot_inv_.push_back(1.0 / pivot_val);

    max_u_entry_ = std::max(max_u_entry_, std::abs(pivot_val));

    // Count work: scanning d vector for eta storage.
    work_.count(static_cast<uint64_t>(dim_));

    ++num_updates_;
}

bool SparseLU::needsRefactorization() const {
    if (num_updates_ >= max_updates_) return true;
    if (max_u_entry_ > kGrowthLimit) return true;
    return false;
}

}  // namespace mipx
