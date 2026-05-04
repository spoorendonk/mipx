#include "mipx/lu.h"

#include "mipx/sparse_matrix.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

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
//  BTF detection via maximum matching + Tarjan's SCC
// --------------------------------------------------------------------------

Index SparseLU::detectBTF(Index dim, const SparseMatrix& matrix, std::span<const Index> basis_cols,
                          std::vector<Index>& btf_row_perm, std::vector<Index>& btf_col_perm,
                          std::vector<Index>& btf_block_start) {
    // Minimum dimension to attempt BTF detection. BTF builds a matching +
    // Tarjan SCC over the basis pattern; for small bases the fixed cost
    // (allocations + scans) outweighs any savings, since a single-block LU
    // factorization is already cheap on small matrices.
    static constexpr Index kBtfMinDim = 100;

    if (dim < kBtfMinDim) {
        return 0;
    }

    const auto n = static_cast<std::size_t>(dim);

    // Step 1: Build column-to-row adjacency for the basis submatrix.
    // col_adj_start[j]..col_adj_start[j+1] gives the row indices in column j.
    std::vector<Index> col_adj_start(n + 1, 0);
    std::vector<Index> col_adj_rows;
    col_adj_rows.reserve(n * 4);  // heuristic

    for (Index j = 0; j < dim; ++j) {
        auto colview = matrix.col(basis_cols[j]);
        for (Index k = 0; k < colview.size(); ++k) {
            Index row = colview.indices[k];
            if (row < dim && std::abs(colview.values[k]) > kZeroTol) {
                col_adj_rows.push_back(row);
            }
        }
        col_adj_start[static_cast<std::size_t>(j + 1)] = static_cast<Index>(col_adj_rows.size());
    }

    // Skip BTF for matrices where the matching work would be excessive.
    // The augmenting-path matching is O(V*E) worst case, and for general LP
    // bases (which are typically not block-structured), it adds overhead
    // without benefit.
    const auto total_nnz = static_cast<Index>(col_adj_rows.size());
    if (static_cast<long long>(dim) * static_cast<long long>(total_nnz) > 2000000LL) {
        return 0;
    }

    // Step 2: Maximum matching using augmenting paths (Koenig's algorithm).
    // match_row[i] = column matched to row i (-1 if unmatched).
    std::vector<Index> match_row(n, -1);
    std::vector<Index> visited(n, -1);  // epoch-based visited marking

    // Try to find an augmenting path from column j.
    // Returns true if an augmenting path was found.
    auto try_augment = [&](Index j, Index epoch, auto& self) -> bool {
        for (Index k = col_adj_start[j]; k < col_adj_start[j + 1]; ++k) {
            Index row = col_adj_rows[static_cast<std::size_t>(k)];
            if (visited[static_cast<std::size_t>(row)] == epoch) {
                continue;
            }
            visited[static_cast<std::size_t>(row)] = epoch;
            if (match_row[static_cast<std::size_t>(row)] < 0 ||
                self(match_row[static_cast<std::size_t>(row)], epoch, self)) {
                match_row[static_cast<std::size_t>(row)] = j;
                return true;
            }
        }
        return false;
    };

    Index matching_size = 0;
    for (Index j = 0; j < dim; ++j) {
        if (try_augment(j, j, try_augment)) {
            ++matching_size;
        }
    }

    // If we don't have a perfect matching, BTF is not applicable.
    if (matching_size < dim) {
        return 0;
    }

    // Step 3: Build the directed graph for Tarjan's SCC.
    // After matching, the directed graph has an edge from SCC node i to SCC node j
    // if column match_row^{-1}[i] has a nonzero in row j (and j != i).
    // Here "SCC node i" corresponds to the matched pair (row i, col match_row[i]).
    // The directed edges are: for each column j = match_row[i], for each row r in
    // column j where r != i, we have edge i -> r.

    // Tarjan's SCC algorithm.
    std::vector<Index> scc_index(n, -1);
    std::vector<Index> scc_lowlink(n, -1);
    std::vector<bool> scc_on_stack(n, false);
    std::vector<Index> scc_stack;
    scc_stack.reserve(n);
    Index scc_counter = 0;
    Index num_sccs = 0;
    std::vector<Index> scc_id(n, -1);  // which SCC each node belongs to

    auto tarjan_visit = [&](Index node, auto& self) -> void {
        scc_index[static_cast<std::size_t>(node)] = scc_counter;
        scc_lowlink[static_cast<std::size_t>(node)] = scc_counter;
        ++scc_counter;
        scc_stack.push_back(node);
        scc_on_stack[static_cast<std::size_t>(node)] = true;

        // Edges from node: match_row[i] = column j means row i is matched to column j.
        // For each row r (r != node) that appears in column match_row[node],
        // we have an edge node -> r.
        Index col = match_row[static_cast<std::size_t>(node)];
        for (Index k = col_adj_start[col]; k < col_adj_start[col + 1]; ++k) {
            Index target = col_adj_rows[static_cast<std::size_t>(k)];
            if (target == node) {
                continue;
            }
            if (scc_index[static_cast<std::size_t>(target)] < 0) {
                self(target, self);
                scc_lowlink[static_cast<std::size_t>(node)] =
                    std::min(scc_lowlink[static_cast<std::size_t>(node)],
                             scc_lowlink[static_cast<std::size_t>(target)]);
            } else if (scc_on_stack[static_cast<std::size_t>(target)]) {
                scc_lowlink[static_cast<std::size_t>(node)] =
                    std::min(scc_lowlink[static_cast<std::size_t>(node)],
                             scc_index[static_cast<std::size_t>(target)]);
            }
        }

        // If node is a root of an SCC, pop the SCC from the stack.
        if (scc_lowlink[static_cast<std::size_t>(node)] ==
            scc_index[static_cast<std::size_t>(node)]) {
            Index scc = num_sccs++;
            while (true) {
                Index w = scc_stack.back();
                scc_stack.pop_back();
                scc_on_stack[static_cast<std::size_t>(w)] = false;
                scc_id[static_cast<std::size_t>(w)] = scc;
                if (w == node) {
                    break;
                }
            }
        }
    };

    // For large matrices, use an iterative Tarjan to avoid stack overflow.
    // Recursive lambda frames are ~100+ bytes; keep limit conservative.
    if (dim < 2000) {
        for (Index i = 0; i < dim; ++i) {
            if (scc_index[static_cast<std::size_t>(i)] < 0) {
                tarjan_visit(i, tarjan_visit);
            }
        }
    } else {
        // Iterative Tarjan's algorithm for large matrices.
        struct Frame {
            Index node;
            Index k;  // current edge index in col adjacency
            Index k_end;
        };
        std::vector<Frame> call_stack;
        call_stack.reserve(n);

        for (Index start = 0; start < dim; ++start) {
            if (scc_index[static_cast<std::size_t>(start)] >= 0) {
                continue;
            }

            Index col = match_row[static_cast<std::size_t>(start)];
            scc_index[static_cast<std::size_t>(start)] = scc_counter;
            scc_lowlink[static_cast<std::size_t>(start)] = scc_counter;
            ++scc_counter;
            scc_stack.push_back(start);
            scc_on_stack[static_cast<std::size_t>(start)] = true;

            call_stack.push_back({start, col_adj_start[col], col_adj_start[col + 1]});

            while (!call_stack.empty()) {
                auto& frame = call_stack.back();
                bool pushed_child = false;

                while (frame.k < frame.k_end) {
                    Index target = col_adj_rows[static_cast<std::size_t>(frame.k)];
                    ++frame.k;

                    if (target == frame.node) {
                        continue;
                    }

                    if (scc_index[static_cast<std::size_t>(target)] < 0) {
                        // Push new frame for unvisited target.
                        scc_index[static_cast<std::size_t>(target)] = scc_counter;
                        scc_lowlink[static_cast<std::size_t>(target)] = scc_counter;
                        ++scc_counter;
                        scc_stack.push_back(target);
                        scc_on_stack[static_cast<std::size_t>(target)] = true;

                        Index tcol = match_row[static_cast<std::size_t>(target)];
                        call_stack.push_back(
                            {target, col_adj_start[tcol], col_adj_start[tcol + 1]});
                        pushed_child = true;
                        break;
                    } else if (scc_on_stack[static_cast<std::size_t>(target)]) {
                        scc_lowlink[static_cast<std::size_t>(frame.node)] =
                            std::min(scc_lowlink[static_cast<std::size_t>(frame.node)],
                                     scc_index[static_cast<std::size_t>(target)]);
                    }
                }

                if (pushed_child) {
                    continue;
                }

                // All edges processed for this node.
                if (scc_lowlink[static_cast<std::size_t>(frame.node)] ==
                    scc_index[static_cast<std::size_t>(frame.node)]) {
                    Index scc = num_sccs++;
                    while (true) {
                        Index w = scc_stack.back();
                        scc_stack.pop_back();
                        scc_on_stack[static_cast<std::size_t>(w)] = false;
                        scc_id[static_cast<std::size_t>(w)] = scc;
                        if (w == frame.node) {
                            break;
                        }
                    }
                }

                Index finished_node = frame.node;
                call_stack.pop_back();

                // Update parent's lowlink.
                if (!call_stack.empty()) {
                    auto& parent = call_stack.back();
                    scc_lowlink[static_cast<std::size_t>(parent.node)] =
                        std::min(scc_lowlink[static_cast<std::size_t>(parent.node)],
                                 scc_lowlink[static_cast<std::size_t>(finished_node)]);
                }
            }
        }
    }

    // If there's only one SCC, BTF gives no benefit.
    if (num_sccs <= 1) {
        return 0;
    }

    // Step 4: Build the BTF ordering.
    // Tarjan's algorithm yields SCCs in reverse topological order.
    // We want blocks ordered so that block 0 comes first (earliest in elimination).
    // SCC id `num_sccs - 1` is the first in topological order.

    // Count nodes per SCC and build block starts.
    std::vector<Index> scc_size(static_cast<std::size_t>(num_sccs), 0);
    for (Index i = 0; i < dim; ++i) {
        ++scc_size[static_cast<std::size_t>(scc_id[i])];
    }

    // Reverse the SCC ordering so block 0 is topologically first.
    // Tarjan gives SCCs in reverse topological order: scc_id 0 is the LAST SCC found,
    // which is topologically LAST. We want it reversed.
    btf_block_start.resize(static_cast<std::size_t>(num_sccs + 1));
    btf_block_start[0] = 0;
    for (Index b = 0; b < num_sccs; ++b) {
        // Block b in our ordering = SCC (num_sccs - 1 - b) in Tarjan's ordering.
        btf_block_start[static_cast<std::size_t>(b + 1)] =
            btf_block_start[static_cast<std::size_t>(b)] +
            scc_size[static_cast<std::size_t>(num_sccs - 1 - b)];
    }

    // Place each row into its block position.
    // Remap SCC ids: new_scc = num_sccs - 1 - old_scc.
    std::vector<Index> block_cursor = btf_block_start;

    btf_row_perm.resize(n);
    btf_col_perm.resize(n);

    for (Index i = 0; i < dim; ++i) {
        Index block = num_sccs - 1 - scc_id[static_cast<std::size_t>(i)];
        Index pos = block_cursor[static_cast<std::size_t>(block)]++;
        btf_row_perm[static_cast<std::size_t>(pos)] = i;
        // Column matched to row i goes to the same position.
        btf_col_perm[static_cast<std::size_t>(pos)] = match_row[static_cast<std::size_t>(i)];
    }

    return num_sccs;
}

// --------------------------------------------------------------------------
//  Markowitz LU factorization
// --------------------------------------------------------------------------

void SparseLU::factorize(const SparseMatrix& matrix, std::span<const Index> basis_cols) {
    static constexpr Index kFastPivotMinDim = 500;
    // Maximum columns to inspect per nnz bucket in the general Markowitz search.
    static constexpr Index kMaxColsPerBucket = 8;
    // Accept a pivot early if its Markowitz cost is at or below this fraction of dim.
    static constexpr Real kEarlyAcceptFraction = 0.05;

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
    // Reset update workspace so the first update after a refactorization
    // cannot read stale values when using sparse touch clearing.
    update_work_.clear();

    // --- BTF pre-ordering ---
    // Detect block upper-triangular form to improve pivot selection.
    // When BTF detects multiple blocks, we reorder rows and columns so the
    // Markowitz code processes diagonal blocks first with less fill-in.
    std::vector<Index> btf_row_perm;
    std::vector<Index> btf_col_perm;
    std::vector<Index> btf_block_start;
    Index num_blocks =
        detectBTF(dim_, matrix, basis_cols, btf_row_perm, btf_col_perm, btf_block_start);

    // Build the effective row and column mapping.
    // If BTF detected blocks, we use the BTF permutation to reorder the
    // active submatrix. btf_row_perm[p] = original row at position p,
    // btf_col_perm[p] = which basis column index goes to position p.
    //
    // row_map[original_row] = internal row index in the active submatrix
    // col_order[internal_col] = basis_cols index for that column
    // row_to_orig[internal_row] = original row index
    // col_to_orig[internal_col] = original basis position index
    static thread_local std::vector<Index> tl_row_map;
    static thread_local std::vector<Index> tl_col_order;
    static thread_local std::vector<Index> tl_row_to_orig;
    static thread_local std::vector<Index> tl_col_to_orig;
    auto& row_map = tl_row_map;
    auto& col_order = tl_col_order;
    auto& row_to_orig = tl_row_to_orig;
    auto& col_to_orig = tl_col_to_orig;
    row_map.resize(static_cast<std::size_t>(dim_));
    col_order.resize(static_cast<std::size_t>(dim_));
    row_to_orig.resize(static_cast<std::size_t>(dim_));
    col_to_orig.resize(static_cast<std::size_t>(dim_));

    if (num_blocks > 1) {
        // Build inverse row mapping: row_map[orig_row] = BTF position.
        for (Index p = 0; p < dim_; ++p) {
            row_map[static_cast<std::size_t>(btf_row_perm[p])] = p;
            row_to_orig[static_cast<std::size_t>(p)] = btf_row_perm[p];
        }
        // Column order: internal col p uses basis column btf_col_perm[p].
        for (Index p = 0; p < dim_; ++p) {
            col_order[static_cast<std::size_t>(p)] = btf_col_perm[p];
            col_to_orig[static_cast<std::size_t>(p)] = btf_col_perm[p];
        }
    } else {
        // Identity mapping: no BTF reordering.
        for (Index i = 0; i < dim_; ++i) {
            row_map[static_cast<std::size_t>(i)] = i;
            col_order[static_cast<std::size_t>(i)] = i;
            row_to_orig[static_cast<std::size_t>(i)] = i;
            col_to_orig[static_cast<std::size_t>(i)] = i;
        }
    }

    // Build dense-ish active submatrix from selected columns.
    // active[i][j] = value at (original_row i, basis_position j).
    // We use a flat vector + linked-list structures for rows/cols.

    // Collect nonzeros into column-oriented lists.
    // For the active submatrix we track: row indices, column indices,
    // values, and counts for Markowitz.

    const std::size_t dim_hint = static_cast<std::size_t>(std::max<Index>(0, dim_));
    std::size_t basis_nnz_hint = 0;
    for (Index j = 0; j < dim_; ++j) {
        basis_nnz_hint += static_cast<std::size_t>(matrix.col(basis_cols[j]).size());
    }
    // Estimate fill-in: use a multiplier based on average column density.
    // Denser matrices produce more fill; scale reservation accordingly.
    const Real avg_col_nnz =
        dim_hint > 0 ? static_cast<Real>(basis_nnz_hint) / static_cast<Real>(dim_hint) : 1.0;
    // Fill multiplier: 2x for very sparse, up to 5x for denser matrices.
    const std::size_t fill_mult = static_cast<std::size_t>(std::clamp(avg_col_nnz, 2.0, 5.0));
    const std::size_t reserve_nnz = std::max(dim_hint * fill_mult, basis_nnz_hint * std::size_t{2});
    static thread_local std::vector<Index> tl_e_row;
    static thread_local std::vector<Index> tl_e_col;
    static thread_local std::vector<Real> tl_e_val;
    static thread_local std::vector<Index> tl_e_next_row;
    static thread_local std::vector<Index> tl_e_next_col;
    static thread_local std::vector<Index> tl_e_prev_row;
    static thread_local std::vector<Index> tl_e_prev_col;
    static thread_local std::vector<uint8_t> tl_e_alive;
    auto& e_row = tl_e_row;
    auto& e_col = tl_e_col;
    auto& e_val = tl_e_val;
    auto& e_next_row = tl_e_next_row;
    auto& e_next_col = tl_e_next_col;
    auto& e_prev_row = tl_e_prev_row;
    auto& e_prev_col = tl_e_prev_col;
    auto& e_alive = tl_e_alive;
    e_row.clear();
    e_col.clear();
    e_val.clear();
    e_next_row.clear();
    e_next_col.clear();
    e_prev_row.clear();
    e_prev_col.clear();
    e_alive.clear();
    e_row.reserve(reserve_nnz);
    e_col.reserve(reserve_nnz);
    e_val.reserve(reserve_nnz);
    e_next_row.reserve(reserve_nnz);
    e_next_col.reserve(reserve_nnz);
    e_prev_row.reserve(reserve_nnz);
    e_prev_col.reserve(reserve_nnz);
    e_alive.reserve(reserve_nnz);

    // Track alive entry count for compaction decisions.
    Index alive_count = 0;

    // Head of linked list for each row/col (-1 = empty).
    static thread_local std::vector<Index> tl_row_head;
    static thread_local std::vector<Index> tl_col_head;
    static thread_local std::vector<Index> tl_row_count;
    static thread_local std::vector<Index> tl_col_count;
    static thread_local std::vector<uint8_t> tl_row_active;
    static thread_local std::vector<uint8_t> tl_col_active;
    auto& row_head = tl_row_head;
    auto& col_head = tl_col_head;
    auto& row_count = tl_row_count;
    auto& col_count = tl_col_count;
    auto& row_active = tl_row_active;
    auto& col_active = tl_col_active;
    row_head.assign(static_cast<std::size_t>(dim_), -1);
    col_head.assign(static_cast<std::size_t>(dim_), -1);
    row_count.assign(static_cast<std::size_t>(dim_), 0);
    col_count.assign(static_cast<std::size_t>(dim_), 0);
    row_active.assign(static_cast<std::size_t>(dim_), uint8_t{1});
    col_active.assign(static_cast<std::size_t>(dim_), uint8_t{1});

    // HFactor-style count buckets: intrusive lists keyed by count.
    static thread_local std::vector<Index> tl_row_bucket_head;
    static thread_local std::vector<Index> tl_col_bucket_head;
    static thread_local std::vector<Index> tl_row_bucket_prev;
    static thread_local std::vector<Index> tl_row_bucket_next;
    static thread_local std::vector<Index> tl_col_bucket_prev;
    static thread_local std::vector<Index> tl_col_bucket_next;
    auto& row_bucket_head = tl_row_bucket_head;
    auto& col_bucket_head = tl_col_bucket_head;
    auto& row_bucket_prev = tl_row_bucket_prev;
    auto& row_bucket_next = tl_row_bucket_next;
    auto& col_bucket_prev = tl_col_bucket_prev;
    auto& col_bucket_next = tl_col_bucket_next;
    row_bucket_head.assign(static_cast<std::size_t>(dim_ + 1), -1);
    col_bucket_head.assign(static_cast<std::size_t>(dim_ + 1), -1);
    row_bucket_prev.assign(static_cast<std::size_t>(dim_), -1);
    row_bucket_next.assign(static_cast<std::size_t>(dim_), -1);
    col_bucket_prev.assign(static_cast<std::size_t>(dim_), -1);
    col_bucket_next.assign(static_cast<std::size_t>(dim_), -1);

    auto bucketAdd = [](std::vector<Index>& head, std::vector<Index>& prev,
                        std::vector<Index>& next, Index count, Index id) {
        if (count <= 0) {
            return;
        }
        const std::size_t cs = static_cast<std::size_t>(count);
        const std::size_t ids = static_cast<std::size_t>(id);
        const Index first = head[cs];
        prev[ids] = -1;
        next[ids] = first;
        if (first >= 0) {
            prev[static_cast<std::size_t>(first)] = id;
        }
        head[cs] = id;
    };

    auto bucketRemove = [](std::vector<Index>& head, std::vector<Index>& prev,
                           std::vector<Index>& next, Index count, Index id) {
        if (count <= 0) {
            return;
        }
        const std::size_t cs = static_cast<std::size_t>(count);
        const std::size_t ids = static_cast<std::size_t>(id);
        const Index p = prev[ids];
        const Index n = next[ids];
        if (p < 0 && n < 0 && head[cs] != id) {
            return;
        }
        if (p >= 0) {
            next[static_cast<std::size_t>(p)] = n;
        } else {
            head[cs] = n;
        }
        if (n >= 0) {
            prev[static_cast<std::size_t>(n)] = p;
        }
        prev[ids] = -1;
        next[ids] = -1;
    };

    auto deactivate = [](std::vector<uint8_t>& active, Index idx) {
        active[static_cast<std::size_t>(idx)] = uint8_t{0};
    };

    // Populate from matrix columns using BTF-reordered indices.
    // Internal column j corresponds to basis_cols[col_order[j]].
    // Internal row index = row_map[original_row].
    for (Index j = 0; j < dim_; ++j) {
        auto colview = matrix.col(basis_cols[col_order[j]]);
        for (Index k = 0; k < colview.size(); ++k) {
            Index orig_row = colview.indices[k];
            Real val = colview.values[k];
            if (std::abs(val) < kZeroTol) {
                continue;
            }
            if (orig_row >= dim_) {
                continue;  // skip rows outside basis dimension
            }

            Index row = row_map[static_cast<std::size_t>(orig_row)];

            Index eidx = static_cast<Index>(e_row.size());
            e_row.push_back(row);
            e_col.push_back(j);
            e_val.push_back(val);
            e_next_row.push_back(row_head[row]);
            e_next_col.push_back(col_head[j]);
            e_prev_row.push_back(-1);
            e_prev_col.push_back(-1);
            e_alive.push_back(uint8_t{1});
            ++alive_count;

            // Update row linked list.
            if (row_head[row] >= 0) {
                e_prev_row[static_cast<std::size_t>(row_head[row])] = eidx;
            }
            row_head[row] = eidx;
            ++row_count[row];

            // Update col linked list.
            if (col_head[j] >= 0) {
                e_prev_col[static_cast<std::size_t>(col_head[j])] = eidx;
            }
            col_head[j] = eidx;
            ++col_count[j];
        }
    }

    for (Index i = 0; i < dim_; ++i) {
        bucketAdd(row_bucket_head, row_bucket_prev, row_bucket_next, row_count[i], i);
    }
    for (Index j = 0; j < dim_; ++j) {
        bucketAdd(col_bucket_head, col_bucket_prev, col_bucket_next, col_count[j], j);
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
        if (e_alive[static_cast<std::size_t>(eidx)] == 0) {
            return;
        }
        e_alive[static_cast<std::size_t>(eidx)] = uint8_t{0};
        --alive_count;
        const Index row = e_row[static_cast<std::size_t>(eidx)];
        const Index col = e_col[static_cast<std::size_t>(eidx)];
        const Index prev_in_row = e_prev_row[static_cast<std::size_t>(eidx)];
        const Index next_in_row = e_next_row[static_cast<std::size_t>(eidx)];
        const Index prev_in_col = e_prev_col[static_cast<std::size_t>(eidx)];
        const Index next_in_col = e_next_col[static_cast<std::size_t>(eidx)];

        // Remove from row list.
        if (prev_in_row >= 0) {
            e_next_row[static_cast<std::size_t>(prev_in_row)] = next_in_row;
        } else {
            row_head[row] = next_in_row;
        }
        if (next_in_row >= 0) {
            e_prev_row[static_cast<std::size_t>(next_in_row)] = prev_in_row;
        }
        if (row_active[static_cast<std::size_t>(row)] != 0) {
            const Index old_count = row_count[row];
            bucketRemove(row_bucket_head, row_bucket_prev, row_bucket_next, old_count, row);
            --row_count[row];
            bucketAdd(row_bucket_head, row_bucket_prev, row_bucket_next, row_count[row], row);
        } else {
            --row_count[row];
        }

        // Remove from col list.
        if (prev_in_col >= 0) {
            e_next_col[static_cast<std::size_t>(prev_in_col)] = next_in_col;
        } else {
            col_head[col] = next_in_col;
        }
        if (next_in_col >= 0) {
            e_prev_col[static_cast<std::size_t>(next_in_col)] = prev_in_col;
        }
        if (col_active[static_cast<std::size_t>(col)] != 0) {
            const Index old_count = col_count[col];
            bucketRemove(col_bucket_head, col_bucket_prev, col_bucket_next, old_count, col);
            --col_count[col];
            bucketAdd(col_bucket_head, col_bucket_prev, col_bucket_next, col_count[col], col);
        } else {
            --col_count[col];
        }
    };

    // Helper: add entry to row and col lists.
    auto addEntry = [&](Index row, Index col, Real val) -> Index {
        Index eidx = static_cast<Index>(e_row.size());
        e_row.push_back(row);
        e_col.push_back(col);
        e_val.push_back(val);
        e_next_row.push_back(row_head[row]);
        e_next_col.push_back(col_head[col]);
        e_prev_row.push_back(-1);
        e_prev_col.push_back(-1);
        e_alive.push_back(uint8_t{1});
        ++alive_count;
        if (row_head[row] >= 0) {
            e_prev_row[static_cast<std::size_t>(row_head[row])] = eidx;
        }
        row_head[row] = eidx;
        if (row_active[static_cast<std::size_t>(row)] != 0) {
            const Index old_count = row_count[row];
            bucketRemove(row_bucket_head, row_bucket_prev, row_bucket_next, old_count, row);
            ++row_count[row];
            bucketAdd(row_bucket_head, row_bucket_prev, row_bucket_next, row_count[row], row);
        } else {
            ++row_count[row];
        }

        if (col_head[col] >= 0) {
            e_prev_col[static_cast<std::size_t>(col_head[col])] = eidx;
        }
        col_head[col] = eidx;
        if (col_active[static_cast<std::size_t>(col)] != 0) {
            const Index old_count = col_count[col];
            bucketRemove(col_bucket_head, col_bucket_prev, col_bucket_next, old_count, col);
            ++col_count[col];
            bucketAdd(col_bucket_head, col_bucket_prev, col_bucket_next, col_count[col], col);
        } else {
            ++col_count[col];
        }
        return eidx;
    };

    // Work vectors reused across reinversions to reduce allocator traffic.
    static thread_local std::vector<Real> tl_work;
    static thread_local std::vector<Index> tl_work_indices;
    static thread_local std::vector<Index> tl_row_entry_for_col;
    static thread_local std::vector<Index> tl_row_touched_cols;
    static thread_local std::vector<std::pair<Index, Real>> tl_rows_to_update;
    auto& work = tl_work;
    auto& work_indices = tl_work_indices;
    auto& row_entry_for_col = tl_row_entry_for_col;
    auto& row_touched_cols = tl_row_touched_cols;
    auto& rows_to_update = tl_rows_to_update;
    work.assign(static_cast<std::size_t>(dim_), 0.0);
    work_indices.clear();
    work_indices.reserve(static_cast<std::size_t>(dim_));
    row_entry_for_col.assign(static_cast<std::size_t>(dim_), -1);
    row_touched_cols.clear();
    row_touched_cols.reserve(static_cast<std::size_t>(dim_));
    rows_to_update.clear();
    rows_to_update.reserve(static_cast<std::size_t>(dim_));

    for (Index step = 0; step < dim_; ++step) {
        // ---- Markowitz pivot selection ----
        // Find pivot minimizing (row_nnz - 1) * (col_nnz - 1)
        // among entries with |a_ij| >= kPivotTol * max|a_*j|.

        Index best_entry = -1;
        long long best_markowitz = static_cast<long long>(dim_) * dim_;
        Real best_abs = 0.0;

        // First check singleton columns from count buckets (Markowitz = 0).
        for (Index j = col_bucket_head[1]; j >= 0;
             j = col_bucket_next[static_cast<std::size_t>(j)]) {
            if (col_active[static_cast<std::size_t>(j)] == 0 || col_count[j] != 1) {
                continue;
            }
            const Index eidx = col_head[j];
            if (eidx < 0 || e_alive[static_cast<std::size_t>(eidx)] == 0) {
                continue;
            }
            const Real absval = std::abs(e_val[static_cast<std::size_t>(eidx)]);
            if (absval <= kZeroTol) {
                continue;
            }
            const long long m =
                static_cast<long long>(row_count[e_row[static_cast<std::size_t>(eidx)]] - 1) *
                static_cast<long long>(col_count[j] - 1);
            if (m < best_markowitz || (m == best_markowitz && absval > best_abs)) {
                best_markowitz = m;
                best_entry = eidx;
                best_abs = absval;
            }
        }

        // Check singleton rows from count buckets.
        if (best_markowitz > 0) {
            for (Index i = row_bucket_head[1]; i >= 0;
                 i = row_bucket_next[static_cast<std::size_t>(i)]) {
                if (row_active[static_cast<std::size_t>(i)] == 0 || row_count[i] != 1) {
                    continue;
                }
                const Index eidx = row_head[i];
                if (eidx < 0 || e_alive[static_cast<std::size_t>(eidx)] == 0) {
                    continue;
                }
                const Real absval = std::abs(e_val[static_cast<std::size_t>(eidx)]);
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
            // Dynamic early-accept: accept a pivot whose Markowitz cost is
            // small relative to the remaining active dimension.
            const Index remaining = dim_ - step;
            const long long early_accept = std::max(
                static_cast<long long>(1),
                static_cast<long long>(kEarlyAcceptFraction * static_cast<Real>(remaining)));

            for (Index nnz = 2; nnz <= dim_; ++nnz) {
                if (col_bucket_head[static_cast<std::size_t>(nnz)] < 0) {
                    continue;
                }

                const long long col_factor = static_cast<long long>(nnz - 1);
                // If even the best possible row-factor (=1) is already worse, stop.
                if (col_factor > best_markowitz) {
                    break;
                }

                Index cols_inspected = 0;
                for (Index j = col_bucket_head[static_cast<std::size_t>(nnz)]; j >= 0;
                     j = col_bucket_next[static_cast<std::size_t>(j)]) {
                    if (col_active[static_cast<std::size_t>(j)] == 0 || col_count[j] != nnz) {
                        continue;
                    }
                    // Limit columns inspected per bucket to control search time.
                    if (++cols_inspected > kMaxColsPerBucket && best_entry >= 0) {
                        break;
                    }

                    if (use_fast_large_pivot) {
                        // Large-dimension fast path: evaluate one strongest candidate per column.
                        Index best_col_entry = -1;
                        Real col_max = 0.0;
                        for (Index eidx = col_head[j]; eidx >= 0;
                             eidx = e_next_col[static_cast<std::size_t>(eidx)]) {
                            if (e_alive[static_cast<std::size_t>(eidx)] == 0) {
                                continue;
                            }
                            const Real absval = std::abs(e_val[static_cast<std::size_t>(eidx)]);
                            if (absval > col_max) {
                                col_max = absval;
                                best_col_entry = eidx;
                            }
                        }
                        if (best_col_entry >= 0 && col_max > kZeroTol) {
                            const Index ri = e_row[static_cast<std::size_t>(best_col_entry)];
                            const long long m =
                                static_cast<long long>(row_count[ri] - 1) * col_factor;
                            if (m < best_markowitz || (m == best_markowitz && col_max > best_abs)) {
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
                         eidx = e_next_col[static_cast<std::size_t>(eidx)]) {
                        if (e_alive[static_cast<std::size_t>(eidx)] != 0) {
                            col_max =
                                std::max(col_max, std::abs(e_val[static_cast<std::size_t>(eidx)]));
                        }
                    }
                    const Real threshold = kPivotTol * col_max;
                    for (Index eidx = col_head[j]; eidx >= 0;
                         eidx = e_next_col[static_cast<std::size_t>(eidx)]) {
                        if (e_alive[static_cast<std::size_t>(eidx)] == 0) {
                            continue;
                        }
                        const Real absval = std::abs(e_val[static_cast<std::size_t>(eidx)]);
                        if (absval < threshold) {
                            continue;
                        }

                        const Index ri = e_row[static_cast<std::size_t>(eidx)];
                        const long long row_factor = static_cast<long long>(row_count[ri] - 1);
                        if (row_factor > (best_markowitz / col_factor)) {
                            continue;
                        }
                        const long long m = row_factor * col_factor;
                        if (m < best_markowitz || (m == best_markowitz && absval > best_abs)) {
                            best_markowitz = m;
                            best_entry = eidx;
                            best_abs = absval;
                        }
                    }
                }

                // Early exit if we found a pivot with low enough Markowitz cost.
                if (best_markowitz <= early_accept) {
                    break;
                }
            }
        }

        if (best_entry < 0 || best_abs < kZeroTol) {
            throw std::runtime_error("SparseLU::factorize: singular basis matrix");
        }

        Index pivot_row = e_row[static_cast<std::size_t>(best_entry)];
        Index pivot_col = e_col[static_cast<std::size_t>(best_entry)];
        Real pivot_val = e_val[static_cast<std::size_t>(best_entry)];

        // Record permutations in original index space.
        // pivot_row/pivot_col are in internal (BTF-reordered) space.
        Index orig_pivot_row = row_to_orig[static_cast<std::size_t>(pivot_row)];
        Index orig_pivot_col = col_to_orig[static_cast<std::size_t>(pivot_col)];
        row_perm_[step] = orig_pivot_row;
        col_perm_[step] = orig_pivot_col;
        row_perm_inv_[orig_pivot_row] = step;
        col_perm_inv_[orig_pivot_col] = step;

        // ---- Extract pivot row into U ----
        // Gather all entries in the pivot row.
        for (Index eidx = row_head[pivot_row]; eidx >= 0;
             eidx = e_next_row[static_cast<std::size_t>(eidx)]) {
            if (e_alive[static_cast<std::size_t>(eidx)] == 0) {
                continue;
            }
            Index c = e_col[static_cast<std::size_t>(eidx)];
            if (c != pivot_col) {
                u_col_.push_back(c);  // will remap to elimination order later
                const Real v = e_val[static_cast<std::size_t>(eidx)];
                u_val_.push_back(v);
                max_u_entry_ = std::max(max_u_entry_, std::abs(v));
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
             eidx = e_next_row[static_cast<std::size_t>(eidx)]) {
            if (e_alive[static_cast<std::size_t>(eidx)] == 0) {
                continue;
            }
            Index c = e_col[static_cast<std::size_t>(eidx)];
            if (c != pivot_col) {
                work[c] = e_val[static_cast<std::size_t>(eidx)];
                work_indices.push_back(c);
            }
        }

        // Collect rows to update (from pivot column, excluding pivot row).
        rows_to_update.clear();
        for (Index eidx = col_head[pivot_col]; eidx >= 0;
             eidx = e_next_col[static_cast<std::size_t>(eidx)]) {
            if (e_alive[static_cast<std::size_t>(eidx)] == 0) {
                continue;
            }
            const Index row = e_row[static_cast<std::size_t>(eidx)];
            if (row == pivot_row) {
                continue;
            }
            rows_to_update.emplace_back(row, e_val[static_cast<std::size_t>(eidx)] / pivot_val);
        }

        // Store L eta vector (using original row indices).
        for (auto& [ri, mult] : rows_to_update) {
            eta_index_.push_back(row_to_orig[static_cast<std::size_t>(ri)]);
            eta_value_.push_back(mult);
        }
        eta_start_.push_back(static_cast<Index>(eta_index_.size()));

        // Update rows.
        for (auto& [ri, mult] : rows_to_update) {
            // Build a temporary col->entry index map for this row, reused across rows.
            row_touched_cols.clear();
            for (Index eidx = row_head[ri]; eidx >= 0;
                 eidx = e_next_row[static_cast<std::size_t>(eidx)]) {
                if (e_alive[static_cast<std::size_t>(eidx)] == 0 ||
                    e_col[static_cast<std::size_t>(eidx)] == pivot_col) {
                    continue;
                }
                Index c = e_col[static_cast<std::size_t>(eidx)];
                row_entry_for_col[c] = eidx;
                row_touched_cols.push_back(c);
            }

            for (Index c : work_indices) {
                Real update_val = mult * work[c];
                Index row_entry = row_entry_for_col[c];
                if (row_entry >= 0) {
                    Real new_val = e_val[static_cast<std::size_t>(row_entry)] - update_val;
                    if (std::abs(new_val) < kZeroTol) {
                        // Remove entry.
                        removeEntry(row_entry);
                    } else {
                        e_val[static_cast<std::size_t>(row_entry)] = new_val;
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
                Index next = e_next_row[static_cast<std::size_t>(eidx)];
                if (e_alive[static_cast<std::size_t>(eidx)] != 0) {
                    removeEntry(eidx);
                }
                eidx = next;
            }
        }
        // Remove all entries in pivot column.
        {
            Index eidx = col_head[pivot_col];
            while (eidx >= 0) {
                Index next = e_next_col[static_cast<std::size_t>(eidx)];
                if (e_alive[static_cast<std::size_t>(eidx)] != 0) {
                    removeEntry(eidx);
                }
                eidx = next;
            }
        }

        if (row_count[pivot_row] > 0) {
            bucketRemove(row_bucket_head, row_bucket_prev, row_bucket_next, row_count[pivot_row],
                         pivot_row);
        }
        if (col_count[pivot_col] > 0) {
            bucketRemove(col_bucket_head, col_bucket_prev, col_bucket_next, col_count[pivot_col],
                         pivot_col);
        }
        deactivate(row_active, pivot_row);
        deactivate(col_active, pivot_col);

        // Compact fill-in storage when alive ratio drops too low.
        // This reclaims dead entries and improves cache locality.
        const auto total_entries = static_cast<Index>(e_row.size());
        if (alive_count > 0 &&
            static_cast<long long>(total_entries) > 4LL * static_cast<long long>(alive_count) &&
            total_entries > static_cast<Index>(reserve_nnz)) {
            // Build a compacted copy: remap alive entries to contiguous storage.
            // First, build an old-to-new index map.
            static thread_local std::vector<Index> tl_old_to_new;
            auto& old_to_new = tl_old_to_new;
            old_to_new.assign(static_cast<std::size_t>(total_entries), -1);
            Index new_count = 0;
            for (Index e = 0; e < total_entries; ++e) {
                if (e_alive[static_cast<std::size_t>(e)] != 0) {
                    old_to_new[static_cast<std::size_t>(e)] = new_count++;
                }
            }
            // Compact arrays in-place.
            for (Index e = 0; e < total_entries; ++e) {
                const Index ne = old_to_new[static_cast<std::size_t>(e)];
                if (ne < 0) {
                    continue;
                }
                const auto es = static_cast<std::size_t>(e);
                const auto nes = static_cast<std::size_t>(ne);
                e_row[nes] = e_row[es];
                e_col[nes] = e_col[es];
                e_val[nes] = e_val[es];
                e_alive[nes] = uint8_t{1};
                // Remap linked list pointers.
                const Index pr = e_prev_row[es];
                const Index nr = e_next_row[es];
                const Index pc = e_prev_col[es];
                const Index nc = e_next_col[es];
                e_prev_row[nes] = pr >= 0 ? old_to_new[static_cast<std::size_t>(pr)] : -1;
                e_next_row[nes] = nr >= 0 ? old_to_new[static_cast<std::size_t>(nr)] : -1;
                e_prev_col[nes] = pc >= 0 ? old_to_new[static_cast<std::size_t>(pc)] : -1;
                e_next_col[nes] = nc >= 0 ? old_to_new[static_cast<std::size_t>(nc)] : -1;
            }
            // Truncate to new size.
            const auto ns = static_cast<std::size_t>(new_count);
            e_row.resize(ns);
            e_col.resize(ns);
            e_val.resize(ns);
            e_next_row.resize(ns);
            e_next_col.resize(ns);
            e_prev_row.resize(ns);
            e_prev_col.resize(ns);
            e_alive.resize(ns);
            // Update row/col heads.
            for (Index i = 0; i < dim_; ++i) {
                if (row_head[i] >= 0) {
                    row_head[i] = old_to_new[static_cast<std::size_t>(row_head[i])];
                }
                if (col_head[i] >= 0) {
                    col_head[i] = old_to_new[static_cast<std::size_t>(col_head[i])];
                }
            }
        }

        // Clean up work vector.
        for (Index c : work_indices) {
            work[c] = 0.0;
        }
    }

    // Remap U column indices from internal to elimination order.
    // u_col_ contains internal (BTF-reordered) column indices; convert to
    // original basis positions first, then to elimination order.
    for (auto& c : u_col_) {
        c = col_perm_inv_[col_to_orig[static_cast<std::size_t>(c)]];
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

    // --- Supernodal detection ---
    // Identify groups of consecutive elimination steps whose eta patterns
    // form nested subsets (fundamental supernodes). For groups >= kSupernodeMinWidth,
    // build dense column-major panels for efficient L-solve.
    supernodes_.clear();
    snode_panel_values_.clear();
    snode_panel_row_indices_.clear();

    // Lower guard: supernode detection performs a linear scan with sorted
    // target-set comparisons and panel allocations. For small bases the
    // cost of detection itself dominates the savings, since the L-solve
    // is already cheap. Only run on bases where dense-panel L-solve can
    // amortize the detection cost.
    static constexpr Index kSupernodeDetectMinDim = 64;
    if (dim_ >= kSupernodeDetectMinDim) {
        // Detect fundamental supernodes: consecutive steps where each step's
        // eta target set is a subset of the next step's. Use a rolling
        // comparison of sorted target sets to keep cost linear.
        static thread_local std::vector<Index> tl_prev_targets;
        static thread_local std::vector<Index> tl_curr_targets;
        static thread_local std::vector<Index> tl_union_targets;
        auto& prev_targets = tl_prev_targets;
        auto& curr_targets = tl_curr_targets;
        auto& union_targets = tl_union_targets;

        Index snode_start = 0;
        prev_targets.clear();

        for (Index step = 0; step <= dim_; ++step) {
            bool extend_snode = false;

            if (step < dim_) {
                // Build sorted target set for this step.
                curr_targets.clear();
                for (Index k = eta_start_[step]; k < eta_start_[step + 1]; ++k) {
                    curr_targets.push_back(eta_target_[k]);
                }
                std::sort(curr_targets.begin(), curr_targets.end());

                if (step > snode_start && !prev_targets.empty()) {
                    // Check if prev_targets is a subset of curr_targets.
                    // This is the fundamental supernode condition.
                    std::size_t pi = 0;
                    std::size_t ci = 0;
                    bool is_subset = true;
                    while (pi < prev_targets.size() && ci < curr_targets.size()) {
                        if (prev_targets[pi] == curr_targets[ci]) {
                            ++pi;
                            ++ci;
                        } else if (prev_targets[pi] > curr_targets[ci]) {
                            ++ci;
                        } else {
                            is_subset = false;
                            break;
                        }
                    }
                    if (pi < prev_targets.size()) {
                        is_subset = false;
                    }
                    extend_snode = is_subset;
                }
            }

            if (!extend_snode) {
                // Finalize the current supernode [snode_start, step).
                Index snode_width = step - snode_start;
                if (snode_width >= kSupernodeMinWidth) {
                    // Build the union of all target rows in this supernode.
                    union_targets.clear();
                    for (Index s = snode_start; s < step; ++s) {
                        for (Index k = eta_start_[s]; k < eta_start_[s + 1]; ++k) {
                            union_targets.push_back(eta_target_[k]);
                        }
                    }
                    std::sort(union_targets.begin(), union_targets.end());
                    union_targets.erase(std::unique(union_targets.begin(), union_targets.end()),
                                        union_targets.end());

                    Index panel_rows = static_cast<Index>(union_targets.size());
                    if (panel_rows > 0) {
                        // Build row index -> panel row mapping.
                        // We'll store the panel in column-major order: panel[r][c]
                        // at offset panel_offset + c * panel_rows + r.
                        Index panel_offset = static_cast<Index>(snode_panel_values_.size());
                        Index row_offset = static_cast<Index>(snode_panel_row_indices_.size());

                        snode_panel_values_.resize(
                            static_cast<std::size_t>(panel_offset + panel_rows * snode_width), 0.0);

                        // Store row indices.
                        for (Index t : union_targets) {
                            snode_panel_row_indices_.push_back(t);
                        }

                        // Build target -> panel_row map using the sorted union.
                        // Since union_targets is sorted, use binary search.
                        for (Index c = 0; c < snode_width; ++c) {
                            Index s = snode_start + c;
                            for (Index k = eta_start_[s]; k < eta_start_[s + 1]; ++k) {
                                Index target = eta_target_[k];
                                // Binary search in union_targets.
                                auto it = std::lower_bound(union_targets.begin(),
                                                           union_targets.end(), target);
                                Index r = static_cast<Index>(it - union_targets.begin());
                                snode_panel_values_[static_cast<std::size_t>(
                                    panel_offset + c * panel_rows + r)] = eta_value_[k];
                            }
                        }

                        supernodes_.push_back(
                            {snode_start, snode_width, panel_rows, panel_offset, row_offset});
                    }
                }
                snode_start = step;
            }

            std::swap(prev_targets, curr_targets);
        }
    }

    // Reset EMA density predictions (fresh factorization).
    for (int s = 0; s < kNumSolveStages; ++s) {
        ema_density_[s] = 0.0;
    }

    // Count work: total nonzeros stored in L and U.
    work_.count(static_cast<uint64_t>(eta_index_.size()) + static_cast<uint64_t>(u_col_.size()));

    // Build FP32 copies of L and U factors if mixed-precision is enabled.
    if (mixed_precision_enabled_) {
        mixed_precision_active_ = buildFp32Factors();
        if (mixed_precision_active_) {
            ir_matrix_ = &matrix;
            ir_basis_cols_.assign(basis_cols.begin(), basis_cols.end());
        }
    } else {
        mixed_precision_active_ = false;
    }
    ir_step_ema_count_ = 0;
    ir_solves_count_ = 0;
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
        if (pivot_x == 0.0) {
            continue;
        }
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
        if (xp == 0.0) {
            continue;
        }
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
    // Tracks output density via EMA (stage 3) to guide upstream solve decisions.
    if (num_updates_ == 0) {
        return;
    }

    const Index n = static_cast<Index>(x.size());

    for (Index u = num_updates_ - 1; u >= 0; --u) {
        Index pos = ft_pivot_pos_[u];
        Real sum = 0.0;
        if (ft_is_dense_[u] != 0) {
            // Dense-stored eta: use SIMD-accelerated dense dot product.
            Index dense_off = ft_dense_offset_[u];
            const Real* dense = ft_dense_value_.data() + dense_off;
            sum = denseDot(dense, x);
        } else {
            // Sparse-stored eta: use indexed dot product (only touches eta's
            // nonzero positions, efficient regardless of input density).
            const Index start = ft_start_[u];
            const Index end = ft_start_[u + 1];
            for (Index k = start; k < end; ++k) {
                sum += ft_value_[k] * x[ft_index_[k]];
            }
        }
        x[pos] = (x[pos] - sum) * ft_pivot_inv_[u];
    }

    // Count output nonzeros and update EMA for BTRAN FT-transpose density (stage 3).
    Index output_nnz = 0;
    for (Index i = 0; i < n; ++i) {
        if (std::abs(x[i]) > kZeroTol) {
            ++output_nnz;
        }
    }
    {
        const Real actual_density =
            static_cast<Real>(output_nnz) / static_cast<Real>(std::max<Index>(n, 1));
        ema_density_[3] = kEmaAlpha * actual_density + (1.0 - kEmaAlpha) * ema_density_[3];
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
        if (val == 0.0) {
            continue;
        }
        Index start = u_start_[step];
        Index end = u_start_[step + 1];
        for (Index k = start; k < end; ++k) {
            x[u_col_[k]] -= u_val_[k] * val;
        }
    }
}

Index SparseLU::solveUTransposeSparse(std::span<Real> x) const {
    // Hyper-sparse forward substitution with U^T using reachability.
    // Only process steps reachable from the initial nonzero positions.
    if (static_cast<Index>(sparse_epoch_.size()) < dim_) {
        sparse_epoch_.resize(dim_, 0U);
    }
    ++sparse_epoch_id_;
    if (sparse_epoch_id_ == 0U) {
        std::fill(sparse_epoch_.begin(), sparse_epoch_.begin() + dim_, 0U);
        sparse_epoch_id_ = 1U;
    }
    sparse_steps_.clear();

    // Seed: all steps with nonzero x[step].
    for (Index step = 0; step < dim_; ++step) {
        if (std::abs(x[step]) > kZeroTol) {
            sparse_epoch_[static_cast<std::size_t>(step)] = sparse_epoch_id_;
            sparse_steps_.push_back(step);
        }
    }

    // Forward reachability: step j scatters into u_col_[k] for its row entries.
    // So if step j is active, all columns it touches become active.
    for (std::size_t idx = 0; idx < sparse_steps_.size(); ++idx) {
        Index step = sparse_steps_[idx];
        for (Index k = u_start_[step]; k < u_start_[step + 1]; ++k) {
            Index col = u_col_[k];
            if (sparse_epoch_[static_cast<std::size_t>(col)] != sparse_epoch_id_) {
                sparse_epoch_[static_cast<std::size_t>(col)] = sparse_epoch_id_;
                sparse_steps_.push_back(col);
            }
        }
    }

    // Sort reachable steps and process in forward order.
    std::sort(sparse_steps_.begin(), sparse_steps_.end());

    const Index bailout_threshold =
        static_cast<Index>(kHyperSparseMaxDensity * static_cast<Real>(dim_) * 2.0);
    Index out_nnz = 0;

    // Process all reachable steps in forward order. The scatter must run for
    // every step (regardless of bailout) because subsequent steps depend on
    // the contributions written into x[u_col_[k]] here. The bailout check
    // only counts output density and runs after the scatter is committed.
    for (std::size_t sorted_idx = 0; sorted_idx < sparse_steps_.size(); ++sorted_idx) {
        Index step = sparse_steps_[sorted_idx];
        x[step] *= u_diag_inv_[step];
        Real val = x[step];
        if (val == 0.0) {
            continue;
        }
        for (Index k = u_start_[step]; k < u_start_[step + 1]; ++k) {
            x[u_col_[k]] -= u_val_[k] * val;
        }
        if (std::abs(val) > kZeroTol) {
            ++out_nnz;
        }
    }

    // bailout_threshold is recorded for caller cost analysis but does not
    // truncate the loop: skipping any step's scatter loses correctness.
    (void)bailout_threshold;

    return out_nnz;
}

void SparseLU::ftran(std::span<Real> rhs) const {
    assert(static_cast<Index>(rhs.size()) == dim_);

    // Mixed-precision path: solve in FP32 with FP64 iterative refinement.
    // Factorization is in FP32 (half memory). Solves use FP32 factors, then
    // compute residual r = b - B'*x using the original FP64 matrix and FT etas,
    // and iterate: x <- x + solve32(r).
    //
    // B' = B * E_1 * ... * E_n, so B' * x = B * (E_1 * ... * E_n * x).
    // We compute z = E_1 * ... * E_n * x, then B * z from the original matrix.
    if (mixed_precision_active_ && ir_matrix_ != nullptr) {
        // Count work: FP32 solve + IR residual computation.
        work_.count(static_cast<uint64_t>(eta_index_.size()) +
                    static_cast<uint64_t>(u_col_.size()) + static_cast<uint64_t>(ft_index_.size()) +
                    ft_dense_nnz_ + static_cast<uint64_t>(dim_) * 2);

        // Save original RHS for residual computation (reuse persistent buffer).
        if (static_cast<Index>(ir_rhs_save_.size()) < dim_) {
            ir_rhs_save_.resize(dim_);
        }
        std::copy(rhs.begin(), rhs.end(), ir_rhs_save_.begin());
        std::span<const Real> b_save(ir_rhs_save_.data(), static_cast<std::size_t>(dim_));

        // Initial solve in FP32.
        ftranFp32(rhs);

        // Iterative refinement loop.
        int steps_used = 0;
        if (static_cast<Index>(ir_residual_.size()) < dim_) {
            ir_residual_.resize(dim_);
        }
        if (static_cast<Index>(ir_z_.size()) < dim_) {
            ir_z_.resize(dim_);
        }
        std::span<Real> residual(ir_residual_.data(), static_cast<std::size_t>(dim_));
        std::span<Real> z(ir_z_.data(), static_cast<std::size_t>(dim_));
        for (int ir = 0; ir < kMaxRefinementSteps; ++ir) {
            // Compute z = E_1 * ... * E_n * x (undo FT updates to get original-basis coords).
            // Process from u = n-1 down to 0 (innermost E_n first, then E_{n-1}, etc.)
            // E_u * y: y[i] += d_u[i] * y[p_u] for i != p, then y[p] *= d_u[p].
            std::copy(rhs.begin(), rhs.end(), z.begin());
            for (Index u = num_updates_ - 1; u >= 0; --u) {
                Index pos = ft_pivot_pos_[u];
                Real yp = z[pos];
                if (ft_is_dense_[u] != 0) {
                    Index dense_off = ft_dense_offset_[u];
                    for (Index i = 0; i < dim_; ++i) {
                        if (i != pos) {
                            z[i] += ft_dense_value_[static_cast<std::size_t>(dense_off + i)] * yp;
                        }
                    }
                } else {
                    const Index start = ft_start_[u];
                    const Index end = ft_start_[u + 1];
                    for (Index k = start; k < end; ++k) {
                        z[ft_index_[k]] += ft_value_[k] * yp;
                    }
                }
                z[pos] = yp * ft_pivot_val_[u];
            }

            // Compute B * z using original matrix columns.
            std::fill(residual.begin(), residual.end(), 0.0);
            for (Index j = 0; j < dim_; ++j) {
                Real zj = z[j];
                if (zj == 0.0) {
                    continue;
                }
                auto colview = ir_matrix_->col(ir_basis_cols_[j]);
                for (Index k = 0; k < colview.size(); ++k) {
                    Index row = colview.indices[k];
                    if (row < dim_) {
                        residual[row] += colview.values[k] * zj;
                    }
                }
            }

            // Residual r = b - B' * x.
            Real max_residual = 0.0;
            for (Index i = 0; i < dim_; ++i) {
                residual[i] = b_save[i] - residual[i];
                max_residual = std::max(max_residual, std::abs(residual[i]));
            }

            if (max_residual < kRefinementTol) {
                break;
            }
            ++steps_used;

            // Solve correction in FP32.
            ftranFp32(residual);

            // Update solution in FP64.
            for (Index i = 0; i < dim_; ++i) {
                rhs[i] += residual[i];
            }
        }

        // Track refinement effort for automatic fallback.
        ++ir_solves_count_;
        if (ir_solves_count_ <= 1) {
            ir_step_ema_count_ = steps_used;
        } else {
            ir_step_ema_count_ =
                static_cast<int>(0.3 * steps_used + 0.7 * ir_step_ema_count_ + 0.5);
        }
        if (ir_solves_count_ >= 10 && ir_step_ema_count_ >= kFallbackRefinementCount) {
            mixed_precision_active_ = false;
        }

        return;
    }

    // Count work: L nnz + U nnz + FT nnz + permutation overhead.
    work_.count(static_cast<uint64_t>(eta_index_.size()) + static_cast<uint64_t>(u_col_.size()) +
                static_cast<uint64_t>(ft_index_.size()) + ft_dense_nnz_ +
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

    // Step 2: Apply L^{-1}. Use EMA-predicted density or measured density to
    // choose between hyper-sparse reachability solve and dense sequential scan.
    const Real predicted_l_density = ema_density_[0];
    bool use_hypersparse_l =
        dim_ >= kHyperSparseMinDim && !eta_index_.empty() &&
        (predicted_l_density > 0.0
             ? predicted_l_density < kHyperSparseMaxDensity
             : work_nnz <= static_cast<Index>(kHyperSparseMaxDensity * static_cast<Real>(dim_)));

    Index l_out_nnz = work_nnz;
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

        // Mid-solve bailout threshold: switch to dense if nonzero count exceeds this.
        const Index bailout_threshold =
            static_cast<Index>(kHyperSparseMaxDensity * static_cast<Real>(dim_) * 2.0);
        bool bailed_out = false;

        if (max_step >= min_step) {
            const Index range_len = max_step - min_step + 1;
            const Index reach_nnz = static_cast<Index>(sparse_steps_.size());
            if (range_len <= 4 * reach_nnz) {
                l_out_nnz = 0;
                Index bailout_step = max_step + 1;
                for (Index step = min_step; step <= max_step; ++step) {
                    if (sparse_epoch_[static_cast<std::size_t>(step)] != sparse_epoch_id_) {
                        continue;
                    }
                    Real wk = work[step];
                    if (std::abs(wk) <= kZeroTol) {
                        continue;
                    }
                    // Scatter must run before any bailout — later steps depend
                    // on this contribution. Bailout only switches the *iteration*
                    // strategy (sparse → dense scan) for remaining steps.
                    for (Index k = eta_start_[step]; k < eta_start_[step + 1]; ++k) {
                        work[eta_target_[k]] -= eta_value_[k] * wk;
                    }
                    ++l_out_nnz;
                    if (l_out_nnz > bailout_threshold) {
                        bailout_step = step + 1;
                        bailed_out = true;
                        break;
                    }
                }
                // Dense fallback for remaining steps after bailout.
                if (bailed_out) {
                    for (Index step = bailout_step; step < dim_; ++step) {
                        Real wk = work[step];
                        if (wk == 0.0) {
                            continue;
                        }
                        for (Index k = eta_start_[step]; k < eta_start_[step + 1]; ++k) {
                            work[eta_target_[k]] -= eta_value_[k] * wk;
                        }
                    }
                }
            } else {
                std::sort(sparse_steps_.begin(), sparse_steps_.end());
                l_out_nnz = 0;
                std::size_t bailout_idx = sparse_steps_.size();
                for (std::size_t si = 0; si < sparse_steps_.size(); ++si) {
                    Index step = sparse_steps_[si];
                    Real wk = work[step];
                    if (std::abs(wk) <= kZeroTol) {
                        continue;
                    }
                    for (Index k = eta_start_[step]; k < eta_start_[step + 1]; ++k) {
                        work[eta_target_[k]] -= eta_value_[k] * wk;
                    }
                    ++l_out_nnz;
                    if (l_out_nnz > bailout_threshold) {
                        bailout_idx = si + 1;
                        bailed_out = true;
                        break;
                    }
                }
                // Dense fallback for remaining steps after bailout.
                if (bailed_out && bailout_idx < sparse_steps_.size()) {
                    Index resume_from = sparse_steps_[bailout_idx];
                    for (Index step = resume_from; step < dim_; ++step) {
                        Real wk = work[step];
                        if (wk == 0.0) {
                            continue;
                        }
                        for (Index k = eta_start_[step]; k < eta_start_[step + 1]; ++k) {
                            work[eta_target_[k]] -= eta_value_[k] * wk;
                        }
                    }
                }
            }
        }
    } else if (!supernodes_.empty()) {
        // Dense L-solve with supernodal acceleration.
        // Process steps sequentially, but use dense panel operations for supernodes.
        std::size_t snode_idx = 0;
        for (Index step = 0; step < dim_;) {
            // Check if this step starts a supernode.
            if (snode_idx < supernodes_.size() && supernodes_[snode_idx].start == step) {
                const auto& sn = supernodes_[snode_idx];
                // Apply dense panel: for each column c in the supernode,
                // subtract panel[:,c] * work[step+c] from the target rows.
                const Real* panel = snode_panel_values_.data() + sn.panel_offset;
                const Index* row_idx = snode_panel_row_indices_.data() + sn.row_offset;
                for (Index c = 0; c < sn.width; ++c) {
                    Real wk = work[step + c];
                    if (wk == 0.0) {
                        continue;
                    }
                    const Real* col_panel = panel + c * sn.panel_rows;
                    for (Index r = 0; r < sn.panel_rows; ++r) {
                        work[row_idx[r]] -= col_panel[r] * wk;
                    }
                }
                step += sn.width;
                ++snode_idx;
            } else {
                // Scalar path for steps not in a supernode.
                Real wk = work[step];
                if (wk != 0.0) {
                    for (Index k = eta_start_[step]; k < eta_start_[step + 1]; ++k) {
                        work[eta_target_[k]] -= eta_value_[k] * wk;
                    }
                }
                ++step;
            }
        }
        // Count output nonzeros for EMA.
        l_out_nnz = 0;
        for (Index step = 0; step < dim_; ++step) {
            if (std::abs(work[step]) > kZeroTol) {
                ++l_out_nnz;
            }
        }
    } else {
        // Standard dense L-solve (no supernodes).
        for (Index step = 0; step < dim_; ++step) {
            Real wk = work[step];
            if (wk == 0.0) {
                continue;
            }
            for (Index k = eta_start_[step]; k < eta_start_[step + 1]; ++k) {
                work[eta_target_[k]] -= eta_value_[k] * wk;
            }
        }
        // Count output nonzeros for EMA.
        l_out_nnz = 0;
        for (Index step = 0; step < dim_; ++step) {
            if (std::abs(work[step]) > kZeroTol) {
                ++l_out_nnz;
            }
        }
    }

    // Update EMA for FTRAN L-solve density (stage 0).
    {
        const Real actual_density =
            static_cast<Real>(l_out_nnz) / static_cast<Real>(std::max<Index>(dim_, 1));
        ema_density_[0] = kEmaAlpha * actual_density + (1.0 - kEmaAlpha) * ema_density_[0];
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
    btranImpl(rhs, nullptr);
}

void SparseLU::btran(std::span<Real> rhs, std::vector<Index>& nonzero_rows) const {
    btranImpl(rhs, &nonzero_rows);
}

void SparseLU::btranImpl(std::span<Real> rhs, std::vector<Index>* nonzero_rows) const {
    assert(static_cast<Index>(rhs.size()) == dim_);

    // Mixed-precision path: solve B'^T y = c in FP32 with FP64 iterative refinement.
    // B'^T = E_n^T * ... * E_1^T * B^T.
    // Residual: r = c - B'^T * y = c - E_n^T * ... * E_1^T * B^T * y.
    // First compute w = B^T * y using original matrix, then apply E^T etas.
    if (mixed_precision_active_ && ir_matrix_ != nullptr) {
        // Count work: FP32 solve + IR residual computation.
        work_.count(static_cast<uint64_t>(eta_index_.size()) +
                    static_cast<uint64_t>(u_col_.size()) + static_cast<uint64_t>(ft_index_.size()) +
                    ft_dense_nnz_ + static_cast<uint64_t>(dim_) * 2);

        // Save original RHS (reuse persistent buffer).
        if (static_cast<Index>(ir_rhs_save_.size()) < dim_) {
            ir_rhs_save_.resize(dim_);
        }
        std::copy(rhs.begin(), rhs.end(), ir_rhs_save_.begin());
        std::span<const Real> c_save(ir_rhs_save_.data(), static_cast<std::size_t>(dim_));

        // Initial solve in FP32.
        btranFp32(rhs);

        // Iterative refinement loop.
        int steps_used = 0;
        if (static_cast<Index>(ir_residual_.size()) < dim_) {
            ir_residual_.resize(dim_);
        }
        std::span<Real> residual(ir_residual_.data(), static_cast<std::size_t>(dim_));
        for (int ir = 0; ir < kMaxRefinementSteps; ++ir) {
            // Compute w = B^T * y using original matrix.
            // (B^T * y)[j] = col(basis_cols[j])^T * y.
            for (Index j = 0; j < dim_; ++j) {
                Real dot = 0.0;
                auto colview = ir_matrix_->col(ir_basis_cols_[j]);
                for (Index k = 0; k < colview.size(); ++k) {
                    Index row = colview.indices[k];
                    if (row < dim_) {
                        dot += colview.values[k] * rhs[row];
                    }
                }
                residual[j] = dot;
            }

            // Apply E_n^T * ... * E_1^T to w.
            // E_u^T * z: z[p] = d[p] * z[p] + sum_{i!=p} d[i] * z[i]; other z[i] unchanged.
            // Process from u = num_updates_-1 down to 0.
            for (Index u = num_updates_ - 1; u >= 0; --u) {
                Index pos = ft_pivot_pos_[u];
                Real sum = 0.0;
                if (ft_is_dense_[u] != 0) {
                    Index dense_off = ft_dense_offset_[u];
                    for (Index i = 0; i < dim_; ++i) {
                        if (i != pos) {
                            sum += ft_dense_value_[static_cast<std::size_t>(dense_off + i)] *
                                   residual[i];
                        }
                    }
                } else {
                    const Index start = ft_start_[u];
                    const Index end = ft_start_[u + 1];
                    for (Index k = start; k < end; ++k) {
                        sum += ft_value_[k] * residual[ft_index_[k]];
                    }
                }
                residual[pos] = ft_pivot_val_[u] * residual[pos] + sum;
            }

            // Residual r = c - B'^T * y.
            Real max_residual = 0.0;
            for (Index i = 0; i < dim_; ++i) {
                residual[i] = c_save[i] - residual[i];
                max_residual = std::max(max_residual, std::abs(residual[i]));
            }

            if (max_residual < kRefinementTol) {
                break;
            }
            ++steps_used;

            // Solve correction in FP32.
            btranFp32(residual);

            // Update solution in FP64.
            for (Index i = 0; i < dim_; ++i) {
                rhs[i] += residual[i];
            }
        }

        // Track refinement effort.
        ++ir_solves_count_;
        if (ir_solves_count_ <= 1) {
            ir_step_ema_count_ = steps_used;
        } else {
            ir_step_ema_count_ =
                static_cast<int>(0.3 * steps_used + 0.7 * ir_step_ema_count_ + 0.5);
        }
        if (ir_solves_count_ >= 10 && ir_step_ema_count_ >= kFallbackRefinementCount) {
            mixed_precision_active_ = false;
        }

        // Collect nonzero rows if requested.
        if (nonzero_rows != nullptr) {
            nonzero_rows->clear();
            for (Index i = 0; i < dim_; ++i) {
                if (std::abs(rhs[i]) > kZeroTol) {
                    nonzero_rows->push_back(i);
                }
            }
        }
        return;
    }

    // Count work: same structure as ftran.
    work_.count(static_cast<uint64_t>(eta_index_.size()) + static_cast<uint64_t>(u_col_.size()) +
                static_cast<uint64_t>(ft_index_.size()) + ft_dense_nnz_ +
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
    // Use hyper-sparse path when input is sparse enough, guided by EMA prediction.
    Index ut_input_nnz = 0;
    for (Index step = 0; step < dim_; ++step) {
        if (std::abs(work[step]) > kZeroTol) {
            ++ut_input_nnz;
        }
    }

    const Real predicted_ut_density = ema_density_[2];
    const bool use_hypersparse_ut =
        dim_ >= kHyperSparseMinDim && !u_col_.empty() &&
        (predicted_ut_density > 0.0 ? predicted_ut_density < kHyperSparseMaxDensity
                                    : ut_input_nnz <= static_cast<Index>(kHyperSparseMaxDensity *
                                                                         static_cast<Real>(dim_)));

    Index ut_out_nnz;
    if (use_hypersparse_ut) {
        ut_out_nnz = solveUTransposeSparse(work);
    } else {
        solveUTranspose(work);
        ut_out_nnz = 0;
        for (Index step = 0; step < dim_; ++step) {
            if (std::abs(work[step]) > kZeroTol) {
                ++ut_out_nnz;
            }
        }
    }

    // Update EMA for BTRAN U^T-solve density (stage 2).
    {
        const Real actual_density =
            static_cast<Real>(ut_out_nnz) / static_cast<Real>(std::max<Index>(dim_, 1));
        ema_density_[2] = kEmaAlpha * actual_density + (1.0 - kEmaAlpha) * ema_density_[2];
    }

    // Step 4: Apply L^{-T}. Use reverse-reach hyper-sparse solve when possible,
    // guided by EMA prediction for BTRAN L^T-solve (stage 1).
    Index work_nnz = ut_out_nnz;
    const Real predicted_lt_density = ema_density_[1];
    bool use_hypersparse_lt =
        dim_ >= kHyperSparseMinDim && !eta_index_.empty() &&
        (predicted_lt_density > 0.0
             ? predicted_lt_density < kHyperSparseMaxDensity
             : work_nnz <= static_cast<Index>(kHyperSparseMaxDensity * static_cast<Real>(dim_)));

    Index lt_out_nnz = work_nnz;
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

        // Mid-solve bailout threshold.
        const Index bailout_threshold =
            static_cast<Index>(kHyperSparseMaxDensity * static_cast<Real>(dim_) * 2.0);
        bool bailed_out = false;

        if (max_step >= min_step) {
            const Index range_len = max_step - min_step + 1;
            const Index reach_nnz = static_cast<Index>(sparse_steps_.size());
            if (range_len <= 4 * reach_nnz) {
                lt_out_nnz = 0;
                Index bailout_step = min_step - 1;
                for (Index step = max_step; step >= min_step; --step) {
                    if (sparse_epoch_[static_cast<std::size_t>(step)] != sparse_epoch_id_) {
                        if (step == min_step) {
                            break;
                        }
                        continue;
                    }
                    Real sum = 0.0;
                    for (Index k = eta_start_[step]; k < eta_start_[step + 1]; ++k) {
                        sum += eta_value_[k] * work[eta_target_[k]];
                    }
                    work[step] -= sum;
                    if (std::abs(work[step]) > kZeroTol) {
                        ++lt_out_nnz;
                        if (lt_out_nnz > bailout_threshold) {
                            bailout_step = step - 1;
                            bailed_out = true;
                            break;
                        }
                    }
                    if (step == min_step) {
                        break;
                    }
                }
                // Dense fallback for remaining steps after bailout.
                if (bailed_out) {
                    for (Index step = bailout_step; step >= 0; --step) {
                        Real sum = 0.0;
                        for (Index k = eta_start_[step]; k < eta_start_[step + 1]; ++k) {
                            sum += eta_value_[k] * work[eta_target_[k]];
                        }
                        work[step] -= sum;
                    }
                }
            } else {
                std::sort(sparse_steps_.begin(), sparse_steps_.end());
                lt_out_nnz = 0;
                auto it = sparse_steps_.rbegin();
                for (; it != sparse_steps_.rend(); ++it) {
                    const Index step = *it;
                    Real sum = 0.0;
                    for (Index k = eta_start_[step]; k < eta_start_[step + 1]; ++k) {
                        sum += eta_value_[k] * work[eta_target_[k]];
                    }
                    work[step] -= sum;
                    if (std::abs(work[step]) > kZeroTol) {
                        ++lt_out_nnz;
                        if (lt_out_nnz > bailout_threshold) {
                            ++it;
                            bailed_out = true;
                            break;
                        }
                    }
                }
                // Dense fallback for remaining steps after bailout.
                if (bailed_out && it != sparse_steps_.rend()) {
                    Index resume_to = *it;
                    for (Index step = resume_to; step >= 0; --step) {
                        Real sum = 0.0;
                        for (Index k = eta_start_[step]; k < eta_start_[step + 1]; ++k) {
                            sum += eta_value_[k] * work[eta_target_[k]];
                        }
                        work[step] -= sum;
                    }
                }
            }
        }
    } else if (!supernodes_.empty()) {
        // Dense L^T-solve with supernodal acceleration (reverse order).
        auto snode_idx = static_cast<std::ptrdiff_t>(supernodes_.size()) - 1;
        for (Index step = dim_ - 1; step >= 0;) {
            // Check if this step ends a supernode.
            if (snode_idx >= 0 && supernodes_[static_cast<std::size_t>(snode_idx)].start +
                                          supernodes_[static_cast<std::size_t>(snode_idx)].width -
                                          1 ==
                                      step) {
                const auto& sn = supernodes_[static_cast<std::size_t>(snode_idx)];
                // L^T-solve for the supernode: process columns in reverse.
                // For column c: work[sn.start+c] -= dot(panel[:,c], work[targets])
                const Real* panel = snode_panel_values_.data() + sn.panel_offset;
                const Index* row_idx = snode_panel_row_indices_.data() + sn.row_offset;
                for (Index c = sn.width - 1; c >= 0; --c) {
                    const Real* col_panel = panel + c * sn.panel_rows;
                    Real sum = 0.0;
                    for (Index r = 0; r < sn.panel_rows; ++r) {
                        sum += col_panel[r] * work[row_idx[r]];
                    }
                    work[sn.start + c] -= sum;
                }
                step = sn.start - 1;
                --snode_idx;
            } else {
                // Scalar path.
                Real sum = 0.0;
                for (Index k = eta_start_[step]; k < eta_start_[step + 1]; ++k) {
                    sum += eta_value_[k] * work[eta_target_[k]];
                }
                work[step] -= sum;
                --step;
            }
        }
        // Count output nonzeros for EMA.
        lt_out_nnz = 0;
        for (Index step = 0; step < dim_; ++step) {
            if (std::abs(work[step]) > kZeroTol) {
                ++lt_out_nnz;
            }
        }
    } else {
        // Standard dense L^T-solve (no supernodes).
        for (Index step = dim_ - 1; step >= 0; --step) {
            Real sum = 0.0;
            for (Index k = eta_start_[step]; k < eta_start_[step + 1]; ++k) {
                sum += eta_value_[k] * work[eta_target_[k]];
            }
            work[step] -= sum;
        }
        // Count output nonzeros for EMA.
        lt_out_nnz = 0;
        for (Index step = 0; step < dim_; ++step) {
            if (std::abs(work[step]) > kZeroTol) {
                ++lt_out_nnz;
            }
        }
    }

    // Update EMA for BTRAN L^T-solve density (stage 1).
    {
        const Real actual_density =
            static_cast<Real>(lt_out_nnz) / static_cast<Real>(std::max<Index>(dim_, 1));
        ema_density_[1] = kEmaAlpha * actual_density + (1.0 - kEmaAlpha) * ema_density_[1];
    }

    // Step 5: y = P^T * work. y[row_perm[step]] = work[step].
    // Actually P^T: y[i] = work[row_perm_inv[i]].
    // But we want: for each step, rhs[row_perm[step]] = work[step].
    // That's P^{-1} = P^T since P is a permutation.
    if (nonzero_rows != nullptr) {
        nonzero_rows->clear();
    }
    for (Index step = 0; step < dim_; ++step) {
        const Real value = work[step];
        rhs[row_perm_[step]] = value;
        if (nonzero_rows != nullptr && std::abs(value) > kZeroTol) {
            nonzero_rows->push_back(row_perm_[step]);
        }
    }
}

// --------------------------------------------------------------------------
//  Forrest-Tomlin update
// --------------------------------------------------------------------------

void SparseLU::update(Index pivot_pos, std::span<const Index> indices,
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

void SparseLU::updateFromFtranColumn(Index pivot_pos, std::span<const Real> transformed_col) {
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
        if (i == pivot_pos) {
            continue;
        }
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
            if (i == pivot_pos) {
                continue;
            }
            const Real v = (std::abs(transformed_col[i]) > drop_tol) ? transformed_col[i] : 0.0;
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
            if (i == pivot_pos) {
                continue;
            }
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

    // Maintain FP32 copies for mixed-precision solves.
    if (mixed_precision_active_) {
        // Extend FP32 sparse FT values to match FP64 ft_value_.
        while (ft_value_f32_.size() < ft_value_.size()) {
            ft_value_f32_.push_back(static_cast<float>(ft_value_[ft_value_f32_.size()]));
        }
        ft_pivot_val_f32_.push_back(static_cast<float>(pivot_val));
        ft_pivot_inv_f32_.push_back(static_cast<float>(1.0 / pivot_val));
        // Extend FP32 dense FT values to match FP64 ft_dense_value_.
        while (ft_dense_value_f32_.size() < ft_dense_value_.size()) {
            ft_dense_value_f32_.push_back(
                static_cast<float>(ft_dense_value_[ft_dense_value_f32_.size()]));
        }
    }

    max_u_entry_ = std::max(max_u_entry_, std::abs(pivot_val));

    // Count work: scanning d vector for eta storage.
    work_.count(static_cast<uint64_t>(dim_));

    ++num_updates_;
}

bool SparseLU::needsRefactorization() const {
    if (num_updates_ >= max_updates_) {
        return true;
    }
    if (max_u_entry_ > kGrowthLimit) {
        return true;
    }
    // Cost-based trigger: refactor when accumulated FT update overhead
    // dominates the base FTRAN/BTRAN cost. Each pivot adds roughly one
    // eta to ft_index_, so for a long sequence of updates (especially on
    // small bases) the per-solve cost balloons relative to the eta+U
    // base. The crossover at 2x base cost is empirically near the point
    // where amortizing one O(nnz) refactor pays off.
    const uint64_t base_cost =
        static_cast<uint64_t>(eta_index_.size()) + static_cast<uint64_t>(u_col_.size());
    const uint64_t ft_cost = static_cast<uint64_t>(ft_index_.size()) + ft_dense_nnz_;
    if (base_cost > 0 && ft_cost > 2 * base_cost && num_updates_ >= kMinUpdatesForCostRefactor) {
        return true;
    }
    return false;
}

// --------------------------------------------------------------------------
//  Mixed-precision FP32 support
// --------------------------------------------------------------------------

bool SparseLU::buildFp32Factors() {
    // Check element growth: if any factor entry exceeds FP32 safe range, bail out.
    if (max_u_entry_ > kFp32GrowthLimit) {
        return false;
    }

    // Build FP32 copies of L eta values.
    eta_value_f32_.resize(eta_value_.size());
    for (std::size_t i = 0; i < eta_value_.size(); ++i) {
        eta_value_f32_[i] = static_cast<float>(eta_value_[i]);
    }

    // Build FP32 copies of U values.
    u_val_f32_.resize(u_val_.size());
    for (std::size_t i = 0; i < u_val_.size(); ++i) {
        u_val_f32_[i] = static_cast<float>(u_val_[i]);
    }
    u_diag_f32_.resize(u_diag_.size());
    u_diag_inv_f32_.resize(u_diag_inv_.size());
    for (std::size_t i = 0; i < u_diag_.size(); ++i) {
        u_diag_f32_[i] = static_cast<float>(u_diag_[i]);
        u_diag_inv_f32_[i] = static_cast<float>(u_diag_inv_[i]);
    }

    // Initialize empty FT FP32 arrays (will be populated by updateFromFtranColumn).
    ft_value_f32_.clear();
    ft_pivot_val_f32_.clear();
    ft_pivot_inv_f32_.clear();
    ft_dense_value_f32_.clear();

    return true;
}

void SparseLU::applyL32(std::span<float> x) const {
    // Apply L^{-1} in FP32, operating in elimination order.
    // x is indexed by elimination step.
    for (Index step = 0; step < dim_; ++step) {
        float wk = x[step];
        if (wk == 0.0f) {
            continue;
        }
        Index start = eta_start_[step];
        Index end = eta_start_[step + 1];
        for (Index k = start; k < end; ++k) {
            x[eta_target_[k]] -= eta_value_f32_[k] * wk;
        }
    }
}

void SparseLU::applyLTranspose32(std::span<float> x) const {
    // Apply L^{-T} in FP32, operating in elimination order.
    for (Index step = dim_ - 1; step >= 0; --step) {
        float sum = 0.0f;
        Index start = eta_start_[step];
        Index end = eta_start_[step + 1];
        for (Index k = start; k < end; ++k) {
            sum += eta_value_f32_[k] * x[eta_target_[k]];
        }
        x[step] -= sum;
    }
}

void SparseLU::solveU32(std::span<float> x) const {
    for (Index step = dim_ - 1; step >= 0; --step) {
        float rhs = x[step];
        Index start = u_start_[step];
        Index end = u_start_[step + 1];
        for (Index k = start; k < end; ++k) {
            rhs -= u_val_f32_[k] * x[u_col_[k]];
        }
        x[step] = rhs * u_diag_inv_f32_[step];
    }
}

void SparseLU::solveUTranspose32(std::span<float> x) const {
    for (Index step = 0; step < dim_; ++step) {
        x[step] *= u_diag_inv_f32_[step];
        float val = x[step];
        if (val == 0.0f) {
            continue;
        }
        Index start = u_start_[step];
        Index end = u_start_[step + 1];
        for (Index k = start; k < end; ++k) {
            x[u_col_[k]] -= u_val_f32_[k] * val;
        }
    }
}

void SparseLU::applyFT32(std::span<float> x) const {
    for (Index u = 0; u < num_updates_; ++u) {
        Index pos = ft_pivot_pos_[u];
        x[pos] *= ft_pivot_inv_f32_[u];
        float xp = x[pos];
        if (xp == 0.0f) {
            continue;
        }
        if (ft_is_dense_[u] != 0) {
            Index dense_off = ft_dense_offset_[u];
            for (Index i = 0; i < dim_; ++i) {
                x[i] -= ft_dense_value_f32_[static_cast<std::size_t>(dense_off + i)] * xp;
            }
            continue;
        }
        const Index start = ft_start_[u];
        const Index end = ft_start_[u + 1];
        for (Index k = start; k < end; ++k) {
            x[ft_index_[k]] -= ft_value_f32_[k] * xp;
        }
    }
}

void SparseLU::applyFTTranspose32(std::span<float> x) const {
    for (Index u = num_updates_ - 1; u >= 0; --u) {
        Index pos = ft_pivot_pos_[u];
        float sum = 0.0f;
        if (ft_is_dense_[u] != 0) {
            Index dense_off = ft_dense_offset_[u];
            for (Index i = 0; i < dim_; ++i) {
                sum += ft_dense_value_f32_[static_cast<std::size_t>(dense_off + i)] * x[i];
            }
        } else {
            const Index start = ft_start_[u];
            const Index end = ft_start_[u + 1];
            for (Index k = start; k < end; ++k) {
                sum += ft_value_f32_[k] * x[ft_index_[k]];
            }
        }
        x[pos] = (x[pos] - sum) * ft_pivot_inv_f32_[u];
    }
}

void SparseLU::ftranFp32(std::span<Real> rhs) const {
    // Allocate FP32 scratch.
    const auto dim_sz = static_cast<std::size_t>(dim_);
    if (solve_work_f32_.size() < dim_sz) {
        solve_work_f32_.resize(dim_sz);
    }
    std::span<float> work(solve_work_f32_.data(), dim_sz);

    // Step 1: w = P * b, cast to FP32.
    for (Index step = 0; step < dim_; ++step) {
        work[step] = static_cast<float>(rhs[row_perm_[step]]);
    }

    // Step 2: Apply L^{-1} in FP32.
    applyL32(work);

    // Step 3: Solve U in FP32.
    solveU32(work);

    // Step 4: x = Q * y, cast back to FP64.
    for (Index step = 0; step < dim_; ++step) {
        rhs[col_perm_[step]] = static_cast<Real>(work[step]);
    }

    // Step 5: Apply FT etas in FP32 (reuse solve_work_f32_ scratch).
    if (num_updates_ > 0) {
        for (Index i = 0; i < dim_; ++i) {
            solve_work_f32_[i] = static_cast<float>(rhs[i]);
        }
        applyFT32(std::span<float>(solve_work_f32_.data(), dim_sz));
        for (Index i = 0; i < dim_; ++i) {
            rhs[i] = static_cast<Real>(solve_work_f32_[i]);
        }
    }
}

void SparseLU::btranFp32(std::span<Real> rhs) const {
    const auto dim_sz = static_cast<std::size_t>(dim_);

    // Step 1: Apply FT transpose in FP32 (reuse solve_work_f32_ scratch).
    if (num_updates_ > 0) {
        if (solve_work_f32_.size() < dim_sz) {
            solve_work_f32_.resize(dim_sz);
        }
        for (Index i = 0; i < dim_; ++i) {
            solve_work_f32_[i] = static_cast<float>(rhs[i]);
        }
        applyFTTranspose32(std::span<float>(solve_work_f32_.data(), dim_sz));
        for (Index i = 0; i < dim_; ++i) {
            rhs[i] = static_cast<Real>(solve_work_f32_[i]);
        }
    }

    // Step 2: w = Q^T * rhs in FP32.
    if (solve_work_f32_.size() < dim_sz) {
        solve_work_f32_.resize(dim_sz);
    }
    std::span<float> work(solve_work_f32_.data(), dim_sz);
    for (Index step = 0; step < dim_; ++step) {
        work[step] = static_cast<float>(rhs[col_perm_[step]]);
    }

    // Step 3: Solve U^T in FP32.
    solveUTranspose32(work);

    // Step 4: Apply L^{-T} in FP32.
    applyLTranspose32(work);

    // Step 5: y = P^T * work, cast back to FP64.
    for (Index step = 0; step < dim_; ++step) {
        rhs[row_perm_[step]] = static_cast<Real>(work[step]);
    }
}

}  // namespace mipx
