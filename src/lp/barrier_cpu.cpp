// CPU Newton-step backends for the barrier solver:
//   CpuCholeskySolver  — Normal Equations + sparse LL' (Mosek/Gurobi style)
//   CpuAugmentedSolver — Augmented system + sparse LDL' (HiPO/Zanetti-Gondzio)

#include "newton_solver.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <set>
#include <vector>

namespace mipx {

namespace {

// ============================================================================
// Approximate Minimum Degree (AMD) ordering
// ============================================================================

void computeAmd(Index n,
                const std::vector<Index>& col_starts,
                const std::vector<Index>& row_indices,
                std::vector<Index>& perm,
                std::vector<Index>& iperm) {
    perm.resize(static_cast<size_t>(n));
    iperm.resize(static_cast<size_t>(n));
    if (n == 0) return;

    // Build symmetric adjacency lists (no self-loops).
    std::vector<std::set<Index>> adj(static_cast<size_t>(n));
    for (Index j = 0; j < n; ++j) {
        for (Index p = col_starts[j]; p < col_starts[j + 1]; ++p) {
            Index i = row_indices[p];
            if (i != j) {
                adj[j].insert(i);
                adj[i].insert(j);
            }
        }
    }

    std::vector<bool> eliminated(static_cast<size_t>(n), false);

    for (Index step = 0; step < n; ++step) {
        // Pick uneliminated node with minimum degree.
        Index best = -1;
        Index best_deg = n + 1;
        for (Index j = 0; j < n; ++j) {
            if (!eliminated[j]) {
                auto deg = static_cast<Index>(adj[j].size());
                if (deg < best_deg) {
                    best = j;
                    best_deg = deg;
                }
            }
        }

        perm[step] = best;
        iperm[best] = step;
        eliminated[best] = true;

        // Collect uneliminated neighbors.
        std::vector<Index> nbrs;
        nbrs.reserve(adj[best].size());
        for (Index nb : adj[best]) {
            if (!eliminated[nb]) nbrs.push_back(nb);
        }

        // Form clique among neighbors.
        for (size_t a = 0; a < nbrs.size(); ++a) {
            for (size_t b = a + 1; b < nbrs.size(); ++b) {
                adj[nbrs[a]].insert(nbrs[b]);
                adj[nbrs[b]].insert(nbrs[a]);
            }
        }

        // Remove eliminated node from neighbor lists.
        for (Index nb : adj[best]) adj[nb].erase(best);
        adj[best].clear();
    }
}

// ============================================================================
// Symbolic factorization
// ============================================================================

struct SymbolicFact {
    Index n = 0;
    std::vector<Index> perm;   // perm[new] = old
    std::vector<Index> iperm;  // iperm[old] = new

    // L in CSC, below-diagonal only.
    std::vector<Index> l_col_ptr;
    std::vector<Index> l_row_idx;
    Index l_nnz = 0;

    // Left-looking helper: for row i, columns k < i with L[i,k] != 0.
    std::vector<std::vector<Index>> l_row_to_cols;
};

// Permute a CSC lower-triangle pattern using iperm.
static void permutePattern(Index n,
                           const std::vector<Index>& orig_cptr,
                           const std::vector<Index>& orig_ridx,
                           const std::vector<Index>& iperm,
                           std::vector<Index>& perm_cptr,
                           std::vector<Index>& perm_ridx) {
    std::vector<std::vector<Index>> cols(static_cast<size_t>(n));
    for (Index j = 0; j < n; ++j) {
        for (Index p = orig_cptr[j]; p < orig_cptr[j + 1]; ++p) {
            Index i = orig_ridx[p];
            Index ni = iperm[i];
            Index nj = iperm[j];
            if (ni < nj) std::swap(ni, nj);
            cols[nj].push_back(ni);
        }
    }
    perm_cptr.resize(static_cast<size_t>(n + 1));
    perm_ridx.clear();
    perm_cptr[0] = 0;
    for (Index j = 0; j < n; ++j) {
        auto& c = cols[j];
        std::sort(c.begin(), c.end());
        c.erase(std::unique(c.begin(), c.end()), c.end());
        // Remove self-loop (diagonal) — we store below-diagonal only.
        c.erase(std::remove(c.begin(), c.end(), j), c.end());
        for (Index i : c) perm_ridx.push_back(i);
        perm_cptr[j + 1] = static_cast<Index>(perm_ridx.size());
    }
}

// Run AMD + symbolic Cholesky on a CSC lower-triangle pattern.
static void symbolicAnalyze(Index n,
                            const std::vector<Index>& col_starts,
                            const std::vector<Index>& row_indices,
                            SymbolicFact& sym) {
    sym.n = n;
    if (n == 0) {
        sym.perm.clear();
        sym.iperm.clear();
        sym.l_col_ptr.assign(1, 0);
        sym.l_row_idx.clear();
        sym.l_nnz = 0;
        sym.l_row_to_cols.clear();
        return;
    }

    // AMD.
    computeAmd(n, col_starts, row_indices, sym.perm, sym.iperm);

    // Permute pattern.
    std::vector<Index> pcptr, pridx;
    permutePattern(n, col_starts, row_indices, sym.iperm, pcptr, pridx);

    // Build row→column adjacency for etree computation.
    std::vector<std::vector<Index>> row_adj(static_cast<size_t>(n));
    for (Index j = 0; j < n; ++j) {
        for (Index p = pcptr[j]; p < pcptr[j + 1]; ++p) {
            Index i = pridx[p];
            if (i > j) row_adj[i].push_back(j);
        }
    }

    // Elimination tree.
    std::vector<Index> parent(static_cast<size_t>(n), -1);
    {
        std::vector<Index> ancestor(static_cast<size_t>(n), -1);
        for (Index k = 0; k < n; ++k) {
            for (Index j : row_adj[k]) {
                Index i = j;
                while (i != -1 && i != k) {
                    Index inext = ancestor[i];
                    ancestor[i] = k;
                    if (inext == -1) parent[i] = k;
                    i = inext;
                }
            }
        }
    }

    // Symbolic factorization: walk etree from each off-diagonal entry.
    sym.l_col_ptr.resize(static_cast<size_t>(n + 1));
    sym.l_row_idx.clear();
    sym.l_row_to_cols.assign(static_cast<size_t>(n), {});

    std::vector<Index> flag(static_cast<size_t>(n), -1);

    for (Index j = 0; j < n; ++j) {
        sym.l_col_ptr[j] = static_cast<Index>(sym.l_row_idx.size());
        flag[j] = j;

        for (Index p = pcptr[j]; p < pcptr[j + 1]; ++p) {
            Index i = pridx[p];
            if (i <= j) continue;
            Index k = i;
            while (k != -1 && flag[k] != j) {
                flag[k] = j;
                sym.l_row_idx.push_back(k);
                k = parent[k];
            }
        }

        // Sort this column.
        auto beg = sym.l_row_idx.begin() + sym.l_col_ptr[j];
        auto end = sym.l_row_idx.end();
        std::sort(beg, end);

        // Build row→col index.
        for (auto it = beg; it != end; ++it) {
            sym.l_row_to_cols[*it].push_back(j);
        }
    }
    sym.l_col_ptr[n] = static_cast<Index>(sym.l_row_idx.size());
    sym.l_nnz = static_cast<Index>(sym.l_row_idx.size());
}

// ============================================================================
// NE sparsity pattern: lower-triangle CSC of M = A*A' (structure only)
// ============================================================================

static void computeNePattern(const SparseMatrix& A, Index m, Index n,
                             std::vector<Index>& col_starts,
                             std::vector<Index>& row_indices) {
    // Column→row lists.
    std::vector<std::vector<Index>> col_rows(static_cast<size_t>(n));
    for (Index i = 0; i < m; ++i) {
        auto rv = A.row(i);
        for (Index k = 0; k < rv.size(); ++k) {
            col_rows[rv.indices[k]].push_back(i);
        }
    }

    // Pairs from each column → NE entries (lower triangle incl. diagonal).
    std::vector<std::set<Index>> ne_cols(static_cast<size_t>(m));
    for (Index k = 0; k < n; ++k) {
        auto& rows = col_rows[k];
        std::sort(rows.begin(), rows.end());
        rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
        for (size_t a = 0; a < rows.size(); ++a) {
            for (size_t b = a; b < rows.size(); ++b) {
                ne_cols[rows[a]].insert(rows[b]);  // rows[a] <= rows[b]
            }
        }
    }

    col_starts.resize(static_cast<size_t>(m + 1));
    row_indices.clear();
    col_starts[0] = 0;
    for (Index j = 0; j < m; ++j) {
        for (Index i : ne_cols[j]) {
            row_indices.push_back(i);
        }
        col_starts[j + 1] = static_cast<Index>(row_indices.size());
    }
}

// Merge-join rows of A to compute a single NE entry.
static Real mergeJoinNE(const SparseMatrix& A, Index row_i, Index row_j,
                        std::span<const Real> theta) {
    auto ri = A.row(row_i);
    auto rj = A.row(row_j);
    Real val = 0.0;
    Index pi = 0, pj = 0;
    while (pi < ri.size() && pj < rj.size()) {
        if (ri.indices[pi] == rj.indices[pj]) {
            val += ri.values[pi] * theta[ri.indices[pi]] * rj.values[pj];
            ++pi;
            ++pj;
        } else if (ri.indices[pi] < rj.indices[pj]) {
            ++pi;
        } else {
            ++pj;
        }
    }
    return val;
}

// ============================================================================
// Augmented sparsity pattern
// ============================================================================

// Lower-triangle CSC of K = [[-Theta^{-1}, A']; [A, delta*I]]
// Dimension: (n + m) × (n + m).
static void computeAugPattern(const SparseMatrix& A, Index m, Index n,
                              std::vector<Index>& col_starts,
                              std::vector<Index>& row_indices) {
    const Index dim = n + m;
    std::vector<std::vector<Index>> cols(static_cast<size_t>(dim));

    // Block (1,1): diagonal only, rows 0..n-1.
    for (Index j = 0; j < n; ++j) {
        cols[j].push_back(j);
    }

    // Block (2,1): A entries → row n+i, column j (lower triangle since n+i > j).
    for (Index i = 0; i < m; ++i) {
        auto rv = A.row(i);
        for (Index k = 0; k < rv.size(); ++k) {
            if (std::abs(rv.values[k]) > 0.0) {
                cols[rv.indices[k]].push_back(n + i);
            }
        }
    }

    // Block (2,2): diagonal only, rows n..n+m-1.
    for (Index i = 0; i < m; ++i) {
        cols[n + i].push_back(n + i);
    }

    // Convert to CSC.
    col_starts.resize(static_cast<size_t>(dim + 1));
    row_indices.clear();
    col_starts[0] = 0;
    for (Index j = 0; j < dim; ++j) {
        auto& c = cols[j];
        std::sort(c.begin(), c.end());
        c.erase(std::unique(c.begin(), c.end()), c.end());
        for (Index i : c) row_indices.push_back(i);
        col_starts[j + 1] = static_cast<Index>(row_indices.size());
    }
}

// ============================================================================
// Forward / backward solves
// ============================================================================

// Forward solve: L * x = b  (L lower triangular with explicit diagonal).
// Overwrites b in-place.
static void forwardSolveLL(const SymbolicFact& sym,
                           const std::vector<Real>& l_val,
                           const std::vector<Real>& l_diag,
                           std::span<Real> b) {
    const Index n = sym.n;
    for (Index j = 0; j < n; ++j) {
        b[j] /= l_diag[j];
        Real bj = b[j];
        for (Index p = sym.l_col_ptr[j]; p < sym.l_col_ptr[j + 1]; ++p) {
            b[sym.l_row_idx[p]] -= l_val[p] * bj;
        }
    }
}

// Backward solve: L' * x = b  (L' upper triangular).
// Overwrites b in-place.
static void backwardSolveLL(const SymbolicFact& sym,
                            const std::vector<Real>& l_val,
                            const std::vector<Real>& l_diag,
                            std::span<Real> b) {
    const Index n = sym.n;
    for (Index j = n - 1; j >= 0; --j) {
        for (Index p = sym.l_col_ptr[j]; p < sym.l_col_ptr[j + 1]; ++p) {
            b[j] -= l_val[p] * b[sym.l_row_idx[p]];
        }
        b[j] /= l_diag[j];
    }
}

// LDL' forward solve: L * x = b  (L unit lower triangular).
static void forwardSolveLDL(const SymbolicFact& sym,
                            const std::vector<Real>& l_val,
                            std::span<Real> b) {
    const Index n = sym.n;
    for (Index j = 0; j < n; ++j) {
        Real bj = b[j];
        for (Index p = sym.l_col_ptr[j]; p < sym.l_col_ptr[j + 1]; ++p) {
            b[sym.l_row_idx[p]] -= l_val[p] * bj;
        }
    }
}

// LDL' backward solve: L' * x = b  (L' unit upper triangular).
static void backwardSolveLDL(const SymbolicFact& sym,
                             const std::vector<Real>& l_val,
                             std::span<Real> b) {
    const Index n = sym.n;
    for (Index j = n - 1; j >= 0; --j) {
        for (Index p = sym.l_col_ptr[j]; p < sym.l_col_ptr[j + 1]; ++p) {
            b[j] -= l_val[p] * b[sym.l_row_idx[p]];
        }
    }
}

// LDL' diagonal scale: x = D^{-1} * b.
static void diagScaleLDL(const std::vector<Real>& d,
                         std::span<Real> b) {
    for (Index j = 0; j < static_cast<Index>(d.size()); ++j) {
        b[j] /= d[j];
    }
}

}  // anonymous namespace

// ============================================================================
// CpuCholeskySolver: Normal Equations + sparse LL'
// ============================================================================

class CpuCholeskySolver final : public NewtonSolver {
public:
    bool setup(const SparseMatrix& A, Index m, Index n,
               const BarrierOptions& opts) override {
        a_ = &A;
        m_ = m;
        n_ = n;
        ir_steps_ = opts.ir_steps;

        // NE sparsity pattern.
        std::vector<Index> ne_cptr, ne_ridx;
        computeNePattern(A, m, n, ne_cptr, ne_ridx);

        // Symbolic analysis.
        symbolicAnalyze(m, ne_cptr, ne_ridx, sym_);

        l_val_.resize(static_cast<size_t>(sym_.l_nnz), 0.0);
        l_diag_.resize(static_cast<size_t>(m), 0.0);
        theta_.resize(static_cast<size_t>(n), 0.0);
        s_copy_.resize(static_cast<size_t>(n), 0.0);

        return true;
    }

    bool factorize(std::span<const Real> z,
                   std::span<const Real> s, Real reg) override {
        reg_ = std::max(reg, 1e-12);

        for (Index j = 0; j < n_; ++j) {
            theta_[j] = z[j] / std::max(s[j], 1e-20);
            s_copy_[j] = s[j];
        }

        return numericLL();
    }

    bool solveNewton(std::span<const Real> rp,
                     std::span<const Real> rd,
                     std::span<const Real> rc,
                     std::span<Real> dz,
                     std::span<Real> dy,
                     std::span<Real> ds) override {
        // h[j] = rc[j]/s[j] - theta[j]*rd[j]
        std::vector<Real> h(static_cast<size_t>(n_));
        for (Index j = 0; j < n_; ++j) {
            Real sj = std::max(s_copy_[j], 1e-20);
            h[j] = rc[j] / sj - theta_[j] * rd[j];
        }

        // rhs = rp - A*h
        std::vector<Real> ah(static_cast<size_t>(m_), 0.0);
        a_->multiply(h, ah);
        std::vector<Real> rhs(static_cast<size_t>(m_));
        for (Index i = 0; i < m_; ++i) rhs[i] = rp[i] - ah[i];

        // Solve M*dy = rhs  (with iterative refinement)
        std::vector<Real> dy_perm(static_cast<size_t>(m_));
        for (Index i = 0; i < m_; ++i) dy_perm[sym_.iperm[i]] = rhs[i];
        forwardSolveLL(sym_, l_val_, l_diag_, dy_perm);
        backwardSolveLL(sym_, l_val_, l_diag_, dy_perm);

        // Iterative refinement.
        for (Int ir = 0; ir < ir_steps_; ++ir) {
            // residual = rhs - M*dy  (compute M*dy via A*Theta*A'*dy + reg*dy)
            std::vector<Real> dy_orig(static_cast<size_t>(m_));
            for (Index i = 0; i < m_; ++i) dy_orig[i] = dy_perm[sym_.perm[i]];

            std::vector<Real> at_dy(static_cast<size_t>(n_), 0.0);
            a_->multiplyTranspose(dy_orig, at_dy);
            for (Index j = 0; j < n_; ++j) at_dy[j] *= theta_[j];
            std::vector<Real> m_dy(static_cast<size_t>(m_), 0.0);
            a_->multiply(at_dy, m_dy);
            for (Index i = 0; i < m_; ++i) m_dy[i] += reg_ * dy_orig[i];

            // residual = rhs - m_dy
            std::vector<Real> resid(static_cast<size_t>(m_));
            for (Index i = 0; i < m_; ++i) resid[i] = rhs[i] - m_dy[i];

            // correction
            std::vector<Real> corr(static_cast<size_t>(m_));
            for (Index i = 0; i < m_; ++i) corr[sym_.iperm[i]] = resid[i];
            forwardSolveLL(sym_, l_val_, l_diag_, corr);
            backwardSolveLL(sym_, l_val_, l_diag_, corr);

            for (Index i = 0; i < m_; ++i) dy_perm[i] += corr[i];
        }

        // Unpermute dy.
        for (Index i = 0; i < m_; ++i) dy[i] = dy_perm[sym_.iperm[i]];

        // dz[j] = h[j] + theta[j] * (A'*dy)[j]
        std::vector<Real> at_dy(static_cast<size_t>(n_), 0.0);
        a_->multiplyTranspose(dy, at_dy);
        for (Index j = 0; j < n_; ++j) {
            dz[j] = h[j] + theta_[j] * at_dy[j];
        }

        // ds = rd - A'*dy
        for (Index j = 0; j < n_; ++j) {
            ds[j] = rd[j] - at_dy[j];
        }

        return true;
    }

private:
    // Left-looking numeric LL' factorization.
    bool numericLL() {
        const Index m = m_;
        std::fill(l_val_.begin(), l_val_.end(), 0.0);
        std::fill(l_diag_.begin(), l_diag_.end(), 0.0);

        std::vector<Real> w(static_cast<size_t>(m), 0.0);

        for (Index j = 0; j < m; ++j) {
            // Scatter: compute NE column j values into workspace.
            Index orig_j = sym_.perm[j];

            // Diagonal: M[orig_j, orig_j] = sum_k A[orig_j,k]^2 * theta[k] + reg
            w[j] = mergeJoinNE(*a_, orig_j, orig_j, theta_) + reg_;

            // Off-diagonal entries.
            for (Index p = sym_.l_col_ptr[j]; p < sym_.l_col_ptr[j + 1]; ++p) {
                Index i = sym_.l_row_idx[p];
                Index orig_i = sym_.perm[i];
                w[i] = mergeJoinNE(*a_, orig_i, orig_j, theta_);
            }

            // Column modifications from previous columns k with L[j,k] != 0.
            for (Index k : sym_.l_row_to_cols[j]) {
                // Find L[j,k].
                Real ljk = 0.0;
                for (Index p = sym_.l_col_ptr[k]; p < sym_.l_col_ptr[k + 1]; ++p) {
                    if (sym_.l_row_idx[p] == j) {
                        ljk = l_val_[p];
                        break;
                    }
                }

                // Diagonal: w[j] -= ljk^2
                w[j] -= ljk * ljk;

                // Off-diagonal: w[i] -= L[i,k] * ljk  for i > j in column k
                for (Index p = sym_.l_col_ptr[k]; p < sym_.l_col_ptr[k + 1]; ++p) {
                    Index i = sym_.l_row_idx[p];
                    if (i > j) {
                        w[i] -= l_val_[p] * ljk;
                    }
                }
            }

            // Pivot.
            if (w[j] < 1e-11) {
                w[j] += reg_;  // regularize small/negative pivot
            }
            if (w[j] <= 0.0) {
                w[j] = reg_;
            }
            l_diag_[j] = std::sqrt(w[j]);
            if (l_diag_[j] < 1e-30) return false;

            // Scale off-diagonal and store.
            Real inv_diag = 1.0 / l_diag_[j];
            for (Index p = sym_.l_col_ptr[j]; p < sym_.l_col_ptr[j + 1]; ++p) {
                Index i = sym_.l_row_idx[p];
                l_val_[p] = w[i] * inv_diag;
                w[i] = 0.0;
            }
            w[j] = 0.0;
        }

        return true;
    }

    const SparseMatrix* a_ = nullptr;
    Index m_ = 0, n_ = 0;
    Int ir_steps_ = 2;
    Real reg_ = 1e-8;

    SymbolicFact sym_;
    std::vector<Real> l_val_;
    std::vector<Real> l_diag_;
    std::vector<Real> theta_;
    std::vector<Real> s_copy_;
};

// ============================================================================
// CpuAugmentedSolver: Augmented system + sparse LDL'
// ============================================================================

class CpuAugmentedSolver final : public NewtonSolver {
public:
    bool setup(const SparseMatrix& A, Index m, Index n,
               const BarrierOptions& opts) override {
        a_ = &A;
        m_ = m;
        n_ = n;
        ir_steps_ = opts.ir_steps;
        dim_ = n + m;

        // Augmented sparsity pattern.
        std::vector<Index> aug_cptr, aug_ridx;
        computeAugPattern(A, m, n, aug_cptr, aug_ridx);

        // Symbolic analysis.
        symbolicAnalyze(dim_, aug_cptr, aug_ridx, sym_);

        l_val_.resize(static_cast<size_t>(sym_.l_nnz), 0.0);
        d_.resize(static_cast<size_t>(dim_), 0.0);

        return true;
    }

    bool factorize(std::span<const Real> z,
                   std::span<const Real> s, Real reg) override {
        reg_ = std::max(reg, 1e-12);

        z_copy_.assign(z.begin(), z.end());
        s_copy_.assign(s.begin(), s.end());

        return numericLDL();
    }

    bool solveNewton(std::span<const Real> rp,
                     std::span<const Real> rd,
                     std::span<const Real> rc,
                     std::span<Real> dz,
                     std::span<Real> dy,
                     std::span<Real> ds) override {
        // Form augmented RHS:
        //   rhs_x[j] = rd[j] - rc[j]/z[j]   for j = 0..n-1
        //   rhs_y[i] = rp[i]                  for i = 0..m-1
        std::vector<Real> rhs(static_cast<size_t>(dim_));
        for (Index j = 0; j < n_; ++j) {
            rhs[j] = rd[j] - rc[j] / std::max(z_copy_[j], 1e-20);
        }
        for (Index i = 0; i < m_; ++i) {
            rhs[n_ + i] = rp[i];
        }

        // Permute RHS.
        std::vector<Real> rhs_perm(static_cast<size_t>(dim_));
        for (Index i = 0; i < dim_; ++i) rhs_perm[sym_.iperm[i]] = rhs[i];

        // Solve K * sol = rhs  via  L * D * L' * sol = rhs.
        forwardSolveLDL(sym_, l_val_, rhs_perm);
        diagScaleLDL(d_, rhs_perm);
        backwardSolveLDL(sym_, l_val_, rhs_perm);

        // Iterative refinement.
        for (Int ir = 0; ir < ir_steps_; ++ir) {
            // Unpermute current solution.
            std::vector<Real> sol(static_cast<size_t>(dim_));
            for (Index i = 0; i < dim_; ++i) sol[i] = rhs_perm[sym_.perm[i]];

            // Compute K * sol.
            std::vector<Real> ksol(static_cast<size_t>(dim_), 0.0);
            // Block (1,1): -Theta^{-1} * sol_x
            for (Index j = 0; j < n_; ++j) {
                ksol[j] = -(s_copy_[j] / std::max(z_copy_[j], 1e-20)) * sol[j];
            }
            // Block (1,2): A' * sol_y
            std::span<const Real> sol_y(sol.data() + n_,
                                        static_cast<size_t>(m_));
            std::vector<Real> at_sol_y(static_cast<size_t>(n_), 0.0);
            a_->multiplyTranspose(sol_y, at_sol_y);
            for (Index j = 0; j < n_; ++j) ksol[j] += at_sol_y[j];

            // Block (2,1): A * sol_x
            std::span<const Real> sol_x(sol.data(), static_cast<size_t>(n_));
            std::vector<Real> a_sol_x(static_cast<size_t>(m_), 0.0);
            a_->multiply(sol_x, a_sol_x);
            for (Index i = 0; i < m_; ++i) {
                ksol[n_ + i] = a_sol_x[i] + reg_ * sol[n_ + i];
            }

            // Residual = rhs - K*sol.
            std::vector<Real> resid(static_cast<size_t>(dim_));
            for (Index i = 0; i < dim_; ++i) resid[i] = rhs[i] - ksol[i];

            // Solve correction.
            std::vector<Real> corr(static_cast<size_t>(dim_));
            for (Index i = 0; i < dim_; ++i) corr[sym_.iperm[i]] = resid[i];
            forwardSolveLDL(sym_, l_val_, corr);
            diagScaleLDL(d_, corr);
            backwardSolveLDL(sym_, l_val_, corr);

            for (Index i = 0; i < dim_; ++i) rhs_perm[i] += corr[i];
        }

        // Unpermute and recover.
        std::vector<Real> sol(static_cast<size_t>(dim_));
        for (Index i = 0; i < dim_; ++i) sol[i] = rhs_perm[sym_.perm[i]];

        // dz = sol[0..n-1], dy = sol[n..n+m-1]
        for (Index j = 0; j < n_; ++j) dz[j] = sol[j];
        for (Index i = 0; i < m_; ++i) dy[i] = sol[n_ + i];

        // ds = (rc - s*dz) / z
        for (Index j = 0; j < n_; ++j) {
            ds[j] = (rc[j] - s_copy_[j] * dz[j]) / std::max(z_copy_[j], 1e-20);
        }

        return true;
    }

private:
    // Left-looking numeric LDL' (unit diagonal L, diagonal D can be negative).
    bool numericLDL() {
        std::fill(l_val_.begin(), l_val_.end(), 0.0);
        std::fill(d_.begin(), d_.end(), 0.0);

        std::vector<Real> w(static_cast<size_t>(dim_), 0.0);

        for (Index j = 0; j < dim_; ++j) {
            // Scatter augmented matrix column j into w.
            scatterAugColumn(j, w);

            // Column modifications: for each k < j with L[j,k] != 0.
            for (Index k : sym_.l_row_to_cols[j]) {
                Real ljk = 0.0;
                for (Index p = sym_.l_col_ptr[k]; p < sym_.l_col_ptr[k + 1];
                     ++p) {
                    if (sym_.l_row_idx[p] == j) {
                        ljk = l_val_[p];
                        break;
                    }
                }

                Real ljk_dk = ljk * d_[k];

                // Diagonal: w[j] -= ljk * D[k] * ljk
                w[j] -= ljk_dk * ljk;

                // Off-diagonal: w[i] -= L[i,k] * D[k] * ljk
                for (Index p = sym_.l_col_ptr[k]; p < sym_.l_col_ptr[k + 1];
                     ++p) {
                    if (sym_.l_row_idx[p] > j) {
                        w[sym_.l_row_idx[p]] -= l_val_[p] * ljk_dk;
                    }
                }
            }

            // Pivot (D[j] = w[j], can be negative for indefinite systems).
            d_[j] = w[j];

            // Per-pivot regularization (HiPO-style).
            constexpr Real eps_s = 1e-12;  // static
            constexpr Real eps_d = 1e-7;   // dynamic boost
            if (std::abs(d_[j]) < eps_d) {
                // Determine sign: variables should be negative, constraints positive.
                Index orig = sym_.perm[j];
                Real sign = (orig < n_) ? -1.0 : 1.0;
                d_[j] = sign * std::max(std::abs(d_[j]) + eps_d, eps_d);
            }
            d_[j] += (d_[j] >= 0 ? eps_s : -eps_s);

            if (std::abs(d_[j]) < 1e-30) return false;

            // Scale off-diagonal.
            Real inv_d = 1.0 / d_[j];
            for (Index p = sym_.l_col_ptr[j]; p < sym_.l_col_ptr[j + 1];
                 ++p) {
                Index i = sym_.l_row_idx[p];
                l_val_[p] = w[i] * inv_d;
                w[i] = 0.0;
            }
            w[j] = 0.0;
        }

        return true;
    }

    // Scatter one column of the augmented matrix into dense workspace.
    void scatterAugColumn(Index j, std::vector<Real>& w) const {
        Index orig_j = sym_.perm[j];

        if (orig_j < n_) {
            // Variable column: diagonal = -s/z  (i.e., -Theta^{-1}).
            w[j] = -(s_copy_[orig_j] / std::max(z_copy_[orig_j], 1e-20));

            // A entries: for each row i with A[i, orig_j] != 0,
            // augmented row = n_ + i, permuted row = iperm[n_ + i].
            auto cv = a_->col(orig_j);
            for (Index k = 0; k < cv.size(); ++k) {
                Index aug_row = n_ + cv.indices[k];
                Index perm_row = sym_.iperm[aug_row];
                if (perm_row > j) {
                    w[perm_row] = cv.values[k];
                }
                // If perm_row < j, this entry was already handled as an
                // above-diagonal entry in column perm_row.
            }
        } else {
            // Constraint column: diagonal = reg (δI block).
            w[j] = reg_;
        }
    }

    const SparseMatrix* a_ = nullptr;
    Index m_ = 0, n_ = 0, dim_ = 0;
    Int ir_steps_ = 2;
    Real reg_ = 1e-8;

    SymbolicFact sym_;
    std::vector<Real> l_val_;
    std::vector<Real> d_;

    std::vector<Real> z_copy_;
    std::vector<Real> s_copy_;
};

// ============================================================================
// Factory functions
// ============================================================================

std::unique_ptr<NewtonSolver> createCpuCholeskySolver() {
    return std::make_unique<CpuCholeskySolver>();
}

std::unique_ptr<NewtonSolver> createCpuAugmentedSolver() {
    return std::make_unique<CpuAugmentedSolver>();
}

}  // namespace mipx
