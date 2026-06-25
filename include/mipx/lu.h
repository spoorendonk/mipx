#pragma once

#include "mipx/core.h"
#include "mipx/work_units.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <span>
#include <vector>

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
    void btran(std::span<Real> rhs, std::vector<Index>& nonzero_rows) const;

    /// Forrest-Tomlin rank-1 update: replace basis column at position `pivot_pos`
    /// with new column `entering_col` (given in terms of original row indices/values).
    void update(Index pivot_pos, std::span<const Index> indices, std::span<const Real> values);
    /// Forrest-Tomlin rank-1 update with pre-transformed column `d = B^{-1} a_q`.
    /// `transformed_col` is in basis-position order and must have size dimension().
    void updateFromFtranColumn(Index pivot_pos, std::span<const Real> transformed_col);

    /// Check if refactorization is needed.
    [[nodiscard]] bool needsRefactorization() const;
    [[nodiscard]] Index numUpdates() const { return num_updates_; }
    [[nodiscard]] Index dimension() const { return dim_; }
    void setMaxUpdates(Index limit) { max_updates_ = std::max<Index>(1, limit); }
    [[nodiscard]] Index maxUpdates() const { return max_updates_; }
    void setFtDropTolerance(Real tol) { ft_drop_tol_ = std::max<Real>(0.0, tol); }
    [[nodiscard]] Real ftDropTolerance() const { return ft_drop_tol_; }

    /// Enable/disable mixed-precision mode (FP32 factorization + FP64 iterative refinement).
    void setMixedPrecision(bool enable) { mixed_precision_enabled_ = enable; }
    [[nodiscard]] bool mixedPrecision() const { return mixed_precision_active_; }
    [[nodiscard]] bool mixedPrecisionEnabled() const { return mixed_precision_enabled_; }

    /// Access work unit counter.
    [[nodiscard]] const WorkUnits& workUnits() const { return work_; }
    void resetWorkUnits() { work_.reset(); }

private:
    /// Apply Forrest-Tomlin update etas forward.
    void applyFT(std::span<Real> x) const;

    /// Apply Forrest-Tomlin update etas in reverse, with density-guided switching.
    void applyFTTranspose(std::span<Real> x) const;

    void btranImpl(std::span<Real> rhs, std::vector<Index>* nonzero_rows) const;

    /// Backward solve with U (upper triangular, row-wise).
    void solveU(std::span<Real> x) const;

    /// Forward solve with U^T.
    void solveUTranspose(std::span<Real> x) const;

    /// Hyper-sparse forward solve with U^T: only process columns reachable from
    /// the nonzero positions in x. Returns the number of output nonzeros.
    Index solveUTransposeSparse(std::span<Real> x) const;

    /// BTF detection: find block upper-triangular form via maximum matching
    /// and Tarjan's SCC algorithm. Returns the number of blocks found.
    /// Populates btf_row_perm and btf_col_perm with the BTF ordering, and
    /// btf_block_start with the block boundaries.
    static Index detectBTF(Index dim, const SparseMatrix& matrix, std::span<const Index> basis_cols,
                           std::vector<Index>& btf_row_perm, std::vector<Index>& btf_col_perm,
                           std::vector<Index>& btf_block_start);

    // --- FP32 mixed-precision solve helpers ---

    /// Apply L eta vectors forward in FP32.
    void applyL32(std::span<float> x) const;

    /// Apply L^T eta vectors backward in FP32.
    void applyLTranspose32(std::span<float> x) const;

    /// Backward solve with U in FP32.
    void solveU32(std::span<float> x) const;

    /// Forward solve with U^T in FP32.
    void solveUTranspose32(std::span<float> x) const;

    /// Apply Forrest-Tomlin update etas forward in FP32.
    void applyFT32(std::span<float> x) const;

    /// Apply Forrest-Tomlin update etas in reverse in FP32.
    void applyFTTranspose32(std::span<float> x) const;

    /// Full FP32 FTRAN solve (no permutation, operates in elimination order).
    void ftranFp32(std::span<Real> rhs) const;

    /// Full FP32 BTRAN solve.
    void btranFp32(std::span<Real> rhs) const;

    /// Build FP32 copies of L and U factors from FP64 data.
    /// Returns false if element growth exceeds FP32 range.
    bool buildFp32Factors();

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
    // Elimination-order target index for each eta entry (same size as eta_index_).
    std::vector<Index> eta_target_;
    // Reverse adjacency: for elimination index t,
    // eta_rev_src_[eta_rev_start_[t]..eta_rev_start_[t+1]) are steps s such that eta has an edge s
    // -> t.
    std::vector<Index> eta_rev_start_;
    std::vector<Index> eta_rev_src_;

    // Supernodal grouping for dense panel L-solve.
    // A supernode is a group of consecutive elimination steps where the eta
    // patterns are nested. For supernodes >= kSupernodeMinWidth, we store a
    // dense panel for efficient application.
    static constexpr Index kSupernodeMinWidth = 4;
    struct Supernode {
        Index start;         // first elimination step
        Index width;         // number of steps in the supernode
        Index panel_rows;    // number of affected rows (dense panel height)
        Index panel_offset;  // offset into snode_panel_values_
        // Row indices offset into snode_panel_row_indices_.
        Index row_offset;
    };
    std::vector<Supernode> supernodes_;
    // Dense panel values: column-major, panel_rows x width.
    std::vector<Real> snode_panel_values_;
    // Row indices for each supernode (in original row space).
    std::vector<Index> snode_panel_row_indices_;

    // U stored row-wise (in elimination order).
    // u_start_[k] .. u_start_[k+1] gives the range for row k.
    // u_col_[p] is the column index (in elimination order), u_val_[p] the value.
    // Diagonal is stored as the last entry in each row.
    std::vector<Index> u_start_;
    std::vector<Index> u_col_;
    std::vector<Real> u_val_;
    std::vector<Real> u_diag_;  // u_diag_[k] = U(k,k)
    std::vector<Real> u_diag_inv_;

    // Forrest-Tomlin update etas.
    // Each update stores an eta vector and the position of the replaced column.
    std::vector<Index> ft_start_;
    std::vector<Index> ft_index_;
    std::vector<Real> ft_value_;
    std::vector<Index> ft_pivot_pos_;  // column position in elimination order
    std::vector<Real> ft_pivot_val_;   // new diagonal value
    std::vector<Real> ft_pivot_inv_;
    std::vector<uint8_t> ft_is_dense_;
    std::vector<Index> ft_dense_offset_;
    std::vector<Real> ft_dense_value_;
    uint64_t ft_dense_nnz_ = 0;

    Index num_updates_ = 0;

    // Reusable dense scratch buffers for hot-path solves/updates.
    mutable std::vector<Real> solve_work_;
    std::vector<Real> update_work_;
    std::vector<Index> update_touched_;
    mutable std::vector<Index> sparse_steps_;
    mutable std::vector<uint32_t> sparse_epoch_;
    mutable uint32_t sparse_epoch_id_ = 1;

    Index max_updates_ = 500;
    static constexpr Real kPivotTol = 0.1;
    static constexpr Real kZeroTol = 1e-13;
    static constexpr Real kGrowthLimit = 1e12;
    // Floor on update count before the cost-based refactor trigger applies.
    // Prevents repeated tiny re-factorizations when ft_index_ briefly bursts.
    static constexpr Index kMinUpdatesForCostRefactor = 8;
    // Upper bound on basis dimension for the cost-based refactor trigger.
    // The trigger targets tiny bases (e.g. dim ~74) where max_updates=500 lets
    // Forrest-Tomlin etas balloon to many times the base eta+U cost before a
    // refactor fires. On larger bases a refactorization is O(nnz)-expensive and
    // firing it speculatively (the 2x-base ratio is easy to reach when L/U are
    // sparse) causes a refactor storm — this regressed MIPLIB "gen" (dim ~780)
    // from optimal to a 20s timeout. Restrict the trigger to small bases.
    static constexpr Index kCostRefactorMaxDim = 256;
    static constexpr Index kHyperSparseMinDim = 256;
    static constexpr Real kHyperSparseMaxDensity = 0.10;
    static constexpr Index kFtDenseMinDim = 512;
    static constexpr Real kFtDenseThreshold = 0.85;

    /// Exponential moving average of solve density per stage.
    /// Stages: 0 = FTRAN L-solve, 1 = BTRAN L^T-solve, 2 = BTRAN U^T-solve,
    ///         3 = BTRAN FT-transpose.
    static constexpr int kNumSolveStages = 4;
    static constexpr Real kEmaAlpha = 0.15;  // smoothing factor
    mutable Real ema_density_[kNumSolveStages] = {0.0, 0.0, 0.0, 0.0};

    Real max_u_entry_ = 0.0;  // track growth
    Real ft_drop_tol_ = 1e-13;

    // --- Mixed-precision (FP32) state ---
    bool mixed_precision_enabled_ = false;  // user opt-in
    mutable bool mixed_precision_active_ =
        false;  // actually using FP32 (mutable: fallback in const solves)

    // Stored references for residual computation during iterative refinement.
    // Lifetime: the matrix must outlive the SparseLU (caller's responsibility).
    const SparseMatrix* ir_matrix_ = nullptr;
    std::vector<Index> ir_basis_cols_;

    // FP32 copies of L eta vectors.
    std::vector<float> eta_value_f32_;

    // FP32 copies of U values.
    std::vector<float> u_val_f32_;
    std::vector<float> u_diag_f32_;
    std::vector<float> u_diag_inv_f32_;

    // FP32 copies of Forrest-Tomlin update etas.
    std::vector<float> ft_value_f32_;
    std::vector<float> ft_pivot_val_f32_;
    std::vector<float> ft_pivot_inv_f32_;
    std::vector<float> ft_dense_value_f32_;

    // FP32 scratch buffers for solve.
    mutable std::vector<float> solve_work_f32_;

    // Iterative refinement scratch buffers (mutable to avoid allocation per solve).
    mutable std::vector<Real> ir_residual_;
    mutable std::vector<Real> ir_z_;
    mutable std::vector<Real> ir_rhs_save_;

    // Iterative refinement tracking.
    static constexpr Real kFp32GrowthLimit = 1e4;  // FP32 growth -> fallback to FP64
    static constexpr int kMaxRefinementSteps = 3;  // max IR steps per solve
    static constexpr Real kRefinementTol = 1e-10;  // residual convergence tolerance
    static constexpr int kFallbackRefinementCount =
        3;                               // if IR consistently needs this many, fall back
    mutable int ir_step_ema_count_ = 0;  // exponential moving count of refinement steps needed
    mutable int ir_solves_count_ = 0;    // number of solves performed with IR

    // Deterministic work counter.
    mutable WorkUnits work_;
};

}  // namespace mipx
