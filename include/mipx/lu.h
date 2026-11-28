#pragma once

#include "mipx/core.h"
#include "mipx/work_units.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <span>
#include <vector>

namespace mipx {

class Deadline;
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

    /// Minimum basis dimension at which BTF detection is attempted. Below this,
    /// the matching + Tarjan SCC scan costs more than a single-block Markowitz
    /// factorization saves.
    static constexpr Index kBtfMinDim = 100;

    /// Minimum basis dimension at which supernode detection is attempted.
    static constexpr Index kSupernodeDetectMinDim = 64;

    /// Minimum number of consecutive elimination steps needed before a group of
    /// nested eta patterns is stored as a dense panel.
    static constexpr Index kSupernodeMinWidth = 4;

    /// Factorize a square basis matrix (given as columns from constraint matrix).
    /// basis_cols[i] = column index in the original matrix for basis position i.
    void factorize(const SparseMatrix& matrix, std::span<const Index> basis_cols);

    /// Like factorize(), but returns false if the deadline stops the build.
    /// After false, factorize again before using the factors.
    [[nodiscard]] bool factorizeUntil(const SparseMatrix& matrix, std::span<const Index> basis_cols,
                                      const Deadline& deadline);

    /// Solve B*x = rhs in-place.
    void ftran(std::span<Real> rhs) const;

    /// Solve B^T*y = rhs in-place.
    void btran(std::span<Real> rhs) const;
    /// As btran(rhs), and `nonzero_rows` is set to the rows whose result exceeds
    /// 1e-13 in magnitude (an output only: its content on entry is ignored).
    void btran(std::span<Real> rhs, std::vector<Index>& nonzero_rows) const;

    /// Solve B*x = rhs in-place, given the right-hand side's nonzero pattern
    /// and returning the result's, so the solve costs the patterns it walks
    /// rather than the dimension (hyper-sparse, Gilbert-Peierls reach).
    ///
    /// On entry `pattern` lists the rows where rhs may be nonzero, and rhs is
    /// zero at every other row. On return rhs holds x, and `pattern` lists,
    /// without duplicates, every basis position where x is nonzero; x is
    /// exactly zero elsewhere. No value is dropped, however small: the result
    /// is the dense solve's up to round-off. The L and U stages each run
    /// hyper-sparse or dense on their predicted result density; the update
    /// etas go by pattern (an eta stored dense costs the dimension).
    void ftranSparse(std::span<Real> rhs, std::vector<Index>& pattern) const;

    /// Solve B^T*y = rhs in-place by pattern, as ftranSparse: `pattern` lists
    /// basis positions on entry and rows on return. The update etas go by
    /// their row-wise copy, so they cost the entries in the rows the result
    /// reaches rather than all eta nonzeros.
    void btranSparse(std::span<Real> rhs, std::vector<Index>& pattern) const;

    /// Forrest-Tomlin rank-1 update: replace basis column at position `pivot_pos`
    /// with new column `entering_col` (given in terms of original row indices/values).
    void update(Index pivot_pos, std::span<const Index> indices, std::span<const Real> values);
    /// Forrest-Tomlin rank-1 update with pre-transformed column `d = B^{-1} a_q`.
    /// `transformed_col` is in basis-position order and must have size dimension().
    void updateFromFtranColumn(Index pivot_pos, std::span<const Real> transformed_col);
    /// As above, with `pattern` listing without duplicates every position
    /// where `transformed_col` is nonzero (ftranSparse's output pattern), so
    /// only those are read.
    void updateFromFtranColumn(Index pivot_pos, std::span<const Real> transformed_col,
                               std::span<const Index> pattern);

    /// Test-only: how each triangular stage chooses between its hyper-sparse
    /// and dense kernels. Auto decides on predicted density; the forced modes
    /// exist so tests can check both kernels on the same basis.
    enum class SolveKernel : uint8_t { Auto, AlwaysHyperSparse, AlwaysDense };
    void setSolveKernel(SolveKernel kernel) { solve_kernel_ = kernel; }
    [[nodiscard]] SolveKernel solveKernel() const { return solve_kernel_; }

    /// Why a refactorization is needed, the first of the tests that holds
    /// (#517): the update count reached its limit, an updated U entry passed
    /// the growth limit, or the Forrest-Tomlin updates cost more than twice
    /// the base factors (small bases only).
    enum class RefactorReason : uint8_t { None, UpdateLimit, Growth, UpdateCost };
    [[nodiscard]] RefactorReason refactorReason() const;
    /// Check if refactorization is needed.
    [[nodiscard]] bool needsRefactorization() const {
        return refactorReason() != RefactorReason::None;
    }
    [[nodiscard]] Index numUpdates() const { return num_updates_; }
    [[nodiscard]] Index dimension() const { return dim_; }
    void setMaxUpdates(Index limit) { max_updates_ = std::max<Index>(1, limit); }
    [[nodiscard]] Index maxUpdates() const { return max_updates_; }
    void setFtDropTolerance(Real tol) { ft_drop_tol_ = std::max<Real>(0.0, tol); }
    [[nodiscard]] Real ftDropTolerance() const { return ft_drop_tol_; }

    /// Use thresholded Markowitz search on large bases and search longer
    /// before accepting a pivot. Used by the opt-in barrier crossover pushes.
    void setLowFillPivotSearch(bool enable) { low_fill_pivot_search_ = enable; }

    /// Enable/disable mixed-precision mode (FP32 factorization + FP64 iterative refinement).
    void setMixedPrecision(bool enable) { mixed_precision_enabled_ = enable; }
    [[nodiscard]] bool mixedPrecision() const { return mixed_precision_active_; }
    [[nodiscard]] bool mixedPrecisionEnabled() const { return mixed_precision_enabled_; }

    /// Enable/disable BTF block-wise elimination: when on, the Markowitz pivot
    /// search is confined to one BTF diagonal block at a time, so each block is
    /// factorized independently.
    ///
    /// OFF BY DEFAULT, pending the benchmark in issue #179. Confinement makes
    /// the factorization itself cheaper (the search only ever ranges over one
    /// block) but gives up the global Markowitz freedom that keeps the
    /// off-diagonal region sparse. Measured on the Netlib bases that reach
    /// kBtfMinDim, it costs 0.7-14% more L+U nonzeros -- and L+U nonzeros are
    /// the per-iteration FTRAN/BTRAN cost, paid thousands of times per LP,
    /// against a one-off factorization gain. Turning it into a win needs the
    /// off-diagonal blocks left unfactorized and handled by block forward/back
    /// substitution in the solves, which is a change to the solve engine rather
    /// than to pivot selection.
    ///
    /// With this off, factorize() behaves exactly as it did before block-wise
    /// elimination existed: BTF is still detected and still used to relabel
    /// rows/columns, only the pivot confinement is skipped.
    void setBtfBlockElimination(bool enable) { btf_block_elimination_ = enable; }
    [[nodiscard]] bool btfBlockElimination() const { return btf_block_elimination_; }

    /// Number of BTF diagonal blocks the last factorization was *confined to*.
    /// 1 means no confinement happened -- block-wise elimination is disabled,
    /// the basis was below kBtfMinDim, or no usable block structure was found --
    /// and a single global Markowitz pass was used.
    [[nodiscard]] Index numBtfBlocks() const { return num_btf_blocks_; }

    /// Number of dense L panels (supernodes of width >= kSupernodeMinWidth)
    /// built by the last factorization.
    [[nodiscard]] Index numSupernodes() const { return static_cast<Index>(supernodes_.size()); }

    /// Test-only: force the scalar eta L/L^T solve by suppressing supernode
    /// detection. This is not a tuning knob -- it exists so tests can check the
    /// dense-panel path against the scalar path on the same basis. Production
    /// callers should leave it enabled.
    void setSupernodalEnabled(bool enable) { supernodal_enabled_ = enable; }
    [[nodiscard]] bool supernodalEnabled() const { return supernodal_enabled_; }

    /// rowPermutation()[k] = original matrix row eliminated at step k.
    [[nodiscard]] std::span<const Index> rowPermutation() const { return row_perm_; }
    /// colPermutation()[k] = basis position (index into basis_cols) pivoted at step k.
    [[nodiscard]] std::span<const Index> colPermutation() const { return col_perm_; }

    /// Access work unit counter.
    [[nodiscard]] const WorkUnits& workUnits() const { return work_; }
    void resetWorkUnits() { work_.reset(); }

private:
    /// Solve stages whose result density is predicted (index into ema_density_).
    enum SolveStage : uint8_t {
        kStageFtranL = 0,
        kStageBtranLTranspose = 1,
        kStageBtranUTranspose = 2,
        kStageFtranU = 3,
    };

    /// Apply Forrest-Tomlin update etas forward (dense x).
    void applyFT(std::span<Real> x) const;

    /// Apply Forrest-Tomlin update etas in reverse (dense x).
    void applyFTTranspose(std::span<Real> x) const;

    /// The same by pattern: `pattern` lists where x may be nonzero, and is
    /// extended by the positions the etas make nonzero.
    void applyFTSparse(std::span<Real> x, std::vector<Index>& pattern) const;
    void applyFTTransposeSparse(std::span<Real> x, std::vector<Index>& pattern) const;
    /// applyFTTransposeSparse's hyper-sparse kernel, by the etas' row-wise copy.
    /// `mark`/`id` must be a fresh epoch, nothing marked: the kernel marks the
    /// positions of `pattern` itself, and skips one it finds already marked
    /// as a repeat.
    void applyFTTransposeByRow(std::span<Real> x, std::vector<Index>& pattern, uint32_t* mark,
                               uint32_t id) const;

    /// Dense L and L^T solves in elimination order (supernodal when panels exist).
    void solveLDense(std::span<Real> x) const;
    void solveLTransposeDense(std::span<Real> x) const;

    /// Backward solve with U (upper triangular, row-wise).
    void solveU(std::span<Real> x) const;

    /// Forward solve with U^T.
    void solveUTranspose(std::span<Real> x) const;

    /// One triangular stage in elimination order by pattern: hyper-sparse over
    /// the reach of `pattern`, or the dense kernel; `pattern` is replaced by
    /// the stage result's pattern either way.
    void solveLStage(std::span<Real> x, std::vector<Index>& pattern) const;
    void solveUStage(std::span<Real> x, std::vector<Index>& pattern) const;
    void solveUTransposeStage(std::span<Real> x, std::vector<Index>& pattern) const;
    void solveLTransposeStage(std::span<Real> x, std::vector<Index>& pattern) const;

    /// Whether `stage` runs hyper-sparse on an input of `nnz` nonzeros.
    [[nodiscard]] bool useHyperSparse(SolveStage stage, std::size_t nnz) const;
    /// Fold a stage's result size into its density prediction.
    void recordDensity(SolveStage stage, std::size_t nnz) const;

    /// Depth-first reach of `seeds` in the graph whose node v has successors
    /// adj[adj_start[v] .. adj_start[v+1]). On success sparse_steps_ holds the
    /// reached nodes in postorder (so its reverse is a topological order:
    /// every node before its successors). Returns false, leaving the reach
    /// incomplete, once more than `limit` nodes are reached.
    bool reach(std::span<const Index> seeds, const Index* adj_start, const Index* adj,
               Index limit) const;
    /// Node count past which a hyper-sparse stage gives up for the dense kernel.
    [[nodiscard]] Index reachLimit() const;
    /// Start a fresh marking in sparse_epoch_ (all entries unmarked).
    void newMarkEpoch() const;

    /// Mixed-precision (FP32 + iterative refinement) solves on dense rhs.
    void ftranMixed(std::span<Real> rhs) const;
    void btranMixed(std::span<Real> rhs) const;

    /// BTF detection: find block upper-triangular form via maximum matching
    /// and Tarjan's SCC algorithm. Returns the number of blocks found.
    /// Populates btf_row_perm and btf_col_perm with the BTF ordering, and
    /// btf_block_start with the block boundaries.
    static Index detectBTF(Index dim, const SparseMatrix& matrix, std::span<const Index> basis_cols,
                           std::vector<Index>& btf_row_perm, std::vector<Index>& btf_col_perm,
                           std::vector<Index>& btf_block_start, const Deadline& deadline);

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
    bool buildFp32Factors(const Deadline& deadline);

    Index dim_ = 0;
    // Number of BTF diagonal blocks the last factorization was confined to
    // (1 = no usable block structure, i.e. one global Markowitz pass).
    Index num_btf_blocks_ = 1;

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
    struct Supernode {
        Index start;         // first elimination step
        Index width;         // number of steps in the supernode
        Index panel_rows;    // number of affected rows (dense panel height)
        Index panel_offset;  // offset into snode_panel_values_
        // Row indices offset into snode_panel_row_indices_.
        Index row_offset;
    };
    std::vector<Supernode> supernodes_;
    bool supernodal_enabled_ = true;
    // BTF block-wise elimination: off by default, see setBtfBlockElimination().
    bool btf_block_elimination_ = false;
    bool low_fill_pivot_search_ = false;
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
    // U's off-diagonal entries again, column-wise, for the hyper-sparse back
    // substitution: ucol_row_/ucol_val_[ucol_start_[j] .. ucol_start_[j+1])
    // are the rows (elimination order) and values of column j. U is fixed
    // from factorize() on (updates are etas), so this copy stays valid.
    std::vector<Index> ucol_start_;
    std::vector<Index> ucol_row_;
    std::vector<Real> ucol_val_;

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
    // The sparse etas' entries again, by position, for BTRAN by pattern.
    // ft_row_head_[i] is the newest entry k (index into ft_index_) at
    // position i, ft_row_next_[k] the next older one there (-1 ends both),
    // and ft_entry_eta_[k] the eta holding entry k.
    std::vector<Index> ft_row_head_;
    std::vector<Index> ft_row_next_;
    std::vector<Index> ft_entry_eta_;
    // ft_last_pivot_[i] is the newest eta pivoting at position i (-1: none).
    // For eta u, ft_prev_pivot_[u] is the previous eta pivoting at the same
    // position (-1: none), and ft_pivot_head_[u] was ft_row_head_ at that
    // position when u was added: the newest entry there of an eta before u.
    std::vector<Index> ft_last_pivot_;
    std::vector<Index> ft_prev_pivot_;
    std::vector<Index> ft_pivot_head_;
    // Per-eta dot products accumulated by applyFTTransposeByRow.
    mutable std::vector<Real> ft_acc_;

    Index num_updates_ = 0;

    // Reusable dense scratch buffers for hot-path solves/updates.
    // solve_work_ is all zero between solves: each solve leaves it so by
    // zeroing the pattern it read out.
    mutable std::vector<Real> solve_work_;
    std::vector<Real> update_work_;
    // update_work_'s nonzero pattern (zero elsewhere), from the last update().
    std::vector<Index> update_touched_;
    mutable std::vector<Index> sparse_steps_;
    mutable std::vector<uint32_t> sparse_epoch_;
    mutable uint32_t sparse_epoch_id_ = 1;
    // Depth-first search stack: node and next successor position.
    mutable std::vector<Index> dfs_node_;
    mutable std::vector<Index> dfs_pos_;
    // Elimination-order pattern inside a solve, and the pattern the dense
    // entry points build for the pattern solves.
    mutable std::vector<Index> stage_pattern_;
    mutable std::vector<Index> api_pattern_;
    SolveKernel solve_kernel_ = SolveKernel::Auto;

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

    /// Exponential moving average of each stage's result density (SolveStage).
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
