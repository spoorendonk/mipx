#pragma once

#include <vector>

#include "mipx/core.h"
#include "mipx/lp_problem.h"
#include "mipx/lp_solver.h"
#include "mipx/lu.h"
#include "mipx/sparse_matrix.h"
#include "mipx/work_units.h"

namespace mipx {

struct DualSimplexOptions {
    // Pricing controls.
    bool enable_partial_pricing = true;
    Int partial_pricing_chunk_min = 512;
    Int partial_pricing_full_scan_freq = 25;

    // Refactorization controls.
    bool enable_adaptive_refactorization = true;
    Int adaptive_refactor_min_updates = 24;
    Int adaptive_refactor_stall_pivots = 32;
    Real adaptive_refactor_degenerate_pivot_tol = 1e-10;

    // Runtime SIMD controls for dense vector kernels.
    bool enable_simd_kernels = true;
    Int simd_min_length = 64;

    // SIP-style parallel candidate scan (CHUZC), guarded and off by default.
    bool enable_sip_parallel_candidates = false;
    bool enable_sip_parallel_dual_scan = false;
    Int sip_parallel_min_nonbasic = 4096;
    Int sip_parallel_grain = 512;
    Int sip_parallel_min_threads = 2;
    bool sip_parallel_disable_on_stall = true;
    Int sip_parallel_stall_pivots = 32;
    bool enable_sip_parallel_candidate_sort = false;
    Int sip_parallel_sort_min_candidates = 4096;

    // SIP-style parallel leaving-row scan (CHUZR), guarded and off by default.
    bool enable_sip_parallel_chuzr = false;
    Int sip_parallel_min_rows = 2048;
    Int sip_parallel_row_grain = 256;
};

class DualSimplexSolver : public LpSolver {
public:
    DualSimplexSolver() = default;

    void load(const LpProblem& problem) override;
    LpResult solve() override;

    Status getStatus() const override { return status_; }
    Real getObjective() const override;

    std::vector<Real> getPrimalValues() const override;
    std::vector<Real> getDualValues() const override;
    std::vector<Real> getReducedCosts() const override;

    std::vector<BasisStatus> getBasis() const override;
    void setBasis(std::span<const BasisStatus> basis) override;

    void addRows(std::span<const Index> starts,
                 std::span<const Index> indices,
                 std::span<const Real> values,
                 std::span<const Real> lower,
                 std::span<const Real> upper) override;
    void removeRows(std::span<const Index> rows) override;

    void setColBounds(Index col, Real lower, Real upper) override;
    void setObjective(std::span<const Real> obj) override;

    void setIterationLimit(Int limit) { iter_limit_ = limit; }
    void setOptions(const DualSimplexOptions& options) { options_ = options; }
    [[nodiscard]] const DualSimplexOptions& getOptions() const { return options_; }

    /// Access work unit counter (includes LU work).
    [[nodiscard]] const WorkUnits& workUnits() const { return work_; }
    void resetWorkUnits() { work_.reset(); lu_.resetWorkUnits(); }

    /// Get a tableau row for basis position `basis_pos` in external (unscaled) space.
    /// Returns a dense vector of size num_cols + num_rows (structural + slacks).
    /// The tableau row satisfies: x_i + sum_{j nonbasic} alpha_j * x_j = 0
    /// where the sum includes both structural and slack variables.
    void getTableauRow(Index basis_pos, std::vector<Real>& tableau_row);

    /// Get the basis position of a variable, or -1 if nonbasic.
    [[nodiscard]] Index basisPosition(Index var) const {
        if (var < 0 || var >= static_cast<Index>(basis_pos_.size())) return -1;
        return basis_pos_[var];
    }

    /// Get the number of rows in the current problem.
    [[nodiscard]] Index numRows() const { return num_rows_; }

    /// Get the number of structural columns in the current problem.
    [[nodiscard]] Index numCols() const { return num_cols_; }

private:
    // Total number of variables = num_cols (structural) + num_rows (slacks).
    Index numVars() const { return num_cols_ + num_rows_; }

    // Build the augmented matrix [A | I].
    void buildAugmentedMatrix();

    // Set up initial basis (all slacks basic).
    void setupInitialBasis();

    // Compute primal values for basic variables: x_B = B^{-1} * b.
    void computePrimals();

    // Compute dual values and reduced costs.
    void computeDuals();

    // Refactorize the basis.
    void refactorize();

    // Scaling.
    void computeScaling();
    void applyScaling();
    void unscaleResults();

    // Variable bounds (structural + slack).
    Real varLower(Index k) const;
    Real varUpper(Index k) const;
    Real varCost(Index k) const;

    // Problem data (internal, after sign flip for maximize).
    Index num_cols_ = 0;
    Index num_rows_ = 0;
    Sense sense_ = Sense::Minimize;
    Real obj_offset_ = 0.0;
    std::vector<Real> obj_;           // size num_cols
    std::vector<Real> col_lower_;     // size num_cols
    std::vector<Real> col_upper_;     // size num_cols
    std::vector<Real> row_lower_;     // size num_rows
    std::vector<Real> row_upper_;     // size num_rows
    SparseMatrix matrix_{0, 0};       // original constraint matrix
    SparseMatrix augmented_{0, 0};    // [A | I]

    // Scaling factors.
    std::vector<Real> row_scale_;     // size num_rows
    std::vector<Real> col_scale_;     // size num_cols
    bool scaled_ = false;

    // Basis representation.
    // basis_[i] = variable index (0..numVars-1) in basis position i.
    std::vector<Index> basis_;
    // nonbasic_ = list of nonbasic variable indices.
    std::vector<Index> nonbasic_;
    // nonbasic_pos_[k] = position in nonbasic_ if nonbasic, -1 otherwise.
    std::vector<Index> nonbasic_pos_;
    // basis_pos_[k] = position in basis_ if basic, -1 otherwise.
    std::vector<Index> basis_pos_;
    // var_status_[k] = status of variable k.
    std::vector<BasisStatus> var_status_;

    // Solution data.
    std::vector<Real> primal_;        // size numVars
    std::vector<Real> dual_;          // size num_rows
    std::vector<Real> reduced_cost_;  // size numVars

    // LU factorization.
    SparseLU lu_;

    // Solver state.
    Status status_ = Status::Error;
    Int iterations_ = 0;
    Int iter_limit_ = 1000000;
    bool loaded_ = false;
    bool has_basis_ = false;  // true after first solve or setBasis()

    // Devex pricing weights.
    std::vector<Real> devex_weights_;  // size num_rows, one per basis position
    Int devex_reset_count_ = 0;
    static constexpr Int kDevexResetFreq = 200;  // reset every N pivots

    // Tolerances.
    static constexpr Real kPrimalTol = 1e-7;
    static constexpr Real kDualTol = 1e-7;
    static constexpr Real kPivotTol = 1e-7;
    static constexpr Real kZeroTol = 1e-13;
    static constexpr Int kLogFrequency = 200;

    // Deterministic work counter.
    WorkUnits work_;
    DualSimplexOptions options_{};
};

}  // namespace mipx
