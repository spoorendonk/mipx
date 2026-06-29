#pragma once

#include "mipx/core.h"
#include "mipx/lp_problem.h"

#include <cstdio>
#include <variant>
#include <vector>

namespace mipx {

// ---- Barrier postsolve operations ------------------------------------------
// These record how to reconstruct the original-space primal and dual solution
// from the reduced problem's solution.

/// A variable was fixed (lb == ub) and removed.
struct BpFixVariable {
    Index orig_col;
    Real value;
    Real obj_coeff;
    // Row indices and coefficients for rows this variable participated in
    // (needed to adjust duals during postsolve).
    std::vector<Index> row_indices;
    std::vector<Real> row_coeffs;
};

/// A singleton row (single nonzero) was removed and its bound was tightened.
struct BpSingletonRow {
    Index orig_row;
    Index orig_col;
    Real coeff;      // The single coefficient a_{row,col}.
    Real row_lower;  // Original row bounds.
    Real row_upper;
};

/// A singleton column (appears in one row) was substituted out.
struct BpSingletonCol {
    Index orig_col;
    Index orig_row;
    Real coeff;      // The coefficient a_{row,col}.
    Real obj_coeff;  // Original objective coefficient.
    Real col_lower;
    Real col_upper;
    Real row_lower;  // Original row bounds (before adjustment).
    Real row_upper;
    // The other columns in the eliminated row, for primal reconstruction:
    // x_j must absorb the row's residual slack, not merely sit at a bound.
    std::vector<Index> row_col_indices;
    std::vector<Real> row_col_coeffs;
};

/// A free column singleton (free variable in a single constraint) was
/// eliminated, removing both the variable and the constraint.
struct BpFreeColSingleton {
    Index orig_col;
    Index orig_row;
    Real coeff;
    Real obj_coeff;
    // The other columns in the eliminated row, for dual reconstruction.
    std::vector<Index> row_col_indices;
    std::vector<Real> row_col_coeffs;
    Real row_lower;
    Real row_upper;
};

/// An empty row (zero nonzeros) was removed.
struct BpEmptyRow {
    Index orig_row;
    Real row_lower;
    Real row_upper;
};

/// An empty column (zero nonzeros after other reductions) was removed.
struct BpEmptyCol {
    Index orig_col;
    Real obj_coeff;
    Real col_lower;
    Real col_upper;
    Real value;  // Fixing value chosen by presolve (sense-aware).
};

using BpPostsolveOp = std::variant<BpFixVariable, BpSingletonRow, BpSingletonCol,
                                   BpFreeColSingleton, BpEmptyRow, BpEmptyCol>;

// ---- Barrier presolve statistics -------------------------------------------

struct BarrierPresolveStats {
    Index fixed_vars = 0;
    Index singleton_rows = 0;
    Index singleton_cols = 0;
    Index free_col_singletons = 0;
    Index empty_rows = 0;
    Index empty_cols = 0;
    Index implied_bounds_tightened = 0;
    Index redundant_rows = 0;
    Index orig_rows = 0;
    Index orig_cols = 0;
    Index orig_nnz = 0;
    Index reduced_rows = 0;
    Index reduced_cols = 0;
    Index reduced_nnz = 0;

    void print() const {
        std::printf(
            "Barrier presolve: %d rows, %d cols, %d nnz -> "
            "%d rows, %d cols, %d nnz\n",
            orig_rows, orig_cols, orig_nnz, reduced_rows, reduced_cols, reduced_nnz);
        std::printf(
            "  fixed_vars=%d singleton_rows=%d singleton_cols=%d "
            "free_col_singletons=%d empty_rows=%d empty_cols=%d "
            "implied_bounds=%d redundant_rows=%d\n",
            fixed_vars, singleton_rows, singleton_cols, free_col_singletons, empty_rows, empty_cols,
            implied_bounds_tightened, redundant_rows);
    }
};

// ---- BarrierPresolver ------------------------------------------------------

class BarrierPresolver {
public:
    /// Run barrier-specific presolve on the problem.
    /// Returns a reduced LpProblem suitable for standard-form conversion.
    LpProblem presolve(const LpProblem& problem);

    /// Postsolve: given primal/dual solution to the reduced problem,
    /// reconstruct the original-space primal and dual solution.
    /// @param primal_reduced  primal values indexed by reduced col
    /// @param dual_reduced    dual values indexed by reduced row
    /// @param rc_reduced      reduced costs indexed by reduced col
    void postsolve(const std::vector<Real>& primal_reduced, const std::vector<Real>& dual_reduced,
                   const std::vector<Real>& rc_reduced, std::vector<Real>& primal_orig,
                   std::vector<Real>& dual_orig, std::vector<Real>& rc_orig) const;

    [[nodiscard]] bool isInfeasible() const { return infeasible_; }
    [[nodiscard]] const BarrierPresolveStats& stats() const { return stats_; }
    [[nodiscard]] Real objectiveOffset() const { return obj_offset_delta_; }

    /// Column mapping: reduced col -> original col.
    [[nodiscard]] const std::vector<Index>& colMapping() const { return col_mapping_; }

    /// Row mapping: reduced row -> original row.
    [[nodiscard]] const std::vector<Index>& rowMapping() const { return row_mapping_; }

private:
    // Individual reductions. Return number of changes made.
    Index removeFixedVariables(LpProblem& lp, std::vector<bool>& col_removed,
                               std::vector<bool>& row_removed, std::vector<Index>& row_nnz,
                               std::vector<Index>& col_nnz);

    Index removeSingletonRows(LpProblem& lp, std::vector<bool>& col_removed,
                              std::vector<bool>& row_removed, std::vector<Index>& row_nnz,
                              std::vector<Index>& col_nnz);

    Index removeSingletonCols(LpProblem& lp, std::vector<bool>& col_removed,
                              std::vector<bool>& row_removed, std::vector<Index>& row_nnz,
                              std::vector<Index>& col_nnz);

    Index removeFreeColSingletons(LpProblem& lp, std::vector<bool>& col_removed,
                                  std::vector<bool>& row_removed, std::vector<Index>& row_nnz,
                                  std::vector<Index>& col_nnz);

    Index removeEmptyRowsCols(LpProblem& lp, std::vector<bool>& col_removed,
                              std::vector<bool>& row_removed, std::vector<Index>& row_nnz,
                              std::vector<Index>& col_nnz);

    Index tightenImpliedBounds(LpProblem& lp, const std::vector<bool>& col_removed,
                               const std::vector<bool>& row_removed,
                               const std::vector<Index>& row_nnz);

    Index removeRedundantRows(LpProblem& lp, const std::vector<bool>& col_removed,
                              std::vector<bool>& row_removed, std::vector<Index>& row_nnz,
                              std::vector<Index>& col_nnz);

    LpProblem buildReducedProblem(const LpProblem& lp, const std::vector<bool>& col_removed,
                                  const std::vector<bool>& row_removed);

    std::vector<BpPostsolveOp> ops_;
    std::vector<Index> col_mapping_;
    std::vector<Index> row_mapping_;
    Index orig_num_cols_ = 0;
    Index orig_num_rows_ = 0;
    Real obj_offset_delta_ = 0.0;
    BarrierPresolveStats stats_{};
    bool infeasible_ = false;

    static constexpr Real kTol = 1e-10;
};

}  // namespace mipx
