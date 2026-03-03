#pragma once

#include <string>
#include <unordered_map>
#include <variant>
#include <vector>
#include <cstdint>

#include "mipx/core.h"
#include "mipx/lp_problem.h"

namespace mipx {

// ---- Postsolve operations ---------------------------------------------------

/// A variable was fixed to a value and removed.
struct PostsolveFixVariable {
    Index orig_col;
    Real value;
};

/// A singleton row was removed (variable bound was tightened instead).
struct PostsolveSingletonRow {
    Index orig_row;
};

/// A singleton column was substituted out.
struct PostsolveSingletonCol {
    Index orig_col;
    Index orig_row;       // The single constraint it appeared in.
    Real coeff;           // The coefficient a_{row,col}.
    Real obj_coeff;       // Original objective coefficient.
    Real row_lower;       // Original row bounds.
    Real row_upper;
    VarType var_type;     // Original variable type.
    Real col_lower;       // Original variable bounds.
    Real col_upper;
};

/// A forcing constraint removed variables.
struct PostsolveForcingRow {
    Index orig_row;
    struct FixedVar {
        Index orig_col;
        Real value;
    };
    std::vector<FixedVar> fixed_vars;
};

/// A dominated (redundant) row was removed.
struct PostsolveDominatedRow {
    Index orig_row;
};

/// Coefficient was tightened for an integer variable.
struct PostsolveCoeffTightening {
    Index orig_row;
    Index orig_col;
    Real old_coeff;
    Real new_coeff;
    Real old_rhs;       // The row_upper that was adjusted.
    Real new_rhs;
};

/// A doubleton equality row was used to substitute out a variable.
struct PostsolveDoubletonEquality {
    Index eliminated_col;
    Index kept_col;
    Real a_eliminated;
    Real a_kept;
    Real rhs;
    Real eliminated_lower;
    Real eliminated_upper;
};

using PostsolveOp = std::variant<
    PostsolveFixVariable,
    PostsolveSingletonRow,
    PostsolveSingletonCol,
    PostsolveForcingRow,
    PostsolveDominatedRow,
    PostsolveCoeffTightening,
    PostsolveDoubletonEquality
>;

// ---- PostsolveStack ---------------------------------------------------------

class PostsolveStack {
public:
    /// Push a postsolve operation.
    void push(PostsolveOp op);

    /// Clear all recorded operations.
    void clear();

    /// Given a solution to the presolved problem (indexed by presolved column),
    /// reconstruct the full solution (indexed by original column).
    /// `col_mapping` maps presolved col index -> original col index.
    std::vector<Real> postsolve(const std::vector<Real>& presolved_solution,
                                 const std::vector<Index>& col_mapping,
                                 Index orig_num_cols) const;

    [[nodiscard]] Index size() const {
        return static_cast<Index>(ops_.size());
    }

private:
    std::vector<PostsolveOp> ops_;
};

// ---- Presolve statistics ----------------------------------------------------

struct PresolveStats {
    Index vars_removed = 0;
    Index rows_removed = 0;
    Index bounds_tightened = 0;
    Index coeffs_tightened = 0;
    Index fixed_var_changes = 0;
    Index singleton_row_changes = 0;
    Index singleton_col_changes = 0;
    Index forcing_row_changes = 0;
    Index dominated_row_changes = 0;
    Index coeff_tightening_changes = 0;
    Index implied_equation_changes = 0;
    Index activity_bound_tightening_changes = 0;
    Index dual_fixing_changes = 0;
    Index empty_col_changes = 0;
    Index duplicate_row_changes = 0;
    Index parallel_row_changes = 0;
    Index doubleton_eq_changes = 0;
    Index rounds = 0;
    Index rounds_with_changes = 0;
    Index rows_examined = 0;
    Index cols_examined = 0;
    Real time_seconds = 0.0;
};

struct PresolveOptions {
    bool enable_forcing_rows = true;
    bool enable_dual_fixing = true;
    bool enable_coefficient_tightening = false;
    bool enable_doubleton_aggregation = true;
    bool enable_parallel_rows = true;
};

// ---- Presolver --------------------------------------------------------------

class Presolver {
public:
    Presolver() = default;

    /// Run presolve on the given problem. Returns the reduced problem.
    /// The postsolve stack and column mapping are stored internally.
    LpProblem presolve(const LpProblem& problem);

    /// Postsolve: given solution to presolved problem, reconstruct full solution.
    std::vector<Real> postsolve(const std::vector<Real>& presolved_solution) const;

    /// Returns true if presolve detected infeasibility.
    [[nodiscard]] bool isInfeasible() const { return infeasible_; }

    /// Access statistics.
    [[nodiscard]] const PresolveStats& stats() const { return stats_; }

    /// Access the column mapping (presolved col -> original col).
    [[nodiscard]] const std::vector<Index>& colMapping() const {
        return col_mapping_;
    }

    /// Set maximum number of presolve rounds.
    void setMaxRounds(Index rounds) { max_rounds_ = rounds; }

    void setOptions(const PresolveOptions& options) { options_ = options; }
    [[nodiscard]] const PresolveOptions& options() const { return options_; }

private:
    // Individual reductions. Return number of changes made.
    Index removeFixedVariables(LpProblem& lp, std::vector<bool>& col_removed,
                               std::vector<bool>& row_removed,
                               std::vector<Index>& row_active_nnz,
                               std::vector<Index>& col_active_nnz,
                               const std::vector<Index>& dirty_cols,
                               std::vector<uint8_t>& next_dirty_rows,
                               std::vector<uint8_t>& next_dirty_cols);
    Index removeSingletonRows(LpProblem& lp, std::vector<bool>& col_removed,
                               std::vector<bool>& row_removed,
                               std::vector<Index>& row_active_nnz,
                               std::vector<Index>& col_active_nnz,
                               const std::vector<Index>& dirty_rows,
                               std::vector<uint8_t>& next_dirty_rows,
                               std::vector<uint8_t>& next_dirty_cols);
    Index removeSingletonCols(LpProblem& lp, std::vector<bool>& col_removed,
                               std::vector<bool>& row_removed,
                               std::vector<Index>& row_active_nnz,
                               std::vector<Index>& col_active_nnz,
                               const std::vector<Index>& dirty_cols,
                               std::vector<uint8_t>& next_dirty_rows,
                               std::vector<uint8_t>& next_dirty_cols);
    Index removeForcingRows(LpProblem& lp, std::vector<bool>& col_removed,
                             std::vector<bool>& row_removed,
                             std::vector<Index>& row_active_nnz,
                             std::vector<Index>& col_active_nnz,
                             const std::vector<Index>& dirty_rows,
                             std::vector<uint8_t>& next_dirty_rows,
                             std::vector<uint8_t>& next_dirty_cols);
    Index removeDominatedRows(LpProblem& lp, std::vector<bool>& col_removed,
                               std::vector<bool>& row_removed,
                               std::vector<Index>& row_active_nnz,
                               std::vector<Index>& col_active_nnz,
                               const std::vector<Index>& dirty_rows,
                               std::vector<uint8_t>& next_dirty_rows,
                               std::vector<uint8_t>& next_dirty_cols);
    Index detectImpliedEquations(LpProblem& lp, std::vector<bool>& col_removed,
                                  std::vector<bool>& row_removed,
                                  const std::vector<Index>& dirty_rows,
                                  std::vector<uint8_t>& next_dirty_rows,
                                  std::vector<uint8_t>& next_dirty_cols);
    Index activityBoundTightening(LpProblem& lp, std::vector<bool>& col_removed,
                                   std::vector<bool>& row_removed,
                                   const std::vector<Index>& dirty_rows,
                                   std::vector<uint8_t>& next_dirty_rows,
                                   std::vector<uint8_t>& next_dirty_cols);
    Index dualFixing(LpProblem& lp, std::vector<bool>& col_removed,
                      std::vector<bool>& row_removed,
                      std::vector<Index>& row_active_nnz,
                      std::vector<Index>& col_active_nnz,
                      const std::vector<Index>& dirty_cols,
                      std::vector<uint8_t>& next_dirty_rows,
                      std::vector<uint8_t>& next_dirty_cols);
    Index removeEmptyColumns(LpProblem& lp, std::vector<bool>& col_removed,
                              std::vector<bool>& row_removed,
                              std::vector<Index>& row_active_nnz,
                              std::vector<Index>& col_active_nnz,
                              const std::vector<Index>& dirty_cols,
                              std::vector<uint8_t>& next_dirty_rows,
                              std::vector<uint8_t>& next_dirty_cols);
    Index removeDuplicateRows(LpProblem& lp, std::vector<bool>& col_removed,
                               std::vector<bool>& row_removed,
                               std::vector<Index>& row_active_nnz,
                               std::vector<Index>& col_active_nnz,
                               const std::vector<Index>& dirty_rows,
                               std::vector<uint8_t>& next_dirty_rows,
                               std::vector<uint8_t>& next_dirty_cols);
    Index removeParallelRows(LpProblem& lp, std::vector<bool>& col_removed,
                              std::vector<bool>& row_removed,
                              std::vector<Index>& row_active_nnz,
                              std::vector<Index>& col_active_nnz,
                              const std::vector<Index>& dirty_rows,
                              std::vector<uint8_t>& next_dirty_rows,
                              std::vector<uint8_t>& next_dirty_cols);
    Index aggregateDoubletonEqualities(LpProblem& lp, std::vector<bool>& col_removed,
                                        std::vector<bool>& row_removed,
                                        std::vector<Index>& row_active_nnz,
                                        std::vector<Index>& col_active_nnz,
                                        const std::vector<Index>& dirty_rows,
                                        std::vector<uint8_t>& next_dirty_rows,
                                        std::vector<uint8_t>& next_dirty_cols);
    Index tightenCoefficients(LpProblem& lp, std::vector<bool>& col_removed,
                               std::vector<bool>& row_removed,
                               std::vector<uint8_t>& next_dirty_rows,
                               std::vector<uint8_t>& next_dirty_cols);

    // Build the reduced problem from the original with removed rows/cols.
    LpProblem buildReducedProblem(const LpProblem& lp,
                                   const std::vector<bool>& col_removed,
                                   const std::vector<bool>& row_removed);

    PostsolveStack postsolve_stack_;
    std::vector<Index> col_mapping_;   // presolved col -> original col
    Index orig_num_cols_ = 0;
    PresolveStats stats_;
    PresolveOptions options_{};
    Index max_rounds_ = 20;
    bool infeasible_ = false;
    std::unordered_map<uint64_t, Real> coeff_overrides_{};

    static constexpr Real kTol = 1e-8;
};

}  // namespace mipx
