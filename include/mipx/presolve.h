#pragma once

#include "mipx/core.h"
#include "mipx/lp_problem.h"

#include <cstdint>
#include <span>
#include <string>
#include <variant>
#include <vector>

namespace mipx {

class VariableBoundStore;
class PresolveMatrix;

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
    Index orig_row;  // The single constraint it appeared in.
    Real coeff;      // The coefficient a_{row,col}.
    Real obj_coeff;  // Original objective coefficient.
    Real row_lower;  // Original row bounds.
    Real row_upper;
    VarType var_type;  // Original variable type.
    Real col_lower;    // Original variable bounds.
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
    Real old_rhs;  // The row_upper that was adjusted.
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

/// Two continuous parallel columns of equal cost ratio, A_removed = scale *
/// A_kept and c_removed = scale * c_kept, were merged into the kept one:
/// y = x_kept + scale * x_removed, with y's bounds the sum of theirs (#308).
/// Postsolve splits y back within both columns' original bounds.
struct PostsolveParallelCol {
    Index kept_col;
    Index removed_col;
    Real scale;
    Real kept_lower;  ///< The kept column's bounds before the merge.
    Real kept_upper;
    Real removed_lower;
    Real removed_upper;
};

/// An implied-free continuous column singleton was substituted out of its
/// row (#324): x_col = (rhs - sum a_k x_k) / coeff over the row's other
/// columns as they were then, rhs the row's equation side or the side the
/// column's cost pushes it to. The row and the column were removed; the
/// row's terms are kept for postsolve (and for a dual postsolve, #323).
struct PostsolveFreeColSubstitution {
    Index orig_col;
    Index orig_row;
    Real coeff;
    Real rhs;
    std::vector<Index> row_cols;
    std::vector<Real> row_coeffs;
};

/// A zero-cost continuous column singleton in a row with both sides finite
/// was that row's slack (#359, HiGHS's zero-cost singleton): the column was
/// removed and the row's sides widened to row_lower - max(a l, a u) and
/// row_upper - min(a l, a u) over the other columns. Postsolve picks x_col
/// in [col_lower, col_upper] that puts a x_col + (the row's other terms)
/// back in [row_lower, row_upper]; the row's terms are kept for that.
struct PostsolveZeroCostSingleton {
    Index orig_col;
    Index orig_row;
    Real coeff;
    Real row_lower;  ///< The row's sides before the column went.
    Real row_upper;
    Real new_row_lower;  ///< And after, over the other columns.
    Real new_row_upper;
    Real col_lower;
    Real col_upper;
    std::vector<Index> row_cols;
    std::vector<Real> row_coeffs;
};

using PostsolveOp =
    std::variant<PostsolveFixVariable, PostsolveSingletonRow, PostsolveSingletonCol,
                 PostsolveForcingRow, PostsolveDominatedRow, PostsolveCoeffTightening,
                 PostsolveDoubletonEquality, PostsolveParallelCol, PostsolveFreeColSubstitution,
                 PostsolveZeroCostSingleton>;

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
                                const std::vector<Index>& col_mapping, Index orig_num_cols) const;

    [[nodiscard]] Index size() const { return static_cast<Index>(ops_.size()); }

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
    /// Of dual_fixing_changes, the columns fixed by the cases
    /// enable_extended_dual_fixing adds (#366).
    Index extended_dual_fixing_changes = 0;
    /// Of dual_fixing_changes, the columns fixed by the single-equation
    /// extension enable_single_equation_dual_fixing adds (#367).
    Index single_equation_dual_fixing_changes = 0;
    Index empty_col_changes = 0;
    Index duplicate_row_changes = 0;
    Index parallel_row_changes = 0;
    /// Parallel continuous columns fixed (dominated) or merged (#308).
    Index parallel_col_changes = 0;
    /// Implied-free column singletons substituted out with their row (#324).
    Index free_col_substitution_changes = 0;
    Index doubleton_eq_changes = 0;
    /// Entries a doubleton equation's substitution inserted into rows that
    /// did not hold the kept column (enable_doubleton_fill_in, #343).
    Index doubleton_fill_in = 0;
    /// Implied-free columns substituted out of an equation, with it, by the
    /// aggregator (enable_aggregator, #348), and the entries it inserted.
    Index aggregator_changes = 0;
    Index aggregator_fill_in = 0;
    /// Zero-cost column singletons removed as their ranged row's or
    /// equation's slack (enable_zero_cost_singletons, #359).
    Index zero_cost_singleton_changes = 0;
    Index probing_changes = 0;
    Index probing_fixings = 0;
    Index probing_coeff_strengthenings = 0;
    /// Bounds of an LP's columns that presolve tightened and then put back
    /// to their original value, as the remaining rows still imply the
    /// tightened ones; bounds_tightened still counts them. The barrier pays
    /// a row for every column boxed by a bound it does not need (#279).
    Index implied_bounds_relaxed = 0;
    Index rounds = 0;
    Index rounds_with_changes = 0;
    Index rows_examined = 0;
    Index cols_examined = 0;
    /// Wall time of presolve's setup (copying the problem, rounding integer
    /// bounds, counting row and column lengths) and its reduction loop;
    /// building the reduced problem after it is not included.
    Real time_seconds = 0.0;
    /// Wall time in each reduction, summed over the rounds, so a slow
    /// presolve can be traced to the pass that takes it (#287). What
    /// time_seconds holds beyond their sum is the setup and the loop's own
    /// bookkeeping.
    Real fixed_var_seconds = 0.0;
    Real singleton_row_seconds = 0.0;
    Real singleton_col_seconds = 0.0;
    Real doubleton_eq_seconds = 0.0;
    Real forcing_row_seconds = 0.0;
    Real dominated_row_seconds = 0.0;
    Real duplicate_row_seconds = 0.0;
    Real parallel_row_seconds = 0.0;
    Real parallel_col_seconds = 0.0;
    Real free_col_substitution_seconds = 0.0;
    Real aggregator_seconds = 0.0;
    Real zero_cost_singleton_seconds = 0.0;
    Real implied_equation_seconds = 0.0;
    Real activity_bound_tightening_seconds = 0.0;
    Real dual_fixing_seconds = 0.0;
    Real empty_col_seconds = 0.0;
    Real coeff_tightening_seconds = 0.0;
    Real probing_seconds = 0.0;
    Real implied_bounds_relaxed_seconds = 0.0;
    /// Finite implied row-dual bounds (lower and upper counted apart) the
    /// enable_dominated_columns derivation found (#375).
    Index implied_row_dual_bounds = 0;
    Real dual_bounds_seconds = 0.0;
    /// Columns the enable_dominated_columns pass fixed as dominated, strongly
    /// or weakly (#376).
    Index dominated_col_changes = 0;
    /// Of dominated_col_changes, the weakly dominated ones.
    Index weakly_dominated_col_changes = 0;
    Real dominated_col_seconds = 0.0;
};

struct PresolveOptions {
    bool enable_forcing_rows = true;
    bool enable_dual_fixing = true;
    bool enable_coefficient_tightening = false;
    bool enable_doubleton_aggregation = true;
    /// Let a doubleton equation's substitution x_e = (r - a_k x_k) / a_e
    /// insert x_k into x_e's rows that do not hold it, instead of refusing
    /// the candidate (#343). Off by default until benchmarked. No cap on the
    /// entries added: one substitution never grows the active matrix (the
    /// working matrix's pool of entries does grow; see
    /// aggregateDoubletonEqualities). Nor on their size: an inserted entry
    /// is -a_row_e times the ratio a_k / a_e, however large, as a changed
    /// one already was without fill-in.
    bool enable_doubleton_fill_in = false;
    bool enable_parallel_rows = true;
    /// Fix dominated and merge equal-cost parallel continuous columns (#308).
    /// Off by default until benchmarked, like every new reduction.
    bool enable_parallel_cols = false;
    /// Substitute implied-free continuous column singletons out of their row,
    /// removing both (#324). Off by default until benchmarked.
    bool enable_free_col_substitution = false;
    /// The aggregator (#348), after HiGHS's: substitute an implied-free
    /// continuous column out of an equation into every other row it is in,
    /// filling in, and remove both. Runs once the other passes reach a
    /// fixpoint. Off by default until benchmarked.
    bool enable_aggregator = false;
    /// Remove a zero-cost continuous column singleton whose row has both
    /// sides finite as that row's slack, widening the row's sides (#359,
    /// HiGHS's zero-cost singleton). Off by default until benchmarked.
    bool enable_zero_cost_singletons = false;
    /// Dual fixing's cases from HiGHS's dualFixing that the cost-sign rule
    /// misses (#366): a column is fixed at a finite bound its cost does not
    /// push it away from when every row that locks it the other way holds
    /// there whatever its other columns take (HiGHS's worst-case bound,
    /// applied as a fix), e.g. a zero-cost column with no up locks at its
    /// upper bound. Off by default until benchmarked.
    bool enable_extended_dual_fixing = false;
    /// Dual fixing's single-equation extension (#367; HiGHS's
    /// handleSingleEquation, Achterberg et al. 2020, section 6.1): a column
    /// whose one lock in a direction is an equation triggers a check of that
    /// equation, which fixes every column of it that all its other rows and
    /// its cost push the same way, when the equation's activity bounds allow.
    /// Off by default until benchmarked.
    bool enable_single_equation_dual_fixing = false;
    /// Dominated columns from implied row-dual bounds, as HiGHS's
    /// detectDominatedCol (#361), on an LP: a column whose reduced cost the
    /// implied row-dual bounds (#375) keep positive (negative) is fixed at
    /// its lower (upper) bound, and one they keep nonnegative (nonpositive)
    /// too, as weakly dominated, when that bound is finite (#376). The
    /// bounds are also counted once before the #279 put-back, in
    /// PresolveStats::implied_row_dual_bounds. Off by default until
    /// benchmarked.
    bool enable_dominated_columns = false;
    /// Run root probing inside the reduction loop (fixings + implication-based
    /// coefficient strengthening). Off by default: probing is the most
    /// expensive pass and its default is decided by benchmarking.
    bool enable_probing = false;
    /// Budget of that probing pass, per presolve call.
    Int probing_max_rounds = 2;
    Int probing_max_probes_per_round = 5000;
    double probing_time_limit = 5.0;  // seconds
};

// ---- Presolver --------------------------------------------------------------

class Presolver {
public:
    /// A row presolve drops must hold to this, the MIP's feasibility
    /// tolerance, as no later step checks it again.
    static constexpr Real kDropTol = 1e-6;

    Presolver() = default;

    /// Run presolve on the given problem. Returns the reduced problem.
    /// The postsolve stack and column mapping are stored internally.
    /// Throws std::invalid_argument when the problem has semi columns, SOS or
    /// indicator constraints: linearize them with linearizeModelFeatures first.
    /// The matrix must hold each (row, col) at most once with columns sorted
    /// within each row, as SparseMatrix's triplet constructor leaves it: the
    /// working matrix looks entries up by binary search, so a repeated or
    /// out-of-order column would be missed (asserted in debug builds).
    LpProblem presolve(const LpProblem& problem);

    /// Postsolve: given solution to presolved problem, reconstruct full solution.
    std::vector<Real> postsolve(const std::vector<Real>& presolved_solution) const;

    /// Returns true if presolve detected infeasibility.
    [[nodiscard]] bool isInfeasible() const { return infeasible_; }

    /// Access statistics.
    [[nodiscard]] const PresolveStats& stats() const { return stats_; }

    /// Access the column mapping (presolved col -> original col).
    [[nodiscard]] const std::vector<Index>& colMapping() const { return col_mapping_; }

    /// Set maximum number of presolve rounds.
    void setMaxRounds(Index rounds) { max_rounds_ = rounds; }

    void setOptions(const PresolveOptions& options) { options_ = options; }
    [[nodiscard]] const PresolveOptions& options() const { return options_; }

private:
    // Individual reductions. Return number of changes made.
    Index removeFixedVariables(LpProblem& lp, std::vector<bool>& col_removed,
                               std::vector<bool>& row_removed, std::vector<Index>& row_active_nnz,
                               std::vector<Index>& col_active_nnz,
                               const std::vector<Index>& dirty_cols,
                               std::vector<uint8_t>& next_dirty_rows,
                               std::vector<uint8_t>& next_dirty_cols);
    Index removeSingletonRows(LpProblem& lp, std::vector<bool>& col_removed,
                              std::vector<bool>& row_removed, std::vector<Index>& row_active_nnz,
                              std::vector<Index>& col_active_nnz,
                              const std::vector<Index>& dirty_rows,
                              std::vector<uint8_t>& next_dirty_rows,
                              std::vector<uint8_t>& next_dirty_cols);
    Index removeSingletonCols(LpProblem& lp, std::vector<bool>& col_removed,
                              std::vector<bool>& row_removed, std::vector<Index>& row_active_nnz,
                              std::vector<Index>& col_active_nnz,
                              const std::vector<Index>& dirty_cols,
                              std::vector<uint8_t>& next_dirty_rows,
                              std::vector<uint8_t>& next_dirty_cols);
    Index removeForcingRows(LpProblem& lp, std::vector<bool>& col_removed,
                            std::vector<bool>& row_removed, std::vector<Index>& row_active_nnz,
                            std::vector<Index>& col_active_nnz,
                            const std::vector<Index>& dirty_rows,
                            std::vector<uint8_t>& next_dirty_rows,
                            std::vector<uint8_t>& next_dirty_cols);
    Index removeDominatedRows(LpProblem& lp, std::vector<bool>& col_removed,
                              std::vector<bool>& row_removed, std::vector<Index>& row_active_nnz,
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
    Index dualFixing(LpProblem& lp, std::vector<bool>& col_removed, std::vector<bool>& row_removed,
                     std::vector<Index>& row_active_nnz, std::vector<Index>& col_active_nnz,
                     const std::vector<Index>& dirty_cols, std::vector<uint8_t>& next_dirty_rows,
                     std::vector<uint8_t>& next_dirty_cols);
    Index removeDominatedColumns(LpProblem& lp, std::vector<bool>& col_removed,
                                 std::vector<bool>& row_removed, std::vector<Index>& row_active_nnz,
                                 std::vector<Index>& col_active_nnz,
                                 std::vector<uint8_t>& next_dirty_rows,
                                 std::vector<uint8_t>& next_dirty_cols);
    Index removeEmptyColumns(LpProblem& lp, std::vector<bool>& col_removed,
                             std::vector<bool>& row_removed, std::vector<Index>& row_active_nnz,
                             std::vector<Index>& col_active_nnz,
                             const std::vector<Index>& dirty_cols,
                             std::vector<uint8_t>& next_dirty_rows,
                             std::vector<uint8_t>& next_dirty_cols);
    Index removeDuplicateRows(LpProblem& lp, std::vector<bool>& col_removed,
                              std::vector<bool>& row_removed, std::vector<Index>& row_active_nnz,
                              std::vector<Index>& col_active_nnz,
                              const std::vector<Index>& dirty_rows,
                              std::vector<uint8_t>& next_dirty_rows,
                              std::vector<uint8_t>& next_dirty_cols);
    Index removeParallelRows(LpProblem& lp, std::vector<bool>& col_removed,
                             std::vector<bool>& row_removed, std::vector<Index>& row_active_nnz,
                             std::vector<Index>& col_active_nnz,
                             const std::vector<Index>& dirty_rows,
                             std::vector<uint8_t>& next_dirty_rows,
                             std::vector<uint8_t>& next_dirty_cols);
    Index removeParallelCols(LpProblem& lp, std::vector<bool>& col_removed,
                             const std::vector<bool>& row_removed,
                             std::vector<Index>& row_active_nnz, std::vector<Index>& col_active_nnz,
                             const std::vector<Index>& dirty_cols,
                             std::vector<uint8_t>& next_dirty_rows,
                             std::vector<uint8_t>& next_dirty_cols);
    Index substituteFreeColSingletons(LpProblem& lp, std::vector<bool>& col_removed,
                                      std::vector<bool>& row_removed,
                                      std::vector<Index>& row_active_nnz,
                                      std::vector<Index>& col_active_nnz,
                                      const std::vector<Index>& dirty_cols,
                                      std::vector<uint8_t>& next_dirty_rows,
                                      std::vector<uint8_t>& next_dirty_cols);
    Index removeZeroCostSingletons(LpProblem& lp, std::vector<bool>& col_removed,
                                   std::vector<bool>& row_removed,
                                   std::vector<Index>& row_active_nnz,
                                   std::vector<Index>& col_active_nnz,
                                   const std::vector<Index>& dirty_cols,
                                   std::vector<uint8_t>& next_dirty_rows,
                                   std::vector<uint8_t>& next_dirty_cols);
    Index aggregateDoubletonEqualities(LpProblem& lp, std::vector<bool>& col_removed,
                                       std::vector<bool>& row_removed,
                                       std::vector<Index>& row_active_nnz,
                                       std::vector<Index>& col_active_nnz,
                                       const std::vector<Index>& dirty_rows,
                                       std::vector<uint8_t>& next_dirty_rows,
                                       std::vector<uint8_t>& next_dirty_cols);
    /// The aggregator (enable_aggregator, #348): every active equation and
    /// implied-free continuous column in it, in HiGHS's order and under its
    /// pivot and fill limits (see the definition).
    Index aggregateImpliedFreeColumns(LpProblem& lp, std::vector<bool>& col_removed,
                                      std::vector<bool>& row_removed,
                                      std::vector<Index>& row_active_nnz,
                                      std::vector<Index>& col_active_nnz,
                                      std::vector<uint8_t>& next_dirty_rows,
                                      std::vector<uint8_t>& next_dirty_cols);
    // Root probing: fixes variables whose opposite branch propagates to
    // infeasibility, and strengthens coefficients with the learned variable
    // bounds. Runs at most once per presolve call.
    Index probingPass(LpProblem& lp, const std::vector<bool>& col_removed,
                      const std::vector<bool>& row_removed, std::vector<uint8_t>& next_dirty_rows,
                      std::vector<uint8_t>& next_dirty_cols);
    Index strengthenCoefficientsFromVarBounds(LpProblem& lp, const VariableBoundStore& vb_store,
                                              const std::vector<bool>& col_removed,
                                              const std::vector<bool>& row_removed,
                                              std::vector<uint8_t>& next_dirty_rows,
                                              std::vector<uint8_t>& next_dirty_cols);
    Index tightenCoefficients(LpProblem& lp, std::vector<bool>& col_removed,
                              std::vector<bool>& row_removed, const std::vector<Index>& dirty_rows,
                              std::vector<uint8_t>& next_dirty_rows,
                              std::vector<uint8_t>& next_dirty_cols);

    /// For an LP, puts back each column's original bound where presolve
    /// tightened it and a remaining row still implies the tightened one
    /// (#279). Returns the number of bounds put back.
    Index relaxImpliedBounds(LpProblem& lp, const LpProblem& original,
                             const std::vector<bool>& col_removed,
                             const std::vector<bool>& row_removed);

    // Build the reduced problem from the original with removed rows/cols.
    LpProblem buildReducedProblem(const LpProblem& lp, const std::vector<bool>& col_removed,
                                  const std::vector<bool>& row_removed);

    PostsolveStack postsolve_stack_;
    std::vector<Index> col_mapping_;  // presolved col -> original col
    Index orig_num_cols_ = 0;
    PresolveStats stats_;
    PresolveOptions options_{};
    Index max_rounds_ = 20;
    bool infeasible_ = false;
    bool probing_done_ = false;
    /// The working matrix (#326): every pass reads it, and every coefficient
    /// change is made in it; the input's matrix stays as given. Set only
    /// while presolve() runs, to a matrix it owns.
    PresolveMatrix* work_ = nullptr;

    /// Subtract a fixed column's term from the row's finite sides, and keep
    /// the term's size in the row's magnitude.
    void shiftRowSides(LpProblem& lp, Index row, Real shift);
    /// Infeasibility tolerance of a row: kTol relative to its magnitude, the
    /// largest of 1, its original sides and every term folded into them. The
    /// sides carry the rounding of those terms, so an emptied row's rhs can be
    /// off zero by far more than an absolute kTol (#236).
    [[nodiscard]] Real rowTol(Index row) const { return kTol * row_scale_[row]; }

    std::vector<Real> row_scale_;

    static constexpr Real kTol = 1e-8;
};

/// Whether `x` holds `model`: every bound and row to 1e-6, and integral where
/// a column's type says so. This is the test every incumbent passes (#240).
/// Presolve holds its reductions to its own tolerances, so a postsolved point
/// can miss the model as loaded by more (#244); check it here before calling
/// it an answer (#257).
[[nodiscard]] bool pointHoldsModel(const LpProblem& model, std::span<const Real> x);

/// The largest amount by which `x` misses a bound or row of `model`, in model
/// units; infinite when `x` has the wrong size or a non-finite entry.
/// Integrality is not measured. Comparing it on the reduced and the
/// postsolved point tells a miss presolve added from one the solver left.
[[nodiscard]] Real pointViolation(const LpProblem& model, std::span<const Real> x);

}  // namespace mipx
