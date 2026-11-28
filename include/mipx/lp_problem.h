#pragma once

#include "mipx/core.h"
#include "mipx/sparse_matrix.h"

#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace mipx {

inline constexpr Real kInf = std::numeric_limits<Real>::infinity();

struct LpProblem {
    std::string name;
    Sense sense = Sense::Minimize;

    // Variables.
    Index num_cols = 0;
    std::vector<Real> obj;
    std::vector<Real> col_lower;
    std::vector<Real> col_upper;
    std::vector<VarType> col_type;
    // For semi-continuous / semi-integer vars: active lower bound L.
    // Ignored for other variable types.
    std::vector<Real> col_semi_lower;
    std::vector<std::string> col_names;

    // Constraints (ranged form: row_lower <= Ax <= row_upper).
    Index num_rows = 0;
    SparseMatrix matrix{0, 0};
    std::vector<Real> row_lower;
    std::vector<Real> row_upper;
    std::vector<std::string> row_names;

    struct SosConstraint {
        enum class Type {
            Sos1,
            Sos2,
        };
        Type type = Type::Sos1;
        std::vector<Index> vars;
        std::vector<Real> weights;
        std::string name;
    };

    struct IndicatorConstraint {
        Index binary_var = -1;
        bool active_value = true;
        std::vector<Index> indices;
        std::vector<Real> values;
        Real lower = -kInf;
        Real upper = kInf;
        std::string name;
    };

    std::vector<SosConstraint> sos_constraints;
    std::vector<IndicatorConstraint> indicator_constraints;

    Real obj_offset = 0.0;

    /// Ordered SOS2 member columns retained for native MIP search after lowering.
    std::vector<std::vector<Index>> native_sos2;

    /// Returns true if the problem has any integer or binary variables.
    [[nodiscard]] bool hasIntegers() const;
};

/// Throws std::runtime_error if advanced feature declarations are invalid.
void validateModelFeatures(const LpProblem& problem);

/// Returns true when unlowered SOS, indicator or semi-variable declarations are present.
[[nodiscard]] bool hasAdvancedModelFeatures(const LpProblem& problem);

/// Linearize SOS/indicator/semi features into core row/bound form.
LpProblem linearizeModelFeatures(const LpProblem& problem);

/// Optionally use adjacent interval selectors for SOS2 sets.
LpProblem linearizeModelFeatures(const LpProblem& problem, bool sos2_adjacent_intervals);

enum class Sos2Formulation { Pairwise, AdjacentIntervals, Native };
LpProblem linearizeModelFeatures(const LpProblem& problem, Sos2Formulation sos2_formulation);

/// First and last active positions in a weight-ordered set, or size() for no support.
std::pair<std::size_t, std::size_t> sos2Support(const LpProblem& problem,
                                                std::span<const Index> members,
                                                std::span<const Real> values, Real tolerance = 0.0);

/// Round the bounds of each Integer and Binary column in to the nearest
/// integers inside them (within @p tol), so no reduction or search step ever
/// fixes such a column at a fractional bound. Returns false when some column's
/// rounded bounds cross, i.e. it has no integer value and the model is
/// infeasible. Semi columns are left alone: their domain is not an interval.
bool roundIntegerBounds(LpProblem& problem, Real tol = 1e-6);

/// Check a batch of columns given column-wise, as LpSolver::addCols takes it,
/// against a model with @p num_rows rows, and return the number of columns.
/// Throws std::invalid_argument when obj, lower and upper differ in size,
/// indices and values differ in size, starts has neither k nor k + 1 entries
/// (then with starts[k] == indices.size()), starts is not nondecreasing
/// within [0, indices.size()], a row index is out of range, a cost, bound or
/// coefficient is NaN, a cost or coefficient is infinite, or a lower bound is
/// +inf or an upper bound -inf. Entries of
/// one column on the same row add up.
Index checkColumnBatch(Index num_rows, std::span<const Index> starts,
                       std::span<const Index> indices, std::span<const Real> values,
                       std::span<const Real> obj, std::span<const Real> lower,
                       std::span<const Real> upper);

/// End of column @p t of a batch that checkColumnBatch accepted.
[[nodiscard]] inline Index columnBatchEnd(std::span<const Index> starts, Index t,
                                          Index num_entries) {
    return (t + 1 < static_cast<Index>(starts.size())) ? starts[t + 1] : num_entries;
}

/// Append a batch that checkColumnBatch accepted to @p problem, as
/// continuous unnamed columns.
void appendColumns(LpProblem& problem, std::span<const Index> starts,
                   std::span<const Index> indices, std::span<const Real> values,
                   std::span<const Real> obj, std::span<const Real> lower,
                   std::span<const Real> upper);

/// The columns of @p cols sorted and without repeats. Throws
/// std::invalid_argument on one outside [0, num_cols).
std::vector<Index> sortedColumnSet(Index num_cols, std::span<const Index> cols);

/// Remove the columns @p sorted_cols (as sortedColumnSet returns them) from
/// @p problem; the survivors keep their order.
void removeColumns(LpProblem& problem, std::span<const Index> sorted_cols);

/// True when column @p j is a 0/1 variable by the problem's own bounds: typed
/// VarType::Binary, or VarType::Integer with bounds exactly [0, 1]. The MPS
/// reader types only a BV bound as Binary, so the MARKER INTORG columns with
/// UP 1 that MIPLIB uses for its 0/1 variables stay Integer. The bounds read
/// are those of @p problem, so a general integer a node has tightened to
/// [0, 1] counts only in a problem carrying the node's bounds.
[[nodiscard]] inline bool isBinaryColumn(const LpProblem& problem, Index j) {
    const VarType type = problem.col_type[j];
    return type == VarType::Binary ||
           (type == VarType::Integer && problem.col_lower[j] == 0.0 && problem.col_upper[j] == 1.0);
}

/// Column @p j's type, with every isBinaryColumn column read as
/// VarType::Binary: for code that keys on a type rather than asking about one
/// column, so a MARKER integer over [0, 1] and a BV column compare equal.
[[nodiscard]] inline VarType binaryAwareType(const LpProblem& problem, Index j) {
    return isBinaryColumn(problem, j) ? VarType::Binary : problem.col_type[j];
}

}  // namespace mipx
