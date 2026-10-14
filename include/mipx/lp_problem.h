#pragma once

#include "mipx/core.h"
#include "mipx/sparse_matrix.h"

#include <limits>
#include <string>
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

    /// Returns true if the problem has any integer or binary variables.
    [[nodiscard]] bool hasIntegers() const;
};

/// Throws std::runtime_error if advanced feature declarations are invalid.
void validateModelFeatures(const LpProblem& problem);

/// Returns true when SOS / indicator / semi-variable features are present.
[[nodiscard]] bool hasAdvancedModelFeatures(const LpProblem& problem);

/// Linearize SOS/indicator/semi features into core row/bound form.
LpProblem linearizeModelFeatures(const LpProblem& problem);

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

}  // namespace mipx
