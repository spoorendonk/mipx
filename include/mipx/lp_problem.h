#pragma once

#include <limits>
#include <string>
#include <vector>

#include "mipx/core.h"
#include "mipx/sparse_matrix.h"

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
    std::vector<std::string> col_names;

    // Constraints (ranged form: row_lower <= Ax <= row_upper).
    Index num_rows = 0;
    SparseMatrix matrix{0, 0};
    std::vector<Real> row_lower;
    std::vector<Real> row_upper;
    std::vector<std::string> row_names;

    Real obj_offset = 0.0;

    /// Returns true if the problem has any integer or binary variables.
    [[nodiscard]] bool hasIntegers() const;
};

}  // namespace mipx
