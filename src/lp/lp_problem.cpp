#include "mipx/lp_problem.h"

#include <algorithm>

namespace mipx {

bool LpProblem::hasIntegers() const {
    return std::ranges::any_of(col_type, [](VarType t) {
        return t == VarType::Integer || t == VarType::Binary;
    });
}

}  // namespace mipx
