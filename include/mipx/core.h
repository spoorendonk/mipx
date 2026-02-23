#pragma once

namespace mipx {

using Real = double;
using Int = int;
using Index = int;

enum class Status {
    Optimal,
    Infeasible,
    Unbounded,
    IterLimit,
    TimeLimit,
    NodeLimit,
    Error,
};

enum class Sense {
    Minimize,
    Maximize,
};

enum class VarType {
    Continuous,
    Integer,
    Binary,
};

enum class ConstraintSense {
    Leq,
    Geq,
    Eq,
    Range,
};

}  // namespace mipx
