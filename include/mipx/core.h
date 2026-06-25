#pragma once

#include <cmath>

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
    SemiContinuous,
    SemiInteger,
};

enum class ConstraintSense {
    Leq,
    Geq,
    Eq,
    Range,
};

/// Compare objectives respecting optimization sense.
/// A non-finite incumbent (e.g. +/-kInf) means "no incumbent yet", so any
/// candidate is considered better.
[[nodiscard]] inline bool betterObjective(Sense sense, Real candidate, Real incumbent) {
    if (!std::isfinite(incumbent)) {
        return true;
    }
    if (sense == Sense::Minimize) {
        return candidate < incumbent - 1e-6;
    }
    return candidate > incumbent + 1e-6;
}

}  // namespace mipx
