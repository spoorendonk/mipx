#pragma once

#include "mipx/core.h"

namespace mipx {

/// A branching bound applied at a node.
struct BranchDecision {
    Index variable = -1;
    Real bound = 0.0;
    bool is_upper = false;
    Int level = 0;  // Zero for a bound without recorded decision depth.
};

}  // namespace mipx
