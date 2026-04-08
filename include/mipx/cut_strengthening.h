#pragma once

#include <span>

#include "mipx/core.h"
#include "mipx/cut_pool.h"

namespace mipx {

class LpProblem;

/// Tighten cut coefficients using variable bound information.
bool strengthenCut(Cut& cut, const LpProblem& problem);

/// Substitute bound-shifted variables to improve cut quality.
bool complementCut(Cut& cut, const LpProblem& problem,
                   std::span<const Real> primals);

/// Clean up numerically unsafe coefficients and scale for conditioning.
bool makeNumericallySafe(Cut& cut);

}  // namespace mipx
