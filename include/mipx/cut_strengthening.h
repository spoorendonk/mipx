#pragma once

#include "mipx/core.h"
#include "mipx/cut_pool.h"

namespace mipx {

class LpProblem;

/// Tighten the coefficients of a one-sided cut on its integer columns, using
/// the column bounds of @p problem (coefficient tightening, Savelsbergh 1994).
///
/// In <= form, an integer column j with a_j > 0 whose cut is slack for every
/// x_j <= u_j - 1 by d = b - a_j (u_j - 1) - R_j > 0, where R_j is the largest
/// activity the other terms reach within their bounds, has its coefficient
/// lowered to a_j - d and the right-hand side to b - d u_j (symmetrically at
/// l_j for a_j < 0). Every integer point inside the bounds that satisfies the
/// cut still satisfies the result, and inside the bounds the result implies
/// the cut, so the LP relaxation only gets tighter.
///
/// The bounds must hold wherever the cut is used: pass the global bounds for
/// a global cut. An integer column's bounds are first rounded in to integers. Two-sided cuts, cuts
/// touching an unbounded column, cuts with an index outside @p problem and cuts the bounds already
/// satisfy are left alone. d is shrunk by a margin that covers the rounding in the activity sums,
/// which can only weaken the result. Returns the number of coefficients changed; the cut keeps its
/// support.
Int strengthenCut(Cut& cut, const LpProblem& problem);

}  // namespace mipx
