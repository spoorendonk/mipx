#pragma once

#include "mipx/core.h"

namespace mipx {

struct LpProblem;

/// Objective integrality (#261, after SCIP's SCIPprobCheckObjIntegral and
/// SCIPprobScaleObj, Achterberg 2007, Sec. 3.3.7). When every unfixed column
/// with a nonzero objective coefficient takes integer values and the
/// coefficients are integer multiples of one scale s, every solution's
/// objective is offset + k s for an integer k. A solution better than the
/// incumbent is then better by at least s, and a node whose LP bound comes
/// within s of the incumbent holds none.
struct ObjectiveLattice {
    /// The lattice spacing s, > 0 when the objective was found integral.
    Real scale = 0.0;
    /// The objective offset plus the fixed columns' contribution.
    Real offset = 0.0;

    [[nodiscard]] bool detected() const { return scale > 0.0; }

    /// The bound a node's LP objective must stay below to hold a solution
    /// better than the incumbent: incumbent - 1e-6 without a lattice, else the
    /// last lattice point at or below that, plus a margin for the LP bound's
    /// rounding (on the lattice, incumbent - s + margin). Never above
    /// incumbent - 1e-6, which it falls back to when the margin would
    /// swallow s.
    [[nodiscard]] Real pruneBound(Real incumbent) const;
};

/// The lattice of problem's objective, or a default (undetected) one. Columns
/// fixed by their bounds go into the offset whatever their type; any other
/// continuous or semi-continuous column with a nonzero coefficient means no
/// lattice. The scale is found much as SCIP finds its objective scalar: a
/// rational p/q within 1e-9 of each coefficient with q <= 1000, the lcm m of
/// the q's at most 1e6, and s = gcd(m c_j) / m.
[[nodiscard]] ObjectiveLattice detectObjectiveLattice(const LpProblem& problem);

}  // namespace mipx
