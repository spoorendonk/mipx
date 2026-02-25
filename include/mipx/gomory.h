#pragma once

#include <span>
#include <vector>

#include "mipx/core.h"
#include "mipx/cut_pool.h"
#include "mipx/dual_simplex.h"
#include "mipx/lp_problem.h"

namespace mipx {

/// Generate Gomory mixed-integer rounding (MIR) cuts from a simplex tableau.
///
/// For each basic integer variable with fractional value, generates a cut
/// from the tableau row using the MIR procedure.
class GomorySeparator {
public:
    GomorySeparator() = default;

    /// Generate Gomory MIR cuts and add them to the cut pool.
    /// Returns the number of cuts generated.
    Int separate(DualSimplexSolver& lp,
                 const LpProblem& problem,
                 std::span<const Real> primals,
                 CutPool& pool);

    /// Set the maximum number of cuts per round.
    void setMaxCuts(Int m) { max_cuts_ = m; }

    /// Set the minimum violation for a cut to be accepted.
    void setMinViolation(Real v) { min_violation_ = v; }

private:
    Int max_cuts_ = 50;
    Real min_violation_ = 1e-4;
    static constexpr Real kIntTol = 1e-6;
    static constexpr Real kCoeffTol = 1e-10;
};

}  // namespace mipx
