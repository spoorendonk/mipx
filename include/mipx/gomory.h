#pragma once

#include "mipx/core.h"
#include "mipx/cut_pool.h"
#include "mipx/dual_simplex.h"
#include "mipx/lp_problem.h"

#include <span>
#include <vector>

namespace mipx {

/// Per-call accounting for one GomorySeparator::separate round.
///
/// Every candidate row ends in exactly one of the counters below, so
/// `candidate_rows == rows_skipped + rejected_empty + rejected_screen +
/// rejected_violation + rejected_pool + accepted` and
/// `cuts_built == rejected_screen + rejected_violation + rejected_pool +
/// accepted`. The split matters because the shared numerical screen and the
/// minimum-violation threshold reject for very different reasons, and issue
/// #211 asks for accepted-versus-rejected numbers rather than a raw count.
struct GomoryStats {
    Int candidate_rows = 0;      // fractional basic integer variables examined
    Int rows_skipped = 0;        // row abandoned: nonbasic status/bound unusable
    Int rejected_empty = 0;      // cut had no support left after sparsification
    Int rejected_screen = 0;     // isNumericallySafeCut rejected the cut
    Int rejected_violation = 0;  // violation below min_violation_
    Int rejected_pool = 0;       // CutPool rejected (parallel or low efficacy)
    Int cuts_built = 0;          // cuts fully constructed and offered onward
    Int accepted = 0;            // cuts added to the pool
    Int local_cuts = 0;          // accepted cuts valid only in the current subtree
};

/// Generate Gomory mixed-integer (GMI) cuts from a simplex tableau.
///
/// For each basic integer variable with fractional value, generates a cut
/// from the tableau row using the GMI procedure.
class GomorySeparator {
public:
    GomorySeparator() = default;

    /// Generate Gomory MIR cuts and add them to the cut pool.
    /// Returns the number of cuts generated.
    Int separate(DualSimplexSolver& lp, const LpProblem& problem, std::span<const Real> primals,
                 CutPool& pool);

    /// Set the maximum number of cuts per round.
    void setMaxCuts(Int m) { max_cuts_ = m; }

    /// Set the minimum violation for a cut to be accepted.
    void setMinViolation(Real v) { min_violation_ = v; }

    /// Declare how many leading LP rows are valid in the whole tree.
    ///
    /// A GMI cut that substitutes the logical of row r is only valid where row
    /// r is, so a cut built off a node-local row must be flagged local. The
    /// separator cannot tell a globally valid cut row from a node-local one on
    /// its own -- both are rows the LP grew after load() -- so the caller says
    /// where the globally valid prefix ends. Negative (the default) means
    /// "only the rows of the LpProblem passed to separate()", the safe reading
    /// when the caller says nothing.
    void setGlobalRowCount(Index n) { global_row_count_ = n; }

    /// Accounting for the most recent separate() call.
    [[nodiscard]] const GomoryStats& stats() const { return stats_; }

private:
    Int max_cuts_ = 50;
    Real min_violation_ = 1e-4;
    Index global_row_count_ = -1;
    GomoryStats stats_{};
    static constexpr Real kIntTol = 1e-6;
    static constexpr Real kCoeffTol = 1e-10;
};

}  // namespace mipx
