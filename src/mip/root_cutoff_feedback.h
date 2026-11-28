#pragma once

#include "mipx/dual_simplex.h"
#include "mipx/mip_solver.h"

namespace mipx {

struct RootCutBound {
    Index col;
    Real lower;
    Real upper;
};

/// Restore a root cut round's rows, staged column bounds and optimal basis.
/// The certificate holds the prior column bounds and mirrors the added rows.
LpResult backOutRootCutRound(DualSimplexSolver& lp, Index keep_rows,
                             std::span<const BasisStatus> basis, LpProblem* certificate,
                             std::span<const RootCutBound> staged_bounds);

/// A root domain valid for solutions at or below one incumbent objective.
/// Its bounds and LP never become inputs to unconditional proofs.
class RootCutoffFeedback {
public:
    RootCutoffFeedback(const LpProblem& problem, Real cutoff);

    /// Update from a newly solved unconditional cut LP. A failed re-solve
    /// leaves the previous domain and guide intact for cut-round rollback.
    /// With no work left, retain the cut LP and clear any older guide.
    LpResult update(const LpProblem& rows, const DualSimplexSolver& lp, Int& row_visits_left,
                    Deadline deadline, MipCutStats& stats, double work_left = kInf);

    [[nodiscard]] std::span<const Real> primals() const { return primals_; }
    [[nodiscard]] std::span<const Real> lower() const { return lower_; }
    [[nodiscard]] std::span<const Real> upper() const { return upper_; }

private:
    Real cutoff_;
    ObjectivePropagator objective_;
    ReducedCostFixer rc_fixer_;
    std::vector<Real> lower_;
    std::vector<Real> upper_;
    std::vector<Real> primals_;
};

}  // namespace mipx
