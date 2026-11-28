#pragma once

#include "mipx/dual_simplex.h"
#include "mipx/mip_solver.h"

namespace mipx {

struct RootCutBound {
    Index col;
    Real lower;
    Real upper;
};

struct RootCutoffInference {
    std::optional<Index> first_new_row = std::nullopt;
    const CliqueTable* cliques = nullptr;
    const ImplicationGraph* implications = nullptr;
};

/// Restore a root cut round's rows, staged column bounds and optimal basis.
/// The certificate holds the prior column bounds and mirrors the added rows.
LpResult backOutRootCutRound(DualSimplexSolver& lp, Index keep_rows,
                             std::span<const BasisStatus> basis, LpProblem* certificate,
                             std::span<const RootCutBound> staged_bounds);

/// Replace an optimal working LP only after its cutoff trial also solves.
LpResult tryScopedCutBounds(DualSimplexSolver& lp, std::span<const RootCutBound> staged_bounds);

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

    /// Feed the same cutoff domain from a fresh optimal LP or a stronger
    /// validated incumbent, charging the scans and their woken rows together.
    LpResult updateScheduled(const LpProblem& rows, const DualSimplexSolver& lp, Int& visits_left,
                             DomainPropagationWork& work, PropagationScheduler& history,
                             Deadline deadline, MipCutStats& stats, double work_left = kInf,
                             const RootCutoffInference& inference = {});

    /// Existing cutoff bounds remain valid for a stronger validated cutoff.
    void strengthenCutoff(Real cutoff);

    [[nodiscard]] std::span<const Real> primals() const { return primals_; }
    [[nodiscard]] std::span<const Real> lower() const { return lower_; }
    [[nodiscard]] std::span<const Real> upper() const { return upper_; }

private:
    LpResult resolve(const DualSimplexSolver& lp, std::span<const Real> lp_lower,
                     std::span<const Real> lp_upper, std::vector<Real> lower,
                     std::vector<Real> upper, Deadline deadline, MipCutStats& stats);

    Real cutoff_;
    ObjectivePropagator objective_;
    ReducedCostFixer rc_fixer_;
    std::vector<Real> lower_;
    std::vector<Real> upper_;
    std::vector<Real> primals_;
};

}  // namespace mipx
