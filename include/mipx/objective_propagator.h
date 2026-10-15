#pragma once

#include "mipx/core.h"
#include "mipx/lp_problem.h"

#include <memory>
#include <mutex>
#include <span>
#include <vector>

namespace mipx {

class CliqueTable;
class ImplicationGraph;

/// Objective propagation (#117): the objective cutoff c^T x + offset <= cutoff
/// treated as one more row, with the residual min-activity reasoning the
/// domain propagator applies to real rows.
///
/// The row is used non-strictly, so a point whose objective equals the cutoff
/// survives every tightening; only the infeasibility test uses the pruning
/// tolerance of the tree, reporting a domain whose min activity is at or above
/// cutoff - kPruneTol, where no point improves on the incumbent.
///
/// The model is expected in the solver's internal form: a minimization, with
/// the offset the LP objective carries.
class ObjectivePropagator {
public:
    struct BoundUpdate {
        Index col = -1;
        Real new_lower = 0.0;
        Real new_upper = 0.0;
    };

    void load(const LpProblem& problem);

    /// Back to the unloaded state, in which propagate() tightens nothing.
    void reset() { *this = ObjectivePropagator{}; }

    [[nodiscard]] bool loaded() const { return loaded_; }

    /// Tighten the domain [lower, upper] against the cutoff. Each update holds
    /// the column's new bounds, rounded for integer columns, and is reported
    /// only when it moves a bound by more than the propagation tolerance.
    /// Returns false when the domain cannot hold a point better than the
    /// cutoff, or when a tightening empties a column's domain.
    bool propagate(Real cutoff, std::span<const Real> lower, std::span<const Real> upper,
                   std::vector<BoundUpdate>& updates) const;

    /// Tolerance for "the domain cannot improve on the incumbent", the one
    /// the node queue and processNode prune by.
    static constexpr Real kPruneTol = 1e-6;

private:
    bool loaded_ = false;
    // The nonzero objective coefficients, as a sparse row.
    std::vector<Index> obj_cols_;
    std::vector<Real> obj_vals_;
    Real obj_offset_ = 0.0;
    std::vector<VarType> col_type_;

    static constexpr Real kTol = 1e-8;
};

/// Statistics for objective propagation.
struct ObjPropStats {
    /// Bounds tightened by the objective row inside node propagation.
    Int node_tightenings = 0;
    /// Nodes found unable to improve on the incumbent by the objective row.
    Int node_infeasible = 0;
    /// Global re-propagations run because a new incumbent lowered the cutoff.
    Int repropagations = 0;
    /// Bounds the latest global re-propagation tightened against the root.
    Int global_tightenings = 0;
};

/// Bounds implied globally by the incumbent cutoff: the root domain propagated
/// with every row plus the objective row, recomputed whenever a new incumbent
/// lowers the cutoff and replayed on every node, like the global reduced-cost
/// fixings. Safe to use from concurrent tree workers.
class ObjectiveCutoffBounds {
public:
    /// Take the root model. The objective propagator, implication graph and
    /// clique table must outlive this object; the graph and table may be null.
    void load(const LpProblem& root, const ObjectivePropagator* objective,
              const ImplicationGraph* implications, const CliqueTable* cliques);

    void reset();

    /// Re-propagate the root domain when `cutoff` is below the cutoff of the
    /// current bounds. Returns true when it re-propagated.
    bool refresh(Real cutoff);

    /// Tighten [lower, upper] to the current global bounds, reporting each
    /// column it changes. Returns false when the global bounds are infeasible
    /// (no point improves on the incumbent) or cross the given domain.
    bool enforce(std::vector<Real>& lower, std::vector<Real>& upper,
                 std::vector<Index>& tightened) const;

    /// Number of bounds the current global bounds tighten against the root.
    [[nodiscard]] Int numTightenings() const;

private:
    struct Snapshot {
        Real cutoff = kInf;
        bool infeasible = false;
        std::vector<Index> cols;
        std::vector<Real> lower;
        std::vector<Real> upper;
    };

    [[nodiscard]] std::shared_ptr<const Snapshot> current() const;

    /// A crossing up to this is the propagator's tolerance, not infeasibility.
    static constexpr Real kCrossTol = 1e-8;

    LpProblem root_;
    const ObjectivePropagator* objective_ = nullptr;
    const ImplicationGraph* implications_ = nullptr;
    const CliqueTable* cliques_ = nullptr;

    mutable std::mutex mutex_;
    // Serializes refresh() so two workers seeing the same new incumbent do
    // not both re-propagate; readers only take mutex_.
    std::mutex refresh_mutex_;
    std::shared_ptr<const Snapshot> snapshot_;
};

}  // namespace mipx
