#pragma once

#include "mipx/bound_reasons.h"
#include "mipx/branch_decision.h"
#include "mipx/core.h"
#include "mipx/lp_problem.h"
#include "mipx/sparse_matrix.h"

#include <vector>

namespace mipx {

class CliqueTable;
class ImplicationGraph;
class ObjectivePropagator;

class DomainPropagator {
public:
    /// Bound tolerance: a column crossed by up to kTol counts as feasible,
    /// and a row violated by up to kTol times its magnitude (the largest of
    /// 1, its finite sides and its finite activity terms).
    static constexpr Real kTol = 1e-8;
    /// Integer column bounds round to the nearest integer within this.
    static constexpr Real kIntRoundTol = 1e-6;
    /// Row propagations per propagate() call, in passes over the rows. Two
    /// integer columns of one row can push each other a unit at a time for
    /// ever when their other bounds are infinite, every step on the undo
    /// stack (#242: 2e7 steps in 7 s, 3 GB). Like the objective cap, it only
    /// stops tightening early.
    static constexpr Int kMaxRowRounds = 100;

    /// Initialize with problem data. When recording, rows after global_rows
    /// are local to this domain. A negative global_rows makes all rows global.
    void load(const LpProblem& problem, bool record_reasons = false, Index global_rows = -1);

    [[nodiscard]] const BoundReasonTrail* boundReasons() const {
        return reasons_ ? &*reasons_ : nullptr;
    }
    void setReasonLevel(Int level) { reason_level_ = level; }
    void setBranchBound(Index col, bool is_upper, Real value, Int level);

    /// Set an implication graph to use during propagation. Optional.
    /// The graph pointer must remain valid for the lifetime of the propagator.
    void setImplicationGraph(const ImplicationGraph* graph, BoundReasonScope scope = {}) {
        implication_graph_ = graph;
        implication_scope_ = scope;
    }

    /// Attach a clique table for clique-based propagation.
    /// The clique table must outlive the propagator.
    void setCliqueTable(const CliqueTable* table, BoundReasonScope scope = {});

    /// True once a clique table has been attached. Lets a caller assert that
    /// the attachment actually happened rather than assuming it.
    [[nodiscard]] bool hasCliqueTable() const { return clique_table_ != nullptr; }

    /// Attach the objective cutoff row c^T x + offset <= cutoff (#117). It is
    /// propagated after the rows and cliques reach a fixpoint, and whatever it
    /// tightens is fed back to them. The propagator must outlive this object.
    /// Restore any deductions made at the old cutoff before weakening it.
    void setObjectiveCutoff(const ObjectivePropagator* objective, Real cutoff);

    /// Bounds the objective row tightened since load().
    [[nodiscard]] Int numObjectiveTightenings() const { return objective_tightenings_; }

    /// True when the last propagate() failed on the objective row, i.e. the
    /// domain cannot hold a point better than the cutoff.
    [[nodiscard]] bool objectiveInfeasible() const { return objective_infeasible_; }

    /// Set variable bounds. A supplied path explains the bounds imported
    /// here without separately applying or scheduling each decision.
    void setBound(Index col, Real lower, Real upper, std::span<const BranchDecision> path = {});

    /// Get variable bounds.
    [[nodiscard]] Real getLower(Index col) const;
    [[nodiscard]] Real getUpper(Index col) const;

    /// Run propagation. Returns false if infeasibility detected.
    /// @param max_iterations  Maximum row propagation iterations (0 = the
    ///        default budget, kMaxRowRounds passes' worth over the rows).
    bool propagate(Int max_iterations = 0);

    /// Checkpoint/restore for tree search backtracking.
    void pushCheckpoint();
    void popCheckpoint();

    /// Get number of tightened bounds since last checkpoint.
    [[nodiscard]] Int numTightened() const;

private:
    void setBound(Index col, Real lower, Real upper, std::span<const BranchDecision> path,
                  bool branch, Int level);
    void recordInputBound(BoundEndpoint endpoint, std::span<const BranchDecision> path, bool branch,
                          Int level);
    [[nodiscard]] bool checkCrossing(Index col);
    BoundReasonId recordRounding(Index col, bool is_upper, Real value, bool applied = true);
    [[nodiscard]] BoundReasonSupports::Snapshot activityReasons(SparseVectorView row,
                                                                bool minimum) const;

    /// Propagate a single row. Returns false if infeasibility detected.
    bool propagateRow(Index row);

    /// propagate() after `objective_passes` passes of the objective row,
    /// with `iterations_left` (> 0) row propagations still allowed.
    bool propagate(Int iterations_left, Int objective_passes);

    /// Propagate the objective cutoff row. Returns false if no point of the
    /// domain improves on the cutoff.
    bool propagateObjective();

    /// Propagate clique constraints. Returns false if infeasibility detected.
    bool propagateCliques();

    /// Record a bound change for the undo stack.
    void recordChange(Index col);

    /// Add all rows containing column col to the propagation queue.
    void enqueueColumn(Index col);

    /// Propagate binary implications when a binary variable gets fixed.
    /// Returns false if infeasibility detected.
    bool propagateImplications(Index col);

    // Problem data
    Index num_cols_ = 0;
    Index num_rows_ = 0;
    SparseMatrix matrix_{0, 0};
    std::vector<Real> row_lower_;
    std::vector<Real> row_upper_;
    std::vector<VarType> col_type_;

    // Current bounds
    std::vector<Real> lower_;
    std::vector<Real> upper_;

    // Change stack for checkpoint/restore
    struct BoundChange {
        Index col;
        Real old_lower;
        Real old_upper;
    };
    struct Checkpoint {
        Index changes;
        Int level;
    };
    std::vector<BoundChange> changes_;
    std::vector<Checkpoint> checkpoint_stack_;

    std::optional<BoundReasonTrail> reasons_;
    Int reason_level_ = 0;
    Index global_rows_ = 0;

    // Propagation queue
    std::vector<bool> row_in_queue_;
    std::vector<Index> queue_;

    // Optional implication graph for enhanced propagation.
    const ImplicationGraph* implication_graph_ = nullptr;
    BoundReasonScope implication_scope_;

    // Clique table for clique-based propagation (not owned).
    const CliqueTable* clique_table_ = nullptr;
    BoundReasonScope clique_scope_;

    // Objective cutoff row (not owned).
    const ObjectivePropagator* objective_ = nullptr;
    Real objective_cutoff_ = kInf;
    Int objective_tightenings_ = 0;
    bool objective_infeasible_ = false;

    /// Objective passes per propagate() call. Each pass that tightens hands
    /// the rows another round, and a continuous column can creep towards its
    /// limit through such rounds without ever reaching a fixpoint; the cap
    /// only stops tightening early, so it never costs validity.
    static constexpr Int kMaxObjectivePasses = 32;
};

}  // namespace mipx
