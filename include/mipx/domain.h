#pragma once

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
    /// Initialize with problem data.
    void load(const LpProblem& problem);

    /// Set an implication graph to use during propagation. Optional.
    /// The graph pointer must remain valid for the lifetime of the propagator.
    void setImplicationGraph(const ImplicationGraph* graph) { implication_graph_ = graph; }

    /// Attach a clique table for clique-based propagation.
    /// The clique table must outlive the propagator.
    void setCliqueTable(const CliqueTable* table);

    /// True once a clique table has been attached. Lets a caller assert that
    /// the attachment actually happened rather than assuming it.
    [[nodiscard]] bool hasCliqueTable() const { return clique_table_ != nullptr; }

    /// Attach the objective cutoff row c^T x + offset <= cutoff (#117). It is
    /// propagated after the rows and cliques reach a fixpoint, and whatever it
    /// tightens is fed back to them. The propagator must outlive this object.
    void setObjectiveCutoff(const ObjectivePropagator* objective, Real cutoff);

    /// Bounds the objective row tightened since load().
    [[nodiscard]] Int numObjectiveTightenings() const { return objective_tightenings_; }

    /// True when the last propagate() failed on the objective row, i.e. the
    /// domain cannot hold a point better than the cutoff.
    [[nodiscard]] bool objectiveInfeasible() const { return objective_infeasible_; }

    /// Set variable bounds.
    void setBound(Index col, Real lower, Real upper);

    /// Get variable bounds.
    [[nodiscard]] Real getLower(Index col) const;
    [[nodiscard]] Real getUpper(Index col) const;

    /// Run propagation. Returns false if infeasibility detected.
    /// @param max_iterations  Maximum row propagation iterations (0 = unlimited).
    bool propagate(Int max_iterations = 0);

    /// Checkpoint/restore for tree search backtracking.
    void pushCheckpoint();
    void popCheckpoint();

    /// Get number of tightened bounds since last checkpoint.
    [[nodiscard]] Int numTightened() const;

private:
    /// Propagate a single row. Returns false if infeasibility detected.
    bool propagateRow(Index row);

    /// propagate() after `objective_passes` passes of the objective row.
    bool propagate(Int max_iterations, Int objective_passes);

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
    std::vector<BoundChange> changes_;
    std::vector<Index> checkpoint_stack_;

    // Propagation queue
    std::vector<bool> row_in_queue_;
    std::vector<Index> queue_;

    // Optional implication graph for enhanced propagation.
    const ImplicationGraph* implication_graph_ = nullptr;

    // Clique table for clique-based propagation (not owned).
    const CliqueTable* clique_table_ = nullptr;

    // Objective cutoff row (not owned).
    const ObjectivePropagator* objective_ = nullptr;
    Real objective_cutoff_ = kInf;
    Int objective_tightenings_ = 0;
    bool objective_infeasible_ = false;

    static constexpr Real kTol = 1e-8;

    /// Objective passes per propagate() call. Each pass that tightens hands
    /// the rows another round, and a continuous column can creep towards its
    /// limit through such rounds without ever reaching a fixpoint; the cap
    /// only stops tightening early, so it never costs validity.
    static constexpr Int kMaxObjectivePasses = 32;
};

}  // namespace mipx
