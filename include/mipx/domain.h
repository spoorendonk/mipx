#pragma once

#include <vector>

#include "mipx/core.h"
#include "mipx/lp_problem.h"
#include "mipx/sparse_matrix.h"

namespace mipx {

class CliqueTable;
class ImplicationGraph;

class DomainPropagator {
public:
    /// Initialize with problem data.
    void load(const LpProblem& problem);

    /// Set an implication graph to use during propagation. Optional.
    /// The graph pointer must remain valid for the lifetime of the propagator.
    void setImplicationGraph(const ImplicationGraph* graph) {
        implication_graph_ = graph;
    }

    /// Attach a clique table for clique-based propagation.
    /// The clique table must outlive the propagator.
    void setCliqueTable(const CliqueTable* table);

    /// Set variable bounds.
    void setBound(Index col, Real lower, Real upper);

    /// Get variable bounds.
    [[nodiscard]] Real getLower(Index col) const;
    [[nodiscard]] Real getUpper(Index col) const;

    /// Run propagation. Returns false if infeasibility detected.
    bool propagate();

    /// Checkpoint/restore for tree search backtracking.
    void pushCheckpoint();
    void popCheckpoint();

    /// Get number of tightened bounds since last checkpoint.
    [[nodiscard]] Int numTightened() const;

private:
    /// Propagate a single row. Returns false if infeasibility detected.
    bool propagateRow(Index row);

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

    static constexpr Real kTol = 1e-8;
};

}  // namespace mipx
