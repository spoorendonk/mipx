#pragma once

#include <vector>

#include "mipx/core.h"
#include "mipx/lp_solver.h"

namespace mipx {

/// A branching decision applied at a node.
struct BranchDecision {
    Index variable = -1;
    Real bound = 0.0;
    bool is_upper = false;
};

/// A node in the branch-and-bound tree.
struct BnbNode {
    Int id = -1;
    Int parent_id = -1;
    Int depth = 0;
    Real lp_bound = -kInf;
    bool is_solved = false;
    bool is_pruned = false;

    /// Branching decision that created this node (from parent).
    BranchDecision branch;

    /// Basis snapshot for warm-starting LP.
    std::vector<BasisStatus> basis;

    /// Bound changes accumulated from root to this node.
    std::vector<BranchDecision> bound_changes;
};

/// Node selection policies.
enum class NodePolicy {
    BestFirst,
    DepthFirst,
};

/// Priority queue for branch-and-bound nodes.
class NodeQueue {
public:
    explicit NodeQueue(NodePolicy policy = NodePolicy::BestFirst);

    /// Add a node to the queue. Assigns a unique id.
    void push(BnbNode node);

    /// Remove and return the next node to process.
    BnbNode pop();

    /// Best (lowest) LP bound in the queue.
    Real bestBound() const;

    Int size() const;
    bool empty() const;

    /// Remove nodes with bound >= cutoff.
    void prune(Real cutoff);

    NodePolicy policy() const { return policy_; }

private:
    NodePolicy policy_;
    std::vector<BnbNode> nodes_;
    Int next_id_ = 0;
};

}  // namespace mipx
