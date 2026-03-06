#pragma once

#include <map>
#include <set>
#include <vector>

#include "mipx/core.h"
#include "mipx/cut_pool.h"
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
    Real estimate = -kInf;
    bool is_solved = false;
    bool is_pruned = false;

    /// Branching decision that created this node (from parent).
    BranchDecision branch;

    /// Basis snapshot for warm-starting LP.
    std::vector<BasisStatus> basis;
    Index basis_rows = 0;

    /// Bound changes accumulated from root to this node.
    std::vector<BranchDecision> bound_changes;

    /// Local cuts inherited along this subtree branch.
    std::vector<Cut> local_cuts;
};

/// Node selection policies.
enum class NodePolicy {
    BestFirst,
    DepthFirst,
    BestEstimate,
    DepthBiased,
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
    void setPolicy(NodePolicy policy) { policy_ = policy; }

    /// Remove nodes with bound >= cutoff.
    void prune(Real cutoff);

    /// Extract all nodes from the queue.
    std::vector<BnbNode> takeAll();

    /// Replace queue contents with the given nodes.
    void replaceAll(std::vector<BnbNode> nodes);

    NodePolicy policy() const { return policy_; }

private:
    struct NodeKey {
        Int id = -1;
        Real lp_bound = kInf;
        Real estimate = kInf;
        Int depth = 0;
    };

    struct BestBoundLess {
        bool operator()(const NodeKey& a, const NodeKey& b) const;
    };
    struct BestFirstLess {
        bool operator()(const NodeKey& a, const NodeKey& b) const;
    };
    struct DepthFirstLess {
        bool operator()(const NodeKey& a, const NodeKey& b) const;
    };
    struct BestEstimateLess {
        bool operator()(const NodeKey& a, const NodeKey& b) const;
    };
    struct DepthBiasedLess {
        bool operator()(const NodeKey& a, const NodeKey& b) const;
    };

    [[nodiscard]] static NodeKey makeKey(const BnbNode& node);
    void insertNode(BnbNode node);
    void eraseById(Int id);

    NodePolicy policy_;
    std::map<Int, BnbNode> nodes_;
    std::set<NodeKey, BestBoundLess> best_bound_;
    std::set<NodeKey, BestFirstLess> best_first_;
    std::set<NodeKey, DepthFirstLess> depth_first_;
    std::set<NodeKey, BestEstimateLess> best_estimate_;
    std::set<NodeKey, DepthBiasedLess> depth_biased_;
    Int next_id_ = 0;
};

}  // namespace mipx
