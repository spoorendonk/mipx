#include "mipx/bnb_node.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mipx {

NodeQueue::NodeQueue(NodePolicy policy) : policy_(policy) {}

namespace {

bool finiteLess(Real a, Real b) {
    const bool a_nan = std::isnan(a);
    const bool b_nan = std::isnan(b);
    if (a_nan || b_nan) {
        if (a_nan == b_nan) return false;
        return !a_nan;
    }
    return a < b;
}

bool orderedRealLess(Real a, Real b) {
    if (finiteLess(a, b)) return true;
    if (finiteLess(b, a)) return false;
    return false;
}

}  // namespace

bool NodeQueue::BestBoundLess::operator()(const NodeKey& a,
                                          const NodeKey& b) const {
    if (orderedRealLess(a.lp_bound, b.lp_bound)) return true;
    if (orderedRealLess(b.lp_bound, a.lp_bound)) return false;
    return a.id < b.id;
}

bool NodeQueue::BestFirstLess::operator()(const NodeKey& a,
                                          const NodeKey& b) const {
    if (orderedRealLess(a.lp_bound, b.lp_bound)) return true;
    if (orderedRealLess(b.lp_bound, a.lp_bound)) return false;
    if (a.depth != b.depth) return a.depth > b.depth;
    return a.id < b.id;
}

bool NodeQueue::DepthFirstLess::operator()(const NodeKey& a,
                                           const NodeKey& b) const {
    if (a.depth != b.depth) return a.depth > b.depth;
    if (orderedRealLess(a.lp_bound, b.lp_bound)) return true;
    if (orderedRealLess(b.lp_bound, a.lp_bound)) return false;
    return a.id < b.id;
}

bool NodeQueue::BestEstimateLess::operator()(const NodeKey& a,
                                             const NodeKey& b) const {
    const Real a_score = std::isfinite(a.estimate) ? a.estimate : a.lp_bound;
    const Real b_score = std::isfinite(b.estimate) ? b.estimate : b.lp_bound;
    if (orderedRealLess(a_score, b_score)) return true;
    if (orderedRealLess(b_score, a_score)) return false;
    if (a.depth != b.depth) return a.depth > b.depth;
    return a.id < b.id;
}

bool NodeQueue::DepthBiasedLess::operator()(const NodeKey& a,
                                            const NodeKey& b) const {
    const Real a_score = a.lp_bound - 1e-3 * static_cast<Real>(a.depth);
    const Real b_score = b.lp_bound - 1e-3 * static_cast<Real>(b.depth);
    if (orderedRealLess(a_score, b_score)) return true;
    if (orderedRealLess(b_score, a_score)) return false;
    if (orderedRealLess(a.lp_bound, b.lp_bound)) return true;
    if (orderedRealLess(b.lp_bound, a.lp_bound)) return false;
    return a.id < b.id;
}

NodeQueue::NodeKey NodeQueue::makeKey(const BnbNode& node) {
    return NodeKey{
        .id = node.id,
        .lp_bound = node.lp_bound,
        .estimate = node.estimate,
        .depth = node.depth,
    };
}

void NodeQueue::insertNode(BnbNode node) {
    if (node.id < 0) {
        throw std::runtime_error("NodeQueue::insertNode() requires non-negative id");
    }
    const NodeKey key = makeKey(node);
    auto [it, inserted] = nodes_.emplace(node.id, std::move(node));
    if (!inserted) {
        throw std::runtime_error("NodeQueue duplicate node id");
    }
    best_bound_.insert(key);
    best_first_.insert(key);
    depth_first_.insert(key);
    best_estimate_.insert(key);
    depth_biased_.insert(key);
}

void NodeQueue::eraseById(Int id) {
    auto it = nodes_.find(id);
    if (it == nodes_.end()) return;
    const NodeKey key = makeKey(it->second);
    best_bound_.erase(key);
    best_first_.erase(key);
    depth_first_.erase(key);
    best_estimate_.erase(key);
    depth_biased_.erase(key);
    nodes_.erase(it);
}

void NodeQueue::push(BnbNode node) {
    node.id = next_id_++;
    insertNode(std::move(node));
}

BnbNode NodeQueue::pop() {
    if (nodes_.empty()) {
        throw std::runtime_error("NodeQueue::pop() called on empty queue");
    }

    const NodeKey* key = nullptr;
    switch (policy_) {
        case NodePolicy::BestFirst: key = &(*best_first_.begin()); break;
        case NodePolicy::DepthFirst: key = &(*depth_first_.begin()); break;
        case NodePolicy::BestEstimate: key = &(*best_estimate_.begin()); break;
        case NodePolicy::DepthBiased: key = &(*depth_biased_.begin()); break;
    }
    if (key == nullptr) {
        throw std::runtime_error("NodeQueue policy selection failed");
    }
    auto it = nodes_.find(key->id);
    if (it == nodes_.end()) {
        throw std::runtime_error("NodeQueue internal inconsistency on pop");
    }
    BnbNode result = std::move(it->second);
    eraseById(key->id);
    return result;
}

Real NodeQueue::bestBound() const {
    if (best_bound_.empty()) return kInf;
    return best_bound_.begin()->lp_bound;
}

Int NodeQueue::size() const {
    return static_cast<Int>(nodes_.size());
}

bool NodeQueue::empty() const {
    return nodes_.empty();
}

void NodeQueue::prune(Real cutoff) {
    std::vector<Int> to_erase;
    to_erase.reserve(nodes_.size() / 4 + 1);
    for (const auto& [id, node] : nodes_) {
        if (node.lp_bound >= cutoff) to_erase.push_back(id);
    }
    for (Int id : to_erase) eraseById(id);
}

std::vector<BnbNode> NodeQueue::takeAll() {
    std::vector<BnbNode> out;
    out.reserve(nodes_.size());
    for (auto& [id, node] : nodes_) {
        static_cast<void>(id);
        out.push_back(std::move(node));
    }
    nodes_.clear();
    best_bound_.clear();
    best_first_.clear();
    depth_first_.clear();
    best_estimate_.clear();
    depth_biased_.clear();
    return out;
}

void NodeQueue::replaceAll(std::vector<BnbNode> nodes) {
    nodes_.clear();
    best_bound_.clear();
    best_first_.clear();
    depth_first_.clear();
    best_estimate_.clear();
    depth_biased_.clear();

    std::set<Int> seen_ids;
    for (auto& node : nodes) {
        bool id_ok = (node.id >= 0) && !seen_ids.contains(node.id);
        if (!id_ok) {
            Int fresh_id = std::max<Int>(0, next_id_);
            while (seen_ids.contains(fresh_id)) ++fresh_id;
            node.id = fresh_id;
        }
        seen_ids.insert(node.id);
        next_id_ = std::max(next_id_, node.id + 1);
        insertNode(std::move(node));
    }
}

}  // namespace mipx
