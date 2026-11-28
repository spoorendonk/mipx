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
        if (a_nan == b_nan)
            return false;
        return !a_nan;
    }
    return a < b;
}

bool orderedRealLess(Real a, Real b) {
    if (finiteLess(a, b))
        return true;
    if (finiteLess(b, a))
        return false;
    return false;
}

}  // namespace

bool NodeQueue::BestBoundLess::operator()(const NodeKey& a, const NodeKey& b) const {
    if (orderedRealLess(a.lp_bound, b.lp_bound))
        return true;
    if (orderedRealLess(b.lp_bound, a.lp_bound))
        return false;
    return a.id < b.id;
}

bool NodeQueue::BestFirstLess::operator()(const NodeKey& a, const NodeKey& b) const {
    if (orderedRealLess(a.lp_bound, b.lp_bound))
        return true;
    if (orderedRealLess(b.lp_bound, a.lp_bound))
        return false;
    if (a.depth != b.depth)
        return a.depth > b.depth;
    return a.id < b.id;
}

bool NodeQueue::DepthFirstLess::operator()(const NodeKey& a, const NodeKey& b) const {
    if (a.depth != b.depth)
        return a.depth > b.depth;
    if (orderedRealLess(a.lp_bound, b.lp_bound))
        return true;
    if (orderedRealLess(b.lp_bound, a.lp_bound))
        return false;
    return a.id < b.id;
}

bool NodeQueue::BestEstimateLess::operator()(const NodeKey& a, const NodeKey& b) const {
    const Real a_score = std::isfinite(a.estimate) ? a.estimate : a.lp_bound;
    const Real b_score = std::isfinite(b.estimate) ? b.estimate : b.lp_bound;
    if (orderedRealLess(a_score, b_score))
        return true;
    if (orderedRealLess(b_score, a_score))
        return false;
    if (a.depth != b.depth)
        return a.depth > b.depth;
    return a.id < b.id;
}

bool NodeQueue::DepthBiasedLess::operator()(const NodeKey& a, const NodeKey& b) const {
    const Real a_score = a.lp_bound - 1e-3 * static_cast<Real>(a.depth);
    const Real b_score = b.lp_bound - 1e-3 * static_cast<Real>(b.depth);
    if (orderedRealLess(a_score, b_score))
        return true;
    if (orderedRealLess(b_score, a_score))
        return false;
    if (orderedRealLess(a.lp_bound, b.lp_bound))
        return true;
    if (orderedRealLess(b.lp_bound, a.lp_bound))
        return false;
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
    if (it == nodes_.end())
        return;
    const NodeKey key = makeKey(it->second);
    best_bound_.erase(key);
    best_first_.erase(key);
    depth_first_.erase(key);
    best_estimate_.erase(key);
    depth_biased_.erase(key);
    nodes_.erase(it);
}

Int NodeQueue::push(BnbNode node) {
    const Int id = next_id_++;
    node.id = id;
    insertNode(std::move(node));
    return id;
}

Int NodeQueue::nextId() const {
    if (nodes_.empty()) {
        throw std::runtime_error("NodeQueue policy selection on empty queue");
    }
    switch (policy_) {
        case NodePolicy::BestFirst:
            return best_first_.begin()->id;
        case NodePolicy::DepthFirst:
            return depth_first_.begin()->id;
        case NodePolicy::BestEstimate:
            return best_estimate_.begin()->id;
        case NodePolicy::DepthBiased:
            return depth_biased_.begin()->id;
    }
    throw std::runtime_error("NodeQueue policy selection failed");
}

BnbNode NodeQueue::pop() {
    auto node = popById(nextId());
    if (!node) {
        throw std::runtime_error("NodeQueue internal inconsistency on pop");
    }
    return std::move(*node);
}

bool NodeQueue::IncumbentScanBudget::visit() {
    if (work_left == 0 || deadline.passed()) {
        stopped = true;
        return false;
    }
    --work_left;
    ++visits;
    return true;
}

std::optional<Int> NodeQueue::incumbentCompatibleId(std::span<const Real> incumbent,
                                                    std::span<const VarType> types,
                                                    IncumbentScanBudget* budget) const {
    if (incumbent.empty() || incumbent.size() != types.size()) {
        return std::nullopt;
    }
    const auto compatible = [&](const BranchDecision& decision) {
        if (budget != nullptr && !budget->visit()) {
            return false;
        }
        if (decision.variable < 0 || static_cast<std::size_t>(decision.variable) >= types.size()) {
            return false;
        }
        const VarType type = types[decision.variable];
        const Real raw = incumbent[decision.variable];
        const bool discrete =
            type == VarType::Integer || type == VarType::Binary || type == VarType::SemiInteger;
        const Real value = discrete ? std::round(raw) : raw;
        constexpr Real tolerance = 1e-7;
        return decision.is_upper ? value <= decision.bound + tolerance
                                 : value >= decision.bound - tolerance;
    };
    for (const auto& [id, node] : nodes_) {
        if (budget != nullptr && !budget->visit()) {
            break;
        }
        if (std::ranges::all_of(node.bound_changes, compatible)) {
            return id;
        }
        if (budget != nullptr && budget->stopped) {
            break;
        }
    }
    return std::nullopt;
}

std::optional<BnbNode> NodeQueue::popById(Int id) {
    auto it = nodes_.find(id);
    if (it == nodes_.end()) {
        return std::nullopt;
    }
    BnbNode result = std::move(it->second);
    eraseById(id);
    return result;
}

Real NodeQueue::bestBound() const {
    if (best_bound_.empty())
        return kInf;
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
        if (node.lp_bound >= cutoff)
            to_erase.push_back(id);
    }
    for (Int id : to_erase)
        eraseById(id);
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
            while (seen_ids.contains(fresh_id))
                ++fresh_id;
            node.id = fresh_id;
        }
        seen_ids.insert(node.id);
        next_id_ = std::max(next_id_, node.id + 1);
        insertNode(std::move(node));
    }
}

}  // namespace mipx
