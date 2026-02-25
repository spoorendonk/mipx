#include "mipx/bnb_node.h"

#include <algorithm>
#include <stdexcept>

namespace mipx {

NodeQueue::NodeQueue(NodePolicy policy) : policy_(policy) {}

void NodeQueue::push(BnbNode node) {
    node.id = next_id_++;
    nodes_.push_back(std::move(node));
}

BnbNode NodeQueue::pop() {
    if (nodes_.empty()) {
        throw std::runtime_error("NodeQueue::pop() called on empty queue");
    }

    auto best = nodes_.begin();

    if (policy_ == NodePolicy::BestFirst) {
        // Select node with smallest lp_bound; break ties by deeper node.
        for (auto it = nodes_.begin() + 1; it != nodes_.end(); ++it) {
            if (it->lp_bound < best->lp_bound ||
                (it->lp_bound == best->lp_bound && it->depth > best->depth)) {
                best = it;
            }
        }
    } else {
        // DepthFirst: deepest node first; break ties by smaller lp_bound.
        for (auto it = nodes_.begin() + 1; it != nodes_.end(); ++it) {
            if (it->depth > best->depth ||
                (it->depth == best->depth && it->lp_bound < best->lp_bound)) {
                best = it;
            }
        }
    }

    BnbNode result = std::move(*best);
    // Swap-and-pop for O(1) removal.
    *best = std::move(nodes_.back());
    nodes_.pop_back();
    return result;
}

Real NodeQueue::bestBound() const {
    if (nodes_.empty()) return kInf;
    Real best = nodes_[0].lp_bound;
    for (size_t i = 1; i < nodes_.size(); ++i) {
        if (nodes_[i].lp_bound < best) {
            best = nodes_[i].lp_bound;
        }
    }
    return best;
}

Int NodeQueue::size() const {
    return static_cast<Int>(nodes_.size());
}

bool NodeQueue::empty() const {
    return nodes_.empty();
}

void NodeQueue::prune(Real cutoff) {
    std::erase_if(nodes_, [cutoff](const BnbNode& n) {
        return n.lp_bound >= cutoff;
    });
}

}  // namespace mipx
