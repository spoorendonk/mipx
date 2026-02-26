#include "mipx/heuristics.h"

#include <algorithm>

namespace mipx {

bool HeuristicBudgetManager::allowRootHeuristic(double total_work_units) const {
    if (max_work_share_ <= 0.0) return true;
    if (total_work_units <= 0.0) return true;
    return total_work_units_ <= max_work_share_ * total_work_units;
}

bool HeuristicBudgetManager::allowTreeHeuristic(Int node_count, double total_work_units) const {
    if (!allowRootHeuristic(total_work_units)) return false;
    return node_count >= next_tree_node_;
}

void HeuristicBudgetManager::recordHeuristicCall(Int node_count, double work_units, bool improved) {
    ++calls_;
    total_work_units_ += std::max(0.0, work_units);
    if (improved) ++successes_;

    const Int base = std::max<Int>(1, base_tree_frequency_);
    if (improved) {
        next_tree_node_ = node_count + base;
        return;
    }
    next_tree_node_ = node_count + currentTreeFrequency();
}

Int HeuristicBudgetManager::currentTreeFrequency() const {
    const Int base = std::max<Int>(1, base_tree_frequency_);
    const Int misses = std::max<Int>(0, calls_ - successes_);
    const Int denom = std::max<Int>(1, successes_ + 1);
    Int scale = 1 + misses / denom;
    scale = std::min(scale, std::max<Int>(1, max_frequency_scale_));
    return base * scale;
}

}  // namespace mipx
