#include "mipx/implication_graph.h"

#include <algorithm>
#include <queue>
#include <unordered_set>

namespace mipx {

const std::vector<BinaryImplication> ImplicationGraph::kEmpty{};

void ImplicationGraph::init(Index num_cols, const std::vector<Index>& binary_vars) {
    num_cols_ = num_cols;
    binary_vars_ = binary_vars;
    var_to_index_.assign(static_cast<std::size_t>(num_cols), -1);
    for (Index i = 0; i < static_cast<Index>(binary_vars.size()); ++i) {
        var_to_index_[binary_vars[i]] = i;
    }
    // Two slots per binary variable (value=0 and value=1).
    adj_.assign(2 * binary_vars.size(), {});
    num_implications_ = 0;
}

void ImplicationGraph::clear() {
    for (auto& list : adj_) {
        list.clear();
    }
    num_implications_ = 0;
}

bool ImplicationGraph::addImplication(Index from_var, bool from_val,
                                      Index to_var, bool to_val) {
    if (!hasIndex(from_var) || !hasIndex(to_var)) return true;

    // Self-contradiction: x=v => x=!v
    if (from_var == to_var && from_val != to_val) {
        // This means from_var cannot take value from_val.
        // Not a graph error, but caller should interpret as a fixing.
        return true;
    }

    // Self-tautology: x=v => x=v
    if (from_var == to_var && from_val == to_val) return true;

    Index s = slot(from_var, from_val);

    // Check for duplicate.
    for (const auto& imp : adj_[s]) {
        if (imp.to_var == to_var && imp.to_val == to_val) return true;
    }

    // Check for contradiction with existing implications.
    // If we already have from_var=from_val => to_var=!to_val, that's a contradiction
    // meaning from_var cannot take value from_val (or to_var is fixed).
    bool contradiction = false;
    for (const auto& imp : adj_[s]) {
        if (imp.to_var == to_var && imp.to_val != to_val) {
            contradiction = true;
            break;
        }
    }

    // Add the implication regardless (so detectFixings can find contradictions).
    adj_[s].push_back({to_var, to_val});
    ++num_implications_;

    // Add the contrapositive: to_var=!to_val => from_var=!from_val
    Index contra_s = slot(to_var, !to_val);
    bool has_contra = false;
    for (const auto& imp : adj_[contra_s]) {
        if (imp.to_var == from_var && imp.to_val == !from_val) {
            has_contra = true;
            break;
        }
    }
    if (!has_contra) {
        adj_[contra_s].push_back({from_var, !from_val});
        ++num_implications_;
    }

    return !contradiction;
}

const std::vector<BinaryImplication>& ImplicationGraph::implications(
    Index var, bool val) const {
    if (!hasIndex(var)) return kEmpty;
    return adj_[slot(var, val)];
}

bool ImplicationGraph::isBinary(Index var) const {
    return hasIndex(var);
}

Int ImplicationGraph::computeTransitiveClosure() {
    Int new_implications = 0;
    const Index n = static_cast<Index>(binary_vars_.size());

    // For each starting literal (var, val), BFS through implications
    // and add any missing direct edges.
    for (Index i = 0; i < n; ++i) {
        for (int v = 0; v <= 1; ++v) {
            Index start = 2 * i + v;
            auto& start_adj = adj_[start];

            // BFS to find all reachable literals.
            std::vector<bool> visited(2 * static_cast<std::size_t>(n), false);
            visited[start] = true;
            std::queue<Index> bfs;

            for (const auto& imp : start_adj) {
                Index target = slot(imp.to_var, imp.to_val);
                if (!visited[target]) {
                    visited[target] = true;
                    bfs.push(target);
                }
            }

            while (!bfs.empty()) {
                Index current = bfs.front();
                bfs.pop();

                for (const auto& imp : adj_[current]) {
                    Index target = slot(imp.to_var, imp.to_val);
                    if (!visited[target]) {
                        visited[target] = true;
                        bfs.push(target);
                    }
                }
            }

            // Add missing edges from start to all reachable nodes.
            for (Index j = 0; j < n; ++j) {
                for (int w = 0; w <= 1; ++w) {
                    Index target = 2 * j + w;
                    if (target == start) continue;
                    if (!visited[target]) continue;

                    Index to_var = binary_vars_[j];
                    bool to_val = (w == 1);

                    // Check if edge already exists.
                    bool found = false;
                    for (const auto& imp : start_adj) {
                        if (imp.to_var == to_var && imp.to_val == to_val) {
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        start_adj.push_back({to_var, to_val});
                        ++num_implications_;
                        ++new_implications;
                    }
                }
            }
        }
    }

    return new_implications;
}

std::vector<VariableEquivalence> ImplicationGraph::detectEquivalences() const {
    std::vector<VariableEquivalence> result;
    const Index n = static_cast<Index>(binary_vars_.size());

    for (Index i = 0; i < n; ++i) {
        for (Index j = i + 1; j < n; ++j) {
            Index var_a = binary_vars_[i];
            Index var_b = binary_vars_[j];

            // Check same-sense equivalence: a=0 => b=0 AND b=0 => a=0
            // (equivalently, a=1 => b=1 AND b=1 => a=1)
            bool a0_implies_b0 = false;
            bool b0_implies_a0 = false;
            bool a1_implies_b1 = false;
            bool b1_implies_a1 = false;

            for (const auto& imp : adj_[2 * i]) {  // a=0
                if (imp.to_var == var_b && !imp.to_val) a0_implies_b0 = true;
            }
            for (const auto& imp : adj_[2 * j]) {  // b=0
                if (imp.to_var == var_a && !imp.to_val) b0_implies_a0 = true;
            }
            for (const auto& imp : adj_[2 * i + 1]) {  // a=1
                if (imp.to_var == var_b && imp.to_val) a1_implies_b1 = true;
            }
            for (const auto& imp : adj_[2 * j + 1]) {  // b=1
                if (imp.to_var == var_a && imp.to_val) b1_implies_a1 = true;
            }

            if (a0_implies_b0 && b0_implies_a0 && a1_implies_b1 && b1_implies_a1) {
                result.push_back({var_a, var_b, true});
                continue;
            }

            // Check opposite-sense equivalence: a=0 => b=1 AND b=1 => a=0
            // (equivalently, a=1 => b=0 AND b=0 => a=1)
            bool a0_implies_b1 = false;
            bool b1_implies_a0 = false;
            bool a1_implies_b0 = false;
            bool b0_implies_a1 = false;

            for (const auto& imp : adj_[2 * i]) {
                if (imp.to_var == var_b && imp.to_val) a0_implies_b1 = true;
            }
            for (const auto& imp : adj_[2 * j + 1]) {
                if (imp.to_var == var_a && !imp.to_val) b1_implies_a0 = true;
            }
            for (const auto& imp : adj_[2 * i + 1]) {
                if (imp.to_var == var_b && !imp.to_val) a1_implies_b0 = true;
            }
            for (const auto& imp : adj_[2 * j]) {
                if (imp.to_var == var_a && imp.to_val) b0_implies_a1 = true;
            }

            if (a0_implies_b1 && b1_implies_a0 && a1_implies_b0 && b0_implies_a1) {
                result.push_back({var_a, var_b, false});
            }
        }
    }

    return result;
}

std::vector<std::pair<Index, bool>> ImplicationGraph::detectFixings() const {
    std::vector<std::pair<Index, bool>> result;
    const Index n = static_cast<Index>(binary_vars_.size());

    for (Index i = 0; i < n; ++i) {
        Index var = binary_vars_[i];

        // Check if var=0 leads to contradiction (implies both y=0 and y=1 for some y).
        bool zero_infeasible = false;
        for (const auto& imp1 : adj_[2 * i]) {
            if (zero_infeasible) break;
            for (const auto& imp2 : adj_[2 * i]) {
                if (imp1.to_var == imp2.to_var && imp1.to_val != imp2.to_val) {
                    zero_infeasible = true;
                    break;
                }
            }
        }

        // Check if var=1 leads to contradiction.
        bool one_infeasible = false;
        for (const auto& imp1 : adj_[2 * i + 1]) {
            if (one_infeasible) break;
            for (const auto& imp2 : adj_[2 * i + 1]) {
                if (imp1.to_var == imp2.to_var && imp1.to_val != imp2.to_val) {
                    one_infeasible = true;
                    break;
                }
            }
        }

        if (zero_infeasible && !one_infeasible) {
            result.push_back({var, true});
        } else if (one_infeasible && !zero_infeasible) {
            result.push_back({var, false});
        }
        // Both infeasible = problem infeasible (caller should detect via propagation).
    }

    return result;
}

Int ImplicationGraph::implicationScore(Index var) const {
    if (!hasIndex(var)) return 0;
    Index idx = var_to_index_[var];
    return static_cast<Int>(adj_[2 * idx].size() + adj_[2 * idx + 1].size());
}

bool ImplicationGraph::propagate(Index var, bool val,
                                 std::vector<std::pair<Index, bool>>& propagated) const {
    propagated.clear();
    if (!hasIndex(var)) return true;

    const Index n = static_cast<Index>(binary_vars_.size());
    std::vector<int8_t> state(static_cast<std::size_t>(n), -1);  // -1=unknown, 0=false, 1=true

    Index start_idx = var_to_index_[var];
    state[start_idx] = val ? 1 : 0;

    std::queue<Index> queue;
    queue.push(slot(var, val));

    while (!queue.empty()) {
        Index current = queue.front();
        queue.pop();

        for (const auto& imp : adj_[current]) {
            if (!hasIndex(imp.to_var)) continue;
            Index to_idx = var_to_index_[imp.to_var];
            int8_t required = imp.to_val ? 1 : 0;

            if (state[to_idx] == -1) {
                state[to_idx] = required;
                propagated.push_back({imp.to_var, imp.to_val});
                queue.push(slot(imp.to_var, imp.to_val));
            } else if (state[to_idx] != required) {
                // Contradiction: variable must be both 0 and 1.
                return false;
            }
        }
    }

    return true;
}

Index ImplicationGraph::slot(Index var, bool val) const {
    return 2 * var_to_index_[var] + (val ? 1 : 0);
}

bool ImplicationGraph::hasIndex(Index var) const {
    return var >= 0 && var < num_cols_ && var_to_index_[var] >= 0;
}

}  // namespace mipx
