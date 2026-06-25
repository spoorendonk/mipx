#include "mipx/implication_graph.h"

#include <algorithm>
#include <queue>
#include <stack>
#include <unordered_map>
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

bool ImplicationGraph::addImplication(Index from_var, bool from_val, Index to_var, bool to_val) {
    if (!hasIndex(from_var) || !hasIndex(to_var)) {
        return true;
    }

    // Self-contradiction: x=v => x=!v
    if (from_var == to_var && from_val != to_val) {
        // This means from_var cannot take value from_val.
        // Not a graph error, but caller should interpret as a fixing.
        return true;
    }

    // Self-tautology: x=v => x=v
    if (from_var == to_var && from_val == to_val) {
        return true;
    }

    Index s = slot(from_var, from_val);

    // Check for duplicate.
    for (const auto& imp : adj_[s]) {
        if (imp.to_var == to_var && imp.to_val == to_val) {
            return true;
        }
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

const std::vector<BinaryImplication>& ImplicationGraph::implications(Index var, bool val) const {
    if (!hasIndex(var)) {
        return kEmpty;
    }
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
                    if (target == start) {
                        continue;
                    }
                    if (!visited[target]) {
                        continue;
                    }

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
    if (n == 0) {
        return result;
    }

    // Use Tarjan's SCC on the implication digraph.
    // Nodes are literal slots: 2*i + val for internal index i, value val.
    // Edge: slot s has implications adj_[s], each pointing to slot(to_var, to_val).
    // Variables in the same SCC are equivalent.
    const Index num_nodes = 2 * n;

    std::vector<Index> scc_id(static_cast<std::size_t>(num_nodes), -1);
    std::vector<Index> disc(static_cast<std::size_t>(num_nodes), -1);
    std::vector<Index> low(static_cast<std::size_t>(num_nodes), -1);
    std::vector<bool> on_stack(static_cast<std::size_t>(num_nodes), false);
    std::stack<Index> stk;
    Index timer = 0;
    Index num_sccs = 0;

    // Iterative Tarjan's to avoid deep recursion.
    // We use an explicit call stack with frames.
    struct Frame {
        Index node;
        Index adj_idx;  // Next adjacency to process.
    };

    for (Index start = 0; start < num_nodes; ++start) {
        if (disc[start] >= 0) {
            continue;
        }

        std::stack<Frame> call_stack;
        call_stack.push({start, 0});
        disc[start] = low[start] = timer++;
        on_stack[start] = true;
        stk.push(start);

        while (!call_stack.empty()) {
            auto& frame = call_stack.top();
            Index u = frame.node;
            const auto& neighbors = adj_[u];

            if (frame.adj_idx < static_cast<Index>(neighbors.size())) {
                const auto& imp = neighbors[frame.adj_idx];
                ++frame.adj_idx;

                if (!hasIndex(imp.to_var)) {
                    continue;
                }
                Index w = slot(imp.to_var, imp.to_val);

                if (disc[w] < 0) {
                    disc[w] = low[w] = timer++;
                    on_stack[w] = true;
                    stk.push(w);
                    call_stack.push({w, 0});
                } else if (on_stack[w]) {
                    low[u] = std::min(low[u], disc[w]);
                }
            } else {
                // All neighbors processed; check if u is SCC root.
                if (low[u] == disc[u]) {
                    // Pop SCC members.
                    while (true) {
                        Index v = stk.top();
                        stk.pop();
                        on_stack[v] = false;
                        scc_id[v] = num_sccs;
                        if (v == u) {
                            break;
                        }
                    }
                    ++num_sccs;
                }

                call_stack.pop();
                // Update parent's low-link.
                if (!call_stack.empty()) {
                    Index parent = call_stack.top().node;
                    low[parent] = std::min(low[parent], low[u]);
                }
            }
        }
    }

    // Find equivalences: two literals of the same variable in the same SCC
    // is a contradiction (handled elsewhere). Two different variables whose
    // literal pairs share SCCs are equivalent.
    //
    // Same-sense: a=0 and b=0 in same SCC AND a=1 and b=1 in same SCC
    //   => a == b
    // Opposite-sense: a=0 and b=1 in same SCC AND a=1 and b=0 in same SCC
    //   => a == 1-b

    // Group internal indices by their SCC pairs.
    // For each internal index i, compute (scc_id[2i], scc_id[2i+1]).
    // Variables with the same SCC pair are same-sense equivalent.
    // Variables where one's pair is the reverse of another's are opposite-sense.

    // Build map from scc pair -> list of internal indices.
    struct PairHash {
        std::size_t operator()(const std::pair<Index, Index>& p) const {
            return std::hash<Index>()(p.first) ^ (std::hash<Index>()(p.second) * 2654435761ULL);
        }
    };
    std::unordered_map<std::pair<Index, Index>, std::vector<Index>, PairHash> same_map;

    for (Index i = 0; i < n; ++i) {
        Index scc0 = scc_id[2 * i];
        Index scc1 = scc_id[2 * i + 1];
        same_map[{scc0, scc1}].push_back(i);
    }

    // Same-sense equivalences: variables sharing the same (scc0, scc1) pair.
    for (const auto& [key, indices] : same_map) {
        for (std::size_t a = 0; a < indices.size(); ++a) {
            for (std::size_t b = a + 1; b < indices.size(); ++b) {
                Index var_a = binary_vars_[indices[a]];
                Index var_b = binary_vars_[indices[b]];
                if (var_a > var_b) {
                    std::swap(var_a, var_b);
                }
                result.push_back({var_a, var_b, true});
            }
        }
    }

    // Opposite-sense equivalences: variable i with SCC pair (a, b) is the
    // opposite of variable j with the reversed pair (b, a). So pair up the
    // entries of same_map[(a, b)] with those of same_map[(b, a)].
    std::unordered_set<std::pair<Index, Index>, PairHash> same_keys;
    for (const auto& [key, indices] : same_map) {
        same_keys.insert(key);
    }

    for (const auto& [key, indices] : same_map) {
        auto [scc0, scc1] = key;
        if (scc0 == scc1) {
            continue;  // Same-sense already handled.
        }
        auto reversed = std::make_pair(scc1, scc0);
        if (!same_keys.contains(reversed)) {
            continue;
        }
        // Avoid processing both (a,b) and (b,a) -- only process once.
        if (scc0 > scc1) {
            continue;
        }

        auto it = same_map.find(reversed);
        if (it == same_map.end()) {
            continue;
        }
        const auto& other_indices = it->second;

        for (Index i_idx : indices) {
            for (Index j_idx : other_indices) {
                if (i_idx == j_idx) {
                    continue;
                }
                Index var_a = binary_vars_[i_idx];
                Index var_b = binary_vars_[j_idx];
                if (var_a > var_b) {
                    std::swap(var_a, var_b);
                }
                result.push_back({var_a, var_b, false});
            }
        }
    }

    // Deduplicate results.
    std::sort(result.begin(), result.end(),
              [](const VariableEquivalence& a, const VariableEquivalence& b) {
                  if (a.var_a != b.var_a) {
                      return a.var_a < b.var_a;
                  }
                  if (a.var_b != b.var_b) {
                      return a.var_b < b.var_b;
                  }
                  return a.same_sense && !b.same_sense;
              });
    result.erase(std::unique(result.begin(), result.end(),
                             [](const VariableEquivalence& a, const VariableEquivalence& b) {
                                 return a.var_a == b.var_a && a.var_b == b.var_b &&
                                        a.same_sense == b.same_sense;
                             }),
                 result.end());

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
            if (zero_infeasible) {
                break;
            }
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
            if (one_infeasible) {
                break;
            }
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
    if (!hasIndex(var)) {
        return 0;
    }
    Index idx = var_to_index_[var];
    return static_cast<Int>(adj_[2 * idx].size() + adj_[2 * idx + 1].size());
}

bool ImplicationGraph::propagate(Index var, bool val,
                                 std::vector<std::pair<Index, bool>>& propagated) const {
    propagated.clear();
    if (!hasIndex(var)) {
        return true;
    }

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
            if (!hasIndex(imp.to_var)) {
                continue;
            }
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
