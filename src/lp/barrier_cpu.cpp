// CPU Newton-step backends for the barrier solver:
//   CpuCholeskySolver  — Normal Equations + sparse LL' (Mosek/Gurobi style)
//   CpuAugmentedSolver — Augmented system + sparse LDL' (HiPO/Zanetti-Gondzio)

#include "newton_solver.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <queue>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mipx {

namespace {

// ============================================================================
// Approximate Minimum Degree (AMD) ordering
// ============================================================================
// Based on Amestoy, Davis, Duff (1996) "An Approximate Minimum Degree
// Ordering Algorithm".  Key features vs. the naive implementation:
//   - Approximate external degree (upper bound) instead of exact degree.
//   - Mass elimination of indistinguishable nodes (supervariables).
//   - Aggressive absorption to keep adjacency lists compact.
//   - O(nnz) expected complexity.
//
// Terminology:
//   - Variable (v): an original graph node, possibly absorbed into a principal.
//   - Element (e): an eliminated variable; represents its reach set implicitly.
//   - Supervariable: a set of indistinguishable variables sharing identical
//     adjacency; one is the principal, others are non-principal.

void computeAmd(Index n, const std::vector<Index>& col_starts,
                const std::vector<Index>& row_indices, std::vector<Index>& perm,
                std::vector<Index>& iperm) {
    perm.resize(static_cast<size_t>(n));
    iperm.resize(static_cast<size_t>(n));
    if (n == 0) {
        return;
    }

    // --- Data structures ---
    // For each variable: adjacency to other variables + elements.
    // We store these in a flat array with linked lists per node.
    // adj_var[i] = set of variable neighbors of i (principal only).
    // adj_elem[i] = set of element neighbors of i.

    std::vector<std::unordered_set<Index>> adj_var(static_cast<size_t>(n));
    std::vector<std::unordered_set<Index>> adj_elem(static_cast<size_t>(n));

    // Build symmetric adjacency (no self-loops).
    for (Index j = 0; j < n; ++j) {
        for (Index p = col_starts[j]; p < col_starts[j + 1]; ++p) {
            Index i = row_indices[p];
            if (i != j) {
                adj_var[j].insert(i);
                adj_var[i].insert(j);
            }
        }
    }

    // Element reach sets: for eliminated element e, the set of variables
    // reachable through e that are still alive.
    std::vector<std::unordered_set<Index>> elem_vars(static_cast<size_t>(n));

    // Supervariable tracking.
    // principal[i] = the principal variable for i's supervariable.
    // If principal[i] == i, then i is a principal.
    // weight[i] = number of variables in this supervariable (only valid for principals).
    std::vector<Index> principal(static_cast<size_t>(n));
    std::iota(principal.begin(), principal.end(), 0);
    std::vector<Index> weight(static_cast<size_t>(n), 1);

    // Approximate external degree (upper bound).
    std::vector<Index> degree(static_cast<size_t>(n));
    for (Index i = 0; i < n; ++i) {
        degree[i] = static_cast<Index>(adj_var[i].size());
    }

    // alive[i] = true if variable i has not been eliminated and is a principal.
    std::vector<bool> alive(static_cast<size_t>(n), true);

    // Priority queue: (degree, node). We use lazy deletion.
    using PQ = std::priority_queue<std::pair<Index, Index>, std::vector<std::pair<Index, Index>>,
                                   std::greater<>>;
    PQ heap;
    for (Index i = 0; i < n; ++i) {
        heap.emplace(degree[i], i);
    }

    Index num_eliminated = 0;

    while (num_eliminated < n) {
        // Pick the minimum-degree alive principal variable.
        Index pivot = -1;
        while (!heap.empty()) {
            auto [d, v] = heap.top();
            heap.pop();
            if (alive[v] && principal[v] == v && d == degree[v]) {
                pivot = v;
                break;
            }
        }
        if (pivot < 0) {
            // All remaining nodes must be isolated or non-principal.
            // Find any alive principal.
            for (Index i = 0; i < n; ++i) {
                if (alive[i] && principal[i] == i) {
                    pivot = i;
                    break;
                }
            }
            if (pivot < 0) {
                break;
            }
        }

        // --- Eliminate pivot ---
        // Collect the reach set of pivot: all alive principal variables
        // adjacent to pivot either directly or through shared elements.
        std::unordered_set<Index> reach;

        // Direct variable neighbors.
        for (Index v : adj_var[pivot]) {
            Index pv = principal[v];
            while (pv != principal[pv]) {
                pv = principal[pv];
            }
            principal[v] = pv;
            if (alive[pv] && pv != pivot) {
                reach.insert(pv);
            }
        }

        // Variables reachable through element neighbors.
        for (Index e : adj_elem[pivot]) {
            for (Index v : elem_vars[e]) {
                Index pv = principal[v];
                while (pv != principal[pv]) {
                    pv = principal[pv];
                }
                principal[v] = pv;
                if (alive[pv] && pv != pivot) {
                    reach.insert(pv);
                }
            }
            elem_vars[e].clear();  // Element e is absorbed into new element pivot.
        }

        // Record the elimination of the principal.
        // Non-principal members of its supervariable were already assigned
        // slots during mass elimination.
        perm[num_eliminated] = pivot;
        iperm[pivot] = num_eliminated;
        ++num_eliminated;

        alive[pivot] = false;

        // Create element for pivot.
        elem_vars[pivot] = reach;

        // --- Update reach set ---
        // For each variable v in reach:
        //   1. Remove pivot from adj_var[v].
        //   2. Replace element neighbors that were absorbed.
        //   3. Compute approximate external degree.
        //   4. Detect indistinguishable variables (mass elimination).

        // Aggressive absorption: merge all elements adjacent to pivot into
        // the new element (pivot). For each v in reach, remove old element
        // neighbors that were also adjacent to pivot (they are subsumed).
        std::unordered_set<Index> pivot_elems(adj_elem[pivot].begin(), adj_elem[pivot].end());

        for (Index v : reach) {
            adj_var[v].erase(pivot);

            // Remove absorbed elements, add new element (pivot).
            for (auto it = adj_elem[v].begin(); it != adj_elem[v].end();) {
                if (pivot_elems.count(*it)) {
                    it = adj_elem[v].erase(it);
                } else {
                    ++it;
                }
            }
            adj_elem[v].insert(pivot);

            // Approximate external degree: upper bound.
            // d(v) <= |adj_var(v) \ {pivot}| + sum_{e in adj_elem(v)} |Le|
            //         - weight(v) + 1  (don't count v itself)
            // Clamped to n - num_eliminated - 1.
            Index d = 0;
            for (Index u : adj_var[v]) {
                Index pu = principal[u];
                while (pu != principal[pu]) {
                    pu = principal[pu];
                }
                principal[u] = pu;
                if (alive[pu] && pu != v) {
                    d += weight[pu];
                }
            }
            for (Index e : adj_elem[v]) {
                for (Index u : elem_vars[e]) {
                    Index pu = principal[u];
                    while (pu != principal[pu]) {
                        pu = principal[pu];
                    }
                    principal[u] = pu;
                    if (alive[pu] && pu != v) {
                        d += weight[pu];
                    }
                }
            }
            degree[v] = std::min(d, n - num_eliminated - 1);
        }

        // Mass elimination: detect indistinguishable variables in reach.
        // Two variables u, v are indistinguishable if adj_var(u) == adj_var(v)
        // and adj_elem(u) == adj_elem(v). We use a hash-based approach.
        // For each variable in reach, compute a hash of (adj_var, adj_elem).
        // Group by hash, then verify equality within each group.
        if (reach.size() > 1) {
            auto hash_adj = [&](Index v) -> size_t {
                size_t h = 0;
                for (Index u : adj_var[v]) {
                    Index pu = principal[u];
                    while (pu != principal[pu]) {
                        pu = principal[pu];
                    }
                    if (alive[pu]) {
                        h ^= std::hash<Index>{}(pu) * 2654435761u;
                    }
                }
                for (Index e : adj_elem[v]) {
                    h ^= std::hash<Index>{}(e) * 2246822519u;
                }
                h ^= std::hash<Index>{}(static_cast<Index>(adj_var[v].size()));
                h ^= std::hash<Index>{}(static_cast<Index>(adj_elem[v].size())) << 16;
                return h;
            };

            std::unordered_map<size_t, std::vector<Index>> groups;
            for (Index v : reach) {
                groups[hash_adj(v)].push_back(v);
            }

            for (auto& [h, group] : groups) {
                if (group.size() <= 1) {
                    continue;
                }
                // For each pair in the group, check true equality.
                // Keep first as principal, merge others.
                Index p0 = group[0];
                for (size_t k = 1; k < group.size(); ++k) {
                    Index v = group[k];
                    // Verify same adjacency (sorted comparison).
                    // For efficiency, just check size match + subset (since hash matched).
                    if (adj_var[v].size() != adj_var[p0].size()) {
                        continue;
                    }
                    if (adj_elem[v].size() != adj_elem[p0].size()) {
                        continue;
                    }
                    bool same = true;
                    for (Index u : adj_var[v]) {
                        if (!adj_var[p0].count(u) && u != v && u != p0) {
                            // Account for v being in adj_var[p0] and p0 in adj_var[v].
                            same = false;
                            break;
                        }
                    }
                    if (!same) {
                        continue;
                    }
                    for (Index e : adj_elem[v]) {
                        if (!adj_elem[p0].count(e)) {
                            same = false;
                            break;
                        }
                    }
                    if (!same) {
                        continue;
                    }

                    // Merge v into p0.
                    weight[p0] += weight[v];
                    principal[v] = p0;
                    alive[v] = false;

                    // Clean up v's adjacency (it's no longer principal).
                    for (Index u : adj_var[v]) {
                        adj_var[u].erase(v);
                        if (u != p0) {
                            adj_var[u].insert(p0);
                        }
                    }
                    adj_var[p0].erase(v);
                    adj_var[v].clear();
                    adj_elem[v].clear();

                    // Update element reach sets.
                    for (auto& ev : elem_vars) {
                        if (ev.erase(v)) {
                            ev.insert(p0);
                        }
                    }

                    // v gets eliminated right after p0.
                    perm[num_eliminated] = v;
                    iperm[v] = num_eliminated;
                    ++num_eliminated;
                }
            }
        }

        // Re-insert updated variables into heap.
        for (Index v : reach) {
            if (alive[v] && principal[v] == v) {
                heap.emplace(degree[v], v);
            }
        }

        adj_var[pivot].clear();
        adj_elem[pivot].clear();
    }

    // Fix up: the perm/iperm for supervariable members assigned above may have
    // the principal's slot. We already handle this inline during elimination,
    // but verify correctness.
    // Re-derive iperm from perm.
    for (Index k = 0; k < n; ++k) {
        iperm[perm[k]] = k;
    }
}

// ============================================================================
// Nested Dissection ordering
// ============================================================================
// A recursive bisection approach: partition the graph into two halves with a
// small separator; recursively order each half, then number the separator last.
// This gives O(n log n) fill for 2D meshes and good fill for general graphs.
//
// We use a simple multilevel-ish BFS-based bisection (Kernighan-Lin style
// refinement omitted for simplicity; the quality is sufficient for the
// problems encountered in barrier).

namespace nd_detail {

// BFS from seed to partition vertices into two halves.
void bfsBisect(const std::vector<std::vector<Index>>& adj, const std::vector<Index>& nodes,
               std::vector<Index>& part0, std::vector<Index>& part1,
               std::vector<Index>& separator) {
    if (nodes.empty()) {
        return;
    }

    // Map global IDs → local IDs for BFS.
    std::unordered_map<Index, Index> global_to_local;
    global_to_local.reserve(nodes.size());
    for (size_t i = 0; i < nodes.size(); ++i) {
        global_to_local[nodes[i]] = static_cast<Index>(i);
    }
    Index nv = static_cast<Index>(nodes.size());

    // Build local adjacency.
    std::vector<std::vector<Index>> local_adj(static_cast<size_t>(nv));
    for (Index li = 0; li < nv; ++li) {
        Index gi = nodes[li];
        for (Index gj : adj[gi]) {
            auto it = global_to_local.find(gj);
            if (it != global_to_local.end()) {
                local_adj[li].push_back(it->second);
            }
        }
    }

    // BFS from a pseudo-peripheral node.
    auto bfs_farthest = [&](Index start) -> Index {
        std::vector<Index> dist(static_cast<size_t>(nv), -1);
        std::queue<Index> q;
        dist[start] = 0;
        q.push(start);
        Index farthest = start;
        while (!q.empty()) {
            Index u = q.front();
            q.pop();
            for (Index v : local_adj[u]) {
                if (dist[v] < 0) {
                    dist[v] = dist[u] + 1;
                    q.push(v);
                    farthest = v;
                }
            }
        }
        return farthest;
    };

    // Find pseudo-peripheral pair.
    Index start = bfs_farthest(0);
    start = bfs_farthest(start);

    // BFS from start to get level sets.
    std::vector<Index> dist(static_cast<size_t>(nv), -1);
    std::queue<Index> q;
    dist[start] = 0;
    q.push(start);
    Index max_dist = 0;
    while (!q.empty()) {
        Index u = q.front();
        q.pop();
        for (Index v : local_adj[u]) {
            if (dist[v] < 0) {
                dist[v] = dist[u] + 1;
                if (dist[v] > max_dist) {
                    max_dist = dist[v];
                }
                q.push(v);
            }
        }
    }

    // Bisect at the median level. Separator = vertices at the median level.
    Index median = max_dist / 2;
    for (Index li = 0; li < nv; ++li) {
        Index gi = nodes[li];
        if (dist[li] < 0) {
            // Disconnected component — put in part0.
            part0.push_back(gi);
        } else if (dist[li] < median) {
            part0.push_back(gi);
        } else if (dist[li] == median) {
            separator.push_back(gi);
        } else {
            part1.push_back(gi);
        }
    }

    // If bisection is very unbalanced, fall back to a simple split.
    if (part0.empty() || part1.empty()) {
        part0.clear();
        part1.clear();
        separator.clear();
        Index half = nv / 2;
        for (Index li = 0; li < nv; ++li) {
            if (li < half) {
                part0.push_back(nodes[li]);
            } else {
                part1.push_back(nodes[li]);
            }
        }
    }
}

// Recursive nested dissection.
void ndRecurse(const std::vector<std::vector<Index>>& adj, const std::vector<Index>& nodes,
               std::vector<Index>& perm, Index& next_slot) {
    if (nodes.size() <= 64) {
        // Base case: for small subgraphs, use local AMD ordering via
        // a simple minimum-degree selection.
        // Build local adjacency as CSC for AMD.
        Index nv = static_cast<Index>(nodes.size());
        std::unordered_map<Index, Index> global_to_local;
        global_to_local.reserve(nodes.size());
        for (Index li = 0; li < nv; ++li) {
            global_to_local[nodes[li]] = li;
        }
        std::vector<Index> local_cptr(static_cast<size_t>(nv + 1), 0);
        std::vector<Index> local_ridx;
        for (Index li = 0; li < nv; ++li) {
            Index gi = nodes[li];
            for (Index gj : adj[gi]) {
                auto it = global_to_local.find(gj);
                if (it != global_to_local.end() && it->second != li) {
                    local_ridx.push_back(it->second);
                }
            }
            local_cptr[li + 1] = static_cast<Index>(local_ridx.size());
        }
        std::vector<Index> local_perm, local_iperm;
        computeAmd(nv, local_cptr, local_ridx, local_perm, local_iperm);
        for (Index k = 0; k < nv; ++k) {
            perm[next_slot++] = nodes[local_perm[k]];
        }
        return;
    }

    std::vector<Index> part0, part1, separator;
    bfsBisect(adj, nodes, part0, part1, separator);

    // Recurse on each partition.
    ndRecurse(adj, part0, perm, next_slot);
    ndRecurse(adj, part1, perm, next_slot);

    // Separator is numbered last.
    for (Index v : separator) {
        perm[next_slot++] = v;
    }
}

}  // namespace nd_detail

void computeNd(Index n, const std::vector<Index>& col_starts, const std::vector<Index>& row_indices,
               std::vector<Index>& perm, std::vector<Index>& iperm) {
    perm.resize(static_cast<size_t>(n));
    iperm.resize(static_cast<size_t>(n));
    if (n == 0) {
        return;
    }

    // Build symmetric adjacency lists (no self-loops).
    std::vector<std::vector<Index>> adj(static_cast<size_t>(n));
    for (Index j = 0; j < n; ++j) {
        for (Index p = col_starts[j]; p < col_starts[j + 1]; ++p) {
            Index i = row_indices[p];
            if (i != j) {
                adj[j].push_back(i);
                adj[i].push_back(j);
            }
        }
    }
    // Deduplicate.
    for (Index j = 0; j < n; ++j) {
        auto& a = adj[j];
        std::sort(a.begin(), a.end());
        a.erase(std::unique(a.begin(), a.end()), a.end());
    }

    // Identify connected components via BFS.
    std::vector<bool> visited(static_cast<size_t>(n), false);
    std::vector<std::vector<Index>> components;
    for (Index i = 0; i < n; ++i) {
        if (visited[i]) {
            continue;
        }
        components.emplace_back();
        auto& comp = components.back();
        std::queue<Index> q;
        q.push(i);
        visited[i] = true;
        while (!q.empty()) {
            Index u = q.front();
            q.pop();
            comp.push_back(u);
            for (Index v : adj[u]) {
                if (!visited[v]) {
                    visited[v] = true;
                    q.push(v);
                }
            }
        }
    }

    // Apply nested dissection to each connected component.
    Index next_slot = 0;
    for (auto& comp : components) {
        nd_detail::ndRecurse(adj, comp, perm, next_slot);
    }

    for (Index k = 0; k < n; ++k) {
        iperm[perm[k]] = k;
    }
}

// Auto-select ordering based on problem size.
BarrierOrdering autoSelectOrdering(Index n, Index nnz) {
    // For large problems, nested dissection tends to produce less fill.
    // Threshold: n > 5000 and relatively sparse (nnz/n < 20).
    if (n > 5000 && (nnz < 20 * n)) {
        return BarrierOrdering::Nd;
    }
    return BarrierOrdering::Amd;
}

// ============================================================================
// Symbolic factorization
// ============================================================================

struct SymbolicFact {
    Index n = 0;
    std::vector<Index> perm;   // perm[new] = old
    std::vector<Index> iperm;  // iperm[old] = new

    // L in CSC, below-diagonal only.
    std::vector<Index> l_col_ptr;
    std::vector<Index> l_row_idx;
    Index l_nnz = 0;

    // Left-looking helper: for row i, columns k < i with L[i,k] != 0.
    std::vector<std::vector<Index>> l_row_to_cols;
};

// Permute a CSC lower-triangle pattern using iperm.
static void permutePattern(Index n, const std::vector<Index>& orig_cptr,
                           const std::vector<Index>& orig_ridx, const std::vector<Index>& iperm,
                           std::vector<Index>& perm_cptr, std::vector<Index>& perm_ridx) {
    std::vector<std::vector<Index>> cols(static_cast<size_t>(n));
    for (Index j = 0; j < n; ++j) {
        for (Index p = orig_cptr[j]; p < orig_cptr[j + 1]; ++p) {
            Index i = orig_ridx[p];
            Index ni = iperm[i];
            Index nj = iperm[j];
            if (ni < nj) {
                std::swap(ni, nj);
            }
            cols[nj].push_back(ni);
        }
    }
    perm_cptr.resize(static_cast<size_t>(n + 1));
    perm_ridx.clear();
    perm_cptr[0] = 0;
    for (Index j = 0; j < n; ++j) {
        auto& c = cols[j];
        std::sort(c.begin(), c.end());
        c.erase(std::unique(c.begin(), c.end()), c.end());
        // Remove self-loop (diagonal) — we store below-diagonal only.
        c.erase(std::remove(c.begin(), c.end(), j), c.end());
        for (Index i : c) {
            perm_ridx.push_back(i);
        }
        perm_cptr[j + 1] = static_cast<Index>(perm_ridx.size());
    }
}

// Run ordering + symbolic Cholesky on a CSC lower-triangle pattern.
static void symbolicAnalyze(Index n, const std::vector<Index>& col_starts,
                            const std::vector<Index>& row_indices, SymbolicFact& sym,
                            BarrierOrdering ordering = BarrierOrdering::Auto,
                            bool verbose = false) {
    sym.n = n;
    if (n == 0) {
        sym.perm.clear();
        sym.iperm.clear();
        sym.l_col_ptr.assign(1, 0);
        sym.l_row_idx.clear();
        sym.l_nnz = 0;
        sym.l_row_to_cols.clear();
        return;
    }

    // Select ordering.
    BarrierOrdering effective = ordering;
    if (effective == BarrierOrdering::Auto || effective == BarrierOrdering::CuDSS) {
        Index nnz = static_cast<Index>(row_indices.size());
        effective = autoSelectOrdering(n, nnz);
    }

    if (effective == BarrierOrdering::Nd) {
        computeNd(n, col_starts, row_indices, sym.perm, sym.iperm);
    } else {
        computeAmd(n, col_starts, row_indices, sym.perm, sym.iperm);
    }

    // Permute pattern.
    std::vector<Index> pcptr, pridx;
    permutePattern(n, col_starts, row_indices, sym.iperm, pcptr, pridx);

    // Build row→column adjacency for etree computation.
    std::vector<std::vector<Index>> row_adj(static_cast<size_t>(n));
    for (Index j = 0; j < n; ++j) {
        for (Index p = pcptr[j]; p < pcptr[j + 1]; ++p) {
            Index i = pridx[p];
            if (i > j) {
                row_adj[i].push_back(j);
            }
        }
    }

    // Elimination tree.
    std::vector<Index> parent(static_cast<size_t>(n), -1);
    {
        std::vector<Index> ancestor(static_cast<size_t>(n), -1);
        for (Index k = 0; k < n; ++k) {
            for (Index j : row_adj[k]) {
                Index i = j;
                while (i != -1 && i != k) {
                    Index inext = ancestor[i];
                    ancestor[i] = k;
                    if (inext == -1) {
                        parent[i] = k;
                    }
                    i = inext;
                }
            }
        }
    }

    // Symbolic factorization: walk etree from each off-diagonal entry.
    sym.l_col_ptr.resize(static_cast<size_t>(n + 1));
    sym.l_row_idx.clear();
    sym.l_row_to_cols.assign(static_cast<size_t>(n), {});

    std::vector<Index> flag(static_cast<size_t>(n), -1);

    for (Index j = 0; j < n; ++j) {
        sym.l_col_ptr[j] = static_cast<Index>(sym.l_row_idx.size());
        flag[j] = j;

        for (Index p = pcptr[j]; p < pcptr[j + 1]; ++p) {
            Index i = pridx[p];
            if (i <= j) {
                continue;
            }
            Index k = i;
            while (k != -1 && flag[k] != j) {
                flag[k] = j;
                sym.l_row_idx.push_back(k);
                k = parent[k];
            }
        }

        // Sort this column.
        auto beg = sym.l_row_idx.begin() + sym.l_col_ptr[j];
        auto end = sym.l_row_idx.end();
        std::sort(beg, end);

        // Build row→col index.
        for (auto it = beg; it != end; ++it) {
            sym.l_row_to_cols[*it].push_back(j);
        }
    }
    sym.l_col_ptr[n] = static_cast<Index>(sym.l_row_idx.size());
    sym.l_nnz = static_cast<Index>(sym.l_row_idx.size());

    // Report fill-in quality.
    if (verbose) {
        Index nnz_a = static_cast<Index>(row_indices.size());
        const char* ordering_name = (effective == BarrierOrdering::Nd) ? "ND" : "AMD";
        std::fprintf(stdout, "  Ordering: %s, n=%d, nnz(A)=%d, nnz(L)=%d, fill-in=%.2f\n",
                     ordering_name, n, nnz_a, sym.l_nnz,
                     nnz_a > 0 ? static_cast<double>(sym.l_nnz) / nnz_a : 0.0);
    }
}

// ============================================================================
// NE sparsity pattern: lower-triangle CSC of M = A*A' (structure only)
// ============================================================================

static void computeNePattern(const SparseMatrix& A, Index m, Index n,
                             std::vector<Index>& col_starts, std::vector<Index>& row_indices) {
    // Column→row lists (CSC of A transposed).
    std::vector<std::vector<Index>> col_rows(static_cast<size_t>(n));
    for (Index i = 0; i < m; ++i) {
        auto rv = A.row(i);
        for (Index k = 0; k < rv.size(); ++k) {
            col_rows[rv.indices[k]].push_back(i);
        }
    }
    for (Index k = 0; k < n; ++k) {
        auto& rows = col_rows[k];
        std::sort(rows.begin(), rows.end());
        rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    }

    // Build NE pattern using row-wise scatter: for each column k of A,
    // generate all pairs (rows[a], rows[b]) with a <= b. Use a marker
    // array to avoid duplicate entries instead of std::set.
    std::vector<std::vector<Index>> ne_cols(static_cast<size_t>(m));

    // Estimate total NE nonzeros for reservation.
    Index est_nnz = 0;
    for (Index k = 0; k < n; ++k) {
        Index rk = static_cast<Index>(col_rows[k].size());
        est_nnz += rk * (rk + 1) / 2;
    }
    // This over-counts duplicates, but is a good upper bound.

    // Use a marker array for deduplication.
    std::vector<Index> marker(static_cast<size_t>(m), -1);

    for (Index j = 0; j < m; ++j) {
        // Column j of NE: find all i >= j such that A_row_j and A_row_i
        // share a column.
        // Scatter: for each column k in row j, mark all rows >= j in col k.
        auto rv = A.row(j);
        for (Index p = 0; p < rv.size(); ++p) {
            Index k = rv.indices[p];
            for (Index i : col_rows[k]) {
                if (i >= j && marker[i] != j) {
                    marker[i] = j;
                    ne_cols[j].push_back(i);
                }
            }
        }
        std::sort(ne_cols[j].begin(), ne_cols[j].end());
    }

    col_starts.resize(static_cast<size_t>(m + 1));
    row_indices.clear();
    row_indices.reserve(static_cast<size_t>(est_nnz));
    col_starts[0] = 0;
    for (Index j = 0; j < m; ++j) {
        for (Index i : ne_cols[j]) {
            row_indices.push_back(i);
        }
        col_starts[j + 1] = static_cast<Index>(row_indices.size());
    }
}

// Merge-join rows of A to compute a single NE entry.
[[maybe_unused]]
static Real mergeJoinNE(const SparseMatrix& A, Index row_i, Index row_j,
                        std::span<const Real> theta) {
    auto ri = A.row(row_i);
    auto rj = A.row(row_j);
    Real val = 0.0;
    Index pi = 0, pj = 0;
    while (pi < ri.size() && pj < rj.size()) {
        if (ri.indices[pi] == rj.indices[pj]) {
            val += ri.values[pi] * theta[ri.indices[pi]] * rj.values[pj];
            ++pi;
            ++pj;
        } else if (ri.indices[pi] < rj.indices[pj]) {
            ++pi;
        } else {
            ++pj;
        }
    }
    return val;
}

// ============================================================================
// Augmented sparsity pattern
// ============================================================================

// Lower-triangle CSC of K = [[-Theta^{-1}, A']; [A, delta*I]]
// Dimension: (n + m) × (n + m).
static void computeAugPattern(const SparseMatrix& A, Index m, Index n,
                              std::vector<Index>& col_starts, std::vector<Index>& row_indices) {
    const Index dim = n + m;
    std::vector<std::vector<Index>> cols(static_cast<size_t>(dim));

    // Block (1,1): diagonal only, rows 0..n-1.
    for (Index j = 0; j < n; ++j) {
        cols[j].push_back(j);
    }

    // Block (2,1): A entries → row n+i, column j (lower triangle since n+i > j).
    for (Index i = 0; i < m; ++i) {
        auto rv = A.row(i);
        for (Index k = 0; k < rv.size(); ++k) {
            if (std::abs(rv.values[k]) > 0.0) {
                cols[rv.indices[k]].push_back(n + i);
            }
        }
    }

    // Block (2,2): diagonal only, rows n..n+m-1.
    for (Index i = 0; i < m; ++i) {
        cols[n + i].push_back(n + i);
    }

    // Convert to CSC.
    col_starts.resize(static_cast<size_t>(dim + 1));
    row_indices.clear();
    col_starts[0] = 0;
    for (Index j = 0; j < dim; ++j) {
        auto& c = cols[j];
        std::sort(c.begin(), c.end());
        c.erase(std::unique(c.begin(), c.end()), c.end());
        for (Index i : c) {
            row_indices.push_back(i);
        }
        col_starts[j + 1] = static_cast<Index>(row_indices.size());
    }
}

// ============================================================================
// Forward / backward solves
// ============================================================================

// Forward solve: L * x = b  (L lower triangular with explicit diagonal).
// Overwrites b in-place.
[[maybe_unused]]
static void forwardSolveLL(const SymbolicFact& sym, const std::vector<Real>& l_val,
                           const std::vector<Real>& l_diag, std::span<Real> b) {
    const Index n = sym.n;
    for (Index j = 0; j < n; ++j) {
        b[j] /= l_diag[j];
        Real bj = b[j];
        for (Index p = sym.l_col_ptr[j]; p < sym.l_col_ptr[j + 1]; ++p) {
            b[sym.l_row_idx[p]] -= l_val[p] * bj;
        }
    }
}

// Backward solve: L' * x = b  (L' upper triangular).
// Overwrites b in-place.
[[maybe_unused]]
static void backwardSolveLL(const SymbolicFact& sym, const std::vector<Real>& l_val,
                            const std::vector<Real>& l_diag, std::span<Real> b) {
    const Index n = sym.n;
    for (Index j = n - 1; j >= 0; --j) {
        for (Index p = sym.l_col_ptr[j]; p < sym.l_col_ptr[j + 1]; ++p) {
            b[j] -= l_val[p] * b[sym.l_row_idx[p]];
        }
        b[j] /= l_diag[j];
    }
}

// LDL' forward solve: L * x = b  (L unit lower triangular).
static void forwardSolveLDL(const SymbolicFact& sym, const std::vector<Real>& l_val,
                            std::span<Real> b) {
    const Index n = sym.n;
    for (Index j = 0; j < n; ++j) {
        Real bj = b[j];
        for (Index p = sym.l_col_ptr[j]; p < sym.l_col_ptr[j + 1]; ++p) {
            b[sym.l_row_idx[p]] -= l_val[p] * bj;
        }
    }
}

// LDL' backward solve: L' * x = b  (L' unit upper triangular).
static void backwardSolveLDL(const SymbolicFact& sym, const std::vector<Real>& l_val,
                             std::span<Real> b) {
    const Index n = sym.n;
    for (Index j = n - 1; j >= 0; --j) {
        for (Index p = sym.l_col_ptr[j]; p < sym.l_col_ptr[j + 1]; ++p) {
            b[j] -= l_val[p] * b[sym.l_row_idx[p]];
        }
    }
}

// LDL' diagonal scale: x = D^{-1} * b.
static void diagScaleLDL(const std::vector<Real>& d, std::span<Real> b) {
    for (Index j = 0; j < static_cast<Index>(d.size()); ++j) {
        b[j] /= d[j];
    }
}

// Binary search for L[j,k] in a sorted column of the symbolic factorization.
// Returns index into l_row_idx/l_val, or -1 if not found.
static Index findInColumn(const SymbolicFact& sym, Index k, Index j) {
    auto beg = sym.l_row_idx.begin() + sym.l_col_ptr[k];
    auto end = sym.l_row_idx.begin() + sym.l_col_ptr[k + 1];
    auto it = std::lower_bound(beg, end, j);
    if (it != end && *it == j) {
        return static_cast<Index>(sym.l_col_ptr[k] + (it - beg));
    }
    return -1;
}

}  // anonymous namespace

// ============================================================================
// CpuCholeskySolver: Normal Equations + sparse LL'
// ============================================================================

class CpuCholeskySolver final : public NewtonSolver {
public:
    bool setup(const SparseMatrix& A, Index m, Index n, const BarrierOptions& opts) override {
        a_ = &A;
        m_ = m;
        n_ = n;
        ir_steps_ = opts.ir_steps;

        // NE sparsity pattern.
        std::vector<Index> ne_cptr, ne_ridx;
        computeNePattern(A, m, n, ne_cptr, ne_ridx);

        // Symbolic analysis with selected ordering.
        symbolicAnalyze(m, ne_cptr, ne_ridx, sym_, opts.ordering, opts.verbose);

        l_val_.resize(static_cast<size_t>(sym_.l_nnz), 0.0);
        l_diag_.resize(static_cast<size_t>(m), 0.0);
        theta_.resize(static_cast<size_t>(n), 0.0);
        s_copy_.resize(static_cast<size_t>(n), 0.0);

        return true;
    }

    bool factorize(std::span<const Real> z, std::span<const Real> s, Real reg) override {
        reg_ = std::max(reg, 1e-12);

        for (Index j = 0; j < n_; ++j) {
            theta_[j] = z[j] / std::max(s[j], 1e-20);
            s_copy_[j] = s[j];
        }

        return numericLL();
    }

    bool solveNewton(std::span<const Real> rp, std::span<const Real> rd, std::span<const Real> rc,
                     std::span<Real> dz, std::span<Real> dy, std::span<Real> ds) override {
        // h[j] = rc[j]/s[j] - theta[j]*rd[j]
        std::vector<Real> h(static_cast<size_t>(n_));
        for (Index j = 0; j < n_; ++j) {
            Real sj = std::max(s_copy_[j], 1e-20);
            h[j] = rc[j] / sj - theta_[j] * rd[j];
        }

        // rhs = rp - A*h
        std::vector<Real> ah(static_cast<size_t>(m_), 0.0);
        a_->multiply(h, ah);
        std::vector<Real> rhs(static_cast<size_t>(m_));
        for (Index i = 0; i < m_; ++i) {
            rhs[i] = rp[i] - ah[i];
        }

        // Solve M*dy = rhs using dense Cholesky (no permutation needed).
        std::vector<Real> dy_vec(rhs.begin(), rhs.end());
        denseForwardSolve(dy_vec);
        denseBackwardSolve(dy_vec);

        // Iterative refinement.
        for (Int ir = 0; ir < ir_steps_; ++ir) {
            std::vector<Real> at_dy_ir(static_cast<size_t>(n_), 0.0);
            a_->multiplyTranspose(dy_vec, at_dy_ir);
            for (Index j = 0; j < n_; ++j) {
                at_dy_ir[j] *= theta_[j];
            }
            std::vector<Real> m_dy(static_cast<size_t>(m_), 0.0);
            a_->multiply(at_dy_ir, m_dy);
            for (Index i = 0; i < m_; ++i) {
                m_dy[i] += reg_ * dy_vec[i];
            }

            std::vector<Real> resid(static_cast<size_t>(m_));
            for (Index i = 0; i < m_; ++i) {
                resid[i] = rhs[i] - m_dy[i];
            }

            std::vector<Real> corr(resid.begin(), resid.end());
            denseForwardSolve(corr);
            denseBackwardSolve(corr);

            for (Index i = 0; i < m_; ++i) {
                dy_vec[i] += corr[i];
            }
        }

        for (Index i = 0; i < m_; ++i) {
            dy[i] = dy_vec[i];
        }

        // dz[j] = h[j] + theta[j] * (A'*dy)[j]
        std::vector<Real> at_dy(static_cast<size_t>(n_), 0.0);
        a_->multiplyTranspose(dy, at_dy);
        for (Index j = 0; j < n_; ++j) {
            dz[j] = h[j] + theta_[j] * at_dy[j];
        }

        // ds = rd - A'*dy
        for (Index j = 0; j < n_; ++j) {
            ds[j] = rd[j] - at_dy[j];
        }

        return true;
    }

private:
    // Left-looking numeric LL' factorization with dynamic pivot modification.
    //
    // The NE matrix diagonal uses only the fixed `reg_` (matching what
    // iterative refinement expects).  If a pivot is too small or negative
    // after Cholesky updates, we perturb it just enough to make the factor
    // well-conditioned.  IR then corrects for the perturbation error.
    // Dense NE matrix construction + dense LL' factorization.
    // Correct reference implementation for Netlib-scale problems.
    // For each IPM iteration, forms M = A*Theta*A' + reg*I densely
    // and computes its Cholesky factor L stored densely (column-major,
    // lower triangle).
    bool numericLL() {
        const Index m = m_;
        ne_dense_.assign(static_cast<size_t>(m) * static_cast<size_t>(m), 0.0);

        // Build dense NE matrix M = A * Theta * A' + reg * I.
        // Column-major storage: M[i,j] = ne_dense_[i + j*m].
        std::vector<Real> tmp(static_cast<size_t>(n_), 0.0);
        std::vector<Real> ne_col(static_cast<size_t>(m_), 0.0);

        for (Index j = 0; j < m; ++j) {
            auto rj = a_->row(j);
            for (Index k = 0; k < rj.size(); ++k) {
                tmp[rj.indices[k]] = theta_[rj.indices[k]] * rj.values[k];
            }
            std::fill(ne_col.begin(), ne_col.end(), 0.0);
            a_->multiply(tmp, ne_col);
            for (Index k = 0; k < rj.size(); ++k) {
                tmp[rj.indices[k]] = 0.0;
            }

            for (Index i = 0; i < m; ++i) {
                ne_dense_[i + j * m] = ne_col[i];
            }
            ne_dense_[j + j * m] += reg_;
        }

        // Dense Cholesky LL' factorization (in-place, lower triangle).
        // L is stored in the lower triangle of ne_dense_.
        for (Index j = 0; j < m; ++j) {
            // Diagonal: L[j,j] = sqrt(M[j,j] - sum_{k<j} L[j,k]^2)
            Real sum = 0.0;
            for (Index k = 0; k < j; ++k) {
                Real ljk = ne_dense_[j + k * m];
                sum += ljk * ljk;
            }
            Real diag = ne_dense_[j + j * m] - sum;

            // Pivot perturbation if needed.
            if (diag < 1e-12) {
                diag = std::max(diag + reg_, reg_);
            }

            ne_dense_[j + j * m] = std::sqrt(diag);
            if (!std::isfinite(ne_dense_[j + j * m]) || ne_dense_[j + j * m] < 1e-30) {
                return false;
            }

            Real inv_diag = 1.0 / ne_dense_[j + j * m];

            // Off-diagonal: L[i,j] = (M[i,j] - sum_{k<j} L[i,k]*L[j,k]) / L[j,j]
            for (Index i = j + 1; i < m; ++i) {
                Real s = 0.0;
                for (Index k = 0; k < j; ++k) {
                    s += ne_dense_[i + k * m] * ne_dense_[j + k * m];
                }
                ne_dense_[i + j * m] = (ne_dense_[i + j * m] - s) * inv_diag;
            }
        }

        return true;
    }

    // Dense forward solve: L * x = b (overwrite b).
    void denseForwardSolve(std::span<Real> b) const {
        const Index m = m_;
        for (Index j = 0; j < m; ++j) {
            b[j] /= ne_dense_[j + j * m];
            for (Index i = j + 1; i < m; ++i) {
                b[i] -= ne_dense_[i + j * m] * b[j];
            }
        }
    }

    // Dense backward solve: L' * x = b (overwrite b).
    void denseBackwardSolve(std::span<Real> b) const {
        const Index m = m_;
        for (Index j = m - 1; j >= 0; --j) {
            for (Index i = j + 1; i < m; ++i) {
                b[j] -= ne_dense_[i + j * m] * b[i];
            }
            b[j] /= ne_dense_[j + j * m];
        }
    }

    const SparseMatrix* a_ = nullptr;
    Index m_ = 0, n_ = 0;
    Int ir_steps_ = 2;
    Real reg_ = 1e-8;

    SymbolicFact sym_;
    std::vector<Real> l_val_;
    std::vector<Real> l_diag_;
    std::vector<Real> ne_dense_;  // Dense NE matrix / Cholesky factor (col-major)
    std::vector<Real> theta_;
    std::vector<Real> s_copy_;
};

// ============================================================================
// CpuAugmentedSolver: Augmented system + sparse LDL'
// ============================================================================

class CpuAugmentedSolver final : public NewtonSolver {
public:
    bool setup(const SparseMatrix& A, Index m, Index n, const BarrierOptions& opts) override {
        a_ = &A;
        m_ = m;
        n_ = n;
        ir_steps_ = opts.ir_steps;
        dim_ = n + m;

        // Augmented sparsity pattern.
        std::vector<Index> aug_cptr, aug_ridx;
        computeAugPattern(A, m, n, aug_cptr, aug_ridx);

        // Symbolic analysis with selected ordering.
        symbolicAnalyze(dim_, aug_cptr, aug_ridx, sym_, opts.ordering, opts.verbose);

        l_val_.resize(static_cast<size_t>(sym_.l_nnz), 0.0);
        d_.resize(static_cast<size_t>(dim_), 0.0);

        return true;
    }

    bool factorize(std::span<const Real> z, std::span<const Real> s, Real reg) override {
        reg_ = std::max(reg, 1e-12);

        z_copy_.assign(z.begin(), z.end());
        s_copy_.assign(s.begin(), s.end());

        return numericLDL();
    }

    bool solveNewton(std::span<const Real> rp, std::span<const Real> rd, std::span<const Real> rc,
                     std::span<Real> dz, std::span<Real> dy, std::span<Real> ds) override {
        // Form augmented RHS:
        //   rhs_x[j] = rd[j] - rc[j]/z[j]   for j = 0..n-1
        //   rhs_y[i] = rp[i]                  for i = 0..m-1
        std::vector<Real> rhs(static_cast<size_t>(dim_));
        for (Index j = 0; j < n_; ++j) {
            rhs[j] = rd[j] - rc[j] / std::max(z_copy_[j], 1e-20);
        }
        for (Index i = 0; i < m_; ++i) {
            rhs[n_ + i] = rp[i];
        }

        // Permute RHS.
        std::vector<Real> rhs_perm(static_cast<size_t>(dim_));
        for (Index i = 0; i < dim_; ++i) {
            rhs_perm[sym_.iperm[i]] = rhs[i];
        }

        // Solve K * sol = rhs  via  L * D * L' * sol = rhs.
        forwardSolveLDL(sym_, l_val_, rhs_perm);
        diagScaleLDL(d_, rhs_perm);
        backwardSolveLDL(sym_, l_val_, rhs_perm);

        // Iterative refinement.
        for (Int ir = 0; ir < ir_steps_; ++ir) {
            // Unpermute current solution.
            std::vector<Real> sol(static_cast<size_t>(dim_));
            for (Index i = 0; i < dim_; ++i) {
                sol[i] = rhs_perm[sym_.perm[i]];
            }

            // Compute K * sol.
            std::vector<Real> ksol(static_cast<size_t>(dim_), 0.0);
            // Block (1,1): -Theta^{-1} * sol_x
            for (Index j = 0; j < n_; ++j) {
                ksol[j] = -(s_copy_[j] / std::max(z_copy_[j], 1e-20)) * sol[j];
            }
            // Block (1,2): A' * sol_y
            std::span<const Real> sol_y(sol.data() + n_, static_cast<size_t>(m_));
            std::vector<Real> at_sol_y(static_cast<size_t>(n_), 0.0);
            a_->multiplyTranspose(sol_y, at_sol_y);
            for (Index j = 0; j < n_; ++j) {
                ksol[j] += at_sol_y[j];
            }

            // Block (2,1): A * sol_x
            std::span<const Real> sol_x(sol.data(), static_cast<size_t>(n_));
            std::vector<Real> a_sol_x(static_cast<size_t>(m_), 0.0);
            a_->multiply(sol_x, a_sol_x);
            for (Index i = 0; i < m_; ++i) {
                ksol[n_ + i] = a_sol_x[i] + reg_ * sol[n_ + i];
            }

            // Residual = rhs - K*sol.
            std::vector<Real> resid(static_cast<size_t>(dim_));
            for (Index i = 0; i < dim_; ++i) {
                resid[i] = rhs[i] - ksol[i];
            }

            // Solve correction.
            std::vector<Real> corr(static_cast<size_t>(dim_));
            for (Index i = 0; i < dim_; ++i) {
                corr[sym_.iperm[i]] = resid[i];
            }
            forwardSolveLDL(sym_, l_val_, corr);
            diagScaleLDL(d_, corr);
            backwardSolveLDL(sym_, l_val_, corr);

            for (Index i = 0; i < dim_; ++i) {
                rhs_perm[i] += corr[i];
            }
        }

        // Unpermute and recover.
        std::vector<Real> sol(static_cast<size_t>(dim_));
        for (Index i = 0; i < dim_; ++i) {
            sol[i] = rhs_perm[sym_.perm[i]];
        }

        // dz = sol[0..n-1], dy = sol[n..n+m-1]
        for (Index j = 0; j < n_; ++j) {
            dz[j] = sol[j];
        }
        for (Index i = 0; i < m_; ++i) {
            dy[i] = sol[n_ + i];
        }

        // ds = (rc - s*dz) / z
        for (Index j = 0; j < n_; ++j) {
            ds[j] = (rc[j] - s_copy_[j] * dz[j]) / std::max(z_copy_[j], 1e-20);
        }

        return true;
    }

private:
    // Left-looking numeric LDL' (unit diagonal L, diagonal D can be negative).
    bool numericLDL() {
        std::fill(l_val_.begin(), l_val_.end(), 0.0);
        std::fill(d_.begin(), d_.end(), 0.0);

        std::vector<Real> w(static_cast<size_t>(dim_), 0.0);

        for (Index j = 0; j < dim_; ++j) {
            // Scatter augmented matrix column j into w.
            scatterAugColumn(j, w);

            // Column modifications: for each k < j with L[j,k] != 0.
            for (Index k : sym_.l_row_to_cols[j]) {
                // Find L[j,k] via binary search (l_row_idx is sorted).
                Index pk = findInColumn(sym_, k, j);
                Real ljk = (pk >= 0) ? l_val_[pk] : 0.0;

                Real ljk_dk = ljk * d_[k];

                // Diagonal: w[j] -= ljk * D[k] * ljk
                w[j] -= ljk_dk * ljk;

                // Off-diagonal: w[i] -= L[i,k] * D[k] * ljk
                for (Index p = sym_.l_col_ptr[k]; p < sym_.l_col_ptr[k + 1]; ++p) {
                    if (sym_.l_row_idx[p] > j) {
                        w[sym_.l_row_idx[p]] -= l_val_[p] * ljk_dk;
                    }
                }
            }

            // Pivot (D[j] = w[j], can be negative for indefinite systems).
            d_[j] = w[j];

            // Per-pivot regularization (HiPO-style).
            constexpr Real eps_s = 1e-12;  // static
            constexpr Real eps_d = 1e-7;   // dynamic boost
            if (std::abs(d_[j]) < eps_d) {
                // Determine sign: variables should be negative, constraints positive.
                Index orig = sym_.perm[j];
                Real sign = (orig < n_) ? -1.0 : 1.0;
                d_[j] = sign * std::max(std::abs(d_[j]) + eps_d, eps_d);
            }
            d_[j] += (d_[j] >= 0 ? eps_s : -eps_s);

            if (std::abs(d_[j]) < 1e-30) {
                return false;
            }

            // Scale off-diagonal.
            Real inv_d = 1.0 / d_[j];
            for (Index p = sym_.l_col_ptr[j]; p < sym_.l_col_ptr[j + 1]; ++p) {
                Index i = sym_.l_row_idx[p];
                l_val_[p] = w[i] * inv_d;
                w[i] = 0.0;
            }
            w[j] = 0.0;
        }

        return true;
    }

    // Scatter one column of the augmented matrix into dense workspace.
    void scatterAugColumn(Index j, std::vector<Real>& w) const {
        Index orig_j = sym_.perm[j];

        if (orig_j < n_) {
            // Variable column: diagonal = -s/z  (i.e., -Theta^{-1}).
            w[j] = -(s_copy_[orig_j] / std::max(z_copy_[orig_j], 1e-20));

            // A entries: for each row i with A[i, orig_j] != 0,
            // augmented row = n_ + i, permuted row = iperm[n_ + i].
            auto cv = a_->col(orig_j);
            for (Index k = 0; k < cv.size(); ++k) {
                Index aug_row = n_ + cv.indices[k];
                Index perm_row = sym_.iperm[aug_row];
                if (perm_row > j) {
                    w[perm_row] = cv.values[k];
                }
                // If perm_row < j, this entry was already handled as an
                // above-diagonal entry in column perm_row.
            }
        } else {
            // Constraint column: diagonal = reg (δI block).
            w[j] = reg_;
        }
    }

    const SparseMatrix* a_ = nullptr;
    Index m_ = 0, n_ = 0, dim_ = 0;
    Int ir_steps_ = 2;
    Real reg_ = 1e-8;

    SymbolicFact sym_;
    std::vector<Real> l_val_;
    std::vector<Real> d_;

    std::vector<Real> z_copy_;
    std::vector<Real> s_copy_;
};

// ============================================================================
// Factory functions
// ============================================================================

std::unique_ptr<NewtonSolver> createCpuCholeskySolver() {
    return std::make_unique<CpuCholeskySolver>();
}

std::unique_ptr<NewtonSolver> createCpuAugmentedSolver() {
    return std::make_unique<CpuAugmentedSolver>();
}

}  // namespace mipx
