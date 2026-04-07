#include "mipx/automorphism.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstdint>
#include <functional>
#include <numeric>
#include <stack>
#include <unordered_map>
#include <vector>

#include "mipx/lp_problem.h"

namespace mipx {

// ---------------------------------------------------------------------------
// ColoredGraph
// ---------------------------------------------------------------------------

void ColoredGraph::addEdge(Index u, Index v) {
    adj[static_cast<std::size_t>(u)].push_back(v);
    adj[static_cast<std::size_t>(v)].push_back(u);
}

void ColoredGraph::addVertex(Index color) {
    adj.emplace_back();
    colors.push_back(color);
    ++num_vertices;
}

// ---------------------------------------------------------------------------
// Incidence graph construction
// ---------------------------------------------------------------------------

ColoredGraph buildIncidenceGraph(const LpProblem& problem) {
    ColoredGraph graph;

    // Color encoding for variables:
    // We hash (var_type, obj_bits, lb_bits, ub_bits) into a color.
    // Color offset: 0 for variables, variable colors + 1 for constraints.
    auto hashDouble = [](Real v) -> std::uint64_t {
        return std::bit_cast<std::uint64_t>(v);
    };

    struct VarSig {
        VarType type;
        std::uint64_t obj_bits;
        std::uint64_t lb_bits;
        std::uint64_t ub_bits;
        bool operator==(const VarSig&) const = default;
    };
    struct VarSigHash {
        std::size_t operator()(const VarSig& s) const {
            std::size_t h = std::hash<int>()(static_cast<int>(s.type));
            auto mix = [&](std::uint64_t v) {
                h ^= std::hash<std::uint64_t>()(v) + 0x9e3779b97f4a7c15ULL +
                     (h << 6) + (h >> 2);
            };
            mix(s.obj_bits);
            mix(s.lb_bits);
            mix(s.ub_bits);
            return h;
        }
    };

    std::unordered_map<VarSig, Index, VarSigHash> var_color_map;
    Index next_color = 0;

    // Add variable vertices (0..num_cols-1).
    for (Index j = 0; j < problem.num_cols; ++j) {
        VarSig sig{problem.col_type[j], hashDouble(problem.obj[j]),
                   hashDouble(problem.col_lower[j]),
                   hashDouble(problem.col_upper[j])};
        auto [it, inserted] = var_color_map.try_emplace(sig, next_color);
        if (inserted) ++next_color;
        graph.addVertex(it->second);
    }

    // Color encoding for constraints: hash (row_lower, row_upper).
    struct RowSig {
        std::uint64_t lb_bits;
        std::uint64_t ub_bits;
        bool operator==(const RowSig&) const = default;
    };
    struct RowSigHash {
        std::size_t operator()(const RowSig& s) const {
            std::size_t h = std::hash<std::uint64_t>()(s.lb_bits);
            h ^= std::hash<std::uint64_t>()(s.ub_bits) + 0x9e3779b97f4a7c15ULL +
                 (h << 6) + (h >> 2);
            return h;
        }
    };

    std::unordered_map<RowSig, Index, RowSigHash> row_color_map;

    // Add constraint vertices (num_cols..num_cols+num_rows-1).
    for (Index i = 0; i < problem.num_rows; ++i) {
        RowSig sig{hashDouble(problem.row_lower[i]),
                   hashDouble(problem.row_upper[i])};
        auto [it, inserted] = row_color_map.try_emplace(sig, next_color);
        if (inserted) ++next_color;
        graph.addVertex(it->second);
    }

    // Add edges: for each nonzero a_{i,j}, connect variable j to constraint
    // vertex (num_cols + i). We encode the coefficient value in the edge by
    // using different-colored intermediate vertices for different coefficient
    // values, ensuring the automorphism respects coefficient structure.
    //
    // For each unique coefficient value, we create an edge-color class.
    // We implement this by inserting an intermediate vertex for each
    // (row, col) entry, colored by coefficient value.
    struct CoeffSig {
        std::uint64_t val_bits;
        bool operator==(const CoeffSig&) const = default;
    };
    struct CoeffSigHash {
        std::size_t operator()(const CoeffSig& s) const {
            return std::hash<std::uint64_t>()(s.val_bits);
        }
    };

    std::unordered_map<CoeffSig, Index, CoeffSigHash> coeff_color_map;

    for (Index i = 0; i < problem.num_rows; ++i) {
        auto row = problem.matrix.row(i);
        for (Index k = 0; k < row.size(); ++k) {
            Index col = row.indices[k];
            Real val = row.values[k];

            CoeffSig sig{hashDouble(val)};
            auto [it, inserted] = coeff_color_map.try_emplace(sig, next_color);
            if (inserted) ++next_color;

            // Create intermediate vertex with coefficient color.
            Index mid = graph.num_vertices;
            graph.addVertex(it->second);
            graph.addEdge(col, mid);
            graph.addEdge(mid, problem.num_cols + i);
        }
    }

    return graph;
}

// ---------------------------------------------------------------------------
// Permutation utilities
// ---------------------------------------------------------------------------

void applyPermutation(const Permutation& perm, std::vector<Index>& vec) {
    for (auto& v : vec) {
        if (v >= 0 && v < static_cast<Index>(perm.size())) {
            v = perm[static_cast<std::size_t>(v)];
        }
    }
}

Permutation composePermutations(const Permutation& a, const Permutation& b) {
    Permutation result(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        result[i] = b[static_cast<std::size_t>(a[i])];
    }
    return result;
}

Permutation inversePermutation(const Permutation& perm) {
    Permutation inv(perm.size());
    for (std::size_t i = 0; i < perm.size(); ++i) {
        inv[static_cast<std::size_t>(perm[i])] = static_cast<Index>(i);
    }
    return inv;
}

bool isIdentity(const Permutation& perm) {
    for (std::size_t i = 0; i < perm.size(); ++i) {
        if (perm[i] != static_cast<Index>(i)) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Orbit computation from generators (union-find)
// ---------------------------------------------------------------------------

namespace {

class UnionFind {
public:
    explicit UnionFind(Index n) : parent_(static_cast<std::size_t>(n)),
                                   rank_(static_cast<std::size_t>(n), 0) {
        std::iota(parent_.begin(), parent_.end(), 0);
    }

    Index find(Index x) {
        while (parent_[x] != x) {
            parent_[x] = parent_[parent_[x]];
            x = parent_[x];
        }
        return x;
    }

    void unite(Index x, Index y) {
        x = find(x);
        y = find(y);
        if (x == y) return;
        if (rank_[x] < rank_[y]) std::swap(x, y);
        parent_[y] = x;
        if (rank_[x] == rank_[y]) ++rank_[x];
    }

private:
    std::vector<Index> parent_;
    std::vector<Index> rank_;
};

}  // namespace

std::vector<std::vector<Index>> computeOrbitsFromGenerators(
    const std::vector<Permutation>& generators, Index n) {
    UnionFind uf(n);
    for (const auto& perm : generators) {
        for (Index i = 0; i < n && i < static_cast<Index>(perm.size()); ++i) {
            if (perm[i] != i) {
                uf.unite(i, perm[i]);
            }
        }
    }

    std::unordered_map<Index, std::vector<Index>> orbit_map;
    for (Index i = 0; i < n; ++i) {
        orbit_map[uf.find(i)].push_back(i);
    }

    std::vector<std::vector<Index>> orbits;
    orbits.reserve(orbit_map.size());
    for (auto& [root, members] : orbit_map) {
        if (members.size() > 1) {
            std::sort(members.begin(), members.end());
            orbits.push_back(std::move(members));
        }
    }
    std::sort(orbits.begin(), orbits.end(),
              [](const auto& a, const auto& b) { return a.front() < b.front(); });
    return orbits;
}

std::vector<std::vector<Index>> computeVariableOrbits(
    const std::vector<Permutation>& generators, Index num_vars) {
    // Only consider how generators permute the first num_vars elements.
    UnionFind uf(num_vars);
    for (const auto& perm : generators) {
        for (Index i = 0; i < num_vars && i < static_cast<Index>(perm.size()); ++i) {
            Index target = perm[i];
            if (target < num_vars && target != i) {
                uf.unite(i, target);
            }
        }
    }

    std::unordered_map<Index, std::vector<Index>> orbit_map;
    for (Index i = 0; i < num_vars; ++i) {
        orbit_map[uf.find(i)].push_back(i);
    }

    std::vector<std::vector<Index>> orbits;
    for (auto& [root, members] : orbit_map) {
        if (members.size() > 1) {
            std::sort(members.begin(), members.end());
            orbits.push_back(std::move(members));
        }
    }
    std::sort(orbits.begin(), orbits.end(),
              [](const auto& a, const auto& b) { return a.front() < b.front(); });
    return orbits;
}

// ---------------------------------------------------------------------------
// Partition refinement + individualization-refinement automorphism algorithm
// ---------------------------------------------------------------------------
//
// This is a simplified nauty-style algorithm:
//   1. Start with a partition colored by initial vertex colors.
//   2. Refine partition by degree-to-cells: split cells by adjacency counts
//      to each other cell.
//   3. If partition is discrete, we have a candidate permutation.
//   4. If not discrete, individualize: pick a non-singleton cell, try each
//      vertex as a singleton, and recurse.
//   5. First leaf defines the "canonical form." Subsequent leaves that match
//      yield automorphism generators.
//   6. Pruning: we track the canonical representative (first leaf found)
//      and prune branches that cannot lead to it.

namespace {

/// A partition of vertices into ordered cells.
/// Each cell is a contiguous range in the perm array.
/// cell_starts_[k] gives the beginning of cell k in perm_.
struct Partition {
    std::vector<Index> perm;          // vertex ordering
    std::vector<Index> inv_perm;      // position of each vertex in perm
    std::vector<Index> cell_starts;   // start positions of cells (sorted)
    Index n = 0;

    explicit Partition(Index num_vertices)
        : perm(static_cast<std::size_t>(num_vertices)),
          inv_perm(static_cast<std::size_t>(num_vertices)),
          n(num_vertices) {
        std::iota(perm.begin(), perm.end(), 0);
        std::iota(inv_perm.begin(), inv_perm.end(), 0);
        cell_starts = {0};
    }

    Partition() = default;

    /// Number of cells.
    [[nodiscard]] Index numCells() const {
        return static_cast<Index>(cell_starts.size());
    }

    /// Is the partition discrete (all singletons)?
    [[nodiscard]] bool isDiscrete() const { return numCells() == n; }

    /// Size of cell containing position pos.
    [[nodiscard]] Index cellSize(Index cell_idx) const {
        Index start = cell_starts[cell_idx];
        Index end = (cell_idx + 1 < numCells()) ? cell_starts[cell_idx + 1] : n;
        return end - start;
    }

    /// Find the index of the first non-singleton cell. Returns -1 if discrete.
    [[nodiscard]] Index firstNonSingletonCell() const {
        for (Index c = 0; c < numCells(); ++c) {
            if (cellSize(c) > 1) return c;
        }
        return -1;
    }

    /// Get the cell index for a given vertex.
    [[nodiscard]] Index cellOf(Index vertex) const {
        Index pos = inv_perm[vertex];
        // Binary search in cell_starts for the largest start <= pos.
        auto it = std::upper_bound(cell_starts.begin(), cell_starts.end(), pos);
        return static_cast<Index>(std::distance(cell_starts.begin(), it)) - 1;
    }

    /// Extract the permutation that maps original vertex ordering to
    /// the canonical ordering defined by this discrete partition.
    [[nodiscard]] Permutation toPermutation() const {
        // perm[pos] = vertex at position pos
        // This defines a mapping: vertex perm[pos] -> position pos
        // But we want: for vertex i, which vertex does it map to?
        // If both first and second leaves are discrete, the automorphism
        // is: first.perm[pos] -> second.perm[pos], i.e.,
        //   aut[first.perm[pos]] = second.perm[pos]
        return perm;
    }
};

/// Refine a partition by adjacency to a "target" cell.
/// For each cell, split it by the number of neighbors each vertex has
/// in the target cell. Returns true if any split occurred.
bool refineByCell(Partition& pi,
                  const std::vector<std::vector<Index>>& adj,
                  Index target_cell_start, Index target_cell_end,
                  double& work) {
    bool changed = false;
    // Count adjacency of each vertex to the target cell.
    std::vector<Index> degree(static_cast<std::size_t>(pi.n), 0);
    for (Index pos = target_cell_start; pos < target_cell_end; ++pos) {
        Index v = pi.perm[pos];
        for (Index neighbor : adj[v]) {
            ++degree[neighbor];
            work += 1.0;
        }
    }

    // For each cell, check if vertices have different degrees.
    std::vector<Index> new_cell_starts;
    new_cell_starts.reserve(pi.cell_starts.size() * 2);

    for (Index c = 0; c < pi.numCells(); ++c) {
        Index start = pi.cell_starts[c];
        Index end = (c + 1 < pi.numCells()) ? pi.cell_starts[c + 1] : pi.n;
        new_cell_starts.push_back(start);

        if (end - start <= 1) continue;

        // Sort vertices in this cell by degree.
        std::sort(pi.perm.begin() + start, pi.perm.begin() + end,
                  [&](Index a, Index b) { return degree[a] < degree[b]; });

        // Split by degree boundaries.
        for (Index pos = start + 1; pos < end; ++pos) {
            if (degree[pi.perm[pos]] != degree[pi.perm[pos - 1]]) {
                new_cell_starts.push_back(pos);
                changed = true;
            }
        }
    }

    if (changed) {
        pi.cell_starts = std::move(new_cell_starts);
        // Rebuild inverse permutation.
        for (Index pos = 0; pos < pi.n; ++pos) {
            pi.inv_perm[pi.perm[pos]] = pos;
        }
    }

    return changed;
}

/// Full refinement: repeatedly refine by all cells until stable.
void refine(Partition& pi, const std::vector<std::vector<Index>>& adj,
            double& work) {
    bool changed = true;
    while (changed) {
        changed = false;
        for (Index c = 0; c < pi.numCells(); ++c) {
            Index start = pi.cell_starts[c];
            Index end = (c + 1 < pi.numCells()) ? pi.cell_starts[c + 1] : pi.n;
            if (refineByCell(pi, adj, start, end, work)) {
                changed = true;
                // Restart refinement with the new partition.
                break;
            }
        }
    }
}

/// Create initial partition from vertex colors.
Partition makeInitialPartition(const std::vector<Index>& colors) {
    Index n = static_cast<Index>(colors.size());
    Partition pi(n);

    // Sort vertices by color.
    std::sort(pi.perm.begin(), pi.perm.end(),
              [&](Index a, Index b) {
                  return colors[a] < colors[b];
              });

    // Build cell starts.
    pi.cell_starts.clear();
    pi.cell_starts.push_back(0);
    for (Index pos = 1; pos < n; ++pos) {
        if (colors[pi.perm[pos]] != colors[pi.perm[pos - 1]]) {
            pi.cell_starts.push_back(pos);
        }
    }

    // Rebuild inverse permutation.
    for (Index pos = 0; pos < n; ++pos) {
        pi.inv_perm[pi.perm[pos]] = pos;
    }

    return pi;
}

/// Size limit for the search tree to prevent combinatorial explosion.
static constexpr int kMaxSearchNodes = 50000;

/// Search tree node for individualization-refinement.
struct SearchState {
    Partition pi;
    Index individualized_cell;
    Index individualized_pos;  // within the cell
};

}  // namespace

AutomorphismResult computeAutomorphisms(const ColoredGraph& graph,
                                        Index num_variable_vertices) {
    AutomorphismResult result;
    result.num_vertices = graph.num_vertices;
    result.num_variable_vertices = num_variable_vertices;

    if (graph.num_vertices == 0) return result;

    // Build initial partition from colors.
    Partition initial = makeInitialPartition(graph.colors);

    // Initial refinement.
    refine(initial, graph.adj, result.work_units);

    if (initial.isDiscrete()) {
        // No symmetry possible: the partition is already fully refined.
        result.orbits = computeVariableOrbits(result.generators, num_variable_vertices);
        return result;
    }

    // We use a depth-first search with individualization-refinement.
    // The first discrete leaf found defines the "canonical labeling."
    // Subsequent discrete leaves that produce a valid automorphism are
    // recorded as generators.

    Permutation canonical_leaf;  // perm from first leaf
    bool have_canonical = false;
    int search_nodes = 0;

    // DFS using explicit stack.
    struct Frame {
        Partition pi;
        Index cell_idx;          // which non-singleton cell to individualize
        Index next_pos_in_cell;  // next vertex position within the cell to try
        Index cell_start;
        Index cell_end;
    };

    std::stack<Frame> stack;

    {
        Frame root;
        root.pi = initial;
        root.cell_idx = root.pi.firstNonSingletonCell();
        root.cell_start = root.pi.cell_starts[root.cell_idx];
        root.cell_end = (root.cell_idx + 1 < root.pi.numCells())
                            ? root.pi.cell_starts[root.cell_idx + 1]
                            : root.pi.n;
        root.next_pos_in_cell = root.cell_start;
        stack.push(std::move(root));
    }

    while (!stack.empty()) {
        if (search_nodes >= kMaxSearchNodes) break;

        auto& frame = stack.top();

        if (frame.next_pos_in_cell >= frame.cell_end) {
            stack.pop();
            continue;
        }

        // Individualize: make vertex at frame.next_pos_in_cell a singleton.
        Index pos = frame.next_pos_in_cell;
        ++frame.next_pos_in_cell;
        ++search_nodes;
        result.work_units += 1.0;

        Partition child = frame.pi;

        // Move the selected vertex to the front of its cell and split.
        Index vertex = child.perm[pos];
        // Swap with cell start.
        Index swap_pos = frame.cell_start;
        if (pos != swap_pos) {
            Index other = child.perm[swap_pos];
            child.perm[swap_pos] = vertex;
            child.perm[pos] = other;
            child.inv_perm[vertex] = swap_pos;
            child.inv_perm[other] = pos;
        }

        // Insert new cell boundary after the singleton.
        auto insert_it = std::lower_bound(child.cell_starts.begin(),
                                          child.cell_starts.end(),
                                          swap_pos + 1);
        if (insert_it == child.cell_starts.end() || *insert_it != swap_pos + 1) {
            child.cell_starts.insert(insert_it, swap_pos + 1);
        }
        // Ensure the singleton's cell start is present.
        auto start_it = std::lower_bound(child.cell_starts.begin(),
                                         child.cell_starts.end(),
                                         swap_pos);
        if (start_it == child.cell_starts.end() || *start_it != swap_pos) {
            child.cell_starts.insert(start_it, swap_pos);
        }

        // Refine.
        refine(child, graph.adj, result.work_units);

        if (child.isDiscrete()) {
            // We have a discrete partition = a "labeling."
            if (!have_canonical) {
                canonical_leaf = child.perm;
                have_canonical = true;
            } else {
                // Build automorphism: canonical_leaf[pos] -> child.perm[pos].
                // automorphism maps vertex canonical_leaf[pos] to child.perm[pos].
                Permutation aut(static_cast<std::size_t>(graph.num_vertices));
                for (Index i = 0; i < graph.num_vertices; ++i) {
                    aut[canonical_leaf[i]] = child.perm[i];
                }
                if (!isIdentity(aut)) {
                    // Verify this is actually an automorphism (edges preserved).
                    bool valid = true;
                    for (Index v = 0; v < graph.num_vertices && valid; ++v) {
                        if (graph.colors[v] != graph.colors[aut[v]]) {
                            valid = false;
                            break;
                        }
                        // Check edges: for each neighbor u of v, aut[u] must
                        // be a neighbor of aut[v].
                        for (Index u : graph.adj[v]) {
                            result.work_units += 1.0;
                            Index au = aut[u];
                            Index av = aut[v];
                            bool found = false;
                            for (Index w : graph.adj[av]) {
                                if (w == au) { found = true; break; }
                            }
                            if (!found) { valid = false; break; }
                        }
                    }
                    if (valid) {
                        result.generators.push_back(std::move(aut));
                    }
                }
            }
        } else {
            // Push new frame for deeper individualization.
            Index next_cell = child.firstNonSingletonCell();
            if (next_cell >= 0) {
                Frame next;
                next.cell_idx = next_cell;
                next.cell_start = child.cell_starts[next_cell];
                next.cell_end = (next_cell + 1 < child.numCells())
                                    ? child.cell_starts[next_cell + 1]
                                    : child.n;
                next.next_pos_in_cell = next.cell_start;
                next.pi = std::move(child);
                stack.push(std::move(next));
            }
        }
    }

    // Compute variable orbits from generators.
    result.orbits = computeVariableOrbits(result.generators, num_variable_vertices);
    return result;
}

}  // namespace mipx
