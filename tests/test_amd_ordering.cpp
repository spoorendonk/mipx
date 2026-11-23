#include "mipx/amd_ordering.h"
#include "mipx/sparse_cholesky.h"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

using namespace mipx;

namespace {

// Lower-triangle compressed columns of a symmetric pattern, diagonal included.
struct Graph {
    Index n = 0;
    std::vector<Index> col_starts;
    std::vector<Index> rows;
};

Graph fromEdges(Index n, std::vector<std::pair<Index, Index>> edges) {
    for (Index i = 0; i < n; ++i) {
        edges.emplace_back(i, i);
    }
    for (auto& [a, b] : edges) {
        if (a < b) {
            std::swap(a, b);
        }
    }
    std::sort(edges.begin(), edges.end(), [](const auto& x, const auto& y) {
        return x.second != y.second ? x.second < y.second : x.first < y.first;
    });
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
    Graph g;
    g.n = n;
    g.col_starts.assign(static_cast<std::size_t>(n) + 1, 0);
    for (const auto& [r, c] : edges) {
        ++g.col_starts[static_cast<std::size_t>(c) + 1];
        g.rows.push_back(r);
    }
    std::partial_sum(g.col_starts.begin(), g.col_starts.end(), g.col_starts.begin());
    return g;
}

// Both triangles of a lower-triangle pattern, each off-diagonal entry of
// the lower one twice.
Graph bothTriangles(const Graph& lower) {
    Graph full;
    full.n = lower.n;
    std::vector<std::vector<Index>> cols(static_cast<std::size_t>(lower.n));
    for (Index c = 0; c < lower.n; ++c) {
        for (Index p = lower.col_starts[c]; p < lower.col_starts[c + 1]; ++p) {
            const Index r = lower.rows[p];
            cols[c].push_back(r);
            cols[r].push_back(c);
            cols[c].push_back(r);
        }
    }
    full.col_starts.push_back(0);
    for (const auto& col : cols) {
        full.rows.insert(full.rows.end(), col.begin(), col.end());
        full.col_starts.push_back(static_cast<Index>(full.rows.size()));
    }
    return full;
}

std::vector<Index> order(const Graph& g, const AmdOptions& opts = {}, AmdInfo* info = nullptr) {
    std::vector<Index> perm;
    std::vector<Index> iperm;
    amdOrder(g.n, g.col_starts, g.rows, perm, iperm, opts, info);
    REQUIRE(perm.size() == static_cast<std::size_t>(g.n));
    REQUIRE(iperm.size() == static_cast<std::size_t>(g.n));
    std::vector<Index> sorted = perm;
    std::sort(sorted.begin(), sorted.end());
    for (Index k = 0; k < g.n; ++k) {
        REQUIRE(sorted[k] == k);
        REQUIRE(iperm[perm[k]] == k);
    }
    return perm;
}

std::size_t nnzL(const Graph& g, const std::vector<Index>& perm) {
    SparseCholesky chol;
    REQUIRE(chol.analyze(g.n, g.col_starts, g.rows, perm));
    return chol.stats().nnz_l;
}

std::vector<Index> identity(Index n) {
    std::vector<Index> id(static_cast<std::size_t>(n));
    std::iota(id.begin(), id.end(), 0);
    return id;
}

Graph grid(Index k) {
    std::vector<std::pair<Index, Index>> e;
    for (Index r = 0; r < k; ++r) {
        for (Index c = 0; c < k; ++c) {
            const Index v = r * k + c;
            if (c + 1 < k) {
                e.emplace_back(v, v + 1);
            }
            if (r + 1 < k) {
                e.emplace_back(v, v + k);
            }
        }
    }
    return fromEdges(k * k, e);
}

}  // namespace

TEST_CASE("AMD returns a valid permutation on random patterns", "[amd]") {
    Index tight_compactions = 0;
    for (unsigned seed = 1; seed <= 30; ++seed) {
        std::mt19937 rng(seed);
        const Index n = std::uniform_int_distribution<Index>(1, 300)(rng);
        const double density = std::uniform_real_distribution<double>(0.0, 0.08)(rng);
        std::bernoulli_distribution keep(density);
        std::vector<std::pair<Index, Index>> e;
        for (Index i = 0; i < n; ++i) {
            for (Index j = 0; j < i; ++j) {
                if (keep(rng)) {
                    e.emplace_back(i, j);
                }
            }
        }
        const Graph g = fromEdges(n, e);
        const auto perm = order(g);
        // Never worse than no ordering at all on these random patterns by
        // more than a little: a sanity bound, not a quality claim.
        const std::size_t bound = nnzL(g, identity(n)) + static_cast<std::size_t>(n);
        CHECK(nnzL(g, perm) <= bound);
        AmdOptions no_aggr;
        no_aggr.aggressive = false;
        CHECK(nnzL(g, order(g, no_aggr)) <= bound);
        // No elbow room compacts the workspace often; the order must not change.
        AmdOptions tight;
        tight.elbow_room = 0.0;
        AmdInfo info;
        CHECK(order(g, tight, &info) == perm);
        tight_compactions += info.compactions;
    }
    // The order comparison above is only a test of compaction if it ran.
    CHECK(tight_compactions > 0);
}

TEST_CASE("AMD reads either triangle or both alike", "[amd]") {
    std::mt19937 rng(5);
    std::bernoulli_distribution keep(0.05);
    const Index n = 120;
    std::vector<std::pair<Index, Index>> e;
    for (Index i = 0; i < n; ++i) {
        for (Index j = 0; j < i; ++j) {
            if (keep(rng)) {
                e.emplace_back(i, j);
            }
        }
    }
    const Graph lower = fromEdges(n, e);
    CHECK(order(lower) == order(bothTriangles(lower)));
}

TEST_CASE("AMD finds the no-fill orderings of trees", "[amd]") {
    SECTION("tridiagonal") {
        const Index n = 500;
        std::vector<std::pair<Index, Index>> e;
        for (Index i = 0; i + 1 < n; ++i) {
            e.emplace_back(i, i + 1);
        }
        const Graph g = fromEdges(n, e);
        CHECK(nnzL(g, order(g)) == static_cast<std::size_t>(2 * n - 1));
    }
    SECTION("arrow") {
        const Index n = 60;  // the hub stays below the dense threshold
        std::vector<std::pair<Index, Index>> e;
        for (Index i = 1; i < n; ++i) {
            e.emplace_back(0, i);
        }
        const Graph g = fromEdges(n, e);
        const auto perm = order(g);
        CHECK(nnzL(g, identity(n)) == static_cast<std::size_t>(n * (n + 1) / 2));
        // The hub goes among the last two: once one leaf is left, the two
        // are symmetric.
        CHECK(nnzL(g, perm) == static_cast<std::size_t>(2 * n - 1));
        CHECK((perm[n - 1] == 0 || perm[n - 2] == 0));
    }
    SECTION("random tree") {
        std::mt19937 rng(9);
        const Index n = 400;
        std::vector<std::pair<Index, Index>> e;
        for (Index i = 1; i < n; ++i) {
            e.emplace_back(i, std::uniform_int_distribution<Index>(0, i - 1)(rng));
        }
        const Graph g = fromEdges(n, e);
        CHECK(nnzL(g, order(g)) == static_cast<std::size_t>(2 * n - 1));
    }
}

TEST_CASE("AMD cuts the fill of a 2D grid", "[amd]") {
    const Index k = 40;
    const Graph g = grid(k);
    const std::size_t natural = nnzL(g, identity(k * k));
    const std::size_t amd = nnzL(g, order(g));
    // The natural (banded) order fills ~k^3; minimum degree is O(k^2 log k).
    CHECK(amd * 2 < natural);
}

TEST_CASE("AMD orders dense rows last", "[amd]") {
    const Index n = 2000;  // dense threshold max(16, 10·√n) ≈ 447
    std::vector<std::pair<Index, Index>> e;
    for (Index i = 0; i + 1 < n; ++i) {
        e.emplace_back(i, i + 1);
    }
    const Index hub = 777;
    for (Index i = 0; i < n; ++i) {
        if (i != hub) {
            e.emplace_back(hub, i);
        }
    }
    // About 200 distinct neighbours: not dense, but listed three times over
    // by bothTriangles below it would be, if duplicates counted.
    const Index wide = 1500;
    for (Index i = 0; i < 200; ++i) {
        e.emplace_back(wide, i);
    }
    const Graph g = fromEdges(n, e);
    // Chain, hub and wide node, each column of L about its own pattern.
    const auto little_fill = static_cast<std::size_t>(3 * n + 200);
    AmdInfo info;
    const auto perm = order(g, {}, &info);
    CHECK(perm.back() == hub);
    CHECK(info.dense == 1);
    CHECK(nnzL(g, perm) <= little_fill);
    // Dense rows count their neighbours once, however the pattern lists them.
    CHECK(order(bothTriangles(g), {}, &info) == perm);
    CHECK(info.dense == 1);

    AmdOptions none;
    none.dense_alpha = -1.0;
    const auto perm2 = order(g, none, &info);
    CHECK(info.dense == 0);
    CHECK(nnzL(g, perm2) <= little_fill);
}

TEST_CASE("AMD orders nodes adjacent to nothing but a dense row", "[amd]") {
    // A star: every leaf's only neighbour is the dense hub, so each pivot
    // forms an empty element.
    const Index n = 2000;
    const Index hub = 5;
    std::vector<std::pair<Index, Index>> e;
    for (Index i = 0; i < n; ++i) {
        if (i != hub) {
            e.emplace_back(hub, i);
        }
    }
    const Graph g = fromEdges(n, e);
    AmdInfo info;
    const auto perm = order(g, {}, &info);
    CHECK(info.dense == 1);
    CHECK(perm.back() == hub);
    CHECK(nnzL(g, perm) == static_cast<std::size_t>(2 * n - 1));
}

TEST_CASE("AMD numbers indistinguishable nodes together", "[amd]") {
    // Cliques of 6 joined in a chain by connector nodes, each adjacent to all
    // of two neighbouring cliques. The first pivot, in an end clique, takes
    // its mates with it by mass elimination; after that each connector pivot
    // turns a whole inner clique into one supervariable (five merges).
    const Index cliques = 12;
    const Index size = 6;
    const Index n = cliques * size + (cliques - 1);
    std::vector<std::pair<Index, Index>> e;
    for (Index c = 0; c < cliques; ++c) {
        for (Index a = 0; a < size; ++a) {
            for (Index b = 0; b < a; ++b) {
                e.emplace_back(c * size + a, c * size + b);
            }
        }
    }
    for (Index c = 0; c + 1 < cliques; ++c) {
        const Index y = cliques * size + c;
        for (Index a = 0; a < 2 * size; ++a) {
            e.emplace_back(y, c * size + a);
        }
    }
    const Graph g = fromEdges(n, e);
    AmdInfo info;
    const auto perm = order(g, {}, &info);
    // The runs checked below would also follow from mass elimination alone;
    // this count is what shows the merging happened.
    CHECK(info.merges >= (cliques - 2) * (size - 1));
    std::vector<Index> iperm(static_cast<std::size_t>(n));
    for (Index k = 0; k < n; ++k) {
        iperm[perm[k]] = k;
    }
    for (Index c = 0; c < cliques; ++c) {
        std::vector<Index> pos;
        for (Index a = 0; a < size; ++a) {
            pos.push_back(iperm[c * size + a]);
        }
        std::sort(pos.begin(), pos.end());
        // At most two runs: the first pivot, then the supervariable.
        Index runs = 1;
        for (std::size_t q = 1; q < pos.size(); ++q) {
            if (pos[q] != pos[q - 1] + 1) {
                ++runs;
            }
        }
        CHECK(runs <= 2);
    }
}

TEST_CASE("AMD handles empty and isolated nodes", "[amd]") {
    CHECK(order(fromEdges(1, {})).size() == 1);
    const Graph g = fromEdges(10, {{2, 3}, {3, 7}});
    const auto perm = order(g);
    CHECK(nnzL(g, perm) == 12);
    std::vector<Index> none;
    std::vector<Index> inone;
    const std::vector<Index> starts{0};
    amdOrder(0, starts, {}, none, inone);
    CHECK(none.empty());
}
