// Column changes with the basis kept (#265): addCols, removeCols,
// setRowBounds and setColObjective, checked on a column-generation loop over
// a path formulation of a small multicommodity flow, against cold solves of
// the same LPs and against the arc-flow formulation's optimum.

#include "mipx/barrier.h"
#include "mipx/dual_simplex.h"
#include "mipx/lp_problem.h"
#include "mipx/pdlp.h"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace mipx;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

// A directed grid with arcs both ways between neighbours, and commodities
// between far-apart nodes, sized so that capacities bind.
struct Network {
    Index num_nodes = 0;
    std::vector<Index> tail, head;
    std::vector<Real> cost, cap;
    struct Commodity {
        Index source, sink;
        Real demand;
    };
    std::vector<Commodity> commodities;
};

constexpr Real kArtificialCost = 1000.0;

Network gridNetwork(Index w, Index h) {
    Network net;
    net.num_nodes = w * h;
    auto node = [w](Index x, Index y) { return y * w + x; };
    auto arc = [&](Index a, Index b) {
        const auto id = static_cast<Index>(net.tail.size());
        net.tail.push_back(a);
        net.head.push_back(b);
        net.cost.push_back(1.0 + static_cast<Real>((a * 7 + b * 3) % 5));
        net.cap.push_back(3.0 + static_cast<Real>(id % 4));
    };
    for (Index y = 0; y < h; ++y) {
        for (Index x = 0; x < w; ++x) {
            if (x + 1 < w) {
                arc(node(x, y), node(x + 1, y));
                arc(node(x + 1, y), node(x, y));
            }
            if (y + 1 < h) {
                arc(node(x, y), node(x, y + 1));
                arc(node(x, y + 1), node(x, y));
            }
        }
    }
    net.commodities = {
        {node(0, 0), node(w - 1, h - 1), 6.0}, {node(w - 1, 0), node(0, h - 1), 5.0},
        {node(0, h - 1), node(w - 1, 0), 4.0}, {node(w - 1, h - 1), node(0, 0), 6.0},
        {node(0, 1), node(w - 1, 1), 7.0},     {node(1, 0), node(1, h - 1), 5.0},
    };
    return net;
}

Index numCommodities(const Network& net) {
    return static_cast<Index>(net.commodities.size());
}

// Master rows: one demand row per commodity (= demand), then one capacity
// row per arc (<= cap). The first columns are one artificial per commodity,
// so every master is feasible.
LpProblem initialMaster(const Network& net) {
    const Index k = numCommodities(net);
    const auto arcs = static_cast<Index>(net.tail.size());
    LpProblem lp;
    lp.num_rows = k + arcs;
    for (const auto& c : net.commodities) {
        lp.row_lower.push_back(c.demand);
        lp.row_upper.push_back(c.demand);
    }
    for (Index a = 0; a < arcs; ++a) {
        lp.row_lower.push_back(-kInf);
        lp.row_upper.push_back(net.cap[a]);
    }
    std::vector<Triplet> trips;
    for (Index c = 0; c < k; ++c) {
        lp.obj.push_back(kArtificialCost);
        lp.col_lower.push_back(0.0);
        lp.col_upper.push_back(kInf);
        trips.push_back({c, c, 1.0});
    }
    lp.num_cols = k;
    lp.matrix = SparseMatrix(lp.num_rows, lp.num_cols, std::move(trips));
    return lp;
}

// A batch of columns in addCols' form.
struct Batch {
    std::vector<Index> starts, indices;
    std::vector<Real> values, obj, lower, upper;

    void add(std::vector<Index> rows, std::vector<Real> vals, Real cost) {
        starts.push_back(static_cast<Index>(indices.size()));
        indices.insert(indices.end(), rows.begin(), rows.end());
        values.insert(values.end(), vals.begin(), vals.end());
        obj.push_back(cost);
        lower.push_back(0.0);
        upper.push_back(kInf);
    }
    [[nodiscard]] Index size() const { return static_cast<Index>(obj.size()); }
};

// One shortest path per commodity under the arc weights cost_a - y_a, added
// when its reduced cost c_p - y_k - sum_a y_a is negative. Bellman-Ford,
// since the duals of the capacity rows are only nonpositive up to round-off.
Batch price(const Network& net, const std::vector<Real>& y) {
    const Index k = numCommodities(net);
    const auto arcs = static_cast<Index>(net.tail.size());
    Batch batch;
    for (Index c = 0; c < k; ++c) {
        std::vector<Real> dist(static_cast<std::size_t>(net.num_nodes), kInf);
        std::vector<Index> pred(static_cast<std::size_t>(net.num_nodes), -1);
        dist[net.commodities[c].source] = 0.0;
        for (Index round = 0; round < net.num_nodes; ++round) {
            bool changed = false;
            for (Index a = 0; a < arcs; ++a) {
                const Real w = net.cost[a] - y[k + a];
                if (dist[net.tail[a]] + w < dist[net.head[a]] - 1e-12) {
                    dist[net.head[a]] = dist[net.tail[a]] + w;
                    pred[net.head[a]] = a;
                    changed = true;
                }
            }
            if (!changed) {
                break;
            }
        }
        const Index sink = net.commodities[c].sink;
        if (dist[sink] - y[c] >= -1e-7) {
            continue;
        }
        std::vector<Index> rows = {c};
        Real cost = 0.0;
        for (Index v = sink; v != net.commodities[c].source; v = net.tail[pred[v]]) {
            rows.push_back(k + pred[v]);
            cost += net.cost[pred[v]];
        }
        batch.add(rows, std::vector<Real>(rows.size(), 1.0), cost);
    }
    return batch;
}

// The arc-flow formulation, with an uncapacitated artificial arc per
// commodity at kArtificialCost: the LP the path master converges to.
Real arcFlowOptimum(const Network& net) {
    const Index k = numCommodities(net);
    const auto arcs = static_cast<Index>(net.tail.size());
    const Index n = net.num_nodes;
    LpProblem lp;
    std::vector<Triplet> trips;
    lp.num_rows = k * n + arcs;
    for (Index c = 0; c < k; ++c) {
        for (Index v = 0; v < n; ++v) {
            Real rhs = 0.0;
            if (v == net.commodities[c].source) {
                rhs = net.commodities[c].demand;
            } else if (v == net.commodities[c].sink) {
                rhs = -net.commodities[c].demand;
            }
            lp.row_lower.push_back(rhs);
            lp.row_upper.push_back(rhs);
        }
    }
    for (Index a = 0; a < arcs; ++a) {
        lp.row_lower.push_back(-kInf);
        lp.row_upper.push_back(net.cap[a]);
    }
    auto col = [&](Real cost) {
        lp.obj.push_back(cost);
        lp.col_lower.push_back(0.0);
        lp.col_upper.push_back(kInf);
        return lp.num_cols++;
    };
    for (Index c = 0; c < k; ++c) {
        for (Index a = 0; a < arcs; ++a) {
            const Index j = col(net.cost[a]);
            trips.push_back({c * n + net.tail[a], j, 1.0});
            trips.push_back({c * n + net.head[a], j, -1.0});
            trips.push_back({k * n + a, j, 1.0});
        }
        const Index j = col(kArtificialCost);
        trips.push_back({c * n + net.commodities[c].source, j, 1.0});
        trips.push_back({c * n + net.commodities[c].sink, j, -1.0});
    }
    lp.matrix = SparseMatrix(lp.num_rows, lp.num_cols, std::move(trips));
    DualSimplexSolver solver;
    solver.setVerbose(false);
    solver.load(lp);
    const auto r = solver.solve();
    REQUIRE(r.status == Status::Optimal);
    return r.objective;
}

LpResult solveCold(const LpProblem& lp) {
    DualSimplexSolver solver;
    solver.setVerbose(false);
    solver.load(lp);
    return solver.solve();
}

struct CgRun {
    LpProblem master;  // the last master, kept alongside the solver
    Int warm_iterations = 0;
    Int cold_iterations = 0;
    Index masters = 0;
};

// Column generation to optimality on @p solver, checking each master's warm
// objective against a cold solve of the same LP.
CgRun runColumnGeneration(const Network& net, DualSimplexSolver& solver) {
    CgRun run;
    run.master = initialMaster(net);
    solver.setVerbose(false);
    solver.load(run.master);
    for (;;) {
        const auto warm = solver.solve();
        REQUIRE(warm.status == Status::Optimal);
        const auto cold = solveCold(run.master);
        REQUIRE(cold.status == Status::Optimal);
        CHECK_THAT(warm.objective, WithinRel(cold.objective, 1e-9));
        run.warm_iterations += warm.iterations;
        run.cold_iterations += cold.iterations;
        ++run.masters;

        const auto y = solver.getDualValues();
        // The pricing's reduced cost is the solver's on the columns it has.
        const auto rc = solver.getReducedCosts();
        for (Index j = 0; j < run.master.num_cols; ++j) {
            Real expect = run.master.obj[j];
            auto col = run.master.matrix.col(j);
            for (Index p = 0; p < col.size(); ++p) {
                expect -= col.values[p] * y[col.indices[p]];
            }
            CHECK_THAT(rc[j], WithinAbs(expect, 1e-6));
        }

        const Batch batch = price(net, y);
        if (batch.size() == 0) {
            break;
        }
        solver.addCols(batch.starts, batch.indices, batch.values, batch.obj, batch.lower,
                       batch.upper);
        appendColumns(run.master, batch.starts, batch.indices, batch.values, batch.obj, batch.lower,
                      batch.upper);
        REQUIRE(solver.numCols() == run.master.num_cols);
        REQUIRE(run.masters < 200);
    }
    return run;
}

}  // namespace

TEST_CASE("Column changes: column generation on a kept basis reaches the arc-flow optimum",
          "[column_changes]") {
    const Network net = gridNetwork(4, 4);
    DualSimplexSolver solver;
    const CgRun run = runColumnGeneration(net, solver);

    CHECK(run.masters > 2);
    CHECK_THAT(solver.getObjective(), WithinRel(arcFlowOptimum(net), 1e-9));
    // The point of keeping the basis.
    CHECK(run.warm_iterations < run.cold_iterations);
}

TEST_CASE("Column changes: primal Devex re-solves reach the same optima", "[column_changes]") {
    DualSimplexOptions options;
    options.enable_primal_devex = true;
    for (const auto& [w, h] : {std::pair<Index, Index>{4, 4}, {6, 5}}) {
        const Network net = gridNetwork(w, h);
        DualSimplexSolver solver;
        solver.setOptions(options);
        const CgRun run = runColumnGeneration(net, solver);
        CHECK_THAT(solver.getObjective(), WithinRel(arcFlowOptimum(net), 1e-9));
        CHECK(run.warm_iterations < run.cold_iterations);
    }
}

TEST_CASE("Column changes: removing nonbasic and basic columns re-solves warm as cold",
          "[column_changes]") {
    const Network net = gridNetwork(4, 4);
    DualSimplexSolver solver;
    CgRun run = runColumnGeneration(net, solver);
    const auto basis = solver.getBasis();
    const Index k = numCommodities(net);

    SECTION("nonbasic path columns") {
        std::vector<Index> drop;
        for (Index j = k; j < run.master.num_cols; ++j) {
            if (basis[j] != BasisStatus::Basic) {
                drop.push_back(j);
            }
        }
        REQUIRE_FALSE(drop.empty());
        const Real before = solver.getObjective();
        solver.removeCols(drop);
        removeColumns(run.master, sortedColumnSet(run.master.num_cols, drop));
        REQUIRE(solver.numCols() == run.master.num_cols);
        const auto warm = solver.solve();
        REQUIRE(warm.status == Status::Optimal);
        // The optimal basis survives: nothing left to do.
        CHECK(warm.iterations == 0);
        CHECK_THAT(warm.objective, WithinRel(before, 1e-9));
    }

    SECTION("basic path columns") {
        std::vector<Index> drop;
        for (Index j = k; j < run.master.num_cols && drop.size() < 3; ++j) {
            if (basis[j] == BasisStatus::Basic) {
                drop.push_back(j);
            }
        }
        REQUIRE(drop.size() == 3);
        solver.removeCols(drop);
        removeColumns(run.master, sortedColumnSet(run.master.num_cols, drop));
        const auto kept = solver.getBasis();
        CHECK(std::ranges::count(kept, BasisStatus::Basic) == run.master.num_rows);
        const auto warm = solver.solve();
        const auto cold = solveCold(run.master);
        REQUIRE(warm.status == Status::Optimal);
        REQUIRE(cold.status == Status::Optimal);
        CHECK_THAT(warm.objective, WithinRel(cold.objective, 1e-9));
    }

    SECTION("then adding columns again prices back to the optimum") {
        std::vector<Index> drop;
        for (Index j = k; j < run.master.num_cols; ++j) {
            drop.push_back(j);
        }
        solver.removeCols(drop);
        removeColumns(run.master, sortedColumnSet(run.master.num_cols, drop));
        for (Index round = 0; round < 200; ++round) {
            const auto r = solver.solve();
            REQUIRE(r.status == Status::Optimal);
            const Batch batch = price(net, solver.getDualValues());
            if (batch.size() == 0) {
                break;
            }
            solver.addCols(batch.starts, batch.indices, batch.values, batch.obj, batch.lower,
                           batch.upper);
        }
        CHECK_THAT(solver.getObjective(), WithinRel(arcFlowOptimum(net), 1e-9));
    }
}

TEST_CASE("Column changes: row bounds and single costs re-solve warm as cold", "[column_changes]") {
    const Network net = gridNetwork(4, 3);
    DualSimplexSolver solver;
    CgRun run = runColumnGeneration(net, solver);
    const Index k = numCommodities(net);

    // Tighten two capacities and raise the cost of the first path column.
    const Index row_a = k + 0;
    const Index row_b = k + 5;
    solver.setRowBounds(row_a, -kInf, 1.0);
    run.master.row_upper[row_a] = 1.0;
    solver.setRowBounds(row_b, -kInf, 0.5);
    run.master.row_upper[row_b] = 0.5;
    solver.setColObjective(k, run.master.obj[k] + 3.0);
    run.master.obj[k] += 3.0;

    const auto warm = solver.solve();
    const auto cold = solveCold(run.master);
    REQUIRE(warm.status == Status::Optimal);
    REQUIRE(cold.status == Status::Optimal);
    CHECK_THAT(warm.objective, WithinRel(cold.objective, 1e-9));

    // A nonbasic logical moves with its row's bounds.
    solver.setRowBounds(row_a, -kInf, net.cap[0]);
    run.master.row_upper[row_a] = net.cap[0];
    const auto back = solver.solve();
    REQUIRE(back.status == Status::Optimal);
    CHECK_THAT(back.objective, WithinRel(solveCold(run.master).objective, 1e-9));
}

TEST_CASE("Column changes: addCols on a maximize LP and an unsolved LP", "[column_changes]") {
    // max x + y  s.t.  x + y <= 4,  x <= 3; then a column z with cost 2 in
    // the first row: max moves to z = 4, objective 8.
    LpProblem lp;
    lp.sense = Sense::Maximize;
    lp.num_cols = 2;
    lp.obj = {1.0, 1.0};
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {kInf, kInf};
    lp.num_rows = 2;
    lp.row_lower = {-kInf, -kInf};
    lp.row_upper = {4.0, 3.0};
    lp.matrix = SparseMatrix(2, 2, {{0, 0, 1.0}, {0, 1, 1.0}, {1, 0, 1.0}});

    const std::vector<Index> starts = {0, 1};
    const std::vector<Index> indices = {0};
    const std::vector<Real> values = {1.0};
    const std::vector<Real> obj = {2.0};
    const std::vector<Real> lower = {0.0};
    const std::vector<Real> upper = {kInf};

    SECTION("after a solve") {
        DualSimplexSolver solver;
        solver.setVerbose(false);
        solver.load(lp);
        REQUIRE_THAT(solver.solve().objective, WithinAbs(4.0, 1e-9));
        solver.addCols(starts, indices, values, obj, lower, upper);
        const auto r = solver.solve();
        REQUIRE(r.status == Status::Optimal);
        CHECK_THAT(r.objective, WithinAbs(8.0, 1e-9));
        CHECK_THAT(solver.getPrimalValues()[2], WithinAbs(4.0, 1e-9));
    }

    SECTION("before any solve") {
        DualSimplexSolver solver;
        solver.setVerbose(false);
        solver.load(lp);
        solver.addCols(starts, indices, values, obj, lower, upper);
        const auto r = solver.solve();
        REQUIRE(r.status == Status::Optimal);
        CHECK_THAT(r.objective, WithinAbs(8.0, 1e-9));
    }

    SECTION("malformed input changes nothing") {
        DualSimplexSolver solver;
        solver.setVerbose(false);
        solver.load(lp);
        REQUIRE(solver.solve().status == Status::Optimal);
        const std::vector<Index> bad_row = {7};
        CHECK_THROWS_AS(solver.addCols(starts, bad_row, values, obj, lower, upper),
                        std::invalid_argument);
        const std::vector<Index> bad_starts = {0, 2};
        CHECK_THROWS_AS(solver.addCols(bad_starts, indices, values, obj, lower, upper),
                        std::invalid_argument);
        const std::vector<Real> inf_lower = {kInf};
        CHECK_THROWS_AS(solver.addCols(starts, indices, values, obj, inf_lower, upper),
                        std::invalid_argument);
        const std::vector<Real> nan_value = {std::nan("")};
        CHECK_THROWS_AS(solver.addCols(starts, indices, nan_value, obj, lower, upper),
                        std::invalid_argument);
        const std::vector<Real> inf_value = {kInf};
        CHECK_THROWS_AS(solver.addCols(starts, indices, inf_value, obj, lower, upper),
                        std::invalid_argument);
        const std::vector<Real> inf_cost = {kInf};
        CHECK_THROWS_AS(solver.addCols(starts, indices, values, inf_cost, lower, upper),
                        std::invalid_argument);
        const std::vector<Real> minus_inf_upper = {-kInf};
        CHECK_THROWS_AS(solver.addCols(starts, indices, values, obj, lower, minus_inf_upper),
                        std::invalid_argument);
        const std::vector<Real> nan_bound = {std::nan("")};
        CHECK_THROWS_AS(solver.addCols(starts, indices, values, obj, nan_bound, upper),
                        std::invalid_argument);
        const std::vector<Index> bad_col = {2};
        CHECK_THROWS_AS(solver.removeCols(bad_col), std::invalid_argument);
        CHECK(solver.numCols() == 2);
        const auto r = solver.solve();
        CHECK(r.iterations == 0);
        CHECK_THAT(r.objective, WithinAbs(4.0, 1e-9));
    }
}

TEST_CASE("Column changes: barrier and PDLP solve the changed LP", "[column_changes]") {
    const Network net = gridNetwork(3, 3);
    DualSimplexSolver simplex;
    CgRun run = runColumnGeneration(net, simplex);
    const Index k = numCommodities(net);

    // Rebuild the final master on each solver from the first one by the
    // column calls, then drop one path column and change a row and a cost.
    LpProblem first = initialMaster(net);
    Batch paths;
    for (Index j = k; j < run.master.num_cols; ++j) {
        auto col = run.master.matrix.col(j);
        paths.add({col.indices.begin(), col.indices.end()}, {col.values.begin(), col.values.end()},
                  run.master.obj[j]);
    }
    const std::vector<Index> drop = {k + 1};
    removeColumns(run.master, sortedColumnSet(run.master.num_cols, drop));
    run.master.row_upper[k + 2] = 1.0;
    run.master.obj[k] += 2.0;
    const Real expect = solveCold(run.master).objective;

    auto apply = [&](LpSolver& s) {
        s.load(first);
        s.addCols(paths.starts, paths.indices, paths.values, paths.obj, paths.lower, paths.upper);
        s.removeCols(drop);
        s.setRowBounds(k + 2, -kInf, 1.0);
        s.setColObjective(k, run.master.obj[k]);
        return s.solve();
    };

    SECTION("barrier") {
        BarrierSolver barrier;
        const auto r = apply(barrier);
        REQUIRE(r.status == Status::Optimal);
        CHECK_THAT(r.objective, WithinRel(expect, 1e-6));

        // A change that leaves no transform leaves no solution to read.
        barrier.setColBounds(k, 5.0, 3.0);
        CHECK(barrier.solve().status != Status::Optimal);
        const auto n = static_cast<std::size_t>(run.master.num_cols);
        CHECK(barrier.getPrimalValues() == std::vector<Real>(n, 0.0));
        CHECK(barrier.getReducedCosts() == std::vector<Real>(n, 0.0));
    }
    SECTION("PDLP") {
        PdlpSolver pdlp;
        const auto r = apply(pdlp);
        REQUIRE(r.status == Status::Optimal);
        CHECK_THAT(r.objective, WithinRel(expect, 1e-4));
    }
}
