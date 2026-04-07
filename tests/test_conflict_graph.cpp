#include <catch2/catch_test_macros.hpp>

#include "mipx/conflict_graph.h"

using namespace mipx;

namespace {

LpProblem makeSetPackingProblem() {
    // 3 binary variables: x0 + x1 <= 1, x1 + x2 <= 1
    LpProblem prob;
    prob.num_cols = 3;
    prob.num_rows = 2;
    prob.matrix = SparseMatrix(2, 3, {
        {0, 0, 1.0}, {0, 1, 1.0},
        {1, 1, 1.0}, {1, 2, 1.0},
    });
    prob.row_lower = {-kInf, -kInf};
    prob.row_upper = {1.0, 1.0};
    prob.col_lower = {0.0, 0.0, 0.0};
    prob.col_upper = {1.0, 1.0, 1.0};
    prob.col_type = {VarType::Binary, VarType::Binary, VarType::Binary};
    prob.obj = {0.0, 0.0, 0.0};
    return prob;
}

LpProblem makeKnapsackConflict() {
    // 2 binary variables: 3*x0 + 4*x1 <= 5
    // Both coefficients sum to 7 > 5, so x0 and x1 conflict.
    LpProblem prob;
    prob.num_cols = 2;
    prob.num_rows = 1;
    prob.matrix = SparseMatrix(1, 2, {
        {0, 0, 3.0}, {0, 1, 4.0},
    });
    prob.row_lower = {-kInf};
    prob.row_upper = {5.0};
    prob.col_lower = {0.0, 0.0};
    prob.col_upper = {1.0, 1.0};
    prob.col_type = {VarType::Binary, VarType::Binary};
    prob.obj = {0.0, 0.0};
    return prob;
}

LpProblem makeNoBinaryProblem() {
    LpProblem prob;
    prob.num_cols = 2;
    prob.num_rows = 1;
    prob.matrix = SparseMatrix(1, 2, {
        {0, 0, 1.0}, {0, 1, 1.0},
    });
    prob.row_lower = {-kInf};
    prob.row_upper = {10.0};
    prob.col_lower = {0.0, 0.0};
    prob.col_upper = {10.0, 10.0};
    prob.col_type = {VarType::Continuous, VarType::Continuous};
    prob.obj = {0.0, 0.0};
    return prob;
}

}  // namespace

TEST_CASE("ConflictGraph: set-packing conflicts", "[conflict_graph]") {
    auto prob = makeSetPackingProblem();
    ConflictGraph graph;
    graph.build(prob);

    CHECK(graph.numBinaries() == 3);
    CHECK(graph.numEdges() == 2);

    // x0 and x1 conflict.
    CHECK(graph.conflicts({0, false}, {1, false}));
    // x1 and x2 conflict.
    CHECK(graph.conflicts({1, false}, {2, false}));
    // x0 and x2 do not conflict.
    CHECK_FALSE(graph.conflicts({0, false}, {2, false}));
}

TEST_CASE("ConflictGraph: knapsack conflicts", "[conflict_graph]") {
    auto prob = makeKnapsackConflict();
    ConflictGraph graph;
    graph.build(prob);

    CHECK(graph.numBinaries() == 2);
    CHECK(graph.numEdges() == 1);
    CHECK(graph.conflicts({0, false}, {1, false}));
}

TEST_CASE("ConflictGraph: no binary variables", "[conflict_graph]") {
    auto prob = makeNoBinaryProblem();
    ConflictGraph graph;
    graph.build(prob);

    CHECK(graph.numBinaries() == 0);
    CHECK(graph.numEdges() == 0);
}

TEST_CASE("ConflictGraph: index mappings", "[conflict_graph]") {
    auto prob = makeSetPackingProblem();
    ConflictGraph graph;
    graph.build(prob);

    for (Index j = 0; j < prob.num_cols; ++j) {
        Index bin = graph.toBinaryIndex(j);
        CHECK(bin >= 0);
        CHECK(graph.toOriginalIndex(bin) == j);
    }

    CHECK(graph.toBinaryIndex(-1) == -1);
    CHECK(graph.toBinaryIndex(100) == -1);
    CHECK(graph.toOriginalIndex(-1) == -1);
}

TEST_CASE("ConflictGraph: neighbors", "[conflict_graph]") {
    auto prob = makeSetPackingProblem();
    ConflictGraph graph;
    graph.build(prob);

    auto nbrs = graph.neighbors({1, false});
    CHECK(nbrs.size() == 2);  // x0 and x2 both conflict with x1.

    auto nbrs0 = graph.neighbors({0, false});
    CHECK(nbrs0.size() == 1);
}

TEST_CASE("ConflictGraph: connected components", "[conflict_graph]") {
    auto prob = makeSetPackingProblem();
    ConflictGraph graph;
    graph.build(prob);

    Index num_comp = 0;
    auto comps = graph.connectedComponents(num_comp);
    // All three are connected through x1.
    CHECK(num_comp == 1);
    CHECK(comps[0] == comps[1]);
    CHECK(comps[1] == comps[2]);
}

TEST_CASE("ConflictGraph: fix/unfix variables", "[conflict_graph]") {
    auto prob = makeSetPackingProblem();
    ConflictGraph graph;
    graph.build(prob);

    CHECK_FALSE(graph.isFixed(0));
    graph.fixVariable(0, 1.0);
    CHECK(graph.isFixed(0));
    CHECK(graph.fixedValue(0) == 1.0);

    graph.unfixVariable(0);
    CHECK_FALSE(graph.isFixed(0));
}

TEST_CASE("ConflictGraph: add conflict manually", "[conflict_graph]") {
    auto prob = makeSetPackingProblem();
    ConflictGraph graph;
    graph.build(prob);

    Index edges_before = graph.numEdges();
    // Add conflict x0-x2 which didn't exist.
    graph.addConflict({0, false}, {2, false});
    CHECK(graph.numEdges() == edges_before + 1);
    CHECK(graph.conflicts({0, false}, {2, false}));

    // Adding the same conflict again should be a no-op.
    graph.addConflict({0, false}, {2, false});
    CHECK(graph.numEdges() == edges_before + 1);
}

TEST_CASE("ConflictGraph: disconnected components", "[conflict_graph]") {
    // 4 binaries: x0+x1<=1, x2+x3<=1 (two separate cliques).
    LpProblem prob;
    prob.num_cols = 4;
    prob.num_rows = 2;
    prob.matrix = SparseMatrix(2, 4, {
        {0, 0, 1.0}, {0, 1, 1.0},
        {1, 2, 1.0}, {1, 3, 1.0},
    });
    prob.row_lower = {-kInf, -kInf};
    prob.row_upper = {1.0, 1.0};
    prob.col_lower = {0.0, 0.0, 0.0, 0.0};
    prob.col_upper = {1.0, 1.0, 1.0, 1.0};
    prob.col_type = {VarType::Binary, VarType::Binary,
                     VarType::Binary, VarType::Binary};
    prob.obj = {0.0, 0.0, 0.0, 0.0};

    ConflictGraph graph;
    graph.build(prob);

    Index num_comp = 0;
    auto comps = graph.connectedComponents(num_comp);
    CHECK(num_comp == 2);
    CHECK(comps[0] == comps[1]);
    CHECK(comps[2] == comps[3]);
    CHECK(comps[0] != comps[2]);
}
