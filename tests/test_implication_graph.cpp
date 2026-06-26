#include "mipx/domain.h"
#include "mipx/implication_graph.h"
#include "mipx/lp_problem.h"
#include "mipx/probing.h"
#include "mipx/variable_bounds.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using namespace mipx;
using Catch::Matchers::WithinAbs;

namespace {

LpProblem makeProblem(Index num_cols, Index num_rows, std::vector<Triplet> triplets,
                      std::vector<Real> row_lower, std::vector<Real> row_upper,
                      std::vector<Real> col_lower, std::vector<Real> col_upper,
                      std::vector<VarType> col_type = {}) {
    LpProblem prob;
    prob.num_cols = num_cols;
    prob.num_rows = num_rows;
    prob.matrix = SparseMatrix(num_rows, num_cols, std::move(triplets));
    prob.row_lower = std::move(row_lower);
    prob.row_upper = std::move(row_upper);
    prob.col_lower = std::move(col_lower);
    prob.col_upper = std::move(col_upper);
    prob.obj.assign(num_cols, 0.0);
    if (col_type.empty()) {
        prob.col_type.assign(num_cols, VarType::Continuous);
    } else {
        prob.col_type = std::move(col_type);
    }
    return prob;
}

}  // namespace

// =============================================================================
// ImplicationGraph tests
// =============================================================================

TEST_CASE("ImplicationGraph: init and basic properties", "[implication_graph]") {
    ImplicationGraph graph;
    std::vector<Index> binaries = {0, 2, 5};
    graph.init(6, binaries);

    CHECK(graph.numBinaryVars() == 3);
    CHECK(graph.numImplications() == 0);
    CHECK(graph.isBinary(0));
    CHECK_FALSE(graph.isBinary(1));
    CHECK(graph.isBinary(2));
    CHECK_FALSE(graph.isBinary(3));
    CHECK_FALSE(graph.isBinary(4));
    CHECK(graph.isBinary(5));
}

TEST_CASE("ImplicationGraph: add simple implication", "[implication_graph]") {
    ImplicationGraph graph;
    std::vector<Index> binaries = {0, 1, 2};
    graph.init(3, binaries);

    // x0=1 => x1=0
    CHECK(graph.addImplication(0, true, 1, false));
    // Contrapositive is automatically added: x1=1 => x0=0
    CHECK(graph.numImplications() == 2);

    const auto& imp01 = graph.implications(0, true);
    REQUIRE(imp01.size() == 1);
    CHECK(imp01[0].to_var == 1);
    CHECK(imp01[0].to_val == false);

    const auto& imp10 = graph.implications(1, true);
    REQUIRE(imp10.size() == 1);
    CHECK(imp10[0].to_var == 0);
    CHECK(imp10[0].to_val == false);
}

TEST_CASE("ImplicationGraph: duplicate implication not added", "[implication_graph]") {
    ImplicationGraph graph;
    std::vector<Index> binaries = {0, 1};
    graph.init(2, binaries);

    CHECK(graph.addImplication(0, true, 1, false));
    Int count1 = graph.numImplications();
    CHECK(graph.addImplication(0, true, 1, false));  // duplicate
    CHECK(graph.numImplications() == count1);
}

TEST_CASE("ImplicationGraph: contradiction detection", "[implication_graph]") {
    ImplicationGraph graph;
    std::vector<Index> binaries = {0, 1};
    graph.init(2, binaries);

    // x0=1 => x1=0
    CHECK(graph.addImplication(0, true, 1, false));
    // x0=1 => x1=1 contradicts the above
    CHECK_FALSE(graph.addImplication(0, true, 1, true));
}

TEST_CASE("ImplicationGraph: propagation", "[implication_graph]") {
    ImplicationGraph graph;
    std::vector<Index> binaries = {0, 1, 2, 3};
    graph.init(4, binaries);

    // Chain: x0=1 => x1=0, x1=0 => x2=1, x2=1 => x3=0
    graph.addImplication(0, true, 1, false);
    graph.addImplication(1, false, 2, true);
    graph.addImplication(2, true, 3, false);

    std::vector<std::pair<Index, bool>> propagated;
    CHECK(graph.propagate(0, true, propagated));
    // Should propagate x1=0, x2=1, x3=0
    CHECK(propagated.size() == 3);

    // Check all expected fixings are present.
    bool found1 = false, found2 = false, found3 = false;
    for (const auto& [var, val] : propagated) {
        if (var == 1 && !val) {
            found1 = true;
        }
        if (var == 2 && val) {
            found2 = true;
        }
        if (var == 3 && !val) {
            found3 = true;
        }
    }
    CHECK(found1);
    CHECK(found2);
    CHECK(found3);
}

TEST_CASE("ImplicationGraph: propagation detects contradiction", "[implication_graph]") {
    ImplicationGraph graph;
    std::vector<Index> binaries = {0, 1, 2};
    graph.init(3, binaries);

    // x0=1 => x1=0 and x0=1 => x2=1, but also x2=1 => x1=1
    graph.addImplication(0, true, 1, false);
    graph.addImplication(0, true, 2, true);
    graph.addImplication(2, true, 1, true);

    std::vector<std::pair<Index, bool>> propagated;
    // x0=1 => x1=0 AND x0=1 => x2=1 => x1=1 -- contradiction
    CHECK_FALSE(graph.propagate(0, true, propagated));
}

TEST_CASE("ImplicationGraph: transitive closure", "[implication_graph]") {
    ImplicationGraph graph;
    std::vector<Index> binaries = {0, 1, 2};
    graph.init(3, binaries);

    // x0=1 => x1=0, x1=0 => x2=1
    graph.addImplication(0, true, 1, false);
    graph.addImplication(1, false, 2, true);

    // Before closure, x0=1 does not directly imply x2=1.
    const auto& before = graph.implications(0, true);
    bool has_direct = false;
    for (const auto& imp : before) {
        if (imp.to_var == 2 && imp.to_val == true) {
            has_direct = true;
        }
    }
    CHECK_FALSE(has_direct);

    Int new_imps = graph.computeTransitiveClosure();
    CHECK(new_imps > 0);

    // After closure, x0=1 should directly imply x2=1.
    const auto& after = graph.implications(0, true);
    has_direct = false;
    for (const auto& imp : after) {
        if (imp.to_var == 2 && imp.to_val == true) {
            has_direct = true;
        }
    }
    CHECK(has_direct);
}

TEST_CASE("ImplicationGraph: same-sense equivalence detection", "[implication_graph]") {
    ImplicationGraph graph;
    std::vector<Index> binaries = {0, 1};
    graph.init(2, binaries);

    // x0=0 => x1=0 AND x1=0 => x0=0 (+ contrapositives)
    // This means x0 == x1.
    graph.addImplication(0, false, 1, false);
    graph.addImplication(1, false, 0, false);
    graph.addImplication(0, true, 1, true);
    graph.addImplication(1, true, 0, true);

    auto equivs = graph.detectEquivalences();
    REQUIRE(equivs.size() == 1);
    CHECK(equivs[0].var_a == 0);
    CHECK(equivs[0].var_b == 1);
    CHECK(equivs[0].same_sense == true);
}

TEST_CASE("ImplicationGraph: opposite-sense equivalence detection", "[implication_graph]") {
    ImplicationGraph graph;
    std::vector<Index> binaries = {0, 1};
    graph.init(2, binaries);

    // x0=0 => x1=1 AND x1=1 => x0=0 (+ contrapositives)
    // This means x0 == 1 - x1.
    graph.addImplication(0, false, 1, true);
    graph.addImplication(1, true, 0, false);
    graph.addImplication(0, true, 1, false);
    graph.addImplication(1, false, 0, true);

    auto equivs = graph.detectEquivalences();
    REQUIRE(equivs.size() == 1);
    CHECK(equivs[0].var_a == 0);
    CHECK(equivs[0].var_b == 1);
    CHECK(equivs[0].same_sense == false);
}

TEST_CASE("ImplicationGraph: transitive-chain equivalence detection", "[implication_graph]") {
    // Chain of same-sense pairs: x0 == x1 and x1 == x2 (with contrapositives).
    // The SCC-based detector must also derive the transitive pair x0 == x2,
    // which has no direct implication between x0 and x2.
    ImplicationGraph graph;
    std::vector<Index> binaries = {0, 1, 2};
    graph.init(3, binaries);

    // x0 == x1
    graph.addImplication(0, true, 1, true);
    graph.addImplication(1, true, 0, true);
    graph.addImplication(0, false, 1, false);
    graph.addImplication(1, false, 0, false);
    // x1 == x2
    graph.addImplication(1, true, 2, true);
    graph.addImplication(2, true, 1, true);
    graph.addImplication(1, false, 2, false);
    graph.addImplication(2, false, 1, false);

    auto equivs = graph.detectEquivalences();
    // All three pairs are same-sense equivalent: (0,1), (0,2), (1,2).
    REQUIRE(equivs.size() == 3);
    for (const auto& eq : equivs) {
        CHECK(eq.same_sense == true);
    }

    // The transitive pair (0,2) — not directly linked — must be present.
    bool found_0_2 = false;
    for (const auto& eq : equivs) {
        if (eq.var_a == 0 && eq.var_b == 2) {
            found_0_2 = true;
        }
    }
    CHECK(found_0_2);
}

TEST_CASE("ImplicationGraph: fixing detection", "[implication_graph]") {
    ImplicationGraph graph;
    std::vector<Index> binaries = {0, 1};
    graph.init(2, binaries);

    // x0=0 => x1=0 and x0=0 => x1=1 -- contradiction at x0=0
    // So x0 must be 1.
    graph.addImplication(0, false, 1, false);
    graph.addImplication(0, false, 1, true);

    auto fixings = graph.detectFixings();
    REQUIRE(fixings.size() == 1);
    CHECK(fixings[0].first == 0);
    CHECK(fixings[0].second == true);
}

TEST_CASE("ImplicationGraph: implication score", "[implication_graph]") {
    ImplicationGraph graph;
    std::vector<Index> binaries = {0, 1, 2, 3};
    graph.init(4, binaries);

    // x0 has implications to x1, x2, x3.
    graph.addImplication(0, true, 1, false);
    graph.addImplication(0, true, 2, true);
    graph.addImplication(0, false, 3, false);

    // x0 should have the highest score.
    CHECK(graph.implicationScore(0) > graph.implicationScore(1));
    CHECK(graph.implicationScore(0) > 0);
    CHECK(graph.implicationScore(3) > 0);  // contrapositives
}

TEST_CASE("ImplicationGraph: non-binary variable ignored", "[implication_graph]") {
    ImplicationGraph graph;
    std::vector<Index> binaries = {0, 2};
    graph.init(4, binaries);

    // Variable 1 is not binary; implications involving it return true but do nothing.
    CHECK(graph.addImplication(0, true, 1, false));
    CHECK(graph.numImplications() == 0);
}

TEST_CASE("ImplicationGraph: clear resets state", "[implication_graph]") {
    ImplicationGraph graph;
    std::vector<Index> binaries = {0, 1};
    graph.init(2, binaries);

    graph.addImplication(0, true, 1, false);
    CHECK(graph.numImplications() > 0);

    graph.clear();
    CHECK(graph.numImplications() == 0);
    CHECK(graph.implications(0, true).empty());
}

// =============================================================================
// VariableBoundStore tests
// =============================================================================

TEST_CASE("VariableBoundStore: init and basic properties", "[variable_bounds]") {
    VariableBoundStore store;
    store.init(5);

    CHECK(store.numVUBs() == 0);
    CHECK(store.numVLBs() == 0);
    CHECK_FALSE(store.hasVUB(0));
    CHECK_FALSE(store.hasVLB(0));
}

TEST_CASE("VariableBoundStore: add VUB", "[variable_bounds]") {
    VariableBoundStore store;
    store.init(5);

    // x0 <= 3*y1 + 2  (when y1=0: x0<=2, when y1=1: x0<=5)
    store.addVUB(0, 1, 3.0, 2.0, 0);
    CHECK(store.hasVUB(0));
    CHECK(store.numVUBs() == 1);

    const auto& vubs = store.vubs(0);
    REQUIRE(vubs.size() == 1);
    CHECK(vubs[0].binary_var == 1);
    CHECK_THAT(vubs[0].coeff, WithinAbs(3.0, 1e-10));
    CHECK_THAT(vubs[0].constant, WithinAbs(2.0, 1e-10));
}

TEST_CASE("VariableBoundStore: add VLB", "[variable_bounds]") {
    VariableBoundStore store;
    store.init(5);

    // x0 >= 2*y1 + 1  (when y1=0: x0>=1, when y1=1: x0>=3)
    store.addVLB(0, 1, 2.0, 1.0, 0);
    CHECK(store.hasVLB(0));
    CHECK(store.numVLBs() == 1);

    const auto& vlbs = store.vlbs(0);
    REQUIRE(vlbs.size() == 1);
    CHECK(vlbs[0].binary_var == 1);
    CHECK_THAT(vlbs[0].coeff, WithinAbs(2.0, 1e-10));
    CHECK_THAT(vlbs[0].constant, WithinAbs(1.0, 1e-10));
}

TEST_CASE("VariableBoundStore: duplicate VUB replaces if tighter", "[variable_bounds]") {
    VariableBoundStore store;
    store.init(5);

    // x0 <= 3*y1 + 5  (y=0: 5, y=1: 8)
    store.addVUB(0, 1, 3.0, 5.0);
    // x0 <= 2*y1 + 4  (y=0: 4, y=1: 6) -- tighter
    store.addVUB(0, 1, 2.0, 4.0);
    CHECK(store.numVUBs() == 1);

    const auto& vubs = store.vubs(0);
    REQUIRE(vubs.size() == 1);
    CHECK_THAT(vubs[0].coeff, WithinAbs(2.0, 1e-10));
    CHECK_THAT(vubs[0].constant, WithinAbs(4.0, 1e-10));
}

TEST_CASE("VariableBoundStore: bestVUB with primals", "[variable_bounds]") {
    VariableBoundStore store;
    store.init(5);

    // x0 <= 3*y1 + 2
    store.addVUB(0, 1, 3.0, 2.0);
    // x0 <= 1*y2 + 4
    store.addVUB(0, 2, 1.0, 4.0);

    std::vector<Real> primals = {0.0, 0.5, 0.8, 0.0, 0.0};
    Real best = store.bestVUB(0, primals);
    // y1=0.5: 3*0.5+2 = 3.5
    // y2=0.8: 1*0.8+4 = 4.8
    CHECK_THAT(best, WithinAbs(3.5, 1e-10));
}

TEST_CASE("VariableBoundStore: bestVLB with primals", "[variable_bounds]") {
    VariableBoundStore store;
    store.init(5);

    // x0 >= 2*y1 + 1
    store.addVLB(0, 1, 2.0, 1.0);
    // x0 >= 3*y2 + 0
    store.addVLB(0, 2, 3.0, 0.0);

    std::vector<Real> primals = {0.0, 0.5, 0.8, 0.0, 0.0};
    Real best = store.bestVLB(0, primals);
    // y1=0.5: 2*0.5+1 = 2.0
    // y2=0.8: 3*0.8+0 = 2.4
    CHECK_THAT(best, WithinAbs(2.4, 1e-10));
}

TEST_CASE("VariableBoundStore: clear resets state", "[variable_bounds]") {
    VariableBoundStore store;
    store.init(5);

    store.addVUB(0, 1, 3.0, 2.0);
    store.addVLB(0, 2, 1.0, 0.0);
    CHECK(store.numVUBs() == 1);
    CHECK(store.numVLBs() == 1);

    store.clear();
    CHECK(store.numVUBs() == 0);
    CHECK(store.numVLBs() == 0);
    CHECK_FALSE(store.hasVUB(0));
    CHECK_FALSE(store.hasVLB(0));
}

TEST_CASE("VariableBoundStore: out of range access", "[variable_bounds]") {
    VariableBoundStore store;
    store.init(3);

    CHECK_FALSE(store.hasVUB(-1));
    CHECK_FALSE(store.hasVUB(5));
    CHECK(store.vubs(-1).empty());
    CHECK(store.vubs(5).empty());
}

// =============================================================================
// ProbingEngine tests
// =============================================================================

TEST_CASE("Probing: detects fixing from infeasible branch", "[probing]") {
    // x0 + x1 <= 1, x0 + x2 <= 1, x1 + x2 >= 2
    // All binary.
    // If x0=1: x1<=0, x2<=0, but x1+x2>=2 => infeasible.
    // So x0 must be 0.
    auto prob = makeProblem(
        3, 3, {{0, 0, 1.0}, {0, 1, 1.0}, {1, 0, 1.0}, {1, 2, 1.0}, {2, 1, 1.0}, {2, 2, 1.0}},
        {-kInf, -kInf, 2.0}, {1.0, 1.0, kInf}, {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0},
        {VarType::Binary, VarType::Binary, VarType::Binary});

    ImplicationGraph graph;
    VariableBoundStore vb_store;
    ProbingEngine engine;
    ProbingConfig config;
    config.max_rounds = 1;

    auto stats = engine.probe(prob, graph, vb_store, config);

    CHECK(stats.variables_probed > 0);
    CHECK(stats.fixings_found >= 1);

    // Check that x0 is fixed to 0.
    bool found_x0_fixed = false;
    for (const auto& [var, val] : engine.fixings()) {
        if (var == 0) {
            CHECK_THAT(val, WithinAbs(0.0, 1e-10));
            found_x0_fixed = true;
        }
    }
    CHECK(found_x0_fixed);
}

TEST_CASE("Probing: learns binary implications", "[probing]") {
    // x0 + x1 <= 1 (both binary)
    // This means: x0=1 => x1=0 and x1=1 => x0=0
    auto prob = makeProblem(2, 1, {{0, 0, 1.0}, {0, 1, 1.0}}, {-kInf}, {1.0}, {0.0, 0.0},
                            {1.0, 1.0}, {VarType::Binary, VarType::Binary});

    ImplicationGraph graph;
    VariableBoundStore vb_store;
    ProbingEngine engine;
    ProbingConfig config;
    config.max_rounds = 1;

    auto stats = engine.probe(prob, graph, vb_store, config);

    CHECK(stats.implications_found > 0);

    // x0=1 should imply x1=0.
    const auto& imp = graph.implications(0, true);
    bool found = false;
    for (const auto& i : imp) {
        if (i.to_var == 1 && !i.to_val) {
            found = true;
        }
    }
    CHECK(found);
}

TEST_CASE("Probing: learns VUBs from bound tightening", "[probing]") {
    // x0 + 3*y <= 5, x0 continuous in [0, 10], y binary
    // When y=0: x0 <= 5
    // When y=1: x0 <= 2
    // VUB: x0 <= -3*y + 5
    auto prob = makeProblem(2, 1, {{0, 0, 1.0}, {0, 1, 3.0}}, {-kInf}, {5.0}, {0.0, 0.0},
                            {10.0, 1.0}, {VarType::Continuous, VarType::Binary});

    ImplicationGraph graph;
    VariableBoundStore vb_store;
    ProbingEngine engine;
    ProbingConfig config;
    config.max_rounds = 1;

    auto stats = engine.probe(prob, graph, vb_store, config);

    CHECK(stats.vubs_found > 0);
    CHECK(vb_store.hasVUB(0));
}

TEST_CASE("Probing: no fixings on fully feasible problem", "[probing]") {
    // x0 + x1 <= 3, both binary.
    // Both directions feasible for both variables.
    auto prob = makeProblem(2, 1, {{0, 0, 1.0}, {0, 1, 1.0}}, {-kInf}, {3.0}, {0.0, 0.0},
                            {1.0, 1.0}, {VarType::Binary, VarType::Binary});

    ImplicationGraph graph;
    VariableBoundStore vb_store;
    ProbingEngine engine;
    ProbingConfig config;
    config.max_rounds = 1;

    auto stats = engine.probe(prob, graph, vb_store, config);

    CHECK(stats.fixings_found == 0);
}

TEST_CASE("Probing: skips already fixed variables", "[probing]") {
    // x0 is already fixed (lb==ub==0), x1 binary.
    auto prob = makeProblem(2, 1, {{0, 0, 1.0}, {0, 1, 1.0}}, {-kInf}, {1.0}, {0.0, 0.0},
                            {0.0, 1.0}, {VarType::Binary, VarType::Binary});

    ImplicationGraph graph;
    VariableBoundStore vb_store;
    ProbingEngine engine;
    ProbingConfig config;

    auto stats = engine.probe(prob, graph, vb_store, config);

    // Only x1 should be probed (x0 is fixed).
    CHECK(stats.variables_probed == 1);
}

TEST_CASE("Probing: multiple rounds discover cascading fixings", "[probing]") {
    // x0 + x1 <= 1, x1 + x2 <= 1, x2 + x3 >= 2
    // All binary.
    // x3 must be 1 (since x2+x3>=2 and x2<=1 => x3>=1).
    // Actually: if x2=0, x3>=2 which is >1, infeasible. So x2=1.
    // If x2=1: x1<=0, so x1=0. Then x0 can be 0 or 1.
    auto prob = makeProblem(
        4, 3, {{0, 0, 1.0}, {0, 1, 1.0}, {1, 1, 1.0}, {1, 2, 1.0}, {2, 2, 1.0}, {2, 3, 1.0}},
        {-kInf, -kInf, 2.0}, {1.0, 1.0, kInf}, {0.0, 0.0, 0.0, 0.0}, {1.0, 1.0, 1.0, 1.0},
        {VarType::Binary, VarType::Binary, VarType::Binary, VarType::Binary});

    ImplicationGraph graph;
    VariableBoundStore vb_store;
    ProbingEngine engine;
    ProbingConfig config;
    config.max_rounds = 3;

    auto stats = engine.probe(prob, graph, vb_store, config);

    // Should discover some fixings.
    CHECK(stats.fixings_found >= 1);
}

TEST_CASE("Probing: empty binary set produces no work", "[probing]") {
    // All continuous variables.
    auto prob = makeProblem(2, 1, {{0, 0, 1.0}, {0, 1, 1.0}}, {-kInf}, {10.0}, {0.0, 0.0},
                            {5.0, 5.0}, {VarType::Continuous, VarType::Continuous});

    ImplicationGraph graph;
    VariableBoundStore vb_store;
    ProbingEngine engine;
    ProbingConfig config;

    auto stats = engine.probe(prob, graph, vb_store, config);

    CHECK(stats.variables_probed == 0);
    CHECK(stats.fixings_found == 0);
    CHECK(stats.implications_found == 0);
}

TEST_CASE("Probing: clique constraint learns mutual exclusion", "[probing]") {
    // x0 + x1 + x2 <= 1 (all binary)
    // Probing x0=1 should fix x1=0 and x2=0.
    auto prob =
        makeProblem(3, 1, {{0, 0, 1.0}, {0, 1, 1.0}, {0, 2, 1.0}}, {-kInf}, {1.0}, {0.0, 0.0, 0.0},
                    {1.0, 1.0, 1.0}, {VarType::Binary, VarType::Binary, VarType::Binary});

    ImplicationGraph graph;
    VariableBoundStore vb_store;
    ProbingEngine engine;
    ProbingConfig config;
    config.max_rounds = 1;

    auto stats = engine.probe(prob, graph, vb_store, config);

    CHECK(stats.implications_found > 0);

    // x0=1 => x1=0
    bool found01 = false;
    for (const auto& imp : graph.implications(0, true)) {
        if (imp.to_var == 1 && !imp.to_val) {
            found01 = true;
        }
    }
    CHECK(found01);

    // x0=1 => x2=0
    bool found02 = false;
    for (const auto& imp : graph.implications(0, true)) {
        if (imp.to_var == 2 && !imp.to_val) {
            found02 = true;
        }
    }
    CHECK(found02);
}

TEST_CASE("Probing: equivalence detection through probing", "[probing]") {
    // x0 + x1 = 1 (exactly one must be 1)
    // This means x0 = 1-x1 (opposite-sense equivalence).
    auto prob = makeProblem(2, 1, {{0, 0, 1.0}, {0, 1, 1.0}}, {1.0}, {1.0}, {0.0, 0.0}, {1.0, 1.0},
                            {VarType::Binary, VarType::Binary});

    ImplicationGraph graph;
    VariableBoundStore vb_store;
    ProbingEngine engine;
    ProbingConfig config;
    config.max_rounds = 1;
    config.detect_equivalences = true;

    auto stats = engine.probe(prob, graph, vb_store, config);

    CHECK(stats.equivalences_found == 1);
    REQUIRE(engine.equivalences().size() == 1);
    CHECK(engine.equivalences()[0].same_sense == false);
}

TEST_CASE("Probing: proves infeasibility when both branches fail", "[probing]") {
    // 2*x0 >= 1 forces x0=1; 2*x0 <= 1 forces x0=0. Both probe branches are
    // infeasible under propagation, so probing must report infeasibility.
    auto prob = makeProblem(1, 2, {{0, 0, 2.0}, {1, 0, 2.0}}, {1.0, -kInf}, {kInf, 1.0}, {0.0},
                            {1.0}, {VarType::Binary});

    ImplicationGraph graph;
    VariableBoundStore vb_store;
    ProbingEngine engine;

    auto stats = engine.probe(prob, graph, vb_store);
    CHECK(stats.infeasible);
}
