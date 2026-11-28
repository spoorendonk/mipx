#include "mipx/bound_conflict.h"
#include "mipx/domain.h"
#include "mipx/implication_graph.h"
#include "mipx/lp_problem.h"
#include "mipx/mip_solver.h"
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

TEST_CASE("VariableBoundStore: coefficient strengthening from a VUB", "[variable_bounds]") {
    // Model: y binary (col 0), x continuous in [0, 10] (col 1) with the
    // variable upper bound x <= 5*y, z continuous in [0, 2] (col 2).
    //   row:  3*y + x + z <= 4
    // In the y = 0 branch the VUB forces x = 0, so the other terms reach at
    // most z = 2. The surplus is 4 - 2 = 2 < 3, so the row is equivalent to
    //   1*y + x + z <= 2.
    VariableBoundStore store;
    store.init(3);
    store.addVUB(1, 0, 5.0, 0.0);  // x <= 5*y + 0

    const std::vector<Index> idx = {0, 1, 2};
    const std::vector<Real> vals = {3.0, 1.0, 1.0};
    const std::vector<Real> lower = {0.0, 0.0, 0.0};
    const std::vector<Real> upper = {1.0, 10.0, 2.0};

    const auto res = store.strengthenCoefficient(0, 3.0, 4.0, idx, vals, lower, upper);
    REQUIRE(res.strengthened);
    CHECK_THAT(res.new_coeff, WithinAbs(1.0, 1e-9));
    CHECK_THAT(res.rhs_delta, WithinAbs(-2.0, 1e-9));

    // The strengthened row cuts off no point that is feasible for the original
    // model (row + variable upper bound + column bounds).
    const Real new_rhs = 4.0 + res.rhs_delta;
    bool all_preserved = true;
    for (int yi = 0; yi <= 1 && all_preserved; ++yi) {
        const Real y = static_cast<Real>(yi);
        for (int xi = 0; xi <= 100 && all_preserved; ++xi) {
            const Real x = 0.1 * xi;
            if (x > 5.0 * y + 1e-12)
                continue;  // Violates the VUB.
            for (int zi = 0; zi <= 20; ++zi) {
                const Real z = 0.1 * zi;
                if (3.0 * y + x + z > 4.0 + 1e-9)
                    continue;  // Violates the row.
                if (res.new_coeff * y + x + z > new_rhs + 1e-9) {
                    all_preserved = false;
                    break;
                }
            }
        }
    }
    CHECK(all_preserved);

    // The strengthened row is genuinely tighter: (y, x, z) = (0.5, 2.5, 0)
    // satisfies the original row and the VUB but violates the new one.
    CHECK(3.0 * 0.5 + 2.5 + 0.0 <= 4.0 + 1e-9);
    CHECK(2.5 <= 5.0 * 0.5 + 1e-9);
    CHECK(res.new_coeff * 0.5 + 2.5 > new_rhs + 1e-9);
}

TEST_CASE("VariableBoundStore: no strengthening without a variable bound", "[variable_bounds]") {
    // Same row as above, but without the VUB the y = 0 branch only knows
    // x <= 10, so the row is not slack there and nothing can be deduced.
    VariableBoundStore store;
    store.init(3);

    const std::vector<Index> idx = {0, 1, 2};
    const std::vector<Real> vals = {3.0, 1.0, 1.0};
    const std::vector<Real> lower = {0.0, 0.0, 0.0};
    const std::vector<Real> upper = {1.0, 10.0, 2.0};

    const auto res = store.strengthenCoefficient(0, 3.0, 4.0, idx, vals, lower, upper);
    CHECK_FALSE(res.strengthened);
    CHECK_THAT(res.new_coeff, WithinAbs(3.0, 1e-12));
    CHECK_THAT(res.rhs_delta, WithinAbs(0.0, 1e-12));
}

TEST_CASE("VariableBoundStore: coefficient strengthening from a VLB", "[variable_bounds]") {
    // y binary (col 0), x continuous in [0, 10] (col 1) with x >= 5*y,
    // z continuous in [0, 2] (col 2).
    //   row:  -3*y - x + z <= -4
    // The y = 1 branch is slack (x >= 5, z <= 2 gives activity -3 - 5 + 2),
    // with surplus 2 < 3, so the coefficient of y shrinks to -1.
    VariableBoundStore store;
    store.init(3);
    store.addVLB(1, 0, 5.0, 0.0);  // x >= 5*y + 0

    const std::vector<Index> idx = {0, 1, 2};
    const std::vector<Real> vals = {-3.0, -1.0, 1.0};
    const std::vector<Real> lower = {0.0, 0.0, 0.0};
    const std::vector<Real> upper = {1.0, 10.0, 2.0};

    const auto res = store.strengthenCoefficient(0, -3.0, -4.0, idx, vals, lower, upper);
    REQUIRE(res.strengthened);
    CHECK_THAT(res.new_coeff, WithinAbs(-1.0, 1e-9));
    CHECK_THAT(res.rhs_delta, WithinAbs(0.0, 1e-12));

    const Real new_rhs = -4.0 + res.rhs_delta;
    bool all_preserved = true;
    for (int yi = 0; yi <= 1 && all_preserved; ++yi) {
        const Real y = static_cast<Real>(yi);
        for (int xi = 0; xi <= 100 && all_preserved; ++xi) {
            const Real x = 0.1 * xi;
            if (x < 5.0 * y - 1e-12)
                continue;  // Violates the VLB.
            for (int zi = 0; zi <= 20; ++zi) {
                const Real z = 0.1 * zi;
                if (-3.0 * y - x + z > -4.0 + 1e-9)
                    continue;  // Violates the row.
                if (res.new_coeff * y - x + z > new_rhs + 1e-9) {
                    all_preserved = false;
                    break;
                }
            }
        }
    }
    CHECK(all_preserved);
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

TEST_CASE("Systematic probing publishes unique global facts and uses them in the tree",
          "[probing][mip]") {
    // The first row has a fractional LP optimum and forbids x=y=1.
    // The second row makes z=1 infeasible even though z starts unfixed.
    auto model = makeProblem(3, 2, {{0, 0, 2.0}, {0, 1, 2.0}, {1, 2, 2.0}}, {-kInf, -kInf},
                             {3.0, 1.0}, {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0},
                             {VarType::Binary, VarType::Integer, VarType::Binary});
    model.obj = {-1.0, -1.0, -0.25};

    ImplicationGraph graph;
    VariableBoundStore vb_store;
    ProbingEngine engine;
    ProbingConfig config;
    config.max_probes_per_round = 2;
    config.learn_vubs = false;
    config.time_limit = kInf;
    config.systematic_rows_per_branch = 1;
    const auto limited = engine.probe(model, graph, vb_store, config);
    CHECK(limited.branches_probed == std::array{2, 2});
    CHECK(limited.row_visits == 4);
    CHECK(engine.checkedConflicts().empty());
    CHECK(engine.fixings().empty());

    config.max_probes_per_round = 3;
    const auto full = engine.probe(model, graph, vb_store, config);
    CHECK(full.branches_probed == std::array{3, 3});
    CHECK(full.row_visits == 6);
    CHECK(full.implication_candidates == 7);
    CHECK(full.implications_found == 2);  // One implication and its contrapositive.
    CHECK(graph.numImplications() == full.implications_found);
    REQUIRE(engine.fixings().size() == 1);
    CHECK(engine.fixings().front() == std::pair{2, 0.0});
    REQUIRE(engine.checkedConflicts().size() == 1);
    const auto& forbidden = engine.checkedConflicts().front();
    CHECK(forbidden.variable == 2);
    CHECK_FALSE(forbidden.is_upper);
    CHECK(forbidden.bound == 1.0);

    Real enumerated_optimum = kInf;
    for (unsigned mask = 0; mask < 8; ++mask) {
        const std::array bits{bool(mask & 1), bool(mask & 2), bool(mask & 4)};
        if (2 * bits[0] + 2 * bits[1] > 3 || 2 * bits[2] > 1) {
            continue;
        }
        enumerated_optimum = std::min(enumerated_optimum, -bits[0] - bits[1] - 0.25 * bits[2]);
        for (Index col : graph.binaryVars()) {
            for (const auto& [target, value] : graph.implications(col, bits[col])) {
                CHECK(bits[target] == value);
            }
        }
        for (const auto& [col, value] : engine.fixings()) {
            CHECK(bits[col] == value);
        }
        const Real value = bits[forbidden.variable];
        CHECK_FALSE((forbidden.is_upper ? value <= forbidden.bound : value >= forbidden.bound));
    }

    MipSolver solver;
    solver.setPresolve(false);
    solver.setCutsEnabled(false);
    solver.setStrongBranchingEnabled(false);
    solver.setSymmetryEnabled(false);
    solver.setTreePresolveEnabled(false);
    solver.setObjectiveIntegralityEnabled(false);
    solver.setSystematicProbingMaxPairs(3);
    solver.setSystematicProbingRowsPerBranch(1);
    solver.load(model);
    const auto check_result = [&](const MipResult& result) {
        REQUIRE(result.status == Status::Optimal);
        CHECK(result.objective == enumerated_optimum);
        CHECK(result.best_bound == enumerated_optimum);
        REQUIRE(result.solution.size() == 3);
        CHECK(result.solution[0] + result.solution[1] == 1.0);
        CHECK(result.solution[2] == 0.0);
    };

    check_result(solver.solve());
    CHECK_FALSE(solver.getProbingStats().enabled);
    solver.setSystematicProbingEnabled(true);
    for (int run = 0; run < 2; ++run) {
        check_result(solver.solve());
        const auto& stats = solver.getProbingStats();
        CHECK(stats.systematic);
        CHECK(stats.branches_probed == full.branches_probed);
        CHECK(stats.row_visits == full.row_visits);
        CHECK(stats.implications_found == full.implications_found);
        CHECK(stats.fixings_found == 1);
        CHECK(stats.checked_conflicts_published == 1);
        CHECK(stats.tree_implication_fixings > 0);
    }
    solver.setSystematicProbingEnabled(false);
    check_result(solver.solve());
    CHECK(solver.getImplicationGraph().numImplications() == 0);
    CHECK(solver.getProbingStats().checked_conflicts_published == 0);
    CHECK(solver.getProbingStats().tree_implication_fixings == 0);

    solver.setProbingEnabled(true);
    for (int run = 0; run < 2; ++run) {
        check_result(solver.solve());
        CHECK_FALSE(solver.getProbingStats().systematic);
        CHECK(solver.getProbingStats().fixings_found == 1);
        CHECK(solver.getImplicationGraph().numImplications() > 0);
    }
    solver.setProbingEnabled(false);
    check_result(solver.solve());
    CHECK_FALSE(solver.getProbingStats().enabled);
    CHECK(solver.getImplicationGraph().numImplications() == 0);
}

TEST_CASE("Systematic probing requires a completed proof to exclude a branch", "[probing]") {
    auto model =
        makeProblem(2, 2, {{0, 0, 1.0}, {0, 1, -1.0}, {1, 1, 2.0}}, {-kInf, -kInf}, {0.0, 1.0},
                    {0.0, 0.0}, {1.0, 1.0}, {VarType::Binary, VarType::Binary});
    ProbingEngine engine;
    ProbingConfig config;
    config.max_probes_per_round = 1;
    config.systematic_rows_per_branch = 1;
    ImplicationGraph graph;
    VariableBoundStore vb_store;
    const auto limited = engine.probe(model, graph, vb_store, config);
    CHECK(limited.branches_probed == std::array{1, 1});
    CHECK(engine.checkedConflicts().empty());
    CHECK(engine.fixings().empty());

    config.systematic_rows_per_branch = 4;
    engine.probe(model, graph, vb_store, config);
    REQUIRE(engine.checkedConflicts().size() == 1);
    REQUIRE(engine.fixings().size() == 1);
    CHECK(engine.fixings().front() == std::pair{0, 0.0});

    model.col_type[1] = VarType::SemiContinuous;
    const auto skipped = engine.probe(model, graph, vb_store, config);
    CHECK(skipped.skipped_unsupported_domains);
    CHECK(skipped.branches_probed == std::array{0, 0});
    CHECK(engine.checkedConflicts().empty());
    CHECK(engine.fixings().empty());
    CHECK(graph.numImplications() == 0);
}

TEST_CASE("Global contradiction checking rejects local rows and objective cutoffs", "[probing]") {
    auto model = makeProblem(1, 1, {{0, 0, 2.0}}, {-kInf}, {1.0}, {0.0}, {1.0}, {VarType::Binary});
    const std::array premises{BranchDecision{0, 1.0, false, 1}};
    DomainPropagator dp;
    dp.load(model, true);
    dp.setBranchBound(0, false, 1.0, 1);
    REQUIRE_FALSE(dp.propagate());
    CHECK(checkGlobalBoundContradiction(model, *dp.boundReasons(), premises));
    CHECK_FALSE(checkGlobalBoundContradiction(model, *dp.boundReasons(), {}));

    dp.load(model, true, 0);
    dp.setBranchBound(0, false, 1.0, 1);
    REQUIRE_FALSE(dp.propagate());
    CHECK_FALSE(checkGlobalBoundContradiction(model, *dp.boundReasons(), premises));
    CHECK_FALSE(checkedGlobalBoundConflict(model, *dp.boundReasons(), premises));

    model.row_upper[0] = kInf;
    model.obj[0] = 1.0;
    ObjectivePropagator objective;
    objective.load(model);
    dp.load(model, true);
    dp.setObjectiveCutoff(&objective, 0.5);
    dp.setBranchBound(0, false, 1.0, 1);
    REQUIRE_FALSE(dp.propagate());
    CHECK(dp.objectiveInfeasible());
    CHECK_FALSE(checkGlobalBoundContradiction(model, *dp.boundReasons(), premises));
    CHECK_FALSE(checkedGlobalBoundConflict(model, *dp.boundReasons(), premises));
}

TEST_CASE("Rapid learning imports a two-decision conflict into the main tree",
          "[probing][rapid-learning][mip]") {
    // z is irrelevant. x=y=1 forces u=v=1 in the second row, then
    // contradicts the first. u=v keeps both values of z within the yield
    // window and produces a conflict when either x or y is 1 and u is 0.
    // w has an independent global fixing.
    auto model = makeProblem(6, 4,
                             {{0, 1, 1},
                              {0, 2, 1},
                              {0, 3, 1},
                              {0, 4, 1},
                              {1, 1, -1},
                              {1, 2, -1},
                              {1, 3, 1},
                              {1, 4, 1},
                              {2, 5, 2},
                              {3, 3, 1},
                              {3, 4, -1}},
                             {-kInf, 0, -kInf, 0}, {3, kInf, 1, 0}, std::vector<Real>(6, 0),
                             std::vector<Real>(6, 1), std::vector<VarType>(6, VarType::Binary));
    model.obj = {0, -3, -2, 0, 0, -0.25};
    auto feasible = [](unsigned mask) {
        const unsigned xy = ((mask >> 1) & 1u) + ((mask >> 2) & 1u);
        const unsigned uv = ((mask >> 3) & 1u) + ((mask >> 4) & 1u);
        return xy <= uv && xy + uv <= 3 && uv != 1 && (mask & 32u) == 0u;
    };

    ProbingEngine probing;
    ImplicationGraph graph;
    VariableBoundStore bounds;
    ProbingConfig config;
    config.systematic_rows_per_branch = 100;
    config.learn_vubs = false;
    const auto probe_stats = probing.probe(model, graph, bounds, config);
    CHECK_FALSE(probe_stats.infeasible);
    REQUIRE(probing.checkedConflicts().size() == 1);
    CHECK(probing.checkedConflicts().front().variable == 5);

    const std::array path{BranchDecision{0, 0, true, 1}, BranchDecision{1, 1, false, 2},
                          BranchDecision{2, 1, false, 3}};
    DomainPropagator domain;
    domain.load(model, true);
    for (const auto& branch : path) {
        domain.setBranchBound(branch.variable, branch.is_upper, branch.bound, branch.level);
    }
    REQUIRE_FALSE(domain.propagate());
    const auto proof = checkedGlobalBoundConflict(model, *domain.boundReasons(), path);
    REQUIRE(proof);
    REQUIRE(proof->size() == 2);
    CHECK(std::ranges::none_of(*proof, [](const auto& branch) { return branch.variable == 0; }));

    MipSolver solver;
    solver.setVerbose(false);
    solver.setPresolve(false);
    solver.setCutsEnabled(false);
    solver.setStrongBranchingEnabled(false);
    solver.setSymmetryEnabled(false);
    solver.setTreePresolveEnabled(false);
    solver.setGapTolerance(0);
    std::vector<Real> observed;
    solver.setIncumbentObserver([&](std::span<const Real> point, const std::string& source) {
        if (source == "root_rapid_learning") {
            observed.assign(point.begin(), point.end());
        }
    });
    solver.load(model);
    Real optimum = kInf;
    for (unsigned mask = 0; mask < 64; ++mask) {
        if (feasible(mask)) {
            CHECK_FALSE(std::ranges::all_of(*proof, [&](const auto& branch) {
                return ((mask >> branch.variable) & 1u) == branch.bound;
            }));
            optimum = std::min(optimum, -3.0 * ((mask >> 1) & 1u) - 2.0 * ((mask >> 2) & 1u));
        }
    }
    REQUIRE(solver.solve().objective == optimum);
    CHECK(solver.getRapidLearningStats().stop == RapidLearningStop::NotRun);
    solver.setRootRapidLearningEnabled(true);
    for (int run = 0; run < 2; ++run) {
        const auto result = solver.solve();
        REQUIRE(result.status == Status::Optimal);
        CHECK(result.objective == optimum);
        CHECK(result.best_bound == optimum);
        REQUIRE(observed.size() == 6);
        CHECK(pointHoldsModel(model, observed));
        observed.clear();
        const auto& stats = solver.getRapidLearningStats();
        CHECK(stats.bounds_imported == 1);
        CHECK(stats.conflicts_imported == 3);
        CHECK(stats.duplicate_conflicts > 0);
        CHECK(stats.incumbents_imported > 0);
        CHECK(stats.tree_conflict_tightenings + stats.tree_conflict_cutoffs > 0);
    }
    for (unsigned mask = 0; mask < 64; ++mask) {
        auto fixed = model;
        for (Index col = 0; col < fixed.num_cols; ++col) {
            fixed.col_lower[col] = fixed.col_upper[col] = (mask >> col) & 1u;
        }
        solver.load(fixed);
        CHECK(solver.solve().status == (feasible(mask) ? Status::Optimal : Status::Infeasible));
    }
}

TEST_CASE("Rapid learning stops at its work caps and completed empty blocks",
          "[probing][rapid-learning][mip]") {
    MipSolver solver;
    solver.setPresolve(false);
    solver.setSymmetryEnabled(false);
    solver.setRootRapidLearningEnabled(true);
    auto model = makeProblem(20, 1, {{0, 0, 1}}, {-kInf}, {1}, std::vector<Real>(20, 0),
                             std::vector<Real>(20, 1), std::vector<VarType>(20, VarType::Binary));
    solver.load(model);
    solver.setRootRapidLearningMaxNodes(3);
    REQUIRE(solver.solve().status == Status::Optimal);
    CHECK(solver.getRapidLearningStats().stop == RapidLearningStop::NodeLimit);
    CHECK(solver.getRapidLearningStats().nodes == 3);
    CHECK(solver.getRapidLearningStats().empty_blocks == 0);
    solver.setRootRapidLearningMaxNodes(128);
    solver.setRootRapidLearningRowBudget(2);
    REQUIRE(solver.solve().status == Status::Optimal);
    CHECK(solver.getRapidLearningStats().stop == RapidLearningStop::RowLimit);
    CHECK(solver.getRapidLearningStats().row_visits == 2);
    solver.setRootRapidLearningRowBudget(128);
    REQUIRE(solver.solve().status == Status::Optimal);
    CHECK(solver.getRapidLearningStats().stop == RapidLearningStop::NoYield);
    CHECK(solver.getRapidLearningStats().nodes == 16);
    CHECK(solver.getRapidLearningStats().empty_blocks == 2);
    solver.setRootRapidLearningMaxNodes(0);
    REQUIRE(solver.solve().status == Status::Optimal);
    CHECK(solver.getRapidLearningStats().nodes == 0);
    CHECK(solver.getRapidLearningStats().row_visits == 0);
}

TEST_CASE("Rapid learning rejects a propagated leaf that fails model validation",
          "[probing][rapid-learning][mip]") {
    MipSolver solver;
    solver.setPresolve(false);
    solver.setSymmetryEnabled(false);
    solver.setRootRapidLearningEnabled(true);
    auto model = makeProblem(3, 1, {{0, 1, 1}, {0, 2, 1}}, {0.5}, {0.5}, {0, 0, 0}, {1, 1, 1},
                             {VarType::Binary, VarType::Continuous, VarType::Continuous});
    solver.load(model);
    const auto result = solver.solve();
    REQUIRE(result.status == Status::Optimal);
    CHECK(pointHoldsModel(model, result.solution));
    CHECK(solver.getRapidLearningStats().incumbents_imported == 0);
    CHECK(solver.getRapidLearningStats().incumbents_rejected == 2);
}

TEST_CASE("Rapid learning cannot use a nearly feasible point as a scaled objective cutoff",
          "[probing][rapid-learning][mip]") {
    // At least one binary is 1, so x >= 1. Propagation's continuous
    // endpoint can sit just below 1 and still pass the ordinary tolerance.
    auto model =
        makeProblem(3, 3, {{0, 0, 1}, {0, 1, -1}, {1, 0, 1}, {1, 2, -1}, {2, 1, 2}, {2, 2, 2}},
                    {0, 0, 1}, {kInf, kInf, kInf}, {0, 0, 0}, {2, 1, 1},
                    {VarType::Continuous, VarType::Binary, VarType::Binary});
    model.obj = {1e8, 0.1, 0.1};
    model.obj_offset = -1e8;
    MipSolver solver;
    solver.setPresolve(false);
    solver.setSymmetryEnabled(false);
    solver.setRootRapidLearningEnabled(true);
    solver.setGapTolerance(0);
    solver.load(model);
    const auto result = solver.solve();
    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(0.1, 1e-7));
    CHECK_THAT(result.best_bound, WithinAbs(0.1, 1e-7));
    CHECK(solver.getRapidLearningStats().incumbents_imported == 0);
    CHECK(solver.getRapidLearningStats().incumbents_rejected > 0);
}

TEST_CASE("Rapid learning reports a certified incumbent in loaded columns after presolve",
          "[probing][rapid-learning][mip]") {
    // Presolve removes the fixed continuous column and its objective term.
    // The odd cycle leaves a nontrivial binary problem.
    auto model = makeProblem(6, 5,
                             {{0, 0, 1},
                              {0, 1, 1},
                              {1, 1, 1},
                              {1, 2, 1},
                              {2, 2, 1},
                              {2, 3, 1},
                              {3, 3, 1},
                              {3, 4, 1},
                              {4, 4, 1},
                              {4, 0, 1}},
                             std::vector<Real>(5, -kInf), std::vector<Real>(5, 1),
                             {0, 0, 0, 0, 0, 3}, {1, 1, 1, 1, 1, 3},
                             {VarType::Binary, VarType::Binary, VarType::Binary, VarType::Binary,
                              VarType::Binary, VarType::Continuous});
    model.obj = {-1, -1, -1, -1, -1, 2};
    Presolver presolver;
    const auto reduced = presolver.presolve(model);
    CHECK(reduced.num_cols < model.num_cols);
    REQUIRE(reduced.hasIntegers());

    MipSolver solver;
    solver.setSymmetryEnabled(false);
    solver.setRootRapidLearningEnabled(true);
    std::vector<Real> observed;
    solver.setIncumbentObserver([&](std::span<const Real> point, const std::string& source) {
        if (source == "root_rapid_learning") {
            observed.assign(point.begin(), point.end());
        }
    });
    solver.load(model);
    const auto result = solver.solve();
    REQUIRE(result.status == Status::Optimal);
    CHECK(result.objective == 4);
    REQUIRE(solver.getRapidLearningStats().incumbents_imported > 0);
    REQUIRE(observed.size() == 6);
    CHECK(observed.back() == 3);
    CHECK(pointHoldsModel(model, observed));
}

TEST_CASE("Rapid learning spends one allowance across native SOS2 unbounded confirmation",
          "[probing][rapid-learning][mip]") {
    auto model = makeProblem(5, 0, {}, {}, {}, {0.5, 0, 0, 0, 0}, {1, 1, 1, kInf, 1},
                             {VarType::Continuous, VarType::Continuous, VarType::Continuous,
                              VarType::Continuous, VarType::Binary});
    model.obj = {0, 0, 0, -1, -1};
    model.native_sos2 = {{0, 1, 2}};
    MipSolver solver;
    solver.setPresolve(false);
    solver.setSymmetryEnabled(false);
    solver.setRootRapidLearningEnabled(true);
    solver.setRootRapidLearningMaxNodes(2);
    std::vector<Real> observed;
    solver.setIncumbentObserver([&](std::span<const Real> point, const std::string& source) {
        if (source == "root_rapid_learning") {
            observed.push_back(point[4]);
        }
    });
    solver.load(model);
    for (int run = 0; run < 2; ++run) {
        // The second attempt minimizes zero, so if allowed to run rapid
        // learning again it would visit the binary's other value first.
        REQUIRE(solver.solve().status == Status::Unbounded);
        const auto& stats = solver.getRapidLearningStats();
        CHECK(stats.stop == RapidLearningStop::NodeLimit);
        CHECK(stats.nodes == 2);
        CHECK(stats.incumbents_imported == 1);
        CHECK(observed == std::vector<Real>{1});
        observed.clear();
    }
}

TEST_CASE("Rapid learning charges dense proof checks to the propagation row allowance",
          "[probing][rapid-learning][mip]") {
    constexpr Index n = 32;
    std::vector<Triplet> row;
    for (Index col = 0; col < n; ++col) {
        row.push_back({0, col, 1});
    }
    auto model = makeProblem(n, 1, std::move(row), {n}, {kInf}, std::vector<Real>(n, 0),
                             std::vector<Real>(n, 1), std::vector<VarType>(n, VarType::Binary));
    DomainPropagator domain;
    domain.load(model, true);
    const auto expired = Deadline::after(0);
    Int rows_left = 3;
    CHECK(domain.propagateWithRowBudget(rows_left, &expired));
    CHECK(rows_left == 3);
    CHECK(domain.getLower(0) == 0);
    REQUIRE(domain.propagate());
    CHECK(domain.getLower(0) == 1);
    const auto& trail = *domain.boundReasons();
    const auto proof = trail.current(0, false);
    REQUIRE(checkGlobalBoundDeduction(model, trail, proof));
    BoundProofBudget expired_budget{rows_left, expired};
    CHECK_FALSE(checkGlobalBoundDeduction(model, trail, proof, &expired_budget));
    CHECK(expired_budget.stopped());
    CHECK(rows_left == 3);

    MipSolver solver;
    solver.setPresolve(false);
    solver.setSymmetryEnabled(false);
    solver.setRootRapidLearningEnabled(true);
    solver.setRootRapidLearningRowBudget(3);
    solver.load(model);
    REQUIRE(solver.solve().status == Status::Optimal);
    const auto& stats = solver.getRapidLearningStats();
    CHECK(stats.stop == RapidLearningStop::RowLimit);
    CHECK(stats.nodes == 1);
    CHECK(stats.row_visits == 3);
    CHECK(stats.bounds_imported < n);
    CHECK(stats.proofs_rejected == 0);
    CHECK(stats.incumbents_imported == 0);
}

TEST_CASE("Rapid learning charges candidate validation to the propagation row allowance",
          "[probing][rapid-learning][mip]") {
    const auto model = makeProblem(2, 1, {{0, 0, 1}, {0, 1, 1}}, {-kInf}, {1}, {0, 0}, {1, 1},
                                   {VarType::Binary, VarType::Binary});
    MipSolver solver;
    solver.setPresolve(false);
    solver.setSymmetryEnabled(false);
    solver.setRootRapidLearningEnabled(true);
    solver.setRootRapidLearningRowBudget(4);
    solver.load(model);
    REQUIRE(solver.solve().status == Status::Optimal);
    const auto& stats = solver.getRapidLearningStats();
    CHECK(stats.stop == RapidLearningStop::RowLimit);
    CHECK(stats.nodes == 3);
    CHECK(stats.row_visits == 4);
    CHECK(stats.incumbents_imported == 1);
}
