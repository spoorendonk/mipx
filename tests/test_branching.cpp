#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "mipx/branching.h"
#include "mipx/dual_simplex.h"
#include "mipx/symmetry.h"

using namespace mipx;

TEST_CASE("MostFractional selects variable closest to 0.5", "[branching]") {
    // x0 = 1.1 (frac 0.1), x1 = 2.5 (frac 0.5), x2 = 3.3 (frac 0.3)
    std::vector<Real> vals = {1.1, 2.5, 3.3};
    std::vector<VarType> types = {VarType::Integer, VarType::Integer, VarType::Integer};
    std::vector<Real> lb = {0, 0, 0};
    std::vector<Real> ub = {10, 10, 10};

    MostFractionalBranching rule;
    Index selected = rule.select(vals, types, lb, ub);
    REQUIRE(selected == 1);
}

TEST_CASE("FirstFractional selects first fractional variable", "[branching]") {
    // x0 integral, x1 fractional, x2 fractional
    std::vector<Real> vals = {2.0, 1.7, 3.3};
    std::vector<VarType> types = {VarType::Integer, VarType::Integer, VarType::Integer};
    std::vector<Real> lb = {0, 0, 0};
    std::vector<Real> ub = {10, 10, 10};

    FirstFractionalBranching rule;
    Index selected = rule.select(vals, types, lb, ub);
    REQUIRE(selected == 1);
}

TEST_CASE("All integral returns -1", "[branching]") {
    std::vector<Real> vals = {1.0, 2.0, 3.0};
    std::vector<VarType> types = {VarType::Integer, VarType::Integer, VarType::Integer};
    std::vector<Real> lb = {0, 0, 0};
    std::vector<Real> ub = {10, 10, 10};

    MostFractionalBranching mf;
    REQUIRE(mf.select(vals, types, lb, ub) == -1);

    FirstFractionalBranching ff;
    REQUIRE(ff.select(vals, types, lb, ub) == -1);
}

TEST_CASE("Continuous variables are skipped", "[branching]") {
    // x0 continuous fractional, x1 integer fractional
    std::vector<Real> vals = {1.5, 2.5};
    std::vector<VarType> types = {VarType::Continuous, VarType::Integer};
    std::vector<Real> lb = {0, 0};
    std::vector<Real> ub = {10, 10};

    FirstFractionalBranching rule;
    REQUIRE(rule.select(vals, types, lb, ub) == 1);
}

TEST_CASE("Binary variables handled correctly", "[branching]") {
    // x0 = 0.3 binary, x1 = 0.7 binary
    std::vector<Real> vals = {0.3, 0.7};
    std::vector<VarType> types = {VarType::Binary, VarType::Binary};
    std::vector<Real> lb = {0, 0};
    std::vector<Real> ub = {1, 1};

    MostFractionalBranching rule;
    Index selected = rule.select(vals, types, lb, ub);
    // Both are fractional binary variables; either is a valid selection.
    REQUIRE((selected == 0 || selected == 1));
}

TEST_CASE("Fixed variables are never selected", "[branching]") {
    // x0 fixed (lb==ub), x1 fractional
    std::vector<Real> vals = {2.5, 1.5};
    std::vector<VarType> types = {VarType::Integer, VarType::Integer};
    std::vector<Real> lb = {2.5, 0};
    std::vector<Real> ub = {2.5, 10};

    FirstFractionalBranching rule;
    REQUIRE(rule.select(vals, types, lb, ub) == 1);
}

TEST_CASE("createChildren produces correct child nodes", "[branching]") {
    BnbNode parent;
    parent.id = 5;
    parent.depth = 3;
    parent.basis = {BasisStatus::Basic, BasisStatus::AtLower};

    auto [left, right] = createChildren(parent, 0, 2.7);

    // Left child: x_0 <= 2 (upper bound).
    REQUIRE(left.parent_id == 5);
    REQUIRE(left.depth == 4);
    REQUIRE(left.branch.variable == 0);
    REQUIRE(left.branch.bound == 2.0);
    REQUIRE(left.branch.is_upper == true);
    REQUIRE(left.basis.size() == 2);
    REQUIRE(left.bound_changes.size() == 1);

    // Right child: x_0 >= 3 (lower bound).
    REQUIRE(right.parent_id == 5);
    REQUIRE(right.depth == 4);
    REQUIRE(right.branch.variable == 0);
    REQUIRE(right.branch.bound == 3.0);
    REQUIRE(right.branch.is_upper == false);
    REQUIRE(right.basis.size() == 2);
    REQUIRE(right.bound_changes.size() == 1);
}

TEST_CASE("createChildren accumulates bound changes", "[branching]") {
    BnbNode parent;
    parent.id = 2;
    parent.depth = 1;
    parent.bound_changes = {{0, 1.0, false}};  // existing change

    auto [left, right] = createChildren(parent, 1, 3.5);

    REQUIRE(left.bound_changes.size() == 2);
    REQUIRE(left.bound_changes[0].variable == 0);
    REQUIRE(left.bound_changes[1].variable == 1);

    REQUIRE(right.bound_changes.size() == 2);
}

TEST_CASE("createChildren propagates local subtree cuts", "[branching]") {
    BnbNode parent;
    Cut local_cut;
    local_cut.indices = {0};
    local_cut.values = {1.0};
    local_cut.upper = 2.0;
    local_cut.local = true;
    parent.local_cuts.push_back(local_cut);

    auto [left, right] = createChildren(parent, 0, 1.4);
    REQUIRE(left.local_cuts.size() == 1);
    REQUIRE(right.local_cuts.size() == 1);
    CHECK(left.local_cuts[0].local);
    CHECK(right.local_cuts[0].local);
}

TEST_CASE("isIntegral helper", "[branching]") {
    REQUIRE(isIntegral(3.0));
    REQUIRE(isIntegral(3.0 + 1e-8));
    REQUIRE(isIntegral(3.0 - 1e-8));
    REQUIRE_FALSE(isIntegral(3.5));
    REQUIRE_FALSE(isIntegral(3.1));
    // Custom tolerance.
    REQUIRE(isIntegral(3.01, 0.02));
    REQUIRE_FALSE(isIntegral(3.01, 0.005));
}

TEST_CASE("fractionality helper", "[branching]") {
    REQUIRE(fractionality(3.0) < 1e-12);
    REQUIRE_THAT(fractionality(3.5), Catch::Matchers::WithinAbs(0.5, 1e-12));
    REQUIRE_THAT(fractionality(3.3), Catch::Matchers::WithinAbs(0.3, 1e-12));
    REQUIRE_THAT(fractionality(3.7), Catch::Matchers::WithinAbs(0.3, 1e-12));
}

TEST_CASE("ReliabilityBranching updates pseudocost reliability counters", "[branching]") {
    ReliabilityBranching branching;
    branching.reset(3);
    branching.setReliabilityThreshold(2);

    branching.updatePseudoCost(1, false, 2.0);
    branching.updatePseudoCost(1, true, 6.0);
    CHECK_FALSE(branching.isReliable(1));
    CHECK(branching.downReliability(1) == 1);
    CHECK(branching.upReliability(1) == 1);

    branching.updatePseudoCost(1, false, 4.0);
    branching.updatePseudoCost(1, true, 2.0);
    CHECK(branching.isReliable(1));
    CHECK(branching.downReliability(1) == 2);
    CHECK(branching.upReliability(1) == 2);
    CHECK_THAT(branching.downPseudoCost(1), Catch::Matchers::WithinAbs(3.0, 1e-12));
    CHECK_THAT(branching.upPseudoCost(1), Catch::Matchers::WithinAbs(4.0, 1e-12));
}

TEST_CASE("SymmetryManager adds canonical symmetry cuts", "[symmetry]") {
    LpProblem lp;
    lp.name = "symmetry";
    lp.sense = Sense::Minimize;
    lp.num_cols = 2;
    lp.obj = {1.0, 1.0};
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {1.0, 1.0};
    lp.col_type = {VarType::Binary, VarType::Binary};
    lp.col_names = {"x", "y"};

    lp.num_rows = 1;
    lp.row_lower = {-kInf};
    lp.row_upper = {1.0};
    lp.row_names = {"sum"};
    std::vector<Triplet> trips = {
        {0, 0, 1.0}, {0, 1, 1.0},
    };
    lp.matrix = SparseMatrix(1, 2, std::move(trips));

    SymmetryManager manager;
    manager.detect(lp);
    REQUIRE(manager.hasSymmetry());
    REQUIRE(!manager.orbitalFixes().empty());

    LpProblem working = lp;
    const Index root_rows = working.num_rows;
    const Index cuts = manager.addSymmetryCuts(working);
    CHECK(cuts == static_cast<Index>(manager.orbitalFixes().size()));
    CHECK(working.num_rows == root_rows + cuts);

    const Index sym_row = working.num_rows - 1;
    CHECK(working.row_lower[sym_row] == -kInf);
    CHECK(working.row_upper[sym_row] == 0.0);
    CHECK(working.matrix.coeff(sym_row, 0) == -1.0);
    CHECK(working.matrix.coeff(sym_row, 1) == 1.0);
    CHECK(manager.detectWorkUnits() > 0.0);
    CHECK(manager.cutWorkUnits() > 0.0);
}

TEST_CASE("SymmetryManager orbit and cut order is deterministic", "[symmetry]") {
    LpProblem lp;
    lp.name = "symmetry_order";
    lp.sense = Sense::Minimize;
    lp.num_cols = 4;
    lp.obj = {1.0, 1.0, 2.0, 2.0};
    lp.col_lower = {0.0, 0.0, 0.0, 0.0};
    lp.col_upper = {1.0, 1.0, 1.0, 1.0};
    lp.col_type = {VarType::Binary, VarType::Binary, VarType::Binary, VarType::Binary};
    lp.col_names = {"x0", "x1", "x2", "x3"};

    lp.num_rows = 2;
    lp.row_lower = {-kInf, -kInf};
    lp.row_upper = {1.0, 1.0};
    lp.row_names = {"r0", "r1"};
    std::vector<Triplet> trips = {
        {0, 0, 1.0}, {0, 1, 1.0},
        {1, 2, 1.0}, {1, 3, 1.0},
    };
    lp.matrix = SparseMatrix(2, 4, std::move(trips));

    SymmetryManager manager;
    manager.detect(lp);
    REQUIRE(manager.hasSymmetry());
    REQUIRE(manager.orbits().size() == 2);
    CHECK(manager.orbits()[0].size() == 2);
    CHECK(manager.orbits()[0][0] == 0);
    CHECK(manager.orbits()[0][1] == 1);
    CHECK(manager.orbits()[1].size() == 2);
    CHECK(manager.orbits()[1][0] == 2);
    CHECK(manager.orbits()[1][1] == 3);

    LpProblem working = lp;
    working.row_names = {"r0"};
    const Index cuts = manager.addSymmetryCuts(working);
    CHECK(cuts == 2);
    CHECK(working.row_names.size() == static_cast<std::size_t>(working.num_rows));
    CHECK(working.row_names[2] == "sym_cut_0_1");
    CHECK(working.row_names[3] == "sym_cut_2_3");
    CHECK(manager.cutWorkUnits() == 4.0);
}

TEST_CASE("Symmetry detection enforces canonical branching", "[branching][symmetry]") {
    LpProblem lp;
    lp.name = "symmetry";
    lp.sense = Sense::Minimize;
    lp.num_cols = 2;
    lp.obj = {1.0, 1.0};
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {1.0, 1.0};
    lp.col_type = {VarType::Binary, VarType::Binary};
    lp.col_names = {"x", "y"};

    lp.num_rows = 1;
    lp.row_lower = {-kInf};
    lp.row_upper = {1.0};
    lp.row_names = {"sum"};
    std::vector<Triplet> trips = {
        {0, 0, 1.0}, {0, 1, 1.0},
    };
    lp.matrix = SparseMatrix(1, 2, std::move(trips));

    SymmetryManager manager;
    manager.detect(lp);
    REQUIRE(manager.hasSymmetry());
    CHECK(manager.canonical(0) == 0);
    CHECK(manager.canonical(1) == 0);

    ReliabilityBranching branching;
    branching.reset(lp.num_cols);
    branching.setSymmetryManager(&manager);
    branching.setReliabilityThreshold(1);
    branching.updatePseudoCost(0, false, 1.0);
    branching.updatePseudoCost(0, true, 1.0);
    DualSimplexSolver solver;
    solver.load(lp);
    solver.setVerbose(false);
    solver.solve();

    std::vector<Real> primal = {0.5, 0.5};
    std::vector<Real> lb = lp.col_lower;
    std::vector<Real> ub = lp.col_upper;
    BranchingTelemetry telemetry;
    auto selection = branching.select(solver,
                                      lp,
                                      primal,
                                      lb,
                                      ub,
                                      0.0,
                                      false,
                                      telemetry);
    CHECK(selection.variable == 0);
}
