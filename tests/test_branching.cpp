#include "mipx/branching.h"
#include "mipx/dual_simplex.h"
#include "mipx/symmetry.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

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
    REQUIRE(left.bound_changes[0].level == left.depth);

    // Right child: x_0 >= 3 (lower bound).
    REQUIRE(right.parent_id == 5);
    REQUIRE(right.depth == 4);
    REQUIRE(right.branch.variable == 0);
    REQUIRE(right.branch.bound == 3.0);
    REQUIRE(right.branch.is_upper == false);
    REQUIRE(right.basis.size() == 2);
    REQUIRE(right.bound_changes.size() == 1);
    REQUIRE(right.bound_changes[0].level == right.depth);
}

TEST_CASE("createChildren accumulates bound changes", "[branching]") {
    BnbNode parent;
    parent.id = 2;
    parent.depth = 1;
    parent.bound_changes = {{0, 1.0, false, parent.depth}};

    auto [left, right] = createChildren(parent, 1, 3.5);

    REQUIRE(left.bound_changes.size() == 2);
    REQUIRE(left.bound_changes[0].variable == 0);
    REQUIRE(left.bound_changes[0].level == 1);
    REQUIRE(left.bound_changes[1].variable == 1);
    REQUIRE(left.bound_changes[1].level == 2);

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

TEST_CASE("ReliabilityBranching gives sibling probes the same reliability view", "[branching]") {
    LpProblem lp;
    lp.num_cols = 3;
    lp.obj = {0.0, 0.0, 1.0};
    lp.col_lower = {0.0, 0.0, 0.0};
    lp.col_upper = {1.0, 1.0, kInf};
    lp.col_type = {VarType::Binary, VarType::Binary, VarType::Continuous};
    lp.num_rows = 4;
    // Both siblings have x = 0.5 at optimum. Fixing x costs 3 when z = 0,
    // and 1 when z = 1 because the steeper rows are then slack.
    lp.row_lower = {1.0, -1.0, 3.0, -3.0};
    lp.row_upper.assign(4, kInf);
    lp.matrix = SparseMatrix(4, 3,
                             std::vector<Triplet>{{0, 0, 2.0},
                                                  {0, 2, 1.0},
                                                  {1, 0, -2.0},
                                                  {1, 2, 1.0},
                                                  {2, 0, 6.0},
                                                  {2, 1, 3.0},
                                                  {2, 2, 1.0},
                                                  {3, 0, -6.0},
                                                  {3, 1, 3.0},
                                                  {3, 2, 1.0}});

    ReliabilityBranching branching;
    branching.reset(lp.num_cols);
    for (Int n = 0; n < 3; ++n) {
        branching.updatePseudoCost(0, false, 1.0);
        branching.updatePseudoCost(0, true, 1.0);
    }
    REQUIRE_FALSE(branching.isReliable(0));

    std::vector<StrongBranchPseudocostSample> samples;
    for (const Real fixed : {0.0, 1.0}) {
        auto sibling = lp;
        sibling.col_lower[1] = sibling.col_upper[1] = fixed;
        DualSimplexSolver solver;
        solver.setVerbose(false);
        solver.load(sibling);
        const auto result = solver.solve();
        REQUIRE(result.status == Status::Optimal);
        const auto primals = solver.getPrimalValues();
        REQUIRE_THAT(primals[0], Catch::Matchers::WithinAbs(0.5, 1e-12));
        BranchingTelemetry telemetry;
        const auto selection =
            branching.selectDeferred(solver, sibling, primals, sibling.col_lower, sibling.col_upper,
                                     result.objective, 2, telemetry, samples);
        CHECK(selection.variable == 0);
        CHECK(telemetry.strong_branch_probes == 2);
        CHECK(telemetry.pseudocost_updates_strong == 2);
        CHECK(telemetry.pseudocost_updates_tree == 0);
        CHECK_FALSE(branching.isReliable(0));
        CHECK(branching.upReliability(0) == 3);
        CHECK(branching.downReliability(0) == 3);
    }
    REQUIRE(samples.size() == 4);
    for (const auto& sample : samples) {
        REQUIRE(sample.variable == 0);
        branching.updatePseudoCost(sample.variable, sample.up_direction, sample.gain_per_unit);
    }
    CHECK(branching.isReliable(0));
    CHECK(branching.upReliability(0) == 5);
    CHECK(branching.downReliability(0) == 5);
    CHECK_THAT(branching.upPseudoCost(0), Catch::Matchers::WithinAbs(2.2, 1e-12));
    CHECK_THAT(branching.downPseudoCost(0), Catch::Matchers::WithinAbs(2.2, 1e-12));
}

TEST_CASE("Balanced branch scoring prefers gains in both pseudocost directions", "[branching]") {
    LpProblem lp;
    lp.num_cols = 2;
    lp.col_type = {VarType::Binary, VarType::Binary};
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {1.0, 1.0};
    const std::vector<Real> primals = {0.5, 0.5};
    DualSimplexSolver solver;
    ReliabilityBranching branching;
    branching.reset(lp.num_cols);
    branching.setReliabilityThreshold(1);
    branching.updatePseudoCost(0, false, 2.0);
    branching.updatePseudoCost(0, true, 28.0);
    branching.updatePseudoCost(1, false, 8.0);
    branching.updatePseudoCost(1, true, 8.0);

    BranchingTelemetry telemetry;
    CHECK(branching.select(solver, lp, primals, lp.col_lower, lp.col_upper, 0.0, false, telemetry)
              .variable == 0);
    branching.setBalancedScore(true);
    CHECK(branching.select(solver, lp, primals, lp.col_lower, lp.col_upper, 0.0, false, telemetry)
              .variable == 1);
    CHECK(telemetry.pseudocost_hits == 2);
    CHECK(telemetry.strong_branch_probes == 0);
}

TEST_CASE("Balanced branch scoring prefers gains in both strong probes", "[branching]") {
    LpProblem lp;
    lp.sense = Sense::Minimize;
    lp.num_cols = 4;
    lp.obj = {0.0, 0.0, 1.0, 1.0};
    lp.col_type = {VarType::Binary, VarType::Binary, VarType::Continuous, VarType::Continuous};
    lp.col_lower = {0.0, 0.0, 0.0, 0.0};
    lp.col_upper = {1.0, 1.0, kInf, kInf};
    lp.num_rows = 4;
    // The two epigraph variables drive x and y to 0.5 at the root.
    // Fixing x costs 1 down and 14 up. Fixing y costs 4 in either direction.
    lp.row_lower = {1.0, -14.0, 4.0, -4.0};
    lp.row_upper.assign(4, kInf);
    lp.matrix = SparseMatrix(4, 4,
                             std::vector<Triplet>{{0, 0, 2.0},
                                                  {0, 2, 1.0},
                                                  {1, 0, -28.0},
                                                  {1, 2, 1.0},
                                                  {2, 1, 8.0},
                                                  {2, 3, 1.0},
                                                  {3, 1, -8.0},
                                                  {3, 3, 1.0}});

    DualSimplexSolver solver;
    solver.setVerbose(false);
    solver.load(lp);
    const auto root = solver.solve();
    REQUIRE(root.status == Status::Optimal);
    const auto primals = solver.getPrimalValues();
    REQUIRE_THAT(primals[0], Catch::Matchers::WithinAbs(0.5, 1e-8));
    REQUIRE_THAT(primals[1], Catch::Matchers::WithinAbs(0.5, 1e-8));

    ReliabilityBranching branching;
    branching.reset(lp.num_cols);
    BranchingTelemetry telemetry;
    CHECK(branching
              .select(solver, lp, primals, lp.col_lower, lp.col_upper, root.objective, true,
                      telemetry)
              .variable == 0);
    branching.reset(lp.num_cols);
    branching.setBalancedScore(true);
    CHECK(branching
              .select(solver, lp, primals, lp.col_lower, lp.col_upper, root.objective, true,
                      telemetry)
              .variable == 1);
    CHECK(telemetry.strong_branch_probes == 8);
}

TEST_CASE("Balanced branch scoring retains a strong down gain when the up probe stops",
          "[branching][time_limit]") {
    class InterruptedProbes : public DualSimplexSolver {
    public:
        LpResult solve() override { return results_.at(next_++); }

    private:
        std::size_t next_ = 0;
        const std::vector<LpResult> results_ = {{Status::Optimal, 10.0},
                                                {Status::Optimal, 10.0},
                                                {Status::Infeasible},
                                                {Status::TimeLimit}};
    };

    LpProblem lp;
    lp.num_cols = 2;
    lp.obj = {0.0, 0.0};
    lp.col_type = {VarType::Binary, VarType::Binary};
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {1.0, 1.0};
    lp.num_rows = 1;
    lp.row_lower = {-kInf};
    lp.row_upper = {1.0};
    lp.matrix = SparseMatrix(1, 2, std::vector<Triplet>{{0, 0, 1.0}, {0, 1, 1.0}});

    InterruptedProbes solver;
    solver.setVerbose(false);
    solver.load(lp);
    REQUIRE(solver.DualSimplexSolver::solve().status == Status::Optimal);

    ReliabilityBranching branching;
    branching.reset(lp.num_cols);
    branching.setBalancedScore(true);
    BranchingTelemetry telemetry;
    const std::vector<Real> primals = {0.5, 0.5};
    CHECK(branching.select(solver, lp, primals, lp.col_lower, lp.col_upper, 0.0, true, telemetry)
              .variable == 1);
    CHECK(telemetry.strong_branch_probes == 4);
    CHECK(branching.downReliability(1) == 1);
    CHECK(branching.upReliability(1) == 0);
}

TEST_CASE("ReliabilityBranching learns pseudocosts from solved children", "[branching]") {
    // Issue #199: a child's LP gain over its parent is a pseudocost sample.
    // No strong branching happens here at all, so any change in the
    // pseudocosts comes from the tree update alone.
    ReliabilityBranching branching;
    branching.reset(3);
    BranchingTelemetry telemetry;

    BnbNode parent;
    parent.id = 7;
    parent.lp_bound = 10.0;
    auto [down, up] = createChildren(std::move(parent), 1, 2.25);
    REQUIRE(down.branch.is_upper);
    CHECK(down.parent_objective == 10.0);
    CHECK(down.branch_value == 2.25);

    // Down child: x1 <= 2, distance 0.25, gain 0.5 -> 2 per unit.
    CHECK(branching.updateFromChild(TreePseudocostSample::fromChild(down, Status::Optimal, 10.5),
                                    telemetry));
    // Up child: x1 >= 3, distance 0.75, gain 3 -> 4 per unit.
    CHECK(branching.updateFromChild(TreePseudocostSample::fromChild(up, Status::Optimal, 13.0),
                                    telemetry));

    CHECK(branching.downReliability(1) == 1);
    CHECK(branching.upReliability(1) == 1);
    CHECK_THAT(branching.downPseudoCost(1), Catch::Matchers::WithinAbs(2.0, 1e-12));
    CHECK_THAT(branching.upPseudoCost(1), Catch::Matchers::WithinAbs(4.0, 1e-12));
    CHECK(telemetry.pseudocost_updates_tree == 2);
    CHECK(telemetry.pseudocost_updates_strong == 0);
    CHECK(telemetry.pseudocost_tree_skipped == 0);
}

TEST_CASE("ReliabilityBranching records nothing from children without a gain", "[branching]") {
    ReliabilityBranching branching;
    branching.reset(3);
    BranchingTelemetry telemetry;

    BnbNode parent;
    parent.lp_bound = 10.0;
    auto [down, up] = createChildren(std::move(parent), 1, 2.25);

    // An infeasible child and one whose LP stopped early carry no gain.
    CHECK_FALSE(branching.updateFromChild(
        TreePseudocostSample::fromChild(down, Status::Infeasible, 0.0), telemetry));
    CHECK_FALSE(branching.updateFromChild(
        TreePseudocostSample::fromChild(up, Status::IterLimit, 11.0), telemetry));
    // A child the incumbent cuts off measures the cutoff, not the branching.
    CHECK_FALSE(branching.updateFromChild(
        TreePseudocostSample::fromChild(up, Status::Optimal, 50.0, true), telemetry));
    // The root has no parent and no branching to credit.
    CHECK_FALSE(branching.updateFromChild(
        TreePseudocostSample::fromChild(BnbNode{}, Status::Optimal, 11.0), telemetry));

    CHECK(branching.downReliability(1) == 0);
    CHECK(branching.upReliability(1) == 0);
    CHECK(telemetry.pseudocost_updates_tree == 0);
    CHECK(telemetry.pseudocost_tree_skipped == 3);
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
        {0, 0, 1.0},
        {0, 1, 1.0},
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
        {0, 0, 1.0},
        {0, 1, 1.0},
        {1, 2, 1.0},
        {1, 3, 1.0},
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
        {0, 0, 1.0},
        {0, 1, 1.0},
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
    auto selection = branching.select(solver, lp, primal, lb, ub, 0.0, false, telemetry);
    CHECK(selection.variable == 0);
}

// #351: once the MIP's deadline has passed, the first strong-branching probe
// stops on TimeLimit; the rule probes no more, learns no pseudocost from the
// cut-short probe, and still picks a variable, by pseudocosts.
TEST_CASE("ReliabilityBranching stops strong branching at a passed deadline",
          "[branching][time_limit]") {
    LpProblem lp;
    lp.name = "deadline";
    lp.sense = Sense::Minimize;
    lp.num_cols = 2;
    lp.obj = {-1.0, -1.0};
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {1.0, 1.0};
    lp.col_type = {VarType::Binary, VarType::Binary};
    lp.col_names = {"x", "y"};
    lp.num_rows = 1;
    lp.row_lower = {-kInf};
    lp.row_upper = {1.0};
    lp.row_names = {"sum"};
    lp.matrix = SparseMatrix(1, 2, std::vector<Triplet>{{0, 0, 1.0}, {0, 1, 1.0}});

    DualSimplexSolver solver;
    solver.setVerbose(false);
    solver.load(lp);
    REQUIRE(solver.solve().status == Status::Optimal);
    auto opts = solver.getOptions();
    opts.deadline = Deadline::after(0.0);
    solver.setOptions(opts);

    ReliabilityBranching branching;
    branching.reset(lp.num_cols);
    std::vector<Real> primal = {0.5, 0.5};
    std::vector<Real> lb = lp.col_lower;
    std::vector<Real> ub = lp.col_upper;
    BranchingTelemetry telemetry;
    const auto selection = branching.select(solver, lp, primal, lb, ub, -1.0, false, telemetry);

    CHECK(telemetry.strong_branch_calls == 1);
    CHECK(telemetry.strong_branch_probes == 1);
    CHECK(telemetry.pseudocost_updates_strong == 0);
    CHECK(branching.downReliability(0) == 0);
    CHECK(branching.upReliability(0) == 0);
    CHECK(selection.variable >= 0);
    CHECK(selection.used_pseudocost);
}

TEST_CASE("pseudocostNodeEstimate matches the hand-computed sum", "[branching][search]") {
    LpProblem problem;
    problem.num_cols = 3;
    problem.col_type = {VarType::Integer, VarType::Binary, VarType::Continuous};
    problem.col_lower = {0.0, 0.0, 0.0};
    problem.col_upper = {10.0, 1.0, 10.0};

    ReliabilityBranching rule;
    rule.reset(problem.num_cols);
    rule.setPseudocostFallback(3.0);
    rule.updatePseudoCost(0, /*up_direction=*/false, 2.0);  // down cost of x0 = 2
    rule.updatePseudoCost(1, /*up_direction=*/true, 4.0);   // up cost of x1 = 4

    // x0 = 1.25: min(0.25 * 2.0, 0.75 * 3.0) = 0.5   (up unobserved -> fallback)
    // x1 = 0.50: min(0.50 * 3.0, 0.50 * 4.0) = 1.5   (down unobserved -> fallback)
    // x2 continuous -> skipped.
    const std::vector<Real> primals = {1.25, 0.5, 0.75};
    const Real estimate = pseudocostNodeEstimate(rule, problem, primals, 10.0);
    CHECK_THAT(estimate, Catch::Matchers::WithinAbs(12.0, 1e-12));
}

TEST_CASE("pseudocostNodeEstimate is the LP bound without fractional variables",
          "[branching][search]") {
    LpProblem problem;
    problem.num_cols = 2;
    problem.col_type = {VarType::Integer, VarType::Continuous};
    problem.col_lower = {0.0, 0.0};
    problem.col_upper = {10.0, 10.0};

    ReliabilityBranching rule;
    rule.reset(problem.num_cols);

    const std::vector<Real> primals = {3.0, 1.5};
    CHECK_THAT(pseudocostNodeEstimate(rule, problem, primals, -7.5),
               Catch::Matchers::WithinAbs(-7.5, 1e-12));

    // A non-finite LP bound is passed through untouched (no NaN arithmetic).
    const std::vector<Real> fractional = {3.5, 1.5};
    CHECK(pseudocostNodeEstimate(rule, problem, fractional, -kInf) == -kInf);
}
