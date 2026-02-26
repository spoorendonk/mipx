#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <memory>
#include <thread>

#include "mipx/branching.h"
#include "mipx/dual_simplex.h"
#include "mipx/heuristics.h"
#include "mipx/heuristic_runtime.h"
#include "mipx/lp_problem.h"

using namespace mipx;
using Catch::Matchers::WithinAbs;

// ---------------------------------------------------------------------------
// Helper: check that a solution is feasible for the given problem.
// ---------------------------------------------------------------------------

static bool isFeasible(const LpProblem& problem,
                       std::span<const Real> sol,
                       Real tol = 1e-6) {
    // Check variable bounds.
    for (Index j = 0; j < problem.num_cols; ++j) {
        if (sol[j] < problem.col_lower[j] - tol) return false;
        if (sol[j] > problem.col_upper[j] + tol) return false;
    }
    // Check integrality.
    for (Index j = 0; j < problem.num_cols; ++j) {
        if (problem.col_type[j] == VarType::Continuous) continue;
        if (!isIntegral(sol[j], tol)) return false;
    }
    // Check constraints.
    for (Index i = 0; i < problem.num_rows; ++i) {
        auto row = problem.matrix.row(i);
        Real activity = 0.0;
        for (Index k = 0; k < row.size(); ++k) {
            activity += row.values[k] * sol[row.indices[k]];
        }
        if (activity < problem.row_lower[i] - tol) return false;
        if (activity > problem.row_upper[i] + tol) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Helper: build a MIP where rounding the LP solution gives a feasible MIP.
// min -x - 2y  s.t. x + y <= 4, x,y >= 0, x,y integer
// LP optimal at (0, 4), already integral => rounding works.
// ---------------------------------------------------------------------------

static LpProblem buildEasyRoundingMip() {
    LpProblem lp;
    lp.name = "easy_rounding";
    lp.sense = Sense::Minimize;
    lp.num_cols = 2;
    lp.obj = {-1.0, -2.0};
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {kInf, kInf};
    lp.col_type = {VarType::Integer, VarType::Integer};
    lp.col_names = {"x", "y"};

    lp.num_rows = 3;
    lp.row_lower = {-kInf, -kInf, -kInf};
    lp.row_upper = {4.0, 3.0, 3.0};
    lp.row_names = {"sum", "ub_x", "ub_y"};

    std::vector<Triplet> trips = {
        {0, 0, 1.0}, {0, 1, 1.0},
        {1, 0, 1.0},
        {2, 1, 1.0},
    };
    lp.matrix = SparseMatrix(3, 2, std::move(trips));
    return lp;
}

// ---------------------------------------------------------------------------
// Helper: build a MIP where rounding the LP solution is infeasible.
// min -x - 2y  s.t. x + y <= 4.5, x + 2y >= 8, x,y >= 0, x,y integer
// LP relaxation: fractional, rounding likely violates one constraint.
// ---------------------------------------------------------------------------

static LpProblem buildHardRoundingMip() {
    LpProblem lp;
    lp.name = "hard_rounding";
    lp.sense = Sense::Minimize;
    lp.num_cols = 2;
    lp.obj = {-1.0, -2.0};
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {kInf, kInf};
    lp.col_type = {VarType::Integer, VarType::Integer};
    lp.col_names = {"x", "y"};

    // x + y <= 4.5  AND  x + 2y >= 8
    // Feasible integers: (0,4): 0+4=4<=4.5, 0+8=8>=8 => yes
    //                    (1,4): 1+4=5 > 4.5 => no
    // LP relaxation will have fractional solution.
    lp.num_rows = 2;
    lp.row_lower = {-kInf, 8.0};
    lp.row_upper = {4.5, kInf};
    lp.row_names = {"ub", "lb"};

    std::vector<Triplet> trips = {
        {0, 0, 1.0}, {0, 1, 1.0},
        {1, 0, 1.0}, {1, 1, 2.0},
    };
    lp.matrix = SparseMatrix(2, 2, std::move(trips));
    return lp;
}

// ---------------------------------------------------------------------------
// Helper: binary knapsack
// max 6x1 + 5x2 + 4x3  s.t. 3x1 + 2x2 + 2x3 <= 5, x binary
// = min -6x1 -5x2 -4x3
// Optimal: x1=1, x2=1, x3=0, obj=-11
// LP relaxation likely has fractional x3.
// ---------------------------------------------------------------------------

static LpProblem buildKnapsack() {
    LpProblem lp;
    lp.name = "knapsack";
    lp.sense = Sense::Minimize;
    lp.num_cols = 3;
    lp.obj = {-6.0, -5.0, -4.0};
    lp.col_lower = {0.0, 0.0, 0.0};
    lp.col_upper = {1.0, 1.0, 1.0};
    lp.col_type = {VarType::Binary, VarType::Binary, VarType::Binary};
    lp.col_names = {"x1", "x2", "x3"};

    lp.num_rows = 1;
    lp.row_lower = {-kInf};
    lp.row_upper = {5.0};
    lp.row_names = {"capacity"};

    std::vector<Triplet> trips = {
        {0, 0, 3.0}, {0, 1, 2.0}, {0, 2, 2.0},
    };
    lp.matrix = SparseMatrix(1, 3, std::move(trips));
    return lp;
}

// ---------------------------------------------------------------------------
// Helper: single-integer problem where LP optimum is fractional.
// min -x  s.t. x <= 4.6, x >= 0, x integer
// LP optimum x=4.6; integer optimum x=4.
// ---------------------------------------------------------------------------

static LpProblem buildFractionalSingleIntMip() {
    LpProblem lp;
    lp.name = "single_int_frac";
    lp.sense = Sense::Minimize;
    lp.num_cols = 1;
    lp.obj = {-1.0};
    lp.col_lower = {0.0};
    lp.col_upper = {10.0};
    lp.col_type = {VarType::Integer};
    lp.col_names = {"x"};

    lp.num_rows = 1;
    lp.row_lower = {-kInf};
    lp.row_upper = {4.6};
    lp.row_names = {"ub_x"};

    std::vector<Triplet> trips = {
        {0, 0, 1.0},
    };
    lp.matrix = SparseMatrix(1, 1, std::move(trips));
    return lp;
}

// ===========================================================================
// Rounding tests
// ===========================================================================

TEST_CASE("RoundingHeuristic: finds feasible on easy MIP", "[heuristics]") {
    auto problem = buildEasyRoundingMip();

    DualSimplexSolver lp;
    lp.load(problem);
    auto lr = lp.solve();
    REQUIRE(lr.status == Status::Optimal);

    auto primals = lp.getPrimalValues();

    RoundingHeuristic rounding;
    auto result = rounding.run(problem, lp, primals, kInf);

    // The LP solution is already integral, so rounding should find it.
    REQUIRE(result.has_value());
    CHECK(isFeasible(problem, result->values));
    CHECK_THAT(result->objective, WithinAbs(-7.0, 1e-6));
}

TEST_CASE("RoundingHeuristic: solution is actually feasible", "[heuristics]") {
    auto problem = buildKnapsack();

    DualSimplexSolver lp;
    lp.load(problem);
    auto lr = lp.solve();
    REQUIRE(lr.status == Status::Optimal);

    auto primals = lp.getPrimalValues();

    RoundingHeuristic rounding;
    auto result = rounding.run(problem, lp, primals, kInf);

    // Whether rounding succeeds or not, if it returns a solution it must be feasible.
    if (result.has_value()) {
        CHECK(isFeasible(problem, result->values));
    }
}

TEST_CASE("RoundingHeuristic: respects incumbent cutoff", "[heuristics]") {
    auto problem = buildEasyRoundingMip();

    DualSimplexSolver lp;
    lp.load(problem);
    lp.solve();
    auto primals = lp.getPrimalValues();

    RoundingHeuristic rounding;
    // Set incumbent to something better than what rounding can find.
    auto result = rounding.run(problem, lp, primals, -100.0);

    // Should not return a solution worse than incumbent.
    CHECK(!result.has_value());
}

// ===========================================================================
// Diving tests
// ===========================================================================

TEST_CASE("FractionalDiving: finds feasible on knapsack", "[heuristics]") {
    auto problem = buildKnapsack();

    DualSimplexSolver lp;
    lp.load(problem);
    auto lr = lp.solve();
    REQUIRE(lr.status == Status::Optimal);

    auto primals = lp.getPrimalValues();

    FractionalDiving diving;
    diving.setMaxDives(50);
    auto result = diving.run(problem, lp, primals, kInf);

    if (result.has_value()) {
        CHECK(isFeasible(problem, result->values));
        // Must be at least as good as -11 (optimal).
        CHECK(result->objective <= -9.0);  // At least a decent solution.
    }
}

TEST_CASE("CoefficientDiving: finds feasible on knapsack", "[heuristics]") {
    auto problem = buildKnapsack();

    DualSimplexSolver lp;
    lp.load(problem);
    auto lr = lp.solve();
    REQUIRE(lr.status == Status::Optimal);

    auto primals = lp.getPrimalValues();

    CoefficientDiving diving;
    diving.setMaxDives(50);
    auto result = diving.run(problem, lp, primals, kInf);

    if (result.has_value()) {
        CHECK(isFeasible(problem, result->values));
        CHECK(result->objective <= -9.0);
    }
}

TEST_CASE("FractionalDiving: LP state restored after dive", "[heuristics]") {
    auto problem = buildKnapsack();

    DualSimplexSolver lp;
    lp.load(problem);
    auto lr = lp.solve();
    REQUIRE(lr.status == Status::Optimal);
    Real obj_before = lr.objective;

    auto primals = lp.getPrimalValues();

    FractionalDiving diving;
    diving.setMaxDives(20);
    diving.run(problem, lp, primals, kInf);

    // After diving, LP should be restored to its original state.
    // Re-solve to verify.
    auto lr2 = lp.solve();
    REQUIRE(lr2.status == Status::Optimal);
    CHECK_THAT(lr2.objective, WithinAbs(obj_before, 1e-6));
}

TEST_CASE("Diving: respects backtrack limit", "[heuristics]") {
    // Build a problem where diving will hit infeasibility.
    auto problem = buildHardRoundingMip();

    DualSimplexSolver lp;
    lp.load(problem);
    auto lr = lp.solve();
    // The LP may or may not be feasible depending on the constraints.
    if (lr.status != Status::Optimal) {
        SKIP("LP relaxation infeasible");
    }

    auto primals = lp.getPrimalValues();

    FractionalDiving diving;
    diving.setMaxDives(50);
    diving.setBacktrackLimit(2);
    // Just verify it doesn't crash/hang.
    auto result = diving.run(problem, lp, primals, kInf);

    if (result.has_value()) {
        CHECK(isFeasible(problem, result->values));
    }
}

// ===========================================================================
// RINS tests
// ===========================================================================

TEST_CASE("RinsHeuristic: requires incumbent solution", "[heuristics]") {
    auto problem = buildEasyRoundingMip();

    DualSimplexSolver lp;
    lp.load(problem);
    auto lr = lp.solve();
    REQUIRE(lr.status == Status::Optimal);

    auto primals = lp.getPrimalValues();

    RinsHeuristic rins;
    auto result = rins.run(problem, lp, primals, kInf);

    CHECK(!result.has_value());
    CHECK(rins.lastSkippedNoIncumbent());
    CHECK_FALSE(rins.lastExecutedSolve());
}

TEST_CASE("RinsHeuristic: uses incumbent-agreement fixing", "[heuristics]") {
    auto problem = buildEasyRoundingMip();

    DualSimplexSolver lp;
    lp.load(problem);
    auto lr = lp.solve();
    REQUIRE(lr.status == Status::Optimal);

    auto primals = lp.getPrimalValues();

    RinsHeuristic rins;
    rins.setSubproblemIterLimit(100);
    rins.setMinFixedVars(1);
    rins.setMinFixedRate(0.0);

    std::vector<Real> disagree_incumbent = {0.0, 0.0};
    auto no_fix = rins.run(problem, lp, primals, 100.0, disagree_incumbent);
    CHECK(!no_fix.has_value());

    std::vector<Real> agree_incumbent = primals;
    auto result = rins.run(problem, lp, primals, 100.0, agree_incumbent);
    REQUIRE(result.has_value());
    CHECK(isFeasible(problem, result->values));
    CHECK_THAT(result->objective, WithinAbs(-7.0, 1e-6));
}

TEST_CASE("RinsHeuristic: restores node-specific bounds after run", "[heuristics]") {
    auto problem = buildEasyRoundingMip();

    DualSimplexSolver lp;
    lp.load(problem);
    lp.setColBounds(0, 0.0, 0.0);  // simulate node-local bound
    auto node_lr = lp.solve();
    REQUIRE(node_lr.status == Status::Optimal);
    Real node_obj_before = node_lr.objective;

    auto primals = lp.getPrimalValues();
    auto incumbent_values = primals;

    RinsHeuristic rins;
    rins.setSubproblemIterLimit(200);
    rins.setMinFixedVars(1);
    rins.setMinFixedRate(0.0);
    auto result = rins.run(problem, lp, primals, 100.0, incumbent_values);
    REQUIRE(result.has_value());
    CHECK(isFeasible(problem, result->values));

    Real lb = -kInf;
    Real ub = kInf;
    lp.getColBounds(0, lb, ub);
    CHECK_THAT(lb, WithinAbs(0.0, 1e-9));
    CHECK_THAT(ub, WithinAbs(0.0, 1e-9));

    auto lr2 = lp.solve();
    REQUIRE(lr2.status == Status::Optimal);
    CHECK_THAT(lr2.objective, WithinAbs(node_obj_before, 1e-6));
}

// ===========================================================================
// RENS + Feasibility Pump tests
// ===========================================================================

TEST_CASE("RensHeuristic: finds feasible on easy MIP", "[heuristics]") {
    auto problem = buildEasyRoundingMip();

    DualSimplexSolver lp;
    lp.load(problem);
    auto lr = lp.solve();
    REQUIRE(lr.status == Status::Optimal);

    auto primals = lp.getPrimalValues();

    RensHeuristic rens;
    rens.setSubproblemIterLimit(100);
    rens.setFixTol(1.0);
    rens.setMinFixedVars(1);
    rens.setMinFixedRate(0.0);
    auto result = rens.run(problem, lp, primals, kInf);

    REQUIRE(result.has_value());
    CHECK(isFeasible(problem, result->values));
}

TEST_CASE("RensHeuristic: skips when too few variables can be fixed", "[heuristics]") {
    auto problem = buildHardRoundingMip();

    DualSimplexSolver lp;
    lp.load(problem);
    auto lr = lp.solve();
    if (lr.status != Status::Optimal) {
        SKIP("LP relaxation infeasible");
    }

    auto primals = lp.getPrimalValues();
    RensHeuristic rens;
    rens.setFixTol(1e-8);
    rens.setMinFixedVars(2);
    rens.setMinFixedRate(0.0);
    auto result = rens.run(problem, lp, primals, kInf);
    CHECK(!result.has_value());
    CHECK(rens.lastSkippedFewFixes());
}

TEST_CASE("FeasibilityPumpHeuristic: can produce feasible integer point", "[heuristics]") {
    auto problem = buildFractionalSingleIntMip();

    DualSimplexSolver lp;
    lp.load(problem);
    auto lr = lp.solve();
    REQUIRE(lr.status == Status::Optimal);
    auto primals = lp.getPrimalValues();

    FeasibilityPumpHeuristic fp;
    fp.setMaxIterations(8);
    fp.setSubproblemIterLimit(60);
    auto result = fp.run(problem, lp, primals, kInf);

    REQUIRE(result.has_value());
    CHECK(isFeasible(problem, result->values));
}

TEST_CASE("FeasibilityPumpHeuristic: LP state restored after run", "[heuristics]") {
    auto problem = buildFractionalSingleIntMip();

    DualSimplexSolver lp;
    lp.load(problem);
    auto lr = lp.solve();
    REQUIRE(lr.status == Status::Optimal);
    Real obj_before = lr.objective;
    auto primals = lp.getPrimalValues();

    FeasibilityPumpHeuristic fp;
    fp.setMaxIterations(6);
    fp.setSubproblemIterLimit(40);
    fp.run(problem, lp, primals, kInf);

    auto lr2 = lp.solve();
    REQUIRE(lr2.status == Status::Optimal);
    CHECK_THAT(lr2.objective, WithinAbs(obj_before, 1e-6));
}

TEST_CASE("AuxObjectiveHeuristic: can produce feasible integer point", "[heuristics]") {
    auto problem = buildFractionalSingleIntMip();

    DualSimplexSolver lp;
    lp.load(problem);
    auto lr = lp.solve();
    REQUIRE(lr.status == Status::Optimal);
    auto primals = lp.getPrimalValues();

    AuxObjectiveHeuristic aux;
    aux.setSubproblemIterLimit(40);
    std::vector<Real> incumbent_values = {4.0};
    auto result = aux.run(problem, lp, primals, 100.0, incumbent_values);

    REQUIRE(result.has_value());
    CHECK(isFeasible(problem, result->values));
}

TEST_CASE("AuxObjectiveHeuristic: LP state restored after run", "[heuristics]") {
    auto problem = buildFractionalSingleIntMip();

    DualSimplexSolver lp;
    lp.load(problem);
    auto lr = lp.solve();
    REQUIRE(lr.status == Status::Optimal);
    Real obj_before = lr.objective;
    auto primals = lp.getPrimalValues();

    AuxObjectiveHeuristic aux;
    aux.setSubproblemIterLimit(40);
    (void)aux.run(problem, lp, primals, kInf);

    auto lr2 = lp.solve();
    REQUIRE(lr2.status == Status::Optimal);
    CHECK_THAT(lr2.objective, WithinAbs(obj_before, 1e-6));
}

TEST_CASE("ZeroObjectiveHeuristic: can produce feasible integer point", "[heuristics]") {
    auto problem = buildKnapsack();

    DualSimplexSolver lp;
    lp.load(problem);
    auto lr = lp.solve();
    REQUIRE(lr.status == Status::Optimal);
    auto primals = lp.getPrimalValues();

    ZeroObjectiveHeuristic zeroobj;
    zeroobj.setSubproblemIterLimit(30);
    auto result = zeroobj.run(problem, lp, primals, 100.0);

    REQUIRE(result.has_value());
    CHECK(isFeasible(problem, result->values));
}

TEST_CASE("ZeroObjectiveHeuristic: LP state restored after run", "[heuristics]") {
    auto problem = buildKnapsack();

    DualSimplexSolver lp;
    lp.load(problem);
    auto lr = lp.solve();
    REQUIRE(lr.status == Status::Optimal);
    Real obj_before = lr.objective;
    auto primals = lp.getPrimalValues();

    ZeroObjectiveHeuristic zeroobj;
    zeroobj.setSubproblemIterLimit(30);
    (void)zeroobj.run(problem, lp, primals, 100.0);

    auto lr2 = lp.solve();
    REQUIRE(lr2.status == Status::Optimal);
    CHECK_THAT(lr2.objective, WithinAbs(obj_before, 1e-6));
}

TEST_CASE("LocalBranchingHeuristic: improves incumbent in binary neighborhood", "[heuristics]") {
    auto problem = buildKnapsack();

    DualSimplexSolver lp;
    lp.load(problem);
    auto lr = lp.solve();
    REQUIRE(lr.status == Status::Optimal);
    auto primals = lp.getPrimalValues();

    LocalBranchingHeuristic lb;
    lb.setSubproblemIterLimit(100);
    lb.setNeighborhoodSize(1);
    lb.setMinBinaryVars(1);
    std::vector<Real> incumbent_values = {1.0, 0.0, 0.0};
    auto result = lb.run(problem, lp, primals, -6.0, incumbent_values);

    REQUIRE(result.has_value());
    CHECK(isFeasible(problem, result->values));
    CHECK(result->objective < -6.0);
}

TEST_CASE("LocalBranchingHeuristic: restores LP state after run", "[heuristics]") {
    auto problem = buildKnapsack();

    DualSimplexSolver lp;
    lp.load(problem);
    auto lr = lp.solve();
    REQUIRE(lr.status == Status::Optimal);
    Real obj_before = lr.objective;
    auto primals = lp.getPrimalValues();

    LocalBranchingHeuristic lb;
    lb.setSubproblemIterLimit(50);
    lb.setNeighborhoodSize(1);
    lb.setMinBinaryVars(1);
    std::vector<Real> incumbent_values = {1.0, 0.0, 0.0};
    (void)lb.run(problem, lp, primals, -6.0, incumbent_values);

    auto lr2 = lp.solve();
    REQUIRE(lr2.status == Status::Optimal);
    CHECK_THAT(lr2.objective, WithinAbs(obj_before, 1e-6));
}

TEST_CASE("LocalBranchingHeuristic: larger neighborhoods can unlock improvements", "[heuristics]") {
    auto problem = buildKnapsack();

    DualSimplexSolver lp;
    lp.load(problem);
    auto lr = lp.solve();
    REQUIRE(lr.status == Status::Optimal);
    auto primals = lp.getPrimalValues();

    std::vector<Real> incumbent_values = {1.0, 0.0, 1.0};
    const Real incumbent_obj = -10.0;

    LocalBranchingHeuristic lb_small;
    lb_small.setSubproblemIterLimit(80);
    lb_small.setNeighborhoodSize(1);
    lb_small.setMinBinaryVars(1);
    auto small = lb_small.run(problem, lp, primals, incumbent_obj, incumbent_values);
    CHECK_FALSE(lb_small.lastSkippedNoIncumbent());

    LocalBranchingHeuristic lb_large;
    lb_large.setSubproblemIterLimit(80);
    lb_large.setNeighborhoodSize(2);
    lb_large.setMinBinaryVars(1);
    auto large = lb_large.run(problem, lp, primals, incumbent_obj, incumbent_values);
    CHECK(lb_large.lastExecutedSolve());
    if (large.has_value()) {
        CHECK(isFeasible(problem, large->values));
        CHECK(large->objective < incumbent_obj);
    } else if (small.has_value()) {
        CHECK(small->objective <= incumbent_obj + 1e-6);
    }
}

TEST_CASE("HeuristicBudgetManager: frequency grows on misses", "[heuristics]") {
    HeuristicBudgetManager budget;
    budget.setBaseTreeFrequency(10);
    budget.setMaxFrequencyScale(8);

    CHECK(budget.currentTreeFrequency() == 10);

    budget.recordHeuristicCall(0, 1.0, false);
    CHECK(budget.currentTreeFrequency() == 20);

    budget.recordHeuristicCall(20, 1.0, false);
    CHECK(budget.currentTreeFrequency() >= 20);

    budget.recordHeuristicCall(40, 1.0, true);
    CHECK(budget.currentTreeFrequency() == 20);
}

TEST_CASE("HeuristicBudgetManager: work-share gate and node gate", "[heuristics]") {
    HeuristicBudgetManager budget;
    budget.setBaseTreeFrequency(5);
    budget.setMaxWorkShare(0.10);

    CHECK(budget.allowRootHeuristic(0.0));
    CHECK(budget.allowTreeHeuristic(0, 100.0));

    budget.recordHeuristicCall(0, 6.0, false);
    CHECK_FALSE(budget.allowRootHeuristic(50.0));
    CHECK(budget.allowRootHeuristic(100.0));

    CHECK_FALSE(budget.allowTreeHeuristic(9, 100.0));
    CHECK(budget.allowTreeHeuristic(10, 100.0));
}

// ===========================================================================
// HeuristicScheduler tests
// ===========================================================================

TEST_CASE("HeuristicScheduler: triggers at correct points", "[heuristics]") {
    auto problem = buildEasyRoundingMip();

    DualSimplexSolver lp;
    lp.load(problem);
    lp.solve();
    auto primals = lp.getPrimalValues();

    HeuristicScheduler scheduler;

    // Add rounding at root only.
    scheduler.addHeuristic(std::make_unique<RoundingHeuristic>(),
                           HeuristicTiming::Root);

    CHECK(scheduler.numHeuristics() == 1);

    // Should run at node 0 (root).
    auto r0 = scheduler.run(problem, lp, primals, kInf, 0);
    CHECK(r0.has_value());

    // Should NOT run at node 5.
    auto r5 = scheduler.run(problem, lp, primals, kInf, 5);
    CHECK(!r5.has_value());
}

TEST_CASE("HeuristicScheduler: periodic timing", "[heuristics]") {
    auto problem = buildEasyRoundingMip();

    DualSimplexSolver lp;
    lp.load(problem);
    lp.solve();
    auto primals = lp.getPrimalValues();

    HeuristicScheduler scheduler;
    scheduler.addHeuristic(std::make_unique<RoundingHeuristic>(),
                           HeuristicTiming::Periodic, 5);

    // Should run at: 0, 5, 10, 15, ...
    auto r0 = scheduler.run(problem, lp, primals, kInf, 0);
    CHECK(r0.has_value());

    auto r3 = scheduler.run(problem, lp, primals, kInf, 3);
    CHECK(!r3.has_value());

    auto r5 = scheduler.run(problem, lp, primals, kInf, 5);
    CHECK(r5.has_value());
}

TEST_CASE("HeuristicScheduler: every-node timing", "[heuristics]") {
    auto problem = buildEasyRoundingMip();

    DualSimplexSolver lp;
    lp.load(problem);
    lp.solve();
    auto primals = lp.getPrimalValues();

    HeuristicScheduler scheduler;
    scheduler.addHeuristic(std::make_unique<RoundingHeuristic>(),
                           HeuristicTiming::EveryNode);

    // Should run at every node.
    for (Int n = 0; n < 5; ++n) {
        auto r = scheduler.run(problem, lp, primals, kInf, n);
        CHECK(r.has_value());
    }

    CHECK(scheduler.totalSolutionsFound() == 5);
}

TEST_CASE("HeuristicScheduler: multiple heuristics, best solution wins", "[heuristics]") {
    auto problem = buildKnapsack();

    DualSimplexSolver lp;
    lp.load(problem);
    lp.solve();
    auto primals = lp.getPrimalValues();

    HeuristicScheduler scheduler;
    scheduler.addHeuristic(std::make_unique<RoundingHeuristic>(),
                           HeuristicTiming::Root);
    scheduler.addHeuristic(std::make_unique<FractionalDiving>(),
                           HeuristicTiming::Root);

    CHECK(scheduler.numHeuristics() == 2);

    auto result = scheduler.run(problem, lp, primals, kInf, 0);

    if (result.has_value()) {
        CHECK(isFeasible(problem, result->values));
    }
}

TEST_CASE("SolutionPool: keeps best incumbent", "[heuristics][runtime]") {
    SolutionPool pool(Sense::Minimize);
    CHECK_FALSE(pool.hasSolution());

    CHECK(pool.submit({{1.0}, 5.0}, "h1", 0));
    CHECK_FALSE(pool.submit({{2.0}, 6.0}, "h2", 1));
    CHECK(pool.submit({{3.0}, 4.0}, "h3", 2));

    REQUIRE(pool.hasSolution());
    auto best = pool.bestSolution();
    REQUIRE(best.has_value());
    CHECK_THAT(best->objective, WithinAbs(4.0, 1e-9));
    REQUIRE(best->values.size() == 1);
    CHECK_THAT(best->values[0], WithinAbs(3.0, 1e-9));
}

TEST_CASE("SolutionPool: concurrent submits preserve best solution", "[heuristics][runtime]") {
    SolutionPool pool(Sense::Minimize);
    constexpr Int kThreads = 8;
    constexpr Int kPerThread = 128;
    std::vector<std::thread> workers;
    workers.reserve(kThreads);

    for (Int t = 0; t < kThreads; ++t) {
        workers.emplace_back([&pool, t]() {
            for (Int i = 0; i < kPerThread; ++i) {
                const Real obj = 10'000.0 - static_cast<Real>(t * kPerThread + i);
                pool.submit({{obj}, obj}, "stress", t);
            }
        });
    }
    for (auto& w : workers) w.join();

    const Real expected = 10'000.0 -
                          static_cast<Real>((kThreads - 1) * kPerThread + (kPerThread - 1));
    auto best = pool.bestSolution();
    REQUIRE(best.has_value());
    CHECK_THAT(best->objective, WithinAbs(expected, 1e-9));
    REQUIRE(best->values.size() == 1);
    CHECK_THAT(best->values[0], WithinAbs(expected, 1e-9));
}
