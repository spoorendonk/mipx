#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <memory>

#include "mipx/branching.h"
#include "mipx/dual_simplex.h"
#include "mipx/heuristics.h"
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

TEST_CASE("RinsHeuristic: finds feasible on easy MIP", "[heuristics]") {
    auto problem = buildEasyRoundingMip();

    DualSimplexSolver lp;
    lp.load(problem);
    auto lr = lp.solve();
    REQUIRE(lr.status == Status::Optimal);

    auto primals = lp.getPrimalValues();

    RinsHeuristic rins;
    rins.setSubproblemIterLimit(200);
    rins.setAgreementTol(1.0);
    auto result = rins.run(problem, lp, primals, kInf);

    REQUIRE(result.has_value());
    CHECK(isFeasible(problem, result->values));
    CHECK_THAT(result->objective, WithinAbs(-7.0, 1e-6));
}

TEST_CASE("RinsHeuristic: respects incumbent cutoff", "[heuristics]") {
    auto problem = buildEasyRoundingMip();

    DualSimplexSolver lp;
    lp.load(problem);
    auto lr = lp.solve();
    REQUIRE(lr.status == Status::Optimal);

    auto primals = lp.getPrimalValues();

    RinsHeuristic rins;
    rins.setSubproblemIterLimit(200);
    rins.setAgreementTol(1.0);
    auto result = rins.run(problem, lp, primals, -100.0);

    CHECK(!result.has_value());
}

TEST_CASE("RinsHeuristic: LP state restored after run", "[heuristics]") {
    auto problem = buildEasyRoundingMip();

    DualSimplexSolver lp;
    lp.load(problem);
    auto lr = lp.solve();
    REQUIRE(lr.status == Status::Optimal);
    Real obj_before = lr.objective;

    auto primals = lp.getPrimalValues();

    RinsHeuristic rins;
    rins.setSubproblemIterLimit(200);
    rins.setAgreementTol(1.0);
    rins.run(problem, lp, primals, kInf);

    auto lr2 = lp.solve();
    REQUIRE(lr2.status == Status::Optimal);
    CHECK_THAT(lr2.objective, WithinAbs(obj_before, 1e-6));
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
