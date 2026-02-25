#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>

#include "mipx/dual_simplex.h"

using namespace mipx;
using Catch::Matchers::WithinAbs;

// ---------------------------------------------------------------------------
// Helper: build LP for testing
// min c^T x  s.t.  Ax <= b,  x >= 0
// ---------------------------------------------------------------------------

static LpProblem buildLP_2var() {
    // min -x - 2y  s.t.  x + y <= 4,  x <= 3,  y <= 3,  x,y >= 0
    // Optimal: x=1, y=3, obj=-7
    LpProblem lp;
    lp.name = "twovars";
    lp.sense = Sense::Minimize;
    lp.num_cols = 2;
    lp.obj = {-1.0, -2.0};
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {kInf, kInf};
    lp.col_type = {VarType::Continuous, VarType::Continuous};
    lp.col_names = {"x", "y"};

    lp.num_rows = 3;
    lp.row_lower = {-kInf, -kInf, -kInf};
    lp.row_upper = {4.0, 3.0, 3.0};
    lp.row_names = {"sum", "ub_x", "ub_y"};

    // Row 0: x + y <= 4
    // Row 1: x <= 3
    // Row 2: y <= 3
    std::vector<Triplet> trips = {
        {0, 0, 1.0}, {0, 1, 1.0},
        {1, 0, 1.0},
        {2, 1, 1.0},
    };
    lp.matrix = SparseMatrix(3, 2, std::move(trips));
    return lp;
}

// ---------------------------------------------------------------------------
// Test: setColBounds — tighten bound and re-solve
// ---------------------------------------------------------------------------
TEST_CASE("Incremental: setColBounds tighten and re-solve", "[incremental]") {
    auto lp = buildLP_2var();
    DualSimplexSolver solver;
    solver.load(lp);

    // First solve: optimal x=1, y=3, obj=-7
    auto r1 = solver.solve();
    REQUIRE(r1.status == Status::Optimal);
    CHECK_THAT(r1.objective, WithinAbs(-7.0, 1e-6));

    // Tighten y <= 2 (via column bound).
    solver.setColBounds(1, 0.0, 2.0);
    auto r2 = solver.solve();
    REQUIRE(r2.status == Status::Optimal);
    // Now y=2, x=2, obj = -1*2 + -2*2 = -6
    CHECK_THAT(r2.objective, WithinAbs(-6.0, 1e-6));

    // Warm-start should use fewer iterations than cold start.
    // (Not a hard requirement but should generally hold.)
}

// ---------------------------------------------------------------------------
// Test: setColBounds — fix variable
// ---------------------------------------------------------------------------
TEST_CASE("Incremental: setColBounds fix variable", "[incremental]") {
    auto lp = buildLP_2var();
    DualSimplexSolver solver;
    solver.load(lp);

    auto r1 = solver.solve();
    REQUIRE(r1.status == Status::Optimal);

    // Fix x = 0
    solver.setColBounds(0, 0.0, 0.0);
    auto r2 = solver.solve();
    REQUIRE(r2.status == Status::Optimal);
    // y = 3 (limited by y <= 3 and x + y <= 4), obj = -2*3 = -6
    CHECK_THAT(r2.objective, WithinAbs(-6.0, 1e-6));
    auto primals = solver.getPrimalValues();
    CHECK_THAT(primals[0], WithinAbs(0.0, 1e-6));
    CHECK_THAT(primals[1], WithinAbs(3.0, 1e-6));
}

// ---------------------------------------------------------------------------
// Test: setColBounds — make infeasible
// ---------------------------------------------------------------------------
TEST_CASE("Incremental: setColBounds makes LP infeasible", "[incremental]") {
    auto lp = buildLP_2var();
    DualSimplexSolver solver;
    solver.load(lp);

    auto r1 = solver.solve();
    REQUIRE(r1.status == Status::Optimal);

    // Force x >= 5, but x + y <= 4 and y >= 0 => x <= 4. Contradiction.
    solver.setColBounds(0, 5.0, kInf);
    auto r2 = solver.solve();
    CHECK(r2.status == Status::Infeasible);
}

// ---------------------------------------------------------------------------
// Test: setObjective — change objective and re-solve
// ---------------------------------------------------------------------------
TEST_CASE("Incremental: setObjective and re-solve", "[incremental]") {
    auto lp = buildLP_2var();
    DualSimplexSolver solver;
    solver.load(lp);

    auto r1 = solver.solve();
    REQUIRE(r1.status == Status::Optimal);

    // Change objective: min -2x - y (swap weights)
    std::vector<Real> new_obj = {-2.0, -1.0};
    solver.setObjective(new_obj);
    auto r2 = solver.solve();
    REQUIRE(r2.status == Status::Optimal);
    // Now x=3, y=1, obj = -2*3 + -1*1 = -7
    CHECK_THAT(r2.objective, WithinAbs(-7.0, 1e-6));
    auto primals = solver.getPrimalValues();
    CHECK_THAT(primals[0], WithinAbs(3.0, 1e-6));
    CHECK_THAT(primals[1], WithinAbs(1.0, 1e-6));
}

// ---------------------------------------------------------------------------
// Test: addRows — add a constraint and re-solve
// ---------------------------------------------------------------------------
TEST_CASE("Incremental: addRows and re-solve", "[incremental]") {
    auto lp = buildLP_2var();
    DualSimplexSolver solver;
    solver.load(lp);

    auto r1 = solver.solve();
    REQUIRE(r1.status == Status::Optimal);
    CHECK_THAT(r1.objective, WithinAbs(-7.0, 1e-6));

    // Add constraint: x + 2y <= 5
    // This makes the old optimal (x=1, y=3) infeasible (1 + 6 = 7 > 5).
    // New optimal should be on the intersection of x+y<=4, x+2y<=5.
    // Solving: y = 1, x = 3, obj = -1*3 + -2*1 = -5
    // Or: from x+y=4, x+2y=5: y=1, x=3. Check: x<=3 ok, y<=3 ok.
    std::vector<Index> starts = {0};
    std::vector<Index> indices = {0, 1};
    std::vector<Real> values = {1.0, 2.0};
    std::vector<Real> lower = {-kInf};
    std::vector<Real> upper = {5.0};

    solver.addRows(starts, indices, values, lower, upper);
    auto r2 = solver.solve();
    REQUIRE(r2.status == Status::Optimal);
    CHECK_THAT(r2.objective, WithinAbs(-5.0, 1e-6));
}

// ---------------------------------------------------------------------------
// Test: addRows — add infeasible constraint
// ---------------------------------------------------------------------------
TEST_CASE("Incremental: addRows makes LP infeasible", "[incremental]") {
    auto lp = buildLP_2var();
    DualSimplexSolver solver;
    solver.load(lp);

    auto r1 = solver.solve();
    REQUIRE(r1.status == Status::Optimal);

    // Add constraint: x + y >= 10 (but x+y <= 4 already)
    std::vector<Index> starts = {0};
    std::vector<Index> indices = {0, 1};
    std::vector<Real> values = {1.0, 1.0};
    std::vector<Real> lower = {10.0};
    std::vector<Real> upper = {kInf};

    solver.addRows(starts, indices, values, lower, upper);
    auto r2 = solver.solve();
    CHECK(r2.status == Status::Infeasible);
}

// ---------------------------------------------------------------------------
// Test: setBasis — warm-start with saved basis
// ---------------------------------------------------------------------------
TEST_CASE("Incremental: setBasis warm-start", "[incremental]") {
    auto lp = buildLP_2var();
    DualSimplexSolver solver;
    solver.load(lp);

    auto r1 = solver.solve();
    REQUIRE(r1.status == Status::Optimal);

    // Save basis.
    auto basis = solver.getBasis();

    // Create a fresh solver, load same problem, set basis, solve.
    DualSimplexSolver solver2;
    solver2.load(lp);
    solver2.setBasis(basis);
    auto r2 = solver2.solve();
    REQUIRE(r2.status == Status::Optimal);
    CHECK_THAT(r2.objective, WithinAbs(-7.0, 1e-6));
    // Should solve in 0 iterations (already optimal).
    CHECK(r2.iterations <= 1);
}

// ---------------------------------------------------------------------------
// Test: Multiple modifications — bound change + re-solve multiple times
// ---------------------------------------------------------------------------
TEST_CASE("Incremental: multiple bound changes", "[incremental]") {
    auto lp = buildLP_2var();
    DualSimplexSolver solver;
    solver.load(lp);

    auto r1 = solver.solve();
    REQUIRE(r1.status == Status::Optimal);
    CHECK_THAT(r1.objective, WithinAbs(-7.0, 1e-6));

    // Fix x = 2
    solver.setColBounds(0, 2.0, 2.0);
    auto r2 = solver.solve();
    REQUIRE(r2.status == Status::Optimal);
    // y = min(3, 4-2) = 2, obj = -1*2 + -2*2 = -6
    CHECK_THAT(r2.objective, WithinAbs(-6.0, 1e-6));

    // Now fix x = 3
    solver.setColBounds(0, 3.0, 3.0);
    auto r3 = solver.solve();
    REQUIRE(r3.status == Status::Optimal);
    // y = min(3, 4-3) = 1, obj = -1*3 + -2*1 = -5
    CHECK_THAT(r3.objective, WithinAbs(-5.0, 1e-6));

    // Release x back to [0, inf)
    solver.setColBounds(0, 0.0, kInf);
    auto r4 = solver.solve();
    REQUIRE(r4.status == Status::Optimal);
    CHECK_THAT(r4.objective, WithinAbs(-7.0, 1e-6));
}
