#include "mipx/dual_simplex.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>

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
        {0, 0, 1.0},
        {0, 1, 1.0},
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

// ---------------------------------------------------------------------------
// Test: removeRows that cannot preserve the basis, then addRows and re-solve
// ---------------------------------------------------------------------------

// min -x - 2y  s.t.  x + y <= 4,  x <= 3,  y <= 3,  x + 2y <= 9,  x, y >= 0
// Optimal: x = 1, y = 3, obj = -7.
static LpProblem buildLP_2var4row() {
    LpProblem lp;
    lp.name = "twovars4row";
    lp.sense = Sense::Minimize;
    lp.num_cols = 2;
    lp.obj = {-1.0, -2.0};
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {kInf, kInf};
    lp.col_type = {VarType::Continuous, VarType::Continuous};
    lp.col_names = {"x", "y"};

    lp.num_rows = 4;
    lp.row_lower = {-kInf, -kInf, -kInf, -kInf};
    lp.row_upper = {4.0, 3.0, 3.0, 9.0};
    lp.row_names = {"sum", "ub_x", "ub_y", "mix"};

    std::vector<Triplet> trips = {
        {0, 0, 1.0}, {0, 1, 1.0}, {1, 0, 1.0}, {2, 1, 1.0}, {3, 0, 1.0}, {3, 1, 2.0},
    };
    lp.matrix = SparseMatrix(4, 2, std::move(trips));
    return lp;
}

TEST_CASE("Incremental: removeRows without basis preservation leaves no stale state",
          "[incremental]") {
    // removeRows can only remap the basis when every kept row's basic variable
    // survives. Below, a kept row's basic variable is the logical of a row that
    // is being removed, so it cannot, and the fallback has to bring every
    // per-variable vector down to the new size. Leaving them at the old size is
    // not harmless: addRows runs before the next solve(), resizes them back
    // into range and re-asserts has_basis_, and solve() then walks a nonbasic_
    // still naming variables of rows that no longer exist -- reading, and
    // writing, past the end of the current ones. Found through the in-tree cut
    // loop once Gomory started feeding it (issue #211).
    auto lp = buildLP_2var4row();
    DualSimplexSolver solver;
    solver.load(lp);
    REQUIRE(solver.solve().status == Status::Optimal);

    // Variables: 0, 1 structural; 2..5 the logicals of rows 0..3. setBasis
    // hands out basis positions in index order, so this puts the logical of
    // row 0 -- which is removed below -- at the position of kept row 1.
    std::vector<BasisStatus> basis = {
        BasisStatus::AtLower,  // x
        BasisStatus::Basic,    // y
        BasisStatus::Basic,    // logical of row 0  <- removed
        BasisStatus::Basic,    // logical of row 1
        BasisStatus::Basic,    // logical of row 2
        BasisStatus::AtUpper,  // logical of row 3  <- removed
    };
    solver.setBasis(basis);

    const std::vector<Index> remove = {0, 3};
    solver.removeRows(remove);
    REQUIRE(solver.numRows() == 2);

    // A cut arrives before the next solve. This is what resizes the vectors the
    // fallback left oversized, putting stale variable indices back in range.
    const std::vector<Index> starts = {0};
    const std::vector<Index> indices = {0, 1};
    const std::vector<Real> values = {1.0, 1.0};
    const std::vector<Real> row_lower = {-kInf};
    const std::vector<Real> row_upper = {2.0};
    solver.addRows(starts, indices, values, row_lower, row_upper);
    REQUIRE(solver.numRows() == 3);

    // What is left is x <= 3, y <= 3 and x + y <= 2, so min -x - 2y is at
    // y = 2, x = 0 with objective -4.
    auto result = solver.solve();
    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(-4.0, 1e-6));

    auto primals = solver.getPrimalValues();
    REQUIRE(primals.size() == 2);
    CHECK_THAT(primals[1], WithinAbs(2.0, 1e-6));

    // The basis status vector must describe the LP as it stands now.
    CHECK(static_cast<Index>(solver.getBasis().size()) == solver.numCols() + solver.numRows());
}
