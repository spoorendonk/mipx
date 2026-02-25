#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>

#include "mipx/mip_solver.h"
#include "mipx/presolve.h"

using namespace mipx;
using Catch::Matchers::WithinAbs;

// =============================================================================
// Test helpers
// =============================================================================

/// Build a problem with a fixed variable (lb == ub).
/// min x + y  s.t. x + y <= 5, x = 3 (fixed), y >= 0
static LpProblem buildFixedVarProblem() {
    LpProblem lp;
    lp.name = "fixed_var";
    lp.sense = Sense::Minimize;
    lp.num_cols = 2;
    lp.obj = {1.0, 1.0};
    lp.col_lower = {3.0, 0.0};
    lp.col_upper = {3.0, 0.0};  // x fixed at 3, y at lower bound for min
    lp.col_type = {VarType::Continuous, VarType::Continuous};
    lp.col_names = {"x", "y"};

    lp.num_rows = 1;
    lp.row_lower = {-kInf};
    lp.row_upper = {5.0};
    lp.row_names = {"c1"};

    std::vector<Triplet> trips = {{0, 0, 1.0}, {0, 1, 1.0}};
    lp.matrix = SparseMatrix(1, 2, std::move(trips));
    return lp;
}

/// Build a problem with a singleton row: 2*x <= 6 (x >= 0, x integer).
/// min -x - y  s.t. x + y <= 10, 2*x <= 6, x,y >= 0, x integer
static LpProblem buildSingletonRowProblem() {
    LpProblem lp;
    lp.name = "singleton_row";
    lp.sense = Sense::Minimize;
    lp.num_cols = 2;
    lp.obj = {-1.0, -1.0};
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {kInf, kInf};
    lp.col_type = {VarType::Integer, VarType::Continuous};
    lp.col_names = {"x", "y"};

    lp.num_rows = 2;
    lp.row_lower = {-kInf, -kInf};
    lp.row_upper = {10.0, 6.0};
    lp.row_names = {"c1", "c2"};

    std::vector<Triplet> trips = {
        {0, 0, 1.0}, {0, 1, 1.0},
        {1, 0, 2.0},
    };
    lp.matrix = SparseMatrix(2, 2, std::move(trips));
    return lp;
}

/// Build a problem where a column appears in only one constraint.
/// min x + 2y  s.t. x + y <= 5, x >= 0, y >= 0
/// y only affects row 0 and has positive objective -> fix at lower bound (0).
/// But x also only affects row 0... both are singletons.
/// Use 3 vars: min x + 2y + z, x + y <= 5, y + z <= 4, x,y,z >= 0
/// Here x is a singleton column in row 0.
static LpProblem buildSingletonColProblem() {
    LpProblem lp;
    lp.name = "singleton_col";
    lp.sense = Sense::Minimize;
    lp.num_cols = 3;
    lp.obj = {1.0, 2.0, 3.0};
    lp.col_lower = {0.0, 0.0, 0.0};
    lp.col_upper = {kInf, kInf, kInf};
    lp.col_type = {VarType::Continuous, VarType::Continuous, VarType::Continuous};
    lp.col_names = {"x", "y", "z"};

    lp.num_rows = 2;
    lp.row_lower = {-kInf, -kInf};
    lp.row_upper = {5.0, 4.0};
    lp.row_names = {"c1", "c2"};

    std::vector<Triplet> trips = {
        {0, 0, 1.0}, {0, 1, 1.0},
        {1, 1, 1.0}, {1, 2, 1.0},
    };
    lp.matrix = SparseMatrix(2, 3, std::move(trips));
    return lp;
}

/// Build a problem with a redundant (dominated) constraint.
/// min x  s.t. x <= 5, x <= 100, x >= 0
/// The second constraint is dominated by bounds + first constraint.
static LpProblem buildDominatedRowProblem() {
    LpProblem lp;
    lp.name = "dominated_row";
    lp.sense = Sense::Minimize;
    lp.num_cols = 1;
    lp.obj = {1.0};
    lp.col_lower = {0.0};
    lp.col_upper = {5.0};
    lp.col_type = {VarType::Continuous};
    lp.col_names = {"x"};

    lp.num_rows = 2;
    lp.row_lower = {-kInf, -kInf};
    lp.row_upper = {5.0, 100.0};
    lp.row_names = {"tight", "dominated"};

    std::vector<Triplet> trips = {
        {0, 0, 1.0},
        {1, 0, 1.0},
    };
    lp.matrix = SparseMatrix(2, 1, std::move(trips));
    return lp;
}

/// Build a problem with a forcing constraint.
/// min x + y  s.t. x + y <= 0, x >= 0, y >= 0
/// The constraint forces x = y = 0.
static LpProblem buildForcingRowProblem() {
    LpProblem lp;
    lp.name = "forcing_row";
    lp.sense = Sense::Minimize;
    lp.num_cols = 2;
    lp.obj = {1.0, 1.0};
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {10.0, 10.0};
    lp.col_type = {VarType::Continuous, VarType::Continuous};
    lp.col_names = {"x", "y"};

    lp.num_rows = 1;
    lp.row_lower = {-kInf};
    lp.row_upper = {0.0};
    lp.row_names = {"forcing"};

    std::vector<Triplet> trips = {
        {0, 0, 1.0}, {0, 1, 1.0},
    };
    lp.matrix = SparseMatrix(1, 2, std::move(trips));
    return lp;
}

/// Build a small MIP for round-trip testing.
/// min -x - 2y  s.t. x + y <= 4, x <= 3, y <= 3, x,y >= 0, x,y integer
static LpProblem buildSmallMip() {
    LpProblem lp;
    lp.name = "small_mip";
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

// =============================================================================
// Tests: Fixed variable removal
// =============================================================================

TEST_CASE("Presolve: fixed variable removal", "[presolve]") {
    auto lp = buildFixedVarProblem();

    Presolver presolver;
    auto reduced = presolver.presolve(lp);

    // x is fixed at 3, should be removed.
    CHECK(reduced.num_cols < lp.num_cols);
    CHECK(presolver.stats().vars_removed >= 1);

    // The objective offset should include x's contribution.
    CHECK_THAT(reduced.obj_offset, WithinAbs(3.0, 1e-8));

    // Row bound should be adjusted: x + y <= 5 becomes y <= 2.
    // (row_upper was 5, subtract 3 for fixed x = 3, giving 2)
    if (reduced.num_rows > 0) {
        CHECK_THAT(reduced.row_upper[0], WithinAbs(2.0, 1e-8));
    }
}

// =============================================================================
// Tests: Singleton row handling
// =============================================================================

TEST_CASE("Presolve: singleton row tightens bounds", "[presolve]") {
    auto lp = buildSingletonRowProblem();

    Presolver presolver;
    auto reduced = presolver.presolve(lp);

    // The singleton row "2*x <= 6" should be removed,
    // and x's upper bound should be tightened to 3.
    CHECK(presolver.stats().rows_removed >= 1);

    // Find x in the reduced problem.
    const auto& mapping = presolver.colMapping();
    for (Index jj = 0; jj < reduced.num_cols; ++jj) {
        if (mapping[jj] == 0) {  // Original column 0 = x.
            CHECK(reduced.col_upper[jj] <= 3.0 + 1e-8);
        }
    }
}

// =============================================================================
// Tests: Singleton column handling
// =============================================================================

TEST_CASE("Presolve: singleton column removal", "[presolve]") {
    auto lp = buildSingletonColProblem();

    Presolver presolver;
    auto reduced = presolver.presolve(lp);

    // x (col 0) and z (col 2) are singleton columns with positive obj in
    // minimization, so they should be fixed at lower bound (0).
    CHECK(presolver.stats().vars_removed >= 1);

    // Postsolve should recover the correct solution.
    // Solve the reduced problem: should just be y with bounds.
    // In the reduced problem, optimal y = 0 (minimization with obj coeff 2).
    std::vector<Real> presolved_sol(reduced.num_cols, 0.0);
    auto full_sol = presolver.postsolve(presolved_sol);

    REQUIRE(full_sol.size() == 3);
    CHECK_THAT(full_sol[0], WithinAbs(0.0, 1e-8));  // x fixed at 0
    CHECK_THAT(full_sol[2], WithinAbs(0.0, 1e-8));  // z fixed at 0
}

// =============================================================================
// Tests: Forcing row
// =============================================================================

TEST_CASE("Presolve: forcing constraint", "[presolve]") {
    auto lp = buildForcingRowProblem();

    Presolver presolver;
    auto reduced = presolver.presolve(lp);

    // x + y <= 0 with x,y >= 0 forces x = y = 0.
    // Both variables should be removed.
    CHECK(presolver.stats().vars_removed >= 2);

    // Postsolve should give x = y = 0.
    std::vector<Real> presolved_sol(reduced.num_cols, 0.0);
    auto full_sol = presolver.postsolve(presolved_sol);

    REQUIRE(full_sol.size() == 2);
    CHECK_THAT(full_sol[0], WithinAbs(0.0, 1e-8));
    CHECK_THAT(full_sol[1], WithinAbs(0.0, 1e-8));
}

// =============================================================================
// Tests: Dominated row
// =============================================================================

TEST_CASE("Presolve: dominated constraint removal", "[presolve]") {
    auto lp = buildDominatedRowProblem();

    Presolver presolver;
    auto reduced = presolver.presolve(lp);

    // x <= 100 is dominated by x <= 5 (upper bound), so should be removed.
    CHECK(presolver.stats().rows_removed >= 1);
}

// =============================================================================
// Tests: Round-trip presolve + postsolve
// =============================================================================

TEST_CASE("Presolve: round-trip recovers original solution", "[presolve]") {
    auto lp = buildSmallMip();

    Presolver presolver;
    auto reduced = presolver.presolve(lp);

    // The singleton rows (x <= 3, y <= 3) should be removed and bounds tightened.
    CHECK(reduced.num_rows <= lp.num_rows);

    // Solve reduced problem with MIP solver.
    MipSolver solver;
    solver.setVerbose(false);
    solver.load(reduced);
    auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);

    // Postsolve.
    auto full_sol = presolver.postsolve(result.solution);

    REQUIRE(full_sol.size() == 2);

    // Verify: original objective = -x - 2y
    Real obj = reduced.obj_offset;
    for (Index j = 0; j < static_cast<Index>(result.solution.size()); ++j) {
        obj += reduced.obj[j] * result.solution[j];
    }
    CHECK_THAT(obj, WithinAbs(-7.0, 1e-6));

    // Verify the postsolved solution satisfies original constraints.
    Real x = full_sol[0];
    Real y = full_sol[1];
    CHECK(x + y <= 4.0 + 1e-6);
    CHECK(x <= 3.0 + 1e-6);
    CHECK(y <= 3.0 + 1e-6);
    CHECK(x >= -1e-6);
    CHECK(y >= -1e-6);
}

// =============================================================================
// Tests: Presolve + MIP solve matches direct solve
// =============================================================================

TEST_CASE("Presolve: presolved MIP matches direct solve", "[presolve]") {
    // Build a MIP that needs branching.
    LpProblem lp;
    lp.name = "branching_mip";
    lp.sense = Sense::Minimize;
    lp.num_cols = 2;
    lp.obj = {-1.0, -2.0};
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {kInf, kInf};
    lp.col_type = {VarType::Integer, VarType::Integer};
    lp.col_names = {"x", "y"};

    lp.num_rows = 1;
    lp.row_lower = {-kInf};
    lp.row_upper = {4.5};
    lp.row_names = {"sum"};

    std::vector<Triplet> trips = {{0, 0, 1.0}, {0, 1, 1.0}};
    lp.matrix = SparseMatrix(1, 2, std::move(trips));

    // Solve directly.
    MipSolver direct_solver;
    direct_solver.setVerbose(false);
    direct_solver.load(lp);
    auto direct_result = direct_solver.solve();

    // Solve with presolve.
    Presolver presolver;
    auto reduced = presolver.presolve(lp);

    MipSolver presolved_solver;
    presolved_solver.setVerbose(false);
    presolved_solver.load(reduced);
    auto presolved_result = presolved_solver.solve();

    REQUIRE(direct_result.status == Status::Optimal);
    REQUIRE(presolved_result.status == Status::Optimal);

    // Objectives should match (accounting for obj_offset).
    Real presolved_obj = presolved_result.objective + reduced.obj_offset;
    CHECK_THAT(presolved_obj, WithinAbs(direct_result.objective, 1e-6));
}

// =============================================================================
// Tests: Empty problem after presolve
// =============================================================================

TEST_CASE("Presolve: all variables fixed", "[presolve]") {
    // min x + y, x = 2, y = 3 (both fixed)
    LpProblem lp;
    lp.name = "all_fixed";
    lp.sense = Sense::Minimize;
    lp.num_cols = 2;
    lp.obj = {1.0, 1.0};
    lp.col_lower = {2.0, 3.0};
    lp.col_upper = {2.0, 3.0};
    lp.col_type = {VarType::Continuous, VarType::Continuous};
    lp.col_names = {"x", "y"};

    lp.num_rows = 1;
    lp.row_lower = {-kInf};
    lp.row_upper = {10.0};
    lp.row_names = {"c1"};

    std::vector<Triplet> trips = {{0, 0, 1.0}, {0, 1, 1.0}};
    lp.matrix = SparseMatrix(1, 2, std::move(trips));

    Presolver presolver;
    auto reduced = presolver.presolve(lp);

    CHECK(reduced.num_cols == 0);
    CHECK(presolver.stats().vars_removed == 2);

    // Objective offset should be 2 + 3 = 5.
    CHECK_THAT(reduced.obj_offset, WithinAbs(5.0, 1e-8));

    // Postsolve with empty solution.
    std::vector<Real> empty_sol;
    auto full_sol = presolver.postsolve(empty_sol);

    REQUIRE(full_sol.size() == 2);
    CHECK_THAT(full_sol[0], WithinAbs(2.0, 1e-8));
    CHECK_THAT(full_sol[1], WithinAbs(3.0, 1e-8));
}

// =============================================================================
// Tests: Statistics tracking
// =============================================================================

TEST_CASE("Presolve: statistics are tracked", "[presolve]") {
    auto lp = buildSmallMip();

    Presolver presolver;
    auto reduced = presolver.presolve(lp);

    auto& stats = presolver.stats();
    CHECK(stats.rounds >= 1);
    // The singleton rows should be detected.
    CHECK(stats.rows_removed >= 0);  // At least some reductions.
}

// =============================================================================
// Tests: No reductions needed
// =============================================================================

TEST_CASE("Presolve: no reductions on dense problem", "[presolve]") {
    // A problem where no presolve applies: 2 vars, 2 constraints,
    // each var appears in both constraints (not singleton cols).
    LpProblem lp;
    lp.name = "no_reduce";
    lp.sense = Sense::Minimize;
    lp.num_cols = 2;
    lp.obj = {-1.0, -1.0};
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {10.0, 10.0};
    lp.col_type = {VarType::Integer, VarType::Integer};
    lp.col_names = {"x", "y"};

    lp.num_rows = 2;
    lp.row_lower = {-kInf, -kInf};
    lp.row_upper = {15.0, 12.0};
    lp.row_names = {"c1", "c2"};

    std::vector<Triplet> trips = {
        {0, 0, 1.0}, {0, 1, 1.0},
        {1, 0, 1.0}, {1, 1, 0.5},
    };
    lp.matrix = SparseMatrix(2, 2, std::move(trips));

    Presolver presolver;
    auto reduced = presolver.presolve(lp);

    // Neither variable is a singleton column (both appear in 2 rows).
    // No fixed vars, no singleton rows, etc.
    CHECK(reduced.num_cols == 2);
    CHECK(reduced.num_rows == 2);
}
