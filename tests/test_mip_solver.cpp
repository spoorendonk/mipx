#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <filesystem>

#include "mipx/io.h"
#include "mipx/mip_solver.h"

using namespace mipx;
using Catch::Matchers::WithinAbs;

// ---------------------------------------------------------------------------
// Helper: build a small MIP for testing
// min -x - 2y  s.t. x + y <= 4, x <= 3, y <= 3, x,y >= 0, x,y integer
// LP relaxation optimal: x=1, y=3, obj=-7
// MIP optimal: x=1, y=3, obj=-7 (same, LP solution is integral)
// ---------------------------------------------------------------------------

static LpProblem buildSimpleMip() {
    LpProblem lp;
    lp.name = "simple_mip";
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
// Helper: build a MIP that needs branching
// min -x - 2y  s.t. x + y <= 4.5, x,y >= 0, x,y integer
// LP optimal: x=0, y=4.5, obj=-9
// MIP optimal: x=0, y=4, obj=-8
// (With branching: need to round y down)
// ---------------------------------------------------------------------------

static LpProblem buildBranchingMip() {
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

    std::vector<Triplet> trips = {
        {0, 0, 1.0}, {0, 1, 1.0},
    };
    lp.matrix = SparseMatrix(1, 2, std::move(trips));
    return lp;
}

// ---------------------------------------------------------------------------
// Helper: root LP fractional but rounding can provide an incumbent.
// min -x  s.t. x <= 0.49, x >= 0, x integer
// ---------------------------------------------------------------------------

static LpProblem buildRootRoundingMip() {
    LpProblem lp;
    lp.name = "root_rounding_mip";
    lp.sense = Sense::Minimize;
    lp.num_cols = 1;
    lp.obj = {-1.0};
    lp.col_lower = {0.0};
    lp.col_upper = {kInf};
    lp.col_type = {VarType::Integer};
    lp.col_names = {"x"};

    lp.num_rows = 1;
    lp.row_lower = {-kInf};
    lp.row_upper = {0.49};
    lp.row_names = {"ub"};

    std::vector<Triplet> trips = {
        {0, 0, 1.0},
    };
    lp.matrix = SparseMatrix(1, 1, std::move(trips));
    return lp;
}

// ---------------------------------------------------------------------------
// Helper: root LP fractional where plain rounding fails; FP/RENS can still
// provide a feasible incumbent.
// min -x  s.t. x <= 4.6, x >= 0, x integer
// ---------------------------------------------------------------------------

static LpProblem buildRootFractionalHeuristicMip() {
    LpProblem lp;
    lp.name = "root_fractional_heur_mip";
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

// ---------------------------------------------------------------------------
// Helper: LP-light Scylla-FPR probe model with many binaries.
// min -sum_j w_j x_j
// s.t. sum_j x_j <= 5.5, x_j binary
// LP root is fractional, while integer-feasible incumbents are easy to verify.
// ---------------------------------------------------------------------------

static LpProblem buildLpLightScyllaProbeMip() {
    constexpr Index n = 12;
    LpProblem lp;
    lp.name = "lplight_scylla_probe_mip";
    lp.sense = Sense::Minimize;
    lp.num_cols = n;
    lp.obj = {-12.0, -11.0, -10.0, -9.0, -8.0, -7.0,
              -6.0,  -5.0,  -4.0,  -3.0, -2.0, -1.0};
    lp.col_lower.assign(n, 0.0);
    lp.col_upper.assign(n, 1.0);
    lp.col_type.assign(n, VarType::Binary);
    lp.col_names = {"x1", "x2", "x3", "x4", "x5", "x6",
                    "x7", "x8", "x9", "x10", "x11", "x12"};

    lp.num_rows = 1;
    lp.row_lower = {-kInf};
    lp.row_upper = {5.5};
    lp.row_names = {"cap"};

    std::vector<Triplet> trips;
    trips.reserve(n);
    for (Index j = 0; j < n; ++j) {
        trips.push_back({0, j, 1.0});
    }
    lp.matrix = SparseMatrix(1, n, std::move(trips));
    return lp;
}

// ---------------------------------------------------------------------------
// Helper: LP-light integer-repair probe
// min -x
// s.t. x <= 3.7, x integer
// LP root: x=3.7, while integer rounding can violate and requires +-1 repair.
// ---------------------------------------------------------------------------

static LpProblem buildLpLightIntegerRepairProbeMip() {
    LpProblem lp;
    lp.name = "lplight_integer_repair_probe_mip";
    lp.sense = Sense::Minimize;
    lp.num_cols = 1;
    lp.obj = {-1.0};
    lp.col_lower = {0.0};
    lp.col_upper = {10.0};
    lp.col_type = {VarType::Integer};
    lp.col_names = {"x"};

    lp.num_rows = 1;
    lp.row_lower = {-kInf};
    lp.row_upper = {3.7};
    lp.row_names = {"ub_x"};

    std::vector<Triplet> trips = {
        {0, 0, 1.0},
    };
    lp.matrix = SparseMatrix(1, 1, std::move(trips));
    return lp;
}

// ---------------------------------------------------------------------------
// Helper: build infeasible MIP
// x >= 5, x <= 3, x integer
// ---------------------------------------------------------------------------

static LpProblem buildInfeasibleMip() {
    LpProblem lp;
    lp.name = "infeasible_mip";
    lp.sense = Sense::Minimize;
    lp.num_cols = 1;
    lp.obj = {1.0};
    lp.col_lower = {0.0};
    lp.col_upper = {kInf};
    lp.col_type = {VarType::Integer};
    lp.col_names = {"x"};

    // x >= 5 AND x <= 3 via constraints.
    lp.num_rows = 2;
    lp.row_lower = {5.0, -kInf};
    lp.row_upper = {kInf, 3.0};
    lp.row_names = {"lb", "ub"};

    std::vector<Triplet> trips = {
        {0, 0, 1.0},
        {1, 0, 1.0},
    };
    lp.matrix = SparseMatrix(2, 1, std::move(trips));
    return lp;
}

// ---------------------------------------------------------------------------
// Helper: Knapsack MIP
// max 5x1 + 4x2 + 3x3  s.t. 2x1 + 3x2 + 2x3 <= 5, x binary
// = min -5x1 -4x2 -3x3
// Optimal: x1=1, x2=1, x3=0, obj=-9
// (LP relaxation: x1=1, x2=1, x3=0 is already integral for this problem)
// Try another: max 6x1 + 5x2 + 4x3 s.t. 3x1 + 2x2 + 2x3 <= 5, x binary
// LP optimal: x1=1, x2=1, x3=0 gives 3+2=5<=5, obj=11
//             x1=1, x2=0, x3=1 gives 3+2=5<=5, obj=10
// Opt: x1=1, x2=1, x3=0, obj=-11
// ---------------------------------------------------------------------------

static LpProblem buildKnapsackMip() {
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
// Helper: fractional nodes beyond root to exercise in-tree cut management.
// min -5x1 -4x2 -3x3  s.t. x1 + x2 + x3 <= 2.5, x integer >= 0
// ---------------------------------------------------------------------------

static LpProblem buildTreeCutMip() {
    LpProblem lp;
    lp.name = "tree_cut_mip";
    lp.sense = Sense::Minimize;
    lp.num_cols = 3;
    lp.obj = {-5.0, -4.0, -3.0};
    lp.col_lower = {0.0, 0.0, 0.0};
    lp.col_upper = {kInf, kInf, kInf};
    lp.col_type = {VarType::Integer, VarType::Integer, VarType::Integer};
    lp.col_names = {"x1", "x2", "x3"};

    lp.num_rows = 1;
    lp.row_lower = {-kInf};
    lp.row_upper = {2.5};
    lp.row_names = {"cap"};

    std::vector<Triplet> trips = {
        {0, 0, 1.0}, {0, 1, 1.0}, {0, 2, 1.0},
    };
    lp.matrix = SparseMatrix(1, 3, std::move(trips));
    return lp;
}

// ---------------------------------------------------------------------------
// Helper: regression for presolve objective offset handling.
// min 100*x + y
// s.t. 2 <= y + z <= 100
//      x fixed at 1 (binary), 0 <= y <= 10 (integer), 0 <= z <= 10 (continuous)
// Optimal objective is 100 (x=1, y=0, z=2).
// ---------------------------------------------------------------------------

static LpProblem buildPresolveOffsetRegressionMip() {
    LpProblem lp;
    lp.name = "presolve_offset_regression";
    lp.sense = Sense::Minimize;
    lp.num_cols = 3;
    lp.obj = {100.0, 1.0, 0.0};
    lp.col_lower = {1.0, 0.0, 0.0};
    lp.col_upper = {1.0, 10.0, 10.0};
    lp.col_type = {VarType::Binary, VarType::Integer, VarType::Continuous};
    lp.col_names = {"x", "y", "z"};

    lp.num_rows = 1;
    lp.row_lower = {2.0};
    lp.row_upper = {100.0};
    lp.row_names = {"balance"};

    std::vector<Triplet> trips = {
        {0, 1, 1.0},
        {0, 2, 1.0},
    };
    lp.matrix = SparseMatrix(1, 3, std::move(trips));
    return lp;
}

// ---------------------------------------------------------------------------
// Helper: infeasible binary model with fractional root to exercise conflict
// learning and no-good reuse in the tree.
// x + y = 1.5, x,y binary
// ---------------------------------------------------------------------------

static LpProblem buildConflictLearningMip() {
    LpProblem lp;
    lp.name = "conflict_learning_mip";
    lp.sense = Sense::Minimize;
    lp.num_cols = 2;
    lp.obj = {0.0, 0.0};
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {1.0, 1.0};
    lp.col_type = {VarType::Binary, VarType::Binary};
    lp.col_names = {"x", "y"};

    lp.num_rows = 1;
    lp.row_lower = {1.5};
    lp.row_upper = {1.5};
    lp.row_names = {"eq"};

    std::vector<Triplet> trips = {
        {0, 0, 1.0}, {0, 1, 1.0},
    };
    lp.matrix = SparseMatrix(1, 2, std::move(trips));
    return lp;
}

// ---------------------------------------------------------------------------
// Helper: infeasible parity-like binary model to trigger search stagnation.
// sum x_i = 2.5, x_i binary.
// ---------------------------------------------------------------------------

static LpProblem buildSearchStagnationMip() {
    constexpr Index n = 10;
    LpProblem lp;
    lp.name = "search_stagnation_mip";
    lp.sense = Sense::Minimize;
    lp.num_cols = n;
    lp.obj.assign(n, 0.0);
    lp.col_lower.assign(n, 0.0);
    lp.col_upper.assign(n, 1.0);
    lp.col_type.assign(n, VarType::Binary);
    lp.col_names = {"x1", "x2", "x3", "x4", "x5",
                    "x6", "x7", "x8", "x9", "x10"};

    lp.num_rows = 1;
    lp.row_lower = {4.5};
    lp.row_upper = {4.5};
    lp.row_names = {"eq"};

    std::vector<Triplet> trips;
    trips.reserve(n);
    for (Index j = 0; j < n; ++j) {
        trips.push_back({0, j, 1.0});
    }
    lp.matrix = SparseMatrix(1, n, std::move(trips));
    return lp;
}

// ---------------------------------------------------------------------------
// Helper: symmetric binary pair so symmetry cuts should be generated/applied.
// ---------------------------------------------------------------------------

static LpProblem buildSymmetryProbeMip() {
    LpProblem lp;
    lp.name = "symmetry_probe_mip";
    lp.sense = Sense::Minimize;
    lp.num_cols = 2;
    lp.obj = {-1.0, -1.0};
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {1.0, 1.0};
    lp.col_type = {VarType::Binary, VarType::Binary};
    lp.col_names = {"x0", "x1"};

    lp.num_rows = 1;
    lp.row_lower = {-kInf};
    lp.row_upper = {1.0};
    lp.row_names = {"sum"};

    std::vector<Triplet> trips = {
        {0, 0, 1.0}, {0, 1, 1.0},
    };
    lp.matrix = SparseMatrix(1, 2, std::move(trips));
    return lp;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST_CASE("MipSolver: LP with no integers", "[mip]") {
    auto lp = buildSimpleMip();
    // Make it a pure LP.
    lp.col_type = {VarType::Continuous, VarType::Continuous};

    MipSolver solver;
    solver.setVerbose(false);
    solver.load(lp);
    auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(-7.0, 1e-6));
    CHECK(result.nodes == 0);
}

TEST_CASE("MipSolver: integral LP relaxation", "[mip]") {
    auto lp = buildSimpleMip();

    MipSolver solver;
    solver.setVerbose(false);
    solver.load(lp);
    auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(-7.0, 1e-6));
    CHECK(result.nodes == 1);  // Root is already integral.
}

TEST_CASE("MipSolver: needs branching", "[mip]") {
    auto lp = buildBranchingMip();

    MipSolver solver;
    solver.setVerbose(false);
    solver.load(lp);
    auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(-8.0, 1e-6));
    // With cutting planes enabled, the problem may be solved at the root
    // (cuts can close the integrality gap), so nodes >= 1.
    CHECK(result.nodes >= 1);
    REQUIRE(result.solution.size() == 2);
    // x=0, y=4 or x=4, y=0 or other combos with obj=-8.
    // Valid: any (x,y) with x+y<=4, x,y>=0, integer, -x-2y=-8
    // y=4, x=0 gives -8. y=3, x=2 gives -8.
    Real obj = -result.solution[0] - 2.0 * result.solution[1];
    CHECK_THAT(obj, WithinAbs(-8.0, 1e-6));
}

TEST_CASE("MipSolver: reliability branching collects strong-branch telemetry", "[mip][branching]") {
    auto lp = buildBranchingMip();

    MipSolver solver;
    solver.setVerbose(false);
    solver.setCutsEnabled(false);
    solver.load(lp);
    auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    const auto& stats = solver.getBranchingStats();
    CHECK(stats.selections >= 1);
    CHECK(stats.strong_branch_calls >= 1);
    CHECK(stats.strong_branch_probes >= 2);
    CHECK(stats.strong_branch_probe_iters >= 0);
    CHECK(stats.strong_branch_probe_work_units >= 0.0);
}

TEST_CASE("MipSolver: infeasible", "[mip]") {
    auto lp = buildInfeasibleMip();

    MipSolver solver;
    solver.setVerbose(false);
    solver.load(lp);
    auto result = solver.solve();

    CHECK(result.status == Status::Infeasible);
}

TEST_CASE("MipSolver: knapsack", "[mip]") {
    auto lp = buildKnapsackMip();

    MipSolver solver;
    solver.setVerbose(false);
    solver.setNodeLimit(1000);
    solver.load(lp);
    auto result = solver.solve();

    REQUIRE((result.status == Status::Optimal || result.status == Status::NodeLimit));
    if (result.status == Status::Optimal) {
        CHECK_THAT(result.objective, WithinAbs(-11.0, 1e-6));
    }
}

TEST_CASE("MipSolver: node limit", "[mip]") {
    auto lp = buildBranchingMip();

    MipSolver solver;
    solver.setVerbose(false);
    solver.setNodeLimit(1);  // Only solve root.
    solver.load(lp);
    auto result = solver.solve();

    // With node limit 1, we process root but can't explore children.
    CHECK((result.status == Status::NodeLimit || result.status == Status::Optimal));
}

TEST_CASE("MipSolver: root heuristics provide incumbent before branching", "[mip]") {
    auto lp = buildRootRoundingMip();

    MipSolver solver;
    solver.setVerbose(false);
    solver.setCutsEnabled(false);
    solver.setNodeLimit(1);  // root only
    solver.load(lp);
    auto result = solver.solve();

    CHECK((result.status == Status::NodeLimit || result.status == Status::Optimal));
    REQUIRE(!result.solution.empty());
    CHECK_THAT(result.objective, WithinAbs(0.0, 1e-6));
    CHECK_THAT(result.solution[0], WithinAbs(0.0, 1e-6));
}

TEST_CASE("MipSolver: LP-based root portfolio can bootstrap incumbent", "[mip]") {
    auto lp = buildRootFractionalHeuristicMip();

    MipSolver solver;
    solver.setVerbose(false);
    solver.setCutsEnabled(false);
    solver.setNodeLimit(1);  // root only
    solver.load(lp);
    auto result = solver.solve();

    CHECK((result.status == Status::NodeLimit || result.status == Status::Optimal));
    REQUIRE(!result.solution.empty());
    CHECK(result.solution[0] >= 0.0);
    CHECK(result.solution[0] <= 4.6 + 1e-6);
}

TEST_CASE("MipSolver: in-tree cut telemetry is populated", "[mip][cuts]") {
    auto lp = buildTreeCutMip();

    MipSolver solver;
    solver.setVerbose(false);
    solver.setCutsEnabled(true);
    solver.setMaxCutRounds(0);  // emphasize in-tree cuts over root rounds
    solver.load(lp);
    auto result = solver.solve();

    CHECK((result.status == Status::Optimal ||
           result.status == Status::NodeLimit ||
           result.status == Status::TimeLimit));
    const auto& cut_stats = solver.getCutStats();
    CHECK(cut_stats.tree_nodes_with_cuts + cut_stats.tree_nodes_skipped >= 1);
    CHECK(cut_stats.tree_rounds >= 0);
}

TEST_CASE("MipSolver: symmetry cuts are applied when presolve is off", "[mip][symmetry]") {
    auto lp = buildSymmetryProbeMip();

    MipSolver without_symmetry;
    without_symmetry.setVerbose(false);
    without_symmetry.setCutsEnabled(false);
    without_symmetry.setPresolve(false);
    without_symmetry.setSymmetryEnabled(false);
    without_symmetry.load(lp);
    const auto off = without_symmetry.solve();

    MipSolver with_symmetry;
    with_symmetry.setVerbose(false);
    with_symmetry.setCutsEnabled(false);
    with_symmetry.setPresolve(false);
    with_symmetry.setSymmetryEnabled(true);
    with_symmetry.load(lp);
    const auto on = with_symmetry.solve();

    CHECK((off.status == Status::Optimal ||
           off.status == Status::NodeLimit ||
           off.status == Status::TimeLimit));
    CHECK((on.status == Status::Optimal ||
           on.status == Status::NodeLimit ||
           on.status == Status::TimeLimit));
    CHECK_THAT(on.objective, WithinAbs(off.objective, 1e-9));

    const auto& off_stats = without_symmetry.getSymmetryStats();
    CHECK(off_stats.orbits == 0);
    CHECK(off_stats.cuts_added == 0);
    CHECK_FALSE(off_stats.cuts_applied);

    const auto& on_stats = with_symmetry.getSymmetryStats();
    CHECK(on_stats.orbits == 1);
    CHECK(on_stats.cuts_added == 1);
    CHECK(on_stats.cuts_applied);
    CHECK(on_stats.detect_work_units > 0.0);
    CHECK(on_stats.cut_work_units > 0.0);
}

TEST_CASE("MipSolver: MIPLIB gt2", "[mip][miplib]") {
    std::string path = std::string(TEST_DATA_DIR) + "/miplib/gt2.mps.gz";
    if (!std::filesystem::exists(path)) {
        SKIP("gt2.mps.gz not found (run download_miplib.sh --small)");
    }

    auto problem = readMps(path);
    REQUIRE(problem.hasIntegers());

    MipSolver solver;
    solver.setVerbose(true);
    solver.setNodeLimit(10000);
    solver.setTimeLimit(60.0);
    solver.load(problem);
    auto result = solver.solve();

    // gt2 optimal: 21166.0
    if (result.status == Status::Optimal) {
        CHECK_THAT(result.objective, WithinAbs(21166.0, 1.0));
    }
    // If not solved to optimality, at least it shouldn't crash.
    CHECK((result.status == Status::Optimal ||
           result.status == Status::NodeLimit ||
           result.status == Status::TimeLimit));
}

TEST_CASE("MipSolver: work units are positive", "[mip][work_units]") {
    auto problem = buildBranchingMip();
    MipSolver solver;
    solver.setVerbose(false);
    solver.load(problem);
    auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    CHECK(result.work_units > 0.0);
}

TEST_CASE("MipSolver: exact refinement default remains off and non-regressive",
          "[mip][exact_refinement]") {
    auto lp = buildBranchingMip();

    MipSolver solver_default;
    solver_default.setVerbose(false);
    solver_default.setCutsEnabled(false);
    solver_default.load(lp);
    const auto default_result = solver_default.solve();

    MipSolver solver_explicit_off;
    solver_explicit_off.setVerbose(false);
    solver_explicit_off.setCutsEnabled(false);
    solver_explicit_off.setExactRefinementMode(ExactRefinementMode::Off);
    solver_explicit_off.load(lp);
    const auto off_result = solver_explicit_off.solve();

    REQUIRE(default_result.status == Status::Optimal);
    REQUIRE(off_result.status == Status::Optimal);
    CHECK_THAT(default_result.objective, WithinAbs(off_result.objective, 1e-9));
    CHECK_THAT(default_result.work_units, WithinAbs(off_result.work_units, 1e-9));

    const auto& default_stats = solver_default.getExactRefinementStats();
    CHECK(default_stats.mode == ExactRefinementMode::Off);
    CHECK_FALSE(default_stats.triggered);
    CHECK(default_stats.evaluation_work_units == 0.0);
}

TEST_CASE("MipSolver: exact refinement forced mode is deterministic",
          "[mip][exact_refinement]") {
    auto lp = buildBranchingMip();

    MipSolver solver_a;
    solver_a.setVerbose(false);
    solver_a.setCutsEnabled(false);
    solver_a.setPresolve(false);
    solver_a.setExactRefinementMode(ExactRefinementMode::On);
    solver_a.setExactRefinementRationalCheck(true);
    solver_a.load(lp);
    const auto a = solver_a.solve();

    MipSolver solver_b;
    solver_b.setVerbose(false);
    solver_b.setCutsEnabled(false);
    solver_b.setPresolve(false);
    solver_b.setExactRefinementMode(ExactRefinementMode::On);
    solver_b.setExactRefinementRationalCheck(true);
    solver_b.load(lp);
    const auto b = solver_b.solve();

    REQUIRE(a.status == Status::Optimal);
    REQUIRE(b.status == Status::Optimal);
    CHECK_THAT(a.objective, WithinAbs(b.objective, 1e-9));
    CHECK_THAT(a.work_units, WithinAbs(b.work_units, 1e-9));

    const auto& sa = solver_a.getExactRefinementStats();
    const auto& sb = solver_b.getExactRefinementStats();
    CHECK(sa.mode == ExactRefinementMode::On);
    CHECK(sb.mode == ExactRefinementMode::On);
    CHECK(sa.rational_verification_enabled);
    CHECK(sb.rational_verification_enabled);
    CHECK(sa.triggered);
    CHECK(sb.triggered);
    CHECK(sa.rounds >= 1);
    CHECK(sa.evaluation_work_units > 0.0);
    CHECK_THAT(sa.evaluation_work_units, WithinAbs(sb.evaluation_work_units, 1e-9));
    CHECK(sa.resolve_calls == sb.resolve_calls);
    CHECK(sa.resolve_iterations == sb.resolve_iterations);
}

TEST_CASE("MipSolver: exact refinement auto mode triggers on unsupported rational checks",
          "[mip][exact_refinement]") {
    auto lp = buildBranchingMip();

    MipSolver solver;
    solver.setVerbose(false);
    solver.setCutsEnabled(false);
    solver.setPresolve(false);
    solver.setExactRefinementMode(ExactRefinementMode::Auto);
    solver.setExactRefinementRationalCheck(true);
    solver.setExactRefinementRationalScale(1.0e10);
    solver.load(lp);
    const auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    const auto& stats = solver.getExactRefinementStats();
    CHECK(stats.triggered);
    CHECK_FALSE(stats.rational_supported);
    CHECK_FALSE(stats.rational_certificate_passed);
    CHECK_FALSE(stats.certificate_passed);
}

TEST_CASE("MipSolver: exact refinement evaluates active root LP rows including cuts",
          "[mip][exact_refinement][cuts]") {
    auto lp = buildBranchingMip();

    MipSolver solver;
    solver.setVerbose(false);
    solver.setPresolve(false);
    solver.setSymmetryEnabled(false);
    solver.setCutsEnabled(true);
    solver.setCutEffortMode(CutEffortMode::Aggressive);
    solver.setExactRefinementMode(ExactRefinementMode::On);
    solver.load(lp);
    const auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    const auto& cut_stats = solver.getCutStats();
    REQUIRE(cut_stats.root_cuts_added > 0);

    const auto& stats = solver.getExactRefinementStats();
    CHECK(stats.rows_evaluated == lp.num_rows + cut_stats.root_cuts_added);
    CHECK(stats.cols_evaluated == lp.num_cols);
}

TEST_CASE("MipSolver: exact refinement off mode does not evaluate certificate rows",
          "[mip][exact_refinement][cuts]") {
    auto lp = buildBranchingMip();

    MipSolver solver;
    solver.setVerbose(false);
    solver.setPresolve(false);
    solver.setSymmetryEnabled(false);
    solver.setCutsEnabled(true);
    solver.setCutEffortMode(CutEffortMode::Aggressive);
    solver.setExactRefinementMode(ExactRefinementMode::Off);
    solver.load(lp);
    const auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    const auto& cut_stats = solver.getCutStats();
    REQUIRE(cut_stats.root_cuts_added > 0);

    const auto& stats = solver.getExactRefinementStats();
    CHECK(stats.mode == ExactRefinementMode::Off);
    CHECK_FALSE(stats.triggered);
    CHECK(stats.rows_evaluated == 0);
    CHECK(stats.cols_evaluated == 0);
    CHECK(stats.evaluation_work_units == 0.0);
}

TEST_CASE("MipSolver: deterministic heuristic mode reproduces with same seed",
          "[mip][heuristics]") {
    auto lp = buildRootFractionalHeuristicMip();

    MipSolver solver_a;
    solver_a.setVerbose(false);
    solver_a.setCutsEnabled(false);
    solver_a.setNodeLimit(1);
    solver_a.setParallelMode(ParallelMode::Deterministic);
    solver_a.setHeuristicSeed(1234);
    solver_a.load(lp);
    auto result_a = solver_a.solve();

    MipSolver solver_b;
    solver_b.setVerbose(false);
    solver_b.setCutsEnabled(false);
    solver_b.setNodeLimit(1);
    solver_b.setParallelMode(ParallelMode::Deterministic);
    solver_b.setHeuristicSeed(1234);
    solver_b.load(lp);
    auto result_b = solver_b.solve();

    REQUIRE((result_a.status == Status::NodeLimit || result_a.status == Status::Optimal));
    REQUIRE((result_b.status == Status::NodeLimit || result_b.status == Status::Optimal));
    CHECK_THAT(result_a.objective, WithinAbs(result_b.objective, 1e-9));
    REQUIRE(result_a.solution.size() == result_b.solution.size());
    for (size_t i = 0; i < result_a.solution.size(); ++i) {
        CHECK_THAT(result_a.solution[i], WithinAbs(result_b.solution[i], 1e-9));
    }
}

TEST_CASE("MipSolver: opportunistic heuristic mode solves", "[mip][heuristics]") {
    auto lp = buildRootFractionalHeuristicMip();

    MipSolver solver;
    solver.setVerbose(false);
    solver.setCutsEnabled(false);
    solver.setNodeLimit(1);
    solver.setParallelMode(ParallelMode::Opportunistic);
    solver.setHeuristicSeed(7);
    solver.load(lp);
    const auto result = solver.solve();

    CHECK((result.status == Status::NodeLimit || result.status == Status::Optimal));
    REQUIRE(!result.solution.empty());
    CHECK(result.solution[0] >= 0.0);
    CHECK(result.solution[0] <= 4.6 + 1e-6);
}

TEST_CASE("MipSolver: pre-root LP-free stage is disabled by default",
          "[mip][heuristics][preroot]") {
    auto lp = buildBranchingMip();

    MipSolver solver;
    solver.setVerbose(false);
    solver.setCutsEnabled(false);
    solver.load(lp);
    auto result = solver.solve();

    CHECK((result.status == Status::Optimal ||
           result.status == Status::NodeLimit ||
           result.status == Status::TimeLimit));
    const auto& stats = solver.getPreRootStats();
    CHECK_FALSE(stats.enabled);
    CHECK(stats.calls == 0);
    CHECK(stats.work_units == 0.0);
    CHECK_FALSE(stats.lp_light_enabled);
    CHECK_FALSE(stats.lp_light_available);
    CHECK(stats.lp_light_calls == 0);
}

TEST_CASE("MipSolver: LP-light capability reflects build configuration",
          "[mip][heuristics][preroot][lplight]") {
    MipSolver solver;
#ifdef MIPX_HAS_LP_LIGHT
    CHECK(solver.hasLpLightCapability());
#else
    CHECK_FALSE(solver.hasLpLightCapability());
#endif
}

TEST_CASE("MipSolver: pre-root LP-free stage can hand off incumbent",
          "[mip][heuristics][preroot]") {
    auto lp = buildRootFractionalHeuristicMip();

    MipSolver solver;
    solver.setVerbose(false);
    solver.setCutsEnabled(false);
    solver.setPresolve(false);
    solver.setNodeLimit(1);
    solver.setHeuristicMode(HeuristicRuntimeMode::Deterministic);
    solver.setHeuristicSeed(11);
    solver.setPreRootLpFreeEnabled(true);
    solver.setPreRootLpFreeWorkBudget(2.0e5);
    solver.setPreRootLpFreeMaxRounds(16);
    solver.setPreRootLpFreeEarlyStop(true);
    solver.load(lp);
    const auto result = solver.solve();

    CHECK((result.status == Status::NodeLimit || result.status == Status::Optimal));
    const auto& stats = solver.getPreRootStats();
    CHECK(stats.enabled);
    CHECK(stats.calls > 0);
    CHECK(stats.work_units > 0.0);
    CHECK(stats.feasible_found >= 1);
    CHECK(stats.lp_light_calls == 0);
    CHECK(std::isfinite(stats.incumbent_at_root));
    REQUIRE(!result.solution.empty());
}

TEST_CASE("MipSolver: pre-root LP-free deterministic mode reproduces with seed",
          "[mip][heuristics][preroot]") {
    auto lp = buildRootFractionalHeuristicMip();

    MipSolver solver_a;
    solver_a.setVerbose(false);
    solver_a.setCutsEnabled(false);
    solver_a.setPresolve(false);
    solver_a.setNodeLimit(1);
    solver_a.setHeuristicMode(HeuristicRuntimeMode::Deterministic);
    solver_a.setHeuristicSeed(77);
    solver_a.setPreRootLpFreeEnabled(true);
    solver_a.setPreRootLpFreeWorkBudget(1.0e5);
    solver_a.setPreRootLpFreeMaxRounds(12);
    solver_a.load(lp);
    const auto a = solver_a.solve();

    MipSolver solver_b;
    solver_b.setVerbose(false);
    solver_b.setCutsEnabled(false);
    solver_b.setPresolve(false);
    solver_b.setNodeLimit(1);
    solver_b.setHeuristicMode(HeuristicRuntimeMode::Deterministic);
    solver_b.setHeuristicSeed(77);
    solver_b.setPreRootLpFreeEnabled(true);
    solver_b.setPreRootLpFreeWorkBudget(1.0e5);
    solver_b.setPreRootLpFreeMaxRounds(12);
    solver_b.load(lp);
    const auto b = solver_b.solve();

    const auto& sa = solver_a.getPreRootStats();
    const auto& sb = solver_b.getPreRootStats();
    CHECK(sa.enabled);
    CHECK(sb.enabled);
    CHECK(sa.calls == sb.calls);
    CHECK_THAT(sa.work_units, WithinAbs(sb.work_units, 1e-9));
    CHECK_THAT(sa.incumbent_at_root, WithinAbs(sb.incumbent_at_root, 1e-9));
    CHECK_THAT(a.objective, WithinAbs(b.objective, 1e-9));
}

TEST_CASE("MipSolver: pre-root LP-light arms can run when enabled",
          "[mip][heuristics][preroot][lplight]") {
    auto lp = buildRootFractionalHeuristicMip();

    MipSolver solver;
    solver.setVerbose(false);
    solver.setCutsEnabled(false);
    solver.setPresolve(false);
    solver.setNodeLimit(1);
    solver.setHeuristicMode(HeuristicRuntimeMode::Deterministic);
    solver.setHeuristicSeed(19);
    solver.setPreRootLpFreeEnabled(false);
    solver.setPreRootLpLightEnabled(true);
    solver.setPreRootLpFreeWorkBudget(1.0e5);
    solver.setPreRootLpFreeMaxRounds(10);
    solver.load(lp);
    const auto result = solver.solve();

    CHECK((result.status == Status::NodeLimit || result.status == Status::Optimal));
    const auto& stats = solver.getPreRootStats();
    CHECK(stats.enabled);
    CHECK(stats.lp_light_enabled);
#ifdef MIPX_HAS_LP_LIGHT
    CHECK(stats.lp_light_available);
    CHECK(stats.lp_light_lp_solves >= 1);
    CHECK(stats.lp_light_calls > 0);
    CHECK(stats.lp_light_fpr_calls + stats.lp_light_diving_calls == stats.lp_light_calls);
#else
    CHECK_FALSE(stats.lp_light_available);
    CHECK(stats.lp_light_calls == 0);
#endif
}

TEST_CASE("MipSolver: pre-root LP-light deterministic mode reproduces with seed",
          "[mip][heuristics][preroot][lplight]") {
    auto lp = buildRootFractionalHeuristicMip();

    MipSolver solver_a;
    solver_a.setVerbose(false);
    solver_a.setCutsEnabled(false);
    solver_a.setPresolve(false);
    solver_a.setNodeLimit(1);
    solver_a.setHeuristicMode(HeuristicRuntimeMode::Deterministic);
    solver_a.setHeuristicSeed(99);
    solver_a.setPreRootLpFreeEnabled(false);
    solver_a.setPreRootLpLightEnabled(true);
    solver_a.setPreRootLpFreeWorkBudget(1.0e5);
    solver_a.setPreRootLpFreeMaxRounds(10);
    solver_a.load(lp);
    const auto a = solver_a.solve();

    MipSolver solver_b;
    solver_b.setVerbose(false);
    solver_b.setCutsEnabled(false);
    solver_b.setPresolve(false);
    solver_b.setNodeLimit(1);
    solver_b.setHeuristicMode(HeuristicRuntimeMode::Deterministic);
    solver_b.setHeuristicSeed(99);
    solver_b.setPreRootLpFreeEnabled(false);
    solver_b.setPreRootLpLightEnabled(true);
    solver_b.setPreRootLpFreeWorkBudget(1.0e5);
    solver_b.setPreRootLpFreeMaxRounds(10);
    solver_b.load(lp);
    const auto b = solver_b.solve();

    const auto& sa = solver_a.getPreRootStats();
    const auto& sb = solver_b.getPreRootStats();
    CHECK(sa.enabled == sb.enabled);
    CHECK(sa.lp_light_enabled == sb.lp_light_enabled);
    CHECK(sa.lp_light_available == sb.lp_light_available);
    CHECK(sa.calls == sb.calls);
    CHECK(sa.lp_light_calls == sb.lp_light_calls);
    CHECK_THAT(sa.work_units, WithinAbs(sb.work_units, 1e-9));
    CHECK_THAT(a.objective, WithinAbs(b.objective, 1e-9));
}

TEST_CASE("MipSolver: pre-root LP-light fixed schedule runs Scylla-FPR arm first",
          "[mip][heuristics][preroot][lplight][scylla]") {
    auto lp = buildLpLightScyllaProbeMip();

    MipSolver solver;
    solver.setVerbose(false);
    solver.setCutsEnabled(false);
    solver.setPresolve(false);
    solver.setNodeLimit(1);
    solver.setHeuristicMode(HeuristicRuntimeMode::Deterministic);
    solver.setHeuristicSeed(2026);
    solver.setPreRootLpFreeEnabled(false);
    solver.setPreRootLpLightEnabled(true);
    solver.setPreRootPortfolioEnabled(false);  // fixed schedule
    solver.setPreRootLpFreeEarlyStop(false);
    solver.setPreRootLpFreeMaxRounds(1);       // single arm call
    solver.setPreRootLpFreeWorkBudget(1.0e9);
    solver.load(lp);
    const auto result = solver.solve();

    CHECK((result.status == Status::NodeLimit || result.status == Status::Optimal));
    const auto& stats = solver.getPreRootStats();
    CHECK(stats.enabled);
    CHECK_FALSE(stats.portfolio_enabled);
#ifdef MIPX_HAS_LP_LIGHT
    CHECK(stats.lp_light_available);
    CHECK(stats.calls == 1);
    CHECK(stats.lp_light_calls == 1);
    CHECK(stats.lp_light_fpr_calls == 1);
    CHECK(stats.lp_light_diving_calls == 0);
    CHECK(stats.work_units > 0.0);
#else
    CHECK_FALSE(stats.lp_light_available);
    CHECK(stats.lp_light_calls == 0);
#endif
}

TEST_CASE("MipSolver: pre-root LP-light Scylla path is deterministic on binary probe",
          "[mip][heuristics][preroot][lplight][scylla]") {
    auto lp = buildLpLightScyllaProbeMip();

    MipSolver solver_a;
    solver_a.setVerbose(false);
    solver_a.setCutsEnabled(false);
    solver_a.setPresolve(false);
    solver_a.setNodeLimit(1);
    solver_a.setHeuristicMode(HeuristicRuntimeMode::Deterministic);
    solver_a.setHeuristicSeed(777);
    solver_a.setPreRootLpFreeEnabled(false);
    solver_a.setPreRootLpLightEnabled(true);
    solver_a.setPreRootPortfolioEnabled(false);
    solver_a.setPreRootLpFreeEarlyStop(false);
    solver_a.setPreRootLpFreeMaxRounds(1);
    solver_a.setPreRootLpFreeWorkBudget(1.0e9);
    solver_a.load(lp);
    const auto a = solver_a.solve();

    MipSolver solver_b;
    solver_b.setVerbose(false);
    solver_b.setCutsEnabled(false);
    solver_b.setPresolve(false);
    solver_b.setNodeLimit(1);
    solver_b.setHeuristicMode(HeuristicRuntimeMode::Deterministic);
    solver_b.setHeuristicSeed(777);
    solver_b.setPreRootLpFreeEnabled(false);
    solver_b.setPreRootLpLightEnabled(true);
    solver_b.setPreRootPortfolioEnabled(false);
    solver_b.setPreRootLpFreeEarlyStop(false);
    solver_b.setPreRootLpFreeMaxRounds(1);
    solver_b.setPreRootLpFreeWorkBudget(1.0e9);
    solver_b.load(lp);
    const auto b = solver_b.solve();

    const auto& sa = solver_a.getPreRootStats();
    const auto& sb = solver_b.getPreRootStats();
    CHECK(sa.calls == sb.calls);
    CHECK(sa.lp_light_calls == sb.lp_light_calls);
    CHECK(sa.lp_light_fpr_calls == sb.lp_light_fpr_calls);
    CHECK(sa.lp_light_diving_calls == sb.lp_light_diving_calls);
    CHECK_THAT(sa.work_units, WithinAbs(sb.work_units, 1e-9));
    CHECK_THAT(a.objective, WithinAbs(b.objective, 1e-9));
}

TEST_CASE("MipSolver: pre-root LP-light Scylla repairs integral-step violations",
          "[mip][heuristics][preroot][lplight][scylla]") {
    auto lp = buildLpLightIntegerRepairProbeMip();

    MipSolver solver;
    solver.setVerbose(false);
    solver.setCutsEnabled(false);
    solver.setPresolve(false);
    solver.setNodeLimit(1);
    solver.setHeuristicMode(HeuristicRuntimeMode::Deterministic);
    solver.setHeuristicSeed(17);
    solver.setPreRootLpFreeEnabled(false);
    solver.setPreRootLpLightEnabled(true);
    solver.setPreRootPortfolioEnabled(false);
    solver.setPreRootLpFreeEarlyStop(false);
    solver.setPreRootLpFreeMaxRounds(1);
    solver.setPreRootLpFreeWorkBudget(1.0e9);
    solver.load(lp);
    const auto result = solver.solve();

    CHECK((result.status == Status::NodeLimit || result.status == Status::Optimal));
    const auto& stats = solver.getPreRootStats();
    CHECK(stats.enabled);
    CHECK_FALSE(stats.portfolio_enabled);
#ifdef MIPX_HAS_LP_LIGHT
    CHECK(stats.lp_light_available);
    CHECK(stats.calls == 1);
    CHECK(stats.lp_light_calls == 1);
    CHECK(stats.lp_light_fpr_calls == 1);
    CHECK(stats.lp_light_diving_calls == 0);
    CHECK(stats.feasible_found >= 1);
    CHECK(stats.improvements >= 1);
    REQUIRE(!result.solution.empty());
    CHECK_THAT(result.solution[0], WithinAbs(3.0, 1e-9));
#else
    CHECK_FALSE(stats.lp_light_available);
    CHECK(stats.lp_light_calls == 0);
#endif
}

TEST_CASE("MipSolver: pre-root fixed schedule can be selected",
          "[mip][heuristics][preroot][portfolio]") {
    auto lp = buildRootFractionalHeuristicMip();

    MipSolver solver;
    solver.setVerbose(false);
    solver.setCutsEnabled(false);
    solver.setPresolve(false);
    solver.setNodeLimit(1);
    solver.setHeuristicMode(HeuristicRuntimeMode::Deterministic);
    solver.setHeuristicSeed(5);
    solver.setPreRootLpFreeEnabled(true);
    solver.setPreRootLpLightEnabled(false);
    solver.setPreRootPortfolioEnabled(false);
    solver.setPreRootLpFreeEarlyStop(false);
    solver.setPreRootLpFreeMaxRounds(6);
    solver.setPreRootLpFreeWorkBudget(1.0e9);
    solver.load(lp);
    (void)solver.solve();

    const auto& stats = solver.getPreRootStats();
    CHECK(stats.enabled);
    CHECK_FALSE(stats.portfolio_enabled);
    CHECK(stats.calls == 6);
    CHECK(stats.feasible_found >= 1);
    CHECK(stats.fj_calls == 2);
    CHECK(stats.fpr_calls == 2);
    CHECK(stats.local_mip_calls == 2);
}

TEST_CASE("MipSolver: pre-root fixed schedule skips LocalMip until an incumbent exists",
          "[mip][heuristics][preroot][portfolio]") {
    auto lp = buildConflictLearningMip();

    MipSolver solver;
    solver.setVerbose(false);
    solver.setCutsEnabled(false);
    solver.setPresolve(false);
    solver.setNodeLimit(1);
    solver.setHeuristicMode(HeuristicRuntimeMode::Deterministic);
    solver.setHeuristicSeed(13);
    solver.setPreRootLpFreeEnabled(true);
    solver.setPreRootLpLightEnabled(false);
    solver.setPreRootPortfolioEnabled(false);
    solver.setPreRootLpFreeEarlyStop(false);
    solver.setPreRootLpFreeMaxRounds(9);
    solver.setPreRootLpFreeWorkBudget(1.0e9);
    solver.load(lp);
    (void)solver.solve();

    const auto& stats = solver.getPreRootStats();
    CHECK(stats.enabled);
    CHECK_FALSE(stats.portfolio_enabled);
    CHECK(stats.calls > 0);
    CHECK(stats.feasible_found == 0);
    CHECK(stats.local_mip_calls == 0);
    CHECK(stats.fj_calls + stats.fpr_calls == stats.calls);
}

TEST_CASE("MipSolver: pre-root adaptive portfolio tracks telemetry",
          "[mip][heuristics][preroot][portfolio]") {
    auto lp = buildRootFractionalHeuristicMip();

    MipSolver solver;
    solver.setVerbose(false);
    solver.setCutsEnabled(false);
    solver.setPresolve(false);
    solver.setNodeLimit(1);
    solver.setHeuristicMode(HeuristicRuntimeMode::Deterministic);
    solver.setHeuristicSeed(23);
    solver.setPreRootLpFreeEnabled(true);
    solver.setPreRootLpLightEnabled(true);
    solver.setPreRootPortfolioEnabled(true);
    solver.setPreRootLpFreeEarlyStop(false);
    solver.setPreRootLpFreeMaxRounds(10);
    solver.setPreRootLpFreeWorkBudget(1.0e6);
    solver.load(lp);
    (void)solver.solve();

    const auto& stats = solver.getPreRootStats();
    CHECK(stats.enabled);
    CHECK((stats.portfolio_enabled || stats.calls <= 1));
    CHECK(stats.portfolio_epochs == stats.calls);
    CHECK(stats.effort_scale_final > 0.0);
    CHECK(stats.fj_calls + stats.fpr_calls + stats.local_mip_calls +
              stats.lp_light_fpr_calls + stats.lp_light_diving_calls ==
          stats.calls);
}

TEST_CASE("MipSolver: pre-root opportunistic fixed schedule enables LocalMip after incumbent",
          "[mip][heuristics][preroot][portfolio]") {
    auto lp = buildRootFractionalHeuristicMip();

    MipSolver solver;
    solver.setVerbose(false);
    solver.setCutsEnabled(false);
    solver.setPresolve(false);
    solver.setNodeLimit(1);
    solver.setNumThreads(1);
    solver.setParallelMode(ParallelMode::Opportunistic);
    solver.setHeuristicSeed(31);
    solver.setPreRootLpFreeEnabled(true);
    solver.setPreRootLpLightEnabled(false);
    solver.setPreRootPortfolioEnabled(false);
    solver.setPreRootLpFreeEarlyStop(false);
    solver.setPreRootLpFreeMaxRounds(96);
    solver.setPreRootLpFreeWorkBudget(1.0e9);
    solver.load(lp);
    (void)solver.solve();

    const auto& stats = solver.getPreRootStats();
    CHECK(stats.enabled);
    CHECK_FALSE(stats.portfolio_enabled);
    CHECK(stats.calls >= 1);
    CHECK(stats.feasible_found >= 1);
    CHECK(stats.local_mip_calls > 0);
    CHECK(stats.fj_calls + stats.fpr_calls + stats.local_mip_calls == stats.calls);
}

TEST_CASE("MipSolver: conflict learning learns and reuses no-goods", "[mip][conflicts]") {
    auto lp = buildConflictLearningMip();

    MipSolver solver;
    solver.setVerbose(false);
    solver.setCutsEnabled(false);
    solver.setPresolve(false);
    solver.setNodeLimit(128);
    solver.load(lp);
    const auto result = solver.solve();

    CHECK(result.status == Status::Infeasible);
    const auto& cstats = solver.getConflictStats();
    CHECK(cstats.learned >= 1);
    CHECK(cstats.lp_infeasible_conflicts >= 1);
    CHECK(cstats.minimized_literals >= 0);
}

TEST_CASE("MipSolver: conflict learning preserves feasible optimum", "[mip][conflicts]") {
    auto lp = buildBranchingMip();

    MipSolver with_conflicts;
    with_conflicts.setVerbose(false);
    with_conflicts.setCutsEnabled(false);
    with_conflicts.setConflictsEnabled(true);
    with_conflicts.load(lp);
    const auto with_result = with_conflicts.solve();

    MipSolver without_conflicts;
    without_conflicts.setVerbose(false);
    without_conflicts.setCutsEnabled(false);
    without_conflicts.setConflictsEnabled(false);
    without_conflicts.load(lp);
    const auto without_result = without_conflicts.solve();

    REQUIRE(with_result.status == Status::Optimal);
    REQUIRE(without_result.status == Status::Optimal);
    CHECK_THAT(with_result.objective, WithinAbs(without_result.objective, 1e-9));
}

TEST_CASE("MipSolver: stable search profile is reproducible", "[mip][search]") {
    auto lp = buildSearchStagnationMip();

    MipSolver solver_a;
    solver_a.setVerbose(false);
    solver_a.setCutsEnabled(false);
    solver_a.setPresolve(false);
    solver_a.setSearchProfile(SearchProfile::Stable);
    solver_a.setHeuristicMode(HeuristicRuntimeMode::Deterministic);
    solver_a.setHeuristicSeed(42);
    solver_a.setNodeLimit(300);
    solver_a.load(lp);
    const auto a = solver_a.solve();

    MipSolver solver_b;
    solver_b.setVerbose(false);
    solver_b.setCutsEnabled(false);
    solver_b.setPresolve(false);
    solver_b.setSearchProfile(SearchProfile::Stable);
    solver_b.setHeuristicMode(HeuristicRuntimeMode::Deterministic);
    solver_b.setHeuristicSeed(42);
    solver_b.setNodeLimit(300);
    solver_b.load(lp);
    const auto b = solver_b.solve();

    REQUIRE(a.status == b.status);
    CHECK(a.nodes == b.nodes);
    CHECK_THAT(a.work_units, WithinAbs(b.work_units, 1e-9));
}

TEST_CASE("MipSolver: aggressive search profile switches and restarts", "[mip][search]") {
    auto lp = buildSearchStagnationMip();

    MipSolver solver;
    solver.setVerbose(false);
    solver.setCutsEnabled(false);
    solver.setPresolve(false);
    solver.setSearchProfile(SearchProfile::Aggressive);
    solver.setRestartsEnabled(true);
    solver.setRestartControls(8, 2);
    solver.setNodeLimit(300);
    solver.load(lp);
    const auto result = solver.solve();

    CHECK((result.status == Status::Infeasible ||
           result.status == Status::NodeLimit ||
           result.status == Status::TimeLimit));
    const auto& sstats = solver.getSearchStats();
    CHECK(sstats.policy_switches >= 1);
    CHECK(sstats.restarts >= 1);
    CHECK(sstats.strong_budget_updates >= 1);
}

TEST_CASE("MipSolver: in-tree presolve telemetry is populated", "[mip][presolve][tree]") {
    auto lp = buildSearchStagnationMip();

    MipSolver solver;
    solver.setVerbose(false);
    solver.setCutsEnabled(false);
    solver.setPresolve(false);
    solver.setTreePresolveEnabled(true);
    solver.setNodeLimit(300);
    solver.load(lp);
    const auto result = solver.solve();

    CHECK((result.status == Status::Infeasible ||
           result.status == Status::NodeLimit ||
           result.status == Status::TimeLimit));
    const auto& stats = solver.getTreePresolveStats();
    CHECK(stats.attempts >= 1);
    CHECK((stats.runs >= 1 || stats.infeasible >= 1));
    CHECK(stats.activity_tightenings >= 0);
    CHECK(stats.reduced_cost_tightenings >= 0);
}

TEST_CASE("MipSolver: tree presolve auto tuning classifies small pure-binary models",
          "[mip][presolve][tree]") {
    MipSolver pure_binary;
    pure_binary.setVerbose(false);
    pure_binary.load(buildKnapsackMip());
    CHECK(pure_binary.isTreePresolveAutoTuningEnabled());
    CHECK(pure_binary.isTreePresolveBinaryLiteProfileActive());

    pure_binary.setTreePresolveAutoTuning(false);
    CHECK_FALSE(pure_binary.isTreePresolveBinaryLiteProfileActive());

    MipSolver general_integer;
    general_integer.setVerbose(false);
    general_integer.load(buildBranchingMip());
    CHECK_FALSE(general_integer.isTreePresolveBinaryLiteProfileActive());
}

TEST_CASE("MipSolver: in-tree presolve preserves feasible optimum", "[mip][presolve][tree]") {
    auto lp = buildBranchingMip();

    MipSolver with_tree_presolve;
    with_tree_presolve.setVerbose(false);
    with_tree_presolve.setCutsEnabled(false);
    with_tree_presolve.setPresolve(false);
    with_tree_presolve.setTreePresolveEnabled(true);
    with_tree_presolve.setSearchProfile(SearchProfile::Stable);
    with_tree_presolve.load(lp);
    const auto a = with_tree_presolve.solve();

    MipSolver without_tree_presolve;
    without_tree_presolve.setVerbose(false);
    without_tree_presolve.setCutsEnabled(false);
    without_tree_presolve.setPresolve(false);
    without_tree_presolve.setTreePresolveEnabled(false);
    without_tree_presolve.setSearchProfile(SearchProfile::Stable);
    without_tree_presolve.load(lp);
    const auto b = without_tree_presolve.solve();

    REQUIRE(a.status == Status::Optimal);
    REQUIRE(b.status == Status::Optimal);
    CHECK_THAT(a.objective, WithinAbs(b.objective, 1e-9));
}

TEST_CASE("MipSolver: presolve does not double-count objective offset",
          "[mip][presolve][objective]") {
    const auto lp = buildPresolveOffsetRegressionMip();

    auto solve_with = [&](bool presolve) {
        MipSolver solver;
        solver.setVerbose(false);
        solver.setCutsEnabled(false);
        solver.setTreePresolveEnabled(false);
        solver.setSearchProfile(SearchProfile::Stable);
        solver.setNumThreads(1);
        solver.setPresolve(presolve);
        solver.load(lp);
        return solver.solve();
    };

    const auto on = solve_with(true);
    const auto off = solve_with(false);

    REQUIRE(on.status == Status::Optimal);
    REQUIRE(off.status == Status::Optimal);
    REQUIRE(on.solution.size() == 3);
    REQUIRE(off.solution.size() == 3);

    CHECK_THAT(on.objective, WithinAbs(100.0, 1e-6));
    CHECK_THAT(off.objective, WithinAbs(100.0, 1e-6));
    CHECK_THAT(on.objective, WithinAbs(off.objective, 1e-9));
    CHECK_THAT(on.solution[0], WithinAbs(1.0, 1e-9));
    CHECK_THAT(off.solution[0], WithinAbs(1.0, 1e-9));
}
