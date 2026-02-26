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

TEST_CASE("MipSolver: deterministic heuristic mode reproduces with same seed",
          "[mip][heuristics]") {
    auto lp = buildRootFractionalHeuristicMip();

    MipSolver solver_a;
    solver_a.setVerbose(false);
    solver_a.setCutsEnabled(false);
    solver_a.setNodeLimit(1);
    solver_a.setHeuristicMode(HeuristicRuntimeMode::Deterministic);
    solver_a.setHeuristicSeed(1234);
    solver_a.load(lp);
    auto result_a = solver_a.solve();

    MipSolver solver_b;
    solver_b.setVerbose(false);
    solver_b.setCutsEnabled(false);
    solver_b.setNodeLimit(1);
    solver_b.setHeuristicMode(HeuristicRuntimeMode::Deterministic);
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
    solver.setHeuristicMode(HeuristicRuntimeMode::Opportunistic);
    solver.setHeuristicSeed(7);
    solver.load(lp);
    const auto result = solver.solve();

    CHECK((result.status == Status::NodeLimit || result.status == Status::Optimal));
    REQUIRE(!result.solution.empty());
    CHECK(result.solution[0] >= 0.0);
    CHECK(result.solution[0] <= 4.6 + 1e-6);
}
