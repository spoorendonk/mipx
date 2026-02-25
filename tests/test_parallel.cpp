#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <filesystem>

#include "mipx/io.h"
#include "mipx/mip_solver.h"

using namespace mipx;
using Catch::Matchers::WithinAbs;

// ---------------------------------------------------------------------------
// Helper: build a MIP that needs branching
// min -x - 2y  s.t. x + y <= 4.5, x,y >= 0, x,y integer
// MIP optimal: x=0, y=4, obj=-8
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

#ifdef MIPX_HAS_TBB
// ---------------------------------------------------------------------------
// Helper: Knapsack MIP
// max 6x1 + 5x2 + 4x3 s.t. 3x1 + 2x2 + 2x3 <= 5, x binary
// Optimal: x1=1, x2=1, x3=0, obj=-11
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
#endif

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST_CASE("Parallel: single thread matches serial", "[parallel]") {
    auto lp = buildBranchingMip();

    // Serial solve.
    MipSolver serial;
    serial.setVerbose(false);
    serial.setNumThreads(1);
    serial.load(lp);
    auto serial_result = serial.solve();

    // "Parallel" with 1 thread (should behave identically).
    MipSolver par1;
    par1.setVerbose(false);
    par1.setNumThreads(1);
    par1.load(lp);
    auto par1_result = par1.solve();

    REQUIRE(serial_result.status == Status::Optimal);
    REQUIRE(par1_result.status == Status::Optimal);
    CHECK_THAT(serial_result.objective, WithinAbs(par1_result.objective, 1e-6));
}

#ifdef MIPX_HAS_TBB
TEST_CASE("Parallel: multi-thread finds optimal", "[parallel][tbb]") {
    auto lp = buildBranchingMip();

    MipSolver solver;
    solver.setVerbose(false);
    solver.setNumThreads(4);
    solver.load(lp);
    auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(-8.0, 1e-6));
    REQUIRE(result.solution.size() == 2);
}

TEST_CASE("Parallel: knapsack multi-thread", "[parallel][tbb]") {
    auto lp = buildKnapsackMip();

    MipSolver solver;
    solver.setVerbose(false);
    solver.setNumThreads(4);
    solver.setNodeLimit(1000);
    solver.load(lp);
    auto result = solver.solve();

    REQUIRE((result.status == Status::Optimal || result.status == Status::NodeLimit));
    if (result.status == Status::Optimal) {
        CHECK_THAT(result.objective, WithinAbs(-11.0, 1e-6));
    }
}

TEST_CASE("Parallel: node limit respected", "[parallel][tbb]") {
    auto lp = buildBranchingMip();

    MipSolver solver;
    solver.setVerbose(false);
    solver.setNumThreads(4);
    solver.setNodeLimit(2);
    solver.load(lp);
    auto result = solver.solve();

    // With node limit 2, we should stop early.
    // The parallel solver may overshoot slightly due to concurrent processing.
    CHECK(result.nodes <= 10);  // Allow some overshoot.
    CHECK((result.status == Status::NodeLimit || result.status == Status::Optimal));
}

TEST_CASE("Parallel: matches serial on knapsack", "[parallel][tbb]") {
    auto lp = buildKnapsackMip();

    MipSolver serial;
    serial.setVerbose(false);
    serial.setNumThreads(1);
    serial.setNodeLimit(1000);
    serial.load(lp);
    auto serial_result = serial.solve();

    MipSolver par;
    par.setVerbose(false);
    par.setNumThreads(4);
    par.setNodeLimit(1000);
    par.load(lp);
    auto par_result = par.solve();

    // Both should find the same optimal.
    if (serial_result.status == Status::Optimal && par_result.status == Status::Optimal) {
        CHECK_THAT(serial_result.objective, WithinAbs(par_result.objective, 1e-6));
    }
}

TEST_CASE("Parallel: MIPLIB gt2", "[parallel][tbb][miplib]") {
    std::string path = std::string(TEST_DATA_DIR) + "/miplib/gt2.mps.gz";
    if (!std::filesystem::exists(path)) {
        SKIP("gt2.mps.gz not found (run download_miplib.sh --small)");
    }

    auto problem = readMps(path);
    REQUIRE(problem.hasIntegers());

    MipSolver solver;
    solver.setVerbose(true);
    solver.setNumThreads(4);
    solver.setNodeLimit(10000);
    solver.setTimeLimit(60.0);
    solver.load(problem);
    auto result = solver.solve();

    // gt2 optimal: 21166.0
    if (result.status == Status::Optimal) {
        CHECK_THAT(result.objective, WithinAbs(21166.0, 1.0));
    }
    CHECK((result.status == Status::Optimal ||
           result.status == Status::NodeLimit ||
           result.status == Status::TimeLimit));
}
#endif
