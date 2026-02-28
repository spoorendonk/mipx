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

static LpProblem buildSymmetryBranchingMip() {
    LpProblem lp;
    lp.name = "symmetry_branching_mip";
    lp.sense = Sense::Minimize;
    lp.num_cols = 2;
    lp.obj = {-1.0, -1.0};
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {1.0, 1.0};
    lp.col_type = {VarType::Binary, VarType::Binary};
    lp.col_names = {"x0", "x1"};

    lp.num_rows = 1;
    lp.row_lower = {-kInf};
    lp.row_upper = {1.5};
    lp.row_names = {"sum"};

    std::vector<Triplet> trips = {
        {0, 0, 1.0}, {0, 1, 1.0},
    };
    lp.matrix = SparseMatrix(1, 2, std::move(trips));
    return lp;
}

static LpProblem buildLpLightProbeMip() {
    constexpr Index n = 12;
    LpProblem lp;
    lp.name = "parallel_lplight_probe_mip";
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

TEST_CASE("Parallel: deterministic heuristic mode is reproducible",
          "[parallel][tbb][heuristics]") {
    auto lp = buildBranchingMip();

    MipSolver solver_a;
    solver_a.setVerbose(false);
    solver_a.setNumThreads(4);
    solver_a.setCutsEnabled(false);
    solver_a.setParallelMode(ParallelMode::Deterministic);
    solver_a.setHeuristicSeed(77);
    solver_a.setSearchProfile(SearchProfile::Stable);
    solver_a.load(lp);
    const auto a = solver_a.solve();

    MipSolver solver_b;
    solver_b.setVerbose(false);
    solver_b.setNumThreads(4);
    solver_b.setCutsEnabled(false);
    solver_b.setParallelMode(ParallelMode::Deterministic);
    solver_b.setHeuristicSeed(77);
    solver_b.setSearchProfile(SearchProfile::Stable);
    solver_b.load(lp);
    const auto b = solver_b.solve();

    REQUIRE(a.status == Status::Optimal);
    REQUIRE(b.status == Status::Optimal);
    CHECK(a.nodes == b.nodes);
    CHECK(a.lp_iterations == b.lp_iterations);
    CHECK_THAT(a.objective, WithinAbs(b.objective, 1e-9));
    CHECK_THAT(a.work_units, WithinAbs(b.work_units, 1e-9));
    REQUIRE(a.solution.size() == b.solution.size());
    for (std::size_t i = 0; i < a.solution.size(); ++i) {
        CHECK_THAT(a.solution[i], WithinAbs(b.solution[i], 1e-9));
    }
}

TEST_CASE("Parallel: deterministic heuristic mode preserves objective across thread counts",
          "[parallel][tbb][heuristics]") {
    auto lp = buildBranchingMip();

    MipSolver single_thread;
    single_thread.setVerbose(false);
    single_thread.setNumThreads(1);
    single_thread.setCutsEnabled(false);
    single_thread.setParallelMode(ParallelMode::Deterministic);
    single_thread.setHeuristicSeed(77);
    single_thread.setSearchProfile(SearchProfile::Stable);
    single_thread.load(lp);
    const auto one = single_thread.solve();

    MipSolver four_threads;
    four_threads.setVerbose(false);
    four_threads.setNumThreads(4);
    four_threads.setCutsEnabled(false);
    four_threads.setParallelMode(ParallelMode::Deterministic);
    four_threads.setHeuristicSeed(77);
    four_threads.setSearchProfile(SearchProfile::Stable);
    four_threads.load(lp);
    const auto four = four_threads.solve();

    REQUIRE(one.status == Status::Optimal);
    REQUIRE(four.status == Status::Optimal);
    CHECK_THAT(one.objective, WithinAbs(four.objective, 1e-9));
    REQUIRE(one.solution.size() == four.solution.size());
    for (std::size_t i = 0; i < one.solution.size(); ++i) {
        CHECK_THAT(one.solution[i], WithinAbs(four.solution[i], 1e-9));
    }
}

TEST_CASE("Parallel: deterministic mode with symmetry is reproducible",
          "[parallel][tbb][heuristics][symmetry]") {
    auto lp = buildSymmetryBranchingMip();

    MipSolver solver_a;
    solver_a.setVerbose(false);
    solver_a.setNumThreads(4);
    solver_a.setCutsEnabled(false);
    solver_a.setPresolve(false);
    solver_a.setSymmetryEnabled(true);
    solver_a.setParallelMode(ParallelMode::Deterministic);
    solver_a.setHeuristicSeed(101);
    solver_a.setSearchProfile(SearchProfile::Stable);
    solver_a.load(lp);
    const auto a = solver_a.solve();

    MipSolver solver_b;
    solver_b.setVerbose(false);
    solver_b.setNumThreads(4);
    solver_b.setCutsEnabled(false);
    solver_b.setPresolve(false);
    solver_b.setSymmetryEnabled(true);
    solver_b.setParallelMode(ParallelMode::Deterministic);
    solver_b.setHeuristicSeed(101);
    solver_b.setSearchProfile(SearchProfile::Stable);
    solver_b.load(lp);
    const auto b = solver_b.solve();

    REQUIRE(a.status == Status::Optimal);
    REQUIRE(b.status == Status::Optimal);
    CHECK(a.nodes == b.nodes);
    CHECK(a.lp_iterations == b.lp_iterations);
    CHECK_THAT(a.objective, WithinAbs(b.objective, 1e-9));
    CHECK_THAT(a.work_units, WithinAbs(b.work_units, 1e-9));
    REQUIRE(a.solution.size() == b.solution.size());
    for (std::size_t i = 0; i < a.solution.size(); ++i) {
        CHECK_THAT(a.solution[i], WithinAbs(b.solution[i], 1e-9));
    }
    CHECK(solver_a.getSymmetryStats().cuts_applied);
    CHECK(solver_b.getSymmetryStats().cuts_applied);
}

TEST_CASE("Parallel: opportunistic heuristic mode remains valid",
          "[parallel][tbb][heuristics]") {
    auto lp = buildBranchingMip();

    MipSolver solver;
    solver.setVerbose(false);
    solver.setNumThreads(4);
    solver.setCutsEnabled(false);
    solver.setParallelMode(ParallelMode::Opportunistic);
    solver.setHeuristicSeed(99);
    solver.load(lp);
    const auto result = solver.solve();

    CHECK((result.status == Status::Optimal ||
           result.status == Status::NodeLimit ||
           result.status == Status::TimeLimit));
}

TEST_CASE("Parallel: deterministic multi-thread pre-root LP-free is reproducible",
          "[parallel][tbb][heuristics][preroot][deterministic]") {
    auto lp = buildBranchingMip();

    auto run_once = [&]() {
        MipSolver solver;
        solver.setVerbose(false);
        solver.setNumThreads(4);
        solver.setCutsEnabled(false);
        solver.setPresolve(false);
        solver.setNodeLimit(1);
        solver.setParallelMode(ParallelMode::Deterministic);
        solver.setHeuristicSeed(4242);
        solver.setPreRootLpFreeEnabled(true);
        solver.setPreRootLpLightEnabled(false);
        solver.setPreRootPortfolioEnabled(false);
        solver.setPreRootLpFreeEarlyStop(false);
        solver.setPreRootLpFreeMaxRounds(8);
        solver.setPreRootLpFreeWorkBudget(1.0e6);
        solver.load(lp);
        auto result = solver.solve();
        return std::make_pair(result, solver.getPreRootStats());
    };

    const auto [a_result, a_stats] = run_once();
    const auto [b_result, b_stats] = run_once();

    CHECK((a_result.status == Status::NodeLimit || a_result.status == Status::Optimal));
    CHECK((b_result.status == Status::NodeLimit || b_result.status == Status::Optimal));
    CHECK(a_result.nodes == b_result.nodes);
    CHECK(a_result.lp_iterations == b_result.lp_iterations);
    CHECK_THAT(a_result.objective, WithinAbs(b_result.objective, 1e-9));
    CHECK_THAT(a_result.work_units, WithinAbs(b_result.work_units, 1e-9));
    REQUIRE(a_result.solution.size() == b_result.solution.size());
    for (std::size_t i = 0; i < a_result.solution.size(); ++i) {
        CHECK_THAT(a_result.solution[i], WithinAbs(b_result.solution[i], 1e-9));
    }
    CHECK(a_stats.enabled);
    CHECK(b_stats.enabled);
    CHECK(a_stats.calls == b_stats.calls);
    CHECK(a_stats.fj_calls == b_stats.fj_calls);
    CHECK(a_stats.fpr_calls == b_stats.fpr_calls);
    CHECK(a_stats.local_mip_calls == b_stats.local_mip_calls);
    CHECK_THAT(a_stats.work_units, WithinAbs(b_stats.work_units, 1e-9));
}

TEST_CASE("Parallel: deterministic multi-thread pre-root LP-light is reproducible",
          "[parallel][tbb][heuristics][preroot][lplight][deterministic]") {
    auto lp = buildLpLightProbeMip();

    auto run_once = [&]() {
        MipSolver solver;
        solver.setVerbose(false);
        solver.setNumThreads(4);
        solver.setCutsEnabled(false);
        solver.setPresolve(false);
        solver.setNodeLimit(1);
        solver.setParallelMode(ParallelMode::Deterministic);
        solver.setHeuristicSeed(5151);
        solver.setPreRootLpFreeEnabled(false);
        solver.setPreRootLpLightEnabled(true);
        solver.setPreRootPortfolioEnabled(false);
        solver.setPreRootLpFreeEarlyStop(false);
        solver.setPreRootLpFreeMaxRounds(1);
        solver.setPreRootLpFreeWorkBudget(1.0e9);
        solver.load(lp);
        auto result = solver.solve();
        return std::make_pair(result, solver.getPreRootStats());
    };

    const auto [a_result, a_stats] = run_once();
    const auto [b_result, b_stats] = run_once();

    CHECK((a_result.status == Status::NodeLimit || a_result.status == Status::Optimal));
    CHECK((b_result.status == Status::NodeLimit || b_result.status == Status::Optimal));
    CHECK(a_result.nodes == b_result.nodes);
    CHECK(a_result.lp_iterations == b_result.lp_iterations);
    CHECK_THAT(a_result.objective, WithinAbs(b_result.objective, 1e-9));
    CHECK_THAT(a_result.work_units, WithinAbs(b_result.work_units, 1e-9));
    REQUIRE(a_result.solution.size() == b_result.solution.size());
    for (std::size_t i = 0; i < a_result.solution.size(); ++i) {
        CHECK_THAT(a_result.solution[i], WithinAbs(b_result.solution[i], 1e-9));
    }
    CHECK(a_stats.enabled);
    CHECK(b_stats.enabled);
    CHECK(a_stats.lp_light_enabled == b_stats.lp_light_enabled);
    CHECK(a_stats.lp_light_available == b_stats.lp_light_available);
    CHECK(a_stats.lp_light_calls == b_stats.lp_light_calls);
    CHECK(a_stats.lp_light_fpr_calls == b_stats.lp_light_fpr_calls);
    CHECK(a_stats.lp_light_diving_calls == b_stats.lp_light_diving_calls);
    CHECK_THAT(a_stats.work_units, WithinAbs(b_stats.work_units, 1e-9));
}

TEST_CASE("Parallel: deterministic multi-thread pre-root adaptive request is forced fixed and reproducible",
          "[parallel][tbb][heuristics][preroot][deterministic][portfolio]") {
    auto lp = buildLpLightProbeMip();

    auto run_once = [&]() {
        MipSolver solver;
        solver.setVerbose(false);
        solver.setNumThreads(4);
        solver.setCutsEnabled(false);
        solver.setPresolve(false);
        solver.setNodeLimit(1);
        solver.setParallelMode(ParallelMode::Deterministic);
        solver.setHeuristicSeed(7777);
        solver.setPreRootLpFreeEnabled(true);
        solver.setPreRootLpLightEnabled(true);
        solver.setPreRootPortfolioEnabled(true);  // request adaptive
        solver.setPreRootLpFreeEarlyStop(false);
        solver.setPreRootLpFreeMaxRounds(12);
        solver.setPreRootLpFreeWorkBudget(1.0e9);
        solver.load(lp);
        auto result = solver.solve();
        return std::make_pair(result, solver.getPreRootStats());
    };

    const auto [a_result, a_stats] = run_once();
    const auto [b_result, b_stats] = run_once();

    CHECK((a_result.status == Status::NodeLimit || a_result.status == Status::Optimal));
    CHECK((b_result.status == Status::NodeLimit || b_result.status == Status::Optimal));
    CHECK(a_result.nodes == b_result.nodes);
    CHECK(a_result.lp_iterations == b_result.lp_iterations);
    CHECK_THAT(a_result.objective, WithinAbs(b_result.objective, 1e-9));
    CHECK_THAT(a_result.work_units, WithinAbs(b_result.work_units, 1e-9));
    REQUIRE(a_result.solution.size() == b_result.solution.size());
    for (std::size_t i = 0; i < a_result.solution.size(); ++i) {
        CHECK_THAT(a_result.solution[i], WithinAbs(b_result.solution[i], 1e-9));
    }

    CHECK(a_stats.enabled);
    CHECK(b_stats.enabled);
    CHECK_FALSE(a_stats.portfolio_enabled);
    CHECK_FALSE(b_stats.portfolio_enabled);
    CHECK(a_stats.calls == b_stats.calls);
    CHECK(a_stats.fj_calls == b_stats.fj_calls);
    CHECK(a_stats.fpr_calls == b_stats.fpr_calls);
    CHECK(a_stats.local_mip_calls == b_stats.local_mip_calls);
    CHECK(a_stats.lp_light_calls == b_stats.lp_light_calls);
    CHECK(a_stats.lp_light_fpr_calls == b_stats.lp_light_fpr_calls);
    CHECK(a_stats.lp_light_diving_calls == b_stats.lp_light_diving_calls);
    CHECK_THAT(a_stats.work_units, WithinAbs(b_stats.work_units, 1e-9));
}

TEST_CASE("Parallel: deterministic multi-thread pre-root work-budget cutoff is reproducible",
          "[parallel][tbb][heuristics][preroot][deterministic][work_budget]") {
    auto lp = buildLpLightProbeMip();

    auto run_once = [&]() {
        MipSolver solver;
        solver.setVerbose(false);
        solver.setNumThreads(4);
        solver.setCutsEnabled(false);
        solver.setPresolve(false);
        solver.setNodeLimit(1);
        solver.setParallelMode(ParallelMode::Deterministic);
        solver.setHeuristicSeed(9090);
        solver.setPreRootLpFreeEnabled(true);
        solver.setPreRootLpLightEnabled(true);
        solver.setPreRootPortfolioEnabled(true);  // forced fixed in deterministic MT
        solver.setPreRootLpFreeEarlyStop(false);
        solver.setPreRootLpFreeMaxRounds(1000);
        solver.setPreRootLpFreeWorkBudget(100.0);
        solver.load(lp);
        auto result = solver.solve();
        return std::make_pair(result, solver.getPreRootStats());
    };

    const auto [a_result, a_stats] = run_once();
    const auto [b_result, b_stats] = run_once();

    CHECK((a_result.status == Status::NodeLimit || a_result.status == Status::Optimal));
    CHECK((b_result.status == Status::NodeLimit || b_result.status == Status::Optimal));
    CHECK(a_result.nodes == b_result.nodes);
    CHECK(a_result.lp_iterations == b_result.lp_iterations);
    CHECK_THAT(a_result.objective, WithinAbs(b_result.objective, 1e-9));
    CHECK_THAT(a_result.work_units, WithinAbs(b_result.work_units, 1e-9));
    REQUIRE(a_result.solution.size() == b_result.solution.size());
    for (std::size_t i = 0; i < a_result.solution.size(); ++i) {
        CHECK_THAT(a_result.solution[i], WithinAbs(b_result.solution[i], 1e-9));
    }

    CHECK(a_stats.enabled);
    CHECK(b_stats.enabled);
    CHECK_FALSE(a_stats.portfolio_enabled);
    CHECK_FALSE(b_stats.portfolio_enabled);
    CHECK(a_stats.calls == b_stats.calls);
    CHECK(a_stats.fj_calls == b_stats.fj_calls);
    CHECK(a_stats.fpr_calls == b_stats.fpr_calls);
    CHECK(a_stats.local_mip_calls == b_stats.local_mip_calls);
    CHECK(a_stats.lp_light_calls == b_stats.lp_light_calls);
    CHECK(a_stats.lp_light_fpr_calls == b_stats.lp_light_fpr_calls);
    CHECK(a_stats.lp_light_diving_calls == b_stats.lp_light_diving_calls);
    CHECK_THAT(a_stats.work_units, WithinAbs(b_stats.work_units, 1e-9));
}

TEST_CASE("Parallel: deterministic mode reports reproducible work-units metric",
          "[parallel][tbb][heuristics][deterministic][reporting]") {
    auto lp = buildLpLightProbeMip();

    auto run_once = [&]() {
        MipSolver solver;
        solver.setVerbose(false);
        solver.setNumThreads(4);
        solver.setCutsEnabled(false);
        solver.setPresolve(false);
        solver.setNodeLimit(1);
        solver.setParallelMode(ParallelMode::Deterministic);
        solver.setHeuristicSeed(321);
        solver.setPreRootLpFreeEnabled(true);
        solver.setPreRootLpLightEnabled(true);
        solver.setPreRootPortfolioEnabled(true);
        solver.setPreRootLpFreeEarlyStop(false);
        solver.setPreRootLpFreeMaxRounds(400);
        solver.setPreRootLpFreeWorkBudget(100.0);
        solver.load(lp);
        return solver.solve();
    };

    const auto a = run_once();
    const auto b = run_once();

    CHECK((a.status == Status::NodeLimit || a.status == Status::Optimal));
    CHECK((b.status == Status::NodeLimit || b.status == Status::Optimal));
    CHECK_THAT(a.work_units, WithinAbs(b.work_units, 1e-9));
    CHECK(a.time_seconds >= 0.0);
    CHECK(b.time_seconds >= 0.0);
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
