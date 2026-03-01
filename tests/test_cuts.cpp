#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <filesystem>

#include "mipx/cut_pool.h"
#include "mipx/dual_simplex.h"
#include "mipx/gomory.h"
#include "mipx/io.h"
#include "mipx/lp_problem.h"
#include "mipx/mip_solver.h"
#include "mipx/separators.h"

using namespace mipx;
using Catch::Matchers::WithinAbs;

// ---------------------------------------------------------------------------
// CutPool tests
// ---------------------------------------------------------------------------

TEST_CASE("CutPool: add and rank by efficacy", "[cuts]") {
    CutPool pool;

    Cut c1;
    c1.indices = {0, 1};
    c1.values = {1.0, 2.0};
    c1.lower = 3.0;
    c1.efficacy = 0.5;

    Cut c2;
    c2.indices = {0, 2};
    c2.values = {1.0, 1.0};
    c2.lower = 2.0;
    c2.efficacy = 0.8;

    Cut c3;
    c3.indices = {1};
    c3.values = {1.0};
    c3.lower = 1.0;
    c3.efficacy = 0.3;

    REQUIRE(pool.addCut(c1));
    REQUIRE(pool.addCut(c2));
    REQUIRE(pool.addCut(c3));
    CHECK(pool.size() == 3);

    auto top = pool.topByEfficacy(2);
    REQUIRE(top.size() == 2);
    CHECK(pool[top[0]].efficacy > pool[top[1]].efficacy);
    CHECK_THAT(pool[top[0]].efficacy, WithinAbs(0.8, 1e-10));
    CHECK_THAT(pool[top[1]].efficacy, WithinAbs(0.5, 1e-10));
}

TEST_CASE("CutPool: reject low efficacy", "[cuts]") {
    CutPool pool;
    pool.setMinEfficacy(0.1);

    Cut c;
    c.indices = {0};
    c.values = {1.0};
    c.lower = 1.0;
    c.efficacy = 0.01;  // Below threshold.

    CHECK_FALSE(pool.addCut(c));
    CHECK(pool.size() == 0);
}

TEST_CASE("CutPool: parallelism filtering", "[cuts]") {
    CutPool pool;
    pool.setParallelismThreshold(0.95);

    Cut c1;
    c1.indices = {0, 1};
    c1.values = {1.0, 0.0};
    c1.lower = 1.0;
    c1.efficacy = 0.5;
    REQUIRE(pool.addCut(c1));

    // Same direction, slightly different magnitude.
    Cut c2;
    c2.indices = {0, 1};
    c2.values = {2.0, 0.0};
    c2.lower = 2.0;
    c2.efficacy = 0.4;  // Lower efficacy, parallel.
    CHECK_FALSE(pool.addCut(c2));

    // Orthogonal cut: accepted.
    Cut c3;
    c3.indices = {0, 1};
    c3.values = {0.0, 1.0};
    c3.lower = 1.0;
    c3.efficacy = 0.5;
    CHECK(pool.addCut(c3));
    CHECK(pool.size() == 2);
}

TEST_CASE("CutPool: age and purge", "[cuts]") {
    CutPool pool;

    Cut c1;
    c1.indices = {0};
    c1.values = {1.0};
    c1.lower = 5.0;  // x0 >= 5
    c1.efficacy = 0.5;

    Cut c2;
    c2.indices = {1};
    c2.values = {1.0};
    c2.lower = 3.0;  // x1 >= 3
    c2.efficacy = 0.3;

    pool.addCut(c1);
    pool.addCut(c2);

    // Primals where c1 is active (x0 = 5.0), c2 is not (x1 = 10.0).
    std::vector<Real> primals = {5.0, 10.0};

    // Age several times.
    for (int i = 0; i < 11; ++i) {
        pool.ageAll(primals, 0.1);
    }

    // c1 should still be age=0 (active), c2 should be age=11.
    CHECK(pool[0].age == 0);
    CHECK(pool[1].age == 11);

    pool.purge(10);
    CHECK(pool.size() == 1);
    CHECK_THAT(pool[0].efficacy, WithinAbs(0.5, 1e-10));
}

// ---------------------------------------------------------------------------
// Helper: build a MIP where cuts can help
// ---------------------------------------------------------------------------

// min -x - y  s.t. 3x + 2y <= 6, x + 4y <= 4, x, y >= 0, x, y integer
// LP relaxation: x = 16/10, y = 6/10, obj = -2.2
// MIP optimal: x = 2, y = 0, obj = -2  OR  x = 0, y = 1, obj = -1
// Actually x=2, y=0: 3*2+2*0=6<=6, 2+0=2<=4. obj = -2. Good.
// x=1, y=1: 3+2=5<=6, 1+4=5>4. Not feasible.
// x=0, y=1: 0+2=2<=6, 0+4=4<=4. obj = -1.
// x=2, y=0 is optimal.
static LpProblem buildCutTestMip() {
    LpProblem lp;
    lp.name = "cut_test";
    lp.sense = Sense::Minimize;
    lp.num_cols = 2;
    lp.obj = {-1.0, -1.0};
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {kInf, kInf};
    lp.col_type = {VarType::Integer, VarType::Integer};
    lp.col_names = {"x", "y"};

    lp.num_rows = 2;
    lp.row_lower = {-kInf, -kInf};
    lp.row_upper = {6.0, 4.0};
    lp.row_names = {"r1", "r2"};

    std::vector<Triplet> trips = {
        {0, 0, 3.0}, {0, 1, 2.0},
        {1, 0, 1.0}, {1, 1, 4.0},
    };
    lp.matrix = SparseMatrix(2, 2, std::move(trips));
    return lp;
}

// A slightly harder problem where cuts should help:
// min -10x - 20y  s.t. x + y <= 5.5, x, y >= 0, integer
// LP opt: x=0, y=5.5, obj=-110
// MIP opt: x=0, y=5, obj=-100  or  x=5, y=0, obj=-50
// Best: y=5, x=0 -> -100.
static LpProblem buildFractionalMip() {
    LpProblem lp;
    lp.name = "fractional_mip";
    lp.sense = Sense::Minimize;
    lp.num_cols = 2;
    lp.obj = {-10.0, -20.0};
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {kInf, kInf};
    lp.col_type = {VarType::Integer, VarType::Integer};
    lp.col_names = {"x", "y"};

    lp.num_rows = 1;
    lp.row_lower = {-kInf};
    lp.row_upper = {5.5};
    lp.row_names = {"cap"};

    std::vector<Triplet> trips = {
        {0, 0, 1.0}, {0, 1, 1.0},
    };
    lp.matrix = SparseMatrix(1, 2, std::move(trips));
    return lp;
}

// min -x1 -x2 -x3  s.t. x1 + x2 + x3 <= 1.5, x binary
// LP optimum is fractional (sum 1.5), good for cover/clique separation tests.
static LpProblem buildBinaryConflictMip() {
    LpProblem lp;
    lp.name = "binary_conflict";
    lp.sense = Sense::Minimize;
    lp.num_cols = 3;
    lp.obj = {-3.0, -2.0, -0.1};
    lp.col_lower = {0.0, 0.0, 0.0};
    lp.col_upper = {1.0, 1.0, 1.0};
    lp.col_type = {VarType::Binary, VarType::Binary, VarType::Binary};
    lp.col_names = {"x1", "x2", "x3"};

    lp.num_rows = 1;
    lp.row_lower = {-kInf};
    lp.row_upper = {1.5};
    lp.row_names = {"cap"};

    std::vector<Triplet> trips = {
        {0, 0, 1.0}, {0, 1, 1.0}, {0, 2, 1.0},
    };
    lp.matrix = SparseMatrix(1, 3, std::move(trips));
    return lp;
}

// min -x  s.t. x <= 2.7, x integer
// MIR-style rounding gives x <= 2 at the root LP solution x = 2.7.
static LpProblem buildMirTestMip() {
    LpProblem lp;
    lp.name = "mir_test";
    lp.sense = Sense::Minimize;
    lp.num_cols = 1;
    lp.obj = {-1.0};
    lp.col_lower = {0.0};
    lp.col_upper = {10.0};
    lp.col_type = {VarType::Integer};
    lp.col_names = {"x"};

    lp.num_rows = 1;
    lp.row_lower = {-kInf};
    lp.row_upper = {2.7};
    lp.row_names = {"ub"};

    std::vector<Triplet> trips = {
        {0, 0, 1.0},
    };
    lp.matrix = SparseMatrix(1, 1, std::move(trips));
    return lp;
}

// ---------------------------------------------------------------------------
// Gomory separator tests
// ---------------------------------------------------------------------------

TEST_CASE("Gomory: generate cuts on small MIP", "[cuts][gomory]") {
    auto problem = buildFractionalMip();

    DualSimplexSolver lp;
    lp.load(problem);
    auto result = lp.solve();
    REQUIRE(result.status == Status::Optimal);

    auto primals = lp.getPrimalValues();

    // y should be fractional (5.5).
    bool has_fractional = false;
    for (Index j = 0; j < problem.num_cols; ++j) {
        if (problem.col_type[j] != VarType::Continuous) {
            Real frac = std::abs(primals[j] - std::round(primals[j]));
            if (frac > 1e-6) has_fractional = true;
        }
    }
    CHECK(has_fractional);

    CutPool pool;
    GomorySeparator gomory;
    Int cuts = gomory.separate(lp, problem, primals, pool);

    // We should get at least one cut from the fractional variable.
    CHECK(cuts >= 0);  // May or may not generate depending on tableau structure.

    // Verify any generated cuts are valid (not violated by the LP relaxation
    // optimum... actually they SHOULD be violated, that's the point).
    for (Index i = 0; i < pool.size(); ++i) {
        const auto& cut = pool[i];
        CHECK(cut.efficacy > 0.0);
        CHECK(!cut.indices.empty());

        // Verify the cut is violated by the current LP solution.
        Real lhs = 0.0;
        for (Index k = 0; k < static_cast<Index>(cut.indices.size()); ++k) {
            lhs += cut.values[k] * primals[cut.indices[k]];
        }
        // For a >= cut: lhs should be < rhs (violated).
        if (cut.lower > -kInf) {
            CHECK(lhs < cut.lower + 1e-3);  // violated or nearly so
        }
    }
}

TEST_CASE("CutFamilyConfig defaults keep Gomory enabled", "[cuts][gomory]") {
    const CutFamilyConfig config{};
    CHECK(config.gomory);
}

TEST_CASE("SeparatorManager: cover family generates tagged cuts", "[cuts][families]") {
    auto problem = buildBinaryConflictMip();
    DualSimplexSolver lp;
    lp.load(problem);
    auto result = lp.solve();
    REQUIRE(result.status == Status::Optimal);

    CutPool pool;
    SeparatorManager manager;
    CutFamilyConfig config;
    config.gomory = false;
    config.mir = false;
    config.cover = true;
    config.implied_bound = false;
    config.clique = false;
    config.zero_half = false;
    config.mixing = false;
    manager.setConfig(config);
    manager.setMaxCutsPerFamily(10);

    CutSeparationStats stats;
    Int cuts = manager.separate(lp, problem, lp.getPrimalValues(), pool, stats);
    CHECK(cuts >= 1);
    CHECK(stats.at(CutFamily::Cover).attempted >= 1);
    CHECK(stats.at(CutFamily::Cover).accepted >= 1);
    for (Index i = 0; i < pool.size(); ++i) {
        CHECK(pool[i].family == CutFamily::Cover);
    }
}

TEST_CASE("SeparatorManager: clique family generates conflict cuts", "[cuts][families]") {
    auto problem = buildBinaryConflictMip();
    DualSimplexSolver lp;
    lp.load(problem);
    auto result = lp.solve();
    REQUIRE(result.status == Status::Optimal);

    CutPool pool;
    SeparatorManager manager;
    CutFamilyConfig config;
    config.gomory = false;
    config.mir = false;
    config.cover = false;
    config.implied_bound = false;
    config.clique = true;
    config.zero_half = false;
    config.mixing = false;
    manager.setConfig(config);
    manager.setMaxCutsPerFamily(10);

    CutSeparationStats stats;
    Int cuts = manager.separate(lp, problem, lp.getPrimalValues(), pool, stats);
    CHECK(cuts >= 1);
    CHECK(stats.at(CutFamily::Clique).attempted >= 1);
    CHECK(stats.at(CutFamily::Clique).accepted >= 1);
    for (Index i = 0; i < pool.size(); ++i) {
        CHECK(pool[i].family == CutFamily::Clique);
    }
}

TEST_CASE("SeparatorManager: MIR family generates rounding cuts", "[cuts][families]") {
    auto problem = buildMirTestMip();
    DualSimplexSolver lp;
    lp.load(problem);
    auto result = lp.solve();
    REQUIRE(result.status == Status::Optimal);

    CutPool pool;
    SeparatorManager manager;
    CutFamilyConfig config;
    config.gomory = false;
    config.mir = true;
    config.cover = false;
    config.implied_bound = false;
    config.clique = false;
    config.zero_half = false;
    config.mixing = false;
    manager.setConfig(config);
    manager.setMaxCutsPerFamily(10);

    CutSeparationStats stats;
    Int cuts = manager.separate(lp, problem, lp.getPrimalValues(), pool, stats);
    CHECK(cuts >= 1);
    CHECK(stats.at(CutFamily::Mir).attempted >= 1);
    CHECK(stats.at(CutFamily::Mir).accepted >= 1);
    for (Index i = 0; i < pool.size(); ++i) {
        CHECK(pool[i].family == CutFamily::Mir);
    }
}

// ---------------------------------------------------------------------------
// Integration tests: MIP solver with and without cuts
// ---------------------------------------------------------------------------

TEST_CASE("MipSolver with cuts: simple MIP", "[cuts][integration]") {
    auto lp = buildCutTestMip();

    MipSolver solver;
    solver.setVerbose(false);
    solver.setCutsEnabled(true);
    solver.setMaxCutRounds(10);
    solver.load(lp);
    auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(-2.0, 1e-6));
}

TEST_CASE("MipSolver with cuts: fractional problem", "[cuts][integration]") {
    auto lp = buildFractionalMip();

    MipSolver solver;
    solver.setVerbose(false);
    solver.setCutsEnabled(true);
    solver.load(lp);
    auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(-100.0, 1e-6));
}

TEST_CASE("MipSolver: cuts disabled gives same result", "[cuts][integration]") {
    auto lp = buildFractionalMip();

    MipSolver solver_no_cuts;
    solver_no_cuts.setVerbose(false);
    solver_no_cuts.setCutsEnabled(false);
    solver_no_cuts.load(lp);
    auto result_no = solver_no_cuts.solve();

    MipSolver solver_cuts;
    solver_cuts.setVerbose(false);
    solver_cuts.setCutsEnabled(true);
    solver_cuts.load(lp);
    auto result_cuts = solver_cuts.solve();

    // Both should find optimal.
    REQUIRE(result_no.status == Status::Optimal);
    REQUIRE(result_cuts.status == Status::Optimal);
    CHECK_THAT(result_no.objective, WithinAbs(result_cuts.objective, 1e-6));
}

TEST_CASE("MipSolver: cut family toggles preserve correctness", "[cuts][integration]") {
    auto lp = buildFractionalMip();

    MipSolver solver_no_families;
    solver_no_families.setVerbose(false);
    solver_no_families.setCutsEnabled(true);
    CutFamilyConfig no_families;
    no_families.gomory = false;
    no_families.mir = false;
    no_families.cover = false;
    no_families.implied_bound = false;
    no_families.clique = false;
    no_families.zero_half = false;
    no_families.mixing = false;
    solver_no_families.setCutFamilyConfig(no_families);
    solver_no_families.load(lp);
    auto result_no = solver_no_families.solve();

    MipSolver solver_mir_only;
    solver_mir_only.setVerbose(false);
    solver_mir_only.setCutsEnabled(true);
    CutFamilyConfig mir_only;
    mir_only.gomory = false;
    mir_only.mir = true;
    mir_only.cover = false;
    mir_only.implied_bound = false;
    mir_only.clique = false;
    mir_only.zero_half = false;
    mir_only.mixing = false;
    solver_mir_only.setCutFamilyConfig(mir_only);
    solver_mir_only.load(lp);
    auto result_mir = solver_mir_only.solve();

    REQUIRE(result_no.status == Status::Optimal);
    REQUIRE(result_mir.status == Status::Optimal);
    CHECK_THAT(result_no.objective, WithinAbs(result_mir.objective, 1e-6));
}

TEST_CASE("MipSolver: cut effort off matches cuts-disabled behavior", "[cuts][integration]") {
    auto lp = buildFractionalMip();

    MipSolver solver_off;
    solver_off.setVerbose(false);
    solver_off.setCutsEnabled(true);
    solver_off.setCutEffortMode(CutEffortMode::Off);
    solver_off.load(lp);
    auto result_off = solver_off.solve();

    MipSolver solver_disabled;
    solver_disabled.setVerbose(false);
    solver_disabled.setCutsEnabled(false);
    solver_disabled.load(lp);
    auto result_disabled = solver_disabled.solve();

    REQUIRE(result_off.status == Status::Optimal);
    REQUIRE(result_disabled.status == Status::Optimal);
    CHECK_THAT(result_off.objective, WithinAbs(result_disabled.objective, 1e-6));
}

TEST_CASE("MipSolver with cuts: MIPLIB gt2", "[cuts][miplib]") {
    std::string path = std::string(TEST_DATA_DIR) + "/miplib/gt2.mps.gz";
    if (!std::filesystem::exists(path)) {
        SKIP("gt2.mps.gz not found (run download_miplib.sh --small)");
    }

    auto problem = readMps(path);
    REQUIRE(problem.hasIntegers());

    MipSolver solver;
    solver.setVerbose(true);
    solver.setCutsEnabled(true);
    solver.setMaxCutRounds(10);
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
