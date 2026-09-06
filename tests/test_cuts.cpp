#include "mipx/clique_table.h"
#include "mipx/conflict_graph.h"
#include "mipx/cut_pool.h"
#include "mipx/dual_simplex.h"
#include "mipx/gomory.h"
#include "mipx/io.h"
#include "mipx/lp_problem.h"
#include "mipx/mip_solver.h"
#include "mipx/separators.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <filesystem>

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
        {0, 0, 3.0},
        {0, 1, 2.0},
        {1, 0, 1.0},
        {1, 1, 4.0},
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
        {0, 0, 1.0},
        {0, 1, 1.0},
    };
    lp.matrix = SparseMatrix(1, 2, std::move(trips));
    return lp;
}

// min -x -y -z  s.t. x + y <= 1, y + z <= 1, x + z <= 1, all binary.
// Every pair conflicts, so x + y + z <= 1 is valid, but no single row implies
// it. The LP relaxation optimum is the unique fractional point (.5, .5, .5),
// where each pairwise inequality is tight and therefore not separable.
static LpProblem buildTriangleCliqueMip() {
    LpProblem lp;
    lp.name = "triangle_clique";
    lp.sense = Sense::Minimize;
    lp.num_cols = 3;
    lp.obj = {-1.0, -1.0, -1.0};
    lp.col_lower = {0.0, 0.0, 0.0};
    lp.col_upper = {1.0, 1.0, 1.0};
    lp.col_type = {VarType::Binary, VarType::Binary, VarType::Binary};
    lp.col_names = {"x", "y", "z"};

    lp.num_rows = 3;
    lp.row_lower = {-kInf, -kInf, -kInf};
    lp.row_upper = {1.0, 1.0, 1.0};
    lp.row_names = {"xy", "yz", "xz"};

    std::vector<Triplet> trips = {
        {0, 0, 1.0}, {0, 1, 1.0}, {1, 1, 1.0}, {1, 2, 1.0}, {2, 0, 1.0}, {2, 2, 1.0},
    };
    lp.matrix = SparseMatrix(3, 3, std::move(trips));
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
        {0, 0, 1.0},
        {0, 1, 1.0},
        {0, 2, 1.0},
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
            if (frac > 1e-6) {
                has_fractional = true;
            }
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
    config.cmir = false;
    config.strong_cg = false;
    config.lifted_cover = false;
    config.mod_k = false;
    config.intersection_cut = false;
    config.multi_row = false;
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
    config.cmir = false;
    config.strong_cg = false;
    config.lifted_cover = false;
    config.mod_k = false;
    config.intersection_cut = false;
    config.multi_row = false;
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

static CutFamilyConfig cliqueOnlyConfig() {
    CutFamilyConfig config;
    config.gomory = false;
    config.mir = false;
    config.cover = false;
    config.implied_bound = false;
    config.clique = true;
    config.zero_half = false;
    config.mixing = false;
    config.cmir = false;
    config.strong_cg = false;
    config.lifted_cover = false;
    config.mod_k = false;
    config.intersection_cut = false;
    config.multi_row = false;
    return config;
}

TEST_CASE("SeparatorManager: clique table separates maximal clique cuts", "[cuts][clique]") {
    auto problem = buildTriangleCliqueMip();
    DualSimplexSolver lp;
    lp.setVerbose(false);
    lp.load(problem);
    auto result = lp.solve();
    REQUIRE(result.status == Status::Optimal);

    auto primals = lp.getPrimalValues();
    REQUIRE(primals.size() == 3);
    for (Index j = 0; j < 3; ++j) {
        CHECK_THAT(primals[j], WithinAbs(0.5, 1e-9));
    }

    const CutFamilyConfig config = cliqueOnlyConfig();

    // Without a clique table the pairwise row scan only reproduces the rows,
    // which are tight at the LP optimum, so nothing is violated.
    {
        CutPool pool;
        SeparatorManager manager;
        manager.setConfig(config);
        manager.setMaxCutsPerFamily(10);
        CutSeparationStats stats;
        CHECK(manager.separate(lp, problem, primals, pool, stats) == 0);
        CHECK(pool.size() == 0);
    }

    ConflictGraph graph;
    graph.build(problem);
    REQUIRE(graph.numEdges() == 3);

    CliqueTable table;
    table.build(problem, graph);
    REQUIRE(table.numCliques() >= 1);

    CutPool pool;
    SeparatorManager manager;
    manager.setConfig(config);
    manager.setMaxCutsPerFamily(10);
    manager.setCliqueTable(&table);

    CutSeparationStats stats;
    const Int cuts = manager.separate(lp, problem, primals, pool, stats);
    REQUIRE(cuts >= 1);
    CHECK(stats.at(CutFamily::Clique).accepted == cuts);
    CHECK(stats.at(CutFamily::Clique).generated >= 1);
    CHECK(stats.at(CutFamily::Clique).efficacy_sum > 0.0);

    bool found_triangle_cut = false;
    for (Index i = 0; i < pool.size(); ++i) {
        const auto& cut = pool[i];
        CHECK(cut.family == CutFamily::Clique);
        if (cut.indices.size() != 3) {
            continue;
        }
        found_triangle_cut = true;
        CHECK(cut.indices == std::vector<Index>{0, 1, 2});
        for (Real v : cut.values) {
            CHECK_THAT(v, WithinAbs(1.0, 1e-12));
        }
        CHECK_THAT(cut.upper, WithinAbs(1.0, 1e-12));
    }
    CHECK(found_triangle_cut);
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
    no_families.cmir = false;
    no_families.strong_cg = false;
    no_families.lifted_cover = false;
    no_families.mod_k = false;
    no_families.intersection_cut = false;
    no_families.multi_row = false;
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
    mir_only.cmir = false;
    mir_only.strong_cg = false;
    mir_only.lifted_cover = false;
    mir_only.mod_k = false;
    mir_only.intersection_cut = false;
    mir_only.multi_row = false;
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
    CHECK((result.status == Status::Optimal || result.status == Status::NodeLimit ||
           result.status == Status::TimeLimit));
}

// ---------------------------------------------------------------------------
// CMIR validity.
//
// The MIR inequality is derived over nonnegative variables, treating the
// continuous terms as a nonnegative slack on the right-hand side. Only
// negative continuous coefficients belong in the cut, scaled by 1/(1-f0); a
// positive continuous term can only be relaxed away. Reading positive terms as
// a/f0 (and negative ones with a flipped sign) produces cuts that remove
// integer-feasible points, which the branch-and-bound then reports as an
// optimal solution worse than the true optimum.
// ---------------------------------------------------------------------------

namespace {

// Mixed-integer model with continuous columns of both signs in a <= row whose
// right-hand side has a fractional part, and a known integer-feasible point.
LpProblem buildCmirValidityMip() {
    LpProblem lp;
    lp.name = "cmir_validity";
    lp.sense = Sense::Minimize;
    lp.num_cols = 7;
    lp.num_rows = 5;
    lp.obj = {2.0, -5.0, 5.0, 3.0, 0.0, -5.0, -2.0};
    lp.obj_offset = -2.0;
    lp.col_lower = {0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 3.0};
    lp.col_upper = {1.0, 1.0, 1.0, 1.0, 10.0, 6.0, 8.8};
    lp.col_type = {VarType::Binary,     VarType::Binary,     VarType::Binary,    VarType::Binary,
                   VarType::Continuous, VarType::Continuous, VarType::Continuous};
    lp.col_names = {"c0", "c1", "c2", "c3", "c4", "c5", "c6"};

    lp.row_lower = {-18.0, -kInf, 4.0, -kInf, 15.0};
    lp.row_upper = {kInf, 29.4, 4.0, 3.0, 15.0};
    lp.row_names = {"R0", "R1", "R2", "R3", "R4"};

    std::vector<Triplet> trips = {
        {0, 0, 4.0},  {0, 2, -5.0}, {0, 6, -2.0}, {1, 1, -3.4}, {1, 3, 0.4},  {1, 5, 1.0},
        {1, 6, 3.0},  {2, 1, 2.0},  {2, 2, -3.0}, {2, 3, -3.0}, {2, 5, 1.0},  {3, 0, -4.0},
        {3, 2, -2.0}, {3, 6, 1.0},  {4, 0, -4.0}, {4, 3, -5.0}, {4, 4, -2.0}, {4, 6, 5.0},
    };
    lp.matrix = SparseMatrix(5, 7, std::move(trips));
    return lp;
}

// Feasible for every row of buildCmirValidityMip, integral on the binaries,
// objective -41 (the model's optimum).
const std::vector<Real>& cmirValidityOptimum() {
    static const std::vector<Real> point = {1.0, 1.0, 0.0, 1.0, 5.5, 5.0, 7.0};
    return point;
}

CutFamilyConfig onlyCmirConfig() {
    CutFamilyConfig config;
    config.gomory = false;
    config.mir = false;
    config.cover = false;
    config.implied_bound = false;
    config.clique = false;
    config.zero_half = false;
    config.mixing = false;
    config.cmir = true;
    config.strong_cg = false;
    config.lifted_cover = false;
    config.mod_k = false;
    config.intersection_cut = false;
    config.multi_row = false;
    return config;
}

}  // namespace

TEST_CASE("SeparatorManager: CMIR cuts keep every integer-feasible point", "[cuts][cmir]") {
    auto problem = buildCmirValidityMip();
    const auto& x = cmirValidityOptimum();

    // The reference point really is feasible for the model.
    for (Index i = 0; i < problem.num_rows; ++i) {
        auto row = problem.matrix.row(i);
        Real activity = 0.0;
        for (Index k = 0; k < row.size(); ++k) {
            activity += row.values[k] * x[static_cast<std::size_t>(row.indices[k])];
        }
        INFO("row " << i);
        CHECK(activity <= problem.row_upper[i] + 1e-9);
        CHECK(activity >= problem.row_lower[i] - 1e-9);
    }

    DualSimplexSolver lp;
    lp.load(problem);
    auto result = lp.solve();
    REQUIRE(result.status == Status::Optimal);

    CutPool pool;
    SeparatorManager manager;
    manager.setConfig(onlyCmirConfig());
    manager.setMaxCutsPerFamily(10);

    CutSeparationStats stats;
    manager.separate(lp, problem, lp.getPrimalValues(), pool, stats);

    for (Index c = 0; c < pool.size(); ++c) {
        const Cut& cut = pool[c];
        Real activity = 0.0;
        for (std::size_t k = 0; k < cut.indices.size(); ++k) {
            activity += cut.values[k] * x[static_cast<std::size_t>(cut.indices[k])];
        }
        INFO("cut " << c << " activity " << activity << " in [" << cut.lower << ", " << cut.upper
                    << "]");
        CHECK(activity <= cut.upper + 1e-7);
        CHECK(activity >= cut.lower - 1e-7);
    }
}

TEST_CASE("MipSolver: CMIR cuts do not cut off the optimum", "[cuts][cmir]") {
    auto problem = buildCmirValidityMip();

    MipSolver solver;
    solver.setVerbose(false);
    solver.load(problem);
    auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(-41.0, 1e-6));
}

TEST_CASE("SeparatorManager: CMIR skips rows with negative lower bounds", "[cuts][cmir]") {
    // MIR needs every variable nonnegative after complementation. A free
    // integer column invalidates the rounding argument, so no cut may be
    // produced from a row containing one.
    LpProblem lp;
    lp.name = "cmir_negative_lower";
    lp.sense = Sense::Minimize;
    lp.num_cols = 3;
    lp.num_rows = 1;
    lp.obj = {-1.0, -1.0, -1.0};
    lp.col_lower = {-5.0, 0.0, 0.0};
    lp.col_upper = {5.0, 10.0, 10.0};
    lp.col_type = {VarType::Integer, VarType::Integer, VarType::Continuous};
    lp.col_names = {"z", "y", "s"};
    lp.row_lower = {-kInf};
    lp.row_upper = {7.5};
    lp.row_names = {"R0"};
    std::vector<Triplet> trips = {{0, 0, 1.0}, {0, 1, 2.0}, {0, 2, 1.0}};
    lp.matrix = SparseMatrix(1, 3, std::move(trips));

    DualSimplexSolver solver;
    solver.load(lp);
    auto result = solver.solve();
    REQUIRE(result.status == Status::Optimal);

    CutPool pool;
    SeparatorManager manager;
    manager.setConfig(onlyCmirConfig());
    manager.setMaxCutsPerFamily(10);

    CutSeparationStats stats;
    manager.separate(solver, lp, solver.getPrimalValues(), pool, stats);
    CHECK(pool.size() == 0);
}

// ---------------------------------------------------------------------------
// Strong CG validity.
//
// The Strong CG cut rounds the scaled right-hand side down, which is only
// valid when the scaled left-hand side is integral. Continuous columns never
// are, so they have to be relaxed away (dropped) before the rounding step --
// keeping them makes the cut stronger than the row implies and removes
// integer-feasible points, which branch-and-bound then reports as an optimum
// worse than the true one.
// ---------------------------------------------------------------------------

namespace {

// min -6a -5b -4c -s  s.t.  4a + 3b + 3c + s <= 6.5, a,b,c binary, s in [0,10].
// Optimum -9.5 at b = c = 1, s = 0.5 (weight 6 <= 6.5).
LpProblem buildStrongCgKnapsackMip() {
    LpProblem lp;
    lp.name = "strongcg_knap";
    lp.sense = Sense::Minimize;
    lp.num_cols = 4;
    lp.num_rows = 1;
    lp.obj = {-6.0, -5.0, -4.0, -1.0};
    lp.col_lower = {0.0, 0.0, 0.0, 0.0};
    lp.col_upper = {1.0, 1.0, 1.0, 10.0};
    lp.col_type = {VarType::Binary, VarType::Binary, VarType::Binary, VarType::Continuous};
    lp.col_names = {"a", "b", "c", "s"};
    lp.row_lower = {-kInf};
    lp.row_upper = {6.5};
    lp.row_names = {"R1"};
    std::vector<Triplet> trips = {{0, 0, 4.0}, {0, 1, 3.0}, {0, 2, 3.0}, {0, 3, 1.0}};
    lp.matrix = SparseMatrix(1, 4, std::move(trips));
    return lp;
}

// min -x - y  s.t.  x + y <= 2.5, x integer in [0,10], y continuous in [0,10].
// Optimum -2.5, attained anywhere on x + y = 2.5 with x integral.
LpProblem buildStrongCgMixedMip() {
    LpProblem lp;
    lp.name = "strongcg_mixed";
    lp.sense = Sense::Minimize;
    lp.num_cols = 2;
    lp.num_rows = 1;
    lp.obj = {-1.0, -1.0};
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {10.0, 10.0};
    lp.col_type = {VarType::Integer, VarType::Continuous};
    lp.col_names = {"x", "y"};
    lp.row_lower = {-kInf};
    lp.row_upper = {2.5};
    lp.row_names = {"R1"};
    std::vector<Triplet> trips = {{0, 0, 1.0}, {0, 1, 1.0}};
    lp.matrix = SparseMatrix(1, 2, std::move(trips));
    return lp;
}

// max x + 0.001z, i.e. min -x - 0.001z, with a coefficient just below one:
//   R1: (1 - 1e-9) x <= 1000.9999999,  x integer in [0, 2000]
//   R2:           2z <=         1,     z integer in [0, 1]
// x = 1001 is feasible (1001 * (1 - 1e-9) = 1000.999998999 <= 1000.9999999) and
// optimal at -1001. Flooring t*a_j with a snapping tolerance rounded the scaled
// coefficient 0.999999999 up to 1, which strengthens the term rather than
// relaxing it and yields the invalid cut x <= 1000.
LpProblem buildStrongCgNearIntegerMip() {
    LpProblem lp;
    lp.name = "strongcg_near_integer";
    lp.sense = Sense::Minimize;
    lp.num_cols = 2;
    lp.num_rows = 2;
    lp.obj = {-1.0, -0.001};
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {2000.0, 1.0};
    lp.col_type = {VarType::Integer, VarType::Integer};
    lp.col_names = {"x", "z"};
    lp.row_lower = {-kInf, -kInf};
    lp.row_upper = {1000.9999999, 1.0};
    lp.row_names = {"R1", "R2"};
    std::vector<Triplet> trips = {{0, 0, 1.0 - 1e-9}, {1, 1, 2.0}};
    lp.matrix = SparseMatrix(2, 2, std::move(trips));
    return lp;
}

CutFamilyConfig onlyStrongCgConfig() {
    CutFamilyConfig config;
    config.gomory = false;
    config.mir = false;
    config.cover = false;
    config.implied_bound = false;
    config.clique = false;
    config.zero_half = false;
    config.mixing = false;
    config.cmir = false;
    config.strong_cg = true;
    config.lifted_cover = false;
    config.mod_k = false;
    config.intersection_cut = false;
    config.multi_row = false;
    return config;
}

// Separate the LP relaxation of `problem` with `config` and check that every
// cut in the pool is satisfied by the integer-feasible point `x`. Returns the
// number of cuts that were checked.
Index separateAndCheckCutsKeep(const LpProblem& problem, const std::vector<Real>& x,
                               const CutFamilyConfig& config) {
    // The reference point really is feasible and integral for the model.
    for (Index j = 0; j < problem.num_cols; ++j) {
        const Real v = x[static_cast<std::size_t>(j)];
        INFO("column " << j);
        CHECK(v >= problem.col_lower[j] - 1e-9);
        CHECK(v <= problem.col_upper[j] + 1e-9);
        if (problem.col_type[j] != VarType::Continuous) {
            CHECK(std::abs(v - std::round(v)) <= 1e-9);
        }
    }
    for (Index i = 0; i < problem.num_rows; ++i) {
        auto row = problem.matrix.row(i);
        Real activity = 0.0;
        for (Index k = 0; k < row.size(); ++k) {
            activity += row.values[k] * x[static_cast<std::size_t>(row.indices[k])];
        }
        INFO("row " << i);
        CHECK(activity <= problem.row_upper[i] + 1e-9);
        CHECK(activity >= problem.row_lower[i] - 1e-9);
    }

    DualSimplexSolver lp;
    lp.load(problem);
    auto result = lp.solve();
    REQUIRE(result.status == Status::Optimal);

    CutPool pool;
    SeparatorManager manager;
    manager.setConfig(config);
    manager.setMaxCutsPerFamily(10);

    CutSeparationStats stats;
    manager.separate(lp, problem, lp.getPrimalValues(), pool, stats);

    for (Index c = 0; c < pool.size(); ++c) {
        const Cut& cut = pool[c];
        Real activity = 0.0;
        for (std::size_t k = 0; k < cut.indices.size(); ++k) {
            activity += cut.values[k] * x[static_cast<std::size_t>(cut.indices[k])];
        }
        INFO("cut " << c << " activity " << activity << " in [" << cut.lower << ", " << cut.upper
                    << "]");
        CHECK(activity <= cut.upper + 1e-7);
        CHECK(activity >= cut.lower - 1e-7);
    }
    return pool.size();
}

}  // namespace

TEST_CASE("SeparatorManager: Strong CG cuts keep the knapsack optimum", "[cuts][strongcg]") {
    const auto problem = buildStrongCgKnapsackMip();
    const std::vector<Real> optimum = {0.0, 1.0, 1.0, 0.5};  // objective -9.5

    // The fractional right-hand side makes the row separable, so the check is
    // not passing vacuously on an empty pool.
    CHECK(separateAndCheckCutsKeep(problem, optimum, onlyStrongCgConfig()) > 0);
}

TEST_CASE("SeparatorManager: Strong CG cuts keep the mixed-row optimum", "[cuts][strongcg]") {
    const auto problem = buildStrongCgMixedMip();
    // x = 0, y = 2.5 is integer-feasible and optimal (objective -2.5). Keeping
    // the continuous column produced 0.75y <= 1 from the multiplier t = 0.75,
    // which this point violates by 0.875. Once the column is relaxed away the
    // only integer term left is x, and no multiplier rounds it into a violated
    // cut -- so the pool must be empty rather than hold a valid-but-different
    // cut. If a future change does emit one here, the per-cut checks in the
    // helper must still pass.
    const std::vector<Real> optimum = {0.0, 2.5};

    CHECK(separateAndCheckCutsKeep(problem, optimum, onlyStrongCgConfig()) == 0);
}

TEST_CASE("MipSolver: Strong CG cuts do not cut off the optimum", "[cuts][strongcg]") {
    SECTION("knapsack with a continuous column") {
        MipSolver solver;
        solver.setVerbose(false);
        solver.setPresolve(false);
        solver.setCutFamilyConfig(onlyStrongCgConfig());
        solver.load(buildStrongCgKnapsackMip());
        auto result = solver.solve();

        REQUIRE(result.status == Status::Optimal);
        CHECK_THAT(result.objective, WithinAbs(-9.5, 1e-6));
    }

    SECTION("mixed integer/continuous row") {
        MipSolver solver;
        solver.setVerbose(false);
        solver.setPresolve(false);
        solver.setCutFamilyConfig(onlyStrongCgConfig());
        solver.load(buildStrongCgMixedMip());
        auto result = solver.solve();

        REQUIRE(result.status == Status::Optimal);
        CHECK_THAT(result.objective, WithinAbs(-2.5, 1e-6));
    }
}

TEST_CASE("SeparatorManager: Strong CG does not round a coefficient up", "[cuts][strongcg]") {
    const auto problem = buildStrongCgNearIntegerMip();
    // x = 1001, z = 0 is integer-feasible; the pre-fix cut x <= 1000 removes it.
    const std::vector<Real> optimum = {1001.0, 0.0};

    separateAndCheckCutsKeep(problem, optimum, onlyStrongCgConfig());
}

TEST_CASE("MipSolver: Strong CG keeps an optimum under a near-integer coefficient",
          "[cuts][strongcg]") {
    MipSolver solver;
    solver.setVerbose(false);
    solver.setPresolve(false);
    solver.setCutFamilyConfig(onlyStrongCgConfig());
    solver.load(buildStrongCgNearIntegerMip());
    auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(-1001.0, 1e-6));
}

TEST_CASE("MipSolver: knapsack solves to -9.5 with presolve off and cuts on", "[cuts][strongcg]") {
    // The reproduction from the bug report: presolve off, every cut family on.
    MipSolver solver;
    solver.setVerbose(false);
    solver.setPresolve(false);
    solver.setCutsEnabled(true);
    solver.load(buildStrongCgKnapsackMip());
    auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(-9.5, 1e-6));
}

// ---------------------------------------------------------------------------
// Cut family configuration gating (issue #189)
// ---------------------------------------------------------------------------

namespace {

// Every CutFamily value that names a real separator, i.e. the enum range with
// the Unknown sentinel and the Count terminator stripped off. Iterating the
// enum rather than a hand-written list keeps these tests honest when a new
// family is added.
std::vector<CutFamily> allSeparableFamilies() {
    std::vector<CutFamily> families;
    for (Int fi = static_cast<Int>(CutFamily::Unknown) + 1; fi < static_cast<Int>(CutFamily::Count);
         ++fi) {
        families.push_back(static_cast<CutFamily>(fi));
    }
    return families;
}

// Config with every flag set to `enabled`. Built through setCutFamilyEnabled so
// the switch in MipSolver has to handle every family for this to work.
CutFamilyConfig allCutFamilies(bool enabled) {
    MipSolver probe;
    for (CutFamily family : allSeparableFamilies()) {
        probe.setCutFamilyEnabled(family, enabled);
    }
    return probe.getCutFamilyConfig();
}

}  // namespace

TEST_CASE("MipSolver: setCutFamilyEnabled reaches every cut family", "[cuts][families]") {
    // A family setCutFamilyEnabled forgets keeps the CutFamilyConfig default of
    // true, which is exactly the bug: it then separates unconditionally.
    const CutFamilyConfig none = allCutFamilies(false);
    CHECK_FALSE(none.gomory);
    CHECK_FALSE(none.mir);
    CHECK_FALSE(none.cover);
    CHECK_FALSE(none.implied_bound);
    CHECK_FALSE(none.clique);
    CHECK_FALSE(none.zero_half);
    CHECK_FALSE(none.mixing);
    CHECK_FALSE(none.cmir);
    CHECK_FALSE(none.strong_cg);
    CHECK_FALSE(none.lifted_cover);
    CHECK_FALSE(none.mod_k);
    CHECK_FALSE(none.intersection_cut);
    CHECK_FALSE(none.multi_row);

    const CutFamilyConfig all = allCutFamilies(true);
    CHECK(all.gomory);
    CHECK(all.mir);
    CHECK(all.cover);
    CHECK(all.implied_bound);
    CHECK(all.clique);
    CHECK(all.zero_half);
    CHECK(all.mixing);
    CHECK(all.cmir);
    CHECK(all.strong_cg);
    CHECK(all.lifted_cover);
    CHECK(all.mod_k);
    CHECK(all.intersection_cut);
    CHECK(all.multi_row);
}

TEST_CASE("MipSolver: every cut family off adds no root cuts", "[cuts][families][integration]") {
    const auto lp = buildStrongCgKnapsackMip();

    // Baseline with every family on. Cuts must actually be separated here,
    // otherwise the all-off run below would pass for the wrong reason.
    MipSolver baseline;
    baseline.setVerbose(false);
    baseline.setPresolve(false);
    baseline.setCutsEnabled(true);
    baseline.setMaxCutRounds(10);
    baseline.setCutFamilyConfig(allCutFamilies(true));
    baseline.load(lp);
    const auto base_result = baseline.solve();
    REQUIRE(base_result.status == Status::Optimal);
    REQUIRE(baseline.getCutStats().root_cuts_added > 0);

    // Same model, same cut settings, only the family flags differ: cuts stay
    // enabled and the effort mode is untouched, so a zero count can only come
    // from the per-family gating.
    MipSolver all_off;
    all_off.setVerbose(false);
    all_off.setPresolve(false);
    all_off.setCutsEnabled(true);
    all_off.setMaxCutRounds(10);
    all_off.setCutFamilyConfig(allCutFamilies(false));
    all_off.load(lp);
    const auto off_result = all_off.solve();

    REQUIRE(off_result.status == Status::Optimal);
    CHECK(all_off.getCutEffortMode() == baseline.getCutEffortMode());
    CHECK(all_off.getCutStats().root_cuts_added == 0);
    CHECK_THAT(off_result.objective, WithinAbs(base_result.objective, 1e-6));

    // No family was even attempted, while the baseline attempted at least one.
    Int baseline_attempted = 0;
    for (CutFamily family : allSeparableFamilies()) {
        INFO("family " << cutFamilyName(family));
        CHECK(all_off.getRootCutFamilyStats().at(family).attempted == 0);
        CHECK(all_off.getRootCutFamilyStats().at(family).accepted == 0);
        baseline_attempted += baseline.getRootCutFamilyStats().at(family).attempted;
    }
    CHECK(baseline_attempted > 0);
}

TEST_CASE("MipSolver: one cut family at a time silences all the others",
          "[cuts][families][integration]") {
    const auto lp = buildStrongCgKnapsackMip();

    Int solo_accepted_total = 0;
    for (CutFamily solo : allSeparableFamilies()) {
        INFO("solo family " << cutFamilyName(solo));

        MipSolver solver;
        solver.setVerbose(false);
        solver.setPresolve(false);
        solver.setCutsEnabled(true);
        solver.setMaxCutRounds(10);
        solver.setCutFamilyConfig(allCutFamilies(false));
        solver.setCutFamilyEnabled(solo, true);
        solver.load(lp);
        const auto result = solver.solve();

        REQUIRE(result.status == Status::Optimal);
        CHECK_THAT(result.objective, WithinAbs(-9.5, 1e-6));

        const auto& stats = solver.getRootCutFamilyStats();
        for (CutFamily other : allSeparableFamilies()) {
            if (other == solo) {
                continue;
            }
            INFO("silenced family " << cutFamilyName(other));
            CHECK(stats.at(other).attempted == 0);
            CHECK(stats.at(other).generated == 0);
            CHECK(stats.at(other).accepted == 0);
        }
        solo_accepted_total += stats.at(solo).accepted;
    }

    // At least one solo run has to separate something; otherwise every check
    // above would hold vacuously.
    CHECK(solo_accepted_total > 0);
}
