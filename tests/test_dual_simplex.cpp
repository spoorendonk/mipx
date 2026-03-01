#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <filesystem>

#include "mipx/dual_simplex.h"
#include "mipx/io.h"

using namespace mipx;
using Catch::Matchers::WithinAbs;
namespace fs = std::filesystem;

static std::string testDataDir() { return std::string(TEST_DATA_DIR); }

// ---------------------------------------------------------------------------
// Helper: build a small LP manually
// ---------------------------------------------------------------------------
static LpProblem buildTrivialLP() {
    // min x + y  s.t.  x + y >= 1,  x,y >= 0
    LpProblem lp;
    lp.name = "trivial";
    lp.sense = Sense::Minimize;
    lp.num_cols = 2;
    lp.obj = {1.0, 1.0};
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {kInf, kInf};
    lp.col_type = {VarType::Continuous, VarType::Continuous};
    lp.col_names = {"x", "y"};

    // One constraint: x + y >= 1  =>  1 <= x + y <= inf
    lp.num_rows = 1;
    lp.row_lower = {1.0};
    lp.row_upper = {kInf};
    lp.row_names = {"c1"};

    // Build constraint matrix (CSR): row 0 has entries (col 0, 1.0) and (col 1, 1.0)
    std::vector<Real> values = {1.0, 1.0};
    std::vector<Index> col_indices = {0, 1};
    std::vector<Index> row_starts = {0, 2};
    lp.matrix = SparseMatrix(1, 2, values, col_indices, row_starts);

    return lp;
}

// ---------------------------------------------------------------------------
// Test 1: Trivial LP
// ---------------------------------------------------------------------------
TEST_CASE("DualSimplex: trivial LP (min x+y s.t. x+y>=1)", "[dual_simplex]") {
    auto lp = buildTrivialLP();

    DualSimplexSolver solver;
    solver.load(lp);
    auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(1.0, 1e-6));

    auto primals = solver.getPrimalValues();
    REQUIRE(primals.size() == 2);
    // x + y should equal 1.0 at optimum
    CHECK_THAT(primals[0] + primals[1], WithinAbs(1.0, 1e-6));
}

// ---------------------------------------------------------------------------
// Test 2: Simple bounded LP
// ---------------------------------------------------------------------------
TEST_CASE("DualSimplex: simple bounded (min -x s.t. x<=5)", "[dual_simplex]") {
    LpProblem lp;
    lp.name = "bounded";
    lp.sense = Sense::Minimize;
    lp.num_cols = 1;
    lp.obj = {-1.0};
    lp.col_lower = {0.0};
    lp.col_upper = {kInf};
    lp.col_type = {VarType::Continuous};
    lp.col_names = {"x"};

    // x <= 5  =>  -inf <= x <= 5
    lp.num_rows = 1;
    lp.row_lower = {-kInf};
    lp.row_upper = {5.0};
    lp.row_names = {"c1"};

    std::vector<Real> values = {1.0};
    std::vector<Index> col_indices = {0};
    std::vector<Index> row_starts = {0, 1};
    lp.matrix = SparseMatrix(1, 1, values, col_indices, row_starts);

    DualSimplexSolver solver;
    solver.load(lp);
    auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(-5.0, 1e-6));

    auto primals = solver.getPrimalValues();
    REQUIRE(primals.size() == 1);
    CHECK_THAT(primals[0], WithinAbs(5.0, 1e-6));
}

// ---------------------------------------------------------------------------
// Test 3: Equality constraint
// ---------------------------------------------------------------------------
TEST_CASE("DualSimplex: equality constraint (min x+y s.t. x+y=4, bounded)", "[dual_simplex]") {
    LpProblem lp;
    lp.name = "equality";
    lp.sense = Sense::Minimize;
    lp.num_cols = 2;
    lp.obj = {1.0, 1.0};
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {3.0, 3.0};
    lp.col_type = {VarType::Continuous, VarType::Continuous};
    lp.col_names = {"x", "y"};

    // x + y = 4  =>  4 <= x + y <= 4
    lp.num_rows = 1;
    lp.row_lower = {4.0};
    lp.row_upper = {4.0};
    lp.row_names = {"eq"};

    std::vector<Real> values = {1.0, 1.0};
    std::vector<Index> col_indices = {0, 1};
    std::vector<Index> row_starts = {0, 2};
    lp.matrix = SparseMatrix(1, 2, values, col_indices, row_starts);

    DualSimplexSolver solver;
    solver.load(lp);
    auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(4.0, 1e-6));

    auto primals = solver.getPrimalValues();
    REQUIRE(primals.size() == 2);
    CHECK_THAT(primals[0] + primals[1], WithinAbs(4.0, 1e-6));
    CHECK(primals[0] >= -1e-6);
    CHECK(primals[0] <= 3.0 + 1e-6);
    CHECK(primals[1] >= -1e-6);
    CHECK(primals[1] <= 3.0 + 1e-6);
}

// ---------------------------------------------------------------------------
// Test: Infeasible LP
// ---------------------------------------------------------------------------
TEST_CASE("DualSimplex: infeasible LP", "[dual_simplex]") {
    // x >= 5 and x <= 3 with a single variable — contradictory via constraints.
    // min x s.t. x >= 5, x <= 3, x >= 0
    LpProblem lp;
    lp.name = "infeasible";
    lp.sense = Sense::Minimize;
    lp.num_cols = 1;
    lp.obj = {1.0};
    lp.col_lower = {0.0};
    lp.col_upper = {kInf};
    lp.col_type = {VarType::Continuous};
    lp.col_names = {"x"};

    // Row 0: x >= 5  =>  5 <= x <= inf
    // Row 1: x <= 3  =>  -inf <= x <= 3
    lp.num_rows = 2;
    lp.row_lower = {5.0, -kInf};
    lp.row_upper = {kInf, 3.0};
    lp.row_names = {"ge5", "le3"};

    std::vector<Real> values = {1.0, 1.0};
    std::vector<Index> col_indices = {0, 0};
    std::vector<Index> row_starts = {0, 1, 2};
    lp.matrix = SparseMatrix(2, 1, values, col_indices, row_starts);

    DualSimplexSolver solver;
    solver.load(lp);
    auto result = solver.solve();

    CHECK(result.status == Status::Infeasible);
}

// ---------------------------------------------------------------------------
// Test: Unbounded LP
// ---------------------------------------------------------------------------
TEST_CASE("DualSimplex: unbounded LP", "[dual_simplex]") {
    // min -x, x >= 0, no upper bound on x, no constraints limiting x.
    LpProblem lp;
    lp.name = "unbounded";
    lp.sense = Sense::Minimize;
    lp.num_cols = 1;
    lp.obj = {-1.0};
    lp.col_lower = {0.0};
    lp.col_upper = {kInf};
    lp.col_type = {VarType::Continuous};
    lp.col_names = {"x"};

    // No real constraints — just a dummy row that does not bound x:
    // y >= 0 where y is a different slack, but we need at least one row.
    // Actually: row 0: 0*x >= -1  (trivially satisfied, does not bound x)
    lp.num_rows = 1;
    lp.row_lower = {-1.0};
    lp.row_upper = {kInf};
    lp.row_names = {"dummy"};

    std::vector<Real> values = {0.0};
    std::vector<Index> col_indices = {0};
    std::vector<Index> row_starts = {0, 1};
    lp.matrix = SparseMatrix(1, 1, values, col_indices, row_starts);

    DualSimplexSolver solver;
    solver.load(lp);
    auto result = solver.solve();

    CHECK(result.status == Status::Unbounded);
}

// ---------------------------------------------------------------------------
// Netlib helpers
// ---------------------------------------------------------------------------
static void testNetlib(const std::string& name, Real expected_obj, Real rel_tol = 1e-6) {
    std::string path = testDataDir() + "/netlib/" + name + ".mps.gz";
    if (!fs::exists(path)) {
        SKIP("Netlib instance " + name + " not downloaded");
    }

    auto lp = readMps(path);
    DualSimplexSolver solver;
    solver.load(lp);
    auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);

    Real denom = std::max(1.0, std::abs(expected_obj));
    Real rel_err = std::abs(result.objective - expected_obj) / denom;
    CHECK(rel_err < rel_tol);
}

// ---------------------------------------------------------------------------
// Tests 4-7: Netlib instances
// ---------------------------------------------------------------------------
TEST_CASE("DualSimplex: netlib afiro", "[dual_simplex][netlib]") {
    testNetlib("afiro", -4.6475314286e+02);
}

TEST_CASE("DualSimplex: netlib sc50a", "[dual_simplex][netlib]") {
    testNetlib("sc50a", -6.4575077059e+01);
}

TEST_CASE("DualSimplex: netlib blend", "[dual_simplex][netlib]") {
    testNetlib("blend", -3.0812149846e+01);
}

TEST_CASE("DualSimplex: netlib adlittle", "[dual_simplex][netlib]") {
    testNetlib("adlittle", 2.2549496316e+05);
}

TEST_CASE("DualSimplex: netlib larger known-optimal LPs", "[dual_simplex][netlib][large]") {
    testNetlib("ship12l", 1.4701879193e+06, 1e-6);
    testNetlib("sierra", 1.5394362184e+07, 1e-6);
}

// ---------------------------------------------------------------------------
// Test 8: Solution feasibility check
// ---------------------------------------------------------------------------
TEST_CASE("DualSimplex: solution feasibility", "[dual_simplex]") {
    auto lp = buildTrivialLP();

    DualSimplexSolver solver;
    solver.load(lp);
    auto result = solver.solve();
    REQUIRE(result.status == Status::Optimal);

    auto primals = solver.getPrimalValues();
    REQUIRE(primals.size() == static_cast<std::size_t>(lp.num_cols));

    // Check variable bounds.
    for (Index j = 0; j < lp.num_cols; ++j) {
        CHECK(primals[j] >= lp.col_lower[j] - 1e-6);
        CHECK(primals[j] <= lp.col_upper[j] + 1e-6);
    }

    // Check constraints: row_lower <= A*x <= row_upper.
    // Compute A*x row by row using the sparse matrix.
    const auto& mat = lp.matrix;
    for (Index i = 0; i < lp.num_rows; ++i) {
        Real row_val = 0.0;
        auto rv = mat.row(i);
        for (Index k = 0; k < rv.size(); ++k) {
            row_val += rv.values[k] * primals[rv.indices[k]];
        }
        CHECK(row_val >= lp.row_lower[i] - 1e-6);
        CHECK(row_val <= lp.row_upper[i] + 1e-6);
    }
}

// ---------------------------------------------------------------------------
// Test: Basis recovery — verify getBasis() returns correct size
// ---------------------------------------------------------------------------
TEST_CASE("DualSimplex: basis recovery", "[dual_simplex]") {
    auto lp = buildTrivialLP();

    DualSimplexSolver solver;
    solver.load(lp);
    auto result = solver.solve();
    REQUIRE(result.status == Status::Optimal);

    auto basis = solver.getBasis();
    // Basis should have one entry per variable (structural + slacks = num_cols + num_rows).
    Index total_vars = lp.num_cols + lp.num_rows;
    REQUIRE(basis.size() == static_cast<std::size_t>(total_vars));

    // Count basic variables — should equal num_rows.
    Index num_basic = 0;
    for (auto s : basis) {
        if (s == BasisStatus::Basic) {
            ++num_basic;
        }
    }
    CHECK(num_basic == lp.num_rows);
}

// ---------------------------------------------------------------------------
// Test: Work units are positive after solving
// ---------------------------------------------------------------------------
TEST_CASE("DualSimplex: work units are positive", "[dual_simplex][work_units]") {
    auto lp = buildTrivialLP();

    DualSimplexSolver solver;
    solver.load(lp);
    auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    CHECK(result.work_units > 0.0);

    // Total work should also be accessible from the solver.
    CHECK(solver.workUnits().ticks() > 0);
}

TEST_CASE("DualSimplex: work units are deterministic", "[dual_simplex][work_units]") {
    auto lp = buildTrivialLP();

    // Solve twice, work should be identical.
    DualSimplexSolver solver1;
    solver1.load(lp);
    auto r1 = solver1.solve();

    DualSimplexSolver solver2;
    solver2.load(lp);
    auto r2 = solver2.solve();

    REQUIRE(r1.status == Status::Optimal);
    REQUIRE(r2.status == Status::Optimal);
    CHECK(r1.work_units == r2.work_units);
}

TEST_CASE("DualSimplex: work units scale with problem size", "[dual_simplex][work_units][netlib]") {
    // afiro is tiny, blend is larger — work should reflect that.
    std::string afiro_path = testDataDir() + "/netlib/afiro.mps.gz";
    std::string blend_path = testDataDir() + "/netlib/blend.mps.gz";
    if (!fs::exists(afiro_path) || !fs::exists(blend_path)) {
        SKIP("Netlib instances not downloaded");
    }

    DualSimplexSolver solver_a;
    solver_a.load(readMps(afiro_path));
    auto ra = solver_a.solve();

    DualSimplexSolver solver_b;
    solver_b.load(readMps(blend_path));
    auto rb = solver_b.solve();

    REQUIRE(ra.status == Status::Optimal);
    REQUIRE(rb.status == Status::Optimal);

    // blend has more rows/cols/nnz — should do more work.
    CHECK(rb.work_units > ra.work_units);
}

TEST_CASE("DualSimplex: runtime options can toggle pricing/refactorization paths",
          "[dual_simplex][options]") {
    auto lp = buildTrivialLP();

    DualSimplexSolver solver;
    DualSimplexOptions opts;
    opts.enable_partial_pricing = false;
    opts.enable_bfrt = true;
    opts.enable_adaptive_bfrt = false;
    opts.adaptive_bfrt_max_pinf = 33;
    opts.adaptive_bfrt_progress_window = 77;
    opts.enable_auto_bfrt_wide = false;
    opts.auto_bfrt_min_cols = 1234;
    opts.auto_bfrt_min_col_row_ratio = 5.5;
    opts.enable_adaptive_refactorization = false;
    opts.lu_update_limit = 321;
    opts.lu_ft_drop_tolerance = 1e-9;
    opts.partial_pricing_chunk_min = 8;
    opts.partial_pricing_full_scan_freq = 1;
    opts.adaptive_refactor_min_updates = 8;
    opts.adaptive_refactor_stall_pivots = 8;
    opts.primal_feasible_adaptive_refactor_stall_pivots = 222;
    opts.primal_feasible_adaptive_refactor_min_updates = 111;
    opts.primal_feasible_dual_progress_window = 333;
    opts.primal_feasible_refactor_cooldown = 444;
    opts.primal_feasible_dual_progress_improve_rel_tol = 2e-3;
    opts.max_solve_seconds = 12.5;
    opts.enable_dual_perturbation = false;
    opts.dual_perturbation_stall_pivots = 16;
    opts.dual_perturbation_magnitude = 5e-8;
    opts.enable_bound_perturbation = true;
    opts.bound_perturbation_stall_pivots = 64;
    opts.bound_perturbation_magnitude = 1e-6;
    opts.bound_perturbation_max_activations = 3;
    opts.enable_stall_restart = false;
    opts.stall_restart_pivots = 1234;
    opts.stall_restart_max_restarts = 4;
    opts.enable_idiot_crash = true;
    opts.idiot_crash_passes = 3;
    opts.idiot_crash_max_flips = 128;
    opts.idiot_crash_min_gain = 1e-7;
    opts.enable_structural_crash = false;
    opts.structural_crash_max_swaps = 7;
    opts.structural_crash_min_pivot = 1e-5;
    opts.enable_simd_kernels = false;
    opts.simd_min_length = 16;
    opts.enable_sip_parallel_candidates = true;
    opts.enable_sip_parallel_dual_scan = true;
    opts.sip_parallel_min_nonbasic = 1;
    opts.sip_parallel_grain = 32;
    opts.sip_parallel_min_threads = 2;
    opts.sip_parallel_disable_on_stall = true;
    opts.sip_parallel_stall_pivots = 4;
    opts.enable_sip_parallel_candidate_sort = true;
    opts.sip_parallel_sort_min_candidates = 32;
    opts.enable_sip_parallel_chuzr = true;
    opts.sip_parallel_min_rows = 1;
    opts.sip_parallel_row_grain = 16;
    solver.setOptions(opts);

    const auto& applied = solver.getOptions();
    CHECK_FALSE(applied.enable_partial_pricing);
    CHECK(applied.enable_bfrt);
    CHECK_FALSE(applied.enable_adaptive_bfrt);
    CHECK(applied.adaptive_bfrt_max_pinf == 33);
    CHECK(applied.adaptive_bfrt_progress_window == 77);
    CHECK_FALSE(applied.enable_auto_bfrt_wide);
    CHECK(applied.auto_bfrt_min_cols == 1234);
    CHECK(applied.auto_bfrt_min_col_row_ratio == 5.5);
    CHECK_FALSE(applied.enable_adaptive_refactorization);
    CHECK(applied.lu_update_limit == 321);
    CHECK(applied.lu_ft_drop_tolerance == 1e-9);
    CHECK(applied.primal_feasible_adaptive_refactor_stall_pivots == 222);
    CHECK(applied.primal_feasible_adaptive_refactor_min_updates == 111);
    CHECK(applied.primal_feasible_dual_progress_window == 333);
    CHECK(applied.primal_feasible_refactor_cooldown == 444);
    CHECK(applied.primal_feasible_dual_progress_improve_rel_tol == 2e-3);
    CHECK(applied.max_solve_seconds == 12.5);
    CHECK_FALSE(applied.enable_dual_perturbation);
    CHECK(applied.dual_perturbation_stall_pivots == 16);
    CHECK(applied.dual_perturbation_magnitude == 5e-8);
    CHECK(applied.enable_bound_perturbation);
    CHECK(applied.bound_perturbation_stall_pivots == 64);
    CHECK(applied.bound_perturbation_magnitude == 1e-6);
    CHECK(applied.bound_perturbation_max_activations == 3);
    CHECK_FALSE(applied.enable_stall_restart);
    CHECK(applied.stall_restart_pivots == 1234);
    CHECK(applied.stall_restart_max_restarts == 4);
    CHECK(applied.enable_idiot_crash);
    CHECK(applied.idiot_crash_passes == 3);
    CHECK(applied.idiot_crash_max_flips == 128);
    CHECK(applied.idiot_crash_min_gain == 1e-7);
    CHECK_FALSE(applied.enable_structural_crash);
    CHECK(applied.structural_crash_max_swaps == 7);
    CHECK(applied.structural_crash_min_pivot == 1e-5);
    CHECK_FALSE(applied.enable_simd_kernels);
    CHECK(applied.enable_sip_parallel_candidates);
    CHECK(applied.enable_sip_parallel_dual_scan);
    CHECK(applied.enable_sip_parallel_chuzr);
    CHECK(applied.partial_pricing_chunk_min == 8);
    CHECK(applied.partial_pricing_full_scan_freq == 1);
    CHECK(applied.simd_min_length == 16);
    CHECK(applied.sip_parallel_min_nonbasic == 1);
    CHECK(applied.sip_parallel_grain == 32);
    CHECK(applied.sip_parallel_min_threads == 2);
    CHECK(applied.sip_parallel_disable_on_stall);
    CHECK(applied.sip_parallel_stall_pivots == 4);
    CHECK(applied.enable_sip_parallel_candidate_sort);
    CHECK(applied.sip_parallel_sort_min_candidates == 32);
    CHECK(applied.sip_parallel_min_rows == 1);
    CHECK(applied.sip_parallel_row_grain == 16);

    solver.load(lp);
    auto result = solver.solve();
    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(1.0, 1e-6));
}

TEST_CASE("DualSimplex: max_solve_seconds enforces time limit", "[dual_simplex][time_limit]") {
    auto lp = buildTrivialLP();

    DualSimplexSolver solver;
    DualSimplexOptions opts = solver.getOptions();
    opts.max_solve_seconds = 0.0;
    solver.setOptions(opts);
    solver.load(lp);
    const auto result = solver.solve();

    CHECK(result.status == Status::TimeLimit);
}

TEST_CASE("DualSimplex: primal-feasible guardrails tolerate non-positive options",
          "[dual_simplex][options][guardrail]") {
    auto lp = buildTrivialLP();

    DualSimplexSolver solver;
    DualSimplexOptions opts = solver.getOptions();
    opts.primal_feasible_adaptive_refactor_stall_pivots = -7;
    opts.primal_feasible_adaptive_refactor_min_updates = -1;
    opts.primal_feasible_dual_progress_window = 0;
    opts.primal_feasible_refactor_cooldown = -9;
    opts.primal_feasible_dual_progress_improve_rel_tol = -1e-3;
    opts.lu_ft_drop_tolerance = -1.0;
    opts.auto_bfrt_min_cols = -123;
    opts.auto_bfrt_min_col_row_ratio = -5.0;
    solver.setOptions(opts);
    solver.load(lp);

    const auto result = solver.solve();
    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(1.0, 1e-6));
}

TEST_CASE("DualSimplex: enabled primal progress gate impacts degen3 run without errors",
          "[dual_simplex][netlib][time_limit][regression]") {
    std::string path = testDataDir() + "/netlib/degen3.mps.gz";
    if (!fs::exists(path)) {
        SKIP("Netlib instance degen3 not downloaded");
    }

    auto lp = readMps(path);

    DualSimplexSolver disabled_solver;
    DualSimplexOptions disabled_opts = disabled_solver.getOptions();
    disabled_opts.max_solve_seconds = 1.0;
    disabled_opts.primal_feasible_dual_progress_window = 0;
    disabled_opts.primal_feasible_refactor_cooldown = 0;
    disabled_solver.setOptions(disabled_opts);
    disabled_solver.load(lp);
    const auto disabled_result = disabled_solver.solve();
    REQUIRE(disabled_result.status != Status::Error);

    DualSimplexSolver enabled_solver;
    DualSimplexOptions enabled_opts = enabled_solver.getOptions();
    enabled_opts.max_solve_seconds = 1.0;
    enabled_opts.primal_feasible_dual_progress_window = 1;
    enabled_opts.primal_feasible_refactor_cooldown = 0;
    enabled_opts.primal_feasible_adaptive_refactor_stall_pivots = 1;
    enabled_opts.primal_feasible_adaptive_refactor_min_updates = 0;
    enabled_solver.setOptions(enabled_opts);
    enabled_solver.load(lp);
    const auto enabled_result = enabled_solver.solve();

    REQUIRE(enabled_result.status != Status::Error);
    CHECK((enabled_result.status == Status::TimeLimit ||
           enabled_result.status == Status::Optimal));
    CHECK(enabled_result.iterations > 100);
    CHECK(disabled_result.work_units > 0.0);
    CHECK(enabled_result.work_units > 0.0);
    const bool gate_has_effect =
        std::abs(enabled_result.work_units - disabled_result.work_units) > 1e-9;
    CHECK(gate_has_effect);
    if (enabled_result.status == Status::Optimal) {
        constexpr Real expected_obj = -9.8729400000e+02;
        Real denom = std::max(1.0, std::abs(expected_obj));
        Real rel_err = std::abs(enabled_result.objective - expected_obj) / denom;
        CHECK(rel_err < 1e-6);
    }
}

TEST_CASE("DualSimplex: enabled primal progress gate keeps degen2 solvable",
          "[dual_simplex][netlib][guardrail][regression]") {
    std::string path = testDataDir() + "/netlib/degen2.mps.gz";
    if (!fs::exists(path)) {
        SKIP("Netlib instance degen2 not downloaded");
    }

    DualSimplexSolver solver;
    DualSimplexOptions opts = solver.getOptions();
    opts.max_solve_seconds = 15.0;
    opts.primal_feasible_dual_progress_window = 8;
    opts.primal_feasible_refactor_cooldown = 32;
    opts.primal_feasible_adaptive_refactor_stall_pivots = 32;
    opts.primal_feasible_adaptive_refactor_min_updates = 24;
    solver.setOptions(opts);
    solver.load(readMps(path));

    const auto result = solver.solve();
    REQUIRE(result.status == Status::Optimal);
    constexpr Real expected_obj = -1.4351780000e+03;
    Real denom = std::max(1.0, std::abs(expected_obj));
    Real rel_err = std::abs(result.objective - expected_obj) / denom;
    CHECK(rel_err < 1e-6);
    CHECK(result.iterations > 0);
}

TEST_CASE("DualSimplex: default no-presolve handles degen2 and degen3 within time limit",
          "[dual_simplex][netlib][regression][no_presolve]") {
    struct NetlibCase {
        const char* name;
        Real expected_obj;
    };
    constexpr NetlibCase kCases[] = {
        {"degen2", -1.4351780000e+03},
        {"degen3", -9.8729400000e+02},
    };

    for (const auto& tc : kCases) {
        std::string path = testDataDir() + "/netlib/" + tc.name + ".mps.gz";
        if (!fs::exists(path)) {
            SKIP(std::string("Netlib instance missing: ") + tc.name);
        }

        DualSimplexSolver solver;
        DualSimplexOptions opts = solver.getOptions();
        opts.max_solve_seconds = 5.0;
        solver.setOptions(opts);
        solver.load(readMps(path));

        const auto result = solver.solve();
        INFO("instance=" << tc.name);
        REQUIRE(result.status == Status::Optimal);
        Real denom = std::max(1.0, std::abs(tc.expected_obj));
        Real rel_err = std::abs(result.objective - tc.expected_obj) / denom;
        CHECK(rel_err < 1e-6);
        CHECK(result.work_units > 0.0);
    }
}

TEST_CASE("DualSimplex: singular warm-start basis recovers to valid solve",
          "[dual_simplex][basis]") {
    // Two structural columns are identical, so choosing both as basic creates a
    // singular basis. The solver should recover by rebuilding a valid basis.
    LpProblem lp;
    lp.name = "singular_basis_recovery";
    lp.sense = Sense::Minimize;
    lp.num_cols = 2;
    lp.obj = {1.0, 1.0};
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {kInf, kInf};
    lp.col_type = {VarType::Continuous, VarType::Continuous};
    lp.col_names = {"x", "y"};

    lp.num_rows = 2;
    lp.row_lower = {1.0, -kInf};
    lp.row_upper = {kInf, 2.0};
    lp.row_names = {"r_ge", "r_le"};

    // Row 0: x + y >= 1
    // Row 1: x + y <= 2
    std::vector<Real> values = {1.0, 1.0, 1.0, 1.0};
    std::vector<Index> col_indices = {0, 1, 0, 1};
    std::vector<Index> row_starts = {0, 2, 4};
    lp.matrix = SparseMatrix(2, 2, values, col_indices, row_starts);

    DualSimplexSolver solver;
    solver.load(lp);

    // Force a singular basis: both structural columns basic.
    std::vector<BasisStatus> singular_basis = {
        BasisStatus::Basic,
        BasisStatus::Basic,
        BasisStatus::AtLower,
        BasisStatus::AtLower,
    };
    solver.setBasis(singular_basis);

    const auto result = solver.solve();
    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(1.0, 1e-6));
}
