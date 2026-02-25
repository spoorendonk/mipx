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
    opts.enable_adaptive_refactorization = false;
    opts.partial_pricing_chunk_min = 8;
    opts.partial_pricing_full_scan_freq = 1;
    opts.adaptive_refactor_min_updates = 8;
    opts.adaptive_refactor_stall_pivots = 8;
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
    CHECK_FALSE(applied.enable_adaptive_refactorization);
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
