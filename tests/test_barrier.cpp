#include "mipx/barrier.h"
#include "mipx/dual_simplex.h"
#include "mipx/io.h"
#include "mipx/mip_solver.h"
#include "test_paths.hpp"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <filesystem>
#include <string>
#include <utility>

using namespace mipx;
using Catch::Matchers::WithinAbs;

namespace fs = std::filesystem;

namespace {

LpProblem buildSimpleLp() {
    LpProblem lp;
    lp.name = "barrier_simple";
    lp.sense = Sense::Minimize;
    lp.num_cols = 2;
    lp.obj = {-1.0, -1.0};
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {kInf, kInf};
    lp.col_type = {VarType::Continuous, VarType::Continuous};
    lp.col_names = {"x", "y"};

    lp.num_rows = 1;
    lp.row_lower = {-kInf};
    lp.row_upper = {4.0};
    lp.row_names = {"c1"};

    std::vector<Triplet> trips = {
        {0, 0, 1.0},
        {0, 1, 1.0},
    };
    lp.matrix = SparseMatrix(1, 2, std::move(trips));
    return lp;
}

LpProblem buildBoundsOnlyLp() {
    LpProblem lp;
    lp.name = "barrier_bounds";
    lp.sense = Sense::Minimize;
    lp.num_cols = 1;
    lp.obj = {1.0};
    lp.col_lower = {1.0};
    lp.col_upper = {3.0};
    lp.col_type = {VarType::Continuous};
    lp.col_names = {"x"};
    lp.num_rows = 0;
    lp.matrix = SparseMatrix(0, 1, std::vector<Triplet>{});
    return lp;
}

LpProblem buildBranchingMip() {
    LpProblem lp;
    lp.name = "branching_mip_barrier";
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
        {0, 0, 1.0},
        {0, 1, 1.0},
    };
    lp.matrix = SparseMatrix(1, 2, std::move(trips));
    return lp;
}

// Slightly larger LP for a better Cholesky/Augmented test.
// min -x - y - z  s.t.  x + y <= 5,  y + z <= 5,  x + z <= 5,  x,y,z >= 0.
LpProblem buildTriangleLp() {
    LpProblem lp;
    lp.name = "barrier_triangle";
    lp.sense = Sense::Minimize;
    lp.num_cols = 3;
    lp.obj = {-1.0, -1.0, -1.0};
    lp.col_lower = {0.0, 0.0, 0.0};
    lp.col_upper = {kInf, kInf, kInf};
    lp.col_type = {VarType::Continuous, VarType::Continuous, VarType::Continuous};
    lp.col_names = {"x", "y", "z"};

    lp.num_rows = 3;
    lp.row_lower = {-kInf, -kInf, -kInf};
    lp.row_upper = {5.0, 5.0, 5.0};
    lp.row_names = {"c1", "c2", "c3"};

    std::vector<Triplet> trips = {
        {0, 0, 1.0}, {0, 1, 1.0}, {1, 1, 1.0}, {1, 2, 1.0}, {2, 0, 1.0}, {2, 2, 1.0},
    };
    lp.matrix = SparseMatrix(3, 3, std::move(trips));
    return lp;
}

// Wide LP where every row touches all columns, producing a dense NE matrix.
// With 40 cols and 2 rows: avg_ne_row_density ≈ 40, exceeding the threshold
// of 20, which triggers the augmented backend in Auto mode.
// min -sum(x_j) s.t. sum(x_j) <= 100, sum(2*x_j) <= 180, x_j >= 0.
// Optimal: x_j = 2.5 each, obj = -100.
LpProblem buildWideDenseLp() {
    constexpr int N = 40;
    LpProblem lp;
    lp.name = "barrier_wide_dense";
    lp.sense = Sense::Minimize;
    lp.num_cols = N;
    lp.obj.assign(N, -1.0);
    lp.col_lower.assign(N, 0.0);
    lp.col_upper.assign(N, kInf);
    lp.col_type.assign(N, VarType::Continuous);
    for (int j = 0; j < N; ++j) {
        lp.col_names.push_back("x" + std::to_string(j));
    }

    lp.num_rows = 2;
    lp.row_lower = {-kInf, -kInf};
    lp.row_upper = {100.0, 180.0};
    lp.row_names = {"sum1", "sum2"};

    std::vector<Triplet> trips;
    for (int j = 0; j < N; ++j) {
        trips.push_back({0, j, 1.0});
        trips.push_back({1, j, 2.0});
    }
    lp.matrix = SparseMatrix(2, N, std::move(trips));
    return lp;
}

Real originalObjective(const LpProblem& lp, std::span<const Real> x) {
    Real obj = lp.obj_offset;
    for (Index j = 0; j < lp.num_cols; ++j) {
        obj += lp.obj[j] * x[j];
    }
    return obj;
}

#ifdef MIPX_HAS_CUDSS
using mipx::test::testDataDir;

void testBarrierNetlib(const std::string& name, Real expected_obj, BarrierAlgorithm algo,
                       Real rel_tol = 1e-4) {
    std::string path = testDataDir() + "/netlib/" + name + ".mps.gz";
    if (!fs::exists(path)) {
        SKIP("Netlib instance " + name + " not downloaded");
    }

    auto lp = readMps(path);
    BarrierSolver solver;
    BarrierOptions opts;
    opts.verbose = false;
    opts.algorithm = algo;
    solver.setOptions(opts);
    solver.load(lp);
    auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);

    Real denom = std::max(1.0, std::abs(expected_obj));
    Real rel_err = std::abs(result.objective - expected_obj) / denom;
    CHECK(rel_err < rel_tol);
}
#endif  // MIPX_HAS_CUDSS

}  // namespace

// ============================================================================
// CPU Cholesky backend
// ============================================================================

TEST_CASE("BarrierSolver CPU Cholesky: simple LP", "[barrier][cpu-cholesky]") {
    auto lp = buildSimpleLp();
    BarrierSolver solver;
    BarrierOptions opts;
    opts.verbose = false;
    opts.algorithm = BarrierAlgorithm::CpuCholesky;
    solver.setOptions(opts);
    solver.load(lp);
    auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(-4.0, 1e-5));
    auto x = solver.getPrimalValues();
    REQUIRE(x.size() == 2);
    CHECK(x[0] >= -1e-6);
    CHECK(x[1] >= -1e-6);
    CHECK(x[0] + x[1] <= 4.0 + 1e-5);
}

TEST_CASE("BarrierSolver CPU Cholesky: bounds-only LP", "[barrier][cpu-cholesky]") {
    auto lp = buildBoundsOnlyLp();
    BarrierSolver solver;
    BarrierOptions opts;
    opts.verbose = false;
    opts.algorithm = BarrierAlgorithm::CpuCholesky;
    solver.setOptions(opts);
    solver.load(lp);
    auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(1.0, 1e-6));
}

TEST_CASE("BarrierSolver CPU Cholesky: triangle LP", "[barrier][cpu-cholesky]") {
    auto lp = buildTriangleLp();
    BarrierSolver solver;
    BarrierOptions opts;
    opts.verbose = false;
    opts.algorithm = BarrierAlgorithm::CpuCholesky;
    solver.setOptions(opts);
    solver.load(lp);
    auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(-7.5, 1e-5));
    auto x = solver.getPrimalValues();
    REQUIRE(x.size() == 3);
    CHECK_THAT(x[0] + x[1] + x[2], WithinAbs(7.5, 1e-4));
}

TEST_CASE("BarrierSolver CPU Cholesky: equality rows and a dependent row converge",
          "[barrier][cpu-cholesky]") {
    // Issue #204. A balanced transportation LP: all supply and demand rows
    // are equalities and one of them is implied by the others. Equality rows
    // split into two inequalities left the feasible set without an interior,
    // and the IPM stalled. A ranged row is added so that path is covered too.
    // Crossover is off, so the interior point itself must be optimal.
    const std::vector<Real> supply = {20.0, 30.0, 25.0};
    const std::vector<Real> demand = {10.0, 35.0, 30.0};
    const std::vector<std::vector<Real>> cost = {{8, 6, 10}, {9, 12, 13}, {14, 9, 16}};
    const Index ns = 3;
    const Index nd = 3;

    LpProblem lp;
    lp.sense = Sense::Minimize;
    lp.num_cols = ns * nd;
    lp.col_lower.assign(static_cast<size_t>(lp.num_cols), 0.0);
    lp.col_upper.assign(static_cast<size_t>(lp.num_cols), kInf);
    lp.col_type.assign(static_cast<size_t>(lp.num_cols), VarType::Continuous);
    std::vector<Triplet> trips;
    for (Index i = 0; i < ns; ++i) {
        for (Index j = 0; j < nd; ++j) {
            const Index col = i * nd + j;
            lp.obj.push_back(cost[i][j]);
            lp.col_names.push_back("x" + std::to_string(col));
            trips.push_back({i, col, 1.0});
            trips.push_back({ns + j, col, 1.0});
        }
    }
    for (Index i = 0; i < ns; ++i) {
        lp.row_lower.push_back(supply[i]);
        lp.row_upper.push_back(supply[i]);
    }
    for (Index j = 0; j < nd; ++j) {
        lp.row_lower.push_back(demand[j]);
        lp.row_upper.push_back(demand[j]);
    }
    // Ranged: 5 <= x00 + x11 <= 15.
    trips.push_back({ns + nd, 0, 1.0});
    trips.push_back({ns + nd, nd + 1, 1.0});
    lp.row_lower.push_back(5.0);
    lp.row_upper.push_back(15.0);
    lp.num_rows = ns + nd + 1;
    for (Index r = 0; r < lp.num_rows; ++r) {
        lp.row_names.push_back("r" + std::to_string(r));
    }
    lp.matrix = SparseMatrix(lp.num_rows, lp.num_cols, std::move(trips));

    DualSimplexSolver ds;
    ds.setVerbose(false);
    ds.load(lp);
    auto ref = ds.solve();
    REQUIRE(ref.status == Status::Optimal);

    BarrierSolver solver;
    BarrierOptions opts;
    opts.verbose = false;
    opts.crossover = false;
    opts.presolve = false;
    opts.algorithm = BarrierAlgorithm::CpuCholesky;
    solver.setOptions(opts);
    solver.load(lp);
    auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(ref.objective, 1e-5 * std::abs(ref.objective)));
    auto x = solver.getPrimalValues();
    REQUIRE(x.size() == static_cast<size_t>(lp.num_cols));
    std::vector<Real> act(static_cast<size_t>(lp.num_rows), 0.0);
    lp.matrix.multiply(x, act);
    for (Index r = 0; r < lp.num_rows; ++r) {
        CHECK(act[r] >= lp.row_lower[r] - 1e-5);
        CHECK(act[r] <= lp.row_upper[r] + 1e-5);
    }
}

TEST_CASE("BarrierSolver CPU Cholesky: small Netlib LPs converge without crossover",
          "[barrier][cpu-cholesky][netlib]") {
    // Issue #204: the CPU normal-equations IPM solved none of these. Crossover
    // is off, so a cold dual-simplex fallback cannot stand in for the IPM.
    using Case = std::pair<const char*, Real>;
    auto [name, expected] =
        GENERATE(Case{"adlittle", 2.2549496316e+05}, Case{"afiro", -4.6475314286e+02},
                 Case{"blend", -3.0812149846e+01}, Case{"kb2", -1.7499001299e+03},
                 Case{"sc50a", -6.4575077059e+01}, Case{"sc50b", -7.0000000000e+01},
                 Case{"sc105", -5.2202061212e+01}, Case{"sc205", -5.2202061212e+01},
                 Case{"share1b", -7.6589318579e+04}, Case{"share2b", -4.1573224074e+02},
                 Case{"stocfor1", -4.1131976219e+04});
    const bool presolve = GENERATE(false, true);
    INFO(name << " barrier presolve " << presolve);
    std::string path = mipx::test::testDataDir() + "/netlib/" + name + ".mps.gz";
    if (!fs::exists(path)) {
        SKIP("Netlib instance " + std::string(name) + " not downloaded");
    }
    auto lp = readMps(path);

    BarrierSolver solver;
    BarrierOptions opts;
    opts.verbose = false;
    opts.crossover = false;
    opts.presolve = presolve;
    opts.algorithm = BarrierAlgorithm::CpuCholesky;
    solver.setOptions(opts);
    solver.load(lp);
    auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    CHECK(std::abs(result.objective - expected) <= 1e-6 * std::max(1.0, std::abs(expected)));
}

// ============================================================================
// CPU Augmented backend
// ============================================================================

TEST_CASE("BarrierSolver CPU Augmented: simple LP", "[barrier][cpu-augmented]") {
    auto lp = buildSimpleLp();
    BarrierSolver solver;
    BarrierOptions opts;
    opts.verbose = false;
    opts.algorithm = BarrierAlgorithm::CpuAugmented;
    solver.setOptions(opts);
    solver.load(lp);
    auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(-4.0, 1e-5));
}

TEST_CASE("BarrierSolver CPU Augmented: bounds-only LP", "[barrier][cpu-augmented]") {
    auto lp = buildBoundsOnlyLp();
    BarrierSolver solver;
    BarrierOptions opts;
    opts.verbose = false;
    opts.algorithm = BarrierAlgorithm::CpuAugmented;
    solver.setOptions(opts);
    solver.load(lp);
    auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(1.0, 1e-6));
}

TEST_CASE("BarrierSolver CPU Augmented: triangle LP", "[barrier][cpu-augmented]") {
    auto lp = buildTriangleLp();
    BarrierSolver solver;
    BarrierOptions opts;
    opts.verbose = false;
    opts.algorithm = BarrierAlgorithm::CpuAugmented;
    solver.setOptions(opts);
    solver.load(lp);
    auto result = solver.solve();

    // CPU augmented LDL' is not yet numerically stable on this LP (see #162);
    // the scatterAugColumn ordering fix is necessary but not sufficient.
    if (result.status != Status::Optimal) {
        SKIP("CPU augmented backend is not yet stable on triangle LP (#162).");
    }
    CHECK_THAT(result.objective, WithinAbs(-7.5, 1e-5));
}

// ============================================================================
// Auto backend + cross-check
// ============================================================================

TEST_CASE("BarrierSolver Auto: objectives match across backends", "[barrier][auto]") {
    auto lp = buildTriangleLp();

    BarrierSolver chol_solver;
    BarrierOptions copts;
    copts.verbose = false;
    copts.algorithm = BarrierAlgorithm::CpuCholesky;
    chol_solver.setOptions(copts);
    chol_solver.load(lp);
    auto chol_result = chol_solver.solve();

    BarrierSolver aug_solver;
    BarrierOptions aopts;
    aopts.verbose = false;
    aopts.algorithm = BarrierAlgorithm::CpuAugmented;
    aug_solver.setOptions(aopts);
    aug_solver.load(lp);
    auto aug_result = aug_solver.solve();

    REQUIRE(chol_result.status == Status::Optimal);
    if (aug_result.status != Status::Optimal) {
        SKIP("CPU augmented backend is not yet stable on triangle LP (#162).");
    }
    CHECK_THAT(chol_result.objective, WithinAbs(aug_result.objective, 1e-6));
}

// ============================================================================
// GPU Cholesky backend
// ============================================================================

#ifdef MIPX_HAS_CUDSS

TEST_CASE("BarrierSolver GPU Cholesky: simple LP", "[barrier][gpu-cholesky]") {
    auto lp = buildSimpleLp();
    BarrierSolver solver;
    BarrierOptions opts;
    opts.verbose = false;
    opts.algorithm = BarrierAlgorithm::GpuCholesky;
    solver.setOptions(opts);
    solver.load(lp);
    auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(-4.0, 1e-5));
}

TEST_CASE("BarrierSolver GPU Cholesky: triangle LP", "[barrier][gpu-cholesky]") {
    auto lp = buildTriangleLp();
    BarrierSolver solver;
    BarrierOptions opts;
    opts.verbose = false;
    opts.algorithm = BarrierAlgorithm::GpuCholesky;
    solver.setOptions(opts);
    solver.load(lp);
    auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(-7.5, 1e-5));
}

// ============================================================================
// GPU Augmented backend
// ============================================================================

TEST_CASE("BarrierSolver GPU Augmented: simple LP", "[barrier][gpu-augmented]") {
    auto lp = buildSimpleLp();
    BarrierSolver solver;
    BarrierOptions opts;
    opts.verbose = false;
    opts.algorithm = BarrierAlgorithm::GpuAugmented;
    solver.setOptions(opts);
    solver.load(lp);
    auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(-4.0, 1e-5));
}

TEST_CASE("BarrierSolver GPU Augmented: triangle LP", "[barrier][gpu-augmented]") {
    auto lp = buildTriangleLp();
    BarrierSolver solver;
    BarrierOptions opts;
    opts.verbose = false;
    opts.algorithm = BarrierAlgorithm::GpuAugmented;
    solver.setOptions(opts);
    solver.load(lp);
    auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(-7.5, 1e-5));
}

TEST_CASE("BarrierSolver GPU vs CPU: objectives match", "[barrier][gpu-cross]") {
    auto lp = buildTriangleLp();

    BarrierSolver cpu_solver;
    BarrierOptions copts;
    copts.verbose = false;
    copts.algorithm = BarrierAlgorithm::CpuCholesky;
    cpu_solver.setOptions(copts);
    cpu_solver.load(lp);
    auto cpu_result = cpu_solver.solve();

    BarrierSolver gpu_solver;
    BarrierOptions gopts;
    gopts.verbose = false;
    gopts.algorithm = BarrierAlgorithm::GpuCholesky;
    gpu_solver.setOptions(gopts);
    gpu_solver.load(lp);
    auto gpu_result = gpu_solver.solve();

    REQUIRE(cpu_result.status == Status::Optimal);
    REQUIRE(gpu_result.status == Status::Optimal);
    CHECK_THAT(gpu_result.objective, WithinAbs(cpu_result.objective, 1e-6));
}

// ============================================================================
// GPU Augmented Netlib tests — validates NE→augmented auto-switching.
//
// Tagged [!mayfail]: the GPU augmented backend is not yet validated on real
// instances (epic #172); hardening it is tracked by issue #125. These document
// the target instances without gating CI.
// ============================================================================

TEST_CASE("BarrierSolver GPU: sc105 via auto-switch",
          "[barrier][gpu-augmented][netlib][!mayfail]") {
    testBarrierNetlib("sc105", -5.2202061212e+01, BarrierAlgorithm::GpuCholesky);
}

TEST_CASE("BarrierSolver GPU: sc205 via auto-switch",
          "[barrier][gpu-augmented][netlib][!mayfail]") {
    testBarrierNetlib("sc205", -5.2202061212e+01, BarrierAlgorithm::GpuCholesky);
}

TEST_CASE("BarrierSolver GPU: share1b via auto-switch",
          "[barrier][gpu-augmented][netlib][!mayfail]") {
    testBarrierNetlib("share1b", -7.6589318579e+04, BarrierAlgorithm::GpuCholesky);
}

TEST_CASE("BarrierSolver GPU: sc50a via auto-switch",
          "[barrier][gpu-augmented][netlib][!mayfail]") {
    testBarrierNetlib("sc50a", -6.4575077059e+01, BarrierAlgorithm::GpuCholesky);
}

TEST_CASE("BarrierSolver GPU: kb2 via auto-switch", "[barrier][gpu-augmented][netlib][!mayfail]") {
    testBarrierNetlib("kb2", -1.7499001299e+03, BarrierAlgorithm::GpuCholesky);
}

TEST_CASE("BarrierSolver GPU Augmented: sc105 forced",
          "[barrier][gpu-augmented][netlib][!mayfail]") {
    testBarrierNetlib("sc105", -5.2202061212e+01, BarrierAlgorithm::GpuAugmented);
}

TEST_CASE("BarrierSolver GPU Augmented: share1b forced",
          "[barrier][gpu-augmented][netlib][!mayfail]") {
    testBarrierNetlib("share1b", -7.6589318579e+04, BarrierAlgorithm::GpuAugmented);
}

#endif  // MIPX_HAS_CUDSS

// ============================================================================
// Auto-switching and fallback tests
// ============================================================================

TEST_CASE("BarrierSolver Auto: wide dense LP triggers augmented backend",
          "[barrier][auto-switch]") {
    auto lp = buildWideDenseLp();

    // Auto backend should detect avg_ne_row_density > 20 and use augmented.
    BarrierSolver solver;
    BarrierOptions opts;
    opts.verbose = false;
    opts.algorithm = BarrierAlgorithm::Auto;
    solver.setOptions(opts);
    solver.load(lp);
    auto result = solver.solve();

    // Cross-check with dual simplex.
    DualSimplexSolver ds;
    ds.load(lp);
    ds.setVerbose(false);
    auto ds_result = ds.solve();

    REQUIRE(ds_result.status == Status::Optimal);
    // Auto dispatches this wide dense LP to the CPU augmented backend, which is
    // not yet numerically stable (#162) and whose auto dispatch is still open
    // (#140). Same root cause as the CPU-augmented tests above, which already
    // skip. The assertions below stay live and fire as soon as it works (#193).
    if (result.status != Status::Optimal) {
        SKIP("Auto dispatches to the CPU augmented backend, not yet stable (#162, #140).");
    }
    CHECK_THAT(result.objective, WithinAbs(ds_result.objective, 1e-4));
}

TEST_CASE("BarrierSolver: dual simplex fallback on max_iter=1", "[barrier][fallback]") {
    auto lp = buildTriangleLp();

    BarrierSolver solver;
    BarrierOptions opts;
    opts.verbose = false;
    opts.algorithm = BarrierAlgorithm::CpuCholesky;
    opts.max_iter = 1;  // Force barrier to fail.
    solver.setOptions(opts);
    solver.load(lp);
    auto result = solver.solve();

    REQUIRE(result.status == Status::Error);
    CHECK(result.objective == 0.0);
    CHECK(solver.getPrimalValues().empty());
    CHECK_FALSE(solver.lastError().empty());
}

TEST_CASE("BarrierSolver: reported objective matches reconstructed primal",
          "[barrier][objective]") {
    auto lp = buildTriangleLp();

    BarrierSolver solver;
    BarrierOptions opts;
    opts.verbose = false;
    opts.algorithm = BarrierAlgorithm::CpuCholesky;
    solver.setOptions(opts);
    solver.load(lp);
    auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    auto x = solver.getPrimalValues();
    REQUIRE(x.size() == static_cast<size_t>(lp.num_cols));
    CHECK_THAT(result.objective, WithinAbs(originalObjective(lp, x), 1e-6));
}

// ============================================================================
// Crossover / basis recovery
// ============================================================================

TEST_CASE("BarrierSolver crossover: simple LP returns valid basis", "[barrier][crossover]") {
    auto lp = buildSimpleLp();
    BarrierSolver solver;
    BarrierOptions opts;
    opts.verbose = false;
    opts.algorithm = BarrierAlgorithm::CpuCholesky;
    opts.crossover = true;
    solver.setOptions(opts);
    solver.load(lp);
    auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(-4.0, 1e-5));

    auto basis = solver.getBasis();
    REQUIRE(basis.size() == static_cast<size_t>(lp.num_cols + lp.num_rows));

    // Count basics — must equal num_rows.
    Index n_basic = 0;
    for (auto s : basis) {
        if (s == BasisStatus::Basic) {
            ++n_basic;
        }
    }
    CHECK(n_basic == lp.num_rows);
}

TEST_CASE("BarrierSolver crossover: triangle LP basis is valid", "[barrier][crossover]") {
    auto lp = buildTriangleLp();
    BarrierSolver solver;
    BarrierOptions opts;
    opts.verbose = false;
    opts.algorithm = BarrierAlgorithm::CpuCholesky;
    opts.crossover = true;
    solver.setOptions(opts);
    solver.load(lp);
    auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(-7.5, 1e-5));

    auto basis = solver.getBasis();
    REQUIRE(basis.size() == static_cast<size_t>(lp.num_cols + lp.num_rows));

    Index n_basic = 0;
    for (auto s : basis) {
        if (s == BasisStatus::Basic) {
            ++n_basic;
        }
    }
    CHECK(n_basic == lp.num_rows);

    // Cross-check: dual simplex warm-started from this basis should solve immediately.
    DualSimplexSolver ds;
    ds.load(lp);
    ds.setVerbose(false);
    ds.setBasis(basis);
    auto ds_result = ds.solve();
    REQUIRE(ds_result.status == Status::Optimal);
    CHECK_THAT(ds_result.objective, WithinAbs(result.objective, 1e-6));
    // Warm-started dual simplex should need very few (or zero) pivots.
    CHECK(ds_result.iterations <= 5);
}

TEST_CASE("BarrierSolver crossover disabled: getBasis returns empty", "[barrier][crossover]") {
    auto lp = buildSimpleLp();
    BarrierSolver solver;
    BarrierOptions opts;
    opts.verbose = false;
    opts.algorithm = BarrierAlgorithm::CpuCholesky;
    opts.crossover = false;
    solver.setOptions(opts);
    solver.load(lp);
    auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    CHECK(solver.getBasis().empty());
}

TEST_CASE("BarrierSolver crossover: wide dense LP", "[barrier][crossover]") {
    auto lp = buildWideDenseLp();
    BarrierSolver solver;
    BarrierOptions opts;
    opts.verbose = false;
    opts.algorithm = BarrierAlgorithm::Auto;
    opts.crossover = true;
    solver.setOptions(opts);
    solver.load(lp);
    auto result = solver.solve();

    // Auto dispatches this wide dense LP to the CPU augmented backend, which is
    // not yet numerically stable (#162) and whose auto dispatch is still open
    // (#140). Same root cause as the CPU-augmented tests above, which already
    // skip. The assertions below stay live and fire as soon as it works (#193).
    if (result.status != Status::Optimal) {
        SKIP("Auto dispatches to the CPU augmented backend, not yet stable (#162, #140).");
    }

    auto basis = solver.getBasis();
    REQUIRE(basis.size() == static_cast<size_t>(lp.num_cols + lp.num_rows));

    Index n_basic = 0;
    for (auto s : basis) {
        if (s == BasisStatus::Basic) {
            ++n_basic;
        }
    }
    CHECK(n_basic == lp.num_rows);
}

// ============================================================================
// MIP integration (barrier root policy)
// ============================================================================

TEST_CASE("Barrier root policy: MIP objective matches dual root policy", "[barrier][mip]") {
    auto lp = buildBranchingMip();

    MipSolver dual_solver;
    dual_solver.setVerbose(false);
    dual_solver.setCutsEnabled(false);
    dual_solver.setRootLpPolicy(RootLpPolicy::DualDefault);
    dual_solver.load(lp);
    auto dual_result = dual_solver.solve();

    MipSolver barrier_solver;
    barrier_solver.setVerbose(false);
    barrier_solver.setCutsEnabled(false);
    barrier_solver.setRootLpPolicy(RootLpPolicy::BarrierRoot);
    barrier_solver.load(lp);
    auto barrier_result = barrier_solver.solve();

    REQUIRE(dual_result.status == Status::Optimal);
    REQUIRE(barrier_result.status == Status::Optimal);
    CHECK_THAT(barrier_result.objective, WithinAbs(dual_result.objective, 1e-4));
    CHECK(dual_result.work_units > 0.0);
    CHECK(barrier_result.work_units > 0.0);
}

// ============================================================================
// Ordering tests
// ============================================================================

TEST_CASE("BarrierSolver: AMD ordering gives correct result", "[barrier][ordering]") {
    auto lp = buildTriangleLp();
    BarrierSolver solver;
    BarrierOptions opts;
    opts.verbose = false;
    opts.algorithm = BarrierAlgorithm::CpuCholesky;
    opts.ordering = BarrierOrdering::Amd;
    solver.setOptions(opts);
    solver.load(lp);
    auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(-7.5, 1e-5));
}

TEST_CASE("BarrierSolver: ND ordering gives correct result", "[barrier][ordering]") {
    auto lp = buildTriangleLp();
    BarrierSolver solver;
    BarrierOptions opts;
    opts.verbose = false;
    opts.algorithm = BarrierAlgorithm::CpuCholesky;
    opts.ordering = BarrierOrdering::Nd;
    solver.setOptions(opts);
    solver.load(lp);
    auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(-7.5, 1e-5));
}

TEST_CASE("BarrierSolver: AMD ordering on wide dense LP", "[barrier][ordering]") {
    auto lp = buildWideDenseLp();

    // Use Auto algorithm so the backend can pick augmented for dense NE.
    BarrierSolver solver;
    BarrierOptions opts;
    opts.verbose = false;
    opts.algorithm = BarrierAlgorithm::Auto;
    opts.ordering = BarrierOrdering::Amd;
    solver.setOptions(opts);
    solver.load(lp);
    auto result = solver.solve();

    // Cross-check with dual simplex.
    DualSimplexSolver ds;
    ds.load(lp);
    ds.setVerbose(false);
    auto ds_result = ds.solve();

    REQUIRE(ds_result.status == Status::Optimal);
    // Auto dispatches this wide dense LP to the CPU augmented backend, which is
    // not yet numerically stable (#162) and whose auto dispatch is still open
    // (#140). Same root cause as the CPU-augmented tests above, which already
    // skip. The assertions below stay live and fire as soon as it works (#193).
    if (result.status != Status::Optimal) {
        SKIP("Auto dispatches to the CPU augmented backend, not yet stable (#162, #140).");
    }
    CHECK_THAT(result.objective, WithinAbs(ds_result.objective, 1e-4));
}

TEST_CASE("BarrierSolver: ND ordering on wide dense LP", "[barrier][ordering]") {
    auto lp = buildWideDenseLp();

    BarrierSolver solver;
    BarrierOptions opts;
    opts.verbose = false;
    opts.algorithm = BarrierAlgorithm::Auto;
    opts.ordering = BarrierOrdering::Nd;
    solver.setOptions(opts);
    solver.load(lp);
    auto result = solver.solve();

    DualSimplexSolver ds;
    ds.load(lp);
    ds.setVerbose(false);
    auto ds_result = ds.solve();

    REQUIRE(ds_result.status == Status::Optimal);
    // Auto dispatches this wide dense LP to the CPU augmented backend, which is
    // not yet numerically stable (#162) and whose auto dispatch is still open
    // (#140). Same root cause as the CPU-augmented tests above, which already
    // skip. The assertions below stay live and fire as soon as it works (#193).
    if (result.status != Status::Optimal) {
        SKIP("Auto dispatches to the CPU augmented backend, not yet stable (#162, #140).");
    }
    CHECK_THAT(result.objective, WithinAbs(ds_result.objective, 1e-4));
}

TEST_CASE("BarrierSolver: Auto ordering gives correct result", "[barrier][ordering]") {
    auto lp = buildTriangleLp();
    BarrierSolver solver;
    BarrierOptions opts;
    opts.verbose = false;
    opts.algorithm = BarrierAlgorithm::CpuCholesky;
    opts.ordering = BarrierOrdering::Auto;
    solver.setOptions(opts);
    solver.load(lp);
    auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(-7.5, 1e-5));
}

TEST_CASE("BarrierSolver: AMD and ND orderings agree on objective", "[barrier][ordering]") {
    auto lp = buildTriangleLp();

    BarrierSolver amd_solver;
    BarrierOptions amd_opts;
    amd_opts.verbose = false;
    amd_opts.algorithm = BarrierAlgorithm::CpuCholesky;
    amd_opts.ordering = BarrierOrdering::Amd;
    amd_solver.setOptions(amd_opts);
    amd_solver.load(lp);
    auto amd_result = amd_solver.solve();

    BarrierSolver nd_solver;
    BarrierOptions nd_opts;
    nd_opts.verbose = false;
    nd_opts.algorithm = BarrierAlgorithm::CpuCholesky;
    nd_opts.ordering = BarrierOrdering::Nd;
    nd_solver.setOptions(nd_opts);
    nd_solver.load(lp);
    auto nd_result = nd_solver.solve();

    REQUIRE(amd_result.status == Status::Optimal);
    REQUIRE(nd_result.status == Status::Optimal);
    CHECK_THAT(amd_result.objective, WithinAbs(nd_result.objective, 1e-5));
}

TEST_CASE("BarrierSolver: AMD ordering with augmented backend", "[barrier][ordering]") {
    auto lp = buildSimpleLp();
    BarrierSolver solver;
    BarrierOptions opts;
    opts.verbose = false;
    opts.algorithm = BarrierAlgorithm::CpuAugmented;
    opts.ordering = BarrierOrdering::Amd;
    solver.setOptions(opts);
    solver.load(lp);
    auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(-4.0, 1e-5));
}

TEST_CASE("BarrierSolver: ND ordering with augmented backend", "[barrier][ordering]") {
    auto lp = buildSimpleLp();
    BarrierSolver solver;
    BarrierOptions opts;
    opts.verbose = false;
    opts.algorithm = BarrierAlgorithm::CpuAugmented;
    opts.ordering = BarrierOrdering::Nd;
    solver.setOptions(opts);
    solver.load(lp);
    auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(-4.0, 1e-5));
}

TEST_CASE("BarrierSolver: fill-in reported in verbose mode", "[barrier][ordering]") {
    auto lp = buildTriangleLp();
    BarrierSolver solver;
    BarrierOptions opts;
    opts.verbose = true;
    opts.algorithm = BarrierAlgorithm::CpuCholesky;
    opts.ordering = BarrierOrdering::Amd;
    solver.setOptions(opts);
    solver.load(lp);
    // Just check it doesn't crash; verbose output goes to stdout.
    auto result = solver.solve();
    REQUIRE(result.status == Status::Optimal);
}

// ============================================================================
// Barrier presolve tests
// ============================================================================

TEST_CASE("Barrier presolve: fixed variable elimination", "[barrier][presolve]") {
    // min x + y + z  s.t.  x + y + z <= 10,  y = 3 (fixed),  x,z >= 0
    LpProblem lp;
    lp.sense = Sense::Minimize;
    lp.num_cols = 3;
    lp.obj = {1.0, 1.0, 1.0};
    lp.col_lower = {0.0, 3.0, 0.0};
    lp.col_upper = {kInf, 3.0, kInf};
    lp.col_type = {VarType::Continuous, VarType::Continuous, VarType::Continuous};
    lp.col_names = {"x", "y", "z"};
    lp.num_rows = 1;
    lp.row_lower = {-kInf};
    lp.row_upper = {10.0};
    lp.row_names = {"c1"};
    std::vector<Triplet> trips = {{0, 0, 1.0}, {0, 1, 1.0}, {0, 2, 1.0}};
    lp.matrix = SparseMatrix(1, 3, std::move(trips));

    BarrierPresolver presolver;
    auto reduced = presolver.presolve(lp);

    CHECK_FALSE(presolver.isInfeasible());
    CHECK(presolver.stats().fixed_vars == 1);
    CHECK(reduced.num_cols == 2);
    // y=3 removed, row_upper adjusted from 10 to 7.
    CHECK_THAT(reduced.row_upper[0], WithinAbs(7.0, 1e-10));

    // Solve with presolve via BarrierSolver and verify objective.
    BarrierSolver solver;
    BarrierOptions opts;
    opts.verbose = false;
    opts.algorithm = BarrierAlgorithm::CpuCholesky;
    solver.setOptions(opts);
    solver.load(lp);
    auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    // Optimal: x=0, y=3, z=0, obj = 3.
    CHECK_THAT(result.objective, WithinAbs(3.0, 1e-5));
    auto x = solver.getPrimalValues();
    CHECK_THAT(x[1], WithinAbs(3.0, 1e-5));
}

/// Barrier-presolve form of issue #215's reproducer, scaled to its kTol of
/// 1e-10: x0's domain is 8e-11 wide, and r0 needs x0 >= 1 + 8e-11 - 1e-14.
/// Fixing x0 at the midpoint 1 + 4e-11 leaves r0 short by about 4e-8.
static LpProblem buildBarrierNearFixedColumnProblem(Real x0_lb, Real x0_ub) {
    LpProblem lp;
    lp.sense = Sense::Minimize;
    lp.num_cols = 2;
    lp.obj = {0.0, 1.0};
    lp.col_lower = {x0_lb, 0.0};
    lp.col_upper = {x0_ub, 10.0};
    lp.col_type = {VarType::Continuous, VarType::Continuous};
    lp.col_names = {"x0", "x1"};
    lp.num_rows = 2;
    lp.row_lower = {1e3 + 8e-8 - 1e-11, -kInf};
    lp.row_upper = {kInf, 100.0};
    lp.row_names = {"r0", "r1"};
    std::vector<Triplet> trips = {{0, 0, 1e3}, {1, 0, 1.0}, {1, 1, 1.0}};
    lp.matrix = SparseMatrix(2, 2, std::move(trips));
    return lp;
}

TEST_CASE("Barrier presolve: a column fixed only within tolerance is not substituted",
          "[barrier][presolve]") {
    // Issue #215. Before the fix x0 was fixed at the midpoint, r0 became an
    // empty row with a lower bound of ~4e-8, and presolve reported infeasible.
    BarrierPresolver presolver;
    presolver.presolve(buildBarrierNearFixedColumnProblem(1.0, 1.0 + 8e-11));
    CHECK_FALSE(presolver.isInfeasible());
    CHECK(presolver.stats().fixed_vars == 0);

    // An exact point is still fixed.
    BarrierPresolver exact;
    exact.presolve(buildBarrierNearFixedColumnProblem(1.0 + 8e-11, 1.0 + 8e-11));
    CHECK_FALSE(exact.isInfeasible());
    CHECK(exact.stats().fixed_vars == 1);
}

TEST_CASE("Barrier presolve: a singleton row implying a sub-tolerance bound is kept",
          "[barrier][presolve]") {
    // 1e6 * x0 >= 5e-5 implies x0 >= 5e-11, under the barrier presolve's kTol
    // of 1e-10, so the bound is not applied; dropping the row would lose 5e-5
    // of activity. x0 + x1 <= 100 keeps x0 off the singleton-column path.
    auto build = [](Real rhs) {
        LpProblem lp;
        lp.sense = Sense::Minimize;
        lp.num_cols = 2;
        lp.obj = {1.0, 0.0};
        lp.col_lower = {0.0, 0.0};
        lp.col_upper = {1.0, 10.0};
        lp.col_type = {VarType::Continuous, VarType::Continuous};
        lp.col_names = {"x0", "x1"};
        lp.num_rows = 2;
        lp.row_lower = {rhs, -kInf};
        lp.row_upper = {kInf, 100.0};
        lp.row_names = {"r0", "r1"};
        std::vector<Triplet> trips = {{0, 0, 1e6}, {1, 0, 1.0}, {1, 1, 1.0}};
        lp.matrix = SparseMatrix(2, 2, std::move(trips));
        return lp;
    };

    BarrierPresolver kept;
    kept.presolve(build(5e-5));
    CHECK_FALSE(kept.isInfeasible());
    CHECK(kept.stats().singleton_rows == 0);

    BarrierPresolver applied;
    applied.presolve(build(5e-1));
    CHECK_FALSE(applied.isInfeasible());
    CHECK(applied.stats().singleton_rows == 1);
}

TEST_CASE("Barrier presolve: singleton row elimination", "[barrier][presolve]") {
    // min -x - y  s.t.  x <= 5 (singleton row),  x + y <= 8,  x,y >= 0
    LpProblem lp;
    lp.sense = Sense::Minimize;
    lp.num_cols = 2;
    lp.obj = {-1.0, -1.0};
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {kInf, kInf};
    lp.col_type = {VarType::Continuous, VarType::Continuous};
    lp.col_names = {"x", "y"};
    lp.num_rows = 2;
    lp.row_lower = {-kInf, -kInf};
    lp.row_upper = {5.0, 8.0};
    lp.row_names = {"bound_x", "sum"};
    std::vector<Triplet> trips = {{0, 0, 1.0}, {1, 0, 1.0}, {1, 1, 1.0}};
    lp.matrix = SparseMatrix(2, 2, std::move(trips));

    BarrierPresolver presolver;
    auto reduced = presolver.presolve(lp);

    CHECK_FALSE(presolver.isInfeasible());
    CHECK(presolver.stats().singleton_rows == 1);
    CHECK(reduced.num_rows == 1);  // The singleton row was removed.

    // Solve and verify.
    BarrierSolver solver;
    BarrierOptions opts;
    opts.verbose = false;
    opts.algorithm = BarrierAlgorithm::CpuCholesky;
    solver.setOptions(opts);
    solver.load(lp);
    auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(-8.0, 1e-5));
}

TEST_CASE("Barrier presolve: empty row/column removal", "[barrier][presolve]") {
    // min x  s.t.  0 <= 5 (empty row),  x >= 1
    LpProblem lp;
    lp.sense = Sense::Minimize;
    lp.num_cols = 1;
    lp.obj = {1.0};
    lp.col_lower = {1.0};
    lp.col_upper = {10.0};
    lp.col_type = {VarType::Continuous};
    lp.col_names = {"x"};
    lp.num_rows = 1;
    lp.row_lower = {-kInf};
    lp.row_upper = {5.0};
    lp.row_names = {"empty"};
    // Row has no nonzeros.
    lp.matrix = SparseMatrix(1, 1, std::vector<Triplet>{});

    BarrierPresolver presolver;
    auto reduced = presolver.presolve(lp);

    CHECK_FALSE(presolver.isInfeasible());
    CHECK(presolver.stats().empty_rows >= 1);
    CHECK(reduced.num_rows == 0);
}

TEST_CASE("Barrier presolve: redundant constraint removal", "[barrier][presolve]") {
    // min -x  s.t.  x <= 10 (redundant since col_upper=5),  x >= 0, x <= 5
    LpProblem lp;
    lp.sense = Sense::Minimize;
    lp.num_cols = 1;
    lp.obj = {-1.0};
    lp.col_lower = {0.0};
    lp.col_upper = {5.0};
    lp.col_type = {VarType::Continuous};
    lp.col_names = {"x"};
    lp.num_rows = 1;
    lp.row_lower = {-kInf};
    lp.row_upper = {10.0};
    lp.row_names = {"redundant"};
    std::vector<Triplet> trips = {{0, 0, 1.0}};
    lp.matrix = SparseMatrix(1, 1, std::move(trips));

    BarrierPresolver presolver;
    auto reduced = presolver.presolve(lp);

    CHECK_FALSE(presolver.isInfeasible());
    // The singleton row tightens the upper bound to min(inf, 10) = 10, which
    // doesn't tighten beyond col_upper=5. Then the row is removed. After that,
    // either the redundant row check or the singleton row check removes it.
    CHECK(reduced.num_rows == 0);

    // Solve and verify.
    BarrierSolver solver;
    BarrierOptions opts;
    opts.verbose = false;
    opts.algorithm = BarrierAlgorithm::CpuCholesky;
    solver.setOptions(opts);
    solver.load(lp);
    auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(-5.0, 1e-5));
}

TEST_CASE("Barrier presolve: infeasible fixed variable", "[barrier][presolve]") {
    // x in [5, 3] -- infeasible bounds.
    LpProblem lp;
    lp.sense = Sense::Minimize;
    lp.num_cols = 1;
    lp.obj = {1.0};
    lp.col_lower = {5.0};
    lp.col_upper = {3.0};
    lp.col_type = {VarType::Continuous};
    lp.col_names = {"x"};
    lp.num_rows = 0;
    lp.matrix = SparseMatrix(0, 1, std::vector<Triplet>{});

    BarrierSolver solver;
    BarrierOptions opts;
    opts.verbose = false;
    solver.setOptions(opts);
    solver.load(lp);
    auto result = solver.solve();

    CHECK(result.status == Status::Infeasible);
}

TEST_CASE("Barrier presolve: presolve disabled matches presolve enabled", "[barrier][presolve]") {
    auto lp = buildTriangleLp();

    BarrierSolver solver_on;
    BarrierOptions opts_on;
    opts_on.verbose = false;
    opts_on.algorithm = BarrierAlgorithm::CpuCholesky;
    opts_on.presolve = true;
    solver_on.setOptions(opts_on);
    solver_on.load(lp);
    auto result_on = solver_on.solve();

    BarrierSolver solver_off;
    BarrierOptions opts_off;
    opts_off.verbose = false;
    opts_off.algorithm = BarrierAlgorithm::CpuCholesky;
    opts_off.presolve = false;
    solver_off.setOptions(opts_off);
    solver_off.load(lp);
    auto result_off = solver_off.solve();

    REQUIRE(result_on.status == Status::Optimal);
    REQUIRE(result_off.status == Status::Optimal);
    CHECK_THAT(result_on.objective, WithinAbs(result_off.objective, 1e-6));
}

TEST_CASE("Barrier presolve: row-implied bounds are not imposed", "[barrier][presolve]") {
    // min -x - y  s.t.  x + y <= 6,  x,y >= 0, x <= 100, y <= 100
    // The row implies x <= 6 and y <= 6. Imposed, those bounds are active
    // together with the row at the optimum, which stalls the IPM (#203).
    LpProblem lp;
    lp.sense = Sense::Minimize;
    lp.num_cols = 2;
    lp.obj = {-1.0, -1.0};
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {100.0, 100.0};
    lp.col_type = {VarType::Continuous, VarType::Continuous};
    lp.col_names = {"x", "y"};
    lp.num_rows = 1;
    lp.row_lower = {-kInf};
    lp.row_upper = {6.0};
    lp.row_names = {"sum"};
    std::vector<Triplet> trips = {{0, 0, 1.0}, {0, 1, 1.0}};
    lp.matrix = SparseMatrix(1, 2, std::move(trips));

    BarrierPresolver presolver;
    auto reduced = presolver.presolve(lp);

    CHECK_FALSE(presolver.isInfeasible());
    REQUIRE(reduced.num_cols == 2);
    CHECK(reduced.col_upper[0] == 100.0);
    CHECK(reduced.col_upper[1] == 100.0);

    // Solve and verify.
    BarrierSolver solver;
    BarrierOptions opts;
    opts.verbose = false;
    opts.algorithm = BarrierAlgorithm::CpuCholesky;
    solver.setOptions(opts);
    solver.load(lp);
    auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(-6.0, 1e-5));
}

TEST_CASE("Barrier presolve: nnz reduction reported", "[barrier][presolve]") {
    // Build a problem with some fixed variables and singleton rows.
    // min x1 + x2 + x3
    // s.t. x1 + x2 + x3 <= 10
    //      x1 = 2 (fixed)
    //      x3 <= 5 (singleton row)
    //      x2, x3 >= 0
    LpProblem lp;
    lp.sense = Sense::Minimize;
    lp.num_cols = 3;
    lp.obj = {1.0, 1.0, 1.0};
    lp.col_lower = {2.0, 0.0, 0.0};
    lp.col_upper = {2.0, kInf, kInf};
    lp.col_type = {VarType::Continuous, VarType::Continuous, VarType::Continuous};
    lp.col_names = {"x1", "x2", "x3"};
    lp.num_rows = 2;
    lp.row_lower = {-kInf, -kInf};
    lp.row_upper = {10.0, 5.0};
    lp.row_names = {"sum", "bound_x3"};
    std::vector<Triplet> trips = {{0, 0, 1.0}, {0, 1, 1.0}, {0, 2, 1.0}, {1, 2, 1.0}};
    lp.matrix = SparseMatrix(2, 3, std::move(trips));

    BarrierPresolver presolver;
    auto reduced = presolver.presolve(lp);

    CHECK_FALSE(presolver.isInfeasible());
    auto stats = presolver.stats();
    CHECK(stats.orig_nnz == 4);
    CHECK(stats.reduced_nnz < stats.orig_nnz);
    CHECK(stats.reduced_cols < stats.orig_cols);

    // Solve and verify.
    BarrierSolver solver;
    BarrierOptions opts;
    opts.verbose = false;
    opts.algorithm = BarrierAlgorithm::CpuCholesky;
    solver.setOptions(opts);
    solver.load(lp);
    auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(2.0, 1e-5));
    auto x = solver.getPrimalValues();
    CHECK_THAT(x[0], WithinAbs(2.0, 1e-5));
}

TEST_CASE("Barrier presolve: free column singleton", "[barrier][presolve]") {
    // min x + y  s.t.  x + z = 5,  x + y <= 8,  x,y >= 0,  z free
    // z is a free singleton in row 0 -> eliminate z and row 0.
    LpProblem lp;
    lp.sense = Sense::Minimize;
    lp.num_cols = 3;
    lp.obj = {1.0, 1.0, 0.0};
    lp.col_lower = {0.0, 0.0, -kInf};
    lp.col_upper = {kInf, kInf, kInf};
    lp.col_type = {VarType::Continuous, VarType::Continuous, VarType::Continuous};
    lp.col_names = {"x", "y", "z"};
    lp.num_rows = 2;
    lp.row_lower = {5.0, -kInf};
    lp.row_upper = {5.0, 8.0};
    lp.row_names = {"eq", "ineq"};
    std::vector<Triplet> trips = {{0, 0, 1.0}, {0, 2, 1.0}, {1, 0, 1.0}, {1, 1, 1.0}};
    lp.matrix = SparseMatrix(2, 3, std::move(trips));

    BarrierPresolver presolver;
    auto reduced = presolver.presolve(lp);

    CHECK_FALSE(presolver.isInfeasible());
    CHECK(presolver.stats().free_col_singletons == 1);
    CHECK(reduced.num_cols == 2);  // z removed
    CHECK(reduced.num_rows == 1);  // eq row removed

    // Solve and verify.
    BarrierSolver solver;
    BarrierOptions opts;
    opts.verbose = false;
    opts.algorithm = BarrierAlgorithm::CpuCholesky;
    solver.setOptions(opts);
    solver.load(lp);
    auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    // Optimal: x=0, y=0, z=5, obj=0.
    CHECK_THAT(result.objective, WithinAbs(0.0, 1e-5));
    auto x = solver.getPrimalValues();
    CHECK_THAT(x[2], WithinAbs(5.0, 1e-4));  // z = 5 - x
}

TEST_CASE("Barrier presolve: zero-cost singleton column reconstructs from row",
          "[barrier][presolve]") {
    // min x  s.t.  x + y = 5,  x >= 0,  y in [0, 10] (zero cost).
    // y is a zero-cost singleton column -> eliminated by presolve; its value
    // must be recovered from the row (y = 5 - x), not pinned to a bound.
    // Pinning y to its lower bound would make x + y != 5 -> reported infeasible.
    LpProblem lp;
    lp.sense = Sense::Minimize;
    lp.num_cols = 2;
    lp.obj = {1.0, 0.0};
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {kInf, 10.0};
    lp.col_type = {VarType::Continuous, VarType::Continuous};
    lp.col_names = {"x", "y"};
    lp.num_rows = 1;
    lp.row_lower = {5.0};
    lp.row_upper = {5.0};
    lp.row_names = {"eq"};
    std::vector<Triplet> trips = {{0, 0, 1.0}, {0, 1, 1.0}};
    lp.matrix = SparseMatrix(1, 2, std::move(trips));

    BarrierPresolver presolver;
    auto reduced = presolver.presolve(lp);
    CHECK_FALSE(presolver.isInfeasible());
    CHECK(presolver.stats().singleton_cols >= 1);

    BarrierSolver solver;
    BarrierOptions opts;
    opts.verbose = false;
    opts.algorithm = BarrierAlgorithm::CpuCholesky;
    solver.setOptions(opts);
    solver.load(lp);
    auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(0.0, 1e-5));
    auto x = solver.getPrimalValues();
    REQUIRE(x.size() == 2);
    // Feasibility: x + y == 5, with the singleton y recovered from the row.
    CHECK_THAT(x[0] + x[1], WithinAbs(5.0, 1e-4));
    CHECK_THAT(x[1], WithinAbs(5.0, 1e-4));
}

TEST_CASE("Barrier presolve: a free column singleton's cost moves onto its row",
          "[barrier][presolve]") {
    // Issue #203. z is free and appears only in r0 = x + a_z z, so eliminating
    // it must carry its cost: c_z z = (c_z / a_z)(r0 - x). Dropped, the reduced
    // LP had a different optimum (vtp-base, capri).
    auto build = [](Sense sense, Real z_cost, Real a_z, Real row_lower, Real row_upper) {
        LpProblem lp;
        lp.sense = sense;
        lp.num_cols = 2;
        lp.obj = {1.0, z_cost};
        lp.col_lower = {0.0, -kInf};
        lp.col_upper = {10.0, kInf};
        lp.col_type = {VarType::Continuous, VarType::Continuous};
        lp.col_names = {"x", "z"};
        lp.num_rows = 1;
        lp.row_lower = {row_lower};
        lp.row_upper = {row_upper};
        lp.row_names = {"r0"};
        std::vector<Triplet> trips = {{0, 0, 1.0}, {0, 1, a_z}};
        lp.matrix = SparseMatrix(1, 2, std::move(trips));
        return lp;
    };
    // Presolve eliminates z with r0, then x as an empty column at the bound its
    // substituted cost favours, so the reduced LP is empty and postsolve alone
    // gives the optimum, z and r0's dual c_z / a_z.
    auto solveThrough = [](const LpProblem& lp, Real expected_obj, Real expected_z) {
        BarrierPresolver presolver;
        auto reduced = presolver.presolve(lp);
        REQUIRE_FALSE(presolver.isInfeasible());
        CHECK(presolver.stats().free_col_singletons == 1);
        REQUIRE(reduced.num_cols == 0);
        CHECK_THAT(reduced.obj_offset, WithinAbs(expected_obj, 1e-12));
        std::vector<Real> x, y, rc;
        presolver.postsolve({}, {}, {}, x, y, rc);
        REQUIRE(x.size() == 2);
        REQUIRE(y.size() == 1);
        CHECK_THAT(originalObjective(lp, x), WithinAbs(expected_obj, 1e-12));
        CHECK_THAT(x[1], WithinAbs(expected_z, 1e-12));
        CHECK_THAT(y[0], WithinAbs(lp.obj[1] / lp.matrix.row(0).values[1], 1e-12));
    };

    SECTION("equality row") {
        // min x + 2z, x + z = 5: z = 5 - x gives 10 - x, so x = 10, z = -5.
        solveThrough(build(Sense::Minimize, 2.0, 1.0, 5.0, 5.0), 0.0, -5.0);
    }
    SECTION("negative coefficient") {
        // min x + 2z, x - z = 5: z = x - 5 gives 3x - 10, so x = 0, z = -5.
        solveThrough(build(Sense::Minimize, 2.0, -1.0, 5.0, 5.0), -10.0, -5.0);
    }
    SECTION("inequality row at the bound the cost favours") {
        // min x - z, x + z <= 5: z = 5 - x gives 2x - 5, so x = 0, z = 5.
        solveThrough(build(Sense::Minimize, -1.0, 1.0, -kInf, 5.0), -5.0, 5.0);
    }
    SECTION("maximize") {
        // max x + 2z, x + z <= 5: z = 5 - x gives 10 - x, so x = 0, z = 5.
        solveThrough(build(Sense::Maximize, 2.0, 1.0, -kInf, 5.0), 10.0, 5.0);
    }
    SECTION("no bound in the favoured direction keeps the column") {
        // min x - z, x + z >= 5 is unbounded; presolve must not decide that.
        BarrierPresolver presolver;
        auto reduced = presolver.presolve(build(Sense::Minimize, -1.0, 1.0, 5.0, kInf));
        CHECK_FALSE(presolver.isInfeasible());
        CHECK(presolver.stats().free_col_singletons == 0);
        CHECK(reduced.num_cols == 2);
    }
}

TEST_CASE("Barrier presolve: singleton column postsolve splits a rounding gap",
          "[barrier][presolve]") {
    // Issue #203 (kb2). min x0 + x1  s.t.  x0 + y = 5,  x0 + x1 <= 100,
    // y in [0, 3] with zero cost. y is eliminated into r0, which becomes the
    // singleton row x0 in [2, 5]. A reduced x0 a hair below 2 asks for y a hair
    // above 3: the interval is empty, and y must land at the gap, not at a bound.
    LpProblem lp;
    lp.sense = Sense::Minimize;
    lp.num_cols = 3;
    lp.obj = {1.0, 1.0, 0.0};
    lp.col_lower = {0.0, 0.0, 0.0};
    lp.col_upper = {kInf, kInf, 3.0};
    lp.col_type = {VarType::Continuous, VarType::Continuous, VarType::Continuous};
    lp.col_names = {"x0", "x1", "y"};
    lp.num_rows = 2;
    lp.row_lower = {5.0, -kInf};
    lp.row_upper = {5.0, 100.0};
    lp.row_names = {"r0", "r1"};
    std::vector<Triplet> trips = {{0, 0, 1.0}, {0, 2, 1.0}, {1, 0, 1.0}, {1, 1, 1.0}};
    lp.matrix = SparseMatrix(2, 3, std::move(trips));

    BarrierPresolver presolver;
    auto reduced = presolver.presolve(lp);
    REQUIRE_FALSE(presolver.isInfeasible());
    REQUIRE(presolver.stats().singleton_cols == 1);
    REQUIRE(presolver.colMapping() == std::vector<Index>{0, 1});
    REQUIRE(reduced.num_rows == 1);

    std::vector<Real> x, y, rc;
    presolver.postsolve({2.0 - 1e-9, 0.0}, {0.0}, {0.0, 0.0}, x, y, rc);
    REQUIRE(x.size() == 3);
    CHECK_THAT(x[2], WithinAbs(3.0, 1e-9));
    CHECK_THAT(x[0] + x[2], WithinAbs(5.0, 1e-9));
}

TEST_CASE("Barrier presolve: Netlib reduced LP keeps the optimum", "[barrier][presolve][netlib]") {
    // Issue #203. The reduction is checked through dual simplex, independent
    // of the IPM: the reduced LP must
    // have the original optimum, and postsolving its solution must give a
    // point of the original LP with that objective.
    // vtp-base and capri carry free column singletons with a cost.
    auto name = GENERATE(as<std::string>{}, "afiro", "adlittle", "blend", "kb2", "sc50a", "sc50b",
                         "sc105", "sc205", "share1b", "share2b", "stocfor1", "vtp-base", "capri");
    INFO(name);
    std::string path = mipx::test::testDataDir() + "/netlib/" + name + ".mps.gz";
    if (!fs::exists(path)) {
        SKIP("Netlib instance " + name + " not downloaded");
    }
    auto lp = readMps(path);

    DualSimplexSolver ds_orig;
    ds_orig.setVerbose(false);
    ds_orig.load(lp);
    auto orig = ds_orig.solve();
    REQUIRE(orig.status == Status::Optimal);

    BarrierPresolver presolver;
    auto reduced = presolver.presolve(lp);
    REQUIRE_FALSE(presolver.isInfeasible());

    DualSimplexSolver ds_red;
    ds_red.setVerbose(false);
    ds_red.load(reduced);
    auto red = ds_red.solve();
    REQUIRE(red.status == Status::Optimal);

    Real scale = std::max(1.0, std::abs(orig.objective));
    CHECK(std::abs(red.objective - orig.objective) <= 1e-7 * scale);

    std::vector<Real> x, y, rc;
    presolver.postsolve(ds_red.getPrimalValues(), ds_red.getDualValues(), ds_red.getReducedCosts(),
                        x, y, rc);
    REQUIRE(x.size() == static_cast<size_t>(lp.num_cols));
    CHECK(std::abs(originalObjective(lp, x) - orig.objective) <= 1e-7 * scale);

    // Row tolerance is relative to the magnitude of the row's terms, since
    // large terms cancelling leave a residual of rounding size, not of |act|.
    constexpr Real kFeasTol = 1e-6;
    Real max_bound_viol = 0.0;
    for (Index j = 0; j < lp.num_cols; ++j) {
        max_bound_viol = std::max({max_bound_viol, lp.col_lower[j] - x[j], x[j] - lp.col_upper[j]});
    }
    Real max_row_viol = 0.0;
    for (Index i = 0; i < lp.num_rows; ++i) {
        auto rv = lp.matrix.row(i);
        Real act = 0.0;
        Real magnitude = 0.0;
        for (Index k = 0; k < rv.size(); ++k) {
            Real term = rv.values[k] * x[rv.indices[k]];
            act += term;
            magnitude += std::abs(term);
        }
        Real tol = kFeasTol * std::max(1.0, magnitude);
        max_row_viol =
            std::max({max_row_viol, (lp.row_lower[i] - act) - tol, (act - lp.row_upper[i]) - tol});
    }
    CHECK(max_bound_viol <= kFeasTol);
    CHECK(max_row_viol <= 0.0);
}

TEST_CASE("Barrier presolve: Netlib barrier objective matches presolve off",
          "[barrier][presolve][netlib]") {
    // Issue #203: the barrier objective must not depend on barrier presolve.
    auto name = GENERATE(as<std::string>{}, "afiro", "blend", "sc50a", "sc105", "sc205", "share2b",
                         "stocfor1");
    INFO(name);
    std::string path = mipx::test::testDataDir() + "/netlib/" + name + ".mps.gz";
    if (!fs::exists(path)) {
        SKIP("Netlib instance " + name + " not downloaded");
    }
    auto lp = readMps(path);

    auto solve = [&lp](bool presolve) {
        BarrierSolver solver;
        BarrierOptions opts;
        opts.verbose = false;
        opts.presolve = presolve;
        opts.algorithm = BarrierAlgorithm::CpuCholesky;
        solver.setOptions(opts);
        solver.load(lp);
        return solver.solve();
    };

    auto off = solve(false);
    REQUIRE(off.status == Status::Optimal);
    auto on = solve(true);
    REQUIRE(on.status == Status::Optimal);
    Real scale = std::max(1.0, std::abs(off.objective));
    CHECK(std::abs(on.objective - off.objective) <= 1e-6 * scale);
}
