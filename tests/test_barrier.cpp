#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "mipx/barrier.h"
#include "mipx/dual_simplex.h"
#include "mipx/mip_solver.h"

using namespace mipx;
using Catch::Matchers::WithinAbs;

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
        {0, 0, 1.0}, {0, 1, 1.0},
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
        {0, 0, 1.0}, {0, 1, 1.0},
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
        {0, 0, 1.0}, {0, 1, 1.0},
        {1, 1, 1.0}, {1, 2, 1.0},
        {2, 0, 1.0}, {2, 2, 1.0},
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
    for (int j = 0; j < N; ++j)
        lp.col_names.push_back("x" + std::to_string(j));

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

    REQUIRE(result.status == Status::Optimal);
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
    REQUIRE(aug_result.status == Status::Optimal);
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
    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(ds_result.objective, 1e-4));
}

TEST_CASE("BarrierSolver: dual simplex fallback on max_iter=1",
          "[barrier][fallback]") {
    auto lp = buildTriangleLp();

    BarrierSolver solver;
    BarrierOptions opts;
    opts.verbose = false;
    opts.algorithm = BarrierAlgorithm::CpuCholesky;
    opts.max_iter = 1;  // Force barrier to fail, triggering dual simplex fallback.
    solver.setOptions(opts);
    solver.load(lp);
    auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(-7.5, 1e-5));
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
