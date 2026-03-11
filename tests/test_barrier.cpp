#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <string>

#include "mipx/barrier.h"
#include "mipx/dual_simplex.h"
#include "mipx/mip_solver.h"

using namespace mipx;
using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;

namespace {

BarrierOptions gpuBarrierOpts() {
    BarrierOptions opts;
    opts.verbose = false;
    opts.use_gpu = true;
    opts.gpu_min_rows = 0;
    opts.gpu_min_nnz = 0;
    return opts;
}

LpResult solveBarrierOrSkip(BarrierSolver& solver) {
    try {
        return solver.solve();
    } catch (const std::exception& e) {
        const std::string msg = e.what();
        if (msg.find("Barrier GPU backend unavailable in this build") !=
                std::string::npos ||
            msg.find("Barrier GPU backend initialization failed") !=
                std::string::npos) {
            SKIP(msg);
        }
        throw;
    }
}

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

// Slightly larger LP: 10 variables, 5 constraints.
LpProblem buildMediumLp() {
    LpProblem lp;
    lp.name = "barrier_medium";
    lp.sense = Sense::Minimize;
    lp.num_cols = 10;
    lp.obj.resize(10);
    lp.col_lower.resize(10, 0.0);
    lp.col_upper.resize(10, kInf);
    lp.col_type.resize(10, VarType::Continuous);
    lp.col_names.resize(10);

    for (int j = 0; j < 10; ++j) {
        lp.obj[j] = static_cast<double>(j + 1);
        lp.col_names[j] = "x" + std::to_string(j);
    }

    lp.num_rows = 5;
    lp.row_lower.resize(5, -kInf);
    lp.row_upper.resize(5);
    lp.row_names.resize(5);

    // Constraint 0: x0 + x1 + x2 <= 10
    // Constraint 1: x3 + x4 + x5 <= 15
    // Constraint 2: x6 + x7 + x8 + x9 <= 20
    // Constraint 3: x0 + x3 + x6 <= 8
    // Constraint 4: x1 + x4 + x7 <= 12
    lp.row_upper = {10.0, 15.0, 20.0, 8.0, 12.0};
    for (int i = 0; i < 5; ++i)
        lp.row_names[i] = "c" + std::to_string(i);

    std::vector<Triplet> trips = {
        {0, 0, 1.0}, {0, 1, 1.0}, {0, 2, 1.0},
        {1, 3, 1.0}, {1, 4, 1.0}, {1, 5, 1.0},
        {2, 6, 1.0}, {2, 7, 1.0}, {2, 8, 1.0}, {2, 9, 1.0},
        {3, 0, 1.0}, {3, 3, 1.0}, {3, 6, 1.0},
        {4, 1, 1.0}, {4, 4, 1.0}, {4, 7, 1.0},
    };
    lp.matrix = SparseMatrix(5, 10, std::move(trips));
    return lp;
}

// Maximization problem.
LpProblem buildMaxLp() {
    LpProblem lp;
    lp.name = "barrier_max";
    lp.sense = Sense::Maximize;
    lp.num_cols = 2;
    lp.obj = {3.0, 5.0};
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {4.0, 6.0};
    lp.col_type = {VarType::Continuous, VarType::Continuous};
    lp.col_names = {"x", "y"};

    lp.num_rows = 2;
    lp.row_lower = {-kInf, -kInf};
    lp.row_upper = {10.0, 8.0};
    lp.row_names = {"c1", "c2"};

    // x + y <= 10
    // x + 2y <= 8 (wait, that doesn't make sense with the bounds)
    // Actually: x + y <= 10, 2x + y <= 14
    std::vector<Triplet> trips = {
        {0, 0, 1.0}, {0, 1, 1.0},
        {1, 0, 2.0}, {1, 1, 1.0},
    };
    lp.matrix = SparseMatrix(2, 2, std::move(trips));
    lp.row_upper = {10.0, 14.0};
    return lp;
}

Real originalObjective(const LpProblem& lp, std::span<const Real> x) {
    Real obj = lp.obj_offset;
    for (Index j = 0; j < lp.num_cols; ++j) {
        obj += lp.obj[j] * x[j];
    }
    return obj;
}

}  // namespace

TEST_CASE("BarrierSolver: solves simple LP", "[barrier]") {
    auto lp = buildSimpleLp();

    BarrierSolver solver;
    BarrierOptions opts = gpuBarrierOpts();
    solver.setOptions(opts);
    solver.load(lp);
    auto result = solveBarrierOrSkip(solver);

    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(-4.0, 1e-5));
    auto x = solver.getPrimalValues();
    REQUIRE(x.size() == 2);
    CHECK(x[0] >= -1e-6);
    CHECK(x[1] >= -1e-6);
    CHECK(x[0] + x[1] <= 4.0 + 1e-5);
}

TEST_CASE("BarrierSolver: handles bounds-only LP", "[barrier]") {
    auto lp = buildBoundsOnlyLp();

    BarrierSolver solver;
    BarrierOptions opts = gpuBarrierOpts();
    solver.setOptions(opts);
    solver.load(lp);
    auto result = solveBarrierOrSkip(solver);

    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(1.0, 1e-6));
    auto x = solver.getPrimalValues();
    REQUIRE(x.size() == 1);
    CHECK_THAT(x[0], WithinAbs(1.0, 1e-5));
}

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
    barrier_solver.setBarrierUseGpu(true);
    barrier_solver.setBarrierGpuThresholds(0, 0);
    barrier_solver.load(lp);
    MipResult barrier_result{};
    try {
        barrier_result = barrier_solver.solve();
    } catch (const std::exception& e) {
        const std::string msg = e.what();
        if (msg.find("Barrier GPU backend unavailable in this build") !=
                std::string::npos ||
            msg.find("Barrier GPU thresholds not met") !=
                std::string::npos ||
            msg.find("Barrier GPU backend initialization failed") !=
                std::string::npos) {
            SKIP(msg);
        }
        throw;
    }

    REQUIRE(dual_result.status == Status::Optimal);
    REQUIRE(barrier_result.status == Status::Optimal);
    CHECK_THAT(barrier_result.objective, WithinAbs(dual_result.objective, 1e-4));
    CHECK(dual_result.work_units > 0.0);
    CHECK(barrier_result.work_units > 0.0);
}

TEST_CASE("BarrierSolver: medium LP with scaling", "[barrier]") {
    auto lp = buildMediumLp();

    BarrierSolver solver;
    BarrierOptions opts = gpuBarrierOpts();
    opts.ruiz_iterations = 10;
    solver.setOptions(opts);
    solver.load(lp);
    auto result = solveBarrierOrSkip(solver);

    REQUIRE(result.status == Status::Optimal);
    CHECK(result.objective >= -1e-6);  // All costs positive, lower bounds 0.

    auto x = solver.getPrimalValues();
    REQUIRE(x.size() == 10);
    for (int j = 0; j < 10; ++j) {
        CHECK(x[j] >= -1e-6);
    }

    // Verify against dual simplex.
    DualSimplexSolver ds;
    ds.load(lp);
    ds.setVerbose(false);
    auto ds_result = ds.solve();
    REQUIRE(ds_result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(ds_result.objective, 1e-4));
}

TEST_CASE("BarrierSolver: maximization problem", "[barrier]") {
    auto lp = buildMaxLp();

    BarrierSolver solver;
    BarrierOptions opts = gpuBarrierOpts();
    solver.setOptions(opts);
    solver.load(lp);
    auto result = solveBarrierOrSkip(solver);

    REQUIRE(result.status == Status::Optimal);

    // Compare with dual simplex.
    DualSimplexSolver ds;
    ds.load(lp);
    ds.setVerbose(false);
    auto ds_result = ds.solve();
    REQUIRE(ds_result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(ds_result.objective, 1e-4));
}

TEST_CASE("BarrierSolver: reported objective matches reconstructed primal",
          "[barrier]") {
    auto lp = buildMaxLp();

    BarrierSolver solver;
    BarrierOptions opts = gpuBarrierOpts();
    solver.setOptions(opts);
    solver.load(lp);
    auto result = solveBarrierOrSkip(solver);

    REQUIRE(result.status == Status::Optimal);
    auto x = solver.getPrimalValues();
    REQUIRE(x.size() == static_cast<size_t>(lp.num_cols));
    CHECK_THAT(result.objective, WithinAbs(originalObjective(lp, x), 1e-6));
}

TEST_CASE("BarrierSolver: scaling with no scaling gives same result", "[barrier]") {
    auto lp = buildSimpleLp();

    // With scaling.
    BarrierSolver solver1;
    BarrierOptions opts1 = gpuBarrierOpts();
    opts1.ruiz_iterations = 10;
    solver1.setOptions(opts1);
    solver1.load(lp);
    auto r1 = solveBarrierOrSkip(solver1);

    // Without scaling.
    BarrierSolver solver2;
    BarrierOptions opts2 = gpuBarrierOpts();
    opts2.ruiz_iterations = 0;
    solver2.setOptions(opts2);
    solver2.load(lp);
    auto r2 = solveBarrierOrSkip(solver2);

    REQUIRE(r1.status == Status::Optimal);
    REQUIRE(r2.status == Status::Optimal);
    CHECK_THAT(r1.objective, WithinAbs(r2.objective, 1e-4));
}

TEST_CASE("BarrierSolver: dense column detection", "[barrier]") {
    // Create LP where one column appears in every constraint.
    LpProblem lp;
    lp.name = "barrier_dense_col";
    lp.sense = Sense::Minimize;
    lp.num_cols = 4;
    lp.obj = {1.0, 2.0, 3.0, 4.0};
    lp.col_lower = {0.0, 0.0, 0.0, 0.0};
    lp.col_upper = {kInf, kInf, kInf, kInf};
    lp.col_type.resize(4, VarType::Continuous);
    lp.col_names = {"x0", "x1", "x2", "x3"};

    lp.num_rows = 3;
    lp.row_lower = {-kInf, -kInf, -kInf};
    lp.row_upper = {10.0, 15.0, 20.0};
    lp.row_names = {"c0", "c1", "c2"};

    // x0 appears in all 3 rows — "dense" column.
    std::vector<Triplet> trips = {
        {0, 0, 1.0}, {0, 1, 1.0},
        {1, 0, 1.0}, {1, 2, 1.0},
        {2, 0, 1.0}, {2, 3, 1.0},
    };
    lp.matrix = SparseMatrix(3, 4, std::move(trips));

    BarrierSolver solver;
    BarrierOptions opts = gpuBarrierOpts();
    opts.dense_col_fraction = 0.5;  // threshold: nnz > 0.5 * 3 = 1.5, so col 0 is dense.
    solver.setOptions(opts);
    solver.load(lp);
    auto result = solveBarrierOrSkip(solver);

    REQUIRE(result.status == Status::Optimal);

    // Verify against dual simplex.
    DualSimplexSolver ds;
    ds.load(lp);
    ds.setVerbose(false);
    auto ds_result = ds.solve();
    REQUIRE(ds_result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(ds_result.objective, 1e-4));
}

TEST_CASE("BarrierSolver: reports GPU-only requirement when GPU is disabled",
          "[barrier]") {
    auto lp = buildSimpleLp();

    BarrierSolver solver;
    BarrierOptions opts;
    opts.verbose = false;
    opts.use_gpu = false;
    solver.setOptions(opts);
    solver.load(lp);

    REQUIRE_THROWS_WITH(
        solver.solve(),
        Catch::Matchers::ContainsSubstring(
            "Barrier currently requires GPU backend"));
}

TEST_CASE("BarrierSolver: reports threshold gating explicitly", "[barrier]") {
    auto lp = buildSimpleLp();

    BarrierSolver solver;
    BarrierOptions opts;
    opts.verbose = false;
    opts.use_gpu = true;
    solver.setOptions(opts);
    solver.load(lp);

    REQUIRE_THROWS_WITH(
        solver.solve(),
        Catch::Matchers::ContainsSubstring("Barrier GPU thresholds not met"));
}

TEST_CASE("BarrierSolver: reports augmented backend as unsupported", "[barrier]") {
    auto lp = buildSimpleLp();

    BarrierSolver solver;
    BarrierOptions opts = gpuBarrierOpts();
    opts.backend = BarrierBackend::Augmented;
    solver.setOptions(opts);
    solver.load(lp);

    REQUIRE_THROWS_WITH(solver.solve(),
                        ContainsSubstring("augmented-system backend requested"));
}

TEST_CASE("BarrierSolver: reports AMD ordering as unsupported", "[barrier]") {
    auto lp = buildSimpleLp();

    BarrierSolver solver;
    BarrierOptions opts = gpuBarrierOpts();
    opts.ordering = BarrierOrdering::Amd;
    solver.setOptions(opts);
    solver.load(lp);

    REQUIRE_THROWS_WITH(solver.solve(),
                        ContainsSubstring("AMD ordering requested"));
}

TEST_CASE("BarrierSolver: reports barrier preprocessing options as unsupported",
          "[barrier]") {
    auto lp = buildSimpleLp();

    SECTION("dualize on") {
        BarrierSolver solver;
        BarrierOptions opts = gpuBarrierOpts();
        opts.dualize = BarrierToggle::On;
        solver.setOptions(opts);
        solver.load(lp);
        REQUIRE_THROWS_WITH(solver.solve(),
                            ContainsSubstring("dualization requested"));
    }

    SECTION("folding on") {
        BarrierSolver solver;
        BarrierOptions opts = gpuBarrierOpts();
        opts.folding = BarrierToggle::On;
        solver.setOptions(opts);
        solver.load(lp);
        REQUIRE_THROWS_WITH(solver.solve(),
                            ContainsSubstring("folding requested"));
    }

    SECTION("explicit dual initial point") {
        BarrierSolver solver;
        BarrierOptions opts = gpuBarrierOpts();
        opts.dual_initial_point =
            BarrierDualInitialPoint::LustigMarstenShanno;
        solver.setOptions(opts);
        solver.load(lp);
        REQUIRE_THROWS_WITH(solver.solve(),
                            ContainsSubstring("initial-point modes"));
    }

    SECTION("dense-column elimination") {
        BarrierSolver solver;
        BarrierOptions opts = gpuBarrierOpts();
        opts.eliminate_dense_columns = true;
        solver.setOptions(opts);
        solver.load(lp);
        REQUIRE_THROWS_WITH(solver.solve(),
                            ContainsSubstring("dense-column elimination"));
    }
}

#ifdef MIPX_HAS_CUDA
TEST_CASE("BarrierSolver: GPU backend honors ir_steps setting", "[barrier]") {
    auto lp = buildSimpleLp();

    BarrierSolver solver0;
    BarrierOptions opts0 = gpuBarrierOpts();
    opts0.ir_steps = 0;
    solver0.setOptions(opts0);
    solver0.load(lp);
    auto r0 = solveBarrierOrSkip(solver0);

    BarrierSolver solver3;
    BarrierOptions opts3 = gpuBarrierOpts();
    opts3.ir_steps = 3;
    solver3.setOptions(opts3);
    solver3.load(lp);
    auto r3 = solveBarrierOrSkip(solver3);

    REQUIRE(r0.status == Status::Optimal);
    REQUIRE(r3.status == Status::Optimal);
    CHECK_THAT(r0.objective, WithinAbs(r3.objective, 1e-5));
}
#endif

#ifndef MIPX_HAS_CUDA
TEST_CASE("BarrierSolver: reports unavailable in non-CUDA build", "[barrier]") {
    auto lp = buildSimpleLp();

    BarrierSolver solver;
    BarrierOptions opts = gpuBarrierOpts();
    solver.setOptions(opts);
    solver.load(lp);

    REQUIRE_THROWS_WITH(
        solver.solve(),
        Catch::Matchers::ContainsSubstring("Barrier GPU backend unavailable in this build"));
}
#endif
