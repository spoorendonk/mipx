#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <random>
#include <utility>

#include "mipx/mip_solver.h"
#include "mipx/pdlp.h"

using namespace mipx;
using Catch::Matchers::WithinAbs;

namespace {

LpProblem buildSimpleLp() {
    LpProblem lp;
    lp.name = "pdlp_simple";
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
    lp.name = "pdlp_bounds";
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
    lp.name = "branching_mip_pdlp";
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

}  // namespace

TEST_CASE("PdlpSolver: solves simple LP", "[pdlp]") {
    auto lp = buildSimpleLp();

    PdlpSolver solver;
    PdlpOptions opts;
    opts.verbose = false;
    opts.max_iter = 10000;
    solver.setOptions(opts);
    solver.load(lp);
    auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(-4.0, 1e-4));
    CHECK(result.work_units > 0.0);
    auto x = solver.getPrimalValues();
    REQUIRE(x.size() == 2);
    CHECK(x[0] >= -1e-6);
    CHECK(x[1] >= -1e-6);
    CHECK(x[0] + x[1] <= 4.0 + 1e-3);
}

TEST_CASE("PdlpSolver: handles bounds-only LP", "[pdlp]") {
    auto lp = buildBoundsOnlyLp();

    PdlpSolver solver;
    PdlpOptions opts;
    opts.verbose = false;
    solver.setOptions(opts);
    solver.load(lp);
    auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(1.0, 1e-6));
    auto x = solver.getPrimalValues();
    REQUIRE(x.size() == 1);
    CHECK_THAT(x[0], WithinAbs(1.0, 1e-5));
}

TEST_CASE("PdlpSolver: solves simple LP with default options", "[pdlp]") {
    auto lp = buildSimpleLp();

    PdlpSolver solver;
    PdlpOptions opts;
    opts.verbose = false;
    opts.max_iter = 10000;
    solver.setOptions(opts);
    solver.load(lp);
    auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(-4.0, 1e-4));
}

TEST_CASE("PDLP root policy: MIP objective matches dual root policy", "[pdlp][mip]") {
    auto lp = buildBranchingMip();

    MipSolver dual_solver;
    dual_solver.setVerbose(false);
    dual_solver.setCutsEnabled(false);
    dual_solver.setRootLpPolicy(RootLpPolicy::DualDefault);
    dual_solver.load(lp);
    auto dual_result = dual_solver.solve();

    MipSolver pdlp_solver;
    pdlp_solver.setVerbose(false);
    pdlp_solver.setCutsEnabled(false);
    pdlp_solver.setRootLpPolicy(RootLpPolicy::PdlpRoot);

    pdlp_solver.load(lp);
    auto pdlp_result = pdlp_solver.solve();

    REQUIRE(dual_result.status == Status::Optimal);
    REQUIRE(pdlp_result.status == Status::Optimal);
    CHECK_THAT(pdlp_result.objective, WithinAbs(dual_result.objective, 1e-4));
    CHECK(dual_result.work_units > 0.0);
    CHECK(pdlp_result.work_units > 0.0);
}

TEST_CASE("Concurrent root policy: MIP objective matches dual root policy",
          "[pdlp][mip][concurrent]") {
    auto lp = buildBranchingMip();

    MipSolver dual_solver;
    dual_solver.setVerbose(false);
    dual_solver.setCutsEnabled(false);
    dual_solver.setRootLpPolicy(RootLpPolicy::DualDefault);
    dual_solver.load(lp);
    auto dual_result = dual_solver.solve();

    MipSolver concurrent_solver;
    concurrent_solver.setVerbose(false);
    concurrent_solver.setCutsEnabled(false);
    concurrent_solver.setHeuristicMode(HeuristicRuntimeMode::Deterministic);
    concurrent_solver.setRootLpPolicy(RootLpPolicy::ConcurrentRootExperimental);
    concurrent_solver.setBarrierUseGpu(false);

    concurrent_solver.load(lp);
    auto concurrent_result = concurrent_solver.solve();

    REQUIRE(dual_result.status == Status::Optimal);
    REQUIRE(concurrent_result.status == Status::Optimal);
    CHECK_THAT(concurrent_result.objective, WithinAbs(dual_result.objective, 1e-4));
    CHECK(dual_result.work_units > 0.0);
    CHECK(concurrent_result.work_units > 0.0);
}

TEST_CASE("Concurrent root deterministic mode is reproducible",
          "[pdlp][mip][concurrent]") {
    auto lp = buildBranchingMip();

    auto run_once = [&]() {
        MipSolver solver;
        solver.setVerbose(false);
        solver.setCutsEnabled(false);
        solver.setHeuristicMode(HeuristicRuntimeMode::Deterministic);
        solver.setRootLpPolicy(RootLpPolicy::ConcurrentRootExperimental);
        solver.setBarrierUseGpu(false);

        solver.load(lp);
        auto result = solver.solve();
        return std::make_pair(result, solver.getLpStats());
    };

    const auto [a_result, a_stats] = run_once();
    const auto [b_result, b_stats] = run_once();

    REQUIRE(a_result.status == Status::Optimal);
    REQUIRE(b_result.status == Status::Optimal);
    CHECK_THAT(a_result.objective, WithinAbs(b_result.objective, 1e-9));
    CHECK(a_result.work_units > 0.0);
    CHECK_THAT(a_result.work_units, WithinAbs(b_result.work_units, 1e-9));
    CHECK(a_stats.root_race_runs == 1);
    CHECK(a_stats.root_race_candidates == 3);
    CHECK(a_stats.root_race_dual_wins + a_stats.root_race_barrier_wins +
              a_stats.root_race_pdlp_wins ==
          1);
    CHECK(b_stats.root_race_runs == 1);
}

namespace {

LpProblem buildMediumLp() {
    // A ~50-row LP with dense-ish constraints to exercise both GPU and CPU paths.
    // min sum(j, c_j * x_j)  s.t.  sum(j, A_ij * x_j) <= b_i,  x_j >= 0
    const Index rows = 50;
    const Index cols = 80;

    LpProblem lp;
    lp.name = "gpu_vs_cpu_medium";
    lp.sense = Sense::Minimize;
    lp.num_cols = cols;
    lp.num_rows = rows;

    // Objective: alternating +/- coefficients.
    lp.obj.resize(cols);
    for (Index j = 0; j < cols; ++j)
        lp.obj[j] = (j % 2 == 0) ? 1.0 + 0.1 * j : -(0.5 + 0.05 * j);

    lp.col_lower.assign(cols, 0.0);
    lp.col_upper.assign(cols, 100.0);
    lp.col_type.assign(cols, VarType::Continuous);

    lp.row_lower.assign(rows, -kInf);
    lp.row_upper.resize(rows);
    for (Index i = 0; i < rows; ++i)
        lp.row_upper[i] = 50.0 + 2.0 * i;

    // Build constraint matrix with ~60% density.
    std::vector<Triplet> trips;
    std::mt19937 rng(123);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (Index i = 0; i < rows; ++i) {
        for (Index j = 0; j < cols; ++j) {
            if (dist(rng) < 0.6) {
                trips.push_back({i, j, 0.1 + dist(rng) * 2.0});
            }
        }
    }
    lp.matrix = SparseMatrix(rows, cols, std::move(trips));
    return lp;
}

}  // namespace

TEST_CASE("PdlpSolver: handles empty problem (no variables)", "[pdlp]") {
    LpProblem lp;
    lp.name = "pdlp_empty";
    lp.sense = Sense::Minimize;
    lp.num_cols = 0;
    lp.num_rows = 0;
    lp.obj_offset = 42.0;
    lp.matrix = SparseMatrix(0, 0, std::vector<Triplet>{});

    PdlpSolver solver;
    PdlpOptions opts;
    opts.verbose = false;
    solver.setOptions(opts);
    solver.load(lp);
    auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(42.0, 1e-9));
    CHECK(result.iterations == 0);
}

TEST_CASE("PdlpSolver: detects unbounded bounds-only LP", "[pdlp]") {
    // min -x  s.t. x >= 0, no upper bound, no constraints → unbounded
    LpProblem lp;
    lp.name = "pdlp_unbounded";
    lp.sense = Sense::Minimize;
    lp.num_cols = 1;
    lp.obj = {-1.0};
    lp.col_lower = {0.0};
    lp.col_upper = {kInf};
    lp.col_type = {VarType::Continuous};
    lp.num_rows = 0;
    lp.matrix = SparseMatrix(0, 1, std::vector<Triplet>{});

    PdlpSolver solver;
    PdlpOptions opts;
    opts.verbose = false;
    solver.setOptions(opts);
    solver.load(lp);
    auto result = solver.solve();

    REQUIRE(result.status == Status::Unbounded);
}

TEST_CASE("PdlpSolver: infeasible LP hits iteration limit", "[pdlp]") {
    // min x  s.t. x <= -1, x >= 1 → infeasible
    // PDLP is a first-order method; it won't certify infeasibility but will
    // fail to converge and hit the iteration limit.
    LpProblem lp;
    lp.name = "pdlp_infeasible";
    lp.sense = Sense::Minimize;
    lp.num_cols = 1;
    lp.obj = {1.0};
    lp.col_lower = {1.0};
    lp.col_upper = {kInf};
    lp.col_type = {VarType::Continuous};

    lp.num_rows = 1;
    lp.row_lower = {-kInf};
    lp.row_upper = {-1.0};  // x <= -1
    lp.row_names = {"c1"};

    std::vector<Triplet> trips = {{0, 0, 1.0}};
    lp.matrix = SparseMatrix(1, 1, std::move(trips));

    PdlpSolver solver;
    PdlpOptions opts;
    opts.verbose = false;
    opts.max_iter = 1000;  // low limit so test is fast
    solver.setOptions(opts);
    solver.load(lp);
    auto result = solver.solve();

    CHECK(result.status == Status::IterLimit);
}

TEST_CASE("PdlpSolver: GPU and CPU produce matching objectives", "[pdlp][gpu]") {
    auto lp = buildMediumLp();

    PdlpSolver cpu_solver;
    PdlpOptions cpu_opts;
    cpu_opts.verbose = false;
    cpu_opts.use_gpu = false;
    cpu_solver.setOptions(cpu_opts);
    cpu_solver.load(lp);
    auto cpu_result = cpu_solver.solve();

    PdlpSolver gpu_solver;
    PdlpOptions gpu_opts;
    gpu_opts.verbose = false;
    gpu_opts.use_gpu = true;
    gpu_opts.gpu_min_rows = 0;
    gpu_opts.gpu_min_nnz = 0;
    gpu_solver.setOptions(gpu_opts);
    gpu_solver.load(lp);
    auto gpu_result = gpu_solver.solve();

    REQUIRE(cpu_result.status == Status::Optimal);
    REQUIRE(gpu_result.status == Status::Optimal);
    // If CUDA is available, confirm GPU was actually used.
    // If not, both are CPU — test still passes.
#ifdef MIPX_HAS_CUDA
    CHECK(gpu_solver.usedGpu());
#endif
    CHECK_THAT(gpu_result.objective, WithinAbs(cpu_result.objective, 1e-4));
}

// ---------------------------------------------------------------------------
// Bound-objective rescaling tests
// ---------------------------------------------------------------------------

namespace {

LpResult solvePdlp(const LpProblem& lp, PdlpOptions opts) {
    PdlpSolver solver;
    solver.setOptions(opts);
    solver.load(lp);
    return solver.solve();
}

PdlpOptions quietOpts(Int max_iter = 50000, bool rescaling = true,
                       bool gpu = false) {
    PdlpOptions opts;
    opts.verbose = false;
    opts.max_iter = max_iter;
    opts.do_bound_obj_rescaling = rescaling;
    opts.use_gpu = gpu;
    return opts;
}

}  // namespace

TEST_CASE("PdlpSolver: bound-obj rescaling matches no-rescaling on simple LP",
          "[pdlp][rescaling]") {
    auto lp = buildSimpleLp();

    auto result_on  = solvePdlp(lp, quietOpts(50000, true));
    auto result_off = solvePdlp(lp, quietOpts(50000, false));

    REQUIRE(result_on.status == Status::Optimal);
    REQUIRE(result_off.status == Status::Optimal);
    CHECK_THAT(result_on.objective, WithinAbs(-4.0, 1e-4));
    CHECK_THAT(result_off.objective, WithinAbs(-4.0, 1e-4));
    CHECK_THAT(result_on.objective, WithinAbs(result_off.objective, 1e-3));
}

TEST_CASE("PdlpSolver: bound-obj rescaling on medium LP", "[pdlp][rescaling]") {
    auto lp = buildMediumLp();

    auto result_on  = solvePdlp(lp, quietOpts(500000, true));
    auto result_off = solvePdlp(lp, quietOpts(500000, false));

    REQUIRE(result_on.status == Status::Optimal);
    REQUIRE(result_off.status == Status::Optimal);
    // First-order methods converge to ~1e-4 relative tolerance, so
    // objectives may differ by O(0.1) on this ~425-obj problem.
    CHECK_THAT(result_on.objective, WithinAbs(result_off.objective, 1.0));
}

TEST_CASE("PdlpSolver: rescaling with binding column bounds", "[pdlp][rescaling]") {
    // min -x - 2y  s.t.  x + y <= 100,  0 <= x <= 3,  0 <= y <= 5
    // Optimal: x=3, y=5, obj = -3 - 10 = -13 (column bounds bind, not row).
    LpProblem lp;
    lp.name = "pdlp_col_bound_binding";
    lp.sense = Sense::Minimize;
    lp.num_cols = 2;
    lp.obj = {-1.0, -2.0};
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {3.0, 5.0};
    lp.col_type = {VarType::Continuous, VarType::Continuous};

    lp.num_rows = 1;
    lp.row_lower = {-kInf};
    lp.row_upper = {100.0};
    lp.row_names = {"slack"};

    std::vector<Triplet> trips = {{0, 0, 1.0}, {0, 1, 1.0}};
    lp.matrix = SparseMatrix(1, 2, std::move(trips));

    auto result = solvePdlp(lp, quietOpts(50000, true));

    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(-13.0, 1e-3));
}

TEST_CASE("PdlpSolver: rescaling with large rhs disparity", "[pdlp][rescaling]") {
    // min -x - y  s.t.  x <= 1000,  y <= 2,  x,y >= 0
    // Large rhs disparity exercises constraint_scale normalization.
    // Optimal: x=1000, y=2, obj = -1002.
    LpProblem lp;
    lp.name = "pdlp_large_rhs";
    lp.sense = Sense::Minimize;
    lp.num_cols = 2;
    lp.obj = {-1.0, -1.0};
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {kInf, kInf};
    lp.col_type = {VarType::Continuous, VarType::Continuous};

    lp.num_rows = 2;
    lp.row_lower = {-kInf, -kInf};
    lp.row_upper = {1000.0, 2.0};
    lp.row_names = {"big", "small"};

    std::vector<Triplet> trips = {{0, 0, 1.0}, {1, 1, 1.0}};
    lp.matrix = SparseMatrix(2, 2, std::move(trips));

    auto result = solvePdlp(lp, quietOpts(100000, true));

    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(-1002.0, 1.0));
}

TEST_CASE("PdlpSolver: rescaling with equality constraints", "[pdlp][rescaling]") {
    // min x + 2y  s.t.  x + y = 10,  x,y >= 0
    // Optimal: x=10, y=0, obj = 10.
    LpProblem lp;
    lp.name = "pdlp_equality";
    lp.sense = Sense::Minimize;
    lp.num_cols = 2;
    lp.obj = {1.0, 2.0};
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {kInf, kInf};
    lp.col_type = {VarType::Continuous, VarType::Continuous};

    lp.num_rows = 1;
    lp.row_lower = {10.0};
    lp.row_upper = {10.0};
    lp.row_names = {"eq"};

    std::vector<Triplet> trips = {{0, 0, 1.0}, {0, 1, 1.0}};
    lp.matrix = SparseMatrix(1, 2, std::move(trips));

    auto result = solvePdlp(lp, quietOpts(100000, true));

    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(10.0, 0.1));
}

// ---------------------------------------------------------------------------
// Rescaling regression / GPU consistency
// ---------------------------------------------------------------------------

TEST_CASE("PdlpSolver: rescaling does not regress iteration count on medium LP",
          "[pdlp][rescaling][performance]") {
    auto lp = buildMediumLp();

    auto result_on  = solvePdlp(lp, quietOpts(500000, true));
    auto result_off = solvePdlp(lp, quietOpts(500000, false));

    REQUIRE(result_on.status == Status::Optimal);
    REQUIRE(result_off.status == Status::Optimal);
    // On well-scaled problems rescaling may not help, but shouldn't hurt much.
    CHECK(result_on.iterations <= result_off.iterations * 2);
}

TEST_CASE("PdlpSolver: GPU with rescaling matches CPU", "[pdlp][rescaling][gpu]") {
    auto lp = buildMediumLp();

    auto cpu_result = solvePdlp(lp, quietOpts(500000, true, false));

    auto gpu_opts = quietOpts(500000, true, true);
    gpu_opts.gpu_min_rows = 0;
    gpu_opts.gpu_min_nnz = 0;
    auto gpu_result = solvePdlp(lp, gpu_opts);

    REQUIRE(cpu_result.status == Status::Optimal);
    REQUIRE(gpu_result.status == Status::Optimal);
    CHECK_THAT(gpu_result.objective, WithinAbs(cpu_result.objective, 1e-3));
}
