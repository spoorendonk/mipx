#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <utility>

#include "mipx/dual_simplex.h"
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
    opts.use_gpu = false;
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
    CHECK(x[0] + x[1] <= 4.0 + 1e-5);
}

TEST_CASE("PdlpSolver: handles bounds-only LP", "[pdlp]") {
    auto lp = buildBoundsOnlyLp();

    PdlpSolver solver;
    PdlpOptions opts;
    opts.verbose = false;
    opts.use_gpu = false;
    solver.setOptions(opts);
    solver.load(lp);
    auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(1.0, 1e-6));
    auto x = solver.getPrimalValues();
    REQUIRE(x.size() == 1);
    CHECK_THAT(x[0], WithinAbs(1.0, 1e-5));
}

TEST_CASE("PdlpSolver: GPU fallback keeps correctness", "[pdlp]") {
    auto lp = buildSimpleLp();

    PdlpSolver solver;
    PdlpOptions opts;
    opts.verbose = false;
    opts.use_gpu = true;
    opts.gpu_min_rows = 0;
    opts.gpu_min_nnz = 0;
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
    pdlp_solver.setPdlpUseGpu(false);
    pdlp_solver.load(lp);
    auto pdlp_result = pdlp_solver.solve();

    REQUIRE(dual_result.status == Status::Optimal);
    REQUIRE(pdlp_result.status == Status::Optimal);
    CHECK_THAT(pdlp_result.objective, WithinAbs(dual_result.objective, 1e-4));
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
    concurrent_solver.setPdlpUseGpu(false);
    concurrent_solver.load(lp);
    auto concurrent_result = concurrent_solver.solve();

    REQUIRE(dual_result.status == Status::Optimal);
    REQUIRE(concurrent_result.status == Status::Optimal);
    CHECK_THAT(concurrent_result.objective, WithinAbs(dual_result.objective, 1e-4));
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
        solver.setPdlpUseGpu(false);
        solver.load(lp);
        auto result = solver.solve();
        return std::make_pair(result, solver.getLpStats());
    };

    const auto [a_result, a_stats] = run_once();
    const auto [b_result, b_stats] = run_once();

    REQUIRE(a_result.status == Status::Optimal);
    REQUIRE(b_result.status == Status::Optimal);
    CHECK_THAT(a_result.objective, WithinAbs(b_result.objective, 1e-9));
    CHECK(a_stats.root_race_runs == 1);
    CHECK(a_stats.root_race_candidates == 3);
    CHECK(a_stats.root_race_dual_wins + a_stats.root_race_barrier_wins +
              a_stats.root_race_pdlp_wins ==
          1);
    CHECK(b_stats.root_race_runs == 1);
}
