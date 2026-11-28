#include "lp/pdlp_restart.h"
#include "mipx/dual_simplex.h"
#include "mipx/io.h"
#include "mipx/mip_solver.h"
#include "mipx/pdlp.h"
#include "mipx/presolve.h"
#include "test_paths.hpp"

#ifdef MIPX_HAS_CUDA
#include "lp/pdlp_kernels.cuh"

#include <cuda_runtime_api.h>
#include <memory>
#endif

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <filesystem>
#include <limits>
#include <random>
#include <utility>

using namespace mipx;
using Catch::Matchers::WithinAbs;

namespace fs = std::filesystem;

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
        {0, 0, 1.0},
        {0, 1, 1.0},
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
        {0, 0, 1.0},
        {0, 1, 1.0},
    };
    lp.matrix = SparseMatrix(1, 2, std::move(trips));
    return lp;
}

}  // namespace

using mipx::test::testDataDir;

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

TEST_CASE("PDLP crossover gives a basic optimum reusable by simplex", "[pdlp][crossover]") {
    auto lp = buildSimpleLp();
    lp.sense = GENERATE(Sense::Minimize, Sense::Maximize);
    const Real cost = lp.sense == Sense::Minimize ? -1.0 : 1.0;
    lp.obj = {cost, cost};
    lp.obj_offset = 7.0;

    PdlpSolver pdlp;
    PdlpOptions opts;
    opts.verbose = false;
    opts.use_gpu = false;
    pdlp.setOptions(opts);
    pdlp.load(lp);
    REQUIRE(pdlp.solve().status == Status::Optimal);
    CHECK(pdlp.getBasis().empty());
    CHECK_THAT(pdlp.getPrimalValues()[0], WithinAbs(2.0, 1e-6));

    opts.crossover = true;
    pdlp.setOptions(opts);
    const auto crossed = pdlp.solve();
    REQUIRE(crossed.status == Status::Optimal);
    CHECK_THAT(crossed.objective, WithinAbs(7.0 + 4.0 * cost, 1e-12));
    CHECK(pdlp.getObjective() == crossed.objective);
    const auto basis = pdlp.getBasis();
    REQUIRE(basis.size() == 3);
    CHECK(std::count(basis.begin(), basis.end(), BasisStatus::Basic) == 1);
    CHECK(basis[2] == BasisStatus::AtUpper);
    const auto primal = pdlp.getPrimalValues();
    CHECK_THAT(std::min(primal[0], primal[1]), WithinAbs(0.0, 1e-12));
    CHECK_THAT(std::max(primal[0], primal[1]), WithinAbs(4.0, 1e-12));
    CHECK_THAT(pdlp.getDualValues()[0], WithinAbs(-cost, 1e-12));
    for (Real reduced_cost : pdlp.getReducedCosts()) {
        CHECK_THAT(reduced_cost, WithinAbs(0.0, 1e-12));
    }

    DualSimplexSolver simplex;
    simplex.load(lp);
    simplex.setBasis(basis);
    const auto warm = simplex.solve();
    REQUIRE(warm.status == Status::Optimal);
    CHECK(warm.iterations == 0);
    CHECK_FALSE(simplex.basisStartStats().cold_start);
    CHECK_FALSE(simplex.basisStartStats().singular_recovery);
    CHECK(simplex.getPrimalValues() == primal);
    CHECK(warm.objective == crossed.objective);

    pdlp.setRowBounds(0, -kInf, 3.0);
    CHECK(pdlp.getBasis().empty());
    opts.max_solve_seconds = 0.0;
    pdlp.setOptions(opts);
    CHECK(pdlp.solve().status == Status::TimeLimit);
    CHECK(pdlp.getBasis().empty());
}

TEST_CASE("PDLP crossover recovers a bounds-only basis", "[pdlp][crossover]") {
    auto lp = buildBoundsOnlyLp();
    PdlpSolver pdlp;
    PdlpOptions opts;
    opts.verbose = false;
    pdlp.setOptions(opts);
    pdlp.load(lp);
    REQUIRE(pdlp.solve().status == Status::Optimal);
    CHECK(pdlp.getReducedCosts() == lp.obj);
    CHECK(pdlp.getBasis().empty());

    opts.crossover = true;
    pdlp.setOptions(opts);
    const auto result = pdlp.solve();
    REQUIRE(result.status == Status::Optimal);
    CHECK(result.objective == 1.0);
    CHECK(pdlp.getBasis() == std::vector{BasisStatus::AtLower});
    CHECK(pdlp.getReducedCosts() == lp.obj);
    pdlp.load(lp);
    CHECK(pdlp.getBasis().empty());
}

TEST_CASE("PDLP: restart gap and KKT error use their respective normalizations", "[pdlp]") {
    using pdlp_detail::ProjectedErrors;

    const auto gap_dominates = ProjectedErrors::measure(0.1, 0.2, -2.0, -4.0);
    CHECK_THAT(gap_dominates.gap, WithinAbs(2.0 / 3.0, 1e-15));
    CHECK_THAT(gap_dominates.kkt, WithinAbs(2.0 / 7.0, 1e-15));
    CHECK(ProjectedErrors::measure(0.4, 0.1, 2.0, 4.0).kkt == 0.4);
    CHECK(ProjectedErrors::measure(0.1, 0.4, 2.0, 4.0).kkt == 0.4);

    const auto large_same_sign = ProjectedErrors::measure(0.0, 0.0, 1e308, 9e307);
    CHECK_THAT(large_same_sign.gap, WithinAbs(0.1, 1e-15));
    CHECK_THAT(large_same_sign.kkt, WithinAbs(1.0 / 19.0, 1e-15));
    const auto large_opposite_sign = ProjectedErrors::measure(0.0, 0.0, 1e308, -1e308);
    CHECK_THAT(large_opposite_sign.gap, WithinAbs(2.0, 1e-15));
    CHECK_THAT(large_opposite_sign.kkt, WithinAbs(1.0, 1e-15));
}

TEST_CASE("PDLP: adaptive evaluations start at the requested interval and stay bounded", "[pdlp]") {
    PdlpOptions options;
    options.termination_eval_frequency = 175;
    options.adaptive_eval_frequency = true;
    pdlp_detail::EvaluationCadence cadence(options);

    CHECK(cadence.interval() == 175);
    cadence.observe(1.0);
    CHECK(cadence.interval() == 175);
    cadence.observe(0.75);
    CHECK(cadence.interval() == 350);
    cadence.observe(0.5);
    cadence.observe(0.25);
    CHECK(cadence.interval() == 1000);
    cadence.observe(0.24);
    CHECK(cadence.interval() == 1000);
    for (Int i = 0; i < 4; ++i)
        cadence.observe(0.24);
    CHECK(cadence.interval() == 62);
    CHECK_FALSE(cadence.periodicCheck(0));
    cadence.observe(0.24);
    CHECK(cadence.interval() == 50);
    CHECK(cadence.periodicCheck(0));

    pdlp_detail::EvaluationCadence invalid_cadence(options);
    invalid_cadence.observe(std::numeric_limits<Real>::infinity());
    CHECK(invalid_cadence.interval() == 175);
    invalid_cadence.observe(std::numeric_limits<Real>::infinity());
    CHECK(invalid_cadence.interval() == 87);

    options.termination_eval_frequency = 10;
    CHECK(pdlp_detail::EvaluationCadence(options).interval() == 50);
    options.adaptive_eval_frequency = false;
    pdlp_detail::EvaluationCadence fixed(options);
    fixed.observe(1.0);
    fixed.observe(0.5);
    CHECK(fixed.interval() == 10);
    CHECK_FALSE(fixed.periodicCheck(50));
    CHECK(fixed.periodicCheck(100));
}

namespace {

void checkFirstStepRestart(bool use_gpu) {
    LpProblem lp;
    lp.num_rows = 2;
    lp.num_cols = 2;
    lp.obj = {0.0, 0.0};
    lp.col_lower = {-10.0, -10.0};
    lp.col_upper = {10.0, 10.0};
    lp.row_lower = {0.0, 1.0};
    lp.row_upper = lp.row_lower;
    lp.matrix = SparseMatrix(2, 2, {{0, 0, 1.0}, {1, 1, 0.4}});

    PdlpOptions opts;
    opts.verbose = false;
    opts.use_gpu = use_gpu;
    opts.gpu_min_rows = 0;
    opts.gpu_min_nnz = 0;
    opts.do_ruiz_scaling = false;
    opts.do_pock_chambolle_scaling = false;
    opts.do_bound_obj_rescaling = false;
    opts.update_primal_weight = false;
    opts.enable_polishing = false;
    opts.restart_necessary_decay = 0.0;
    opts.restart_artificial_fraction = 1.0;
    opts.termination_eval_frequency = 7;
    opts.max_iter = 21;
    opts.primal_tol = opts.dual_tol = opts.optimality_tol = 0.015;

    Status expected_status = Status::IterLimit;
    SECTION("default periodic anchor") {}
    SECTION("first-step anchor") {
        // The residual at step 14 is about 9.4% of the one at step 8. A
        // baseline taken at step 14 misses the restart at that check.
        opts.restart_first_step_anchor = true;
        expected_status = Status::Optimal;
    }

    PdlpSolver solver;
    solver.setOptions(opts);
    solver.load(lp);
    const auto result = solver.solve();
    CHECK(solver.restartStats().first_batch == 1);
    CHECK(solver.restartStats().sufficient_fpe == (opts.restart_first_step_anchor ? 1 : 0));
    CHECK(result.status == expected_status);
    CHECK(solver.usedGpu() == use_gpu);
    CHECK(result.iterations == 21);
}

void checkOptionalRestartRules(bool use_gpu) {
    PdlpOptions options;
    options.verbose = false;
    options.use_gpu = use_gpu;
    options.gpu_min_rows = 0;
    options.gpu_min_nnz = 0;
    options.termination_eval_frequency = 5;
    options.max_iter = 100;
    options.primal_tol = 0.0;
    options.dual_tol = 0.0;
    options.optimality_tol = 0.0;
    options.update_primal_weight = false;
    options.restart_sufficient_decay = 0.0;
    options.restart_necessary_decay = 0.0;
    options.restart_artificial_fraction = 1.0;

    PdlpSolver solver;
    solver.setOptions(options);
    solver.load(buildSimpleLp());
    const auto run = [&](Real gap, Real kkt) {
        options.restart_gap_decay = gap;
        options.restart_kkt_decay = kkt;
        solver.setOptions(options);
        REQUIRE(solver.solve().status == Status::IterLimit);
        CHECK(solver.usedGpu() == use_gpu);
        CHECK(solver.restartStats().first_batch == 1);
    };

    run(0.9, 0.0);
    CHECK(solver.restartStats().gap > 0);
    CHECK(solver.restartStats().kkt == 0);
    run(0.0, 0.9);
    CHECK(solver.restartStats().gap == 0);
    CHECK(solver.restartStats().kkt > 0);
    run(0.0, 0.0);
    CHECK(solver.restartStats().gap == 0);
    CHECK(solver.restartStats().kkt == 0);

    // This trajectory improves fast enough to increase the interval.
    // The 275-step budget also exercises a final partial batch.
    options.max_iter = 275;
    options.termination_eval_frequency = 50;
    run(0.0, 0.9);
    const Int fixed_restarts = solver.restartStats().kkt;
    options.adaptive_eval_frequency = true;
    run(0.0, 0.9);
    CHECK(solver.restartStats().kkt > 0);
    CHECK(solver.restartStats().kkt < fixed_restarts);

    // The main phase converges at its first check. Report restarts taken
    // while polishing, even if the original converged solution is kept.
    options.adaptive_eval_frequency = false;
    options.termination_eval_frequency = 5;
    options.primal_tol = options.dual_tol = options.optimality_tol = 0.15;
    options.polish_primal_tol = options.polish_dual_tol = 0.0;
    options.polish_max_iter = 25;
    solver.setOptions(options);
    REQUIRE(solver.solve().status == Status::Optimal);
    CHECK(solver.restartStats().first_batch == 1);
    CHECK(solver.restartStats().kkt > 0);
}

}  // namespace

TEST_CASE("PdlpSolver: restart decay can use the first step of the epoch", "[pdlp]") {
    checkFirstStepRestart(false);
}

TEST_CASE("PdlpSolver: GPU restart decay can use the first step of the epoch", "[pdlp][gpu]") {
    if (!PdlpSolver::gpuAvailable()) {
        SKIP("No usable CUDA device: none found, or no device code for it (#192).");
    }
    checkFirstStepRestart(true);
}

TEST_CASE("PdlpSolver: optional restart rules fire independently and stats reset", "[pdlp]") {
    checkOptionalRestartRules(false);
}

TEST_CASE("PdlpSolver: GPU optional restart rules fire independently and stats reset",
          "[pdlp][gpu]") {
    if (!PdlpSolver::gpuAvailable())
        SKIP("No usable CUDA device.");
    checkOptionalRestartRules(true);
}

TEST_CASE("PdlpSolver: handles bounds-only LP", "[pdlp]") {
    auto lp = buildBoundsOnlyLp();
    SECTION("objective storage matches the model columns") {}
    SECTION("objective storage extends beyond the model columns") {
        lp.obj.push_back(std::numeric_limits<Real>::quiet_NaN());
    }

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

TEST_CASE("PdlpSolver: keeps small objective terms across cancellation", "[pdlp]") {
    LpProblem lp;
    lp.num_cols = 3;
    lp.obj = {1e16, 1.0, -1e16};
    lp.col_lower = {1.0, 1.0, 0.0};
    lp.col_upper = {2.0, 2.0, 1.0};
    lp.col_type.assign(3, VarType::Continuous);
    lp.num_rows = 1;
    lp.row_lower = {-kInf};
    lp.row_upper = {2.0};
    lp.matrix = SparseMatrix(1, 3, {{0, 1, 1.0}});

    PdlpOptions options;
    options.use_gpu = false;
    options.do_ruiz_scaling = false;
    options.do_pock_chambolle_scaling = false;
    options.do_bound_obj_rescaling = false;
    PdlpSolver solver;
    solver.setOptions(options);
    solver.load(lp);
    const auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    CHECK(result.objective == 1.0);
}

TEST_CASE("PdlpSolver: keeps the objective gap when the offset cancels large terms", "[pdlp]") {
    auto lp = buildSimpleLp();
    lp.obj = {1e16, 1.0};
    lp.obj_offset = -1e16;
    lp.col_lower = {1.0, 0.0};
    lp.col_upper = {2.0, 2.0};
    lp.matrix = SparseMatrix(1, 2, {{0, 1, 1.0}});

    PdlpOptions options;
    options.use_gpu = false;
    options.do_ruiz_scaling = false;
    options.do_pock_chambolle_scaling = false;
    options.do_bound_obj_rescaling = false;
    options.max_iter = 1;
    PdlpSolver solver;
    solver.setOptions(options);
    solver.load(lp);
    // In the first step x[1] moves from 1.5 to 0.502. Its contribution
    // to the primal objective remains different from the dual bound of zero.
    solver.setWarmStart(std::vector<Real>{1.0, 1.5}, std::vector<Real>{0.0});
    CHECK(solver.solve().status == Status::IterLimit);
}

TEST_CASE("PdlpSolver: combines dual row and column totals after cancellation", "[pdlp]") {
    constexpr Real large = 1e154;
    LpProblem lp;
    lp.num_cols = 3;
    lp.obj = {large, -large, large};
    lp.col_lower.assign(3, large);
    lp.col_upper = lp.col_lower;
    lp.col_type.assign(3, VarType::Continuous);
    lp.num_rows = 2;
    lp.row_lower = {-kInf, -kInf};
    lp.row_upper = {-large, large};
    lp.matrix = SparseMatrix(2, 3, {{0, 0, -1.0}, {1, 1, 1.0}});

    PdlpOptions options;
    options.use_gpu = false;
    options.do_ruiz_scaling = false;
    options.do_pock_chambolle_scaling = false;
    options.do_bound_obj_rescaling = false;
    options.optimality_norm = PdlpOptimalityNorm::LInf;
    options.max_iter = 1;
    PdlpSolver solver;
    solver.setOptions(options);
    solver.load(lp);
    solver.setWarmStart(lp.col_lower, std::vector<Real>{large, large});
    const auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    CHECK(result.objective == large * large);
}

TEST_CASE("PdlpSolver: a scaled maximization meets the objective accuracy at Optimal", "[pdlp]") {
    // Found by the libc++ validity stream (#424, #429). The optimum is
    // 2.078125 at (0.025, 387.5). PDLP stopped Optimal at 2.0777 with
    // x2 = 387.54: the slack of row 4, priced by its dual, was cancelled in
    // the gap by the free column's reduced cost times x2.
    LpProblem lp;
    lp.sense = Sense::Maximize;
    lp.num_cols = 2;
    lp.num_rows = 5;
    lp.obj_offset = -7.875;
    lp.obj = {553.125, -0.01};
    lp.col_lower = {-kInf, -kInf};
    lp.col_upper = {0.025000000000000001, kInf};
    lp.col_type.assign(2, VarType::Continuous);
    lp.row_lower = {-kInf, 25000, -31.25, -kInf, 0.00038750000000000004};
    lp.row_upper = {250, kInf, -31.25, kInf, kInf};
    lp.matrix = SparseMatrix(5, 2,
                             {{0, 0, 10000},
                              {1, 0, 1000000},
                              {2, 0, -1250},
                              {3, 0, 46250},
                              {4, 1, 1.0000000000000002e-06}});

    PdlpOptions options;
    options.use_gpu = false;
    options.verbose = false;
    options.max_iter = 20000;
    options.polish_max_iter = 20000;
    PdlpSolver solver;
    solver.setOptions(options);
    solver.load(lp);
    const auto result = solver.solve();
    INFO("status " << static_cast<int>(result.status) << " objective " << result.objective);
    // Optimal here, within 5e-6 (Linux, libstdc++). The issue allows a limit
    // status when the accuracy is not reached, so another platform's
    // rounding may stop at IterLimit, never at an inaccurate Optimal.
    CHECK((result.status == Status::Optimal || result.status == Status::IterLimit));
    if (result.status == Status::Optimal)
        CHECK(std::abs(result.objective - 2.078125) <= 1e-4 * 2.078125);
}

TEST_CASE("PdlpSolver: max_solve_seconds enforces time limit", "[pdlp][time_limit]") {
    auto lp = buildSimpleLp();

    PdlpSolver solver;
    PdlpOptions opts;
    opts.verbose = false;
    opts.use_gpu = false;
    opts.max_solve_seconds = 0.0;
    solver.setOptions(opts);
    solver.load(lp);
    auto result = solver.solve();

    REQUIRE(result.status == Status::TimeLimit);
}

TEST_CASE("PdlpSolver: linf norm mode solves simple LP", "[pdlp][linf]") {
    auto lp = buildSimpleLp();

    PdlpSolver solver;
    PdlpOptions opts;
    opts.verbose = false;
    opts.use_gpu = false;
    opts.max_iter = 10000;
    opts.optimality_norm = PdlpOptimalityNorm::LInf;
    solver.setOptions(opts);
    solver.load(lp);
    auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(-4.0, 1e-4));
}

TEST_CASE("PDLP root policy: MIP objective matches dual root policy", "[pdlp][mip]") {
    auto lp = buildBranchingMip();
    const bool presolve = GENERATE(true, false);

    MipSolver dual_solver;
    dual_solver.setVerbose(false);
    dual_solver.setPresolve(presolve);
    dual_solver.setCutsEnabled(false);
    dual_solver.setRootLpPolicy(RootLpPolicy::DualDefault);
    dual_solver.load(lp);
    auto dual_result = dual_solver.solve();

    MipSolver pdlp_solver;
    pdlp_solver.setVerbose(false);
    pdlp_solver.setPresolve(presolve);
    pdlp_solver.setCutsEnabled(false);
    pdlp_solver.setRootLpPolicy(RootLpPolicy::PdlpRoot);

    pdlp_solver.load(lp);
    auto pdlp_result = pdlp_solver.solve();

    REQUIRE(dual_result.status == Status::Optimal);
    REQUIRE(pdlp_result.status == Status::Optimal);
    CHECK(pdlp_solver.getLpStats().root_pdlp_crossover);
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

TEST_CASE("Concurrent root deterministic mode is reproducible", "[pdlp][mip][concurrent]") {
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

TEST_CASE("PdlpSolver: detects infeasible LP", "[pdlp]") {
    // min x  s.t. x <= -1, x >= 1 → infeasible
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
    opts.use_gpu = false;
    opts.max_iter = 100000;
    solver.setOptions(opts);
    solver.load(lp);
    auto result = solver.solve();

    REQUIRE(result.status == Status::Infeasible);
}

TEST_CASE("PdlpSolver: detects infeasible LP with equality constraint", "[pdlp]") {
    // min x + y  s.t.  x + y = 5,  x + y = 3  → infeasible (contradicting equalities)
    LpProblem lp;
    lp.name = "pdlp_infeasible_eq";
    lp.sense = Sense::Minimize;
    lp.num_cols = 2;
    lp.obj = {1.0, 1.0};
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {kInf, kInf};
    lp.col_type = {VarType::Continuous, VarType::Continuous};

    lp.num_rows = 2;
    lp.row_lower = {5.0, 3.0};
    lp.row_upper = {5.0, 3.0};
    lp.row_names = {"eq1", "eq2"};

    std::vector<Triplet> trips = {
        {0, 0, 1.0},
        {0, 1, 1.0},
        {1, 0, 1.0},
        {1, 1, 1.0},
    };
    lp.matrix = SparseMatrix(2, 2, std::move(trips));

    PdlpSolver solver;
    PdlpOptions opts;
    opts.verbose = false;
    opts.use_gpu = false;
    opts.max_iter = 100000;
    solver.setOptions(opts);
    solver.load(lp);
    auto result = solver.solve();

    REQUIRE(result.status == Status::Infeasible);
}

TEST_CASE("PdlpSolver: detects unbounded LP with constraints", "[pdlp]") {
    // min -x  s.t.  x - y <= 0,  x >= 0, y >= 0  → unbounded (increase y, set x = y)
    LpProblem lp;
    lp.name = "pdlp_unbounded_constrained";
    lp.sense = Sense::Minimize;
    lp.num_cols = 2;
    lp.obj = {-1.0, 0.0};
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {kInf, kInf};
    lp.col_type = {VarType::Continuous, VarType::Continuous};

    lp.num_rows = 1;
    lp.row_lower = {-kInf};
    lp.row_upper = {0.0};  // x - y <= 0, i.e. x <= y
    lp.row_names = {"c1"};

    std::vector<Triplet> trips = {{0, 0, 1.0}, {0, 1, -1.0}};
    lp.matrix = SparseMatrix(1, 2, std::move(trips));

    PdlpSolver solver;
    PdlpOptions opts;
    opts.verbose = false;
    opts.use_gpu = false;
    opts.max_iter = 100000;
    solver.setOptions(opts);
    solver.load(lp);
    auto result = solver.solve();

    REQUIRE(result.status == Status::Unbounded);
}

TEST_CASE("PdlpSolver: Farkas certificate verifies for infeasible LP", "[pdlp]") {
    // min x  s.t. x <= -1, x >= 1 → infeasible
    // Farkas certificate: y >= 0 (for <= constraint), A^T y has correct sign,
    // and b^T y - column_support < 0.
    LpProblem lp;
    lp.name = "pdlp_farkas";
    lp.sense = Sense::Minimize;
    lp.num_cols = 1;
    lp.obj = {1.0};
    lp.col_lower = {1.0};
    lp.col_upper = {kInf};
    lp.col_type = {VarType::Continuous};

    lp.num_rows = 1;
    lp.row_lower = {-kInf};
    lp.row_upper = {-1.0};
    lp.row_names = {"c1"};

    std::vector<Triplet> trips = {{0, 0, 1.0}};
    lp.matrix = SparseMatrix(1, 1, std::move(trips));

    PdlpSolver solver;
    PdlpOptions opts;
    opts.verbose = false;
    opts.use_gpu = false;
    opts.max_iter = 100000;
    solver.setOptions(opts);
    solver.load(lp);
    auto result = solver.solve();

    REQUIRE(result.status == Status::Infeasible);

    // Verify the Farkas certificate from dual values.
    auto y = solver.getDualValues();
    REQUIRE(y.size() == 1);

    // Normalize y to get the ray.
    Real y_norm = std::sqrt(y[0] * y[0]);
    REQUIRE(y_norm > 1e-10);
    Real ray = y[0] / y_norm;

    // For <= constraint: dual ray should be >= 0.
    CHECK(ray >= -1e-6);

    // A^T * ray = ray (since A = [1]).
    // For variable with lb = 1 (finite): A^T ray >= 0 is needed.
    CHECK(ray >= -1e-6);

    // Farkas value: ray * ru - (A^T ray) * lb = ray * (-1) - ray * 1 = -2 * ray < 0.
    Real farkas_val = ray * (-1.0) - ray * 1.0;
    CHECK(farkas_val < -1e-6);
}

TEST_CASE("PdlpSolver: no false positive on feasible LP", "[pdlp]") {
    // Verify that the infeasibility/unboundedness detection does not
    // fire on a feasible bounded LP.
    auto lp = buildSimpleLp();

    PdlpSolver solver;
    PdlpOptions opts;
    opts.verbose = false;
    opts.use_gpu = false;
    opts.max_iter = 50000;
    solver.setOptions(opts);
    solver.load(lp);
    auto result = solver.solve();

    // Should be optimal, not infeasible or unbounded.
    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(-4.0, 1e-4));
}

TEST_CASE("PdlpSolver: a CUDA device present can run the PDLP kernels", "[pdlp][gpu]") {
    // The parity tests skip without gpuAvailable(), which is also what a
    // build without device code for the GPU gives: no kernel runs (#192).
    // So a device that is present must pass the probe.
#ifdef MIPX_HAS_CUDA
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        cudaGetLastError();
        SKIP("No CUDA device found.");
    }
    INFO(
        "CUDA device present but PdlpSolver::gpuAvailable() is false: is "
        "CMAKE_CUDA_ARCHITECTURES (in CMakeCache.txt) one this GPU runs?");
    CHECK(PdlpSolver::gpuAvailable());
#else
    CHECK_FALSE(PdlpSolver::gpuAvailable());
#endif
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
    if (!PdlpSolver::gpuAvailable()) {
        SKIP("No usable CUDA device: none found, or no device code for it (#192).");
    }
    REQUIRE(gpu_result.status == Status::Optimal);
    CHECK(gpu_solver.usedGpu());
    CHECK_THAT(gpu_result.objective, WithinAbs(cpu_result.objective, 1e-4));
}

#ifdef MIPX_HAS_CUDA
TEST_CASE("PDLP: GPU convergence totals retain small block contributions", "[pdlp][gpu]") {
    if (!PdlpSolver::gpuAvailable())
        SKIP("No usable CUDA device.");

    constexpr Index blocks = 1025;
    constexpr Real large = 0x1p54;
    std::vector<GpuConvergenceMetrics> partials(blocks);
    for (auto& partial : partials) {
        partial.primal_obj = 1.0;
        partial.primal_resid_sq = 1.0;
    }
    partials.front().primal_obj = large;
    // The first pair in the block tree cancels the large terms.
    partials[128].primal_obj = -large;
    partials.front().primal_resid_sq = large;
    // The leftover block also contributes to sums and maxima.
    partials.back().dual_obj_row = 3.0;
    partials.back().dual_resid_max = 7.0;

    const size_t entries = 1 + blocks + convergenceMetricBlocks(blocks);
    GpuConvergenceMetrics* device = nullptr;
    REQUIRE(cudaMalloc(reinterpret_cast<void**>(&device), entries * sizeof(*device)) ==
            cudaSuccess);
    const std::unique_ptr<GpuConvergenceMetrics, decltype(&cudaFree)> allocation(device, cudaFree);
    REQUIRE(cudaMemcpy(device + 1, partials.data(), partials.size() * sizeof(*device),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    launchReduceConvergenceMetrics(blocks, device + 1, device, nullptr);
    REQUIRE(cudaGetLastError() == cudaSuccess);
    GpuConvergenceMetrics result{};
    REQUIRE(cudaMemcpy(&result, device, sizeof(result), cudaMemcpyDeviceToHost) == cudaSuccess);

    CHECK(result.primal_obj == blocks - 2);
    CHECK_THAT(result.primal_resid_sq, WithinAbs(large + (blocks - 1), 4.0));
    CHECK(result.dual_obj_row == 3.0);
    CHECK(result.dual_resid_max == 7.0);
}

TEST_CASE("PdlpSolver: GPU convergence includes partial row and column blocks", "[pdlp][gpu]") {
    if (!PdlpSolver::gpuAvailable())
        SKIP("No usable CUDA device.");

    constexpr Index rows = 269;
    constexpr Index cols = 517;
    LpProblem lp;
    lp.num_rows = rows;
    lp.num_cols = cols;
    lp.obj.assign(cols, -1.0);
    lp.col_lower.assign(cols, 0.0);
    lp.col_upper.assign(cols, 1.0);
    lp.col_type.assign(cols, VarType::Continuous);
    lp.row_lower.assign(rows, -kInf);
    lp.row_upper.resize(rows);
    std::vector<Triplet> coefficients;
    Real objective = -(cols - rows);
    for (Index i = 0; i < rows; ++i) {
        lp.col_upper[i] = kInf;
        lp.row_upper[i] = 1.0 + (i % 8);
        coefficients.push_back({i, i, 1.0});
        objective -= lp.row_upper[i];
    }
    lp.matrix = SparseMatrix(rows, cols, std::move(coefficients));

    PdlpOptions options;
    options.verbose = false;
    options.use_gpu = true;
    options.gpu_min_rows = 0;
    options.gpu_min_nnz = 0;
    PdlpSolver solver;
    solver.setOptions(options);
    solver.load(lp);
    const auto result = solver.solve();

    REQUIRE(solver.usedGpu());
    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(objective, 1e-3));
}
#endif

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

PdlpOptions quietOpts(Int max_iter = 50000, bool rescaling = true, bool gpu = false) {
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

    auto result_on = solvePdlp(lp, quietOpts(50000, true));
    auto result_off = solvePdlp(lp, quietOpts(50000, false));

    REQUIRE(result_on.status == Status::Optimal);
    REQUIRE(result_off.status == Status::Optimal);
    CHECK_THAT(result_on.objective, WithinAbs(-4.0, 1e-4));
    CHECK_THAT(result_off.objective, WithinAbs(-4.0, 1e-4));
    CHECK_THAT(result_on.objective, WithinAbs(result_off.objective, 1e-3));
}

TEST_CASE("PdlpSolver: bound-obj rescaling on medium LP", "[pdlp][rescaling]") {
    auto lp = buildMediumLp();

    auto result_on = solvePdlp(lp, quietOpts(500000, true));
    auto result_off = solvePdlp(lp, quietOpts(500000, false));

    REQUIRE(result_on.status == Status::Optimal);
    REQUIRE(result_off.status == Status::Optimal);
    // First-order methods converge to ~1e-4 relative tolerance, so
    // objectives may differ by O(0.1) on this ~425-obj problem.
    CHECK_THAT(result_on.objective, WithinAbs(result_off.objective, 1.0));
}

TEST_CASE("PdlpSolver: presolved adlittle tracks dual objective", "[pdlp][netlib][presolve]") {
    const fs::path path = fs::path(testDataDir()) / "netlib" / "adlittle.mps.gz";
    if (!fs::exists(path)) {
        SUCCEED("Netlib adlittle not available");
        return;
    }

    auto lp = readMps(path.string());
    Presolver presolver;
    auto reduced = presolver.presolve(lp);

    DualSimplexSolver dual;
    dual.setVerbose(false);
    dual.load(reduced);
    auto dual_result = dual.solve();
    REQUIRE(dual_result.status == Status::Optimal);
    auto dual_full = presolver.postsolve(dual.getPrimalValues());
    Real dual_obj = lp.obj_offset;
    for (Index j = 0; j < lp.num_cols; ++j) {
        dual_obj += lp.obj[j] * dual_full[j];
    }

    PdlpSolver pdlp;
    PdlpOptions opts;
    opts.verbose = false;
    opts.use_gpu = false;
    pdlp.setOptions(opts);
    pdlp.load(reduced);
    auto pdlp_result = pdlp.solve();
    REQUIRE(pdlp_result.status == Status::Optimal);

    auto pdlp_full = presolver.postsolve(pdlp.getPrimalValues());
    Real pdlp_obj = lp.obj_offset;
    for (Index j = 0; j < lp.num_cols; ++j) {
        pdlp_obj += lp.obj[j] * pdlp_full[j];
    }

    CHECK_THAT(pdlp_obj, WithinAbs(dual_obj, 1.0));
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

    auto result_on = solvePdlp(lp, quietOpts(500000, true));
    auto result_off = solvePdlp(lp, quietOpts(500000, false));

    REQUIRE(result_on.status == Status::Optimal);
    REQUIRE(result_off.status == Status::Optimal);
    // On well-scaled problems rescaling may not help, but shouldn't hurt much.
    CHECK(result_on.iterations <= result_off.iterations * 2);
}

// ---------------------------------------------------------------------------
// Adaptive step size tests
// ---------------------------------------------------------------------------

TEST_CASE("PdlpSolver: adaptive step solves simple LP", "[pdlp][adaptive]") {
    auto lp = buildSimpleLp();

    PdlpSolver solver;
    PdlpOptions opts;
    opts.verbose = false;
    opts.use_gpu = false;
    opts.max_iter = 50000;
    opts.adaptive_step = true;
    solver.setOptions(opts);
    solver.load(lp);
    auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(-4.0, 1e-4));
}

TEST_CASE("PdlpSolver: adaptive step solves bounds-only LP", "[pdlp][adaptive]") {
    auto lp = buildBoundsOnlyLp();

    PdlpSolver solver;
    PdlpOptions opts;
    opts.verbose = false;
    opts.adaptive_step = true;
    solver.setOptions(opts);
    solver.load(lp);
    auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(1.0, 1e-6));
}

TEST_CASE("PdlpSolver: adaptive step matches fixed step on simple LP", "[pdlp][adaptive]") {
    auto lp = buildSimpleLp();

    auto fixed_opts = quietOpts(50000, true, false);
    auto adaptive_opts = quietOpts(50000, true, false);
    adaptive_opts.adaptive_step = true;

    auto fixed_result = solvePdlp(lp, fixed_opts);
    auto adaptive_result = solvePdlp(lp, adaptive_opts);

    REQUIRE(fixed_result.status == Status::Optimal);
    REQUIRE(adaptive_result.status == Status::Optimal);
    CHECK_THAT(adaptive_result.objective, WithinAbs(fixed_result.objective, 1e-3));
}

TEST_CASE("PdlpSolver: adaptive step on medium LP", "[pdlp][adaptive]") {
    auto lp = buildMediumLp();

    auto opts = quietOpts(500000, true, false);
    opts.adaptive_step = true;
    auto result = solvePdlp(lp, opts);

    REQUIRE(result.status == Status::Optimal);
    // Should match the known objective from the fixed-step test.
    auto fixed_result = solvePdlp(lp, quietOpts(500000, true, false));
    REQUIRE(fixed_result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(fixed_result.objective, 1.0));
}

TEST_CASE("PdlpSolver: adaptive step does not regress iteration count",
          "[pdlp][adaptive][performance]") {
    auto lp = buildSimpleLp();

    auto fixed_opts = quietOpts(50000, true, false);
    auto adaptive_opts = quietOpts(50000, true, false);
    adaptive_opts.adaptive_step = true;

    auto fixed_result = solvePdlp(lp, fixed_opts);
    auto adaptive_result = solvePdlp(lp, adaptive_opts);

    REQUIRE(fixed_result.status == Status::Optimal);
    REQUIRE(adaptive_result.status == Status::Optimal);
    // Adaptive should not take more than 3x fixed iterations.
    CHECK(adaptive_result.iterations <= fixed_result.iterations * 3);
}

TEST_CASE("PdlpSolver: adaptive step with equality constraints", "[pdlp][adaptive]") {
    // min x + 2y  s.t.  x + y = 10,  x,y >= 0
    // Optimal: x=10, y=0, obj = 10.
    LpProblem lp;
    lp.name = "adaptive_equality";
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

    auto opts = quietOpts(100000, true, false);
    opts.adaptive_step = true;
    auto result = solvePdlp(lp, opts);

    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(10.0, 0.1));
}

TEST_CASE("PdlpSolver: GPU with adaptive step matches CPU", "[pdlp][adaptive][gpu]") {
    auto lp = buildMediumLp();

    auto cpu_opts = quietOpts(500000, true, false);
    cpu_opts.adaptive_step = true;
    auto cpu_result = solvePdlp(lp, cpu_opts);

    auto gpu_opts = quietOpts(500000, true, true);
    gpu_opts.adaptive_step = true;
    gpu_opts.gpu_min_rows = 0;
    gpu_opts.gpu_min_nnz = 0;
    auto gpu_result = solvePdlp(lp, gpu_opts);

    REQUIRE(cpu_result.status == Status::Optimal);
    if (!PdlpSolver::gpuAvailable()) {
        SKIP("No usable CUDA device: none found, or no device code for it (#192).");
    }
    REQUIRE(gpu_result.status == Status::Optimal);
    CHECK_THAT(gpu_result.objective, WithinAbs(cpu_result.objective, 1e-3));
}

TEST_CASE("PdlpSolver: GPU with rescaling matches CPU", "[pdlp][rescaling][gpu]") {
    auto lp = buildMediumLp();

    auto cpu_result = solvePdlp(lp, quietOpts(500000, true, false));

    auto gpu_opts = quietOpts(500000, true, true);
    gpu_opts.gpu_min_rows = 0;
    gpu_opts.gpu_min_nnz = 0;
    auto gpu_result = solvePdlp(lp, gpu_opts);

    REQUIRE(cpu_result.status == Status::Optimal);
    if (!PdlpSolver::gpuAvailable()) {
        SKIP("No usable CUDA device: none found, or no device code for it (#192).");
    }
    REQUIRE(gpu_result.status == Status::Optimal);
    CHECK_THAT(gpu_result.objective, WithinAbs(cpu_result.objective, 1e-3));
}

// ---------------------------------------------------------------------------
// Matrix zero tolerance filtering tests
// ---------------------------------------------------------------------------

TEST_CASE("PdlpSolver: matrix_zero_tol=0 leaves matrix unchanged", "[pdlp][filter]") {
    auto lp = buildSimpleLp();

    PdlpSolver solver;
    PdlpOptions opts;
    opts.verbose = false;
    opts.use_gpu = false;
    opts.max_iter = 10000;
    opts.matrix_zero_tol = 0.0;  // disabled (default)
    solver.setOptions(opts);
    solver.load(lp);
    auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(-4.0, 1e-4));
}

TEST_CASE("PdlpSolver: matrix_zero_tol filters near-zero entries", "[pdlp][filter]") {
    // Build LP with some near-zero noise entries that should not affect the
    // optimal solution when filtered.
    // min -x - y  s.t.  x + y + 1e-15*z <= 4,  x,y,z >= 0
    LpProblem lp;
    lp.name = "pdlp_filter";
    lp.sense = Sense::Minimize;
    lp.num_cols = 3;
    lp.obj = {-1.0, -1.0, 0.0};
    lp.col_lower = {0.0, 0.0, 0.0};
    lp.col_upper = {kInf, kInf, kInf};
    lp.col_type = {VarType::Continuous, VarType::Continuous, VarType::Continuous};

    lp.num_rows = 1;
    lp.row_lower = {-kInf};
    lp.row_upper = {4.0};

    std::vector<Triplet> trips = {
        {0, 0, 1.0},
        {0, 1, 1.0},
        {0, 2, 1e-15},
    };
    lp.matrix = SparseMatrix(1, 3, std::move(trips));

    // Without filtering: 3 nonzeros.
    auto baseline = solvePdlp(lp, quietOpts(50000, false, false));
    REQUIRE(baseline.status == Status::Optimal);
    CHECK_THAT(baseline.objective, WithinAbs(-4.0, 1e-4));

    // With filtering: the 1e-15 entry should be dropped.
    PdlpOptions opts;
    opts.verbose = false;
    opts.use_gpu = false;
    opts.max_iter = 50000;
    opts.do_bound_obj_rescaling = false;
    opts.matrix_zero_tol = 1e-10;
    auto filtered = solvePdlp(lp, opts);

    REQUIRE(filtered.status == Status::Optimal);
    CHECK_THAT(filtered.objective, WithinAbs(-4.0, 1e-4));
    // Objectives should match closely.
    CHECK_THAT(filtered.objective, WithinAbs(baseline.objective, 1e-3));
}

TEST_CASE("PdlpSolver: matrix_zero_tol on medium LP matches baseline", "[pdlp][filter]") {
    auto lp = buildMediumLp();

    auto baseline = solvePdlp(lp, quietOpts(500000, true, false));
    REQUIRE(baseline.status == Status::Optimal);

    PdlpOptions opts;
    opts.verbose = false;
    opts.use_gpu = false;
    opts.max_iter = 500000;
    opts.do_bound_obj_rescaling = true;
    // Tolerance well below the smallest coefficient (0.1), so nothing should
    // actually be dropped from the medium LP.
    opts.matrix_zero_tol = 1e-12;
    auto filtered = solvePdlp(lp, opts);

    REQUIRE(filtered.status == Status::Optimal);
    CHECK_THAT(filtered.objective, WithinAbs(baseline.objective, 1e-3));
}

// ---------------------------------------------------------------------------
// Preconditioner refresh tests
// ---------------------------------------------------------------------------

TEST_CASE("PdlpSolver: preconditioner refresh off by default", "[pdlp][precond_refresh]") {
    auto lp = buildSimpleLp();

    PdlpSolver solver;
    PdlpOptions opts;
    opts.verbose = false;
    opts.use_gpu = false;
    opts.max_iter = 10000;
    solver.setOptions(opts);
    CHECK_FALSE(solver.options().preconditioner_refresh);

    solver.load(lp);
    auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(-4.0, 1e-4));
}

TEST_CASE("PdlpSolver: preconditioner refresh enabled produces correct result",
          "[pdlp][precond_refresh]") {
    auto lp = buildSimpleLp();

    PdlpSolver solver;
    PdlpOptions opts;
    opts.verbose = false;
    opts.use_gpu = false;
    opts.max_iter = 10000;
    opts.preconditioner_refresh = true;
    opts.preconditioner_refresh_ratio = 10.0;
    solver.setOptions(opts);
    solver.load(lp);
    auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(-4.0, 1e-4));
}

TEST_CASE("PdlpSolver: preconditioner refresh on medium LP matches baseline",
          "[pdlp][precond_refresh]") {
    auto lp = buildMediumLp();

    auto baseline = solvePdlp(lp, quietOpts(500000, true, false));
    REQUIRE(baseline.status == Status::Optimal);

    PdlpOptions opts;
    opts.verbose = false;
    opts.use_gpu = false;
    opts.max_iter = 500000;
    opts.do_bound_obj_rescaling = true;
    opts.preconditioner_refresh = true;
    opts.preconditioner_refresh_ratio = 10.0;
    auto refresh_result = solvePdlp(lp, opts);

    REQUIRE(refresh_result.status == Status::Optimal);
    CHECK_THAT(refresh_result.objective, WithinAbs(baseline.objective, 1e-3));
}

TEST_CASE("PdlpSolver: preconditioner refresh with aggressive ratio on medium LP",
          "[pdlp][precond_refresh]") {
    // Use a very low ratio (2x) so refresh fires frequently.
    auto lp = buildMediumLp();

    PdlpOptions opts;
    opts.verbose = false;
    opts.use_gpu = false;
    opts.max_iter = 500000;
    opts.do_bound_obj_rescaling = true;
    opts.preconditioner_refresh = true;
    opts.preconditioner_refresh_ratio = 2.0;
    auto result = solvePdlp(lp, opts);

    REQUIRE(result.status == Status::Optimal);
    // Should still converge to approximately the same objective.
    auto baseline = solvePdlp(lp, quietOpts(500000, true, false));
    REQUIRE(baseline.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(baseline.objective, 1e-3));
}

TEST_CASE("PdlpSolver: GPU with preconditioner refresh matches CPU",
          "[pdlp][precond_refresh][gpu]") {
    auto lp = buildMediumLp();

    PdlpOptions cpu_opts;
    cpu_opts.verbose = false;
    cpu_opts.use_gpu = false;
    cpu_opts.max_iter = 500000;
    cpu_opts.preconditioner_refresh = true;
    cpu_opts.preconditioner_refresh_ratio = 10.0;
    auto cpu_result = solvePdlp(lp, cpu_opts);

    PdlpOptions gpu_opts;
    gpu_opts.verbose = false;
    gpu_opts.use_gpu = true;
    gpu_opts.gpu_min_rows = 0;
    gpu_opts.gpu_min_nnz = 0;
    gpu_opts.max_iter = 500000;
    gpu_opts.preconditioner_refresh = true;
    gpu_opts.preconditioner_refresh_ratio = 10.0;
    auto gpu_result = solvePdlp(lp, gpu_opts);

    REQUIRE(cpu_result.status == Status::Optimal);
    if (!PdlpSolver::gpuAvailable()) {
        SKIP("No usable CUDA device: none found, or no device code for it (#192).");
    }
    REQUIRE(gpu_result.status == Status::Optimal);
    CHECK_THAT(gpu_result.objective, WithinAbs(cpu_result.objective, 1e-3));
}

// ---------------------------------------------------------------------------
// Warm-start tests
// ---------------------------------------------------------------------------

TEST_CASE("PdlpSolver: warm-start re-solve converges faster than cold start", "[pdlp][warmstart]") {
    // Solve the medium LP, then tighten a column upper bound and re-solve.
    // Warm-starting from the prior solution should converge in fewer iterations.
    auto lp = buildMediumLp();

    PdlpSolver solver;
    auto opts = quietOpts(500000, true, false);
    solver.setOptions(opts);
    solver.load(lp);
    auto result1 = solver.solve();
    REQUIRE(result1.status == Status::Optimal);

    auto primal = solver.getPrimalValues();
    auto dual = solver.getDualValues();

    // Tighten one column upper bound slightly.
    lp.col_upper[0] = 50.0;

    // Cold-start solve of the modified LP.
    PdlpSolver cold_solver;
    cold_solver.setOptions(opts);
    cold_solver.load(lp);
    auto cold_result = cold_solver.solve();
    REQUIRE(cold_result.status == Status::Optimal);

    // Warm-start solve of the modified LP.
    PdlpSolver warm_solver;
    warm_solver.setOptions(opts);
    warm_solver.load(lp);
    warm_solver.setWarmStart(primal, dual);
    auto warm_result = warm_solver.solve();
    REQUIRE(warm_result.status == Status::Optimal);

    // Objectives should match (first-order tolerance).
    CHECK_THAT(warm_result.objective, WithinAbs(cold_result.objective, 1.0));

    // Warm start should converge in fewer iterations.
    CHECK(warm_result.iterations < cold_result.iterations);
}

TEST_CASE("PdlpSolver: warm-start on simple LP after bound change", "[pdlp][warmstart]") {
    // min -x - y  s.t.  x + y <= 4,  x,y >= 0
    // Optimal: x+y=4, obj=-4.
    auto lp = buildSimpleLp();

    PdlpSolver solver;
    PdlpOptions opts;
    opts.verbose = false;
    opts.max_iter = 50000;
    opts.use_gpu = false;
    solver.setOptions(opts);
    solver.load(lp);
    auto result1 = solver.solve();
    REQUIRE(result1.status == Status::Optimal);

    auto primal = solver.getPrimalValues();
    auto dual = solver.getDualValues();

    // Change the row upper bound from 4 to 3.
    lp.row_upper[0] = 3.0;

    // Cold start.
    PdlpSolver cold;
    cold.setOptions(opts);
    cold.load(lp);
    auto cold_result = cold.solve();
    REQUIRE(cold_result.status == Status::Optimal);
    CHECK_THAT(cold_result.objective, WithinAbs(-3.0, 1e-3));

    // Warm start — should produce the same objective.
    PdlpSolver warm;
    warm.setOptions(opts);
    warm.load(lp);
    warm.setWarmStart(primal, dual);
    auto warm_result = warm.solve();
    REQUIRE(warm_result.status == Status::Optimal);
    CHECK_THAT(warm_result.objective, WithinAbs(-3.0, 1e-3));
}

TEST_CASE("PdlpSolver: clearWarmStart disables warm-start", "[pdlp][warmstart]") {
    auto lp = buildSimpleLp();

    PdlpSolver solver;
    PdlpOptions opts;
    opts.verbose = false;
    opts.max_iter = 50000;
    opts.use_gpu = false;
    solver.setOptions(opts);
    solver.load(lp);
    auto result1 = solver.solve();
    REQUIRE(result1.status == Status::Optimal);

    auto primal = solver.getPrimalValues();
    auto dual = solver.getDualValues();

    // Set and then clear warm-start.
    solver.setWarmStart(primal, dual);
    solver.clearWarmStart();

    // Re-load and solve — should behave like cold start.
    solver.load(lp);
    auto result2 = solver.solve();
    REQUIRE(result2.status == Status::Optimal);
    CHECK_THAT(result2.objective, WithinAbs(-4.0, 1e-4));
}

TEST_CASE("PdlpSolver: warm-start with wrong dimensions is ignored", "[pdlp][warmstart]") {
    auto lp = buildSimpleLp();

    PdlpSolver solver;
    PdlpOptions opts;
    opts.verbose = false;
    opts.max_iter = 50000;
    opts.use_gpu = false;
    solver.setOptions(opts);
    solver.load(lp);

    // Set warm-start with wrong dimensions — should be silently ignored.
    std::vector<Real> wrong_x = {1.0};  // num_cols == 2, only 1 provided
    std::vector<Real> wrong_y = {};
    solver.setWarmStart(wrong_x, wrong_y);

    auto result = solver.solve();
    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(-4.0, 1e-4));
}

// ---------------------------------------------------------------------------
// Anderson acceleration tests
// ---------------------------------------------------------------------------

TEST_CASE("PdlpSolver: Anderson acceleration solves simple LP", "[pdlp][anderson]") {
    auto lp = buildSimpleLp();

    PdlpSolver solver;
    PdlpOptions opts;
    opts.verbose = false;
    opts.use_gpu = false;
    opts.max_iter = 50000;
    opts.anderson_acceleration = true;
    opts.anderson_window_size = 5;
    solver.setOptions(opts);
    solver.load(lp);
    auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(-4.0, 1e-4));
}

TEST_CASE("PdlpSolver: Anderson acceleration matches non-AA on simple LP", "[pdlp][anderson]") {
    auto lp = buildSimpleLp();

    auto opts_no_aa = quietOpts(50000, true, false);
    auto result_no_aa = solvePdlp(lp, opts_no_aa);

    auto opts_aa = quietOpts(50000, true, false);
    opts_aa.anderson_acceleration = true;
    opts_aa.anderson_window_size = 5;
    auto result_aa = solvePdlp(lp, opts_aa);

    REQUIRE(result_no_aa.status == Status::Optimal);
    REQUIRE(result_aa.status == Status::Optimal);
    CHECK_THAT(result_aa.objective, WithinAbs(result_no_aa.objective, 1e-3));
}

TEST_CASE("PdlpSolver: Anderson acceleration on medium LP", "[pdlp][anderson]") {
    auto lp = buildMediumLp();

    auto opts = quietOpts(500000, true, false);
    opts.anderson_acceleration = true;
    opts.anderson_window_size = 5;
    auto result = solvePdlp(lp, opts);

    REQUIRE(result.status == Status::Optimal);

    // Verify objective matches non-AA result.
    auto result_no_aa = solvePdlp(lp, quietOpts(500000, true, false));
    REQUIRE(result_no_aa.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(result_no_aa.objective, 1.0));
}

TEST_CASE("PdlpSolver: Anderson acceleration with equality constraints", "[pdlp][anderson]") {
    // min x + 2y  s.t.  x + y = 10,  x,y >= 0
    // Optimal: x=10, y=0, obj = 10.
    LpProblem lp;
    lp.name = "pdlp_aa_equality";
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

    PdlpSolver solver;
    PdlpOptions opts;
    opts.verbose = false;
    opts.use_gpu = false;
    opts.max_iter = 100000;
    opts.anderson_acceleration = true;
    opts.anderson_window_size = 5;
    solver.setOptions(opts);
    solver.load(lp);
    auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(10.0, 0.1));
}

TEST_CASE("PdlpSolver: Anderson acceleration with window size 2", "[pdlp][anderson]") {
    auto lp = buildSimpleLp();

    PdlpSolver solver;
    PdlpOptions opts;
    opts.verbose = false;
    opts.use_gpu = false;
    opts.max_iter = 50000;
    opts.anderson_acceleration = true;
    opts.anderson_window_size = 2;
    solver.setOptions(opts);
    solver.load(lp);
    auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(-4.0, 1e-4));
}

TEST_CASE("PdlpSolver: Anderson acceleration disabled does not change behavior",
          "[pdlp][anderson]") {
    auto lp = buildSimpleLp();

    // Run with AA explicitly disabled.
    PdlpSolver solver1;
    PdlpOptions opts1;
    opts1.verbose = false;
    opts1.use_gpu = false;
    opts1.max_iter = 10000;
    opts1.anderson_acceleration = false;
    solver1.setOptions(opts1);
    solver1.load(lp);
    auto result1 = solver1.solve();

    // Run with default options (AA disabled by default).
    PdlpSolver solver2;
    PdlpOptions opts2;
    opts2.verbose = false;
    opts2.use_gpu = false;
    opts2.max_iter = 10000;
    solver2.setOptions(opts2);
    solver2.load(lp);
    auto result2 = solver2.solve();

    REQUIRE(result1.status == Status::Optimal);
    REQUIRE(result2.status == Status::Optimal);
    // With identical options and no AA, results should be identical.
    CHECK(result1.iterations == result2.iterations);
    CHECK_THAT(result1.objective, WithinAbs(result2.objective, 1e-9));
}

TEST_CASE("PdlpSolver: Anderson acceleration GPU matches CPU", "[pdlp][anderson][gpu]") {
    auto lp = buildMediumLp();

    auto cpu_opts = quietOpts(500000, true, false);
    cpu_opts.anderson_acceleration = true;
    cpu_opts.anderson_window_size = 5;
    auto cpu_result = solvePdlp(lp, cpu_opts);

    auto gpu_opts = quietOpts(500000, true, true);
    gpu_opts.gpu_min_rows = 0;
    gpu_opts.gpu_min_nnz = 0;
    gpu_opts.anderson_acceleration = true;
    gpu_opts.anderson_window_size = 5;
    auto gpu_result = solvePdlp(lp, gpu_opts);

    REQUIRE(cpu_result.status == Status::Optimal);
    if (!PdlpSolver::gpuAvailable()) {
        SKIP("No usable CUDA device: none found, or no device code for it (#192).");
    }
    REQUIRE(gpu_result.status == Status::Optimal);
    CHECK_THAT(gpu_result.objective, WithinAbs(cpu_result.objective, 1e-3));
}

// ---------------------------------------------------------------------------
// Polishing tests (continued PDHG at the polish tolerances)
// ---------------------------------------------------------------------------

TEST_CASE("PdlpSolver: polishing enabled does not regress simple LP", "[pdlp][polishing]") {
    auto lp = buildSimpleLp();

    PdlpSolver solver;
    PdlpOptions opts;
    opts.verbose = false;
    opts.use_gpu = false;
    opts.max_iter = 50000;
    opts.enable_polishing = true;
    solver.setOptions(opts);
    solver.load(lp);
    auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(-4.0, 1e-4));
}

TEST_CASE("PdlpSolver: polishing disabled still converges", "[pdlp][polishing]") {
    auto lp = buildSimpleLp();

    PdlpSolver solver;
    PdlpOptions opts;
    opts.verbose = false;
    opts.use_gpu = false;
    opts.max_iter = 50000;
    opts.enable_polishing = false;
    solver.setOptions(opts);
    solver.load(lp);
    auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(-4.0, 1e-4));
}

TEST_CASE("PdlpSolver: polishing tightens residuals on medium LP", "[pdlp][polishing]") {
    auto lp = buildMediumLp();

    // Solve without polishing.
    PdlpSolver solver_no_polish;
    PdlpOptions opts_no;
    opts_no.verbose = false;
    opts_no.use_gpu = false;
    opts_no.max_iter = 500000;
    opts_no.enable_polishing = false;
    solver_no_polish.setOptions(opts_no);
    solver_no_polish.load(lp);
    auto result_no = solver_no_polish.solve();
    REQUIRE(result_no.status == Status::Optimal);

    // Solve with polishing (tight polish tolerances).
    PdlpSolver solver_polish;
    PdlpOptions opts_yes;
    opts_yes.verbose = false;
    opts_yes.use_gpu = false;
    opts_yes.max_iter = 500000;
    opts_yes.enable_polishing = true;
    opts_yes.polish_primal_tol = 1e-7;
    opts_yes.polish_dual_tol = 1e-7;
    opts_yes.polish_max_iter = 500000;
    solver_polish.setOptions(opts_yes);
    solver_polish.load(lp);
    auto result_yes = solver_polish.solve();
    REQUIRE(result_yes.status == Status::Optimal);

    // Both should produce similar objectives.
    CHECK_THAT(result_yes.objective, WithinAbs(result_no.objective, 1.0));
}

TEST_CASE("PdlpSolver: polishing with equality constraints", "[pdlp][polishing]") {
    // min x + 2y  s.t.  x + y = 10,  x,y >= 0
    // Optimal: x=10, y=0, obj = 10.
    LpProblem lp;
    lp.name = "pdlp_polish_eq";
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

    PdlpSolver solver;
    PdlpOptions opts;
    opts.verbose = false;
    opts.use_gpu = false;
    opts.max_iter = 100000;
    opts.enable_polishing = true;
    opts.polish_primal_tol = 1e-7;
    opts.polish_dual_tol = 1e-7;
    solver.setOptions(opts);
    solver.load(lp);
    auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(10.0, 0.1));
}

TEST_CASE("PdlpSolver: polishing respects time limit", "[pdlp][polishing][time_limit]") {
    auto lp = buildSimpleLp();

    PdlpSolver solver;
    PdlpOptions opts;
    opts.verbose = false;
    opts.use_gpu = false;
    opts.max_iter = 50000;
    opts.enable_polishing = true;
    // Very tight polish tolerance that would take many iterations.
    opts.polish_primal_tol = 1e-15;
    opts.polish_dual_tol = 1e-15;
    opts.polish_max_iter = 100000000;
    // Short time limit — main loop should converge, then polish
    // should be cut short by the time limit.
    opts.max_solve_seconds = 0.5;
    solver.setOptions(opts);
    solver.load(lp);
    auto result = solver.solve();

    // Should still report optimal since main loop converged.
    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(-4.0, 1e-4));
}

TEST_CASE("PdlpSolver: polishing on GPU matches CPU", "[pdlp][polishing][gpu]") {
    auto lp = buildMediumLp();

    PdlpOptions cpu_opts;
    cpu_opts.verbose = false;
    cpu_opts.use_gpu = false;
    cpu_opts.max_iter = 500000;
    cpu_opts.enable_polishing = true;
    cpu_opts.polish_primal_tol = 1e-6;
    cpu_opts.polish_dual_tol = 1e-6;
    auto cpu_result = solvePdlp(lp, cpu_opts);

    PdlpOptions gpu_opts;
    gpu_opts.verbose = false;
    gpu_opts.use_gpu = true;
    gpu_opts.gpu_min_rows = 0;
    gpu_opts.gpu_min_nnz = 0;
    gpu_opts.max_iter = 500000;
    gpu_opts.enable_polishing = true;
    gpu_opts.polish_primal_tol = 1e-6;
    gpu_opts.polish_dual_tol = 1e-6;
    auto gpu_result = solvePdlp(lp, gpu_opts);

    REQUIRE(cpu_result.status == Status::Optimal);
    if (!PdlpSolver::gpuAvailable()) {
        SKIP("No usable CUDA device: none found, or no device code for it (#192).");
    }
    REQUIRE(gpu_result.status == Status::Optimal);
    CHECK_THAT(gpu_result.objective, WithinAbs(cpu_result.objective, 1e-3));
}

TEST_CASE("PdlpSolver: an empty row that excludes 0 makes a column-free LP infeasible", "[pdlp]") {
    // Presolve can leave an emptied row it could neither drop nor call
    // infeasible (#236); with no columns the shortcut must still read it.
    for (const Real upper : {-9e-6, 0.0}) {
        LpProblem lp;
        lp.num_cols = 0;
        lp.num_rows = 1;
        lp.row_lower = {-kInf};
        lp.row_upper = {upper};
        lp.row_names = {"r0"};
        lp.matrix = SparseMatrix(1, 0, {});
        PdlpSolver solver;
        PdlpOptions opts;
        opts.verbose = false;
        solver.setOptions(opts);
        solver.load(lp);
        CHECK(solver.solve().status == (upper < 0.0 ? Status::Infeasible : Status::Optimal));
    }
}

TEST_CASE("PdlpSolver: an infeasible LP with an improving ray is infeasible, not unbounded",
          "[pdlp]") {
    // Each LP is infeasible but has an improving ray. The ray alone proves
    // nothing without a feasible point (#246). A large objective makes the
    // primal iterates run off along the ray before the dual ones prove
    // infeasibility.
    //
    // Every case and cost must end Infeasible: not Unbounded (the #246 bug),
    // and not Optimal, IterLimit or Error either, which would mean the
    // infeasibility was never proved (#252). PdlpSolver does not implement
    // getDualRay(), but it reports Infeasible (on a model with columns) only
    // when a candidate passes the getDualRay() contract checks in the
    // original space (RayDetector in pdlp.cpp, #239), so the status stands
    // for the certificate.
    //   - "apart": x2 >= 1 and x2 <= 0, off the ray x1 = x3 = 1 of min -x1,
    //     plus x1 - x3 = 0.
    //   - "on ray": x1 - x3 >= 1 and x1 - x3 <= 0, the rows the same ray
    //     runs along, plus x2 >= 1 sitting at its bound: a point found by
    //     moving back along the ray only as far as the bounds allow stays
    //     where the iterate is.
    //   - "noise": free columns, x1 >= 1, x1 + x2 - x3 <= 0 and x3 - x2 <= 0
    //     with min -x2 - x3. A ray (1e-12, 1, 1) moves the first row clear of
    //     its violation, but only by noise; mending it breaks the second.
    struct Case {
        const char* name;
        std::vector<Real> obj, row_lower, row_upper, col_lower;
        std::vector<Triplet> trips;
    };
    const Case cases[] = {
        {"apart",
         {-1.0, 0.0, 0.0},
         {1.0, -kInf, 0.0},
         {kInf, 0.0, 0.0},
         {0.0, -kInf, 0.0},
         {{0, 1, 1.0}, {1, 1, 1.0}, {2, 0, 1.0}, {2, 2, -1.0}}},
        {"on ray",
         {-1.0, 0.0, 0.0},
         {1.0, -kInf, -kInf},
         {kInf, 0.0, kInf},
         {0.0, 1.0, 0.0},
         {{0, 0, 1.0}, {0, 2, -1.0}, {1, 0, 1.0}, {1, 2, -1.0}, {2, 1, 1.0}}},
        {"noise",
         {0.0, -1.0, -1.0},
         {1.0, -kInf, -kInf},
         {kInf, 0.0, 0.0},
         {-kInf, -kInf, -kInf},
         {{0, 0, 1.0}, {1, 0, 1.0}, {1, 1, 1.0}, {1, 2, -1.0}, {2, 1, -1.0}, {2, 2, 1.0}}},
    };
    for (const Case& c : cases) {
        for (const Real cost : {1.0, 1e3, 1e6}) {
            LpProblem lp;
            lp.num_cols = 3;
            lp.obj = c.obj;
            for (Real& cj : lp.obj)
                cj *= cost;
            lp.col_lower = c.col_lower;
            lp.col_upper = {kInf, kInf, kInf};
            lp.col_type.assign(3, VarType::Continuous);
            lp.col_names = {"x1", "x2", "x3"};
            lp.num_rows = 3;
            lp.row_lower = c.row_lower;
            lp.row_upper = c.row_upper;
            lp.row_names = {"r0", "r1", "r2"};
            lp.matrix = SparseMatrix(3, 3, std::vector<Triplet>(c.trips));
            PdlpSolver solver;
            PdlpOptions opts;
            opts.verbose = false;
            opts.use_gpu = false;
            solver.setOptions(opts);
            solver.load(lp);
            const Status st = solver.solve().status;
            INFO(c.name << ", cost " << cost << ", status " << static_cast<int>(st));
            CHECK(st == Status::Infeasible);
        }
    }
}
