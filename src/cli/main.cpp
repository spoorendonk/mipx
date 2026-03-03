#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <string>
#include <thread>
#include <chrono>

#include "mipx/dual_simplex.h"
#include "mipx/barrier.h"
#include "mipx/pdlp.h"
#include "mipx/io.h"
#include "mipx/logger.h"
#include "mipx/mip_solver.h"
#include "mipx/presolve.h"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::fprintf(stdout,
            "Usage: mipx-solve <mps-file> [--threads N] [--time-limit S] "
            "[--node-limit N] [--gap-tol G] [--no-cuts|--cuts] "
            "[--no-presolve|--presolve] [--barrier|--pdlp|--dual|--concurrent-root] "
            "[--presolve-forcing-rows|--no-presolve-forcing-rows] "
            "[--presolve-dual-fixing|--no-presolve-dual-fixing] "
            "[--presolve-coeff-tightening|--no-presolve-coeff-tightening] "
            "[--parallel-mode deterministic|opportunistic] [--seed N] "
            "[--heur-deterministic|--heur-opportunistic] "
            "[--pre-root-lpfree|--no-pre-root-lpfree] [--pre-root-work W] "
            "[--pre-root-rounds N] [--pre-root-no-early-stop] "
            "[--pre-root-lplight|--no-pre-root-lplight] "
            "[--pre-root-portfolio|--pre-root-fixed] [--no-symmetry] "
            "[--exact-refine-off|--exact-refine-auto|--exact-refine-on] "
            "[--exact-rational-check|--exact-no-rational-check] "
            "[--exact-warning-tol T] [--exact-cert-tol T] "
            "[--exact-max-rounds N] [--exact-repair-passes N] "
            "[--exact-rational-scale S] "
            "[--dual-idiot-crash|--dual-no-idiot-crash] "
            "[--search-stable|--search-default|--search-aggressive] "
            "[--gpu|--no-gpu] [--gpu-min-rows N] [--gpu-min-nnz N] "
            "[--relax-integrality] "
            "[--verbose|--quiet]\n");
        return 1;
    }

    std::string filename = argv[1];
    int num_threads = 1;
    double time_limit = -1.0;  // negative = use solver default
    mipx::Int node_limit = 1000000;
    double gap_tol = 1e-4;
    bool verbose = true;
    bool presolve = true;
    bool presolve_forcing_rows = true;
    bool presolve_dual_fixing = true;
    bool presolve_coefficient_tightening = false;
    bool cuts_enabled = true;
    enum class LpMode { Dual, Barrier, Pdlp, Concurrent };
    LpMode lp_mode = LpMode::Dual;
    bool barrier_gpu = true;
    bool relax_integrality = false;
    mipx::Int barrier_gpu_min_rows = 512;
    mipx::Int barrier_gpu_min_nnz = 10000;
    mipx::ParallelMode parallel_mode = mipx::ParallelMode::Deterministic;
    uint64_t heuristic_seed = 1;
    bool pre_root_lpfree = false;
    double pre_root_work = 5.0e4;
    mipx::Int pre_root_rounds = 24;
    bool pre_root_early_stop = true;
    bool pre_root_lplight = false;
    bool pre_root_portfolio = true;
    mipx::SearchProfile search_profile = mipx::SearchProfile::Default;
    bool symmetry_enabled = true;
    mipx::ExactRefinementMode exact_refine_mode =
        mipx::ExactRefinementMode::Off;
    bool exact_rational_check = false;
    double exact_warning_tol = 1e-7;
    double exact_cert_tol = 1e-8;
    mipx::Int exact_max_rounds = 2;
    mipx::Int exact_repair_passes = 2;
    double exact_rational_scale = 1.0e6;
    bool dual_idiot_crash = false;

    // Parse optional arguments.
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--threads" && i + 1 < argc) {
            num_threads = std::atoi(argv[++i]);
        } else if (arg == "--time-limit" && i + 1 < argc) {
            time_limit = std::atof(argv[++i]);
        } else if (arg == "--node-limit" && i + 1 < argc) {
            node_limit = static_cast<mipx::Int>(std::atoll(argv[++i]));
        } else if (arg == "--gap-tol" && i + 1 < argc) {
            gap_tol = std::atof(argv[++i]);
        } else if (arg == "--no-cuts") {
            cuts_enabled = false;
        } else if (arg == "--cuts") {
            cuts_enabled = true;
        } else if (arg == "--no-presolve") {
            presolve = false;
        } else if (arg == "--presolve") {
            presolve = true;
        } else if (arg == "--presolve-forcing-rows") {
            presolve_forcing_rows = true;
        } else if (arg == "--no-presolve-forcing-rows") {
            presolve_forcing_rows = false;
        } else if (arg == "--presolve-dual-fixing") {
            presolve_dual_fixing = true;
        } else if (arg == "--no-presolve-dual-fixing") {
            presolve_dual_fixing = false;
        } else if (arg == "--presolve-coeff-tightening") {
            presolve_coefficient_tightening = true;
        } else if (arg == "--no-presolve-coeff-tightening") {
            presolve_coefficient_tightening = false;
        } else if (arg == "--verbose") {
            verbose = true;
        } else if (arg == "--quiet") {
            verbose = false;
        } else if (arg == "--barrier") {
            lp_mode = LpMode::Barrier;
        } else if (arg == "--pdlp") {
            lp_mode = LpMode::Pdlp;
        } else if (arg == "--dual") {
            lp_mode = LpMode::Dual;
        } else if (arg == "--concurrent-root") {
            lp_mode = LpMode::Concurrent;
        } else if (arg == "--parallel-mode" && i + 1 < argc) {
            const std::string mode = argv[++i];
            if (mode == "deterministic") {
                parallel_mode = mipx::ParallelMode::Deterministic;
            } else if (mode == "opportunistic") {
                parallel_mode = mipx::ParallelMode::Opportunistic;
            } else {
                std::fprintf(stderr,
                             "Invalid --parallel-mode value: %s "
                             "(expected deterministic or opportunistic)\n",
                             mode.c_str());
                return 1;
            }
        } else if (arg == "--heur-deterministic") {
            parallel_mode = mipx::ParallelMode::Deterministic;
        } else if (arg == "--heur-opportunistic") {
            parallel_mode = mipx::ParallelMode::Opportunistic;
        } else if (arg == "--seed" && i + 1 < argc) {
            heuristic_seed = static_cast<uint64_t>(
                std::strtoull(argv[++i], nullptr, 10));
        } else if (arg == "--pre-root-lpfree") {
            pre_root_lpfree = true;
        } else if (arg == "--no-pre-root-lpfree") {
            pre_root_lpfree = false;
        } else if (arg == "--pre-root-work" && i + 1 < argc) {
            pre_root_work = std::max(1.0, std::atof(argv[++i]));
        } else if (arg == "--pre-root-rounds" && i + 1 < argc) {
            pre_root_rounds =
                std::max<mipx::Int>(1, static_cast<mipx::Int>(std::atoll(argv[++i])));
        } else if (arg == "--pre-root-no-early-stop") {
            pre_root_early_stop = false;
        } else if (arg == "--pre-root-lplight") {
            pre_root_lplight = true;
        } else if (arg == "--no-pre-root-lplight") {
            pre_root_lplight = false;
        } else if (arg == "--pre-root-portfolio") {
            pre_root_portfolio = true;
        } else if (arg == "--pre-root-fixed") {
            pre_root_portfolio = false;
        } else if (arg == "--no-symmetry") {
            symmetry_enabled = false;
        } else if (arg == "--exact-refine-off") {
            exact_refine_mode = mipx::ExactRefinementMode::Off;
        } else if (arg == "--exact-refine-auto") {
            exact_refine_mode = mipx::ExactRefinementMode::Auto;
        } else if (arg == "--exact-refine-on") {
            exact_refine_mode = mipx::ExactRefinementMode::On;
        } else if (arg == "--exact-rational-check") {
            exact_rational_check = true;
        } else if (arg == "--exact-no-rational-check") {
            exact_rational_check = false;
        } else if (arg == "--exact-warning-tol" && i + 1 < argc) {
            exact_warning_tol = std::max(1e-12, std::atof(argv[++i]));
        } else if (arg == "--exact-cert-tol" && i + 1 < argc) {
            exact_cert_tol = std::max(1e-12, std::atof(argv[++i]));
        } else if (arg == "--exact-max-rounds" && i + 1 < argc) {
            exact_max_rounds =
                std::max<mipx::Int>(1, static_cast<mipx::Int>(std::atoll(argv[++i])));
        } else if (arg == "--exact-repair-passes" && i + 1 < argc) {
            exact_repair_passes =
                std::max<mipx::Int>(1, static_cast<mipx::Int>(std::atoll(argv[++i])));
        } else if (arg == "--exact-rational-scale" && i + 1 < argc) {
            exact_rational_scale = std::max(1.0, std::atof(argv[++i]));
        } else if (arg == "--dual-idiot-crash") {
            dual_idiot_crash = true;
        } else if (arg == "--dual-no-idiot-crash") {
            dual_idiot_crash = false;
        } else if (arg == "--search-stable") {
            search_profile = mipx::SearchProfile::Stable;
        } else if (arg == "--search-default") {
            search_profile = mipx::SearchProfile::Default;
        } else if (arg == "--search-aggressive") {
            search_profile = mipx::SearchProfile::Aggressive;
        } else if (arg == "--gpu") {
            barrier_gpu = true;
        } else if (arg == "--no-gpu") {
            barrier_gpu = false;
        } else if (arg == "--gpu-min-rows" && i + 1 < argc) {
            barrier_gpu_min_rows =
                std::max<mipx::Int>(0, static_cast<mipx::Int>(std::atoll(argv[++i])));
        } else if (arg == "--gpu-min-nnz" && i + 1 < argc) {
            barrier_gpu_min_nnz =
                std::max<mipx::Int>(0, static_cast<mipx::Int>(std::atoll(argv[++i])));
        } else if (arg == "--relax-integrality") {
            relax_integrality = true;
        } else {
            std::fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            return 1;
        }
    }

    try {
        auto lp = mipx::readMps(filename);
        if (relax_integrality && lp.hasIntegers()) {
            for (auto& t : lp.col_type) t = mipx::VarType::Continuous;
        }
        mipx::Logger log;
        mipx::PresolveOptions presolve_opts;
        presolve_opts.enable_forcing_rows = presolve_forcing_rows;
        presolve_opts.enable_dual_fixing = presolve_dual_fixing;
        presolve_opts.enable_coefficient_tightening =
            presolve_coefficient_tightening;

        if (lp.hasIntegers()) {
            // MIP solve — MipSolver prints its own banner.
            mipx::MipSolver solver;
            solver.setNumThreads(num_threads);
            if (time_limit >= 0.0) solver.setTimeLimit(time_limit);
            solver.setNodeLimit(node_limit);
            solver.setGapTolerance(gap_tol);
            solver.setVerbose(verbose);
            solver.setPresolve(presolve);
            solver.setRootPresolveOptions(presolve_opts);
            solver.setCutsEnabled(cuts_enabled);
            if (lp_mode == LpMode::Barrier) {
                solver.setRootLpPolicy(mipx::RootLpPolicy::BarrierRoot);
            } else if (lp_mode == LpMode::Pdlp) {
                solver.setRootLpPolicy(mipx::RootLpPolicy::PdlpRoot);
            } else if (lp_mode == LpMode::Concurrent) {
                solver.setRootLpPolicy(mipx::RootLpPolicy::ConcurrentRootExperimental);
            } else {
                solver.setRootLpPolicy(mipx::RootLpPolicy::DualDefault);
            }
            solver.setBarrierAlgorithm(barrier_gpu ? mipx::BarrierAlgorithm::Auto
                                                    : mipx::BarrierAlgorithm::CpuCholesky);
            solver.setPdlpUseGpu(barrier_gpu);
            solver.setPdlpGpuThresholds(barrier_gpu_min_rows, barrier_gpu_min_nnz);
            solver.setParallelMode(parallel_mode);
            solver.setHeuristicSeed(heuristic_seed);
            solver.setPreRootLpFreeEnabled(pre_root_lpfree);
            solver.setPreRootLpFreeWorkBudget(pre_root_work);
            solver.setPreRootLpFreeMaxRounds(pre_root_rounds);
            solver.setPreRootLpFreeEarlyStop(pre_root_early_stop);
            solver.setPreRootLpLightEnabled(pre_root_lplight);
            solver.setPreRootPortfolioEnabled(pre_root_portfolio);
            solver.setSymmetryEnabled(symmetry_enabled);
            solver.setExactRefinementMode(exact_refine_mode);
            solver.setExactRefinementRationalCheck(exact_rational_check);
            solver.setExactRefinementWarningTolerance(exact_warning_tol);
            solver.setExactRefinementCertificateTolerance(exact_cert_tol);
            solver.setExactRefinementMaxRounds(exact_max_rounds);
            solver.setExactRefinementRepairPasses(exact_repair_passes);
            solver.setExactRefinementRationalScale(exact_rational_scale);
            solver.setSearchProfile(search_profile);
            solver.load(lp);
            auto result = solver.solve();

            if (result.status == mipx::Status::Optimal &&
                result.gap_limit_reached) {
                log.log("Status: Gap limit\n");
            } else {
                switch (result.status) {
                    case mipx::Status::Optimal:    log.log("Status: Optimal\n"); break;
                    case mipx::Status::Infeasible: log.log("Status: Infeasible\n"); break;
                    case mipx::Status::Unbounded:  log.log("Status: Unbounded\n"); break;
                    case mipx::Status::NodeLimit:  log.log("Status: Node limit\n"); break;
                    case mipx::Status::TimeLimit:  log.log("Status: Time limit\n"); break;
                    default:                       log.log("Status: Error\n"); break;
                }
            }
            log.log("Objective: %.10e\n", result.objective);
            log.log("Nodes: %d\n", result.nodes);
            log.log("LP iterations: %d\n", result.lp_iterations);
            log.log("Work units: %.2f\n", result.work_units);
            log.log("Time: %.2fs\n", result.time_seconds);
        } else {
            // LP solve — print banner from CLI.
            log.log("mipx v0.3\n");

            unsigned logical = std::thread::hardware_concurrency();
            const char* tbb_str = "";
            const char* simd_str = "";
#ifdef MIPX_HAS_TBB
            tbb_str = ", TBB";
#endif
#ifdef __AVX512F__
            simd_str = ", AVX-512";
#elif defined(__AVX2__)
            simd_str = ", AVX2";
#elif defined(__AVX__)
            simd_str = ", AVX";
#elif defined(__SSE4_2__)
            simd_str = ", SSE4.2";
#endif
            log.log("Thread count: %u logical processors, using up to 1 thread%s%s\n",
                     logical, tbb_str, simd_str);

            // Presolve.
            mipx::Presolver presolver;
            presolver.setOptions(presolve_opts);
            auto working = lp;
            bool did_presolve = false;
            if (presolve) {
                working = presolver.presolve(lp);
                const auto& stats = presolver.stats();
                if (stats.vars_removed > 0 || stats.rows_removed > 0) {
                    did_presolve = true;
                } else {
                    working = lp;
                }
            }

            log.log("Solving LP with:\n");
            log.log("  %d rows, %d cols, %d nonzeros\n",
                     working.num_rows, working.num_cols,
                     working.matrix.numNonzeros());
            log.log("\n");

            if (did_presolve) {
                const auto& stats = presolver.stats();
                log.log("Presolve: %d vars removed, %d rows removed, "
                         "%d bounds tightened, %d rounds (%d changed), %.3fs "
                         "[rules: forcing=%d implied=%d abt=%d dual=%d coeff=%d "
                         "doubleton=%d empty_col=%d dup_row=%d par_row=%d] "
                         "[examined: %d rows, %d cols]\n\n",
                         stats.vars_removed, stats.rows_removed,
                         stats.bounds_tightened, stats.rounds,
                         stats.rounds_with_changes, stats.time_seconds,
                         stats.forcing_row_changes,
                         stats.implied_equation_changes,
                         stats.activity_bound_tightening_changes,
                         stats.dual_fixing_changes,
                         stats.coeff_tightening_changes,
                         stats.doubleton_eq_changes,
                         stats.empty_col_changes,
                         stats.duplicate_row_changes,
                         stats.parallel_row_changes,
                         stats.rows_examined, stats.cols_examined);
            }

            auto t0 = std::chrono::steady_clock::now();
            mipx::LpResult result;
            std::vector<mipx::Real> primals;
            bool used_gpu_backend = false;
            if (lp_mode == LpMode::Barrier) {
                mipx::BarrierSolver solver;
                mipx::BarrierOptions bopts;
                bopts.verbose = verbose;
                bopts.algorithm = barrier_gpu ? mipx::BarrierAlgorithm::Auto
                                              : mipx::BarrierAlgorithm::CpuCholesky;
                solver.setOptions(bopts);
                solver.load(working);
                result = solver.solve();
                primals = solver.getPrimalValues();
                used_gpu_backend = solver.usedGpu();
            } else if (lp_mode == LpMode::Pdlp) {
                mipx::PdlpSolver solver;
                mipx::PdlpOptions popts;
                popts.verbose = verbose;
                popts.use_gpu = barrier_gpu;
                popts.gpu_min_rows = barrier_gpu_min_rows;
                popts.gpu_min_nnz = barrier_gpu_min_nnz;
                solver.setOptions(popts);
                solver.load(working);
                result = solver.solve();
                primals = solver.getPrimalValues();
                used_gpu_backend = solver.usedGpu();
            } else {
                mipx::DualSimplexSolver solver;
                solver.setVerbose(verbose);
                mipx::DualSimplexOptions dopts = solver.getOptions();
                dopts.enable_idiot_crash = dual_idiot_crash;
                if (time_limit >= 0.0) {
                    dopts.max_solve_seconds = time_limit;
                }
                solver.setOptions(dopts);
                solver.load(working);
                result = solver.solve();
                primals = solver.getPrimalValues();
            }
            auto t1 = std::chrono::steady_clock::now();
            double solve_seconds = std::chrono::duration<double>(t1 - t0).count();

            if (lp_mode == LpMode::Barrier && verbose) {
                log.log("Barrier backend: %s\n", used_gpu_backend ? "GPU" : "CPU");
            }
            if (lp_mode == LpMode::Pdlp && verbose) {
                log.log("PDLP backend: %s\n", used_gpu_backend ? "GPU" : "CPU");
            }

            // Postsolve if needed.
            double obj = result.objective;
            if (did_presolve && result.status == mipx::Status::Optimal) {
                auto postsolved = presolver.postsolve(primals);
                obj = lp.obj_offset;
                for (mipx::Index j = 0; j < lp.num_cols; ++j) {
                    obj += lp.obj[j] * postsolved[j];
                }
            }

            std::fflush(stdout);  // flush LP solver's printf output before Logger write()
            log.log("\n");
            const char* status_text = "Error";
            switch (result.status) {
                case mipx::Status::Optimal:
                    status_text = "Optimal";
                    log.log("Optimal: %.10e (%d iterations)\n", obj, result.iterations);
                    break;
                case mipx::Status::Infeasible:
                    status_text = "Infeasible";
                    log.log("Status: Infeasible\n");
                    break;
                case mipx::Status::Unbounded:
                    status_text = "Unbounded";
                    log.log("Status: Unbounded\n");
                    break;
                case mipx::Status::IterLimit:
                    status_text = "Iteration limit";
                    log.log("Status: Iteration limit\n");
                    break;
                case mipx::Status::TimeLimit:
                    status_text = "Time limit";
                    log.log("Status: Time limit\n");
                    break;
                default:
                    status_text = "Error";
                    log.log("Status: Error\n");
                    break;
            }
            // Stable summary lines consumed by benchmark scripts.
            log.log("Status: %s\n", status_text);
            log.log("Objective: %.10e\n", obj);
            log.log("Iterations: %d\n", result.iterations);
            log.log("Work units: %.2f\n", result.work_units);
            log.log("Time: %.2fs\n", solve_seconds);
        }

    } catch (const std::exception& e) {
        std::fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }

    return 0;
}
