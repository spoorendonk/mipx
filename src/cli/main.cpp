#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <thread>

#include "mipx/dual_simplex.h"
#include "mipx/io.h"
#include "mipx/logger.h"
#include "mipx/mip_solver.h"
#include "mipx/presolve.h"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::fprintf(stdout, "Usage: mipx-solve <mps-file> [--threads N]\n");
        return 1;
    }

    std::string filename = argv[1];
    int num_threads = 1;

    // Parse optional arguments.
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--threads" && i + 1 < argc) {
            num_threads = std::atoi(argv[++i]);
        }
    }

    try {
        auto lp = mipx::readMps(filename);
        mipx::Logger log;

        if (lp.hasIntegers()) {
            // MIP solve — MipSolver prints its own banner.
            mipx::MipSolver solver;
            solver.setNumThreads(num_threads);
            solver.load(lp);
            auto result = solver.solve();

            switch (result.status) {
                case mipx::Status::Optimal:    log.log("Status: Optimal\n"); break;
                case mipx::Status::Infeasible: log.log("Status: Infeasible\n"); break;
                case mipx::Status::Unbounded:  log.log("Status: Unbounded\n"); break;
                case mipx::Status::NodeLimit:  log.log("Status: Node limit\n"); break;
                case mipx::Status::TimeLimit:  log.log("Status: Time limit\n"); break;
                default:                       log.log("Status: Error\n"); break;
            }
            log.log("Objective: %.10e\n", result.objective);
            log.log("Nodes: %d\n", result.nodes);
            log.log("LP iterations: %d\n", result.lp_iterations);
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
            auto working = presolver.presolve(lp);
            bool did_presolve = false;
            const auto& stats = presolver.stats();
            if (stats.vars_removed > 0 || stats.rows_removed > 0) {
                did_presolve = true;
            } else {
                working = lp;
            }

            log.log("Solving LP with:\n");
            log.log("  %d rows, %d cols, %d nonzeros\n",
                     working.num_rows, working.num_cols,
                     working.matrix.numNonzeros());
            log.log("\n");

            if (did_presolve) {
                log.log("Presolve: %d vars removed, %d rows removed, "
                         "%d bounds tightened, %d rounds\n\n",
                         stats.vars_removed, stats.rows_removed,
                         stats.bounds_tightened, stats.rounds);
            }

            mipx::DualSimplexSolver solver;
            solver.load(working);
            auto result = solver.solve();

            // Postsolve if needed.
            double obj = result.objective;
            if (did_presolve && result.status == mipx::Status::Optimal) {
                auto primals = solver.getPrimalValues();
                auto postsolved = presolver.postsolve(primals);
                obj = lp.obj_offset;
                for (mipx::Index j = 0; j < lp.num_cols; ++j) {
                    obj += lp.obj[j] * postsolved[j];
                }
            }

            std::fflush(stdout);  // flush LP solver's printf output before Logger write()
            log.log("\n");
            switch (result.status) {
                case mipx::Status::Optimal:
                    log.log("Optimal: %.10e (%d iterations)\n", obj, result.iterations);
                    break;
                case mipx::Status::Infeasible:
                    log.log("Status: Infeasible\n");
                    break;
                case mipx::Status::Unbounded:
                    log.log("Status: Unbounded\n");
                    break;
                case mipx::Status::IterLimit:
                    log.log("Status: Iteration limit\n");
                    break;
                default:
                    log.log("Status: Error\n");
                    break;
            }
        }

    } catch (const std::exception& e) {
        std::fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }

    return 0;
}
