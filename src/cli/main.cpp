#include <cstdlib>
#include <exception>
#include <print>
#include <string>
#include <thread>

#include "mipx/dual_simplex.h"
#include "mipx/io.h"
#include "mipx/mip_solver.h"
#include "mipx/presolve.h"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::println("Usage: mipx-solve <mps-file> [--threads N]");
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

        if (lp.hasIntegers()) {
            // MIP solve — MipSolver prints its own banner.
            mipx::MipSolver solver;
            solver.setNumThreads(num_threads);
            solver.load(lp);
            auto result = solver.solve();

            switch (result.status) {
                case mipx::Status::Optimal:
                    std::println("Status: Optimal");
                    break;
                case mipx::Status::Infeasible:
                    std::println("Status: Infeasible");
                    break;
                case mipx::Status::Unbounded:
                    std::println("Status: Unbounded");
                    break;
                case mipx::Status::NodeLimit:
                    std::println("Status: Node limit");
                    break;
                case mipx::Status::TimeLimit:
                    std::println("Status: Time limit");
                    break;
                default:
                    std::println("Status: Error");
                    break;
            }
            std::println("Objective: {:.10e}", result.objective);
            std::println("Nodes: {}", result.nodes);
            std::println("LP iterations: {}", result.lp_iterations);
            std::println("Time: {:.2f}s", result.time_seconds);
        } else {
            // LP solve — print banner from CLI.
            std::println("mipx v0.3");

            unsigned logical = std::thread::hardware_concurrency();
            std::string capabilities;
#ifdef MIPX_HAS_TBB
            capabilities += ", TBB";
#endif
#ifdef __AVX512F__
            capabilities += ", AVX-512";
#elif defined(__AVX2__)
            capabilities += ", AVX2";
#elif defined(__AVX__)
            capabilities += ", AVX";
#elif defined(__SSE4_2__)
            capabilities += ", SSE4.2";
#endif
            std::println("Thread count: {} logical processors, using up to 1 thread{}",
                         logical, capabilities);

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

            std::println("Solving LP with:");
            std::println("  {} rows, {} cols, {} nonzeros",
                         working.num_rows, working.num_cols,
                         working.matrix.numNonzeros());
            std::println("");

            if (did_presolve) {
                std::println("Presolve: {} vars removed, {} rows removed, "
                             "{} bounds tightened, {} rounds",
                             stats.vars_removed, stats.rows_removed,
                             stats.bounds_tightened, stats.rounds);
                std::println("");
            }

            mipx::DualSimplexSolver solver;
            solver.load(working);
            auto result = solver.solve();

            // Postsolve if needed.
            double obj = result.objective;
            if (did_presolve && result.status == mipx::Status::Optimal) {
                auto primals = solver.getPrimalValues();
                auto postsolved = presolver.postsolve(primals);
                // Recompute objective from original problem.
                obj = lp.obj_offset;
                for (mipx::Index j = 0; j < lp.num_cols; ++j) {
                    obj += lp.obj[j] * postsolved[j];
                }
                if (lp.sense == mipx::Sense::Maximize) {
                    // obj is already in original sense from postsolve
                }
            }

            std::println("");
            switch (result.status) {
                case mipx::Status::Optimal:
                    std::println("Optimal: {:.10e} ({} iterations)", obj, result.iterations);
                    break;
                case mipx::Status::Infeasible:
                    std::println("Status: Infeasible");
                    break;
                case mipx::Status::Unbounded:
                    std::println("Status: Unbounded");
                    break;
                case mipx::Status::IterLimit:
                    std::println("Status: Iteration limit");
                    break;
                default:
                    std::println("Status: Error");
                    break;
            }
        }

    } catch (const std::exception& e) {
        std::println(stderr, "Error: {}", e.what());
        return 1;
    }

    return 0;
}
