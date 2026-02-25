#include <cstdlib>
#include <exception>
#include <print>
#include <string>

#include "mipx/dual_simplex.h"
#include "mipx/io.h"
#include "mipx/mip_solver.h"

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
        std::println("mipx-solve v0.3");
        std::println("Reading: {}", filename);

        auto lp = mipx::readMps(filename);
        std::println("Problem: {} ({} rows, {} cols, {} nonzeros)",
                     lp.name, lp.num_rows, lp.num_cols,
                     lp.matrix.numNonzeros());

        if (lp.hasIntegers()) {
            // MIP solve.
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
            // LP solve.
            mipx::DualSimplexSolver solver;
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
                case mipx::Status::IterLimit:
                    std::println("Status: Iteration limit");
                    break;
                default:
                    std::println("Status: Error");
                    break;
            }
            std::println("Objective: {:.10e}", result.objective);
            std::println("Iterations: {}", result.iterations);
        }

    } catch (const std::exception& e) {
        std::println(stderr, "Error: {}", e.what());
        return 1;
    }

    return 0;
}
