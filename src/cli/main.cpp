#include <cstdlib>
#include <exception>
#include <print>
#include <string>

#include "mipx/dual_simplex.h"
#include "mipx/io.h"
#include "mipx/mip_solver.h"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::println(
            "Usage: mipx-solve <mps-file> [--threads N] [--time-limit S] "
            "[--node-limit N] [--gap-tol G] [--no-cuts|--cuts] "
            "[--no-presolve|--presolve] [--verbose|--quiet]"
        );
        return 1;
    }

    std::string filename = argv[1];
    int num_threads = 1;
    double time_limit = 3600.0;
    mipx::Int node_limit = 1000000;
    double gap_tol = 1e-4;
    bool verbose = true;
    bool presolve = true;
    bool cuts_enabled = true;

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
        } else if (arg == "--verbose") {
            verbose = true;
        } else if (arg == "--quiet") {
            verbose = false;
        } else {
            std::println(stderr, "Unknown argument: {}", arg);
            return 1;
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
            solver.setTimeLimit(time_limit);
            solver.setNodeLimit(node_limit);
            solver.setGapTolerance(gap_tol);
            solver.setVerbose(verbose);
            solver.setPresolve(presolve);
            solver.setCutsEnabled(cuts_enabled);
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
            std::println("Work units: {:.2f}", result.work_units);
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
            std::println("Work units: {:.2f}", result.work_units);
        }

    } catch (const std::exception& e) {
        std::println(stderr, "Error: {}", e.what());
        return 1;
    }

    return 0;
}
