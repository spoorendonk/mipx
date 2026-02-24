#include <cstdlib>
#include <exception>
#include <print>
#include <string>

#include "mipx/dual_simplex.h"
#include "mipx/io.h"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::println("Usage: mipx-solve <mps-file>");
        return 1;
    }

    std::string filename = argv[1];

    try {
        std::println("mipx-solve v0.1");
        std::println("Reading: {}", filename);

        auto lp = mipx::readMps(filename);
        std::println("Problem: {} ({} rows, {} cols, {} nonzeros)",
                     lp.name, lp.num_rows, lp.num_cols,
                     lp.matrix.numNonzeros());

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
            case mipx::Status::TimeLimit:
                std::println("Status: Time limit");
                break;
            default:
                std::println("Status: Error");
                break;
        }

        std::println("Objective: {:.10e}", result.objective);
        std::println("Iterations: {}", result.iterations);

    } catch (const std::exception& e) {
        std::println(stderr, "Error: {}", e.what());
        return 1;
    }

    return 0;
}
