#include "mipx/mip_solver.h"
#include "slow_root_set_cover.hpp"

#include <charconv>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <system_error>

namespace {

const char* statusName(mipx::Status status) {
    using enum mipx::Status;
    switch (status) {
        case Optimal:
            return "Optimal";
        case Infeasible:
            return "Infeasible";
        case Unbounded:
            return "Unbounded";
        case IterLimit:
            return "IterLimit";
        case TimeLimit:
            return "TimeLimit";
        case NodeLimit:
            return "NodeLimit";
        case Error:
            return "Error";
    }
    return "Unknown";
}

bool parseLimit(std::string_view input, double& seconds) {
    const auto end = input.data() + input.size();
    const auto parsed = std::from_chars(input.data(), end, seconds);
    return parsed.ec == std::errc{} && parsed.ptr == end && std::isfinite(seconds) &&
           seconds >= 0.0;
}

}  // namespace

int main(int argc, char** argv) {
    double limit = 8.0;
    if (argc > 2 || (argc == 2 && !parseLimit(argv[1], limit))) {
        std::cerr << "Usage: mipx-root-heuristic-deadline-bench [time-limit-seconds]\n";
        return 2;
    }

    mipx::MipSolver solver;
    solver.setVerbose(false);
    solver.setNumThreads(1);
    solver.setRootHeuristicThresholds(1000, 100000);
    solver.setTimeLimit(limit);
    solver.load(mipx::test::buildSlowRootSetCover(400, 80000, 8));

    const auto start = std::chrono::steady_clock::now();
    const auto result = solver.solve();
    const double wall =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    std::cout << "status,nodes,objective,best_bound,work_units,limit_seconds,solve_seconds,"
                 "wall_seconds,wall_minus_limit_seconds\n"
              << std::setprecision(10) << statusName(result.status) << ',' << result.nodes << ','
              << result.objective << ',' << result.best_bound << ',' << result.work_units << ','
              << limit << ',' << result.time_seconds << ',' << wall << ',' << wall - limit << '\n';
}
