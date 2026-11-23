// The barrier honours BarrierOptions::max_solve_seconds (#275).

#include "mipx/barrier.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <vector>

using namespace mipx;

namespace {

// min sum_j (1 + j/n) x_j  s.t.  sum_j x_j >= n/2,  0 <= x_j <= 1.
LpProblem halfKnapsackLp(Index n) {
    LpProblem lp;
    lp.sense = Sense::Minimize;
    lp.num_cols = n;
    lp.num_rows = 1;
    lp.obj.resize(n);
    std::vector<Triplet> entries;
    entries.reserve(n);
    for (Index j = 0; j < n; ++j) {
        lp.obj[j] = 1.0 + static_cast<Real>(j) / static_cast<Real>(n);
        entries.push_back({0, j, 1.0});
    }
    lp.col_lower.assign(n, 0.0);
    lp.col_upper.assign(n, 1.0);
    lp.col_type.assign(n, VarType::Continuous);
    lp.col_names.assign(n, "");
    lp.row_lower = {static_cast<Real>(n / 2)};
    lp.row_upper = {kInf};
    lp.row_names.assign(1, "");
    lp.matrix = SparseMatrix(1, n, entries);
    return lp;
}

LpResult solveWithLimit(BarrierAlgorithm algo, Real seconds) {
    BarrierOptions opts;
    opts.verbose = false;
    opts.presolve = false;
    opts.algorithm = algo;
    opts.max_solve_seconds = seconds;
    BarrierSolver solver;
    solver.setOptions(opts);
    solver.load(halfKnapsackLp(200));
    const auto r = solver.solve();
    if (r.status != Status::Optimal) {
        UNSCOPED_INFO("error: " << solver.lastError());
    }
    return r;
}

}  // namespace

TEST_CASE("Barrier with a zero time limit returns TimeLimit", "[barrier][time-limit]") {
    // A limit already passed stops the starting point's factorization from
    // inside the backend, on every CPU backend.
    const auto algo = GENERATE(BarrierAlgorithm::CpuCholesky, BarrierAlgorithm::CpuAugmented,
                               BarrierAlgorithm::CpuSparseCholesky);
    INFO("algo " << static_cast<int>(algo));
    const auto result = solveWithLimit(algo, 0.0);
    CHECK(result.status == Status::TimeLimit);
    CHECK(result.iterations == 0);
}

TEST_CASE("Barrier with a generous time limit still solves", "[barrier][time-limit]") {
    // Not CpuAugmented: its LDL' fails this LP with no limit at all.
    const auto algo = GENERATE(BarrierAlgorithm::CpuCholesky, BarrierAlgorithm::CpuSparseCholesky);
    INFO("algo " << static_cast<int>(algo));
    CHECK(solveWithLimit(algo, 600.0).status == Status::Optimal);
    CHECK(solveWithLimit(algo, -1.0).status == Status::Optimal);
}
