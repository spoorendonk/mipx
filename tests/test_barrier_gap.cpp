// The barrier's stopping test is a relative duality gap, not x'z/n (#253).

#include "mipx/barrier.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cmath>
#include <utility>
#include <vector>

using namespace mipx;

namespace {

// min sum_j (1 + j/n) x_j  s.t.  sum_j x_j >= n/2,  0 <= x_j <= 1. The n/2
// cheapest columns sit at 1: objective n/2 + (n/2)(n/2 - 1) / (2n).
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

}  // namespace

TEST_CASE("Barrier stops on the relative duality gap, not x'z/n, at large n", "[barrier][gap]") {
    // n = 1e4 pairs: a stop on x'z/n at 1e-8 leaves a relative gap up to ~1e-4.
    constexpr Index kN = 10000;
    using Case = std::pair<BarrierAlgorithm, bool>;
    const auto [algo, implicit] = GENERATE(Case{BarrierAlgorithm::CpuSparseCholesky, false},
                                           Case{BarrierAlgorithm::CpuSparseCholesky, true},
                                           Case{BarrierAlgorithm::CpuCholesky, true});
    INFO("algo " << static_cast<int>(algo) << " implicit " << implicit);
    const Real half = kN / 2;
    const Real expected = half + half * (half - 1.0) / (2.0 * kN);

    BarrierOptions opts;
    opts.verbose = false;
    opts.crossover = false;  // the IPM's own point
    opts.presolve = false;
    opts.algorithm = algo;
    opts.implicit_upper_bounds = implicit;
    BarrierSolver solver;
    solver.setOptions(opts);
    solver.load(halfKnapsackLp(kN));
    const auto result = solver.solve();
    REQUIRE(result.status == Status::Optimal);
    CHECK(std::abs(result.objective - expected) <= 1e-6 * std::abs(expected));
}
