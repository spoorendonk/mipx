// The barrier honours BarrierOptions::max_solve_seconds (#275).

#include "lp/newton_solver.h"
#include "mipx/barrier.h"
#include "mipx/sparse_cholesky.h"

#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <memory>
#include <string>
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
    // A limit already passed ends the solve before its first iteration, on
    // every CPU backend. Which check stops it (the backend's factorization or
    // the loop's own) is not observable here; the factorize() test below
    // covers the backends' checks.
    const auto algo = GENERATE(BarrierAlgorithm::CpuCholesky, BarrierAlgorithm::CpuAugmented,
                               BarrierAlgorithm::CpuSparseCholesky);
    INFO("algo " << static_cast<int>(algo));
    const auto result = solveWithLimit(algo, 0.0);
    CHECK(result.status == Status::TimeLimit);
    CHECK(result.iterations == 0);
}

TEST_CASE("Barrier with a generous time limit still solves", "[barrier][time-limit]") {
    const auto algo = GENERATE(BarrierAlgorithm::CpuCholesky, BarrierAlgorithm::CpuAugmented,
                               BarrierAlgorithm::CpuSparseCholesky);
    INFO("algo " << static_cast<int>(algo));
    CHECK(solveWithLimit(algo, 600.0).status == Status::Optimal);
    CHECK(solveWithLimit(algo, -1.0).status == Status::Optimal);
}

TEST_CASE("Barrier backends stop factorizing once the deadline has passed",
          "[barrier][time-limit]") {
    // The checks inside factorize() themselves: a solve through solve() is
    // caught by the IPM loop's own check too, so it cannot tell them apart.
    // A chain of rows: row i holds columns i and i + 1.
    constexpr Index kRows = 30;
    constexpr Index kCols = kRows + 1;
    std::vector<Triplet> t;
    for (Index i = 0; i < kRows; ++i) {
        t.push_back({i, i, 1.0});
        t.push_back({i, i + 1, 2.0});
    }
    const SparseMatrix A(kRows, kCols, t);
    const std::vector<Real> ones(kCols, 1.0);
    BarrierOptions opts;
    opts.verbose = false;

    const int backend = GENERATE(0, 1, 2);
    INFO("backend " << backend);
    std::unique_ptr<NewtonSolver> solver = backend == 0   ? createCpuCholeskySolver()
                                           : backend == 1 ? createCpuAugmentedSolver()
                                                          : createCpuSparseCholeskySolver();
    REQUIRE(solver->setup(A, kRows, kCols, opts));
    REQUIRE(solver->factorize(ones, ones, 1e-8));
    solver->setDeadline(Deadline::after(0.0));
    CHECK_FALSE(solver->factorize(ones, ones, 1e-8));
    solver->setDeadline(Deadline{});
    CHECK(solver->factorize(ones, ones, 1e-8));

    // A raised stop flag, with no time set, stops it the same way.
    std::atomic<bool> stop{true};
    solver->setDeadline(Deadline{}.orStop(&stop));
    CHECK_FALSE(solver->factorize(ones, ones, 1e-8));
    stop.store(false);
    CHECK(solver->factorize(ones, ones, 1e-8));
}

TEST_CASE("Barrier with a raised stop flag returns IterLimit, not TimeLimit",
          "[barrier][time-limit]") {
    // The flag is up before the starting point's factorization, which then
    // fails inside the backend: that must still read as a stop, and the
    // empty iterate must not be unscaled (an ASan build catches that). The
    // backend test above is the one that shows the factorization stopping.
    const auto algo = GENERATE(BarrierAlgorithm::CpuCholesky, BarrierAlgorithm::CpuAugmented,
                               BarrierAlgorithm::CpuSparseCholesky);
    INFO("algo " << static_cast<int>(algo));
    const std::atomic<bool> stop{true};
    BarrierOptions opts;
    opts.verbose = false;
    opts.presolve = false;
    opts.algorithm = algo;
    opts.stop_flag = &stop;
    BarrierSolver solver;
    solver.setOptions(opts);
    solver.load(halfKnapsackLp(200));
    const auto result = solver.solve();
    CHECK(result.status == Status::IterLimit);
    CHECK(result.iterations == 0);
}

TEST_CASE("SparseCholesky stops at a passed deadline and says so", "[time-limit][sparse-ne]") {
    // M = 4 I on 3 rows, the diagonal pattern only.
    const std::vector<Index> starts = {0, 1, 2, 3};
    const std::vector<Index> rows = {0, 1, 2};
    const std::vector<Index> perm = {0, 1, 2};
    const std::vector<Real> values = {4.0, 4.0, 4.0};
    SparseCholesky chol;
    REQUIRE(chol.analyze(3, starts, rows, perm));
    const SparseCholesky::PivotRule rule;

    CHECK_FALSE(chol.factorize(values, rule, Deadline::after(0.0)));
    CHECK(chol.error().find("time limit") != std::string::npos);
    // A later factorization without one succeeds and clears the message.
    CHECK(chol.factorize(values, rule));
    CHECK(chol.error().empty());

    const std::atomic<bool> stop{true};
    CHECK_FALSE(chol.factorize(values, rule, Deadline{}.orStop(&stop)));
    CHECK(chol.error().find("stopped") != std::string::npos);
}
