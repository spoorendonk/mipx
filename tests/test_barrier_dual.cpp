#include "mipx/barrier.h"

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <utility>

using namespace mipx;
using Catch::Matchers::WithinAbs;

namespace {

LpProblem mixedBoundsLp(Sense sense, Real range_sign) {
    LpProblem lp;
    lp.sense = sense;
    lp.num_cols = 5;
    lp.obj = {11.0, -1.0, 1.5, -3.0, 2.375};
    if (sense == Sense::Maximize) {
        for (Real& cost : lp.obj) {
            cost = -cost;
        }
    }
    lp.obj_offset = 17.0;
    lp.col_lower = {-kInf, 2.0, -kInf, 3.0, 4.0};
    lp.col_upper = {kInf, 6.0, -1.0, kInf, 4.0};
    lp.col_type.assign(lp.num_cols, VarType::Continuous);
    lp.num_rows = 6;
    lp.row_lower = {11.0, 11.0, range_sign > 0 ? 7.0 : -12.0, -kInf, 2.0, -kInf};
    lp.row_upper = {11.0, 11.0, range_sign > 0 ? 12.0 : -7.0, 10.0, kInf, kInf};
    // Duplicate the first equality to make its price nonunique. The final
    // row is free and carries no price.
    lp.matrix = SparseMatrix(6, 5,
                             {{0, 0, 2.0},
                              {0, 1, 1.0},
                              {0, 4, 0.25},
                              {1, 0, 2.0},
                              {1, 1, 1.0},
                              {1, 4, 0.25},
                              {2, 0, 4.0 * range_sign},
                              {2, 2, range_sign},
                              {3, 1, 0.5},
                              {3, 3, 1.0},
                              {4, 2, 1.0},
                              {4, 3, 1.0},
                              {5, 0, 1.0},
                              {5, 2, 1.0}});
    return lp;
}

BarrierOptions dualOptions() {
    BarrierOptions options;
    options.algorithm = BarrierAlgorithm::CpuCholesky;
    options.dualize = BarrierDualize::On;
    options.presolve = false;
    options.crossover = false;
    options.verbose = false;
    return options;
}

LpProblem tallLp(Index rows) {
    LpProblem lp;
    lp.num_cols = 2;
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {kInf, kInf};
    lp.col_type.assign(lp.num_cols, VarType::Continuous);
    lp.obj = {1.0, 1.0};
    lp.num_rows = rows;
    lp.row_upper.assign(rows, kInf);
    std::vector<Triplet> trips;
    for (Index i = 0; i < rows; ++i) {
        const Real fraction = static_cast<Real>(i) / (rows - 1);
        const Real slope = 0.5 + fraction;
        trips.push_back({i, 0, slope});
        trips.push_back({i, 1, 1.0});
        // The extreme slopes are the only active rows at (1, 1).
        lp.row_lower.push_back(1.0 + slope - fraction * (1.0 - fraction));
    }
    lp.matrix = SparseMatrix(rows, 2, std::move(trips));
    return lp;
}

}  // namespace

TEST_CASE("Barrier dualization restores mixed-bound solutions in original coordinates",
          "[barrier][dualize]") {
    const Sense sense = GENERATE(Sense::Minimize, Sense::Maximize);
    const Real range_sign = GENERATE(1.0, -1.0);
    auto opts = dualOptions();
    opts.presolve = GENERATE(false, true);
    opts.crossover = GENERATE(false, true);
    opts.implicit_upper_bounds = true;
    BarrierSolver solver;
    solver.setOptions(opts);
    const auto lp = mixedBoundsLp(sense, range_sign);
    solver.load(lp);
    REQUIRE(solver.usedDualization());
    CHECK(solver.standardFormRows() == 4);
    const Real sign = sense == Sense::Minimize ? 1.0 : -1.0;
    const auto verify = [&] {
        const auto result = solver.solve();
        INFO(solver.lastError());
        REQUIRE(result.status == Status::Optimal);
        CHECK_THAT(result.objective, WithinAbs(17.0 + sign * 3.0, 2e-6));
        const auto x = solver.getPrimalValues();
        const auto y = solver.getDualValues();
        const auto rc = solver.getReducedCosts();
        REQUIRE(x.size() == 5);
        REQUIRE(y.size() == 6);
        REQUIRE(rc.size() == 5);
        const std::array<Real, 5> expected{2.0, 6.0, -1.0, 7.0, 4.0};
        std::array<Real, 5> aty;
        lp.matrix.multiplyTranspose(y, aty);
        for (Index j = 0; j < lp.num_cols; ++j) {
            CHECK_THAT(x[j], WithinAbs(expected[j], 2e-6));
            CHECK_THAT(aty[j] + rc[j], WithinAbs(lp.obj[j], 2e-6));
        }
        CHECK(range_sign * sign * y[2] > 0.0);
        CHECK(sign * y[3] < 0.0);
        CHECK_THAT(y[4], WithinAbs(0.0, 2e-6));   // slack lower side
        CHECK_THAT(y[5], WithinAbs(0.0, 2e-6));   // free row
        CHECK_THAT(rc[0], WithinAbs(0.0, 2e-6));  // free column
        CHECK(sign * rc[1] < 0.0);                // boxed column at its upper bound
        CHECK(sign * rc[2] <= 2e-6);              // upper-only column can be basic and degenerate
        CHECK_THAT(rc[3], WithinAbs(0.0, 2e-6));  // interior column
        if (opts.crossover) {
            CHECK(solver.getBasis().size() == 11);
        }
    };
    verify();
    verify();  // Ruiz scales still map a repeat solve to the input coordinates.
}

TEST_CASE("Barrier auto dualization can solve a tall LP within the dense factor cap",
          "[barrier][dualize]") {
    const auto lp = tallLp(GENERATE(128, 256, 512));
    auto opts = dualOptions();
    opts.dualize = BarrierDualize::Off;
    opts.ne_max_bytes = 4096;
    BarrierSolver solver;
    solver.setOptions(opts);
    solver.load(lp);
    REQUIRE(solver.standardFormRows() == lp.num_rows);
    CHECK(solver.solve().status == Status::Error);
    CHECK_FALSE(solver.usedDualization());

    opts.dualize = BarrierDualize::Auto;
    solver.setOptions(opts);
    REQUIRE(solver.usedDualization());
    CHECK(solver.standardFormRows() == lp.num_cols);
    const auto result = solver.solve();
    INFO(solver.lastError());
    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(2.0, 1e-6));
}

TEST_CASE("Barrier auto compares sparse factors after dense-column splitting",
          "[barrier][dualize]") {
    const bool split_columns = GENERATE(false, true);
    LpProblem lp;
    lp.num_cols = split_columns ? 64 : 24;
    lp.num_rows = 2 * lp.num_cols + 1;
    lp.col_lower.assign(lp.num_cols, 0.0);
    lp.col_upper.assign(lp.num_cols, kInf);
    lp.col_type.assign(lp.num_cols, VarType::Continuous);
    lp.obj.assign(lp.num_cols, 1.0);
    lp.row_lower.assign(lp.num_rows, -kInf);
    lp.row_upper.assign(lp.num_rows, 2.0);
    lp.row_lower[0] = 1.0;
    lp.row_upper[0] = kInf;
    std::vector<Triplet> trips;
    for (Index j = 0; j < lp.num_cols; ++j) {
        trips.push_back({0, j, 1.0});
    }
    for (Index i = 1; i < lp.num_rows; ++i) {
        trips.push_back({i, (i - 1) % lp.num_cols, 1.0});
    }
    lp.matrix = SparseMatrix(lp.num_rows, lp.num_cols, std::move(trips));

    auto opts = dualOptions();
    opts.algorithm = BarrierAlgorithm::CpuSparseCholesky;
    opts.dualize = BarrierDualize::Auto;
    if (split_columns) {
        // The primal matrix alone exceeds the cap. Peeling the dual's
        // one dense column lets its diagonal sparse factor fit.
        opts.ne_max_bytes = 5000;
    } else {
        // Without the split, the wide sum row makes the dual factor dense.
        opts.ne_max_dense_cols = 0;
    }
    BarrierSolver solver;
    solver.setOptions(opts);
    solver.load(lp);
    CHECK(solver.usedDualization() == split_columns);
    const auto result = solver.solve();
    INFO(solver.lastError());
    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(1.0, 1e-6));
}

TEST_CASE("Barrier auto avoids a cheaper dense factor above the cap", "[barrier][dualize]") {
    LpProblem lp;
    lp.num_cols = 1;
    lp.col_lower = {0.0};
    lp.col_upper = {kInf};
    lp.obj = {1.0};
    lp.col_type = {VarType::Continuous};
    lp.num_rows = 2;
    lp.row_lower = {1.0, 1.0};
    lp.row_upper = lp.row_lower;
    lp.matrix = SparseMatrix(2, 1, {{0, 0, 1.0}, {1, 0, 1.0}});

    auto opts = dualOptions();
    opts.dualize = BarrierDualize::Auto;
    BarrierSolver solver;
    solver.setOptions(opts);
    solver.load(lp);
    REQUIRE_FALSE(solver.usedDualization());
    opts.ne_max_bytes = 16;
    solver.setOptions(opts);
    CHECK(solver.usedDualization());
    const auto result = solver.solve();
    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(1.0, 1e-6));
}

TEST_CASE("Barrier dualization postsolves the price of a bound introduced by presolve",
          "[barrier][dualize]") {
    LpProblem lp;
    lp.sense = Sense::Maximize;
    lp.num_cols = 1;
    lp.col_lower = {-kInf};
    lp.col_upper = {kInf};
    lp.col_type = {VarType::Continuous};
    lp.obj = {2.0};
    lp.num_rows = 1;
    lp.row_lower = {-6.0};
    lp.row_upper = {kInf};
    lp.matrix = SparseMatrix(1, 1, {{0, 0, -2.0}});

    auto opts = dualOptions();
    opts.presolve = true;
    BarrierSolver solver;
    solver.setOptions(opts);
    solver.load(lp);
    const auto result = solver.solve();
    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(6.0, 1e-6));
    CHECK_THAT(solver.getDualValues().at(0), WithinAbs(-1.0, 1e-6));
    CHECK_THAT(solver.getReducedCosts().at(0), WithinAbs(0.0, 1e-6));
}

TEST_CASE("Barrier dualized HSD reports infeasibility and unboundedness in primal space",
          "[barrier][dualize][hsd]") {
    LpProblem lp;
    lp.num_cols = 2;
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {kInf, kInf};
    lp.col_type.assign(lp.num_cols, VarType::Continuous);
    Status expected = Status::Infeasible;
    auto opts = dualOptions();
    SECTION("Primal infeasible") {
        lp.obj = {1.0, 1.0};
        lp.num_rows = 2;
        lp.row_lower = {-kInf, 2.0};
        lp.row_upper = {1.0, kInf};
        lp.matrix = SparseMatrix(2, 2, {{0, 0, 1.0}, {0, 1, 1.0}, {1, 0, 1.0}, {1, 1, 1.0}});
    }
    SECTION("Primal unbounded") {
        lp.obj = {-1.0, -1.0};
        lp.num_rows = 1;
        lp.row_lower = {-kInf};
        lp.row_upper = {1.0};
        lp.matrix = SparseMatrix(1, 2, {{0, 0, 1.0}, {0, 1, -1.0}});
        expected = Status::Unbounded;
    }
    SECTION("Both primal and dual infeasible") {
        lp.obj = {-1.0, -1.0};
        lp.num_rows = 2;
        lp.row_lower = {1.0, 1.0};
        lp.row_upper = {kInf, kInf};
        lp.matrix = SparseMatrix(2, 2, {{0, 0, 1.0}, {0, 1, -1.0}, {1, 0, -1.0}, {1, 1, 1.0}});
    }
    SECTION("The unboundedness probe also needs the compact factor") {
        lp.col_lower[1] = -kInf;
        lp.obj = {0.0, -1.0};
        lp.num_rows = 3;
        lp.row_lower = {1.0, 1.0, 1.0};
        lp.row_upper = {kInf, kInf, kInf};
        lp.matrix = SparseMatrix(3, 2, {{0, 0, 1.0}, {1, 0, 1.0}, {2, 0, 1.0}});
        opts.ne_max_bytes = 32;
        expected = Status::Unbounded;
    }
    opts.homogeneous = true;
    BarrierSolver solver;
    solver.setOptions(opts);
    solver.load(lp);
    const auto result = solver.solve();
    INFO(solver.lastError());
    CHECK(result.status == expected);
    CHECK(solver.getPrimalValues().empty());
}
