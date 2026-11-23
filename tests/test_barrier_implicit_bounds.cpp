// Upper bounds inside the barrier IPM instead of as rows (#299, #328).

#include "mipx/barrier.h"
#include "mipx/io.h"
#include "test_paths.hpp"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

using namespace mipx;
using Catch::Matchers::WithinAbs;

namespace {

LpProblem makeLp(Sense sense, std::vector<Real> obj, std::vector<Real> col_lower,
                 std::vector<Real> col_upper, std::vector<Real> row_lower,
                 std::vector<Real> row_upper, std::vector<Triplet> trips) {
    LpProblem lp;
    lp.sense = sense;
    lp.num_cols = static_cast<Index>(obj.size());
    lp.num_rows = static_cast<Index>(row_lower.size());
    lp.obj = std::move(obj);
    lp.col_lower = std::move(col_lower);
    lp.col_upper = std::move(col_upper);
    lp.col_type.assign(static_cast<size_t>(lp.num_cols), VarType::Continuous);
    lp.col_names.assign(static_cast<size_t>(lp.num_cols), "");
    lp.row_lower = std::move(row_lower);
    lp.row_upper = std::move(row_upper);
    lp.row_names.assign(static_cast<size_t>(lp.num_rows), "");
    lp.matrix = SparseMatrix(lp.num_rows, lp.num_cols, std::move(trips));
    return lp;
}

// min -x - 2y  s.t.  x + y <= 3,  0 <= x <= 2,  0 <= y <= 1.5.
// Optimum x = y = 1.5, objective -4.5; y sits at its upper bound with reduced
// cost -1 (row dual -1), x is basic with reduced cost 0.
LpProblem boxedLp() {
    return makeLp(Sense::Minimize, {-1.0, -2.0}, {0.0, 0.0}, {2.0, 1.5}, {-kInf}, {3.0},
                  {{0, 0, 1.0}, {0, 1, 1.0}});
}

BarrierOptions quietOptions(bool implicit, BarrierAlgorithm algo = BarrierAlgorithm::CpuCholesky) {
    BarrierOptions opts;
    opts.verbose = false;
    opts.crossover = false;
    opts.presolve = false;
    opts.algorithm = algo;
    opts.implicit_upper_bounds = implicit;
    return opts;
}

}  // namespace

TEST_CASE("Barrier implicit bounds: boxed LP keeps one row per LP row",
          "[barrier][implicit-bounds]") {
    // Not CpuAugmented: its LDL' fails on this LP in either mode (#162).
    const auto algo = GENERATE(BarrierAlgorithm::CpuCholesky, BarrierAlgorithm::CpuSparseCholesky);
    const bool implicit = GENERATE(false, true);
    INFO("implicit " << implicit << " algo " << static_cast<int>(algo));
    BarrierSolver solver;
    solver.setOptions(quietOptions(implicit, algo));
    solver.load(boxedLp());
    // One LP row; the explicit form adds a row per boxed column.
    CHECK(solver.standardFormRows() == (implicit ? 1 : 3));
    CHECK(solver.standardFormCols() == (implicit ? 3 : 5));

    const auto result = solver.solve();
    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(-4.5, 1e-6));
    const auto x = solver.getPrimalValues();
    CHECK_THAT(x[0], WithinAbs(1.5, 1e-6));
    CHECK_THAT(x[1], WithinAbs(1.5, 1e-6));
}

TEST_CASE("Barrier implicit bounds: reduced cost at an upper bound is negative",
          "[barrier][implicit-bounds]") {
    // Minimisation: a column at its upper bound has reduced cost c - A'y <= 0,
    // here -1; the basic one has 0. Maximisation flips the signs.
    const bool maximize = GENERATE(false, true);
    INFO("maximize " << maximize);
    auto lp = boxedLp();
    if (maximize) {
        lp.sense = Sense::Maximize;
        for (auto& c : lp.obj) {
            c = -c;
        }
    }
    BarrierSolver solver;
    solver.setOptions(quietOptions(true));
    solver.load(lp);
    const auto result = solver.solve();
    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(maximize ? 4.5 : -4.5, 1e-6));
    const auto rc = solver.getReducedCosts();
    const Real sign = maximize ? -1.0 : 1.0;
    CHECK_THAT(rc[0], WithinAbs(0.0, 1e-6));
    CHECK_THAT(rc[1], WithinAbs(sign * -1.0, 1e-6));
}

TEST_CASE("Barrier implicit bounds: ranged row slack carries the range",
          "[barrier][implicit-bounds]") {
    // max y  s.t.  1 <= x - y <= 2 (ranged),  x + y <= 4,  x, y >= 0,
    // x <= 10: optimum x = 2.5, y = 1.5 with the range's lower side active,
    // i.e. the slack of x - y + s = 2 at its upper bound s = 1. Without that
    // bound the optimum moves to y = 4.
    auto lp = makeLp(Sense::Maximize, {0.0, 1.0}, {0.0, 0.0}, {10.0, kInf}, {1.0, -kInf},
                     {2.0, 4.0}, {{0, 0, 1.0}, {0, 1, -1.0}, {1, 0, 1.0}, {1, 1, 1.0}});
    const bool implicit = GENERATE(false, true);
    INFO("implicit " << implicit);
    BarrierSolver solver;
    solver.setOptions(quietOptions(implicit));
    solver.load(lp);
    // Explicit: two LP rows, the range row and x's bound row.
    CHECK(solver.standardFormRows() == (implicit ? 2 : 4));
    const auto result = solver.solve();
    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(1.5, 1e-6));
    const auto x = solver.getPrimalValues();
    CHECK_THAT(x[0], WithinAbs(2.5, 1e-6));
    CHECK_THAT(x[1], WithinAbs(1.5, 1e-6));
}

TEST_CASE("Barrier implicit bounds: scaled columns keep their upper bounds",
          "[barrier][implicit-bounds]") {
    // Non-unit coefficients, so Ruiz scales each column differently and the
    // bounds must follow (u / C). min -3x - 2y  s.t.  2x + 0.5y <= 5,
    // 0.1x + 4y <= 8,  0 <= x <= 2,  0 <= y <= 1.5: both columns end at their
    // upper bounds with the rows slack, so the reduced costs are the costs.
    auto lp = makeLp(Sense::Minimize, {-3.0, -2.0}, {0.0, 0.0}, {2.0, 1.5}, {-kInf, -kInf},
                     {5.0, 8.0}, {{0, 0, 2.0}, {0, 1, 0.5}, {1, 0, 0.1}, {1, 1, 4.0}});
    const bool implicit = GENERATE(false, true);
    INFO("implicit " << implicit);
    BarrierSolver solver;
    solver.setOptions(quietOptions(implicit));
    solver.load(lp);
    const auto result = solver.solve();
    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(-9.0, 1e-6));
    const auto x = solver.getPrimalValues();
    CHECK_THAT(x[0], WithinAbs(2.0, 1e-6));
    CHECK_THAT(x[1], WithinAbs(1.5, 1e-6));
    if (implicit) {
        const auto rc = solver.getReducedCosts();
        CHECK_THAT(rc[0], WithinAbs(-3.0, 1e-6));
        CHECK_THAT(rc[1], WithinAbs(-2.0, 1e-6));
    }
}

TEST_CASE("Barrier implicit bounds: bounds-only LP goes to the upper bound",
          "[barrier][implicit-bounds]") {
    // min -x, 1 <= x <= 3, no rows: the standard form has none either.
    auto lp = makeLp(Sense::Minimize, {-1.0}, {1.0}, {3.0}, {}, {}, {});
    BarrierSolver solver;
    solver.setOptions(quietOptions(true));
    solver.load(lp);
    CHECK(solver.standardFormRows() == 0);
    const auto result = solver.solve();
    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(-3.0, 1e-9));
    CHECK_THAT(solver.getPrimalValues()[0], WithinAbs(3.0, 1e-9));
}

TEST_CASE("Barrier implicit bounds: setOptions after load rebuilds the standard form",
          "[barrier][implicit-bounds]") {
    BarrierSolver solver;
    solver.setOptions(quietOptions(false));
    solver.load(boxedLp());
    CHECK(solver.standardFormRows() == 3);
    solver.setOptions(quietOptions(true));
    CHECK(solver.standardFormRows() == 1);
    const auto result = solver.solve();
    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(-4.5, 1e-6));
}

TEST_CASE("Barrier implicit bounds: crossover from the bounded IPM gives a basis",
          "[barrier][implicit-bounds][crossover]") {
    BarrierSolver solver;
    auto opts = quietOptions(true);
    opts.crossover = true;
    solver.setOptions(opts);
    solver.load(boxedLp());
    const auto result = solver.solve();
    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(-4.5, 1e-9));
    const auto basis = solver.getBasis();
    REQUIRE(basis.size() == 3);
    CHECK(basis[1] == BasisStatus::AtUpper);
}

TEST_CASE("Barrier implicit bounds: Netlib LPs reach the same objectives",
          "[barrier][implicit-bounds][netlib]") {
    using Case = std::pair<const char*, Real>;
    auto [name, expected] =
        GENERATE(Case{"adlittle", 2.2549496316e+05}, Case{"afiro", -4.6475314286e+02},
                 Case{"blend", -3.0812149846e+01}, Case{"kb2", -1.7499001299e+03},
                 Case{"sc50a", -6.4575077059e+01}, Case{"sc50b", -7.0000000000e+01},
                 Case{"sc105", -5.2202061212e+01}, Case{"sc205", -5.2202061212e+01},
                 Case{"share1b", -7.6589318579e+04}, Case{"share2b", -4.1573224074e+02},
                 Case{"stocfor1", -4.1131976219e+04}, Case{"bore3d", 1.3730803942e+03},
                 Case{"recipe", -2.6661600000e+02});
    const bool presolve = GENERATE(false, true);
    const auto algo = GENERATE(BarrierAlgorithm::CpuCholesky, BarrierAlgorithm::CpuSparseCholesky);
    INFO(name << " presolve " << presolve << " algo " << static_cast<int>(algo));
    const std::string path = mipx::test::testDataDir() + "/netlib/" + name + ".mps.gz";
    if (!std::filesystem::exists(path)) {
        SKIP("Netlib instance " + std::string(name) + " not downloaded");
    }
    const auto lp = readMps(path);

    BarrierSolver solver;
    auto opts = quietOptions(true, algo);
    opts.presolve = presolve;
    solver.setOptions(opts);
    solver.load(lp);
    const auto result = solver.solve();
    REQUIRE(result.status == Status::Optimal);
    CHECK(std::abs(result.objective - expected) <= 1e-6 * std::max(1.0, std::abs(expected)));
}
