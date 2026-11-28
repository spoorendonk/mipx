// Starting points of the CPU barrier's Mehrotra loop (#144).

#include "mipx/barrier.h"
#include "mipx/io.h"
#include "test_paths.hpp"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cmath>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

using namespace mipx;

namespace {

BarrierOptions quietOptions(BarrierInitialPoint start, BarrierAlgorithm algo, bool implicit) {
    BarrierOptions opts;
    opts.verbose = false;
    opts.crossover = false;
    opts.presolve = false;
    opts.algorithm = algo;
    opts.implicit_upper_bounds = implicit;
    opts.initial_point = start;
    return opts;
}

// min -x - 2y + z  s.t.  x + y + z <= 3,  x - y >= -1,  0 <= x <= 2,
// 0 <= y <= 1.5, z >= 0.5. Optimum x = 1, y = 1.5, z = 0.5, objective -3.5.
LpProblem smallLp() {
    LpProblem lp;
    lp.sense = Sense::Minimize;
    lp.num_cols = 3;
    lp.num_rows = 2;
    lp.obj = {-1.0, -2.0, 1.0};
    lp.col_lower = {0.0, 0.0, 0.5};
    lp.col_upper = {2.0, 1.5, kInf};
    lp.col_type.assign(3, VarType::Continuous);
    lp.col_names.assign(3, "");
    lp.row_lower = {-kInf, -1.0};
    lp.row_upper = {3.0, kInf};
    lp.row_names.assign(2, "");
    lp.matrix =
        SparseMatrix(2, 3, {{0, 0, 1.0}, {0, 1, 1.0}, {0, 2, 1.0}, {1, 0, 1.0}, {1, 1, -1.0}});
    return lp;
}

// min -x1  s.t.  x1 - x2 = 0,  0 <= x1, x2 <= 1. With the bounds in the IPM
// the standard form's b is zero, which HiPO's start guards; Mehrotra's gets a
// positive point from the bound slacks. Optimum x1 = x2 = 1, objective -1.
LpProblem zeroRhsLp() {
    LpProblem lp;
    lp.sense = Sense::Minimize;
    lp.num_cols = 2;
    lp.num_rows = 1;
    lp.obj = {-1.0, 0.0};
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {1.0, 1.0};
    lp.col_type.assign(2, VarType::Continuous);
    lp.col_names.assign(2, "");
    lp.row_lower = {0.0};
    lp.row_upper = {0.0};
    lp.row_names.assign(1, "");
    lp.matrix = SparseMatrix(1, 2, {{0, 0, 1.0}, {0, 1, -1.0}});
    return lp;
}

// min x1  s.t.  x1 - x2 = 0,  x1, x2 >= 0: b = 0 and no bounded column, so
// Mehrotra's least-squares z is zero and its centring falls back to the unit
// start. Optimum x1 = x2 = 0, objective 0.
LpProblem zeroRhsUnboundedLp() {
    LpProblem lp;
    lp.sense = Sense::Minimize;
    lp.num_cols = 2;
    lp.num_rows = 1;
    lp.obj = {1.0, 0.0};
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {kInf, kInf};
    lp.col_type.assign(2, VarType::Continuous);
    lp.col_names.assign(2, "");
    lp.row_lower = {0.0};
    lp.row_upper = {0.0};
    lp.row_names.assign(1, "");
    lp.matrix = SparseMatrix(1, 2, {{0, 0, 1.0}, {0, 1, -1.0}});
    return lp;
}

LpResult solveWith(const LpProblem& lp, const BarrierOptions& opts) {
    BarrierSolver solver;
    solver.setOptions(opts);
    solver.load(lp);
    return solver.solve();
}

}  // namespace

TEST_CASE("Barrier initial point: Auto is Mehrotra's", "[barrier][initial_point]") {
    CHECK(BarrierOptions{}.initial_point == BarrierInitialPoint::Auto);
    const auto algo = GENERATE(BarrierAlgorithm::CpuCholesky, BarrierAlgorithm::CpuSparseCholesky);
    const bool implicit = GENERATE(false, true);
    INFO("algo " << static_cast<int>(algo) << " implicit " << implicit);
    const auto lp = smallLp();
    const auto autostart = solveWith(lp, quietOptions(BarrierInitialPoint::Auto, algo, implicit));
    const auto mehrotra =
        solveWith(lp, quietOptions(BarrierInitialPoint::Mehrotra, algo, implicit));
    REQUIRE(autostart.status == Status::Optimal);
    REQUIRE(mehrotra.status == Status::Optimal);
    // The same start, so the same run.
    CHECK(autostart.iterations == mehrotra.iterations);
    CHECK(autostart.objective == mehrotra.objective);
}

TEST_CASE("Barrier initial point: HiPO's start reaches the optimum", "[barrier][initial_point]") {
    const auto algo = GENERATE(BarrierAlgorithm::CpuCholesky, BarrierAlgorithm::CpuSparseCholesky,
                               BarrierAlgorithm::CpuAugmented);
    const bool implicit = GENERATE(false, true);
    INFO("algo " << static_cast<int>(algo) << " implicit " << implicit);
    const auto mehrotra =
        solveWith(smallLp(), quietOptions(BarrierInitialPoint::Mehrotra, algo, implicit));
    const auto hipo = solveWith(smallLp(), quietOptions(BarrierInitialPoint::Hipo, algo, implicit));
    REQUIRE(hipo.status == Status::Optimal);
    // Crossover is off: the IPM's own point, within its stop at 1e-8 (#253).
    CHECK(std::abs(hipo.objective - (-3.5)) <= 1e-6);
    // Another start, so another run: the option reaches the loop.
    if (mehrotra.status == Status::Optimal) {
        CHECK((hipo.iterations != mehrotra.iterations || hipo.objective != mehrotra.objective));
    }
}

TEST_CASE("Barrier initial point: a zero right-hand side", "[barrier][initial_point]") {
    const auto start = GENERATE(BarrierInitialPoint::Mehrotra, BarrierInitialPoint::Hipo);
    const auto algo = GENERATE(BarrierAlgorithm::CpuCholesky, BarrierAlgorithm::CpuSparseCholesky);
    INFO("start " << static_cast<int>(start) << " algo " << static_cast<int>(algo));
    const auto result = solveWith(zeroRhsLp(), quietOptions(start, algo, true));
    REQUIRE(result.status == Status::Optimal);
    CHECK(std::abs(result.objective - (-1.0)) <= 1e-6);
    // Mehrotra's unit fallback, and HiPO's guard on the same b = 0.
    const auto fallback = solveWith(zeroRhsUnboundedLp(), quietOptions(start, algo, false));
    REQUIRE(fallback.status == Status::Optimal);
    CHECK(std::abs(fallback.objective) <= 1e-6);
}

TEST_CASE("Barrier initial point: Netlib LPs reach their objectives from HiPO's start",
          "[barrier][initial_point][netlib]") {
    using Case = std::pair<const char*, Real>;
    // scfxm2 and vtp-base: fewer iterations than from Mehrotra's (#144).
    auto [name, expected] =
        GENERATE(Case{"adlittle", 2.2549496316e+05}, Case{"afiro", -4.6475314286e+02},
                 Case{"blend", -3.0812149846e+01}, Case{"kb2", -1.7499001299e+03},
                 Case{"sc50a", -6.4575077059e+01}, Case{"sc105", -5.2202061212e+01},
                 Case{"share2b", -4.1573224074e+02}, Case{"bore3d", 1.3730803942e+03},
                 Case{"scfxm2", 3.6660261565e+04}, Case{"vtp-base", 1.2983146246e+05});
    const auto algo = GENERATE(BarrierAlgorithm::CpuCholesky, BarrierAlgorithm::CpuSparseCholesky);
    const bool implicit = GENERATE(false, true);
    INFO(name << " algo " << static_cast<int>(algo) << " implicit " << implicit);
    const std::string path = mipx::test::testDataDir() + "/netlib/" + name + ".mps.gz";
    if (!std::filesystem::exists(path)) {
        SKIP("Netlib instance " + std::string(name) + " not downloaded");
    }
    const auto lp = readMps(path);
    auto opts = quietOptions(BarrierInitialPoint::Hipo, algo, implicit);
    // As in the Gondzio tests: the IPM's stop can leave an original row off
    // by more than the feasibility check once unscaled, and such a point goes
    // to crossover by design.
    opts.crossover = true;
    opts.presolve = true;
    const auto result = solveWith(lp, opts);
    REQUIRE(result.status == Status::Optimal);
    CHECK(std::abs(result.objective - expected) <= 1e-6 * std::max(1.0, std::abs(expected)));
}
