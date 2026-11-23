// The CPU barrier on the homogeneous self-dual model (#292, #370).

#include "lp/newton_solver.h"
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

BarrierOptions quietOptions(bool homogeneous, BarrierAlgorithm algo) {
    BarrierOptions opts;
    opts.verbose = false;
    opts.crossover = false;
    opts.presolve = false;
    opts.algorithm = algo;
    opts.homogeneous = homogeneous;
    return opts;
}

// min -x - 2y + z  s.t.  x + y + z <= 3,  x - y >= -1,  0 <= x <= 2,
// 0 <= y <= 1.5, z >= 0.5. Optimum x = 1, y = 1.5, z = 0.5 (the first row
// tight, y at its upper bound), objective -3.5.
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

struct Solved {
    LpResult result;
    std::vector<Real> primals;
    bool homogeneous = false;
};

Solved solveWith(const LpProblem& lp, const BarrierOptions& opts) {
    BarrierSolver solver;
    solver.setOptions(opts);
    solver.load(lp);
    Solved out;
    out.result = solver.solve();
    out.primals = solver.getPrimalValues();
    out.homogeneous = solver.usedHomogeneous();
    return out;
}

}  // namespace

TEST_CASE("Barrier HSD: off by default", "[barrier][hsd]") {
    CHECK_FALSE(BarrierOptions{}.homogeneous);
    const auto off = solveWith(smallLp(), quietOptions(false, BarrierAlgorithm::CpuCholesky));
    REQUIRE(off.result.status == Status::Optimal);
    CHECK_FALSE(off.homogeneous);
}

TEST_CASE("Barrier HSD: dtau denominator is bounded below by u_z' S Z^-1 u_z", "[barrier][hsd]") {
    // q = 2 * 1^2 / 1 + 1 * (-1)^2 / 2 = 2.5 and kappa / tau = 0.5. With c = 0
    // and b = 1 the computed b'u_y - c'u_z is u_y.
    const std::vector<Real> b{1.0};
    const std::vector<Real> c{0.0, 0.0};
    const std::vector<Real> uz{1.0, -1.0};
    const std::vector<Real> z{1.0, 2.0};
    const std::vector<Real> s{2.0, 1.0};
    const Real tau = 0.5;
    const Real kappa = 0.25;
    auto denom = [&](Real uy) {
        const std::vector<Real> uyv{uy};
        return hsdGapDenominator(b, c, uz, uyv, z, s, tau, kappa);
    };
    // An exact solve: the computed value is q.
    CHECK(std::abs(denom(2.5) - 3.0) <= 1e-15);
    // Regularization adds u_y'(reg I) u_y: the computed value is kept.
    CHECK(std::abs(denom(4.0) - 4.5) <= 1e-15);
    // Rounding takes it below q, or below zero (blend, dense NE): q bounds it.
    CHECK(std::abs(denom(1.0) - 3.0) <= 1e-15);
    CHECK(std::abs(denom(-7.0) - 3.0) <= 1e-15);
    // c enters with a minus sign: c'u_z = 1 * 1 + 3 * (-1) = -2, so 4 + 2 = 6.
    const std::vector<Real> c2{1.0, 3.0};
    const std::vector<Real> uy4{4.0};
    CHECK(std::abs(hsdGapDenominator(b, c2, uz, uy4, z, s, tau, kappa) - 6.5) <= 1e-15);
}

TEST_CASE("Barrier HSD: small LP reaches its optimum on the IPM's own point", "[barrier][hsd]") {
    const auto algo = GENERATE(BarrierAlgorithm::CpuCholesky, BarrierAlgorithm::CpuSparseCholesky);
    INFO("algo " << static_cast<int>(algo));
    // Crossover is off: the objective is that of the HSD loop's x = z / tau.
    const auto [result, primals, homogeneous] = solveWith(smallLp(), quietOptions(true, algo));
    CHECK(homogeneous);
    REQUIRE(result.status == Status::Optimal);
    CHECK(std::abs(result.objective - (-3.5)) <= 1e-6);
    REQUIRE(primals.size() == 3);
    CHECK(std::abs(primals[0] - 1.0) <= 1e-6);
    CHECK(std::abs(primals[1] - 1.5) <= 1e-6);
    CHECK(std::abs(primals[2] - 0.5) <= 1e-6);
}

TEST_CASE("Barrier HSD: implicit upper bounds or Gondzio fall back to Mehrotra's loop",
          "[barrier][hsd]") {
    const bool implicit = GENERATE(false, true);
    INFO("implicit " << implicit);
    auto opts = quietOptions(true, BarrierAlgorithm::CpuCholesky);
    opts.implicit_upper_bounds = implicit;
    opts.gondzio_correctors = implicit ? 0 : 4;
    const auto [result, primals, homogeneous] = solveWith(smallLp(), opts);
    CHECK_FALSE(homogeneous);
    REQUIRE(result.status == Status::Optimal);
    CHECK(std::abs(result.objective - (-3.5)) <= 1e-6);
}

TEST_CASE("Barrier HSD: Netlib LPs reach the same objectives", "[barrier][hsd][netlib]") {
    using Case = std::pair<const char*, Real>;
    auto [name, expected] =
        GENERATE(Case{"adlittle", 2.2549496316e+05}, Case{"afiro", -4.6475314286e+02},
                 Case{"blend", -3.0812149846e+01}, Case{"kb2", -1.7499001299e+03},
                 Case{"sc50a", -6.4575077059e+01}, Case{"sc50b", -7.0000000000e+01},
                 Case{"sc105", -5.2202061212e+01}, Case{"sc205", -5.2202061212e+01},
                 Case{"share1b", -7.6589318579e+04}, Case{"share2b", -4.1573224074e+02},
                 Case{"stocfor1", -4.1131976219e+04}, Case{"bore3d", 1.3730803942e+03},
                 Case{"recipe", -2.6661600000e+02});
    const auto algo = GENERATE(BarrierAlgorithm::CpuCholesky, BarrierAlgorithm::CpuSparseCholesky);
    INFO(name << " algo " << static_cast<int>(algo));
    const std::string path = mipx::test::testDataDir() + "/netlib/" + name + ".mps.gz";
    if (!std::filesystem::exists(path)) {
        SKIP("Netlib instance " + std::string(name) + " not downloaded");
    }
    const auto lp = readMps(path);
    auto opts = quietOptions(true, algo);
    opts.presolve = true;
    const auto [result, primals, homogeneous] = solveWith(lp, opts);
    CHECK(homogeneous);
    REQUIRE(result.status == Status::Optimal);
    CHECK(std::abs(result.objective - expected) <= 1e-6 * std::max(1.0, std::abs(expected)));
}
