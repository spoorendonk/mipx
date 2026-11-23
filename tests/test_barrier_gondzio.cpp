// Gondzio's multiple centrality correctors in the CPU barrier loop (#293).

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

BarrierOptions quietOptions(Int correctors, BarrierAlgorithm algo, bool implicit) {
    BarrierOptions opts;
    opts.verbose = false;
    opts.crossover = false;
    opts.presolve = false;
    opts.algorithm = algo;
    opts.implicit_upper_bounds = implicit;
    opts.gondzio_correctors = correctors;
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
    Int corrections = 0;
};

Solved solveWith(const LpProblem& lp, const BarrierOptions& opts) {
    BarrierSolver solver;
    solver.setOptions(opts);
    solver.load(lp);
    Solved out;
    out.result = solver.solve();
    out.corrections = solver.gondzioCorrections();
    return out;
}

}  // namespace

TEST_CASE("Barrier Gondzio: off by default", "[barrier][gondzio]") {
    CHECK(BarrierOptions{}.gondzio_correctors == 0);
    const auto off = solveWith(smallLp(), quietOptions(0, BarrierAlgorithm::CpuCholesky, false));
    REQUIRE(off.result.status == Status::Optimal);
    CHECK(off.corrections == 0);
}

TEST_CASE("Barrier Gondzio: small LP reaches the same optimum with correctors",
          "[barrier][gondzio]") {
    // Not CpuAugmented: it fails this LP with correctors off as well.
    const auto algo = GENERATE(BarrierAlgorithm::CpuCholesky, BarrierAlgorithm::CpuSparseCholesky);
    const bool implicit = GENERATE(false, true);
    INFO("algo " << static_cast<int>(algo) << " implicit " << implicit);
    const auto lp = smallLp();
    const auto off = solveWith(lp, quietOptions(0, algo, implicit)).result;
    const auto on = solveWith(lp, quietOptions(4, algo, implicit)).result;
    REQUIRE(off.status == Status::Optimal);
    REQUIRE(on.status == Status::Optimal);
    // Crossover is off, so this is the IPM's own point: its stop on x'z/n at
    // 1e-8 (#253) allows a gap of about n_compl 1e-8 (1 + |obj|), ~2e-7 here.
    CHECK(std::abs(on.objective - (-3.5)) <= 1e-6);
    CHECK(std::abs(on.objective - off.objective) <= 1e-6);
}

TEST_CASE("Barrier Gondzio: Netlib LPs reach the same objectives with correctors",
          "[barrier][gondzio][netlib]") {
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
    const bool implicit = GENERATE(false, true);
    INFO(name << " algo " << static_cast<int>(algo) << " implicit " << implicit);
    const std::string path = mipx::test::testDataDir() + "/netlib/" + name + ".mps.gz";
    if (!std::filesystem::exists(path)) {
        SKIP("Netlib instance " + std::string(name) + " not downloaded");
    }
    const auto lp = readMps(path);
    auto opts = quietOptions(4, algo, implicit);
    // The IPM's stop at primal_dual_tol can leave an original row off by more
    // than the 1e-6 feasibility check once unscaled (recipe, sparse NE with
    // implicit bounds: pinf 5e-10 scaled; fine at 1e-10), and such a point
    // goes to crossover by design. The stopping test is #253's.
    opts.crossover = true;
    const auto [result, corrections] = solveWith(lp, opts);
    REQUIRE(result.status == Status::Optimal);
    CHECK(std::abs(result.objective - expected) <= 1e-6 * std::max(1.0, std::abs(expected)));
    // The correctors ran and some were taken, not all rejected.
    CHECK(corrections > 0);
}
