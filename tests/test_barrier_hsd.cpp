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
#include <limits>
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

// A solve with crossover on (#294): what crossover did after the IPM.
struct Crossed {
    LpResult result;
    bool homogeneous = false;
    std::vector<BasisStatus> basis;
    Int iters = 0;  // crossover's simplex iterations
};

Crossed crossWith(const LpProblem& lp, BarrierOptions opts) {
    opts.crossover = true;
    BarrierSolver solver;
    solver.setOptions(opts);
    solver.load(lp);
    Crossed out;
    out.result = solver.solve();
    out.homogeneous = solver.usedHomogeneous();
    out.basis = solver.getBasis();
    out.iters = solver.crossoverIterations();
    return out;
}

// The basis crossover hands back: one status per column and row, as many
// basic as there are rows.
void checkBasis(const std::vector<BasisStatus>& basis, const LpProblem& lp) {
    REQUIRE(basis.size() == static_cast<size_t>(lp.num_cols + lp.num_rows));
    const auto basic = std::count(basis.begin(), basis.end(), BasisStatus::Basic);
    CHECK(basic == lp.num_rows);
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

namespace {

// x + y >= 3 and x + y <= 1 with x, y >= 0: no feasible point.
LpProblem infeasibleLp() {
    LpProblem lp;
    lp.sense = Sense::Minimize;
    lp.num_cols = 2;
    lp.num_rows = 2;
    lp.obj = {1.0, 1.0};
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {kInf, kInf};
    lp.col_type.assign(2, VarType::Continuous);
    lp.col_names.assign(2, "");
    lp.row_lower = {3.0, -kInf};
    lp.row_upper = {kInf, 1.0};
    lp.row_names.assign(2, "");
    lp.matrix = SparseMatrix(2, 2, {{0, 0, 1.0}, {0, 1, 1.0}, {1, 0, 1.0}, {1, 1, 1.0}});
    return lp;
}

// min -x - y s.t. x - y <= 1, x, y >= 0: feasible, unbounded along (1, 1).
LpProblem unboundedLp() {
    LpProblem lp;
    lp.sense = Sense::Minimize;
    lp.num_cols = 2;
    lp.num_rows = 1;
    lp.obj = {-1.0, -1.0};
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {kInf, kInf};
    lp.col_type.assign(2, VarType::Continuous);
    lp.col_names.assign(2, "");
    lp.row_lower = {-kInf};
    lp.row_upper = {1.0};
    lp.row_names.assign(1, "");
    lp.matrix = SparseMatrix(1, 2, {{0, 0, 1.0}, {0, 1, -1.0}});
    return lp;
}

// min obj'x over two rows lo <= a_i'x <= hi and x >= 0, x in R^2.
LpProblem twoByTwo(std::vector<Real> obj, std::vector<Real> a, std::vector<Real> lo,
                   std::vector<Real> hi) {
    LpProblem lp;
    lp.sense = Sense::Minimize;
    lp.num_cols = 2;
    lp.num_rows = 2;
    lp.obj = std::move(obj);
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {kInf, kInf};
    lp.col_type.assign(2, VarType::Continuous);
    lp.col_names.assign(2, "");
    lp.row_lower = std::move(lo);
    lp.row_upper = std::move(hi);
    lp.row_names.assign(2, "");
    lp.matrix = SparseMatrix(2, 2, {{0, 0, a[0]}, {0, 1, a[1]}, {1, 0, a[2]}, {1, 1, a[3]}});
    return lp;
}

// n columns x >= 0 and n rows lo <= a_i'x <= hi, row 0 first_lo <= a_0'x <= first_hi.
LpProblem chainLp(Index n, std::vector<Real> obj, std::vector<Triplet> entries, Real lo, Real hi,
                  Real first_lo, Real first_hi) {
    LpProblem lp;
    lp.sense = Sense::Minimize;
    lp.num_cols = n;
    lp.num_rows = n;
    lp.obj = std::move(obj);
    lp.col_lower.assign(static_cast<size_t>(n), 0.0);
    lp.col_upper.assign(static_cast<size_t>(n), kInf);
    lp.col_type.assign(static_cast<size_t>(n), VarType::Continuous);
    lp.col_names.assign(static_cast<size_t>(n), "");
    lp.row_lower.assign(static_cast<size_t>(n), lo);
    lp.row_upper.assign(static_cast<size_t>(n), hi);
    lp.row_lower.front() = first_lo;
    lp.row_upper.front() = first_hi;
    lp.row_names.assign(static_cast<size_t>(n), "");
    lp.matrix = SparseMatrix(n, n, std::move(entries));
    return lp;
}

// min sum x s.t. x_0 + x_1 >= 3, x_j - 2 x_{j-1} >= 0 (0 < j < n): b and c
// of size 1, optimum x_j = 2^j, objective 2^n - 1.
LpProblem chainPrimal(Index n) {
    std::vector<Triplet> e{{0, 0, 1.0}, {0, 1, 1.0}};
    for (Index j = 1; j < n; ++j) {
        e.push_back({j, j, 1.0});
        e.push_back({j, j - 1, -2.0});
    }
    return chainLp(n, std::vector<Real>(static_cast<size_t>(n), 1.0), e, 0.0, kInf, 3.0, kInf);
}

// Its dual mirror: min -x_0 - x_1 s.t. x_{n-2} + x_{n-1} <= 3,
// x_{j-1} - 2 x_j <= 0 (0 < j < n): optimum x_{n-1} = 1, x_j = 2^(n-1-j),
// objective -3 2^(n-2).
LpProblem chainDual(Index n) {
    std::vector<Triplet> e{{0, n - 2, 1.0}, {0, n - 1, 1.0}};
    for (Index j = 1; j < n; ++j) {
        e.push_back({j, j - 1, 1.0});
        e.push_back({j, j, -2.0});
    }
    std::vector<Real> obj(static_cast<size_t>(n), 0.0);
    obj[0] = -1.0;
    obj[1] = -1.0;
    return chainLp(n, std::move(obj), e, -kInf, 0.0, -kInf, 3.0);
}

// min -x with x >= 0 and no rows: unbounded.
LpProblem noRowsLp() {
    LpProblem lp;
    lp.sense = Sense::Minimize;
    lp.num_cols = 1;
    lp.num_rows = 0;
    lp.obj = {-1.0};
    lp.col_lower = {0.0};
    lp.col_upper = {kInf};
    lp.col_type.assign(1, VarType::Continuous);
    lp.col_names.assign(1, "");
    lp.matrix = SparseMatrix(0, 1);
    return lp;
}

}  // namespace

TEST_CASE("Barrier HSD: feasible LPs with large b or c are not read as infeasible",
          "[barrier][hsd]") {
    // The rays' bounds are relative to |b| and |c|: unnormalized, the first
    // ended Infeasible at tau ~ 5e-9, the second Unbounded at iteration 1.
    using Case = std::pair<LpProblem, Real>;
    auto [lp, expected] = GENERATE(
        // min x + y, x + 2y >= 1e9, 3x + y >= 5e8: optimum (0, 5e8).
        Case{twoByTwo({1.0, 1.0}, {1.0, 2.0, 3.0, 1.0}, {1e9, 5e8}, {kInf, kInf}), 5e8},
        // min -1e9 (x + y), x + 2y <= 1, 3x + y <= 0.5: optimum (0, 0.5).
        Case{twoByTwo({-1e9, -1e9}, {1.0, 2.0, 3.0, 1.0}, {-kInf, -kInf}, {1.0, 0.5}), -5e8});
    const auto algo = GENERATE(BarrierAlgorithm::CpuCholesky, BarrierAlgorithm::CpuSparseCholesky);
    INFO("expected " << expected << " algo " << static_cast<int>(algo));
    const auto [result, primals, homogeneous] = solveWith(lp, quietOptions(true, algo));
    CHECK(homogeneous);
    REQUIRE(result.status == Status::Optimal);
    CHECK(std::abs(result.objective - expected) <= 1e-6 * std::abs(expected));
}

TEST_CASE("Barrier HSD: feasible LPs with solutions far larger than b and c give no certificate",
          "[barrier][hsd]") {
    // b and c of size 1, solutions of size 2^30: on the rays alone they ended
    // Infeasible and Unbounded at iteration 12, mid-run, at tau ~ mu ~ 1e-7;
    // the mu and tau gate holds the verdict back. Whether the loop then
    // reaches 1e-8 on them depends on the row order (one stalls at dinf ~ 2e-8
    // on the sparse factor); the certificate is what this tests.
    using Case = std::pair<LpProblem, Real>;
    auto [lp, expected] = GENERATE(Case{chainPrimal(30), std::ldexp(1.0, 30) - 1.0},
                                   Case{chainDual(30), -3.0 * std::ldexp(1.0, 28)});
    const auto algo = GENERATE(BarrierAlgorithm::CpuCholesky, BarrierAlgorithm::CpuSparseCholesky);
    INFO("expected " << expected << " algo " << static_cast<int>(algo));
    const auto [result, primals, homogeneous] = solveWith(lp, quietOptions(true, algo));
    CHECK(homogeneous);
    CHECK(result.status != Status::Infeasible);
    CHECK(result.status != Status::Unbounded);
    if (result.status == Status::Optimal) {
        CHECK(std::abs(result.objective - expected) <= 1e-6 * std::abs(expected));
    }
}

TEST_CASE("Barrier HSD: an infeasible LP ends Infeasible, an unbounded one Unbounded",
          "[barrier][hsd]") {
    const auto algo = GENERATE(BarrierAlgorithm::CpuCholesky, BarrierAlgorithm::CpuSparseCholesky);
    const bool crossover = GENERATE(false, true);
    INFO("algo " << static_cast<int>(algo) << " crossover " << crossover);
    auto opts = quietOptions(true, algo);
    opts.crossover = crossover;
    const auto infeasible = solveWith(infeasibleLp(), opts);
    CHECK(infeasible.homogeneous);
    CHECK(infeasible.result.status == Status::Infeasible);
    const auto unbounded = solveWith(unboundedLp(), opts);
    CHECK(unbounded.homogeneous);
    CHECK(unbounded.result.status == Status::Unbounded);
    // Infeasible through its upper bounds, which the HSD loop's standard form
    // carries as rows: x + y >= 5 with x, y <= 2.
    const auto bounded = solveWith(readLp(mipx::test::testDataDir() + "/infeasible_lp.lp"), opts);
    CHECK(bounded.homogeneous);
    CHECK(bounded.result.status == Status::Infeasible);
    // Infeasible both ways (x - y >= d, y - x >= d, min -x - y): Infeasible,
    // whichever ray the loop meets first. With d = 1e-3 the primal ray is
    // weak and the dual one came first, reading Unbounded before the
    // zero-objective check.
    for (const Real d : {1.0, 1e-3}) {
        INFO("d " << d);
        const auto both =
            solveWith(twoByTwo({-1.0, -1.0}, {1.0, -1.0, -1.0, 1.0}, {d, d}, {kInf, kInf}), opts);
        CHECK(both.homogeneous);
        CHECK(both.result.status == Status::Infeasible);
    }
    // No rows in the standard form: the m = 0 path's dual ray.
    CHECK(solveWith(noRowsLp(), opts).result.status == Status::Unbounded);
    // A feasible bounded LP still ends Optimal.
    CHECK(solveWith(smallLp(), opts).result.status == Status::Optimal);
    if (crossover) {
        // A verdict is reported as it stands, crossover never runs (#294).
        for (const auto& lp : {infeasibleLp(), unboundedLp()}) {
            const auto crossed = crossWith(lp, opts);
            CHECK(crossed.homogeneous);
            CHECK(crossed.iters == 0);
            CHECK(crossed.basis.empty());
        }
        // An optimum: the basis guessed at the IPM's point is the optimal
        // one, no pivot (a crash-basis start takes one here).
        const auto lp = smallLp();
        const auto crossed = crossWith(lp, opts);
        REQUIRE(crossed.result.status == Status::Optimal);
        CHECK(crossed.iters == 0);
        checkBasis(crossed.basis, lp);
    }
}

TEST_CASE("Barrier HSD: infeasibility test reads the Farkas rays", "[barrier][hsd]") {
    // One row, two columns: az = A z, at_y = A'y.
    const std::vector<Real> b{1.0};
    const std::vector<Real> c{1.0, 1.0};
    const Real tol = 1e-8;
    const Real tau = 1e-9;  // <= tol kappa at kappa = 1
    const Real mu = 1e-9;   // <= tol mu0
    // A'y + s = (1e-12, 0): a primal ray for b'y > 0.
    const std::vector<Real> at_y{-1.0, -2.0};
    const std::vector<Real> s{1.0 + 1e-12, 2.0};
    const std::vector<Real> az_far{1e-3};
    const std::vector<Real> az_ray{0.0};
    // b'y = 1 > 0 and |A'y + s| = 1e-12 <= 1e-8: primal infeasible.
    CHECK(hsdInfeasibility(b, c, az_far, at_y, s, tau, 1.0, mu, 0.0, 1.0, tol) ==
          BarrierIpmOutcome::PrimalInfeasible);
    // No verdict before the run gets there: tau above tol kappa, or mu above
    // tol mu0, whatever the rays.
    CHECK(hsdInfeasibility(b, c, az_ray, at_y, s, 1e-3, 1.0, mu, -1.0, 1.0, tol) ==
          BarrierIpmOutcome::Failed);
    CHECK(hsdInfeasibility(b, c, az_ray, at_y, s, tau, 1.0, 1e-3, -1.0, 1.0, tol) ==
          BarrierIpmOutcome::Failed);
    // b'y <= 0: no primal ray; c'z = -1 < 0 but |A z| = 1e-3 > 1e-8: none.
    CHECK(hsdInfeasibility(b, c, az_far, at_y, s, tau, 1.0, mu, -1.0, 0.0, tol) ==
          BarrierIpmOutcome::Failed);
    // A z = 0: a dual ray.
    CHECK(hsdInfeasibility(b, c, az_ray, at_y, s, tau, 1.0, mu, -1.0, 0.0, tol) ==
          BarrierIpmOutcome::DualInfeasible);
    // |A z| = 1e-12 meets tol (-c'z) with |c| = 1, not tol (-c'z) / |c| with
    // |c| = 1e9: the bound is relative to the data.
    const std::vector<Real> az_near{1e-12};
    CHECK(hsdInfeasibility(b, c, az_near, at_y, s, tau, 1.0, mu, -1.0, 0.0, tol) ==
          BarrierIpmOutcome::DualInfeasible);
    const std::vector<Real> c_big{1e9, 1e9};
    CHECK(hsdInfeasibility(b, c_big, az_near, at_y, s, tau, 1.0, mu, -1.0, 0.0, tol) ==
          BarrierIpmOutcome::Failed);
    // Both: primal first.
    CHECK(hsdInfeasibility(b, c, az_ray, at_y, s, tau, 1.0, mu, -1.0, 1.0, tol) ==
          BarrierIpmOutcome::PrimalInfeasible);
    // A NaN gives no verdict: in the ray, or in b'y.
    const Real nan = std::numeric_limits<Real>::quiet_NaN();
    const std::vector<Real> s_nan{nan, 2.0};
    CHECK(hsdInfeasibility(b, c, az_far, at_y, s_nan, tau, 1.0, mu, 0.0, 1.0, tol) ==
          BarrierIpmOutcome::Failed);
    const std::vector<Real> az_nan{nan};
    CHECK(hsdInfeasibility(b, c, az_nan, at_y, s, tau, 1.0, mu, -1.0, 0.0, tol) ==
          BarrierIpmOutcome::Failed);
    CHECK(hsdInfeasibility(b, c, az_ray, at_y, s, tau, 1.0, mu, -1.0, nan, tol) ==
          BarrierIpmOutcome::Failed);
}

TEST_CASE("Barrier HSD: implicit upper bounds fall back to Mehrotra's loop", "[barrier][hsd]") {
    auto opts = quietOptions(true, BarrierAlgorithm::CpuCholesky);
    opts.implicit_upper_bounds = true;
    const auto [result, primals, homogeneous] = solveWith(smallLp(), opts);
    CHECK_FALSE(homogeneous);
    REQUIRE(result.status == Status::Optimal);
    CHECK(std::abs(result.objective - (-3.5)) <= 1e-6);
}

// Gondzio's correctors in the homogeneous loop (#335): the same optimum
// as without them, with correctors taken.
TEST_CASE("Barrier HSD: Gondzio correctors run in the homogeneous loop", "[barrier][hsd]") {
    const auto algo = GENERATE(BarrierAlgorithm::CpuCholesky, BarrierAlgorithm::CpuSparseCholesky);
    INFO("algo " << static_cast<int>(algo));
    auto opts = quietOptions(true, algo);
    opts.gondzio_correctors = 4;
    BarrierSolver solver;
    solver.setOptions(opts);
    solver.load(smallLp());
    const auto result = solver.solve();
    CHECK(solver.usedHomogeneous());
    REQUIRE(result.status == Status::Optimal);
    CHECK(std::abs(result.objective - (-3.5)) <= 1e-6);
    CHECK(solver.gondzioCorrections() > 0);
    // An infeasible LP still ends on its certificate.
    CHECK(solveWith(infeasibleLp(), opts).result.status == Status::Infeasible);
    CHECK(solveWith(unboundedLp(), opts).result.status == Status::Unbounded);
}

// stair and scfxm2 fail with Gondzio's correctors in Mehrotra's loop, from
// a primal drift that predates them (#293, decision #397); the homogeneous
// loop with them reaches the optimum (#335).
TEST_CASE("Barrier HSD: Gondzio correctors solve stair and scfxm2", "[barrier][hsd][netlib]") {
    using Case = std::pair<const char*, Real>;
    auto [name, expected] =
        GENERATE(Case{"stair", -2.5126695119e+02}, Case{"scfxm2", 3.6660261565e+04});
    INFO(name);
    const std::string path = mipx::test::testDataDir() + "/netlib/" + name + ".mps.gz";
    if (!std::filesystem::exists(path)) {
        SKIP("Netlib instance " + std::string(name) + " not downloaded");
    }
    auto opts = quietOptions(true, BarrierAlgorithm::CpuSparseCholesky);
    opts.presolve = true;
    opts.gondzio_correctors = 4;
    BarrierSolver solver;
    solver.setOptions(opts);
    solver.load(readMps(path));
    const auto result = solver.solve();
    CHECK(solver.usedHomogeneous());
    CHECK(solver.gondzioCorrections() > 0);
    REQUIRE(result.status == Status::Optimal);
    CHECK(std::abs(result.objective - expected) <= 1e-6 * std::abs(expected));
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
    const Int gondzio = GENERATE(0, 4);  // correctors off and on (#335)
    INFO(name << " algo " << static_cast<int>(algo) << " gondzio " << gondzio);
    const std::string path = mipx::test::testDataDir() + "/netlib/" + name + ".mps.gz";
    if (!std::filesystem::exists(path)) {
        SKIP("Netlib instance " + std::string(name) + " not downloaded");
    }
    const auto lp = readMps(path);
    auto opts = quietOptions(true, algo);
    opts.presolve = true;
    opts.gondzio_correctors = gondzio;
    const auto [result, primals, homogeneous] = solveWith(lp, opts);
    CHECK(homogeneous);
    REQUIRE(result.status == Status::Optimal);
    CHECK(std::abs(result.objective - expected) <= 1e-6 * std::max(1.0, std::abs(expected)));
}

// Crossover from the homogeneous loop's iterate (#294): the LP point
// (x, y, s) / tau it hands over reaches the same objective and a basis.
// Not a test of the point itself: on most of these LPs crossover's basis
// guess is singular, from either loop, and the simplex restarts from a
// crash basis (#398).
TEST_CASE("Barrier HSD: Netlib LPs cross over to a basis", "[barrier][hsd][netlib][crossover]") {
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
    const auto hsd = crossWith(lp, opts);
    CHECK(hsd.homogeneous);
    REQUIRE(hsd.result.status == Status::Optimal);
    CHECK(std::abs(hsd.result.objective - expected) <= 1e-6 * std::max(1.0, std::abs(expected)));
    checkBasis(hsd.basis, lp);
}

// A homogeneous run that stops short of tolerance hands its best iterate to
// crossover as NearOptimal, as the infeasible-start loop does (#294). A
// tolerance of 0 is never met, so the run ends on its stall test (max_iter
// 60) or on the iteration limit (max_iter 12, past the 1e-6 hand-off);
// without crossover that is an error, with it an optimum.
TEST_CASE("Barrier HSD: a stop short of tolerance seeds crossover", "[barrier][hsd][crossover]") {
    const auto algo = GENERATE(BarrierAlgorithm::CpuCholesky, BarrierAlgorithm::CpuSparseCholesky);
    const Int max_iter = GENERATE(12, 60);
    INFO("algo " << static_cast<int>(algo) << " max_iter " << max_iter);
    const auto lp = smallLp();
    auto opts = quietOptions(true, algo);
    opts.primal_dual_tol = 0.0;
    opts.max_iter = max_iter;
    BarrierSolver solver;
    solver.setOptions(opts);
    solver.load(lp);
    const auto off = solver.solve();
    CHECK(solver.usedHomogeneous());
    CHECK(off.status == Status::Error);
    CHECK(solver.lastError() == "Barrier stopped short of tolerance and crossover is disabled.");
    // Which of the two stops ended the run.
    if (max_iter == 60) {
        CHECK(off.iterations < max_iter);
    } else {
        CHECK(off.iterations == max_iter);
    }

    const auto on = crossWith(lp, opts);
    CHECK(on.homogeneous);
    REQUIRE(on.result.status == Status::Optimal);
    CHECK(std::abs(on.result.objective - (-3.5)) <= 1e-9);
    // The best iterate is the optimum to 1e-6, so the basis guessed at it is
    // the optimal one: no pivot (a crash-basis start takes one here).
    CHECK(on.iters == 0);
    checkBasis(on.basis, lp);
}
