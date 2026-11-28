// The CPU barrier on the homogeneous self-dual model (#292, #370).

#include "lp/newton_solver.h"
#include "mipx/barrier.h"
#include "mipx/dual_simplex.h"
#include "mipx/io.h"
#include "test_paths.hpp"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cmath>
#include <filesystem>
#include <limits>
#include <memory>
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
    BarrierSolver::CrossoverStats stats;
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
    out.stats = solver.crossoverStats();
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

TEST_CASE("Barrier HSD: a direction solves all five rows of the HSD Newton system",
          "[barrier][hsd][gondzio]") {
    // From the backend's two solves, hsdDirection must give a d that solves
    //   A dz - b dtau = eta rp,  A'dy + ds - c dtau = eta rd,
    //   -c'dz + b'dy - dkappa = eta rg,  S dz + Z ds = rc,  kappa dtau + tau dkappa = rk:
    // Mehrotra's direction (eta != 0) and Gondzio's corrector (#335), which is
    // the same system for (0, 0, 0, t_c, t_k).
    constexpr Index m = 2;
    constexpr Index n = 3;
    const SparseMatrix A(m, n, {{0, 0, 1.0}, {0, 1, 1.0}, {1, 1, 1.0}, {1, 2, 2.0}});
    const std::vector<Real> b{1.0, 2.0};
    const std::vector<Real> c{1.0, -1.0, 0.5};
    const std::vector<Real> z{0.7, 1.3, 0.4};
    const std::vector<Real> s{0.9, 0.2, 1.6};
    const Real tau = 0.8;
    const Real kappa = 0.3;
    const std::vector<Real> rc{0.25, -0.5, 0.125};
    const Real rk = -0.375;

    const bool corrector = GENERATE(false, true);
    const int backend = GENERATE(0, 1, 2);
    INFO("corrector " << corrector << ", backend " << backend);
    // eta times rp, rd, rg: zero for the corrector.
    const Real eta = corrector ? 0.0 : 0.6;
    std::vector<Real> rp{0.3, -0.2};
    std::vector<Real> rd{-0.1, 0.4, 0.2};
    const Real rg = 0.15;
    for (Real& v : rp) {
        v *= eta;
    }
    for (Real& v : rd) {
        v *= eta;
    }

    std::unique_ptr<NewtonSolver> solver = backend == 0   ? createCpuCholeskySolver()
                                           : backend == 1 ? createCpuAugmentedSolver()
                                                          : createCpuSparseCholeskySolver();
    BarrierOptions opts;
    opts.verbose = false;
    REQUIRE(solver->setup(A, m, n, opts));
    REQUIRE(solver->factorize(z, s, 1e-12));
    const std::vector<Real> zero_n(n, 0.0);
    std::vector<Real> uz(n), uy(m), us(n), pz(n), py(m), ps(n);
    REQUIRE(solver->solveNewton(b, c, zero_n, uz, uy, us));
    REQUIRE(solver->solveNewton(rp, rd, rc, pz, py, ps));
    const Real denom = hsdGapDenominator(b, c, uz, uy, z, s, tau, kappa);
    REQUIRE(denom > 0.0);

    std::vector<Real> dz(n), dy(m), ds(n);
    Real dtau = 0.0;
    Real dkappa = 0.0;
    REQUIRE(hsdDirection(b, c, pz, py, ps, uz, uy, us, tau, kappa, denom, eta * rg, rk, dz, dy, ds,
                         dtau, dkappa));
    CHECK(std::abs(dtau) > 1e-3);  // the u terms are exercised

    constexpr Real kTol = 1e-8;
    std::vector<Real> adz(m, 0.0), atdy(n, 0.0);
    A.multiply(dz, adz);
    A.multiplyTranspose(dy, atdy);
    for (Index i = 0; i < m; ++i) {
        CHECK(std::abs(adz[i] - b[i] * dtau - rp[i]) <= kTol);
    }
    Real ctdz = 0.0;
    Real btdy = 0.0;
    for (Index j = 0; j < n; ++j) {
        CHECK(std::abs(atdy[j] + ds[j] - c[j] * dtau - rd[j]) <= kTol);
        CHECK(std::abs(s[j] * dz[j] + z[j] * ds[j] - rc[j]) <= kTol);
        ctdz += c[j] * dz[j];
    }
    for (Index i = 0; i < m; ++i) {
        btdy += b[i] * dy[i];
    }
    CHECK(std::abs(-ctdz + btdy - dkappa - eta * rg) <= kTol);
    CHECK(std::abs(kappa * dtau + tau * dkappa - rk) <= kTol);

    // A non-finite entry of d is reported (#335 review), also one that leaves
    // dtau and dkappa finite: p_s enters neither.
    std::vector<Real> bad_ps = ps;
    bad_ps[1] = std::numeric_limits<Real>::quiet_NaN();
    CHECK_FALSE(hsdDirection(b, c, pz, py, bad_ps, uz, uy, us, tau, kappa, denom, eta * rg, rk, dz,
                             dy, ds, dtau, dkappa));
    CHECK(std::isfinite(dtau));
}

TEST_CASE("Barrier HSD: bounded direction solves the upper rows and gap row", "[barrier][hsd]") {
    // One boxed column with A = [1]. The two solves for fixed dtau have
    // dz determined by the equality and dv by the upper row, so we can
    // compute p and the unit-tau direction directly.
    const std::vector<Real> b{1.25};
    const std::vector<Real> c{2.0};
    const std::vector<Real> z{1.0};
    const std::vector<Real> s{0.5};
    const Real upper = 3.25;
    const Real v = 2.0;
    const Real w = 1.5;
    const Real tau = 0.8;
    const Real kappa = 0.3;
    const Real eta = GENERATE(0.0, 0.6);
    const Real rp = 0.3;
    const Real rd = -0.1;
    const Real ru = 0.7;
    const Real rg = 0.15;
    const Real rc = 0.25;
    const Real rw = -0.2;
    const Real rk = -0.375;

    const std::vector<Real> uz{b[0]};
    const Real uv = upper - uz[0];
    const std::vector<Real> us{-s[0] * uz[0] / z[0]};
    const Real uw = -w * uv / v;
    const std::vector<Real> uy{c[0] - us[0] + uw};

    const std::vector<Real> pz{eta * rp};
    const Real pv = eta * ru - pz[0];
    const std::vector<Real> ps{(rc - s[0] * pz[0]) / z[0]};
    const Real pw = (rw - w * pv) / v;
    const std::vector<Real> py{eta * rd - ps[0] + pw};

    const auto denominator = [&](std::span<const Real> dual) {
        return hsdGapDenominator(b, c, uz, dual, z, s, tau, kappa, upper * uw, w * uv * uv / v);
    };
    constexpr Real kTol = 1e-12;
    const Real denom = denominator(uy);
    CHECK(std::abs(denom - 4.15625) <= kTol);
    // Force each side of the denominator's max to win to check both upper terms.
    const std::vector<Real> regularized_y{uy[0] + 1.0};
    const std::vector<Real> rounded_y{uy[0] - 1.0};
    CHECK(std::abs(denominator(regularized_y) - (denom + b[0])) <= kTol);
    CHECK(std::abs(denominator(rounded_y) - denom) <= kTol);
    std::vector<Real> dz(1), dy(1), ds(1);
    Real dtau = 0.0;
    Real dkappa = 0.0;
    REQUIRE(hsdDirection(b, c, pz, py, ps, uz, uy, us, tau, kappa, denom, eta * rg, rk, dz, dy, ds,
                         dtau, dkappa, upper * pw));
    REQUIRE(std::abs(dtau) > 1e-3);
    const Real dv = pv + dtau * uv;
    const Real dw = pw + dtau * uw;
    CHECK(std::abs(dz[0] - b[0] * dtau - eta * rp) <= kTol);
    CHECK(std::abs(dy[0] + ds[0] - dw - c[0] * dtau - eta * rd) <= kTol);
    CHECK(std::abs(dz[0] + dv - upper * dtau - eta * ru) <= kTol);
    CHECK(std::abs(s[0] * dz[0] + z[0] * ds[0] - rc) <= kTol);
    CHECK(std::abs(w * dv + v * dw - rw) <= kTol);
    CHECK(std::abs(-c[0] * dz[0] + b[0] * dy[0] - upper * dw - dkappa - eta * rg) <= kTol);
    CHECK(std::abs(kappa * dtau + tau * dkappa - rk) <= kTol);
}

TEST_CASE("Barrier HSD: small LP reaches its optimum on the IPM's own point", "[barrier][hsd]") {
    const auto algo = GENERATE(BarrierAlgorithm::CpuCholesky, BarrierAlgorithm::CpuSparseCholesky);
    const bool implicit = GENERATE(false, true);
    INFO("algo " << static_cast<int>(algo) << " implicit " << implicit);
    auto opts = quietOptions(true, algo);
    opts.implicit_upper_bounds = implicit;
    BarrierSolver solver;
    solver.setOptions(opts);
    solver.load(smallLp());
    CHECK(solver.standardFormRows() == (implicit ? 2 : 4));
    // Crossover is off: the objective is that of the HSD loop's x = z / tau.
    const auto result = solver.solve();
    CHECK(solver.usedHomogeneous());
    REQUIRE(result.status == Status::Optimal);
    CHECK(std::abs(result.objective - (-3.5)) <= 1e-6);
    const auto primals = solver.getPrimalValues();
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
    const bool implicit = GENERATE(false, true);
    INFO("algo " << static_cast<int>(algo) << " crossover " << crossover << " implicit "
                 << implicit);
    auto opts = quietOptions(true, algo);
    opts.crossover = crossover;
    opts.implicit_upper_bounds = implicit;
    const auto infeasible = solveWith(infeasibleLp(), opts);
    CHECK(infeasible.homogeneous);
    CHECK(infeasible.result.status == Status::Infeasible);
    const auto unbounded = solveWith(unboundedLp(), opts);
    CHECK(unbounded.homogeneous);
    CHECK(unbounded.result.status == Status::Unbounded);
    // Infeasible through its upper bounds: x + y >= 5 with x, y <= 2.
    const auto bounded = solveWith(readLp(mipx::test::testDataDir() + "/infeasible_lp.lp"), opts);
    CHECK(bounded.homogeneous);
    CHECK(bounded.result.status == Status::Infeasible);
    // The dual ray can still move y, but its boxed x component must vanish.
    auto mixed = unboundedLp();
    mixed.col_upper[0] = 2.0;
    const auto mixed_result = solveWith(mixed, opts);
    CHECK(mixed_result.homogeneous);
    CHECK(mixed_result.result.status == Status::Unbounded);
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
    // A'y <= 0: a primal ray for b'y > 0 with slack chosen from A'y.
    const std::vector<Real> at_y{-1.0, -2.0};
    const std::vector<Real> az_far{1e-3};
    const std::vector<Real> az_ray{0.0};
    // The scaled bound is 1e-17, below the IPM's 1e-12 slack floor.
    const std::vector<Real> b_big{1e9};
    const std::vector<Real> at_y_on_boundary{-1.0, 0.0};
    CHECK(hsdInfeasibility(b_big, c, az_far, at_y_on_boundary, tau, 1.0, mu, 0.0, 1.0, tol) ==
          BarrierIpmOutcome::PrimalInfeasible);
    const std::vector<Real> at_y_violated{-1.0, 1e-12};
    CHECK(hsdInfeasibility(b_big, c, az_far, at_y_violated, tau, 1.0, mu, 0.0, 1.0, tol) ==
          BarrierIpmOutcome::Failed);
    CHECK(hsdInfeasibility(b, c, az_far, at_y_violated, tau, 1.0, mu, 0.0, 1.0, tol) ==
          BarrierIpmOutcome::PrimalInfeasible);
    CHECK(hsdInfeasibility(b, c, az_far, at_y_violated, tau, 1.0, mu, 0.0, 1.0, tol, false, 1e9) ==
          BarrierIpmOutcome::Failed);
    // No verdict before the run gets there: tau above tol kappa, or mu above
    // tol mu0, whatever the rays.
    CHECK(hsdInfeasibility(b, c, az_ray, at_y, 1e-3, 1.0, mu, -1.0, 1.0, tol) ==
          BarrierIpmOutcome::Failed);
    CHECK(hsdInfeasibility(b, c, az_ray, at_y, tau, 1.0, 1e-3, -1.0, 1.0, tol) ==
          BarrierIpmOutcome::Failed);
    // b'y <= 0: no primal ray; c'z = -1 < 0 but |A z| = 1e-3 > 1e-8: none.
    CHECK(hsdInfeasibility(b, c, az_far, at_y, tau, 1.0, mu, -1.0, 0.0, tol) ==
          BarrierIpmOutcome::Failed);
    // A z = 0: a dual ray.
    CHECK(hsdInfeasibility(b, c, az_ray, at_y, tau, 1.0, mu, -1.0, 0.0, tol) ==
          BarrierIpmOutcome::DualInfeasible);
    // |A z| = 1e-12 meets tol (-c'z) with |c| = 1, not tol (-c'z) / |c| with
    // |c| = 1e9: the bound is relative to the data.
    const std::vector<Real> az_near{1e-12};
    CHECK(hsdInfeasibility(b, c, az_near, at_y, tau, 1.0, mu, -1.0, 0.0, tol) ==
          BarrierIpmOutcome::DualInfeasible);
    const std::vector<Real> c_big{1e9, 1e9};
    CHECK(hsdInfeasibility(b, c_big, az_near, at_y, tau, 1.0, mu, -1.0, 0.0, tol) ==
          BarrierIpmOutcome::Failed);
    // Both: primal first.
    CHECK(hsdInfeasibility(b, c, az_ray, at_y, tau, 1.0, mu, -1.0, 1.0, tol) ==
          BarrierIpmOutcome::PrimalInfeasible);
    // A NaN gives no verdict: in the ray, or in b'y.
    const Real nan = std::numeric_limits<Real>::quiet_NaN();
    const std::vector<Real> at_y_nan{nan, -2.0};
    CHECK(hsdInfeasibility(b, c, az_far, at_y_nan, tau, 1.0, mu, 0.0, 1.0, tol) ==
          BarrierIpmOutcome::Failed);
    const std::vector<Real> az_nan{nan};
    CHECK(hsdInfeasibility(b, c, az_nan, at_y, tau, 1.0, mu, -1.0, 0.0, tol) ==
          BarrierIpmOutcome::Failed);
    CHECK(hsdInfeasibility(b, c, az_ray, at_y, tau, 1.0, mu, -1.0, nan, tol) ==
          BarrierIpmOutcome::Failed);
    // Upper rows are part of the explicit-form tolerance. Their activity
    // can veto an otherwise acceptable dual infeasibility ray.
    CHECK(hsdInfeasibility(b, c, az_ray, at_y, tau, 1.0, mu, -1.0, 0.0, tol, false, 2.0,
                           2.0 * tol) == BarrierIpmOutcome::Failed);
    CHECK(hsdInfeasibility(b, c, az_ray, at_y, tau, 1.0, mu, -1.0, 0.0, tol, false, 2.0, nan) ==
          BarrierIpmOutcome::Failed);
}

namespace {

// x <= 1 and x >= lower2 on 0 <= x <= 10, with a second column y in
// [y_lower, kInf] on the first row: x + y <= 1. Infeasible for lower2 > 1
// once y >= 0, by the ray (1, -1).
LpProblem twoRowLp(Real lower2, Real y_lower) {
    LpProblem lp;
    lp.sense = Sense::Minimize;
    lp.num_cols = 2;
    lp.num_rows = 2;
    lp.obj = {0.0, 0.0};
    lp.col_lower = {0.0, y_lower};
    lp.col_upper = {10.0, kInf};
    lp.col_type.assign(2, VarType::Continuous);
    lp.col_names.assign(2, "");
    lp.row_lower = {-kInf, lower2};
    lp.row_upper = {1.0, kInf};
    lp.row_names.assign(2, "");
    lp.matrix = SparseMatrix(2, 2, {{0, 0, 1.0}, {0, 1, 1.0}, {1, 0, 1.0}});
    return lp;
}

}  // namespace

TEST_CASE("Barrier crossover: a Farkas ray is rechecked in the model's data",
          "[barrier][crossover]") {
    const auto lp = twoRowLp(2.0, 0.0);
    const std::vector<Real> ray{1.0, -1.0};
    CHECK(farkasRayProvesInfeasible(lp, ray));
    // Any positive multiple proves the same.
    const std::vector<Real> scaled{1e-3, -1e-3};
    CHECK(farkasRayProvesInfeasible(lp, scaled));
    // A multiplier on an infinite side is dropped: (1, 1) leaves the ray
    // (1, 0), which proves nothing.
    const std::vector<Real> wrong_sign{1.0, 1.0};
    CHECK_FALSE(farkasRayProvesInfeasible(lp, wrong_sign));
    CHECK_FALSE(farkasRayProvesInfeasible(lp, std::vector<Real>{0.0, 0.0}));
    CHECK_FALSE(farkasRayProvesInfeasible(lp, std::vector<Real>{1.0}));
    const Real nan = std::numeric_limits<Real>::quiet_NaN();
    CHECK_FALSE(farkasRayProvesInfeasible(lp, std::vector<Real>{nan, -1.0}));
    // Infeasible by 1e-12 against terms of 2: a sliver, no proof.
    CHECK_FALSE(farkasRayProvesInfeasible(twoRowLp(1.0 + 1e-12, 0.0), ray));
    // By 1e-6 it is one.
    CHECK(farkasRayProvesInfeasible(twoRowLp(1.0 + 1e-6, 0.0), ray));
    // y unbounded below: d_y = 1 meets an infinite lower bound, and the LP
    // is feasible.
    CHECK_FALSE(farkasRayProvesInfeasible(twoRowLp(2.0, -kInf), ray));

    // Free x and z: x + z <= 0 and x + c z >= 1. For c = 1 the ray (1, -1)
    // gives A^T y = 0 exactly, a proof. For c = 1 + 1e-13 the LP is feasible
    // (z = 1e13, x = -z): d_z = -1e-13 is far above the sums' round-off, so
    // it meets z's infinite upper bound and proves nothing. `pad` empty
    // columns make the LP large: d_z's round-off is its own sum's, not the
    // whole LP's.
    const auto freeLp = [](Real c, Index pad) {
        LpProblem lp;
        lp.sense = Sense::Minimize;
        lp.num_cols = 2 + pad;
        lp.num_rows = 2;
        lp.obj.assign(lp.num_cols, 0.0);
        lp.col_lower.assign(lp.num_cols, -kInf);
        lp.col_upper.assign(lp.num_cols, kInf);
        lp.col_type.assign(lp.num_cols, VarType::Continuous);
        lp.col_names.assign(lp.num_cols, "");
        lp.row_lower = {-kInf, 1.0};
        lp.row_upper = {0.0, kInf};
        lp.row_names.assign(2, "");
        lp.matrix =
            SparseMatrix(2, lp.num_cols, {{0, 0, 1.0}, {0, 1, 1.0}, {1, 0, 1.0}, {1, 1, c}});
        return lp;
    };
    CHECK(farkasRayProvesInfeasible(freeLp(1.0, 0), ray));
    CHECK_FALSE(farkasRayProvesInfeasible(freeLp(1.0 + 1e-13, 0), ray));
    CHECK_FALSE(farkasRayProvesInfeasible(freeLp(1.0 + 1e-13, 1'000'000), ray));
}

// Netlib cplex2 is infeasible by 1.6e-11 per row: from the HSD run's stalled
// point the warm crossover lands within the simplex's tolerance and ended
// Optimal (#408). Crossover now ends Infeasible on a ray that rechecks, by
// whichever path the run takes: the probe after a stop short is the one
// measured.
TEST_CASE("Barrier HSD: crossover proves Netlib cplex2 infeasible", "[barrier][hsd][crossover]") {
    const auto lp = readMps(mipx::test::testDataDir() + "/cplex2.mps");
    // The cold dual simplex's ray rechecks in the model's data.
    DualSimplexSolver cold;
    cold.load(lp);
    cold.setVerbose(false);
    REQUIRE(cold.solve().status == Status::Infeasible);
    CHECK(farkasRayProvesInfeasible(lp, cold.getDualRay()));

    auto opts = quietOptions(true, BarrierAlgorithm::CpuSparseCholesky);
    opts.presolve = true;
    opts.threads = 1;
    const auto crossed = crossWith(lp, opts);
    CAPTURE(crossed.stats.attempted, crossed.stats.infeasibility_probe,
            crossed.stats.probe_iterations, crossed.stats.cold_fallback);
    CHECK(crossed.homogeneous);
    CHECK(crossed.result.status == Status::Infeasible);
}

// Gondzio's correctors in the homogeneous loop (#335): the same optimum
// as without them, with correctors taken.
TEST_CASE("Barrier HSD: Gondzio correctors run in the homogeneous loop", "[barrier][hsd]") {
    const auto algo = GENERATE(BarrierAlgorithm::CpuCholesky, BarrierAlgorithm::CpuSparseCholesky);
    const bool implicit = GENERATE(false, true);
    INFO("algo " << static_cast<int>(algo) << " implicit " << implicit);
    auto opts = quietOptions(true, algo);
    opts.implicit_upper_bounds = implicit;
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

TEST_CASE("Barrier: Netlib crossover uses the proposed basis",
          "[barrier][hsd][netlib][crossover]") {
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
    const bool homogeneous = GENERATE(false, true);
    CAPTURE(name, static_cast<int>(algo), homogeneous);
    const std::string path = mipx::test::testDataDir() + "/netlib/" + name + ".mps.gz";
    if (!std::filesystem::exists(path)) {
        SKIP("Netlib instance " + std::string(name) + " not downloaded");
    }
    const auto lp = readMps(path);
    auto opts = quietOptions(homogeneous, algo);
    opts.presolve = true;
    const auto crossed = crossWith(lp, opts);
    DualSimplexSolver cold;
    cold.load(lp);
    cold.setVerbose(false);
    const auto cold_result = cold.solve();
    CAPTURE(cold_result.iterations, crossed.iters);
    REQUIRE(cold_result.status == Status::Optimal);
    CHECK(crossed.homogeneous == homogeneous);
    REQUIRE(crossed.result.status == Status::Optimal);
    CHECK(std::abs(crossed.result.objective - expected) <=
          1e-6 * std::max(1.0, std::abs(expected)));
    CHECK(crossed.stats.attempted);
    CHECK_FALSE(crossed.stats.singular_recovery);
    CHECK_FALSE(crossed.stats.cold_fallback);
    checkBasis(crossed.basis, lp);
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
    // The stop short ran the infeasibility probe, whose cold run found no
    // ray: the warm optimum stands (#408).
    CHECK(on.stats.infeasibility_probe);
    CHECK_FALSE(on.stats.cold_fallback);
}
