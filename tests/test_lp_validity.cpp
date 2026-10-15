// Regressions from the LP validity audit (#237): models the dual simplex
// answered wrongly, fixed as literals, each answer checked without trusting
// the solver.

#include "mipx/dual_simplex.h"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>

using namespace mipx;

namespace {

// Relative tolerances of the certificate checks. The dual simplex works to
// 1e-7 in its scaled space; these allow an order of magnitude on top for the
// unscaling back to the model as given.
constexpr long double kPrimalRel = 1e-6L;
constexpr long double kDualRel = 1e-6L;
constexpr long double kObjRel = 1e-6L;
// ---------------------------------------------------------------------------
// Certificate checks, all in long double against the model as given.
// ---------------------------------------------------------------------------

std::string describe(const LpProblem& lp) {
    std::ostringstream os;
    os.precision(17);
    os << (lp.sense == Sense::Maximize ? "max" : "min") << " offset " << lp.obj_offset << "\n  c:";
    for (Real v : lp.obj) {
        os << ' ' << v;
    }
    os << "\n  cols:";
    for (Index j = 0; j < lp.num_cols; ++j) {
        os << " [" << lp.col_lower[j] << ", " << lp.col_upper[j] << "]";
    }
    for (Index i = 0; i < lp.num_rows; ++i) {
        os << "\n  " << lp.row_lower[i] << " <=";
        const auto rv = lp.matrix.row(i);
        for (Index k = 0; k < rv.size(); ++k) {
            os << ' ' << rv.values[k] << "*x" << rv.indices[k];
        }
        os << " <= " << lp.row_upper[i];
    }
    return os.str();
}

// Returns an empty string when (x, y) proves `reported` optimal for lp.
std::string checkOptimal(const LpProblem& lp, const std::vector<Real>& x,
                         const std::vector<Real>& y, Real reported,
                         long double primal_rel = kPrimalRel) {
    std::ostringstream err;
    err.precision(17);
    if (static_cast<Index>(x.size()) != lp.num_cols ||
        static_cast<Index>(y.size()) != lp.num_rows) {
        return "solution vectors have the wrong size";
    }
    const long double s = lp.sense == Sense::Maximize ? -1.0L : 1.0L;

    // Objective as reported equals c^T x + offset.
    long double cx = lp.obj_offset;
    long double cx_mag = std::abs(static_cast<long double>(lp.obj_offset));
    for (Index j = 0; j < lp.num_cols; ++j) {
        cx += static_cast<long double>(lp.obj[j]) * x[j];
        cx_mag += std::abs(static_cast<long double>(lp.obj[j]) * x[j]);
    }
    if (std::abs(cx - reported) > kObjRel * std::max(1.0L, cx_mag)) {
        err << "objective " << reported << " != c^T x + offset " << static_cast<double>(cx) << "; ";
    }

    // Primal feasibility, relative to the row's magnitude.
    std::vector<long double> act(lp.num_rows, 0.0L);
    std::vector<long double> act_mag(lp.num_rows, 0.0L);
    for (Index i = 0; i < lp.num_rows; ++i) {
        const auto rv = lp.matrix.row(i);
        for (Index k = 0; k < rv.size(); ++k) {
            const long double t = static_cast<long double>(rv.values[k]) * x[rv.indices[k]];
            act[i] += t;
            act_mag[i] = std::max(act_mag[i], std::abs(t));
        }
    }
    auto finiteMag = [](Real v) {
        return std::isfinite(v) ? std::abs(static_cast<long double>(v)) : 0.0L;
    };
    std::vector<long double> row_tol(lp.num_rows);
    for (Index i = 0; i < lp.num_rows; ++i) {
        row_tol[i] = primal_rel * std::max({1.0L, act_mag[i], finiteMag(lp.row_lower[i]),
                                            finiteMag(lp.row_upper[i])});
        if (act[i] < lp.row_lower[i] - row_tol[i] || act[i] > lp.row_upper[i] + row_tol[i]) {
            err << "row " << i << " activity " << static_cast<double>(act[i]) << " outside ["
                << lp.row_lower[i] << ", " << lp.row_upper[i] << "]; ";
        }
    }
    // A bound may be missed by as much as moves no row by more than its own
    // tolerance: the dual simplex holds bounds to 1e-7 in its scaled space,
    // where a column of tiny coefficients is stretched (#236 policy). On such
    // a column that is far more than 1e-6 in the model's units.
    std::vector<long double> col_tol(lp.num_cols);
    for (Index j = 0; j < lp.num_cols; ++j) {
        col_tol[j] = primal_rel * std::max({1.0L, finiteMag(lp.col_lower[j]),
                                            finiteMag(lp.col_upper[j]), finiteMag(x[j])});
    }
    {
        std::vector<long double> reach(lp.num_cols, kInf);
        for (Index i = 0; i < lp.num_rows; ++i) {
            const auto rv = lp.matrix.row(i);
            for (Index k = 0; k < rv.size(); ++k) {
                if (rv.values[k] != 0.0) {
                    auto& r = reach[rv.indices[k]];
                    r = std::min(r, row_tol[i] / std::abs(static_cast<long double>(rv.values[k])));
                }
            }
        }
        for (Index j = 0; j < lp.num_cols; ++j) {
            if (std::isfinite(reach[j])) {
                col_tol[j] = std::max(col_tol[j], reach[j]);
            }
        }
    }
    for (Index j = 0; j < lp.num_cols; ++j) {
        if (x[j] < lp.col_lower[j] - col_tol[j] || x[j] > lp.col_upper[j] + col_tol[j]) {
            err << "x" << j << " = " << x[j] << " outside [" << lp.col_lower[j] << ", "
                << lp.col_upper[j] << "]; ";
        }
    }

    // Reduced costs in minimization terms: d = s*c - A^T (s*y), since y is
    // reported in the model's own sense.
    std::vector<long double> d(lp.num_cols);
    std::vector<long double> d_mag(lp.num_cols);
    for (Index j = 0; j < lp.num_cols; ++j) {
        d[j] = s * lp.obj[j];
        d_mag[j] = std::abs(d[j]);
    }
    for (Index i = 0; i < lp.num_rows; ++i) {
        const long double yi = s * y[i];
        const auto rv = lp.matrix.row(i);
        for (Index k = 0; k < rv.size(); ++k) {
            const long double t = static_cast<long double>(rv.values[k]) * yi;
            d[rv.indices[k]] -= t;
            d_mag[rv.indices[k]] = std::max(d_mag[rv.indices[k]], std::abs(t));
        }
    }

    // Dual feasibility (each multiplier's sign is allowed by the side or
    // bound it prices) and the duality gap, term by term: for a feasible pair
    // each term is >= 0 and their sum is c^T x minus the dual objective, so
    // each must vanish (complementary slackness).
    //
    // A multiplier's sign is checked by its weight in the dual, |y_i| times
    // the row's largest coefficient: the dual simplex holds it to tolerance
    // in row-equilibrated units, so on a row of 1e-7 coefficients a wrong-
    // signed y_i of 1e-6 weighs nothing.
    long double dual_scale = 1.0L;
    for (Index j = 0; j < lp.num_cols; ++j) {
        dual_scale = std::max(dual_scale, d_mag[j]);
    }
    for (Index i = 0; i < lp.num_rows; ++i) {
        const long double yi = s * y[i];
        if (yi == 0.0L) {
            continue;
        }
        const Real side = yi > 0.0L ? lp.row_lower[i] : lp.row_upper[i];
        if (!std::isfinite(side)) {
            long double row_max = 0.0L;
            const auto rv = lp.matrix.row(i);
            for (Index k = 0; k < rv.size(); ++k) {
                row_max = std::max(row_max, std::abs(static_cast<long double>(rv.values[k])));
            }
            if (std::abs(yi) * row_max > kDualRel * dual_scale) {
                err << "y" << i << " = " << y[i] << " needs a finite "
                    << (yi > 0.0L ? "lower" : "upper") << " side; ";
            }
            continue;
        }
        const long double term = yi * (act[i] - side);  // >= 0 when complementary
        const long double tol =
            std::abs(yi) * row_tol[i] * 10.0L + kObjRel * std::max(1.0L, std::abs(yi * side));
        if (std::abs(term) > tol) {
            err << "row " << i << " complementarity y=" << y[i] << " slack "
                << static_cast<double>(act[i] - side) << "; ";
        }
    }
    for (Index j = 0; j < lp.num_cols; ++j) {
        const long double tol_d = kDualRel * std::max(1.0L, d_mag[j]);
        if (d[j] > tol_d && lp.col_lower[j] == -kInf) {
            err << "d" << j << " = " << static_cast<double>(d[j])
                << " > 0 with no lower bound (dual infeasible); ";
            continue;
        }
        if (d[j] < -tol_d && lp.col_upper[j] == kInf) {
            err << "d" << j << " = " << static_cast<double>(d[j])
                << " < 0 with no upper bound (dual infeasible); ";
            continue;
        }
        if (std::abs(d[j]) <= tol_d) {
            continue;
        }
        const Real bound = d[j] > 0.0L ? lp.col_lower[j] : lp.col_upper[j];
        const long double term = d[j] * (x[j] - bound);
        const long double tol =
            std::abs(d[j]) * col_tol[j] * 10.0L + kObjRel * std::max(1.0L, std::abs(d[j] * bound));
        if (std::abs(term) > tol) {
            err << "x" << j << " complementarity d=" << static_cast<double>(d[j]) << " gap "
                << static_cast<double>(x[j] - bound) << "; ";
        }
    }
    return err.str();
}

// Returns an empty string when y satisfies the LpSolver::getDualRay() contract.
//
// d = A^T y is accumulated in long double, and an entry meeting a finite
// bound always counts at its value. An entry meeting an infinite bound counts
// as zero when it is below the pivot tolerance of the ray's size in
// row-equilibrated units, max_i |y_i| * max_k |a_ik|: that is an entry the
// ratio test skipped (or one within a capped repair step), as the contract
// allows. Measuring it against max |y| alone would read rows of very
// different scales wrongly.
//
// With `marginal`, a ray whose alpha falls short of beta within tolerance is
// accepted: it proves the model infeasible up to tolerance, which is all a
// model built without a known status (the warm-start rounds) can promise.
std::string checkFarkas(const LpProblem& lp, const std::vector<Real>& y, bool marginal = false) {
    if (static_cast<Index>(y.size()) != lp.num_rows) {
        return "dual ray has size " + std::to_string(y.size());
    }
    long double size = 0.0L;
    long double beta = 0.0L;
    long double mag = 0.0L;
    std::vector<long double> d(lp.num_cols, 0.0L);
    for (Index i = 0; i < lp.num_rows; ++i) {
        const long double yi = y[i];
        if (yi > 0.0L) {
            if (lp.row_upper[i] == kInf) {
                return "y" + std::to_string(i) + " > 0 on a row with no upper side";
            }
            beta += yi * lp.row_upper[i];
            mag += std::abs(yi * lp.row_upper[i]);
        } else if (yi < 0.0L) {
            if (lp.row_lower[i] == -kInf) {
                return "y" + std::to_string(i) + " < 0 on a row with no lower side";
            }
            beta += yi * lp.row_lower[i];
            mag += std::abs(yi * lp.row_lower[i]);
        }
        const auto rv = lp.matrix.row(i);
        for (Index k = 0; k < rv.size(); ++k) {
            const long double t = static_cast<long double>(rv.values[k]) * yi;
            d[rv.indices[k]] += t;
            size = std::max(size, std::abs(t));
        }
    }
    if (std::all_of(y.begin(), y.end(), [](Real v) { return v == 0.0; })) {
        return "dual ray is zero";
    }
    long double alpha = 0.0L;
    for (Index j = 0; j < lp.num_cols; ++j) {
        if (d[j] == 0.0L) {
            continue;
        }
        const Real bound = d[j] > 0.0L ? lp.col_lower[j] : lp.col_upper[j];
        if (std::isfinite(bound)) {
            alpha += d[j] * bound;
            mag += std::abs(d[j] * bound);
        } else if (std::abs(d[j]) > 1e-7L * size) {
            std::ostringstream os;
            os << "d" << j << " = " << static_cast<double>(d[j]) << " meets an infinite bound";
            return os.str();
        }
    }
    const long double margin =
        marginal ? -kPrimalRel * std::max(1.0L, mag) : 1e-9L * std::max(size, mag);
    if (alpha - beta <= margin) {
        std::ostringstream os;
        os << "alpha " << static_cast<double>(alpha) << " does not exceed beta "
           << static_cast<double>(beta);
        return os.str();
    }
    return {};
}

// ---------------------------------------------------------------------------
// Models the audit found wrong answers on (#237), fixed as literals so they
// do not depend on the generator or the standard library's distributions.
// ---------------------------------------------------------------------------

LpProblem literalLp(Sense sense, Real offset, std::vector<Real> obj, std::vector<Real> lo,
                    std::vector<Real> up, std::vector<Real> rl, std::vector<Real> ru,
                    std::vector<Triplet> entries) {
    LpProblem lp;
    lp.sense = sense;
    lp.obj_offset = offset;
    lp.num_cols = static_cast<Index>(obj.size());
    lp.num_rows = static_cast<Index>(rl.size());
    lp.obj = std::move(obj);
    lp.col_lower = std::move(lo);
    lp.col_upper = std::move(up);
    lp.col_type.assign(static_cast<std::size_t>(lp.num_cols), VarType::Continuous);
    lp.row_lower = std::move(rl);
    lp.row_upper = std::move(ru);
    lp.matrix = SparseMatrix(lp.num_rows, lp.num_cols, std::move(entries));
    return lp;
}

LpResult solveLiteral(DualSimplexSolver& solver, const LpProblem& lp) {
    solver.setVerbose(false);
    solver.setIterationLimit(10000);
    solver.load(lp);
    return solver.solve();
}

void requireCertifiedOptimal(const LpProblem& lp) {
    DualSimplexSolver solver;
    const LpResult res = solveLiteral(solver, lp);
    INFO(describe(lp));
    REQUIRE(res.status == Status::Optimal);
    CHECK(checkOptimal(lp, solver.getPrimalValues(), solver.getDualValues(), res.objective) == "");
}

void requireCertifiedInfeasible(const LpProblem& lp) {
    DualSimplexSolver solver;
    const LpResult res = solveLiteral(solver, lp);
    INFO(describe(lp));
    REQUIRE(res.status == Status::Infeasible);
    CHECK(checkFarkas(lp, solver.getDualRay()) == "");
}

}  // namespace

TEST_CASE("Regression #237: an empty ratio test proves infeasibility without dual feasibility",
          "[lp_validity]") {
    // The leaving row has no eligible entering column, but some nonbasic
    // reduced cost has the wrong sign. The solver refactorized four times and
    // then reported Error.
    requireCertifiedInfeasible(literalLp(Sense::Maximize, -7.125,
                                         {17.8125, -4.890625, -13.0625, 11.9375},
                                         {-kInf, -kInf, -kInf, -3.375}, {kInf, 6.25, 7.375, -1.875},
                                         {-12.75, 12.375, -kInf, -12.3125, -24.186764122050818},
                                         {-12.75, kInf, 0.828125, -10.3125, kInf},
                                         {{0, 0, 3.25},
                                          {0, 1, -3.25},
                                          {0, 3, 4.5},
                                          {1, 0, 2},
                                          {1, 1, 1.625},
                                          {2, 1, 2.5},
                                          {2, 2, -3.625},
                                          {3, 0, -4.625},
                                          {3, 1, -2.875},
                                          {3, 2, 4.75},
                                          {4, 0, -3.7050438596491224},
                                          {4, 1, -4.4828956820931642},
                                          {4, 2, 1.7499999999999998},
                                          {4, 3, 1.875}}));
}

TEST_CASE("Regression #237: a free column with a tiny coefficient repairs the leaving row",
          "[lp_validity]") {
    // x1 is free and enters rows 0 and 1 only through coefficients near
    // 2e-8, below the pivot tolerance. The dual ratio test skipped it and
    // read the empty row as a proof: Infeasible on a feasible LP.
    requireCertifiedOptimal(
        literalLp(Sense::Maximize, -2.75, {36.25, 0.000421875, 16875, -2.6875, 11.25},
                  {-0.26250000000000001, -kInf, -kInf, -0.125, -0.16250000000000001},
                  {-0.26250000000000001, kInf, kInf, -0.125, kInf},
                  {-0.0014578125000000001, 0.001453125, -kInf, -1.9828125000000001},
                  {kInf, 0.001840625, 12656.25, -1.7078125000000002},
                  {{0, 0, 0.002},
                   {0, 1, -2.1250000000000002e-08},
                   {0, 2, 0.37500000000000006},
                   {0, 4, -0.0015},
                   {1, 0, -0.002875},
                   {1, 1, -1.6250000000000003e-08},
                   {1, 2, -5},
                   {1, 3, 0.00015000000000000001},
                   {1, 4, -0.0032500000000000003},
                   {2, 1, 0.375},
                   {3, 0, 2.25},
                   {3, 2, 3375},
                   {3, 3, -0.0625},
                   {3, 4, 2.25}}));
}

TEST_CASE("Regression #237: a repair that needs an astronomically long step does not count",
          "[lp_validity]") {
    // Infeasible by construction. Repairing the leaving row through entries
    // near 1e-9 would move a column by about 1e14, where its value is
    // round-off; doing so reported Optimal at |x| near 1e16.
    requireCertifiedInfeasible(
        literalLp(Sense::Maximize, 3.125,
                  {-0.00093749999999999997, 4484.375, -0.00092656250000000009, 1839.0625,
                   0.082656250000000001, -0.00175},
                  {-kInf, -kInf, 22500, 0.021250000000000002, -kInf, 375},
                  {kInf, 0.000125, kInf, kInf, kInf, kInf},
                  {-13593.75, -1.65625, -0.0046249999999999998, -2203.125, -26014.622338392852},
                  {-13593.75, 0.96875, -0.0046249999999999998, kInf, kInf},
                  {{0, 0, 0.25},
                   {0, 1, -4375000},
                   {0, 2, 0.087500000000000008},
                   {0, 3, -312500},
                   {0, 4, -28.75},
                   {1, 0, -0.000125},
                   {1, 1, -3125},
                   {1, 2, -0.00030000000000000003},
                   {1, 3, 387.5},
                   {2, 0, -1.2499999999999999e-07},
                   {2, 2, -2.0000000000000002e-07},
                   {3, 3, -250000},
                   {3, 4, 21.25},
                   {3, 5, 1.625},
                   {4, 0, -0.78125001785714265},
                   {4, 1, 1.375},
                   {4, 2, -1.2499999549999998},
                   {4, 3, -0.023928571428571428},
                   {4, 4, 4.9285714285714286e-06}}));
}
