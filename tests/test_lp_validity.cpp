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
