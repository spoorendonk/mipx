// LP validity audit for the dual simplex (#237).
//
// Every model is built around a known answer, so each result is checked
// without trusting the solver:
//
//   - Optimal models are built from a KKT point: a primal point x*, row
//     multipliers y and reduced costs d with the signs that make x* optimal,
//     and the objective c = A^T y + d. The optimum is c^T x* plus the offset.
//     The solver's own x and y are then checked as a certificate: primal
//     feasibility, dual feasibility of y (with d recomputed here as
//     c - A^T y) and a zero duality gap, term by term (complementary
//     slackness).
//   - Infeasible models add to a feasible one the row g^T x >= beta + delta,
//     where g is a positive combination of rows whose sides bound g^T x by
//     beta. The Farkas ray the solver returns is checked against the
//     LpSolver::getDualRay() contract.
//   - Unbounded models open a feasible one along a ray r (every row and
//     bound that r moves towards is made infinite) and tilt c so c^T r < 0.
//     The primal ray the solver returns is checked against the
//     LpSolver::getPrimalRay() contract (#238), as is every Unbounded the
//     warm rounds produce.
//
// Shapes: equality, ranged, one-sided and free rows; free, fixed, boxed and
// half-bounded columns; degenerate points (zero multipliers on active rows
// and bounds, more active rows than columns); row and column scalings that
// spread the coefficients over 1e-9..1e9; minimize and maximize. Each optimal
// model is also re-solved warm after bound changes, as branch-and-bound does,
// and compared with a cold solve of the changed model.
//
// Multipliers follow the minimization convention after the sense is undone:
// for min, y_i >= 0 where row i sits at its lower side and y_i <= 0 at its
// upper; d_j likewise for column bounds. getDualValues() and
// getReducedCosts() negate for maximize, so the same test applies to the
// objective -c of the equivalent minimization.

#include "mipx/barrier.h"
#include "mipx/dual_simplex.h"
#include "mipx/mip_solver.h"
#include "mipx/pdlp.h"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstdint>
#include <random>
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
// Warm rounds cut bounds at arbitrary points, so their optima often sit on
// rows of mixed scale exactly at a bound; the scaled 1e-7 maps to a little
// more than 1e-6 relative there (seen up to 2e-6).
constexpr long double kWarmRel = 1e-5L;
// How far two optima that both hold to kWarmRel may differ (seen up to
// 2.5e-5 relative); a larger gap fails whatever the certificates say.
constexpr Real kWarmObjRel = 1e-4;

enum class Shape : std::uint8_t { Optimal, Infeasible, Unbounded };

struct Generated {
    LpProblem lp;
    Shape shape = Shape::Optimal;
    Real optimum = 0.0;  // in the model's own sense, offset included
};

class Rng {
public:
    explicit Rng(std::uint32_t seed) : gen_(seed) {}
    Real uniform(Real lo, Real hi) { return std::uniform_real_distribution<Real>(lo, hi)(gen_); }
    int pick(int lo, int hi) { return std::uniform_int_distribution<int>(lo, hi)(gen_); }
    bool coin(Real p) { return uniform(0.0, 1.0) < p; }
    // A value with a few significant digits, so sums stay exact-ish.
    Real value(Real lo, Real hi) { return std::round(uniform(lo, hi) * 8.0) / 8.0; }

private:
    std::mt19937 gen_;
};

// Column shapes and where x*_j sits in them.
enum class ColKind : std::uint8_t { Boxed, Fixed, Free, LowerOnly, UpperOnly };
// Row shapes.
enum class RowKind : std::uint8_t { Equality, Lower, Upper, Ranged, Free };

struct Options {
    bool scaled = false;      // spread coefficients over 1e-9..1e9
    bool degenerate = false;  // many active rows and zero multipliers
    bool maximize = false;
};

// Build an LP with known optimum c^T x* (min convention) from a KKT point.
Generated generateOptimal(Rng& rng, const Options& opt) {
    const int n = rng.pick(2, 7);
    const int m = rng.pick(1, opt.degenerate ? 9 : 6);

    std::vector<Real> x(n), cl(n), cu(n), d(n, 0.0);
    for (int j = 0; j < n; ++j) {
        const auto kind = static_cast<ColKind>(rng.pick(0, 4));
        x[j] = rng.value(-4.0, 4.0);
        const Real w = rng.value(0.5, 5.0);
        switch (kind) {
            case ColKind::Boxed: {
                const int at = rng.pick(0, 2);  // 0 lower, 1 upper, 2 interior
                cl[j] = at == 0 ? x[j] : x[j] - w;
                cu[j] = at == 1 ? x[j] : x[j] + w;
                if (at == 0) {
                    d[j] = rng.coin(opt.degenerate ? 0.5 : 0.2) ? 0.0 : rng.value(0.125, 3.0);
                } else if (at == 1) {
                    d[j] = rng.coin(opt.degenerate ? 0.5 : 0.2) ? 0.0 : -rng.value(0.125, 3.0);
                }
                break;
            }
            case ColKind::Fixed:
                cl[j] = cu[j] = x[j];
                d[j] = rng.value(-3.0, 3.0);
                break;
            case ColKind::Free:
                cl[j] = -kInf;
                cu[j] = kInf;
                break;
            case ColKind::LowerOnly:
                cu[j] = kInf;
                if (rng.coin(0.6)) {
                    cl[j] = x[j];
                    d[j] = rng.coin(opt.degenerate ? 0.5 : 0.2) ? 0.0 : rng.value(0.125, 3.0);
                } else {
                    cl[j] = x[j] - w;
                }
                break;
            case ColKind::UpperOnly:
                cl[j] = -kInf;
                if (rng.coin(0.6)) {
                    cu[j] = x[j];
                    d[j] = rng.coin(opt.degenerate ? 0.5 : 0.2) ? 0.0 : -rng.value(0.125, 3.0);
                } else {
                    cu[j] = x[j] + w;
                }
                break;
        }
    }

    std::vector<std::vector<Real>> a(m, std::vector<Real>(n, 0.0));
    std::vector<Real> rl(m), ru(m), y(m, 0.0);
    for (int i = 0; i < m; ++i) {
        bool any = false;
        for (int j = 0; j < n; ++j) {
            if (rng.coin(0.6)) {
                a[i][j] = rng.value(-5.0, 5.0);
                any = any || a[i][j] != 0.0;
            }
        }
        if (!any) {
            a[i][rng.pick(0, n - 1)] = 1.0;
        }
        Real act = 0.0;
        for (int j = 0; j < n; ++j) {
            act += a[i][j] * x[j];
        }
        const auto kind = static_cast<RowKind>(rng.pick(0, 4));
        const Real slack = rng.value(0.5, 4.0);
        const bool active = rng.coin(opt.degenerate ? 0.85 : 0.6);
        const Real zero_y = opt.degenerate ? 0.5 : 0.15;
        switch (kind) {
            case RowKind::Equality:
                rl[i] = ru[i] = act;
                y[i] = rng.value(-3.0, 3.0);
                break;
            case RowKind::Lower:
                rl[i] = active ? act : act - slack;
                ru[i] = kInf;
                if (active && !rng.coin(zero_y)) {
                    y[i] = rng.value(0.125, 3.0);
                }
                break;
            case RowKind::Upper:
                rl[i] = -kInf;
                ru[i] = active ? act : act + slack;
                if (active && !rng.coin(zero_y)) {
                    y[i] = -rng.value(0.125, 3.0);
                }
                break;
            case RowKind::Ranged: {
                const int at = active ? rng.pick(0, 1) : 2;
                rl[i] = at == 0 ? act : act - slack;
                ru[i] = at == 1 ? act : act + slack;
                if (at < 2 && !rng.coin(zero_y)) {
                    y[i] = (at == 0 ? 1.0 : -1.0) * rng.value(0.125, 3.0);
                }
                break;
            }
            case RowKind::Free:
                rl[i] = -kInf;
                ru[i] = kInf;
                break;
        }
    }

    // c = A^T y + d, so x* with (y, d) satisfies the KKT conditions.
    std::vector<Real> c(d);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            c[j] += a[i][j] * y[i];
        }
    }

    // Optional scaling: A' = R A S, x' = S^{-1} x, c' = S c, rows times R.
    std::vector<Real> rs(m, 1.0), cs(n, 1.0);
    if (opt.scaled) {
        for (auto& v : rs) {
            v = std::pow(10.0, rng.pick(-4, 4));
        }
        for (auto& v : cs) {
            v = std::pow(10.0, rng.pick(-4, 4));
        }
    }

    Generated g;
    LpProblem& lp = g.lp;
    lp.name = "lp_validity";
    lp.num_cols = n;
    lp.num_rows = m;
    lp.obj_offset = rng.value(-10.0, 10.0);
    lp.col_type.assign(n, VarType::Continuous);
    lp.obj.resize(n);
    lp.col_lower.resize(n);
    lp.col_upper.resize(n);
    Real opt_min = 0.0;
    for (int j = 0; j < n; ++j) {
        lp.obj[j] = c[j] * cs[j];
        lp.col_lower[j] = cl[j] == -kInf ? -kInf : cl[j] / cs[j];
        lp.col_upper[j] = cu[j] == kInf ? kInf : cu[j] / cs[j];
        opt_min += c[j] * x[j];
    }
    std::vector<Triplet> entries;
    lp.row_lower.resize(m);
    lp.row_upper.resize(m);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (a[i][j] != 0.0) {
                entries.push_back({i, j, a[i][j] * rs[i] * cs[j]});
            }
        }
        lp.row_lower[i] = rl[i] == -kInf ? -kInf : rl[i] * rs[i];
        lp.row_upper[i] = ru[i] == kInf ? kInf : ru[i] * rs[i];
    }
    lp.matrix = SparseMatrix(m, n, std::move(entries));
    if (opt.maximize) {
        lp.sense = Sense::Maximize;
        for (auto& v : lp.obj) {
            v = -v;
        }
        g.optimum = -opt_min + lp.obj_offset;
    } else {
        g.optimum = opt_min + lp.obj_offset;
    }
    return g;
}

std::vector<std::vector<Real>> denseRows(const LpProblem& lp) {
    std::vector<std::vector<Real>> a(lp.num_rows, std::vector<Real>(lp.num_cols, 0.0));
    for (Index i = 0; i < lp.num_rows; ++i) {
        const auto rv = lp.matrix.row(i);
        for (Index k = 0; k < rv.size(); ++k) {
            a[i][rv.indices[k]] += rv.values[k];
        }
    }
    return a;
}

void setRows(LpProblem& lp, const std::vector<std::vector<Real>>& a) {
    std::vector<Triplet> entries;
    for (Index i = 0; i < static_cast<Index>(a.size()); ++i) {
        for (Index j = 0; j < lp.num_cols; ++j) {
            if (a[i][j] != 0.0) {
                entries.push_back({i, j, a[i][j]});
            }
        }
    }
    lp.num_rows = static_cast<Index>(a.size());
    lp.matrix = SparseMatrix(lp.num_rows, lp.num_cols, std::move(entries));
}

// Add g^T x >= beta + delta, where g = sum_i w_i sigma_i a_i over rows with the
// side sigma_i finite, so every point meeting the rows has g^T x <= beta.
Generated generateInfeasible(Rng& rng, const Options& opt) {
    Generated g = generateOptimal(rng, opt);
    LpProblem& lp = g.lp;
    auto a = denseRows(lp);
    std::vector<Real> gv(lp.num_cols, 0.0);
    long double beta = 0.0L;
    long double scale = 0.0L;
    bool used = false;
    for (Index i = 0; i < lp.num_rows; ++i) {
        const bool up = lp.row_upper[i] != kInf;
        const bool lo = lp.row_lower[i] != -kInf;
        if ((!up && !lo) || !rng.coin(0.7)) {
            continue;
        }
        const Real sigma = up && (!lo || rng.coin(0.5)) ? 1.0 : -1.0;
        const Real side = sigma > 0.0 ? lp.row_upper[i] : lp.row_lower[i];
        // Weight by the row's size, so scaled rows contribute evenly.
        Real rmax = 0.0;
        for (Real v : a[i]) {
            rmax = std::max(rmax, std::abs(v));
        }
        const Real w = rng.value(0.5, 3.0) / rmax;
        for (Index j = 0; j < lp.num_cols; ++j) {
            gv[j] += w * sigma * a[i][j];
        }
        beta += static_cast<long double>(w) * sigma * side;
        scale = std::max(scale, std::abs(static_cast<long double>(w) * side));
        used = true;
    }
    if (!used) {
        // No row qualified: contradict a column bound instead.
        Index j = 0;
        while (j < lp.num_cols && lp.col_upper[j] == kInf && lp.col_lower[j] == -kInf) {
            ++j;
        }
        if (j == lp.num_cols) {
            j = 0;
            lp.col_upper[0] = lp.col_lower[0] = 0.0;
        }
        std::fill(gv.begin(), gv.end(), 0.0);
        if (lp.col_upper[j] != kInf) {
            gv[j] = 1.0;
            beta = lp.col_upper[j];
        } else {
            gv[j] = -1.0;
            beta = -lp.col_lower[j];
        }
        scale = std::abs(beta);
    }
    a.push_back(gv);
    setRows(lp, a);
    const long double delta = 1.0L + 0.1L * scale;
    lp.row_lower.push_back(static_cast<Real>(beta + delta));
    lp.row_upper.push_back(kInf);
    g.shape = Shape::Infeasible;
    return g;
}

// Open the optimal model along a ray r and tilt the objective along -r.
Generated generateUnbounded(Rng& rng, const Options& opt) {
    Generated g = generateOptimal(rng, opt);
    LpProblem& lp = g.lp;
    const Index n = lp.num_cols;
    std::vector<Real> r(n, 0.0);
    const int support = rng.pick(1, std::min<int>(3, n));
    for (int k = 0; k < support; ++k) {
        const Index j = rng.pick(0, n - 1);
        r[j] = rng.value(0.25, 2.0) * (rng.coin(0.5) ? 1.0 : -1.0);
        if (lp.col_lower[j] == lp.col_upper[j]) {
            lp.col_lower[j] = -kInf;  // a fixed column cannot move; free it
            lp.col_upper[j] = kInf;
        }
        if (r[j] > 0.0) {
            lp.col_upper[j] = kInf;
        } else {
            lp.col_lower[j] = -kInf;
        }
    }
    const auto a = denseRows(lp);
    for (Index i = 0; i < lp.num_rows; ++i) {
        long double ar = 0.0L;
        for (Index j = 0; j < n; ++j) {
            ar += static_cast<long double>(a[i][j]) * r[j];
        }
        if (ar > 0.0L) {
            lp.row_upper[i] = kInf;
        } else if (ar < 0.0L) {
            lp.row_lower[i] = -kInf;
        }
    }
    // Minimization terms: c_min = obj (min) or -obj (max). Make c_min^T r < 0.
    const Real s = lp.sense == Sense::Maximize ? -1.0 : 1.0;
    long double cr = 0.0L;
    long double rr = 0.0L;
    for (Index j = 0; j < n; ++j) {
        cr += static_cast<long double>(s * lp.obj[j]) * r[j];
        rr += static_cast<long double>(r[j]) * r[j];
    }
    const long double t = (cr + 1.0L + 0.5L * std::abs(cr)) / rr;
    for (Index j = 0; j < n; ++j) {
        lp.obj[j] -= static_cast<Real>(s * t * r[j]);
    }
    g.shape = Shape::Unbounded;
    return g;
}

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
    // The widened tolerance below excuses a bound miss only. A column off
    // its bound on the feasible side still breaks complementarity at this.
    const std::vector<long double> col_tol_strict = col_tol;
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
        // > 0: off the bound on the feasible side; < 0: past the bound.
        const long double term = d[j] * (x[j] - bound);
        const long double off = term > 0.0L ? col_tol_strict[j] : col_tol[j];
        const long double tol =
            std::abs(d[j]) * off * 10.0L + kObjRel * std::max(1.0L, std::abs(d[j] * bound));
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

// Returns an empty string when r satisfies the LpSolver::getPrimalRay()
// contract.
//
// q = A r is accumulated in long double. The column signs must hold exactly.
// A row entry meeting a finite side counts as zero when it is below the
// pivot tolerance of the ray's size in the row's units, max_j |a_ij| * |r|:
// round-off in the ray's small entries, as the contract allows. Measuring it
// against the row's own terms max_j |a_ij r_j| would not do: an entry of
// 1e-12 against |r| = 3e-3, alone in its row, is still round-off. c^T r must
// be negative (in minimization terms) by more than round-off of its terms.
std::string checkPrimalRay(const LpProblem& lp, const std::vector<Real>& r) {
    if (static_cast<Index>(r.size()) != lp.num_cols) {
        return "primal ray has size " + std::to_string(r.size());
    }
    if (std::all_of(r.begin(), r.end(), [](Real v) { return v == 0.0; })) {
        return "primal ray is zero";
    }
    long double rmax = 0.0L;
    for (Real v : r) {
        rmax = std::max(rmax, std::abs(static_cast<long double>(v)));
    }
    const long double s = lp.sense == Sense::Maximize ? -1.0L : 1.0L;
    long double cr = 0.0L;
    long double cmag = 0.0L;
    for (Index j = 0; j < lp.num_cols; ++j) {
        if ((r[j] > 0.0 && lp.col_upper[j] != kInf) || (r[j] < 0.0 && lp.col_lower[j] != -kInf)) {
            std::ostringstream os;
            os << "r" << j << " = " << r[j] << " moves towards a finite bound";
            return os.str();
        }
        const long double t = s * static_cast<long double>(lp.obj[j]) * r[j];
        cr += t;
        cmag += std::abs(t);
    }
    if (cr >= -1e-9L * cmag) {
        std::ostringstream os;
        os << "c^T r = " << static_cast<double>(cr) << " (minimization terms) is not negative";
        return os.str();
    }
    for (Index i = 0; i < lp.num_rows; ++i) {
        long double q = 0.0L;
        long double amax = 0.0L;
        const auto rv = lp.matrix.row(i);
        for (Index k = 0; k < rv.size(); ++k) {
            q += static_cast<long double>(rv.values[k]) * r[rv.indices[k]];
            amax = std::max(amax, std::abs(static_cast<long double>(rv.values[k])));
        }
        const bool blocked =
            (q > 0.0L && lp.row_upper[i] != kInf) || (q < 0.0L && lp.row_lower[i] != -kInf);
        if (blocked && std::abs(q) > 1e-7L * amax * rmax) {
            std::ostringstream os;
            os << "(A r)" << i << " = " << static_cast<double>(q) << " meets a finite side";
            return os.str();
        }
    }
    return {};
}

const char* statusName(Status s) {
    switch (s) {
        case Status::Optimal:
            return "Optimal";
        case Status::Infeasible:
            return "Infeasible";
        case Status::Unbounded:
            return "Unbounded";
        case Status::IterLimit:
            return "IterLimit";
        case Status::TimeLimit:
            return "TimeLimit";
        case Status::NodeLimit:
            return "NodeLimit";
        case Status::Error:
            return "Error";
    }
    return "?";
}

struct Audit {
    int models = 0;
    // Warm rounds where the two solves differ only within tolerance: one
    // proved the model infeasible up to tolerance and the other returned a
    // point, or both optima hold to tolerance at different objectives.
    int tolerance_calls = 0;
    std::vector<std::string> failures;

    void fail(const std::string& what, const LpProblem& lp) {
        if (failures.size() < 8) {
            failures.push_back(what + "\n" + describe(lp));
        } else {
            failures.emplace_back(what);
        }
    }
};

// Solve with the solver as branch-and-bound uses it and check the verdict.
// Returns the status, so a caller can go on to warm starts.
Status checkSolve(DualSimplexSolver& solver, const Generated& g, const std::string& tag,
                  Audit& audit, bool check_known_optimum = true) {
    const LpResult res = solver.solve();
    const LpProblem& lp = g.lp;
    const auto want = g.shape == Shape::Optimal      ? Status::Optimal
                      : g.shape == Shape::Infeasible ? Status::Infeasible
                                                     : Status::Unbounded;
    if (res.status == Status::Unbounded) {
        const std::string err = checkPrimalRay(lp, solver.getPrimalRay());
        if (!err.empty()) {
            std::ostringstream os;
            os.precision(17);
            os << tag << ": primal ray: " << err << "\n  r:";
            for (Real v : solver.getPrimalRay()) {
                os << ' ' << v;
            }
            audit.fail(os.str(), lp);
        }
    } else if (!solver.getPrimalRay().empty()) {
        audit.fail(tag + ": primal ray after status " + statusName(res.status), lp);
    }
    if (res.status != want) {
        audit.fail(tag + ": status " + statusName(res.status) + ", expected " + statusName(want),
                   lp);
        return res.status;
    }
    if (want == Status::Optimal) {
        const std::string err =
            checkOptimal(lp, solver.getPrimalValues(), solver.getDualValues(), res.objective);
        if (!err.empty()) {
            audit.fail(tag + ": optimal certificate: " + err, lp);
        }
        if (std::abs(res.objective - solver.getObjective()) >
            kObjRel * std::max(1.0, std::abs(res.objective))) {
            audit.fail(tag + ": LpResult.objective differs from getObjective()", lp);
        }
        if (check_known_optimum &&
            std::abs(res.objective - g.optimum) > kObjRel * std::max(1.0, std::abs(g.optimum))) {
            std::ostringstream os;
            os.precision(17);
            os << tag << ": objective " << res.objective << ", known optimum " << g.optimum;
            audit.fail(os.str(), lp);
        }
    } else if (want == Status::Infeasible) {
        const std::string err = checkFarkas(lp, solver.getDualRay());
        if (!err.empty()) {
            audit.fail(tag + ": Farkas ray: " + err, lp);
        }
    }
    return res.status;
}

void report(const Audit& audit) {
    INFO("models " << audit.models << ", tolerance calls " << audit.tolerance_calls << ", failures "
                   << audit.failures.size());
    for (const auto& f : audit.failures) {
        UNSCOPED_INFO(f);
    }
    CHECK(audit.failures.empty());
}

Options optionsFor(int variant) {
    return Options{.scaled = (variant & 1) != 0,
                   .degenerate = (variant & 2) != 0,
                   .maximize = (variant & 4) != 0};
}

std::string tagFor(const char* kind, std::uint32_t seed, int variant) {
    const Options o = optionsFor(variant);
    std::ostringstream os;
    os << kind << " seed " << seed << (o.scaled ? " scaled" : "")
       << (o.degenerate ? " degenerate" : "") << (o.maximize ? " max" : " min");
    return os.str();
}

// 8 variants per seed. Sweeps of 60000 seeds (1.9M models) were run while
// fixing #237; this many keeps the suite inside its time budget.
constexpr std::uint32_t kSeeds = 300;

// One model through rounds of branching-style bound changes, re-solved warm
// each round and compared with a cold solve of the changed model.
void auditWarm(std::uint32_t seed, int variant, Audit& audit) {
    Rng rng(300000U + seed * 8U + static_cast<std::uint32_t>(variant));
    Generated g = generateOptimal(rng, optionsFor(variant));
    DualSimplexSolver warm;
    warm.setVerbose(false);
    warm.load(g.lp);
    const std::string tag = tagFor("warm", seed, variant);
    ++audit.models;
    if (checkSolve(warm, g, tag, audit) != Status::Optimal) {
        return;
    }
    for (int round = 0; round < 3; ++round) {
        const auto x = warm.getPrimalValues();
        const int changes = rng.pick(1, 2);
        for (int k = 0; k < changes; ++k) {
            const Index j = rng.pick(0, g.lp.num_cols - 1);
            Real lo = g.lp.col_lower[j];
            Real hi = g.lp.col_upper[j];
            const Real step = std::max(1e-3, 0.25 * std::abs(x[j])) * rng.value(0.5, 2.0);
            switch (rng.pick(0, 3)) {
                case 0:  // down branch: cut x_j off from above
                    hi = x[j] - step;
                    break;
                case 1:  // up branch
                    lo = x[j] + step;
                    break;
                case 2:  // fix at the current value
                    lo = hi = x[j];
                    break;
                default:  // relax back to free
                    lo = -kInf;
                    hi = kInf;
                    break;
            }
            if (lo > hi) {
                std::swap(lo, hi);  // keep the box nonempty
            }
            g.lp.col_lower[j] = lo;
            g.lp.col_upper[j] = hi;
            warm.setColBounds(j, lo, hi);
        }
        DualSimplexSolver cold;
        cold.setVerbose(false);
        cold.load(g.lp);
        const LpResult cr = cold.solve();
        const LpResult wr = warm.solve();
        const std::string rtag = tag + " round " + std::to_string(round);
        // Neither solve's Unbounded is known to be right here: check each ray.
        for (const auto* sv : {&warm, &cold}) {
            const char* who = sv == &warm ? " warm" : " cold";
            if (sv->getStatus() == Status::Unbounded) {
                const std::string err = checkPrimalRay(g.lp, sv->getPrimalRay());
                if (!err.empty()) {
                    audit.fail(rtag + who + " primal ray: " + err, g.lp);
                    return;
                }
            } else if (!sv->getPrimalRay().empty()) {
                audit.fail(rtag + who + " primal ray after status " + statusName(sv->getStatus()),
                           g.lp);
                return;
            }
        }
        // Bounds cut at arbitrary points leave models that are feasible, or
        // infeasible, only to within the solver's scaled tolerance. A ray
        // valid up to tolerance on one side then makes a point from the
        // other side a tolerance call rather than a wrong answer.
        const bool split = (cr.status == Status::Optimal && wr.status == Status::Infeasible) ||
                           (cr.status == Status::Infeasible && wr.status == Status::Optimal);
        if (split) {
            const auto& inf = wr.status == Status::Infeasible ? warm : cold;
            if (checkFarkas(g.lp, inf.getDualRay(), true).empty()) {
                ++audit.tolerance_calls;
                return;
            }
        }
        if (cr.status != wr.status) {
            std::ostringstream os;
            os.precision(17);
            os << rtag << ": warm " << statusName(wr.status) << ", cold " << statusName(cr.status);
            for (const auto* sv : {&warm, &cold}) {
                if (sv->getStatus() == Status::Optimal) {
                    os << "\n  " << (sv == &warm ? "warm" : "cold") << " obj " << sv->getObjective()
                       << " certificate: "
                       << checkOptimal(g.lp, sv->getPrimalValues(), sv->getDualValues(),
                                       sv->getObjective());
                    os << "\n  x:";
                    for (Real v : sv->getPrimalValues()) {
                        os << ' ' << v;
                    }
                } else if (sv->getStatus() == Status::Infeasible) {
                    os << "\n  " << (sv == &warm ? "warm" : "cold")
                       << " Farkas: " << checkFarkas(g.lp, sv->getDualRay(), true);
                }
            }
            audit.fail(os.str(), g.lp);
            return;
        }
        if (wr.status == Status::Optimal) {
            const std::string err = checkOptimal(g.lp, warm.getPrimalValues(), warm.getDualValues(),
                                                 wr.objective, kWarmRel);
            if (!err.empty()) {
                audit.fail(rtag + ": warm optimal certificate: " + err, g.lp);
                return;
            }
            // Both optima hold to tolerance, so they may differ by as much
            // as the tolerance moves the optimum; flag it only when the cold
            // one fails its own certificate.
            const Real obj_gap =
                std::abs(wr.objective - cr.objective) / std::max(1.0, std::abs(cr.objective));
            if (obj_gap > kObjRel && obj_gap <= kWarmObjRel &&
                checkOptimal(g.lp, cold.getPrimalValues(), cold.getDualValues(), cr.objective,
                             kWarmRel)
                    .empty()) {
                ++audit.tolerance_calls;
            } else if (std::abs(wr.objective - cr.objective) >
                       kObjRel * std::max(1.0, std::abs(cr.objective))) {
                std::ostringstream os;
                os.precision(17);
                os << rtag << ": warm objective " << wr.objective << ", cold " << cr.objective;
                audit.fail(os.str(), g.lp);
                return;
            }
        } else if (wr.status == Status::Infeasible) {
            const std::string err = checkFarkas(g.lp, warm.getDualRay(), true);
            if (!err.empty()) {
                std::ostringstream os;
                os.precision(17);
                os << rtag << ": warm Farkas ray: " << err
                   << " (cold ray: " << checkFarkas(g.lp, cold.getDualRay(), true) << ")\n  y:";
                for (Real v : warm.getDualRay()) {
                    os << ' ' << v;
                }
                audit.fail(os.str(), g.lp);
            }
            return;
        } else {
            return;
        }
    }
}

// Empty when the solver's verdict carries a certificate valid to tolerance for
// a model of unknown status: a point at kWarmRel, a Farkas ray up to
// tolerance, or a primal ray. Any other status certifies nothing.
std::string certify(const DualSimplexSolver& sv, const LpProblem& lp) {
    switch (sv.getStatus()) {
        case Status::Optimal:
            return checkOptimal(lp, sv.getPrimalValues(), sv.getDualValues(), sv.getObjective(),
                                kWarmRel);
        case Status::Infeasible:
            return checkFarkas(lp, sv.getDualRay(), true);
        case Status::Unbounded:
            return checkPrimalRay(lp, sv.getPrimalRay());
        default:
            return std::string("status ") + statusName(sv.getStatus());
    }
}

// One model through rounds of cut-like rows added with addRows (#241), as
// branch-and-cut adds them: each round cuts off the current LP point by a
// margin, is re-solved warm, and is compared with a cold solve of the
// extended model. A warm answer fails only when it carries no certificate
// while the cold one does; two certified answers that differ are tolerance
// calls, since cuts at arbitrary points leave models that are feasible only
// to within tolerance.
void auditCuts(std::uint32_t seed, int variant, Audit& audit) {
    Rng rng(700000U + seed * 8U + static_cast<std::uint32_t>(variant));
    const Options opt = optionsFor(variant);
    Generated g = generateOptimal(rng, opt);
    DualSimplexSolver warm;
    warm.setVerbose(false);
    warm.load(g.lp);
    const std::string tag = tagFor("cuts", seed, variant);
    ++audit.models;
    if (checkSolve(warm, g, tag, audit) != Status::Optimal) {
        return;
    }
    const Index n = g.lp.num_cols;
    for (int round = 0; round < 3; ++round) {
        const auto x = warm.getPrimalValues();
        std::vector<Index> idx;
        std::vector<Real> val;
        long double gx = 0.0L;
        Real gmax = 0.0;
        for (Index j = 0; j < n; ++j) {
            if (!rng.coin(0.6)) {
                continue;
            }
            Real v = rng.value(-5.0, 5.0);
            if (v == 0.0) {
                continue;
            }
            if (opt.scaled) {
                v *= std::pow(10.0, rng.pick(-4, 4));
            }
            idx.push_back(j);
            val.push_back(v);
            gx += static_cast<long double>(v) * x[j];
            gmax = std::max(gmax, std::abs(v * x[j]));
        }
        if (idx.empty()) {
            const Index j = rng.pick(0, n - 1);
            idx.push_back(j);
            val.push_back(1.0);
            gx = x[j];
            gmax = std::abs(x[j]);
        }
        const Real delta = 0.05 * std::max(1.0, gmax) * rng.uniform(0.25, 2.0);
        Real lo = -kInf;
        Real hi = kInf;
        if (rng.coin(0.5)) {
            lo = static_cast<Real>(gx) + delta;
        } else {
            hi = static_cast<Real>(gx) - delta;
        }
        const std::vector<Index> starts{0};
        const std::vector<Real> los{lo};
        const std::vector<Real> his{hi};
        warm.addRows(starts, idx, val, los, his);
        auto rows = denseRows(g.lp);
        rows.emplace_back(n, 0.0);
        for (std::size_t k = 0; k < idx.size(); ++k) {
            rows.back()[idx[k]] = val[k];
        }
        setRows(g.lp, rows);
        g.lp.row_lower.push_back(lo);
        g.lp.row_upper.push_back(hi);

        DualSimplexSolver cold;
        cold.setVerbose(false);
        cold.load(g.lp);
        const LpResult cr = cold.solve();
        const LpResult wr = warm.solve();
        const std::string rtag = tag + " round " + std::to_string(round);
        const std::string werr = certify(warm, g.lp);
        const std::string cerr = certify(cold, g.lp);
        if (!werr.empty() && cerr.empty()) {
            std::ostringstream os;
            os.precision(17);
            os << rtag << ": warm " << statusName(wr.status) << " uncertified (" << werr
               << "), cold " << statusName(cr.status) << " obj " << cr.objective;
            auto dump = [&os](const char* name, const std::vector<Real>& v) {
                os << "\n  " << name << ':';
                for (Real e : v) {
                    os << ' ' << e;
                }
            };
            if (wr.status == Status::Optimal) {
                os << "\n  warm obj " << wr.objective;
                dump("x", warm.getPrimalValues());
                dump("y", warm.getDualValues());
            } else if (wr.status == Status::Infeasible) {
                dump("ray", warm.getDualRay());
            } else if (wr.status == Status::Unbounded) {
                dump("ray", warm.getPrimalRay());
            }
            audit.fail(os.str(), g.lp);
            return;
        }
        if (!werr.empty() || wr.status != Status::Optimal) {
            return;  // nothing further to cut, or no certified point to cut
        }
        if (cerr.empty() &&
            (cr.status != Status::Optimal || std::abs(wr.objective - cr.objective) >
                                                 kObjRel * std::max(1.0, std::abs(cr.objective)))) {
            ++audit.tolerance_calls;
        }
    }
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

// The interior-point backends on the same generators (#239). They return no
// rays, so the audit checks the status against the known one: a limit or
// Error is allowed, any other wrong status is not. An Optimal must also
// match the known optimum.
Generated generate(Shape shape, std::uint32_t seed, int variant) {
    const Options opt = optionsFor(variant);
    const auto v = static_cast<std::uint32_t>(variant);
    switch (shape) {
        case Shape::Optimal: {
            Rng rng(seed * 8U + v + 1U);
            return generateOptimal(rng, opt);
        }
        case Shape::Infeasible: {
            Rng rng(100000U + seed * 8U + v);
            return generateInfeasible(rng, opt);
        }
        case Shape::Unbounded:
            break;
    }
    Rng rng(200000U + seed * 8U + v);
    return generateUnbounded(rng, opt);
}

Status checkStatus(LpSolver& solver, const Generated& g, const std::string& tag, Real obj_rel,
                   Audit& audit) {
    const LpResult res = solver.solve();
    const auto want = g.shape == Shape::Optimal      ? Status::Optimal
                      : g.shape == Shape::Infeasible ? Status::Infeasible
                                                     : Status::Unbounded;
    const bool definite = res.status == Status::Optimal || res.status == Status::Infeasible ||
                          res.status == Status::Unbounded;
    if (definite && res.status != want) {
        audit.fail(tag + ": status " + statusName(res.status) + ", expected " + statusName(want),
                   g.lp);
    } else if (res.status == Status::Optimal &&
               std::abs(res.objective - g.optimum) > obj_rel * std::max(1.0, std::abs(g.optimum))) {
        std::ostringstream os;
        os.precision(17);
        os << tag << ": objective " << res.objective << ", known optimum " << g.optimum;
        audit.fail(os.str(), g.lp);
    }
    return res.status;
}

// PDLP gets a fixed iteration budget, not a time limit, so the audit does not
// depend on the machine. Its Optimal is held to 1e-4, the accuracy its
// default tolerances promise (#205).
constexpr std::uint32_t kPdlpSeeds = 100;
constexpr Real kPdlpObjRel = 1e-4;

PdlpOptions pdlpOptions() {
    PdlpOptions opts;
    opts.use_gpu = false;
    opts.verbose = false;
    opts.max_iter = 20000;
    opts.polish_max_iter = 20000;
    return opts;
}

constexpr Shape kShapes[] = {Shape::Optimal, Shape::Infeasible, Shape::Unbounded};

const char* shapeName(Shape shape) {
    return shape == Shape::Optimal      ? "optimal"
           : shape == Shape::Infeasible ? "infeasible"
                                        : "unbounded";
}

}  // namespace

TEST_CASE("LP validity: optimal LPs give a valid primal-dual certificate", "[lp_validity]") {
    Audit audit;
    for (std::uint32_t seed = 0; seed < kSeeds; ++seed) {
        for (int variant = 0; variant < 8; ++variant) {
            Rng rng(seed * 8U + static_cast<std::uint32_t>(variant) + 1U);
            const Generated g = generateOptimal(rng, optionsFor(variant));
            DualSimplexSolver solver;
            solver.setVerbose(false);
            solver.load(g.lp);
            ++audit.models;
            checkSolve(solver, g, tagFor("optimal", seed, variant), audit);
        }
    }
    report(audit);
}

TEST_CASE("LP validity: infeasible LPs give a valid Farkas ray", "[lp_validity]") {
    Audit audit;
    for (std::uint32_t seed = 0; seed < kSeeds; ++seed) {
        for (int variant = 0; variant < 8; ++variant) {
            Rng rng(100000U + seed * 8U + static_cast<std::uint32_t>(variant));
            const Generated g = generateInfeasible(rng, optionsFor(variant));
            DualSimplexSolver solver;
            solver.setVerbose(false);
            solver.load(g.lp);
            ++audit.models;
            checkSolve(solver, g, tagFor("infeasible", seed, variant), audit);
        }
    }
    report(audit);
}

TEST_CASE("LP validity: unbounded LPs give a valid primal ray", "[lp_validity]") {
    Audit audit;
    for (std::uint32_t seed = 0; seed < kSeeds; ++seed) {
        for (int variant = 0; variant < 8; ++variant) {
            Rng rng(200000U + seed * 8U + static_cast<std::uint32_t>(variant));
            const Generated g = generateUnbounded(rng, optionsFor(variant));
            DualSimplexSolver solver;
            solver.setVerbose(false);
            solver.load(g.lp);
            ++audit.models;
            checkSolve(solver, g, tagFor("unbounded", seed, variant), audit);
        }
    }
    report(audit);
}

TEST_CASE("LP validity: warm re-solves after bound changes match a cold solve", "[lp_validity]") {
    Audit audit;
    for (std::uint32_t seed = 0; seed < kSeeds; ++seed) {
        for (int variant = 0; variant < 8; ++variant) {
            auditWarm(seed, variant, audit);
        }
    }
    report(audit);
}

TEST_CASE("LP validity: warm re-solves after addRows are certified when a cold solve is",
          "[lp_validity]") {
    Audit audit;
    for (std::uint32_t seed = 0; seed < kSeeds; ++seed) {
        for (int variant = 0; variant < 8; ++variant) {
            auditCuts(seed, variant, audit);
        }
    }
    report(audit);
}

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

TEST_CASE("Regression #237: a basic variable just past its bound blocks the primal ray",
          "[lp_validity]") {
    // In the primal phase, row 2's logical sat a hair above its upper bound,
    // so its ratio-test step came out below -1e-7 and the row was skipped as
    // not blocking: Unbounded on a bounded LP (optimum 10.2109375).
    const LpProblem lp =
        literalLp(Sense::Maximize, 0.75, {-0.097187499999999996, -47812.5, 0.004875, 0.0, -7500},
                  {-225.5922724253347, -kInf, -kInf, -kInf, -1.2500000000000001e-05},
                  {-225.5922724253347, kInf, kInf, 0.0061250000000000002, 0.00033750000000000002},
                  {-kInf, -kInf, 0.080937499999999996, 0.0087187500000000008},
                  {-26562.5, kInf, 0.093437500000000007, 0.0092187499999999995},
                  {{0, 0, -362.5},
                   {0, 1, -412500000},
                   {1, 2, 3.875},
                   {1, 3, -1250000},
                   {2, 0, -0.00022499999999999999},
                   {2, 1, 162.5},
                   {2, 2, 3.2500000000000004e-05},
                   {3, 0, -4.125e-05},
                   {3, 3, -3.125}});
    requireCertifiedOptimal(lp);
}

TEST_CASE("Regression #237: a small entry that blocks first is pivoted on, not overshot",
          "[lp_validity]") {
    // The optimal basis pivots on row 0's -7.5e-9. The primal ratio test
    // skipped it and stepped past row 0's bound; the dual phase stepped back,
    // and the two cycled until the iteration limit.
    const LpProblem lp =
        literalLp(Sense::Maximize, -6.125, {-8.4375000000000004e-05, 4656.25}, {-kInf, -kInf},
                  {-20000, -0.0040000000000000001}, {-kInf, -137500, -kInf, -kInf},
                  {-0.00067500000000000004, kInf, kInf, 875},
                  {{0, 0, -7.500000000000001e-09},
                   {0, 1, 0.22500000000000003},
                   {1, 0, 3.875},
                   {2, 1, -0.48750000000000004},
                   {3, 0, -0.022500000000000003}});
    requireCertifiedOptimal(lp);
    DualSimplexSolver solver;
    CHECK(std::abs(solveLiteral(solver, lp).objective - -22.21875) < 1e-6);
}

TEST_CASE("Regression #238: the primal ray does not outlive a change to the model",
          "[lp_validity]") {
    // min -x0 s.t. x0 - x1 <= 1, x >= 0: unbounded along r = (1, 1).
    const LpProblem lp = literalLp(Sense::Minimize, 0.0, {-1.0, 0.0}, {0.0, 0.0}, {kInf, kInf},
                                   {-kInf}, {1.0}, {{0, 0, 1.0}, {0, 1, -1.0}});
    const std::vector<Index> starts{0, 1};
    const std::vector<Index> idx{0};
    const std::vector<Real> vals{1.0};
    const std::vector<Real> lo{-kInf};
    const std::vector<Real> up{5.0};
    for (int change = 0; change < 4; ++change) {
        DualSimplexSolver solver;
        REQUIRE(solveLiteral(solver, lp).status == Status::Unbounded);
        REQUIRE(checkPrimalRay(lp, solver.getPrimalRay()) == "");
        switch (change) {
            case 0:
                solver.addRows(starts, idx, vals, lo, up);  // x0 <= 5 stops the ray
                break;
            case 1: {
                const std::vector<Index> rows{0};
                solver.removeRows(rows);
                break;
            }
            case 2:
                solver.setColBounds(0, 0.0, 5.0);
                break;
            default: {
                const std::vector<Real> obj{1.0, 0.0};
                solver.setObjective(obj);
                break;
            }
        }
        INFO("change " << change);
        CHECK(solver.getPrimalRay().empty());
    }
}

TEST_CASE("Regression #243: a cold solve of a one-point feasible set is not Infeasible",
          "[lp_validity]") {
    // x2 is fixed, row 4 fixes x0 at its upper bound, and row 3 is tight at
    // x1's lower bound: x = (0.03875, 16250, 0.0003375) is the only feasible
    // point. A cold solve returned Infeasible (audit seed 27267, variant 7).
    requireCertifiedOptimal(literalLp(Sense::Maximize, 5.25, {-567.1875, 0.000125, -30000},
                                      {0.02375, 16250, 0.00033750000000000002},
                                      {0.03875, 32500, 0.00033750000000000002},
                                      {-kInf, -kInf, -kInf, -kInf, -121.09375},
                                      {kInf, kInf, kInf, -0.00059843750000000001, -121.09375},
                                      {{0, 1, -0.625},
                                       {1, 1, 2.5000000000000002e-06},
                                       {2, 0, 0.26250000000000001},
                                       {3, 1, 1.2500000000000001e-08},
                                       {3, 2, -2.375},
                                       {4, 0, -3125}}));
}

TEST_CASE("Regression #243: a row repaired exactly to its bound is not a Farkas proof",
          "[lp_validity]") {
    // Audit seed 44060, variant 3. Like the case above, a cold solve returned
    // Infeasible: the skipped small entries could repair the leaving row by
    // exactly its violation, and rounding put the repair just short of it.
    requireCertifiedOptimal(literalLp(
        Sense::Minimize, 9.75,
        {0.00013750000000000001, -146093.75, 5250, 0.00037500000000000001, 0.00021875},
        {-28750, -0.0001125, -0.00075000000000000002, -kInf, -2875},
        {-28750, -0.0001125, -0.00075000000000000002, -22500, -2000},
        {-kInf, 0.00013593750000000002, -kInf, -435.9375, -kInf, -0.0087500000000000008, 69531.25},
        {kInf, 0.00013593750000000002, kInf, kInf, kInf, -0.0087500000000000008, kInf},
        {{0, 1, 25000},
         {0, 2, -625},
         {0, 3, -0.0004125},
         {1, 1, -3.8750000000000004},
         {1, 2, -0.20000000000000001},
         {1, 3, 2e-08},
         {2, 0, 2},
         {2, 3, 0.125},
         {2, 4, 28.75},
         {3, 0, 0.033750000000000002},
         {3, 1, -4750000},
         {4, 0, -1.0000000000000002e-06},
         {4, 1, -462.5},
         {4, 2, -48.75},
         {4, 3, 4.2500000000000008e-06},
         {4, 4, -2.5000000000000002e-06},
         {5, 1, 400},
         {5, 2, -45},
         {5, 4, -1.2500000000000001e-06},
         {6, 0, -2.125},
         {6, 1, -75000000}}));
}

TEST_CASE("Regression #245: observed incumbents hold the model as loaded", "[lp_validity]") {
    // A model of the #244 sweep: presolved, its root LP incumbent held the
    // reduced model but missed row 2 of this one by 1.24e-6.
    const std::uint32_t seed = 14373;
    const int variant = 5;
    Rng rng(500000U + seed * 8U + static_cast<std::uint32_t>(variant));
    Generated g = generateOptimal(rng, optionsFor(variant));
    for (Index j = 0; j < g.lp.num_cols; ++j) {
        if (rng.coin(0.6)) {
            g.lp.col_type[j] = VarType::Integer;
        }
    }
    const LpProblem& lp = g.lp;
    MipSolver solver;
    solver.setVerbose(false);
    solver.setNumThreads(1);
    solver.setNodeLimit(300);
    solver.setTimeLimit(60.0);
    Int observed = 0;
    // The gate's own test: every bound and row to 1e-6, integers to 1e-6.
    constexpr Real kTol = 1e-6;
    solver.setIncumbentObserver([&](const std::vector<Real>& x, const std::string& source) {
        ++observed;
        INFO("source " << source);
        REQUIRE(static_cast<Index>(x.size()) == lp.num_cols);
        for (Index j = 0; j < lp.num_cols; ++j) {
            CHECK(x[j] >= lp.col_lower[j] - kTol);
            CHECK(x[j] <= lp.col_upper[j] + kTol);
            if (lp.col_type[j] != VarType::Continuous) {
                CHECK(std::abs(x[j] - std::round(x[j])) <= kTol);
            }
        }
        for (Index i = 0; i < lp.num_rows; ++i) {
            const auto row = lp.matrix.row(i);
            Real activity = 0.0;
            for (Index k = 0; k < row.size(); ++k) {
                activity += row.values[k] * x[row.indices[k]];
            }
            INFO("row " << i << " activity " << activity);
            CHECK(activity >= lp.row_lower[i] - kTol);
            CHECK(activity <= lp.row_upper[i] + kTol);
        }
    });
    solver.load(lp);
    const auto result = solver.solve();
    CHECK(result.status == Status::Optimal);
    CHECK(observed > 0);
}

TEST_CASE("LP validity: barrier gives no wrong status on generated LPs", "[lp_validity]") {
    // Seeds 83 (variant 5) and 90 (variant 6) of the optimal shape crashed
    // in crossover (#239).
    Audit audit;
    for (const Shape shape : kShapes) {
        for (std::uint32_t seed = 0; seed < 100; ++seed) {
            for (int variant = 0; variant < 8; ++variant) {
                const Generated g = generate(shape, seed, variant);
                BarrierSolver solver;
                BarrierOptions opts;
                opts.verbose = false;
                solver.setOptions(opts);
                solver.load(g.lp);
                ++audit.models;
                checkStatus(solver, g, tagFor(shapeName(shape), seed, variant), kObjRel, audit);
            }
        }
    }
    report(audit);
}

TEST_CASE("LP validity: PDLP gives no wrong status on generated LPs", "[lp_validity]") {
    // Rays were accepted on absolute tolerances: Optimal and Unbounded
    // models came back Infeasible or Unbounded, and the dual ray took the
    // objective's sign, so no maximization was found infeasible (#239).
    // Termination compared residual norms with the norm of c, which hid the
    // dual infeasibility of an unbounded model (#239, #205).
    Audit audit;
    int proved = 0;
    for (const Shape shape : kShapes) {
        for (std::uint32_t seed = 0; seed < kPdlpSeeds; ++seed) {
            for (int variant = 0; variant < 8; ++variant) {
                const Generated g = generate(shape, seed, variant);
                PdlpSolver solver;
                solver.setOptions(pdlpOptions());
                solver.load(g.lp);
                ++audit.models;
                const Status st = checkStatus(solver, g, tagFor(shapeName(shape), seed, variant),
                                              kPdlpObjRel, audit);
                if (g.shape != Shape::Optimal &&
                    st == (g.shape == Shape::Infeasible ? Status::Infeasible : Status::Unbounded)) {
                    ++proved;
                }
            }
        }
    }
    report(audit);
    // At least 3 in 4 of the infeasible and unbounded models are proved
    // within the budget (1504 of 1600 when written; 1490 since a primal ray
    // also needs a feasible point, #246).
    CHECK(proved >= static_cast<int>(kPdlpSeeds) * 12);
}
