#include "mipx/barrier.h"

#include "mipx/dual_simplex.h"
#include "newton_solver.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <numeric>
#include <stop_token>
#include <string>
#include <thread>
#include <utility>

namespace mipx {

namespace {

inline bool isFinite(Real v) {
    return std::isfinite(v);
}

inline Real infNorm(std::span<const Real> v) {
    Real n = 0.0;
    for (Real x : v) {
        n = std::max(n, std::abs(x));
    }
    return n;
}

inline Real dot(std::span<const Real> a, std::span<const Real> b) {
    Real s = 0.0;
    for (Index i = 0; i < static_cast<Index>(a.size()); ++i) {
        s += a[i] * b[i];
    }
    return s;
}

inline Real computeOriginalObjective(const LpProblem& problem, std::span<const Real> x) {
    Real obj = problem.obj_offset;
    for (Index j = 0; j < problem.num_cols; ++j) {
        obj += problem.obj[j] * x[j];
    }
    return obj;
}

inline Real maxStepToBoundary(std::span<const Real> x, std::span<const Real> dx, Real fraction) {
    Real alpha = 1.0;
    for (Index i = 0; i < static_cast<Index>(x.size()); ++i) {
        if (dx[i] < 0.0) {
            alpha = std::min(alpha, -fraction * x[i] / dx[i]);
        }
    }
    return std::clamp(alpha, 0.0, 1.0);
}

// ============================================================================
// Ruiz equilibration scaling
// ============================================================================

void computeRuizScaling(SparseMatrix& A, std::vector<Real>& b, std::vector<Real>& c,
                        std::vector<Real>& row_scale, std::vector<Real>& col_scale,
                        Int iterations) {
    const Index m = A.numRows();
    const Index n = A.numCols();
    row_scale.assign(static_cast<size_t>(m), 1.0);
    col_scale.assign(static_cast<size_t>(n), 1.0);

    if (iterations <= 0 || m == 0 || n == 0) {
        return;
    }

    // Work on CSR data directly.  After scaling we rebuild the matrix.
    auto vals = A.csr_values();
    auto cols = A.csr_col_indices();
    auto rows = A.csr_row_starts();
    std::vector<Real> values(vals.begin(), vals.end());
    std::vector<Index> col_indices(cols.begin(), cols.end());
    std::vector<Index> row_starts(rows.begin(), rows.end());

    for (Int iter = 0; iter < iterations; ++iter) {
        // Row scaling: r_i = 1 / sqrt(max_j |A[i,j]|)
        std::vector<Real> r(static_cast<size_t>(m), 0.0);
        for (Index i = 0; i < m; ++i) {
            Real mx = 0.0;
            for (Index p = row_starts[i]; p < row_starts[i + 1]; ++p) {
                mx = std::max(mx, std::abs(values[p]));
            }
            r[i] = (mx > 1e-30) ? 1.0 / std::sqrt(mx) : 1.0;
        }

        // Column scaling: c_j = 1 / sqrt(max_i |A[i,j]|)
        std::vector<Real> cs(static_cast<size_t>(n), 0.0);
        for (Index i = 0; i < m; ++i) {
            for (Index p = row_starts[i]; p < row_starts[i + 1]; ++p) {
                Index j = col_indices[p];
                cs[j] = std::max(cs[j], std::abs(values[p]) * r[i]);
            }
        }
        for (Index j = 0; j < n; ++j) {
            cs[j] = (cs[j] > 1e-30) ? 1.0 / std::sqrt(cs[j]) : 1.0;
        }

        // Apply: A[i,j] *= r[i] * cs[j]
        for (Index i = 0; i < m; ++i) {
            for (Index p = row_starts[i]; p < row_starts[i + 1]; ++p) {
                values[p] *= r[i] * cs[col_indices[p]];
            }
        }

        // Accumulate into overall scale factors.
        for (Index i = 0; i < m; ++i) {
            row_scale[i] *= r[i];
        }
        for (Index j = 0; j < n; ++j) {
            col_scale[j] *= cs[j];
        }
    }

    // Rebuild matrix.
    A = SparseMatrix(m, n, std::move(values), std::move(col_indices), std::move(row_starts));

    // Scale b and c.
    for (Index i = 0; i < m; ++i) {
        b[i] *= row_scale[i];
    }
    for (Index j = 0; j < n; ++j) {
        c[j] *= col_scale[j];
    }
}

// ============================================================================
// Density detection for backend auto-selection
// ============================================================================

struct DensityInfo {
    Index dense_cols = 0;
    Index dense_rows = 0;
    Real avg_ne_row_density = 0.0;
};

DensityInfo detectDensity(const SparseMatrix& A, Real dense_fraction) {
    const Index m = A.numRows();
    const Index n = A.numCols();
    DensityInfo info;

    Index thresh = std::max<Index>(1, static_cast<Index>(dense_fraction * m));
    for (Index j = 0; j < n; ++j) {
        auto cv = A.col(j);
        if (cv.size() > thresh) {
            ++info.dense_cols;
        }
    }

    Index col_thresh = std::max<Index>(1, static_cast<Index>(dense_fraction * n));
    for (Index i = 0; i < m; ++i) {
        auto rv = A.row(i);
        if (rv.size() > col_thresh) {
            ++info.dense_rows;
        }
    }

    // Estimate NE row density.
    if (m > 0) {
        double total_nnz = 0.0;
        // Quick estimate: for each row, NE row nnz ≈ sum of intersections.
        // Approximate: nnz(NE) ≈ nnz(A)^2 / n for random structure.
        double a_nnz = static_cast<double>(A.numNonzeros());
        double a_density = a_nnz / (static_cast<double>(m) * n);
        total_nnz = a_density * a_nnz;
        info.avg_ne_row_density = total_nnz / m;
    }

    return info;
}

// ============================================================================
// Shared Mehrotra predictor-corrector IPM loop
// ============================================================================

// Mehrotra's starting point (Mehrotra 1992, sec. 7; Wright, "Primal-Dual
// Interior-Point Methods", ch. 10): the least-squares solutions of A z = b and
// A'y + s = c, shifted to be positive and then centred. A start at z = s = 1
// ignores b and c, so on badly scaled models the first steps are tiny and mu
// grows instead of shrinking (#204).
//
// With upper bounds (#299), each bounded column j also gets its slack
// v_j = u_j - z_j and the dual w_j of z_j <= u_j, with s_j - w_j the
// least-squares reduced cost split into its positive and negative parts. The
// shifts and centring then run over (z, v) and (s, w) together, which keeps
// s - w and so the dual residual.
bool mehrotraStartingPoint(NewtonSolver& solver, std::span<const Real> b, std::span<const Real> c,
                           std::span<const Real> u, std::span<const Index> upper_cols, Real reg,
                           std::vector<Real>& z, std::vector<Real>& y, std::vector<Real>& s,
                           std::vector<Real>& v, std::vector<Real>& w) {
    const auto m = b.size();
    const auto n = c.size();
    const std::vector<Real> ones(n, 1.0);
    const std::vector<Real> zero_m(m, 0.0);
    const std::vector<Real> zero_n(n, 0.0);
    std::vector<Real> dz(n);
    std::vector<Real> dy(m);
    std::vector<Real> ds(n);

    // With z = s = 1 the Newton matrix is A A' (+ reg I). Retry with boosted
    // regularization, as the main loop does.
    if (!solver.factorize(ones, ones, reg) && !solver.factorize(ones, ones, reg * 100.0)) {
        return false;
    }
    // rp = b, rd = rc = 0 gives dz = A'(AA')^{-1} b, the least-norm z.
    if (!solver.solveNewton(b, zero_n, zero_n, dz, dy, ds)) {
        return false;
    }
    z = dz;
    // rp = rc = 0, rd = c gives dy = (AA')^{-1} A c and ds = c - A'dy.
    if (!solver.solveNewton(zero_m, c, zero_n, dz, dy, ds)) {
        return false;
    }
    y = dy;
    s = ds;
    v.assign(n, 0.0);
    w.assign(n, 0.0);
    for (const Index j : upper_cols) {
        v[j] = u[j] - z[j];
        w[j] = std::max(-s[j], 0.0);
        s[j] = std::max(s[j], 0.0);
    }

    Real min_z = 0.0;
    Real min_s = 0.0;
    for (size_t j = 0; j < n; ++j) {
        min_z = std::min(min_z, z[j]);
        min_s = std::min(min_s, s[j]);
    }
    for (const Index j : upper_cols) {
        min_z = std::min(min_z, v[j]);
    }
    const Real shift_z = -1.5 * min_z;
    const Real shift_s = -1.5 * min_s;
    Real sum_z = 0.0;
    Real sum_s = 0.0;
    Real zs = 0.0;
    for (size_t j = 0; j < n; ++j) {
        z[j] += shift_z;
        s[j] += shift_s;
        sum_z += z[j];
        sum_s += s[j];
        zs += z[j] * s[j];
    }
    for (const Index j : upper_cols) {
        v[j] += shift_z;
        w[j] += shift_s;
        sum_z += v[j];
        sum_s += w[j];
        zs += v[j] * w[j];
    }
    // The centring shifts are undefined when z or s is zero everywhere
    // (b = 0 or c in the row space of A); fall back to a unit start then.
    if (!(sum_z > 0.0) || !(sum_s > 0.0) || !(zs > 0.0) || !std::isfinite(zs)) {
        z.assign(n, 1.0);
        s.assign(n, 1.0);
        y.assign(m, 0.0);
        for (const Index j : upper_cols) {
            v[j] = 1.0;
            w[j] = 1.0;
        }
        return true;
    }
    const Real centre_z = 0.5 * zs / sum_s;
    const Real centre_s = 0.5 * zs / sum_z;
    for (size_t j = 0; j < n; ++j) {
        z[j] += centre_z;
        s[j] += centre_s;
    }
    for (const Index j : upper_cols) {
        v[j] += centre_z;
        w[j] += centre_s;
    }
    return true;
}

// The bounded columns' part of a Newton step (#299). With upper slack v and
// its dual w, the step solves
//   A dz = rp,  dz + dv = ru,  A'dy + ds - dw = rd,
//   S dz + Z ds = rc,  W dv + V dw = rw.
// Eliminating ds = (rc - S dz)/Z, dv = ru - dz and dw = (rw - W dv)/V leaves
//   A'dy - (S/Z + W/V) dz = rd - rc/Z + (rw - W ru)/V,
// which the backend's system with z = 1, s = S/Z + W/V and rc = 0 has, so
// every NewtonSolver serves unchanged. Columns without an upper bound pass
// through as they are.
struct UpperBoundStep {
    std::vector<Real> z_eff;
    std::vector<Real> s_eff;
    std::vector<Real> rd_eff;
    std::vector<Real> rc_eff;

    void scaling(std::span<const Index> cols, std::span<const Real> z, std::span<const Real> s,
                 std::span<const Real> v, std::span<const Real> w) {
        z_eff.assign(z.begin(), z.end());
        s_eff.assign(s.begin(), s.end());
        for (const Index j : cols) {
            z_eff[j] = 1.0;
            s_eff[j] = s[j] / z[j] + w[j] / v[j];
        }
    }

    // rd_eff, rc_eff for residuals rd, rc and the bound rows' ru, rw.
    void fold(std::span<const Index> cols, std::span<const Real> z, std::span<const Real> v,
              std::span<const Real> w, std::span<const Real> rd, std::span<const Real> rc,
              std::span<const Real> ru, std::span<const Real> rw) {
        rd_eff.assign(rd.begin(), rd.end());
        rc_eff.assign(rc.begin(), rc.end());
        for (const Index j : cols) {
            rd_eff[j] = rd[j] - rc[j] / z[j] + (rw[j] - w[j] * ru[j]) / v[j];
            rc_eff[j] = 0.0;
        }
    }

    // ds, dv, dw of the bounded columns once the backend has given dz.
    static void recover(std::span<const Index> cols, std::span<const Real> z,
                        std::span<const Real> s, std::span<const Real> v, std::span<const Real> w,
                        std::span<const Real> rc, std::span<const Real> ru,
                        std::span<const Real> rw, std::span<const Real> dz, std::span<Real> ds,
                        std::span<Real> dv, std::span<Real> dw) {
        for (const Index j : cols) {
            ds[j] = (rc[j] - s[j] * dz[j]) / z[j];
            dv[j] = ru[j] - dz[j];
            dw[j] = (rw[j] - w[j] * dv[j]) / v[j];
        }
    }
};

inline Real maxStepOn(std::span<const Index> cols, std::span<const Real> x,
                      std::span<const Real> dx, Real fraction, Real alpha) {
    for (const Index j : cols) {
        if (dx[j] < 0.0) {
            alpha = std::min(alpha, -fraction * x[j] / dx[j]);
        }
    }
    return std::clamp(alpha, 0.0, 1.0);
}

// Gondzio's multiple centrality correctors (J. Gondzio, "Multiple centrality
// corrections in a primal-dual method for linear programming", COAP 6, 1996;
// as restated in M. Colombo and J. Gondzio, "Further development of multiple
// centrality correctors for interior point methods", COAP 41, 2008, sec. 2.2).
//
// Given the predictor-corrector direction d with feasible steps a_p, a_d,
// aim at the enlarged steps a~ = min(a + delta, 1) in each space, form the
// trial products v~ = (z + a~_p dz)(s + a~_d ds), and move only the outliers
// into [beta_min, beta_max] * sigma mu (CG08 eq. 7, gamma = 0.1 there):
//   t = beta_min sigma mu - v~  where v~ <= beta_min sigma mu,
//   t = beta_max sigma mu - v~  where v~ >= beta_max sigma mu,  0 otherwise.
// The corrector d_m solves the Newton system for (0, 0, t) with the current
// factorization. In each space short of a full step, d + d_m replaces d when
// that space's step grows by at least a fraction of the increase aimed at;
// correcting goes on, at most k times, while either space takes it. t has no
// lower cap: CG08 eq. 7 states none. The bound pairs (v, w) of #299 are
// complementarity pairs like (z, s) and take part the same way.
struct GondzioCorrectors {
    static constexpr Real kAspiration = 0.1;  // delta, Gondzio 1996
    static constexpr Real kBetaMin = 0.1;     // gamma of CG08's neighbourhood
    static constexpr Real kBetaMax = 10.0;    // 1 / gamma
    static constexpr Real kAcceptFraction = 0.1;

    std::span<const Index> cols;  // bounded columns
    std::span<const Real> z, s, v, w;
    std::span<Real> dz, dy, ds, dv, dw;

    std::vector<Real> zero_m{}, zero_n{}, t_c{}, t_w{};
    std::vector<Real> mz{}, my{}, ms{}, mv{}, mw{};  // corrector
    std::vector<Real> tz{}, ts{}, tv{}, tw{};        // d + d_m

    void steps(std::span<const Real> ddz, std::span<const Real> dds, std::span<const Real> ddv,
               std::span<const Real> ddw, Real& a_p, Real& a_d) const {
        a_p = maxStepToBoundary(z, ddz, 1.0);
        a_d = maxStepToBoundary(s, dds, 1.0);
        if (!cols.empty()) {
            a_p = maxStepOn(cols, v, ddv, 1.0, a_p);
            a_d = maxStepOn(cols, w, ddw, 1.0, a_d);
        }
    }

    static Real target(Real product, Real lo, Real hi) {
        if (product <= lo) {
            return lo - product;
        }
        return product >= hi ? hi - product : 0.0;
    }

    // Correctors accepted into d. solve(rp, rd, rc, ru, rw, dz, dy, ds, dv, dw)
    // is the Newton solve with the current factorization.
    template <class Solve>
    Int correct(Int k_max, Real sigma_mu, const Solve& solve) {
        const auto n = dz.size();
        const auto m = dy.size();
        if (zero_n.size() != n || zero_m.size() != m) {
            zero_m.assign(m, 0.0);
            zero_n.assign(n, 0.0);
            for (auto* vec : {&t_c, &t_w, &mz, &ms, &mv, &mw, &tz, &ts, &tv, &tw}) {
                vec->assign(n, 0.0);
            }
            my.assign(m, 0.0);
        }
        // sigma mu = 0 (sigma clamped to 0) has no neighbourhood to aim at:
        // every target would be -v~, a second affine step, not a centring one.
        if (!(sigma_mu > 0.0)) {
            return 0;
        }
        const Real lo = kBetaMin * sigma_mu;
        const Real hi = kBetaMax * sigma_mu;
        Real a_p = 0.0;
        Real a_d = 0.0;
        steps(dz, ds, dv, dw, a_p, a_d);
        Int used = 0;
        while (used < k_max && (a_p < 1.0 || a_d < 1.0)) {
            const Real aim_p = std::min(a_p + kAspiration, 1.0);
            const Real aim_d = std::min(a_d + kAspiration, 1.0);
            for (std::size_t j = 0; j < n; ++j) {
                t_c[j] = target((z[j] + aim_p * dz[j]) * (s[j] + aim_d * ds[j]), lo, hi);
            }
            for (const Index j : cols) {
                t_w[j] = target((v[j] + aim_p * dv[j]) * (w[j] + aim_d * dw[j]), lo, hi);
            }
            if (!solve(zero_m, zero_n, t_c, zero_n, t_w, mz, my, ms, mv, mw)) {
                break;
            }
            for (std::size_t j = 0; j < n; ++j) {
                tz[j] = dz[j] + mz[j];
                ts[j] = ds[j] + ms[j];
            }
            for (const Index j : cols) {
                tv[j] = dv[j] + mv[j];
                tw[j] = dw[j] + mw[j];
            }
            Real new_p = 0.0;
            Real new_d = 0.0;
            steps(tz, ts, tv, tw, new_p, new_d);
            // Each space takes the corrector on its own, as in CG08 sec. 5,
            // when its step grows by a fraction of the aspiration (sec. 2.2's
            // test; sec. 5's 1.01 a goes with its weighted correctors, not
            // used here). A dz_m = 0 (up to the primal regularization, reg
            // dy_m), dz_m + dv_m = 0 and A' dy_m + ds_m - dw_m = 0, so either
            // half keeps its feasibility equations whatever the other does. A
            // space already at a full step keeps its direction.
            const bool take_p = a_p < 1.0 && new_p >= a_p + kAcceptFraction * (aim_p - a_p);
            const bool take_d = a_d < 1.0 && new_d >= a_d + kAcceptFraction * (aim_d - a_d);
            if (!take_p && !take_d) {
                break;
            }
            if (take_p) {
                std::copy(tz.begin(), tz.end(), dz.begin());
                for (const Index j : cols) {
                    dv[j] = tv[j];
                }
                a_p = new_p;
            }
            if (take_d) {
                for (std::size_t i = 0; i < m; ++i) {
                    dy[i] += my[i];
                }
                std::copy(ts.begin(), ts.end(), ds.begin());
                for (const Index j : cols) {
                    dw[j] = tw[j];
                }
                a_d = new_d;
            }
            ++used;
        }
        return used;
    }
};

// Correctors per iteration: k with (1 + k) solves costing about one
// factorization, at least 1 (the first corrector is always worth its solve),
// capped by BarrierOptions::gondzio_correctors. Gondzio 1996 (sec. 3.3) sets
// it from this ratio of factorization to solve effort; the estimate
// (1 + k) E_s ~ E_f is the one of arXiv:2508.04370 without its refinement
// terms. The cap alone where the backend gives no cost estimate. The caller
// counts once, at the first factorization: a Schur complement that grows
// later does not change it.
Int gondzioCorrectorCount(Int cap, const NewtonSolver::FactorStats& st) {
    if (cap <= 0) {
        return 0;
    }
    if (st.factor_flops <= 0.0 || st.solve_flops <= 0.0) {
        return cap;
    }
    const double k = std::floor(st.factor_flops / st.solve_flops) - 1.0;
    return static_cast<Int>(std::clamp(k, 1.0, static_cast<double>(cap)));
}

// A run that stops short of primal_dual_tol (stall, iteration limit, failed
// factorization) still hands its best iterate on when every measure is within
// this tolerance. Such a point is only ever used as a crossover start, which
// proves optimality by simplex, so it need not meet primal_dual_tol itself.
// Its gap measure is mu / (1 + |pobj|), not the stop's relative gap (#253).
constexpr Real kNearOptimalTol = 1e-6;

BarrierIpmOutcome runMehrotraIpm(NewtonSolver& solver, const SparseMatrix& A,
                                 std::span<const Real> b, std::span<const Real> c,
                                 std::span<const Real> u, const BarrierOptions& opts,
                                 Real obj_offset, std::vector<Real>& z, std::vector<Real>& y,
                                 std::vector<Real>& s, Int& iters, Int& corrections) {
    const Index m = A.numRows();
    const Index n = A.numCols();
    const Real reg = std::max(opts.regularization, 1e-12);
    corrections = 0;

    // Columns with an upper bound (#299); empty unless implicit_upper_bounds,
    // and then the loop below is the plain standard-form one.
    std::vector<Index> upper_cols;
    for (Index j = 0; j < n; ++j) {
        if (isFinite(u[j])) {
            upper_cols.push_back(j);
        }
    }
    const bool bounded = !upper_cols.empty();
    const Index n_compl = n + static_cast<Index>(upper_cols.size());
    std::vector<Real> v;  // upper slack u - z
    std::vector<Real> w;  // dual of z <= u
    std::vector<Real> ru(static_cast<size_t>(n), 0.0);
    std::vector<Real> rw(static_cast<size_t>(n), 0.0);
    std::vector<Real> dv(static_cast<size_t>(n), 0.0);
    std::vector<Real> dw(static_cast<size_t>(n), 0.0);
    std::vector<Real> dv_aff(static_cast<size_t>(n), 0.0);
    std::vector<Real> dw_aff(static_cast<size_t>(n), 0.0);
    UpperBoundStep ub_step;

    std::vector<Real> az(static_cast<size_t>(m));
    std::vector<Real> at_y(static_cast<size_t>(n));
    std::vector<Real> rp(static_cast<size_t>(m));
    std::vector<Real> rd(static_cast<size_t>(n));
    std::vector<Real> rc(static_cast<size_t>(n));

    std::vector<Real> dz(static_cast<size_t>(n));
    std::vector<Real> dy(static_cast<size_t>(m));
    std::vector<Real> ds(static_cast<size_t>(n));
    std::vector<Real> dz_aff(static_cast<size_t>(n));
    std::vector<Real> dy_aff(static_cast<size_t>(m));
    std::vector<Real> ds_aff(static_cast<size_t>(n));

    // The time limit (#275) and the caller's stop flag: the backend's
    // factorize() fails once either is reached. A stop ends as Failed, which
    // solve() reads as a stop (IterLimit), a passed time as TimeLimit.
    const Deadline& deadline = solver.deadline();
    auto interrupted = [&] {
        return deadline.stopped() ? BarrierIpmOutcome::Failed : BarrierIpmOutcome::TimeLimit;
    };
    if (!mehrotraStartingPoint(solver, b, c, u, upper_cols, reg, z, y, s, v, w)) {
        iters = 0;
        return deadline.passed() ? interrupted() : BarrierIpmOutcome::Failed;
    }
    // After the starting point, which sizes v and w.
    GondzioCorrectors gondzio{upper_cols, z, s, v, w, dz, dy, ds, dv, dw};
    Int gondzio_k = -1;  // set once the backend has factorized

    // The caller reads s as the reduced cost c - A'y, which is s - w on a
    // bounded column (negative at an upper bound).
    auto reducedCosts = [&](std::vector<Real>& s_out, const std::vector<Real>& w_in) {
        for (const Index j : upper_cols) {
            s_out[j] -= w_in[j];
        }
    };

    // Best iterate seen, by the largest of the three termination measures.
    Real best_err = std::numeric_limits<Real>::infinity();
    std::vector<Real> best_z;
    std::vector<Real> best_y;
    std::vector<Real> best_s;
    std::vector<Real> best_w;
    auto stopShort = [&](Int iter) {
        iters = iter;
        if (best_err > kNearOptimalTol) {
            return BarrierIpmOutcome::Failed;
        }
        z = std::move(best_z);
        y = std::move(best_y);
        s = std::move(best_s);
        reducedCosts(s, best_w);
        return BarrierIpmOutcome::NearOptimal;
    };

    // Stall detection: track previous gap.
    Real prev_gap = 1e30;
    Int stall_count = 0;
    constexpr Int max_stall = 10;
    Index schur_order_shown = -1;
    Index schur_dropped_shown = -1;

    Real b_norm = infNorm(b);
    for (const Index j : upper_cols) {
        b_norm = std::max(b_norm, std::abs(u[j]));
    }
    const Real inv_b = 1.0 / (1.0 + b_norm);
    const Real inv_c = 1.0 / (1.0 + infNorm(c));

    for (Int iter = 0; iter < opts.max_iter; ++iter) {
        if (opts.stop_flag != nullptr && opts.stop_flag->load(std::memory_order_relaxed)) {
            iters = iter;
            return BarrierIpmOutcome::Failed;
        }

        // Residuals.
        std::fill(az.begin(), az.end(), 0.0);
        std::fill(at_y.begin(), at_y.end(), 0.0);
        A.multiply(z, az);
        A.multiplyTranspose(y, at_y);

        for (Index i = 0; i < m; ++i) {
            rp[i] = b[i] - az[i];
        }
        for (Index j = 0; j < n; ++j) {
            rd[j] = c[j] - at_y[j] - s[j];
        }
        Real zs_sum = dot(z, s);
        Real ru_norm = 0.0;
        for (const Index j : upper_cols) {
            rd[j] += w[j];
            ru[j] = u[j] - z[j] - v[j];
            ru_norm = std::max(ru_norm, std::abs(ru[j]));
            zs_sum += v[j] * w[j];
        }

        Real mu = zs_sum / std::max<Index>(n_compl, 1);
        Real pinf = std::max(infNorm(rp), ru_norm) * inv_b;
        Real dinf = infNorm(rd) * inv_c;
        // Relative duality gap: z's (bound pairs included) is the gap of a
        // feasible pair, not mu = z's / n, which would stop n times early
        // (#253).
        Real gap = std::abs(zs_sum) / (1.0 + std::abs(obj_offset + dot(c, z)));

        if (opts.verbose) {
            std::printf("IPM %4d  pobj=% .10e  pinf=% .2e  dinf=% .2e  gap=% .2e  mu=% .2e\n", iter,
                        obj_offset + dot(c, z), pinf, dinf, gap, mu);
        }

        if (pinf < opts.primal_dual_tol && dinf < opts.primal_dual_tol &&
            gap < opts.primal_dual_tol) {
            iters = iter;
            reducedCosts(s, w);
            return BarrierIpmOutcome::Optimal;
        }

        // The crossover hand-off keeps mu / (1 + |pobj|), the scale
        // kNearOptimalTol was set on: crossover proves optimality itself.
        const Real gap_mu = gap / static_cast<Real>(std::max<Index>(n_compl, 1));
        const Real err = std::max({pinf, dinf, gap_mu});
        if (err < best_err) {
            best_err = err;
            best_z = z;
            best_y = y;
            best_s = s;
            if (bounded) {
                best_w = w;
            }
        }

        // Stall detection.
        if (gap >= prev_gap * 0.999) {
            ++stall_count;
            if (stall_count >= max_stall) {
                return stopShort(iter);
            }
        } else {
            stall_count = 0;
        }
        prev_gap = gap;

        // After the stopping test, so an iterate that converged on the last
        // step is still reported optimal.
        if (deadline.passed()) {
            iters = iter;
            return interrupted();
        }

        // Factorize.
        if (bounded) {
            ub_step.scaling(upper_cols, z, s, v, w);
        }
        const std::span<const Real> z_fac = bounded ? ub_step.z_eff : z;
        const std::span<const Real> s_fac = bounded ? ub_step.s_eff : s;
        if (!solver.factorize(z_fac, s_fac, reg)) {
            // Retry with boosted regularization, unless out of time.
            if (deadline.passed() || !solver.factorize(z_fac, s_fac, reg * 100.0)) {
                if (deadline.passed()) {
                    iters = iter;
                    return interrupted();
                }
                return stopShort(iter);
            }
        }
        if (gondzio_k < 0) {
            gondzio_k = gondzioCorrectorCount(opts.gondzio_correctors, solver.factorStats());
        }
        // One Newton solve for a right-hand side (rp, rd, rc, ru, rw).
        auto newtonFor = [&](std::span<const Real> rp_in, std::span<const Real> rd_in,
                             std::span<const Real> rc_in, std::span<const Real> ru_in,
                             std::span<const Real> rw_in, std::span<Real> dz_out,
                             std::span<Real> dy_out, std::span<Real> ds_out, std::span<Real> dv_out,
                             std::span<Real> dw_out) {
            if (!bounded) {
                return solver.solveNewton(rp_in, rd_in, rc_in, dz_out, dy_out, ds_out);
            }
            ub_step.fold(upper_cols, z, v, w, rd_in, rc_in, ru_in, rw_in);
            if (!solver.solveNewton(rp_in, ub_step.rd_eff, ub_step.rc_eff, dz_out, dy_out,
                                    ds_out)) {
                return false;
            }
            UpperBoundStep::recover(upper_cols, z, s, v, w, rc_in, ru_in, rw_in, dz_out, ds_out,
                                    dv_out, dw_out);
            return true;
        };
        auto newton = [&](std::span<Real> dz_out, std::span<Real> dy_out, std::span<Real> ds_out,
                          std::span<Real> dv_out, std::span<Real> dw_out) {
            return newtonFor(rp, rd, rc, ru, rw, dz_out, dy_out, ds_out, dv_out, dw_out);
        };
        if (opts.verbose) {
            // The dense-column Schur complement of the sparse NE backend
            // (#314), printed when it changes: it grows with shifted pivots.
            const auto st = solver.factorStats();
            if (st.dense_cols > 0 &&
                (st.schur_order != schur_order_shown || st.schur_dropped != schur_dropped_shown)) {
                std::printf("  NE Schur: order=%d (dense_cols=%d) dropped=%d\n", st.schur_order,
                            st.dense_cols, st.schur_dropped);
                schur_order_shown = st.schur_order;
                schur_dropped_shown = st.schur_dropped;
            }
        }

        // Affine direction (predictor).
        for (Index j = 0; j < n; ++j) {
            rc[j] = -z[j] * s[j];
        }
        for (const Index j : upper_cols) {
            rw[j] = -v[j] * w[j];
        }
        if (!newton(dz_aff, dy_aff, ds_aff, dv_aff, dw_aff)) {
            return stopShort(iter);
        }

        Real alpha_aff_p = maxStepToBoundary(z, dz_aff, 1.0);
        Real alpha_aff_d = maxStepToBoundary(s, ds_aff, 1.0);
        if (bounded) {
            alpha_aff_p = maxStepOn(upper_cols, v, dv_aff, 1.0, alpha_aff_p);
            alpha_aff_d = maxStepOn(upper_cols, w, dw_aff, 1.0, alpha_aff_d);
        }

        Real mu_aff = 0.0;
        for (Index j = 0; j < n; ++j) {
            mu_aff += (z[j] + alpha_aff_p * dz_aff[j]) * (s[j] + alpha_aff_d * ds_aff[j]);
        }
        for (const Index j : upper_cols) {
            mu_aff += (v[j] + alpha_aff_p * dv_aff[j]) * (w[j] + alpha_aff_d * dw_aff[j]);
        }
        mu_aff /= std::max<Index>(n_compl, 1);

        Real sigma = std::pow(std::max(mu_aff, 0.0) / std::max(mu, 1e-30), 3.0);
        sigma = std::clamp(sigma, 0.0, 1.0);

        // Corrector direction.
        for (Index j = 0; j < n; ++j) {
            rc[j] = sigma * mu - z[j] * s[j] - dz_aff[j] * ds_aff[j];
        }
        for (const Index j : upper_cols) {
            rw[j] = sigma * mu - v[j] * w[j] - dv_aff[j] * dw_aff[j];
        }
        if (!newton(dz, dy, ds, dv, dw)) {
            return stopShort(iter);
        }

        if (gondzio_k > 0) {
            const Int used = gondzio.correct(gondzio_k, sigma * mu, newtonFor);
            corrections += used;
            if (opts.verbose) {
                std::printf("        gondzio correctors: %d of %d\n", used, gondzio_k);
            }
        }

        // Step to boundary.
        Real alpha_p = maxStepToBoundary(z, dz, opts.step_fraction);
        Real alpha_d = maxStepToBoundary(s, ds, opts.step_fraction);
        if (bounded) {
            alpha_p = maxStepOn(upper_cols, v, dv, opts.step_fraction, alpha_p);
            alpha_d = maxStepOn(upper_cols, w, dw, opts.step_fraction, alpha_d);
        }

        if (opts.verbose) {
            Real dz_norm = infNorm(std::span<const Real>(dz.data(), static_cast<size_t>(n)));
            Real dy_norm = infNorm(std::span<const Real>(dy.data(), static_cast<size_t>(m)));
            Real ds_norm = infNorm(std::span<const Real>(ds.data(), static_cast<size_t>(n)));
            std::printf(
                "        alpha_p=%.4e alpha_d=%.4e |dz|=%.2e |dy|=%.2e |ds|=%.2e sigma=%.4e\n",
                alpha_p, alpha_d, dz_norm, dy_norm, ds_norm, sigma);
        }

        for (Index j = 0; j < n; ++j) {
            z[j] += alpha_p * dz[j];
            s[j] += alpha_d * ds[j];
            z[j] = std::max(z[j], 1e-12);
            s[j] = std::max(s[j], 1e-12);
        }
        for (const Index j : upper_cols) {
            v[j] = std::max(v[j] + alpha_p * dv[j], 1e-12);
            w[j] = std::max(w[j] + alpha_d * dw[j], 1e-12);
        }
        for (Index i = 0; i < m; ++i) {
            y[i] += alpha_d * dy[i];
        }
    }

    return stopShort(opts.max_iter);
}

}  // anonymous namespace

// ============================================================================
// BarrierSolver public interface (unchanged from before)
// ============================================================================

void BarrierSolver::load(const LpProblem& problem) {
    original_ = problem;
    loaded_ = true;
    status_ = Status::Error;
    objective_ = 0.0;
    scaled_obj_ = 0.0;
    iterations_ = 0;
    crossover_iters_ = 0;
    last_error_.clear();
    primal_orig_.assign(static_cast<size_t>(original_.num_cols), 0.0);
    dual_eq_.clear();
    reduced_costs_std_.clear();
    basis_orig_.clear();
    used_gpu_ = false;
    rebuildTransform();
}

void BarrierSolver::setOptions(const BarrierOptions& options) {
    const bool reshape = options.implicit_upper_bounds != options_.implicit_upper_bounds;
    options_ = options;
    if (loaded_ && reshape) {
        rebuildTransform();
    }
}

bool BarrierSolver::buildStandardForm() {
    // Use the presolved problem if presolve is active, otherwise original.
    const auto& prob = presolve_active_ ? presolved_ : original_;

    transformed_infeasible_ = false;
    beq_.clear();
    cstd_.clear();
    ustd_.clear();
    const bool implicit_ub = options_.implicit_upper_bounds;
    col_expr_.assign(static_cast<size_t>(prob.num_cols), OriginalColExpr{});
    // The standard form minimises obj_sign * c, so its offset carries the same sign.
    const Real obj_sign = (prob.sense == Sense::Minimize) ? 1.0 : -1.0;
    std_obj_offset_ = obj_sign * prob.obj_offset;

    std::vector<Triplet> trips;

    auto addStdCol = [&](Real cost, Real upper = kInf) {
        Index idx = static_cast<Index>(cstd_.size());
        cstd_.push_back(cost);
        ustd_.push_back(upper);
        return idx;
    };

    auto addEqRow = [&](const std::vector<std::pair<Index, Real>>& entries, Real rhs) {
        Index row = static_cast<Index>(beq_.size());
        beq_.push_back(rhs);
        for (const auto& [col, val] : entries) {
            if (std::abs(val) > 0.0) {
                trips.push_back({row, col, val});
            }
        }
    };

    for (Index j = 0; j < prob.num_cols; ++j) {
        Real lb = prob.col_lower[j];
        Real ub = prob.col_upper[j];
        Real obj = obj_sign * prob.obj[j];

        bool lb_finite = isFinite(lb);
        bool ub_finite = isFinite(ub);

        OriginalColExpr expr;

        if (lb_finite && ub_finite) {
            if (ub < lb - 1e-12) {
                transformed_infeasible_ = true;
                return false;
            }
            Index w = addStdCol(obj, implicit_ub ? std::max(ub - lb, 0.0) : kInf);
            expr.offset = lb;
            expr.col_a = w;
            expr.coeff_a = 1.0;
            std_obj_offset_ += obj * lb;
            if (!implicit_ub) {
                Index t = addStdCol(0.0);
                addEqRow({{w, 1.0}, {t, 1.0}}, ub - lb);
            }
        } else if (lb_finite && !ub_finite) {
            Index w = addStdCol(obj);
            expr.offset = lb;
            expr.col_a = w;
            expr.coeff_a = 1.0;
            std_obj_offset_ += obj * lb;
        } else if (!lb_finite && ub_finite) {
            Index w = addStdCol(-obj);
            expr.offset = ub;
            expr.col_a = w;
            expr.coeff_a = -1.0;
            std_obj_offset_ += obj * ub;
        } else {
            Index p = addStdCol(obj);
            Index n = addStdCol(-obj);
            expr.col_a = p;
            expr.coeff_a = 1.0;
            expr.col_b = n;
            expr.coeff_b = -1.0;
        }

        col_expr_[j] = expr;
    }

    // One standard-form row per original row. An equality row gets no slack:
    // splitting it into two inequalities would force both slacks to zero and
    // leave the feasible set without an interior, which pins the barrier
    // parameter and stalls the IPM (#204). A ranged row gets a slack bounded
    // by the range through a second row, as a bounded column does.
    std::vector<std::pair<Index, Real>> entries;
    for (Index i = 0; i < prob.num_rows; ++i) {
        const Real rl = prob.row_lower[i];
        const Real ru = prob.row_upper[i];
        const bool rl_finite = isFinite(rl);
        const bool ru_finite = isFinite(ru);
        if (!rl_finite && !ru_finite) {
            continue;
        }
        if (rl_finite && ru_finite && ru < rl - 1e-12) {
            transformed_infeasible_ = true;
            return false;
        }

        // Rows with only a lower bound are negated into upper-bound form.
        const Real sign = ru_finite ? 1.0 : -1.0;
        Real rhs = ru_finite ? ru : -rl;
        entries.clear();
        auto rv = prob.matrix.row(i);
        for (Index k = 0; k < rv.size(); ++k) {
            const Real a = sign * rv.values[k];
            const auto& e = col_expr_[rv.indices[k]];
            rhs -= a * e.offset;
            if (e.col_a >= 0) {
                entries.push_back({e.col_a, a * e.coeff_a});
            }
            if (e.col_b >= 0) {
                entries.push_back({e.col_b, a * e.coeff_b});
            }
        }

        const bool equality = rl_finite && ru_finite && ru - rl <= 1e-12;
        if (equality) {
            if (entries.empty()) {
                // An empty equality row is either 0 = 0 or infeasible.
                if (std::abs(rhs) > 1e-9) {
                    transformed_infeasible_ = true;
                    return false;
                }
                continue;
            }
            addEqRow(entries, rhs);
            continue;
        }
        const bool ranged = rl_finite && ru_finite;
        Index slack = addStdCol(0.0, implicit_ub && ranged ? ru - rl : kInf);
        entries.push_back({slack, 1.0});
        addEqRow(entries, rhs);
        if (ranged && !implicit_ub) {
            Index t = addStdCol(0.0);
            addEqRow({{slack, 1.0}, {t, 1.0}}, ru - rl);
        }
    }

    aeq_ = SparseMatrix(static_cast<Index>(beq_.size()), static_cast<Index>(cstd_.size()),
                        std::move(trips));

    dual_eq_.assign(beq_.size(), 0.0);
    reduced_costs_std_.assign(cstd_.size(), 0.0);
    return true;
}

void BarrierSolver::rebuildTransform() {
    presolve_active_ = false;
    // Column changes resize the model; the solution vector, which postsolve
    // and the feasibility checks index by column, follows it.
    if (static_cast<Index>(primal_orig_.size()) != original_.num_cols) {
        primal_orig_.assign(static_cast<size_t>(original_.num_cols), 0.0);
    }
    if (static_cast<Index>(basis_orig_.size()) != original_.num_cols + original_.num_rows) {
        basis_orig_.clear();
    }
    if (options_.presolve) {
        presolver_ = BarrierPresolver{};
        presolved_ = presolver_.presolve(original_);
        if (presolver_.isInfeasible()) {
            transformed_ok_ = false;
            transformed_infeasible_ = true;
            clearTransformedSolution();
            return;
        }
        if (presolved_.num_cols < original_.num_cols || presolved_.num_rows < original_.num_rows) {
            presolve_active_ = true;
            if (options_.verbose) {
                presolver_.stats().print();
            }
        }
    }
    transformed_ok_ = buildStandardForm();
    if (!transformed_ok_) {
        clearTransformedSolution();
    }
}

// A model change that leaves no transform also leaves no solution: nothing
// from the previous model may be read back against this one.
void BarrierSolver::clearTransformedSolution() {
    col_expr_.clear();
    dual_eq_.clear();
    reduced_costs_std_.clear();
    basis_orig_.clear();
    primal_orig_.assign(static_cast<size_t>(original_.num_cols), 0.0);
}

std::vector<Real> BarrierSolver::getPrimalValues() const {
    return primal_orig_;
}

std::vector<Real> BarrierSolver::getDualValues() const {
    return std::vector<Real>(static_cast<size_t>(original_.num_rows), 0.0);
}

std::vector<Real> BarrierSolver::getReducedCosts() const {
    const auto& prob = presolve_active_ ? presolved_ : original_;
    const Real sense_sign = (prob.sense == Sense::Minimize) ? 1.0 : -1.0;
    // No transform for this model (presolve proved a changed model
    // infeasible before building one): nothing to map back.
    if (static_cast<Index>(col_expr_.size()) != prob.num_cols) {
        return std::vector<Real>(static_cast<size_t>(original_.num_cols), 0.0);
    }

    // Compute reduced costs in the presolved/original space.
    std::vector<Real> rc_prob(static_cast<size_t>(prob.num_cols), 0.0);
    for (Index j = 0; j < prob.num_cols; ++j) {
        const auto& e = col_expr_[j];
        Real v = 0.0;
        if (e.col_a >= 0 && e.col_a < static_cast<Index>(reduced_costs_std_.size())) {
            v += e.coeff_a * reduced_costs_std_[e.col_a];
        }
        if (e.col_b >= 0 && e.col_b < static_cast<Index>(reduced_costs_std_.size())) {
            v += e.coeff_b * reduced_costs_std_[e.col_b];
        }
        rc_prob[j] = sense_sign * v;
    }

    if (!presolve_active_) {
        return rc_prob;
    }

    // Map back to original space via presolver postsolve.
    std::vector<Real> rc(static_cast<size_t>(original_.num_cols), 0.0);
    const auto& col_map = presolver_.colMapping();
    for (Index jj = 0; jj < static_cast<Index>(col_map.size()); ++jj) {
        rc[col_map[jj]] = rc_prob[jj];
    }
    return rc;
}

std::vector<BasisStatus> BarrierSolver::getBasis() const {
    return basis_orig_;
}

void BarrierSolver::setBasis(std::span<const BasisStatus> basis) {
    (void)basis;
}

void BarrierSolver::addRows(std::span<const Index> starts, std::span<const Index> indices,
                            std::span<const Real> values, std::span<const Real> lower,
                            std::span<const Real> upper) {
    if (!loaded_) {
        return;
    }
    if (starts.empty()) {
        return;
    }

    Index rows_to_add = static_cast<Index>(lower.size());
    if (rows_to_add != static_cast<Index>(upper.size())) {
        return;
    }
    if (static_cast<Index>(starts.size()) != rows_to_add + 1) {
        return;
    }

    for (Index r = 0; r < rows_to_add; ++r) {
        Index s = starts[r];
        Index e = starts[r + 1];
        if (s < 0 || e < s || e > static_cast<Index>(indices.size()) ||
            e > static_cast<Index>(values.size())) {
            return;
        }
        std::span<const Index> ri(indices.data() + s, static_cast<size_t>(e - s));
        std::span<const Real> rv(values.data() + s, static_cast<size_t>(e - s));
        original_.matrix.addRow(ri, rv);
        original_.row_lower.push_back(lower[r]);
        original_.row_upper.push_back(upper[r]);
        original_.row_names.push_back("");
    }
    original_.num_rows += rows_to_add;
    rebuildTransform();
}

void BarrierSolver::removeRows(std::span<const Index> rows) {
    if (!loaded_) {
        return;
    }
    if (rows.empty()) {
        return;
    }

    std::vector<Index> sorted(rows.begin(), rows.end());
    std::sort(sorted.begin(), sorted.end());
    sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());

    for (auto it = sorted.rbegin(); it != sorted.rend(); ++it) {
        Index r = *it;
        if (r < 0 || r >= original_.num_rows) {
            continue;
        }
        original_.matrix.removeRowStable(r);
        original_.row_lower.erase(original_.row_lower.begin() + r);
        original_.row_upper.erase(original_.row_upper.begin() + r);
        if (r < static_cast<Index>(original_.row_names.size())) {
            original_.row_names.erase(original_.row_names.begin() + r);
        }
        --original_.num_rows;
    }
    rebuildTransform();
}

void BarrierSolver::setColBounds(Index col, Real lower, Real upper) {
    if (!loaded_) {
        return;
    }
    if (col < 0 || col >= original_.num_cols) {
        return;
    }
    original_.col_lower[col] = lower;
    original_.col_upper[col] = upper;
    rebuildTransform();
}

void BarrierSolver::setObjective(std::span<const Real> obj) {
    if (!loaded_) {
        return;
    }
    if (static_cast<Index>(obj.size()) != original_.num_cols) {
        return;
    }
    original_.obj.assign(obj.begin(), obj.end());
    rebuildTransform();
}

void BarrierSolver::addCols(std::span<const Index> starts, std::span<const Index> indices,
                            std::span<const Real> values, std::span<const Real> obj,
                            std::span<const Real> lower, std::span<const Real> upper) {
    if (!loaded_) {
        return;
    }
    if (checkColumnBatch(original_.num_rows, starts, indices, values, obj, lower, upper) == 0) {
        return;
    }
    appendColumns(original_, starts, indices, values, obj, lower, upper);
    rebuildTransform();
}

void BarrierSolver::removeCols(std::span<const Index> cols) {
    if (!loaded_) {
        return;
    }
    const std::vector<Index> sorted = sortedColumnSet(original_.num_cols, cols);
    if (sorted.empty()) {
        return;
    }
    removeColumns(original_, sorted);
    rebuildTransform();
}

void BarrierSolver::setRowBounds(Index row, Real lower, Real upper) {
    if (!loaded_ || row < 0 || row >= original_.num_rows) {
        return;
    }
    original_.row_lower[row] = lower;
    original_.row_upper[row] = upper;
    rebuildTransform();
}

void BarrierSolver::setColObjective(Index col, Real cost) {
    if (!loaded_ || col < 0 || col >= original_.num_cols) {
        return;
    }
    original_.obj[col] = cost;
    rebuildTransform();
}

void BarrierSolver::reconstructOriginalPrimals(const std::vector<Real>& z) {
    const auto& prob = presolve_active_ ? presolved_ : original_;

    // Reconstruct primals in the (possibly presolved) problem space.
    std::vector<Real> primal_prob(static_cast<size_t>(prob.num_cols), 0.0);
    for (Index j = 0; j < prob.num_cols; ++j) {
        const auto& e = col_expr_[j];
        Real x = e.offset;
        if (e.col_a >= 0 && e.col_a < static_cast<Index>(z.size())) {
            x += e.coeff_a * z[e.col_a];
        }
        if (e.col_b >= 0 && e.col_b < static_cast<Index>(z.size())) {
            x += e.coeff_b * z[e.col_b];
        }
        if (isFinite(prob.col_lower[j])) {
            x = std::max(x, prob.col_lower[j]);
        }
        if (isFinite(prob.col_upper[j])) {
            x = std::min(x, prob.col_upper[j]);
        }
        primal_prob[j] = x;
    }

    if (presolve_active_) {
        // Postsolve to recover original-space primals.
        std::vector<Real> dual_prob;  // Not used for primal reconstruction.
        std::vector<Real> rc_prob;
        std::vector<Real> dual_orig;
        std::vector<Real> rc_orig;
        presolver_.postsolve(primal_prob, dual_prob, rc_prob, primal_orig_, dual_orig, rc_orig);

        // Clamp to original bounds.
        for (Index j = 0; j < original_.num_cols; ++j) {
            if (isFinite(original_.col_lower[j])) {
                primal_orig_[j] = std::max(primal_orig_[j], original_.col_lower[j]);
            }
            if (isFinite(original_.col_upper[j])) {
                primal_orig_[j] = std::min(primal_orig_[j], original_.col_upper[j]);
            }
        }
    } else {
        primal_orig_ = std::move(primal_prob);
    }
}

bool BarrierSolver::checkOriginalPrimalFeasibility(std::span<const Real> x) const {
    const Real tol = 1e-5;
    for (Index j = 0; j < original_.num_cols; ++j) {
        if (isFinite(original_.col_lower[j]) && x[j] < original_.col_lower[j] - tol) {
            return false;
        }
        if (isFinite(original_.col_upper[j]) && x[j] > original_.col_upper[j] + tol) {
            return false;
        }
    }

    // A row's tolerance is 1e-6 times the magnitude of its terms, floored at
    // 1e-6 absolute. Large terms cancelling leave a rounding-size residual,
    // and the old fixed 1e-5 let a row of tiny terms be badly violated: a
    // generated infeasible LP passed that way once the IPM converged on it
    // (#204).
    constexpr Real kRowTol = 1e-6;
    for (Index i = 0; i < original_.num_rows; ++i) {
        const auto rv = original_.matrix.row(i);
        Real act = 0.0;
        Real magnitude = 0.0;
        for (Index k = 0; k < rv.size(); ++k) {
            const Real term = rv.values[k] * x[rv.indices[k]];
            act += term;
            magnitude += std::abs(term);
        }
        const Real row_tol = kRowTol * std::max(1.0, magnitude);
        if (isFinite(original_.row_lower[i]) && act < original_.row_lower[i] - row_tol) {
            return false;
        }
        if (isFinite(original_.row_upper[i]) && act > original_.row_upper[i] + row_tol) {
            return false;
        }
    }
    return true;
}

// ============================================================================
// solveStandardForm — dispatch + shared IPM
// ============================================================================

BarrierIpmOutcome BarrierSolver::solveStandardForm(std::vector<Real>& z, std::vector<Real>& y,
                                                   std::vector<Real>& s, Int& iters) {
    const Index m = aeq_.numRows();
    const Index n = aeq_.numCols();
    gondzio_corrections_ = 0;
    if (n == 0) {
        z.clear();
        y.assign(static_cast<size_t>(m), 0.0);
        s.clear();
        iters = 0;
        return BarrierIpmOutcome::Optimal;
    }
    if (m == 0) {
        z.assign(static_cast<size_t>(n), 0.0);
        y.clear();
        s = cstd_;
        for (Index j = 0; j < n; ++j) {
            if (s[j] < -1e-12) {
                // A negative cost pushes the column to its upper bound.
                if (!isFinite(ustd_[j])) {
                    return BarrierIpmOutcome::Failed;
                }
                z[j] = ustd_[j];
            }
        }
        iters = 0;
        return BarrierIpmOutcome::Optimal;
    }

    // Ruiz equilibration.
    computeRuizScaling(aeq_, beq_, cstd_, row_scale_, col_scale_, options_.ruiz_iterations);
    // Column scaling x = C x_s moves an upper bound to u / C.
    bool has_upper = false;
    for (Index j = 0; j < n; ++j) {
        if (isFinite(ustd_[j])) {
            ustd_[j] /= col_scale_[j];
            has_upper = true;
        }
    }

    // Density detection for backend selection.
    auto density = detectDensity(aeq_, options_.dense_col_fraction);
    bool use_augmented =
        (density.dense_cols > 50 || density.dense_rows > 10 || density.avg_ne_row_density > 20.0);

    // Select backend.
    std::unique_ptr<NewtonSolver> solver;
    auto algo = options_.algorithm;

    [[maybe_unused]] const auto requested_algo = options_.algorithm;
    if (algo == BarrierAlgorithm::Auto) {
#ifdef MIPX_HAS_CUDSS
        algo = use_augmented ? BarrierAlgorithm::GpuAugmented : BarrierAlgorithm::GpuCholesky;
#else
        algo = use_augmented ? BarrierAlgorithm::CpuAugmented : BarrierAlgorithm::CpuCholesky;
#endif
    }

    // The GPU path solves the standard form without upper bounds; implicit
    // ones (#299) and Gondzio's correctors (#293) are in the CPU loop only.
    // The CPU backend is the one a build without CUDA picks under Auto; the
    // CPU default itself (dense or sparse NE) is #295's.
    const bool cpu_only = has_upper || options_.gondzio_correctors > 0;
    if (cpu_only &&
        (algo == BarrierAlgorithm::GpuCholesky || algo == BarrierAlgorithm::GpuAugmented)) {
        if (options_.verbose) {
            std::printf(
                "Barrier: implicit upper bounds and Gondzio correctors are CPU-only; using the "
                "CPU backend.\n");
        }
        algo = use_augmented ? BarrierAlgorithm::CpuAugmented : BarrierAlgorithm::CpuCholesky;
    }

    bool ok = false;
    BarrierIpmOutcome outcome = BarrierIpmOutcome::Failed;
    std::string gpu_fallback_error;

#ifdef MIPX_HAS_CUDSS
    // GPU device-resident path: handles NE/Augmented internally with auto-switching.
    if (algo == BarrierAlgorithm::GpuCholesky || algo == BarrierAlgorithm::GpuAugmented) {
        bool prefer_aug = (algo == BarrierAlgorithm::GpuAugmented) || use_augmented;
        std::string gpu_error;
        // The GPU loop checks only a stop flag, so under a time limit (#275)
        // it gets one of its own, raised by a watcher at the deadline or when
        // the caller's flag is.
        BarrierOptions gpu_opts = options_;
        std::atomic<bool> gpu_stop{false};
        std::jthread watcher;
        if (deadline_.isSet()) {
            gpu_opts.stop_flag = &gpu_stop;
            watcher = std::jthread([this, &gpu_stop](const std::stop_token& token) {
                while (!token.stop_requested()) {
                    if (deadline_.passed() ||
                        (options_.stop_flag != nullptr &&
                         options_.stop_flag->load(std::memory_order_relaxed))) {
                        gpu_stop.store(true, std::memory_order_relaxed);
                        return;
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
            });
        }
        ok = solveBarrierGpu(aeq_, m, n, beq_, cstd_, gpu_opts, std_obj_offset_, prefer_aug, z, y,
                             s, iters, &gpu_error);
        watcher = std::jthread();  // stops and joins it
        // No CPU fallback after a stop or at the time limit. A stop reads as
        // one (Failed, then IterLimit in solve()), as on the CPU path.
        if (!ok && options_.stop_flag != nullptr &&
            options_.stop_flag->load(std::memory_order_relaxed)) {
            return BarrierIpmOutcome::Failed;
        }
        if (!ok && deadline_.passed()) {
            return BarrierIpmOutcome::TimeLimit;
        }
        if (ok) {
            used_gpu_ = true;
            outcome = BarrierIpmOutcome::Optimal;
        } else {
            gpu_fallback_error = gpu_error.empty() ? "Barrier GPU solve failed." : gpu_error;
            if (options_.verbose && !gpu_error.empty()) {
                if (requested_algo == BarrierAlgorithm::Auto) {
                    std::fprintf(stderr, "Barrier GPU backend failed, falling back to CPU: %s\n",
                                 gpu_error.c_str());
                } else {
                    std::fprintf(stderr, "Barrier GPU backend failed: %s\n", gpu_error.c_str());
                }
            }
            if (requested_algo == BarrierAlgorithm::Auto) {
                algo =
                    use_augmented ? BarrierAlgorithm::CpuAugmented : BarrierAlgorithm::CpuCholesky;
            } else {
                last_error_ = gpu_fallback_error;
                return BarrierIpmOutcome::Failed;
            }
        }
    }
#else
    if (algo == BarrierAlgorithm::GpuCholesky || algo == BarrierAlgorithm::GpuAugmented) {
        // No CUDA — use CPU.
        algo = use_augmented ? BarrierAlgorithm::CpuAugmented : BarrierAlgorithm::CpuCholesky;
    }
#endif

    if (!ok && !solver) {
        switch (algo) {
            case BarrierAlgorithm::CpuCholesky:
                solver = createCpuCholeskySolver();
                break;
            case BarrierAlgorithm::CpuAugmented:
                solver = createCpuAugmentedSolver();
                break;
            case BarrierAlgorithm::CpuSparseCholesky:
                solver = createCpuSparseCholeskySolver();
                break;
            default:
                solver = createCpuCholeskySolver();
                break;
        }
    }

    if (solver) {
        // A factor too large for memory ends the solve with an error status
        // rather than an exception out of solve() (#280).
        try {
            // The stop flag too, so a raised one ends a long factorization
            // rather than waiting it out (the MIP root race joins this arm).
            solver->setDeadline(deadline_.orStop(options_.stop_flag));
            // Setup can take seconds (the sparse NE's pattern, ordering and
            // analysis; the dense NE's memory is first touched later, in the
            // first factorize), so it starts only with time left, and a
            // setup cut short by the deadline is a time limit, not a failure
            // (#347). A stop ends as Failed, which solve() reads as a stop.
            const Deadline& backend_deadline = solver->deadline();
            auto interrupted = [&] {
                return backend_deadline.stopped() ? BarrierIpmOutcome::Failed
                                                  : BarrierIpmOutcome::TimeLimit;
            };
            if (backend_deadline.passed()) {
                return interrupted();
            }
            if (!solver->setup(aeq_, m, n, options_)) {
                if (backend_deadline.passed()) {
                    return interrupted();
                }
                if (last_error_.empty()) {
                    last_error_ = solver->lastError().empty() ? "Barrier backend setup failed."
                                                              : solver->lastError();
                }
                return BarrierIpmOutcome::Failed;
            }
            outcome = runMehrotraIpm(*solver, aeq_, beq_, cstd_, ustd_, options_, std_obj_offset_,
                                     z, y, s, iters, gondzio_corrections_);
        } catch (const std::bad_alloc&) {
            last_error_ = "Barrier ran out of memory in its Newton-system backend.";
            return BarrierIpmOutcome::Failed;
        }
        if (outcome == BarrierIpmOutcome::TimeLimit) {
            return outcome;  // no iterate to unscale or hand on
        }
        ok = outcome != BarrierIpmOutcome::Failed;
        if (!ok && last_error_.empty() &&
            !(options_.stop_flag != nullptr &&
              options_.stop_flag->load(std::memory_order_relaxed))) {
            if (!gpu_fallback_error.empty()) {
                last_error_ = gpu_fallback_error + " CPU fallback also failed.";
            } else if (!solver->lastError().empty()) {
                last_error_ = solver->lastError();
            } else {
                last_error_ = "Barrier solve failed.";
            }
        }
        // A failed or stopped run has no iterate that solve() reads, and one
        // whose starting point failed (a stop inside its factorization) has
        // none at all: z is empty, and unscaling it would read past its end.
        if (!ok) {
            return outcome;
        }
    }

    // Compute objective while z is still in scaled space.
    // In scaled space: c_s'*z_s = (C*c)'*z_s = c'*(C*z_s) = c'*z_orig,
    // so dot(cstd_, z) gives the correct original objective.
    // After unscaling, cstd_ remains scaled but z is unscaled, which would
    // give c'*C^2*z_s — wrong.
    scaled_obj_ = std_obj_offset_ + dot(cstd_, z);

    // Unscale solution.
    if (!col_scale_.empty()) {
        for (Index j = 0; j < n; ++j) {
            z[j] *= col_scale_[j];
            s[j] /= col_scale_[j];
        }
        for (Index i = 0; i < m; ++i) {
            y[i] *= row_scale_[i];
        }
    }

    return outcome;
}

// ============================================================================
// runCrossover() — extract a basic feasible solution from the IPM iterate
// ============================================================================

bool BarrierSolver::runCrossover() {
    crossover_status_ = Status::Error;
    const Index ncols = original_.num_cols;
    const Index nrows = original_.num_rows;
    const Real tol = 1e-6;

    // Phase 1: Variable pushing — classify each original variable.
    std::vector<BasisStatus> basis(static_cast<size_t>(ncols + nrows));

    for (Index j = 0; j < ncols; ++j) {
        Real x = primal_orig_[j];
        Real lb = original_.col_lower[j];
        Real ub = original_.col_upper[j];
        bool lb_finite = isFinite(lb);
        bool ub_finite = isFinite(ub);

        if (lb_finite && ub_finite && std::abs(ub - lb) < tol) {
            basis[j] = BasisStatus::Fixed;
        } else if (lb_finite && std::abs(x - lb) < tol) {
            basis[j] = BasisStatus::AtLower;
        } else if (ub_finite && std::abs(x - ub) < tol) {
            basis[j] = BasisStatus::AtUpper;
        } else if (!lb_finite && !ub_finite) {
            // Free variable — must be basic.
            basis[j] = BasisStatus::Basic;
        } else {
            // Interior point — candidate for basic.
            basis[j] = BasisStatus::Basic;
        }
    }

    // Compute row activities: ax[i] = sum_j A[i,j] * x[j].
    std::vector<Real> ax(static_cast<size_t>(nrows), 0.0);
    original_.matrix.multiply(primal_orig_, ax);

    // Phase 1 (continued): Classify row slacks.
    // Dual simplex basis has structural vars [0..ncols) then slacks [ncols..ncols+nrows).
    for (Index i = 0; i < nrows; ++i) {
        Real rl = original_.row_lower[i];
        Real ru = original_.row_upper[i];
        bool rl_finite = isFinite(rl);
        bool ru_finite = isFinite(ru);

        if (rl_finite && ru_finite && std::abs(ru - rl) < tol) {
            // Equality row — slack is fixed at zero.
            basis[ncols + i] = BasisStatus::Fixed;
        } else if (ru_finite && std::abs(ax[i] - ru) < tol) {
            // Row active at its upper bound: slack value = activity = row_upper,
            // so the slack is nonbasic at its upper bound. (mipx slack convention:
            // s_i = a_i^T x with row_lower_i <= s_i <= row_upper_i.)
            basis[ncols + i] = BasisStatus::AtUpper;
        } else if (rl_finite && std::abs(ax[i] - rl) < tol) {
            // Row active at its lower bound: slack value = activity = row_lower.
            basis[ncols + i] = BasisStatus::AtLower;
        } else {
            // Slack is interior — basic candidate.
            basis[ncols + i] = BasisStatus::Basic;
        }
    }

    // Count basics. We need exactly nrows basic variables.
    Index n_basic = 0;
    for (auto s : basis) {
        if (s == BasisStatus::Basic) {
            ++n_basic;
        }
    }

    // If too many basics, push some to bounds using reduced cost indicators.
    // Variables with larger |reduced_cost| are less likely to be basic.
    if (n_basic > nrows) {
        auto rc = getReducedCosts();
        // Build list of basic structural variables with their |rc| values.
        std::vector<std::pair<Real, Index>> basic_vars;
        for (Index j = 0; j < ncols; ++j) {
            if (basis[j] == BasisStatus::Basic) {
                basic_vars.push_back({std::abs(rc[j]), j});
            }
        }
        // Sort by |rc| descending — push the ones with largest |rc| first.
        std::sort(basic_vars.begin(), basic_vars.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });

        for (const auto& [rc_val, j] : basic_vars) {
            if (n_basic <= nrows) {
                break;
            }
            Real lb = original_.col_lower[j];
            Real ub = original_.col_upper[j];
            Real x = primal_orig_[j];
            bool lb_finite = isFinite(lb);
            bool ub_finite = isFinite(ub);

            if (lb_finite && ub_finite) {
                // Push to nearer bound.
                basis[j] = (x - lb <= ub - x) ? BasisStatus::AtLower : BasisStatus::AtUpper;
                --n_basic;
            } else if (lb_finite) {
                basis[j] = BasisStatus::AtLower;
                --n_basic;
            } else if (ub_finite) {
                basis[j] = BasisStatus::AtUpper;
                --n_basic;
            }
            // Free variables cannot be pushed to a bound.
        }

        // If still too many basics, push basic slacks.
        if (n_basic > nrows) {
            for (Index i = 0; i < nrows; ++i) {
                if (n_basic <= nrows) {
                    break;
                }
                if (basis[ncols + i] != BasisStatus::Basic) {
                    continue;
                }
                Real rl = original_.row_lower[i];
                Real ru = original_.row_upper[i];
                if (isFinite(ru)) {
                    basis[ncols + i] = BasisStatus::AtLower;
                    --n_basic;
                } else if (isFinite(rl)) {
                    basis[ncols + i] = BasisStatus::AtUpper;
                    --n_basic;
                }
            }
        }
    }

    // If too few basics, promote nonbasic variables that are interior.
    if (n_basic < nrows) {
        // First try slacks.
        for (Index i = 0; i < nrows; ++i) {
            if (n_basic >= nrows) {
                break;
            }
            if (basis[ncols + i] == BasisStatus::Basic) {
                continue;
            }
            Real rl = original_.row_lower[i];
            Real ru = original_.row_upper[i];
            bool can_promote = isFinite(rl) || isFinite(ru);
            if (can_promote && basis[ncols + i] != BasisStatus::Fixed) {
                basis[ncols + i] = BasisStatus::Basic;
                ++n_basic;
            }
        }
        // Then try structural variables.
        for (Index j = 0; j < ncols; ++j) {
            if (n_basic >= nrows) {
                break;
            }
            if (basis[j] == BasisStatus::Basic) {
                continue;
            }
            if (basis[j] != BasisStatus::Fixed) {
                basis[j] = BasisStatus::Basic;
                ++n_basic;
            }
        }
    }

    // Phase 2: Basis cleanup using dual simplex.
    DualSimplexSolver ds;
    ds.load(original_);
    ds.setVerbose(false);
    ds.setBasis(basis);

    DualSimplexOptions ds_opts;
    ds_opts.stop_flag = options_.stop_flag;
    ds_opts.enable_scaling = true;
    ds_opts.max_solve_seconds = deadline_.remainingSeconds();  // -1: none (#275)
    ds.setOptions(ds_opts);

    auto result = ds.solve();
    crossover_iters_ = result.iterations;
    crossover_status_ = result.status;

    if (result.status == Status::Optimal) {
        primal_orig_ = ds.getPrimalValues();
        basis_orig_ = ds.getBasis();
        objective_ = result.objective;
        return true;
    }
    // No time left for a cold run. A warm proof of infeasibility or
    // unboundedness stands, even when the deadline passed as it ended.
    if (result.status != Status::Infeasible && result.status != Status::Unbounded &&
        (result.status == Status::TimeLimit || deadline_.passed())) {
        crossover_status_ = Status::TimeLimit;
        return false;
    }

    // Crossover failed — fall back to cold dual simplex.
    if (options_.verbose) {
        std::printf(
            "Crossover cleanup failed (status %d, %d iters), "
            "falling back to cold dual simplex.\n",
            static_cast<int>(result.status), result.iterations);
    }

    DualSimplexSolver ds_cold;
    ds_cold.load(original_);
    ds_cold.setVerbose(false);
    // Bound the cold fallback to avoid unbounded runtime in MIP node context.
    // 3 * (m + n) is generous for a fresh LP and matches typical cold-start
    // expectations in branch-and-cut.
    const Int dim_sum = static_cast<Int>(original_.num_rows) + static_cast<Int>(original_.num_cols);
    ds_cold.setIterationLimit(3 * dim_sum);
    DualSimplexOptions cold_opts;
    cold_opts.stop_flag = options_.stop_flag;
    cold_opts.max_solve_seconds = deadline_.remainingSeconds();
    ds_cold.setOptions(cold_opts);

    auto cold_result = ds_cold.solve();
    crossover_iters_ += cold_result.iterations;
    // Keep a warm run's proof of infeasibility or unboundedness: a cold run
    // that only hits its iteration limit must not hide the contradiction.
    if (crossover_status_ != Status::Infeasible && crossover_status_ != Status::Unbounded) {
        crossover_status_ = cold_result.status;
    }

    if (cold_result.status == Status::Optimal) {
        primal_orig_ = ds_cold.getPrimalValues();
        basis_orig_ = ds_cold.getBasis();
        objective_ = cold_result.objective;
        return true;
    }

    return false;
}

// ============================================================================
// solve() — top-level entry point
// ============================================================================

LpResult BarrierSolver::solve() {
    last_error_.clear();
    gondzio_corrections_ = 0;
    deadline_ = Deadline::after(options_.max_solve_seconds);
    if (!loaded_) {
        status_ = Status::Error;
        last_error_ = "Barrier solver has no loaded problem.";
        return {status_, 0.0, 0, 0.0};
    }

    if (!transformed_ok_) {
        status_ = transformed_infeasible_ ? Status::Infeasible : Status::Error;
        if (status_ == Status::Error) {
            last_error_ = "Barrier standard-form transformation failed.";
        }
        return {status_, 0.0, 0, 0.0};
    }

    std::vector<Real> z, y, s;
    Int iters = 0;

    const BarrierIpmOutcome outcome = solveStandardForm(z, y, s, iters);

    auto timeLimit = [&](Int iterations) {
        dual_eq_.clear();
        reduced_costs_std_.clear();
        primal_orig_.clear();
        basis_orig_.clear();
        status_ = Status::TimeLimit;
        objective_ = 0.0;
        iterations_ = iterations;
        last_error_.clear();
        if (options_.verbose) {
            std::printf("Barrier: time limit of %g s reached.\n", options_.max_solve_seconds);
        }
        return LpResult{status_, objective_, iterations_, 0.0};
    };
    if (outcome == BarrierIpmOutcome::TimeLimit) {
        return timeLimit(iters);
    }

    if (outcome == BarrierIpmOutcome::Failed) {
        if (options_.stop_flag != nullptr && options_.stop_flag->load(std::memory_order_relaxed)) {
            // Clear as the other early ends do, so a reused solver does not
            // hand out the previous solve's results.
            dual_eq_.clear();
            reduced_costs_std_.clear();
            primal_orig_.clear();
            basis_orig_.clear();
            status_ = Status::IterLimit;
            objective_ = 0.0;
            iterations_ = iters;
            last_error_.clear();
            return {status_, objective_, iterations_, 0.0};
        }
        dual_eq_.clear();
        reduced_costs_std_.clear();
        primal_orig_.clear();
        basis_orig_.clear();
        status_ = Status::Error;
        objective_ = 0.0;
        iterations_ = iters;
        if (last_error_.empty()) {
            last_error_ = "Barrier solve failed.";
        }
        return {status_, objective_, iterations_, 0.0};
    }

    reconstructOriginalPrimals(z);
    const bool near_optimal = outcome == BarrierIpmOutcome::NearOptimal;
    const bool feasible = checkOriginalPrimalFeasibility(primal_orig_);
    // A point short of tolerance, or one that misses an original row once
    // unscaled, can still start crossover, which then proves optimality by
    // simplex. Without crossover it is an error.
    const bool needs_crossover = near_optimal || !feasible;
    auto fail = [&](const char* reason) {
        dual_eq_.clear();
        reduced_costs_std_.clear();
        primal_orig_.clear();
        basis_orig_.clear();
        status_ = Status::Error;
        objective_ = 0.0;
        iterations_ = iters;
        last_error_ = reason;
        return LpResult{status_, objective_, iterations_, 0.0};
    };
    if (needs_crossover && !options_.crossover) {
        return fail(near_optimal ? "Barrier stopped short of tolerance and crossover is disabled."
                                 : "Barrier produced a primal-infeasible solution.");
    }

    dual_eq_ = y;
    reduced_costs_std_ = s;

    objective_ = computeOriginalObjective(original_, primal_orig_);
    status_ = Status::Optimal;
    iterations_ = iters;
    last_error_.clear();

    // Run crossover to extract a basic feasible solution.
    if (options_.crossover) {
        if (!runCrossover()) {
            if (crossover_status_ == Status::Infeasible || crossover_status_ == Status::Unbounded) {
                // Simplex on the same LP contradicts the barrier's optimum,
                // so neither verdict is reported.
                return fail("Barrier optimum contradicted by crossover's simplex.");
            }
            if (needs_crossover && crossover_status_ == Status::TimeLimit) {
                return timeLimit(iters + crossover_iters_);
            }
            if (needs_crossover) {
                return fail(near_optimal
                                ? "Barrier stopped short of tolerance and crossover failed."
                                : "Barrier produced a primal-infeasible solution and crossover "
                                  "failed.");
            }
            // Crossover failed but barrier itself converged.
            // Keep barrier solution with empty basis.
            basis_orig_.clear();
            if (options_.verbose) {
                std::printf("Crossover failed; barrier solution retained without basis.\n");
            }
        } else {
            iterations_ += crossover_iters_;
        }
    }

    // Work units must be a deterministic function of the computation
    // performed (barrier iterations x per-iteration cost), never wall-clock
    // time. Folding elapsed seconds into the work measure makes it
    // non-reproducible under machine load, which breaks the deterministic
    // concurrent-root race (the race sums every arm's work_units).
    double work = static_cast<double>(iters) * (4.0 * static_cast<double>(aeq_.numNonzeros()) +
                                                10.0 * static_cast<double>(aeq_.numRows()));

    return {status_, objective_, iterations_, work};
}

}  // namespace mipx
