#include "mipx/exact_refinement.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace mipx {

namespace {

constexpr Real kMinTol = 1e-12;
constexpr Real kMinRationalScale = 1.0;

bool isFinite(Real v) {
    return std::isfinite(v);
}

Real absViolation(Real lhs, Real lower, Real upper) {
    Real violation = 0.0;
    if (isFinite(lower) && lhs < lower) {
        violation = std::max(violation, lower - lhs);
    }
    if (isFinite(upper) && lhs > upper) {
        violation = std::max(violation, lhs - upper);
    }
    return violation;
}

long double safeValue(std::span<const Real> values, Index idx) {
    if (idx < 0 || idx >= static_cast<Index>(values.size())) return 0.0L;
    return static_cast<long double>(values[idx]);
}

bool canScaleToI64(long double v, long double scale) {
    if (!std::isfinite(v)) return false;
    const long double max_mag =
        static_cast<long double>(std::numeric_limits<std::int64_t>::max()) / scale;
    return std::abs(v) <= max_mag;
}

long double scaledRound(long double v, long double scale) {
    return std::llround(v * scale);
}

}  // namespace

LpCertificateMetrics evaluateLpCertificate(const LpProblem& problem,
                                           std::span<const Real> primals,
                                           Real reported_objective,
                                           bool rational_check,
                                           Real certificate_tol,
                                           Real rational_scale,
                                           double* work_units) {
    LpCertificateMetrics metrics;
    metrics.rows_evaluated = problem.num_rows;
    metrics.cols_evaluated = problem.num_cols;
    const Real tol = std::max(kMinTol, certificate_tol);
    const long double ltol = static_cast<long double>(tol);

    for (Index j = 0; j < problem.num_cols; ++j) {
        const Real x = static_cast<Real>(safeValue(primals, j));
        if (isFinite(problem.col_lower[j])) {
            metrics.max_col_violation = std::max(metrics.max_col_violation,
                                                 std::max<Real>(0.0, problem.col_lower[j] - x));
        }
        if (isFinite(problem.col_upper[j])) {
            metrics.max_col_violation = std::max(metrics.max_col_violation,
                                                 std::max<Real>(0.0, x - problem.col_upper[j]));
        }
        if (work_units != nullptr) {
            *work_units += 1e-6;
        }
    }

    for (Index i = 0; i < problem.num_rows; ++i) {
        const auto row = problem.matrix.row(i);
        long double activity = 0.0L;
        for (Index k = 0; k < row.size(); ++k) {
            const Index j = row.indices[k];
            activity += static_cast<long double>(row.values[k]) * safeValue(primals, j);
        }
        const Real lhs = static_cast<Real>(activity);
        metrics.max_row_violation = std::max(
            metrics.max_row_violation,
            absViolation(lhs, problem.row_lower[i], problem.row_upper[i]));
        if (work_units != nullptr) {
            *work_units += static_cast<double>(row.size()) * 1e-6;
        }
    }

    long double recomputed_obj = static_cast<long double>(problem.obj_offset);
    for (Index j = 0; j < problem.num_cols; ++j) {
        recomputed_obj += static_cast<long double>(problem.obj[j]) * safeValue(primals, j);
    }
    metrics.recomputed_objective = static_cast<Real>(recomputed_obj);
    metrics.objective_mismatch = static_cast<Real>(
        std::abs(recomputed_obj - static_cast<long double>(reported_objective)));
    if (work_units != nullptr) {
        *work_units += static_cast<double>(problem.num_cols) * 1e-6;
    }

    if (!rational_check) {
        metrics.rational_supported = true;
        metrics.rational_ok = true;
        return metrics;
    }

    const long double scale =
        std::max(static_cast<long double>(kMinRationalScale),
                 static_cast<long double>(rational_scale));
    const long double scale_sq = scale * scale;
    const long double col_tol_q = ltol * scale + 1.0L;
    const long double row_tol_q = ltol * scale_sq + 1.0L;

    for (Index j = 0; j < problem.num_cols; ++j) {
        const long double x = safeValue(primals, j);
        if (!canScaleToI64(x, scale)) {
            metrics.rational_supported = false;
            metrics.rational_ok = false;
            return metrics;
        }
        const long double xq = scaledRound(x, scale);

        if (isFinite(problem.col_lower[j])) {
            const long double lb = static_cast<long double>(problem.col_lower[j]);
            if (!canScaleToI64(lb, scale)) {
                metrics.rational_supported = false;
                metrics.rational_ok = false;
                return metrics;
            }
            const long double lbq = scaledRound(lb, scale);
            if (xq + col_tol_q < lbq) {
                metrics.rational_ok = false;
            }
        }
        if (isFinite(problem.col_upper[j])) {
            const long double ub = static_cast<long double>(problem.col_upper[j]);
            if (!canScaleToI64(ub, scale)) {
                metrics.rational_supported = false;
                metrics.rational_ok = false;
                return metrics;
            }
            const long double ubq = scaledRound(ub, scale);
            if (xq - col_tol_q > ubq) {
                metrics.rational_ok = false;
            }
        }
        if (work_units != nullptr) {
            *work_units += 1e-6;
        }
    }

    for (Index i = 0; i < problem.num_rows; ++i) {
        const auto row = problem.matrix.row(i);
        long double activity_q = 0.0L;
        for (Index k = 0; k < row.size(); ++k) {
            const Index j = row.indices[k];
            const long double a = static_cast<long double>(row.values[k]);
            const long double x = safeValue(primals, j);
            if (!canScaleToI64(a, scale) || !canScaleToI64(x, scale)) {
                metrics.rational_supported = false;
                metrics.rational_ok = false;
                return metrics;
            }
            const long double aq = scaledRound(a, scale);
            const long double xq = scaledRound(x, scale);
            activity_q += aq * xq;
        }

        if (isFinite(problem.row_lower[i])) {
            const long double lb = static_cast<long double>(problem.row_lower[i]);
            if (!canScaleToI64(lb, scale_sq)) {
                metrics.rational_supported = false;
                metrics.rational_ok = false;
                return metrics;
            }
            const long double lbq = scaledRound(lb, scale_sq);
            if (activity_q + row_tol_q < lbq) {
                metrics.rational_ok = false;
            }
        }
        if (isFinite(problem.row_upper[i])) {
            const long double ub = static_cast<long double>(problem.row_upper[i]);
            if (!canScaleToI64(ub, scale_sq)) {
                metrics.rational_supported = false;
                metrics.rational_ok = false;
                return metrics;
            }
            const long double ubq = scaledRound(ub, scale_sq);
            if (activity_q - row_tol_q > ubq) {
                metrics.rational_ok = false;
            }
        }
        if (work_units != nullptr) {
            *work_units += static_cast<double>(row.size()) * 1e-6;
        }
    }

    return metrics;
}

void iterativePrimalRepair(const LpProblem& problem,
                           std::vector<Real>& primals,
                           Real feasibility_tol,
                           Int max_passes,
                           double* work_units) {
    if (max_passes <= 0 || problem.num_cols <= 0 || problem.num_rows <= 0) return;

    if (primals.size() < static_cast<std::size_t>(problem.num_cols)) {
        primals.resize(problem.num_cols, 0.0);
    }

    const Real tol = std::max(kMinTol, feasibility_tol);

    auto clamp_col = [&](Index j) {
        if (isFinite(problem.col_lower[j]) && primals[j] < problem.col_lower[j]) {
            primals[j] = problem.col_lower[j];
        }
        if (isFinite(problem.col_upper[j]) && primals[j] > problem.col_upper[j]) {
            primals[j] = problem.col_upper[j];
        }
    };

    for (Int pass = 0; pass < max_passes; ++pass) {
        bool changed = false;

        for (Index j = 0; j < problem.num_cols; ++j) {
            const Real before = primals[j];
            clamp_col(j);
            if (primals[j] != before) {
                changed = true;
            }
            if (work_units != nullptr) {
                *work_units += 1e-6;
            }
        }

        for (Index i = 0; i < problem.num_rows; ++i) {
            const auto row = problem.matrix.row(i);
            long double activity_ld = 0.0L;
            for (Index k = 0; k < row.size(); ++k) {
                const Index j = row.indices[k];
                activity_ld += static_cast<long double>(row.values[k]) *
                               static_cast<long double>(primals[j]);
            }
            Real activity = static_cast<Real>(activity_ld);
            if (work_units != nullptr) {
                *work_units += static_cast<double>(row.size()) * 1e-6;
            }

            auto adjust = [&](bool enforce_upper, Real rhs) {
                Real violation = enforce_upper ? (activity - rhs) : (rhs - activity);
                if (violation <= tol) return;

                Index best_col = -1;
                Real best_delta = 0.0;
                Real best_abs_coeff = 0.0;
                Real best_direction = 0.0;
                Real best_coeff = 0.0;

                for (Index k = 0; k < row.size(); ++k) {
                    const Index j = row.indices[k];
                    const Real a = row.values[k];
                    if (std::abs(a) <= kMinTol) continue;

                    bool increase_x = false;
                    if (enforce_upper) {
                        increase_x = (a < 0.0);
                    } else {
                        increase_x = (a > 0.0);
                    }

                    const Real abs_a = std::abs(a);
                    Real cap = kInf;
                    if (increase_x) {
                        if (isFinite(problem.col_upper[j])) {
                            cap = std::max<Real>(0.0, problem.col_upper[j] - primals[j]);
                        }
                    } else if (isFinite(problem.col_lower[j])) {
                        cap = std::max<Real>(0.0, primals[j] - problem.col_lower[j]);
                    }

                    Real delta = violation / abs_a;
                    if (isFinite(cap)) delta = std::min(delta, cap);
                    if (!std::isfinite(delta) || delta <= tol) continue;

                    if (best_col < 0 ||
                        abs_a > best_abs_coeff + kMinTol ||
                        (std::abs(abs_a - best_abs_coeff) <= kMinTol && delta > best_delta + kMinTol)) {
                        best_col = j;
                        best_delta = delta;
                        best_abs_coeff = abs_a;
                        best_direction = increase_x ? 1.0 : -1.0;
                        best_coeff = a;
                    }
                }

                if (best_col < 0 || best_delta <= tol) return;
                const Real before_value = primals[best_col];
                primals[best_col] += best_direction * best_delta;
                clamp_col(best_col);
                const Real applied_delta = primals[best_col] - before_value;
                activity += best_coeff * applied_delta;
                changed = true;
            };

            if (isFinite(problem.row_upper[i]) && activity > problem.row_upper[i] + tol) {
                adjust(true, problem.row_upper[i]);
            }
            if (isFinite(problem.row_lower[i]) && activity < problem.row_lower[i] - tol) {
                adjust(false, problem.row_lower[i]);
            }
        }

        if (!changed) break;
    }
}

}  // namespace mipx
