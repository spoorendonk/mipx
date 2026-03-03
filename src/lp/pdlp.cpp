#include "mipx/pdlp.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>

namespace mipx {

namespace {

inline bool isFinite(Real v) {
    return std::isfinite(v);
}

inline Real l2Norm(std::span<const Real> v) {
    Real sum = 0.0;
    for (Real x : v) sum += x * x;
    return std::sqrt(sum);
}

inline Real dot(std::span<const Real> a, std::span<const Real> b) {
    Real s = 0.0;
    for (Index i = 0; i < static_cast<Index>(a.size()); ++i) s += a[i] * b[i];
    return s;
}

}  // namespace

// ---------------------------------------------------------------------------
// load / buildScaledProblem
// ---------------------------------------------------------------------------

void PdlpSolver::load(const LpProblem& problem) {
    original_ = problem;
    loaded_ = true;
    status_ = Status::Error;
    objective_ = 0.0;
    iterations_ = 0;
    primal_orig_.assign(static_cast<size_t>(original_.num_cols), 0.0);
    dual_orig_.assign(static_cast<size_t>(original_.num_rows), 0.0);
    buildScaledProblem();
}

void PdlpSolver::buildScaledProblem() {
    const Index m = original_.num_rows;
    const Index n = original_.num_cols;

    obj_sign_ = (original_.sense == Sense::Minimize) ? 1.0 : -1.0;

    // Copy objective with sense normalization.
    cscaled_.resize(static_cast<size_t>(n));
    for (Index j = 0; j < n; ++j) {
        cscaled_[j] = obj_sign_ * original_.obj[j];
    }

    // Copy bounds (will be scaled below).
    scaled_col_lower_.assign(original_.col_lower.begin(), original_.col_lower.end());
    scaled_col_upper_.assign(original_.col_upper.begin(), original_.col_upper.end());
    scaled_row_lower_.assign(original_.row_lower.begin(), original_.row_lower.end());
    scaled_row_upper_.assign(original_.row_upper.begin(), original_.row_upper.end());

    // Initialize scales to 1.
    row_scale_.assign(static_cast<size_t>(m), 1.0);
    col_scale_.assign(static_cast<size_t>(n), 1.0);

    // Copy matrix data for in-place scaling.
    auto values_span = original_.matrix.csr_values();
    auto cols_span = original_.matrix.csr_col_indices();
    auto rows_span = original_.matrix.csr_row_starts();
    std::vector<Real> values(values_span.begin(), values_span.end());
    std::vector<Index> col_indices(cols_span.begin(), cols_span.end());
    std::vector<Index> row_starts(rows_span.begin(), rows_span.end());

    // Ruiz scaling: alternating row/col inf-norm equilibration.
    if (options_.do_ruiz_scaling && m > 0 && n > 0 && !values.empty()) {
        std::vector<Real> col_norm(static_cast<size_t>(n), 0.0);
        for (Index it = 0; it < options_.ruiz_iterations; ++it) {
            // Row scaling.
            for (Index i = 0; i < m; ++i) {
                Real row_max = 0.0;
                for (Index k = row_starts[i]; k < row_starts[i + 1]; ++k) {
                    row_max = std::max(row_max, std::abs(values[k]));
                }
                if (row_max <= 1e-12) continue;
                Real scale = 1.0 / std::sqrt(row_max);
                row_scale_[i] *= scale;
                for (Index k = row_starts[i]; k < row_starts[i + 1]; ++k) {
                    values[k] *= scale;
                }
            }

            // Column scaling.
            std::fill(col_norm.begin(), col_norm.end(), 0.0);
            for (Index k = 0; k < static_cast<Index>(values.size()); ++k) {
                Index j = col_indices[k];
                col_norm[j] = std::max(col_norm[j], std::abs(values[k]));
            }
            for (Index j = 0; j < n; ++j) {
                if (col_norm[j] <= 1e-12) continue;
                Real scale = 1.0 / std::sqrt(col_norm[j]);
                col_scale_[j] *= scale;
            }
            for (Index k = 0; k < static_cast<Index>(values.size()); ++k) {
                Index j = col_indices[k];
                if (col_norm[j] > 1e-12) {
                    values[k] *= 1.0 / std::sqrt(col_norm[j]);
                }
            }
        }
    }

    // Scale objective and bounds.
    for (Index j = 0; j < n; ++j) {
        cscaled_[j] *= col_scale_[j];
        if (isFinite(scaled_col_lower_[j])) scaled_col_lower_[j] /= col_scale_[j];
        if (isFinite(scaled_col_upper_[j])) scaled_col_upper_[j] /= col_scale_[j];
    }
    for (Index i = 0; i < m; ++i) {
        if (isFinite(scaled_row_lower_[i])) scaled_row_lower_[i] *= row_scale_[i];
        if (isFinite(scaled_row_upper_[i])) scaled_row_upper_[i] *= row_scale_[i];
    }

    scaled_a_ = SparseMatrix(m, n, std::move(values), std::move(col_indices),
                              std::move(row_starts));

    // Pock-Chambolle diagonal preconditioning.
    sigma_base_.assign(static_cast<size_t>(m), 1.0);
    tau_base_.assign(static_cast<size_t>(n), 1.0);
    if (options_.do_pock_chambolle_scaling && m > 0 && n > 0) {
        std::vector<Real> row_sq(static_cast<size_t>(m), 0.0);
        std::vector<Real> col_sq(static_cast<size_t>(n), 0.0);
        auto vals = scaled_a_.csr_values();
        auto cols = scaled_a_.csr_col_indices();
        auto rows = scaled_a_.csr_row_starts();
        for (Index i = 0; i < m; ++i) {
            for (Index k = rows[i]; k < rows[i + 1]; ++k) {
                Real a = vals[k];
                Index j = cols[k];
                row_sq[i] += a * a;
                col_sq[j] += a * a;
            }
        }
        for (Index i = 0; i < m; ++i) {
            sigma_base_[i] = 1.0 / std::max(std::sqrt(row_sq[i]), 1e-8);
        }
        for (Index j = 0; j < n; ++j) {
            tau_base_[j] = 1.0 / std::max(std::sqrt(col_sq[j]), 1e-8);
        }
    }
}

// ---------------------------------------------------------------------------
// Power iteration for spectral norm estimate
// ---------------------------------------------------------------------------

Real PdlpSolver::estimateSpectralNorm() const {
    const Index n = scaled_a_.numCols();
    const Index m = scaled_a_.numRows();
    if (n == 0 || m == 0) return 1.0;

    std::mt19937 rng(42);
    std::normal_distribution<Real> dist(0.0, 1.0);

    std::vector<Real> x(static_cast<size_t>(n));
    for (auto& v : x) v = dist(rng);
    Real xnorm = l2Norm(x);
    if (xnorm < 1e-15) xnorm = 1.0;
    for (auto& v : x) v /= xnorm;

    std::vector<Real> y(static_cast<size_t>(m));
    std::vector<Real> x_new(static_cast<size_t>(n));

    Real sigma_sq = 1.0;
    for (Index iter = 0; iter < options_.sv_max_iter; ++iter) {
        scaled_a_.multiply(x, y);
        scaled_a_.multiplyTranspose(y, x_new);

        sigma_sq = dot(x, x_new);

        // Residual check: ||x_new - sigma_sq * x||.
        Real resid_sq = 0.0;
        for (Index j = 0; j < n; ++j) {
            Real d = x_new[j] - sigma_sq * x[j];
            resid_sq += d * d;
        }
        Real resid = std::sqrt(resid_sq);

        x.swap(x_new);
        xnorm = l2Norm(x);
        if (xnorm < 1e-15) break;
        for (auto& v : x) v /= xnorm;

        if (resid < options_.sv_tol * std::abs(sigma_sq)) break;
    }
    return std::sqrt(std::max(sigma_sq, 1e-12));
}

// ---------------------------------------------------------------------------
// solve — Halpern PDHG
// ---------------------------------------------------------------------------

LpResult PdlpSolver::solve() {
    if (!loaded_) {
        status_ = Status::Error;
        return {status_, 0.0, 0, 0.0};
    }

    auto t0 = std::chrono::steady_clock::now();

    const Index m = scaled_a_.numRows();
    const Index n = scaled_a_.numCols();

    // Trivial: no variables.
    if (n == 0) {
        status_ = Status::Optimal;
        objective_ = original_.obj_offset;
        iterations_ = 0;
        primal_orig_.assign(static_cast<size_t>(original_.num_cols), 0.0);
        dual_orig_.assign(static_cast<size_t>(original_.num_rows), 0.0);
        return {status_, objective_, 0, 0.0};
    }

    // Bounds-only LP (m == 0): clamp to bounds in cost-minimizing direction.
    if (m == 0) {
        primal_orig_.resize(static_cast<size_t>(n));
        Real obj_val = original_.obj_offset;
        bool unbounded = false;
        for (Index j = 0; j < n; ++j) {
            Real c = cscaled_[j];  // already obj_sign * obj * col_scale
            Real lb = scaled_col_lower_[j];
            Real ub = scaled_col_upper_[j];
            Real xj;
            if (c > 0.0) {
                if (!isFinite(lb)) { unbounded = true; break; }
                xj = lb;
            } else if (c < 0.0) {
                if (!isFinite(ub)) { unbounded = true; break; }
                xj = ub;
            } else {
                xj = isFinite(lb) ? lb : (isFinite(ub) ? ub : 0.0);
            }
            // Unscale.
            primal_orig_[j] = xj * col_scale_[j];
            obj_val += original_.obj[j] * primal_orig_[j];
        }
        if (unbounded) {
            status_ = Status::Unbounded;
            return {status_, 0.0, 0, 0.0};
        }
        dual_orig_.assign(static_cast<size_t>(original_.num_rows), 0.0);
        objective_ = obj_val;
        status_ = Status::Optimal;
        iterations_ = 0;
        return {status_, objective_, 0, 0.0};
    }

    // Step size from spectral norm estimate.
    Real spectral = estimateSpectralNorm();
    Real step = 0.998 / spectral;

    // Primal weight initialization.
    Real primal_weight = options_.primal_weight;

    // Halpern PDHG state.
    const Int N = options_.termination_eval_frequency;

    std::vector<Real> current_x(static_cast<size_t>(n), 0.0);
    std::vector<Real> current_y(static_cast<size_t>(m), 0.0);
    std::vector<Real> initial_x(static_cast<size_t>(n), 0.0);
    std::vector<Real> initial_y(static_cast<size_t>(m), 0.0);
    std::vector<Real> pdhg_x(static_cast<size_t>(n), 0.0);
    std::vector<Real> pdhg_y(static_cast<size_t>(m), 0.0);
    std::vector<Real> reflected_x(static_cast<size_t>(n), 0.0);
    std::vector<Real> reflected_y(static_cast<size_t>(m), 0.0);

    std::vector<Real> at_y(static_cast<size_t>(n), 0.0);
    std::vector<Real> a_xrefl(static_cast<size_t>(m), 0.0);
    std::vector<Real> ax(static_cast<size_t>(m), 0.0);
    std::vector<Real> delta_y_vec(static_cast<size_t>(m), 0.0);
    std::vector<Real> at_delta_y(static_cast<size_t>(n), 0.0);

    // Initialize at origin (or could warm-start later).
    // initial_x = current_x = 0, initial_y = current_y = 0.

    Int inner_count = 0;
    Int total_count = 0;

    Real initial_fpe = -1.0;
    Real last_trial_fpe = std::numeric_limits<Real>::infinity();

    // PI controller state.
    Real pw_error_sum = 0.0;
    Real pw_last_error = 0.0;
    Real best_primal_weight = primal_weight;
    Real best_pw_score = std::numeric_limits<Real>::infinity();

    // Precompute norms for relative termination.
    Real c_norm = l2Norm(cscaled_);
    Real bound_norm = 0.0;
    for (Index i = 0; i < m; ++i) {
        Real rl = scaled_row_lower_[i];
        Real ru = scaled_row_upper_[i];
        if (isFinite(rl)) bound_norm += rl * rl;
        if (isFinite(ru)) bound_norm += ru * ru;
    }
    bound_norm = std::sqrt(bound_norm);

    bool converged = false;

    while (total_count < options_.max_iter) {
        if (options_.stop_flag != nullptr &&
            options_.stop_flag->load(std::memory_order_relaxed)) {
            status_ = Status::IterLimit;
            iterations_ = total_count;
            return {status_, 0.0, iterations_, 0.0};
        }

        // Run a batch of N Halpern PDHG iterations.
        Int batch = std::min(N, options_.max_iter - total_count);
        for (Int k_offset = 1; k_offset <= batch; ++k_offset) {
            Real lambda = static_cast<Real>(inner_count + k_offset) /
                          static_cast<Real>(inner_count + k_offset + 1);

            // 1. A^T * current_y → at_y.
            scaled_a_.multiplyTranspose(current_y, at_y);

            // 2. Primal step + project + reflect + Halpern.
            for (Index j = 0; j < n; ++j) {
                Real tau = step * tau_base_[j] / primal_weight;
                Real temp = current_x[j] - tau * (cscaled_[j] + at_y[j]);
                pdhg_x[j] = std::clamp(temp, scaled_col_lower_[j], scaled_col_upper_[j]);
                reflected_x[j] = 2.0 * pdhg_x[j] - current_x[j];
                current_x[j] = lambda * reflected_x[j] + (1.0 - lambda) * initial_x[j];
            }

            // 3. A * reflected_x → a_xrefl.
            scaled_a_.multiply(reflected_x, a_xrefl);

            // 4. Dual step + proximal + reflect + Halpern.
            for (Index i = 0; i < m; ++i) {
                Real sigma = step * sigma_base_[i] * primal_weight;
                Real v = current_y[i] + sigma * a_xrefl[i];
                Real rl = scaled_row_lower_[i];
                Real ru = scaled_row_upper_[i];

                Real pdhg_yi;
                if (rl == ru) {
                    // Equality constraint.
                    pdhg_yi = v - sigma * rl;
                } else if (!isFinite(rl)) {
                    // Upper-bound only: y >= 0 after projection.
                    pdhg_yi = std::max(0.0, v - sigma * ru);
                } else if (!isFinite(ru)) {
                    // Lower-bound only: y <= 0 after projection.
                    pdhg_yi = std::min(0.0, v - sigma * rl);
                } else {
                    // Ranged constraint.
                    if (v >= sigma * ru)
                        pdhg_yi = v - sigma * ru;
                    else if (v <= sigma * rl)
                        pdhg_yi = v - sigma * rl;
                    else
                        pdhg_yi = 0.0;
                }
                pdhg_y[i] = pdhg_yi;
                reflected_y[i] = 2.0 * pdhg_yi - current_y[i];
                current_y[i] = lambda * reflected_y[i] + (1.0 - lambda) * initial_y[i];
            }
        }
        inner_count += batch;
        total_count += batch;

        // --- Fixed-point error for restart decision ---
        // delta_x = pdhg_x - initial_x, delta_y = pdhg_y - initial_y.
        Real movement = 0.0;
        Real delta_x_sq = 0.0, delta_y_sq = 0.0;
        for (Index j = 0; j < n; ++j) {
            Real dx = pdhg_x[j] - initial_x[j];
            delta_x_sq += dx * dx;
        }
        for (Index i = 0; i < m; ++i) {
            Real dy = pdhg_y[i] - initial_y[i];
            delta_y_sq += dy * dy;
        }
        movement = delta_x_sq * primal_weight + delta_y_sq / primal_weight;

        // Interaction term: 2 * step * dot(A^T * delta_y, delta_x).
        for (Index i = 0; i < m; ++i) delta_y_vec[i] = pdhg_y[i] - initial_y[i];
        scaled_a_.multiplyTranspose(delta_y_vec, at_delta_y);
        Real interaction = 0.0;
        for (Index j = 0; j < n; ++j) {
            interaction += at_delta_y[j] * (pdhg_x[j] - initial_x[j]);
        }
        interaction *= 2.0 * step;

        Real fpe = std::sqrt(std::max(movement + interaction, 0.0));

        if (initial_fpe < 0.0) initial_fpe = fpe;

        // --- Convergence check on pdhg_x, pdhg_y ---
        scaled_a_.multiply(pdhg_x, ax);

        // Primal residual.
        Real primal_resid_sq = 0.0;
        for (Index i = 0; i < m; ++i) {
            Real violation = ax[i] - std::clamp(ax[i], scaled_row_lower_[i],
                                                 scaled_row_upper_[i]);
            primal_resid_sq += violation * violation;
        }
        Real abs_primal_resid = std::sqrt(primal_resid_sq);
        Real rel_primal_resid = abs_primal_resid / (1.0 + bound_norm);

        // Dual residual: for each j, check reduced cost feasibility.
        scaled_a_.multiplyTranspose(pdhg_y, at_y);
        Real dual_resid_sq = 0.0;
        for (Index j = 0; j < n; ++j) {
            Real grad = cscaled_[j] + at_y[j];
            Real xj = pdhg_x[j];
            Real lb = scaled_col_lower_[j];
            Real ub = scaled_col_upper_[j];
            // At interior: grad should be 0.
            // At lower bound: grad >= 0.
            // At upper bound: grad <= 0.
            Real dr = 0.0;
            if (xj <= lb + 1e-12 && isFinite(lb)) {
                dr = std::min(grad, 0.0);
            } else if (xj >= ub - 1e-12 && isFinite(ub)) {
                dr = std::max(grad, 0.0);
            } else {
                dr = grad;
            }
            dual_resid_sq += dr * dr;
        }
        Real abs_dual_resid = std::sqrt(dual_resid_sq);
        Real rel_dual_resid = abs_dual_resid / (1.0 + c_norm);

        // Primal objective.
        Real pobj_scaled = dot(cscaled_, std::span<const Real>(pdhg_x));
        Real pobj = original_.obj_offset + obj_sign_ * pobj_scaled;

        // Dual objective via support function.
        // dobj = -delta*_{[lb,ub]}(-grad) - delta*_{[rl,ru]}(y)
        // where delta*(z) = sup_{s in [l,u]} z*s = max(0,z)*u + min(0,z)*l.
        // Corrected dual objective: project reduced costs and dual values
        // to be sign-feasible, then compute dual function.
        // d(y) = sum_j col_term[j] + sum_i row_term[i] where:
        //   col_term[j] = g_j * lb_j if g_j > 0, g_j * ub_j if g_j < 0
        //   row_term[i] = -y_i * ru_i if y_i > 0, -y_i * rl_i if y_i < 0
        Real dobj_col = 0.0;
        for (Index j = 0; j < n; ++j) {
            Real grad = cscaled_[j] + at_y[j];
            Real lb = scaled_col_lower_[j];
            Real ub = scaled_col_upper_[j];
            // Project grad to dual-feasible region.
            if (!isFinite(lb)) grad = std::min(grad, 0.0);
            if (!isFinite(ub)) grad = std::max(grad, 0.0);
            if (grad > 0.0 && isFinite(lb))
                dobj_col += grad * lb;
            else if (grad < 0.0 && isFinite(ub))
                dobj_col += grad * ub;
        }
        Real dobj_row = 0.0;
        for (Index i = 0; i < m; ++i) {
            Real yi = pdhg_y[i];
            Real rl = scaled_row_lower_[i];
            Real ru = scaled_row_upper_[i];
            // Project y to sign-feasible region.
            if (!isFinite(ru)) yi = std::min(yi, 0.0);
            if (!isFinite(rl)) yi = std::max(yi, 0.0);
            if (yi > 0.0 && isFinite(ru))
                dobj_row -= yi * ru;
            else if (yi < 0.0 && isFinite(rl))
                dobj_row -= yi * rl;
        }
        Real dobj = original_.obj_offset + obj_sign_ * (dobj_col + dobj_row);

        Real rel_gap = std::abs(pobj - dobj) /
                       (1.0 + std::abs(pobj) + std::abs(dobj));

        if (options_.verbose && (total_count <= N || total_count % (10 * N) == 0)) {
            std::printf(
                "PDLP %7d  pobj=% .8e  pinf=% .2e  dinf=% .2e  gap=% .2e  "
                "fpe=% .2e  pw=% .2e  step=% .2e\n",
                total_count, pobj, rel_primal_resid, rel_dual_resid, rel_gap,
                fpe, primal_weight, step);
        }

        if (rel_primal_resid <= options_.primal_tol &&
            rel_dual_resid <= options_.dual_tol &&
            rel_gap <= options_.optimality_tol) {
            converged = true;
            break;
        }

        // --- Restart decision ---
        bool do_restart = false;
        if (total_count == N) {
            // Always restart after first batch.
            do_restart = true;
        } else if (initial_fpe > 0.0 && fpe <= options_.restart_sufficient_decay * initial_fpe) {
            do_restart = true;
        } else if (initial_fpe > 0.0 && fpe <= options_.restart_necessary_decay * initial_fpe &&
                   fpe > last_trial_fpe) {
            do_restart = true;
        } else if (inner_count >= static_cast<Int>(options_.restart_artificial_fraction *
                                                    static_cast<Real>(total_count))) {
            do_restart = true;
        }

        last_trial_fpe = fpe;

        if (do_restart) {
            // PI controller for primal weight update.
            if (options_.update_primal_weight) {
                Real primal_dist = 0.0, dual_dist = 0.0;
                for (Index j = 0; j < n; ++j) {
                    Real d = pdhg_x[j] - initial_x[j];
                    primal_dist += d * d;
                }
                primal_dist = std::sqrt(primal_dist);
                for (Index i = 0; i < m; ++i) {
                    Real d = pdhg_y[i] - initial_y[i];
                    dual_dist += d * d;
                }
                dual_dist = std::sqrt(dual_dist);

                if (primal_dist > 1e-15 && dual_dist > 1e-15) {
                    Real error = std::log(dual_dist) - std::log(primal_dist) -
                                 std::log(primal_weight);
                    pw_error_sum = options_.pid_i_smooth * pw_error_sum + error;
                    Real delta_error = error - pw_last_error;
                    primal_weight *= std::exp(
                        options_.pid_kp * error +
                        options_.pid_ki * pw_error_sum +
                        options_.pid_kd * delta_error);
                    primal_weight = std::clamp(primal_weight, 1e-8, 1e8);
                    pw_last_error = error;

                    // Track best weight.
                    Real score = std::abs(
                        std::log10(std::max(rel_dual_resid, 1e-15)) -
                        std::log10(std::max(rel_primal_resid, 1e-15)));
                    if (score < best_pw_score) {
                        best_pw_score = score;
                        best_primal_weight = primal_weight;
                    }
                } else {
                    primal_weight = best_primal_weight;
                }
            }

            // Restart iterates.
            for (Index j = 0; j < n; ++j) {
                initial_x[j] = pdhg_x[j];
                current_x[j] = pdhg_x[j];
            }
            for (Index i = 0; i < m; ++i) {
                initial_y[i] = pdhg_y[i];
                current_y[i] = pdhg_y[i];
            }
            inner_count = 0;
            initial_fpe = -1.0;  // Will be set on next batch.
            last_trial_fpe = std::numeric_limits<Real>::infinity();
        }
    }

    iterations_ = total_count;

    if (converged) {
        // Reconstruct original solution.
        primal_orig_.resize(static_cast<size_t>(original_.num_cols));
        for (Index j = 0; j < n; ++j) {
            Real xj = pdhg_x[j] * col_scale_[j];
            if (isFinite(original_.col_lower[j]))
                xj = std::max(xj, original_.col_lower[j]);
            if (isFinite(original_.col_upper[j]))
                xj = std::min(xj, original_.col_upper[j]);
            primal_orig_[j] = xj;
        }
        dual_orig_.resize(static_cast<size_t>(original_.num_rows));
        for (Index i = 0; i < m; ++i) {
            dual_orig_[i] = obj_sign_ * pdhg_y[i] * row_scale_[i];
        }

        // Compute objective from original primals.
        Real obj_val = original_.obj_offset;
        for (Index j = 0; j < n; ++j) {
            obj_val += original_.obj[j] * primal_orig_[j];
        }
        objective_ = obj_val;
        status_ = Status::Optimal;
    } else {
        status_ = Status::IterLimit;
        objective_ = 0.0;
    }

    double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    double work = static_cast<double>(total_count) *
                  (4.0 * static_cast<double>(scaled_a_.numNonzeros()) +
                   2.0 * static_cast<double>(m + n));
    work += seconds * 1e-6;
    return {status_, objective_, iterations_, work};
}

// ---------------------------------------------------------------------------
// Solution accessors
// ---------------------------------------------------------------------------

std::vector<Real> PdlpSolver::getPrimalValues() const {
    return primal_orig_;
}

std::vector<Real> PdlpSolver::getDualValues() const {
    return dual_orig_;
}

std::vector<Real> PdlpSolver::getReducedCosts() const {
    std::vector<Real> rc(static_cast<size_t>(original_.num_cols), 0.0);
    if (!loaded_ || dual_orig_.empty()) return rc;

    std::vector<Real> at_y(static_cast<size_t>(original_.num_cols), 0.0);
    original_.matrix.multiplyTranspose(dual_orig_, at_y);
    for (Index j = 0; j < original_.num_cols; ++j) {
        rc[j] = original_.obj[j] + at_y[j];
    }
    return rc;
}

std::vector<BasisStatus> PdlpSolver::getBasis() const {
    return {};
}

void PdlpSolver::setBasis(std::span<const BasisStatus> basis) {
    (void)basis;
}

// ---------------------------------------------------------------------------
// Incremental modifications
// ---------------------------------------------------------------------------

void PdlpSolver::addRows(std::span<const Index> starts,
                          std::span<const Index> indices,
                          std::span<const Real> values,
                          std::span<const Real> lower,
                          std::span<const Real> upper) {
    if (!loaded_) return;
    if (starts.empty()) return;

    Index rows_to_add = static_cast<Index>(lower.size());
    if (rows_to_add != static_cast<Index>(upper.size())) return;
    if (static_cast<Index>(starts.size()) != rows_to_add + 1) return;

    for (Index r = 0; r < rows_to_add; ++r) {
        Index s = starts[r];
        Index e = starts[r + 1];
        if (s < 0 || e < s || e > static_cast<Index>(indices.size()) ||
            e > static_cast<Index>(values.size())) {
            return;
        }
        std::span<const Index> row_idx(indices.data() + s, static_cast<size_t>(e - s));
        std::span<const Real> row_val(values.data() + s, static_cast<size_t>(e - s));
        original_.matrix.addRow(row_idx, row_val);
        original_.row_lower.push_back(lower[r]);
        original_.row_upper.push_back(upper[r]);
        original_.row_names.push_back("");
    }
    original_.num_rows += rows_to_add;
    buildScaledProblem();
}

void PdlpSolver::removeRows(std::span<const Index> rows) {
    if (!loaded_) return;
    if (rows.empty()) return;

    std::vector<Index> sorted(rows.begin(), rows.end());
    std::sort(sorted.begin(), sorted.end());
    sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());

    for (auto it = sorted.rbegin(); it != sorted.rend(); ++it) {
        Index r = *it;
        if (r < 0 || r >= original_.num_rows) continue;
        original_.matrix.removeRowStable(r);
        original_.row_lower.erase(original_.row_lower.begin() + r);
        original_.row_upper.erase(original_.row_upper.begin() + r);
        if (r < static_cast<Index>(original_.row_names.size())) {
            original_.row_names.erase(original_.row_names.begin() + r);
        }
        --original_.num_rows;
    }
    buildScaledProblem();
}

void PdlpSolver::setColBounds(Index col, Real lower, Real upper) {
    if (!loaded_) return;
    if (col < 0 || col >= original_.num_cols) return;
    original_.col_lower[col] = lower;
    original_.col_upper[col] = upper;
    buildScaledProblem();
}

void PdlpSolver::setObjective(std::span<const Real> obj) {
    if (!loaded_) return;
    if (static_cast<Index>(obj.size()) != original_.num_cols) return;
    original_.obj.assign(obj.begin(), obj.end());
    buildScaledProblem();
}

}  // namespace mipx
