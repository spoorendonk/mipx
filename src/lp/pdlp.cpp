#include "mipx/pdlp.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>

#ifdef MIPX_HAS_CUDA
#include <cuda_runtime_api.h>
#include <cusparse.h>
#include "pdlp_kernels.cuh"
#endif

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

    // Build explicit A^T in CSR format (used by GPU path).
    buildTransposeCSR();

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

void PdlpSolver::buildTransposeCSR() {
    // Build A^T in CSR format via standard CSR→CSC transpose.
    // A^T in CSR with n rows and m cols is equivalent to A in CSC.
    const Index m = scaled_a_.numRows();
    const Index n = scaled_a_.numCols();
    const Index nnz = scaled_a_.numNonzeros();

    auto vals = scaled_a_.csr_values();
    auto cols = scaled_a_.csr_col_indices();
    auto rows = scaled_a_.csr_row_starts();

    at_row_starts_.assign(static_cast<size_t>(n + 1), 0);
    at_col_indices_.resize(static_cast<size_t>(nnz));
    at_values_.resize(static_cast<size_t>(nnz));

    if (nnz == 0) return;

    // Count nnz per column of A (= nnz per row of A^T).
    for (Index k = 0; k < nnz; ++k) {
        ++at_row_starts_[static_cast<size_t>(cols[k] + 1)];
    }
    // Prefix sum.
    for (Index j = 1; j <= n; ++j) {
        at_row_starts_[j] += at_row_starts_[j - 1];
    }
    // Scatter.
    std::vector<Index> cursor(at_row_starts_.begin(), at_row_starts_.begin() + n);
    for (Index i = 0; i < m; ++i) {
        for (Index k = rows[i]; k < rows[i + 1]; ++k) {
            Index j = cols[k];
            Index pos = cursor[j]++;
            at_col_indices_[pos] = i;
            at_values_[pos] = vals[k];
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

#ifdef MIPX_HAS_CUDA
    // GPU path: all iterate vectors stay on device.
    bool should_use_gpu = options_.use_gpu &&
                          m >= options_.gpu_min_rows &&
                          scaled_a_.numNonzeros() >= options_.gpu_min_nnz;
    if (should_use_gpu) {
        return solveGpu();
    }
#endif

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
// GPU solve path
// ---------------------------------------------------------------------------

#ifdef MIPX_HAS_CUDA

namespace {

inline bool cudaOk(cudaError_t code) { return code == cudaSuccess; }
inline bool cusparseOk(cusparseStatus_t code) { return code == CUSPARSE_STATUS_SUCCESS; }

}  // namespace

LpResult PdlpSolver::solveGpu() {
    auto t0 = std::chrono::steady_clock::now();

    const Index m = scaled_a_.numRows();
    const Index n = scaled_a_.numCols();
    const Index nnz = scaled_a_.numNonzeros();
    const Index at_nnz = static_cast<Index>(at_values_.size());

    used_gpu_ = false;

    // Arena layout: 12n + 11m + 5 doubles.
    // Iterate vectors (8n + 7m):
    //   current_x(n), current_y(m), initial_x(n), initial_y(m),
    //   pdhg_x(n), pdhg_y(m), reflected_x(n), reflected_y(m),
    //   at_y(n), a_xrefl(m), ax(m), delta_y(m), at_delta_y(n)
    // Constant vectors (4n + 4m):
    //   cscaled(n), col_lower(n), col_upper(n), tau_base(n),
    //   row_lower(m), row_upper(m), sigma_base(m)
    // Power iteration (2n + m): pi_x(n), pi_y(m), pi_x_new(n)
    // Scalars (5): scratch(1), lambda(1), inner_count(1 Int, 1-double slot),
    //              step(1), primal_weight(1)
    const size_t sn = static_cast<size_t>(n);
    const size_t sm = static_cast<size_t>(m);
    const size_t arena_doubles = 12 * sn + 11 * sm + 5;

    Real* d_arena = nullptr;
    if (!cudaOk(cudaMalloc(reinterpret_cast<void**>(&d_arena),
                            arena_doubles * sizeof(Real)))) {
        // Fall back to CPU path by returning a sentinel.
        return solve();  // Will skip GPU because used_gpu_ check won't re-enter.
    }

    // Pointers into arena.
    Real* d_current_x   = d_arena;
    Real* d_current_y   = d_current_x + n;
    Real* d_initial_x   = d_current_y + m;
    Real* d_initial_y   = d_initial_x + n;
    Real* d_pdhg_x      = d_initial_y + m;
    Real* d_pdhg_y      = d_pdhg_x + n;
    Real* d_reflected_x = d_pdhg_y + m;
    Real* d_reflected_y = d_reflected_x + n;  // unused but reserved
    Real* d_at_y        = d_reflected_y + m;
    Real* d_a_xrefl     = d_at_y + n;
    Real* d_ax          = d_a_xrefl + m;
    Real* d_delta_y     = d_ax + m;
    Real* d_at_delta_y  = d_delta_y + m;
    Real* d_cscaled     = d_at_delta_y + n;
    Real* d_col_lower   = d_cscaled + n;
    Real* d_col_upper   = d_col_lower + n;
    Real* d_row_lower   = d_col_upper + n;
    Real* d_row_upper   = d_row_lower + m;
    Real* d_tau_base    = d_row_upper + m;
    Real* d_sigma_base  = d_tau_base + n;
    Real* d_pi_x        = d_sigma_base + m;
    Real* d_pi_y        = d_pi_x + n;
    Real* d_pi_x_new    = d_pi_y + m;
    Real* d_scratch     = d_pi_x_new + n;
    Real* d_lambda      = d_scratch + 1;
    Int*  d_inner_count = reinterpret_cast<Int*>(d_lambda + 1);
    // d_inner_count occupies 1 Int within a double-aligned slot; skip a full
    // double (not just sizeof(Int)) to keep subsequent Real* 8-byte aligned.
    Real* d_step         = d_lambda + 2;  // skip lambda(1) + inner_count_slot(1)
    Real* d_primal_weight = d_step + 1;

    // Zero all iterate vectors (constants are overwritten by uploads below).
    cudaMemset(d_arena, 0, arena_doubles * sizeof(Real));

    // Upload constant vectors.
    cudaMemcpy(d_cscaled, cscaled_.data(), sn * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_lower, scaled_col_lower_.data(), sn * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_upper, scaled_col_upper_.data(), sn * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_lower, scaled_row_lower_.data(), sm * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_upper, scaled_row_upper_.data(), sm * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tau_base, tau_base_.data(), sn * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sigma_base, sigma_base_.data(), sm * sizeof(Real), cudaMemcpyHostToDevice);

    // Upload A CSR to device.
    auto a_vals = scaled_a_.csr_values();
    auto a_cols = scaled_a_.csr_col_indices();
    auto a_rows = scaled_a_.csr_row_starts();

    Real* d_a_values = nullptr;
    Index* d_a_col_indices = nullptr;
    Index* d_a_row_starts = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&d_a_values), sizeof(Real) * nnz);
    cudaMalloc(reinterpret_cast<void**>(&d_a_col_indices), sizeof(Index) * nnz);
    cudaMalloc(reinterpret_cast<void**>(&d_a_row_starts), sizeof(Index) * (m + 1));
    cudaMemcpy(d_a_values, a_vals.data(), sizeof(Real) * nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_col_indices, a_cols.data(), sizeof(Index) * nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_row_starts, a_rows.data(), sizeof(Index) * (m + 1), cudaMemcpyHostToDevice);

    // Upload A^T CSR to device.
    Real* d_at_values = nullptr;
    Index* d_at_col_indices = nullptr;
    Index* d_at_row_starts = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&d_at_values), sizeof(Real) * at_nnz);
    cudaMalloc(reinterpret_cast<void**>(&d_at_col_indices), sizeof(Index) * at_nnz);
    cudaMalloc(reinterpret_cast<void**>(&d_at_row_starts), sizeof(Index) * (n + 1));
    cudaMemcpy(d_at_values, at_values_.data(), sizeof(Real) * at_nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_at_col_indices, at_col_indices_.data(), sizeof(Index) * at_nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_at_row_starts, at_row_starts_.data(), sizeof(Index) * (n + 1), cudaMemcpyHostToDevice);

    // cuSPARSE setup.
    cusparseHandle_t handle = nullptr;
    cusparseCreate(&handle);
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    cusparseSetStream(handle, stream);

    // A sparse matrix descriptor (m x n).
    cusparseSpMatDescr_t a_desc = nullptr;
    cusparseCreateCsr(&a_desc, m, n, nnz, d_a_row_starts, d_a_col_indices, d_a_values,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

    // A^T sparse matrix descriptor (n x m).
    cusparseSpMatDescr_t at_desc = nullptr;
    cusparseCreateCsr(&at_desc, n, m, at_nnz, d_at_row_starts, d_at_col_indices, d_at_values,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

    // Dense vector descriptors.
    auto makeVecDesc = [](cusparseDnVecDescr_t& desc, int64_t size, Real* data) {
        cusparseCreateDnVec(&desc, size, data, CUDA_R_64F);
    };
    cusparseDnVecDescr_t desc_current_y = nullptr, desc_at_y = nullptr;
    cusparseDnVecDescr_t desc_reflected_x = nullptr, desc_a_xrefl = nullptr;
    cusparseDnVecDescr_t desc_pdhg_x = nullptr, desc_ax = nullptr;
    cusparseDnVecDescr_t desc_pdhg_y = nullptr;
    cusparseDnVecDescr_t desc_delta_y = nullptr, desc_at_delta_y = nullptr;
    cusparseDnVecDescr_t desc_pi_x = nullptr, desc_pi_y = nullptr, desc_pi_x_new = nullptr;

    makeVecDesc(desc_current_y, m, d_current_y);
    makeVecDesc(desc_at_y, n, d_at_y);
    makeVecDesc(desc_reflected_x, n, d_reflected_x);
    makeVecDesc(desc_a_xrefl, m, d_a_xrefl);
    makeVecDesc(desc_pdhg_x, n, d_pdhg_x);
    makeVecDesc(desc_ax, m, d_ax);
    makeVecDesc(desc_pdhg_y, m, d_pdhg_y);
    makeVecDesc(desc_delta_y, m, d_delta_y);
    makeVecDesc(desc_at_delta_y, n, d_at_delta_y);
    makeVecDesc(desc_pi_x, n, d_pi_x);
    makeVecDesc(desc_pi_y, m, d_pi_y);
    makeVecDesc(desc_pi_x_new, n, d_pi_x_new);

    const Real alpha_one = 1.0, beta_zero = 0.0;

    // Allocate SpMV buffers.
    // A * x (non-transpose): A_desc * desc_reflected_x -> desc_a_xrefl
    size_t buf_a_size = 0, buf_at_size = 0;
    cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha_one, a_desc, desc_reflected_x, &beta_zero, desc_a_xrefl,
                            CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, &buf_a_size);
    // A^T * y (non-transpose on A^T desc): AT_desc * desc_current_y -> desc_at_y
    cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha_one, at_desc, desc_current_y, &beta_zero, desc_at_y,
                            CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, &buf_at_size);

    void* d_buf_a = nullptr;
    void* d_buf_at = nullptr;
    if (buf_a_size > 0) cudaMalloc(&d_buf_a, buf_a_size);
    if (buf_at_size > 0) cudaMalloc(&d_buf_at, buf_at_size);

    // Convergence metrics.
    GpuConvergenceMetrics* d_metrics = nullptr;
    GpuConvergenceMetrics* h_metrics = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&d_metrics), sizeof(GpuConvergenceMetrics));
    cudaMallocHost(reinterpret_cast<void**>(&h_metrics), sizeof(GpuConvergenceMetrics));

    used_gpu_ = true;

    // SpMV helper lambdas.
    auto spmv_a = [&](cusparseDnVecDescr_t in, cusparseDnVecDescr_t out) {
        cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                     &alpha_one, a_desc, in, &beta_zero, out,
                     CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, d_buf_a);
    };
    auto spmv_at = [&](cusparseDnVecDescr_t in, cusparseDnVecDescr_t out) {
        cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                     &alpha_one, at_desc, in, &beta_zero, out,
                     CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, d_buf_at);
    };

    // -----------------------------------------------------------------------
    // Power iteration on device for spectral norm.
    // -----------------------------------------------------------------------
    {
        std::mt19937 rng(42);
        std::normal_distribution<Real> dist(0.0, 1.0);
        std::vector<Real> x_host(sn);
        for (auto& v : x_host) v = dist(rng);
        Real xnorm = l2Norm(x_host);
        if (xnorm < 1e-15) xnorm = 1.0;
        for (auto& v : x_host) v /= xnorm;
        cudaMemcpy(d_pi_x, x_host.data(), sn * sizeof(Real), cudaMemcpyHostToDevice);

        Real sigma_sq = 1.0;
        for (Index iter = 0; iter < options_.sv_max_iter; ++iter) {
            // y = A * x
            spmv_a(desc_pi_x, desc_pi_y);
            // x_new = A^T * y
            spmv_at(desc_pi_y, desc_pi_x_new);

            sigma_sq = launchDotProduct(n, d_pi_x, d_pi_x_new, d_scratch, stream);

            // Residual: ||x_new - sigma_sq * x||
            // Compute x_new = x_new - sigma_sq * x, then norm
            launchAxpby(n, 1.0, d_pi_x_new, -sigma_sq, d_pi_x, stream);
            // Now d_pi_x = x_new - sigma_sq * x_old. Get its norm.
            Real resid_sq = launchDotProduct(n, d_pi_x, d_pi_x, d_scratch, stream);
            Real resid = std::sqrt(resid_sq);

            // Restore: swap pi_x and pi_x_new (just copy x_new → pi_x and normalize).
            xnorm = std::sqrt(launchDotProduct(n, d_pi_x_new, d_pi_x_new, d_scratch, stream));
            if (xnorm < 1e-15) break;
            // pi_x = pi_x_new / xnorm
            cudaMemcpy(d_pi_x, d_pi_x_new, sn * sizeof(Real), cudaMemcpyDeviceToDevice);
            launchScale(n, 1.0 / xnorm, d_pi_x, stream);

            if (resid < options_.sv_tol * std::abs(sigma_sq)) break;
        }

        Real spectral = std::sqrt(std::max(sigma_sq, 1e-12));
        // Step size from spectral norm.
        Real step_init = 0.998 / spectral;
        // Store into a variable accessible below.
        // We'll use 'step' directly in the main loop.
        cudaMemcpy(d_scratch, &step_init, sizeof(Real), cudaMemcpyHostToDevice);
    }

    // Retrieve step from scratch (stored during power iteration).
    Real step;
    cudaMemcpy(&step, d_scratch, sizeof(Real), cudaMemcpyDeviceToHost);

    // Upload step and primal_weight to device-resident scalars so kernels
    // read via pointer indirection — this allows CUDA graphs to survive restarts.
    Real primal_weight = options_.primal_weight;
    cudaMemcpy(d_step, &step, sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_primal_weight, &primal_weight, sizeof(Real), cudaMemcpyHostToDevice);
    const Int N = options_.termination_eval_frequency;

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

    // Initialize d_inner_count on device.
    cudaMemset(d_inner_count, 0, sizeof(Int));

    // CUDA graph capture state.
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graphExec = nullptr;
    bool graphCaptureFailed = false;

    // Helper: launch one batch of N inner iterations (no graph).
    auto launchBatch = [&](Int batch) {
        for (Int k_offset = 1; k_offset <= batch; ++k_offset) {
            launchComputeLambda(d_lambda, d_inner_count, k_offset, stream);
            spmv_at(desc_current_y, desc_at_y);
            launchPrimalHalpernStep(
                n, d_lambda, d_step, d_primal_weight,
                d_current_x, d_initial_x, d_pdhg_x, d_reflected_x,
                d_cscaled, d_at_y, d_tau_base, d_col_lower, d_col_upper,
                stream);
            spmv_a(desc_reflected_x, desc_a_xrefl);
            launchDualHalpernStep(
                m, d_lambda, d_step, d_primal_weight,
                d_current_y, d_initial_y, d_pdhg_y, d_reflected_y,
                d_a_xrefl, d_sigma_base, d_row_lower, d_row_upper,
                stream);
        }
    };

    // -----------------------------------------------------------------------
    // Main PDHG loop (GPU-resident).
    // -----------------------------------------------------------------------
    while (total_count < options_.max_iter) {
        if (options_.stop_flag != nullptr &&
            options_.stop_flag->load(std::memory_order_relaxed)) {
            status_ = Status::IterLimit;
            iterations_ = total_count;
            goto cleanup;
        }

        {
            Int batch = std::min(N, options_.max_iter - total_count);

            if (batch == N && !graphCaptureFailed) {
                if (graphExec == nullptr) {
                    // First full batch: capture the graph.
                    cudaMemcpyAsync(d_inner_count, &inner_count, sizeof(Int),
                                    cudaMemcpyHostToDevice, stream);
                    cudaStreamSynchronize(stream);
                    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
                    launchBatch(batch);
                    cudaStreamEndCapture(stream, &graph);
                    if (graph == nullptr) {
                        // Graph capture failed permanently — fall back to
                        // non-graph launches for remainder of solve.
                        graphCaptureFailed = true;
                        cudaMemcpyAsync(d_inner_count, &inner_count, sizeof(Int),
                                        cudaMemcpyHostToDevice, stream);
                        launchBatch(batch);
                    } else {
                        cudaGraphInstantiate(&graphExec, graph, 0);
                        // Execute the captured graph for this first batch.
                        cudaMemcpyAsync(d_inner_count, &inner_count, sizeof(Int),
                                        cudaMemcpyHostToDevice, stream);
                        cudaGraphLaunch(graphExec, stream);
                    }
                } else {
                    // Subsequent full batches: update d_inner_count and replay.
                    cudaMemcpyAsync(d_inner_count, &inner_count, sizeof(Int),
                                    cudaMemcpyHostToDevice, stream);
                    cudaGraphLaunch(graphExec, stream);
                }
            } else {
                // Last partial batch: fall back to non-graph launches.
                cudaMemcpyAsync(d_inner_count, &inner_count, sizeof(Int),
                                cudaMemcpyHostToDevice, stream);
                launchBatch(batch);
            }

            inner_count += batch;
            total_count += batch;
        }

        // --- Convergence check ---
        // Compute A * pdhg_x, A^T * pdhg_y, delta_y, A^T * delta_y.
        spmv_a(desc_pdhg_x, desc_ax);
        spmv_at(desc_pdhg_y, desc_at_y);
        launchSubtract(m, d_pdhg_y, d_initial_y, d_delta_y, stream);
        spmv_at(desc_delta_y, desc_at_delta_y);

        // Fused reduction.
        launchZeroMetrics(d_metrics, stream);
        launchConvergenceMetricsCol(
            n, d_pdhg_x, d_initial_x, d_cscaled, d_at_y,
            d_col_lower, d_col_upper, d_at_delta_y,
            primal_weight, step, d_metrics, stream);
        launchConvergenceMetricsRow(
            m, d_pdhg_y, d_initial_y, d_ax,
            d_row_lower, d_row_upper, primal_weight, d_metrics, stream);

        // Transfer metrics to host.
        cudaMemcpyAsync(h_metrics, d_metrics, sizeof(GpuConvergenceMetrics),
                         cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        // Host-side convergence computation.
        Real abs_primal_resid = std::sqrt(h_metrics->primal_resid_sq);
        Real rel_primal_resid = abs_primal_resid / (1.0 + bound_norm);
        Real abs_dual_resid = std::sqrt(h_metrics->dual_resid_sq);
        Real rel_dual_resid = abs_dual_resid / (1.0 + c_norm);

        Real pobj_scaled = h_metrics->primal_obj;
        Real pobj = original_.obj_offset + obj_sign_ * pobj_scaled;
        Real dobj = original_.obj_offset + obj_sign_ * (h_metrics->dual_obj_col + h_metrics->dual_obj_row);
        Real rel_gap = std::abs(pobj - dobj) / (1.0 + std::abs(pobj) + std::abs(dobj));

        Real movement = h_metrics->fpe_movement;
        Real interaction = h_metrics->fpe_interaction;
        Real fpe = std::sqrt(std::max(movement + interaction, 0.0));

        if (initial_fpe < 0.0) initial_fpe = fpe;

        if (options_.verbose && (total_count <= N || total_count % (10 * N) == 0)) {
            std::printf(
                "PDLP %7d  pobj=% .8e  pinf=% .2e  dinf=% .2e  gap=% .2e  "
                "fpe=% .2e  pw=% .2e  step=% .2e  [GPU]\n",
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
            // PI controller for primal weight.
            if (options_.update_primal_weight) {
                Real primal_dist = std::sqrt(h_metrics->primal_dist_sq);
                Real dual_dist = std::sqrt(h_metrics->dual_dist_sq);

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

            // Upload updated primal_weight to device — the graph reads it
            // via pointer indirection, so no graph re-capture needed.
            cudaMemcpyAsync(d_primal_weight, &primal_weight, sizeof(Real),
                            cudaMemcpyHostToDevice, stream);

            launchRestartCopy(n, m, d_pdhg_x, d_pdhg_y,
                              d_initial_x, d_initial_y,
                              d_current_x, d_current_y, stream);
            inner_count = 0;
            // d_inner_count will be updated at the start of the next batch.
            initial_fpe = -1.0;
            last_trial_fpe = std::numeric_limits<Real>::infinity();
        }
    }

    iterations_ = total_count;

    if (converged) {
        // Download solution.
        std::vector<Real> pdhg_x_host(sn);
        std::vector<Real> pdhg_y_host(sm);
        cudaMemcpy(pdhg_x_host.data(), d_pdhg_x, sn * sizeof(Real), cudaMemcpyDeviceToHost);
        cudaMemcpy(pdhg_y_host.data(), d_pdhg_y, sm * sizeof(Real), cudaMemcpyDeviceToHost);

        // Unscale to original space.
        primal_orig_.resize(sn);
        for (Index j = 0; j < n; ++j) {
            Real xj = pdhg_x_host[j] * col_scale_[j];
            if (isFinite(original_.col_lower[j]))
                xj = std::max(xj, original_.col_lower[j]);
            if (isFinite(original_.col_upper[j]))
                xj = std::min(xj, original_.col_upper[j]);
            primal_orig_[j] = xj;
        }
        dual_orig_.resize(sm);
        for (Index i = 0; i < m; ++i) {
            dual_orig_[i] = obj_sign_ * pdhg_y_host[i] * row_scale_[i];
        }

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

cleanup:
    // Destroy CUDA graph.
    if (graphExec) cudaGraphExecDestroy(graphExec);
    if (graph) cudaGraphDestroy(graph);

    // Destroy cuSPARSE descriptors.
    if (desc_current_y) cusparseDestroyDnVec(desc_current_y);
    if (desc_at_y) cusparseDestroyDnVec(desc_at_y);
    if (desc_reflected_x) cusparseDestroyDnVec(desc_reflected_x);
    if (desc_a_xrefl) cusparseDestroyDnVec(desc_a_xrefl);
    if (desc_pdhg_x) cusparseDestroyDnVec(desc_pdhg_x);
    if (desc_ax) cusparseDestroyDnVec(desc_ax);
    if (desc_pdhg_y) cusparseDestroyDnVec(desc_pdhg_y);
    if (desc_delta_y) cusparseDestroyDnVec(desc_delta_y);
    if (desc_at_delta_y) cusparseDestroyDnVec(desc_at_delta_y);
    if (desc_pi_x) cusparseDestroyDnVec(desc_pi_x);
    if (desc_pi_y) cusparseDestroyDnVec(desc_pi_y);
    if (desc_pi_x_new) cusparseDestroyDnVec(desc_pi_x_new);
    if (a_desc) cusparseDestroySpMat(a_desc);
    if (at_desc) cusparseDestroySpMat(at_desc);

    // Free device memory.
    if (d_buf_a) cudaFree(d_buf_a);
    if (d_buf_at) cudaFree(d_buf_at);
    if (d_metrics) cudaFree(d_metrics);
    if (h_metrics) cudaFreeHost(h_metrics);
    if (d_a_values) cudaFree(d_a_values);
    if (d_a_col_indices) cudaFree(d_a_col_indices);
    if (d_a_row_starts) cudaFree(d_a_row_starts);
    if (d_at_values) cudaFree(d_at_values);
    if (d_at_col_indices) cudaFree(d_at_col_indices);
    if (d_at_row_starts) cudaFree(d_at_row_starts);
    if (d_arena) cudaFree(d_arena);

    if (stream) cudaStreamDestroy(stream);
    if (handle) cusparseDestroy(handle);

    double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    double work = static_cast<double>(total_count) *
                  (4.0 * static_cast<double>(nnz) +
                   2.0 * static_cast<double>(m + n));
    work += seconds * 1e-6;
    return {status_, objective_, iterations_, work};
}

#endif  // MIPX_HAS_CUDA

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
