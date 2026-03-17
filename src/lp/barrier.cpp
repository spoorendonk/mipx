#include "mipx/barrier.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <utility>

#include "newton_solver.h"

namespace mipx {

namespace {

inline bool isFinite(Real v) {
    return std::isfinite(v);
}

inline Real infNorm(std::span<const Real> v) {
    Real n = 0.0;
    for (Real x : v) n = std::max(n, std::abs(x));
    return n;
}

inline Real dot(std::span<const Real> a, std::span<const Real> b) {
    Real s = 0.0;
    for (Index i = 0; i < static_cast<Index>(a.size()); ++i) s += a[i] * b[i];
    return s;
}

inline Real computeOriginalObjective(const LpProblem& problem,
                                     std::span<const Real> x) {
    Real obj = problem.obj_offset;
    for (Index j = 0; j < problem.num_cols; ++j) obj += problem.obj[j] * x[j];
    return obj;
}

inline Real maxStepToBoundary(std::span<const Real> x, std::span<const Real> dx,
                              Real fraction) {
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

void computeRuizScaling(SparseMatrix& A,
                        std::vector<Real>& b,
                        std::vector<Real>& c,
                        std::vector<Real>& row_scale,
                        std::vector<Real>& col_scale,
                        Int iterations) {
    const Index m = A.numRows();
    const Index n = A.numCols();
    row_scale.assign(static_cast<size_t>(m), 1.0);
    col_scale.assign(static_cast<size_t>(n), 1.0);

    if (iterations <= 0 || m == 0 || n == 0) return;

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
        for (Index i = 0; i < m; ++i) row_scale[i] *= r[i];
        for (Index j = 0; j < n; ++j) col_scale[j] *= cs[j];
    }

    // Rebuild matrix.
    A = SparseMatrix(m, n, std::move(values), std::move(col_indices),
                     std::move(row_starts));

    // Scale b and c.
    for (Index i = 0; i < m; ++i) b[i] *= row_scale[i];
    for (Index j = 0; j < n; ++j) c[j] *= col_scale[j];
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
        if (cv.size() > thresh) ++info.dense_cols;
    }

    Index col_thresh = std::max<Index>(1, static_cast<Index>(dense_fraction * n));
    for (Index i = 0; i < m; ++i) {
        auto rv = A.row(i);
        if (rv.size() > col_thresh) ++info.dense_rows;
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

bool runMehrotraIpm(NewtonSolver& solver,
                    const SparseMatrix& A,
                    std::span<const Real> b,
                    std::span<const Real> c,
                    const BarrierOptions& opts,
                    Real obj_offset,
                    std::vector<Real>& z,
                    std::vector<Real>& y,
                    std::vector<Real>& s,
                    Int& iters) {
    const Index m = A.numRows();
    const Index n = A.numCols();
    const Real reg = std::max(opts.regularization, 1e-12);

    // Mehrotra starting point.
    z.assign(static_cast<size_t>(n), 1.0);
    y.assign(static_cast<size_t>(m), 0.0);
    s.assign(static_cast<size_t>(n), 1.0);

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

    // Stall detection: track previous gap.
    Real prev_gap = 1e30;
    Int stall_count = 0;
    constexpr Int max_stall = 10;

    const Real inv_b = 1.0 / (1.0 + infNorm(b));
    const Real inv_c = 1.0 / (1.0 + infNorm(c));

    for (Int iter = 0; iter < opts.max_iter; ++iter) {
        if (opts.stop_flag != nullptr &&
            opts.stop_flag->load(std::memory_order_relaxed)) {
            iters = iter;
            return false;
        }

        // Residuals.
        std::fill(az.begin(), az.end(), 0.0);
        std::fill(at_y.begin(), at_y.end(), 0.0);
        A.multiply(z, az);
        A.multiplyTranspose(y, at_y);

        for (Index i = 0; i < m; ++i) rp[i] = b[i] - az[i];
        for (Index j = 0; j < n; ++j) rd[j] = c[j] - at_y[j] - s[j];

        Real mu = dot(z, s) / std::max<Index>(n, 1);
        Real pinf = infNorm(rp) * inv_b;
        Real dinf = infNorm(rd) * inv_c;
        Real gap = std::abs(mu) / (1.0 + std::abs(obj_offset + dot(c, z)));

        if (opts.verbose) {
            std::printf("IPM %4d  pobj=% .10e  pinf=% .2e  dinf=% .2e  gap=% .2e\n",
                        iter, obj_offset + dot(c, z), pinf, dinf, gap);
        }

        if (pinf < opts.primal_dual_tol &&
            dinf < opts.primal_dual_tol &&
            gap < opts.primal_dual_tol) {
            iters = iter;
            return true;
        }

        // Stall detection.
        if (gap >= prev_gap * 0.999) {
            ++stall_count;
            if (stall_count >= max_stall) {
                iters = iter;
                return false;
            }
        } else {
            stall_count = 0;
        }
        prev_gap = gap;

        // Factorize.
        if (!solver.factorize(z, s, reg)) {
            // Retry with boosted regularization.
            if (!solver.factorize(z, s, reg * 100.0)) {
                iters = iter;
                return false;
            }
        }

        // Affine direction (predictor).
        for (Index j = 0; j < n; ++j) rc[j] = -z[j] * s[j];
        if (!solver.solveNewton(rp, rd, rc, dz_aff, dy_aff, ds_aff)) {
            iters = iter;
            return false;
        }

        Real alpha_aff_p = maxStepToBoundary(z, dz_aff, 1.0);
        Real alpha_aff_d = maxStepToBoundary(s, ds_aff, 1.0);

        Real mu_aff = 0.0;
        for (Index j = 0; j < n; ++j) {
            mu_aff += (z[j] + alpha_aff_p * dz_aff[j]) *
                      (s[j] + alpha_aff_d * ds_aff[j]);
        }
        mu_aff /= std::max<Index>(n, 1);

        Real sigma = std::pow(std::max(mu_aff, 0.0) / std::max(mu, 1e-30), 3.0);
        sigma = std::clamp(sigma, 0.0, 1.0);

        // Corrector direction.
        for (Index j = 0; j < n; ++j) {
            rc[j] = sigma * mu - z[j] * s[j] - dz_aff[j] * ds_aff[j];
        }
        if (!solver.solveNewton(rp, rd, rc, dz, dy, ds)) {
            iters = iter;
            return false;
        }

        // Step to boundary.
        Real alpha_p = maxStepToBoundary(z, dz, opts.step_fraction);
        Real alpha_d = maxStepToBoundary(s, ds, opts.step_fraction);

        for (Index j = 0; j < n; ++j) {
            z[j] += alpha_p * dz[j];
            s[j] += alpha_d * ds[j];
            z[j] = std::max(z[j], 1e-12);
            s[j] = std::max(s[j], 1e-12);
        }
        for (Index i = 0; i < m; ++i) y[i] += alpha_d * dy[i];
    }

    iters = opts.max_iter;
    return false;
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
    last_error_.clear();
    primal_orig_.assign(static_cast<size_t>(original_.num_cols), 0.0);
    dual_eq_.clear();
    reduced_costs_std_.clear();
    used_gpu_ = false;
    transformed_ok_ = buildStandardForm();
}

bool BarrierSolver::buildStandardForm() {
    transformed_infeasible_ = false;
    beq_.clear();
    cstd_.clear();
    col_expr_.assign(static_cast<size_t>(original_.num_cols), OriginalColExpr{});
    std_obj_offset_ = original_.obj_offset;

    std::vector<Triplet> trips;

    auto addStdCol = [&](Real cost) {
        Index idx = static_cast<Index>(cstd_.size());
        cstd_.push_back(cost);
        return idx;
    };

    auto addEqRow = [&](const std::vector<std::pair<Index, Real>>& entries,
                        Real rhs) {
        Index row = static_cast<Index>(beq_.size());
        beq_.push_back(rhs);
        for (const auto& [col, val] : entries) {
            if (std::abs(val) > 0.0) {
                trips.push_back({row, col, val});
            }
        }
    };

    const Real obj_sign = (original_.sense == Sense::Minimize) ? 1.0 : -1.0;

    for (Index j = 0; j < original_.num_cols; ++j) {
        Real lb = original_.col_lower[j];
        Real ub = original_.col_upper[j];
        Real obj = obj_sign * original_.obj[j];

        bool lb_finite = isFinite(lb);
        bool ub_finite = isFinite(ub);

        OriginalColExpr expr;

        if (lb_finite && ub_finite) {
            if (ub < lb - 1e-12) {
                transformed_infeasible_ = true;
                return false;
            }
            Index w = addStdCol(obj);
            Index t = addStdCol(0.0);
            expr.offset = lb;
            expr.col_a = w;
            expr.coeff_a = 1.0;
            std_obj_offset_ += obj * lb;
            addEqRow({{w, 1.0}, {t, 1.0}}, ub - lb);
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

    std::vector<std::pair<Index, Real>> entries;
    for (Index i = 0; i < original_.num_rows; ++i) {
        auto rv = original_.matrix.row(i);

        if (isFinite(original_.row_upper[i])) {
            entries.clear();
            Real rhs = original_.row_upper[i];
            for (Index k = 0; k < rv.size(); ++k) {
                Index j = rv.indices[k];
                Real a = rv.values[k];
                const auto& e = col_expr_[j];
                rhs -= a * e.offset;
                if (e.col_a >= 0) entries.push_back({e.col_a, a * e.coeff_a});
                if (e.col_b >= 0) entries.push_back({e.col_b, a * e.coeff_b});
            }
            Index s = addStdCol(0.0);
            entries.push_back({s, 1.0});
            addEqRow(entries, rhs);
        }

        if (isFinite(original_.row_lower[i])) {
            entries.clear();
            Real rhs = -original_.row_lower[i];
            for (Index k = 0; k < rv.size(); ++k) {
                Index j = rv.indices[k];
                Real a = rv.values[k];
                const auto& e = col_expr_[j];
                rhs += a * e.offset;
                if (e.col_a >= 0) entries.push_back({e.col_a, -a * e.coeff_a});
                if (e.col_b >= 0) entries.push_back({e.col_b, -a * e.coeff_b});
            }
            Index s = addStdCol(0.0);
            entries.push_back({s, 1.0});
            addEqRow(entries, rhs);
        }
    }

    aeq_ = SparseMatrix(static_cast<Index>(beq_.size()),
                        static_cast<Index>(cstd_.size()),
                        std::move(trips));

    dual_eq_.assign(beq_.size(), 0.0);
    reduced_costs_std_.assign(cstd_.size(), 0.0);
    return true;
}

std::vector<Real> BarrierSolver::getPrimalValues() const {
    return primal_orig_;
}

std::vector<Real> BarrierSolver::getDualValues() const {
    return std::vector<Real>(static_cast<size_t>(original_.num_rows), 0.0);
}

std::vector<Real> BarrierSolver::getReducedCosts() const {
    std::vector<Real> rc(static_cast<size_t>(original_.num_cols), 0.0);
    const Real sense_sign = (original_.sense == Sense::Minimize) ? 1.0 : -1.0;
    for (Index j = 0; j < original_.num_cols; ++j) {
        const auto& e = col_expr_[j];
        Real v = 0.0;
        if (e.col_a >= 0 && e.col_a < static_cast<Index>(reduced_costs_std_.size()))
            v += e.coeff_a * reduced_costs_std_[e.col_a];
        if (e.col_b >= 0 && e.col_b < static_cast<Index>(reduced_costs_std_.size()))
            v += e.coeff_b * reduced_costs_std_[e.col_b];
        rc[j] = sense_sign * v;
    }
    return rc;
}

std::vector<BasisStatus> BarrierSolver::getBasis() const { return {}; }

void BarrierSolver::setBasis(std::span<const BasisStatus> basis) {
    (void)basis;
}

void BarrierSolver::addRows(std::span<const Index> starts,
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
            e > static_cast<Index>(values.size()))
            return;
        std::span<const Index> ri(indices.data() + s, static_cast<size_t>(e - s));
        std::span<const Real> rv(values.data() + s, static_cast<size_t>(e - s));
        original_.matrix.addRow(ri, rv);
        original_.row_lower.push_back(lower[r]);
        original_.row_upper.push_back(upper[r]);
        original_.row_names.push_back("");
    }
    original_.num_rows += rows_to_add;
    transformed_ok_ = buildStandardForm();
}

void BarrierSolver::removeRows(std::span<const Index> rows) {
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
        if (r < static_cast<Index>(original_.row_names.size()))
            original_.row_names.erase(original_.row_names.begin() + r);
        --original_.num_rows;
    }
    transformed_ok_ = buildStandardForm();
}

void BarrierSolver::setColBounds(Index col, Real lower, Real upper) {
    if (!loaded_) return;
    if (col < 0 || col >= original_.num_cols) return;
    original_.col_lower[col] = lower;
    original_.col_upper[col] = upper;
    transformed_ok_ = buildStandardForm();
}

void BarrierSolver::setObjective(std::span<const Real> obj) {
    if (!loaded_) return;
    if (static_cast<Index>(obj.size()) != original_.num_cols) return;
    original_.obj.assign(obj.begin(), obj.end());
    transformed_ok_ = buildStandardForm();
}

void BarrierSolver::reconstructOriginalPrimals(const std::vector<Real>& z) {
    primal_orig_.assign(static_cast<size_t>(original_.num_cols), 0.0);
    for (Index j = 0; j < original_.num_cols; ++j) {
        const auto& e = col_expr_[j];
        Real x = e.offset;
        if (e.col_a >= 0 && e.col_a < static_cast<Index>(z.size()))
            x += e.coeff_a * z[e.col_a];
        if (e.col_b >= 0 && e.col_b < static_cast<Index>(z.size()))
            x += e.coeff_b * z[e.col_b];
        if (isFinite(original_.col_lower[j]))
            x = std::max(x, original_.col_lower[j]);
        if (isFinite(original_.col_upper[j]))
            x = std::min(x, original_.col_upper[j]);
        primal_orig_[j] = x;
    }
}

bool BarrierSolver::checkOriginalPrimalFeasibility(std::span<const Real> x) const {
    const Real tol = 1e-5;
    for (Index j = 0; j < original_.num_cols; ++j) {
        if (isFinite(original_.col_lower[j]) && x[j] < original_.col_lower[j] - tol)
            return false;
        if (isFinite(original_.col_upper[j]) && x[j] > original_.col_upper[j] + tol)
            return false;
    }

    std::vector<Real> ax(static_cast<size_t>(original_.num_rows), 0.0);
    original_.matrix.multiply(x, ax);
    for (Index i = 0; i < original_.num_rows; ++i) {
        if (isFinite(original_.row_lower[i]) && ax[i] < original_.row_lower[i] - tol)
            return false;
        if (isFinite(original_.row_upper[i]) && ax[i] > original_.row_upper[i] + tol)
            return false;
    }
    return true;
}

// ============================================================================
// solveStandardForm — dispatch + shared IPM
// ============================================================================

bool BarrierSolver::solveStandardForm(std::vector<Real>& z,
                                      std::vector<Real>& y,
                                      std::vector<Real>& s,
                                      Int& iters) {
    const Index m = aeq_.numRows();
    const Index n = aeq_.numCols();
    if (n == 0) {
        z.clear();
        y.assign(static_cast<size_t>(m), 0.0);
        s.clear();
        iters = 0;
        return true;
    }
    if (m == 0) {
        z.assign(static_cast<size_t>(n), 0.0);
        y.clear();
        s = cstd_;
        for (Index j = 0; j < n; ++j) {
            if (s[j] < -1e-12) return false;
        }
        iters = 0;
        return true;
    }

    // Ruiz equilibration.
    computeRuizScaling(aeq_, beq_, cstd_, row_scale_, col_scale_,
                       options_.ruiz_iterations);

    // Density detection for backend selection.
    auto density = detectDensity(aeq_, options_.dense_col_fraction);
    bool use_augmented = (density.dense_cols > 50 ||
                          density.dense_rows > 10 ||
                          density.avg_ne_row_density > 20.0);

    // Select backend.
    std::unique_ptr<NewtonSolver> solver;
    auto algo = options_.algorithm;

    const auto requested_algo = options_.algorithm;
    if (algo == BarrierAlgorithm::Auto) {
#ifdef MIPX_HAS_CUDSS
        algo = use_augmented ? BarrierAlgorithm::GpuAugmented
                             : BarrierAlgorithm::GpuCholesky;
#else
        algo = use_augmented ? BarrierAlgorithm::CpuAugmented
                             : BarrierAlgorithm::CpuCholesky;
#endif
    }

    bool ok = false;
    std::string gpu_fallback_error;

#ifdef MIPX_HAS_CUDSS
    // GPU device-resident path: handles NE/Augmented internally with auto-switching.
    if (algo == BarrierAlgorithm::GpuCholesky || algo == BarrierAlgorithm::GpuAugmented) {
        bool prefer_aug = (algo == BarrierAlgorithm::GpuAugmented) || use_augmented;
        std::string gpu_error;
        ok = solveBarrierGpu(aeq_, m, n, beq_, cstd_, options_,
                             std_obj_offset_, prefer_aug, z, y, s, iters, &gpu_error);
        if (ok) {
            used_gpu_ = true;
        } else {
            gpu_fallback_error =
                gpu_error.empty() ? "Barrier GPU solve failed." : gpu_error;
            if (options_.verbose && !gpu_error.empty()) {
                if (requested_algo == BarrierAlgorithm::Auto) {
                    std::fprintf(stderr,
                                 "Barrier GPU backend failed, falling back to CPU: %s\n",
                                 gpu_error.c_str());
                } else {
                    std::fprintf(stderr, "Barrier GPU backend failed: %s\n",
                                 gpu_error.c_str());
                }
            }
            if (requested_algo == BarrierAlgorithm::Auto) {
                algo = use_augmented ? BarrierAlgorithm::CpuAugmented
                                     : BarrierAlgorithm::CpuCholesky;
            } else {
                last_error_ = gpu_fallback_error;
                return false;
            }
        }
    }
#else
    if (algo == BarrierAlgorithm::GpuCholesky || algo == BarrierAlgorithm::GpuAugmented) {
        // No CUDA — use CPU.
        algo = use_augmented ? BarrierAlgorithm::CpuAugmented
                             : BarrierAlgorithm::CpuCholesky;
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
        default:
            solver = createCpuCholeskySolver();
            break;
        }
    }

    if (solver) {
        if (!solver->setup(aeq_, m, n, options_)) {
            if (last_error_.empty()) {
                last_error_ = "Barrier backend setup failed.";
            }
            return false;
        }
        ok = runMehrotraIpm(*solver, aeq_, beq_, cstd_, options_,
                             std_obj_offset_, z, y, s, iters);
        if (!ok && last_error_.empty() &&
            !(options_.stop_flag != nullptr &&
              options_.stop_flag->load(std::memory_order_relaxed))) {
            if (!gpu_fallback_error.empty()) {
                last_error_ =
                    gpu_fallback_error + " CPU fallback also failed.";
            } else {
                last_error_ = "Barrier solve failed.";
            }
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

    return ok;
}

// ============================================================================
// solve() — top-level entry point
// ============================================================================

LpResult BarrierSolver::solve() {
    last_error_.clear();
    if (!loaded_) {
        status_ = Status::Error;
        last_error_ = "Barrier solver has no loaded problem.";
        return {status_, 0.0, 0, 0.0};
    }

    auto t0 = std::chrono::steady_clock::now();

    if (!transformed_ok_) {
        status_ = transformed_infeasible_ ? Status::Infeasible : Status::Error;
        if (status_ == Status::Error) {
            last_error_ = "Barrier standard-form transformation failed.";
        }
        return {status_, 0.0, 0, 0.0};
    }

    std::vector<Real> z, y, s;
    Int iters = 0;

    bool ok = solveStandardForm(z, y, s, iters);

    if (!ok) {
        if (options_.stop_flag != nullptr &&
            options_.stop_flag->load(std::memory_order_relaxed)) {
            status_ = Status::IterLimit;
            objective_ = 0.0;
            iterations_ = iters;
            last_error_.clear();
            return {status_, objective_, iterations_, 0.0};
        }
        dual_eq_.clear();
        reduced_costs_std_.clear();
        primal_orig_.clear();
        status_ = Status::Error;
        objective_ = 0.0;
        iterations_ = iters;
        if (last_error_.empty()) {
            last_error_ = "Barrier solve failed.";
        }
        return {status_, objective_, iterations_, 0.0};
    }

    reconstructOriginalPrimals(z);
    if (!checkOriginalPrimalFeasibility(primal_orig_)) {
        dual_eq_.clear();
        reduced_costs_std_.clear();
        primal_orig_.clear();
        status_ = Status::Error;
        objective_ = 0.0;
        iterations_ = iters;
        last_error_ = "Barrier produced a primal-infeasible solution.";
        return {status_, objective_, iterations_, 0.0};
    }

    dual_eq_ = y;
    reduced_costs_std_ = s;

    objective_ = computeOriginalObjective(original_, primal_orig_);
    status_ = Status::Optimal;
    iterations_ = iters;
    last_error_.clear();

    double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();

    double work = static_cast<double>(iters) *
                  (4.0 * static_cast<double>(aeq_.numNonzeros()) +
                   10.0 * static_cast<double>(aeq_.numRows()));
    work += seconds * 1e-6;

    return {status_, objective_, iterations_, work};
}

}  // namespace mipx
