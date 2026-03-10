#include "mipx/barrier.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <utility>

#ifdef MIPX_HAS_CUDA
// Defined in barrier_gpu.cu — thin C++ linkage bridge.
namespace mipx::detail {
bool gpuBarrierSolve(const SparseMatrix& aeq, std::span<const Real> beq,
                     std::span<const Real> cstd,
                     const BarrierOptions& opts, Real std_obj_offset,
                     std::vector<Real>& z, std::vector<Real>& y,
                     std::vector<Real>& s, Int& iters,
                     bool& gpu_initialized, std::string& error_msg);
}  // namespace mipx::detail
#endif

namespace mipx {

// ---------------------------------------------------------------------------
// Helpers (anonymous namespace)
// ---------------------------------------------------------------------------

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

}  // namespace

// ---------------------------------------------------------------------------
// Ruiz equilibration
// ---------------------------------------------------------------------------

void BarrierSolver::computeRuizScaling() {
    const Index m = aeq_.numRows();
    const Index n = aeq_.numCols();

    row_scale_.assign(static_cast<size_t>(m), 1.0);
    col_scale_.assign(static_cast<size_t>(n), 1.0);

    for (Int iter = 0; iter < options_.ruiz_iterations; ++iter) {
        // Row scaling: r_i = 1/sqrt(max_j |A(i,j)|).
        std::vector<Real> row_factor(static_cast<size_t>(m), 1.0);
        for (Index i = 0; i < m; ++i) {
            auto rv = aeq_.row(i);
            Real mx = 0.0;
            for (Index k = 0; k < rv.size(); ++k) {
                mx = std::max(mx, std::abs(rv.values[k]));
            }
            if (mx > 1e-20) {
                row_factor[i] = 1.0 / std::sqrt(mx);
            }
        }

        // Apply row scaling to matrix and b.
        {
            auto vals = aeq_.csr_values();
            auto cols_span = aeq_.csr_col_indices();
            auto rows_span = aeq_.csr_row_starts();
            std::vector<Real> new_vals(vals.begin(), vals.end());
            std::vector<Index> new_cols(cols_span.begin(), cols_span.end());
            std::vector<Index> new_rows(rows_span.begin(), rows_span.end());
            for (Index i = 0; i < m; ++i) {
                for (Index p = new_rows[i]; p < new_rows[i + 1]; ++p) {
                    new_vals[p] *= row_factor[i];
                }
                beq_[i] *= row_factor[i];
                row_scale_[i] *= row_factor[i];
            }
            aeq_ = SparseMatrix(m, n, std::move(new_vals), std::move(new_cols),
                                std::move(new_rows));
        }

        // Col scaling: c_j = 1/sqrt(max_i |A(i,j)|).
        std::vector<Real> col_factor(static_cast<size_t>(n), 1.0);
        {
            std::vector<Real> col_max(static_cast<size_t>(n), 0.0);
            for (Index i = 0; i < m; ++i) {
                auto rv = aeq_.row(i);
                for (Index k = 0; k < rv.size(); ++k) {
                    col_max[rv.indices[k]] =
                        std::max(col_max[rv.indices[k]], std::abs(rv.values[k]));
                }
            }
            for (Index j = 0; j < n; ++j) {
                if (col_max[j] > 1e-20) {
                    col_factor[j] = 1.0 / std::sqrt(col_max[j]);
                }
            }
        }

        // Apply col scaling to matrix and c.
        {
            auto vals = aeq_.csr_values();
            auto cols_span = aeq_.csr_col_indices();
            auto rows_span = aeq_.csr_row_starts();
            std::vector<Real> new_vals(vals.begin(), vals.end());
            std::vector<Index> new_cols(cols_span.begin(), cols_span.end());
            std::vector<Index> new_rows(rows_span.begin(), rows_span.end());
            for (Index i = 0; i < m; ++i) {
                for (Index p = new_rows[i]; p < new_rows[i + 1]; ++p) {
                    new_vals[p] *= col_factor[new_cols[p]];
                }
            }
            for (Index j = 0; j < n; ++j) {
                cstd_[j] *= col_factor[j];
                col_scale_[j] *= col_factor[j];
            }
            aeq_ = SparseMatrix(m, n, std::move(new_vals), std::move(new_cols),
                                std::move(new_rows));
        }
    }

    scaling_applied_ = (options_.ruiz_iterations > 0);
}

void BarrierSolver::applyScaling() {
    // No-op: scaling is applied incrementally in computeRuizScaling.
}

void BarrierSolver::unscaleResult(std::vector<Real>& z, std::vector<Real>& y,
                                  std::vector<Real>& s) {
    if (!scaling_applied_) return;
    const Index m = aeq_.numRows();
    const Index n = aeq_.numCols();

    for (Index j = 0; j < n && j < static_cast<Index>(z.size()); ++j) {
        z[j] *= col_scale_[j];
    }
    for (Index i = 0; i < m && i < static_cast<Index>(y.size()); ++i) {
        y[i] *= row_scale_[i];
    }
    for (Index j = 0; j < n && j < static_cast<Index>(s.size()); ++j) {
        s[j] /= col_scale_[j];
    }
}

// ---------------------------------------------------------------------------
// Dense column detection
// ---------------------------------------------------------------------------

void BarrierSolver::detectDenseColumns() {
    const Index m = aeq_.numRows();
    const Index n = aeq_.numCols();
    dense_cols_.clear();

    if (m == 0 || n == 0) return;

    const Index threshold = static_cast<Index>(options_.dense_col_fraction * m);

    // Count nnz per column.
    std::vector<Index> col_nnz(static_cast<size_t>(n), 0);
    for (Index i = 0; i < m; ++i) {
        auto rv = aeq_.row(i);
        for (Index k = 0; k < rv.size(); ++k) {
            ++col_nnz[rv.indices[k]];
        }
    }

    for (Index j = 0; j < n; ++j) {
        if (col_nnz[j] > threshold) {
            dense_cols_.push_back(j);
        }
    }
}

// ---------------------------------------------------------------------------
// Standard form transformation (preserved from original)
// ---------------------------------------------------------------------------

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

    auto addEqRow = [&](const std::vector<std::pair<Index, Real>>& entries, Real rhs) {
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

    aeq_ = SparseMatrix(static_cast<Index>(beq_.size()), static_cast<Index>(cstd_.size()),
                        std::move(trips));

    dual_eq_.assign(beq_.size(), 0.0);
    reduced_costs_std_.assign(cstd_.size(), 0.0);
    return true;
}

// ---------------------------------------------------------------------------
// Load
// ---------------------------------------------------------------------------

void BarrierSolver::load(const LpProblem& problem) {
    original_ = problem;
    loaded_ = true;
    status_ = Status::Error;
    objective_ = 0.0;
    iterations_ = 0;
    primal_orig_.assign(static_cast<size_t>(original_.num_cols), 0.0);
    dual_eq_.clear();
    reduced_costs_std_.clear();
    used_gpu_ = false;
    scaling_applied_ = false;
    row_scale_.clear();
    col_scale_.clear();
    dense_cols_.clear();
    last_error_.clear();
    transformed_ok_ = buildStandardForm();
}

// ---------------------------------------------------------------------------
// Solution accessors (preserved from original)
// ---------------------------------------------------------------------------

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
        if (e.col_a >= 0 && e.col_a < static_cast<Index>(reduced_costs_std_.size())) {
            v += e.coeff_a * reduced_costs_std_[e.col_a];
        }
        if (e.col_b >= 0 && e.col_b < static_cast<Index>(reduced_costs_std_.size())) {
            v += e.coeff_b * reduced_costs_std_[e.col_b];
        }
        rc[j] = sense_sign * v;
    }
    return rc;
}

std::vector<BasisStatus> BarrierSolver::getBasis() const {
    return {};
}

void BarrierSolver::setBasis(std::span<const BasisStatus> basis) {
    (void)basis;
}

// ---------------------------------------------------------------------------
// Incremental modifications (preserved from original)
// ---------------------------------------------------------------------------

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
        if (r < static_cast<Index>(original_.row_names.size())) {
            original_.row_names.erase(original_.row_names.begin() + r);
        }
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
        if (e.col_a >= 0 && e.col_a < static_cast<Index>(z.size())) x += e.coeff_a * z[e.col_a];
        if (e.col_b >= 0 && e.col_b < static_cast<Index>(z.size())) x += e.coeff_b * z[e.col_b];
        if (isFinite(original_.col_lower[j])) x = std::max(x, original_.col_lower[j]);
        if (isFinite(original_.col_upper[j])) x = std::min(x, original_.col_upper[j]);
        primal_orig_[j] = x;
    }
}

bool BarrierSolver::checkOriginalPrimalFeasibility(std::span<const Real> x) const {
    const Real tol = 1e-5;
    for (Index j = 0; j < original_.num_cols; ++j) {
        if (isFinite(original_.col_lower[j]) && x[j] < original_.col_lower[j] - tol) return false;
        if (isFinite(original_.col_upper[j]) && x[j] > original_.col_upper[j] + tol) return false;
    }

    std::vector<Real> ax(static_cast<size_t>(original_.num_rows), 0.0);
    original_.matrix.multiply(x, ax);
    for (Index i = 0; i < original_.num_rows; ++i) {
        if (isFinite(original_.row_lower[i]) && ax[i] < original_.row_lower[i] - tol) return false;
        if (isFinite(original_.row_upper[i]) && ax[i] > original_.row_upper[i] + tol) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// solveStandardForm: dispatch to GPU or CPU
// ---------------------------------------------------------------------------

bool BarrierSolver::solveStandardForm(std::vector<Real>& z,
                                      std::vector<Real>& y,
                                      std::vector<Real>& s,
                                      Int& iters) {
    last_error_.clear();
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
            if (s[j] < -1e-12) {
                return false;
            }
        }
        iters = 0;
        return true;
    }

    // Apply Ruiz scaling.
    computeRuizScaling();

    // Detect dense columns.
    detectDenseColumns();

    if (!options_.use_gpu) {
        last_error_ =
            "Barrier currently requires GPU backend; run with --gpu "
            "or enable CUDA barrier backends.";
        return false;
    }

    if (aeq_.numRows() < options_.gpu_min_rows ||
        aeq_.numNonzeros() < options_.gpu_min_nnz) {
        char msg[256];
        std::snprintf(
            msg, sizeof(msg),
            "Barrier GPU thresholds not met: rows=%d (min=%d), nnz=%d (min=%d).",
            static_cast<int>(aeq_.numRows()),
            static_cast<int>(options_.gpu_min_rows),
            static_cast<int>(aeq_.numNonzeros()),
            static_cast<int>(options_.gpu_min_nnz));
        last_error_ = msg;
        return false;
    }

#ifdef MIPX_HAS_CUDA
    bool ok = solveStandardFormGpu(z, y, s, iters);
    if (ok) return true;
    if (last_error_.empty()) {
        last_error_ = used_gpu_
                          ? "Barrier GPU solve failed."
                          : "Barrier GPU backend initialization failed.";
    }
    return false;
#else
    last_error_ =
        "Barrier GPU backend unavailable in this build "
        "(rebuild with -DMIPX_USE_CUDA=ON and a detected CUDA toolkit).";
    return false;
#endif
}

// ---------------------------------------------------------------------------
// GPU IPM solve (delegates to GpuBarrierImpl)
// ---------------------------------------------------------------------------

bool BarrierSolver::solveStandardFormGpu(
    [[maybe_unused]] std::vector<Real>& z,
    [[maybe_unused]] std::vector<Real>& y,
    [[maybe_unused]] std::vector<Real>& s,
    [[maybe_unused]] Int& iters) {
#ifdef MIPX_HAS_CUDA
    bool gpu_initialized = false;
    std::string gpu_error;
    bool ok = detail::gpuBarrierSolve(aeq_, beq_, cstd_, options_,
                                       std_obj_offset_, z, y, s, iters,
                                       gpu_initialized, gpu_error);
    if (gpu_initialized) used_gpu_ = true;
    if (!ok && last_error_.empty()) {
        if (!gpu_error.empty()) {
            last_error_ = gpu_error;
        } else if (!gpu_initialized) {
            last_error_ = "Barrier GPU backend initialization failed.";
        } else {
            last_error_ = "Barrier GPU solve failed.";
        }
    }
    return ok;
#else
    if (last_error_.empty()) {
        last_error_ =
            "Barrier GPU backend unavailable in this build "
            "(rebuild with -DMIPX_USE_CUDA=ON and a detected CUDA toolkit).";
    }
    return false;
#endif
}

// ---------------------------------------------------------------------------
// Solve (top-level, preserved structure from original)
// ---------------------------------------------------------------------------

LpResult BarrierSolver::solve() {
    if (!loaded_) {
        status_ = Status::Error;
        return {status_, 0.0, 0, 0.0};
    }

    auto t0 = std::chrono::steady_clock::now();

    if (!transformed_ok_) {
        status_ = transformed_infeasible_ ? Status::Infeasible : Status::Error;
        return {status_, 0.0, 0, 0.0};
    }

    std::vector<Real> z;
    std::vector<Real> y;
    std::vector<Real> s;
    Int iters = 0;

    bool ok = solveStandardForm(z, y, s, iters);

    if (!ok) {
        if (options_.stop_flag != nullptr &&
            options_.stop_flag->load(std::memory_order_relaxed)) {
            status_ = Status::IterLimit;
            objective_ = 0.0;
            iterations_ = iters;
            return {status_, objective_, iterations_, 0.0};
        }
        status_ = Status::Error;
        objective_ = 0.0;
        iterations_ = iters;
        std::string msg =
            last_error_.empty() ? "Barrier solve failed." : last_error_;
        throw std::runtime_error(msg + " (dual-simplex fallback disabled).");
    }

    // Compute objective in scaled space (before unscaling).
    Real min_obj = std_obj_offset_ + dot(cstd_, z);

    // Unscale solution back to original space.
    unscaleResult(z, y, s);

    reconstructOriginalPrimals(z);
    if (!checkOriginalPrimalFeasibility(primal_orig_)) {
        status_ = Status::Error;
        objective_ = 0.0;
        iterations_ = iters;
        throw std::runtime_error(
            "Barrier produced a primal-infeasible solution "
            "(dual-simplex fallback disabled).");
    }

    dual_eq_ = y;
    reduced_costs_std_ = s;
    objective_ = (original_.sense == Sense::Minimize) ? min_obj : -min_obj;
    status_ = Status::Optimal;
    iterations_ = iters;

    double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();

    double work = static_cast<double>(iters) *
                  (4.0 * static_cast<double>(aeq_.numNonzeros()) +
                   10.0 * static_cast<double>(aeq_.numRows()));
    work += seconds * 1e-6;

    return {status_, objective_, iterations_, work};
}

}  // namespace mipx
