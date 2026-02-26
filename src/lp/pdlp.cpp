#include "mipx/pdlp.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <numeric>
#include <utility>

#include "mipx/dual_simplex.h"

#ifdef MIPX_HAS_CUDA
#include <cuda_runtime_api.h>
#include <cusparse.h>
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

class LinearBackend {
public:
    virtual ~LinearBackend() = default;
    virtual bool multiply(std::span<const Real> x, std::span<Real> y) = 0;
    virtual bool multiplyTranspose(std::span<const Real> x, std::span<Real> y) = 0;
    [[nodiscard]] virtual bool isGpu() const = 0;
};

class CpuBackend final : public LinearBackend {
public:
    explicit CpuBackend(const SparseMatrix& a) : a_(a) {}

    bool multiply(std::span<const Real> x, std::span<Real> y) override {
        a_.multiply(x, y);
        return true;
    }

    bool multiplyTranspose(std::span<const Real> x, std::span<Real> y) override {
        a_.multiplyTranspose(x, y);
        return true;
    }

    [[nodiscard]] bool isGpu() const override { return false; }

private:
    const SparseMatrix& a_;
};

#ifdef MIPX_HAS_CUDA

inline bool cudaOk(cudaError_t code) { return code == cudaSuccess; }
inline bool cusparseOk(cusparseStatus_t code) { return code == CUSPARSE_STATUS_SUCCESS; }

class CudaBackend final : public LinearBackend {
public:
    explicit CudaBackend(const SparseMatrix& a)
        : rows_(a.numRows()), cols_(a.numCols()), nnz_(a.numNonzeros()) {

        auto vals = a.csr_values();
        auto cols = a.csr_col_indices();
        auto rows = a.csr_row_starts();

        if (vals.size() != static_cast<size_t>(nnz_) ||
            cols.size() != static_cast<size_t>(nnz_) ||
            rows.size() != static_cast<size_t>(rows_ + 1)) {
            return;
        }

        if (!cusparseOk(cusparseCreate(&handle_))) return;

        if (!cudaOk(cudaMalloc(reinterpret_cast<void**>(&d_values_), sizeof(Real) * vals.size())))
            return;
        if (!cudaOk(cudaMalloc(reinterpret_cast<void**>(&d_col_indices_), sizeof(Index) * cols.size())))
            return;
        if (!cudaOk(cudaMalloc(reinterpret_cast<void**>(&d_row_starts_), sizeof(Index) * rows.size())))
            return;

        if (!cudaOk(cudaMemcpy(d_values_, vals.data(), sizeof(Real) * vals.size(),
                               cudaMemcpyHostToDevice))) {
            return;
        }
        if (!cudaOk(cudaMemcpy(d_col_indices_, cols.data(), sizeof(Index) * cols.size(),
                               cudaMemcpyHostToDevice))) {
            return;
        }
        if (!cudaOk(cudaMemcpy(d_row_starts_, rows.data(), sizeof(Index) * rows.size(),
                               cudaMemcpyHostToDevice))) {
            return;
        }

        if (!cusparseOk(cusparseCreateCsr(
                &mat_, static_cast<int64_t>(rows_), static_cast<int64_t>(cols_),
                static_cast<int64_t>(nnz_), d_row_starts_, d_col_indices_, d_values_,
                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F))) {
            return;
        }

        ok_ = true;
    }

    ~CudaBackend() override {
        destroyVectorDescriptors();
        if (mat_) cusparseDestroySpMat(mat_);
        if (d_x_) cudaFree(d_x_);
        if (d_y_) cudaFree(d_y_);
        if (d_values_) cudaFree(d_values_);
        if (d_col_indices_) cudaFree(d_col_indices_);
        if (d_row_starts_) cudaFree(d_row_starts_);
        if (buffer_n_) cudaFree(buffer_n_);
        if (buffer_t_) cudaFree(buffer_t_);
        if (handle_) cusparseDestroy(handle_);
    }

    [[nodiscard]] bool initialized() const { return ok_; }

    bool multiply(std::span<const Real> x, std::span<Real> y) override {
        if (!ok_) return false;
        if (static_cast<Index>(x.size()) != cols_ || static_cast<Index>(y.size()) != rows_) {
            return false;
        }
        if (!ensureVectorCapacity(cols_, rows_)) return false;
        if (!ensureVectors(CUSPARSE_OPERATION_NON_TRANSPOSE)) return false;
        if (!ensureBuffer(CUSPARSE_OPERATION_NON_TRANSPOSE)) return false;
        if (!cudaOk(cudaMemcpy(d_x_, x.data(), sizeof(Real) * x.size(), cudaMemcpyHostToDevice)))
            return false;

        const Real alpha = 1.0;
        const Real beta = 0.0;
        if (!cusparseOk(cusparseSpMV(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                     &alpha, mat_, vec_x_n_, &beta, vec_y_n_,
                                     CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                                     buffer_n_))) {
            return false;
        }
        if (!cudaOk(cudaMemcpy(y.data(), d_y_, sizeof(Real) * y.size(), cudaMemcpyDeviceToHost)))
            return false;
        return true;
    }

    bool multiplyTranspose(std::span<const Real> x, std::span<Real> y) override {
        if (!ok_) return false;
        if (static_cast<Index>(x.size()) != rows_ || static_cast<Index>(y.size()) != cols_) {
            return false;
        }
        if (!ensureVectorCapacity(rows_, cols_)) return false;
        if (!ensureVectors(CUSPARSE_OPERATION_TRANSPOSE)) return false;
        if (!ensureBuffer(CUSPARSE_OPERATION_TRANSPOSE)) return false;
        if (!cudaOk(cudaMemcpy(d_x_, x.data(), sizeof(Real) * x.size(), cudaMemcpyHostToDevice)))
            return false;

        const Real alpha = 1.0;
        const Real beta = 0.0;
        if (!cusparseOk(cusparseSpMV(handle_, CUSPARSE_OPERATION_TRANSPOSE,
                                     &alpha, mat_, vec_x_t_, &beta, vec_y_t_,
                                     CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                                     buffer_t_))) {
            return false;
        }
        if (!cudaOk(cudaMemcpy(y.data(), d_y_, sizeof(Real) * y.size(), cudaMemcpyDeviceToHost)))
            return false;
        return true;
    }

    [[nodiscard]] bool isGpu() const override { return true; }

private:
    bool ensureVectorCapacity(Index in_size, Index out_size) {
        bool resized = false;
        size_t need_x = sizeof(Real) * static_cast<size_t>(in_size);
        size_t need_y = sizeof(Real) * static_cast<size_t>(out_size);
        if (need_x > x_cap_bytes_) {
            if (d_x_) cudaFree(d_x_);
            d_x_ = nullptr;
            if (!cudaOk(cudaMalloc(reinterpret_cast<void**>(&d_x_), need_x))) return false;
            x_cap_bytes_ = need_x;
            resized = true;
        }
        if (need_y > y_cap_bytes_) {
            if (d_y_) cudaFree(d_y_);
            d_y_ = nullptr;
            if (!cudaOk(cudaMalloc(reinterpret_cast<void**>(&d_y_), need_y))) return false;
            y_cap_bytes_ = need_y;
            resized = true;
        }
        if (resized) destroyVectorDescriptors();
        return true;
    }

    bool ensureVectors(cusparseOperation_t op) {
        if (op == CUSPARSE_OPERATION_NON_TRANSPOSE) {
            if (!vec_x_n_ &&
                !cusparseOk(cusparseCreateDnVec(&vec_x_n_, cols_, d_x_, CUDA_R_64F))) {
                return false;
            }
            if (!vec_y_n_ &&
                !cusparseOk(cusparseCreateDnVec(&vec_y_n_, rows_, d_y_, CUDA_R_64F))) {
                return false;
            }
            return true;
        }
        if (!vec_x_t_ &&
            !cusparseOk(cusparseCreateDnVec(&vec_x_t_, rows_, d_x_, CUDA_R_64F))) {
            return false;
        }
        if (!vec_y_t_ &&
            !cusparseOk(cusparseCreateDnVec(&vec_y_t_, cols_, d_y_, CUDA_R_64F))) {
            return false;
        }
        return true;
    }

    bool ensureBuffer(cusparseOperation_t op) {
        size_t* target_size = (op == CUSPARSE_OPERATION_NON_TRANSPOSE)
                                  ? &buffer_n_size_
                                  : &buffer_t_size_;
        void** target_buf = (op == CUSPARSE_OPERATION_NON_TRANSPOSE)
                                ? &buffer_n_
                                : &buffer_t_;
        if (*target_size > 0) return true;
        cusparseDnVecDescr_t vec_x =
            (op == CUSPARSE_OPERATION_NON_TRANSPOSE) ? vec_x_n_ : vec_x_t_;
        cusparseDnVecDescr_t vec_y =
            (op == CUSPARSE_OPERATION_NON_TRANSPOSE) ? vec_y_n_ : vec_y_t_;
        if (!vec_x || !vec_y) return false;

        const Real alpha = 1.0;
        const Real beta = 0.0;
        size_t bytes = 0;
        if (!cusparseOk(cusparseSpMV_bufferSize(handle_, op,
                                                &alpha, mat_, vec_x, &beta, vec_y,
                                                CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                                                &bytes))) {
            return false;
        }
        if (bytes == 0) {
            *target_size = 1;
            return true;
        }
        if (!cudaOk(cudaMalloc(target_buf, bytes))) return false;
        *target_size = bytes;
        return true;
    }

    void destroyVectorDescriptors() {
        if (vec_x_n_) {
            cusparseDestroyDnVec(vec_x_n_);
            vec_x_n_ = nullptr;
        }
        if (vec_y_n_) {
            cusparseDestroyDnVec(vec_y_n_);
            vec_y_n_ = nullptr;
        }
        if (vec_x_t_) {
            cusparseDestroyDnVec(vec_x_t_);
            vec_x_t_ = nullptr;
        }
        if (vec_y_t_) {
            cusparseDestroyDnVec(vec_y_t_);
            vec_y_t_ = nullptr;
        }
    }

    Index rows_ = 0;
    Index cols_ = 0;
    Index nnz_ = 0;
    bool ok_ = false;

    cusparseHandle_t handle_ = nullptr;
    cusparseSpMatDescr_t mat_ = nullptr;

    Real* d_values_ = nullptr;
    Index* d_col_indices_ = nullptr;
    Index* d_row_starts_ = nullptr;
    Real* d_x_ = nullptr;
    Real* d_y_ = nullptr;
    cusparseDnVecDescr_t vec_x_n_ = nullptr;
    cusparseDnVecDescr_t vec_y_n_ = nullptr;
    cusparseDnVecDescr_t vec_x_t_ = nullptr;
    cusparseDnVecDescr_t vec_y_t_ = nullptr;

    size_t x_cap_bytes_ = 0;
    size_t y_cap_bytes_ = 0;

    void* buffer_n_ = nullptr;
    void* buffer_t_ = nullptr;
    size_t buffer_n_size_ = 0;
    size_t buffer_t_size_ = 0;
};

#endif

}  // namespace

void PdlpSolver::load(const LpProblem& problem) {
    original_ = problem;
    loaded_ = true;
    status_ = Status::Error;
    objective_ = 0.0;
    iterations_ = 0;
    primal_orig_.assign(static_cast<size_t>(original_.num_cols), 0.0);
    dual_eq_.clear();
    reduced_costs_std_.clear();
    used_gpu_ = false;
    transformed_ok_ = buildStandardForm();
    if (transformed_ok_) buildScaledProblem();
}

bool PdlpSolver::buildStandardForm() {
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

void PdlpSolver::buildScaledProblem() {
    const Index m = aeq_.numRows();
    const Index n = aeq_.numCols();

    row_scale_.assign(static_cast<size_t>(m), 1.0);
    col_scale_.assign(static_cast<size_t>(n), 1.0);
    bscaled_ = beq_;
    cscaled_ = cstd_;

    auto values_span = aeq_.csr_values();
    auto cols_span = aeq_.csr_col_indices();
    auto rows_span = aeq_.csr_row_starts();
    std::vector<Real> values(values_span.begin(), values_span.end());
    std::vector<Index> col_indices(cols_span.begin(), cols_span.end());
    std::vector<Index> row_starts(rows_span.begin(), rows_span.end());

    if (options_.do_ruiz_scaling && m > 0 && n > 0 && !values.empty()) {
        std::vector<Real> col_mult(static_cast<size_t>(n), 1.0);
        std::vector<Real> col_norm(static_cast<size_t>(n), 0.0);
        for (Index it = 0; it < options_.ruiz_iterations; ++it) {
            for (Index i = 0; i < m; ++i) {
                Index rs = row_starts[i];
                Index re = row_starts[i + 1];
                Real row_max = 0.0;
                for (Index k = rs; k < re; ++k) {
                    row_max = std::max(row_max, std::abs(values[k]));
                }
                if (row_max <= 1e-12) continue;
                Real scale = 1.0 / std::sqrt(row_max);
                row_scale_[i] *= scale;
                bscaled_[i] *= scale;
                for (Index k = rs; k < re; ++k) values[k] *= scale;
            }

            std::fill(col_norm.begin(), col_norm.end(), 0.0);
            std::fill(col_mult.begin(), col_mult.end(), 1.0);
            for (Index k = 0; k < static_cast<Index>(values.size()); ++k) {
                Index j = col_indices[k];
                col_norm[j] = std::max(col_norm[j], std::abs(values[k]));
            }
            for (Index j = 0; j < n; ++j) {
                if (col_norm[j] <= 1e-12) continue;
                Real scale = 1.0 / std::sqrt(col_norm[j]);
                col_scale_[j] *= scale;
                cscaled_[j] *= scale;
                col_mult[j] = scale;
            }
            for (Index k = 0; k < static_cast<Index>(values.size()); ++k) {
                values[k] *= col_mult[col_indices[k]];
            }
        }
    }

    scaled_aeq_ = SparseMatrix(m, n, std::move(values), std::move(col_indices),
                               std::move(row_starts));

    sigma_base_.assign(static_cast<size_t>(m), 1.0);
    tau_base_.assign(static_cast<size_t>(n), 1.0);
    if (options_.do_pock_chambolle_scaling && m > 0 && n > 0) {
        std::vector<Real> row_sq(static_cast<size_t>(m), 0.0);
        std::vector<Real> col_sq(static_cast<size_t>(n), 0.0);
        auto vals = scaled_aeq_.csr_values();
        auto cols = scaled_aeq_.csr_col_indices();
        auto rows = scaled_aeq_.csr_row_starts();

        for (Index i = 0; i < m; ++i) {
            for (Index k = rows[i]; k < rows[i + 1]; ++k) {
                Real a = vals[k];
                Index j = cols[k];
                row_sq[i] += a * a;
                col_sq[j] += a * a;
            }
        }

        const Real alpha = std::max(options_.default_alpha_pock_chambolle_rescaling, 1e-6);
        for (Index i = 0; i < m; ++i) {
            sigma_base_[i] = alpha / std::max(std::sqrt(row_sq[i]), 1e-8);
        }
        for (Index j = 0; j < n; ++j) {
            tau_base_[j] = alpha / std::max(std::sqrt(col_sq[j]), 1e-8);
        }
    }
}

std::vector<Real> PdlpSolver::getPrimalValues() const {
    return primal_orig_;
}

std::vector<Real> PdlpSolver::getDualValues() const {
    // Equality-dual mapping back to ranged rows is not yet exposed.
    return std::vector<Real>(static_cast<size_t>(original_.num_rows), 0.0);
}

std::vector<Real> PdlpSolver::getReducedCosts() const {
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

std::vector<BasisStatus> PdlpSolver::getBasis() const {
    return {};
}

void PdlpSolver::setBasis(std::span<const BasisStatus> basis) {
    (void)basis;
}

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
    transformed_ok_ = buildStandardForm();
    if (transformed_ok_) buildScaledProblem();
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

    transformed_ok_ = buildStandardForm();
    if (transformed_ok_) buildScaledProblem();
}

void PdlpSolver::setColBounds(Index col, Real lower, Real upper) {
    if (!loaded_) return;
    if (col < 0 || col >= original_.num_cols) return;
    original_.col_lower[col] = lower;
    original_.col_upper[col] = upper;
    transformed_ok_ = buildStandardForm();
    if (transformed_ok_) buildScaledProblem();
}

void PdlpSolver::setObjective(std::span<const Real> obj) {
    if (!loaded_) return;
    if (static_cast<Index>(obj.size()) != original_.num_cols) return;
    original_.obj.assign(obj.begin(), obj.end());
    transformed_ok_ = buildStandardForm();
    if (transformed_ok_) buildScaledProblem();
}

void PdlpSolver::reconstructOriginalPrimals(std::span<const Real> z_unscaled) {
    primal_orig_.assign(static_cast<size_t>(original_.num_cols), 0.0);
    for (Index j = 0; j < original_.num_cols; ++j) {
        const auto& e = col_expr_[j];
        Real x = e.offset;
        if (e.col_a >= 0 && e.col_a < static_cast<Index>(z_unscaled.size())) {
            x += e.coeff_a * z_unscaled[e.col_a];
        }
        if (e.col_b >= 0 && e.col_b < static_cast<Index>(z_unscaled.size())) {
            x += e.coeff_b * z_unscaled[e.col_b];
        }
        if (isFinite(original_.col_lower[j])) x = std::max(x, original_.col_lower[j]);
        if (isFinite(original_.col_upper[j])) x = std::min(x, original_.col_upper[j]);
        primal_orig_[j] = x;
    }
}

bool PdlpSolver::checkOriginalPrimalFeasibility(std::span<const Real> x) const {
    const Real tol = 2e-5;
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

bool PdlpSolver::solveStandardForm(std::vector<Real>& z_unscaled,
                                   std::vector<Real>& y_unscaled,
                                   Int& iters) {
    const Index m = scaled_aeq_.numRows();
    const Index n = scaled_aeq_.numCols();
    if (n == 0) {
        z_unscaled.clear();
        y_unscaled.assign(static_cast<size_t>(m), 0.0);
        iters = 0;
        return true;
    }

    std::unique_ptr<LinearBackend> backend = std::make_unique<CpuBackend>(scaled_aeq_);
#ifdef MIPX_HAS_CUDA
    std::unique_ptr<CudaBackend> cuda_backend;
    bool gpu_worthwhile = options_.use_gpu &&
                          scaled_aeq_.numRows() >= options_.gpu_min_rows &&
                          scaled_aeq_.numNonzeros() >= options_.gpu_min_nnz;
    if (gpu_worthwhile) {
        cuda_backend = std::make_unique<CudaBackend>(scaled_aeq_);
        if (cuda_backend->initialized()) {
            used_gpu_ = true;
            backend = std::move(cuda_backend);
        }
    }
#endif

    if (m == 0) {
        z_unscaled.assign(static_cast<size_t>(n), 0.0);
        for (Index j = 0; j < n; ++j) {
            // min c^T z, z >= 0.
            if (cstd_[j] < -1e-12) return false;
        }
        y_unscaled.clear();
        iters = 0;
        return true;
    }

    std::vector<Real> z(static_cast<size_t>(n), 0.0);
    std::vector<Real> z_prev(static_cast<size_t>(n), 0.0);
    std::vector<Real> z_bar(static_cast<size_t>(n), 0.0);
    std::vector<Real> y(static_cast<size_t>(m), 0.0);

    std::vector<Real> az_bar(static_cast<size_t>(m), 0.0);
    std::vector<Real> az(static_cast<size_t>(m), 0.0);
    std::vector<Real> at_y(static_cast<size_t>(n), 0.0);
    std::vector<Real> rc(static_cast<size_t>(n), 0.0);
    std::vector<Real> dual_resid(static_cast<size_t>(n), 0.0);

    const Real inv_b = 1.0 / (1.0 + infNorm(bscaled_));
    const Real inv_c = 1.0 / (1.0 + infNorm(cscaled_));
    const Real eps = std::max(options_.optimality_tol * 0.1, 1e-10);

    Real step = std::clamp(options_.initial_step_size_scaling,
                           options_.min_step_size, options_.max_step_size);

    Real primal_weight = std::max(options_.primal_weight, 1e-6);
    if (options_.update_primal_weight) {
        Real nb = l2Norm(bscaled_);
        Real nc = l2Norm(cscaled_);
        if (nb > 1e-12 && nc > 1e-12) primal_weight = std::clamp(nc / nb, 1e-3, 1e3);
    }

    Real prev_merit = std::numeric_limits<Real>::infinity();
    Real restart_merit = std::numeric_limits<Real>::infinity();

    for (Index iter = 0; iter < options_.max_iter; ++iter) {
        if (options_.stop_flag != nullptr &&
            options_.stop_flag->load(std::memory_order_relaxed)) {
            iters = iter;
            return false;
        }
        if (!backend->multiply(z_bar, az_bar)) return false;
        for (Index i = 0; i < m; ++i) {
            y[i] += step * primal_weight * sigma_base_[i] * (az_bar[i] - bscaled_[i]);
        }

        if (!backend->multiplyTranspose(y, at_y)) return false;
        z_prev = z;
        for (Index j = 0; j < n; ++j) {
            Real tau = step * tau_base_[j] / primal_weight;
            Real grad = cscaled_[j] + at_y[j];
            Real zj = z[j] - tau * grad;
            z[j] = (zj > 0.0) ? zj : 0.0;
        }

        for (Index j = 0; j < n; ++j) {
            z_bar[j] = z[j] + options_.extrapolation_factor * (z[j] - z_prev[j]);
            if (z_bar[j] < 0.0) z_bar[j] = 0.0;
        }

        if (!backend->multiply(z, az)) return false;

        for (Index j = 0; j < n; ++j) {
            rc[j] = cscaled_[j] + at_y[j];
            Real projected = z[j] - std::max<Real>(0.0, z[j] - rc[j]);
            dual_resid[j] = projected;
        }

        for (Index i = 0; i < m; ++i) az[i] -= bscaled_[i];

        Real primal_inf = infNorm(az) * inv_b;
        Real dual_inf = infNorm(dual_resid) * inv_c;

        Real pobj = std_obj_offset_ + dot(cscaled_, z);
        Real dobj = -std::numeric_limits<Real>::infinity();
        bool dual_feasible = true;
        for (Index j = 0; j < n; ++j) {
            if (rc[j] < -10.0 * options_.dual_tol) {
                dual_feasible = false;
                break;
            }
        }
        if (dual_feasible) dobj = std_obj_offset_ + dot(bscaled_, y);

        Real gap;
        if (dual_feasible) {
            gap = std::abs(pobj - dobj) /
                  (1.0 + std::max(std::abs(pobj), std::abs(dobj)));
        } else {
            Real complementarity = std::abs(dot(z, rc));
            gap = complementarity / (1.0 + std::abs(pobj));
        }

        Real merit = std::max(primal_inf, std::max(dual_inf, gap));
        if (options_.verbose && (iter < 20 || (iter + 1) % 50 == 0)) {
            std::printf(
                "PDLP %6d  pobj=% .10e  pinf=% .2e  dinf=% .2e  gap=% .2e  step=% .2e%s\n",
                iter + 1, pobj, primal_inf, dual_inf, gap, step,
                backend->isGpu() ? " [gpu]" : "");
        }

        if (primal_inf <= options_.primal_tol &&
            dual_inf <= options_.dual_tol &&
            gap <= options_.optimality_tol) {
            iters = iter + 1;
            z_unscaled.assign(static_cast<size_t>(n), 0.0);
            y_unscaled.assign(static_cast<size_t>(m), 0.0);
            for (Index j = 0; j < n; ++j) z_unscaled[j] = z[j] * col_scale_[j];
            for (Index i = 0; i < m; ++i) y_unscaled[i] = y[i] * row_scale_[i];
            return true;
        }

        if (iter > 0) {
            if (merit < prev_merit * 0.98) {
                step = std::min(options_.max_step_size, step * options_.step_growth);
            } else if (merit > prev_merit * 1.02) {
                step = std::max(options_.min_step_size, step * options_.step_reduction);
            }
        }
        prev_merit = merit;

        if ((iter + 1) % std::max<Int>(1, options_.major_iteration) == 0) {
            if (!std::isfinite(restart_merit)) restart_merit = merit;
            if (merit < restart_merit * options_.restart_sufficient_reduction) {
                restart_merit = merit;
            } else if (merit > restart_merit * options_.restart_necessary_reduction) {
                z_bar = z;
                step = std::max(options_.min_step_size, step * options_.step_reduction);
                restart_merit = merit;
            }
            if (options_.update_primal_weight) {
                Real ratio = std::sqrt(
                    std::max(primal_inf, eps) / std::max(dual_inf, eps));
                ratio = std::clamp(ratio, 0.5, 2.0);
                primal_weight *= std::pow(ratio, options_.primal_weight_update_smoothing);
                primal_weight = std::clamp(primal_weight, 1e-4, 1e4);
            }
        }
    }

    iters = options_.max_iter;
    return false;
}

LpResult PdlpSolver::solve() {
    if (!loaded_) {
        status_ = Status::Error;
        return {status_, 0.0, 0, 0.0};
    }

    auto t0 = std::chrono::steady_clock::now();

    if (!transformed_ok_) {
        status_ = transformed_infeasible_ ? Status::Infeasible : Status::Error;
        return {status_, 0.0, 0, 0.0};
    }

    std::vector<Real> z_unscaled;
    std::vector<Real> y_unscaled;
    Int iters = 0;
    bool ok = solveStandardForm(z_unscaled, y_unscaled, iters);

    if (!ok) {
        if (options_.stop_flag != nullptr &&
            options_.stop_flag->load(std::memory_order_relaxed)) {
            status_ = Status::IterLimit;
            objective_ = 0.0;
            iterations_ = iters;
            return {status_, objective_, iterations_, 0.0};
        }

        // Robust fallback: preserve correctness if PDLP stalls.
        DualSimplexSolver fallback;
        fallback.load(original_);
        fallback.setVerbose(false);
        auto fb = fallback.solve();
        status_ = fb.status;
        objective_ = fb.objective;
        iterations_ = fb.iterations;
        primal_orig_ = fallback.getPrimalValues();
        dual_eq_.clear();
        reduced_costs_std_.clear();
        return {status_, objective_, iterations_, fb.work_units};
    }

    reconstructOriginalPrimals(z_unscaled);
    if (!checkOriginalPrimalFeasibility(primal_orig_)) {
        DualSimplexSolver fallback;
        fallback.load(original_);
        fallback.setVerbose(false);
        auto fb = fallback.solve();
        status_ = fb.status;
        objective_ = fb.objective;
        iterations_ = fb.iterations;
        primal_orig_ = fallback.getPrimalValues();
        dual_eq_.clear();
        reduced_costs_std_.clear();
        return {status_, objective_, iterations_, fb.work_units};
    }

    dual_eq_ = y_unscaled;
    std::vector<Real> at_y(static_cast<size_t>(cstd_.size()), 0.0);
    aeq_.multiplyTranspose(y_unscaled, at_y);
    reduced_costs_std_.assign(cstd_.size(), 0.0);
    for (Index j = 0; j < static_cast<Index>(cstd_.size()); ++j) {
        reduced_costs_std_[j] = cstd_[j] + at_y[j];
    }

    Real min_obj = std_obj_offset_ + dot(cstd_, z_unscaled);
    objective_ = (original_.sense == Sense::Minimize) ? min_obj : -min_obj;
    status_ = Status::Optimal;
    iterations_ = iters;

    double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    double work = static_cast<double>(iters) *
                  (4.0 * static_cast<double>(scaled_aeq_.numNonzeros()) +
                   2.0 * static_cast<double>(scaled_aeq_.numRows() + scaled_aeq_.numCols()));
    work += seconds * 1e-6;
    return {status_, objective_, iterations_, work};
}

}  // namespace mipx
