#include "mipx/barrier.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
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

inline Real maxStepToBoundary(std::span<const Real> x, std::span<const Real> dx,
                              Real fraction) {
    Real alpha = 1.0;
    for (Index i = 0; i < static_cast<Index>(x.size()); ++i) {
        if (dx[i] < 0.0) {
            alpha = std::min(alpha, -fraction * x[i] / dx[i]);
        }
    }
    if (alpha < 0.0) alpha = 0.0;
    if (alpha > 1.0) alpha = 1.0;
    return alpha;
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

        if (!cudaOk(cudaMemcpy(d_values_, vals.data(), sizeof(Real) * vals.size(), cudaMemcpyHostToDevice)))
            return;
        if (!cudaOk(cudaMemcpy(d_col_indices_, cols.data(), sizeof(Index) * cols.size(), cudaMemcpyHostToDevice)))
            return;
        if (!cudaOk(cudaMemcpy(d_row_starts_, rows.data(), sizeof(Index) * rows.size(), cudaMemcpyHostToDevice)))
            return;

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
        if (static_cast<Index>(x.size()) != cols_ || static_cast<Index>(y.size()) != rows_)
            return false;
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
        if (static_cast<Index>(x.size()) != rows_ || static_cast<Index>(y.size()) != cols_)
            return false;
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
        if (resized) {
            destroyVectorDescriptors();
        }
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
        size_t* target_size = (op == CUSPARSE_OPERATION_NON_TRANSPOSE) ? &buffer_n_size_ : &buffer_t_size_;
        void** target_buf = (op == CUSPARSE_OPERATION_NON_TRANSPOSE) ? &buffer_n_ : &buffer_t_;
        if (*target_size > 0) return true;
        cusparseDnVecDescr_t vec_x = (op == CUSPARSE_OPERATION_NON_TRANSPOSE) ? vec_x_n_ : vec_x_t_;
        cusparseDnVecDescr_t vec_y = (op == CUSPARSE_OPERATION_NON_TRANSPOSE) ? vec_y_n_ : vec_y_t_;
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

std::vector<Real> BarrierSolver::getPrimalValues() const {
    return primal_orig_;
}

std::vector<Real> BarrierSolver::getDualValues() const {
    // Equality-dual mapping back to ranged rows is not yet exposed.
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

    std::unique_ptr<LinearBackend> backend = std::make_unique<CpuBackend>(aeq_);
#ifdef MIPX_HAS_CUDA
    std::unique_ptr<CudaBackend> cuda_backend;
    bool gpu_worthwhile = options_.use_gpu &&
                          aeq_.numRows() >= options_.gpu_min_rows &&
                          aeq_.numNonzeros() >= options_.gpu_min_nnz;
    if (gpu_worthwhile) {
        cuda_backend = std::make_unique<CudaBackend>(aeq_);
        if (cuda_backend->initialized()) {
            used_gpu_ = true;
            backend = std::move(cuda_backend);
        }
    }
#endif

    const Real reg = std::max(options_.regularization, 1e-12);

    if (m == 0) {
        z.assign(static_cast<size_t>(n), 0.0);
        y.clear();
        s = cstd_;
        for (Index j = 0; j < n; ++j) {
            if (s[j] < -1e-12) {
                return false;  // unbounded in transformed space
            }
        }
        iters = 0;
        return true;
    }

    z.assign(static_cast<size_t>(n), 1.0);
    y.assign(static_cast<size_t>(m), 0.0);
    s.assign(static_cast<size_t>(n), 1.0);

    std::vector<Real> az(static_cast<size_t>(m), 0.0);
    std::vector<Real> at_y(static_cast<size_t>(n), 0.0);
    std::vector<Real> rp(static_cast<size_t>(m), 0.0);
    std::vector<Real> rd(static_cast<size_t>(n), 0.0);
    std::vector<Real> rc(static_cast<size_t>(n), 0.0);

    std::vector<Real> theta(static_cast<size_t>(n), 0.0);
    std::vector<Real> h(static_cast<size_t>(n), 0.0);
    std::vector<Real> ah(static_cast<size_t>(m), 0.0);
    std::vector<Real> rhs(static_cast<size_t>(m), 0.0);
    std::vector<Real> dy(static_cast<size_t>(m), 0.0);
    std::vector<Real> at_dy(static_cast<size_t>(n), 0.0);
    std::vector<Real> dz(static_cast<size_t>(n), 0.0);
    std::vector<Real> ds(static_cast<size_t>(n), 0.0);

    std::vector<Real> dz_aff(static_cast<size_t>(n), 0.0);
    std::vector<Real> dy_aff(static_cast<size_t>(m), 0.0);
    std::vector<Real> ds_aff(static_cast<size_t>(n), 0.0);

    std::vector<Real> diag_precond(static_cast<size_t>(m), reg);

    std::vector<Real> cg_r(static_cast<size_t>(m), 0.0);
    std::vector<Real> cg_z(static_cast<size_t>(m), 0.0);
    std::vector<Real> cg_p(static_cast<size_t>(m), 0.0);
    std::vector<Real> cg_q(static_cast<size_t>(m), 0.0);
    std::vector<Real> cg_tmp_n(static_cast<size_t>(n), 0.0);

    auto applyNormalEq = [&](std::span<const Real> v, std::span<Real> out,
                             std::span<const Real> theta_local) -> bool {
        std::fill(cg_tmp_n.begin(), cg_tmp_n.end(), 0.0);
        if (!backend->multiplyTranspose(v, cg_tmp_n)) return false;
        for (Index j = 0; j < n; ++j) cg_tmp_n[j] *= theta_local[j];
        if (!backend->multiply(cg_tmp_n, out)) return false;
        for (Index i = 0; i < m; ++i) out[i] += reg * v[i];
        return true;
    };

    auto solveNormalEq = [&](std::span<const Real> rhs_local,
                             std::span<Real> sol,
                             std::span<const Real> theta_local) -> bool {
        std::fill(sol.begin(), sol.end(), 0.0);

        std::fill(diag_precond.begin(), diag_precond.end(), reg);
        for (Index i = 0; i < m; ++i) {
            auto rv = aeq_.row(i);
            Real d = reg;
            for (Index k = 0; k < rv.size(); ++k) {
                Index j = rv.indices[k];
                Real a = rv.values[k];
                d += theta_local[j] * a * a;
            }
            diag_precond[i] = std::max(d, reg);
        }

        cg_r.assign(rhs_local.begin(), rhs_local.end());
        for (Index i = 0; i < m; ++i) cg_z[i] = cg_r[i] / diag_precond[i];
        cg_p = cg_z;

        Real rz = dot(cg_r, cg_z);
        Real rhs_n = l2Norm(rhs_local);
        if (rhs_n < 1e-30) {
            std::fill(sol.begin(), sol.end(), 0.0);
            return true;
        }

        Real tol_abs = std::max(options_.cg_rel_tol * rhs_n, 1e-14);

        for (Index it = 0; it < options_.max_cg_iter; ++it) {
            std::fill(cg_q.begin(), cg_q.end(), 0.0);
            if (!applyNormalEq(cg_p, cg_q, theta_local)) return false;
            Real pAp = dot(cg_p, cg_q);
            if (std::abs(pAp) < 1e-30) {
                return false;
            }

            Real alpha = rz / pAp;
            for (Index i = 0; i < m; ++i) {
                sol[i] += alpha * cg_p[i];
                cg_r[i] -= alpha * cg_q[i];
            }

            Real rnorm = l2Norm(cg_r);
            if (rnorm <= tol_abs) {
                return true;
            }

            for (Index i = 0; i < m; ++i) cg_z[i] = cg_r[i] / diag_precond[i];
            Real rz_new = dot(cg_r, cg_z);
            if (std::abs(rz) < 1e-30) return false;
            Real beta = rz_new / rz;
            for (Index i = 0; i < m; ++i) {
                cg_p[i] = cg_z[i] + beta * cg_p[i];
            }
            rz = rz_new;
        }

        return false;
    };

    auto solveNewton = [&](std::span<const Real> rp_local,
                           std::span<const Real> rd_local,
                           std::span<const Real> rc_local,
                           std::span<Real> dz_out,
                           std::span<Real> dy_out,
                           std::span<Real> ds_out) -> bool {
        for (Index j = 0; j < n; ++j) {
            Real sj = std::max(s[j], 1e-12);
            theta[j] = z[j] / sj;
            h[j] = rc_local[j] / sj - theta[j] * rd_local[j];
        }

        std::fill(ah.begin(), ah.end(), 0.0);
        if (!backend->multiply(h, ah)) return false;
        for (Index i = 0; i < m; ++i) rhs[i] = rp_local[i] - ah[i];

        if (!solveNormalEq(rhs, dy_out, theta)) return false;

        std::fill(at_dy.begin(), at_dy.end(), 0.0);
        if (!backend->multiplyTranspose(dy_out, at_dy)) return false;

        for (Index j = 0; j < n; ++j) {
            ds_out[j] = rd_local[j] - at_dy[j];
            dz_out[j] = h[j] + theta[j] * at_dy[j];
        }
        return true;
    };

    const Real inv_b = 1.0 / (1.0 + infNorm(beq_));
    const Real inv_c = 1.0 / (1.0 + infNorm(cstd_));

    for (Index iter = 0; iter < options_.max_iter; ++iter) {
        std::fill(az.begin(), az.end(), 0.0);
        std::fill(at_y.begin(), at_y.end(), 0.0);
        if (!backend->multiply(z, az)) return false;
        if (!backend->multiplyTranspose(y, at_y)) return false;

        for (Index i = 0; i < m; ++i) rp[i] = beq_[i] - az[i];
        for (Index j = 0; j < n; ++j) rd[j] = cstd_[j] - at_y[j] - s[j];

        Real mu = dot(z, s) / std::max<Index>(n, 1);
        Real primal_inf = infNorm(rp) * inv_b;
        Real dual_inf = infNorm(rd) * inv_c;
        Real gap = std::abs(mu) / (1.0 + std::abs(std_obj_offset_ + dot(cstd_, z)));

        if (options_.verbose) {
            std::printf("IPM %4d  pobj=% .10e  pinf=% .2e  dinf=% .2e  gap=% .2e%s\n",
                        iter,
                        std_obj_offset_ + dot(cstd_, z),
                        primal_inf, dual_inf, gap,
                        backend->isGpu() ? " [gpu]" : "");
        }

        if (primal_inf < options_.primal_dual_tol &&
            dual_inf < options_.primal_dual_tol &&
            gap < options_.primal_dual_tol) {
            iters = iter;
            return true;
        }

        for (Index j = 0; j < n; ++j) rc[j] = -z[j] * s[j];
        if (!solveNewton(rp, rd, rc, dz_aff, dy_aff, ds_aff)) return false;

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

        for (Index j = 0; j < n; ++j) {
            rc[j] = sigma * mu - z[j] * s[j] - dz_aff[j] * ds_aff[j];
        }

        if (!solveNewton(rp, rd, rc, dz, dy, ds)) return false;

        Real alpha_p = maxStepToBoundary(z, dz, options_.step_fraction);
        Real alpha_d = maxStepToBoundary(s, ds, options_.step_fraction);

        for (Index j = 0; j < n; ++j) {
            z[j] += alpha_p * dz[j];
            s[j] += alpha_d * ds[j];
            z[j] = std::max(z[j], 1e-12);
            s[j] = std::max(s[j], 1e-12);
        }
        for (Index i = 0; i < m; ++i) y[i] += alpha_d * dy[i];
    }

    iters = options_.max_iter;
    return false;
}

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
        // Robust fallback: preserve correctness if IPM fails to converge.
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

        double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        double work = fb.work_units;
        (void)seconds;
        return {status_, objective_, iterations_, work};
    }

    reconstructOriginalPrimals(z);
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

    dual_eq_ = y;
    reduced_costs_std_ = s;

    Real min_obj = std_obj_offset_ + dot(cstd_, z);
    objective_ = (original_.sense == Sense::Minimize) ? min_obj : -min_obj;
    status_ = Status::Optimal;
    iterations_ = iters;

    double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();

    // Deterministic, rough work estimate for benchmarking gates.
    double work = static_cast<double>(iters) *
                  (4.0 * static_cast<double>(aeq_.numNonzeros()) +
                   10.0 * static_cast<double>(aeq_.numRows()));
    work += seconds * 1e-6;

    return {status_, objective_, iterations_, work};
}

}  // namespace mipx
