#include "mipx/pdlp.h"

#ifdef MIPX_HAS_CUDA

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <span>
#include <vector>

#include <cuda_runtime.h>
#include <cusparse.h>

#include "pdlp_kernels.cuh"
#include "mipx/sparse_matrix.h"

namespace mipx {

// ============================================================================
// GpuPdlpBackend — GPU-resident PDLP solver
// ============================================================================

namespace {

inline bool cu_ok(cudaError_t e) { return e == cudaSuccess; }
inline bool cs_ok(cusparseStatus_t e) { return e == CUSPARSE_STATUS_SUCCESS; }

template <typename T>
bool deviceUpload(T* dst, const T* src, int count) {
    return cu_ok(cudaMemcpy(dst, src, sizeof(T) * static_cast<size_t>(count),
                            cudaMemcpyHostToDevice));
}

template <typename T>
bool deviceDownload(T* dst, const T* src, int count) {
    return cu_ok(cudaMemcpy(dst, src, sizeof(T) * static_cast<size_t>(count),
                            cudaMemcpyDeviceToHost));
}

}  // namespace

class GpuPdlpBackend {
public:
    GpuPdlpBackend() = default;
    ~GpuPdlpBackend() { cleanup(); }

    GpuPdlpBackend(const GpuPdlpBackend&) = delete;
    GpuPdlpBackend& operator=(const GpuPdlpBackend&) = delete;

    bool initialize(const SparseMatrix& scaled_aeq,
                    std::span<const Real> c, std::span<const Real> b,
                    std::span<const Real> sigma, std::span<const Real> tau,
                    std::span<const Real> col_scale,
                    std::span<const Real> row_scale) {
        m_ = scaled_aeq.numRows();
        n_ = scaled_aeq.numCols();
        if (m_ <= 0 || n_ <= 0) return false;

        if (!cs_ok(cusparseCreate(&handle_))) return false;
        if (!cu_ok(cudaStreamCreate(&stream_))) return false;
        cusparseSetStream(handle_, stream_);

        if (!uploadCSR(scaled_aeq)) return false;
        if (!buildTransposedCSR(scaled_aeq)) return false;
        if (!createSpMatDescriptors()) return false;

        // Arena allocation: single cudaMalloc for all vectors
        if (!allocateArena()) return false;

        // Zero all arena memory first, then upload constants over the zeros.
        // This ensures iteration vectors (z, y, etc.) start at zero while
        // constant data (c, b, sigma, tau, scales) is properly initialized.
        cudaMemsetAsync(arena_, 0, arena_bytes_, stream_);
        cu_ok(cudaStreamSynchronize(stream_));

        // Upload constant data (overwrites the zeroed constant regions)
        if (!deviceUpload(d_c_, c.data(), n_)) return false;
        if (!deviceUpload(d_b_, b.data(), m_)) return false;
        if (!deviceUpload(d_sigma_, sigma.data(), m_)) return false;
        if (!deviceUpload(d_tau_, tau.data(), n_)) return false;
        if (!deviceUpload(d_col_scale_, col_scale.data(), n_)) return false;
        if (!deviceUpload(d_row_scale_, row_scale.data(), m_)) return false;

        // Create dense vector descriptors (point directly at iteration vectors)
        if (!cs_ok(cusparseCreateDnVec(&vec_x_a_, n_, d_z_, CUDA_R_64F))) return false;
        if (!cs_ok(cusparseCreateDnVec(&vec_y_a_, m_, d_az_, CUDA_R_64F))) return false;
        if (!cs_ok(cusparseCreateDnVec(&vec_x_at_, m_, d_y_, CUDA_R_64F))) return false;
        if (!cs_ok(cusparseCreateDnVec(&vec_y_at_, n_, d_at_y_, CUDA_R_64F))) return false;

        // Allocate SpMV buffers
        const double alpha = 1.0, beta = 0.0;
        size_t buf_size_a = 0;
        if (!cs_ok(cusparseSpMV_bufferSize(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha, mat_a_, vec_x_a_, &beta, vec_y_a_,
                                           CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                                           &buf_size_a))) return false;
        if (buf_size_a > 0) {
            if (!cu_ok(cudaMalloc(&buf_a_, buf_size_a))) return false;
        }

        size_t buf_size_at = 0;
        if (!cs_ok(cusparseSpMV_bufferSize(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha, mat_at_, vec_x_at_, &beta, vec_y_at_,
                                           CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                                           &buf_size_at))) return false;
        if (buf_size_at > 0) {
            if (!cu_ok(cudaMalloc(&buf_at_, buf_size_at))) return false;
        }

        // SpMV preprocess (one-time analysis of sparsity pattern)
        if (buf_a_) {
            cusparseSpMV_preprocess(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                   &alpha, mat_a_, vec_x_a_, &beta, vec_y_a_,
                                   CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, buf_a_);
        }
        if (buf_at_) {
            cusparseSpMV_preprocess(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                   &alpha, mat_at_, vec_x_at_, &beta, vec_y_at_,
                                   CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, buf_at_);
        }

        // Pinned host memory for scalar downloads (avoid double-sync)
        cu_ok(cudaMallocHost(&h_scalars_, sizeof(double) * 4));

        // Device scalars for convergence/movement
        d_metrics_ = reinterpret_cast<gpu::ConvergenceMetrics*>(
            arena_end()); // After arena — allocate separately
        cu_ok(cudaMalloc(reinterpret_cast<void**>(&d_metrics_),
                         sizeof(gpu::ConvergenceMetrics)));
        d_movement_ = reinterpret_cast<double*>(
            reinterpret_cast<char*>(d_metrics_) + sizeof(gpu::ConvergenceMetrics));
        // Actually allocate these separately for simplicity
        cu_ok(cudaMalloc(reinterpret_cast<void**>(&d_movement_), sizeof(double)));
        cu_ok(cudaMalloc(reinterpret_cast<void**>(&d_interaction_), sizeof(double)));

        cu_ok(cudaMalloc(reinterpret_cast<void**>(&d_fp_result_), sizeof(double) * 2));

        if (!d_metrics_ || !d_movement_ || !d_interaction_ || !d_fp_result_) return false;

        ok_ = true;
        return true;
    }

    bool initialized() const { return ok_; }

    // -----------------------------------------------------------------------
    // Halpern PDHG iteration
    // -----------------------------------------------------------------------

    void primalHalpernStep(double step, double primal_weight, double lambda) {
        double step_over_pw = step / primal_weight;
        gpu::launchPrimalHalpernStep(d_z_, d_z_prev_, d_z_refl_,
                                     d_z0_, d_c_, d_at_y_, d_tau_,
                                     step_over_pw, lambda, n_, stream_);
    }

    void dualHalpernStep(double step, double primal_weight, double lambda) {
        double step_pw = step * primal_weight;
        gpu::launchDualHalpernStep(d_y_, d_y_prev_, d_y0_,
                                   d_az_refl_, d_b_, d_sigma_,
                                   step_pw, lambda, m_, stream_);
    }

    // -----------------------------------------------------------------------
    // SpMV using cusparseDnVecSetValues (no D2D copies)
    // -----------------------------------------------------------------------

    bool computeATy() {
        // A^T * y: input=y, output=at_y
        cusparseDnVecSetValues(vec_x_at_, d_y_);
        cusparseDnVecSetValues(vec_y_at_, d_at_y_);
        const double alpha = 1.0, beta = 0.0;
        return cs_ok(cusparseSpMV(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  &alpha, mat_at_, vec_x_at_, &beta, vec_y_at_,
                                  CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, buf_at_));
    }

    bool computeAzReflected() {
        // A * z_refl: input=z_refl, output=az_refl
        cusparseDnVecSetValues(vec_x_a_, d_z_refl_);
        cusparseDnVecSetValues(vec_y_a_, d_az_refl_);
        const double alpha = 1.0, beta = 0.0;
        return cs_ok(cusparseSpMV(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  &alpha, mat_a_, vec_x_a_, &beta, vec_y_a_,
                                  CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, buf_a_));
    }

    bool computeAz() {
        // A * z: input=z, output=az
        cusparseDnVecSetValues(vec_x_a_, d_z_);
        cusparseDnVecSetValues(vec_y_a_, d_az_);
        const double alpha = 1.0, beta = 0.0;
        return cs_ok(cusparseSpMV(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  &alpha, mat_a_, vec_x_a_, &beta, vec_y_a_,
                                  CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, buf_a_));
    }

    bool computeAx(const double* x, double* ax_out) {
        cusparseDnVecSetValues(vec_x_a_, const_cast<double*>(x));
        cusparseDnVecSetValues(vec_y_a_, ax_out);
        const double alpha = 1.0, beta = 0.0;
        return cs_ok(cusparseSpMV(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  &alpha, mat_a_, vec_x_a_, &beta, vec_y_a_,
                                  CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, buf_a_));
    }

    bool computeATx(const double* x, double* atx_out) {
        cusparseDnVecSetValues(vec_x_at_, const_cast<double*>(x));
        cusparseDnVecSetValues(vec_y_at_, atx_out);
        const double alpha = 1.0, beta = 0.0;
        return cs_ok(cusparseSpMV(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  &alpha, mat_at_, vec_x_at_, &beta, vec_y_at_,
                                  CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, buf_at_));
    }

    // -----------------------------------------------------------------------
    // Adaptive step size
    // -----------------------------------------------------------------------

    void saveATyPrev() {
        gpu::launchCopyVector(d_at_y_prev_, d_at_y_, n_, stream_);
    }

    bool computeMovementInteraction(double primal_weight,
                                    double& movement, double& interaction) {
        gpu::launchMovementInteraction(d_movement_, d_interaction_,
                                       d_z_, d_z_prev_,
                                       d_at_y_, d_at_y_prev_,
                                       d_y_, d_y_prev_,
                                       primal_weight, n_, m_, stream_);
        cu_ok(cudaStreamSynchronize(stream_));
        // Use pinned memory for async-friendly download
        if (!deviceDownload(h_scalars_, d_movement_, 1)) return false;
        if (!deviceDownload(h_scalars_ + 1, d_interaction_, 1)) return false;
        movement = h_scalars_[0];
        interaction = 2.0 * std::abs(h_scalars_[1]);
        return true;
    }

    // -----------------------------------------------------------------------
    // Halpern restart / fixed-point error
    // -----------------------------------------------------------------------

    void restartAnchor() {
        // z0 = z, y0 = y
        gpu::launchCopyVector(d_z0_, d_z_, n_, stream_);
        gpu::launchCopyVector(d_y0_, d_y_, m_, stream_);
    }

    bool computeFixedPointError(double& primal_dist_sq, double& dual_dist_sq) {
        gpu::launchFixedPointError(d_fp_result_, d_z_, d_z0_,
                                    d_y_, d_y0_, n_, m_, stream_);
        cu_ok(cudaStreamSynchronize(stream_));
        double results[2];
        if (!deviceDownload(results, d_fp_result_, 2)) return false;
        primal_dist_sq = results[0];
        dual_dist_sq = results[1];
        return true;
    }

    // -----------------------------------------------------------------------
    // Convergence
    // -----------------------------------------------------------------------

    bool computeConvergence(double inv_b, double inv_c, double obj_offset,
                            gpu::ConvergenceMetrics& metrics) {
        gpu::launchConvergence(d_metrics_, d_az_, d_b_,
                               d_z_, d_at_y_, d_c_, d_y_,
                               inv_b, inv_c, obj_offset, m_, n_, stream_);
        cu_ok(cudaStreamSynchronize(stream_));
        return deviceDownload(&metrics, d_metrics_, 1);
    }


    // -----------------------------------------------------------------------
    // Solution download
    // -----------------------------------------------------------------------

    bool downloadSolution(std::span<Real> z_out, std::span<Real> y_out) {
        gpu::launchUnscale(d_z_refl_, d_z_, d_col_scale_, n_, stream_);
        gpu::launchUnscale(d_az_refl_, d_y_, d_row_scale_, m_, stream_);
        cu_ok(cudaStreamSynchronize(stream_));
        if (!deviceDownload(z_out.data(), d_z_refl_, n_)) return false;
        if (!deviceDownload(y_out.data(), d_az_refl_, m_)) return false;
        return true;
    }

    cudaStream_t stream() const { return stream_; }
    int rows() const { return m_; }
    int cols() const { return n_; }

private:
    void cleanup() {
        if (vec_x_a_) cusparseDestroyDnVec(vec_x_a_);
        if (vec_y_a_) cusparseDestroyDnVec(vec_y_a_);
        if (vec_x_at_) cusparseDestroyDnVec(vec_x_at_);
        if (vec_y_at_) cusparseDestroyDnVec(vec_y_at_);
        if (mat_a_) cusparseDestroySpMat(mat_a_);
        if (mat_at_) cusparseDestroySpMat(mat_at_);

        if (buf_a_) cudaFree(buf_a_);
        if (buf_at_) cudaFree(buf_at_);

        if (d_a_values_) cudaFree(d_a_values_);
        if (d_a_col_idx_) cudaFree(d_a_col_idx_);
        if (d_a_row_ptr_) cudaFree(d_a_row_ptr_);
        if (d_at_values_) cudaFree(d_at_values_);
        if (d_at_col_idx_) cudaFree(d_at_col_idx_);
        if (d_at_row_ptr_) cudaFree(d_at_row_ptr_);

        if (arena_) cudaFree(arena_);
        if (d_metrics_) cudaFree(d_metrics_);
        if (d_movement_) cudaFree(d_movement_);
        if (d_interaction_) cudaFree(d_interaction_);
        if (d_fp_result_) cudaFree(d_fp_result_);
        if (h_scalars_) cudaFreeHost(h_scalars_);

        if (handle_) cusparseDestroy(handle_);
        if (stream_) cudaStreamDestroy(stream_);

        handle_ = nullptr;
        stream_ = nullptr;
        arena_ = nullptr;
        ok_ = false;
    }

    void* arena_end() const {
        return reinterpret_cast<char*>(arena_) + arena_bytes_;
    }

    bool uploadCSR(const SparseMatrix& A) {
        auto vals = A.csr_values();
        auto cols = A.csr_col_indices();
        auto rows = A.csr_row_starts();
        nnz_a_ = A.numNonzeros();

        auto alloc = [](auto*& ptr, size_t bytes) -> bool {
            return cu_ok(cudaMalloc(reinterpret_cast<void**>(&ptr), bytes));
        };

        if (!alloc(d_a_values_, sizeof(double) * nnz_a_)) return false;
        if (!alloc(d_a_col_idx_, sizeof(int) * nnz_a_)) return false;
        if (!alloc(d_a_row_ptr_, sizeof(int) * (m_ + 1))) return false;

        if (!deviceUpload(d_a_values_, vals.data(), nnz_a_)) return false;
        if (!deviceUpload(d_a_col_idx_, cols.data(), nnz_a_)) return false;
        if (!deviceUpload(d_a_row_ptr_, rows.data(), m_ + 1)) return false;
        return true;
    }

    bool buildTransposedCSR(const SparseMatrix& A) {
        auto a_vals = A.csr_values();
        auto a_cols = A.csr_col_indices();
        auto a_rows = A.csr_row_starts();
        nnz_at_ = nnz_a_;

        std::vector<int> col_count(n_ + 1, 0);
        for (int k = 0; k < nnz_a_; ++k) col_count[a_cols[k] + 1]++;
        std::vector<int> at_row_ptr(n_ + 1, 0);
        for (int j = 0; j < n_; ++j) at_row_ptr[j + 1] = at_row_ptr[j] + col_count[j + 1];

        std::vector<double> at_vals(nnz_a_);
        std::vector<int> at_cols(nnz_a_);
        std::vector<int> pos(n_, 0);

        for (int i = 0; i < m_; ++i) {
            for (int k = a_rows[i]; k < a_rows[i + 1]; ++k) {
                int j = a_cols[k];
                int dest = at_row_ptr[j] + pos[j];
                at_vals[dest] = a_vals[k];
                at_cols[dest] = i;
                pos[j]++;
            }
        }

        auto alloc = [](auto*& ptr, size_t bytes) -> bool {
            return cu_ok(cudaMalloc(reinterpret_cast<void**>(&ptr), bytes));
        };

        if (!alloc(d_at_values_, sizeof(double) * nnz_at_)) return false;
        if (!alloc(d_at_col_idx_, sizeof(int) * nnz_at_)) return false;
        if (!alloc(d_at_row_ptr_, sizeof(int) * (n_ + 1))) return false;

        if (!deviceUpload(d_at_values_, at_vals.data(), nnz_at_)) return false;
        if (!deviceUpload(d_at_col_idx_, at_cols.data(), nnz_at_)) return false;
        if (!deviceUpload(d_at_row_ptr_, at_row_ptr.data(), n_ + 1)) return false;
        return true;
    }

    bool createSpMatDescriptors() {
        if (!cs_ok(cusparseCreateCsr(
                &mat_a_, int64_t(m_), int64_t(n_), int64_t(nnz_a_),
                d_a_row_ptr_, d_a_col_idx_, d_a_values_,
                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F))) return false;
        if (!cs_ok(cusparseCreateCsr(
                &mat_at_, int64_t(n_), int64_t(m_), int64_t(nnz_at_),
                d_at_row_ptr_, d_at_col_idx_, d_at_values_,
                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F))) return false;
        return true;
    }

    // Single arena allocation for all iteration vectors.
    // Layout: [primal-sized vectors (n)] [dual-sized vectors (m)] [constants]
    bool allocateArena() {
        // Primal-sized: z, z_prev, z_refl, z0, at_y, at_y_prev,
        //               c, tau, col_scale = 9 vectors of size n
        // Dual-sized:   y, y_prev, az_refl, az, y0, b, sigma,
        //               row_scale = 8 vectors of size m
        int num_primal = 9;
        int num_dual = 8;
        size_t primal_bytes = sizeof(double) * static_cast<size_t>(n_) * num_primal;
        size_t dual_bytes = sizeof(double) * static_cast<size_t>(m_) * num_dual;
        arena_bytes_ = primal_bytes + dual_bytes;

        if (!cu_ok(cudaMalloc(&arena_, arena_bytes_))) return false;

        // Carve sub-allocations
        double* base = reinterpret_cast<double*>(arena_);
        int off = 0;

        // Primal-sized (length n each)
        d_z_ = base + off; off += n_;
        d_z_prev_ = base + off; off += n_;
        d_z_refl_ = base + off; off += n_;
        d_z0_ = base + off; off += n_;
        d_at_y_ = base + off; off += n_;
        d_at_y_prev_ = base + off; off += n_;
        d_c_ = base + off; off += n_;
        d_tau_ = base + off; off += n_;
        d_col_scale_ = base + off; off += n_;

        // Dual-sized (length m each)
        d_y_ = base + off; off += m_;
        d_y_prev_ = base + off; off += m_;
        d_az_refl_ = base + off; off += m_;
        d_az_ = base + off; off += m_;
        d_y0_ = base + off; off += m_;
        d_b_ = base + off; off += m_;
        d_sigma_ = base + off; off += m_;
        d_row_scale_ = base + off; off += m_;

        return true;
    }

    // -----------------------------------------------------------------------
    // Member data
    // -----------------------------------------------------------------------

    bool ok_ = false;
    int m_ = 0, n_ = 0;
    int nnz_a_ = 0, nnz_at_ = 0;

    cudaStream_t stream_ = nullptr;
    cusparseHandle_t handle_ = nullptr;

    // A CSR on device
    double* d_a_values_ = nullptr;
    int* d_a_col_idx_ = nullptr;
    int* d_a_row_ptr_ = nullptr;

    // A^T CSR on device
    double* d_at_values_ = nullptr;
    int* d_at_col_idx_ = nullptr;
    int* d_at_row_ptr_ = nullptr;

    // cuSPARSE descriptors
    cusparseSpMatDescr_t mat_a_ = nullptr;
    cusparseSpMatDescr_t mat_at_ = nullptr;
    cusparseDnVecDescr_t vec_x_a_ = nullptr;
    cusparseDnVecDescr_t vec_y_a_ = nullptr;
    cusparseDnVecDescr_t vec_x_at_ = nullptr;
    cusparseDnVecDescr_t vec_y_at_ = nullptr;

    // SpMV workspace buffers
    void* buf_a_ = nullptr;
    void* buf_at_ = nullptr;

    // Arena allocation (single cudaMalloc for all vectors)
    void* arena_ = nullptr;
    size_t arena_bytes_ = 0;

    // Pointers into arena — primal-sized (length n)
    double* d_z_ = nullptr;
    double* d_z_prev_ = nullptr;
    double* d_z_refl_ = nullptr;
    double* d_z0_ = nullptr;
    double* d_at_y_ = nullptr;
    double* d_at_y_prev_ = nullptr;
    double* d_c_ = nullptr;
    double* d_tau_ = nullptr;
    double* d_col_scale_ = nullptr;

    // Pointers into arena — dual-sized (length m)
    double* d_y_ = nullptr;
    double* d_y_prev_ = nullptr;
    double* d_az_refl_ = nullptr;
    double* d_az_ = nullptr;
    double* d_y0_ = nullptr;
    double* d_b_ = nullptr;
    double* d_sigma_ = nullptr;
    double* d_row_scale_ = nullptr;

    // Device scalars (separate allocations)
    gpu::ConvergenceMetrics* d_metrics_ = nullptr;
    double* d_movement_ = nullptr;
    double* d_interaction_ = nullptr;
    double* d_fp_result_ = nullptr;

    // Pinned host memory for scalar downloads
    double* h_scalars_ = nullptr;
};


// ============================================================================
// GPU-resident PDLP solve entry point
// ============================================================================

bool solveStandardFormGpu(
    const SparseMatrix& scaled_aeq,
    std::span<const Real> bscaled,
    std::span<const Real> cscaled,
    std::span<const Real> sigma_base,
    std::span<const Real> tau_base,
    std::span<const Real> col_scale,
    std::span<const Real> row_scale,
    Real std_obj_offset,
    const PdlpOptions& options,
    std::vector<Real>& z_unscaled,
    std::vector<Real>& y_unscaled,
    Int& iters) {

    const Index m = scaled_aeq.numRows();
    const Index n = scaled_aeq.numCols();

    GpuPdlpBackend gpu;
    if (!gpu.initialize(scaled_aeq, cscaled, bscaled, sigma_base, tau_base,
                        col_scale, row_scale)) {
        return false;
    }

    // Normalisation constants
    Real max_b = 0.0, max_c = 0.0;
    for (auto v : bscaled) max_b = std::max(max_b, std::abs(v));
    for (auto v : cscaled) max_c = std::max(max_c, std::abs(v));
    const Real inv_b = 1.0 / (1.0 + max_b);
    const Real inv_c = 1.0 / (1.0 + max_c);

    Real step = std::clamp(options.initial_step_size,
                           options.min_step_size, options.max_step_size);

    Real primal_weight = std::max(options.primal_weight, 1e-6);
    if (options.update_primal_weight) {
        Real nb = 0.0, nc = 0.0;
        for (auto v : bscaled) nb += v * v;
        for (auto v : cscaled) nc += v * v;
        nb = std::sqrt(nb);
        nc = std::sqrt(nc);
        if (nb > 1e-12 && nc > 1e-12) primal_weight = std::clamp(nc / nb, 1e-3, 1e3);
    }

    // Halpern PDHG state
    Int inner_count = 0;

    // Fixed-point restart state
    Real restart_fp = std::numeric_limits<Real>::infinity();
    Real candidate_fp = std::numeric_limits<Real>::infinity();
    Int last_restart_iter = 0;
    Int step_size_updates = 0;

    // PID primal weight state
    Real pw_error_sum = 0.0;

    // Exponential convergence check intervals
    auto isMajorIter = [](Int iter) -> bool {
        if (iter <= 10) return true;
        Int s = 10;
        while (iter >= s * 10) s *= 10;
        return (iter % s == 0);
    };

    for (Index iter = 0; iter < options.max_iter; ++iter) {
        if (options.stop_flag != nullptr &&
            options.stop_flag->load(std::memory_order_relaxed)) {
            iters = iter;
            return false;
        }

        // --- Halpern PDHG iteration (2 SpMV) ---
        double lambda = (inner_count > 0)
            ? static_cast<double>(inner_count) / static_cast<double>(inner_count + 1)
            : 0.0;

        // 1. ATy = A^T * y
        if (!gpu.computeATy()) return false;

        // 2. Primal: project + reflect + Halpern blend (saves z_prev internally)
        gpu.primalHalpernStep(step, primal_weight, lambda);

        // 3. A * z_refl (SpMV on reflected point, NOT Halpern-blended z)
        if (!gpu.computeAzReflected()) return false;

        // 4. Dual: step + reflect + Halpern blend (saves y_prev internally)
        gpu.dualHalpernStep(step, primal_weight, lambda);

        ++inner_count;

        // --- Adaptive step size (movement/interaction) ---
        if (iter > 0 && iter % 4 == 0) {
            gpu.saveATyPrev();
            if (!gpu.computeATy()) {};
            double movement = 0.0, interaction = 0.0;
            if (gpu.computeMovementInteraction(primal_weight, movement, interaction)) {
                if (interaction > 1e-30) {
                    double step_limit = movement / interaction;
                    ++step_size_updates;
                    double k = static_cast<double>(step_size_updates + 1);
                    double first_term = (1.0 - std::pow(k, -options.step_size_reduction_exponent))
                                        * step_limit;
                    double second_term = (1.0 + std::pow(k, -options.step_size_growth_exponent))
                                         * step;
                    step = std::clamp(std::min(first_term, second_term),
                                      options.min_step_size, options.max_step_size);
                }
            }
        }

        // --- Convergence check at major iterations ---
        bool is_major = isMajorIter(iter + 1);

        if (is_major && iter > 0) {
            if (!gpu.computeAz()) return false;

            gpu::ConvergenceMetrics metrics{};
            if (!gpu.computeConvergence(inv_b, inv_c, std_obj_offset, metrics))
                return false;

            Real primal_inf = metrics.primal_inf * inv_b;
            Real dual_inf = metrics.dual_inf * inv_c;
            Real pobj = std_obj_offset + metrics.pobj;
            Real dobj = std_obj_offset + metrics.dobj;

            Real gap = std::abs(pobj - dobj) /
                       (1.0 + std::abs(pobj) + std::abs(dobj));

            if (options.verbose) {
                std::printf(
                    "PDLP %6d  pobj=% .10e  pinf=% .2e  dinf=% .2e  gap=% .2e  step=% .2e [gpu]\n",
                    iter + 1, pobj, primal_inf, dual_inf, gap, step);
            }

            if (primal_inf <= options.primal_tol &&
                dual_inf <= options.dual_tol &&
                gap <= options.optimality_tol) {
                iters = iter + 1;
                z_unscaled.resize(static_cast<size_t>(n));
                y_unscaled.resize(static_cast<size_t>(m));
                gpu.downloadSolution(z_unscaled, y_unscaled);
                return true;
            }

            // --- Fixed-point restart decision ---
            double primal_dist_sq = 0.0, dual_dist_sq = 0.0;
            if (!gpu.computeFixedPointError(primal_dist_sq, dual_dist_sq))
                return false;

            Real fp_error = primal_weight * primal_dist_sq +
                            dual_dist_sq / primal_weight;

            if (!std::isfinite(restart_fp)) restart_fp = fp_error;

            bool do_restart = false;
            if (fp_error < options.restart_sufficient_decay * restart_fp) {
                do_restart = true;
            } else if (fp_error < options.restart_necessary_decay * restart_fp &&
                       fp_error > candidate_fp) {
                do_restart = true;
            } else if (iter - last_restart_iter >
                       static_cast<Int>(options.restart_artificial_fraction *
                                        static_cast<double>(iter + 1))) {
                do_restart = true;
            }

            candidate_fp = fp_error;

            if (do_restart) {
                gpu.restartAnchor();
                inner_count = 0;

                restart_fp = fp_error;
                last_restart_iter = iter;

                // PID primal weight update
                if (options.update_primal_weight &&
                    primal_dist_sq > 1e-20 && dual_dist_sq > 1e-20) {
                    Real error = 0.5 * (std::log(dual_dist_sq) -
                                        std::log(primal_dist_sq)) -
                                 std::log(primal_weight);
                    pw_error_sum = options.pid_i_smooth * pw_error_sum +
                                   (1.0 - options.pid_i_smooth) * error;
                    Real log_pw = std::log(primal_weight) +
                                  options.pid_kp * error +
                                  options.pid_ki * pw_error_sum;
                    primal_weight = std::clamp(std::exp(log_pw), 1e-4, 1e4);
                }
            }
        }
    }

    iters = options.max_iter;
    return false;
}

}  // namespace mipx

#endif  // MIPX_HAS_CUDA
