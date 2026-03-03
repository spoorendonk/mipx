// GPU Newton-step backends for the barrier solver:
//   GpuCholeskySolver  — Normal Equations + cuDSS SPD Cholesky
//   GpuAugmentedSolver — Augmented system + cuDSS SYMMETRIC LDL'
//
// This file provides the CUDA implementation.  The C++ NewtonSolver
// wrappers are in barrier_gpu_wrap.cpp (compiled by g++, not nvcc).
//
// Only compiled when MIPX_USE_CUDA=ON and cuDSS is found.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudss.h>
#include <cusparse.h>

// Use mipx core types directly (avoid including headers with std::span).
using Real = double;
using Index = int;
using Int = int;

namespace mipx {
namespace gpu_detail {

// ---------------------------------------------------------------------------
// Error checking helpers
// ---------------------------------------------------------------------------

static bool cudaOk(cudaError_t e) { return e == cudaSuccess; }
static bool cusparseOk(cusparseStatus_t e) { return e == CUSPARSE_STATUS_SUCCESS; }
[[maybe_unused]] static bool cublasOk(cublasStatus_t e) { return e == CUBLAS_STATUS_SUCCESS; }
static bool cudssOk(cudssStatus_t e) { return e == CUDSS_STATUS_SUCCESS; }

// ---------------------------------------------------------------------------
// CUDA kernels — block size 256, grid-stride loops
// ---------------------------------------------------------------------------

static constexpr int kBlockSize = 256;

static __host__ int gridSize(int n) {
    return std::min((n + kBlockSize - 1) / kBlockSize, 65535);
}

__global__ void kernelTheta(int n, const double* z, const double* s,
                            double* theta) {
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n;
         j += gridDim.x * blockDim.x) {
        theta[j] = z[j] / fmax(s[j], 1e-20);
    }
}

__global__ void kernelScaledRhsH(int n, const double* rc, const double* s,
                                  const double* theta, const double* rd,
                                  double* h) {
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n;
         j += gridDim.x * blockDim.x) {
        double sj = fmax(s[j], 1e-20);
        h[j] = rc[j] / sj - theta[j] * rd[j];
    }
}

__global__ void kernelNormalEqRhs(int m, const double* rp, const double* ah,
                                  double* rhs) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < m;
         i += gridDim.x * blockDim.x) {
        rhs[i] = rp[i] - ah[i];
    }
}

__global__ void kernelSearchDirectionZ(int n, const double* h,
                                       const double* theta,
                                       const double* at_dy, double* dz) {
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n;
         j += gridDim.x * blockDim.x) {
        dz[j] = h[j] + theta[j] * at_dy[j];
    }
}

__global__ void kernelSearchDirectionS(int n, const double* rd,
                                       const double* at_dy, double* ds) {
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n;
         j += gridDim.x * blockDim.x) {
        ds[j] = rd[j] - at_dy[j];
    }
}

__global__ void kernelFormNEValues(int ne_nnz, const int* ne_row_starts,
                                   const int* ne_col_indices,
                                   const int* a_row_starts,
                                   const int* a_col_indices,
                                   const double* a_values,
                                   const double* theta, double reg,
                                   int m, double* ne_values) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < ne_nnz;
         idx += gridDim.x * blockDim.x) {
        int lo = 0, hi = m;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (ne_row_starts[mid + 1] <= idx)
                lo = mid + 1;
            else
                hi = mid;
        }
        int i = lo;
        int j = ne_col_indices[idx];

        int pi = a_row_starts[i], ei = a_row_starts[i + 1];
        int pj = a_row_starts[j], ej = a_row_starts[j + 1];
        double val = 0.0;
        while (pi < ei && pj < ej) {
            int ci = a_col_indices[pi];
            int cj = a_col_indices[pj];
            if (ci == cj) {
                val += a_values[pi] * theta[ci] * a_values[pj];
                ++pi;
                ++pj;
            } else if (ci < cj) {
                ++pi;
            } else {
                ++pj;
            }
        }
        if (i == j) val += reg;
        ne_values[idx] = val;
    }
}

__global__ void kernelResidualIR(int m, const double* rhs, const double* ax,
                                 double* residual) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < m;
         i += gridDim.x * blockDim.x) {
        residual[i] = rhs[i] - ax[i];
    }
}

__global__ void kernelAddCorrection(int m, const double* dx, double* x) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < m;
         i += gridDim.x * blockDim.x) {
        x[i] += dx[i];
    }
}

__global__ void kernelElemMul(int n, const double* a, const double* b,
                              double* out) {
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n;
         j += gridDim.x * blockDim.x) {
        out[j] = a[j] * b[j];
    }
}

__global__ void kernelAugRhsZ(int n, const double* rd, const double* rc,
                               const double* z, double* rz) {
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n;
         j += gridDim.x * blockDim.x) {
        rz[j] = rd[j] - rc[j] / fmax(z[j], 1e-20);
    }
}

__global__ void kernelDsFromAug(int n, const double* rc, const double* s,
                                 const double* dz, const double* z, double* ds) {
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n;
         j += gridDim.x * blockDim.x) {
        ds[j] = (rc[j] - s[j] * dz[j]) / fmax(z[j], 1e-20);
    }
}

__global__ void kernelFillAugValues(int n, int m,
                                     const double* s, const double* z,
                                     double reg_primal, double reg_dual,
                                     const double* a_values,
                                     const int* a_row_starts,
                                     const int* aug_row_starts,
                                     double* aug_values) {
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n;
         j += gridDim.x * blockDim.x) {
        aug_values[j] = -(s[j] / fmax(z[j], 1e-20)) - reg_primal;
    }
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < m;
         i += gridDim.x * blockDim.x) {
        int aug_start = aug_row_starts[n + i];
        int a_start = a_row_starts[i];
        int a_end = a_row_starts[i + 1];
        int k = 0;
        for (int p = a_start; p < a_end; ++p, ++k) {
            aug_values[aug_start + k] = a_values[p];
        }
        aug_values[aug_start + k] = reg_dual;
    }
}

// ---------------------------------------------------------------------------
// GpuContext: shared CUDA resources
// ---------------------------------------------------------------------------

struct GpuContext {
    int m = 0, n = 0, nnz = 0;
    cudaStream_t stream = nullptr;
    cublasHandle_t cublas_handle = nullptr;
    cusparseHandle_t cusparse_handle = nullptr;

    int* d_a_row_starts = nullptr;
    int* d_a_col_indices = nullptr;
    double* d_a_values = nullptr;

    cusparseSpMatDescr_t a_mat = nullptr;
    cusparseDnVecDescr_t vec_n1 = nullptr, vec_n2 = nullptr;
    cusparseDnVecDescr_t vec_m1 = nullptr, vec_m2 = nullptr;
    void* spmv_buffer_n = nullptr;
    void* spmv_buffer_t = nullptr;

    double* d_pool = nullptr;

    // Named pointers into pool.
    double* d_z = nullptr;
    double* d_s = nullptr;
    double* d_theta = nullptr;
    double* d_h = nullptr;
    double* d_ah = nullptr;
    double* d_rhs_ne = nullptr;
    double* d_atdy = nullptr;
    double* d_rp = nullptr;
    double* d_rd = nullptr;
    double* d_rc = nullptr;
    double* d_dz = nullptr;
    double* d_dy = nullptr;
    double* d_ds = nullptr;
    double* d_tmp = nullptr;
    double* d_ir_residual = nullptr;
    double* d_ir_correction = nullptr;
    double* d_aug_rhs = nullptr;
    double* d_aug_sol = nullptr;

    bool init(int m_in, int n_in, int nnz_in,
              const int* h_row_starts, const int* h_col_indices,
              const double* h_values) {
        m = m_in;
        n = n_in;
        nnz = nnz_in;

        if (!cudaOk(cudaStreamCreate(&stream))) return false;
        if (!cublasOk(cublasCreate(&cublas_handle))) return false;
        if (!cusparseOk(cusparseCreate(&cusparse_handle))) return false;
        cublasSetStream(cublas_handle, stream);
        cusparseSetStream(cusparse_handle, stream);

        if (!cudaOk(cudaMalloc(&d_a_row_starts, sizeof(int) * (m + 1)))) return false;
        if (!cudaOk(cudaMalloc(&d_a_col_indices, sizeof(int) * nnz))) return false;
        if (!cudaOk(cudaMalloc(&d_a_values, sizeof(double) * nnz))) return false;

        cudaMemcpyAsync(d_a_row_starts, h_row_starts, sizeof(int) * (m + 1),
                        cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_a_col_indices, h_col_indices, sizeof(int) * nnz,
                        cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_a_values, h_values, sizeof(double) * nnz,
                        cudaMemcpyHostToDevice, stream);

        if (!cusparseOk(cusparseCreateCsr(
                &a_mat, m, n, nnz, d_a_row_starts, d_a_col_indices, d_a_values,
                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F)))
            return false;

        // Allocate pool.
        int mn = std::max(m, n);
        size_t pool_size = static_cast<size_t>(10 * n + 5 * m + mn + 2 * (n + m));
        if (!cudaOk(cudaMalloc(&d_pool, sizeof(double) * pool_size))) return false;

        double* p = d_pool;
        auto take_n = [&]() { double* r = p; p += n; return r; };
        auto take_m = [&]() { double* r = p; p += m; return r; };
        auto take_nm = [&]() { double* r = p; p += (n + m); return r; };

        d_z = take_n(); d_s = take_n(); d_theta = take_n();
        d_h = take_n(); d_atdy = take_n();
        d_rd = take_n(); d_rc = take_n();
        d_dz = take_n(); d_ds = take_n();
        d_ir_correction = take_n();

        d_ah = take_m(); d_rhs_ne = take_m(); d_rp = take_m();
        d_dy = take_m(); d_ir_residual = take_m();

        d_tmp = p; p += mn;
        d_aug_rhs = take_nm();
        d_aug_sol = take_nm();

        // Dense vector descriptors for SpMV.
        if (!cusparseOk(cusparseCreateDnVec(&vec_n1, n, d_z, CUDA_R_64F))) return false;
        if (!cusparseOk(cusparseCreateDnVec(&vec_n2, n, d_h, CUDA_R_64F))) return false;
        if (!cusparseOk(cusparseCreateDnVec(&vec_m1, m, d_ah, CUDA_R_64F))) return false;
        if (!cusparseOk(cusparseCreateDnVec(&vec_m2, m, d_rp, CUDA_R_64F))) return false;

        size_t buf_size_n = 0, buf_size_t = 0;
        double alpha = 1.0, beta = 0.0;
        cusparseSpMV_bufferSize(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, a_mat, vec_n1, &beta, vec_m1,
                                CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &buf_size_n);
        cusparseSpMV_bufferSize(cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
                                &alpha, a_mat, vec_m2, &beta, vec_n2,
                                CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &buf_size_t);
        if (buf_size_n > 0) cudaMalloc(&spmv_buffer_n, buf_size_n);
        if (buf_size_t > 0) cudaMalloc(&spmv_buffer_t, buf_size_t);

        cudaStreamSynchronize(stream);
        return true;
    }

    bool multiplyA(double* d_x_in, double* d_y_out) {
        cusparseDnVecSetValues(vec_n1, d_x_in);
        cusparseDnVecSetValues(vec_m1, d_y_out);
        double alpha = 1.0, beta = 0.0;
        return cusparseOk(cusparseSpMV(cusparse_handle,
                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        &alpha, a_mat, vec_n1, &beta, vec_m1,
                                        CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                                        spmv_buffer_n));
    }

    bool multiplyAT(double* d_x_in, double* d_y_out) {
        cusparseDnVecSetValues(vec_m2, d_x_in);
        cusparseDnVecSetValues(vec_n2, d_y_out);
        double alpha = 1.0, beta = 0.0;
        return cusparseOk(cusparseSpMV(cusparse_handle,
                                        CUSPARSE_OPERATION_TRANSPOSE,
                                        &alpha, a_mat, vec_m2, &beta, vec_n2,
                                        CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                                        spmv_buffer_t));
    }

    ~GpuContext() {
        if (vec_n1) cusparseDestroyDnVec(vec_n1);
        if (vec_n2) cusparseDestroyDnVec(vec_n2);
        if (vec_m1) cusparseDestroyDnVec(vec_m1);
        if (vec_m2) cusparseDestroyDnVec(vec_m2);
        if (a_mat) cusparseDestroySpMat(a_mat);
        if (spmv_buffer_n) cudaFree(spmv_buffer_n);
        if (spmv_buffer_t) cudaFree(spmv_buffer_t);
        cudaFree(d_a_row_starts);
        cudaFree(d_a_col_indices);
        cudaFree(d_a_values);
        cudaFree(d_pool);
        if (cublas_handle) cublasDestroy(cublas_handle);
        if (cusparse_handle) cusparseDestroy(cusparse_handle);
        if (stream) cudaStreamDestroy(stream);
    }
};

// ---------------------------------------------------------------------------
// GpuCholeskyImpl: NE + cuDSS SPD Cholesky
// ---------------------------------------------------------------------------

struct GpuCholeskyImpl {
    GpuContext ctx;
    int m_ = 0, n_ = 0;
    int ir_steps_ = 2;
    double reg_ = 1e-8;

    int ne_m_ = 0, ne_nnz_ = 0;
    int* d_ne_row_starts = nullptr;
    int* d_ne_col_indices = nullptr;
    double* d_ne_values = nullptr;

    cudssHandle_t cudss_handle = nullptr;
    cudssConfig_t cudss_config = nullptr;
    cudssData_t cudss_data = nullptr;
    cudssMatrix_t cudss_mat = nullptr;
    cudssMatrix_t cudss_rhs_mat = nullptr;
    cudssMatrix_t cudss_sol_mat = nullptr;
    bool analyzed_ = false;

    ~GpuCholeskyImpl() {
        if (cudss_rhs_mat) cudssMatrixDestroy(cudss_rhs_mat);
        if (cudss_sol_mat) cudssMatrixDestroy(cudss_sol_mat);
        if (cudss_mat) cudssMatrixDestroy(cudss_mat);
        if (cudss_data) cudssDataDestroy(cudss_handle, cudss_data);
        if (cudss_config) cudssConfigDestroy(cudss_config);
        if (cudss_handle) cudssDestroy(cudss_handle);
        cudaFree(d_ne_row_starts);
        cudaFree(d_ne_col_indices);
        cudaFree(d_ne_values);
    }

    bool setup(int m, int n, int nnz, int ir_steps,
               const int* h_row_starts, const int* h_col_indices,
               const double* h_values) {
        m_ = m; n_ = n; ir_steps_ = ir_steps;
        if (!ctx.init(m, n, nnz, h_row_starts, h_col_indices, h_values))
            return false;

        // Build NE pattern.
        ne_m_ = m;
        std::vector<std::vector<int>> ne_cols(m);
        for (int i = 0; i < m; ++i) ne_cols[i].push_back(i);

        std::vector<std::vector<int>> col_rows(n);
        for (int i = 0; i < m; ++i) {
            for (int p = h_row_starts[i]; p < h_row_starts[i + 1]; ++p)
                col_rows[h_col_indices[p]].push_back(i);
        }

        for (int k = 0; k < n; ++k) {
            auto& rows = col_rows[k];
            for (size_t a = 0; a < rows.size(); ++a)
                for (size_t b = 0; b <= a; ++b)
                    ne_cols[rows[a]].push_back(rows[b]);
        }

        for (int i = 0; i < m; ++i) {
            auto& row = ne_cols[i];
            std::sort(row.begin(), row.end());
            row.erase(std::unique(row.begin(), row.end()), row.end());
        }

        std::vector<int> h_ne_row_starts(m + 1);
        ne_nnz_ = 0;
        for (int i = 0; i < m; ++i) {
            h_ne_row_starts[i] = ne_nnz_;
            ne_nnz_ += static_cast<int>(ne_cols[i].size());
        }
        h_ne_row_starts[m] = ne_nnz_;

        double avg_density = static_cast<double>(ne_nnz_) / std::max(m, 1);
        if (avg_density > 20.0) return false;

        std::vector<int> h_ne_col_indices(ne_nnz_);
        int pos = 0;
        for (int i = 0; i < m; ++i)
            for (int j : ne_cols[i]) h_ne_col_indices[pos++] = j;

        if (!cudaOk(cudaMalloc(&d_ne_row_starts, sizeof(int) * (m + 1)))) return false;
        if (!cudaOk(cudaMalloc(&d_ne_col_indices, sizeof(int) * ne_nnz_))) return false;
        if (!cudaOk(cudaMalloc(&d_ne_values, sizeof(double) * ne_nnz_))) return false;

        cudaMemcpyAsync(d_ne_row_starts, h_ne_row_starts.data(),
                        sizeof(int) * (m + 1), cudaMemcpyHostToDevice, ctx.stream);
        cudaMemcpyAsync(d_ne_col_indices, h_ne_col_indices.data(),
                        sizeof(int) * ne_nnz_, cudaMemcpyHostToDevice, ctx.stream);
        cudaMemsetAsync(d_ne_values, 0, sizeof(double) * ne_nnz_, ctx.stream);

        // cuDSS setup.
        if (!cudssOk(cudssCreate(&cudss_handle))) return false;
        if (!cudssOk(cudssSetStream(cudss_handle, ctx.stream))) return false;
        if (!cudssOk(cudssConfigCreate(&cudss_config))) return false;
        if (!cudssOk(cudssDataCreate(cudss_handle, &cudss_data))) return false;

        if (!cudssOk(cudssMatrixCreateCsr(
                &cudss_mat, ne_m_, ne_m_, ne_nnz_, d_ne_row_starts,
                nullptr, d_ne_col_indices, d_ne_values,
                CUDA_R_32I, CUDA_R_64F,
                CUDSS_MTYPE_SPD, CUDSS_MVIEW_LOWER, CUDSS_BASE_ZERO)))
            return false;

        if (!cudssOk(cudssMatrixCreateDn(
                &cudss_rhs_mat, ne_m_, 1, ne_m_, ctx.d_rhs_ne, CUDA_R_64F,
                CUDSS_LAYOUT_COL_MAJOR)))
            return false;
        if (!cudssOk(cudssMatrixCreateDn(
                &cudss_sol_mat, ne_m_, 1, ne_m_, ctx.d_dy, CUDA_R_64F,
                CUDSS_LAYOUT_COL_MAJOR)))
            return false;

        if (!cudssOk(cudssExecute(cudss_handle, CUDSS_PHASE_ANALYSIS,
                                   cudss_config, cudss_data, cudss_mat,
                                   cudss_sol_mat, cudss_rhs_mat)))
            return false;

        cudaStreamSynchronize(ctx.stream);
        analyzed_ = true;
        return true;
    }

    bool factorize(const double* z, const double* s, int n, double reg) {
        reg_ = std::max(reg, 1e-12);
        cudaMemcpyAsync(ctx.d_z, z, sizeof(double) * n,
                        cudaMemcpyHostToDevice, ctx.stream);
        cudaMemcpyAsync(ctx.d_s, s, sizeof(double) * n,
                        cudaMemcpyHostToDevice, ctx.stream);

        kernelTheta<<<gridSize(n), kBlockSize, 0, ctx.stream>>>(
            n, ctx.d_z, ctx.d_s, ctx.d_theta);

        kernelFormNEValues<<<gridSize(ne_nnz_), kBlockSize, 0, ctx.stream>>>(
            ne_nnz_, d_ne_row_starts, d_ne_col_indices,
            ctx.d_a_row_starts, ctx.d_a_col_indices, ctx.d_a_values,
            ctx.d_theta, reg_, ne_m_, d_ne_values);

        auto phase = analyzed_ ? CUDSS_PHASE_REFACTORIZATION
                               : CUDSS_PHASE_FACTORIZATION;
        auto status = cudssExecute(cudss_handle, phase, cudss_config,
                                    cudss_data, cudss_mat, cudss_sol_mat,
                                    cudss_rhs_mat);
        if (!cudssOk(status)) {
            status = cudssExecute(cudss_handle, CUDSS_PHASE_FACTORIZATION,
                                   cudss_config, cudss_data, cudss_mat,
                                   cudss_sol_mat, cudss_rhs_mat);
            if (!cudssOk(status)) return false;
        }
        return true;
    }

    bool cudssSolveNE(double* d_rhs_in, double* d_sol_out) {
        if (cudss_rhs_mat) cudssMatrixDestroy(cudss_rhs_mat);
        if (cudss_sol_mat) cudssMatrixDestroy(cudss_sol_mat);
        cudss_rhs_mat = nullptr;
        cudss_sol_mat = nullptr;

        if (!cudssOk(cudssMatrixCreateDn(&cudss_rhs_mat, ne_m_, 1, ne_m_,
                                          d_rhs_in, CUDA_R_64F,
                                          CUDSS_LAYOUT_COL_MAJOR)))
            return false;
        if (!cudssOk(cudssMatrixCreateDn(&cudss_sol_mat, ne_m_, 1, ne_m_,
                                          d_sol_out, CUDA_R_64F,
                                          CUDSS_LAYOUT_COL_MAJOR)))
            return false;

        return cudssOk(cudssExecute(cudss_handle, CUDSS_PHASE_SOLVE,
                                     cudss_config, cudss_data, cudss_mat,
                                     cudss_sol_mat, cudss_rhs_mat));
    }

    bool solveNE(double* d_rhs_in, double* d_sol_out) {
        if (!cudssSolveNE(d_rhs_in, d_sol_out)) return false;

        for (int step = 0; step < ir_steps_; ++step) {
            ctx.multiplyAT(d_sol_out, ctx.d_tmp);
            kernelElemMul<<<gridSize(n_), kBlockSize, 0, ctx.stream>>>(
                n_, ctx.d_tmp, ctx.d_theta, ctx.d_tmp);
            ctx.multiplyA(ctx.d_tmp, ctx.d_ir_residual);

            double reg = reg_;
            cublasSetStream(ctx.cublas_handle, ctx.stream);
            cublasDaxpy(ctx.cublas_handle, m_, &reg, d_sol_out, 1,
                        ctx.d_ir_residual, 1);

            kernelResidualIR<<<gridSize(m_), kBlockSize, 0, ctx.stream>>>(
                m_, d_rhs_in, ctx.d_ir_residual, ctx.d_ir_residual);

            if (!cudssSolveNE(ctx.d_ir_residual, ctx.d_ir_correction)) return false;

            kernelAddCorrection<<<gridSize(m_), kBlockSize, 0, ctx.stream>>>(
                m_, ctx.d_ir_correction, d_sol_out);
        }
        return true;
    }

    bool solveNewton(const double* rp, const double* rd, const double* rc,
                     double* dz, double* dy, double* ds) {
        cudaMemcpyAsync(ctx.d_rp, rp, sizeof(double) * m_,
                        cudaMemcpyHostToDevice, ctx.stream);
        cudaMemcpyAsync(ctx.d_rd, rd, sizeof(double) * n_,
                        cudaMemcpyHostToDevice, ctx.stream);
        cudaMemcpyAsync(ctx.d_rc, rc, sizeof(double) * n_,
                        cudaMemcpyHostToDevice, ctx.stream);

        kernelScaledRhsH<<<gridSize(n_), kBlockSize, 0, ctx.stream>>>(
            n_, ctx.d_rc, ctx.d_s, ctx.d_theta, ctx.d_rd, ctx.d_h);
        ctx.multiplyA(ctx.d_h, ctx.d_ah);
        kernelNormalEqRhs<<<gridSize(m_), kBlockSize, 0, ctx.stream>>>(
            m_, ctx.d_rp, ctx.d_ah, ctx.d_rhs_ne);

        if (!solveNE(ctx.d_rhs_ne, ctx.d_dy)) return false;

        ctx.multiplyAT(ctx.d_dy, ctx.d_atdy);
        kernelSearchDirectionZ<<<gridSize(n_), kBlockSize, 0, ctx.stream>>>(
            n_, ctx.d_h, ctx.d_theta, ctx.d_atdy, ctx.d_dz);
        kernelSearchDirectionS<<<gridSize(n_), kBlockSize, 0, ctx.stream>>>(
            n_, ctx.d_rd, ctx.d_atdy, ctx.d_ds);

        cudaMemcpyAsync(dz, ctx.d_dz, sizeof(double) * n_,
                        cudaMemcpyDeviceToHost, ctx.stream);
        cudaMemcpyAsync(dy, ctx.d_dy, sizeof(double) * m_,
                        cudaMemcpyDeviceToHost, ctx.stream);
        cudaMemcpyAsync(ds, ctx.d_ds, sizeof(double) * n_,
                        cudaMemcpyDeviceToHost, ctx.stream);
        cudaStreamSynchronize(ctx.stream);
        return true;
    }
};

// ---------------------------------------------------------------------------
// GpuAugmentedImpl: Augmented system + cuDSS SYMMETRIC LDL'
// ---------------------------------------------------------------------------

struct GpuAugmentedImpl {
    GpuContext ctx;
    int m_ = 0, n_ = 0;
    int aug_dim_ = 0, aug_nnz_ = 0;

    int* d_aug_row_starts = nullptr;
    int* d_aug_col_indices = nullptr;
    double* d_aug_values = nullptr;

    cudssHandle_t cudss_handle = nullptr;
    cudssConfig_t cudss_config = nullptr;
    cudssData_t cudss_data = nullptr;
    cudssMatrix_t cudss_mat = nullptr;
    cudssMatrix_t cudss_rhs_mat = nullptr;
    cudssMatrix_t cudss_sol_mat = nullptr;
    bool analyzed_ = false;

    ~GpuAugmentedImpl() {
        if (cudss_rhs_mat) cudssMatrixDestroy(cudss_rhs_mat);
        if (cudss_sol_mat) cudssMatrixDestroy(cudss_sol_mat);
        if (cudss_mat) cudssMatrixDestroy(cudss_mat);
        if (cudss_data) cudssDataDestroy(cudss_handle, cudss_data);
        if (cudss_config) cudssConfigDestroy(cudss_config);
        if (cudss_handle) cudssDestroy(cudss_handle);
        cudaFree(d_aug_row_starts);
        cudaFree(d_aug_col_indices);
        cudaFree(d_aug_values);
    }

    bool setup(int m, int n, int nnz,
               const int* h_row_starts, const int* h_col_indices,
               const double* h_values) {
        m_ = m; n_ = n;
        aug_dim_ = n + m;
        aug_nnz_ = n + nnz + m;

        if (!ctx.init(m, n, nnz, h_row_starts, h_col_indices, h_values))
            return false;

        // Build augmented pattern.
        std::vector<int> h_aug_rows(aug_dim_ + 1);
        std::vector<int> h_aug_cols(aug_nnz_);

        for (int j = 0; j < n; ++j) {
            h_aug_rows[j] = j;
            h_aug_cols[j] = j;
        }

        int pos = n;
        for (int i = 0; i < m; ++i) {
            h_aug_rows[n + i] = pos;
            for (int p = h_row_starts[i]; p < h_row_starts[i + 1]; ++p)
                h_aug_cols[pos++] = h_col_indices[p];
            h_aug_cols[pos++] = n + i;
        }
        h_aug_rows[aug_dim_] = pos;

        if (!cudaOk(cudaMalloc(&d_aug_row_starts, sizeof(int) * (aug_dim_ + 1)))) return false;
        if (!cudaOk(cudaMalloc(&d_aug_col_indices, sizeof(int) * aug_nnz_))) return false;
        if (!cudaOk(cudaMalloc(&d_aug_values, sizeof(double) * aug_nnz_))) return false;

        cudaMemcpyAsync(d_aug_row_starts, h_aug_rows.data(),
                        sizeof(int) * (aug_dim_ + 1), cudaMemcpyHostToDevice, ctx.stream);
        cudaMemcpyAsync(d_aug_col_indices, h_aug_cols.data(),
                        sizeof(int) * aug_nnz_, cudaMemcpyHostToDevice, ctx.stream);
        cudaMemsetAsync(d_aug_values, 0, sizeof(double) * aug_nnz_, ctx.stream);

        // cuDSS setup.
        if (!cudssOk(cudssCreate(&cudss_handle))) return false;
        if (!cudssOk(cudssSetStream(cudss_handle, ctx.stream))) return false;
        if (!cudssOk(cudssConfigCreate(&cudss_config))) return false;
        if (!cudssOk(cudssDataCreate(cudss_handle, &cudss_data))) return false;

        if (!cudssOk(cudssMatrixCreateCsr(
                &cudss_mat, aug_dim_, aug_dim_, aug_nnz_, d_aug_row_starts,
                nullptr, d_aug_col_indices, d_aug_values,
                CUDA_R_32I, CUDA_R_64F,
                CUDSS_MTYPE_SYMMETRIC, CUDSS_MVIEW_LOWER, CUDSS_BASE_ZERO)))
            return false;

        if (!cudssOk(cudssMatrixCreateDn(
                &cudss_rhs_mat, aug_dim_, 1, aug_dim_, ctx.d_aug_rhs,
                CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR)))
            return false;
        if (!cudssOk(cudssMatrixCreateDn(
                &cudss_sol_mat, aug_dim_, 1, aug_dim_, ctx.d_aug_sol,
                CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR)))
            return false;

        if (!cudssOk(cudssExecute(cudss_handle, CUDSS_PHASE_ANALYSIS,
                                   cudss_config, cudss_data, cudss_mat,
                                   cudss_sol_mat, cudss_rhs_mat)))
            return false;

        cudaStreamSynchronize(ctx.stream);
        analyzed_ = true;
        return true;
    }

    bool factorize(const double* z, const double* s, int n, double reg) {
        double reg_val = std::max(reg, 1e-12);

        cudaMemcpyAsync(ctx.d_z, z, sizeof(double) * n,
                        cudaMemcpyHostToDevice, ctx.stream);
        cudaMemcpyAsync(ctx.d_s, s, sizeof(double) * n,
                        cudaMemcpyHostToDevice, ctx.stream);

        kernelFillAugValues<<<gridSize(std::max(n_, m_)), kBlockSize, 0,
                              ctx.stream>>>(
            n_, m_, ctx.d_s, ctx.d_z, reg_val, reg_val,
            ctx.d_a_values, ctx.d_a_row_starts, d_aug_row_starts,
            d_aug_values);

        auto phase = analyzed_ ? CUDSS_PHASE_REFACTORIZATION
                               : CUDSS_PHASE_FACTORIZATION;
        auto status = cudssExecute(cudss_handle, phase, cudss_config,
                                    cudss_data, cudss_mat, cudss_sol_mat,
                                    cudss_rhs_mat);
        if (!cudssOk(status)) {
            status = cudssExecute(cudss_handle, CUDSS_PHASE_FACTORIZATION,
                                   cudss_config, cudss_data, cudss_mat,
                                   cudss_sol_mat, cudss_rhs_mat);
            if (!cudssOk(status)) return false;
        }
        return true;
    }

    bool cudssSolveAug(double* d_rhs_in, double* d_sol_out) {
        if (cudss_rhs_mat) cudssMatrixDestroy(cudss_rhs_mat);
        if (cudss_sol_mat) cudssMatrixDestroy(cudss_sol_mat);
        cudss_rhs_mat = nullptr;
        cudss_sol_mat = nullptr;

        if (!cudssOk(cudssMatrixCreateDn(&cudss_rhs_mat, aug_dim_, 1, aug_dim_,
                                          d_rhs_in, CUDA_R_64F,
                                          CUDSS_LAYOUT_COL_MAJOR)))
            return false;
        if (!cudssOk(cudssMatrixCreateDn(&cudss_sol_mat, aug_dim_, 1, aug_dim_,
                                          d_sol_out, CUDA_R_64F,
                                          CUDSS_LAYOUT_COL_MAJOR)))
            return false;

        return cudssOk(cudssExecute(cudss_handle, CUDSS_PHASE_SOLVE,
                                     cudss_config, cudss_data, cudss_mat,
                                     cudss_sol_mat, cudss_rhs_mat));
    }

    bool solveNewton(const double* rp, const double* rd, const double* rc,
                     double* dz, double* dy, double* ds) {
        cudaMemcpyAsync(ctx.d_rp, rp, sizeof(double) * m_,
                        cudaMemcpyHostToDevice, ctx.stream);
        cudaMemcpyAsync(ctx.d_rd, rd, sizeof(double) * n_,
                        cudaMemcpyHostToDevice, ctx.stream);
        cudaMemcpyAsync(ctx.d_rc, rc, sizeof(double) * n_,
                        cudaMemcpyHostToDevice, ctx.stream);

        kernelAugRhsZ<<<gridSize(n_), kBlockSize, 0, ctx.stream>>>(
            n_, ctx.d_rd, ctx.d_rc, ctx.d_z, ctx.d_aug_rhs);
        cudaMemcpyAsync(ctx.d_aug_rhs + n_, ctx.d_rp, sizeof(double) * m_,
                        cudaMemcpyDeviceToDevice, ctx.stream);

        if (!cudssSolveAug(ctx.d_aug_rhs, ctx.d_aug_sol)) return false;

        cudaMemcpyAsync(ctx.d_dz, ctx.d_aug_sol, sizeof(double) * n_,
                        cudaMemcpyDeviceToDevice, ctx.stream);
        cudaMemcpyAsync(ctx.d_dy, ctx.d_aug_sol + n_, sizeof(double) * m_,
                        cudaMemcpyDeviceToDevice, ctx.stream);

        kernelDsFromAug<<<gridSize(n_), kBlockSize, 0, ctx.stream>>>(
            n_, ctx.d_rc, ctx.d_s, ctx.d_dz, ctx.d_z, ctx.d_ds);

        cudaMemcpyAsync(dz, ctx.d_dz, sizeof(double) * n_,
                        cudaMemcpyDeviceToHost, ctx.stream);
        cudaMemcpyAsync(dy, ctx.d_dy, sizeof(double) * m_,
                        cudaMemcpyDeviceToHost, ctx.stream);
        cudaMemcpyAsync(ds, ctx.d_ds, sizeof(double) * n_,
                        cudaMemcpyDeviceToHost, ctx.stream);
        cudaStreamSynchronize(ctx.stream);
        return true;
    }
};

// ---------------------------------------------------------------------------
// C-linkage bridge functions called from barrier_gpu_wrap.cpp
// ---------------------------------------------------------------------------

extern "C" {

void* gpuCholeskyCreate() { return new GpuCholeskyImpl(); }
void gpuCholeskyDestroy(void* p) { delete static_cast<GpuCholeskyImpl*>(p); }

bool gpuCholeskySetup(void* p, int m, int n, int nnz, int ir_steps,
                      const int* rows, const int* cols, const double* vals) {
    return static_cast<GpuCholeskyImpl*>(p)->setup(m, n, nnz, ir_steps, rows, cols, vals);
}

bool gpuCholeskyFactorize(void* p, const double* z, const double* s, int n, double reg) {
    return static_cast<GpuCholeskyImpl*>(p)->factorize(z, s, n, reg);
}

bool gpuCholeskySolveNewton(void* p, const double* rp, const double* rd,
                             const double* rc, double* dz, double* dy, double* ds) {
    return static_cast<GpuCholeskyImpl*>(p)->solveNewton(rp, rd, rc, dz, dy, ds);
}

void* gpuAugmentedCreate() { return new GpuAugmentedImpl(); }
void gpuAugmentedDestroy(void* p) { delete static_cast<GpuAugmentedImpl*>(p); }

bool gpuAugmentedSetup(void* p, int m, int n, int nnz,
                        const int* rows, const int* cols, const double* vals) {
    return static_cast<GpuAugmentedImpl*>(p)->setup(m, n, nnz, rows, cols, vals);
}

bool gpuAugmentedFactorize(void* p, const double* z, const double* s, int n, double reg) {
    return static_cast<GpuAugmentedImpl*>(p)->factorize(z, s, n, reg);
}

bool gpuAugmentedSolveNewton(void* p, const double* rp, const double* rd,
                              const double* rc, double* dz, double* dy, double* ds) {
    return static_cast<GpuAugmentedImpl*>(p)->solveNewton(rp, rd, rc, dz, dy, ds);
}

}  // extern "C"

}  // namespace gpu_detail
}  // namespace mipx
