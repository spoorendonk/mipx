// GPU-accelerated barrier solver backend: cuDSS sparse Cholesky + cusparse SpMV.
// This file is only compiled when MIPX_USE_CUDA=ON.

#include "mipx/barrier.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudss.h>
#include <cusparse.h>

namespace mipx {
namespace detail {

// ---------------------------------------------------------------------------
// Error checking helpers
// ---------------------------------------------------------------------------

static bool cudaOk(cudaError_t e) { return e == cudaSuccess; }
static bool cusparseOk(cusparseStatus_t e) { return e == CUSPARSE_STATUS_SUCCESS; }
static bool cublasOk(cublasStatus_t e) { return e == CUBLAS_STATUS_SUCCESS; }
static bool cudssOk(cudssStatus_t e) { return e == CUDSS_STATUS_SUCCESS; }

static const char* cudssStatusName(cudssStatus_t status) {
    switch (status) {
    case CUDSS_STATUS_SUCCESS:
        return "success";
    case CUDSS_STATUS_NOT_INITIALIZED:
        return "not_initialized";
    case CUDSS_STATUS_ALLOC_FAILED:
        return "alloc_failed";
    case CUDSS_STATUS_INVALID_VALUE:
        return "invalid_value";
    case CUDSS_STATUS_NOT_SUPPORTED:
        return "not_supported";
    case CUDSS_STATUS_EXECUTION_FAILED:
        return "execution_failed";
    case CUDSS_STATUS_INTERNAL_ERROR:
        return "internal_error";
    }
    return "unknown";
}

// ---------------------------------------------------------------------------
// CUDA kernels — block size 256, grid-stride loops
// ---------------------------------------------------------------------------

static constexpr int kBlockSize = 256;

static __host__ int gridSize(int n) {
    return std::min((n + kBlockSize - 1) / kBlockSize, 65535);
}

__global__ void kernelResidualPrimal(int m, const double* b, const double* az,
                                     double* rp) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < m;
         i += gridDim.x * blockDim.x) {
        rp[i] = b[i] - az[i];
    }
}

__global__ void kernelResidualDual(int n, const double* c, const double* aty,
                                   const double* s, double* rd) {
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n;
         j += gridDim.x * blockDim.x) {
        rd[j] = c[j] - aty[j] - s[j];
    }
}

__global__ void kernelTheta(int n, const double* z, const double* s,
                            double* theta) {
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n;
         j += gridDim.x * blockDim.x) {
        theta[j] = z[j] / fmax(s[j], 1e-20);
    }
}

__global__ void kernelComplementarityAffine(int n, const double* z,
                                            const double* s, double* rc) {
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n;
         j += gridDim.x * blockDim.x) {
        rc[j] = -z[j] * s[j];
    }
}

__global__ void kernelComplementarityCorrected(int n, double sigma_mu,
                                               const double* z, const double* s,
                                               const double* dz_aff,
                                               const double* ds_aff,
                                               double* rc) {
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n;
         j += gridDim.x * blockDim.x) {
        rc[j] = sigma_mu - z[j] * s[j] - dz_aff[j] * ds_aff[j];
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

__global__ void kernelUpdateIterates(int n, int m, double alpha_p,
                                     double alpha_d, const double* dz,
                                     const double* ds, const double* dy,
                                     double* z, double* s, double* y,
                                     double min_val) {
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n;
         j += gridDim.x * blockDim.x) {
        z[j] = fmax(z[j] + alpha_p * dz[j], min_val);
        s[j] = fmax(s[j] + alpha_d * ds[j], min_val);
    }
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < m;
         i += gridDim.x * blockDim.x) {
        y[i] += alpha_d * dy[i];
    }
}

__global__ void kernelFillOnes(int n, double* v) {
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n;
         j += gridDim.x * blockDim.x) {
        v[j] = 1.0;
    }
}

__global__ void kernelInit(int n, double val, double* v) {
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n;
         j += gridDim.x * blockDim.x) {
        v[j] = val;
    }
}

// Dot product via shared memory reduction.
__global__ void kernelDot(int n, const double* a, const double* b,
                          double* result) {
    __shared__ double sdata[kBlockSize];
    double sum = 0.0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        sum += a[i] * b[i];
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(result, sdata[0]);
}

__global__ void kernelDotPartial(int n, const double* a, const double* b,
                                 double* partials) {
    __shared__ double sdata[kBlockSize];
    double sum = 0.0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        sum += a[i] * b[i];
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) partials[blockIdx.x] = sdata[0];
}

__global__ void kernelDot4Partial(int n, const double* a0, const double* b0,
                                  const double* a1, const double* b1,
                                  const double* a2, const double* b2,
                                  const double* a3, const double* b3,
                                  double* partials) {
    __shared__ double s0[kBlockSize];
    __shared__ double s1[kBlockSize];
    __shared__ double s2[kBlockSize];
    __shared__ double s3[kBlockSize];
    double sum0 = 0.0;
    double sum1 = 0.0;
    double sum2 = 0.0;
    double sum3 = 0.0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        sum0 += a0[i] * b0[i];
        sum1 += a1[i] * b1[i];
        sum2 += a2[i] * b2[i];
        sum3 += a3[i] * b3[i];
    }
    s0[threadIdx.x] = sum0;
    s1[threadIdx.x] = sum1;
    s2[threadIdx.x] = sum2;
    s3[threadIdx.x] = sum3;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s0[threadIdx.x] += s0[threadIdx.x + s];
            s1[threadIdx.x] += s1[threadIdx.x + s];
            s2[threadIdx.x] += s2[threadIdx.x + s];
            s3[threadIdx.x] += s3[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        partials[blockIdx.x] = s0[0];
        partials[gridDim.x + blockIdx.x] = s1[0];
        partials[2 * gridDim.x + blockIdx.x] = s2[0];
        partials[3 * gridDim.x + blockIdx.x] = s3[0];
    }
}

// Inf-norm reduction.
__global__ void kernelInfNorm(int n, const double* v, double* result) {
    __shared__ double sdata[kBlockSize];
    double mx = 0.0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        mx = fmax(mx, fabs(v[i]));
    }
    sdata[threadIdx.x] = mx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] = fmax(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        // Use atomicMax via CAS on unsigned long long for doubles.
        unsigned long long int* addr = (unsigned long long int*)result;
        unsigned long long int old = *addr;
        unsigned long long int assumed;
        do {
            assumed = old;
            double old_val = __longlong_as_double(assumed);
            double new_val = fmax(old_val, sdata[0]);
            old = atomicCAS(addr, assumed, __double_as_longlong(new_val));
        } while (assumed != old);
    }
}

// Max step-to-boundary: find max alpha s.t. x + alpha*dx >= 0.
// Stores result as min of -fraction*x/dx for negative dx.
__global__ void kernelMaxStep(int n, const double* x, const double* dx,
                              double fraction, double* result) {
    __shared__ double sdata[kBlockSize];
    double alpha = 1.0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        if (dx[i] < 0.0) {
            alpha = fmin(alpha, -fraction * x[i] / dx[i]);
        }
    }
    sdata[threadIdx.x] = alpha;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] = fmin(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        unsigned long long int* addr = (unsigned long long int*)result;
        unsigned long long int old = *addr;
        unsigned long long int assumed;
        do {
            assumed = old;
            double old_val = __longlong_as_double(assumed);
            double new_val = fmin(old_val, sdata[0]);
            old = atomicCAS(addr, assumed, __double_as_longlong(new_val));
        } while (assumed != old);
    }
}

// Form NE values: ADA' formation via merge-join of sorted CSR rows.
// ne_row_starts/ne_col_indices: CSR of lower-triangle NE pattern.
// a_row_starts/a_col_indices/a_values: CSR of A.
// For each NE entry (i, j) with i >= j, compute sum_k A(i,k)*theta(k)*A(j,k).
__global__ void kernelFormNEValues(int ne_nnz, const int* ne_row_starts,
                                   const int* ne_col_indices,
                                   const int* a_row_starts,
                                   const int* a_col_indices,
                                   const double* a_values,
                                   const double* theta, double reg,
                                   int m, double* ne_values) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < ne_nnz;
         idx += gridDim.x * blockDim.x) {
        // Find which row this NE entry belongs to.
        // Binary search in ne_row_starts.
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

        // Merge-join A(i,:) and A(j,:).
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

__global__ void kernelAddReg(int m, const int* ne_row_starts,
                             const int* ne_col_indices, double reg,
                             double* ne_values) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < m;
         i += gridDim.x * blockDim.x) {
        // Diagonal is where ne_col_indices[k] == i.
        for (int k = ne_row_starts[i]; k < ne_row_starts[i + 1]; ++k) {
            if (ne_col_indices[k] == i) {
                ne_values[k] += reg;
                break;
            }
        }
    }
}

// Compute z*s elementwise and store in out.
__global__ void kernelZS(int n, const double* z, const double* s, double* out) {
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n;
         j += gridDim.x * blockDim.x) {
        out[j] = z[j] * s[j];
    }
}

// Shift z and s for Mehrotra starting point.
__global__ void kernelShiftPositive(int n, double dz, double dd, double dp,
                                    double* z, double* s) {
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n;
         j += gridDim.x * blockDim.x) {
        z[j] += dz + dp;
        s[j] += dd;
        z[j] = fmax(z[j], 1e-4);
        s[j] = fmax(s[j], 1e-4);
    }
}

// Negate and axpy: out = -a + b*c.
__global__ void kernelResidualIR(int m, const double* rhs, const double* ax,
                                 double* residual) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < m;
         i += gridDim.x * blockDim.x) {
        residual[i] = rhs[i] - ax[i];
    }
}

// Add correction: x += dx.
__global__ void kernelAddCorrection(int m, const double* dx, double* x) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < m;
         i += gridDim.x * blockDim.x) {
        x[i] += dx[i];
    }
}

// Sum reduction for mu computation.
__global__ void kernelSum(int n, const double* v, double* result) {
    __shared__ double sdata[kBlockSize];
    double sum = 0.0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        sum += v[i];
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(result, sdata[0]);
}

__global__ void kernelSumPartial(int n, const double* v, double* partials) {
    __shared__ double sdata[kBlockSize];
    double sum = 0.0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        sum += v[i];
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) partials[blockIdx.x] = sdata[0];
}

// Min reduction for finding min(z) or min(s).
__global__ void kernelMin(int n, const double* v, double* result) {
    __shared__ double sdata[kBlockSize];
    double mn = 1e30;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        mn = fmin(mn, v[i]);
    }
    sdata[threadIdx.x] = mn;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] = fmin(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        unsigned long long int* addr = (unsigned long long int*)result;
        unsigned long long int old = *addr;
        unsigned long long int assumed;
        do {
            assumed = old;
            double old_val = __longlong_as_double(assumed);
            double new_val = fmin(old_val, sdata[0]);
            old = atomicCAS(addr, assumed, __double_as_longlong(new_val));
        } while (assumed != old);
    }
}

// ---------------------------------------------------------------------------
// GpuContext: manages GPU memory for IPM vectors and cusparse SpMV
// ---------------------------------------------------------------------------

struct GpuContext {
    cudaStream_t stream = nullptr;
    cusparseHandle_t cusparse_handle = nullptr;
    cublasHandle_t cublas_handle = nullptr;

    // Matrix A in CSR on device.
    int m = 0, n = 0, nnz = 0;
    int* d_a_row_starts = nullptr;
    int* d_a_col_indices = nullptr;
    double* d_a_values = nullptr;
    cusparseSpMatDescr_t a_mat = nullptr;

    // Pooled GPU memory for all IPM vectors (single cudaMalloc).
    double* d_pool = nullptr;

    // IPM vectors (pointers into d_pool).
    double* d_z = nullptr;
    double* d_y = nullptr;
    double* d_s = nullptr;
    double* d_b = nullptr;
    double* d_c = nullptr;

    // Backup vectors for rollback on numerical instability.
    double* d_z_bak = nullptr;
    double* d_y_bak = nullptr;
    double* d_s_bak = nullptr;

    double* d_az = nullptr;
    double* d_aty = nullptr;
    double* d_rp = nullptr;
    double* d_rd = nullptr;
    double* d_rc = nullptr;
    double* d_theta = nullptr;
    double* d_h = nullptr;
    double* d_ah = nullptr;
    double* d_rhs = nullptr;
    double* d_dy = nullptr;
    double* d_atdy = nullptr;
    double* d_dz = nullptr;
    double* d_ds = nullptr;
    double* d_dz_aff = nullptr;
    double* d_dy_aff = nullptr;
    double* d_ds_aff = nullptr;

    // Scratch for reductions.
    double* d_scalar = nullptr;   // 4 scalars on device
    double* d_tmp = nullptr;      // max(m,n)-sized temp for reductions
    double* d_reduce = nullptr;   // partial reductions, size gridSize(max(m,n))
    std::vector<double> h_reduce;

    // cuSPARSE vector descriptors.
    cusparseDnVecDescr_t vec_n1 = nullptr;  // size n, for input
    cusparseDnVecDescr_t vec_n2 = nullptr;  // size n, for output
    cusparseDnVecDescr_t vec_m1 = nullptr;  // size m, for input
    cusparseDnVecDescr_t vec_m2 = nullptr;  // size m, for output
    void* spmv_buffer_n = nullptr;  // A*x buffer (normal)
    void* spmv_buffer_t = nullptr;  // A'*x buffer (transpose)
    size_t spmv_buffer_n_size = 0;
    size_t spmv_buffer_t_size = 0;

    bool ok = false;

    GpuContext() = default;

    double accumulateReduce(int blocks, int offset) const {
        double sum = 0.0;
        const size_t base = static_cast<size_t>(offset) * blocks;
        for (int i = 0; i < blocks; ++i) sum += h_reduce[base + i];
        return sum;
    }

    double downloadReduceSum(int blocks, int offset) {
        cudaMemcpyAsync(h_reduce.data() + static_cast<size_t>(offset) * blocks,
                        d_reduce + static_cast<size_t>(offset) * blocks,
                        sizeof(double) * blocks, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        return accumulateReduce(blocks, offset);
    }

    bool init(const SparseMatrix& a, std::span<const double> b_host,
              std::span<const double> c_host) {
        m = a.numRows();
        n = a.numCols();
        nnz = a.numNonzeros();

        if (!cudaOk(cudaStreamCreate(&stream))) return false;
        if (!cusparseOk(cusparseCreate(&cusparse_handle))) return false;
        if (!cusparseOk(cusparseSetStream(cusparse_handle, stream))) return false;
        if (!cublasOk(cublasCreate(&cublas_handle))) return false;
        if (!cublasOk(cublasSetStream(cublas_handle, stream))) return false;

        // Upload A.
        auto vals = a.csr_values();
        auto cols = a.csr_col_indices();
        auto rows = a.csr_row_starts();

        if (!cudaOk(cudaMalloc(&d_a_row_starts, sizeof(int) * (m + 1)))) return false;
        if (!cudaOk(cudaMalloc(&d_a_col_indices, sizeof(int) * nnz))) return false;
        if (!cudaOk(cudaMalloc(&d_a_values, sizeof(double) * nnz))) return false;
        if (!cudaOk(cudaMemcpyAsync(d_a_row_starts, rows.data(),
                                     sizeof(int) * (m + 1),
                                     cudaMemcpyHostToDevice, stream)))
            return false;
        if (!cudaOk(cudaMemcpyAsync(d_a_col_indices, cols.data(),
                                     sizeof(int) * nnz,
                                     cudaMemcpyHostToDevice, stream)))
            return false;
        if (!cudaOk(cudaMemcpyAsync(d_a_values, vals.data(),
                                     sizeof(double) * nnz,
                                     cudaMemcpyHostToDevice, stream)))
            return false;

        // cusparse matrix descriptor.
        if (!cusparseOk(cusparseCreateCsr(
                &a_mat, m, n, nnz, d_a_row_starts, d_a_col_indices, d_a_values,
                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F)))
            return false;

        // Allocate all IPM vectors in a single cudaMalloc (reduces init overhead).
        // n-sized: z, s, c, z_bak, aty, rd, rc, theta, h, atdy, dz, ds, dz_aff, ds_aff = 14
        // m-sized: y, b, y_bak, az, rp, ah, rhs, dy, dy_aff = 9
        // s_bak: n, tmp: max(m,n), deterministic reduction partials, scalar: 4
        int mn = std::max(m, n);
        int reduce_slots = 4 * std::max(gridSize(mn), 1);
        size_t pool_size =
            static_cast<size_t>(15 * n + 9 * m + mn + reduce_slots + 4);
        if (!cudaOk(cudaMalloc(&d_pool, sizeof(double) * pool_size))) return false;

        double* p = d_pool;
        auto take_n = [&]() { double* r = p; p += n; return r; };
        auto take_m = [&]() { double* r = p; p += m; return r; };

        d_z = take_n(); d_s = take_n(); d_c = take_n();
        d_z_bak = take_n(); d_aty = take_n(); d_rd = take_n();
        d_rc = take_n(); d_theta = take_n(); d_h = take_n();
        d_atdy = take_n(); d_dz = take_n(); d_ds = take_n();
        d_dz_aff = take_n(); d_ds_aff = take_n(); d_s_bak = take_n();

        d_y = take_m(); d_b = take_m(); d_y_bak = take_m();
        d_az = take_m(); d_rp = take_m(); d_ah = take_m();
        d_rhs = take_m(); d_dy = take_m(); d_dy_aff = take_m();

        d_tmp = p; p += mn;
        d_reduce = p; p += reduce_slots;
        d_scalar = p; p += 4;
        h_reduce.assign(static_cast<size_t>(reduce_slots), 0.0);

        // Upload b, c.
        if (!cudaOk(cudaMemcpyAsync(d_b, b_host.data(), sizeof(double) * m,
                                     cudaMemcpyHostToDevice, stream)))
            return false;
        if (!cudaOk(cudaMemcpyAsync(d_c, c_host.data(), sizeof(double) * n,
                                     cudaMemcpyHostToDevice, stream)))
            return false;

        // Create vector descriptors for SpMV.
        if (!cusparseOk(cusparseCreateDnVec(&vec_n1, n, d_z, CUDA_R_64F)))
            return false;
        if (!cusparseOk(cusparseCreateDnVec(&vec_m1, m, d_az, CUDA_R_64F)))
            return false;
        if (!cusparseOk(cusparseCreateDnVec(&vec_m2, m, d_y, CUDA_R_64F)))
            return false;
        if (!cusparseOk(cusparseCreateDnVec(&vec_n2, n, d_aty, CUDA_R_64F)))
            return false;

        // Allocate SpMV buffers.
        double alpha = 1.0, beta = 0.0;
        size_t bsize = 0;
        if (!cusparseOk(cusparseSpMV_bufferSize(
                cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                a_mat, vec_n1, &beta, vec_m1, CUDA_R_64F,
                CUSPARSE_SPMV_ALG_DEFAULT, &bsize)))
            return false;
        spmv_buffer_n_size = std::max(bsize, (size_t)1);
        if (!cudaOk(cudaMalloc(&spmv_buffer_n, spmv_buffer_n_size)))
            return false;

        if (!cusparseOk(cusparseSpMV_bufferSize(
                cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE, &alpha,
                a_mat, vec_m2, &beta, vec_n2, CUDA_R_64F,
                CUSPARSE_SPMV_ALG_DEFAULT, &bsize)))
            return false;
        spmv_buffer_t_size = std::max(bsize, (size_t)1);
        if (!cudaOk(cudaMalloc(&spmv_buffer_t, spmv_buffer_t_size)))
            return false;

        if (!cudaOk(cudaStreamSynchronize(stream))) return false;
        ok = true;
        return true;
    }

    // y = A * x. x must be device pointer of size n, y of size m.
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

    // y = A' * x. x must be device pointer of size m, y of size n.
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

    // Scalar reductions: download from d_scalar[idx].
    double downloadScalar(int idx) {
        double val = 0.0;
        cudaMemcpyAsync(&val, d_scalar + idx, sizeof(double),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        return val;
    }

    // Compute dot(a, b) on GPU and return.
    double gpuDot(double* d_a, double* d_b, int sz) {
        int blocks = std::max(gridSize(sz), 1);
        kernelDotPartial<<<blocks, kBlockSize, 0, stream>>>(sz, d_a, d_b,
                                                            d_reduce);
        return downloadReduceSum(blocks, 0);
    }

    // Compute inf-norm on GPU.
    double gpuInfNorm(double* d_v, int sz) {
        cudaMemsetAsync(d_scalar, 0, sizeof(double), stream);
        kernelInfNorm<<<gridSize(sz), kBlockSize, 0, stream>>>(sz, d_v,
                                                                d_scalar);
        return downloadScalar(0);
    }

    // Compute max step to boundary on GPU.
    double gpuMaxStep(double* d_x, double* d_dx, int sz, double fraction) {
        // Initialize to 1.0.
        double one = 1.0;
        cudaMemcpyAsync(d_scalar, &one, sizeof(double), cudaMemcpyHostToDevice,
                        stream);
        kernelMaxStep<<<gridSize(sz), kBlockSize, 0, stream>>>(
            sz, d_x, d_dx, fraction, d_scalar);
        double alpha = downloadScalar(0);
        if (alpha < 0.0) alpha = 0.0;
        if (alpha > 1.0) alpha = 1.0;
        return alpha;
    }

    // Compute mu = sum(z*s) / n on GPU.
    double gpuMu(double* d_z_in, double* d_s_in, int sz) {
        kernelZS<<<gridSize(sz), kBlockSize, 0, stream>>>(sz, d_z_in, d_s_in,
                                                           d_tmp);
        int blocks = std::max(gridSize(sz), 1);
        kernelSumPartial<<<blocks, kBlockSize, 0, stream>>>(sz, d_tmp,
                                                            d_reduce);
        return downloadReduceSum(blocks, 0) / std::max(sz, 1);
    }

    // Compute min of vector.
    double gpuMin(double* d_v, int sz) {
        double big = 1e30;
        cudaMemcpyAsync(d_scalar, &big, sizeof(double), cudaMemcpyHostToDevice,
                        stream);
        kernelMin<<<gridSize(sz), kBlockSize, 0, stream>>>(sz, d_v, d_scalar);
        return downloadScalar(0);
    }

    // Compute sum of vector.
    double gpuSum(double* d_v, int sz) {
        int blocks = std::max(gridSize(sz), 1);
        kernelSumPartial<<<blocks, kBlockSize, 0, stream>>>(sz, d_v,
                                                            d_reduce);
        return downloadReduceSum(blocks, 0);
    }

    std::array<double, 4> gpuDot4(double* d_a0, double* d_b0, double* d_a1,
                                  double* d_b1, double* d_a2, double* d_b2,
                                  double* d_a3, double* d_b3, int sz) {
        int blocks = std::max(gridSize(sz), 1);
        kernelDot4Partial<<<blocks, kBlockSize, 0, stream>>>(
            sz, d_a0, d_b0, d_a1, d_b1, d_a2, d_b2, d_a3, d_b3, d_reduce);
        cudaMemcpyAsync(h_reduce.data(), d_reduce, sizeof(double) * 4 * blocks,
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        return {accumulateReduce(blocks, 0), accumulateReduce(blocks, 1),
                accumulateReduce(blocks, 2), accumulateReduce(blocks, 3)};
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
        cudaFree(d_pool);  // All IPM vectors are sub-allocations of d_pool.
        if (cublas_handle) cublasDestroy(cublas_handle);
        if (cusparse_handle) cusparseDestroy(cusparse_handle);
        if (stream) cudaStreamDestroy(stream);
    }
};

// ---------------------------------------------------------------------------
// NormalEqSolver: cuDSS-based sparse Cholesky for normal equations
// ---------------------------------------------------------------------------

struct NormalEqSolver {
    static constexpr double kMaxAvgNeRowDensity = 64.0;

    // cuDSS handles.
    cudssHandle_t cudss_handle = nullptr;
    cudssConfig_t cudss_config = nullptr;
    cudssData_t cudss_data = nullptr;
    cudssMatrix_t cudss_mat = nullptr;
    cudssMatrix_t cudss_rhs_mat = nullptr;
    cudssMatrix_t cudss_sol_mat = nullptr;

    // NE pattern on device (lower triangle CSR).
    int ne_m = 0;
    int ne_nnz = 0;
    int* d_ne_row_starts = nullptr;
    int* d_ne_col_indices = nullptr;
    double* d_ne_values = nullptr;

    // Temp for IR.
    double* d_ir_residual = nullptr;
    double* d_ir_correction = nullptr;

    // Reference to parent context.
    GpuContext* ctx = nullptr;

    // Last regularization used in factorize().
    double last_reg_ = 0.0;
    double avg_ne_row_density_ = 0.0;

    bool analyzed = false;
    bool ok = false;
    std::string last_error;

    NormalEqSolver() = default;

    // Build NE sparsity pattern on CPU, upload, do cuDSS symbolic analysis.
    // Returns false if NE is too dense for cuDSS (caller falls back to CPU).
    bool analyze(const SparseMatrix& a, GpuContext* context) {
        ctx = context;
        ne_m = a.numRows();
        last_error.clear();

        auto fail = [&](const std::string& message) {
            last_error = message;
            return false;
        };
        auto failStatus = [&](const char* op, cudssStatus_t status) {
            return fail(std::string(op) + " failed (" + cudssStatusName(status) +
                        ")");
        };

        auto status = cudssCreate(&cudss_handle);
        if (!cudssOk(status)) return failStatus("cudssCreate", status);
        status = cudssSetStream(cudss_handle, ctx->stream);
        if (!cudssOk(status)) return failStatus("cudssSetStream", status);
        status = cudssConfigCreate(&cudss_config);
        if (!cudssOk(status)) return failStatus("cudssConfigCreate", status);
        status = cudssDataCreate(cudss_handle, &cudss_data);
        if (!cudssOk(status)) return failStatus("cudssDataCreate", status);

        // Compute NE pattern: lower triangle of A*A'.
        int m = ne_m;

        std::vector<std::vector<int>> ne_cols(m);
        for (int i = 0; i < m; ++i)
            ne_cols[i].push_back(i);

        auto csr_starts = a.csr_row_starts();
        auto csr_indices = a.csr_col_indices();

        int nc = a.numCols();
        std::vector<std::vector<int>> col_rows(nc);
        for (int i = 0; i < m; ++i) {
            for (int p = csr_starts[i]; p < csr_starts[i + 1]; ++p)
                col_rows[csr_indices[p]].push_back(i);
        }

        for (int k = 0; k < nc; ++k) {
            auto& rows = col_rows[k];
            for (size_t a_idx = 0; a_idx < rows.size(); ++a_idx) {
                for (size_t b_idx = 0; b_idx <= a_idx; ++b_idx) {
                    ne_cols[rows[a_idx]].push_back(rows[b_idx]);
                }
            }
        }

        for (int i = 0; i < m; ++i) {
            auto& row = ne_cols[i];
            std::sort(row.begin(), row.end());
            row.erase(std::unique(row.begin(), row.end()), row.end());
        }

        // Build CSR.
        std::vector<int> h_ne_row_starts(m + 1, 0);
        ne_nnz = 0;
        for (int i = 0; i < m; ++i) {
            h_ne_row_starts[i] = ne_nnz;
            ne_nnz += static_cast<int>(ne_cols[i].size());
        }
        h_ne_row_starts[m] = ne_nnz;

        // Skip GPU Cholesky if NE matrix is too dense — cuDSS can hang
        // on dense NE patterns due to excessive fill-in during factorization.
        avg_ne_row_density_ =
            static_cast<double>(ne_nnz) / std::max(m, 1);
        if (avg_ne_row_density_ > kMaxAvgNeRowDensity) {
            return fail("normal-equation pattern too dense for cuDSS (avg row "
                        "density " +
                        std::to_string(avg_ne_row_density_) + " > " +
                        std::to_string(kMaxAvgNeRowDensity) + ", rows=" +
                        std::to_string(m) + ", nnz=" +
                        std::to_string(ne_nnz) + ")");
        }

        std::vector<int> h_ne_col_indices(ne_nnz);
        int pos = 0;
        for (int i = 0; i < m; ++i) {
            for (int j : ne_cols[i])
                h_ne_col_indices[pos++] = j;
        }

        // Upload to GPU.
        if (!cudaOk(cudaMalloc(&d_ne_row_starts, sizeof(int) * (m + 1))))
            return fail("cudaMalloc d_ne_row_starts failed");
        if (!cudaOk(cudaMalloc(&d_ne_col_indices, sizeof(int) * ne_nnz)))
            return fail("cudaMalloc d_ne_col_indices failed");
        if (!cudaOk(cudaMalloc(&d_ne_values, sizeof(double) * ne_nnz)))
            return fail("cudaMalloc d_ne_values failed");
        if (!cudaOk(cudaMalloc(&d_ir_residual, sizeof(double) * m)))
            return fail("cudaMalloc d_ir_residual failed");
        if (!cudaOk(cudaMalloc(&d_ir_correction, sizeof(double) * m)))
            return fail("cudaMalloc d_ir_correction failed");

        if (!cudaOk(cudaMemcpyAsync(d_ne_row_starts, h_ne_row_starts.data(),
                                     sizeof(int) * (m + 1),
                                     cudaMemcpyHostToDevice, ctx->stream)))
            return fail("cudaMemcpyAsync d_ne_row_starts failed");
        if (!cudaOk(cudaMemcpyAsync(d_ne_col_indices, h_ne_col_indices.data(),
                                     sizeof(int) * ne_nnz,
                                     cudaMemcpyHostToDevice, ctx->stream)))
            return fail("cudaMemcpyAsync d_ne_col_indices failed");
        if (!cudaOk(cudaMemsetAsync(d_ne_values, 0, sizeof(double) * ne_nnz,
                                     ctx->stream)))
            return fail("cudaMemsetAsync d_ne_values failed");

        int hybrid_mode = 1;
        status = cudssConfigSet(cudss_config, CUDSS_CONFIG_HYBRID_MODE,
                                &hybrid_mode, sizeof(hybrid_mode));
        if (!cudssOk(status)) {
            return failStatus("cudssConfigSet(CUDSS_CONFIG_HYBRID_MODE)",
                              status);
        }

        // Create cuDSS sparse matrix (lower triangle, SPD).
        status = cudssMatrixCreateCsr(&cudss_mat, ne_m, ne_m, ne_nnz,
                                      d_ne_row_starts, nullptr,
                                      d_ne_col_indices, d_ne_values,
                                      CUDA_R_32I, CUDA_R_64F, CUDSS_MTYPE_SPD,
                                      CUDSS_MVIEW_LOWER, CUDSS_BASE_ZERO);
        if (!cudssOk(status)) return failStatus("cudssMatrixCreateCsr", status);

        status = cudssMatrixCreateDn(&cudss_rhs_mat, ne_m, 1, ne_m, ctx->d_rhs,
                                     CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR);
        if (!cudssOk(status))
            return failStatus("cudssMatrixCreateDn(rhs)", status);
        status = cudssMatrixCreateDn(&cudss_sol_mat, ne_m, 1, ne_m, ctx->d_dy,
                                     CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR);
        if (!cudssOk(status))
            return failStatus("cudssMatrixCreateDn(sol)", status);

        // Symbolic analysis.
        status = cudssExecute(cudss_handle, CUDSS_PHASE_ANALYSIS, cudss_config,
                              cudss_data, cudss_mat, cudss_sol_mat,
                              cudss_rhs_mat);
        if (!cudssOk(status))
            return failStatus("cudssExecute(analysis)", status);

        if (!cudaOk(cudaStreamSynchronize(ctx->stream)))
            return fail("cudaStreamSynchronize after analysis failed");

        analyzed = true;
        ok = true;
        return true;
    }

    // Form NE values on GPU and factorize.
    bool factorize(double reg) {
        if (!ok) return false;
        last_error.clear();
        last_reg_ = reg;

        // Form NE values via merge-join kernel.
        kernelFormNEValues<<<gridSize(ne_nnz), kBlockSize, 0, ctx->stream>>>(
            ne_nnz, d_ne_row_starts, d_ne_col_indices, ctx->d_a_row_starts,
            ctx->d_a_col_indices, ctx->d_a_values, ctx->d_theta, reg, ne_m,
            d_ne_values);

        // cuDSS refactorization (reuses symbolic structure).
        auto phase = analyzed ? CUDSS_PHASE_REFACTORIZATION
                              : CUDSS_PHASE_FACTORIZATION;
        auto status = cudssExecute(cudss_handle, phase, cudss_config,
                                    cudss_data, cudss_mat, cudss_sol_mat,
                                    cudss_rhs_mat);
        if (!cudssOk(status)) {
            // Try factorization from scratch if refactorization fails.
            status = cudssExecute(cudss_handle, CUDSS_PHASE_FACTORIZATION,
                                   cudss_config, cudss_data, cudss_mat,
                                   cudss_sol_mat, cudss_rhs_mat);
            if (!cudssOk(status)) {
                last_error = std::string("cudssExecute(factorization) failed (") +
                             cudssStatusName(status) + ")";
                return false;
            }
        }

        if (!cudaOk(cudaStreamSynchronize(ctx->stream))) {
            last_error = "cudaStreamSynchronize after factorization failed";
            return false;
        }

        return true;
    }

    // Solve (ADA' + reg*I) * dy = rhs.
    // d_rhs_in and d_sol_out are device pointers of size m.
    bool solve(double* d_rhs_in, double* d_sol_out) {
        if (!ok) return false;
        last_error.clear();

        // Update dense matrix pointers.
        if (cudss_rhs_mat) cudssMatrixDestroy(cudss_rhs_mat);
        if (cudss_sol_mat) cudssMatrixDestroy(cudss_sol_mat);
        cudss_rhs_mat = nullptr;
        cudss_sol_mat = nullptr;

        if (!cudssOk(cudssMatrixCreateDn(&cudss_rhs_mat, ne_m, 1, ne_m,
                                          d_rhs_in, CUDA_R_64F,
                                          CUDSS_LAYOUT_COL_MAJOR))) {
            last_error = "cudssMatrixCreateDn(rhs) failed";
            return false;
        }
        if (!cudssOk(cudssMatrixCreateDn(&cudss_sol_mat, ne_m, 1, ne_m,
                                          d_sol_out, CUDA_R_64F,
                                          CUDSS_LAYOUT_COL_MAJOR))) {
            last_error = "cudssMatrixCreateDn(sol) failed";
            return false;
        }

        auto status = cudssExecute(cudss_handle, CUDSS_PHASE_SOLVE, cudss_config,
                                   cudss_data, cudss_mat, cudss_sol_mat,
                                   cudss_rhs_mat);
        if (!cudssOk(status)) {
            last_error = std::string("cudssExecute(solve) failed (") +
                         cudssStatusName(status) + ")";
            return false;
        }
        if (!cudaOk(cudaStreamSynchronize(ctx->stream))) {
            last_error = "cudaStreamSynchronize after solve failed";
            return false;
        }
        return true;
    }

    // Solve with iterative refinement.
    // Computes NE matvec as: M*x = A*(theta .* (A'*x)) + reg*x
    // using the original A matrix and current theta values.
    bool solveRefined(double* d_rhs_in, double* d_sol_out, int ir_steps) {
        if (!solve(d_rhs_in, d_sol_out)) return false;
        if (ir_steps <= 0) return true;

        int m = ne_m;
        int n = ctx->n;

        for (int step = 0; step < ir_steps; ++step) {
            // Compute M * sol where M = A*diag(theta)*A' + reg*I.
            // Step 1: tmp_n = A' * sol (size n)
            ctx->multiplyAT(d_sol_out, ctx->d_tmp);

            // Step 2: tmp_n = theta .* tmp_n (element-wise multiply, in-place)
            kernelZS<<<gridSize(n), kBlockSize, 0, ctx->stream>>>(
                n, ctx->d_tmp, ctx->d_theta, ctx->d_tmp);

            // Step 3: ir_residual = A * tmp_n (size m)
            ctx->multiplyA(ctx->d_tmp, d_ir_residual);

            // Step 4: ir_residual += reg * sol
            double reg = last_reg_;
            cublasSetStream(ctx->cublas_handle, ctx->stream);
            cublasDaxpy(ctx->cublas_handle, m, &reg, d_sol_out, 1,
                        d_ir_residual, 1);

            // Step 5: ir_residual = rhs - M*sol (residual)
            kernelResidualIR<<<gridSize(m), kBlockSize, 0, ctx->stream>>>(
                m, d_rhs_in, d_ir_residual, d_ir_residual);

            // Step 6: solve M * correction = residual
            if (!solve(d_ir_residual, d_ir_correction)) return false;

            // Step 7: sol += correction
            kernelAddCorrection<<<gridSize(m), kBlockSize, 0, ctx->stream>>>(
                m, d_ir_correction, d_sol_out);
        }
        return true;
    }

    const std::string& error() const { return last_error; }

    ~NormalEqSolver() {
        if (cudss_rhs_mat) cudssMatrixDestroy(cudss_rhs_mat);
        if (cudss_sol_mat) cudssMatrixDestroy(cudss_sol_mat);
        if (cudss_mat) cudssMatrixDestroy(cudss_mat);
        if (cudss_data) {
            // Data must be destroyed before handle.
            cudssDataDestroy(cudss_handle, cudss_data);
        }
        if (cudss_config) cudssConfigDestroy(cudss_config);
        if (cudss_handle) cudssDestroy(cudss_handle);
        cudaFree(d_ne_row_starts);
        cudaFree(d_ne_col_indices);
        cudaFree(d_ne_values);
        cudaFree(d_ir_residual);
        cudaFree(d_ir_correction);
    }
};

// ---------------------------------------------------------------------------
// GpuBarrierImpl: the full GPU IPM implementation
// ---------------------------------------------------------------------------

struct GpuBarrierImpl {
    GpuContext ctx;
    NormalEqSolver ne_solver;

    bool initialized = false;
    std::string last_error;

    bool init(const SparseMatrix& a, std::span<const double> b,
              std::span<const double> c) {
        if (!ctx.init(a, b, c)) {
            last_error = "Barrier GPU context initialization failed.";
            return false;
        }
        if (!ne_solver.analyze(a, &ctx)) {
            last_error = ne_solver.error().empty()
                             ? "Barrier GPU normal-equation analysis failed."
                             : "Barrier GPU " + ne_solver.error() + ".";
            return false;
        }
        initialized = true;
        return true;
    }

    // Compute Mehrotra starting point on GPU.
    bool computeStartingPoint() {
        int n = ctx.n;

        // Set theta = ones for starting point.
        kernelFillOnes<<<gridSize(n), kBlockSize, 0, ctx.stream>>>(n,
                                                                    ctx.d_theta);

        // Factorize A*I*A' + reg*I = A*A' + reg*I.
        double reg = 1e-8;
        if (!ne_solver.factorize(reg)) {
            last_error = ne_solver.error().empty()
                             ? "Barrier GPU starting-point factorization failed."
                             : "Barrier GPU starting-point factorization failed: " +
                                   ne_solver.error() + ".";
            return false;
        }

        // Solve (A*A')*y0 = b.
        if (!ne_solver.solve(ctx.d_b, ctx.d_y)) {
            last_error = ne_solver.error().empty()
                             ? "Barrier GPU starting-point solve failed."
                             : "Barrier GPU starting-point solve failed: " +
                                   ne_solver.error() + ".";
            return false;
        }

        // z0 = A' * y0.
        if (!ctx.multiplyAT(ctx.d_y, ctx.d_z)) {
            last_error =
                "Barrier GPU starting-point transpose multiply failed.";
            return false;
        }

        // s0 = c - A' * y0 = c - z0.
        kernelResidualDual<<<gridSize(n), kBlockSize, 0, ctx.stream>>>(
            n, ctx.d_c, ctx.d_z, ctx.d_s,
            ctx.d_s);  // Not right — need s = c - z, not c - z - s.

        // Actually: s0[j] = c[j] - z0[j]. Use a custom approach.
        // d_s = d_c (copy), then d_s -= d_z.
        cudaMemcpyAsync(ctx.d_s, ctx.d_c, sizeof(double) * n,
                        cudaMemcpyDeviceToDevice, ctx.stream);
        // d_s = d_s - d_z: use cublas daxpy.
        double neg_one = -1.0;
        cublasSetStream(ctx.cublas_handle, ctx.stream);
        cublasDaxpy(ctx.cublas_handle, n, &neg_one, ctx.d_z, 1, ctx.d_s, 1);

        // Shift to positivity.
        double min_z = ctx.gpuMin(ctx.d_z, n);
        double min_s = ctx.gpuMin(ctx.d_s, n);

        double delta_z = std::max(-1.5 * min_z, 0.0);
        double delta_s = std::max(-1.5 * min_s, 0.0);

        // z_hat = z + delta_z, s_hat = s + delta_s.
        // Then dp = 0.5 * dot(z_hat, s_hat) / sum(s_hat).
        // dd = 0.5 * dot(z_hat, s_hat) / sum(z_hat).
        // Shift z_hat and s_hat.
        double dz_val = delta_z;
        double ds_val = delta_s;

        // Add shifts.
        kernelInit<<<gridSize(n), kBlockSize, 0, ctx.stream>>>(n, dz_val,
                                                                ctx.d_tmp);
        double one = 1.0;
        cublasDaxpy(ctx.cublas_handle, n, &one, ctx.d_tmp, 1, ctx.d_z, 1);

        kernelInit<<<gridSize(n), kBlockSize, 0, ctx.stream>>>(n, ds_val,
                                                                ctx.d_tmp);
        cublasDaxpy(ctx.cublas_handle, n, &one, ctx.d_tmp, 1, ctx.d_s, 1);

        // Compute dp and dd.
        double zs_dot = ctx.gpuDot(ctx.d_z, ctx.d_s, n);
        double z_sum = ctx.gpuSum(ctx.d_z, n);
        double s_sum = ctx.gpuSum(ctx.d_s, n);

        double dp = 0.5 * zs_dot / std::max(s_sum, 1e-30);
        double dd = 0.5 * zs_dot / std::max(z_sum, 1e-30);

        // Final: z += dp, s += dd, clamp to 1e-4.
        kernelShiftPositive<<<gridSize(n), kBlockSize, 0, ctx.stream>>>(
            n, 0.0, dd, dp, ctx.d_z, ctx.d_s);

        cudaStreamSynchronize(ctx.stream);
        return true;
    }

    // Run the full IPM loop on GPU. Returns true if converged.
    bool solve(const BarrierOptions& opts, Real std_obj_offset,
               std::span<const Real> beq, std::span<const Real> cstd,
               std::vector<Real>& z_out, std::vector<Real>& y_out,
               std::vector<Real>& s_out, Int& iters) {
        int m = ctx.m, n = ctx.n;

        // Compute starting point.
        if (!computeStartingPoint()) return false;

        const double base_reg = std::max(opts.regularization, 1e-12);
        const double inv_b = 1.0 / (1.0 + [&] {
            double mx = 0.0;
            for (auto v : beq) mx = std::max(mx, std::abs(v));
            return mx;
        }());
        const double inv_c = 1.0 / (1.0 + [&] {
            double mx = 0.0;
            for (auto v : cstd) mx = std::max(mx, std::abs(v));
            return mx;
        }());

        // Stall detection state.
        double prev_pinf = 1e30, prev_dinf = 1e30, prev_gap = 1e30;
        int stagnant_iters = 0;
        double reg_boost = 1.0;  // multiplicative boost on regularization
        for (int iter = 0; iter < opts.max_iter; ++iter) {
            if (opts.stop_flag != nullptr &&
                opts.stop_flag->load(std::memory_order_relaxed)) {
                iters = iter;
                return false;
            }

            // Residuals.
            ctx.multiplyA(ctx.d_z, ctx.d_az);
            ctx.multiplyAT(ctx.d_y, ctx.d_aty);

            kernelResidualPrimal<<<gridSize(m), kBlockSize, 0, ctx.stream>>>(
                m, ctx.d_b, ctx.d_az, ctx.d_rp);
            kernelResidualDual<<<gridSize(n), kBlockSize, 0, ctx.stream>>>(
                n, ctx.d_c, ctx.d_aty, ctx.d_s, ctx.d_rd);

            // Convergence check.
            double mu = ctx.gpuMu(ctx.d_z, ctx.d_s, n);
            double complementarity = std::abs(mu) * std::max(n, 1);
            double pinf = ctx.gpuInfNorm(ctx.d_rp, m) * inv_b;
            double dinf = ctx.gpuInfNorm(ctx.d_rd, n) * inv_c;
            double pobj = std_obj_offset + ctx.gpuDot(ctx.d_c, ctx.d_z, n);
            double gap = complementarity / (1.0 + std::abs(pobj));

            if (opts.verbose) {
                std::printf(
                    "IPM %4d  pobj=% .10e  pinf=% .2e  dinf=% .2e  "
                    "gap=% .2e  reg=% .1e [gpu-chol]\n",
                    iter, pobj, pinf, dinf, gap, base_reg * reg_boost);
            }

            if (pinf < opts.primal_dual_tol && dinf < opts.primal_dual_tol &&
                gap < opts.primal_dual_tol) {
                iters = iter;
                downloadSolution(z_out, y_out, s_out);
                return true;
            }

            // --- Stall detection ---
            // Detect catastrophic numerical failure: residuals jump >100x.
            if (iter > 0 && (pinf > 100.0 * prev_pinf + 1e-10 ||
                             dinf > 100.0 * prev_dinf + 1e-10)) {
                if (opts.verbose) {
                    std::printf("IPM: numerical instability at iter %d "
                                "(pinf %.2e -> %.2e, dinf %.2e -> %.2e)\n",
                                iter, prev_pinf, pinf, prev_dinf, dinf);
                }
                // Restore backup (last good iterate) and check if it's
                // close enough to declare convergence with relaxed tol.
                cudaMemcpyAsync(ctx.d_z, ctx.d_z_bak, sizeof(double) * n,
                                cudaMemcpyDeviceToDevice, ctx.stream);
                cudaMemcpyAsync(ctx.d_y, ctx.d_y_bak, sizeof(double) * m,
                                cudaMemcpyDeviceToDevice, ctx.stream);
                cudaMemcpyAsync(ctx.d_s, ctx.d_s_bak, sizeof(double) * n,
                                cudaMemcpyDeviceToDevice, ctx.stream);
                cudaStreamSynchronize(ctx.stream);

                // Accept the restored iterate if it is already close enough.
                double relaxed_tol = opts.primal_dual_tol * 50.0;
                if (prev_pinf < relaxed_tol && prev_dinf < relaxed_tol &&
                    prev_gap < relaxed_tol) {
                    if (opts.verbose) {
                        std::printf("IPM: restored last good iterate "
                                    "(pinf=%.2e dinf=%.2e gap=%.2e)\n",
                                    prev_pinf, prev_dinf, prev_gap);
                    }
                    iters = iter;
                    downloadSolution(z_out, y_out, s_out);
                    return true;
                }
                iters = iter;
                char msg[192];
                std::snprintf(
                    msg, sizeof(msg),
                    "Barrier GPU numerical instability at iter %d "
                    "(pinf %.2e -> %.2e, dinf %.2e -> %.2e).",
                    iter, prev_pinf, pinf, prev_dinf, dinf);
                last_error = msg;
                return false;
            }

            // Detect stagnation: gap not improving.
            if (gap >= 0.99 * prev_gap && iter > 5) {
                stagnant_iters++;
                if (stagnant_iters >= 10) {
                    if (opts.verbose) {
                        std::printf("IPM: stagnation detected (%d iters "
                                    "without progress), giving up\n",
                                    stagnant_iters);
                    }
                    iters = iter;
                    char msg[160];
                    std::snprintf(
                        msg, sizeof(msg),
                        "Barrier GPU stagnation after %d consecutive iterations "
                        "without gap improvement.",
                        stagnant_iters);
                    last_error = msg;
                    return false;
                }
            } else {
                stagnant_iters = 0;
            }

            prev_pinf = pinf;
            prev_dinf = dinf;
            prev_gap = gap;

            // Theta = z / s.
            if (iter == 0 && opts.verbose) std::fprintf(stderr, "GPU: iter0 factorize...\n");
            kernelTheta<<<gridSize(n), kBlockSize, 0, ctx.stream>>>(
                n, ctx.d_z, ctx.d_s, ctx.d_theta);

            // Factorize normal equations with regularization.
            double cur_reg = base_reg * reg_boost;
            if (gap < 1e-4) {
                cur_reg *= 100.0;
                if (gap < 5e-5) {
                    cur_reg *= 10.0;
                }
            }
            bool factorized = false;
            for (int attempt = 0; attempt < 3; ++attempt) {
                if (ne_solver.factorize(cur_reg)) {
                    factorized = true;
                    break;
                }
                cur_reg *= 10.0;
                reg_boost *= 10.0;
            }
            if (!factorized) {
                iters = iter;
                char msg[256];
                std::snprintf(
                    msg, sizeof(msg),
                    "Barrier GPU factorization failed at iter %d after regularization retries%s%s.",
                    iter, ne_solver.error().empty() ? "" : ": ",
                    ne_solver.error().c_str());
                last_error = msg;
                return false;
            }

            // --- Predictor step ---
            kernelComplementarityAffine<<<gridSize(n), kBlockSize, 0,
                                          ctx.stream>>>(n, ctx.d_z, ctx.d_s,
                                                         ctx.d_rc);

            if (!solveNewtonStep(ctx.d_rp, ctx.d_rd, ctx.d_rc, ctx.d_dz_aff,
                                 ctx.d_dy_aff, ctx.d_ds_aff, opts.ir_steps)) {
                char msg[256];
                std::snprintf(
                    msg, sizeof(msg),
                    "Barrier GPU predictor Newton solve failed at iter %d%s%s.",
                    iter, ne_solver.error().empty() ? "" : ": ",
                    ne_solver.error().c_str());
                last_error = msg;
                return false;
            }

            double alpha_aff_p =
                ctx.gpuMaxStep(ctx.d_z, ctx.d_dz_aff, n, 1.0);
            double alpha_aff_d =
                ctx.gpuMaxStep(ctx.d_s, ctx.d_ds_aff, n, 1.0);

            // mu_aff = mu(z+ap*dz, s+ad*ds) via dot product decomposition.
            auto [zs, z_ds, dz_s, dz_ds] = ctx.gpuDot4(
                ctx.d_z, ctx.d_s, ctx.d_z, ctx.d_ds_aff, ctx.d_dz_aff,
                ctx.d_s, ctx.d_dz_aff, ctx.d_ds_aff, n);
            double mu_aff =
                (zs + alpha_aff_d * z_ds + alpha_aff_p * dz_s +
                 alpha_aff_p * alpha_aff_d * dz_ds) /
                std::max(n, 1);

            double sigma = std::pow(
                std::max(mu_aff, 0.0) / std::max(mu, 1e-30), 3.0);
            sigma = std::clamp(sigma, 0.0, 1.0);

            // --- Corrector step ---
            double sigma_mu = sigma * mu;
            kernelComplementarityCorrected<<<gridSize(n), kBlockSize, 0,
                                             ctx.stream>>>(
                n, sigma_mu, ctx.d_z, ctx.d_s, ctx.d_dz_aff, ctx.d_ds_aff,
                ctx.d_rc);

            if (!solveNewtonStep(ctx.d_rp, ctx.d_rd, ctx.d_rc, ctx.d_dz,
                                 ctx.d_dy, ctx.d_ds, opts.ir_steps)) {
                char msg[256];
                std::snprintf(
                    msg, sizeof(msg),
                    "Barrier GPU corrector Newton solve failed at iter %d%s%s.",
                    iter, ne_solver.error().empty() ? "" : ": ",
                    ne_solver.error().c_str());
                last_error = msg;
                return false;
            }

            double alpha_p =
                ctx.gpuMaxStep(ctx.d_z, ctx.d_dz, n, opts.step_fraction);
            double alpha_d =
                ctx.gpuMaxStep(ctx.d_s, ctx.d_ds, n, opts.step_fraction);

            // Save current iterate as backup before updating.
            cudaMemcpyAsync(ctx.d_z_bak, ctx.d_z, sizeof(double) * n,
                            cudaMemcpyDeviceToDevice, ctx.stream);
            cudaMemcpyAsync(ctx.d_y_bak, ctx.d_y, sizeof(double) * m,
                            cudaMemcpyDeviceToDevice, ctx.stream);
            cudaMemcpyAsync(ctx.d_s_bak, ctx.d_s, sizeof(double) * n,
                            cudaMemcpyDeviceToDevice, ctx.stream);

            // Update iterates.
            kernelUpdateIterates<<<gridSize(std::max(m, n)), kBlockSize, 0,
                                    ctx.stream>>>(
                n, m, alpha_p, alpha_d, ctx.d_dz, ctx.d_ds, ctx.d_dy, ctx.d_z,
                ctx.d_s, ctx.d_y, 1e-12);
        }

        iters = opts.max_iter;
        last_error = "Barrier GPU reached maximum IPM iterations.";
        return false;
    }

    // Solve Newton step: given rp, rd, rc, compute dz, dy, ds.
    bool solveNewtonStep(double* d_rp, double* d_rd, double* d_rc,
                         double* d_dz_out, double* d_dy_out,
                         double* d_ds_out, int ir_steps) {
        int m = ctx.m, n = ctx.n;

        // h = rc/s - theta*rd.
        kernelScaledRhsH<<<gridSize(n), kBlockSize, 0, ctx.stream>>>(
            n, d_rc, ctx.d_s, ctx.d_theta, d_rd, ctx.d_h);

        // ah = A * h.
        ctx.multiplyA(ctx.d_h, ctx.d_ah);

        // rhs = rp - ah.
        kernelNormalEqRhs<<<gridSize(m), kBlockSize, 0, ctx.stream>>>(
            m, d_rp, ctx.d_ah, ctx.d_rhs);

        // Solve NE system for dy with caller-configured iterative refinement.
        if (!ne_solver.solveRefined(ctx.d_rhs, d_dy_out, ir_steps))
            return false;

        // atdy = A' * dy.
        ctx.multiplyAT(d_dy_out, ctx.d_atdy);

        // dz = h + theta * atdy.
        kernelSearchDirectionZ<<<gridSize(n), kBlockSize, 0, ctx.stream>>>(
            n, ctx.d_h, ctx.d_theta, ctx.d_atdy, d_dz_out);

        // ds = rd - atdy.
        kernelSearchDirectionS<<<gridSize(n), kBlockSize, 0, ctx.stream>>>(
            n, d_rd, ctx.d_atdy, d_ds_out);

        return true;
    }

    void downloadSolution(std::vector<Real>& z, std::vector<Real>& y,
                          std::vector<Real>& s) {
        int m = ctx.m, n = ctx.n;
        z.resize(n);
        y.resize(m);
        s.resize(n);
        cudaMemcpyAsync(z.data(), ctx.d_z, sizeof(double) * n,
                        cudaMemcpyDeviceToHost, ctx.stream);
        cudaMemcpyAsync(y.data(), ctx.d_y, sizeof(double) * m,
                        cudaMemcpyDeviceToHost, ctx.stream);
        cudaMemcpyAsync(s.data(), ctx.d_s, sizeof(double) * n,
                        cudaMemcpyDeviceToHost, ctx.stream);
        cudaStreamSynchronize(ctx.stream);
    }
};

// ---------------------------------------------------------------------------
// Bridge function called from barrier.cpp
// ---------------------------------------------------------------------------

bool gpuBarrierSolve(const SparseMatrix& aeq, std::span<const Real> beq,
                     std::span<const Real> cstd,
                     const BarrierOptions& opts, Real std_obj_offset,
                     std::vector<Real>& z, std::vector<Real>& y,
                     std::vector<Real>& s, Int& iters,
                     bool& gpu_initialized, std::string& error_msg) {
    gpu_initialized = false;
    GpuBarrierImpl impl;
    if (!impl.init(aeq, beq, cstd)) {
        error_msg = impl.last_error;
        return false;
    }
    gpu_initialized = true;
    bool ok = impl.solve(opts, std_obj_offset, beq, cstd, z, y, s, iters);
    if (!ok) error_msg = impl.last_error;
    return ok;
}

}  // namespace detail
}  // namespace mipx
