// GPU device-resident barrier solver with:
//   - NE+Cholesky and Augmented+LDL' backends
//   - Mixed precision (FP32 factorize + FP64 iterative refinement)
//   - Auto-switching NE↔Augmented on factorize failure
//   - Iterate backup/rollback and adaptive regularization
//
// Only compiled when MIPX_USE_CUDA=ON and cuDSS is found.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <vector>

#include <cuda_runtime.h>
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

static bool cudssOk(cudssStatus_t e) { return e == CUDSS_STATUS_SUCCESS; }

// ---------------------------------------------------------------------------
// CUDA kernels — block size 256, grid-stride loops
// ---------------------------------------------------------------------------

static constexpr int kBlockSize = 256;

static __host__ int gridSize(int n) {
    return std::min((n + kBlockSize - 1) / kBlockSize, 65535);
}

// --- Residual / iterate kernels ---

// Fused residual + inf-norm: computes rp[i] = b[i] - az[i] and reduces
// ||rp||_inf into *norm_out via atomic max (warp-shuffle final reduction).
__global__ void kernelResidualPrimalNorm(int m, const double* b, const double* az,
                                         double* rp, double* norm_out) {
    __shared__ double sdata[kBlockSize];
    double mx = 0.0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < m;
         i += gridDim.x * blockDim.x) {
        double v = b[i] - az[i];
        rp[i] = v;
        mx = fmax(mx, fabs(v));
    }
    sdata[threadIdx.x] = mx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] = fmax(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x < 32) {
        double val = fmax(sdata[threadIdx.x], sdata[threadIdx.x + 32]);
        for (int offset = 16; offset > 0; offset >>= 1)
            val = fmax(val, __shfl_down_sync(0xffffffff, val, offset));
        if (threadIdx.x == 0) {
            unsigned long long int* addr = (unsigned long long int*)norm_out;
            unsigned long long int old = *addr;
            unsigned long long int assumed;
            do {
                assumed = old;
                double old_val = __longlong_as_double(assumed);
                double new_val = fmax(old_val, val);
                old = atomicCAS(addr, assumed, __double_as_longlong(new_val));
            } while (assumed != old);
        }
    }
}

// Fused residual + inf-norm: computes rd[j] = c[j] - aty[j] - s[j] and
// reduces ||rd||_inf into *norm_out via atomic max.
__global__ void kernelResidualDualNorm(int n, const double* c, const double* aty,
                                       const double* s, double* rd,
                                       double* norm_out) {
    __shared__ double sdata[kBlockSize];
    double mx = 0.0;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n;
         j += gridDim.x * blockDim.x) {
        double v = c[j] - aty[j] - s[j];
        rd[j] = v;
        mx = fmax(mx, fabs(v));
    }
    sdata[threadIdx.x] = mx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] = fmax(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x < 32) {
        double val = fmax(sdata[threadIdx.x], sdata[threadIdx.x + 32]);
        for (int offset = 16; offset > 0; offset >>= 1)
            val = fmax(val, __shfl_down_sync(0xffffffff, val, offset));
        if (threadIdx.x == 0) {
            unsigned long long int* addr = (unsigned long long int*)norm_out;
            unsigned long long int old = *addr;
            unsigned long long int assumed;
            do {
                assumed = old;
                double old_val = __longlong_as_double(assumed);
                double new_val = fmax(old_val, val);
                old = atomicCAS(addr, assumed, __double_as_longlong(new_val));
            } while (assumed != old);
        }
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

// --- Normal equations kernels ---

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

// --- Augmented system kernels ---

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

// Augmented IR residual z-component:
// r_z[j] = rhs_z[j] - (-s[j]/z[j] - reg_p) * sol_z[j] - aty[j]
__global__ void kernelAugIRResidualZ(int n, const double* rhs_z,
                                      const double* sol_z, const double* aty,
                                      const double* s, const double* z,
                                      double reg_p, double* r_z) {
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n;
         j += gridDim.x * blockDim.x) {
        double diag = -(s[j] / fmax(z[j], 1e-20)) - reg_p;
        r_z[j] = rhs_z[j] - diag * sol_z[j] - aty[j];
    }
}

// Augmented IR residual y-component:
// r_y[i] = rhs_y[i] - az[i] - reg_d * sol_y[i]
__global__ void kernelAugIRResidualY(int m, const double* rhs_y,
                                      const double* az, const double* sol_y,
                                      double reg_d, double* r_y) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < m;
         i += gridDim.x * blockDim.x) {
        r_y[i] = rhs_y[i] - az[i] - reg_d * sol_y[i];
    }
}

// --- IR and correction kernels ---

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

// --- Reduction kernels ---

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
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x < 32) {
        double val = sdata[threadIdx.x] + sdata[threadIdx.x + 32];
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
        if (threadIdx.x == 0) atomicAdd(result, val);
    }
}

__global__ void kernelInfNorm(int n, const double* v, double* result) {
    __shared__ double sdata[kBlockSize];
    double mx = 0.0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        mx = fmax(mx, fabs(v[i]));
    }
    sdata[threadIdx.x] = mx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] = fmax(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x < 32) {
        double val = fmax(sdata[threadIdx.x], sdata[threadIdx.x + 32]);
        for (int offset = 16; offset > 0; offset >>= 1)
            val = fmax(val, __shfl_down_sync(0xffffffff, val, offset));
        if (threadIdx.x == 0) {
            unsigned long long int* addr = (unsigned long long int*)result;
            unsigned long long int old = *addr;
            unsigned long long int assumed;
            do {
                assumed = old;
                double old_val = __longlong_as_double(assumed);
                double new_val = fmax(old_val, val);
                old = atomicCAS(addr, assumed, __double_as_longlong(new_val));
            } while (assumed != old);
        }
    }
}

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
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] = fmin(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x < 32) {
        double val = fmin(sdata[threadIdx.x], sdata[threadIdx.x + 32]);
        for (int offset = 16; offset > 0; offset >>= 1)
            val = fmin(val, __shfl_down_sync(0xffffffff, val, offset));
        if (threadIdx.x == 0) {
            unsigned long long int* addr = (unsigned long long int*)result;
            unsigned long long int old = *addr;
            unsigned long long int assumed;
            do {
                assumed = old;
                double old_val = __longlong_as_double(assumed);
                double new_val = fmin(old_val, val);
                old = atomicCAS(addr, assumed, __double_as_longlong(new_val));
            } while (assumed != old);
        }
    }
}

__global__ void kernelSum(int n, const double* v, double* result) {
    __shared__ double sdata[kBlockSize];
    double sum = 0.0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        sum += v[i];
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x < 32) {
        double val = sdata[threadIdx.x] + sdata[threadIdx.x + 32];
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
        if (threadIdx.x == 0) atomicAdd(result, val);
    }
}

__global__ void kernelMin(int n, const double* v, double* result) {
    __shared__ double sdata[kBlockSize];
    double mn = 1e30;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        mn = fmin(mn, v[i]);
    }
    sdata[threadIdx.x] = mn;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] = fmin(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x < 32) {
        double val = fmin(sdata[threadIdx.x], sdata[threadIdx.x + 32]);
        for (int offset = 16; offset > 0; offset >>= 1)
            val = fmin(val, __shfl_down_sync(0xffffffff, val, offset));
        if (threadIdx.x == 0) {
            unsigned long long int* addr = (unsigned long long int*)result;
            unsigned long long int old = *addr;
            unsigned long long int assumed;
            do {
                assumed = old;
                double old_val = __longlong_as_double(assumed);
                double new_val = fmin(old_val, val);
                old = atomicCAS(addr, assumed, __double_as_longlong(new_val));
            } while (assumed != old);
        }
    }
}

__global__ void kernelZS(int n, const double* z, const double* s, double* out) {
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n;
         j += gridDim.x * blockDim.x) {
        out[j] = z[j] * s[j];
    }
}

// Fused 4-component dot product for mu_aff computation.
// out4[0] = sum(z*s), out4[1] = sum(z*ds), out4[2] = sum(dz*s), out4[3] = sum(dz*ds)
__global__ void kernelMuAffParts(int n, const double* z, const double* s,
                                  const double* dz, const double* ds,
                                  double* out4) {
    __shared__ double sdata[4][kBlockSize];
    double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        double zi = z[i], si = s[i], dzi = dz[i], dsi = ds[i];
        s0 += zi * si;
        s1 += zi * dsi;
        s2 += dzi * si;
        s3 += dzi * dsi;
    }
    sdata[0][threadIdx.x] = s0;
    sdata[1][threadIdx.x] = s1;
    sdata[2][threadIdx.x] = s2;
    sdata[3][threadIdx.x] = s3;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[0][threadIdx.x] += sdata[0][threadIdx.x + stride];
            sdata[1][threadIdx.x] += sdata[1][threadIdx.x + stride];
            sdata[2][threadIdx.x] += sdata[2][threadIdx.x + stride];
            sdata[3][threadIdx.x] += sdata[3][threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x < 32) {
        double v0 = sdata[0][threadIdx.x] + sdata[0][threadIdx.x + 32];
        double v1 = sdata[1][threadIdx.x] + sdata[1][threadIdx.x + 32];
        double v2 = sdata[2][threadIdx.x] + sdata[2][threadIdx.x + 32];
        double v3 = sdata[3][threadIdx.x] + sdata[3][threadIdx.x + 32];
        for (int offset = 16; offset > 0; offset >>= 1) {
            v0 += __shfl_down_sync(0xffffffff, v0, offset);
            v1 += __shfl_down_sync(0xffffffff, v1, offset);
            v2 += __shfl_down_sync(0xffffffff, v2, offset);
            v3 += __shfl_down_sync(0xffffffff, v3, offset);
        }
        if (threadIdx.x == 0) {
            atomicAdd(out4 + 0, v0);
            atomicAdd(out4 + 1, v1);
            atomicAdd(out4 + 2, v2);
            atomicAdd(out4 + 3, v3);
        }
    }
}

// --- Vector arithmetic kernels (replacing cublasDaxpy for small ops) ---

// y[i] += alpha * x[i]
__global__ void kernelAxpy(int n, double alpha, const double* x, double* y) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        y[i] += alpha * x[i];
    }
}

// y[i] = x[i] - z[i]
__global__ void kernelVecSub(int n, const double* x, const double* z, double* y) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        y[i] = x[i] - z[i];
    }
}

// y[i] += scalar
__global__ void kernelAddScalar(int n, double scalar, double* y) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        y[i] += scalar;
    }
}

// --- Mixed precision kernels ---

__global__ void kernelCastF64toF32(int n, const double* src, float* dst) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        dst[i] = __double2float_rn(src[i]);
    }
}

__global__ void kernelCastF32toF64(int n, const float* src, double* dst) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        dst[i] = (double)src[i];
    }
}

// ---------------------------------------------------------------------------
// GpuContext: shared CUDA resources for device-resident IPM
// ---------------------------------------------------------------------------

struct GpuContext {
    int m = 0, n = 0, nnz = 0;
    cudaStream_t stream = nullptr;
    cusparseHandle_t cusparse_handle = nullptr;

    // A in CSR on device.
    int* d_a_row_starts = nullptr;
    int* d_a_col_indices = nullptr;
    double* d_a_values = nullptr;
    cusparseSpMatDescr_t a_mat = nullptr;

    // Pooled GPU memory.
    double* d_pool = nullptr;

    // IPM iterate vectors (device-resident).
    double* d_z = nullptr;
    double* d_y = nullptr;
    double* d_s = nullptr;
    double* d_b = nullptr;
    double* d_c = nullptr;

    // Backup vectors for rollback.
    double* d_z_bak = nullptr;
    double* d_y_bak = nullptr;
    double* d_s_bak = nullptr;

    // SpMV result vectors.
    double* d_az = nullptr;
    double* d_aty = nullptr;

    // Residual vectors.
    double* d_rp = nullptr;
    double* d_rd = nullptr;
    double* d_rc = nullptr;

    // NE-specific work vectors.
    double* d_theta = nullptr;
    double* d_h = nullptr;
    double* d_ah = nullptr;
    double* d_rhs = nullptr;
    double* d_atdy = nullptr;

    // Search directions.
    double* d_dz = nullptr;
    double* d_dy = nullptr;
    double* d_ds = nullptr;

    // Predictor (affine) directions.
    double* d_dz_aff = nullptr;
    double* d_dy_aff = nullptr;
    double* d_ds_aff = nullptr;

    // Augmented system work vectors.
    double* d_aug_rhs = nullptr;
    double* d_aug_sol = nullptr;
    double* d_aug_ir_rhs = nullptr;  // IR scratch for augmented system (lazy alloc)
    double* d_aug_ir_sol = nullptr;  // IR scratch for augmented system (lazy alloc)
    double* d_aug_ir_pool = nullptr; // Separate allocation for aug IR buffers

    // IR scratch (m-sized, shared with NE/Aug solvers).
    double* d_ir_residual = nullptr;
    double* d_ir_correction = nullptr;

    // Scratch.
    double* d_scalar = nullptr;   // 16 scalars on device
    double* d_tmp = nullptr;      // max(m,n)-sized temp

    // cuSPARSE descriptors.
    cusparseDnVecDescr_t vec_n1 = nullptr, vec_n2 = nullptr;
    cusparseDnVecDescr_t vec_m1 = nullptr, vec_m2 = nullptr;
    void* spmv_buffer_n = nullptr;
    void* spmv_buffer_t = nullptr;

    bool init(int m_in, int n_in, int nnz_in,
              const int* h_row_starts, const int* h_col_indices,
              const double* h_values,
              const double* h_b, const double* h_c) {
        m = m_in;
        n = n_in;
        nnz = nnz_in;

        if (!cudaOk(cudaStreamCreate(&stream))) return false;
        if (!cusparseOk(cusparseCreate(&cusparse_handle))) return false;
        cusparseSetStream(cusparse_handle, stream);

        // Upload A.
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

        // Allocate pool: n-sized * 15 + m-sized * 11 + max(m,n) + 16 + 2*(n+m)
        // Augmented IR buffers (2*(n+m)) are allocated lazily on first use.
        int mn = std::max(m, n);
        size_t pool_size = static_cast<size_t>(15 * n + 11 * m + mn + 16 + 2 * (n + m));
        if (!cudaOk(cudaMalloc(&d_pool, sizeof(double) * pool_size))) return false;

        double* p = d_pool;
        auto take_n = [&]() { double* r = p; p += n; return r; };
        auto take_m = [&]() { double* r = p; p += m; return r; };
        auto take_nm = [&]() { double* r = p; p += (n + m); return r; };

        // n-sized (15)
        d_z = take_n(); d_s = take_n(); d_c = take_n();
        d_z_bak = take_n(); d_s_bak = take_n();
        d_aty = take_n(); d_rd = take_n(); d_rc = take_n();
        d_theta = take_n(); d_h = take_n(); d_atdy = take_n();
        d_dz = take_n(); d_ds = take_n();
        d_dz_aff = take_n(); d_ds_aff = take_n();

        // m-sized (11): includes IR scratch vectors
        d_y = take_m(); d_b = take_m(); d_y_bak = take_m();
        d_az = take_m(); d_rp = take_m(); d_ah = take_m();
        d_rhs = take_m(); d_dy = take_m(); d_dy_aff = take_m();
        d_ir_residual = take_m();
        d_ir_correction = take_m();

        d_tmp = p; p += mn;
        d_scalar = p; p += 16;

        d_aug_rhs = take_nm();
        d_aug_sol = take_nm();

        // Upload b, c.
        cudaMemcpyAsync(d_b, h_b, sizeof(double) * m,
                        cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_c, h_c, sizeof(double) * n,
                        cudaMemcpyHostToDevice, stream);

        // Dense vector descriptors for SpMV.
        if (!cusparseOk(cusparseCreateDnVec(&vec_n1, n, d_z, CUDA_R_64F))) return false;
        if (!cusparseOk(cusparseCreateDnVec(&vec_n2, n, d_aty, CUDA_R_64F))) return false;
        if (!cusparseOk(cusparseCreateDnVec(&vec_m1, m, d_az, CUDA_R_64F))) return false;
        if (!cusparseOk(cusparseCreateDnVec(&vec_m2, m, d_y, CUDA_R_64F))) return false;

        size_t buf_size_n = 0, buf_size_t = 0;
        double alpha = 1.0, beta = 0.0;
        cusparseSpMV_bufferSize(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, a_mat, vec_n1, &beta, vec_m1,
                                CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &buf_size_n);
        cusparseSpMV_bufferSize(cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
                                &alpha, a_mat, vec_m2, &beta, vec_n2,
                                CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &buf_size_t);
        buf_size_n = std::max(buf_size_n, (size_t)1);
        buf_size_t = std::max(buf_size_t, (size_t)1);
        if (!cudaOk(cudaMalloc(&spmv_buffer_n, buf_size_n))) return false;
        if (!cudaOk(cudaMalloc(&spmv_buffer_t, buf_size_t))) return false;

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

    // Scalar download helpers.
    double downloadScalar(int idx) {
        double val = 0.0;
        cudaMemcpyAsync(&val, d_scalar + idx, sizeof(double),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        return val;
    }

    // Download count scalars in a single memcpy + sync.
    double h_scalars[16] = {};
    void downloadScalars(int count) {
        cudaMemcpyAsync(h_scalars, d_scalar, sizeof(double) * count,
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    }

    // Zero all scalar slots in one memset (call before a batch of async reductions).
    void zeroScalars(int count) {
        cudaMemsetAsync(d_scalar, 0, sizeof(double) * count, stream);
    }

    // Init scalar slots for MaxStep (need value 1.0, not 0.0).
    double h_init_ones[16] = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
    void initMaxStepScalars(int slot, int count) {
        cudaMemcpyAsync(d_scalar + slot, h_init_ones, sizeof(double) * count,
                        cudaMemcpyHostToDevice, stream);
    }

    // Async reduction launchers — write to d_scalar[slot], no sync.
    // Caller must ensure the slot is zeroed (via zeroScalars) before calling.
    void gpuDotAsync(double* d_a, double* d_b, int sz, int slot) {
        kernelDot<<<gridSize(sz), kBlockSize, 0, stream>>>(sz, d_a, d_b, d_scalar + slot);
    }

    void gpuInfNormAsync(double* d_v, int sz, int slot) {
        kernelInfNorm<<<gridSize(sz), kBlockSize, 0, stream>>>(sz, d_v, d_scalar + slot);
    }

    void gpuMaxStepAsync(double* d_x, double* d_dx, int sz, double fraction, int slot) {
        kernelMaxStep<<<gridSize(sz), kBlockSize, 0, stream>>>(
            sz, d_x, d_dx, fraction, d_scalar + slot);
    }

    void gpuMuAsync(double* d_z_in, double* d_s_in, int sz, int slot) {
        gpuDotAsync(d_z_in, d_s_in, sz, slot);
    }

    // Synchronous wrappers (for starting point and other non-hot paths).
    double gpuDot(double* d_a, double* d_b, int sz) {
        zeroScalars(1);
        gpuDotAsync(d_a, d_b, sz, 0);
        return downloadScalar(0);
    }

    double gpuInfNorm(double* d_v, int sz) {
        zeroScalars(1);
        gpuInfNormAsync(d_v, sz, 0);
        return downloadScalar(0);
    }

    double gpuMaxStep(double* d_x, double* d_dx, int sz, double fraction) {
        initMaxStepScalars(0, 1);
        gpuMaxStepAsync(d_x, d_dx, sz, fraction, 0);
        double alpha = downloadScalar(0);
        return std::clamp(alpha, 0.0, 1.0);
    }

    double gpuMu(double* d_z_in, double* d_s_in, int sz) {
        zeroScalars(1);
        gpuMuAsync(d_z_in, d_s_in, sz, 0);
        return downloadScalar(0) / std::max(sz, 1);
    }

    double gpuMin(double* d_v, int sz) {
        double big = 1e30;
        cudaMemcpyAsync(d_scalar, &big, sizeof(double), cudaMemcpyHostToDevice, stream);
        kernelMin<<<gridSize(sz), kBlockSize, 0, stream>>>(sz, d_v, d_scalar);
        return downloadScalar(0);
    }

    double gpuSum(double* d_v, int sz) {
        cudaMemsetAsync(d_scalar, 0, sizeof(double), stream);
        kernelSum<<<gridSize(sz), kBlockSize, 0, stream>>>(sz, d_v, d_scalar);
        return downloadScalar(0);
    }

    // Lazily allocate augmented IR buffers (called on first augmented use).
    bool allocAugIRBuffers() {
        if (d_aug_ir_pool) return true;  // Already allocated.
        size_t sz = static_cast<size_t>(2) * (n + m);
        if (!cudaOk(cudaMalloc(&d_aug_ir_pool, sizeof(double) * sz))) return false;
        d_aug_ir_rhs = d_aug_ir_pool;
        d_aug_ir_sol = d_aug_ir_pool + (n + m);
        return true;
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
        cudaFree(d_aug_ir_pool);
        if (cusparse_handle) cusparseDestroy(cusparse_handle);
        if (stream) cudaStreamDestroy(stream);
    }
};

// ---------------------------------------------------------------------------
// NormalEqSolver: cuDSS SPD Cholesky with mixed precision support
// ---------------------------------------------------------------------------

struct NormalEqSolver {
    GpuContext* ctx = nullptr;

    int ne_m = 0, ne_nnz = 0;
    int* d_ne_row_starts = nullptr;
    int* d_ne_col_indices = nullptr;
    double* d_ne_values = nullptr;

    // FP32 mixed precision support.
    float* d_ne_values_f32 = nullptr;
    float* d_rhs_f32 = nullptr;
    float* d_sol_f32 = nullptr;
    bool use_fp32 = false;

    // FP64 cuDSS handles.
    cudssHandle_t cudss_handle = nullptr;
    cudssConfig_t cudss_config = nullptr;
    cudssData_t cudss_data = nullptr;
    cudssMatrix_t cudss_mat = nullptr;
    cudssMatrix_t cudss_rhs_mat = nullptr;
    cudssMatrix_t cudss_sol_mat = nullptr;
    bool analyzed = false;

    // FP32 cuDSS handles.
    cudssHandle_t cudss_handle_f32 = nullptr;
    cudssConfig_t cudss_config_f32 = nullptr;
    cudssData_t cudss_data_f32 = nullptr;
    cudssMatrix_t cudss_mat_f32 = nullptr;
    cudssMatrix_t cudss_rhs_mat_f32 = nullptr;
    cudssMatrix_t cudss_sol_mat_f32 = nullptr;
    bool analyzed_f32 = false;

    double last_reg_ = 0.0;
    bool ok = false;

    ~NormalEqSolver() {
        // FP64
        if (cudss_rhs_mat) cudssMatrixDestroy(cudss_rhs_mat);
        if (cudss_sol_mat) cudssMatrixDestroy(cudss_sol_mat);
        if (cudss_mat) cudssMatrixDestroy(cudss_mat);
        if (cudss_data) cudssDataDestroy(cudss_handle, cudss_data);
        if (cudss_config) cudssConfigDestroy(cudss_config);
        if (cudss_handle) cudssDestroy(cudss_handle);
        // FP32
        if (cudss_rhs_mat_f32) cudssMatrixDestroy(cudss_rhs_mat_f32);
        if (cudss_sol_mat_f32) cudssMatrixDestroy(cudss_sol_mat_f32);
        if (cudss_mat_f32) cudssMatrixDestroy(cudss_mat_f32);
        if (cudss_data_f32) cudssDataDestroy(cudss_handle_f32, cudss_data_f32);
        if (cudss_config_f32) cudssConfigDestroy(cudss_config_f32);
        if (cudss_handle_f32) cudssDestroy(cudss_handle_f32);

        cudaFree(d_ne_row_starts);
        cudaFree(d_ne_col_indices);
        cudaFree(d_ne_values);
        cudaFree(d_ne_values_f32);
        cudaFree(d_rhs_f32);
        cudaFree(d_sol_f32);
    }

    // Build NE sparsity pattern, upload, cuDSS analysis.
    // Returns false if NE is too dense.
    bool analyze(int a_m, int a_n,
                 const int* h_row_starts, const int* h_col_indices,
                 GpuContext* context) {
        ctx = context;
        ne_m = a_m;

        // Build NE pattern on CPU.
        std::vector<std::vector<int>> ne_cols(a_m);
        for (int i = 0; i < a_m; ++i) ne_cols[i].push_back(i);

        std::vector<std::vector<int>> col_rows(a_n);
        for (int i = 0; i < a_m; ++i) {
            for (int p = h_row_starts[i]; p < h_row_starts[i + 1]; ++p)
                col_rows[h_col_indices[p]].push_back(i);
        }

        for (int k = 0; k < a_n; ++k) {
            auto& rows = col_rows[k];
            for (size_t a_idx = 0; a_idx < rows.size(); ++a_idx)
                for (size_t b_idx = 0; b_idx <= a_idx; ++b_idx)
                    ne_cols[rows[a_idx]].push_back(rows[b_idx]);
        }

        for (int i = 0; i < a_m; ++i) {
            auto& row = ne_cols[i];
            std::sort(row.begin(), row.end());
            row.erase(std::unique(row.begin(), row.end()), row.end());
        }

        std::vector<int> h_ne_row_starts(a_m + 1);
        ne_nnz = 0;
        for (int i = 0; i < a_m; ++i) {
            h_ne_row_starts[i] = ne_nnz;
            ne_nnz += static_cast<int>(ne_cols[i].size());
        }
        h_ne_row_starts[a_m] = ne_nnz;

        double avg_density = static_cast<double>(ne_nnz) / std::max(a_m, 1);
        if (avg_density > 20.0) return false;

        std::vector<int> h_ne_col_indices(ne_nnz);
        int pos = 0;
        for (int i = 0; i < a_m; ++i)
            for (int j : ne_cols[i]) h_ne_col_indices[pos++] = j;

        // Upload.
        if (!cudaOk(cudaMalloc(&d_ne_row_starts, sizeof(int) * (a_m + 1)))) return false;
        if (!cudaOk(cudaMalloc(&d_ne_col_indices, sizeof(int) * ne_nnz))) return false;
        if (!cudaOk(cudaMalloc(&d_ne_values, sizeof(double) * ne_nnz))) return false;

        cudaMemcpyAsync(d_ne_row_starts, h_ne_row_starts.data(),
                        sizeof(int) * (a_m + 1), cudaMemcpyHostToDevice, ctx->stream);
        cudaMemcpyAsync(d_ne_col_indices, h_ne_col_indices.data(),
                        sizeof(int) * ne_nnz, cudaMemcpyHostToDevice, ctx->stream);
        cudaMemsetAsync(d_ne_values, 0, sizeof(double) * ne_nnz, ctx->stream);

        // FP64 cuDSS setup.
        if (!cudssOk(cudssCreate(&cudss_handle))) return false;
        if (!cudssOk(cudssSetStream(cudss_handle, ctx->stream))) return false;
        if (!cudssOk(cudssConfigCreate(&cudss_config))) return false;
        if (!cudssOk(cudssDataCreate(cudss_handle, &cudss_data))) return false;

        if (!cudssOk(cudssMatrixCreateCsr(
                &cudss_mat, ne_m, ne_m, ne_nnz, d_ne_row_starts,
                nullptr, d_ne_col_indices, d_ne_values,
                CUDA_R_32I, CUDA_R_64F,
                CUDSS_MTYPE_SPD, CUDSS_MVIEW_LOWER, CUDSS_BASE_ZERO)))
            return false;

        if (!cudssOk(cudssMatrixCreateDn(
                &cudss_rhs_mat, ne_m, 1, ne_m, ctx->d_rhs, CUDA_R_64F,
                CUDSS_LAYOUT_COL_MAJOR)))
            return false;
        if (!cudssOk(cudssMatrixCreateDn(
                &cudss_sol_mat, ne_m, 1, ne_m, ctx->d_dy, CUDA_R_64F,
                CUDSS_LAYOUT_COL_MAJOR)))
            return false;

        if (!cudssOk(cudssExecute(cudss_handle, CUDSS_PHASE_ANALYSIS,
                                   cudss_config, cudss_data, cudss_mat,
                                   cudss_sol_mat, cudss_rhs_mat)))
            return false;

        // FP32 cuDSS setup for mixed precision.
        if (!cudaOk(cudaMalloc(&d_ne_values_f32, sizeof(float) * ne_nnz))) return false;
        if (!cudaOk(cudaMalloc(&d_rhs_f32, sizeof(float) * a_m))) return false;
        if (!cudaOk(cudaMalloc(&d_sol_f32, sizeof(float) * a_m))) return false;

        if (!cudssOk(cudssCreate(&cudss_handle_f32))) return false;
        if (!cudssOk(cudssSetStream(cudss_handle_f32, ctx->stream))) return false;
        if (!cudssOk(cudssConfigCreate(&cudss_config_f32))) return false;
        if (!cudssOk(cudssDataCreate(cudss_handle_f32, &cudss_data_f32))) return false;

        if (!cudssOk(cudssMatrixCreateCsr(
                &cudss_mat_f32, ne_m, ne_m, ne_nnz, d_ne_row_starts,
                nullptr, d_ne_col_indices, d_ne_values_f32,
                CUDA_R_32I, CUDA_R_32F,
                CUDSS_MTYPE_SPD, CUDSS_MVIEW_LOWER, CUDSS_BASE_ZERO)))
            return false;

        if (!cudssOk(cudssMatrixCreateDn(
                &cudss_rhs_mat_f32, ne_m, 1, ne_m, d_rhs_f32, CUDA_R_32F,
                CUDSS_LAYOUT_COL_MAJOR)))
            return false;
        if (!cudssOk(cudssMatrixCreateDn(
                &cudss_sol_mat_f32, ne_m, 1, ne_m, d_sol_f32, CUDA_R_32F,
                CUDSS_LAYOUT_COL_MAJOR)))
            return false;

        if (!cudssOk(cudssExecute(cudss_handle_f32, CUDSS_PHASE_ANALYSIS,
                                   cudss_config_f32, cudss_data_f32,
                                   cudss_mat_f32, cudss_sol_mat_f32,
                                   cudss_rhs_mat_f32)))
            return false;

        cudaStreamSynchronize(ctx->stream);
        analyzed = true;
        analyzed_f32 = true;
        use_fp32 = true;  // Default to FP32 with IR.
        ok = true;
        return true;
    }

    // Form NE values and factorize.
    bool factorize(double reg) {
        if (!ok) return false;
        last_reg_ = reg;

        kernelFormNEValues<<<gridSize(ne_nnz), kBlockSize, 0, ctx->stream>>>(
            ne_nnz, d_ne_row_starts, d_ne_col_indices,
            ctx->d_a_row_starts, ctx->d_a_col_indices, ctx->d_a_values,
            ctx->d_theta, reg, ne_m, d_ne_values);

        if (use_fp32) {
            // Cast to FP32 and factorize.
            kernelCastF64toF32<<<gridSize(ne_nnz), kBlockSize, 0, ctx->stream>>>(
                ne_nnz, d_ne_values, d_ne_values_f32);

            auto phase = analyzed_f32 ? CUDSS_PHASE_REFACTORIZATION
                                      : CUDSS_PHASE_FACTORIZATION;
            auto status = cudssExecute(cudss_handle_f32, phase,
                                        cudss_config_f32, cudss_data_f32,
                                        cudss_mat_f32, cudss_sol_mat_f32,
                                        cudss_rhs_mat_f32);
            if (!cudssOk(status)) {
                status = cudssExecute(cudss_handle_f32, CUDSS_PHASE_FACTORIZATION,
                                       cudss_config_f32, cudss_data_f32,
                                       cudss_mat_f32, cudss_sol_mat_f32,
                                       cudss_rhs_mat_f32);
                if (!cudssOk(status)) {
                    // FP32 failed — fall back to FP64.
                    use_fp32 = false;
                    return factorizeFp64();
                }
            }
            return true;
        }

        return factorizeFp64();
    }

    bool factorizeFp64() {
        auto phase = analyzed ? CUDSS_PHASE_REFACTORIZATION
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

    // Raw cuDSS solve (FP32 path: solve in FP32, cast result to FP64).
    bool solveFp32(double* d_rhs_in, double* d_sol_out) {
        int m = ne_m;
        // Cast RHS to FP32.
        kernelCastF64toF32<<<gridSize(m), kBlockSize, 0, ctx->stream>>>(
            m, d_rhs_in, d_rhs_f32);

        // Rebind data pointers without recreating descriptors.
        cudssMatrixSetValues(cudss_rhs_mat_f32, d_rhs_f32);
        cudssMatrixSetValues(cudss_sol_mat_f32, d_sol_f32);

        if (!cudssOk(cudssExecute(cudss_handle_f32, CUDSS_PHASE_SOLVE,
                                   cudss_config_f32, cudss_data_f32,
                                   cudss_mat_f32, cudss_sol_mat_f32,
                                   cudss_rhs_mat_f32)))
            return false;

        // Cast solution back to FP64.
        kernelCastF32toF64<<<gridSize(m), kBlockSize, 0, ctx->stream>>>(
            m, d_sol_f32, d_sol_out);
        return true;
    }

    // Raw cuDSS solve (FP64 path).
    bool solveFp64(double* d_rhs_in, double* d_sol_out) {
        // Rebind data pointers without recreating descriptors.
        cudssMatrixSetValues(cudss_rhs_mat, d_rhs_in);
        cudssMatrixSetValues(cudss_sol_mat, d_sol_out);

        return cudssOk(cudssExecute(cudss_handle, CUDSS_PHASE_SOLVE,
                                     cudss_config, cudss_data, cudss_mat,
                                     cudss_sol_mat, cudss_rhs_mat));
    }

    // Dispatch to FP32 or FP64 solve.
    bool solveOnce(double* rhs, double* sol) {
        return use_fp32 ? solveFp32(rhs, sol) : solveFp64(rhs, sol);
    }

    // Solve with iterative refinement.
    // FP32 path: solve in FP32, then IR steps in FP64.
    // FP64 path: solve in FP64 with standard IR.
    bool solveRefined(double* d_rhs_in, double* d_sol_out, int ir_steps) {
        if (!solveOnce(d_rhs_in, d_sol_out)) return false;
        if (ir_steps <= 0) return true;

        int m = ne_m;
        int n = ctx->n;

        for (int step = 0; step < ir_steps; ++step) {
            // Compute M * sol where M = A*diag(theta)*A' + reg*I.
            ctx->multiplyAT(d_sol_out, ctx->d_tmp);
            kernelZS<<<gridSize(n), kBlockSize, 0, ctx->stream>>>(
                n, ctx->d_tmp, ctx->d_theta, ctx->d_tmp);
            ctx->multiplyA(ctx->d_tmp, ctx->d_ir_residual);

            double reg = last_reg_;
            kernelAxpy<<<gridSize(m), kBlockSize, 0, ctx->stream>>>(
                m, reg, d_sol_out, ctx->d_ir_residual);

            // residual = rhs - M*sol
            kernelResidualIR<<<gridSize(m), kBlockSize, 0, ctx->stream>>>(
                m, d_rhs_in, ctx->d_ir_residual, ctx->d_ir_residual);

            if (!solveOnce(ctx->d_ir_residual, ctx->d_ir_correction)) return false;

            kernelAddCorrection<<<gridSize(m), kBlockSize, 0, ctx->stream>>>(
                m, ctx->d_ir_correction, d_sol_out);
        }
        return true;
    }
};

// ---------------------------------------------------------------------------
// AugmentedSolver: cuDSS SYMMETRIC LDL'
// ---------------------------------------------------------------------------

struct AugmentedSolver {
    GpuContext* ctx = nullptr;
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
    bool ok = false;

    ~AugmentedSolver() {
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

    bool analyze(int a_m, int a_n,
                 const int* h_row_starts, const int* h_col_indices,
                 int a_nnz, GpuContext* context) {
        ctx = context;
        m_ = a_m;
        n_ = a_n;
        aug_dim_ = a_n + a_m;
        aug_nnz_ = a_n + a_nnz + a_m;

        // Build augmented pattern.
        std::vector<int> h_aug_rows(aug_dim_ + 1);
        std::vector<int> h_aug_cols(aug_nnz_);

        for (int j = 0; j < a_n; ++j) {
            h_aug_rows[j] = j;
            h_aug_cols[j] = j;
        }

        int pos = a_n;
        for (int i = 0; i < a_m; ++i) {
            h_aug_rows[a_n + i] = pos;
            for (int p = h_row_starts[i]; p < h_row_starts[i + 1]; ++p)
                h_aug_cols[pos++] = h_col_indices[p];
            h_aug_cols[pos++] = a_n + i;
        }
        h_aug_rows[aug_dim_] = pos;

        if (!cudaOk(cudaMalloc(&d_aug_row_starts, sizeof(int) * (aug_dim_ + 1)))) return false;
        if (!cudaOk(cudaMalloc(&d_aug_col_indices, sizeof(int) * aug_nnz_))) return false;
        if (!cudaOk(cudaMalloc(&d_aug_values, sizeof(double) * aug_nnz_))) return false;

        cudaMemcpyAsync(d_aug_row_starts, h_aug_rows.data(),
                        sizeof(int) * (aug_dim_ + 1), cudaMemcpyHostToDevice, ctx->stream);
        cudaMemcpyAsync(d_aug_col_indices, h_aug_cols.data(),
                        sizeof(int) * aug_nnz_, cudaMemcpyHostToDevice, ctx->stream);
        cudaMemsetAsync(d_aug_values, 0, sizeof(double) * aug_nnz_, ctx->stream);

        // cuDSS setup.
        if (!cudssOk(cudssCreate(&cudss_handle))) return false;
        if (!cudssOk(cudssSetStream(cudss_handle, ctx->stream))) return false;
        if (!cudssOk(cudssConfigCreate(&cudss_config))) return false;
        if (!cudssOk(cudssDataCreate(cudss_handle, &cudss_data))) return false;

        if (!cudssOk(cudssMatrixCreateCsr(
                &cudss_mat, aug_dim_, aug_dim_, aug_nnz_, d_aug_row_starts,
                nullptr, d_aug_col_indices, d_aug_values,
                CUDA_R_32I, CUDA_R_64F,
                CUDSS_MTYPE_SYMMETRIC, CUDSS_MVIEW_LOWER, CUDSS_BASE_ZERO)))
            return false;

        if (!cudssOk(cudssMatrixCreateDn(
                &cudss_rhs_mat, aug_dim_, 1, aug_dim_, ctx->d_aug_rhs,
                CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR)))
            return false;
        if (!cudssOk(cudssMatrixCreateDn(
                &cudss_sol_mat, aug_dim_, 1, aug_dim_, ctx->d_aug_sol,
                CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR)))
            return false;

        if (!cudssOk(cudssExecute(cudss_handle, CUDSS_PHASE_ANALYSIS,
                                   cudss_config, cudss_data, cudss_mat,
                                   cudss_sol_mat, cudss_rhs_mat)))
            return false;

        cudaStreamSynchronize(ctx->stream);
        analyzed_ = true;
        ok = true;
        return true;
    }

    double last_reg_primal_ = 0.0;
    double last_reg_dual_ = 0.0;

    bool factorize(double reg_primal, double reg_dual) {
        if (!ok) return false;
        last_reg_primal_ = reg_primal;
        last_reg_dual_ = reg_dual;

        kernelFillAugValues<<<gridSize(std::max(n_, m_)), kBlockSize, 0,
                              ctx->stream>>>(
            n_, m_, ctx->d_s, ctx->d_z, reg_primal, reg_dual,
            ctx->d_a_values, ctx->d_a_row_starts, d_aug_row_starts,
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

    bool solve(double* d_rhs_in, double* d_sol_out) {
        if (!ok) return false;

        // Rebind data pointers without recreating descriptors.
        cudssMatrixSetValues(cudss_rhs_mat, d_rhs_in);
        cudssMatrixSetValues(cudss_sol_mat, d_sol_out);

        return cudssOk(cudssExecute(cudss_handle, CUDSS_PHASE_SOLVE,
                                     cudss_config, cudss_data, cudss_mat,
                                     cudss_sol_mat, cudss_rhs_mat));
    }

    // Solve with iterative refinement for augmented system.
    // rhs and sol are (n+m)-sized augmented vectors: [z-part; y-part].
    bool solveRefined(double* d_rhs_in, double* d_sol_out, int ir_steps) {
        if (!solve(d_rhs_in, d_sol_out)) return false;
        if (ir_steps <= 0) return true;

        int n = n_, m = m_;
        // Use dedicated IR buffers to avoid clobbering d_aug_rhs/d_aug_sol.
        double* d_ir_rhs = ctx->d_aug_ir_rhs;
        double* d_ir_sol = ctx->d_aug_ir_sol;

        for (int step = 0; step < ir_steps; ++step) {
            // Compute augmented residual: r = rhs - K * sol
            double* sol_z = d_sol_out;
            double* sol_y = d_sol_out + n;

            // A'*sol_y → d_tmp (n-sized, fits in max(m,n))
            ctx->multiplyAT(sol_y, ctx->d_tmp);
            kernelAugIRResidualZ<<<gridSize(n), kBlockSize, 0, ctx->stream>>>(
                n, d_rhs_in, sol_z, ctx->d_tmp, ctx->d_s, ctx->d_z,
                last_reg_primal_, d_ir_rhs);

            // A*sol_z → d_ir_residual (m-sized scratch)
            ctx->multiplyA(sol_z, ctx->d_ir_residual);
            kernelAugIRResidualY<<<gridSize(m), kBlockSize, 0, ctx->stream>>>(
                m, d_rhs_in + n, ctx->d_ir_residual, sol_y,
                last_reg_dual_, d_ir_rhs + n);

            // Solve correction: K * correction = residual
            if (!solve(d_ir_rhs, d_ir_sol)) return false;

            // Accumulate: sol += correction
            kernelAddCorrection<<<gridSize(n + m), kBlockSize, 0, ctx->stream>>>(
                n + m, d_ir_sol, d_sol_out);
        }
        return true;
    }
};

// ---------------------------------------------------------------------------
// GpuBarrierImpl: full device-resident IPM with auto-switching
// ---------------------------------------------------------------------------

struct GpuBarrierImpl {
    GpuContext ctx;
    NormalEqSolver ne_solver;
    AugmentedSolver aug_solver;

    enum class Backend { NE, Augmented };
    Backend active_backend = Backend::NE;
    bool ne_available = false;
    bool aug_available = false;

    int ne_fail_count = 0;
    int ir_steps_ = 2;

    // Host copies for augmented lazy init.
    std::vector<int> h_row_starts;
    std::vector<int> h_col_indices;
    int stored_nnz = 0;

    bool init(int m, int n, int nnz,
              const int* rows, const int* cols, const double* vals,
              const double* b, const double* c,
              bool prefer_augmented) {
        h_row_starts.assign(rows, rows + m + 1);
        h_col_indices.assign(cols, cols + nnz);
        stored_nnz = nnz;

        if (!ctx.init(m, n, nnz, rows, cols, vals, b, c))
            return false;

        if (!prefer_augmented) {
            // Try NE first.
            ne_available = ne_solver.analyze(m, n, rows, cols, &ctx);
            if (ne_available) {
                active_backend = Backend::NE;
            } else {
                // NE too dense, use Augmented.
                aug_available = aug_solver.analyze(m, n, rows, cols, nnz, &ctx);
                if (!aug_available) return false;
                if (!ctx.allocAugIRBuffers()) return false;
                active_backend = Backend::Augmented;
            }
        } else {
            // Prefer augmented.
            aug_available = aug_solver.analyze(m, n, rows, cols, nnz, &ctx);
            if (!aug_available) return false;
            if (!ctx.allocAugIRBuffers()) return false;
            active_backend = Backend::Augmented;
        }

        return true;
    }

    // Switch from NE to Augmented mid-solve.
    bool switchToAugmented() {
        if (aug_available) {
            active_backend = Backend::Augmented;
            return ctx.allocAugIRBuffers();
        }
        // Lazy init augmented.
        int m = ctx.m, n = ctx.n;
        aug_available = aug_solver.analyze(m, n, h_row_starts.data(),
                                           h_col_indices.data(),
                                           stored_nnz, &ctx);
        if (!aug_available) return false;
        if (!ctx.allocAugIRBuffers()) return false;
        active_backend = Backend::Augmented;
        return true;
    }

    // Compute s0, shift z/s to positivity, and apply Mehrotra correction.
    // Assumes d_y and d_z contain initial y0 and z0 (or z0 = A'y0).
    bool computeStartingPointS() {
        int m = ctx.m, n = ctx.n;

        // s0 = c - A'y0. Recompute A'y into d_z, then s = c - z.
        if (!ctx.multiplyAT(ctx.d_y, ctx.d_z)) return false;

        kernelVecSub<<<gridSize(n), kBlockSize, 0, ctx.stream>>>(
            n, ctx.d_c, ctx.d_z, ctx.d_s);

        // Shift to positivity.
        double min_z = ctx.gpuMin(ctx.d_z, n);
        double min_s = ctx.gpuMin(ctx.d_s, n);

        double delta_z = std::max(-1.5 * min_z, 0.0);
        double delta_s = std::max(-1.5 * min_s, 0.0);

        kernelAddScalar<<<gridSize(n), kBlockSize, 0, ctx.stream>>>(
            n, delta_z, ctx.d_z);
        kernelAddScalar<<<gridSize(n), kBlockSize, 0, ctx.stream>>>(
            n, delta_s, ctx.d_s);

        double zs_dot = ctx.gpuDot(ctx.d_z, ctx.d_s, n);
        double z_sum = ctx.gpuSum(ctx.d_z, n);
        double s_sum = ctx.gpuSum(ctx.d_s, n);

        double dp = 0.5 * zs_dot / std::max(s_sum, 1e-30);
        double dd = 0.5 * zs_dot / std::max(z_sum, 1e-30);

        kernelShiftPositive<<<gridSize(n), kBlockSize, 0, ctx.stream>>>(
            n, 0.0, dd, dp, ctx.d_z, ctx.d_s);

        cudaStreamSynchronize(ctx.stream);
        return true;
    }

    // Compute Mehrotra starting point on GPU.
    bool computeStartingPoint() {
        int m = ctx.m, n = ctx.n;

        // Set theta = ones.
        kernelFillOnes<<<gridSize(n), kBlockSize, 0, ctx.stream>>>(n, ctx.d_theta);

        double reg = 1e-8;
        bool factor_ok = false;

        if (active_backend == Backend::NE) {
            factor_ok = ne_solver.factorize(reg);
            if (factor_ok) {
                if (!ne_solver.solveFp64(ctx.d_b, ctx.d_y)) return false;
            }
        }

        if (!factor_ok && active_backend == Backend::NE) {
            // Try augmented for starting point.
            if (!switchToAugmented()) return false;
        }

        if (active_backend == Backend::Augmented) {
            // Initialize z and s to 1 so kernelFillAugValues reads valid data.
            kernelFillOnes<<<gridSize(n), kBlockSize, 0, ctx.stream>>>(n, ctx.d_z);
            kernelFillOnes<<<gridSize(n), kBlockSize, 0, ctx.stream>>>(n, ctx.d_s);
            // For starting point with augmented: factorize and solve.
            if (!aug_solver.factorize(reg, reg)) return false;
            // Set up RHS: [0; b] for augmented system.
            cudaMemsetAsync(ctx.d_aug_rhs, 0, sizeof(double) * n, ctx.stream);
            cudaMemcpyAsync(ctx.d_aug_rhs + n, ctx.d_b, sizeof(double) * m,
                            cudaMemcpyDeviceToDevice, ctx.stream);
            if (!aug_solver.solve(ctx.d_aug_rhs, ctx.d_aug_sol)) return false;
            // Extract z0 from first n, y0 from last m.
            cudaMemcpyAsync(ctx.d_z, ctx.d_aug_sol, sizeof(double) * n,
                            cudaMemcpyDeviceToDevice, ctx.stream);
            cudaMemcpyAsync(ctx.d_y, ctx.d_aug_sol + n, sizeof(double) * m,
                            cudaMemcpyDeviceToDevice, ctx.stream);
            return computeStartingPointS();
        }

        // NE path: z0 = A' * y0.
        if (!ctx.multiplyAT(ctx.d_y, ctx.d_z)) return false;
        return computeStartingPointS();
    }

    // Solve NE Newton step (all on device).
    bool solveNewtonNE(double* d_rp, double* d_rd, double* d_rc,
                       double* d_dz_out, double* d_dy_out, double* d_ds_out) {
        int m = ctx.m, n = ctx.n;

        kernelScaledRhsH<<<gridSize(n), kBlockSize, 0, ctx.stream>>>(
            n, d_rc, ctx.d_s, ctx.d_theta, d_rd, ctx.d_h);
        ctx.multiplyA(ctx.d_h, ctx.d_ah);
        kernelNormalEqRhs<<<gridSize(m), kBlockSize, 0, ctx.stream>>>(
            m, d_rp, ctx.d_ah, ctx.d_rhs);

        if (!ne_solver.solveRefined(ctx.d_rhs, d_dy_out, ir_steps_)) return false;

        ctx.multiplyAT(d_dy_out, ctx.d_atdy);
        kernelSearchDirectionZ<<<gridSize(n), kBlockSize, 0, ctx.stream>>>(
            n, ctx.d_h, ctx.d_theta, ctx.d_atdy, d_dz_out);
        kernelSearchDirectionS<<<gridSize(n), kBlockSize, 0, ctx.stream>>>(
            n, d_rd, ctx.d_atdy, d_ds_out);

        return true;
    }

    // Solve Augmented Newton step (all on device).
    bool solveNewtonAug(double* d_rp, double* d_rd, double* d_rc,
                        double* d_dz_out, double* d_dy_out, double* d_ds_out) {
        int m = ctx.m, n = ctx.n;

        kernelAugRhsZ<<<gridSize(n), kBlockSize, 0, ctx.stream>>>(
            n, d_rd, d_rc, ctx.d_z, ctx.d_aug_rhs);
        cudaMemcpyAsync(ctx.d_aug_rhs + n, d_rp, sizeof(double) * m,
                        cudaMemcpyDeviceToDevice, ctx.stream);

        if (!aug_solver.solveRefined(ctx.d_aug_rhs, ctx.d_aug_sol, ir_steps_)) return false;

        cudaMemcpyAsync(d_dz_out, ctx.d_aug_sol, sizeof(double) * n,
                        cudaMemcpyDeviceToDevice, ctx.stream);
        cudaMemcpyAsync(d_dy_out, ctx.d_aug_sol + n, sizeof(double) * m,
                        cudaMemcpyDeviceToDevice, ctx.stream);

        kernelDsFromAug<<<gridSize(n), kBlockSize, 0, ctx.stream>>>(
            n, d_rc, ctx.d_s, d_dz_out, ctx.d_z, d_ds_out);

        return true;
    }

    // Dispatch Newton solve to active backend.
    bool solveNewtonStep(double* d_rp, double* d_rd, double* d_rc,
                         double* d_dz_out, double* d_dy_out, double* d_ds_out) {
        if (active_backend == Backend::NE)
            return solveNewtonNE(d_rp, d_rd, d_rc, d_dz_out, d_dy_out, d_ds_out);
        else
            return solveNewtonAug(d_rp, d_rd, d_rc, d_dz_out, d_dy_out, d_ds_out);
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

    // Device-resident Mehrotra IPM loop.
    bool solve(int max_iter, double tol, double step_fraction, double base_reg,
               int ir_steps, bool verbose, const void* stop_flag_ptr,
               double obj_offset,
               const double* h_b, const double* h_c, int h_n, int h_m,
               std::vector<Real>& z_out, std::vector<Real>& y_out,
               std::vector<Real>& s_out, double* out_obj,
               int* out_status, int* out_iters) {
        int m = ctx.m, n = ctx.n;
        ir_steps_ = ir_steps;
        // We can't use std::atomic in CUDA code, so cast to volatile bool*.
        // The caller guarantees this is an atomic<bool>, and volatile read
        // gives us the needed visibility on x86.
        const volatile bool* stop_flag =
            static_cast<const volatile bool*>(stop_flag_ptr);

        if (!computeStartingPoint()) return false;

        // Compute inv_b, inv_c from host data.
        double max_b = 0.0, max_c = 0.0;
        for (int i = 0; i < h_m; ++i) max_b = std::max(max_b, std::abs(h_b[i]));
        for (int j = 0; j < h_n; ++j) max_c = std::max(max_c, std::abs(h_c[j]));
        const double inv_b = 1.0 / (1.0 + max_b);
        const double inv_c = 1.0 / (1.0 + max_c);

        // Adaptive regularization.
        double reg_primal = base_reg;
        double reg_dual = base_reg;

        // Stall detection.
        double prev_pinf = 1e30, prev_dinf = 1e30, prev_gap = 1e30;
        int stagnant_iters = 0;

        const char* backend_tag = (active_backend == Backend::NE)
            ? (ne_solver.use_fp32 ? "[gpu-chol-fp32]" : "[gpu-chol]")
            : "[gpu-aug]";

        for (int iter = 0; iter < max_iter; ++iter) {
            if (stop_flag != nullptr && *stop_flag) {
                *out_iters = iter;
                *out_status = 2;  // IterLimit
                return false;
            }

            // Residuals (all on GPU).
            ctx.multiplyA(ctx.d_z, ctx.d_az);
            ctx.multiplyAT(ctx.d_y, ctx.d_aty);

            // Batch: fused residual+norm, mu, pobj, dobj → 1 sync.
            // Slots: 0=sum(z*s), 1=||rp||_inf, 2=||rd||_inf, 3=c'z, 4=b'y
            ctx.zeroScalars(5);
            ctx.gpuMuAsync(ctx.d_z, ctx.d_s, n, 0);
            kernelResidualPrimalNorm<<<gridSize(m), kBlockSize, 0, ctx.stream>>>(
                m, ctx.d_b, ctx.d_az, ctx.d_rp, ctx.d_scalar + 1);
            kernelResidualDualNorm<<<gridSize(n), kBlockSize, 0, ctx.stream>>>(
                n, ctx.d_c, ctx.d_aty, ctx.d_s, ctx.d_rd, ctx.d_scalar + 2);
            ctx.gpuDotAsync(ctx.d_c, ctx.d_z, n, 3);
            ctx.gpuDotAsync(ctx.d_b, ctx.d_y, m, 4);
            ctx.downloadScalars(5);
            double mu = ctx.h_scalars[0] / std::max(n, 1);
            double pinf = ctx.h_scalars[1] * inv_b;
            double dinf = ctx.h_scalars[2] * inv_c;
            double pobj = obj_offset + ctx.h_scalars[3];
            double dobj = obj_offset + ctx.h_scalars[4];
            double gap = std::abs(mu) / (1.0 + std::abs(pobj));

            // Update backend tag when switching.
            backend_tag = (active_backend == Backend::NE)
                ? (ne_solver.use_fp32 ? "[gpu-chol-fp32]" : "[gpu-chol]")
                : "[gpu-aug]";

            if (verbose) {
                std::printf(
                    "IPM %4d  pobj=% .10e  dobj=% .10e  pinf=% .2e  dinf=% .2e  "
                    "gap=% .2e  reg=% .1e %s\n",
                    iter, pobj, dobj, pinf, dinf, gap, reg_primal, backend_tag);
            }

            if (pinf < tol && dinf < tol && gap < tol) {
                *out_iters = iter;
                *out_obj = pobj;
                *out_status = 0;  // Optimal
                downloadSolution(z_out, y_out, s_out);
                return true;
            }

            // Instability detection: residual jump >100x.
            if (iter > 0 && (pinf > 100.0 * prev_pinf + 1e-10 ||
                             dinf > 100.0 * prev_dinf + 1e-10)) {
                if (verbose) {
                    std::printf("IPM: numerical instability at iter %d "
                                "(pinf %.2e -> %.2e, dinf %.2e -> %.2e)\n",
                                iter, prev_pinf, pinf, prev_dinf, dinf);
                }
                // Restore backup.
                cudaMemcpyAsync(ctx.d_z, ctx.d_z_bak, sizeof(double) * n,
                                cudaMemcpyDeviceToDevice, ctx.stream);
                cudaMemcpyAsync(ctx.d_y, ctx.d_y_bak, sizeof(double) * m,
                                cudaMemcpyDeviceToDevice, ctx.stream);
                cudaMemcpyAsync(ctx.d_s, ctx.d_s_bak, sizeof(double) * n,
                                cudaMemcpyDeviceToDevice, ctx.stream);
                cudaStreamSynchronize(ctx.stream);

                double relaxed_tol = tol * 10.0;
                if (prev_pinf < relaxed_tol && prev_dinf < relaxed_tol &&
                    prev_gap < relaxed_tol) {
                    if (verbose) {
                        std::printf("IPM: accepted last good iterate "
                                    "(pinf=%.2e dinf=%.2e gap=%.2e)\n",
                                    prev_pinf, prev_dinf, prev_gap);
                    }
                    *out_iters = iter;
                    *out_obj = pobj;
                    *out_status = 0;
                    downloadSolution(z_out, y_out, s_out);
                    return true;
                }
                *out_iters = iter;
                *out_status = 1;  // Failed
                return false;
            }

            // Stall detection.
            if (gap >= 0.99 * prev_gap && iter > 5) {
                stagnant_iters++;
                if (stagnant_iters >= 10) {
                    if (verbose) {
                        std::printf("IPM: stagnation detected (%d iters)\n",
                                    stagnant_iters);
                    }
                    *out_iters = iter;
                    *out_status = 1;
                    return false;
                }
            } else {
                stagnant_iters = 0;
            }

            // Adaptive regularization: decay when gap improves.
            if (gap < 0.9 * prev_gap) {
                reg_primal = std::max(reg_primal * 0.5, 1e-12);
                reg_dual = std::max(reg_dual * 0.5, 1e-12);
            }

            prev_pinf = pinf;
            prev_dinf = dinf;
            prev_gap = gap;

            // Theta = z / s.
            kernelTheta<<<gridSize(n), kBlockSize, 0, ctx.stream>>>(
                n, ctx.d_z, ctx.d_s, ctx.d_theta);

            // Factorize.
            bool factorized = false;
            double cur_reg = reg_primal;

            if (active_backend == Backend::NE) {
                for (int attempt = 0; attempt < 3; ++attempt) {
                    if (ne_solver.factorize(cur_reg)) {
                        factorized = true;
                        ne_fail_count = 0;
                        break;
                    }
                    cur_reg *= 10.0;
                    reg_primal = std::min(cur_reg, 1e-4);
                }
                if (!factorized) {
                    ne_fail_count++;
                    if (verbose)
                        std::printf("IPM iter %d: NE factorize failed (%d), switching to augmented\n",
                                    iter, ne_fail_count);
                    if (!switchToAugmented()) {
                        *out_iters = iter;
                        *out_status = 1;
                        return false;
                    }
                    // Fall through to augmented factorize below.
                }
            }

            if (active_backend == Backend::Augmented && !factorized) {
                cur_reg = reg_primal;
                for (int attempt = 0; attempt < 3; ++attempt) {
                    if (aug_solver.factorize(cur_reg, reg_dual)) {
                        factorized = true;
                        break;
                    }
                    cur_reg *= 10.0;
                    reg_primal = std::min(cur_reg, 1e-4);
                    reg_dual = std::min(reg_dual * 10.0, 1e-4);
                }
                if (!factorized) {
                    *out_iters = iter;
                    *out_status = 1;
                    return false;
                }
            }

            // Predictor step.
            kernelComplementarityAffine<<<gridSize(n), kBlockSize, 0,
                                          ctx.stream>>>(n, ctx.d_z, ctx.d_s,
                                                         ctx.d_rc);

            if (!solveNewtonStep(ctx.d_rp, ctx.d_rd, ctx.d_rc,
                                 ctx.d_dz_aff, ctx.d_dy_aff, ctx.d_ds_aff)) {
                *out_iters = iter;
                *out_status = 1;
                return false;
            }

            // Batch predictor reductions: step sizes + mu_aff parts → 1 sync.
            // Slots 0-1 = MaxStep (init 1.0), slots 2-5 = dot products (init 0.0).
            ctx.initMaxStepScalars(0, 2);
            cudaMemsetAsync(ctx.d_scalar + 2, 0, sizeof(double) * 4, ctx.stream);
            ctx.gpuMaxStepAsync(ctx.d_z, ctx.d_dz_aff, n, 1.0, 0);
            ctx.gpuMaxStepAsync(ctx.d_s, ctx.d_ds_aff, n, 1.0, 1);
            kernelMuAffParts<<<gridSize(n), kBlockSize, 0, ctx.stream>>>(
                n, ctx.d_z, ctx.d_s, ctx.d_dz_aff, ctx.d_ds_aff, ctx.d_scalar + 2);
            ctx.downloadScalars(6);
            double alpha_aff_p = std::clamp(ctx.h_scalars[0], 0.0, 1.0);
            double alpha_aff_d = std::clamp(ctx.h_scalars[1], 0.0, 1.0);
            double zs = ctx.h_scalars[2], z_ds = ctx.h_scalars[3];
            double dz_s = ctx.h_scalars[4], dz_ds = ctx.h_scalars[5];
            double mu_aff = (zs + alpha_aff_d * z_ds + alpha_aff_p * dz_s +
                             alpha_aff_p * alpha_aff_d * dz_ds) /
                            std::max(n, 1);

            double sigma = std::pow(
                std::max(mu_aff, 0.0) / std::max(mu, 1e-30), 3.0);
            sigma = std::clamp(sigma, 0.0, 1.0);

            // Corrector step.
            double sigma_mu = sigma * mu;
            kernelComplementarityCorrected<<<gridSize(n), kBlockSize, 0,
                                             ctx.stream>>>(
                n, sigma_mu, ctx.d_z, ctx.d_s, ctx.d_dz_aff, ctx.d_ds_aff,
                ctx.d_rc);

            if (!solveNewtonStep(ctx.d_rp, ctx.d_rd, ctx.d_rc,
                                 ctx.d_dz, ctx.d_dy, ctx.d_ds)) {
                *out_iters = iter;
                *out_status = 1;
                return false;
            }

            // Batch corrector step sizes → 1 sync.
            ctx.initMaxStepScalars(0, 2);
            ctx.gpuMaxStepAsync(ctx.d_z, ctx.d_dz, n, step_fraction, 0);
            ctx.gpuMaxStepAsync(ctx.d_s, ctx.d_ds, n, step_fraction, 1);
            ctx.downloadScalars(2);
            double alpha_p = std::clamp(ctx.h_scalars[0], 0.0, 1.0);
            double alpha_d = std::clamp(ctx.h_scalars[1], 0.0, 1.0);

            // Backup before update.
            cudaMemcpyAsync(ctx.d_z_bak, ctx.d_z, sizeof(double) * n,
                            cudaMemcpyDeviceToDevice, ctx.stream);
            cudaMemcpyAsync(ctx.d_y_bak, ctx.d_y, sizeof(double) * m,
                            cudaMemcpyDeviceToDevice, ctx.stream);
            cudaMemcpyAsync(ctx.d_s_bak, ctx.d_s, sizeof(double) * n,
                            cudaMemcpyDeviceToDevice, ctx.stream);

            // Update iterates.
            kernelUpdateIterates<<<gridSize(std::max(m, n)), kBlockSize, 0,
                                    ctx.stream>>>(
                n, m, alpha_p, alpha_d, ctx.d_dz, ctx.d_ds, ctx.d_dy,
                ctx.d_z, ctx.d_s, ctx.d_y, 1e-12);
        }

        *out_iters = max_iter;
        *out_status = 2;  // IterLimit
        return false;
    }
};

// ---------------------------------------------------------------------------
// C-linkage bridge function
// ---------------------------------------------------------------------------

extern "C" {

bool gpuSolveBarrier(
    int m, int n, int nnz,
    const int* rows, const int* cols, const double* vals,
    const double* b, const double* c,
    int max_iter, double tol, double step_fraction, double reg,
    int ir_steps, bool verbose,
    const void* stop_flag,
    bool prefer_augmented,
    double obj_offset,
    double* out_z, double* out_y, double* out_s,
    double* out_obj, int* out_status, int* out_iters) {

    GpuBarrierImpl impl;
    if (!impl.init(m, n, nnz, rows, cols, vals, b, c, prefer_augmented))
        return false;

    std::vector<Real> z_vec, y_vec, s_vec;
    bool ok = impl.solve(max_iter, tol, step_fraction, reg, ir_steps, verbose,
                          stop_flag, obj_offset, b, c, n, m,
                          z_vec, y_vec, s_vec, out_obj, out_status, out_iters);

    if (ok || *out_status == 0) {
        std::copy(z_vec.begin(), z_vec.end(), out_z);
        std::copy(y_vec.begin(), y_vec.end(), out_y);
        std::copy(s_vec.begin(), s_vec.end(), out_s);
    }

    return ok;
}

}  // extern "C"

}  // namespace gpu_detail
}  // namespace mipx
