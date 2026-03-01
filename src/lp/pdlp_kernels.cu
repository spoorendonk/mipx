#include "pdlp_kernels.cuh"

#ifdef MIPX_HAS_CUDA

#include <cfloat>
#include <cmath>

namespace mipx {
namespace gpu {

static constexpr int kBlockSize = 256;

static int gridSize(int n) {
    return (n + kBlockSize - 1) / kBlockSize;
}

// ============================================================================
// Phase 1: Core PDLP iteration kernels
// ============================================================================

__global__ void dualStepKernel(double* y, const double* az_bar, const double* b,
                               const double* sigma, double step_pw, int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) {
        y[i] += step_pw * sigma[i] * (az_bar[i] - b[i]);
    }
}

void launchDualStep(double* y, const double* az_bar, const double* b,
                    const double* sigma, double step_pw, int m,
                    cudaStream_t stream) {
    if (m <= 0) return;
    dualStepKernel<<<gridSize(m), kBlockSize, 0, stream>>>(
        y, az_bar, b, sigma, step_pw, m);
}

__global__ void primalStepKernel(double* z, double* z_prev, const double* c,
                                 const double* at_y, const double* tau,
                                 double step_over_pw, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n) {
        double zj = z[j];
        z_prev[j] = zj;
        double grad = c[j] + at_y[j];
        double zn = zj - step_over_pw * tau[j] * grad;
        z[j] = (zn > 0.0) ? zn : 0.0;
    }
}

void launchPrimalStep(double* z, double* z_prev, const double* c,
                      const double* at_y, const double* tau,
                      double step_over_pw, int n, cudaStream_t stream) {
    if (n <= 0) return;
    primalStepKernel<<<gridSize(n), kBlockSize, 0, stream>>>(
        z, z_prev, c, at_y, tau, step_over_pw, n);
}

__global__ void extrapolateKernel(double* z_bar, const double* z,
                                  const double* z_prev, double factor, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n) {
        double zb = z[j] + factor * (z[j] - z_prev[j]);
        z_bar[j] = (zb > 0.0) ? zb : 0.0;
    }
}

void launchExtrapolate(double* z_bar, const double* z, const double* z_prev,
                       double factor, int n, cudaStream_t stream) {
    if (n <= 0) return;
    extrapolateKernel<<<gridSize(n), kBlockSize, 0, stream>>>(
        z_bar, z, z_prev, factor, n);
}

// ============================================================================
// Phase 2: Reflected primal-dual kernels
// ============================================================================

__global__ void primalReflectedStepKernel(double* z, double* z_prev,
                                          double* z_reflected,
                                          const double* c, const double* at_y,
                                          const double* tau,
                                          double step_over_pw, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n) {
        double zold = z[j];
        z_prev[j] = zold;
        double grad = c[j] + at_y[j];
        double zn = zold - step_over_pw * tau[j] * grad;
        zn = (zn > 0.0) ? zn : 0.0;
        z[j] = zn;
        z_reflected[j] = 2.0 * zn - zold;
    }
}

void launchPrimalReflectedStep(double* z, double* z_prev, double* z_reflected,
                               const double* c, const double* at_y,
                               const double* tau, double step_over_pw, int n,
                               cudaStream_t stream) {
    if (n <= 0) return;
    primalReflectedStepKernel<<<gridSize(n), kBlockSize, 0, stream>>>(
        z, z_prev, z_reflected, c, at_y, tau, step_over_pw, n);
}

__global__ void dualStepReflectedKernel(double* y, const double* az_refl,
                                        const double* b, const double* sigma,
                                        double step_pw, int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) {
        y[i] += step_pw * sigma[i] * (az_refl[i] - b[i]);
    }
}

void launchDualStepReflected(double* y, const double* az_refl, const double* b,
                             const double* sigma, double step_pw, int m,
                             cudaStream_t stream) {
    if (m <= 0) return;
    dualStepReflectedKernel<<<gridSize(m), kBlockSize, 0, stream>>>(
        y, az_refl, b, sigma, step_pw, m);
}

// ============================================================================
// Phase 3: Movement/interaction reduction
// ============================================================================

// Two-pass reduction: first block-level partial sums, then final reduction.
// For simplicity we use atomicAdd for the final accumulation.

__global__ void movementInteractionKernel(
    double* d_movement, double* d_interaction,
    const double* z_next, const double* z,
    const double* at_y_next, const double* at_y,
    const double* y_next, const double* y_old,
    double primal_weight, int n, int m) {

    __shared__ double s_move[kBlockSize];
    __shared__ double s_inter[kBlockSize];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    double local_move = 0.0;
    double local_inter = 0.0;

    // Primal part: indices [0, n)
    if (gid < n) {
        double dz = z_next[gid] - z[gid];
        local_move += primal_weight * dz * dz;
        double day = at_y_next[gid] - at_y[gid];
        local_inter += dz * day;
    }

    // Dual part: indices [0, m) — handled by separate grid range
    int dual_idx = gid - n;
    if (dual_idx >= 0 && dual_idx < m) {
        double dy = y_next[dual_idx] - y_old[dual_idx];
        local_move += dy * dy / primal_weight;
    }

    s_move[tid] = local_move;
    s_inter[tid] = local_inter;
    __syncthreads();

    // Block reduction
    for (int s = kBlockSize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_move[tid] += s_move[tid + s];
            s_inter[tid] += s_inter[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(d_movement, s_move[0]);
        atomicAdd(d_interaction, s_inter[0]);
    }
}

void launchMovementInteraction(double* d_movement, double* d_interaction,
                               const double* z_next, const double* z,
                               const double* at_y_next, const double* at_y,
                               const double* y_next, const double* y,
                               double primal_weight, int n, int m,
                               cudaStream_t stream) {
    // Zero outputs first
    cudaMemsetAsync(d_movement, 0, sizeof(double), stream);
    cudaMemsetAsync(d_interaction, 0, sizeof(double), stream);

    int total = n > m ? n : (n + m);
    if (total <= 0) return;

    // We process primal+dual indices in the same grid
    int grid_elems = n + m;
    int grid = gridSize(grid_elems);
    movementInteractionKernel<<<grid, kBlockSize, 0, stream>>>(
        d_movement, d_interaction, z_next, z, at_y_next, at_y,
        y_next, y, primal_weight, n, m);
}

// ============================================================================
// Phase 4: Average accumulation / copy / zero
// ============================================================================

__global__ void accumulateAverageKernel(double* sum, const double* x,
                                        double weight, int len) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < len) {
        sum[j] += weight * x[j];
    }
}

void launchAccumulateAverage(double* sum, const double* x, double weight,
                             int len, cudaStream_t stream) {
    if (len <= 0) return;
    accumulateAverageKernel<<<gridSize(len), kBlockSize, 0, stream>>>(
        sum, x, weight, len);
}

__global__ void computeAverageKernel(double* avg, const double* sum,
                                     double inv_weight, int len) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < len) {
        avg[j] = sum[j] * inv_weight;
    }
}

void launchComputeAverage(double* avg, const double* sum, double inv_weight,
                          int len, cudaStream_t stream) {
    if (len <= 0) return;
    computeAverageKernel<<<gridSize(len), kBlockSize, 0, stream>>>(
        avg, sum, inv_weight, len);
}

__global__ void copyVectorKernel(double* dst, const double* src, int len) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < len) {
        dst[j] = src[j];
    }
}

void launchCopyVector(double* dst, const double* src, int len,
                      cudaStream_t stream) {
    if (len <= 0) return;
    copyVectorKernel<<<gridSize(len), kBlockSize, 0, stream>>>(dst, src, len);
}

void launchZeroVector(double* dst, int len, cudaStream_t stream) {
    if (len <= 0) return;
    cudaMemsetAsync(dst, 0, sizeof(double) * len, stream);
}

// ============================================================================
// Convergence reduction
// ============================================================================

// Block-level reduction to compute primal_inf, dual_inf, pobj, dobj.
// Primal infeasibility: ||Az - b||_inf * inv_b    (indices [0, m))
// Dual infeasibility: max_j |projected_rc_j| * inv_c  (indices [0, n))
// pobj: sum c[j]*z[j]  (indices [0, n))
// dobj: sum b[i]*y[i]  (indices [0, m))

__global__ void convergenceKernel(ConvergenceMetrics* out,
                                  const double* az, const double* b,
                                  const double* z, const double* at_y,
                                  const double* c, const double* y,
                                  double inv_b, double inv_c,
                                  double obj_offset,
                                  int m, int n) {
    __shared__ double s_pinf[kBlockSize];
    __shared__ double s_dinf[kBlockSize];
    __shared__ double s_pobj[kBlockSize];
    __shared__ double s_dobj[kBlockSize];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    double local_pinf = 0.0;
    double local_dinf = 0.0;
    double local_pobj = 0.0;
    double local_dobj = 0.0;

    int max_dim = (m > n) ? m : n;

    // Grid-stride loop to handle large dimensions
    for (int idx = gid; idx < max_dim; idx += gridDim.x * blockDim.x) {
        if (idx < m) {
            double resid = az[idx] - b[idx];
            double a = fabs(resid);
            if (a > local_pinf) local_pinf = a;
            local_dobj += b[idx] * y[idx];
        }
        if (idx < n) {
            double rc = c[idx] + at_y[idx];
            // Projected reduced cost for z >= 0:
            // If z[j] > 0, rc should be 0 at optimality → residual = |rc|
            // If z[j] = 0, rc should be >= 0 → residual = max(0, -rc)
            double projected = (z[idx] > 1e-12) ? fabs(rc) : fmax(0.0, -rc);
            if (projected > local_dinf) local_dinf = projected;
            local_pobj += c[idx] * z[idx];
        }
    }

    s_pinf[tid] = local_pinf;
    s_dinf[tid] = local_dinf;
    s_pobj[tid] = local_pobj;
    s_dobj[tid] = local_dobj;
    __syncthreads();

    // Block reduction
    for (int s = kBlockSize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_pinf[tid + s] > s_pinf[tid]) s_pinf[tid] = s_pinf[tid + s];
            if (s_dinf[tid + s] > s_dinf[tid]) s_dinf[tid] = s_dinf[tid + s];
            s_pobj[tid] += s_pobj[tid + s];
            s_dobj[tid] += s_dobj[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        // Use atomics to combine across blocks
        // For max: use atomicMax is only for int, so we use a compare-and-swap loop
        // For sum: use atomicAdd

        // Primal inf (max)
        unsigned long long* p_pinf = reinterpret_cast<unsigned long long*>(&out->primal_inf);
        unsigned long long old_val = *p_pinf;
        while (true) {
            double old_d = __longlong_as_double(old_val);
            if (s_pinf[0] <= old_d) break;
            unsigned long long assumed = old_val;
            old_val = atomicCAS(p_pinf, assumed,
                                __double_as_longlong(s_pinf[0]));
            if (old_val == assumed) break;
        }

        // Dual inf (max)
        unsigned long long* p_dinf = reinterpret_cast<unsigned long long*>(&out->dual_inf);
        old_val = *p_dinf;
        while (true) {
            double old_d = __longlong_as_double(old_val);
            if (s_dinf[0] <= old_d) break;
            unsigned long long assumed = old_val;
            old_val = atomicCAS(p_dinf, assumed,
                                __double_as_longlong(s_dinf[0]));
            if (old_val == assumed) break;
        }

        atomicAdd(&out->pobj, s_pobj[0]);
        atomicAdd(&out->dobj, s_dobj[0]);
    }
}

void launchConvergence(ConvergenceMetrics* d_out,
                       const double* az, const double* b,
                       const double* z, const double* at_y, const double* c,
                       const double* y,
                       double inv_b, double inv_c, double obj_offset,
                       int m, int n, cudaStream_t stream) {
    // Zero the output struct first
    cudaMemsetAsync(d_out, 0, sizeof(ConvergenceMetrics), stream);

    int max_dim = (m > n) ? m : n;
    if (max_dim <= 0) return;

    // Use enough blocks to cover the larger dimension, capped for efficiency
    int grid = gridSize(max_dim);
    if (grid > 1024) grid = 1024;

    convergenceKernel<<<grid, kBlockSize, 0, stream>>>(
        d_out, az, b, z, at_y, c, y, inv_b, inv_c, obj_offset, m, n);
}

// ============================================================================
// Unscale
// ============================================================================

__global__ void unscaleKernel(double* out, const double* x, const double* scale,
                              int len) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < len) {
        out[j] = x[j] * scale[j];
    }
}

void launchUnscale(double* z_out, const double* z, const double* scale,
                   int len, cudaStream_t stream) {
    if (len <= 0) return;
    unscaleKernel<<<gridSize(len), kBlockSize, 0, stream>>>(
        z_out, z, scale, len);
}

}  // namespace gpu
}  // namespace mipx

#endif  // MIPX_HAS_CUDA
