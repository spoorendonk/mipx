#include "pdlp_kernels.cuh"

#include <cuda_runtime.h>
#include <cmath>
#include <cfloat>

namespace mipx {

static constexpr int kBlockSize = 256;
static constexpr Real kInf = 1e30;

static inline int gridSize(int n) {
    return (n + kBlockSize - 1) / kBlockSize;
}

__device__ inline bool devIsFinite(Real v) {
    return v > -kInf && v < kInf;
}

// ---------------------------------------------------------------------------
// Compute lambda on device
// ---------------------------------------------------------------------------

__global__ void computeLambdaKernel(Real* d_lambda, const Int* d_inner_count,
                                     Int k_offset) {
    Int k = *d_inner_count + k_offset;
    *d_lambda = static_cast<Real>(k) / static_cast<Real>(k + 1);
}

void launchComputeLambda(Real* d_lambda, const Int* d_inner_count,
                         Int k_offset, cudaStream_t stream) {
    computeLambdaKernel<<<1, 1, 0, stream>>>(d_lambda, d_inner_count, k_offset);
}

// ---------------------------------------------------------------------------
// Primal Halpern step
// ---------------------------------------------------------------------------

__global__ void primalHalpernStepKernel(
    Index n, const Real* __restrict__ d_lambda,
    const Real* __restrict__ d_step, const Real* __restrict__ d_primal_weight,
    Real* __restrict__ current_x, const Real* __restrict__ initial_x,
    Real* __restrict__ pdhg_x, Real* __restrict__ reflected_x,
    const Real* __restrict__ cscaled, const Real* __restrict__ at_y,
    const Real* __restrict__ tau_base,
    const Real* __restrict__ col_lower, const Real* __restrict__ col_upper)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    Real lambda = *d_lambda;
    Real step = *d_step;
    Real primal_weight = *d_primal_weight;
    Real tau = step * tau_base[j] / primal_weight;
    Real temp = current_x[j] - tau * (cscaled[j] + at_y[j]);
    Real lb = col_lower[j];
    Real ub = col_upper[j];
    Real px = temp;
    if (px < lb) px = lb;
    if (px > ub) px = ub;
    pdhg_x[j] = px;
    Real rx = 2.0 * px - current_x[j];
    reflected_x[j] = rx;
    current_x[j] = lambda * rx + (1.0 - lambda) * initial_x[j];
}

void launchPrimalHalpernStep(
    Index n, const Real* d_lambda, const Real* d_step, const Real* d_primal_weight,
    Real* current_x, const Real* initial_x,
    Real* pdhg_x, Real* reflected_x,
    const Real* cscaled, const Real* at_y,
    const Real* tau_base,
    const Real* col_lower, const Real* col_upper,
    cudaStream_t stream)
{
    if (n <= 0) return;
    primalHalpernStepKernel<<<gridSize(n), kBlockSize, 0, stream>>>(
        n, d_lambda, d_step, d_primal_weight,
        current_x, initial_x, pdhg_x, reflected_x,
        cscaled, at_y, tau_base, col_lower, col_upper);
}

// ---------------------------------------------------------------------------
// Dual Halpern step
// ---------------------------------------------------------------------------

__global__ void dualHalpernStepKernel(
    Index m, const Real* __restrict__ d_lambda,
    const Real* __restrict__ d_step, const Real* __restrict__ d_primal_weight,
    Real* __restrict__ current_y, const Real* __restrict__ initial_y,
    Real* __restrict__ pdhg_y, Real* __restrict__ reflected_y,
    const Real* __restrict__ a_xrefl, const Real* __restrict__ sigma_base,
    const Real* __restrict__ row_lower, const Real* __restrict__ row_upper)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;

    Real lambda = *d_lambda;
    Real step = *d_step;
    Real primal_weight = *d_primal_weight;
    Real sigma = step * sigma_base[i] * primal_weight;
    Real v = current_y[i] + sigma * a_xrefl[i];
    Real rl = row_lower[i];
    Real ru = row_upper[i];

    Real py;
    if (rl == ru) {
        // Equality constraint.
        py = v - sigma * rl;
    } else if (!devIsFinite(rl)) {
        // Upper-bound only.
        py = v - sigma * ru;
        if (py < 0.0) py = 0.0;
    } else if (!devIsFinite(ru)) {
        // Lower-bound only.
        py = v - sigma * rl;
        if (py > 0.0) py = 0.0;
    } else {
        // Ranged constraint.
        if (v >= sigma * ru)
            py = v - sigma * ru;
        else if (v <= sigma * rl)
            py = v - sigma * rl;
        else
            py = 0.0;
    }
    pdhg_y[i] = py;
    Real ry = 2.0 * py - current_y[i];
    reflected_y[i] = ry;
    current_y[i] = lambda * ry + (1.0 - lambda) * initial_y[i];
}

void launchDualHalpernStep(
    Index m, const Real* d_lambda, const Real* d_step, const Real* d_primal_weight,
    Real* current_y, const Real* initial_y,
    Real* pdhg_y, Real* reflected_y,
    const Real* a_xrefl, const Real* sigma_base,
    const Real* row_lower, const Real* row_upper,
    cudaStream_t stream)
{
    if (m <= 0) return;
    dualHalpernStepKernel<<<gridSize(m), kBlockSize, 0, stream>>>(
        m, d_lambda, d_step, d_primal_weight,
        current_y, initial_y, pdhg_y, reflected_y,
        a_xrefl, sigma_base, row_lower, row_upper);
}

// ---------------------------------------------------------------------------
// Convergence metrics — column pass
// ---------------------------------------------------------------------------

__global__ void convergenceMetricsColKernel(
    Index n,
    const Real* __restrict__ pdhg_x, const Real* __restrict__ initial_x,
    const Real* __restrict__ cscaled, const Real* __restrict__ at_y,
    const Real* __restrict__ col_lower, const Real* __restrict__ col_upper,
    const Real* __restrict__ at_delta_y,
    Real primal_weight, Real step,
    GpuConvergenceMetrics* __restrict__ metrics)
{
    __shared__ Real s_pobj[kBlockSize];
    __shared__ Real s_dobj_col[kBlockSize];
    __shared__ Real s_dual_resid_sq[kBlockSize];
    __shared__ Real s_movement[kBlockSize];
    __shared__ Real s_interaction[kBlockSize];
    __shared__ Real s_dist_sq[kBlockSize];

    int tid = threadIdx.x;
    int j = blockIdx.x * blockDim.x + tid;

    Real pobj = 0.0, dobj_col = 0.0, drsq = 0.0;
    Real mov = 0.0, interact = 0.0, dist = 0.0;

    if (j < n) {
        Real xj = pdhg_x[j];
        Real dx = xj - initial_x[j];
        Real lb = col_lower[j];
        Real ub = col_upper[j];

        pobj = cscaled[j] * xj;

        Real grad = cscaled[j] + at_y[j];
        Real dr = 0.0;
        if (xj <= lb + 1e-12 && devIsFinite(lb)) {
            if (grad < 0.0) dr = grad;
        } else if (xj >= ub - 1e-12 && devIsFinite(ub)) {
            if (grad > 0.0) dr = grad;
        } else {
            dr = grad;
        }
        drsq = dr * dr;

        Real pg = grad;
        if (!devIsFinite(lb)) pg = (pg < 0.0) ? pg : 0.0;
        if (!devIsFinite(ub)) pg = (pg > 0.0) ? pg : 0.0;
        if (pg > 0.0 && devIsFinite(lb))
            dobj_col = pg * lb;
        else if (pg < 0.0 && devIsFinite(ub))
            dobj_col = pg * ub;

        mov = dx * dx * primal_weight;
        interact = at_delta_y[j] * dx * 2.0 * step;
        dist = dx * dx;
    }

    s_pobj[tid] = pobj;
    s_dobj_col[tid] = dobj_col;
    s_dual_resid_sq[tid] = drsq;
    s_movement[tid] = mov;
    s_interaction[tid] = interact;
    s_dist_sq[tid] = dist;
    __syncthreads();

    for (int s = kBlockSize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_pobj[tid] += s_pobj[tid + s];
            s_dobj_col[tid] += s_dobj_col[tid + s];
            s_dual_resid_sq[tid] += s_dual_resid_sq[tid + s];
            s_movement[tid] += s_movement[tid + s];
            s_interaction[tid] += s_interaction[tid + s];
            s_dist_sq[tid] += s_dist_sq[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(&metrics->primal_obj, s_pobj[0]);
        atomicAdd(&metrics->dual_obj_col, s_dobj_col[0]);
        atomicAdd(&metrics->dual_resid_sq, s_dual_resid_sq[0]);
        atomicAdd(&metrics->fpe_movement, s_movement[0]);
        atomicAdd(&metrics->fpe_interaction, s_interaction[0]);
        atomicAdd(&metrics->primal_dist_sq, s_dist_sq[0]);
    }
}

void launchConvergenceMetricsCol(
    Index n,
    const Real* pdhg_x, const Real* initial_x,
    const Real* cscaled, const Real* at_y,
    const Real* col_lower, const Real* col_upper,
    const Real* at_delta_y,
    Real primal_weight, Real step,
    GpuConvergenceMetrics* d_metrics,
    cudaStream_t stream)
{
    if (n <= 0) return;
    convergenceMetricsColKernel<<<gridSize(n), kBlockSize, 0, stream>>>(
        n, pdhg_x, initial_x, cscaled, at_y, col_lower, col_upper,
        at_delta_y, primal_weight, step, d_metrics);
}

// ---------------------------------------------------------------------------
// Convergence metrics — row pass
// ---------------------------------------------------------------------------

__global__ void convergenceMetricsRowKernel(
    Index m,
    const Real* __restrict__ pdhg_y, const Real* __restrict__ initial_y,
    const Real* __restrict__ ax,
    const Real* __restrict__ row_lower, const Real* __restrict__ row_upper,
    Real primal_weight,
    GpuConvergenceMetrics* __restrict__ metrics)
{
    __shared__ Real s_presid_sq[kBlockSize];
    __shared__ Real s_dobj_row[kBlockSize];
    __shared__ Real s_movement[kBlockSize];
    __shared__ Real s_dist_sq[kBlockSize];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    Real presid = 0.0, dobj_row = 0.0, mov = 0.0, dist = 0.0;

    if (i < m) {
        Real axi = ax[i];
        Real rl = row_lower[i];
        Real ru = row_upper[i];

        // Primal residual.
        Real clamped = axi;
        if (clamped < rl) clamped = rl;
        if (clamped > ru) clamped = ru;
        Real violation = axi - clamped;
        presid = violation * violation;

        // Dual objective row contribution.
        Real yi = pdhg_y[i];
        Real pyi = yi;
        if (!devIsFinite(ru)) pyi = (pyi < 0.0) ? pyi : 0.0;
        if (!devIsFinite(rl)) pyi = (pyi > 0.0) ? pyi : 0.0;
        if (pyi > 0.0 && devIsFinite(ru))
            dobj_row = -pyi * ru;
        else if (pyi < 0.0 && devIsFinite(rl))
            dobj_row = -pyi * rl;

        // FPE movement (dual part).
        Real dy = yi - initial_y[i];
        mov = dy * dy / primal_weight;
        dist = dy * dy;
    }

    s_presid_sq[tid] = presid;
    s_dobj_row[tid] = dobj_row;
    s_movement[tid] = mov;
    s_dist_sq[tid] = dist;
    __syncthreads();

    for (int s = kBlockSize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_presid_sq[tid] += s_presid_sq[tid + s];
            s_dobj_row[tid] += s_dobj_row[tid + s];
            s_movement[tid] += s_movement[tid + s];
            s_dist_sq[tid] += s_dist_sq[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(&metrics->primal_resid_sq, s_presid_sq[0]);
        atomicAdd(&metrics->dual_obj_row, s_dobj_row[0]);
        atomicAdd(&metrics->fpe_movement, s_movement[0]);
        atomicAdd(&metrics->dual_dist_sq, s_dist_sq[0]);
    }
}

void launchConvergenceMetricsRow(
    Index m,
    const Real* pdhg_y, const Real* initial_y,
    const Real* ax,
    const Real* row_lower, const Real* row_upper,
    Real primal_weight,
    GpuConvergenceMetrics* d_metrics,
    cudaStream_t stream)
{
    if (m <= 0) return;
    convergenceMetricsRowKernel<<<gridSize(m), kBlockSize, 0, stream>>>(
        m, pdhg_y, initial_y, ax, row_lower, row_upper,
        primal_weight, d_metrics);
}

// ---------------------------------------------------------------------------
// Restart copy kernel
// ---------------------------------------------------------------------------

__global__ void restartCopyKernelImpl(
    Index n, Index m,
    const Real* __restrict__ pdhg_x, const Real* __restrict__ pdhg_y,
    Real* __restrict__ initial_x, Real* __restrict__ initial_y,
    Real* __restrict__ current_x, Real* __restrict__ current_y)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        initial_x[idx] = pdhg_x[idx];
        current_x[idx] = pdhg_x[idx];
    }
    if (idx < m) {
        initial_y[idx] = pdhg_y[idx];
        current_y[idx] = pdhg_y[idx];
    }
}

void launchRestartCopy(
    Index n, Index m,
    const Real* pdhg_x, const Real* pdhg_y,
    Real* initial_x, Real* initial_y,
    Real* current_x, Real* current_y,
    cudaStream_t stream)
{
    int total = (n > m) ? n : m;
    if (total <= 0) return;
    restartCopyKernelImpl<<<gridSize(total), kBlockSize, 0, stream>>>(
        n, m, pdhg_x, pdhg_y, initial_x, initial_y, current_x, current_y);
}

// ---------------------------------------------------------------------------
// Subtract kernel: result = a - b
// ---------------------------------------------------------------------------

__global__ void subtractKernelImpl(
    Index count, const Real* __restrict__ a, const Real* __restrict__ b,
    Real* __restrict__ result)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) result[i] = a[i] - b[i];
}

void launchSubtract(
    Index count, const Real* a, const Real* b, Real* result,
    cudaStream_t stream)
{
    if (count <= 0) return;
    subtractKernelImpl<<<gridSize(count), kBlockSize, 0, stream>>>(
        count, a, b, result);
}

// ---------------------------------------------------------------------------
// Zero metrics
// ---------------------------------------------------------------------------

void launchZeroMetrics(GpuConvergenceMetrics* d_metrics, cudaStream_t stream) {
    cudaMemsetAsync(d_metrics, 0, sizeof(GpuConvergenceMetrics), stream);
}

// ---------------------------------------------------------------------------
// Dot product reduction (for power iteration)
// ---------------------------------------------------------------------------

__global__ void dotProductKernel(
    Index n, const Real* __restrict__ a, const Real* __restrict__ b,
    Real* __restrict__ result)
{
    __shared__ Real s_partial[kBlockSize];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    s_partial[tid] = (i < n) ? a[i] * b[i] : 0.0;
    __syncthreads();

    for (int s = kBlockSize / 2; s > 0; s >>= 1) {
        if (tid < s) s_partial[tid] += s_partial[tid + s];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(result, s_partial[0]);
}

Real launchDotProduct(Index n, const Real* a, const Real* b,
                      Real* d_scratch, cudaStream_t stream)
{
    if (n <= 0) return 0.0;
    cudaMemsetAsync(d_scratch, 0, sizeof(Real), stream);
    dotProductKernel<<<gridSize(n), kBlockSize, 0, stream>>>(n, a, b, d_scratch);
    Real result;
    cudaMemcpyAsync(&result, d_scratch, sizeof(Real), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    return result;
}

// ---------------------------------------------------------------------------
// axpby: y = alpha*x + beta*y
// ---------------------------------------------------------------------------

__global__ void axpbyKernel(
    Index n, Real alpha, const Real* __restrict__ x, Real beta,
    Real* __restrict__ y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = alpha * x[i] + beta * y[i];
}

void launchAxpby(Index n, Real alpha, const Real* x, Real beta, Real* y,
                 cudaStream_t stream)
{
    if (n <= 0) return;
    axpbyKernel<<<gridSize(n), kBlockSize, 0, stream>>>(n, alpha, x, beta, y);
}

// ---------------------------------------------------------------------------
// Scale: x *= alpha
// ---------------------------------------------------------------------------

__global__ void scaleKernel(Index n, Real alpha, Real* __restrict__ x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= alpha;
}

void launchScale(Index n, Real alpha, Real* x, cudaStream_t stream) {
    if (n <= 0) return;
    scaleKernel<<<gridSize(n), kBlockSize, 0, stream>>>(n, alpha, x);
}

}  // namespace mipx
