#include "pdlp_kernels.cuh"

#include <cuda_runtime.h>
#include <cmath>

namespace mipx {

static constexpr int kBlockSize = 256;

static inline int gridSize(int n) {
    return (n + kBlockSize - 1) / kBlockSize;
}

Index convergenceMetricBlocks(Index count) {
    return count / kBlockSize + (count % kBlockSize != 0);
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
    const Real* __restrict__ row_lower, const Real* __restrict__ row_upper,
    const Int* __restrict__ row_constraint_type)
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

    // Constraint type: 0=equality, 1=upper-only, 2=lower-only, 3=ranged.
    Int ct = row_constraint_type[i];
    Real py;
    if (ct == 0) {
        // Equality constraint.
        py = v - sigma * rl;
    } else if (ct == 1) {
        // Upper-bound only.
        py = v - sigma * ru;
        if (py < 0.0) py = 0.0;
    } else if (ct == 2) {
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
    const Int* row_constraint_type,
    cudaStream_t stream)
{
    if (m <= 0) return;
    dualHalpernStepKernel<<<gridSize(m), kBlockSize, 0, stream>>>(
        m, d_lambda, d_step, d_primal_weight,
        current_y, initial_y, pdhg_y, reflected_y,
        a_xrefl, sigma_base, row_lower, row_upper, row_constraint_type);
}

// ---------------------------------------------------------------------------
// Convergence metrics — column pass
// ---------------------------------------------------------------------------

__global__ void convergenceMetricsColKernel(
    Index n,
    const Real* __restrict__ pdhg_x, const Real* __restrict__ initial_x,
    const Real* __restrict__ reflected_x,
    const Real* __restrict__ cscaled, const Real* __restrict__ at_y,
    const Real* __restrict__ col_lower, const Real* __restrict__ col_upper,
    const Real* __restrict__ finite_col_lower, const Real* __restrict__ finite_col_upper,
    const Real* __restrict__ dual_resid_scale,
    const Real* __restrict__ at_delta_y,
    const Real* __restrict__ tau_base,
    Real primal_weight,
    GpuConvergenceMetrics* __restrict__ partials)
{
    __shared__ Real s_pobj[kBlockSize];
    __shared__ Real s_dobj_col[kBlockSize];
    __shared__ Real s_dual_resid_sq[kBlockSize];
    __shared__ Real s_dual_resid_max[kBlockSize];
    __shared__ Real s_movement[kBlockSize];
    __shared__ Real s_cross[kBlockSize];
    __shared__ Real s_dist_sq[kBlockSize];

    int tid = threadIdx.x;
    int j = blockIdx.x * blockDim.x + tid;

    Real pobj = 0.0, dobj_col = 0.0, drsq = 0.0, drmax = 0.0;
    Real mov = 0.0, cross = 0.0, dist = 0.0;

    if (j < n) {
        Real xj = pdhg_x[j];
        Real dx = xj - initial_x[j];
        // z − T(z), primal part: reflected − pdhg = pdhg − z (#251).
        Real fx = reflected_x[j] - xj;
        Real lb = col_lower[j];
        Real ub = col_upper[j];

        pobj = cscaled[j] * xj;

        // Dual residual: reduced cost projected to feasible sign.
        // When lb is ±inf, xj <= lb + eps is always false (IEEE arithmetic),
        // so no finiteness guard is needed on the original bound arrays.
        Real grad = cscaled[j] + at_y[j];
        Real dr = 0.0;
        if (xj <= lb + 1e-12) {
            if (grad < 0.0) dr = grad;
        } else if (xj >= ub - 1e-12) {
            if (grad > 0.0) dr = grad;
        } else {
            dr = grad;
        }
        Real dr_orig = dr * dual_resid_scale[j];
        drsq = dr_orig * dr_orig;
        drmax = fabs(dr_orig);

        // Dual objective column contribution (branchless).
        // finite_col_lower/upper have inf replaced with 0, so the
        // multiplication naturally yields 0 for infinite bounds.
        Real pg_pos = fmax(grad, 0.0);
        Real pg_neg = fmin(grad, 0.0);
        dobj_col = pg_pos * finite_col_lower[j] + pg_neg * finite_col_upper[j];

        mov = fx * fx / tau_base[j] * primal_weight;
        cross = at_delta_y[j] * fx;
        dist = dx * dx;
    }

    s_pobj[tid] = pobj;
    s_dobj_col[tid] = dobj_col;
    s_dual_resid_sq[tid] = drsq;
    s_dual_resid_max[tid] = drmax;
    s_movement[tid] = mov;
    s_cross[tid] = cross;
    s_dist_sq[tid] = dist;
    __syncthreads();

    for (int s = kBlockSize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_pobj[tid] += s_pobj[tid + s];
            s_dobj_col[tid] += s_dobj_col[tid + s];
            s_dual_resid_sq[tid] += s_dual_resid_sq[tid + s];
            s_dual_resid_max[tid] = fmax(s_dual_resid_max[tid], s_dual_resid_max[tid + s]);
            s_movement[tid] += s_movement[tid + s];
            s_cross[tid] += s_cross[tid + s];
            s_dist_sq[tid] += s_dist_sq[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        GpuConvergenceMetrics result{};
        result.primal_obj = s_pobj[0];
        result.dual_obj_col = s_dobj_col[0];
        result.dual_resid_sq = s_dual_resid_sq[0];
        result.dual_resid_max = s_dual_resid_max[0];
        result.fpe_movement = s_movement[0];
        result.fpe_cross = s_cross[0];
        result.primal_dist_sq = s_dist_sq[0];
        partials[blockIdx.x] = result;
    }
}

void launchConvergenceMetricsCol(
    Index n,
    const Real* pdhg_x, const Real* initial_x,
    const Real* reflected_x,
    const Real* cscaled, const Real* at_y,
    const Real* col_lower, const Real* col_upper,
    const Real* finite_col_lower, const Real* finite_col_upper,
    const Real* dual_resid_scale,
    const Real* at_delta_y,
    const Real* tau_base,
    Real primal_weight,
    GpuConvergenceMetrics* d_partials,
    cudaStream_t stream)
{
    if (n <= 0) return;
    convergenceMetricsColKernel<<<convergenceMetricBlocks(n), kBlockSize, 0, stream>>>(
        n, pdhg_x, initial_x, reflected_x, cscaled, at_y, col_lower, col_upper,
        finite_col_lower, finite_col_upper, dual_resid_scale,
        at_delta_y, tau_base, primal_weight, d_partials);
}

// ---------------------------------------------------------------------------
// Convergence metrics — row pass
// ---------------------------------------------------------------------------

__global__ void convergenceMetricsRowKernel(
    Index m,
    const Real* __restrict__ pdhg_y, const Real* __restrict__ initial_y,
    const Real* __restrict__ reflected_y,
    const Real* __restrict__ ax,
    const Real* __restrict__ row_lower, const Real* __restrict__ row_upper,
    const Real* __restrict__ finite_row_lower, const Real* __restrict__ finite_row_upper,
    const Real* __restrict__ primal_resid_scale,
    const Real* __restrict__ sigma_base,
    Real primal_weight,
    GpuConvergenceMetrics* __restrict__ partials)
{
    __shared__ Real s_presid_sq[kBlockSize];
    __shared__ Real s_presid_max[kBlockSize];
    __shared__ Real s_dobj_row[kBlockSize];
    __shared__ Real s_movement[kBlockSize];
    __shared__ Real s_dist_sq[kBlockSize];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    Real presid = 0.0, presid_max = 0.0, dobj_row = 0.0, mov = 0.0, dist = 0.0;

    if (i < m) {
        Real axi = ax[i];
        Real rl = row_lower[i];
        Real ru = row_upper[i];

        // Primal residual.
        Real clamped = axi;
        if (clamped < rl) clamped = rl;
        if (clamped > ru) clamped = ru;
        Real violation = (axi - clamped) * primal_resid_scale[i];
        presid = violation * violation;
        presid_max = fabs(violation);

        // Dual objective row contribution (branchless).
        // finite_row_lower/upper have inf replaced with 0, so the
        // multiplication naturally yields 0 for infinite bounds.
        Real yi = pdhg_y[i];
        Real pyi_pos = fmax(yi, 0.0);
        Real pyi_neg = fmin(yi, 0.0);
        dobj_row = -pyi_pos * finite_row_upper[i] - pyi_neg * finite_row_lower[i];

        // Fixed-point residual movement, dual part: reflected − pdhg (#251).
        Real fy = reflected_y[i] - yi;
        mov = fy * fy / sigma_base[i] / primal_weight;
        // Distance since the last restart, for the primal weight.
        Real dy = yi - initial_y[i];
        dist = dy * dy;
    }

    s_presid_sq[tid] = presid;
    s_presid_max[tid] = presid_max;
    s_dobj_row[tid] = dobj_row;
    s_movement[tid] = mov;
    s_dist_sq[tid] = dist;
    __syncthreads();

    for (int s = kBlockSize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_presid_sq[tid] += s_presid_sq[tid + s];
            s_presid_max[tid] = fmax(s_presid_max[tid], s_presid_max[tid + s]);
            s_dobj_row[tid] += s_dobj_row[tid + s];
            s_movement[tid] += s_movement[tid + s];
            s_dist_sq[tid] += s_dist_sq[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        GpuConvergenceMetrics result{};
        result.primal_resid_sq = s_presid_sq[0];
        result.primal_resid_max = s_presid_max[0];
        result.dual_obj_row = s_dobj_row[0];
        result.fpe_movement = s_movement[0];
        result.dual_dist_sq = s_dist_sq[0];
        partials[blockIdx.x] = result;
    }
}

void launchConvergenceMetricsRow(
    Index m,
    const Real* pdhg_y, const Real* initial_y,
    const Real* reflected_y,
    const Real* ax,
    const Real* row_lower, const Real* row_upper,
    const Real* finite_row_lower, const Real* finite_row_upper,
    const Real* primal_resid_scale,
    const Real* sigma_base,
    Real primal_weight,
    GpuConvergenceMetrics* d_partials,
    cudaStream_t stream)
{
    if (m <= 0) return;
    convergenceMetricsRowKernel<<<convergenceMetricBlocks(m), kBlockSize, 0, stream>>>(
        m, pdhg_y, initial_y, reflected_y, ax, row_lower, row_upper,
        finite_row_lower, finite_row_upper, primal_resid_scale,
        sigma_base, primal_weight, d_partials);
}

// Reduce the partials in fixed pairs at every level, including across blocks.
__global__ void convergenceMetricsReduceKernel(Index count,
                                               const GpuConvergenceMetrics* __restrict__ input,
                                               GpuConvergenceMetrics* __restrict__ output) {
    __shared__ GpuConvergenceMetrics partials[kBlockSize];
    const int tid = threadIdx.x;
    const Index i = blockIdx.x * blockDim.x + tid;
    partials[tid] = i < count ? input[i] : GpuConvergenceMetrics{};
    __syncthreads();

    for (int stride = kBlockSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            auto& left = partials[tid];
            const auto& right = partials[tid + stride];
            left.primal_resid_sq += right.primal_resid_sq;
            left.dual_resid_sq += right.dual_resid_sq;
            left.primal_resid_max = fmax(left.primal_resid_max, right.primal_resid_max);
            left.dual_resid_max = fmax(left.dual_resid_max, right.dual_resid_max);
            left.primal_obj += right.primal_obj;
            left.dual_obj_col += right.dual_obj_col;
            left.dual_obj_row += right.dual_obj_row;
            left.fpe_movement += right.fpe_movement;
            left.fpe_cross += right.fpe_cross;
            left.primal_dist_sq += right.primal_dist_sq;
            left.dual_dist_sq += right.dual_dist_sq;
        }
        __syncthreads();
    }
    if (tid == 0)
        output[blockIdx.x] = partials[0];
}

void launchReduceConvergenceMetrics(Index block_count, GpuConvergenceMetrics* d_partials,
                                    GpuConvergenceMetrics* d_result, cudaStream_t stream) {
    auto* input = d_partials;
    auto* scratch = d_partials + block_count;
    while (block_count > kBlockSize) {
        const Index groups = convergenceMetricBlocks(block_count);
        convergenceMetricsReduceKernel<<<groups, kBlockSize, 0, stream>>>(block_count, input,
                                                                         scratch);
        auto* previous = input;
        input = scratch;
        scratch = previous;
        block_count = groups;
    }
    convergenceMetricsReduceKernel<<<1, kBlockSize, 0, stream>>>(block_count, input, d_result);
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

// ---------------------------------------------------------------------------
// Probe
// ---------------------------------------------------------------------------

__global__ void probeKernel() {}

cudaError_t probeKernels() {
    probeKernel<<<1, 1>>>();
    const cudaError_t launched = cudaGetLastError();
    const cudaError_t ran = cudaDeviceSynchronize();
    return launched != cudaSuccess ? launched : ran;
}

}  // namespace mipx
