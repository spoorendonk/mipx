#pragma once

#include <cstdint>

namespace mipx {

using Real = double;
using Int = int;
using Index = int;

struct GpuConvergenceMetrics {
    Real primal_resid_sq;
    Real dual_resid_sq;
    Real primal_resid_max;
    Real dual_resid_max;
    Real primal_obj;
    Real dual_obj_col;
    Real dual_obj_row;
    Real fpe_movement;
    Real fpe_interaction;
    Real primal_dist_sq;
    Real dual_dist_sq;
};

// Compute lambda on device: lambda = (inner_count + k_offset) / (inner_count + k_offset + 1).
void launchComputeLambda(Real* d_lambda, const Int* d_inner_count, Int k_offset,
                         cudaStream_t stream);

// Halpern PDHG step kernels.
void launchPrimalHalpernStep(
    Index n, const Real* d_lambda, const Real* d_step, const Real* d_primal_weight,
    Real* current_x, const Real* initial_x,
    Real* pdhg_x, Real* reflected_x,
    const Real* cscaled, const Real* at_y,
    const Real* tau_base,
    const Real* col_lower, const Real* col_upper,
    cudaStream_t stream);

void launchDualHalpernStep(
    Index m, const Real* d_lambda, const Real* d_step, const Real* d_primal_weight,
    Real* current_y, const Real* initial_y,
    Real* pdhg_y, Real* reflected_y,
    const Real* a_xrefl, const Real* sigma_base,
    const Real* row_lower, const Real* row_upper,
    cudaStream_t stream);

// Convergence metric reduction.
void launchConvergenceMetricsCol(
    Index n,
    const Real* pdhg_x, const Real* initial_x,
    const Real* cscaled, const Real* at_y,
    const Real* col_lower, const Real* col_upper,
    const Real* dual_resid_scale,
    const Real* at_delta_y,
    Real primal_weight, Real step,
    GpuConvergenceMetrics* d_metrics,
    cudaStream_t stream);

void launchConvergenceMetricsRow(
    Index m,
    const Real* pdhg_y, const Real* initial_y,
    const Real* ax,
    const Real* row_lower, const Real* row_upper,
    const Real* primal_resid_scale,
    Real primal_weight,
    GpuConvergenceMetrics* d_metrics,
    cudaStream_t stream);

// Utility kernels.
void launchRestartCopy(
    Index n, Index m,
    const Real* pdhg_x, const Real* pdhg_y,
    Real* initial_x, Real* initial_y,
    Real* current_x, Real* current_y,
    cudaStream_t stream);

void launchSubtract(
    Index count, const Real* a, const Real* b, Real* result,
    cudaStream_t stream);

void launchZeroMetrics(GpuConvergenceMetrics* d_metrics, cudaStream_t stream);

// Reduction helpers for power iteration.
Real launchDotProduct(Index n, const Real* a, const Real* b,
                      Real* d_scratch, cudaStream_t stream);

void launchAxpby(Index n, Real alpha, const Real* x, Real beta, Real* y,
                 cudaStream_t stream);

void launchScale(Index n, Real alpha, Real* x, cudaStream_t stream);

}  // namespace mipx
