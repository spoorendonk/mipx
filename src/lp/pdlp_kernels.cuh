#pragma once

#ifdef MIPX_HAS_CUDA

#include <cuda_runtime.h>

namespace mipx {
namespace gpu {

// ---------------------------------------------------------------------------
// Phase 1: Core PDLP iteration kernels
// ---------------------------------------------------------------------------

// Fused dual step: y[i] += step_pw * sigma[i] * (az_bar[i] - b[i])
void launchDualStep(double* y, const double* az_bar, const double* b,
                    const double* sigma, double step_pw, int m,
                    cudaStream_t stream);

// Fused primal step with projection:
//   z_prev[j] = z[j]
//   z[j] = max(0, z[j] - step_over_pw * tau[j] * (c[j] + at_y[j]))
void launchPrimalStep(double* z, double* z_prev, const double* c,
                      const double* at_y, const double* tau,
                      double step_over_pw, int n, cudaStream_t stream);

// Extrapolation with projection:
//   z_bar[j] = max(0, z[j] + factor * (z[j] - z_prev[j]))
void launchExtrapolate(double* z_bar, const double* z, const double* z_prev,
                       double factor, int n, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Phase 2: Reflected primal-dual kernels
// ---------------------------------------------------------------------------

// Primal step + reflection in one pass:
//   z_prev[j] = z[j]  (saved before update)
//   z_next = max(0, z[j] - step_over_pw * tau[j] * (c[j] + at_y[j]))
//   z_reflected[j] = 2 * z_next - z_prev[j]
//   z[j] = z_next
void launchPrimalReflectedStep(double* z, double* z_prev, double* z_reflected,
                               const double* c, const double* at_y,
                               const double* tau, double step_over_pw, int n,
                               cudaStream_t stream);

// Dual step (equality constraints, no projection):
//   y[i] += step_pw * sigma[i] * (az_refl[i] - b[i])
void launchDualStepReflected(double* y, const double* az_refl, const double* b,
                             const double* sigma, double step_pw, int m,
                             cudaStream_t stream);

// ---------------------------------------------------------------------------
// Phase 3: Adaptive step size (movement/interaction) kernels
// ---------------------------------------------------------------------------

// Compute movement and interaction norms in a single reduction:
//   movement = pw * ||z_next - z||^2 + (1/pw) * ||y_next - y||^2
//   interaction = 2 * |<z_next - z, at_y_next - at_y>|
// Output: d_movement[0], d_interaction[0]
void launchMovementInteraction(double* d_movement, double* d_interaction,
                               const double* z_next, const double* z,
                               const double* at_y_next, const double* at_y,
                               const double* y_next, const double* y,
                               double primal_weight, int n, int m,
                               cudaStream_t stream);

// ---------------------------------------------------------------------------
// Phase 4: KKT restart / weighted average kernels
// ---------------------------------------------------------------------------

// Accumulate into weighted average: sum[j] += weight * x[j]
void launchAccumulateAverage(double* sum, const double* x, double weight,
                             int len, cudaStream_t stream);

// Compute average from accumulated sum: avg[j] = sum[j] / total_weight
void launchComputeAverage(double* avg, const double* sum, double inv_weight,
                          int len, cudaStream_t stream);

// Copy a device vector: dst[j] = src[j]
void launchCopyVector(double* dst, const double* src, int len,
                      cudaStream_t stream);

// Zero a device vector: dst[j] = 0
void launchZeroVector(double* dst, int len, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Convergence kernel
// ---------------------------------------------------------------------------

struct ConvergenceMetrics {
    double primal_inf;   // ||Az - b||_inf * inv_b
    double dual_inf;     // ||projected_rc||_inf * inv_c
    double pobj;         // obj_offset + c^T z
    double dobj;         // obj_offset + b^T y  (or -inf if dual infeasible)
};

// Batched convergence computation — single reduction.
// Computes primal_inf, dual_inf, pobj, dobj.
void launchConvergence(ConvergenceMetrics* d_out,
                       const double* az, const double* b,
                       const double* z, const double* at_y, const double* c,
                       const double* y,
                       double inv_b, double inv_c, double obj_offset,
                       int m, int n, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Unscale solution for download
// ---------------------------------------------------------------------------

// z_out[j] = z[j] * col_scale[j]
void launchUnscale(double* z_out, const double* z, const double* scale,
                   int len, cudaStream_t stream);

}  // namespace gpu
}  // namespace mipx

#endif  // MIPX_HAS_CUDA
