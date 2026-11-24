#pragma once

#include "mipx/core.h"
#include "mipx/sparse_matrix.h"
#include "pdlp_sum.h"

#include <algorithm>
#include <cmath>
#include <span>
#include <vector>

namespace mipx {

/// The fixed-point residual r(z) = ‖z − T(z)‖_P that PDLP's restart rules
/// compare (Lu & Yang's restarted Halpern PDHG, as in cuPDLPx's
/// compute_fixed_point_error, src/termination.cu). z is the iterate a PDHG
/// step started from and T(z) the step's pdhg point; the step overwrites z
/// with its Halpern combination, so z − T(z) is read back from the
/// reflection, reflected − pdhg = pdhg − z, as cuPDLPx does (#251).
///
/// P is the PDHG metric of mipx's step, x⁺ = proj(x − τ(c + Aᵀy)) and
/// y⁺ = prox(y + σA(2x⁺ − x)) with τ_j = ητ_base,j/ω and σ_i = ηωσ_base,i,
/// scaled by η:
///   ‖(dx, dy)‖²_P = ω Σ_j dx_j²/τ_base,j + Σ_i dy_i²/(ωσ_base,i) − 2η dxᵀAᵀdy.
/// cuPDLPx adds the cross term because its dual carries the opposite sign
/// (x − τ(c − Aᵀy)); with y → −y the two norms are the same. It folds its
/// preconditioning into A. The cross term in mipx's coordinates still uses A.
///
/// ‖(dx, dy)‖_P from its diagonal movement and cross = dxᵀAᵀdy.
/// The GPU path reduces those parts on the device.
inline Real pdlpPNorm(Real movement, Real cross, Real norm_step) {
    return std::sqrt(std::max(movement - 2.0 * norm_step * cross, 0.0));
}

/// Cap the residual's step at the initial step, and ensure positivity even if
/// the diagonal bases amplify A. With B = sqrt(Σ_base) A sqrt(T_base), the
/// Schur complement requires η‖B‖₂ ≤ 1. Use ‖B‖₂ ≤ sqrt(‖B‖₁‖B‖∞) to avoid
/// relying on the unpreconditioned power iteration as an upper bound.
inline Real pdlpPreconditionedNormStepCap(const SparseMatrix& a, std::span<const Real> tau_base,
                                          std::span<const Real> sigma_base, Real initial_step) {
    // Unit bases preserve the original cap. Use the stored bases because
    // setOptions can change the scaling flag after the matrix was loaded.
    const auto is_unit = [](Real base) { return base == 1.0; };
    if (std::ranges::all_of(tau_base, is_unit) && std::ranges::all_of(sigma_base, is_unit))
        return initial_step;

    const auto values = a.csr_values();
    const auto cols = a.csr_col_indices();
    const auto rows = a.csr_row_starts();
    std::vector<Real> sqrt_tau(tau_base.size());
    std::ranges::transform(tau_base, sqrt_tau.begin(), [](Real tau) { return std::sqrt(tau); });
    std::vector<Real> col_sums(tau_base.size(), 0.0);
    Real max_row_sum = 0.0;
    for (Index i = 0; i < a.numRows(); ++i) {
        const Real sqrt_sigma = std::sqrt(sigma_base[i]);
        Real row_sum = 0.0;
        for (Index k = rows[i]; k < rows[i + 1]; ++k) {
            const Index j = cols[k];
            const Real abs_b = std::abs(values[k]) * sqrt_tau[j] * sqrt_sigma;
            row_sum += abs_b;
            col_sums[j] += abs_b;
        }
        max_row_sum = std::max(max_row_sum, row_sum);
    }
    if (max_row_sum == 0.0)
        return initial_step;
    const Real bound = std::sqrt(max_row_sum) * std::sqrt(std::ranges::max(col_sums));
    return std::min(initial_step, 0.998 / bound);
}

/// delta_y and at_delta_y are scratch of the dual and primal sizes.
inline Real pdlpFixedPointResidual(const SparseMatrix& a, std::span<const Real> pdhg_x,
                                   std::span<const Real> reflected_x, std::span<const Real> pdhg_y,
                                   std::span<const Real> reflected_y,
                                   std::span<const Real> tau_base, std::span<const Real> sigma_base,
                                   Real primal_weight, Real norm_step, std::vector<Real>& delta_y,
                                   std::vector<Real>& at_delta_y) {
    const auto n = pdhg_x.size();
    const auto m = pdhg_y.size();
    delta_y.resize(m);
    at_delta_y.resize(n);
    pdlp_detail::ConvergenceSum dx_sq;
    for (std::size_t j = 0; j < n; ++j) {
        const Real dx = reflected_x[j] - pdhg_x[j];
        dx_sq.add(dx * dx / tau_base[j]);
    }
    pdlp_detail::ConvergenceSum dy_sq;
    for (std::size_t i = 0; i < m; ++i) {
        delta_y[i] = reflected_y[i] - pdhg_y[i];
        dy_sq.add(delta_y[i] * delta_y[i] / sigma_base[i]);
    }
    pdlp_detail::ConvergenceSum cross;
    if (m > 0 && n > 0) {
        a.multiplyTranspose(delta_y, at_delta_y);
        for (std::size_t j = 0; j < n; ++j) {
            cross.add(at_delta_y[j] * (reflected_x[j] - pdhg_x[j]));
        }
    }
    return pdlpPNorm(dx_sq.value() * primal_weight + dy_sq.value() / primal_weight, cross.value(),
                     norm_step);
}

}  // namespace mipx
