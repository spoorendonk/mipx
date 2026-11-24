#pragma once

#include "mipx/core.h"
#include "mipx/sparse_matrix.h"

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
/// y⁺ = prox(y + σA(2x⁺ − x)) with τ = η/ω, σ = ηω, scaled by η:
///   ‖(dx, dy)‖²_P = ω‖dx‖² + ‖dy‖²/ω − 2η dxᵀAᵀdy.
/// cuPDLPx adds the cross term because its dual carries the opposite sign
/// (x − τ(c − Aᵀy)); with y → −y the two norms are the same. The step is
/// norm_step = min(step, base_step), as in current cuPDLPx (P is indefinite
/// for a step boosted past the global one; the port-source commit 83f9ffd
/// still used the step itself). The Pock-Chambolle per-entry step bases
/// stay out of the weighting, as they did before: mipx applies them on top of
/// a rescaled matrix whose bases are near 1, and cuPDLPx has none.
///
/// ‖(dx, dy)‖_P from its parts: movement = ω‖dx‖² + ‖dy‖²/ω and
/// cross = dxᵀAᵀdy. The GPU path reduces the parts on the device.
inline Real pdlpPNorm(Real movement, Real cross, Real norm_step) {
    return std::sqrt(std::max(movement - 2.0 * norm_step * cross, 0.0));
}

/// delta_y and at_delta_y are scratch of the dual and primal sizes.
inline Real pdlpFixedPointResidual(const SparseMatrix& a, std::span<const Real> pdhg_x,
                                   std::span<const Real> reflected_x, std::span<const Real> pdhg_y,
                                   std::span<const Real> reflected_y, Real primal_weight,
                                   Real norm_step, std::vector<Real>& delta_y,
                                   std::vector<Real>& at_delta_y) {
    const auto n = pdhg_x.size();
    const auto m = pdhg_y.size();
    delta_y.resize(m);
    at_delta_y.resize(n);
    Real dx_sq = 0.0;
    for (std::size_t j = 0; j < n; ++j) {
        const Real dx = reflected_x[j] - pdhg_x[j];
        dx_sq += dx * dx;
    }
    Real dy_sq = 0.0;
    for (std::size_t i = 0; i < m; ++i) {
        delta_y[i] = reflected_y[i] - pdhg_y[i];
        dy_sq += delta_y[i] * delta_y[i];
    }
    Real cross = 0.0;
    if (m > 0 && n > 0) {
        a.multiplyTranspose(delta_y, at_delta_y);
        for (std::size_t j = 0; j < n; ++j) {
            cross += at_delta_y[j] * (reflected_x[j] - pdhg_x[j]);
        }
    }
    return pdlpPNorm(dx_sq * primal_weight + dy_sq / primal_weight, cross, norm_step);
}

}  // namespace mipx
