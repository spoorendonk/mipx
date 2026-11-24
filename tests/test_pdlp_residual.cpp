#include "lp/pdlp_residual.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <vector>

using namespace mipx;

// One PDHG step by hand on min x1 + x2 s.t. x1 + 2 x2 >= 2, x >= 0, from
// z = (x = (1, 1), y = 0), with ω = 1, η = 0.25 (τ = σ = 0.25):
//   x⁺ = proj(x − τ(c + Aᵀy)) = (0.75, 0.75), reflected 2x⁺ − x = (0.5, 0.5);
//   v = y + σ A(2x⁺ − x) = 0.25 · 1.5 = 0.375, and the row is lower-only, so
//   y⁺ = min(0, v − σ·2) = −0.125, reflected 2y⁺ − y = −0.25.
// z − T(z) read as reflected − pdhg: dx = (−0.25, −0.25), dy = −0.125.
//   movement = ω‖dx‖² + ‖dy‖²/ω = 0.125 + 0.015625 = 0.140625
//   cross    = dxᵀAᵀdy = (−0.25)(−0.125) + (−0.25)(−0.25) = 0.09375
//   ‖z − T(z)‖²_P = 0.140625 − 2 · 0.25 · 0.09375 = 0.09375
TEST_CASE("PDLP fixed-point residual matches a hand-computed PDHG step", "[pdlp][pdlp_residual]") {
    const SparseMatrix a(1, 2, {{0, 0, 1.0}, {0, 1, 2.0}});
    const std::vector<Real> pdhg_x{0.75, 0.75};
    const std::vector<Real> reflected_x{0.5, 0.5};
    const std::vector<Real> pdhg_y{-0.125};
    const std::vector<Real> reflected_y{-0.25};
    std::vector<Real> delta_y;
    std::vector<Real> at_delta_y;
    const Real r = pdlpFixedPointResidual(a, pdhg_x, reflected_x, pdhg_y, reflected_y,
                                          std::vector<Real>{1.0, 1.0}, std::vector<Real>{1.0}, 1.0,
                                          0.25, delta_y, at_delta_y);
    CHECK(r == Catch::Approx(std::sqrt(0.09375)).epsilon(1e-15));
}

// With ω = 2 and nonuniform bases, movement is
// 2 · (0.0625 / 0.5 + 0.0625 / 0.25) + 0.015625 / (2 · 2) = 0.75390625.
// The cross term stays 0.09375, so the step subtracts 0.046875.
TEST_CASE("PDLP fixed-point residual uses the primal weight and diagonal step bases",
          "[pdlp][pdlp_residual]") {
    const SparseMatrix a(1, 2, {{0, 0, 1.0}, {0, 1, 2.0}});
    std::vector<Real> delta_y;
    std::vector<Real> at_delta_y;
    const Real r = pdlpFixedPointResidual(a, std::vector<Real>{0.75, 0.75},
                                          std::vector<Real>{0.5, 0.5}, std::vector<Real>{-0.125},
                                          std::vector<Real>{-0.25}, std::vector<Real>{0.5, 0.25},
                                          std::vector<Real>{2.0}, 2.0, 0.25, delta_y, at_delta_y);
    CHECK(r == Catch::Approx(std::sqrt(0.75390625 - 0.046875)).epsilon(1e-15));
}

TEST_CASE("PDLP caps the P-norm step when diagonal scaling amplifies the matrix",
          "[pdlp][pdlp_residual]") {
    const SparseMatrix a(1, 1, {{0, 0, -0.25}});
    const std::vector<Real> base{4.0};
    // The unscaled norm is 0.25, but the preconditioned norm is 1.
    CHECK(pdlpPreconditionedNormStepCap(a, base, base, 0.998 / 0.25) == Catch::Approx(0.998));
    CHECK(pdlpPreconditionedNormStepCap(a, base, base, 0.25) == 0.25);
    const std::vector<Real> unit{1.0};
    CHECK(pdlpPreconditionedNormStepCap(a, unit, unit, 5.0) == 5.0);
}

TEST_CASE("PDLP fixed-point residual keeps small squared movements", "[pdlp][pdlp_residual]") {
    constexpr Index n = 1025;
    const SparseMatrix a(0, n);
    const std::vector<Real> pdhg_x(n, 0.0);
    std::vector<Real> reflected_x(n, 1.0);
    reflected_x.front() = 0x1p27;
    const std::vector<Real> tau_base(n, 1.0);
    std::vector<Real> delta_y;
    std::vector<Real> at_delta_y;

    const Real r = pdlpFixedPointResidual(a, pdhg_x, reflected_x, {}, {}, tau_base, {}, 1.0, 0.25,
                                          delta_y, at_delta_y);
    CHECK(r == std::sqrt(0x1p54 + (n - 1)));
}

// The cross term enters with a minus sign: mipx's step is Chambolle-Pock with
// K = A, whose metric is [I/τ, −Aᵀ; −A, I/σ]. Only then is the PDHG step
// nonexpansive in it, which the restart rules' decay assumes.
TEST_CASE("PDLP P-norm subtracts the step-weighted cross term", "[pdlp][pdlp_residual]") {
    CHECK(pdlpPNorm(1.0, 0.5, 0.25) == Catch::Approx(std::sqrt(0.75)).epsilon(1e-15));
    CHECK(pdlpPNorm(1.0, -0.5, 0.25) == Catch::Approx(std::sqrt(1.25)).epsilon(1e-15));
    CHECK(pdlpPNorm(0.0, 1.0, 0.25) == 0.0);  // round-off below zero clamps
}
