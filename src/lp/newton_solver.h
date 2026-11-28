#pragma once

#include "mipx/barrier.h"
#include "mipx/deadline.h"
#include "mipx/sparse_matrix.h"

#include <cstddef>
#include <memory>
#include <span>
#include <string>

namespace mipx {

/// Abstract Newton-step solver for the barrier IPM.
///
/// Each backend (CPU Cholesky, CPU Augmented, GPU Cholesky, GPU Augmented)
/// implements this interface.  The shared Mehrotra predictor-corrector loop
/// in barrier.cpp calls setup() once, then factorize()+solveNewton() per
/// iteration.
class NewtonSolver {
public:
    virtual ~NewtonSolver() = default;

    /// One-time setup: analyze sparsity, allocate storage.
    virtual bool setup(const SparseMatrix& A, Index m, Index n, const BarrierOptions& opts) = 0;

    /// Per-iteration: form and factor the linear system from current z, s.
    virtual bool factorize(std::span<const Real> z, std::span<const Real> s, Real reg) = 0;

    /// Solve Newton system: given residuals (rp, rd, rc), produce search
    /// directions (dz, dy, ds).
    virtual bool solveNewton(std::span<const Real> rp, std::span<const Real> rd,
                             std::span<const Real> rc, std::span<Real> dz, std::span<Real> dy,
                             std::span<Real> ds) = 0;

    /// Why setup() or factorize() last failed, when the backend can say
    /// (e.g. the factor does not fit in memory); empty otherwise.
    [[nodiscard]] const std::string& lastError() const { return error_; }

    /// The solve's time limit (#275), with the caller's stop flag. A CPU
    /// backend checks it inside its factorization, so one long factorize()
    /// cannot overrun it (or a stop) by minutes, and returns false once it
    /// has passed; the sparse normal equations also inside setup()'s
    /// ordering and analysis (#347). Set it before setup().
    void setDeadline(const Deadline& deadline) { deadline_ = deadline; }
    [[nodiscard]] const Deadline& deadline() const { return deadline_; }

    /// The factor's size once set up, where the backend counts it (the sizes:
    /// sparse normal equations; the flop estimates: both normal-equations
    /// backends; zeros elsewhere), for tests, diagnostics and #293's K.
    struct FactorStats {
        std::size_t nnz_l = 0;
        Index dense_cols = 0;     // columns of A left out of the factor
        Index schur_order = 0;    // dense columns + shifted pivots, last factorize
        Index schur_dropped = 0;  // of those, rows dropped as dependent in M
        // Flop estimates of one factorize() and one solveNewton(), for the
        // number of centrality correctors worth a factorization (#293); zero
        // where the backend does not estimate them.
        double factor_flops = 0.0;
        double solve_flops = 0.0;
    };
    [[nodiscard]] virtual FactorStats factorStats() const { return {}; }

protected:
    std::string error_;
    Deadline deadline_;
};

// CPU backends (always available).
std::unique_ptr<NewtonSolver> createCpuCholeskySolver();
// max_schur_order bounds the dense-column Schur complement (#314): order^2
// reals and one solve per column each iteration. Past it, the dense columns
// go back into the factor, or, if that factor does not fit, S may grow to
// twice the cap before the factorization fails. Smaller only in tests.
std::unique_ptr<NewtonSolver> createCpuSparseCholeskySolver(Index max_schur_order = 4096);
std::unique_ptr<NewtonSolver> createCpuAugmentedSolver();

// Compare barrier reformulations with a bounded symbolic factorization for
// small patterns and a linear-time degree estimate on large ones.
double estimateSparseNewtonWork(const SparseMatrix& matrix, const BarrierOptions& options,
                                bool augmented, bool peel_dense_columns);

// The denominator of the homogeneous self-dual loop's dtau (#370), given
// u = (u_z, u_y) solved from (b, c, 0) at the iterate (z, s, tau, kappa):
// max(b'u_y - c'u_z, q) + kappa / tau, with q = u_z' S Z^-1 u_z. For an exact
// solve b'u_y - c'u_z = q; the backends' regularization and pivot shifts add
// u_y'(reg I + shifts) u_y >= 0, and rounding near the optimum can take the
// computed value below q, even below zero, where q bounds it.
Real hsdGapDenominator(std::span<const Real> b, std::span<const Real> c, std::span<const Real> uz,
                       std::span<const Real> uy, std::span<const Real> z, std::span<const Real> s,
                       Real tau, Real kappa);

// One direction of the homogeneous self-dual loop (#370, #335) from the
// backend's two solves at the iterate's (tau, kappa): p = (p_z, p_y, p_s) for
// (eta rp, eta rd, rc) and u for (b, c, 0), with denom = hsdGapDenominator.
// dtau = (eta_rg + c'p_z - b'p_y + rk / tau) / denom,
// dkappa = (rk - kappa dtau) / tau and d = p + dtau u, where eta_rg is eta
// times the gap residual (zero for a Gondzio corrector, whose rk is t_k).
// Returns false if dtau, dkappa or an entry of d is not finite.
bool hsdDirection(std::span<const Real> b, std::span<const Real> c, std::span<const Real> pz,
                  std::span<const Real> py, std::span<const Real> ps, std::span<const Real> uz,
                  std::span<const Real> uy, std::span<const Real> us, Real tau, Real kappa,
                  Real denom, Real eta_rg, Real rk, std::span<Real> dz, std::span<Real> dy,
                  std::span<Real> ds, Real& dtau, Real& dkappa);

// The HSD loop's infeasibility test (#371) at an iterate (z, y, s, tau,
// kappa) with az = A z, at_y = A'y, c'z and b'y, and mu_rel its mu relative
// to the starting one (scale-invariant, see the definition):
// PrimalInfeasible or DualInfeasible on a Farkas ray within tol, else
// Failed. No verdict unless tau <= tol kappa and mu_rel <= tol.
BarrierIpmOutcome hsdInfeasibility(std::span<const Real> b, std::span<const Real> c,
                                   std::span<const Real> az, std::span<const Real> at_y, Real tau,
                                   Real kappa, Real mu_rel, Real ctz, Real bty, Real tol,
                                   bool verbose = false);

// Whether `ray`, one multiplier per row of `lp`, proves `lp` infeasible in its
// own data, rechecked as getDualRay()'s contract leaves to a consumer that
// needs rigor (#408): in long double, alpha - beta above round-off of every
// term that makes it up. A multiplier whose side is infinite is dropped, so
// the proof is that of the rest of the ray.
bool farkasRayProvesInfeasible(const LpProblem& lp, std::span<const Real> ray);

// GPU backends now use the device-resident path (solveBarrierGpu in barrier.h).
#ifdef MIPX_HAS_CUDSS
// Also report the eliminated column count of the backend that finished.
bool solveBarrierGpuWithStats(const SparseMatrix& a, Index m, Index n, std::span<const Real> b,
                              std::span<const Real> c, const BarrierOptions& opts, Real obj_offset,
                              bool prefer_augmented, std::vector<Real>& z, std::vector<Real>& y,
                              std::vector<Real>& s, Int& iters, std::string* error_msg,
                              Real dualized_obj_sign, Index* eliminated_columns);
#endif

}  // namespace mipx
