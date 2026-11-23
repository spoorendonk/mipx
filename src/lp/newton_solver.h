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
    /// has passed.
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

// GPU backends now use the device-resident path (solveBarrierGpu in barrier.h).
// No NewtonSolver GPU factories needed.

}  // namespace mipx
