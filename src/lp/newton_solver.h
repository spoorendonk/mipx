#pragma once

#include "mipx/barrier.h"
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

    /// The factor's size once set up, where the backend counts it (the sparse
    /// normal equations; zeros elsewhere), for tests and diagnostics.
    struct FactorStats {
        std::size_t nnz_l = 0;
        Index dense_cols = 0;  // columns of A left out of the factor
    };
    [[nodiscard]] virtual FactorStats factorStats() const { return {}; }

protected:
    std::string error_;
};

// CPU backends (always available).
std::unique_ptr<NewtonSolver> createCpuCholeskySolver();
std::unique_ptr<NewtonSolver> createCpuSparseCholeskySolver();
std::unique_ptr<NewtonSolver> createCpuAugmentedSolver();

// GPU backends now use the device-resident path (solveBarrierGpu in barrier.h).
// No NewtonSolver GPU factories needed.

}  // namespace mipx
