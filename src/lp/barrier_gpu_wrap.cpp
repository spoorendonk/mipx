// C++ wrappers for GPU barrier backends.
// This file is compiled by g++ (not nvcc), so it can use std::span and
// other C++23 features.  It delegates to the CUDA implementation via
// extern "C" bridge functions defined in barrier_gpu.cu.

#include "newton_solver.h"

#include <vector>

namespace mipx {

// Bridge declarations (defined in barrier_gpu.cu).
namespace gpu_detail {
extern "C" {
void* gpuCholeskyCreate();
void gpuCholeskyDestroy(void* p);
bool gpuCholeskySetup(void* p, int m, int n, int nnz, int ir_steps,
                      const int* rows, const int* cols, const double* vals);
bool gpuCholeskyFactorize(void* p, const double* z, const double* s, int n, double reg);
bool gpuCholeskySolveNewton(void* p, const double* rp, const double* rd,
                             const double* rc, double* dz, double* dy, double* ds);

void* gpuAugmentedCreate();
void gpuAugmentedDestroy(void* p);
bool gpuAugmentedSetup(void* p, int m, int n, int nnz,
                        const int* rows, const int* cols, const double* vals);
bool gpuAugmentedFactorize(void* p, const double* z, const double* s, int n, double reg);
bool gpuAugmentedSolveNewton(void* p, const double* rp, const double* rd,
                              const double* rc, double* dz, double* dy, double* ds);
}
}  // namespace gpu_detail

// ---------------------------------------------------------------------------
// GpuCholeskySolver: NewtonSolver wrapper
// ---------------------------------------------------------------------------

class GpuCholeskySolver final : public NewtonSolver {
public:
    GpuCholeskySolver() : impl_(gpu_detail::gpuCholeskyCreate()) {}
    ~GpuCholeskySolver() override { gpu_detail::gpuCholeskyDestroy(impl_); }

    bool setup(const SparseMatrix& A, Index m, Index n,
               const BarrierOptions& opts) override {
        m_ = m; n_ = n;
        // Extract CSR data as contiguous int/double arrays.
        auto rows = A.csr_row_starts();
        auto cols = A.csr_col_indices();
        auto vals = A.csr_values();
        std::vector<int> h_rows(rows.begin(), rows.end());
        std::vector<int> h_cols(cols.begin(), cols.end());
        std::vector<double> h_vals(vals.begin(), vals.end());
        return gpu_detail::gpuCholeskySetup(impl_, m, n,
                                             static_cast<int>(A.numNonzeros()),
                                             opts.ir_steps,
                                             h_rows.data(), h_cols.data(),
                                             h_vals.data());
    }

    bool factorize(std::span<const Real> z,
                   std::span<const Real> s, Real reg) override {
        return gpu_detail::gpuCholeskyFactorize(impl_, z.data(), s.data(),
                                                 n_, reg);
    }

    bool solveNewton(std::span<const Real> rp,
                     std::span<const Real> rd,
                     std::span<const Real> rc,
                     std::span<Real> dz,
                     std::span<Real> dy,
                     std::span<Real> ds) override {
        return gpu_detail::gpuCholeskySolveNewton(impl_, rp.data(), rd.data(),
                                                   rc.data(), dz.data(),
                                                   dy.data(), ds.data());
    }

private:
    void* impl_;
    int m_ = 0, n_ = 0;
};

// ---------------------------------------------------------------------------
// GpuAugmentedSolver: NewtonSolver wrapper
// ---------------------------------------------------------------------------

class GpuAugmentedSolver final : public NewtonSolver {
public:
    GpuAugmentedSolver() : impl_(gpu_detail::gpuAugmentedCreate()) {}
    ~GpuAugmentedSolver() override { gpu_detail::gpuAugmentedDestroy(impl_); }

    bool setup(const SparseMatrix& A, Index m, Index n,
               const BarrierOptions& /*opts*/) override {
        m_ = m; n_ = n;
        auto rows = A.csr_row_starts();
        auto cols = A.csr_col_indices();
        auto vals = A.csr_values();
        std::vector<int> h_rows(rows.begin(), rows.end());
        std::vector<int> h_cols(cols.begin(), cols.end());
        std::vector<double> h_vals(vals.begin(), vals.end());
        return gpu_detail::gpuAugmentedSetup(impl_, m, n,
                                              static_cast<int>(A.numNonzeros()),
                                              h_rows.data(), h_cols.data(),
                                              h_vals.data());
    }

    bool factorize(std::span<const Real> z,
                   std::span<const Real> s, Real reg) override {
        return gpu_detail::gpuAugmentedFactorize(impl_, z.data(), s.data(),
                                                   n_, reg);
    }

    bool solveNewton(std::span<const Real> rp,
                     std::span<const Real> rd,
                     std::span<const Real> rc,
                     std::span<Real> dz,
                     std::span<Real> dy,
                     std::span<Real> ds) override {
        return gpu_detail::gpuAugmentedSolveNewton(impl_, rp.data(), rd.data(),
                                                     rc.data(), dz.data(),
                                                     dy.data(), ds.data());
    }

private:
    void* impl_;
    int m_ = 0, n_ = 0;
};

// ---------------------------------------------------------------------------
// Factory functions
// ---------------------------------------------------------------------------

std::unique_ptr<NewtonSolver> createGpuCholeskySolver() {
    return std::make_unique<GpuCholeskySolver>();
}

std::unique_ptr<NewtonSolver> createGpuAugmentedSolver() {
    return std::make_unique<GpuAugmentedSolver>();
}

}  // namespace mipx
