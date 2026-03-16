// C++ wrapper for the GPU device-resident barrier solver.
// This file is compiled by g++ (not nvcc).  It delegates to the single
// gpuSolveBarrier() bridge function defined in barrier_gpu.cu.

#include "mipx/barrier.h"
#include "mipx/sparse_matrix.h"

#include <vector>

namespace mipx {

// Bridge declaration (defined in barrier_gpu.cu).
namespace gpu_detail {
extern "C" {
bool gpuSolveBarrier(
    int m, int n, int nnz,
    const int* rows, const int* cols, const double* vals,
    const double* b, const double* c,
    int max_iter, double tol, double step_fraction, double reg,
    int ir_steps, bool verbose,
    const void* stop_flag,
    bool prefer_augmented,
    double dense_col_fraction,
    double obj_offset,
    double* out_z, double* out_y, double* out_s,
    double* out_obj, int* out_status, int* out_iters);
}
const char* gpuLastBarrierError();
}  // namespace gpu_detail

bool solveBarrierGpu(const SparseMatrix& A, Index m, Index n,
                     std::span<const Real> b, std::span<const Real> c,
                     const BarrierOptions& opts, Real obj_offset,
                     bool prefer_augmented,
                     std::vector<Real>& z, std::vector<Real>& y,
                     std::vector<Real>& s, Int& iters,
                     std::string* error_msg) {
    auto rows = A.csr_row_starts();
    auto cols = A.csr_col_indices();
    auto vals = A.csr_values();
    std::vector<int> h_rows(rows.begin(), rows.end());
    std::vector<int> h_cols(cols.begin(), cols.end());
    std::vector<double> h_vals(vals.begin(), vals.end());

    z.resize(static_cast<size_t>(n));
    y.resize(static_cast<size_t>(m));
    s.resize(static_cast<size_t>(n));

    double out_obj = 0.0;
    int out_status = 1;
    int out_iters = 0;

    bool ok = gpu_detail::gpuSolveBarrier(
        m, n, static_cast<int>(A.numNonzeros()),
        h_rows.data(), h_cols.data(), h_vals.data(),
        b.data(), c.data(),
        opts.max_iter, opts.primal_dual_tol, opts.step_fraction,
        std::max(opts.regularization, 1e-12),
        opts.ir_steps, opts.verbose,
        static_cast<const void*>(opts.stop_flag),
        prefer_augmented,
        opts.dense_col_fraction,
        obj_offset,
        z.data(), y.data(), s.data(),
        &out_obj, &out_status, &out_iters);

    iters = out_iters;
    if (!ok && out_status != 0 && error_msg != nullptr && error_msg->empty()) {
        if (const char* gpu_error = gpu_detail::gpuLastBarrierError()) {
            *error_msg = gpu_error;
        } else {
            *error_msg = "Barrier GPU solve failed before convergence.";
        }
    }
    return ok || out_status == 0;
}

}  // namespace mipx
