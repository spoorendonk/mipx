#pragma once

#include "mipx/barrier_presolve.h"
#include "mipx/core.h"
#include "mipx/lp_problem.h"
#include "mipx/lp_solver.h"

#include <atomic>
#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace mipx {

enum class BarrierAlgorithm {
    Auto,
    CpuCholesky,  // normal equations, dense LL'
    CpuAugmented,
    GpuCholesky,
    GpuAugmented,
    CpuSparseCholesky,  // normal equations, sparse supernodal LL' (#290); opt-in
};

enum class BarrierOrdering {
    Auto,   // Pick based on problem size/structure.
    Amd,    // Approximate minimum degree.
    Nd,     // Nested dissection.
    CuDSS,  // cuDSS internal ordering (GPU only).
};

// How an interior-point run ended. NearOptimal is a run that stopped short of
// primal_dual_tol but whose best iterate is close enough to start crossover.
enum class BarrierIpmOutcome {
    Optimal,
    NearOptimal,
    Failed,
};

struct BarrierOptions {
    Int max_iter = 100;
    Real primal_dual_tol = 1e-8;
    Real regularization = 1e-8;
    Real step_fraction = 0.995;
    BarrierAlgorithm algorithm = BarrierAlgorithm::Auto;
    // Fill-reducing ordering for the sparse CPU paths (augmented LDL' and
    // sparse normal equations). The dense normal-equations path has no
    // sparsity to preserve, and the GPU path delegates ordering to cuDSS, so
    // this is ignored by both.
    BarrierOrdering ordering = BarrierOrdering::Auto;
    // Threads of the BLAS behind the sparse normal-equations factorization.
    Int threads = 1;
    // Cap in bytes on a CPU normal-equations factor (dense or sparse); one
    // above it ends the solve with Status::Error. The sparse path counts the
    // arrays of M itself against the same cap. 0: the memory available when
    // the backend is set up (MemAvailable, or the cgroup's headroom if lower).
    std::size_t ne_max_bytes = 0;
    // Fraction of m above which a column of A counts as dense. The sparse
    // normal-equations path takes max(dense_col_fraction * m, 16) nonzeros as
    // its bar and leaves at most ne_max_dense_cols such columns (the densest)
    // out of its Cholesky factor, solving with them by a Schur complement
    // (#314); ne_max_dense_cols = 0 keeps them all in.
    Real dense_col_fraction = 0.1;
    Int ne_max_dense_cols = 100;
    Int ir_steps = 2;
    Int ruiz_iterations = 10;
    // Upper bounds inside the IPM (#299): a column with finite bounds, and a
    // ranged row's slack, stays one standard-form column with an upper bound,
    // its slack and dual carried by the IPM loop, instead of two columns and
    // a row w + t = u - l. Opt-in, CPU backends only: with it on, a model with
    // upper bounds skips the GPU path.
    bool implicit_upper_bounds = false;
    // Gondzio's multiple centrality correctors (#293) after Mehrotra's
    // corrector in the CPU IPM loop: at most this many per iteration, fewer
    // when the backend's factorization is cheap next to its solve. 0: off.
    // Opt-in, CPU backends only: above 0 the GPU path is skipped, an explicit
    // GPU backend included (reported when verbose).
    Int gondzio_correctors = 0;
    bool verbose = true;
    bool crossover = true;
    bool presolve = true;
    const std::atomic<bool>* stop_flag = nullptr;
};

class BarrierSolver : public LpSolver {
public:
    BarrierSolver() = default;

    void load(const LpProblem& problem) override;
    LpResult solve() override;

    Status getStatus() const override { return status_; }
    Real getObjective() const override { return objective_; }

    std::vector<Real> getPrimalValues() const override;
    std::vector<Real> getDualValues() const override;
    std::vector<Real> getReducedCosts() const override;

    std::vector<BasisStatus> getBasis() const override;
    void setBasis(std::span<const BasisStatus> basis) override;

    void addRows(std::span<const Index> starts, std::span<const Index> indices,
                 std::span<const Real> values, std::span<const Real> lower,
                 std::span<const Real> upper) override;
    void removeRows(std::span<const Index> rows) override;
    void addCols(std::span<const Index> starts, std::span<const Index> indices,
                 std::span<const Real> values, std::span<const Real> obj,
                 std::span<const Real> lower, std::span<const Real> upper) override;
    void removeCols(std::span<const Index> cols) override;

    void setColBounds(Index col, Real lower, Real upper) override;
    void setRowBounds(Index row, Real lower, Real upper) override;
    void setObjective(std::span<const Real> obj) override;
    void setColObjective(Index col, Real cost) override;

    // Rebuilds the standard form of a loaded model when the options change
    // its shape (implicit_upper_bounds).
    void setOptions(const BarrierOptions& options);
    [[nodiscard]] const BarrierOptions& options() const { return options_; }
    // Size of the standard form the IPM solves (rows = normal-equations
    // dimension), for tests and diagnostics.
    [[nodiscard]] Index standardFormRows() const { return aeq_.numRows(); }
    [[nodiscard]] Index standardFormCols() const { return aeq_.numCols(); }
    [[nodiscard]] bool usedGpu() const { return used_gpu_; }
    // Gondzio correctors accepted over the last IPM run, for tests.
    [[nodiscard]] Int gondzioCorrections() const { return gondzio_corrections_; }
    [[nodiscard]] const std::string& lastError() const { return last_error_; }

private:
    struct OriginalColExpr {
        Real offset = 0.0;
        Index col_a = -1;
        Real coeff_a = 0.0;
        Index col_b = -1;
        Real coeff_b = 0.0;
    };

    bool buildStandardForm();
    void rebuildTransform();
    void clearTransformedSolution();
    BarrierIpmOutcome solveStandardForm(std::vector<Real>& z, std::vector<Real>& y,
                                        std::vector<Real>& s, Int& iters);
    void reconstructOriginalPrimals(const std::vector<Real>& z);
    bool checkOriginalPrimalFeasibility(std::span<const Real> x) const;
    bool runCrossover();

    BarrierOptions options_{};
    LpProblem original_;
    LpProblem presolved_;  // After barrier presolve (may alias original_).
    BarrierPresolver presolver_;
    bool presolve_active_ = false;
    bool loaded_ = false;
    bool transformed_ok_ = false;
    bool transformed_infeasible_ = false;

    SparseMatrix aeq_{0, 0};
    std::vector<Real> beq_;
    std::vector<Real> cstd_;
    // Upper bound of each standard-form column, +inf where none. Finite only
    // under implicit_upper_bounds; otherwise the bounds are rows of aeq_.
    std::vector<Real> ustd_;
    std::vector<OriginalColExpr> col_expr_;
    Real std_obj_offset_ = 0.0;
    Real scaled_obj_ = 0.0;  // objective computed before Ruiz unscaling

    // Ruiz scaling factors.
    std::vector<Real> row_scale_;
    std::vector<Real> col_scale_;

    std::vector<Real> primal_orig_;
    std::vector<Real> dual_eq_;
    std::vector<Real> reduced_costs_std_;
    std::vector<BasisStatus> basis_orig_;
    Status status_ = Status::Error;
    Real objective_ = 0.0;
    Int iterations_ = 0;
    Int crossover_iters_ = 0;
    Status crossover_status_ =
        Status::Error;  // Infeasible/Unbounded if either crossover run proved it
    bool used_gpu_ = false;
    Int gondzio_corrections_ = 0;
    std::string last_error_;
};

// GPU device-resident barrier solver (available when compiled with MIPX_HAS_CUDSS).
#ifdef MIPX_HAS_CUDSS
bool solveBarrierGpu(const SparseMatrix& A, Index m, Index n, std::span<const Real> b,
                     std::span<const Real> c, const BarrierOptions& opts, Real obj_offset,
                     bool prefer_augmented, std::vector<Real>& z, std::vector<Real>& y,
                     std::vector<Real>& s, Int& iters, std::string* error_msg = nullptr);
#endif

}  // namespace mipx
