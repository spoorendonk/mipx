#pragma once

#include "mipx/barrier_presolve.h"
#include "mipx/core.h"
#include "mipx/deadline.h"
#include "mipx/lp_problem.h"
#include "mipx/lp_solver.h"

#include <atomic>
#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace mipx {

enum class BarrierAlgorithm {
    // On the CPU: the dense normal equations, or the augmented system when
    // the NE's m^3/3 is over ten times the augmented factor's flops from a
    // symbolic analysis (HiGHS HiPO's ratio); either one hands over to the
    // other when it fails (#140). With cuDSS, the GPU backend, which still
    // picks NE or augmented by a density proxy and switches at run time.
    Auto,
    CpuCholesky,   // normal equations, dense LL'
    CpuAugmented,  // augmented system, sparse LDL'
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

enum class BarrierDualize { Off, On, Auto };

// The starting point of the CPU Mehrotra loop (#144).
enum class BarrierInitialPoint {
    // Mehrotra's: on Netlib neither start beats the other on the whole, so
    // Auto keeps the one that was there (#144)
    Auto,
    Mehrotra,  // least-squares z and (y, s), shifted positive and centred
    // HiGHS HiPO's variant of Mehrotra's: shifts of at least 1, centring sums
    // from 1, a guard on b = 0 or c = 0, and a bounded column's reduced cost
    // split in halves
    Hipo,
};

// How an interior-point run ended. NearOptimal is a run that stopped short of
// primal_dual_tol but whose best iterate is close enough to start crossover.
// TimeLimit is a run stopped by BarrierOptions::max_solve_seconds.
enum class BarrierIpmOutcome {
    Optimal,
    NearOptimal,
    Failed,
    TimeLimit,
    // The homogeneous self-dual loop's certificates (#371): Farkas rays of
    // the standard form, so the LP is infeasible, or its dual is.
    PrimalInfeasible,
    DualInfeasible,
};

struct BarrierOptions {
    Int max_iter = 100;
    // The IPM stops at scaled primal and dual infeasibility and relative
    // duality gap x'z / (1 + |c'x|) all below this (#253).
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
    // arrays of M itself against the same cap. 0 uses the memory available at
    // setup where the platform reports it (on Linux, bounded by cgroup headroom).
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
    // The CPU IPM on the homogeneous self-dual model (#292; Xu, Hung & Ye
    // 1996, as in Andersen & Andersen 2000) instead of Mehrotra's loop on the
    // LP itself, with gondzio_correctors too (#335). Opt-in, CPU backends
    // only: on, the GPU path is skipped. Combines with implicit_upper_bounds
    // (#443), keeping upper-bound pairs in the homogeneous loop.
    bool homogeneous = false;
    // Starting point of the CPU Mehrotra loop (#144). Opt-in, CPU backends
    // only: a value other than Auto or Mehrotra skips the GPU path, whose
    // start is its own. The homogeneous loop starts all complementarity
    // variables at 1 whatever this says.
    BarrierInitialPoint initial_point = BarrierInitialPoint::Auto;
    // Wall-clock limit on solve() in seconds, crossover included; below 0:
    // none (#275). The CPU IPM checks it every iteration and inside its
    // factorizations, and solve() then returns Status::TimeLimit; crossover
    // gets what is left. The clock starts in solve(): load()'s presolve is
    // outside it. The GPU loop checks it at the start of each iteration (a
    // watcher raises its stop flag), so one long GPU factorization can still
    // overrun it.
    Real max_solve_seconds = -1.0;
    bool verbose = true;
    bool crossover = true;
    bool presolve = true;
    // Solve the compact dual of the presolved LP. Auto compares the estimated
    // Newton-system costs of both standard forms.
    BarrierDualize dualize = BarrierDualize::Off;
    // Raised by the caller to stop the solve, which then returns
    // Status::IterLimit. The CPU IPM checks it every iteration and inside its
    // factorizations, as it does max_solve_seconds.
    const std::atomic<bool>* stop_flag = nullptr;
    // Opt-in cap on columns left out of the GPU normal-equations factor.
    // Among columns above the GPU density threshold, the densest go first.
    Int gpu_ne_max_dense_cols = 0;
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

    // Rebuilds the transform of a loaded model when its construction or
    // equilibration settings change.
    void setOptions(const BarrierOptions& options);
    [[nodiscard]] const BarrierOptions& options() const { return options_; }
    // Size of the standard form the IPM solves (rows = normal-equations
    // dimension), for tests and diagnostics.
    [[nodiscard]] Index standardFormRows() const { return aeq_.numRows(); }
    [[nodiscard]] Index standardFormCols() const { return aeq_.numCols(); }
    [[nodiscard]] bool usedDualization() const { return dualized_; }
    [[nodiscard]] bool usedGpu() const { return used_gpu_; }
    // The Newton-system backend the last solve tried last: the one Auto
    // chose, or the other after it failed (#140). Auto itself when none ran
    // (an empty standard form, an error before it). On the GPU, cuDSS may still
    // switch from the normal equations to the augmented system inside a
    // GpuCholesky run.
    [[nodiscard]] BarrierAlgorithm usedAlgorithm() const { return used_algorithm_; }
    // Gondzio correctors accepted over the last IPM run, for tests.
    [[nodiscard]] Int gondzioCorrections() const { return gondzio_corrections_; }
    // Whether the last IPM run was the homogeneous self-dual loop, for tests.
    [[nodiscard]] bool usedHomogeneous() const { return used_homogeneous_; }
    // Simplex iterations of the last solve's crossover, cold fallback
    // included; 0 when it did not run (or its basis guess was optimal).
    [[nodiscard]] Int crossoverIterations() const { return crossover_iters_; }
    /// How the simplex cleanup ran in the most recent barrier solve.
    struct CrossoverStats {
        bool attempted = false;
        bool singular_recovery = false;
        bool cold_fallback = false;
        // A barrier stop short of tolerance ran a cold dual simplex after a
        // warm optimum, for a rigorous Farkas ray (#408).
        bool infeasibility_probe = false;
        Int probe_iterations = 0;
    };
    [[nodiscard]] CrossoverStats crossoverStats() const { return crossover_stats_; }
    [[nodiscard]] const std::string& lastError() const { return last_error_; }

private:
    struct OriginalColExpr {
        Real offset = 0.0;
        Index col_a = -1;
        Real coeff_a = 0.0;
        Index col_b = -1;
        Real coeff_b = 0.0;
    };

    struct DualForm {
        SparseMatrix matrix{0, 0};
        std::vector<Real> rhs;
        std::vector<Real> cost;
        Real offset = 0.0;
        std::vector<OriginalColExpr> primals;
        std::vector<OriginalColExpr> row_prices;
    };

    static std::optional<DualForm> buildDualForm(const LpProblem& problem);
    void useDualForm(DualForm form);
    bool buildStandardForm();
    void rebuildTransform();
    void clearTransformedSolution();
    BarrierIpmOutcome solveStandardForm(std::vector<Real>& z, std::vector<Real>& y,
                                        std::vector<Real>& s, Int& iters);
    void reconstructOriginalSolution(const std::vector<Real>& z, const std::vector<Real>& y);
    bool checkOriginalPrimalFeasibility(std::span<const Real> x) const;
    bool runCrossover(bool probe_infeasibility);

    BarrierOptions options_{};
    LpProblem original_;
    LpProblem presolved_;  // After barrier presolve (may alias original_).
    BarrierPresolver presolver_;
    bool presolve_active_ = false;
    bool loaded_ = false;
    bool transformed_ok_ = false;
    bool transformed_infeasible_ = false;
    bool dualized_ = false;

    SparseMatrix aeq_{0, 0};
    std::vector<Real> beq_;
    std::vector<Real> cstd_;
    // Upper bound of each standard-form column, +inf where none. Finite only
    // under implicit_upper_bounds; otherwise the bounds are rows of aeq_.
    std::vector<Real> ustd_;
    std::vector<OriginalColExpr> col_expr_;
    std::vector<OriginalColExpr> dual_row_expr_;
    Real std_obj_offset_ = 0.0;
    Real scaled_obj_ = 0.0;  // objective computed before Ruiz unscaling

    // Ruiz scaling factors.
    std::vector<Real> row_scale_;
    std::vector<Real> col_scale_;

    std::vector<Real> primal_orig_;
    std::vector<Real> dual_orig_;
    std::vector<Real> dualized_reduced_costs_;
    std::vector<Real> reduced_costs_std_;
    std::vector<BasisStatus> basis_orig_;
    Status status_ = Status::Error;
    Real objective_ = 0.0;
    Int iterations_ = 0;
    Int crossover_iters_ = 0;
    CrossoverStats crossover_stats_;
    Status crossover_status_ =
        Status::Error;  // Infeasible/Unbounded if either crossover run proved it
    // Crossover's simplex ended Infeasible on a ray farkasRayProvesInfeasible
    // accepts: the barrier reports Infeasible, not a contradiction (#408).
    bool crossover_proved_infeasible_ = false;
    bool used_gpu_ = false;
    BarrierAlgorithm used_algorithm_ = BarrierAlgorithm::Auto;
    Int gondzio_corrections_ = 0;
    bool used_homogeneous_ = false;
    Deadline deadline_;  // of the current solve(), from max_solve_seconds
    std::string last_error_;
};

// GPU device-resident barrier solver (available when compiled with MIPX_HAS_CUDSS).
#ifdef MIPX_HAS_CUDSS
bool solveBarrierGpu(const SparseMatrix& A, Index m, Index n, std::span<const Real> b,
                     std::span<const Real> c, const BarrierOptions& opts, Real obj_offset,
                     bool prefer_augmented, std::vector<Real>& z, std::vector<Real>& y,
                     std::vector<Real>& s, Int& iters, std::string* error_msg = nullptr,
                     Real dualized_obj_sign = 0.0);
#endif

}  // namespace mipx
