#pragma once

#include <atomic>
#include <span>
#include <vector>

#include "mipx/core.h"
#include "mipx/lp_problem.h"
#include "mipx/lp_solver.h"

namespace mipx {

enum class BarrierAlgorithm {
    Auto,
    CpuCholesky,
    CpuAugmented,
    GpuCholesky,
    GpuAugmented,
};

struct BarrierOptions {
    Int max_iter = 100;
    Real primal_dual_tol = 1e-8;
    Real regularization = 1e-8;
    Real step_fraction = 0.995;
    BarrierAlgorithm algorithm = BarrierAlgorithm::Auto;
    Real dense_col_fraction = 0.1;
    Int ir_steps = 2;
    Int ruiz_iterations = 10;
    bool verbose = true;
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

    void addRows(std::span<const Index> starts,
                 std::span<const Index> indices,
                 std::span<const Real> values,
                 std::span<const Real> lower,
                 std::span<const Real> upper) override;
    void removeRows(std::span<const Index> rows) override;

    void setColBounds(Index col, Real lower, Real upper) override;
    void setObjective(std::span<const Real> obj) override;

    void setOptions(const BarrierOptions& options) { options_ = options; }
    [[nodiscard]] const BarrierOptions& options() const { return options_; }
    [[nodiscard]] bool usedGpu() const { return used_gpu_; }

private:
    struct OriginalColExpr {
        Real offset = 0.0;
        Index col_a = -1;
        Real coeff_a = 0.0;
        Index col_b = -1;
        Real coeff_b = 0.0;
    };

    bool buildStandardForm();
    bool solveStandardForm(std::vector<Real>& z, std::vector<Real>& y,
                           std::vector<Real>& s, Int& iters);
    void reconstructOriginalPrimals(const std::vector<Real>& z);
    bool checkOriginalPrimalFeasibility(std::span<const Real> x) const;

    BarrierOptions options_{};
    LpProblem original_;
    bool loaded_ = false;
    bool transformed_ok_ = false;
    bool transformed_infeasible_ = false;

    SparseMatrix aeq_{0, 0};
    std::vector<Real> beq_;
    std::vector<Real> cstd_;
    std::vector<OriginalColExpr> col_expr_;
    Real std_obj_offset_ = 0.0;

    // Ruiz scaling factors.
    std::vector<Real> row_scale_;
    std::vector<Real> col_scale_;

    std::vector<Real> primal_orig_;
    std::vector<Real> dual_eq_;
    std::vector<Real> reduced_costs_std_;
    Status status_ = Status::Error;
    Real objective_ = 0.0;
    Int iterations_ = 0;
    bool used_gpu_ = false;
};

// GPU device-resident barrier solver (available when compiled with MIPX_HAS_CUDSS).
#ifdef MIPX_HAS_CUDSS
bool solveBarrierGpu(const SparseMatrix& A, Index m, Index n,
                     std::span<const Real> b, std::span<const Real> c,
                     const BarrierOptions& opts, Real obj_offset,
                     bool prefer_augmented,
                     std::vector<Real>& z, std::vector<Real>& y,
                     std::vector<Real>& s, Int& iters);
#endif

}  // namespace mipx
