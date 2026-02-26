#pragma once

#include <atomic>
#include <span>
#include <vector>

#include "mipx/core.h"
#include "mipx/lp_problem.h"
#include "mipx/lp_solver.h"
#include "mipx/sparse_matrix.h"

namespace mipx {

struct PdlpOptions {
    Int max_iter = 20000;
    Real optimality_tol = 1e-6;
    Real primal_tol = 1e-6;
    Real dual_tol = 1e-6;
    Int major_iteration = 64;

    // Adaptive step-size update.
    Real initial_step_size_scaling = 0.95;
    Real step_growth = 1.03;
    Real step_reduction = 0.7;
    Real min_step_size = 1e-6;
    Real max_step_size = 1e2;

    // Primal/dual balancing and restart.
    Real primal_weight = 1.0;
    Real primal_weight_update_smoothing = 0.5;
    Real restart_sufficient_reduction = 0.2;
    Real restart_necessary_reduction = 0.85;
    Real extrapolation_factor = 1.0;
    bool update_primal_weight = true;

    // Scaling/preconditioning.
    bool do_ruiz_scaling = true;
    Int ruiz_iterations = 8;
    bool do_pock_chambolle_scaling = true;
    Real default_alpha_pock_chambolle_rescaling = 1.0;

    // Execution.
    bool use_gpu = true;
    Int gpu_min_rows = 512;
    Int gpu_min_nnz = 10000;
    bool verbose = true;
    const std::atomic<bool>* stop_flag = nullptr;
};

class PdlpSolver : public LpSolver {
public:
    PdlpSolver() = default;

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

    void setOptions(const PdlpOptions& options) { options_ = options; }
    [[nodiscard]] const PdlpOptions& options() const { return options_; }
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
    void buildScaledProblem();
    bool solveStandardForm(std::vector<Real>& z_unscaled, std::vector<Real>& y_unscaled,
                           Int& iters);
    void reconstructOriginalPrimals(std::span<const Real> z_unscaled);
    bool checkOriginalPrimalFeasibility(std::span<const Real> x) const;

    PdlpOptions options_{};
    LpProblem original_;
    bool loaded_ = false;
    bool transformed_ok_ = false;
    bool transformed_infeasible_ = false;

    SparseMatrix aeq_{0, 0};
    std::vector<Real> beq_;
    std::vector<Real> cstd_;
    std::vector<OriginalColExpr> col_expr_;
    Real std_obj_offset_ = 0.0;

    SparseMatrix scaled_aeq_{0, 0};
    std::vector<Real> bscaled_;
    std::vector<Real> cscaled_;
    std::vector<Real> row_scale_;
    std::vector<Real> col_scale_;
    std::vector<Real> sigma_base_;
    std::vector<Real> tau_base_;

    std::vector<Real> primal_orig_;
    std::vector<Real> dual_eq_;
    std::vector<Real> reduced_costs_std_;
    Status status_ = Status::Error;
    Real objective_ = 0.0;
    Int iterations_ = 0;
    bool used_gpu_ = false;
};

}  // namespace mipx
