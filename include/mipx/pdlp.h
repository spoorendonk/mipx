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
    Int max_iter = 4000000;
    Real optimality_tol = 1e-4;
    Real primal_tol = 1e-4;
    Real dual_tol = 1e-4;

    // Step size (power iteration for spectral norm).
    Int sv_max_iter = 5000;
    Real sv_tol = 1e-4;

    // Primal weight (PI controller at restart).
    Real primal_weight = 1.0;
    bool update_primal_weight = true;
    Real pid_kp = 0.99;
    Real pid_ki = 0.01;
    Real pid_kd = 0.0;
    Real pid_i_smooth = 0.3;

    // Restart (fixed-point based).
    Real restart_sufficient_decay = 0.2;
    Real restart_necessary_decay = 0.5;
    Real restart_artificial_fraction = 0.36;
    Int termination_eval_frequency = 200;

    // Scaling.
    bool do_ruiz_scaling = true;
    Int ruiz_iterations = 10;
    bool do_pock_chambolle_scaling = true;
    bool do_bound_obj_rescaling = true;

    // GPU acceleration.
    bool use_gpu = true;
    Int gpu_min_rows = 512;
    Int gpu_min_nnz = 10000;

    // Execution.
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
    void buildScaledProblem();
    void buildTransposeCSR();
    Real estimateSpectralNorm() const;
#ifdef MIPX_HAS_CUDA
    LpResult solveGpu();
#endif

    PdlpOptions options_{};
    LpProblem original_;
    bool loaded_ = false;

    Real obj_sign_ = 1.0;
    SparseMatrix scaled_a_{0, 0};
    std::vector<Real> cscaled_;
    std::vector<Real> scaled_col_lower_, scaled_col_upper_;
    std::vector<Real> scaled_row_lower_, scaled_row_upper_;
    std::vector<Real> row_scale_, col_scale_;
    Real objective_scale_ = 1.0;
    Real constraint_scale_ = 1.0;
    std::vector<Real> sigma_base_, tau_base_;

    // Explicit A^T CSR storage (for GPU path).
    std::vector<Real> at_values_;
    std::vector<Index> at_col_indices_;
    std::vector<Index> at_row_starts_;

    std::vector<Real> primal_orig_;
    std::vector<Real> dual_orig_;
    Status status_ = Status::Error;
    Real objective_ = 0.0;
    Int iterations_ = 0;
    bool used_gpu_ = false;
};

}  // namespace mipx
