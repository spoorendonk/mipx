#pragma once

#include "mipx/core.h"
#include "mipx/lp_problem.h"
#include "mipx/lp_solver.h"
#include "mipx/sparse_matrix.h"

#include <atomic>
#include <span>
#include <vector>

namespace mipx {

enum class PdlpOptimalityNorm {
    L2,
    LInf,
};

struct PdlpOptions {
    Int max_iter = 4000000;
    Real optimality_tol = 5e-5;
    Real primal_tol = 5e-5;
    Real dual_tol = 5e-5;
    Real max_solve_seconds = -1.0;
    PdlpOptimalityNorm optimality_norm = PdlpOptimalityNorm::L2;

    // Step size (power iteration for spectral norm).
    Int sv_max_iter = 5000;
    Real sv_tol = 1e-4;

    // Adaptive step size (Malitsky-Tam style line search).
    bool adaptive_step = false;
    Real adaptive_step_down = 0.7;  // Shrink factor when descent violated.
    Real adaptive_step_up = 1.0;    // Grow factor on successful steps.
    Real adaptive_step_min = 1e-6;  // Lower bound relative to initial step.
    Real adaptive_step_max = 1e3;   // Upper bound relative to initial step.

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

    // Matrix filtering: drop entries with |a_ij| < matrix_zero_tol before
    // scaling/solve. 0 = disabled (default).
    Real matrix_zero_tol = 0.0;

    // Scaling.
    bool do_ruiz_scaling = true;
    Int ruiz_iterations = 10;
    bool do_pock_chambolle_scaling = true;
    bool do_bound_obj_rescaling = true;

    // Preconditioner refresh: recompute sigma_base/tau_base at restart when
    // primal_weight has drifted far from its value at last preconditioning.
    // Off by default — experimental.
    bool preconditioner_refresh = false;
    Real preconditioner_refresh_ratio = 10.0;

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

    void addRows(std::span<const Index> starts, std::span<const Index> indices,
                 std::span<const Real> values, std::span<const Real> lower,
                 std::span<const Real> upper) override;
    void removeRows(std::span<const Index> rows) override;

    void setColBounds(Index col, Real lower, Real upper) override;
    void setObjective(std::span<const Real> obj) override;

    /// Set warm-start primal/dual solution from the original (unscaled) space.
    /// Vectors are scaled internally through the Ruiz/Pock-Chambolle/bound-obj
    /// pipeline at solve() time.  Silently ignored when vector dimensions do
    /// not match the loaded problem (e.g. after presolve changes the variable
    /// space).  Cleared automatically by load().
    void setWarmStart(std::span<const Real> x, std::span<const Real> y);
    void clearWarmStart();

    void setOptions(const PdlpOptions& options) { options_ = options; }
    [[nodiscard]] const PdlpOptions& options() const { return options_; }
    [[nodiscard]] bool usedGpu() const { return used_gpu_; }

private:
    void buildScaledProblem();
    void buildTransposeCSR();
    void refreshPreconditioner(Real primal_weight);
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
    Real precond_primal_weight_ = 1.0;  // primal_weight when preconditioner was last computed

    // Finite-bound arrays for GPU kernels: infinities replaced with zero
    // so that multiplication by a bound never produces ±inf in reductions.
    std::vector<Real> finite_col_lower_, finite_col_upper_;
    std::vector<Real> finite_row_lower_, finite_row_upper_;

    // Constraint type per row: 0=equality, 1=upper-only, 2=lower-only, 3=ranged.
    std::vector<Int> row_constraint_type_;

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

    // Warm-start (original/unscaled space).
    std::vector<Real> warm_start_x_;
    std::vector<Real> warm_start_y_;
    bool has_warm_start_ = false;
};

}  // namespace mipx
