#pragma once

#include "mipx/core.h"
#include "mipx/deadline.h"
#include "mipx/lp_problem.h"
#include "mipx/lp_solver.h"
#include "mipx/lu.h"
#include "mipx/sparse_matrix.h"
#include "mipx/work_units.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace mipx {

struct LpDualDegeneracy {
    Int eligible = 0;
    Int zero_reduced_cost = 0;
};

namespace test {
struct DualSimplexPricingAccess;
}

/// Pricing strategy for the dual simplex CHUZR.
enum class PricingStrategy : int {
    Devex = 0,         // Devex only (original behavior)
    SteepestEdge = 1,  // DSE only (no fallback)
    Auto = 2,          // DSE with automatic Devex fallback (default)
};

enum class ScalingStrategy {
    Auto,           // Select based on coefficient range.
    Equilibration,  // Single-pass max-norm equilibration (current default).
    GeometricMean,  // Alternating geometric mean scaling passes.
    Ruiz,           // Iterative Ruiz equilibration (infinity-norm).
};

struct DualSimplexOptions {
    // Pricing strategy: Auto = DSE with Devex fallback.
    PricingStrategy pricing = PricingStrategy::Auto;

    // DSE fallback controls.
    // Short warmup window: evaluate DSE cost after this many initial pivots.
    // If DSE is expensive in this early window, fall back immediately.
    Int dse_warmup_window = 30;
    // Number of pivots over which to measure DSE cost/benefit ratio.
    Int dse_fallback_window = 50;
    // If DSE BTRAN time exceeds this fraction of total iteration time,
    // fall back to Devex.
    Real dse_fallback_cost_ratio = 0.20;

    // DSE weight initialization.
    // Exact init (w_i = ||B^{-T} e_i||^2) requires m BTRANs and by default
    // is repeated after every refactorization. Each exact init costs O(m^2) work,
    // which dominates total work on small/medium bases that solve in a few
    // hundred pivots. Default is approximate init (w_i = 1.0) — Goldfarb-Forrest
    // converges within O(m) pivots from this starting point, matching HiGHS/CLP.
    bool dse_exact_init = false;
    // Upper bound on basis dimension for exact init; above this, exact init
    // is too expensive even when enabled.
    Int dse_exact_init_max_dim = 1000;

    // Pricing controls.
    // Devex pricing for the primal pivots taken once the basis is primal
    // feasible (re-solves after addCols spend their pivots there), in place
    // of |rc| or the static column-norm score. Off pending a benchmark
    // (#266, #268).
    bool enable_primal_devex = false;
    bool enable_partial_pricing = false;
    bool enable_bfrt = false;
    bool enable_adaptive_bfrt = true;
    Int adaptive_bfrt_max_pinf = 128;
    Int adaptive_bfrt_progress_window = 512;
    // Auto-enable BFRT on very wide LPs where it consistently reduces
    // stalling/degenerate pivot tails.
    bool enable_auto_bfrt_wide = true;
    Int auto_bfrt_min_cols = 5000;
    Real auto_bfrt_min_col_row_ratio = 4.0;
    Int partial_pricing_chunk_min = 512;
    Int partial_pricing_full_scan_freq = 25;

    // Adaptive Harris ratio test controls.
    bool enable_adaptive_harris = true;
    Real harris_tight_tol = 1e-10;
    Real harris_expand_factor = 10.0;
    Real harris_max_tol = 1e-5;
    // Minimum relative pivot size |alpha|/||column|| for pivot quality.
    Real harris_min_relative_pivot = 1e-8;

    // BFRT numerical hygiene controls.
    // Maximum accumulated primal change from bound flips before stopping.
    Real bfrt_max_accumulated_change = 1e6;
    // Maximum number of bound flips per iteration before stopping.
    Int bfrt_max_flips = 1000;
    // Trigger early refactorization after iterations with many flips.
    Int bfrt_high_flip_refactor_threshold = 50;

    // Refactorization controls.
    bool enable_adaptive_refactorization = true;
    // Maximum Forrest-Tomlin updates before mandatory reinversion.
    // Non-positive enables auto sizing from basis dimension.
    Int lu_update_limit = 0;
    // Drop tolerance for non-pivot FT update entries (absolute magnitude).
    // Non-positive enables auto sizing from basis dimension.
    Real lu_ft_drop_tolerance = 0.0;
    Int adaptive_refactor_min_updates = 32;
    Int adaptive_refactor_stall_pivots = 64;
    Real adaptive_refactor_degenerate_pivot_tol = 1e-10;
    // Extra guardrails for primal-feasible/dual-infeasible pivot phase.
    Int primal_feasible_adaptive_refactor_stall_pivots = 24;
    Int primal_feasible_adaptive_refactor_min_updates = 16;
    // Non-positive disables the dual-progress stall gate.
    Int primal_feasible_dual_progress_window = 8;
    Int primal_feasible_refactor_cooldown = 32;
    Real primal_feasible_dual_progress_improve_rel_tol = 1e-3;

    // Runtime SIMD controls for dense vector kernels.
    bool enable_simd_kernels = true;
    Int simd_min_length = 64;

    // Matrix/bound scaling.
    bool enable_scaling = true;
    ScalingStrategy scaling_strategy = ScalingStrategy::Auto;
    Int scaling_max_passes = 10;

    // Solve time limit for stand-alone LP usage. Negative means disabled.
    double max_solve_seconds = -1.0;
    // A wall-clock point every solve stops by, whichever of it and
    // max_solve_seconds comes first, with Status::TimeLimit (#351). Set once
    // on a solver whose solves share one budget (a MIP tree's); none by
    // default. Its stop flag, if any, is not read: stop_flag is.
    Deadline deadline;

    // Cost perturbation against dual degeneracy (#278), after HiGHS's
    // HEkk::initialiseCost: each bounded structural's cost moves by
    // (1 + |c_j|) * base * (1 + u_j) toward its dual-feasible side, base =
    // multiplier * 5e-7 * C with C from max |c_j|. Taken off when the
    // perturbed LP is optimal; the primal-feasible phase then repairs the
    // dual infeasibilities that leaves. Applied on a cold start only.
    bool enable_cost_perturbation = true;
    Real cost_perturbation_multiplier = 1.0;

    // Variants of the perturbation's shape (#428), each opt-in; the defaults
    // are the shape above. They change the stall-triggered one below too.
    //
    // ByTheBook sizes column j's move as multiplier * (1e-5 |c_j| +
    // 1e-7 max|c| + 10 * the dual tolerance) * (1 + u_j), after the account of
    // HiGHS's perturbation in Bach, Black, Huiberts and Kafer, "Beyond
    // Smoothed Analysis: Analyzing the Simplex Method by the Book"
    // (arXiv:2510.21613): max |c| undamped, no boxed-rate cap, and an
    // absolute term that moves zero costs at any cost scale.
    enum class CostPerturbationShape { HiGHS, ByTheBook };
    CostPerturbationShape cost_perturbation_shape = CostPerturbationShape::HiGHS;
    // u_j from a splitmix64 stream seeded with cost_perturbation_seed at each
    // solve, column j taking the j-th draw and logical i the (n + i)-th, in
    // place of the per-index multiplicative hash. Deterministic either way.
    bool cost_perturbation_seeded_random = false;
    uint64_t cost_perturbation_seed = 0;
    // HiGHS's jitter on the logicals' zero costs: logical i's cost moves by
    // (0.5 - u_{n+i}) * multiplier * 1e-12, either sign. Logicals have no
    // cost slot here, so the move goes on the structurals of row i (c_j +=
    // delta * a_ij), the same LP on the feasible set. Cold-start perturbation
    // only, and it comes off with it. A move under an ulp of c_j (|c_j|
    // around 1e4 and up) rounds away, so on large costs it is weaker than
    // HiGHS's; with the HiGHS shape and every cost zero there is none.
    bool cost_perturbation_logical_jitter = false;

    // Stall-triggered cost perturbation (#427), for the warm starts the one
    // above skips. Off by default. A run of cost_perturbation_stall_pivots
    // consecutive zero dual steps (refactors do not end it, a nonzero dual
    // step does; zero steps in a cleanup phase are not counted) moves the
    // cost of every nonbasic structural at a bound toward its dual-feasible
    // side, up at its lower bound, down at its upper, by the cold-start
    // perturbation's size from the costs at the solve's entry; fixed and free
    // columns stay. The basic costs, so the duals, do not move, and the basis
    // stays dual feasible. Each further run of that length in the same
    // simplex pass (one solve() can make up to three: warm, cold fallback,
    // unscaled clean-up) moves them again by 10 times the previous
    // activation's size, up to cost_perturbation_stall_max_activations
    // activations per pass (below 1: none; keep it small, the size grows
    // tenfold each time). It comes off with the cold-start one, at primal
    // feasibility. It fires on any start and needs
    // cost_perturbation_multiplier > 0, not enable_cost_perturbation.
    bool enable_stall_cost_perturbation = false;
    Int cost_perturbation_stall_pivots = 8;
    Int cost_perturbation_stall_max_activations = 3;

    // Deterministic anti-degeneracy reduced-cost perturbation.
    bool enable_dual_perturbation = false;
    Int dual_perturbation_stall_pivots = 64;
    Real dual_perturbation_magnitude = 1e-8;

    // Stall-triggered primal bound perturbation (deterministic).
    bool enable_bound_perturbation = true;
    Int bound_perturbation_stall_pivots = 512;
    Real bound_perturbation_magnitude = 1e-8;
    Int bound_perturbation_max_activations = 4;

    // Expanding perturbation: start at initial_scale * base, escalate by
    // expand_factor each time degeneracy persists, up to max_level levels.
    Real bound_perturbation_initial_scale = 0.1;
    Real bound_perturbation_expand_factor = 10.0;
    Int bound_perturbation_max_level = 3;

    // Outward mode of the stall-triggered perturbation (#426), after SoPlex
    // and Bach, Black, Huiberts, Kafer, arXiv:2510.21613. Off by default.
    // Instead of shrinking the box (lb + eps, ub - eps) and coming off at the
    // first non-degenerate pivot, a stall widens it with the a priori
    // perturbation's shape (see enable_primal_bound_perturbation, sized by
    // primal_bound_perturbation_base), keeps it until the apparent optimum
    // and removes it through the same cleanup. The stall is counted the
    // SoPlex way: one up per degenerate pivot, halved by a non-degenerate one
    // (not reset by it or by a refactor; it starts again from zero when the
    // perturbation comes off at an optimum). The perturbation comes on
    // when the count reaches bound_perturbation_stall_pivots and escalates
    // each time it gains that many again; bound_perturbation_max_activations
    // and bound_perturbation_max_level bound it as in the inward mode, and
    // it needs enable_bound_perturbation.
    bool bound_perturbation_outward = false;

    // Cleanup phase: bounded iterations to resolve infeasibilities after
    // removing perturbation at optimality. A cleanup that runs out of them is
    // accepted as Optimal only if the basis is primal and dual feasible for
    // the unperturbed LP; otherwise the solve goes on without the budget.
    Int bound_perturbation_cleanup_iter_limit = 500;

    // A priori outward bound perturbation for the primal phase (#425), after
    // HiGHS's HEkk::initialiseBound for its primal simplex and Bach, Black,
    // Huiberts, Kafer, arXiv:2510.21613 (outward, random, kept until an
    // apparent optimum). Off by default. The first time a simplex pass (the
    // warm solve, a cold fallback, the unscaled clean-up) reaches the
    // primal-feasible, dual-infeasible phase, every finite bound but a
    // nonbasic fixed variable's moves outward by base * r_k * max(1, |bound|),
    // r_k in [0, 1) from a fixed seed. Unlike HiGHS, which moves nonbasics
    // onto the perturbed bounds, nonbasics stay where they are, inside the
    // wider box: the basis stays feasible and no value moves. It replaces the
    // stall-triggered perturbation while it is on: a stall of
    // bound_perturbation_stall_pivots escalates it by
    // bound_perturbation_expand_factor, up to bound_perturbation_max_level.
    // It comes off at the apparent optimum, through the same cleanup.
    bool enable_primal_bound_perturbation = false;
    Real primal_bound_perturbation_base = 5e-7;

    // Stall-triggered deterministic restart/re-crash.
    bool enable_stall_restart = false;
    Int stall_restart_pivots = 2000;
    Int stall_restart_max_restarts = 2;

    // Optional "idiot crash" style bound-flip heuristic before first factorization.
    bool enable_idiot_crash = false;
    Int idiot_crash_passes = 2;
    Int idiot_crash_max_flips = 10000;
    Real idiot_crash_min_gain = 1e-9;
    bool enable_structural_crash = false;
    Int structural_crash_max_swaps = 64;
    Real structural_crash_min_pivot = 1e-7;

    // SIP-style parallel candidate scan (CHUZC), guarded and off by default.
    bool enable_sip_parallel_candidates = false;
    bool enable_sip_parallel_dual_scan = false;
    Int sip_parallel_min_nonbasic = 4096;
    Int sip_parallel_grain = 512;
    Int sip_parallel_min_threads = 2;
    bool sip_parallel_disable_on_stall = true;
    Int sip_parallel_stall_pivots = 32;
    bool enable_sip_parallel_candidate_sort = false;
    Int sip_parallel_sort_min_candidates = 4096;

    // SIP-style parallel leaving-row scan (CHUZR), guarded and off by default.
    bool enable_sip_parallel_chuzr = false;
    Int sip_parallel_min_rows = 2048;
    Int sip_parallel_row_grain = 256;

    // Hyper-sparse pivot-row ("row price") assembly (#131).
    // Forces the dense assembly kernel unconditionally, bypassing both the
    // BTRAN-density gate and the EMA history. Provided so the sparse and dense
    // kernels can be compared directly; the adaptive path is the default.
    bool force_dense_pivot_row = false;
    // Charge each pivot-row assembly the work it does (one unit per row
    // scanned, plus the nonzeros and slack entry of each row with a nonzero
    // BTRAN entry) instead of the whole matrix's nonzero count (#379, decided
    // in #378).
    // Off by default: work units pace the DSE fallback and the MIP's
    // heuristics, so the default waits on an LP and MIP benchmark.
    bool row_price_actual_work = false;

    // Optional cooperative stop flag for externally orchestrated solves.
    const std::atomic<bool>* stop_flag = nullptr;

    // Keep DSE weights for the current basis across reinversions and warm solves.
    bool retain_dse_weights = false;
};

class DualSimplexSolver : public LpSolver {
public:
    DualSimplexSolver() = default;

    void load(const LpProblem& problem) override;
    LpResult solve() override;

    Status getStatus() const override { return status_; }
    Real getObjective() const override;

    std::vector<Real> getPrimalValues() const override;
    std::vector<Real> getDualValues() const override;
    /// After Optimal: c - A^T y of getDualValues(), to round-off. On an
    /// ill-conditioned basis a column can show a wrong-signed d_j below what
    /// the basis resolves (about |c_B|^T |B^-1 a_j| eps): the primal path
    /// prices out a ray whose gain is round-off of its terms (#241, #319).
    std::vector<Real> getReducedCosts() const override;

    /// Nonbasic structural columns at one active bound in the current optimum.
    /// Applies the dual feasibility tolerance in the units of the accepted solve.
    std::optional<LpDualDegeneracy> measureDualDegeneracy() const;
    /// See LpSolver::getDualRay(). The ray is the row of B^{-1} for the basic
    /// variable that the dual ratio test proved cannot reach its bound.
    std::vector<Real> getDualRay() const override;
    /// See LpSolver::getPrimalRay(). The ray is the entering column's
    /// direction (B^{-1} a_q, signed by the entering move) from the primal
    /// ratio test that found nothing blocking.
    std::vector<Real> getPrimalRay() const override;

    /// Empty before the first solve or setBasis after a load().
    std::vector<BasisStatus> getBasis() const override;
    /// Ignored, keeping the current basis, unless it has one entry per
    /// column and row and exactly one Basic entry per row.
    void setBasis(std::span<const BasisStatus> basis) override;

    void addRows(std::span<const Index> starts, std::span<const Index> indices,
                 std::span<const Real> values, std::span<const Real> lower,
                 std::span<const Real> upper) override;
    /// Remove rows, keeping the basis when every remaining basic variable
    /// survives the removal.
    ///
    /// Postcondition: re-solve before reading a solution. When the basis is
    /// kept, primal and reduced-cost values of the surviving variables carry
    /// over and the duals are zero. When it cannot be kept, there is no basis
    /// and getPrimalValues(), the reduced costs and the duals read zero, and
    /// getObjective() the objective offset, rather than stale values from the
    /// old problem.
    void removeRows(std::span<const Index> rows) override;

    /// See LpSolver::addCols(). Each new column is scaled on its own, to a
    /// largest entry of 1 against the current row scaling; the rest of the
    /// model keeps its scaling. The factors stay valid (the basis matrix does
    /// not change), and the next solve() starts from the kept basis.
    void addCols(std::span<const Index> starts, std::span<const Index> indices,
                 std::span<const Real> values, std::span<const Real> obj,
                 std::span<const Real> lower, std::span<const Real> upper) override;
    /// See LpSolver::removeCols(). Each removed basic column is swapped for
    /// the logical of a row on which its basis row of B^{-1} is largest, one
    /// Forrest-Tomlin update at a time, so the basis stays nonsingular. When
    /// the current basis cannot be factorized there is no basis afterwards,
    /// as after removeRows().
    ///
    /// Postcondition: re-solve before reading a solution.
    void removeCols(std::span<const Index> cols) override;

    void setColBounds(Index col, Real lower, Real upper) override;
    void setRowBounds(Index row, Real lower, Real upper) override;
    void setObjective(std::span<const Real> obj) override;
    void setColObjective(Index col, Real cost) override;

    void setIterationLimit(Int limit) { iter_limit_ = limit; }
    /// Forget the current basis, so the next solve() starts cold from the
    /// crash basis, as a first solve does.
    void clearBasis() { dropBasis(); }
    [[nodiscard]] Int getIterationLimit() const { return iter_limit_; }
    void getColBounds(Index col, Real& lower, Real& upper) const;
    void setVerbose(bool v) { verbose_ = v; }
    void setOptions(const DualSimplexOptions& options);
    [[nodiscard]] const DualSimplexOptions& getOptions() const { return options_; }

    /// Access work unit counter (includes LU work).
    [[nodiscard]] const WorkUnits& workUnits() const { return work_; }
    void resetWorkUnits() {
        work_.reset();
        lu_.resetWorkUnits();
    }

    /// Get a tableau row for basis position `basis_pos` in external (unscaled) space.
    /// Returns a dense vector of size num_cols + num_rows (structural + slacks).
    /// If no factored basis is available, the vector is all zeros.
    /// The tableau row satisfies: x_i + sum_{j nonbasic} alpha_j * x_j = 0
    /// where the sum includes both structural and slack variables.
    void getTableauRow(Index basis_pos, std::vector<Real>& tableau_row);

    /// Get row `row` of the constraint matrix in external (unscaled) space,
    /// together with its external row bounds.
    ///
    /// The logical of row i is the row activity s_i = a_i^T x bounded by
    /// [lower, upper], so this is what a cut generator needs to substitute a
    /// nonbasic logical back into structural columns. Rows added after load()
    /// are included, which is the point: their coefficients are not in the
    /// LpProblem the solver was loaded from.
    void getRowExternal(Index row, std::vector<Index>& indices, std::vector<Real>& values,
                        Real& lower, Real& upper) const;

    /// Get the basis position of a variable, or -1 if nonbasic.
    [[nodiscard]] Index basisPosition(Index var) const {
        if (var < 0 || var >= static_cast<Index>(basis_pos_.size())) {
            return -1;
        }
        return basis_pos_[var];
    }

    /// Get the number of rows in the current problem.
    [[nodiscard]] Index numRows() const { return num_rows_; }

    /// Get the number of structural columns in the current problem.
    [[nodiscard]] Index numCols() const { return num_cols_; }

    /// Diagnostics for the adaptive Harris ratio test (#159), accumulated over
    /// the most recent solve(). Exposed for observability and testing.
    struct AdaptiveHarrisStats {
        Int primal_expansion_pivots = 0;   // primal pivots needing tolerance expansion
        Int rejected_pivots = 0;           // candidates rejected for poor relative pivot quality
        Int bfrt_high_flip_refactors = 0;  // refactorizations forced by high BFRT flip counts
    };
    [[nodiscard]] const AdaptiveHarrisStats& getAdaptiveHarrisStats() const {
        return adaptive_harris_stats_;
    }

    /// Diagnostics for the hyper-sparse pivot-row ("row price") kernel (#131),
    /// accumulated over the most recent solve(), all of its internal passes
    /// (cold retry, unscaled clean-up) included. Exposed for observability and
    /// testing, so the path actually taken can be asserted.
    struct RowPriceStats {
        Int sparse_assemblies = 0;     // assemblies completed by the sparse kernel
        Int dense_assemblies = 0;      // assemblies completed by the dense kernel
        Int fill_in_fallbacks = 0;     // sparse assemblies abandoned mid-way for fill-in
        Real ema_alpha_density = 0.0;  // EMA of assembled alpha-row density (live state)
    };
    [[nodiscard]] const RowPriceStats& getRowPriceStats() const { return row_price_stats_; }

    /// DSE-to-Devex fallbacks of the most recent solve(), all of its internal
    /// passes included, by cause. The cost check reads work units, so a
    /// change to how work is charged can move it (#379).
    struct DseFallbackStats {
        Int cost = 0;        // DSE's extra BTRAN work over its share of the total
        Int degenerate = 0;  // a degenerate pivot streak under DSE
    };
    [[nodiscard]] const DseFallbackStats& getDseFallbackStats() const {
        return dse_fallback_stats_;
    }

    /// Basis starts and recoveries across all passes of the most recent solve().
    struct BasisStartStats {
        bool cold_start = false;
        bool singular_recovery = false;
    };
    [[nodiscard]] BasisStartStats basisStartStats() const { return basis_start_stats_; }

    /// Degeneracy and perturbation counters of the most recent solve() (#422),
    /// summed over all of its passes (cold retry, unscaled clean-up included).
    /// A pivot is degenerate when its primal step is within
    /// adaptive_refactor_degenerate_pivot_tol of zero, the test the stall
    /// streak behind the bound perturbation counts (a primal-phase bound flip
    /// counts as a pivot). A dual pivot stalls the dual objective on a zero
    /// dual step instead, which dual_zero_steps counts. dual_pivots +
    /// primal_pivots + retry_iterations == LpResult::iterations, but for a
    /// solve that ends in Error.
    struct DegeneracyStats {
        Int dual_pivots = 0;
        Int dual_degenerate_pivots = 0;  // dual pivots with a zero primal step
        Int dual_zero_steps = 0;         // dual pivots with a zero dual step (dual degeneracy)
        Int max_dual_zero_step_streak = 0;
        // Zero dual steps that moved the entering cost (a nonzero shift).
        Int zero_step_cost_shifts = 0;
        Int primal_pivots = 0;
        Int primal_degenerate_pivots = 0;
        // The longest run of degenerate_pivot_streak, the stall streak itself:
        // in the dual phase every refactor ends it, the primal phase carries
        // it over its routine refactors. max_dual_zero_step_streak only ends on a nonzero
        // dual step.
        Int max_degenerate_streak = 0;
        // Iterations without a pivot: a refactor and retry, a ray guard's
        // price-out, a stall restart.
        Int retry_iterations = 0;
        bool cost_perturbed = false;  // the cold-start cost perturbation was applied
        // Times the stall-triggered cost perturbation was applied (#427).
        Int stall_cost_perturbations = 0;
        // Times the cold-start or stall-triggered perturbation, with the
        // zero-step and dual correction shifts held with it, came off at
        // primal feasibility. The phase-1 cost shifts coming off are not
        // counted.
        Int cost_perturbation_removals = 0;
        Int bound_perturbation_activations = 0;
        Int bound_perturbation_escalations = 0;
        // Times the a priori outward perturbation (#425) was applied; its
        // escalations count in bound_perturbation_escalations.
        Int primal_bound_perturbation_activations = 0;
        // Cleanups entered at an apparent optimum after the bound
        // perturbation came off or nonbasics were put back on their bounds.
        Int cleanup_phases = 0;
        Int cleanup_budget_exhausted = 0;  // cleanups declared Optimal on an exhausted budget
        // Cleanups whose budget ran out on a basis infeasible for the
        // unperturbed LP, so the solve went on (#426).
        Int cleanup_budget_rejected = 0;
    };
    [[nodiscard]] const DegeneracyStats& getDegeneracyStats() const { return degeneracy_stats_; }

    /// Basis factorizations of the most recent solve() by what asked for
    /// them (#517), summed over all of its passes (cold retry, unscaled
    /// clean-up included). Each counts an attempt, so one that fails or
    /// throws is counted too. edits counts those of the edit calls since
    /// the solve before it or the last load() (removeCols swapping a basic
    /// column out).
    struct RefactorStats {
        // Each pass's first factorization of the basis it starts from: the
        // warm basis after the edits, or a cold start's crash basis.
        Int start = 0;
        // Scheduled: the LU's update count reached lu_update_limit, or its
        // Forrest-Tomlin updates cost more than twice the base factors
        // (small bases only).
        Int update_limit = 0;
        Int update_cost = 0;
        // Numerical trouble: an updated U entry passed the growth limit; a
        // pivot too small on use; row and column pivots that disagree; an
        // empty ratio test on a dual-infeasible basis; no BFRT group; the
        // primal ray guard on updated factors; a singular basis replaced by
        // a crash basis.
        Int growth = 0;
        Int pivot_rejected = 0;
        Int pivot_mismatch = 0;
        Int empty_ratio_test = 0;
        Int no_bfrt_group = 0;
        Int ray_guard = 0;
        Int singular_recovery = 0;
        // Adaptive: a degenerate pivot streak, consecutive high-flip BFRT
        // iterations, a stall restart from a crash basis.
        Int degenerate_stall = 0;
        Int bfrt_high_flip = 0;
        Int stall_restart = 0;
        // After the loop: the warm point put back after a cold retry that
        // stopped on a limit, and the unscaled clean-up's scaled basis.
        Int cold_restore = 0;
        Int unscaled_cleanup = 0;
        Int edits = 0;
    };
    [[nodiscard]] const RefactorStats& getRefactorStats() const { return refactor_stats_; }

private:
    friend class BarrierSolver;
    friend class MipSolver;
    friend class PdlpSolver;
    friend struct test::DualSimplexPricingAccess;

    // A crossover load shares its caller's solve deadline. A stopped load
    // leaves the simplex solver unloaded.
    [[nodiscard]] bool loadUntil(const LpProblem& problem, const Deadline& deadline);

    // Total number of variables = num_cols (structural) + num_rows (slacks).
    Index numVars() const { return num_cols_ + num_rows_; }

    /// Build augmented_ = [A | -I], the matrix the basis is factorized from.
    [[nodiscard]] static std::optional<SparseMatrix> makeBasisMatrix(const SparseMatrix& matrix,
                                                                     const Deadline& deadline);
    [[nodiscard]] bool buildBasisMatrixSource(const Deadline& deadline = {});
    /// Swap each basic column in @p cols for a row's logical (removeCols).
    /// False, with the basis unchanged in meaning, when it cannot be done.
    bool swapOutBasicColumns(std::span<const Index> cols);
    /// Drop the basis, sizing every per-variable vector to the model.
    void dropBasis();

    // Set up initial basis (all slacks basic).
    void setupInitialBasis();

    // Compute primal values for basic variables: x_B = B^{-1} * b.
    void computePrimals();

    // Compute dual values and reduced costs.
    void computeDuals();

    // Refactorize the basis, counting it in refactor_stats_ under @p trigger
    // (#517); the no-deadline form is an edit call's, counted in
    // pending_edit_refactors_ until the next solve() (a load() drops it).
    void refactorize();
    [[nodiscard]] bool refactorize(const Deadline& deadline, Int RefactorStats::* trigger);

    // Scaling.
    [[nodiscard]] bool computeScaling(const Deadline& deadline = {});
    [[nodiscard]] bool computeEquilibrationScaling(const Deadline& deadline);
    [[nodiscard]] bool computeGeometricMeanScaling(const Deadline& deadline);
    [[nodiscard]] bool computeRuizScaling(const Deadline& deadline);
    [[nodiscard]] bool applyScaling(const Deadline& deadline = {});
    // The inverse of applyScaling(): the matrix, sides, bounds and objective
    // back in the model's units, with row_scale_/col_scale_ left as they were.
    [[nodiscard]] bool unapplyScaling(const Deadline& deadline = {});
    // Move primal_, dual_ and reduced_cost_ between the scaled and the model's
    // units with the current row_scale_/col_scale_.
    void convertValues(bool to_model);
    void rescaleKeepingBasis();
    LpResult solveFromCurrentBasis(const Deadline& deadline);
    // solve() less the unscaled clean-up: a warm verdict that fails its check
    // gets a cold re-solve. `checked` says the verdict returned passed
    // verdictHolds(), so the clean-up need not run it again.
    LpResult solveWithColdFallback(const Deadline& deadline, bool& checked);
    // An Optimal or Infeasible the model's units refute gets a re-solve of the
    // unscaled model from its basis (#307, #306, #305).
    LpResult cleanUpUnscaled(LpResult res, const Deadline& deadline, bool checked);
    bool dualRayMeetsContract() const;
    bool optimumHoldsUnscaled() const;
    // `tight`: absolute, as pointViolation() reads a point (#304).
    bool primalHoldsUnscaled(bool tight = false) const;
    // False for an Optimal or Infeasible status whose certificate fails the
    // checks above; true for any other status.
    bool verdictHolds() const;
    // Replace an uncertified verdict by `status`, a non-verdict, keeping the
    // point and the basis (#249).
    LpResult withdrawVerdict(LpResult res, Status status);
    void unscaleResults();

    // Record the Farkas ray proven by basis position `row` (rho = B^{-T} e_row,
    // internal scaled space): sigma = +1 when the basic variable sits below
    // its lower bound, -1 when above its upper.
    void storeDualRay(std::span<const Real> rho, Real sigma);

    // Record the unbounded direction of the primal ratio test: the entering
    // variable moves by delta_dir, basis position i by -delta_dir *
    // pivot_col[i] (internal scaled space).
    void storePrimalRay(std::span<const Real> pivot_col, Index entering, Real delta_dir);

    // Variable bounds (structural + slack).
    Real varLower(Index k) const;
    /// Bring one nonbasic variable's status and value in line with its bounds.
    void reconcileNonbasicStatus(Index k);
    /// The same against the given bounds (internal units, unperturbed).
    void reconcileNonbasicStatus(Index k, Real lo, Real up);
    Real varUpper(Index k) const;
    Real varCost(Index k) const;

    // Problem data (internal, after sign flip for maximize).
    Index num_cols_ = 0;
    Index num_rows_ = 0;
    Sense sense_ = Sense::Minimize;
    Real obj_offset_ = 0.0;
    std::vector<Real> obj_;         // size num_cols
    std::vector<Real> col_lower_;   // size num_cols
    std::vector<Real> col_upper_;   // size num_cols
    std::vector<Real> row_lower_;   // size num_rows
    std::vector<Real> row_upper_;   // size num_rows
    SparseMatrix matrix_{0, 0};     // original constraint matrix
    SparseMatrix augmented_{0, 0};  // [A | -I]

    // Scaling factors.
    std::vector<Real> row_scale_;  // size num_rows
    std::vector<Real> col_scale_;  // size num_cols
    bool scaled_ = false;

    // Basis representation.
    // basis_[i] = variable index (0..numVars-1) in basis position i.
    std::vector<Index> basis_;
    // nonbasic_ = list of nonbasic variable indices.
    std::vector<Index> nonbasic_;
    // nonbasic_pos_[k] = position in nonbasic_ if nonbasic, -1 otherwise.
    std::vector<Index> nonbasic_pos_;
    // basis_pos_[k] = position in basis_ if basic, -1 otherwise.
    std::vector<Index> basis_pos_;
    // var_status_[k] = status of variable k.
    std::vector<BasisStatus> var_status_;

    // Solution data.
    std::vector<Real> primal_;        // size numVars
    std::vector<Real> dual_;          // size num_rows
    std::vector<Real> reduced_cost_;  // size numVars
    // Farkas ray in external row space, set when a solve ends Infeasible.
    std::vector<Real> dual_ray_;
    // Primal ray in external column space, set when a solve ends Unbounded.
    std::vector<Real> primal_ray_;

    // LU factorization.
    SparseLU lu_;

    // Solver state.
    Status status_ = Status::Error;
    bool optimal_basis_current_ = false;
    bool optimal_basis_from_unscaled_cleanup_ = false;
    Int iterations_ = 0;
    Int iter_limit_ = 1000000;
    bool loaded_ = false;
    bool has_basis_ = false;          // true after first solve or setBasis()
    bool factors_available_ = false;  // whether LU matches the installed basis
    bool verbose_ = true;
    BasisStartStats basis_start_stats_;

    // Pricing weights (one per basis position, size num_rows).
    // When DSE is active, these estimate steepest-edge weights w_i = ||B^{-T} e_i||^2.
    // When Devex is active, these hold approximate Devex weights.
    std::vector<Real> devex_weights_;
    Int devex_reset_count_ = 0;
    // The one Devex weight that may sit below kDualTol: the last update's
    // leaving row (1 / pivot^2). The next update clamps it, as the dense loop
    // it replaced clamped every row. kDevexClampAll asks for a full pass, after
    // weights were restored from elsewhere.
    static constexpr Index kDevexClampAll = -2;
    Index devex_clamp_row_ = -1;
    static constexpr Int kDevexResetFreq = 200;  // reset every N pivots
    // Primal Devex restarts its reference framework when the entering
    // column's kept weight and its exact one differ by more than this factor
    // (Forrest & Goldfarb 1992: the weights only bound the reference norms
    // and drift; an error past a small factor means the framework is stale).
    static constexpr Real kPrimalDevexErrorRatio = 3.0;

    // DSE state.
    bool dse_active_ = false;         // true when using DSE pricing
    bool dse_fell_back_ = false;      // true once auto-fallback switched to Devex
    bool dse_weights_valid_ = false;  // opted-in weights correspond to the current basis
    Int dse_pivots_since_init_ = 0;   // pivots since last DSE weight init
    double dse_btran_work_ = 0.0;     // cumulative DSE BTRAN work in window
    double dse_total_work_ = 0.0;     // cumulative total iteration work in window
    Int dse_window_pivots_ = 0;       // pivots counted in current window

    // Tolerances.
    static constexpr Real kPrimalTol = 1e-7;
    static constexpr Real kDualTol = 1e-7;
    // A dual step this small counts as zero in DegeneracyStats (#422).
    static constexpr Real kDualZeroStepTol = 1e-12;
    static constexpr Real kPivotTol = 1e-7;
    static constexpr Real kZeroTol = 1e-13;
    // Smallest pivot the ratio test admits when no entry reaches kPivotTol
    // but a small one, over its column's range, can still repair the row.
    static constexpr Real kRepairPivotTol = 1e-9;
    // Relative disagreement between the row- and column-wise pivot element
    // above which updated factors are refreshed before pivoting.
    static constexpr Real kAlphaTroubleTol = 1e-7;
    // A primal ray must improve the objective by more than this fraction of
    // the magnitude of the terms its reduced cost sums.
    static constexpr Real kRayGainRelTol = 1e-9;
    // Relative tolerance of the check of an optimum in the model's units.
    static constexpr Real kUnscaledRelTol = 1e-6;
    // An optimum's point is cleaned up unscaled when it misses a row or bound
    // by more than kUnscaledAbsTol in the model's units, unless that is below
    // what double arithmetic resolves on the row: kResolvableRelTol of its
    // largest term (#304).
    static constexpr Real kUnscaledAbsTol = 1e-6;
    static constexpr Real kResolvableRelTol = 1e-12;
    // addRows rescales the whole model when a new row's coefficients, in the
    // current column scaling, spread more than this or sit further than this
    // from 1.
    static constexpr Real kRescaleSpread = 1e3;
    // Largest move (scaled units) a repair may count on. Reaching a bound
    // only past it puts the column where round-off in its value exceeds the
    // primal tolerance, so such a row still stands as a proof.
    static constexpr Real kRepairMaxStep = 1e9;
    static constexpr Int kLogFrequency = 200;

    // Hyper-sparse pivot-row ("row price") switching constants (#131).
    // Lifted from the EMA-density-guided switching in the LU hyper-sparse
    // solves (SparseLU::kHyperSparseMaxDensity / kEmaAlpha): below this
    // density the indexed walk over the BTRAN support beats a full O(m) scan
    // of rho; above it the sequential dense scan wins on locality.
    static constexpr Real kRowPriceMaxDensity = 0.12;
    static constexpr Real kRowPriceEmaAlpha = 0.15;
    // Mid-assembly bail-out budget, as a multiple of kRowPriceMaxDensity.
    // Matches the LU solves, which bail from the hyper-sparse iteration to a
    // dense scan once the output exceeds 2x the entry threshold.
    static constexpr Real kRowPriceFillInFactor = 2.0;

    // Timing for iteration log.
    std::chrono::steady_clock::time_point solve_start_;

    // Deterministic work counter.
    WorkUnits work_;
    bool bound_perturb_active_ = false;
    Int bound_perturb_activations_ = 0;
    Int bound_perturb_level_ = 0;  // current escalation level (0 = initial)
    // The active bound perturbation is an outward one (the a priori one,
    // #425, or the stall-triggered one in outward mode, #426): negative
    // entries in the perturbation arrays, kept until an optimum.
    bool bound_perturb_outward_ = false;
    Int bfrt_high_flip_iters_ = 0;  // consecutive iterations with high flip count
    // Adaptive Harris (#159) numerical-incident tracking, reset each solve().
    AdaptiveHarrisStats adaptive_harris_stats_{};
    DegeneracyStats degeneracy_stats_{};
    RefactorStats refactor_stats_{};
    Int pending_edit_refactors_ = 0;  // edit calls' since the last solve() or load()
    // Hyper-sparse row-price (#131) path tracking and density history,
    // reset each solve().
    RowPriceStats row_price_stats_{};
    DseFallbackStats dse_fallback_stats_{};
    // Added to finite lowers and subtracted from finite uppers: positive
    // shrinks the box (the stall-triggered perturbation), negative widens it
    // (the outward one).
    std::vector<Real> lower_bound_perturb_;
    std::vector<Real> upper_bound_perturb_;
    DualSimplexOptions options_{};
};

}  // namespace mipx
