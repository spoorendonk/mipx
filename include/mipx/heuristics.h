#pragma once

#include <algorithm>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include "mipx/core.h"
#include "mipx/lp_problem.h"

namespace mipx {

// Forward declarations.
class DualSimplexSolver;

/// A feasible MIP solution found by a heuristic.
struct HeuristicSolution {
    std::vector<Real> values;
    Real objective = 0.0;
};

/// When should a heuristic be invoked?
enum class HeuristicTiming {
    Root,       ///< After root LP solve.
    EveryNode,  ///< After every node LP solve.
    Periodic,   ///< Every N nodes.
};

/// Abstract base class for primal heuristics.
class Heuristic {
public:
    virtual ~Heuristic() = default;

    /// Try to find a feasible MIP solution from the current LP relaxation.
    /// @param problem    The MIP problem data.
    /// @param lp         The LP solver (already solved at current node).
    /// @param primals    The LP relaxation primal values.
    /// @param incumbent  Current best objective (kInf if none).
    /// @return A feasible solution, or std::nullopt.
    virtual std::optional<HeuristicSolution> run(
        const LpProblem& problem,
        DualSimplexSolver& lp,
        std::span<const Real> primals,
        Real incumbent) = 0;

    /// Name for logging.
    [[nodiscard]] virtual const char* name() const = 0;
};

/// Simple rounding: round each fractional integer variable to nearest integer,
/// check feasibility of the resulting solution.
class RoundingHeuristic : public Heuristic {
public:
    std::optional<HeuristicSolution> run(
        const LpProblem& problem,
        DualSimplexSolver& lp,
        std::span<const Real> primals,
        Real incumbent) override;

    [[nodiscard]] const char* name() const override { return "rounding"; }
};

/// Diving heuristic base: iteratively fix integer variables and re-solve LP.
class DivingHeuristic : public Heuristic {
public:
    /// Maximum number of dives (LP re-solves) before giving up.
    void setMaxDives(Int limit) { max_dives_ = limit; }

    /// Maximum number of consecutive infeasible dives before backtracking.
    void setBacktrackLimit(Int limit) { backtrack_limit_ = limit; }

protected:
    /// Select the variable to fix and which bound to fix it to.
    /// Returns (variable_index, fixed_value), or (-1, 0) if no candidate.
    virtual std::pair<Index, Real> selectVariable(
        const LpProblem& problem,
        std::span<const Real> primals,
        std::span<const Real> obj) const = 0;

    /// Common diving loop implementation.
    std::optional<HeuristicSolution> dive(
        const LpProblem& problem,
        DualSimplexSolver& lp,
        std::span<const Real> primals,
        Real incumbent);

    Int max_dives_ = 100;
    Int backtrack_limit_ = 3;
};

/// Fractional diving: fix the most fractional variable to its nearest integer.
class FractionalDiving : public DivingHeuristic {
public:
    std::optional<HeuristicSolution> run(
        const LpProblem& problem,
        DualSimplexSolver& lp,
        std::span<const Real> primals,
        Real incumbent) override;

    [[nodiscard]] const char* name() const override { return "fracdiving"; }

protected:
    std::pair<Index, Real> selectVariable(
        const LpProblem& problem,
        std::span<const Real> primals,
        std::span<const Real> obj) const override;
};

/// Coefficient diving: fix the variable with smallest objective coefficient
/// impact per unit rounding.
class CoefficientDiving : public DivingHeuristic {
public:
    std::optional<HeuristicSolution> run(
        const LpProblem& problem,
        DualSimplexSolver& lp,
        std::span<const Real> primals,
        Real incumbent) override;

    [[nodiscard]] const char* name() const override { return "coefdiving"; }

protected:
    std::pair<Index, Real> selectVariable(
        const LpProblem& problem,
        std::span<const Real> primals,
        std::span<const Real> obj) const override;
};

/// RINS (Relaxation Induced Neighborhood Search)-style heuristic.
/// Fix integer variables where LP and incumbent values agree, then solve
/// the restricted neighborhood LP.
class RinsHeuristic : public Heuristic {
public:
    /// Maximum LP iterations for the restricted subproblem solve.
    void setSubproblemIterLimit(Int limit) { subproblem_iter_limit_ = limit; }

    /// Agreement tolerance for fixing integer variables.
    /// If |x - round(x)| <= tol, value is treated as integral.
    void setAgreementTol(Real tol) { agreement_tol_ = tol; }

    /// Minimum number of variables that must be fixed before solving.
    void setMinFixedVars(Int count) { min_fixed_vars_ = count; }

    /// Minimum fraction (0..1) of integer vars to fix before solving.
    void setMinFixedRate(Real rate) { min_fixed_rate_ = rate; }

    /// If true, try rounded-repair when restricted LP is fractional.
    void setEnableRoundingRepair(bool enable) { enable_rounding_repair_ = enable; }

    std::optional<HeuristicSolution> run(
        const LpProblem& problem,
        DualSimplexSolver& lp,
        std::span<const Real> primals,
        Real incumbent) override;

    /// Classic incumbent-guided RINS.
    std::optional<HeuristicSolution> run(
        const LpProblem& problem,
        DualSimplexSolver& lp,
        std::span<const Real> primals,
        Real incumbent,
        std::span<const Real> incumbent_values);

    [[nodiscard]] Int lastFixedCount() const { return last_fixed_count_; }
    [[nodiscard]] bool lastExecutedSolve() const { return last_executed_solve_; }
    [[nodiscard]] bool lastSkippedNoIncumbent() const { return last_skipped_no_incumbent_; }
    [[nodiscard]] bool lastSkippedFewFixes() const { return last_skipped_few_fixes_; }
    [[nodiscard]] bool lastFoundSolution() const { return last_found_solution_; }
    [[nodiscard]] Int lastLpIterations() const { return last_lp_iterations_; }
    [[nodiscard]] double lastWorkUnits() const { return last_work_units_; }

    [[nodiscard]] const char* name() const override { return "rins"; }

private:
    Int subproblem_iter_limit_ = 120;
    Real agreement_tol_ = 1e-4;
    Int min_fixed_vars_ = 8;
    Real min_fixed_rate_ = 0.03;
    bool enable_rounding_repair_ = true;

    Int last_fixed_count_ = 0;
    bool last_executed_solve_ = false;
    bool last_skipped_no_incumbent_ = false;
    bool last_skipped_few_fixes_ = false;
    bool last_found_solution_ = false;
    Int last_lp_iterations_ = 0;
    double last_work_units_ = 0.0;
};

/// RENS (Relaxation Enforced Neighborhood Search)-style heuristic.
/// Fix near-integral LP integer variables without requiring an incumbent.
class RensHeuristic : public Heuristic {
public:
    void setSubproblemIterLimit(Int limit) { subproblem_iter_limit_ = limit; }
    void setFixTol(Real tol) { fix_tol_ = tol; }
    void setMinFixedVars(Int count) { min_fixed_vars_ = count; }
    void setMinFixedRate(Real rate) { min_fixed_rate_ = rate; }
    void setEnableRoundingRepair(bool enable) { enable_rounding_repair_ = enable; }

    std::optional<HeuristicSolution> run(
        const LpProblem& problem,
        DualSimplexSolver& lp,
        std::span<const Real> primals,
        Real incumbent) override;

    [[nodiscard]] Int lastFixedCount() const { return last_fixed_count_; }
    [[nodiscard]] bool lastExecutedSolve() const { return last_executed_solve_; }
    [[nodiscard]] bool lastSkippedFewFixes() const { return last_skipped_few_fixes_; }
    [[nodiscard]] bool lastFoundSolution() const { return last_found_solution_; }
    [[nodiscard]] Int lastLpIterations() const { return last_lp_iterations_; }
    [[nodiscard]] double lastWorkUnits() const { return last_work_units_; }

    [[nodiscard]] const char* name() const override { return "rens"; }

private:
    Int subproblem_iter_limit_ = 80;
    Real fix_tol_ = 1e-5;
    Int min_fixed_vars_ = 12;
    Real min_fixed_rate_ = 0.2;
    bool enable_rounding_repair_ = true;

    Int last_fixed_count_ = 0;
    bool last_executed_solve_ = false;
    bool last_skipped_few_fixes_ = false;
    bool last_found_solution_ = false;
    Int last_lp_iterations_ = 0;
    double last_work_units_ = 0.0;
};

/// Lightweight feasibility pump:
/// alternate integer rounding and LP projection with a linear guidance objective.
class FeasibilityPumpHeuristic : public Heuristic {
public:
    void setMaxIterations(Int iters) { max_iterations_ = iters; }
    void setSubproblemIterLimit(Int limit) { subproblem_iter_limit_ = limit; }
    void setCyclePerturbationPeriod(Int period) { cycle_perturb_period_ = period; }

    std::optional<HeuristicSolution> run(
        const LpProblem& problem,
        DualSimplexSolver& lp,
        std::span<const Real> primals,
        Real incumbent) override;

    [[nodiscard]] Int lastLpIterations() const { return last_lp_iterations_; }
    [[nodiscard]] double lastWorkUnits() const { return last_work_units_; }

    [[nodiscard]] const char* name() const override { return "feaspump"; }

private:
    Int max_iterations_ = 6;
    Int subproblem_iter_limit_ = 40;
    Int cycle_perturb_period_ = 2;
    Int last_lp_iterations_ = 0;
    double last_work_units_ = 0.0;
};

/// Local Branching heuristic for binary MIPs:
/// solve a neighborhood around an incumbent with
///   sum_j |x_j - x^*_j| <= k  (for binary variables).
class LocalBranchingHeuristic : public Heuristic {
public:
    void setSubproblemIterLimit(Int limit) { subproblem_iter_limit_ = limit; }
    void setNeighborhoodSize(Int size) { neighborhood_size_ = size; }
    void setMinBinaryVars(Int count) { min_binary_vars_ = count; }
    void setEnableRoundingRepair(bool enable) { enable_rounding_repair_ = enable; }

    std::optional<HeuristicSolution> run(
        const LpProblem& problem,
        DualSimplexSolver& lp,
        std::span<const Real> primals,
        Real incumbent) override;

    /// Incumbent-guided local branching.
    std::optional<HeuristicSolution> run(
        const LpProblem& problem,
        DualSimplexSolver& lp,
        std::span<const Real> primals,
        Real incumbent,
        std::span<const Real> incumbent_values);

    [[nodiscard]] bool lastExecutedSolve() const { return last_executed_solve_; }
    [[nodiscard]] bool lastSkippedNoIncumbent() const { return last_skipped_no_incumbent_; }
    [[nodiscard]] bool lastSkippedTooSmall() const { return last_skipped_too_small_; }
    [[nodiscard]] Int lastBinaryCount() const { return last_binary_count_; }
    [[nodiscard]] Int lastLpIterations() const { return last_lp_iterations_; }
    [[nodiscard]] double lastWorkUnits() const { return last_work_units_; }

    [[nodiscard]] const char* name() const override { return "localbranching"; }

private:
    Int subproblem_iter_limit_ = 80;
    Int neighborhood_size_ = 12;
    Int min_binary_vars_ = 8;
    bool enable_rounding_repair_ = true;

    bool last_executed_solve_ = false;
    bool last_skipped_no_incumbent_ = false;
    bool last_skipped_too_small_ = false;
    Int last_binary_count_ = 0;
    Int last_lp_iterations_ = 0;
    double last_work_units_ = 0.0;
};

/// Auxiliary-objective heuristic:
/// temporarily replace the original objective by an integrality-guidance
/// objective and solve a short LP to obtain a nearby integer-feasible point.
class AuxObjectiveHeuristic : public Heuristic {
public:
    void setSubproblemIterLimit(Int limit) { subproblem_iter_limit_ = limit; }
    void setMinActiveIntegerVars(Int count) { min_active_integer_vars_ = count; }
    void setEnableRoundingRepair(bool enable) { enable_rounding_repair_ = enable; }

    std::optional<HeuristicSolution> run(
        const LpProblem& problem,
        DualSimplexSolver& lp,
        std::span<const Real> primals,
        Real incumbent) override;

    std::optional<HeuristicSolution> run(
        const LpProblem& problem,
        DualSimplexSolver& lp,
        std::span<const Real> primals,
        Real incumbent,
        std::span<const Real> incumbent_values);

    [[nodiscard]] bool lastExecutedSolve() const { return last_executed_solve_; }
    [[nodiscard]] bool lastSkippedTooSmall() const { return last_skipped_too_small_; }
    [[nodiscard]] Int lastActiveIntegerVars() const { return last_active_integer_vars_; }
    [[nodiscard]] Int lastLpIterations() const { return last_lp_iterations_; }
    [[nodiscard]] double lastWorkUnits() const { return last_work_units_; }

    [[nodiscard]] const char* name() const override { return "auxobj"; }

private:
    Int subproblem_iter_limit_ = 40;
    Int min_active_integer_vars_ = 1;
    bool enable_rounding_repair_ = true;

    bool last_executed_solve_ = false;
    bool last_skipped_too_small_ = false;
    Int last_active_integer_vars_ = 0;
    Int last_lp_iterations_ = 0;
    double last_work_units_ = 0.0;
};

/// Zero-objective (min-relaxation-style) heuristic:
/// solve a short LP with objective 0 to focus on feasibility structure,
/// then try integer repair.
class ZeroObjectiveHeuristic : public Heuristic {
public:
    void setSubproblemIterLimit(Int limit) { subproblem_iter_limit_ = limit; }
    void setEnableRoundingRepair(bool enable) { enable_rounding_repair_ = enable; }

    std::optional<HeuristicSolution> run(
        const LpProblem& problem,
        DualSimplexSolver& lp,
        std::span<const Real> primals,
        Real incumbent) override;

    [[nodiscard]] bool lastExecutedSolve() const { return last_executed_solve_; }
    [[nodiscard]] Int lastLpIterations() const { return last_lp_iterations_; }
    [[nodiscard]] double lastWorkUnits() const { return last_work_units_; }

    [[nodiscard]] const char* name() const override { return "zeroobj"; }

private:
    Int subproblem_iter_limit_ = 40;
    bool enable_rounding_repair_ = true;

    bool last_executed_solve_ = false;
    Int last_lp_iterations_ = 0;
    double last_work_units_ = 0.0;
};

// ---------------------------------------------------------------------------
// Feasibility Jump: constraint-violation-guided variable flipping.
// ---------------------------------------------------------------------------

/// Feasibility Jump heuristic: iteratively flip integer variables to
/// reduce total constraint violation, producing feasible MIP solutions
/// without LP solves.
class FeasibilityJumpHeuristic : public Heuristic {
public:
    void setMaxIterations(Int iters) { max_iterations_ = iters; }
    void setMaxRestarts(Int restarts) { max_restarts_ = restarts; }

    std::optional<HeuristicSolution> run(
        const LpProblem& problem,
        DualSimplexSolver& lp,
        std::span<const Real> primals,
        Real incumbent) override;

    [[nodiscard]] const char* name() const override { return "feasjump"; }

private:
    Int max_iterations_ = 500;
    Int max_restarts_ = 4;
};

// ---------------------------------------------------------------------------
// Crossover: combine two incumbent solutions.
// ---------------------------------------------------------------------------

/// Crossover heuristic: fix variables where two solutions agree,
/// optimize the disagreements via a restricted LP.
class CrossoverHeuristic : public Heuristic {
public:
    void setSubproblemIterLimit(Int limit) { subproblem_iter_limit_ = limit; }
    void setMinFixedVars(Int count) { min_fixed_vars_ = count; }
    void setEnableRoundingRepair(bool enable) { enable_rounding_repair_ = enable; }

    std::optional<HeuristicSolution> run(
        const LpProblem& problem,
        DualSimplexSolver& lp,
        std::span<const Real> primals,
        Real incumbent) override;

    /// Crossover of LP relaxation and incumbent solution.
    std::optional<HeuristicSolution> run(
        const LpProblem& problem,
        DualSimplexSolver& lp,
        std::span<const Real> primals,
        Real incumbent,
        std::span<const Real> incumbent_values);

    [[nodiscard]] Int lastFixedCount() const { return last_fixed_count_; }
    [[nodiscard]] bool lastExecutedSolve() const { return last_executed_solve_; }
    [[nodiscard]] Int lastLpIterations() const { return last_lp_iterations_; }
    [[nodiscard]] double lastWorkUnits() const { return last_work_units_; }

    [[nodiscard]] const char* name() const override { return "crossover"; }

private:
    Int subproblem_iter_limit_ = 80;
    Int min_fixed_vars_ = 4;
    bool enable_rounding_repair_ = true;

    Int last_fixed_count_ = 0;
    bool last_executed_solve_ = false;
    Int last_lp_iterations_ = 0;
    double last_work_units_ = 0.0;
};

// ---------------------------------------------------------------------------
// Proximity Search (Fischetti & Monaci): minimize Hamming distance.
// ---------------------------------------------------------------------------

/// Proximity search: minimize Hamming distance to incumbent subject to
/// an improving objective cutoff constraint.
class ProximitySearchHeuristic : public Heuristic {
public:
    void setSubproblemIterLimit(Int limit) { subproblem_iter_limit_ = limit; }
    void setMinBinaryVars(Int count) { min_binary_vars_ = count; }
    void setEnableRoundingRepair(bool enable) { enable_rounding_repair_ = enable; }

    std::optional<HeuristicSolution> run(
        const LpProblem& problem,
        DualSimplexSolver& lp,
        std::span<const Real> primals,
        Real incumbent) override;

    std::optional<HeuristicSolution> run(
        const LpProblem& problem,
        DualSimplexSolver& lp,
        std::span<const Real> primals,
        Real incumbent,
        std::span<const Real> incumbent_values);

    [[nodiscard]] bool lastExecutedSolve() const { return last_executed_solve_; }
    [[nodiscard]] Int lastLpIterations() const { return last_lp_iterations_; }
    [[nodiscard]] double lastWorkUnits() const { return last_work_units_; }

    [[nodiscard]] const char* name() const override { return "proximity"; }

private:
    Int subproblem_iter_limit_ = 80;
    Int min_binary_vars_ = 4;
    bool enable_rounding_repair_ = true;

    bool last_executed_solve_ = false;
    Int last_lp_iterations_ = 0;
    double last_work_units_ = 0.0;
};

// ---------------------------------------------------------------------------
// Undercover: fix maximum LP-feasible variable set, solve residual.
// ---------------------------------------------------------------------------

/// Undercover heuristic: fix the maximum set of variables that keeps the
/// LP feasible, then solve the residual restricted LP.
class UncoverHeuristic : public Heuristic {
public:
    void setSubproblemIterLimit(Int limit) { subproblem_iter_limit_ = limit; }
    void setMinFreeVars(Int count) { min_free_vars_ = count; }
    void setEnableRoundingRepair(bool enable) { enable_rounding_repair_ = enable; }

    std::optional<HeuristicSolution> run(
        const LpProblem& problem,
        DualSimplexSolver& lp,
        std::span<const Real> primals,
        Real incumbent) override;

    [[nodiscard]] bool lastExecutedSolve() const { return last_executed_solve_; }
    [[nodiscard]] Int lastFixedCount() const { return last_fixed_count_; }
    [[nodiscard]] Int lastLpIterations() const { return last_lp_iterations_; }
    [[nodiscard]] double lastWorkUnits() const { return last_work_units_; }

    [[nodiscard]] const char* name() const override { return "undercover"; }

private:
    Int subproblem_iter_limit_ = 80;
    Int min_free_vars_ = 2;
    bool enable_rounding_repair_ = true;

    bool last_executed_solve_ = false;
    Int last_fixed_count_ = 0;
    Int last_lp_iterations_ = 0;
    double last_work_units_ = 0.0;
};

// ---------------------------------------------------------------------------
// Rounding variants.
// ---------------------------------------------------------------------------

/// ZI-rounding: round integer variables without worsening the objective.
class ZiRoundingHeuristic : public Heuristic {
public:
    std::optional<HeuristicSolution> run(
        const LpProblem& problem,
        DualSimplexSolver& lp,
        std::span<const Real> primals,
        Real incumbent) override;

    [[nodiscard]] const char* name() const override { return "zirounding"; }
};

/// Shifting heuristic: fix violated constraints by shifting variable values
/// along feasible directions.
class ShiftingHeuristic : public Heuristic {
public:
    void setMaxPasses(Int passes) { max_passes_ = passes; }

    std::optional<HeuristicSolution> run(
        const LpProblem& problem,
        DualSimplexSolver& lp,
        std::span<const Real> primals,
        Real incumbent) override;

    [[nodiscard]] const char* name() const override { return "shifting"; }

private:
    Int max_passes_ = 10;
};

/// Randomized rounding: multiple random roundings with feasibility repair.
class RandomizedRoundingHeuristic : public Heuristic {
public:
    void setNumTrials(Int trials) { num_trials_ = trials; }
    void setMaxRepairPasses(Int passes) { max_repair_passes_ = passes; }

    std::optional<HeuristicSolution> run(
        const LpProblem& problem,
        DualSimplexSolver& lp,
        std::span<const Real> primals,
        Real incumbent) override;

    [[nodiscard]] const char* name() const override { return "randrounding"; }

private:
    Int num_trials_ = 10;
    Int max_repair_passes_ = 5;
};

// ---------------------------------------------------------------------------
// Reduced-cost and propagation-based.
// ---------------------------------------------------------------------------

/// Root reduced-cost heuristic: fix variables by reduced cost, solve
/// restricted LP.
class ReducedCostHeuristic : public Heuristic {
public:
    void setSubproblemIterLimit(Int limit) { subproblem_iter_limit_ = limit; }
    void setRcThreshold(Real threshold) { rc_threshold_ = threshold; }
    void setMinFixedVars(Int count) { min_fixed_vars_ = count; }
    void setEnableRoundingRepair(bool enable) { enable_rounding_repair_ = enable; }

    std::optional<HeuristicSolution> run(
        const LpProblem& problem,
        DualSimplexSolver& lp,
        std::span<const Real> primals,
        Real incumbent) override;

    [[nodiscard]] bool lastExecutedSolve() const { return last_executed_solve_; }
    [[nodiscard]] Int lastFixedCount() const { return last_fixed_count_; }
    [[nodiscard]] Int lastLpIterations() const { return last_lp_iterations_; }
    [[nodiscard]] double lastWorkUnits() const { return last_work_units_; }

    [[nodiscard]] const char* name() const override { return "reducedcost"; }

private:
    Int subproblem_iter_limit_ = 80;
    Real rc_threshold_ = 0.5;
    Int min_fixed_vars_ = 4;
    bool enable_rounding_repair_ = true;

    bool last_executed_solve_ = false;
    Int last_fixed_count_ = 0;
    Int last_lp_iterations_ = 0;
    double last_work_units_ = 0.0;
};

/// Propagation-completion heuristic: propagate fixings to maximum extent,
/// round remainder.
class PropagationCompletionHeuristic : public Heuristic {
public:
    void setMaxPropagationRounds(Int rounds) { max_propagation_rounds_ = rounds; }

    std::optional<HeuristicSolution> run(
        const LpProblem& problem,
        DualSimplexSolver& lp,
        std::span<const Real> primals,
        Real incumbent) override;

    [[nodiscard]] const char* name() const override { return "propcompletion"; }

private:
    Int max_propagation_rounds_ = 10;
};

// ---------------------------------------------------------------------------
// Neighborhood search framework.
// ---------------------------------------------------------------------------

/// 1-opt local search: single variable improvement moves on incumbent.
class OneOptHeuristic : public Heuristic {
public:
    std::optional<HeuristicSolution> run(
        const LpProblem& problem,
        DualSimplexSolver& lp,
        std::span<const Real> primals,
        Real incumbent) override;

    std::optional<HeuristicSolution> run(
        const LpProblem& problem,
        std::span<const Real> incumbent_values,
        Real incumbent);

    [[nodiscard]] const char* name() const override { return "oneopt"; }
};

/// 2-opt local search: pairwise variable swaps on incumbent.
class TwoOptHeuristic : public Heuristic {
public:
    void setMaxPairs(Int pairs) { max_pairs_ = pairs; }

    std::optional<HeuristicSolution> run(
        const LpProblem& problem,
        DualSimplexSolver& lp,
        std::span<const Real> primals,
        Real incumbent) override;

    std::optional<HeuristicSolution> run(
        const LpProblem& problem,
        std::span<const Real> incumbent_values,
        Real incumbent);

    [[nodiscard]] const char* name() const override { return "twoopt"; }

private:
    Int max_pairs_ = 500;
};

/// Adaptive budget controller for primal heuristics.
/// Tracks heuristic effort and adapts tree-call spacing from outcomes.
class HeuristicBudgetManager {
public:
    void setBaseTreeFrequency(Int freq) { base_tree_frequency_ = std::max<Int>(1, freq); }
    void setMaxFrequencyScale(Int scale) { max_frequency_scale_ = std::max<Int>(1, scale); }
    void setMaxWorkShare(Real share) { max_work_share_ = std::max<Real>(0.0, share); }

    [[nodiscard]] bool allowRootHeuristic(double total_work_units) const;
    [[nodiscard]] bool allowTreeHeuristic(Int node_count, double total_work_units) const;

    void recordHeuristicCall(Int node_count, double work_units, bool improved);

    [[nodiscard]] Int currentTreeFrequency() const;
    [[nodiscard]] Int totalCalls() const { return calls_; }
    [[nodiscard]] Int totalSuccesses() const { return successes_; }
    [[nodiscard]] double totalWorkUnits() const { return total_work_units_; }

private:
    Int base_tree_frequency_ = 64;
    Int max_frequency_scale_ = 8;
    Real max_work_share_ = 0.20;

    Int next_tree_node_ = 0;
    Int calls_ = 0;
    Int successes_ = 0;
    double total_work_units_ = 0.0;
};

/// Schedules and manages heuristic execution.
class HeuristicScheduler {
public:
    /// Add a heuristic with a timing policy.
    void addHeuristic(std::unique_ptr<Heuristic> heuristic,
                      HeuristicTiming timing,
                      Int frequency = 10);

    /// Run heuristics appropriate for the current context.
    /// @param problem    The MIP problem.
    /// @param lp         The LP solver (already solved).
    /// @param primals    LP primal values.
    /// @param incumbent  Current best objective.
    /// @param node_count Current node count (0 = root).
    /// @return Best solution found (if any).
    std::optional<HeuristicSolution> run(
        const LpProblem& problem,
        DualSimplexSolver& lp,
        std::span<const Real> primals,
        Real incumbent,
        Int node_count);

    /// Number of registered heuristics.
    [[nodiscard]] Int numHeuristics() const {
        return static_cast<Int>(entries_.size());
    }

    /// Total solutions found across all heuristics.
    [[nodiscard]] Int totalSolutionsFound() const { return total_solutions_; }

private:
    struct Entry {
        std::unique_ptr<Heuristic> heuristic;
        HeuristicTiming timing;
        Int frequency;   ///< For Periodic timing: run every N nodes.
        Int solutions;   ///< Number of solutions found by this heuristic.
    };

    std::vector<Entry> entries_;
    Int total_solutions_ = 0;
};

}  // namespace mipx
