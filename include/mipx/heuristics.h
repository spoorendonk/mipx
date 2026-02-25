#pragma once

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
