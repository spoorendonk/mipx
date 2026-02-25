#include "mipx/heuristics.h"

#include <cmath>

#include "mipx/branching.h"
#include "mipx/dual_simplex.h"

namespace mipx {

// ---------------------------------------------------------------------------
// DivingHeuristic: common diving loop
// ---------------------------------------------------------------------------

std::optional<HeuristicSolution> DivingHeuristic::dive(
    const LpProblem& problem,
    DualSimplexSolver& lp,
    std::span<const Real> primals,
    Real incumbent) {

    const Index n = problem.num_cols;

    // Save original bounds so we can restore them after diving.
    std::vector<Real> orig_lower(problem.col_lower.begin(), problem.col_lower.end());
    std::vector<Real> orig_upper(problem.col_upper.begin(), problem.col_upper.end());

    // Track current effective bounds (modified during diving).
    std::vector<Real> cur_lower = orig_lower;
    std::vector<Real> cur_upper = orig_upper;

    // Work with a mutable copy of primals.
    std::vector<Real> current_primals(primals.begin(), primals.end());

    Int consecutive_infeasible = 0;
    std::optional<HeuristicSolution> best;

    for (Int dive = 0; dive < max_dives_; ++dive) {
        // Select variable to fix.
        auto [var, fix_val] = selectVariable(problem, current_primals, problem.obj);
        if (var < 0) {
            // All integer variables are integral - we have a feasible solution!
            Real obj = problem.obj_offset;
            for (Index j = 0; j < n; ++j) {
                obj += problem.obj[j] * current_primals[j];
            }
            if (obj < incumbent - 1e-6) {
                best = HeuristicSolution{current_primals, obj};
            }
            break;
        }

        // Fix the variable by setting both bounds to fix_val.
        cur_lower[var] = fix_val;
        cur_upper[var] = fix_val;
        lp.setColBounds(var, fix_val, fix_val);

        // Re-solve the LP.
        lp.setIterationLimit(200);  // Limit iterations per dive.
        auto result = lp.solve();

        if (result.status != Status::Optimal) {
            ++consecutive_infeasible;
            // Undo the fixing.
            cur_lower[var] = orig_lower[var];
            cur_upper[var] = orig_upper[var];
            lp.setColBounds(var, orig_lower[var], orig_upper[var]);

            if (consecutive_infeasible >= backtrack_limit_) {
                break;
            }
            continue;
        }

        consecutive_infeasible = 0;
        current_primals = lp.getPrimalValues();

        // Check if we improved over incumbent.
        if (result.objective >= incumbent - 1e-6) {
            break;  // Can't beat incumbent, stop diving.
        }

        // Check if solution is integer feasible.
        bool feasible = true;
        for (Index j = 0; j < n; ++j) {
            if (problem.col_type[j] == VarType::Continuous) continue;
            if (!isIntegral(current_primals[j])) {
                feasible = false;
                break;
            }
        }

        if (feasible) {
            Real obj = problem.obj_offset;
            for (Index j = 0; j < n; ++j) {
                obj += problem.obj[j] * current_primals[j];
            }
            if (obj < incumbent - 1e-6) {
                best = HeuristicSolution{current_primals, obj};
            }
            break;
        }
    }

    // Restore original bounds.
    for (Index j = 0; j < n; ++j) {
        if (cur_lower[j] != orig_lower[j] || cur_upper[j] != orig_upper[j]) {
            lp.setColBounds(j, orig_lower[j], orig_upper[j]);
        }
    }

    // Restore iteration limit.
    lp.setIterationLimit(1000000);

    // Re-solve to restore the LP state for the B&B solver.
    // Only needed if we modified bounds during diving.
    lp.solve();

    return best;
}

// ---------------------------------------------------------------------------
// FractionalDiving
// ---------------------------------------------------------------------------

std::optional<HeuristicSolution> FractionalDiving::run(
    const LpProblem& problem,
    DualSimplexSolver& lp,
    std::span<const Real> primals,
    Real incumbent) {
    return dive(problem, lp, primals, incumbent);
}

std::pair<Index, Real> FractionalDiving::selectVariable(
    const LpProblem& problem,
    std::span<const Real> primals,
    [[maybe_unused]] std::span<const Real> obj) const {

    // Select the most fractional integer variable (closest to 0.5).
    Index best_var = -1;
    Real best_frac = 0.0;

    for (Index j = 0; j < problem.num_cols; ++j) {
        if (problem.col_type[j] == VarType::Continuous) continue;
        if (isIntegral(primals[j])) continue;

        Real frac = fractionality(primals[j]);
        Real dist_to_half = std::abs(frac - 0.5);
        // We want the one closest to 0.5 (most fractional).
        if (best_var < 0 || dist_to_half < best_frac) {
            best_var = j;
            best_frac = dist_to_half;
        }
    }

    if (best_var < 0) return {-1, 0.0};

    // Round to nearest integer.
    Real fix_val = std::round(primals[best_var]);
    fix_val = std::max(fix_val, problem.col_lower[best_var]);
    fix_val = std::min(fix_val, problem.col_upper[best_var]);
    return {best_var, fix_val};
}

// ---------------------------------------------------------------------------
// CoefficientDiving
// ---------------------------------------------------------------------------

std::optional<HeuristicSolution> CoefficientDiving::run(
    const LpProblem& problem,
    DualSimplexSolver& lp,
    std::span<const Real> primals,
    Real incumbent) {
    return dive(problem, lp, primals, incumbent);
}

std::pair<Index, Real> CoefficientDiving::selectVariable(
    const LpProblem& problem,
    std::span<const Real> primals,
    std::span<const Real> obj) const {

    // Select the fractional integer variable where rounding costs the least
    // in terms of objective degradation per unit change.
    Index best_var = -1;
    Real best_score = kInf;

    for (Index j = 0; j < problem.num_cols; ++j) {
        if (problem.col_type[j] == VarType::Continuous) continue;
        if (isIntegral(primals[j])) continue;

        Real val = primals[j];
        Real floor_val = std::floor(val);
        Real ceil_val = std::ceil(val);

        // Compute objective cost of rounding down vs up.
        // For minimization: positive obj means rounding down is cheaper.
        Real cost_down = std::abs(obj[j]) * (val - floor_val);
        Real cost_up = std::abs(obj[j]) * (ceil_val - val);

        Real score = std::min(cost_down, cost_up);
        if (score < best_score) {
            best_score = score;
            best_var = j;
        }
    }

    if (best_var < 0) return {-1, 0.0};

    // Round in the cheaper direction.
    Real val = primals[best_var];
    Real floor_val = std::floor(val);
    Real ceil_val = std::ceil(val);

    Real fix_val;
    // For minimization: if obj > 0, prefer rounding down; if obj < 0, prefer rounding up.
    if (obj[best_var] >= 0) {
        fix_val = floor_val;
    } else {
        fix_val = ceil_val;
    }

    fix_val = std::max(fix_val, problem.col_lower[best_var]);
    fix_val = std::min(fix_val, problem.col_upper[best_var]);
    return {best_var, fix_val};
}

// ---------------------------------------------------------------------------
// HeuristicScheduler
// ---------------------------------------------------------------------------

void HeuristicScheduler::addHeuristic(std::unique_ptr<Heuristic> heuristic,
                                       HeuristicTiming timing,
                                       Int frequency) {
    entries_.push_back({std::move(heuristic), timing, frequency, 0});
}

std::optional<HeuristicSolution> HeuristicScheduler::run(
    const LpProblem& problem,
    DualSimplexSolver& lp,
    std::span<const Real> primals,
    Real incumbent,
    Int node_count) {

    std::optional<HeuristicSolution> best;

    for (auto& entry : entries_) {
        bool should_run = false;

        switch (entry.timing) {
            case HeuristicTiming::Root:
                should_run = (node_count == 0);
                break;
            case HeuristicTiming::EveryNode:
                should_run = true;
                break;
            case HeuristicTiming::Periodic:
                should_run = (node_count == 0) || (node_count % entry.frequency == 0);
                break;
        }

        if (!should_run) continue;

        auto result = entry.heuristic->run(problem, lp, primals, incumbent);
        if (result) {
            ++entry.solutions;
            ++total_solutions_;
            // Keep the best solution found.
            if (!best || result->objective < best->objective) {
                best = std::move(result);
                incumbent = best->objective;  // Update for subsequent heuristics.
            }
        }
    }

    return best;
}

}  // namespace mipx
