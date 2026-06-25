#include "common.h"
#include "mipx/dual_simplex.h"
#include "mipx/heuristics.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace mipx {

using namespace heuristic_detail;

std::optional<HeuristicSolution> UncoverHeuristic::run(const LpProblem& problem,
                                                       DualSimplexSolver& lp,
                                                       std::span<const Real> primals,
                                                       Real incumbent) {
    last_executed_solve_ = false;
    last_fixed_count_ = 0;
    last_lp_iterations_ = 0;
    last_work_units_ = 0.0;

    const Index n = problem.num_cols;

    // Strategy: fix integer variables that are near-integral in the LP
    // relaxation to their rounded values. For each variable, estimate
    // whether fixing it would maintain LP feasibility by checking
    // its "coverage": how many constraints it appears in with other
    // free integer variables.

    struct VarInfo {
        Index col;
        Real frac;
        Int row_count;  // Number of constraint rows containing this variable.
    };

    std::vector<VarInfo> int_vars;
    for (Index j = 0; j < n; ++j) {
        if (!isIntegerVar(problem.col_type[j])) {
            continue;
        }
        Real frac = std::abs(primals[j] - std::round(primals[j]));
        auto col_view = problem.matrix.col(j);
        int_vars.push_back({j, frac, col_view.size()});
    }

    if (int_vars.empty()) {
        return std::nullopt;
    }

    // Sort by fractionality (ascending): fix near-integral variables first.
    // Among tied fractionalities, prefer variables appearing in fewer rows
    // (less constraint coverage = safer to fix).
    std::sort(int_vars.begin(), int_vars.end(), [](const VarInfo& a, const VarInfo& b) {
        if (std::abs(a.frac - b.frac) > 1e-8) {
            return a.frac < b.frac;
        }
        return a.row_count < b.row_count;
    });

    // Fix as many as possible, leaving at least min_free_vars_ free.
    const Int max_fixable = static_cast<Int>(int_vars.size()) - min_free_vars_;
    if (max_fixable <= 0) {
        return std::nullopt;
    }

    struct SavedBound {
        Index col;
        Real lower;
        Real upper;
    };

    const Int previous_iter_limit = lp.getIterationLimit();
    const auto saved_basis = lp.getBasis();
    std::vector<SavedBound> saved_bounds;
    saved_bounds.reserve(int_vars.size());

    Int fixed = 0;
    for (const auto& info : int_vars) {
        if (fixed >= max_fixable) {
            break;
        }

        Real rounded = std::round(primals[info.col]);
        Real lower = -kInf;
        Real upper = kInf;
        lp.getColBounds(info.col, lower, upper);
        rounded = std::max(rounded, lower);
        rounded = std::min(rounded, upper);

        // Skip if already effectively fixed.
        if (std::abs(lower - rounded) <= kFeasTol && std::abs(upper - rounded) <= kFeasTol) {
            continue;
        }

        saved_bounds.push_back({info.col, lower, upper});
        lp.setColBounds(info.col, rounded, rounded);
        ++fixed;
    }

    last_fixed_count_ = fixed;

    auto restoreState = [&]() {
        for (const auto& sb : saved_bounds) {
            lp.setColBounds(sb.col, sb.lower, sb.upper);
        }
        lp.setBasis(saved_basis);
        lp.setIterationLimit(previous_iter_limit);
    };

    if (fixed == 0) {
        restoreState();
        return std::nullopt;
    }

    lp.setIterationLimit(subproblem_iter_limit_);
    auto result = lp.solve();
    last_executed_solve_ = true;
    last_lp_iterations_ = result.iterations;
    last_work_units_ = result.work_units;

    std::optional<HeuristicSolution> best;
    if (result.status == Status::Optimal &&
        betterObjective(problem.sense, result.objective, incumbent)) {
        auto candidate = lp.getPrimalValues();
        bool integer_feasible = true;
        for (Index j = 0; j < n; ++j) {
            if (!isIntegerVar(problem.col_type[j])) {
                continue;
            }
            if (!isIntegral(candidate[j], kFeasTol)) {
                integer_feasible = false;
                break;
            }
        }

        if (integer_feasible) {
            best = HeuristicSolution{std::move(candidate), result.objective};
        } else if (enable_rounding_repair_) {
            for (Index j = 0; j < n; ++j) {
                if (!isIntegerVar(problem.col_type[j])) {
                    continue;
                }
                Real lb = -kInf;
                Real ub = kInf;
                lp.getColBounds(j, lb, ub);
                Real rounded = std::round(candidate[j]);
                if (lb != -kInf) {
                    rounded = std::max(rounded, lb);
                }
                if (ub != kInf) {
                    rounded = std::min(rounded, ub);
                }
                candidate[j] = rounded;
            }
            if (isRowFeasible(problem, candidate)) {
                Real repaired_obj = computeObjective(problem, candidate);
                if (betterObjective(problem.sense, repaired_obj, incumbent)) {
                    best = HeuristicSolution{std::move(candidate), repaired_obj};
                }
            }
        }
    }

    restoreState();
    return best;
}

}  // namespace mipx
