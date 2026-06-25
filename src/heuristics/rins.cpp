#include "common.h"
#include "mipx/dual_simplex.h"
#include "mipx/heuristics.h"

#include <algorithm>
#include <cmath>

namespace mipx {

using namespace heuristic_detail;

namespace {

Int countIntegerColumns(const LpProblem& problem) {
    Int count = 0;
    for (Index j = 0; j < problem.num_cols; ++j) {
        if (isIntegerVar(problem.col_type[j])) {
            ++count;
        }
    }
    return count;
}

}  // namespace

std::optional<HeuristicSolution> RinsHeuristic::run(const LpProblem& problem, DualSimplexSolver& lp,
                                                    std::span<const Real> primals, Real incumbent) {
    return run(problem, lp, primals, incumbent, {});
}

std::optional<HeuristicSolution> RinsHeuristic::run(const LpProblem& problem, DualSimplexSolver& lp,
                                                    std::span<const Real> primals, Real incumbent,
                                                    std::span<const Real> incumbent_values) {
    last_fixed_count_ = 0;
    last_executed_solve_ = false;
    last_skipped_no_incumbent_ = false;
    last_skipped_few_fixes_ = false;
    last_found_solution_ = false;
    last_lp_iterations_ = 0;
    last_work_units_ = 0.0;

    const Index n = problem.num_cols;
    if (incumbent == kInf || static_cast<Index>(incumbent_values.size()) != n) {
        last_skipped_no_incumbent_ = true;
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
    saved_bounds.reserve(n);

    const Int integer_cols = countIntegerColumns(problem);
    const Int min_by_rate = static_cast<Int>(std::ceil(
        std::max(0.0, min_fixed_rate_) * static_cast<Real>(std::max<Int>(1, integer_cols))));
    const Int min_required = std::max<Int>(min_fixed_vars_, min_by_rate);

    for (Index j = 0; j < n; ++j) {
        if (!isIntegerVar(problem.col_type[j])) {
            continue;
        }

        Real lp_round = std::round(primals[j]);
        if (std::abs(primals[j] - lp_round) > agreement_tol_) {
            continue;
        }

        Real inc_round = std::round(incumbent_values[j]);
        if (std::abs(incumbent_values[j] - inc_round) > agreement_tol_) {
            continue;
        }
        if (std::abs(lp_round - inc_round) > agreement_tol_) {
            continue;
        }

        Real lower = -kInf;
        Real upper = kInf;
        lp.getColBounds(j, lower, upper);
        if (lp_round < lower - agreement_tol_ || lp_round > upper + agreement_tol_) {
            continue;
        }
        if (std::abs(lower - lp_round) <= agreement_tol_ &&
            std::abs(upper - lp_round) <= agreement_tol_) {
            continue;
        }

        saved_bounds.push_back({j, lower, upper});
        lp.setColBounds(j, lp_round, lp_round);
    }

    last_fixed_count_ = static_cast<Int>(saved_bounds.size());

    auto restoreState = [&]() {
        for (const auto& sb : saved_bounds) {
            lp.setColBounds(sb.col, sb.lower, sb.upper);
        }
        lp.setBasis(saved_basis);
        lp.setIterationLimit(previous_iter_limit);
    };

    if (last_fixed_count_ < min_required) {
        last_skipped_few_fixes_ = true;
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
                Real lower = -kInf;
                Real upper = kInf;
                lp.getColBounds(j, lower, upper);
                Real rounded = std::round(candidate[j]);
                if (lower != -kInf) {
                    rounded = std::max(rounded, lower);
                }
                if (upper != kInf) {
                    rounded = std::min(rounded, upper);
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

    if (best.has_value()) {
        last_found_solution_ = true;
    }

    restoreState();
    return best;
}

}  // namespace mipx
