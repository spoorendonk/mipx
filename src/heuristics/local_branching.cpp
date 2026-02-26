#include "mipx/heuristics.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "mipx/branching.h"
#include "mipx/dual_simplex.h"

namespace mipx {

namespace {

constexpr Real kObjectiveTol = 1e-6;
constexpr Real kFeasTol = 1e-6;

bool isIntegerVar(VarType t) {
    return t != VarType::Continuous;
}

bool isBinaryVar(const LpProblem& problem, Index j) {
    if (problem.col_type[j] == VarType::Binary) return true;
    if (problem.col_type[j] == VarType::Continuous) return false;
    return problem.col_lower[j] >= -kFeasTol && problem.col_lower[j] <= kFeasTol &&
           problem.col_upper[j] >= 1.0 - kFeasTol && problem.col_upper[j] <= 1.0 + kFeasTol;
}

Real computeObjective(const LpProblem& problem, std::span<const Real> values) {
    Real obj = problem.obj_offset;
    for (Index j = 0; j < problem.num_cols; ++j) {
        obj += problem.obj[j] * values[j];
    }
    return obj;
}

bool isRowFeasible(const LpProblem& problem, std::span<const Real> values) {
    for (Index i = 0; i < problem.num_rows; ++i) {
        auto row = problem.matrix.row(i);
        Real activity = 0.0;
        for (Index k = 0; k < row.size(); ++k) {
            activity += row.values[k] * values[row.indices[k]];
        }
        if (activity < problem.row_lower[i] - kFeasTol) return false;
        if (activity > problem.row_upper[i] + kFeasTol) return false;
    }
    return true;
}

}  // namespace

std::optional<HeuristicSolution> LocalBranchingHeuristic::run(
    const LpProblem& problem,
    DualSimplexSolver& lp,
    std::span<const Real> primals,
    Real incumbent) {
    return run(problem, lp, primals, incumbent, {});
}

std::optional<HeuristicSolution> LocalBranchingHeuristic::run(
    const LpProblem& problem,
    DualSimplexSolver& lp,
    std::span<const Real> /*primals*/,
    Real incumbent,
    std::span<const Real> incumbent_values) {

    last_executed_solve_ = false;
    last_skipped_no_incumbent_ = false;
    last_skipped_too_small_ = false;
    last_binary_count_ = 0;
    last_lp_iterations_ = 0;
    last_work_units_ = 0.0;

    const Index n = problem.num_cols;
    if (incumbent == kInf || static_cast<Index>(incumbent_values.size()) != n) {
        last_skipped_no_incumbent_ = true;
        return std::nullopt;
    }

    if (neighborhood_size_ <= 0) {
        last_skipped_too_small_ = true;
        return std::nullopt;
    }

    std::vector<Index> bin_cols;
    bin_cols.reserve(n);
    for (Index j = 0; j < n; ++j) {
        if (isBinaryVar(problem, j)) {
            bin_cols.push_back(j);
        }
    }

    last_binary_count_ = static_cast<Int>(bin_cols.size());
    if (last_binary_count_ < min_binary_vars_) {
        last_skipped_too_small_ = true;
        return std::nullopt;
    }

    std::vector<Real> row_vals;
    row_vals.reserve(bin_cols.size());
    Real constant = 0.0;
    for (Index col : bin_cols) {
        const bool inc_one = incumbent_values[col] >= 0.5;
        if (inc_one) {
            row_vals.push_back(-1.0);
            constant += 1.0;
        } else {
            row_vals.push_back(1.0);
        }
    }

    const Real rhs = static_cast<Real>(neighborhood_size_) - constant;
    if (rhs < -kFeasTol) {
        last_skipped_too_small_ = true;
        return std::nullopt;
    }

    const Int previous_iter_limit = lp.getIterationLimit();
    const auto saved_basis = lp.getBasis();
    const Index added_row = lp.numRows();

    std::vector<Index> starts = {0, static_cast<Index>(bin_cols.size())};
    std::vector<Real> lower = {-kInf};
    std::vector<Real> upper = {rhs};
    lp.addRows(starts, bin_cols, row_vals, lower, upper);

    lp.setIterationLimit(subproblem_iter_limit_);
    auto result = lp.solve();
    last_executed_solve_ = true;
    last_lp_iterations_ = result.iterations;
    last_work_units_ = result.work_units;

    std::optional<HeuristicSolution> best;
    if (result.status == Status::Optimal && result.objective < incumbent - kObjectiveTol) {
        auto candidate = lp.getPrimalValues();
        bool integer_feasible = true;
        for (Index j = 0; j < n; ++j) {
            if (!isIntegerVar(problem.col_type[j])) continue;
            if (!isIntegral(candidate[j], kFeasTol)) {
                integer_feasible = false;
                break;
            }
        }

        if (integer_feasible) {
            best = HeuristicSolution{std::move(candidate), result.objective};
        } else if (enable_rounding_repair_) {
            for (Index j = 0; j < n; ++j) {
                if (!isIntegerVar(problem.col_type[j])) continue;
                Real lb = -kInf;
                Real ub = kInf;
                lp.getColBounds(j, lb, ub);
                Real rounded = std::round(candidate[j]);
                if (lb != -kInf) rounded = std::max(rounded, lb);
                if (ub != kInf) rounded = std::min(rounded, ub);
                candidate[j] = rounded;
            }
            if (isRowFeasible(problem, candidate)) {
                Real repaired_obj = computeObjective(problem, candidate);
                if (repaired_obj < incumbent - kObjectiveTol) {
                    best = HeuristicSolution{std::move(candidate), repaired_obj};
                }
            }
        }
    }

    const std::vector<Index> remove_rows = {added_row};
    lp.removeRows(remove_rows);
    lp.setBasis(saved_basis);
    lp.setIterationLimit(previous_iter_limit);

    return best;
}

}  // namespace mipx
