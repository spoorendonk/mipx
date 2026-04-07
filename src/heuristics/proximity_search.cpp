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

std::optional<HeuristicSolution> ProximitySearchHeuristic::run(
    [[maybe_unused]] const LpProblem& problem,
    [[maybe_unused]] DualSimplexSolver& lp,
    [[maybe_unused]] std::span<const Real> primals,
    [[maybe_unused]] Real incumbent) {
    // Proximity search requires an incumbent solution.
    return std::nullopt;
}

std::optional<HeuristicSolution> ProximitySearchHeuristic::run(
    const LpProblem& problem,
    DualSimplexSolver& lp,
    [[maybe_unused]] std::span<const Real> primals,
    Real incumbent,
    std::span<const Real> incumbent_values) {

    last_executed_solve_ = false;
    last_lp_iterations_ = 0;
    last_work_units_ = 0.0;

    const Index n = problem.num_cols;
    if (incumbent == kInf || static_cast<Index>(incumbent_values.size()) != n) {
        return std::nullopt;
    }

    // Collect binary variables for the Hamming-distance objective.
    std::vector<Index> bin_cols;
    bin_cols.reserve(n);
    for (Index j = 0; j < n; ++j) {
        if (isBinaryVar(problem, j)) bin_cols.push_back(j);
    }
    if (static_cast<Int>(bin_cols.size()) < min_binary_vars_) {
        return std::nullopt;
    }

    // Proximity search: replace objective with Hamming distance to incumbent,
    // add cutoff constraint on original objective.
    // min sum_{j binary} |x_j - x*_j|
    //   = sum_{j: x*_j=1} (1 - x_j) + sum_{j: x*_j=0} x_j
    //   = sum_{j: x*_j=1} (-x_j) + sum_{j: x*_j=0} (x_j) + constant

    const Int previous_iter_limit = lp.getIterationLimit();
    const auto saved_basis = lp.getBasis();
    const std::vector<Real> original_obj(problem.obj.begin(), problem.obj.end());

    // Build Hamming distance objective.
    std::vector<Real> prox_obj(n, 0.0);
    for (Index j : bin_cols) {
        const bool inc_one = incumbent_values[j] >= 0.5;
        prox_obj[j] = inc_one ? -1.0 : 1.0;
    }

    // Add cutoff constraint: c^T x <= incumbent - epsilon.
    const Real cutoff = incumbent - kObjectiveTol;
    std::vector<Index> cutoff_cols;
    std::vector<Real> cutoff_vals;
    cutoff_cols.reserve(n);
    cutoff_vals.reserve(n);
    for (Index j = 0; j < n; ++j) {
        if (std::abs(problem.obj[j]) > 1e-15) {
            cutoff_cols.push_back(j);
            cutoff_vals.push_back(problem.obj[j]);
        }
    }

    const Index added_row = lp.numRows();
    std::vector<Index> starts = {0, static_cast<Index>(cutoff_cols.size())};
    std::vector<Real> lower = {-kInf};
    std::vector<Real> upper = {cutoff - problem.obj_offset};
    lp.addRows(starts, cutoff_cols, cutoff_vals, lower, upper);

    lp.setObjective(prox_obj);
    lp.setIterationLimit(subproblem_iter_limit_);
    auto result = lp.solve();
    last_executed_solve_ = true;
    last_lp_iterations_ = result.iterations;
    last_work_units_ = result.work_units;

    std::optional<HeuristicSolution> best;
    if (result.status == Status::Optimal) {
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
            Real candidate_obj = computeObjective(problem, candidate);
            if (candidate_obj < incumbent - kObjectiveTol) {
                best = HeuristicSolution{std::move(candidate), candidate_obj};
            }
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

    // Restore LP state.
    const std::vector<Index> remove_rows = {added_row};
    lp.removeRows(remove_rows);
    lp.setObjective(original_obj);
    lp.setBasis(saved_basis);
    lp.setIterationLimit(previous_iter_limit);

    return best;
}

}  // namespace mipx
