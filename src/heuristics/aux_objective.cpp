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

Real nearestFeasibleInteger(Real x, Real lb, Real ub) {
    Real lo = std::ceil(lb - kFeasTol);
    Real hi = std::floor(ub + kFeasTol);
    if (lo > hi) return x;
    Real r = std::round(x);
    if (r < lo) r = lo;
    if (r > hi) r = hi;
    return r;
}

}  // namespace

std::optional<HeuristicSolution> AuxObjectiveHeuristic::run(
    const LpProblem& problem,
    DualSimplexSolver& lp,
    std::span<const Real> primals,
    Real incumbent) {
    return run(problem, lp, primals, incumbent, {});
}

std::optional<HeuristicSolution> AuxObjectiveHeuristic::run(
    const LpProblem& problem,
    DualSimplexSolver& lp,
    std::span<const Real> primals,
    Real incumbent,
    std::span<const Real> incumbent_values) {

    last_executed_solve_ = false;
    last_skipped_too_small_ = false;
    last_active_integer_vars_ = 0;
    last_lp_iterations_ = 0;
    last_work_units_ = 0.0;

    const Index n = problem.num_cols;
    if (static_cast<Index>(primals.size()) != n) {
        last_skipped_too_small_ = true;
        return std::nullopt;
    }

    std::vector<Real> aux_obj(n, 0.0);
    std::vector<Real> original_obj(problem.obj.begin(), problem.obj.end());
    const bool have_incumbent_values = static_cast<Index>(incumbent_values.size()) == n;

    for (Index j = 0; j < n; ++j) {
        if (!isIntegerVar(problem.col_type[j])) continue;

        Real lb = -kInf;
        Real ub = kInf;
        lp.getColBounds(j, lb, ub);
        Real target = primals[j];
        if (have_incumbent_values) {
            target = nearestFeasibleInteger(incumbent_values[j], lb, ub);
        } else {
            target = nearestFeasibleInteger(primals[j], lb, ub);
        }

        if (primals[j] > target + kFeasTol) {
            aux_obj[j] = 1.0;
            ++last_active_integer_vars_;
        } else if (primals[j] < target - kFeasTol) {
            aux_obj[j] = -1.0;
            ++last_active_integer_vars_;
        }
    }

    if (last_active_integer_vars_ < min_active_integer_vars_) {
        last_skipped_too_small_ = true;
        return std::nullopt;
    }

    const Int previous_iter_limit = lp.getIterationLimit();
    const auto saved_basis = lp.getBasis();

    lp.setObjective(aux_obj);
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

        if (integer_feasible && isRowFeasible(problem, candidate)) {
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
                candidate[j] = nearestFeasibleInteger(candidate[j], lb, ub);
            }
            if (isRowFeasible(problem, candidate)) {
                Real repaired_obj = computeObjective(problem, candidate);
                if (repaired_obj < incumbent - kObjectiveTol) {
                    best = HeuristicSolution{std::move(candidate), repaired_obj};
                }
            }
        }
    }

    lp.setObjective(original_obj);
    lp.setBasis(saved_basis);
    lp.setIterationLimit(previous_iter_limit);

    return best;
}

}  // namespace mipx
