#include "mipx/heuristics.h"

#include <algorithm>
#include <cmath>

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

}  // namespace

std::optional<HeuristicSolution> FeasibilityPumpHeuristic::run(
    const LpProblem& problem,
    DualSimplexSolver& lp,
    std::span<const Real> primals,
    Real incumbent) {

    const Index n = problem.num_cols;
    const Int previous_iter_limit = lp.getIterationLimit();
    const auto saved_basis = lp.getBasis();
    const std::vector<Real> original_obj(problem.obj.begin(), problem.obj.end());
    last_lp_iterations_ = 0;
    last_work_units_ = 0.0;

    std::vector<Real> current_lp(primals.begin(), primals.end());
    std::vector<Real> rounded(n, 0.0);
    std::vector<Real> previous_rounded(n, 0.0);
    std::vector<Real> guide_obj(n, 0.0);

    bool have_previous_rounded = false;
    Int cycle_count = 0;

    std::optional<HeuristicSolution> best;

    for (Int iter = 0; iter < max_iterations_; ++iter) {
        // Round the current LP point to an integer candidate.
        rounded = current_lp;
        for (Index j = 0; j < n; ++j) {
            if (!isIntegerVar(problem.col_type[j])) continue;
            rounded[j] = std::round(current_lp[j]);
            rounded[j] = std::max(rounded[j], problem.col_lower[j]);
            rounded[j] = std::min(rounded[j], problem.col_upper[j]);
        }

        // If rounded point is feasible, evaluate incumbent improvement.
        if (isRowFeasible(problem, rounded)) {
            Real rounded_obj = computeObjective(problem, rounded);
            if (rounded_obj < incumbent - kObjectiveTol) {
                best = HeuristicSolution{rounded, rounded_obj};
                break;
            }
        }

        std::fill(guide_obj.begin(), guide_obj.end(), 0.0);
        Int active_guide = 0;
        for (Index j = 0; j < n; ++j) {
            if (!isIntegerVar(problem.col_type[j])) continue;
            Real diff = current_lp[j] - rounded[j];
            if (diff > 1e-9) {
                guide_obj[j] = 1.0;   // push variable down
                ++active_guide;
            } else if (diff < -1e-9) {
                guide_obj[j] = -1.0;  // push variable up
                ++active_guide;
            }
        }

        // Break FP cycling by perturbing one coefficient deterministically.
        bool same_as_previous = false;
        if (have_previous_rounded) {
            same_as_previous = true;
            for (Index j = 0; j < n; ++j) {
                if (!isIntegerVar(problem.col_type[j])) continue;
                if (std::abs(rounded[j] - previous_rounded[j]) > 1e-9) {
                    same_as_previous = false;
                    break;
                }
            }
        }
        previous_rounded = rounded;
        have_previous_rounded = true;

        if (same_as_previous) {
            ++cycle_count;
        } else {
            cycle_count = 0;
        }

        if (cycle_count >= cycle_perturb_period_) {
            Index frac_var = -1;
            Real best_frac = -1.0;
            for (Index j = 0; j < n; ++j) {
                if (!isIntegerVar(problem.col_type[j])) continue;
                Real frac = fractionality(current_lp[j]);
                if (frac > best_frac) {
                    best_frac = frac;
                    frac_var = j;
                }
            }
            if (frac_var >= 0) {
                if (std::abs(guide_obj[frac_var]) < 0.5) {
                    guide_obj[frac_var] = (current_lp[frac_var] >= rounded[frac_var]) ? 1.0 : -1.0;
                    ++active_guide;
                } else {
                    guide_obj[frac_var] = -guide_obj[frac_var];
                }
            }
            cycle_count = 0;
        }

        if (active_guide == 0) break;

        lp.setObjective(guide_obj);
        lp.setIterationLimit(subproblem_iter_limit_);
        auto result = lp.solve();
        last_lp_iterations_ += result.iterations;
        last_work_units_ += result.work_units;
        if (result.status != Status::Optimal) break;
        current_lp = lp.getPrimalValues();
    }

    lp.setObjective(original_obj);
    lp.setBasis(saved_basis);
    lp.setIterationLimit(previous_iter_limit);

    return best;
}

}  // namespace mipx
