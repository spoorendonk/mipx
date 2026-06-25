#include "common.h"
#include "mipx/dual_simplex.h"
#include "mipx/heuristics.h"

#include <algorithm>
#include <cmath>

namespace mipx {

using namespace heuristic_detail;

std::optional<HeuristicSolution> FeasibilityPumpHeuristic::run(const LpProblem& problem,
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
            if (!isIntegerVar(problem.col_type[j])) {
                continue;
            }
            rounded[j] = std::round(current_lp[j]);
            rounded[j] = std::max(rounded[j], problem.col_lower[j]);
            rounded[j] = std::min(rounded[j], problem.col_upper[j]);
        }

        // If rounded point is feasible, evaluate incumbent improvement.
        if (isRowFeasible(problem, rounded)) {
            Real rounded_obj = computeObjective(problem, rounded);
            if (betterObjective(problem.sense, rounded_obj, incumbent)) {
                best = HeuristicSolution{rounded, rounded_obj};
                break;
            }
        }

        std::fill(guide_obj.begin(), guide_obj.end(), 0.0);
        Int active_guide = 0;
        for (Index j = 0; j < n; ++j) {
            if (!isIntegerVar(problem.col_type[j])) {
                continue;
            }
            Real diff = current_lp[j] - rounded[j];
            if (diff > 1e-9) {
                guide_obj[j] = 1.0;  // push variable down
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
                if (!isIntegerVar(problem.col_type[j])) {
                    continue;
                }
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
                if (!isIntegerVar(problem.col_type[j])) {
                    continue;
                }
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

        if (active_guide == 0) {
            break;
        }

        lp.setObjective(guide_obj);
        lp.setIterationLimit(subproblem_iter_limit_);
        auto result = lp.solve();
        last_lp_iterations_ += result.iterations;
        last_work_units_ += result.work_units;
        if (result.status != Status::Optimal) {
            break;
        }
        current_lp = lp.getPrimalValues();
    }

    lp.setObjective(original_obj);
    lp.setBasis(saved_basis);
    lp.setIterationLimit(previous_iter_limit);

    return best;
}

}  // namespace mipx
