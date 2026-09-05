#include "mipx/reduced_cost_fixer.h"

#include <algorithm>
#include <cmath>

namespace mipx {

void ReducedCostFixer::load(const LpProblem& problem) {
    num_cols_ = problem.num_cols;
    col_type_ = problem.col_type;
    global_lower_ = problem.col_lower;
    global_upper_ = problem.col_upper;
    global_changes_.clear();
    stats_ = {};
    loaded_ = true;
}

void ReducedCostFixer::reset() {
    // Roll the recorded tightenings back out of the global bound arrays, in
    // reverse order, so the arrays return to the bounds seen at load(). Simply
    // dropping the change log would leave the tightened arrays in place, and
    // enforceGlobalFixings() would then be free to re-apply bounds that are no
    // longer justified as soon as a new change is recorded.
    for (auto it = global_changes_.rbegin(); it != global_changes_.rend(); ++it) {
        global_lower_[it->variable] = it->old_lower;
        global_upper_[it->variable] = it->old_upper;
    }
    global_changes_.clear();
    stats_ = {};
}

Int ReducedCostFixer::rcTighten(std::span<const Real> reduced_costs, std::span<const Real> primals,
                                Real gap, std::vector<Real>& col_lower,
                                std::vector<Real>& col_upper,
                                std::vector<Index>& tightened_vars) const {
    if (gap <= 0.0 || !std::isfinite(gap))
        return 0;
    if (static_cast<Index>(reduced_costs.size()) < num_cols_)
        return 0;
    // Without the primal point complementarity cannot be verified, and the
    // tightening below would be unsound for any non-vertex LP solution.
    if (static_cast<Index>(primals.size()) < num_cols_)
        return 0;

    Int count = 0;

    for (Index j = 0; j < num_cols_; ++j) {
        if (!std::isfinite(col_lower[j]) || !std::isfinite(col_upper[j])) {
            continue;
        }
        if (col_lower[j] >= col_upper[j] - kBoundTol)
            continue;

        const Real rc = reduced_costs[j];

        // For minimization:
        //   Positive RC => variable wants to decrease => at lower bound.
        //   rc * delta_ub >= gap => new_ub = lb + gap / rc.
        //   Negative RC => variable wants to increase => at upper bound.
        //   |rc| * delta_lb >= gap => new_lb = ub - gap / |rc| = ub + gap / rc.
        const Real x = primals[j];

        if (rc > kRcTol) {
            // Variable must actually sit at its lower bound for the reduced
            // cost to price that bound; otherwise skip it.
            if (std::abs(x - col_lower[j]) > kPrimalTol)
                continue;
            // New upper bound: lb + gap / rc.
            const Real new_ub = col_lower[j] + gap / rc;
            if (new_ub < col_upper[j] - kBoundTol) {
                Real bounded_ub = new_ub;
                // Integer rounding: floor for upper bound.
                if (col_type_[j] != VarType::Continuous) {
                    bounded_ub = std::floor(bounded_ub + kFeasTol);
                }
                bounded_ub = std::max(bounded_ub, col_lower[j]);
                if (bounded_ub < col_upper[j] - kBoundTol) {
                    if (bounded_ub < col_lower[j] - kBoundTol) {
                        return -1;  // Infeasible.
                    }
                    col_upper[j] = bounded_ub;
                    tightened_vars.push_back(j);
                    ++count;
                }
            }
        } else if (rc < -kRcTol) {
            // Mirror case: the variable must actually sit at its upper bound.
            if (std::abs(x - col_upper[j]) > kPrimalTol)
                continue;
            // New lower bound: ub + gap / rc (rc is negative, so this increases lb).
            const Real new_lb = col_upper[j] + gap / rc;
            if (new_lb > col_lower[j] + kBoundTol) {
                Real bounded_lb = new_lb;
                // Integer rounding: ceil for lower bound.
                if (col_type_[j] != VarType::Continuous) {
                    bounded_lb = std::ceil(bounded_lb - kFeasTol);
                }
                bounded_lb = std::min(bounded_lb, col_upper[j]);
                if (bounded_lb > col_lower[j] + kBoundTol) {
                    if (bounded_lb > col_upper[j] + kBoundTol) {
                        return -1;  // Infeasible.
                    }
                    col_lower[j] = bounded_lb;
                    tightened_vars.push_back(j);
                    ++count;
                }
            }
        }
    }

    return count;
}

bool ReducedCostFixer::applyGlobalFixing(std::span<const Real> reduced_costs,
                                         std::span<const Real> primals, Real lp_objective,
                                         Real incumbent, std::vector<Real>& col_lower,
                                         std::vector<Real>& col_upper,
                                         std::vector<Index>& tightened_vars) {
    if (!loaded_)
        return true;
    if (incumbent >= kInf)
        return true;
    if (lp_objective >= incumbent - kBoundTol)
        return true;

    const Real gap = incumbent - lp_objective;

    Int count = rcTighten(reduced_costs, primals, gap, col_lower, col_upper, tightened_vars);

    if (count < 0)
        return false;

    // Record global changes and update global bounds. Only the entries this
    // call appended are ours -- the caller may pass a scratch vector that
    // already holds indices, and re-recording those would promote another
    // pass's node-local tightening to a global bound.
    const auto first_new = tightened_vars.size() - static_cast<std::size_t>(count);
    for (std::size_t i = first_new; i < tightened_vars.size(); ++i) {
        const Index j = tightened_vars[i];
        if (col_lower[j] > global_lower_[j] + kBoundTol ||
            col_upper[j] < global_upper_[j] - kBoundTol) {
            global_changes_.push_back({
                .variable = j,
                .old_lower = global_lower_[j],
                .old_upper = global_upper_[j],
                .new_lower = col_lower[j],
                .new_upper = col_upper[j],
                .is_global = true,
            });
            global_lower_[j] = std::max(global_lower_[j], col_lower[j]);
            global_upper_[j] = std::min(global_upper_[j], col_upper[j]);

            if (std::abs(global_lower_[j] - global_upper_[j]) <= kBoundTol) {
                ++stats_.root_global_fixings;
            } else {
                ++stats_.root_global_tightenings;
            }
        }
    }

    return true;
}

bool ReducedCostFixer::applyLocalFixing(std::span<const Real> reduced_costs,
                                        std::span<const Real> primals, Real node_objective,
                                        Real incumbent, std::vector<Real>& col_lower,
                                        std::vector<Real>& col_upper,
                                        std::vector<Index>& tightened_vars,
                                        RcFixingStats& delta) const {
    if (!loaded_)
        return true;
    if (incumbent >= kInf)
        return true;
    if (node_objective >= incumbent - kBoundTol)
        return true;

    const Real gap = incumbent - node_objective;

    Int count = rcTighten(reduced_costs, primals, gap, col_lower, col_upper, tightened_vars);

    if (count < 0)
        return false;

    // Count fixings vs tightenings among the newly appended entries. Node-local
    // tightening deliberately touches neither global_lower_/global_upper_ nor
    // global_changes_: it is valid only in this node's subtree.
    const auto start = tightened_vars.size() - static_cast<std::size_t>(count);
    for (std::size_t i = start; i < tightened_vars.size(); ++i) {
        Index j = tightened_vars[i];
        if (std::abs(col_lower[j] - col_upper[j]) <= kBoundTol) {
            ++delta.tree_local_fixings;
        } else {
            ++delta.tree_local_tightenings;
        }
    }

    return true;
}

bool ReducedCostFixer::applyLocalFixing(std::span<const Real> reduced_costs,
                                        std::span<const Real> primals, Real node_objective,
                                        Real incumbent, std::vector<Real>& col_lower,
                                        std::vector<Real>& col_upper,
                                        std::vector<Index>& tightened_vars) {
    RcFixingStats delta{};
    const bool ok = applyLocalFixing(reduced_costs, primals, node_objective, incumbent, col_lower,
                                     col_upper, tightened_vars, delta);
    stats_.tree_local_fixings += delta.tree_local_fixings;
    stats_.tree_local_tightenings += delta.tree_local_tightenings;
    stats_.propagation_triggers += delta.propagation_triggers;
    return ok;
}

bool ReducedCostFixer::enforceGlobalFixings(std::vector<Real>& col_lower,
                                            std::vector<Real>& col_upper,
                                            std::vector<Index>& tightened_vars) const {
    if (!loaded_)
        return true;

    for (const auto& change : global_changes_) {
        Index j = change.variable;
        bool changed = false;

        if (global_lower_[j] > col_lower[j] + kBoundTol) {
            col_lower[j] = global_lower_[j];
            changed = true;
        }
        if (global_upper_[j] < col_upper[j] - kBoundTol) {
            col_upper[j] = global_upper_[j];
            changed = true;
        }

        // Report the change before bailing out: the caller must be able to
        // undo every bound this call touched, including the one that crossed.
        if (changed) {
            tightened_vars.push_back(j);
        }

        if (col_lower[j] > col_upper[j] + kBoundTol) {
            return false;
        }
    }

    return true;
}

}  // namespace mipx
