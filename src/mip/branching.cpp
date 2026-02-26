#include "mipx/branching.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace mipx {

Index MostFractionalBranching::select(
    std::span<const Real> primal_values,
    std::span<const VarType> var_types,
    std::span<const Real> col_lower,
    std::span<const Real> col_upper) const {

    Index best = -1;
    Real best_score = -1.0;  // Best = closest to 0.5.

    for (Index j = 0; j < static_cast<Index>(var_types.size()); ++j) {
        if (var_types[j] == VarType::Continuous) continue;
        // Skip fixed variables.
        if (col_lower[j] == col_upper[j]) continue;

        Real val = primal_values[j];
        if (isIntegral(val)) continue;

        Real frac = fractionality(val);
        // Score: closer to 0.5 is better. Use 0.5 - |frac - 0.5|.
        Real score = 0.5 - std::abs(frac - 0.5);
        if (score > best_score) {
            best_score = score;
            best = j;
        }
    }
    return best;
}

Index FirstFractionalBranching::select(
    std::span<const Real> primal_values,
    std::span<const VarType> var_types,
    std::span<const Real> col_lower,
    std::span<const Real> col_upper) const {

    for (Index j = 0; j < static_cast<Index>(var_types.size()); ++j) {
        if (var_types[j] == VarType::Continuous) continue;
        if (col_lower[j] == col_upper[j]) continue;

        if (!isIntegral(primal_values[j])) {
            return j;
        }
    }
    return -1;
}

void ReliabilityBranching::reset(Index num_cols) {
    pseudocosts_.assign(static_cast<std::size_t>(std::max<Index>(0, num_cols)),
                        PseudoCost{});
}

bool ReliabilityBranching::isReliable(Index var) const {
    if (!inRange(var)) return false;
    const auto& pc = pseudocosts_[var];
    return pc.up_count >= reliability_threshold_ &&
           pc.down_count >= reliability_threshold_;
}

Real ReliabilityBranching::upPseudoCost(Index var) const {
    if (!inRange(var)) return pseudocost_fallback_;
    const auto& pc = pseudocosts_[var];
    if (pc.up_count <= 0) return pseudocost_fallback_;
    return std::max<Real>(pc.up, 1e-8);
}

Real ReliabilityBranching::downPseudoCost(Index var) const {
    if (!inRange(var)) return pseudocost_fallback_;
    const auto& pc = pseudocosts_[var];
    if (pc.down_count <= 0) return pseudocost_fallback_;
    return std::max<Real>(pc.down, 1e-8);
}

Int ReliabilityBranching::upReliability(Index var) const {
    if (!inRange(var)) return 0;
    return pseudocosts_[var].up_count;
}

Int ReliabilityBranching::downReliability(Index var) const {
    if (!inRange(var)) return 0;
    return pseudocosts_[var].down_count;
}

void ReliabilityBranching::updatePseudoCost(Index var, bool up_direction, Real gain_per_unit) {
    if (!inRange(var) || !std::isfinite(gain_per_unit)) return;
    gain_per_unit = std::max<Real>(gain_per_unit, 0.0);

    auto& pc = pseudocosts_[var];
    if (up_direction) {
        const Real weighted = pc.up * static_cast<Real>(pc.up_count);
        ++pc.up_count;
        pc.up = (weighted + gain_per_unit) / static_cast<Real>(pc.up_count);
    } else {
        const Real weighted = pc.down * static_cast<Real>(pc.down_count);
        ++pc.down_count;
        pc.down = (weighted + gain_per_unit) / static_cast<Real>(pc.down_count);
    }
}

Real ReliabilityBranching::safeUpCost(Index var) const {
    return upPseudoCost(var);
}

Real ReliabilityBranching::safeDownCost(Index var) const {
    return downPseudoCost(var);
}

Real ReliabilityBranching::blendScore(Real frac, Real pseudo_score) {
    const Real frac_term = 0.5 - std::abs(frac - 0.5);
    const Real pseudo_term = std::log1p(std::max<Real>(0.0, pseudo_score));
    return frac_term + 0.15 * pseudo_term;
}

BranchingSelection ReliabilityBranching::select(DualSimplexSolver& lp,
                                                const LpProblem& problem,
                                                std::span<const Real> primal_values,
                                                std::span<const Real> col_lower,
                                                std::span<const Real> col_upper,
                                                Real node_objective,
                                                bool force_strong_branch,
                                                BranchingTelemetry& telemetry) {
    BranchingSelection selection;
    ++telemetry.selections;

    std::vector<Candidate> candidates;
    candidates.reserve(static_cast<std::size_t>(problem.num_cols));

    for (Index j = 0; j < problem.num_cols; ++j) {
        if (problem.col_type[j] == VarType::Continuous) continue;
        if (col_lower[j] >= col_upper[j] - 1e-12) continue;

        const Real value = primal_values[j];
        if (isIntegral(value)) continue;

        const Real floor_v = std::floor(value);
        const Real ceil_v = std::ceil(value);
        const Real down_dist = std::max<Real>(0.0, value - floor_v);
        const Real up_dist = std::max<Real>(0.0, ceil_v - value);
        if (down_dist <= 1e-8 || up_dist <= 1e-8) continue;

        Candidate c;
        c.var = j;
        c.value = value;
        c.frac = fractionality(value);
        c.down_dist = down_dist;
        c.up_dist = up_dist;
        c.reliable = isReliable(j);
        c.pseudo_score = std::max(down_dist * safeDownCost(j),
                                  up_dist * safeUpCost(j));
        c.prefilter_score = blendScore(c.frac, c.pseudo_score);
        candidates.push_back(c);
    }

    if (candidates.empty()) return selection;

    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) {
                  if (a.prefilter_score != b.prefilter_score) {
                      return a.prefilter_score > b.prefilter_score;
                  }
                  return a.var < b.var;
              });

    const Index max_candidates = std::min<Index>(
        static_cast<Index>(candidates.size()), strong_branch_max_candidates_);
    candidates.resize(static_cast<std::size_t>(max_candidates));

    Index best_reliable = -1;
    Real best_reliable_score = -std::numeric_limits<Real>::infinity();
    for (const auto& c : candidates) {
        if (!c.reliable) continue;
        if (c.pseudo_score > best_reliable_score) {
            best_reliable_score = c.pseudo_score;
            best_reliable = c.var;
        }
    }

    const bool run_strong_branch = force_strong_branch || best_reliable < 0;
    Index best_strong = -1;
    Real best_strong_score = -std::numeric_limits<Real>::infinity();

    if (run_strong_branch) {
        ++telemetry.strong_branch_calls;

        const auto saved_basis = lp.getBasis();
        const Int old_limit = lp.getIterationLimit();
        lp.setIterationLimit(strong_branch_iter_limit_);

        Int probe_budget = strong_branch_probe_budget_;
        for (const auto& c : candidates) {
            if (probe_budget < 2) break;
            if (!force_strong_branch && c.reliable) continue;

            const Real orig_lb = col_lower[c.var];
            const Real orig_ub = col_upper[c.var];
            const Real down_ub = std::floor(c.value);
            const Real up_lb = std::ceil(c.value);

            Real down_gain = 0.0;
            Real up_gain = 0.0;

            if (down_ub >= orig_lb - 1e-12) {
                lp.setBasis(saved_basis);
                lp.setColBounds(c.var, orig_lb, down_ub);
                const auto down = lp.solve();
                ++telemetry.strong_branch_probes;
                telemetry.strong_branch_probe_iters += down.iterations;
                telemetry.strong_branch_probe_work_units += down.work_units;
                lp.setColBounds(c.var, orig_lb, orig_ub);

                if (down.status == Status::Infeasible) {
                    down_gain = kStrongInfeasibleGain;
                } else if (down.status == Status::Optimal) {
                    down_gain = std::max<Real>(0.0, down.objective - node_objective);
                } else {
                    down_gain = c.down_dist * safeDownCost(c.var);
                }
            } else {
                down_gain = kStrongInfeasibleGain;
            }
            if (c.down_dist > 1e-8) {
                updatePseudoCost(c.var, false, down_gain / c.down_dist);
            }
            --probe_budget;

            if (up_lb <= orig_ub + 1e-12) {
                lp.setBasis(saved_basis);
                lp.setColBounds(c.var, up_lb, orig_ub);
                const auto up = lp.solve();
                ++telemetry.strong_branch_probes;
                telemetry.strong_branch_probe_iters += up.iterations;
                telemetry.strong_branch_probe_work_units += up.work_units;
                lp.setColBounds(c.var, orig_lb, orig_ub);

                if (up.status == Status::Infeasible) {
                    up_gain = kStrongInfeasibleGain;
                } else if (up.status == Status::Optimal) {
                    up_gain = std::max<Real>(0.0, up.objective - node_objective);
                } else {
                    up_gain = c.up_dist * safeUpCost(c.var);
                }
            } else {
                up_gain = kStrongInfeasibleGain;
            }
            if (c.up_dist > 1e-8) {
                updatePseudoCost(c.var, true, up_gain / c.up_dist);
            }
            --probe_budget;

            const Real strong_score = std::max(down_gain, up_gain);
            if (strong_score > best_strong_score) {
                best_strong_score = strong_score;
                best_strong = c.var;
            }
        }

        lp.setIterationLimit(old_limit);
        lp.setBasis(saved_basis);
    }

    if (best_strong >= 0) {
        selection.variable = best_strong;
        return selection;
    }

    if (best_reliable >= 0) {
        selection.variable = best_reliable;
        selection.used_pseudocost = true;
        selection.used_reliable_pseudocost = true;
        ++telemetry.pseudocost_uses;
        ++telemetry.pseudocost_hits;
        return selection;
    }

    const auto best_it = std::max_element(
        candidates.begin(), candidates.end(),
        [](const Candidate& a, const Candidate& b) {
            if (a.pseudo_score != b.pseudo_score) {
                return a.pseudo_score < b.pseudo_score;
            }
            return a.var > b.var;
        });

    selection.variable = (best_it == candidates.end()) ? -1 : best_it->var;
    if (selection.variable >= 0) {
        selection.used_pseudocost = true;
        ++telemetry.pseudocost_uses;
    }
    return selection;
}

std::pair<BnbNode, BnbNode> createChildren(const BnbNode& parent,
                                            Index branch_var,
                                            Real branch_val) {
    Real floor_val = std::floor(branch_val);
    Real ceil_val = std::ceil(branch_val);

    BnbNode left;
    left.parent_id = parent.id;
    left.depth = parent.depth + 1;
    left.basis = parent.basis;
    left.estimate = parent.estimate;
    left.basis_rows = parent.basis_rows;
    left.branch = {branch_var, floor_val, true};  // x_j <= floor(v)
    left.bound_changes = parent.bound_changes;
    left.bound_changes.push_back(left.branch);
    left.local_cuts = parent.local_cuts;

    BnbNode right;
    right.parent_id = parent.id;
    right.depth = parent.depth + 1;
    right.basis = parent.basis;
    right.estimate = parent.estimate;
    right.basis_rows = parent.basis_rows;
    right.branch = {branch_var, ceil_val, false};  // x_j >= ceil(v)
    right.bound_changes = parent.bound_changes;
    right.bound_changes.push_back(right.branch);
    right.local_cuts = parent.local_cuts;

    return {std::move(left), std::move(right)};
}

}  // namespace mipx
