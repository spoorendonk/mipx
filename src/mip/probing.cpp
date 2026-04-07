#include "mipx/probing.h"

#include <algorithm>
#include <chrono>
#include <cmath>

#include "mipx/domain.h"
#include "mipx/lp_problem.h"

namespace mipx {

ProbingStats ProbingEngine::probe(const LpProblem& problem,
                                  ImplicationGraph& graph,
                                  VariableBoundStore& vb_store,
                                  const ProbingConfig& config) {
    ProbingStats stats;
    fixings_.clear();
    equivalences_.clear();
    dominated_.clear();

    auto t0 = std::chrono::steady_clock::now();
    auto elapsed = [&]() -> double {
        return std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
    };

    // Collect binary variables.
    std::vector<Index> binary_vars;
    binary_vars.reserve(static_cast<std::size_t>(problem.num_cols));
    for (Index j = 0; j < problem.num_cols; ++j) {
        if (problem.col_type[j] == VarType::Binary ||
            (problem.col_type[j] == VarType::Integer &&
             problem.col_lower[j] >= -kTol && problem.col_lower[j] <= kTol &&
             problem.col_upper[j] >= 1.0 - kTol && problem.col_upper[j] <= 1.0 + kTol)) {
            // Only probe variables that aren't already fixed.
            if (problem.col_upper[j] - problem.col_lower[j] > kTol) {
                binary_vars.push_back(j);
            }
        }
    }

    if (binary_vars.empty()) {
        stats.time_seconds = elapsed();
        return stats;
    }

    // Initialize graph and VB store if not already done.
    graph.init(problem.num_cols, binary_vars);
    vb_store.init(problem.num_cols);

    // Multiple rounds of probing to discover cascading implications.
    for (Int round = 0; round < config.max_rounds; ++round) {
        ++stats.rounds;
        Int fixings_before = stats.fixings_found;
        Int implications_before = stats.implications_found;
        Int probes_this_round = 0;

        for (Index var : binary_vars) {
            if (elapsed() > config.time_limit) break;
            if (probes_this_round >= config.max_probes_per_round) break;

            // Skip already fixed variables.
            if (problem.col_lower[var] > 1.0 - kTol ||
                problem.col_upper[var] < kTol) continue;

            // Check if fixed by earlier probing in this round.
            bool already_fixed = false;
            for (const auto& [fvar, fval] : fixings_) {
                if (fvar == var) {
                    already_fixed = true;
                    break;
                }
            }
            if (already_fixed) continue;

            DomainPropagator dp;
            dp.load(problem);

            // Apply known fixings.
            for (const auto& [fvar, fval] : fixings_) {
                dp.setBound(fvar, fval, fval);
            }

            if (!probeVariable(var, problem, dp, graph, vb_store, stats, config)) {
                // Variable was fixed. Will be picked up in next round.
            }
            ++probes_this_round;
        }

        // Check if we learned anything new.
        if (stats.fixings_found == fixings_before &&
            stats.implications_found == implications_before) {
            break;  // No progress, stop early.
        }
    }

    // Post-probing: detect equivalences and dominated variables.
    if (config.detect_equivalences) {
        equivalences_ = graph.detectEquivalences();
        stats.equivalences_found = static_cast<Int>(equivalences_.size());
    }

    if (config.detect_dominated) {
        detectDominated(graph, stats);
    }

    // Also detect fixings from contradiction analysis in the graph.
    auto graph_fixings = graph.detectFixings();
    for (const auto& [var, val] : graph_fixings) {
        bool already_known = false;
        for (const auto& [fvar, fval] : fixings_) {
            if (fvar == var) {
                already_known = true;
                break;
            }
        }
        if (!already_known) {
            fixings_.push_back({var, val ? 1.0 : 0.0});
            ++stats.fixings_found;
        }
    }

    stats.time_seconds = elapsed();
    return stats;
}

bool ProbingEngine::probeVariable(Index var,
                                  const LpProblem& problem,
                                  DomainPropagator& dp,
                                  ImplicationGraph& graph,
                                  VariableBoundStore& vb_store,
                                  ProbingStats& stats,
                                  const ProbingConfig& config) {
    ++stats.variables_probed;

    // Save base bounds for VUB/VLB learning.
    std::vector<Real> base_lower(static_cast<std::size_t>(problem.num_cols));
    std::vector<Real> base_upper(static_cast<std::size_t>(problem.num_cols));
    for (Index j = 0; j < problem.num_cols; ++j) {
        base_lower[j] = dp.getLower(j);
        base_upper[j] = dp.getUpper(j);
    }

    // Probe var=0.
    dp.pushCheckpoint();
    dp.setBound(var, 0.0, 0.0);
    bool feasible_at_0 = dp.propagate();

    if (feasible_at_0) {
        if (config.learn_implications) {
            learnImplications(var, false, problem, dp, graph, stats);
        }
        if (config.learn_vubs) {
            learnVariableBounds(var, false, problem, dp, vb_store,
                                base_lower, base_upper, stats);
        }
    } else {
        ++stats.infeasible_probes;
    }
    dp.popCheckpoint();

    // Probe var=1.
    dp.pushCheckpoint();
    dp.setBound(var, 1.0, 1.0);
    bool feasible_at_1 = dp.propagate();

    if (feasible_at_1) {
        if (config.learn_implications) {
            learnImplications(var, true, problem, dp, graph, stats);
        }
        if (config.learn_vubs) {
            learnVariableBounds(var, true, problem, dp, vb_store,
                                base_lower, base_upper, stats);
        }
    } else {
        ++stats.infeasible_probes;
    }
    dp.popCheckpoint();

    // If one direction is infeasible, the variable is fixed.
    if (!feasible_at_0 && feasible_at_1) {
        fixings_.push_back({var, 1.0});
        ++stats.fixings_found;
        return false;
    }
    if (feasible_at_0 && !feasible_at_1) {
        fixings_.push_back({var, 0.0});
        ++stats.fixings_found;
        return false;
    }
    if (!feasible_at_0 && !feasible_at_1) {
        // Both directions infeasible - problem is infeasible.
        // Record as fixing to 0 (the caller should check propagation).
        fixings_.push_back({var, 0.0});
        ++stats.fixings_found;
        return false;
    }

    return true;
}

void ProbingEngine::learnImplications(Index fixed_var, bool fixed_val,
                                      const LpProblem& problem,
                                      DomainPropagator& dp,
                                      ImplicationGraph& graph,
                                      ProbingStats& stats) {
    for (Index j = 0; j < problem.num_cols; ++j) {
        if (j == fixed_var) continue;
        if (!graph.isBinary(j)) continue;

        Real lb = dp.getLower(j);
        Real ub = dp.getUpper(j);

        // If binary variable j is fixed to 1 after propagation.
        if (lb > 1.0 - kTol) {
            if (graph.addImplication(fixed_var, fixed_val, j, true)) {
                ++stats.implications_found;
            }
        }
        // If binary variable j is fixed to 0 after propagation.
        if (ub < kTol) {
            if (graph.addImplication(fixed_var, fixed_val, j, false)) {
                ++stats.implications_found;
            }
        }
    }
}

void ProbingEngine::learnVariableBounds(Index fixed_var, bool fixed_val,
                                        const LpProblem& problem,
                                        DomainPropagator& dp,
                                        VariableBoundStore& vb_store,
                                        const std::vector<Real>& base_lower,
                                        const std::vector<Real>& base_upper,
                                        ProbingStats& stats) {
    for (Index j = 0; j < problem.num_cols; ++j) {
        if (j == fixed_var) continue;
        // VUBs/VLBs are primarily useful for non-binary variables.
        if (problem.col_type[j] == VarType::Binary) continue;

        Real new_lb = dp.getLower(j);
        Real new_ub = dp.getUpper(j);
        Real old_lb = base_lower[j];
        Real old_ub = base_upper[j];

        // If upper bound tightened, we learned a VUB.
        // x_j <= new_ub when y = fixed_val.
        // VUB form: x_j <= a * y + b
        // If fixed_val = 0: b = new_ub, and a + b = old_ub => a = old_ub - new_ub
        // If fixed_val = 1: a + b = new_ub, and b = old_ub => a = new_ub - old_ub
        if (new_ub < old_ub - kTol && std::isfinite(new_ub) && std::isfinite(old_ub)) {
            Real a, b;
            if (!fixed_val) {
                b = new_ub;
                a = old_ub - new_ub;
            } else {
                b = old_ub;
                a = new_ub - old_ub;
            }
            vb_store.addVUB(j, fixed_var, a, b);
            ++stats.vubs_found;
        }

        // If lower bound tightened, we learned a VLB.
        // x_j >= new_lb when y = fixed_val.
        // VLB form: x_j >= a * y + b
        // If fixed_val = 0: b = new_lb, and a + b = old_lb => a = old_lb - new_lb
        // If fixed_val = 1: a + b = new_lb, and b = old_lb => a = new_lb - old_lb
        if (new_lb > old_lb + kTol && std::isfinite(new_lb) && std::isfinite(old_lb)) {
            Real a, b;
            if (!fixed_val) {
                b = new_lb;
                a = old_lb - new_lb;
            } else {
                b = old_lb;
                a = new_lb - old_lb;
            }
            vb_store.addVLB(j, fixed_var, a, b);
            ++stats.vlbs_found;
        }
    }
}

void ProbingEngine::detectDominated(const ImplicationGraph& graph,
                                    ProbingStats& stats) {
    dominated_.clear();

    // Variable x dominates variable y if:
    // x=1 => y=1 (or x=0 => y=0) and y has a subset of x's implications.
    // We use a simpler check: x dominates y if fixing x gives strictly more
    // implications than fixing y in the same direction.
    const auto& binary_vars = graph.numBinaryVars();
    if (binary_vars == 0) return;

    // For each pair of binary variables, check if one's implications
    // are a superset of the other's.
    // This is O(n^2) in number of binary vars, so we limit it.
    constexpr Int kMaxDominationChecks = 5000;
    Int checks = 0;

    // Use implication scores as a quick filter: a variable with fewer
    // implications cannot dominate one with more.
    // For now, just detect simple domination patterns.
    // A variable x dominates y if x=1 => y=1 and all implications of y=1
    // are also implications of x=1.
    // We skip the full check and just record when x=1 => y=1 as a signal.
    // This is used heuristically by the branching rule.

    // Placeholder: actual domination detection is expensive.
    // We record basic domination signals from direct implications.
    stats.dominated_vars = static_cast<Int>(dominated_.size());
    (void)checks;
    (void)kMaxDominationChecks;
}

}  // namespace mipx
