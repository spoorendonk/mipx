#include "mipx/probing.h"

#include "mipx/domain.h"
#include "mipx/lp_problem.h"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace mipx {

ProbingStats ProbingEngine::probe(const LpProblem& problem, ImplicationGraph& graph,
                                  VariableBoundStore& vb_store, const ProbingConfig& config) {
    ProbingStats stats;
    fixings_.clear();
    equivalences_.clear();
    dominated_.clear();

    auto t0 = std::chrono::steady_clock::now();
    auto elapsed = [&]() -> double {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    };

    // Collect binary variables.
    std::vector<Index> binary_vars;
    binary_vars.reserve(static_cast<std::size_t>(problem.num_cols));
    for (Index j = 0; j < problem.num_cols; ++j) {
        if (problem.col_type[j] == VarType::Binary ||
            (problem.col_type[j] == VarType::Integer && problem.col_lower[j] >= -kTol &&
             problem.col_lower[j] <= kTol && problem.col_upper[j] >= 1.0 - kTol &&
             problem.col_upper[j] <= 1.0 + kTol)) {
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

    // Create domain propagator once and reuse across all probes.
    DomainPropagator dp;
    dp.load(problem);

    // Multiple rounds of probing to discover cascading implications.
    for (Int round = 0; round < config.max_rounds; ++round) {
        ++stats.rounds;
        Int fixings_before = stats.fixings_found;
        Int implications_before = stats.implications_found;
        Int probes_this_round = 0;

        // Build O(1) lookup for already-fixed variables.
        std::vector<bool> fixed_lookup(static_cast<std::size_t>(problem.num_cols), false);
        for (const auto& [fvar, fval] : fixings_) {
            fixed_lookup[fvar] = true;
        }

        for (Index var : binary_vars) {
            if (elapsed() > config.time_limit) {
                break;
            }
            if (probes_this_round >= config.max_probes_per_round) {
                break;
            }

            // Skip already fixed variables.
            if (problem.col_lower[var] > 1.0 - kTol || problem.col_upper[var] < kTol) {
                continue;
            }

            // Check if fixed by earlier probing in this round (O(1) lookup).
            if (fixed_lookup[var]) {
                continue;
            }

            // Reset propagator to base state and apply known fixings.
            dp.pushCheckpoint();
            for (const auto& [fvar, fval] : fixings_) {
                dp.setBound(fvar, fval, fval);
            }

            if (!probeVariable(var, problem, dp, graph, vb_store, stats, config)) {
                // Variable was fixed (or problem proven infeasible).
                if (stats.infeasible) {
                    dp.popCheckpoint();
                    stats.time_seconds = elapsed();
                    return stats;
                }
                // Mark newly fixed variable in the lookup.
                if (!fixings_.empty()) {
                    fixed_lookup[fixings_.back().first] = true;
                }
            }
            dp.popCheckpoint();
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
    // Build O(1) lookup for known fixings.
    std::vector<bool> known_fixing(static_cast<std::size_t>(problem.num_cols), false);
    for (const auto& [fvar, fval] : fixings_) {
        known_fixing[fvar] = true;
    }
    auto graph_fixings = graph.detectFixings();
    for (const auto& [var, val] : graph_fixings) {
        if (!known_fixing[var]) {
            fixings_.push_back({var, val ? 1.0 : 0.0});
            ++stats.fixings_found;
            known_fixing[var] = true;
        }
    }

    stats.time_seconds = elapsed();
    return stats;
}

bool ProbingEngine::probeVariable(Index var, const LpProblem& problem, DomainPropagator& dp,
                                  ImplicationGraph& graph, VariableBoundStore& vb_store,
                                  ProbingStats& stats, const ProbingConfig& config) {
    ++stats.variables_probed;

    // Save base bounds for VUB/VLB learning.
    std::vector<Real> base_lower(static_cast<std::size_t>(problem.num_cols));
    std::vector<Real> base_upper(static_cast<std::size_t>(problem.num_cols));
    for (Index j = 0; j < problem.num_cols; ++j) {
        base_lower[j] = dp.getLower(j);
        base_upper[j] = dp.getUpper(j);
    }

    // Probe var=0. Propagate to fixpoint: probing depends on full propagation
    // to discover cascading implications and fixings.
    dp.pushCheckpoint();
    dp.setBound(var, 0.0, 0.0);
    bool feasible_at_0 = dp.propagate();

    if (feasible_at_0) {
        if (config.learn_implications) {
            learnImplications(var, false, problem, dp, graph, stats);
        }
        if (config.learn_vubs) {
            learnVariableBounds(var, false, problem, dp, vb_store, base_lower, base_upper, stats);
        }
    } else {
        ++stats.infeasible_probes;
    }
    dp.popCheckpoint();

    // Probe var=1. Propagate to fixpoint (see var=0 above).
    dp.pushCheckpoint();
    dp.setBound(var, 1.0, 1.0);
    bool feasible_at_1 = dp.propagate();

    if (feasible_at_1) {
        if (config.learn_implications) {
            learnImplications(var, true, problem, dp, graph, stats);
        }
        if (config.learn_vubs) {
            learnVariableBounds(var, true, problem, dp, vb_store, base_lower, base_upper, stats);
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
        // Both directions infeasible - problem is proven infeasible.
        stats.infeasible = true;
        return false;
    }

    return true;
}

void ProbingEngine::learnImplications(Index fixed_var, bool fixed_val, const LpProblem& problem,
                                      DomainPropagator& dp, ImplicationGraph& graph,
                                      ProbingStats& stats) {
    for (Index j = 0; j < problem.num_cols; ++j) {
        if (j == fixed_var) {
            continue;
        }
        if (!graph.isBinary(j)) {
            continue;
        }

        Real lb = dp.getLower(j);
        Real ub = dp.getUpper(j);

        // If binary variable j is fixed to 1 after propagation.
        if (lb > 1.0 - kTol) {
            Int before = graph.numImplications();
            graph.addImplication(fixed_var, fixed_val, j, true);
            Int after = graph.numImplications();
            stats.implications_found += (after - before);
        }
        // If binary variable j is fixed to 0 after propagation.
        if (ub < kTol) {
            Int before = graph.numImplications();
            graph.addImplication(fixed_var, fixed_val, j, false);
            Int after = graph.numImplications();
            stats.implications_found += (after - before);
        }
    }
}

void ProbingEngine::learnVariableBounds(Index fixed_var, bool fixed_val, const LpProblem& problem,
                                        DomainPropagator& dp, VariableBoundStore& vb_store,
                                        const std::vector<Real>& base_lower,
                                        const std::vector<Real>& base_upper, ProbingStats& stats) {
    for (Index j = 0; j < problem.num_cols; ++j) {
        if (j == fixed_var) {
            continue;
        }
        // VUBs/VLBs are primarily useful for non-binary variables.
        if (problem.col_type[j] == VarType::Binary) {
            continue;
        }

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

void ProbingEngine::detectDominated(const ImplicationGraph& graph, ProbingStats& stats) {
    dominated_.clear();

    const Int num_binaries = graph.numBinaryVars();
    if (num_binaries == 0) {
        return;
    }

    // Guard: skip domination detection if too many binaries to avoid O(n^2).
    constexpr Int kMaxBinariesForDomination = 1000;
    if (num_binaries > kMaxBinariesForDomination) {
        stats.dominated_vars = 0;
        return;
    }

    // Variable x dominates variable y if implications(x=1) supseteq
    // implications(y=1). We check this using sorted implication lists
    // and a merge-based O(k) subset check.

    const auto& bin_vars = graph.binaryVars();

    // Build sorted implication targets for val=1 for each binary variable.
    // Encode each target as (to_var * 2 + to_val) for sortability.
    struct ImpsEntry {
        Index var;
        std::vector<Int> targets;
    };
    std::vector<ImpsEntry> entries;
    entries.reserve(bin_vars.size());

    for (Index var : bin_vars) {
        const auto& imps = graph.implications(var, true);
        ImpsEntry entry;
        entry.var = var;
        entry.targets.reserve(imps.size());
        for (const auto& imp : imps) {
            entry.targets.push_back(imp.to_var * 2 + (imp.to_val ? 1 : 0));
        }
        std::sort(entry.targets.begin(), entry.targets.end());
        entries.push_back(std::move(entry));
    }

    // Merge-based O(k) subset check on sorted vectors.
    auto isSubset = [](const std::vector<Int>& small, const std::vector<Int>& large) -> bool {
        if (small.size() > large.size()) {
            return false;
        }
        std::size_t si = 0;
        std::size_t li = 0;
        while (si < small.size() && li < large.size()) {
            if (small[si] == large[li]) {
                ++si;
                ++li;
            } else if (large[li] < small[si]) {
                ++li;
            } else {
                return false;
            }
        }
        return si == small.size();
    };

    for (std::size_t i = 0; i < entries.size(); ++i) {
        for (std::size_t j = i + 1; j < entries.size(); ++j) {
            // Check if entries[i] dominates entries[j]:
            // implications(i=1) supseteq implications(j=1)
            if (entries[i].targets.size() >= entries[j].targets.size() &&
                !entries[j].targets.empty() && isSubset(entries[j].targets, entries[i].targets)) {
                dominated_.push_back({entries[j].var, entries[i].var});
            }
            // Check reverse: entries[j] dominates entries[i].
            else if (entries[j].targets.size() >= entries[i].targets.size() &&
                     !entries[i].targets.empty() &&
                     isSubset(entries[i].targets, entries[j].targets)) {
                dominated_.push_back({entries[i].var, entries[j].var});
            }
        }
    }

    stats.dominated_vars = static_cast<Int>(dominated_.size());
}

}  // namespace mipx
