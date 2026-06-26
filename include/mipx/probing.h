#pragma once

#include "mipx/core.h"
#include "mipx/implication_graph.h"
#include "mipx/variable_bounds.h"

#include <vector>

namespace mipx {

// Forward declarations.
struct LpProblem;
class DomainPropagator;

/// Statistics collected during probing.
struct ProbingStats {
    Int variables_probed = 0;
    Int fixings_found = 0;
    Int implications_found = 0;
    Int vubs_found = 0;
    Int vlbs_found = 0;
    Int infeasible_probes = 0;
    Int equivalences_found = 0;
    Int dominated_vars = 0;
    Int rounds = 0;
    double time_seconds = 0.0;
    bool infeasible = false;  // True if problem is proven infeasible.
};

/// Configuration for the probing engine.
struct ProbingConfig {
    Int max_rounds = 3;                // Maximum probing rounds.
    Int max_probes_per_round = 10000;  // Work budget per round.
    Int max_depth = 1;                 // Propagation depth (1 = direct probing).
    bool detect_equivalences = true;   // Find variable equivalences.
    bool detect_dominated = false;     // Find dominated variables (off: O(n^2), output unused).
    bool learn_vubs = true;            // Learn VUBs/VLBs from probing.
    bool learn_implications = true;    // Learn binary implications.
    double time_limit = 30.0;          // Time limit in seconds.
};

/// Root probing engine.
///
/// For each binary variable, fixes it to 0 and 1, propagates bounds,
/// and learns:
/// - Binary implications (x=v => y=w)
/// - Variable upper/lower bounds (x <= a*y + b)
/// - Variable fixings (x must be 0 or 1)
/// - Infeasibility of one branch (implies fixing)
/// - Variable equivalences
/// - Dominated variables
class ProbingEngine {
public:
    ProbingEngine() = default;

    /// Run probing on the given problem.
    /// Populates the implication graph and variable bound store.
    /// Returns the probing statistics.
    ProbingStats probe(const LpProblem& problem, ImplicationGraph& graph,
                       VariableBoundStore& vb_store, const ProbingConfig& config = {});

    /// Get variable fixings discovered during the last probe run.
    /// Each entry is (variable_index, fixed_value: 0.0 or 1.0).
    [[nodiscard]] const std::vector<std::pair<Index, Real>>& fixings() const { return fixings_; }

    /// Get variable equivalences discovered during the last probe run.
    [[nodiscard]] const std::vector<VariableEquivalence>& equivalences() const {
        return equivalences_;
    }

    /// Get dominated variable pairs. Each entry is (dominated, dominator):
    /// variable `dominated` is dominated by variable `dominator`.
    [[nodiscard]] const std::vector<std::pair<Index, Index>>& dominatedVars() const {
        return dominated_;
    }

private:
    /// Probe a single variable in both directions.
    /// Returns false if the variable gets fixed (infeasible in one direction).
    bool probeVariable(Index var, const LpProblem& problem, DomainPropagator& dp,
                       ImplicationGraph& graph, VariableBoundStore& vb_store, ProbingStats& stats,
                       const ProbingConfig& config);

    /// Learn implications from bound changes after fixing a binary variable.
    void learnImplications(Index fixed_var, bool fixed_val, const LpProblem& problem,
                           DomainPropagator& dp, ImplicationGraph& graph, ProbingStats& stats);

    /// Learn VUBs/VLBs from bound changes after fixing a binary variable.
    void learnVariableBounds(Index fixed_var, bool fixed_val, const LpProblem& problem,
                             DomainPropagator& dp, VariableBoundStore& vb_store,
                             const std::vector<Real>& base_lower,
                             const std::vector<Real>& base_upper, ProbingStats& stats);

    /// Detect dominated variables from implication structure.
    void detectDominated(const ImplicationGraph& graph, ProbingStats& stats);

    std::vector<std::pair<Index, Real>> fixings_;
    std::vector<VariableEquivalence> equivalences_;
    std::vector<std::pair<Index, Index>> dominated_;

    static constexpr Real kTol = 1e-8;
};

}  // namespace mipx
