#pragma once

#include <vector>

#include "mipx/core.h"

namespace mipx {

/// Represents a binary implication: fixing variable `from_var` to `from_val`
/// implies that variable `to_var` must take value `to_val`.
/// Only applies to binary variables (from_val and to_val are 0 or 1).
struct BinaryImplication {
    Index to_var = -1;
    bool to_val = false;
};

/// Variable equivalence detected via implication chains.
/// Represents: var_a == var_b (if same_sense) or var_a == 1 - var_b (if !same_sense).
struct VariableEquivalence {
    Index var_a = -1;
    Index var_b = -1;
    bool same_sense = true;  // true: a=b, false: a=1-b
};

/// Persistent implication graph for binary variables.
///
/// Stores binary implications of the form:
///   "x_i = v  =>  x_j = w"  where v, w in {0, 1}
///
/// Uses forward adjacency lists indexed by (variable, value) pairs.
/// Supports transitive closure, variable equivalence detection, and
/// integration with domain propagation.
class ImplicationGraph {
public:
    ImplicationGraph() = default;

    /// Initialize for a problem with the given number of binary variables.
    /// `binary_vars` contains the indices of binary variables in the original problem.
    void init(Index num_cols, const std::vector<Index>& binary_vars);

    /// Reset all implications (keeps the variable set).
    void clear();

    /// Add an implication: fixing `from_var` to `from_val` implies `to_var` = `to_val`.
    /// Returns false if a contradiction is detected (e.g., x=0 => x=1).
    bool addImplication(Index from_var, bool from_val, Index to_var, bool to_val);

    /// Get all implications triggered by fixing `var` to `val`.
    [[nodiscard]] const std::vector<BinaryImplication>& implications(
        Index var, bool val) const;

    /// Number of implications stored.
    [[nodiscard]] Int numImplications() const { return num_implications_; }

    /// Number of binary variables tracked.
    [[nodiscard]] Int numBinaryVars() const {
        return static_cast<Int>(binary_vars_.size());
    }

    /// Check if a variable is tracked as binary.
    [[nodiscard]] bool isBinary(Index var) const;

    /// Compute transitive closure of current implications.
    /// Returns number of new implications added.
    Int computeTransitiveClosure();

    /// Detect variable equivalences from implication chains.
    /// x=0 => y=1 AND y=1 => x=0 means x and 1-y are equivalent.
    [[nodiscard]] std::vector<VariableEquivalence> detectEquivalences() const;

    /// Detect variables that are fixed by the implication graph.
    /// If x=0 leads to contradiction, x must be 1 (and vice versa).
    /// Returns pairs of (variable, fixed_value).
    [[nodiscard]] std::vector<std::pair<Index, bool>> detectFixings() const;

    /// Get the implication score for a variable (number of implications it triggers).
    /// Useful for branching prioritization.
    [[nodiscard]] Int implicationScore(Index var) const;

    /// Propagate implications given a variable fixing.
    /// Fills `propagated` with all implied fixings.
    /// Returns false if a contradiction is found.
    bool propagate(Index var, bool val,
                   std::vector<std::pair<Index, bool>>& propagated) const;

private:
    /// Adjacency list node index: 2*internal_idx + val gives the slot.
    [[nodiscard]] Index slot(Index var, bool val) const;

    /// Check if a variable has an internal index.
    [[nodiscard]] bool hasIndex(Index var) const;

    Index num_cols_ = 0;
    std::vector<Index> binary_vars_;       // Original variable indices.
    std::vector<Index> var_to_index_;      // Original var -> internal index (-1 if not binary).
    // Adjacency lists: adj_[2*i] = implications when var i is fixed to 0,
    //                  adj_[2*i+1] = implications when var i is fixed to 1.
    std::vector<std::vector<BinaryImplication>> adj_;
    Int num_implications_ = 0;

    static const std::vector<BinaryImplication> kEmpty;
};

}  // namespace mipx
