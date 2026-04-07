#pragma once

#include <span>
#include <vector>

#include "mipx/core.h"
#include "mipx/lp_problem.h"
#include "mipx/sparse_matrix.h"

namespace mipx {

/// A literal represents a binary variable at a specific value (0 or 1).
/// For binary variable j: literal 2*j represents x_j = 1, literal 2*j+1
/// represents x_j = 0 (i.e., complement).
struct Literal {
    Index var;
    bool complemented;  // true means (1 - x_j), false means x_j.

    [[nodiscard]] Index id() const { return 2 * var + (complemented ? 1 : 0); }
    [[nodiscard]] Literal complement() const { return {var, !complemented}; }
};

/// Explicit conflict graph for binary variables.
///
/// Two literals conflict if they cannot both be 1 simultaneously.
/// Equivalently, a conflict between x_i and x_j means x_i + x_j <= 1.
/// The graph also handles complemented literals: a conflict between x_i
/// and (1 - x_j) means x_i + (1 - x_j) <= 1, i.e., x_i <= x_j.
class ConflictGraph {
public:
    ConflictGraph() = default;

    /// Build the conflict graph from the constraint matrix.
    /// Detects conflicts from:
    ///  - Set-packing/partitioning rows (all binary, all coeffs ~1, rhs ~1)
    ///  - Knapsack analysis (pairs of binaries whose coefficients exceed rhs)
    void build(const LpProblem& problem);

    /// Add a conflict edge between two literals.
    void addConflict(Literal a, Literal b);

    /// Number of binary variables tracked.
    [[nodiscard]] Index numBinaries() const { return num_binaries_; }

    /// Number of conflict edges (undirected).
    [[nodiscard]] Index numEdges() const { return num_edges_; }

    /// Return all literals conflicting with the given literal.
    [[nodiscard]] std::span<const Literal> neighbors(Literal lit) const;

    /// Check if two literals conflict.
    [[nodiscard]] bool conflicts(Literal a, Literal b) const;

    /// Mapping from original variable index to internal binary index.
    /// Returns -1 if the variable is not binary.
    [[nodiscard]] Index toBinaryIndex(Index col) const;

    /// Mapping from internal binary index to original variable index.
    [[nodiscard]] Index toOriginalIndex(Index bin_idx) const;

    /// Find connected components. Returns component id per binary variable.
    [[nodiscard]] std::vector<Index> connectedComponents(
        Index& num_components) const;

    /// Fix a variable during tree search, updating adjacency info.
    /// This does not structurally remove nodes; it marks them as fixed.
    void fixVariable(Index col, Real value);

    /// Unfix a variable (backtrack).
    void unfixVariable(Index col);

    /// Check if a variable is fixed.
    [[nodiscard]] bool isFixed(Index col) const;

    /// Get fixed value. Undefined if not fixed.
    [[nodiscard]] Real fixedValue(Index col) const;

private:
    /// Number of binary variables.
    Index num_binaries_ = 0;

    /// Total number of literals = 2 * num_binaries_.
    [[nodiscard]] Index numLiterals() const { return 2 * num_binaries_; }

    /// Adjacency lists indexed by literal id.
    std::vector<std::vector<Literal>> adj_;

    /// Mapping: original col -> binary index (-1 if not binary).
    std::vector<Index> col_to_bin_;

    /// Mapping: binary index -> original col.
    std::vector<Index> bin_to_col_;

    /// Fixed status.
    std::vector<bool> fixed_;
    std::vector<Real> fixed_value_;

    Index num_edges_ = 0;
};

}  // namespace mipx
