#pragma once

#include <string>
#include <vector>

#include "mipx/automorphism.h"
#include "mipx/core.h"

namespace mipx {

// Forward declarations.
struct LpProblem;

/// Types of symmetry-breaking constraints.
enum class SymbreakType {
    Lexicographic,  ///< Lex-leader: x_{sigma(j)} <= x_j for first moved var
    Symresack,      ///< Full symresack inequality for a permutation
    Orbitope,       ///< Orbitope constraint for a set of symmetric columns
};

/// A symmetry-breaking constraint to be added to the formulation.
struct SymbreakConstraint {
    SymbreakType type = SymbreakType::Lexicographic;
    std::vector<Index> col_indices;
    std::vector<Real> coefficients;
    Real lower = -1e30;
    Real upper = 1e30;
    std::string name;
};

/// Generate symmetry-breaking constraints from automorphism generators.
///
/// For each generator sigma, adds a symresack or lex-leader constraint:
///   For binary variables with generator sigma:
///     sum_{j: sigma(j)!=j} (x_j - x_{sigma(j)}) >= 0
///   enforces lexicographic ordering x >= sigma(x).
///
/// For orbitopes (matrix of symmetric binary columns):
///   Detects when variables form a matrix structure where columns are
///   permuted by the symmetry group, and adds full orbitope constraints.
class SymbreakGenerator {
public:
    SymbreakGenerator() = default;

    /// Generate symmetry-breaking constraints from generators.
    /// @param generators   Automorphism generators (on variable indices).
    /// @param col_type     Variable types.
    /// @param num_cols     Number of variables.
    /// @param max_constraints  Maximum number of constraints to generate.
    /// @return Vector of symmetry-breaking constraints.
    std::vector<SymbreakConstraint> generate(
        const std::vector<Permutation>& generators,
        const std::vector<VarType>& col_type,
        Index num_cols,
        Index max_constraints = 64) const;

    /// Add symmetry-breaking constraints to the problem formulation.
    /// @param problem      The LP/MIP problem (modified in place).
    /// @param constraints  Constraints to add.
    /// @return Number of constraints actually added.
    static Index addConstraints(LpProblem& problem,
                                const std::vector<SymbreakConstraint>& constraints);

    /// Generate and add lexicographic fixing constraints.
    /// For each orbit, fix the canonical (smallest index) variable to be
    /// lexicographically first: x_canon >= x_j for all j in orbit.
    static std::vector<SymbreakConstraint> generateLexFixing(
        const std::vector<std::vector<Index>>& orbits,
        const std::vector<VarType>& col_type,
        Index num_cols);

    /// Aggregate symmetric constraints in the formulation.
    /// If two constraints are symmetric (one maps to the other under a
    /// generator), replace them with a single tighter constraint.
    /// @return Number of constraints removed by aggregation.
    static Index aggregateSymmetricConstraints(
        LpProblem& problem,
        const std::vector<Permutation>& generators);

private:
    /// Generate a symresack constraint for a single generator.
    static SymbreakConstraint generateSymresack(
        const Permutation& generator,
        const std::vector<VarType>& col_type,
        Index num_cols,
        Index constraint_idx);

    /// Generate a lex-leader constraint for a single generator.
    static SymbreakConstraint generateLexLeader(
        const Permutation& generator,
        const std::vector<VarType>& col_type,
        Index num_cols,
        Index constraint_idx);

    /// Detect orbitope structure in a set of generators.
    /// Returns column groups that form an orbitope, or empty if none found.
    static std::vector<std::vector<Index>> detectOrbitope(
        const std::vector<Permutation>& generators,
        const std::vector<VarType>& col_type,
        Index num_cols);
};

/// Isomorphism pruning for branch-and-bound.
///
/// During tree search, checks if the current node's domain is isomorphic
/// to a previously explored node's domain. If so, the subtree can be pruned.
class IsomorphismPruner {
public:
    IsomorphismPruner() = default;

    /// Set the symmetry generators.
    void setGenerators(const std::vector<Permutation>& generators,
                       Index num_vars);

    /// Check if the current node's bound state can be pruned due to
    /// symmetry with an already-explored configuration.
    /// @param col_lower  Current lower bounds at this node.
    /// @param col_upper  Current upper bounds at this node.
    /// @return true if this node can be pruned.
    [[nodiscard]] bool canPrune(const std::vector<Real>& col_lower,
                                const std::vector<Real>& col_upper) const;

    /// Record a node's bound configuration as explored.
    void recordExplored(const std::vector<Real>& col_lower,
                        const std::vector<Real>& col_upper);

    /// Clear all recorded configurations.
    void clear();

    /// Get number of pruned nodes.
    [[nodiscard]] Index numPruned() const { return num_pruned_; }

private:
    /// Compute a canonical hash for a bound configuration under symmetry.
    [[nodiscard]] std::size_t canonicalHash(
        const std::vector<Real>& col_lower,
        const std::vector<Real>& col_upper) const;

    std::vector<Permutation> generators_;
    Index num_vars_ = 0;
    std::vector<std::size_t> explored_hashes_;
    mutable Index num_pruned_ = 0;
};

}  // namespace mipx
