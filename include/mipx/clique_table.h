#pragma once

#include <span>
#include <vector>

#include "mipx/conflict_graph.h"
#include "mipx/core.h"
#include "mipx/cut_pool.h"
#include "mipx/lp_problem.h"

namespace mipx {

/// A clique is a set of literals that are mutually conflicting.
/// At most one of the literals in a clique can be 1 simultaneously.
struct Clique {
    std::vector<Literal> literals;
    bool is_equality = false;  // If true, exactly one literal must be 1.
    bool from_sos1 = false;    // If true, originated from an SOS1 constraint.
};

/// Persistent clique table with maximal clique detection, merging,
/// and subsumption.
class CliqueTable {
public:
    CliqueTable() = default;

    /// Build the clique table from problem data and conflict graph.
    /// Detects maximal cliques via greedy extension from the conflict graph.
    /// Also extracts cliques from SOS1 constraints.
    void build(const LpProblem& problem, const ConflictGraph& graph);

    /// Extract cliques from a new cut added to the pool.
    /// Returns the number of new cliques found.
    Int extractFromCut(const Cut& cut, const LpProblem& problem,
                       const ConflictGraph& graph);

    /// Number of cliques stored.
    [[nodiscard]] Index numCliques() const {
        return static_cast<Index>(cliques_.size());
    }

    /// Access a clique by index.
    [[nodiscard]] const Clique& clique(Index i) const { return cliques_[i]; }

    /// Return all clique indices containing a given literal.
    [[nodiscard]] std::span<const Index> cliquesOf(Literal lit) const;

    /// Perform clique merging and subsumption.
    /// Removes cliques that are subsets of other cliques.
    /// Returns the number of cliques removed.
    Int mergeAndSubsume();

    /// Separate violated clique cover inequalities from the current LP solution.
    /// Returns the number of cuts added to the pool.
    Int separateCliqueCover(const LpProblem& problem,
                            std::span<const Real> primals, CutPool& pool,
                            Real min_violation = 1e-5,
                            Int max_cuts = 50) const;

    /// Get objective cliques from 0-1 knapsack analysis of the objective.
    /// These identify sets of variables where at most one should be 1
    /// based on objective function structure.
    [[nodiscard]] std::vector<Clique> findObjectiveCliques(
        const LpProblem& problem, const ConflictGraph& graph) const;

    /// Clique partitioning for branching: partition binary variables into
    /// clique covers. Returns a vector of cliques covering the fractional
    /// binary variables.
    [[nodiscard]] std::vector<Clique> cliquePartition(
        std::span<const Real> primals,
        const LpProblem& problem) const;

    /// Propagate cliques: if a literal in a clique is fixed to 1,
    /// fix all other literals to 0. Returns false if infeasibility detected.
    /// Outputs the (col, new_lower, new_upper) tuples for tightened variables.
    struct BoundUpdate {
        Index col;
        Real new_lower;
        Real new_upper;
    };
    bool propagate(std::span<const Real> lower, std::span<const Real> upper,
                   std::vector<BoundUpdate>& updates) const;

    /// Clique substitutions in presolve: identify variables that can be
    /// eliminated via clique structure. Returns pairs (eliminated_var,
    /// substitute_var) where eliminated_var = 1 - substitute_var within a
    /// 2-clique.
    struct Substitution {
        Index eliminated;
        Index substitute;
        bool complement;  // eliminated = complement ? (1 - substitute)
                          //                         : substitute
    };
    [[nodiscard]] std::vector<Substitution> findSubstitutions(
        const LpProblem& problem) const;

private:
    /// Greedy extension of a partial clique to a maximal clique.
    void extendClique(Clique& clique, const ConflictGraph& graph) const;

    /// Add a clique, maintaining the literal-to-clique index.
    void addClique(Clique clique);

    /// All stored cliques.
    std::vector<Clique> cliques_;

    /// For each literal (indexed by literal id), the list of clique indices
    /// containing it.
    std::vector<std::vector<Index>> lit_to_cliques_;

    /// Number of original binary variables (for sizing lit_to_cliques_).
    Index num_cols_ = 0;
};

}  // namespace mipx
