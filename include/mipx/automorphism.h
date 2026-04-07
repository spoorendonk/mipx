#pragma once

#include <cstddef>
#include <functional>
#include <vector>

#include "mipx/core.h"

namespace mipx {

// Forward declarations.
struct LpProblem;

/// A permutation of vertices, represented as an image array:
/// perm[i] = j means vertex i maps to vertex j.
using Permutation = std::vector<Index>;

/// A colored graph for automorphism detection.
/// Vertices are labeled 0..n-1 with integer colors.
/// Edges are undirected, stored as adjacency lists.
struct ColoredGraph {
    Index num_vertices = 0;
    std::vector<std::vector<Index>> adj;   // adjacency lists
    std::vector<Index> colors;             // vertex colors

    void addEdge(Index u, Index v);
    void addVertex(Index color);
};

/// Build the constraint-variable incidence graph for automorphism detection.
/// Returns a colored graph where:
///   - Vertices 0..num_cols-1 are variable nodes
///   - Vertices num_cols..num_cols+num_rows-1 are constraint nodes
///   - Edges connect variables to constraints they appear in
///   - Colors encode variable type/bounds/objective and constraint type
ColoredGraph buildIncidenceGraph(const LpProblem& problem);

/// Result of automorphism computation: a set of generators for the
/// automorphism group, plus computed orbits.
struct AutomorphismResult {
    std::vector<Permutation> generators;
    std::vector<std::vector<Index>> orbits;  // variable-only orbits
    Index num_vertices = 0;
    Index num_variable_vertices = 0;         // first N vertices are variables
    double work_units = 0.0;
};

/// Compute automorphisms of a colored graph using partition refinement
/// with individualization-refinement (simplified nauty-style).
///
/// Callback is invoked for each discovered generator.
/// Only generators that move at least one vertex are reported.
AutomorphismResult computeAutomorphisms(const ColoredGraph& graph,
                                        Index num_variable_vertices);

/// Apply a permutation to a vector of indices.
void applyPermutation(const Permutation& perm, std::vector<Index>& vec);

/// Compose two permutations: result[i] = b[a[i]].
Permutation composePermutations(const Permutation& a, const Permutation& b);

/// Compute the inverse permutation.
Permutation inversePermutation(const Permutation& perm);

/// Check if a permutation is the identity.
bool isIdentity(const Permutation& perm);

/// Compute orbits from a set of generators on n elements.
/// Returns orbits as groups of indices, each sorted.
std::vector<std::vector<Index>> computeOrbitsFromGenerators(
    const std::vector<Permutation>& generators, Index n);

/// Compute orbits restricted to a subset of elements (variable indices).
std::vector<std::vector<Index>> computeVariableOrbits(
    const std::vector<Permutation>& generators, Index num_vars);

}  // namespace mipx
