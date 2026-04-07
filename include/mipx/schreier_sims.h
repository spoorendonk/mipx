#pragma once

#include <cstddef>
#include <optional>
#include <unordered_map>
#include <vector>

#include "mipx/automorphism.h"
#include "mipx/core.h"

namespace mipx {

/// Schreier-Sims representation of a permutation group.
///
/// Provides efficient orbit computation and group membership testing
/// using a base and strong generating set (BSGS).
///
/// The base B = [b_0, b_1, ..., b_{k-1}] is an ordered sequence of points
/// such that the only group element fixing all base points is the identity.
///
/// For each base point b_i, we store the orbit of b_i under the stabilizer
/// G^(i) = Stab(b_0, ..., b_{i-1}) and coset representatives (transversal).
class SchreierSims {
public:
    SchreierSims() = default;

    /// Build the BSGS from a set of generators on n elements.
    /// Only tracks orbits for the first num_tracked_points elements
    /// (typically variable vertices only).
    void build(const std::vector<Permutation>& generators, Index n,
               Index num_tracked_points = -1);

    /// Check if a permutation belongs to the group.
    [[nodiscard]] bool isMember(const Permutation& perm) const;

    /// Get the orbit of a point under the full group.
    [[nodiscard]] std::vector<Index> orbit(Index point) const;

    /// Get the orbit of a point under the stabilizer of all points
    /// in the given fixed set. Useful for orbital fixing at tree nodes.
    [[nodiscard]] std::vector<Index> orbitUnderStabilizer(
        Index point, const std::vector<Index>& fixed_points) const;

    /// Get the base.
    [[nodiscard]] const std::vector<Index>& base() const { return base_; }

    /// Get the number of elements.
    [[nodiscard]] Index size() const { return n_; }

    /// Get the strong generating set.
    [[nodiscard]] const std::vector<Permutation>& strongGenerators() const {
        return strong_generators_;
    }

    /// Get all computed orbits (non-trivial only).
    [[nodiscard]] std::vector<std::vector<Index>> allOrbits() const;

    /// Get work units consumed during build.
    [[nodiscard]] double buildWorkUnits() const { return work_units_; }

    /// Check if the structure has been built.
    [[nodiscard]] bool isBuilt() const { return built_; }

private:
    /// A level in the Schreier-Sims structure, corresponding to one base point.
    struct Level {
        Index base_point = -1;
        std::vector<Index> orbit;                         // orbit of base_point
        std::unordered_map<Index, Permutation> transversal;  // coset reps
        std::vector<Permutation> stabilizer_generators;   // generators for next level
    };

    /// Compute the orbit and transversal for a given point and set of generators.
    static Level computeOrbitAndTransversal(
        Index point, const std::vector<Permutation>& generators, Index n);

    /// Sift a permutation through the BSGS. Returns the residual
    /// (identity if the element is in the group).
    [[nodiscard]] Permutation sift(const Permutation& perm) const;

    /// Add a new generator to the structure, propagating through levels.
    void addGenerator(const Permutation& perm);

    std::vector<Level> levels_;
    std::vector<Index> base_;
    std::vector<Permutation> strong_generators_;
    Index n_ = 0;
    Index num_tracked_ = -1;
    bool built_ = false;
    double work_units_ = 0.0;
};

}  // namespace mipx
