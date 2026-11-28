#pragma once

#include "mipx/bound_conflict.h"
#include "mipx/core.h"

#include <optional>
#include <span>
#include <vector>

namespace mipx {

struct BranchDecision;
struct LpProblem;

/// Recheck a node LP ray on the global rows and the given persistent bounds.
/// The ray is in LP row order, with the sorted local row indices interleaved.
/// On success, greedily drop bounds from the coalesced path sorted by column
/// and then by direction, lower first. Keep at least one literal so a proof
/// of the whole root box does not enter the path pool.
/// A failed recheck leaves the path intact.
bool shortenConflictWithRay(const LpProblem& global_rows, std::span<const Real> ray,
                            std::span<const Index> local_rows, std::span<const Real> global_lower,
                            std::span<const Real> global_upper, std::vector<BranchDecision>& path);

/// Prove that c*x + offset is strictly above cutoff in the minimization
/// model's units. The unscaled simplex duals use the opposite row signs to a
/// Farkas ray. On success, keep only the necessary path bounds, as for a ray.
bool shortenConflictWithCutoff(const LpProblem& global_rows, std::span<const Real> dual,
                               std::span<const Index> local_rows, Real cutoff,
                               std::vector<BranchDecision>& path);

/// Recheck an LP proof using explained endpoints from a replay of the branch
/// path, then resolve the retained premises through the reason trail. Here
/// weights are a simplex dual for an objective proof or a Farkas ray otherwise.
std::optional<LearnedBoundConflict> checkCombinedLpConflict(
    const LpProblem& global_rows, std::span<const Real> weights, std::span<const Index> local_rows,
    const BoundReasonTrail& trail, BoundReasonScope scope, bool objective_proof);

}  // namespace mipx
