#pragma once

#include "mipx/core.h"

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

}  // namespace mipx
