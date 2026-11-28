#pragma once

#include "mipx/bound_reasons.h"
#include "mipx/branch_decision.h"

namespace mipx {

struct LearnedBoundConflict {
    // The conjunction of these conditions is forbidden within scope.
    std::vector<BranchDecision> literals;
    std::vector<BranchDecision> removed_literals;
    BoundReasonScope scope;
    Int lbd = 0;
};

/// Resolve a contradiction to the first unique condition at the highest
/// participating level. Local or opaque proofs fall back to path learning.
[[nodiscard]] std::optional<LearnedBoundConflict> analyzeBoundConflict(
    const BoundReasonTrail& trail);

/// Resolve the bound premises of an independently checked contradiction.
[[nodiscard]] std::optional<LearnedBoundConflict> analyzeBoundConflict(
    const BoundReasonTrail& trail, const BoundReasonSupports& supports, BoundReasonScope scope);

struct BoundConflictInference {
    // An engaged inference without a tightening cuts off the domain.
    std::optional<BoundEndpoint> tightening;
    // Borrowed from the clause. Together these spans are the entailed premises.
    std::span<const BranchDecision> antecedents_before;
    std::span<const BranchDecision> antecedents_after;
    BoundReasonScope scope;
};

/// Check one clause against the current domain. A scheduler may call this
/// after any change to a clause column or to the active cutoff. Returns an
/// integer complement for a unit clause, or a cutoff if all literals hold.
[[nodiscard]] std::optional<BoundConflictInference> evaluateBoundConflict(
    std::span<const BranchDecision> literals, BoundReasonScope scope, std::span<const Real> lower,
    std::span<const Real> upper, std::span<const VarType> types, Real cutoff = kInf);

}  // namespace mipx
