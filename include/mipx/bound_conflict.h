#pragma once

#include "mipx/bound_reasons.h"
#include "mipx/branch_decision.h"
#include "mipx/cut_pool.h"
#include "mipx/deadline.h"

namespace mipx {

struct LearnedBoundConflict {
    // The conjunction of these conditions is forbidden within scope.
    std::vector<BranchDecision> literals;
    std::vector<BranchDecision> removed_literals;
    // Forbidden binary conditions for the implication between two UIPs.
    std::vector<BranchDecision> reconvergence;
    BoundReasonScope scope;
    Int lbd = 0;
};

/// Resolve a contradiction to the first unique condition at the highest
/// participating level. Local or opaque proofs fall back to path learning.
[[nodiscard]] std::optional<LearnedBoundConflict> analyzeBoundConflict(
    const BoundReasonTrail& trail, bool find_reconvergence = false);

/// Check a recorded contradiction using the global model and only the given
/// branch premises. Reject local rows, cutoff-dependent reasons and opaque inputs.
[[nodiscard]] bool checkGlobalBoundContradiction(const LpProblem& model,
                                                 const BoundReasonTrail& trail,
                                                 std::span<const BranchDecision> premises);

/// Allowance shared by propagation and its independent proof checker.
struct BoundProofBudget {
    Int& rows_left;
    const Deadline& deadline;
    bool interrupted = false;

    [[nodiscard]] bool debitRow() {
        if (rows_left <= 0 || deadline.passed()) {
            interrupted = true;
            return false;
        }
        --rows_left;
        return true;
    }
};

/// Return the branch premises used by a checked global contradiction.
[[nodiscard]] std::optional<std::vector<BranchDecision>> checkedGlobalBoundConflict(
    const LpProblem& model, const BoundReasonTrail& trail, std::span<const BranchDecision> premises,
    BoundProofBudget* budget = nullptr);

/// Check a deduction against the global model without branch premises.
[[nodiscard]] bool checkGlobalBoundDeduction(const LpProblem& model, const BoundReasonTrail& trail,
                                             BoundReasonId id, BoundProofBudget* budget = nullptr);

/// Recheck a reconvergence clause using only the global model and return its
/// binary inequality if the forbidden assignment can be ruled out.
[[nodiscard]] std::optional<Cut> checkedReconvergenceCut(std::span<const BranchDecision> forbidden,
                                                         const LpProblem& global_model);

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
