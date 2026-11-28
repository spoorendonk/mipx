#pragma once

#include "mipx/bnb_node.h"
#include "mipx/core.h"
#include "mipx/dual_simplex.h"
#include "mipx/lp_problem.h"
#include "mipx/symmetry.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>
#include <vector>

namespace mipx {

/// Check if a value is integer within tolerance.
inline bool isIntegral(Real val, Real tol = 1e-6) {
    return std::abs(val - std::round(val)) <= tol;
}

/// Fractionality: distance to nearest integer.
inline Real fractionality(Real val) {
    return std::abs(val - std::round(val));
}

/// Interface for branching variable selection.
class BranchingRule {
public:
    virtual ~BranchingRule() = default;

    /// Select a fractional integer variable to branch on.
    /// Returns the variable index, or -1 if all integer variables are integral.
    virtual Index select(std::span<const Real> primal_values, std::span<const VarType> var_types,
                         std::span<const Real> col_lower,
                         std::span<const Real> col_upper) const = 0;
};

/// Branch on the variable with fractionality closest to 0.5.
class MostFractionalBranching : public BranchingRule {
public:
    Index select(std::span<const Real> primal_values, std::span<const VarType> var_types,
                 std::span<const Real> col_lower, std::span<const Real> col_upper) const override;
};

/// Branch on the first fractional integer variable found.
class FirstFractionalBranching : public BranchingRule {
public:
    Index select(std::span<const Real> primal_values, std::span<const VarType> var_types,
                 std::span<const Real> col_lower, std::span<const Real> col_upper) const override;
};

struct BranchingSelection {
    Index variable = -1;
    bool used_pseudocost = false;
    bool used_reliable_pseudocost = false;
    Index inference_baseline = -1;
    Index cutoff_baseline = -1;
};

struct MipDirectionalInferenceStats {
    uint64_t samples = 0;
    uint64_t tightened_columns = 0;
    uint64_t cutoffs = 0;
    uint64_t skipped = 0;
};

struct MipBranchInferenceStats {
    MipDirectionalInferenceStats down;
    MipDirectionalInferenceStats up;
};

struct MipDirectionalCutoffStats {
    uint64_t qualified_attempts = 0;
    uint64_t cutoffs = 0;
};

struct MipBranchCutoffStats {
    MipDirectionalCutoffStats down;
    MipDirectionalCutoffStats up;
};

struct InferenceCoverageBand {
    /// A challenger exists within the score gap.
    uint64_t close = 0;
    /// The winner and a close challenger both have at least four samples per direction.
    uint64_t ready = 0;
    /// A ready challenger has a strictly better weaker-direction mean.
    uint64_t favored = 0;
};

struct InferenceCoveragePath {
    static constexpr std::array<int, 3> kGapPercents = {0, 1, 5};
    uint64_t decisions = 0;
    /// Cumulative bands. Each decision contributes at most once per counter.
    std::array<InferenceCoverageBand, kGapPercents.size()> bands{};
    uint64_t selection_changes = 0;
    /// The final branch differs from the baseline after applying arbitration to both.
    uint64_t surviving_changes = 0;
};

struct BranchInferenceCoverageStats {
    InferenceCoveragePath reliable;
    InferenceCoveragePath fallback;

    /// Both columns are the results of the same downstream arbitration rule.
    void recordArbitration(const BranchingSelection& selection, Index chosen,
                           Index baseline_chosen);
};

/// Live first-pass histories in original model column order. Each working
/// column maps to an entry in history. An empty mapping uses the working index.
struct BranchInferenceCoverage {
    std::span<const MipBranchInferenceStats> history;
    std::span<const Index> original_columns;
    BranchInferenceCoverageStats& stats;
    bool select_by_inference = false;
};

struct CutoffComparisonPath {
    uint64_t decisions = 0;
    uint64_t ready_pairs = 0;
    uint64_t favored_decisions = 0;
    uint64_t selection_changes = 0;
    uint64_t surviving_changes = 0;
};

enum class BranchCutoffScoreMode { Off, RecordOnly, Score };

struct BranchCutoffComparisonStats {
    CutoffComparisonPath reliable;
    CutoffComparisonPath fallback;

    void recordArbitration(const BranchingSelection& selection, Index chosen,
                           Index baseline_chosen);
};

/// Live first-LP dispositions in original model column order.
struct BranchCutoffComparison {
    std::span<const MipBranchCutoffStats> history;
    std::span<const Index> original_columns;
    BranchCutoffComparisonStats& stats;
    bool select_by_cutoff = false;
};

struct BranchSelectionHistories {
    const BranchInferenceCoverage* inference = nullptr;
    const BranchCutoffComparison* cutoff = nullptr;
};

struct BranchingTelemetry {
    Int selections = 0;
    Int strong_branch_calls = 0;
    Int strong_branch_probes = 0;
    Int strong_branch_probe_iters = 0;
    double strong_branch_probe_work_units = 0.0;
    Int pseudocost_uses = 0;
    Int pseudocost_hits = 0;
    /// Pseudocost samples recorded from strong-branch probes.
    Int pseudocost_updates_strong = 0;
    /// Pseudocost samples recorded from solved child nodes in the tree.
    Int pseudocost_updates_tree = 0;
    /// Child nodes whose LP ended infeasible, cut off or without an optimum,
    /// so gave no gain to record.
    Int pseudocost_tree_skipped = 0;

    BranchingTelemetry& operator+=(const BranchingTelemetry& other) {
        selections += other.selections;
        strong_branch_calls += other.strong_branch_calls;
        strong_branch_probes += other.strong_branch_probes;
        strong_branch_probe_iters += other.strong_branch_probe_iters;
        strong_branch_probe_work_units += other.strong_branch_probe_work_units;
        pseudocost_uses += other.pseudocost_uses;
        pseudocost_hits += other.pseudocost_hits;
        pseudocost_updates_strong += other.pseudocost_updates_strong;
        pseudocost_updates_tree += other.pseudocost_updates_tree;
        pseudocost_tree_skipped += other.pseudocost_tree_skipped;
        return *this;
    }
};

/// One directional observation from a strong-branch probe.
struct StrongBranchPseudocostSample {
    Index variable = -1;
    bool up_direction = false;
    Real gain_per_unit = 0.0;
};

/// What a solved child node's LP says about the branching that created it:
/// the branching decision, the parent's LP value of the variable and LP
/// objective, and the child's LP outcome (#199).
struct TreePseudocostSample {
    BranchDecision branch;
    Real branch_value = 0.0;
    Real parent_objective = -kInf;
    Status status = Status::Error;
    Real objective = 0.0;
    /// The child's LP reached the incumbent's objective, so the node is cut
    /// off; its gain is bounded by the cutoff, not measured by the branching.
    bool cutoff = false;

    [[nodiscard]] static TreePseudocostSample fromChild(const BnbNode& child, Status status,
                                                        Real objective, bool cutoff = false) {
        return {child.branch, child.branch_value, child.parent_objective,
                status,       objective,          cutoff};
    }
};

class ReliabilityBranching {
public:
    static constexpr Int kDefaultStrongBranchProbeBudget = 8;

    ReliabilityBranching() = default;

    void reset(Index num_cols);
    void setBalancedScore(bool enabled) { balanced_score_ = enabled; }
    void setReliabilityThreshold(Int t) { reliability_threshold_ = std::max<Int>(1, t); }
    void setStrongBranchMaxCandidates(Int c) {
        strong_branch_max_candidates_ = std::max<Int>(1, c);
    }
    /// Directional probes per strong-branch call. A pair is the smallest
    /// useful budget; below 2 no probe runs, which disables strong branching.
    void setStrongBranchProbeBudget(Int b) { strong_branch_probe_budget_ = std::max<Int>(0, b); }
    [[nodiscard]] Int strongBranchProbeBudget() const { return strong_branch_probe_budget_; }
    void setStrongBranchIterLimit(Int iters) {
        strong_branch_iter_limit_ = std::max<Int>(1, iters);
    }
    void setPseudocostFallback(Real fallback) {
        pseudocost_fallback_ = std::max<Real>(1e-8, fallback);
    }
    void setSymmetryManager(const SymmetryManager* manager) { symmetry_manager_ = manager; }

    [[nodiscard]] bool isReliable(Index var) const;
    [[nodiscard]] Real upPseudoCost(Index var) const;
    [[nodiscard]] Real downPseudoCost(Index var) const;
    [[nodiscard]] Int upReliability(Index var) const;
    [[nodiscard]] Int downReliability(Index var) const;

    void updatePseudoCost(Index var, bool up_direction, Real gain_per_unit);

    /// Record the pseudocost sample a solved child node yields: the gain of
    /// its LP objective over the parent's, per unit of branching distance.
    /// Only an optimal child LP that is not cut off carries a gain; an
    /// infeasible or cut-off child, or one whose LP stopped early, is counted
    /// as skipped and records nothing.
    /// Returns true when a sample was recorded.
    bool updateFromChild(const TreePseudocostSample& sample, BranchingTelemetry& telemetry);

    BranchingSelection select(DualSimplexSolver& lp, const LpProblem& problem,
                              std::span<const Real> primal_values, std::span<const Real> col_lower,
                              std::span<const Real> col_upper, Real node_objective,
                              bool force_strong_branch, BranchingTelemetry& telemetry);

    /// Nonnegative conflict activity in model column order biases pseudocost
    /// ranking. An empty span disables the bias. Gain observations are unweighted.
    BranchingSelection selectWithConflictActivity(DualSimplexSolver& lp, const LpProblem& problem,
                                                  std::span<const Real> primal_values,
                                                  std::span<const Real> col_lower,
                                                  std::span<const Real> col_upper,
                                                  Real node_objective, bool force_strong_branch,
                                                  BranchingTelemetry& telemetry,
                                                  std::span<const Real> conflict_activity);

    /// Optionally observe pseudocost selection with the same conflict activity scoring.
    BranchingSelection selectWithInferenceCoverage(
        DualSimplexSolver& lp, const LpProblem& problem, std::span<const Real> primal_values,
        std::span<const Real> col_lower, std::span<const Real> col_upper, Real node_objective,
        bool force_strong_branch, BranchingTelemetry& telemetry,
        std::span<const Real> conflict_activity, const BranchInferenceCoverage* coverage);

    BranchingSelection selectWithHistories(DualSimplexSolver& lp, const LpProblem& problem,
                                           std::span<const Real> primal_values,
                                           std::span<const Real> col_lower,
                                           std::span<const Real> col_upper, Real node_objective,
                                           bool force_strong_branch, BranchingTelemetry& telemetry,
                                           std::span<const Real> conflict_activity,
                                           BranchSelectionHistories histories);

    /// Read the batch's pseudocosts and collect this node's probe observations
    /// for ordered application after every node in the batch has finished.
    BranchingSelection selectDeferred(DualSimplexSolver& lp, const LpProblem& problem,
                                      std::span<const Real> primal_values,
                                      std::span<const Real> col_lower,
                                      std::span<const Real> col_upper, Real node_objective,
                                      Int probe_budget, BranchingTelemetry& telemetry,
                                      std::vector<StrongBranchPseudocostSample>& samples) const;

private:
    template <typename RecordProbe>
    BranchingSelection selectWithProbeSink(
        DualSimplexSolver& lp, const LpProblem& problem, std::span<const Real> primal_values,
        std::span<const Real> col_lower, std::span<const Real> col_upper, Real node_objective,
        bool force_strong_branch, Int probe_budget, BranchingTelemetry& telemetry,
        std::span<const Real> conflict_activity, BranchSelectionHistories histories,
        RecordProbe record_probe) const;

    struct PseudoCost {
        Real up = 0.0;
        Real down = 0.0;
        Int up_count = 0;
        Int down_count = 0;
    };

    struct Candidate {
        Index var = -1;
        Real value = 0.0;
        Real down_dist = 0.0;
        Real up_dist = 0.0;
        Real objective_score = 0.0;
        Real pseudo_score = 0.0;
        Real prefilter_score = 0.0;
        bool reliable = false;

        [[nodiscard]] bool winsHistoryTie(const Candidate& other, bool objective_only) const {
            const Real score = objective_only ? objective_score : pseudo_score;
            const Real other_score = objective_only ? other.objective_score : other.pseudo_score;
            return score > other_score || (score == other_score && var < other.var);
        }
    };

    static void compareInference(const BranchInferenceCoverage& coverage,
                                 std::span<const Candidate> candidates, Real winner_score,
                                 BranchingSelection& selection);
    static Index compareCutoffs(const BranchCutoffComparison& comparison,
                                std::span<const Candidate> candidates, Real winner_score,
                                const BranchingSelection& selection);

    [[nodiscard]] bool inRange(Index var) const {
        return var >= 0 && var < static_cast<Index>(pseudocosts_.size());
    }

    [[nodiscard]] Real safeUpCost(Index var) const;
    [[nodiscard]] Real safeDownCost(Index var) const;
    [[nodiscard]] Real directionalScore(Real down_gain, Real up_gain) const;
    [[nodiscard]] static Real blendScore(Real frac, Real pseudo_score);

    const SymmetryManager* symmetry_manager_ = nullptr;
    [[nodiscard]] bool isCanonicalCandidate(Index var, std::span<const Real> primal_values) const {
        if (symmetry_manager_ == nullptr) {
            return true;
        }
        const Index canon = symmetry_manager_->canonical(var);
        if (canon == var) {
            return true;
        }
        return isIntegral(primal_values[canon]);
    }

    std::vector<PseudoCost> pseudocosts_;
    Int reliability_threshold_ = 4;
    Int strong_branch_max_candidates_ = 8;
    Int strong_branch_probe_budget_ = kDefaultStrongBranchProbeBudget;  // Directional probes.
    Int strong_branch_iter_limit_ = 60;
    Real pseudocost_fallback_ = 1.0;
    bool balanced_score_ = false;
    static constexpr Real kStrongInfeasibleGain = 1e4;
};

/// Pseudocost-based node estimate for a solved node.
///
///   estimate = lp_bound + sum_{j fractional} min(f_j * down_j, (1 - f_j) * up_j)
///
/// where f_j is the fractional part of x_j, and down_j / up_j are the rule's
/// per-unit pseudocosts. Directions with no observations fall back to the
/// rule's pseudocost fallback (see ReliabilityBranching::setPseudocostFallback),
/// so the estimate is always finite when lp_bound is finite. A non-finite
/// lp_bound is returned unchanged.
[[nodiscard]] Real pseudocostNodeEstimate(const ReliabilityBranching& rule,
                                          const LpProblem& problem,
                                          std::span<const Real> primal_values, Real lp_bound);

/// Create two child nodes by branching on variable branch_var with value branch_val.
/// Left child: x_j <= floor(branch_val), right child: x_j >= ceil(branch_val).
std::pair<BnbNode, BnbNode> createChildren(BnbNode parent, Index branch_var, Real branch_val);

/// Split a violated SOS2 at an interior position. Both sides keep that position.
std::vector<BnbNode> createSos2Children(BnbNode parent, std::span<const Index> members,
                                        std::size_t split, std::span<const Real> lower,
                                        std::span<const Real> upper);

}  // namespace mipx
