#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <span>
#include <utility>
#include <vector>

#include "mipx/bnb_node.h"
#include "mipx/core.h"
#include "mipx/dual_simplex.h"
#include "mipx/lp_problem.h"
#include "mipx/symmetry.h"

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
    virtual Index select(std::span<const Real> primal_values,
                         std::span<const VarType> var_types,
                         std::span<const Real> col_lower,
                         std::span<const Real> col_upper) const = 0;
};

/// Branch on the variable with fractionality closest to 0.5.
class MostFractionalBranching : public BranchingRule {
public:
    Index select(std::span<const Real> primal_values,
                 std::span<const VarType> var_types,
                 std::span<const Real> col_lower,
                 std::span<const Real> col_upper) const override;
};

/// Branch on the first fractional integer variable found.
class FirstFractionalBranching : public BranchingRule {
public:
    Index select(std::span<const Real> primal_values,
                 std::span<const VarType> var_types,
                 std::span<const Real> col_lower,
                 std::span<const Real> col_upper) const override;
};

struct BranchingSelection {
    Index variable = -1;
    bool used_pseudocost = false;
    bool used_reliable_pseudocost = false;
};

struct BranchingTelemetry {
    Int selections = 0;
    Int strong_branch_calls = 0;
    Int strong_branch_probes = 0;
    Int strong_branch_probe_iters = 0;
    double strong_branch_probe_work_units = 0.0;
    Int pseudocost_uses = 0;
    Int pseudocost_hits = 0;
};

class ReliabilityBranching {
public:
    ReliabilityBranching() = default;

    void reset(Index num_cols);
    void setReliabilityThreshold(Int t) { reliability_threshold_ = std::max<Int>(1, t); }
    void setStrongBranchMaxCandidates(Int c) { strong_branch_max_candidates_ = std::max<Int>(1, c); }
    void setStrongBranchProbeBudget(Int b) { strong_branch_probe_budget_ = std::max<Int>(2, b); }
    void setStrongBranchIterLimit(Int iters) { strong_branch_iter_limit_ = std::max<Int>(1, iters); }
    void setPseudocostFallback(Real fallback) { pseudocost_fallback_ = std::max<Real>(1e-8, fallback); }
    void setSymmetryManager(const SymmetryManager* manager) { symmetry_manager_ = manager; }

    [[nodiscard]] bool isReliable(Index var) const;
    [[nodiscard]] Real upPseudoCost(Index var) const;
    [[nodiscard]] Real downPseudoCost(Index var) const;
    [[nodiscard]] Int upReliability(Index var) const;
    [[nodiscard]] Int downReliability(Index var) const;

    void updatePseudoCost(Index var, bool up_direction, Real gain_per_unit);

    BranchingSelection select(DualSimplexSolver& lp,
                              const LpProblem& problem,
                              std::span<const Real> primal_values,
                              std::span<const Real> col_lower,
                              std::span<const Real> col_upper,
                              Real node_objective,
                              bool force_strong_branch,
                              BranchingTelemetry& telemetry);

private:
    struct PseudoCost {
        Real up = 0.0;
        Real down = 0.0;
        Int up_count = 0;
        Int down_count = 0;
    };

    struct Candidate {
        Index var = -1;
        Real value = 0.0;
        Real frac = 0.0;
        Real down_dist = 0.0;
        Real up_dist = 0.0;
        Real pseudo_score = 0.0;
        Real prefilter_score = 0.0;
        bool reliable = false;
    };

    [[nodiscard]] bool inRange(Index var) const {
        return var >= 0 && var < static_cast<Index>(pseudocosts_.size());
    }

    [[nodiscard]] Real safeUpCost(Index var) const;
    [[nodiscard]] Real safeDownCost(Index var) const;
    [[nodiscard]] static Real blendScore(Real frac, Real pseudo_score);

    const SymmetryManager* symmetry_manager_ = nullptr;
    [[nodiscard]] bool isCanonicalCandidate(Index var,
                                            std::span<const Real> primal_values) const {
        if (symmetry_manager_ == nullptr) return true;
        const Index canon = symmetry_manager_->canonical(var);
        if (canon == var) return true;
        return isIntegral(primal_values[canon]);
    }

    std::vector<PseudoCost> pseudocosts_;
    Int reliability_threshold_ = 4;
    Int strong_branch_max_candidates_ = 8;
    Int strong_branch_probe_budget_ = 8;  // Directional probes.
    Int strong_branch_iter_limit_ = 60;
    Real pseudocost_fallback_ = 1.0;
    static constexpr Real kStrongInfeasibleGain = 1e4;
};

/// Create two child nodes by branching on variable branch_var with value branch_val.
/// Left child: x_j <= floor(branch_val), right child: x_j >= ceil(branch_val).
std::pair<BnbNode, BnbNode> createChildren(BnbNode parent,
                                            Index branch_var,
                                            Real branch_val);

}  // namespace mipx
