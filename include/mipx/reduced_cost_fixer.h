#pragma once

#include <span>
#include <vector>

#include "mipx/core.h"
#include "mipx/lp_problem.h"

namespace mipx {

/// Statistics for reduced-cost fixing.
struct RcFixingStats {
    Int root_global_fixings = 0;
    Int root_global_tightenings = 0;
    Int tree_local_fixings = 0;
    Int tree_local_tightenings = 0;
    Int propagation_triggers = 0;
};

/// A single bound change produced by reduced-cost fixing, with provenance.
struct RcBoundChange {
    Index variable = -1;
    Real old_lower = 0.0;
    Real old_upper = 0.0;
    Real new_lower = 0.0;
    Real new_upper = 0.0;
    bool is_global = false;
};

/// Standalone reduced-cost fixing engine.
///
/// Called at the root (after LP solve) to derive global fixings that persist
/// across all nodes, and at each tree node (after node LP solve) to tighten
/// local bounds.
///
/// Global fixings: when rc * (ub - lb) >= gap, a variable can be permanently
/// fixed to its bound value.
///
/// Local fixings: tighten bounds using node-local LP reduced costs and the
/// current incumbent gap.
class ReducedCostFixer {
public:
    /// Initialize with problem dimensions and variable types.
    void load(const LpProblem& problem);

    /// Reset all state (global fixings, stats).
    void reset();

    /// Apply global reduced-cost fixing at the root node.
    ///
    /// \param reduced_costs  Reduced costs from the root LP solution.
    /// \param primals        Primal values from the root LP solution.
    /// \param lp_objective   Root LP objective value.
    /// \param incumbent      Current best incumbent (kInf if none).
    /// \param col_lower      Current lower bounds (modified in place).
    /// \param col_upper      Current upper bounds (modified in place).
    /// \param tightened_vars Output: indices of variables whose bounds changed.
    /// \return false if infeasibility detected.
    bool applyGlobalFixing(std::span<const Real> reduced_costs,
                           std::span<const Real> primals,
                           Real lp_objective,
                           Real incumbent,
                           std::vector<Real>& col_lower,
                           std::vector<Real>& col_upper,
                           std::vector<Index>& tightened_vars);

    /// Apply local reduced-cost tightening at a tree node.
    ///
    /// \param reduced_costs  Reduced costs from the node LP solution.
    /// \param primals        Primal values from the node LP solution.
    /// \param node_objective Node LP objective value.
    /// \param incumbent      Current best incumbent.
    /// \param col_lower      Current lower bounds (modified in place).
    /// \param col_upper      Current upper bounds (modified in place).
    /// \param tightened_vars Output: indices of variables whose bounds changed.
    /// \return false if infeasibility detected.
    bool applyLocalFixing(std::span<const Real> reduced_costs,
                          std::span<const Real> primals,
                          Real node_objective,
                          Real incumbent,
                          std::vector<Real>& col_lower,
                          std::vector<Real>& col_upper,
                          std::vector<Index>& tightened_vars);

    /// Enforce previously computed global fixings on the given bounds.
    /// Call at each tree node to carry forward root fixings without
    /// recomputation.
    ///
    /// \param col_lower  Current lower bounds (modified in place).
    /// \param col_upper  Current upper bounds (modified in place).
    /// \param tightened_vars Output: indices of variables whose bounds changed.
    /// \return false if infeasibility detected.
    bool enforceGlobalFixings(std::vector<Real>& col_lower,
                              std::vector<Real>& col_upper,
                              std::vector<Index>& tightened_vars) const;

    /// Get the number of globally fixed variables.
    [[nodiscard]] Int numGlobalFixings() const {
        return static_cast<Int>(global_changes_.size());
    }

    /// Get the global lower bound for a variable (after global RC fixing).
    [[nodiscard]] Real globalLower(Index col) const { return global_lower_[col]; }

    /// Get the global upper bound for a variable (after global RC fixing).
    [[nodiscard]] Real globalUpper(Index col) const { return global_upper_[col]; }

    /// Statistics.
    [[nodiscard]] const RcFixingStats& stats() const { return stats_; }

    /// Check whether the engine has been loaded.
    [[nodiscard]] bool loaded() const { return loaded_; }

private:
    /// Core RC tightening logic shared by global and local fixing.
    /// Returns number of bounds tightened, or -1 on infeasibility.
    Int rcTighten(std::span<const Real> reduced_costs,
                  std::span<const Real> primals,
                  Real gap,
                  std::vector<Real>& col_lower,
                  std::vector<Real>& col_upper,
                  std::vector<Index>& tightened_vars);

    bool loaded_ = false;
    Index num_cols_ = 0;
    std::vector<VarType> col_type_;

    // Global bounds derived from root RC fixing.
    std::vector<Real> global_lower_;
    std::vector<Real> global_upper_;

    // Record of which variables were globally tightened.
    std::vector<RcBoundChange> global_changes_;

    RcFixingStats stats_{};

    static constexpr Real kRcTol = 1e-7;
    static constexpr Real kBoundTol = 1e-9;
    static constexpr Real kFeasTol = 1e-6;
};

}  // namespace mipx
