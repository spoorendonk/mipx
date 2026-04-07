#pragma once

#include <vector>

#include "mipx/core.h"

namespace mipx {

// Forward declarations.
class SchreierSims;

/// Result of orbital fixing at a node.
struct OrbitalFixingResult {
    /// Variables whose bounds were tightened.
    std::vector<Index> tightened_vars;
    /// Number of fixings applied.
    Index num_fixings = 0;
    /// Whether infeasibility was detected.
    bool infeasible = false;
    /// Work units consumed.
    double work_units = 0.0;
};

/// Orbital fixing exploits symmetry at tree nodes.
///
/// When a variable x_j is fixed to a value v in a branch-and-bound node,
/// all variables in the orbit of x_j (under the symmetry group stabilizing
/// already-fixed variables) can also be fixed to v or have their bounds
/// tightened accordingly.
///
/// For binary variables fixed to 0 or 1:
///   - If x_j = 1, all variables in orbit(j) under Stab(fixed) get lb >= 1.
///   - If x_j = 0, all variables in orbit(j) under Stab(fixed) get ub <= 0.
///
/// For general integer variables, we tighten bounds:
///   - lb(orbit member) >= lb(j), ub(orbit member) <= ub(j)
class OrbitalFixer {
public:
    OrbitalFixer() = default;

    /// Set the Schreier-Sims structure for orbit computation.
    void setSchreierSims(const SchreierSims* ss) { schreier_sims_ = ss; }

    /// Apply orbital fixing at a node.
    /// @param col_lower    Current lower bounds (may be modified).
    /// @param col_upper    Current upper bounds (may be modified).
    /// @param col_type     Variable types.
    /// @param fixed_vars   Variables fixed at this node (bound equal).
    /// @param num_cols     Number of columns.
    /// @return Result describing tightenings and infeasibility.
    OrbitalFixingResult fix(std::vector<Real>& col_lower,
                            std::vector<Real>& col_upper,
                            const std::vector<VarType>& col_type,
                            const std::vector<Index>& fixed_vars,
                            Index num_cols) const;

    /// Lightweight orbital fixing using precomputed orbit representatives
    /// (for use when Schreier-Sims is not available or too expensive).
    /// This uses the simple canonical[] mapping from SymmetryManager.
    static OrbitalFixingResult fixFromCanonical(
        std::vector<Real>& col_lower,
        std::vector<Real>& col_upper,
        const std::vector<Index>& canonical,
        Index num_cols);

private:
    const SchreierSims* schreier_sims_ = nullptr;
};

}  // namespace mipx
