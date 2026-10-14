#pragma once

#include "mipx/core.h"
#include "mipx/lp_problem.h"

#include <span>
#include <vector>

namespace mipx {

struct LpResult {
    Status status = Status::Error;
    Real objective = 0.0;
    Int iterations = 0;
    double work_units = 0.0;  // deterministic work measure
};

enum class BasisStatus {
    Basic,
    AtLower,
    AtUpper,
    Fixed,
    Free,
};

class LpSolver {
public:
    virtual ~LpSolver() = default;

    /// Load a problem.
    virtual void load(const LpProblem& problem) = 0;

    /// Solve the LP relaxation.
    virtual LpResult solve() = 0;

    /// Get solution status.
    virtual Status getStatus() const = 0;
    virtual Real getObjective() const = 0;

    /// Get solution vectors.
    virtual std::vector<Real> getPrimalValues() const = 0;
    virtual std::vector<Real> getDualValues() const = 0;
    virtual std::vector<Real> getReducedCosts() const = 0;

    /// Farkas certificate of primal infeasibility (a dual ray), one entry per
    /// row, in the space of the loaded problem (unscaled, original row order).
    ///
    /// Returned when the last solve() ended Status::Infeasible; empty after any
    /// other status, and from backends that do not produce one (the default).
    ///
    /// Contract, for the ray y and d = A^T y:
    ///   - y_i > 0 only where row_upper_i is finite, y_i < 0 only where
    ///     row_lower_i is finite (so y_i >= 0 on a <= row, y_i <= 0 on a >= row,
    ///     either sign on an equality or ranged row);
    ///   - every feasible x satisfies y^T A x <= beta, with
    ///       beta = sum_i (y_i > 0 ? y_i * row_upper_i : y_i * row_lower_i);
    ///   - over the column bounds, d^T x is at least
    ///       alpha = sum_j (d_j > 0 ? d_j * col_lower_j : d_j * col_upper_j),
    ///     and alpha > beta, so no x within its bounds meets the rows.
    /// For x >= 0 this is the textbook form: the sign conditions plus
    /// beta < 0. The ray's length is arbitrary, and it does not depend on the
    /// objective or its sense.
    ///
    /// Floating point: the row sign conditions hold exactly, but d is only
    /// right up to the solver's tolerances -- treat |d_j| below a tolerance
    /// (relative to |y|) as zero before using an infinite column bound. The
    /// dual simplex may prove infeasibility against bounds its internal
    /// perturbation has tightened by a small epsilon, so alpha - beta can fall
    /// short against the original bounds by about that much. A consumer that
    /// needs a rigorous proof recomputes alpha and beta itself.
    virtual std::vector<Real> getDualRay() const { return {}; }

    /// Basis operations.
    virtual std::vector<BasisStatus> getBasis() const = 0;
    virtual void setBasis(std::span<const BasisStatus> basis) = 0;

    /// Incremental modifications (for branch-and-cut).
    virtual void addRows(std::span<const Index> starts, std::span<const Index> indices,
                         std::span<const Real> values, std::span<const Real> lower,
                         std::span<const Real> upper) = 0;
    virtual void removeRows(std::span<const Index> rows) = 0;

    /// Bound and objective changes.
    virtual void setColBounds(Index col, Real lower, Real upper) = 0;
    virtual void setObjective(std::span<const Real> obj) = 0;
};

}  // namespace mipx
