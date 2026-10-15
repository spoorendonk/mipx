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
    /// Returned when the last solve() proved the LP infeasible; empty after any
    /// other status, and from backends that do not produce one (the default).
    /// Also empty when the dual simplex reports Infeasible only as a status
    /// carried over from an earlier solve (issue #222).
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

    /// Certificate of unboundedness (a primal ray), one entry per column, in
    /// the space of the loaded problem (unscaled, original column order).
    ///
    /// Returned when the last solve() proved the LP unbounded; empty after any
    /// other status, after any change to the model since that solve (rows,
    /// column bounds or objective; the status still reads Unbounded), and
    /// from backends that do not produce one (the default; only the dual
    /// simplex does). It certifies unboundedness only together with a
    /// feasible point, which the Unbounded status does not provide.
    ///
    /// Contract, for the ray r and q = A r:
    ///   - c^T r < 0 for minimize, c^T r > 0 for maximize;
    ///   - r_j > 0 only where col_upper_j is infinite, r_j < 0 only where
    ///     col_lower_j is infinite;
    ///   - q_i > 0 only where row_upper_i is infinite, q_i < 0 only where
    ///     row_lower_i is infinite;
    /// so x + t r stays feasible for every t >= 0 from any feasible x, and the
    /// objective improves without bound. The ray's length is arbitrary.
    ///
    /// Floating point: the column sign conditions hold exactly, but q and the
    /// sign of c^T r only up to the solver's tolerances -- treat |q_i| below
    /// a tolerance (relative to |r| times the row's coefficients) as zero
    /// before using an infinite row side. The dual simplex drops entries its
    /// ratio test skipped; their size is bounded relative to the whole
    /// direction in its scaled space, slacks included, so on badly scaled
    /// models q_i can exceed that tolerance. A consumer that needs a rigorous
    /// proof recomputes q and c^T r itself.
    virtual std::vector<Real> getPrimalRay() const { return {}; }

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
