#pragma once

#include <span>
#include <vector>

#include "mipx/core.h"
#include "mipx/lp_problem.h"

namespace mipx {

struct LpResult {
    Status status = Status::Error;
    Real objective = 0.0;
    Int iterations = 0;
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

    /// Basis operations.
    virtual std::vector<BasisStatus> getBasis() const = 0;
    virtual void setBasis(std::span<const BasisStatus> basis) = 0;

    /// Incremental modifications (for branch-and-cut).
    virtual void addRows(std::span<const Index> starts,
                         std::span<const Index> indices,
                         std::span<const Real> values,
                         std::span<const Real> lower,
                         std::span<const Real> upper) = 0;
    virtual void removeRows(std::span<const Index> rows) = 0;

    /// Bound and objective changes.
    virtual void setColBounds(Index col, Real lower, Real upper) = 0;
    virtual void setObjective(std::span<const Real> obj) = 0;
};

}  // namespace mipx
