#pragma once

#include "mipx/lp_solver.h"

#include <optional>
#include <span>
#include <vector>

namespace mipx {

class Deadline;
class DualSimplexSolver;

struct CrossoverBound {
    BasisStatus status;
    Real distance;
};

CrossoverBound nearestCrossoverBound(Real value, Real lower, Real upper);

// Construct a full-rank simplex start in the original model's space from an
// interior solution. The basis still needs simplex cleanup to prove optimality.
std::optional<std::vector<BasisStatus>> crossoverBasis(const LpProblem& lp,
                                                       std::span<const Real> primal,
                                                       std::span<const Real> reduced_costs,
                                                       const Deadline& deadline);

// Clean up the candidate in an already loaded simplex solver, leaving the
// factorization and solution available to callers that need to reoptimize.
LpResult solveCrossoverBasis(DualSimplexSolver& simplex, std::span<const BasisStatus> basis);

}  // namespace mipx
