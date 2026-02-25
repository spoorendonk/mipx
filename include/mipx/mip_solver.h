#pragma once

#include <chrono>
#include <vector>

#include "mipx/bnb_node.h"
#include "mipx/branching.h"
#include "mipx/core.h"
#include "mipx/domain.h"
#include "mipx/dual_simplex.h"
#include "mipx/lp_problem.h"

namespace mipx {

struct MipResult {
    Status status = Status::Error;
    Real objective = 0.0;
    Real best_bound = -kInf;
    Real gap = kInf;
    Int nodes = 0;
    Int lp_iterations = 0;
    double time_seconds = 0.0;
    std::vector<Real> solution;
};

class MipSolver {
public:
    MipSolver() = default;

    void load(const LpProblem& problem);
    MipResult solve();

    // Parameter setters.
    void setNodeLimit(Int limit) { node_limit_ = limit; }
    void setTimeLimit(double seconds) { time_limit_ = seconds; }
    void setGapTolerance(Real tol) { gap_tol_ = tol; }
    void setVerbose(bool v) { verbose_ = v; }

private:
    // Check if all integer variables are integral in the given solution.
    bool isFeasibleMip(const std::vector<Real>& primals) const;

    // Compute optimality gap.
    Real computeGap(Real incumbent, Real best_bound) const;

    // Log a progress line.
    void logProgress(Int nodes, Int open, Int lp_iters,
                     Real incumbent, Real best_bound, double elapsed) const;

    // Problem data.
    LpProblem problem_;
    bool loaded_ = false;

    // Parameters.
    Int node_limit_ = 1000000;
    double time_limit_ = 3600.0;
    Real gap_tol_ = 1e-4;
    bool verbose_ = true;

    static constexpr Real kIntTol = 1e-6;
    static constexpr Int kLogFrequency = 100;
};

}  // namespace mipx
