#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <mutex>
#include <vector>

#include "mipx/bnb_node.h"
#include "mipx/branching.h"
#include "mipx/core.h"
#include "mipx/cut_pool.h"
#include "mipx/domain.h"
#include "mipx/dual_simplex.h"
#include "mipx/gomory.h"
#include "mipx/lp_problem.h"
#include "mipx/presolve.h"

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
    void setPresolve(bool p) { presolve_ = p; }
    void setMaxCutRounds(Int r) { max_cut_rounds_ = r; }
    void setMaxCutsPerRound(Int c) { max_cuts_per_round_ = c; }
    void setCutsEnabled(bool e) { cuts_enabled_ = e; }
    void setNumThreads(Int n) { num_threads_ = n; }

private:
    /// Run cutting plane rounds at the root node.
    Int runCuttingPlanes(DualSimplexSolver& lp, Int& total_lp_iters);

    /// Serial branch-and-bound loop.
    void solveSerial(DualSimplexSolver& lp, NodeQueue& queue,
                     Int& nodes_explored, Int& total_lp_iters,
                     Real& incumbent, std::vector<Real>& best_solution,
                     Real root_bound,
                     const std::function<double()>& elapsed);

    /// Parallel branch-and-bound loop using TBB.
    void solveParallel(const DualSimplexSolver& root_lp, NodeQueue& queue,
                       Int& nodes_explored, Int& total_lp_iters,
                       Real& incumbent, std::vector<Real>& best_solution,
                       Real root_bound,
                       const std::function<double()>& elapsed);

    // Check if all integer variables are integral in the given solution.
    bool isFeasibleMip(const std::vector<Real>& primals) const;

    // Compute optimality gap.
    Real computeGap(Real incumbent, Real best_bound) const;

    // Log a progress line.
    void logProgress(Int nodes, Int open, Int lp_iters,
                     Real incumbent, Real best_bound, double elapsed) const;

    /// Process a single node. Returns true if children were created.
    bool processNode(DualSimplexSolver& lp, BnbNode& node,
                     Real incumbent_snapshot,
                     std::vector<BnbNode>& children_out,
                     Real& node_obj_out,
                     std::vector<Real>& node_primals_out,
                     Int& node_iters_out);

    // Problem data.
    LpProblem problem_;
    bool loaded_ = false;

    // Parameters.
    Int node_limit_ = 1000000;
    double time_limit_ = 3600.0;
    Real gap_tol_ = 1e-4;
    bool verbose_ = true;
    bool presolve_ = true;
    Int num_threads_ = 1;

    // Cutting plane parameters.
    Int max_cut_rounds_ = 20;
    Int max_cuts_per_round_ = 50;
    bool cuts_enabled_ = true;

    static constexpr Real kIntTol = 1e-6;
    static constexpr Real kCutImprovementTol = 1e-6;
    static constexpr Int kLogFrequency = 100;
};

}  // namespace mipx
