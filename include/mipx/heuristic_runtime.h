#pragma once

#include <cstdint>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "mipx/core.h"
#include "mipx/heuristics.h"
#include "mipx/lp_problem.h"

namespace mipx {

enum class HeuristicRuntimeMode {
    Deterministic,
    Opportunistic,
};

enum class RestartStrategy {
    Uniform,
    TournamentCrossover,
    Perturbation,
    PolarityGuided,
    ActivityGuided,
    DistanceGuided,
};

struct HeuristicRuntimeStats {
    Int calls = 0;
    Int improvements = 0;
    Int lp_iterations = 0;
    double work_units = 0.0;
};

class HeuristicCallback {
public:
    virtual ~HeuristicCallback() = default;

    virtual void onHeader(const char*, HeuristicRuntimeMode) {}
    virtual void onIncumbent(const char*, Real, Int, Int) {}
    virtual void onHeartbeat(Int, const HeuristicRuntimeStats&) {}
    virtual void onFinish(const HeuristicRuntimeStats&) {}
};

class SolutionPool {
public:
    explicit SolutionPool(Sense sense = Sense::Minimize) : sense_(sense) {}

    bool submit(HeuristicSolution solution,
                const char* source = "",
                Int thread_id = -1);

    [[nodiscard]] std::optional<HeuristicSolution> bestSolution() const;
    [[nodiscard]] bool hasSolution() const;
    [[nodiscard]] Real bestObjective() const;
    [[nodiscard]] uint64_t updateCount() const;

private:
    struct Entry {
        HeuristicSolution solution;
        std::string source;
        Int thread_id = -1;
    };

    static bool better(Sense sense, Real lhs, Real rhs);

    Sense sense_ = Sense::Minimize;
    mutable std::mutex mutex_;
    std::optional<Entry> best_;
    uint64_t update_count_ = 0;
};

class RestartStrategyEngine {
public:
    explicit RestartStrategyEngine(uint64_t seed = 1) : seed_(seed) {}

    void setSeed(uint64_t seed) { seed_ = seed; }

    [[nodiscard]] RestartStrategy choose(HeuristicRuntimeMode mode,
                                         Int epoch,
                                         Int stagnation_epochs) const;
    [[nodiscard]] static const char* name(RestartStrategy s);

private:
    static uint64_t splitmix64(uint64_t x);

    uint64_t seed_ = 1;
};

struct HeuristicRuntimeConfig {
    HeuristicRuntimeMode mode = HeuristicRuntimeMode::Deterministic;
    uint64_t seed = 1;

    Int rins_node_frequency = 64;
    Int rins_subproblem_iter_limit = 64;
    Real rins_agreement_tol = 1e-4;
    Int rins_max_int_inf_for_run = 24;
    Int rins_min_fixed_vars = 12;
    Real rins_min_fixed_rate = 0.08;
    Real rins_max_relative_gap_for_run = 0.10;

    Int root_max_int_inf = 12;
    Int root_max_int_vars = 96;
    Int root_feaspump_max_iter = 3;
    Int root_feaspump_subproblem_iter_limit = 20;
    Int root_auxobj_subproblem_iter_limit = 30;
    Int root_auxobj_min_active_integer_vars = 1;
    Int root_zeroobj_subproblem_iter_limit = 20;
    Int root_rens_subproblem_iter_limit = 40;
    Int root_rens_min_fixed_vars = 16;
    Real root_rens_min_fixed_rate = 0.25;
    Int root_local_branching_subproblem_iter_limit = 40;
    Int root_local_branching_neighborhood_small = 8;
    Int root_local_branching_neighborhood_medium = 16;
    Int root_local_branching_neighborhood_large = 24;
    Int root_local_branching_min_binary_vars = 8;

    Real budget_max_work_share = 0.20;
    Int budget_max_frequency_scale = 8;
};

struct RootHeuristicContext {
    const LpProblem& problem;
    DualSimplexSolver& lp;
    std::span<const Real> primals;
    Int root_int_inf = 0;
    Int root_int_vars = 0;
    Int node_count = 0;
    Int thread_id = -1;
    double total_work_units = 0.0;
    SolutionPool* solution_pool = nullptr;
};

struct RootHeuristicOutcome {
    bool basis_dirty = false;
    Int calls = 0;
    Int improvements = 0;
    Int lp_iterations = 0;
    double work_units = 0.0;
    Int rounding_calls = 0;
    Int rounding_improvements = 0;
    Int auxobj_calls = 0;
    Int auxobj_improvements = 0;
    Int zeroobj_calls = 0;
    Int zeroobj_improvements = 0;
    Int feaspump_calls = 0;
    Int feaspump_improvements = 0;
    Int rens_calls = 0;
    Int rens_improvements = 0;
    Int rins_calls = 0;
    Int rins_improvements = 0;
    Int localbranching_calls = 0;
    Int localbranching_improvements = 0;
};

struct WorkerHeuristicContext {
    const LpProblem& problem;
    DualSimplexSolver& lp;
    std::span<const Real> primals;
    Int node_count = 0;
    Int int_inf = -1;
    Real node_objective = 0.0;
    Real incumbent = kInf;
    std::span<const Real> incumbent_values;
    Int thread_id = 0;
    double total_work_units = 0.0;
};

struct WorkerHeuristicOutcome {
    bool attempted = false;
    bool improved = false;
    bool executed_solve = false;
    bool skipped_no_incumbent = false;
    bool skipped_few_fixes = false;
    Int fixed_count = 0;
    Int lp_iterations = 0;
    double work_units = 0.0;
    RestartStrategy restart_strategy = RestartStrategy::Uniform;
    std::optional<HeuristicSolution> solution;
};

class HeuristicRuntime {
public:
    explicit HeuristicRuntime(const HeuristicRuntimeConfig& config = {});

    void setCallback(HeuristicCallback* callback) { callback_ = callback; }
    void resetForSolve();
    void finish();

    [[nodiscard]] HeuristicRuntimeMode mode() const { return config_.mode; }
    [[nodiscard]] const HeuristicRuntimeStats& stats() const { return stats_; }

    RootHeuristicOutcome runRootPortfolio(const RootHeuristicContext& ctx,
                                          Real& incumbent,
                                          std::vector<Real>& best_solution);

    WorkerHeuristicOutcome runTreeWorker(const WorkerHeuristicContext& ctx);

private:
    static double canonicalWorkUnits(double work_units);
    static bool hasIncumbent(Real incumbent);
    static Real relativeGap(Real incumbent, Real bound);
    static bool isImprovement(Sense sense, Real candidate, Real incumbent);

    bool allowRootCall(double total_work_units) const;
    bool allowTreeCall(Int node_count, double total_work_units) const;
    void recordCall(Int node_count, double work_units, bool improved);

    HeuristicRuntimeConfig config_;
    HeuristicBudgetManager budget_;
    RestartStrategyEngine restart_engine_;
    HeuristicRuntimeStats stats_;
    HeuristicCallback* callback_ = nullptr;
    mutable std::mutex state_mutex_;
    Int stagnation_epochs_ = 0;
    bool emitted_root_header_ = false;
    bool emitted_worker_header_ = false;
};

}  // namespace mipx
