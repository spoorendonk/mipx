#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>
#include <span>
#include <unordered_map>
#include <vector>

#include "mipx/bnb_node.h"
#include "mipx/branching.h"
#include "mipx/core.h"
#include "mipx/cut_manager.h"
#include "mipx/cut_pool.h"
#include "mipx/domain.h"
#include "mipx/dual_simplex.h"
#include "mipx/heuristic_runtime.h"
#include "mipx/logger.h"
#include "mipx/lp_problem.h"
#include "mipx/presolve.h"
#include "mipx/separators.h"

namespace mipx {

enum class RootLpPolicy {
    DualDefault,
    BarrierRoot,
    PdlpRoot,
    ConcurrentRootExperimental,
};

enum class SearchProfile {
    Stable,
    Default,
    Aggressive,
};

struct MipLpStats {
    RootLpPolicy root_policy = RootLpPolicy::DualDefault;
    double root_lp_seconds = 0.0;
    double root_race_winner_seconds = 0.0;
    double root_cut_lp_seconds = 0.0;
    double node_bound_apply_seconds = 0.0;
    double node_basis_set_seconds = 0.0;
    double node_lp_solve_seconds = 0.0;
    Int nodes_solved = 0;
    Int warm_starts = 0;
    Int cold_starts = 0;
    Int root_race_runs = 0;
    Int root_race_candidates = 0;
    Int root_race_cancelled = 0;
    Int root_race_dual_wins = 0;
    Int root_race_barrier_wins = 0;
    Int root_race_pdlp_wins = 0;
};

struct MipCutStats {
    Int root_rounds = 0;
    Int root_cuts_added = 0;
    Int tree_nodes_with_cuts = 0;
    Int tree_nodes_skipped = 0;
    Int tree_rounds = 0;
    Int tree_cuts_local = 0;
    Int tree_cuts_global = 0;
    Int tree_cuts_purged = 0;
    Int tree_cuts_revived = 0;
    Real tree_lp_delta = 0.0;
};

struct MipConflictStats {
    Int learned = 0;
    Int reused = 0;
    Int pruned = 0;
    Int purged = 0;
    Int minimized_literals = 0;
    Int lp_infeasible_conflicts = 0;
    Int bound_infeasible_conflicts = 0;
    Int branch_score_overrides = 0;
};

struct MipPreRootStats {
    bool enabled = false;
    bool lp_light_enabled = false;
    bool lp_light_available = false;
    bool portfolio_enabled = false;
    Int rounds = 0;
    Int calls = 0;
    Int improvements = 0;
    Int feasible_found = 0;
    Int early_stops = 0;
    Int portfolio_epochs = 0;
    Int portfolio_wins = 0;
    Int portfolio_stagnant = 0;
    Int fj_calls = 0;
    Int fpr_calls = 0;
    Int local_mip_calls = 0;
    Int lp_light_calls = 0;
    Int lp_light_fpr_calls = 0;
    Int lp_light_diving_calls = 0;
    Int fj_improvements = 0;
    Int fpr_improvements = 0;
    Int local_mip_improvements = 0;
    Int lp_light_fpr_improvements = 0;
    Int lp_light_diving_improvements = 0;
    Int lp_light_lp_solves = 0;
    Int lp_light_lp_iterations = 0;
    double work_units = 0.0;
    double lp_light_lp_work = 0.0;
    double fj_reward = 0.0;
    double fpr_reward = 0.0;
    double local_mip_reward = 0.0;
    double lp_light_fpr_reward = 0.0;
    double lp_light_diving_reward = 0.0;
    double effort_scale_final = 1.0;
    double time_seconds = 0.0;
    double time_to_first_feasible = kInf;
    Real incumbent_at_root = kInf;
};

struct MipSearchStats {
    Int policy_switches = 0;
    Int restarts = 0;
    Int restart_nodes_dropped = 0;
    Int sibling_cache_hits = 0;
    Int sibling_cache_misses = 0;
    Int strong_budget_updates = 0;
};

struct MipTreePresolveStats {
    Int attempts = 0;
    Int runs = 0;
    Int skipped = 0;
    Int infeasible = 0;
    Int activity_tightenings = 0;
    Int reduced_cost_tightenings = 0;
    Int lp_resolves = 0;
    Real lp_delta = 0.0;
};

struct MipResult {
    Status status = Status::Error;
    Real objective = 0.0;
    Real best_bound = -kInf;
    Real gap = kInf;
    Int nodes = 0;
    Int lp_iterations = 0;
    double work_units = 0.0;   // deterministic work measure
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
    void setVerbose(bool v) { verbose_ = v; log_.setEnabled(v); }
    void setPresolve(bool p) { presolve_ = p; }
    void setMaxCutRounds(Int r) { max_cut_rounds_ = r; }
    void setMaxCutsPerRound(Int c) { max_cuts_per_round_ = c; }
    void setCutsEnabled(bool e) { cuts_enabled_ = e; }
    void setCutFamilyEnabled(CutFamily family, bool enabled) {
        switch (family) {
            case CutFamily::Gomory: cut_family_config_.gomory = enabled; break;
            case CutFamily::Mir: cut_family_config_.mir = enabled; break;
            case CutFamily::Cover: cut_family_config_.cover = enabled; break;
            case CutFamily::ImpliedBound: cut_family_config_.implied_bound = enabled; break;
            case CutFamily::Clique: cut_family_config_.clique = enabled; break;
            case CutFamily::ZeroHalf: cut_family_config_.zero_half = enabled; break;
            case CutFamily::Mixing: cut_family_config_.mixing = enabled; break;
            case CutFamily::Unknown:
            case CutFamily::Count:
            default: break;
        }
    }
    void setCutFamilyConfig(const CutFamilyConfig& config) { cut_family_config_ = config; }
    [[nodiscard]] const CutFamilyConfig& getCutFamilyConfig() const { return cut_family_config_; }
    void setCutEffortMode(CutEffortMode mode) { cut_effort_mode_ = mode; }
    [[nodiscard]] CutEffortMode getCutEffortMode() const { return cut_effort_mode_; }
    void setCutWorkBudgets(double per_node, double per_round, double global) {
        cut_per_node_work_budget_ = std::max(1.0, per_node);
        cut_per_round_work_budget_ = std::max(1.0, per_round);
        cut_global_work_budget_ = std::max(1.0, global);
    }
    void setNumThreads(Int n) { num_threads_ = n; }
    void setRootLpPolicy(RootLpPolicy policy) { root_lp_policy_ = policy; }
    void setBarrierUseGpu(bool use_gpu) { barrier_use_gpu_ = use_gpu; }
    void setBarrierGpuThresholds(Int min_rows, Int min_nnz) {
        barrier_gpu_min_rows_ = std::max<Int>(0, min_rows);
        barrier_gpu_min_nnz_ = std::max<Int>(0, min_nnz);
    }
    void setPdlpUseGpu(bool use_gpu) { pdlp_use_gpu_ = use_gpu; }
    void setPdlpGpuThresholds(Int min_rows, Int min_nnz) {
        pdlp_gpu_min_rows_ = std::max<Int>(0, min_rows);
        pdlp_gpu_min_nnz_ = std::max<Int>(0, min_nnz);
    }
    void setHeuristicMode(HeuristicRuntimeMode mode) { heuristic_mode_ = mode; }
    void setHeuristicSeed(uint64_t seed) { heuristic_seed_ = seed; }
    void setPreRootLpFreeEnabled(bool enabled) { pre_root_lp_free_enabled_ = enabled; }
    void setPreRootLpFreeWorkBudget(double max_work_units) {
        pre_root_lp_free_work_budget_ = std::max(1.0, max_work_units);
    }
    void setPreRootLpFreeMaxRounds(Int rounds) {
        pre_root_lp_free_max_rounds_ = std::max<Int>(1, rounds);
    }
    void setPreRootLpFreeEarlyStop(bool enabled) { pre_root_lp_free_early_stop_ = enabled; }
    void setPreRootLpLightEnabled(bool enabled) { pre_root_lp_light_enabled_ = enabled; }
    void setPreRootPortfolioEnabled(bool enabled) { pre_root_portfolio_enabled_ = enabled; }
    void setConflictsEnabled(bool enabled) { conflicts_enabled_ = enabled; }
    void setSearchProfile(SearchProfile profile) { search_profile_ = profile; }
    void setRestartsEnabled(bool enabled) { restarts_enabled_ = enabled; }
    void setRestartControls(Int stagnation_nodes, Int keep_nodes) {
        restart_stagnation_nodes_ = std::max<Int>(8, stagnation_nodes);
        restart_keep_nodes_ = std::max<Int>(2, keep_nodes);
    }
    void setTreePresolveEnabled(bool enabled) { tree_presolve_enabled_ = enabled; }
    void setTreePresolveControls(Int max_depth, Int min_frac, Int depth_frequency) {
        tree_presolve_max_depth_ = std::max<Int>(1, max_depth);
        tree_presolve_min_frac_ = std::max<Int>(1, min_frac);
        tree_presolve_depth_frequency_ = std::max<Int>(1, depth_frequency);
    }
    const MipLpStats& getLpStats() const { return lp_stats_; }
    const MipCutStats& getCutStats() const { return cut_stats_; }
    const MipConflictStats& getConflictStats() const { return conflict_stats_; }
    const MipPreRootStats& getPreRootStats() const { return pre_root_stats_; }
    [[nodiscard]] bool hasLpLightCapability() const;
    const MipSearchStats& getSearchStats() const { return search_stats_; }
    const MipTreePresolveStats& getTreePresolveStats() const { return tree_presolve_stats_; }
    const BranchingTelemetry& getBranchingStats() const { return branching_stats_; }

private:
    struct NodeWorkStats {
        double bound_apply_seconds = 0.0;
        double basis_set_seconds = 0.0;
        double lp_solve_seconds = 0.0;
        Int nodes_solved = 0;
        Int warm_starts = 0;
        Int cold_starts = 0;
    };

    /// Run cutting plane rounds at the root node.
    Int runCuttingPlanes(DualSimplexSolver& lp, Int& total_lp_iters, double& total_work);

    /// Serial branch-and-bound loop.
    void solveSerial(DualSimplexSolver& lp, NodeQueue& queue,
                     Int& nodes_explored, Int& total_lp_iters,
                     double& total_work,
                     HeuristicRuntime& heuristic_runtime,
                     SolutionPool& solution_pool,
                     Real& incumbent, std::vector<Real>& best_solution,
                     Real root_bound,
                     const std::function<double()>& elapsed);

    /// Parallel branch-and-bound loop using TBB.
    void solveParallel(const DualSimplexSolver& root_lp, NodeQueue& queue,
                       Int& nodes_explored, Int& total_lp_iters,
                       double& total_work,
                       HeuristicRuntime& heuristic_runtime,
                       SolutionPool& solution_pool,
                       Real& incumbent, std::vector<Real>& best_solution,
                       Real root_bound,
                       const std::function<double()>& elapsed);

    // Check if all integer variables are integral in the given solution.
    bool isFeasibleMip(const std::vector<Real>& primals) const;
    bool isFeasibleLp(const std::vector<Real>& primals) const;

    // Compute optimality gap.
    Real computeGap(Real incumbent, Real best_bound) const;

    // Log a progress line.
    void logProgress(Int nodes, Int open, Int lp_iters,
                     Real incumbent, Real best_bound, double elapsed,
                     bool new_incumbent = false, Int int_inf = -1) const;

    /// Process a single node. Returns true if children were created.
    bool processNode(DualSimplexSolver& lp, BnbNode& node,
                     Real incumbent_snapshot,
                     std::vector<BnbNode>& children_out,
                     Real& node_obj_out,
                     std::vector<Real>& node_primals_out,
                     Int& node_iters_out,
                     double& node_work_out,
                     std::vector<Real>& current_lower,
                     std::vector<Real>& current_upper,
                     std::vector<Index>& touched_vars,
                     NodeWorkStats& node_stats,
                     Int& int_inf_out);
    void ageConflictPool();
    void learnConflictFromNode(const std::vector<BranchDecision>& bound_changes,
                               bool lp_infeasible);
    bool isConflictTriggered(std::span<const Index> vars,
                             std::span<const Real> node_lb,
                             std::span<const Real> node_ub);
    Index selectConflictAwareBranchVariable(std::span<const Real> primals,
                                            std::span<const Real> current_lower,
                                            std::span<const Real> current_upper,
                                            Index default_var);
    HeuristicRuntimeConfig makeHeuristicRuntimeConfig() const;

    struct ConflictLiteral {
        Index variable = -1;
        Real bound = 0.0;
        bool is_upper = false;
    };

    struct ConflictClause {
        std::vector<ConflictLiteral> literals;
        Int age = 0;
        Int hits = 0;
    };

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
    CutFamilyConfig cut_family_config_{};
    CutEffortMode cut_effort_mode_ = CutEffortMode::Auto;
    double cut_per_node_work_budget_ = 2.5e5;
    double cut_per_round_work_budget_ = 5.0e4;
    double cut_global_work_budget_ = 1.0e6;
    RootLpPolicy root_lp_policy_ = RootLpPolicy::DualDefault;
    bool barrier_use_gpu_ = true;
    Int barrier_gpu_min_rows_ = 512;
    Int barrier_gpu_min_nnz_ = 10000;
    bool pdlp_use_gpu_ = true;
    Int pdlp_gpu_min_rows_ = 512;
    Int pdlp_gpu_min_nnz_ = 10000;
    HeuristicRuntimeMode heuristic_mode_ = HeuristicRuntimeMode::Deterministic;
    uint64_t heuristic_seed_ = 1;
    bool pre_root_lp_free_enabled_ = false;
    bool pre_root_lp_free_early_stop_ = true;
    Int pre_root_lp_free_max_rounds_ = 24;
    double pre_root_lp_free_work_budget_ = 5.0e4;
    bool pre_root_lp_light_enabled_ = false;
    bool pre_root_portfolio_enabled_ = true;
    bool conflicts_enabled_ = true;
    Int conflict_max_pool_size_ = 512;
    Int conflict_max_age_ = 64;
    SearchProfile search_profile_ = SearchProfile::Default;
    bool restarts_enabled_ = false;
    Int restart_stagnation_nodes_ = 96;
    Int restart_keep_nodes_ = 32;
    bool tree_presolve_enabled_ = true;
    Int tree_presolve_max_depth_ = 24;
    Int tree_presolve_min_frac_ = 4;
    Int tree_presolve_depth_frequency_ = 3;
    MipLpStats lp_stats_{};
    MipCutStats cut_stats_{};
    MipConflictStats conflict_stats_{};
    MipPreRootStats pre_root_stats_{};
    MipSearchStats search_stats_{};
    MipTreePresolveStats tree_presolve_stats_{};
    std::vector<ConflictClause> conflict_pool_{};
    std::vector<Real> conflict_scores_{};
    std::unordered_map<Int, Index> sibling_branch_cache_{};
    ReliabilityBranching branching_rule_;
    BranchingTelemetry branching_stats_{};
    std::mutex branching_mutex_;
    mutable Logger log_;

    static constexpr Real kIntTol = 1e-6;
    static constexpr Real kCutImprovementTol = 1e-6;
    static constexpr double kLogInterval = 5.0;  // seconds between heartbeat lines
    static constexpr Int kRinsNodeFrequency = 64;
    static constexpr Int kRinsSubproblemIterLimit = 64;
    static constexpr Real kRinsAgreementTol = 1e-4;
    static constexpr Int kRinsMaxIntInfForRun = 24;
    static constexpr Int kRinsMinFixedVars = 12;
    static constexpr Real kRinsMinFixedRate = 0.08;
    static constexpr Real kRinsMaxRelativeGapForRun = 0.10;
    static constexpr Int kRootHeuristicMaxIntInf = 12;
    static constexpr Int kRootHeuristicMaxIntVars = 96;
    static constexpr Int kRootFeasPumpMaxIter = 3;
    static constexpr Int kRootFeasPumpSubproblemIterLimit = 20;
    static constexpr Int kRootAuxObjSubproblemIterLimit = 30;
    static constexpr Int kRootAuxObjMinActiveIntegerVars = 1;
    static constexpr Int kRootZeroObjSubproblemIterLimit = 20;
    static constexpr Int kRootRensSubproblemIterLimit = 40;
    static constexpr Int kRootRensMinFixedVars = 16;
    static constexpr Real kRootRensMinFixedRate = 0.25;
    static constexpr Int kRootLocalBranchingSubproblemIterLimit = 40;
    static constexpr Int kRootLocalBranchingNeighborhoodSmall = 8;
    static constexpr Int kRootLocalBranchingNeighborhoodMedium = 16;
    static constexpr Int kRootLocalBranchingNeighborhoodLarge = 24;
    static constexpr Int kRootLocalBranchingMinBinaryVars = 8;
    static constexpr Real kHeurBudgetMaxWorkShare = 0.20;
    static constexpr Int kHeurBudgetMaxFrequencyScale = 8;
};

}  // namespace mipx
