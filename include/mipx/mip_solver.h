#pragma once

#include "mipx/barrier.h"
#include "mipx/bnb_node.h"
#include "mipx/branching.h"
#include "mipx/clique_table.h"
#include "mipx/conflict_graph.h"
#include "mipx/core.h"
#include "mipx/cut_manager.h"
#include "mipx/cut_pool.h"
#include "mipx/domain.h"
#include "mipx/dual_simplex.h"
#include "mipx/exact_refinement.h"
#include "mipx/heuristic_runtime.h"
#include "mipx/implication_graph.h"
#include "mipx/logger.h"
#include "mipx/lp_problem.h"
#include "mipx/objective_integrality.h"
#include "mipx/objective_propagator.h"
#include "mipx/presolve.h"
#include "mipx/probing.h"
#include "mipx/reduced_cost_fixer.h"
#include "mipx/separators.h"
#include "mipx/symmetry.h"
#include "mipx/variable_bounds.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

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

enum class ParallelMode {
    Deterministic,
    Opportunistic,
};

enum class ExactRefinementMode {
    Off,
    Auto,
    On,
};

struct Sos2PropagationStats {
    /// Columns fixed to zero. A column already fixed at the node counts zero times.
    uint64_t fixings = 0;
    /// Member bounds inspected while finding forced support or applying its fixings.
    uint64_t member_visits = 0;
    /// LPs skipped because forced support spanned more than one adjacent window.
    Int prunes = 0;
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
    bool root_pdlp_crossover = false;  // the tree reused the PDLP crossover solve
    /// Node LPs that ended without a verdict (neither Optimal nor
    /// Infeasible: Error, an iteration limit, or Unbounded) and were
    /// re-solved cold. One the MIP's time limit stopped is not (#351).
    Int node_lp_no_verdict = 0;
    /// Of those, the ones the cold re-solve gave a verdict.
    Int node_lp_no_verdict_recovered = 0;
    /// Nodes left unexplored because no re-solve gave a verdict (#283).
    Int nodes_dropped_no_verdict = 0;
    /// Nodes left unexplored because the time limit stopped their LP
    /// (#351): the tree stopped on the limit, whatever the queue holds.
    Int nodes_dropped_time_limit = 0;
    /// The least bound among the nodes of either count; +inf when none was
    /// dropped. The search counts as complete only while this is not below
    /// the incumbent.
    Real dropped_node_bound = kInf;
    Int pre_lp_propagation_calls = 0;
    Int pre_lp_propagation_pruned = 0;
    Int pre_lp_propagation_fixings = 0;
    /// Cumulative across internal retries and unboundedness confirmation passes.
    Sos2PropagationStats root_sos2_propagation;
    Sos2PropagationStats tree_sos2_propagation;
};

struct MipCutStats {
    Int root_rounds = 0;
    Int root_cuts_added = 0;
    /// Root rounds whose LP did not solve to optimality and were removed.
    Int root_rounds_backed_out = 0;
    Int tree_nodes_with_cuts = 0;
    Int tree_nodes_skipped = 0;
    Int tree_rounds = 0;
    Int tree_cuts_local = 0;
    Int tree_cuts_global = 0;
    Int tree_cuts_purged = 0;
    Int tree_cuts_revived = 0;
    Real tree_lp_delta = 0.0;
    Int root_cut_bound_tightenings = 0;
    Int root_cut_bound_tightenings_backed_out = 0;
    /// Includes visits in rounds whose LP changes were backed out.
    std::uint64_t root_cut_propagation_row_visits = 0;
    /// Work attempted by root cut inference. Tightenings count retained updates.
    DomainPropagationWork root_cut_inference;
    /// Counts a pass that spends its entire cap, even if its last visit also
    /// reaches a row fixpoint or its final cut batch is backed out.
    Int root_cut_propagation_budget_exhaustions = 0;
    /// Cutoff-scoped deductions attempted, including those in backed-out rounds.
    Int root_cutoff_objective_tightenings = 0;
    Int root_cutoff_reduced_cost_tightenings = 0;
    Int root_cutoff_row_tightenings = 0;
    /// Successful LP re-solves with cutoff-scoped bounds.
    Int root_cutoff_feedbacks = 0;
    Int root_cutoff_resolves_backed_out = 0;
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
    Int ray_qualified = 0;
    Int ray_shortened = 0;
    Int ray_literals_dropped = 0;
    Int ray_reused = 0;
    std::uint64_t watch_candidates = 0;
    std::uint64_t watch_body_scans = 0;
    std::uint64_t watch_literal_checks = 0;
    Int bound_conflicts_learned = 0;
    Int bound_conflict_subsumed = 0;
    Int bound_conflict_tightenings = 0;
    Int bound_conflict_cutoffs = 0;
    Int lp_cutoff_proofs_learned = 0;
    Int lp_combined_proofs_learned = 0;
    Int reconvergence_cuts_added = 0;
    std::uint64_t linear_clause_checks = 0;
    std::uint64_t linear_literal_checks = 0;
    std::optional<Real> root_bound_before_tree = std::nullopt;
    bool bound_conflict_learning_in_tree = false;
    bool reconvergence_cuts_in_tree = false;
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
    /// Nodes entered by diving into a child instead of popping the queue.
    Int plunge_nodes = 0;
    /// Backtracks because the plunged child produced no children (infeasible,
    /// integral, or pruned by the incumbent before it could be processed).
    Int plunge_backtracks_infeasible = 0;
    /// Backtracks because the plunge depth limit was reached.
    Int plunge_backtracks_depth = 0;
    /// Dives declined because the child LP bound degrades the queue best bound
    /// by more than the configured quotient.
    Int plunge_backtracks_bound = 0;
    /// Rounding heuristic calls made at a plunge leaf before backtracking.
    Int plunge_rounding_calls = 0;
    /// Plunge-leaf rounding calls that improved the incumbent.
    Int plunge_rounding_improvements = 0;
    /// Openings of the work-unit heuristic gate in the deterministic parallel tree.
    uint64_t det_heur_gate_openings = 0;
    /// Serial restarts whose conflict activity reached the configured threshold.
    Int conflict_restarts = 0;
    Real last_conflict_restart_activity = 0.0;
};

struct MipSerialNodePolicyStats {
    std::optional<NodePolicy> policy;
    Int nodes_selected = 0;
    /// Selected nodes resolved without children. Queue prunes and unresolved LPs are separate.
    Int nodes_closed = 0;
    Int nodes_pruned_from_queue = 0;
    Int incumbents_accepted = 0;
    Int restarts = 0;
    /// Net gain in objective units in the final attempt, when both bounds are finite.
    Real proof_bound_improvement = 0.0;
    /// The final serial tree exhausted the queue and resolved all outstanding bounds.
    bool proof_completed = false;
};

struct MipTreePresolveStats {
    Int attempts = 0;
    Int runs = 0;
    Int skipped = 0;
    Int infeasible = 0;
    /// Activity tightenings on runs credited by the tree presolve benefit heuristic.
    Int activity_tightenings = 0;
    Int reduced_cost_tightenings = 0;
    Int lp_resolves = 0;
    Real lp_delta = 0.0;
};

struct MipSymmetryStats {
    Int orbits = 0;
    Int cuts_added = 0;
    Int symbreak_added = 0;
    Int constraints_aggregated = 0;
    Int generators = 0;
    Int isomorphism_prunes = 0;
    bool cuts_applied = false;
    bool used_full_detection = false;
    double detect_work_units = 0.0;
    double cut_work_units = 0.0;
};

struct MipExactRefinementStats {
    ExactRefinementMode mode = ExactRefinementMode::Off;
    bool rational_verification_enabled = false;
    bool triggered = false;
    bool certificate_passed = true;
    bool rational_supported = true;
    bool rational_certificate_passed = true;
    Int rounds = 0;
    Int repair_passes = 0;
    Int resolve_calls = 0;
    Int resolve_iterations = 0;
    Int rows_evaluated = 0;
    Int cols_evaluated = 0;
    double evaluation_work_units = 0.0;
    double resolve_work_units = 0.0;
    Real max_row_violation_before = 0.0;
    Real max_row_violation_after = 0.0;
    Real max_col_violation_before = 0.0;
    Real max_col_violation_after = 0.0;
    Real objective_mismatch_before = 0.0;
    Real objective_mismatch_after = 0.0;
};

struct MipProbingStats {
    bool enabled = false;
    Int variables_probed = 0;
    bool systematic = false;
    std::array<Int, 2> branches_probed{};
    std::uint64_t row_visits = 0;
    std::uint64_t implication_candidates = 0;
    Int checked_conflicts_published = 0;
    std::uint64_t tree_implication_fixings = 0;
    std::uint64_t tree_implication_cutoffs = 0;
    Int fixings_found = 0;
    Int implications_found = 0;
    Int vubs_found = 0;
    Int vlbs_found = 0;
    Int equivalences_found = 0;
    Int rounds = 0;
    double time_seconds = 0.0;
};

enum class RapidLearningStop {
    NotRun,
    Exhausted,
    NodeLimit,
    RowLimit,
    NoYield,
    Deadline,
    Infeasible
};

struct MipRapidLearningStats {
    RapidLearningStop stop = RapidLearningStop::NotRun;
    Int nodes = 0;
    Int row_visits = 0;
    Int empty_blocks = 0;
    std::uint64_t bounds_imported = 0;
    Int conflicts_imported = 0;
    Int incumbents_imported = 0;
    std::uint64_t proofs_rejected = 0;
    Int duplicate_conflicts = 0;
    Int incumbents_rejected = 0;
    std::uint64_t tree_conflict_tightenings = 0;
    std::uint64_t tree_conflict_cutoffs = 0;
};

/// Statistics for the root conflict graph / clique table.
struct MipCliqueStats {
    bool enabled = false;
    Int num_binaries = 0;
    Int num_conflict_edges = 0;
    Int num_cliques = 0;        // after merging and subsumption
    Int cliques_from_cuts = 0;  // cliques extracted from root cuts
    double build_time_seconds = 0.0;
    bool skipped_too_large = false;  // binary count exceeded the build cap
    Int node_attachments = 0;        // node propagators given the table
};

struct MipResult {
    Status status = Status::Error;
    Real objective = 0.0;
    Real best_bound = -kInf;
    Real gap = kInf;
    bool gap_limit_reached = false;
    Int nodes = 0;
    Int lp_iterations = 0;
    double work_units = 0.0;  // deterministic work measure
    double time_seconds = 0.0;
    std::vector<Real> solution;
};

class ConflictWatchIndex;

class MipSolver {
public:
    MipSolver();
    ~MipSolver();

    void load(const LpProblem& problem);
    MipResult solve();

    // Parameter setters.
    /// Use adjacent interval selectors when linearizing SOS2 sets on subsequent loads.
    void setSos2AdjacentIntervals(bool enabled) { sos2_adjacent_intervals_ = enabled; }
    /// Branch directly on SOS2 sets on subsequent loads. Takes precedence over selectors.
    void setSos2NativeBranching(bool enabled) { sos2_native_branching_ = enabled; }
    /// Apply local forced-support fixings when solving a model with native SOS2 sets.
    void setSos2Propagation(bool enabled) { sos2_propagation_enabled_ = enabled; }
    void setNodeLimit(Int limit) { node_limit_ = limit; }
    void setTimeLimit(double seconds) { time_limit_ = seconds; }
    void setGapTolerance(Real tol) { gap_tol_ = tol; }
    void setVerbose(bool v) {
        verbose_ = v;
        log_.setEnabled(v);
    }
    void setPresolve(bool p) { presolve_ = p; }
    void setRootPresolveOptions(const PresolveOptions& options) { presolve_options_ = options; }
    [[nodiscard]] const PresolveOptions& getRootPresolveOptions() const {
        return presolve_options_;
    }
    void setMaxCutRounds(Int r) { max_cut_rounds_ = r; }
    void setMaxCutsPerRound(Int c) { max_cuts_per_round_ = c; }
    void setCutsEnabled(bool e) { cuts_enabled_ = e; }
    /// Opt in to selecting uninstalled root cuts by violation at the current LP point.
    void setRootCutRecheck(bool enabled) { root_cut_recheck_ = enabled; }
    /// Propagate installed model and cut rows during root cut rounds. Off by default.
    void setRootCutPropagationEnabled(bool enabled) { root_cut_propagation_enabled_ = enabled; }
    /// Feed global cliques, implications and bounds back into root cut rounds.
    void setRootCutInferenceEnabled(bool enabled) { root_cut_inference_enabled_ = enabled; }
    /// Feed incumbent-derived bounds back into root separation. Off by default.
    void setRootCutoffFeedbackEnabled(bool enabled) { root_cutoff_feedback_enabled_ = enabled; }
    /// Limit visits across all root cut rounds, shared by root propagators.
    /// Zero keeps only the per-round limit. A negative value throws std::invalid_argument.
    void setRootCutPropagationRowBudget(Int visits);
    void setCutFamilyEnabled(CutFamily family, bool enabled) {
        switch (family) {
            case CutFamily::Gomory:
                cut_family_config_.gomory = enabled;
                break;
            case CutFamily::Mir:
                cut_family_config_.mir = enabled;
                break;
            case CutFamily::Cover:
                cut_family_config_.cover = enabled;
                break;
            case CutFamily::ImpliedBound:
                cut_family_config_.implied_bound = enabled;
                break;
            case CutFamily::Clique:
                cut_family_config_.clique = enabled;
                break;
            case CutFamily::ZeroHalf:
                cut_family_config_.zero_half = enabled;
                break;
            case CutFamily::Mixing:
                cut_family_config_.mixing = enabled;
                break;
            case CutFamily::Cmir:
                cut_family_config_.cmir = enabled;
                break;
            case CutFamily::StrongCg:
                cut_family_config_.strong_cg = enabled;
                break;
            case CutFamily::LiftedCover:
                cut_family_config_.lifted_cover = enabled;
                break;
            case CutFamily::ModK:
                cut_family_config_.mod_k = enabled;
                break;
            case CutFamily::IntersectionCut:
                cut_family_config_.intersection_cut = enabled;
                break;
            case CutFamily::MultiRow:
                cut_family_config_.multi_row = enabled;
                break;
            case CutFamily::FlowCover:
                cut_family_config_.flow_cover = enabled;
                break;
            case CutFamily::Unknown:
            case CutFamily::Count:
            default:
                break;
        }
    }
    void setCutFamilyConfig(const CutFamilyConfig& config) { cut_family_config_ = config; }
    [[nodiscard]] const CutFamilyConfig& getCutFamilyConfig() const { return cut_family_config_; }
    /// Separate singleton signed flow covers at the root. Off by default.
    void setRootFlowCoverEnabled(bool enabled) { cut_family_config_.flow_cover = enabled; }
    void setCutEffortMode(CutEffortMode mode) { cut_effort_mode_ = mode; }
    [[nodiscard]] CutEffortMode getCutEffortMode() const { return cut_effort_mode_; }
    void setCutWorkBudgets(double per_node, double per_round, double global) {
        cut_per_node_work_budget_ = std::max(1.0, per_node);
        cut_per_round_work_budget_ = std::max(1.0, per_round);
        cut_global_work_budget_ = std::max(1.0, global);
    }
    void setNumThreads(Int n) { num_threads_ = n; }
    void setRootLpPolicy(RootLpPolicy policy) { root_lp_policy_ = policy; }
    /// Charge dual-simplex pivot-row assembly for the work it performs.
    void setDualRowPriceActualWork(bool enabled) { dual_row_price_actual_work_ = enabled; }
    void setDualRetainDseWeights(bool enabled) { dual_retain_dse_weights_ = enabled; }
    void setBarrierAlgorithm(BarrierAlgorithm algo) { barrier_algorithm_ = algo; }
    void setBarrierOrdering(BarrierOrdering ord) { barrier_ordering_ = ord; }
    void setBarrierDualize(BarrierDualize mode) { barrier_dualize_ = mode; }
    void setBarrierUseGpu(bool use_gpu) {
        barrier_algorithm_ = use_gpu ? BarrierAlgorithm::Auto : BarrierAlgorithm::CpuCholesky;
    }
    void setBarrierGpuThresholds(Int, Int) {
        // Legacy — thresholds now controlled via BarrierAlgorithm dispatch.
    }
    void setParallelMode(ParallelMode mode) { parallel_mode_ = mode; }
    /// Keep opportunistic TBB workers available while other workers own nodes.
    void setPersistentTreeWorkers(bool enabled) { persistent_tree_workers_ = enabled; }
    /// Use a fixed pseudocost view within each deterministic TBB tree batch.
    void setDeterministicBatchBranching(bool enabled) { deterministic_batch_branching_ = enabled; }
    void setHeuristicMode(HeuristicRuntimeMode mode) {
        parallel_mode_ = (mode == HeuristicRuntimeMode::Opportunistic)
                             ? ParallelMode::Opportunistic
                             : ParallelMode::Deterministic;
    }
    void setHeuristicSeed(uint64_t seed) { heuristic_seed_ = seed; }
    void setRootHeuristicThresholds(Int max_int_inf, Int max_int_vars) {
        root_heuristic_max_int_inf_ = std::max<Int>(1, max_int_inf);
        root_heuristic_max_int_vars_ = std::max<Int>(1, max_int_vars);
    }
    /// Limit root feasibility-jump passes, including the initial pass. Zero skips the call.
    /// Throws std::invalid_argument outside [0, default], so the control only reduces the budget.
    void setRootFeasibilityJumpRestarts(Int restarts);
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
    void setSymmetryEnabled(bool enabled) { symmetry_enabled_ = enabled; }
    void setExactRefinementMode(ExactRefinementMode mode) { exact_refinement_mode_ = mode; }
    void setExactRefinementRationalCheck(bool enabled) {
        exact_refinement_rational_check_ = enabled;
    }
    void setExactRefinementWarningTolerance(Real tol) {
        exact_refinement_warning_tol_ = std::max<Real>(1e-12, tol);
    }
    void setExactRefinementCertificateTolerance(Real tol) {
        exact_refinement_certificate_tol_ = std::max<Real>(1e-12, tol);
    }
    void setExactRefinementMaxRounds(Int rounds) {
        exact_refinement_max_rounds_ = std::max<Int>(1, rounds);
    }
    void setExactRefinementRepairPasses(Int passes) {
        exact_refinement_repair_passes_ = std::max<Int>(1, passes);
    }
    void setExactRefinementRationalScale(Real scale) {
        exact_refinement_rational_scale_ = std::max<Real>(1.0, scale);
    }
    void setConflictsEnabled(bool enabled) { conflicts_enabled_ = enabled; }
    /// Use decayed conflict activity in serial pseudocost ranking. Off by default.
    void setConflictActivityBranchingEnabled(bool enabled) {
        conflict_activity_branching_enabled_ = enabled;
    }
    /// Index learned path no-goods in the serial tree. Off by default.
    void setConflictWatchesEnabled(bool enabled) { conflict_watches_enabled_ = enabled; }
    /// Collect extra comparison counters for serial conflict benchmarks.
    void setConflictBenchStats(bool enabled) { conflict_bench_stats_ = enabled; }

    /// Record propagation explanations in the node domains (off by default).
    void setBoundReasonsEnabled(bool enabled) { bound_reasons_enabled_ = enabled; }
    /// Learn and propagate 1-UIP bound conflicts in the serial all-discrete tree.
    /// Off by default.
    void setBoundConflictLearningEnabled(bool enabled) {
        bound_conflict_learning_enabled_ = enabled;
    }
    /// Add checked binary reconvergence cuts in serial all-discrete search.
    void setReconvergenceCutsEnabled(bool enabled) { reconvergence_cuts_enabled_ = enabled; }
    /// Shorten serial tree no-goods with rechecked LP infeasibility rays. Off by default.
    void setConflictRayShorteningEnabled(bool enabled) {
        conflict_ray_shortening_enabled_ = enabled;
    }
    /// Learn from checked LP cutoff and combined proofs in the serial tree.
    /// Off by default.
    void setConflictLpProofsEnabled(bool enabled) { conflict_lp_proofs_enabled_ = enabled; }
    /// Turn strong branching off: branching then rests on pseudocosts alone,
    /// learned from the tree. Mainly for tests and experiments.
    void setStrongBranchingEnabled(bool enabled) { strong_branching_enabled_ = enabled; }
    /// Give the weaker branch direction more weight in reliability branching.
    /// Off by default, which scores a branch by its larger directional gain.
    void setBalancedBranchScore(bool enabled) { branching_rule_.setBalancedScore(enabled); }
    /// Measure the first domain propagation pass of each serial variable-branch child.
    void setDirectionalInferenceEnabled(bool enabled) { directional_inference_enabled_ = enabled; }
    /// Measure live sample coverage at serial tree pseudocost decisions after the root.
    /// Only feasible first-pass propagation contributes samples. Off by default.
    void setInferenceBranchCoverageEnabled(bool enabled) {
        inference_branch_coverage_enabled_ = enabled;
    }
    /// Prefer a ready inference challenger within 1% of the serial pseudocost winner.
    void setInferenceBranchScoreEnabled(bool enabled) { inference_branch_score_enabled_ = enabled; }
    /// Count qualified first-LP dispositions of actual serial variable children.
    void setBranchCutoffStatsEnabled(bool enabled) { branch_cutoff_stats_enabled_ = enabled; }

    /// Collect LP dual degeneracy at fresh serial reliability decisions.
    /// Root heuristics or reduced-cost fixing can leave the root basis stale,
    /// in which case its decision is counted as unavailable.
    void setBranchDualDegeneracyEnabled(bool enabled) { branch_dual_degeneracy_enabled_ = enabled; }
    /// Compare smoothed cutoff rates on the serial pseudocost shortlist.
    /// Score takes precedence over inference ranking. Both may record coverage.
    void setBranchCutoffScoreMode(BranchCutoffScoreMode mode) { branch_cutoff_score_mode_ = mode; }
    /// Learn pseudocosts from the gains of solved child nodes (#199). On by
    /// default; off restores the search where only strong-branch probes feed
    /// pseudocosts, as an A/B switch for benchmarking.
    void setTreePseudocostsEnabled(bool enabled) { tree_pseudocosts_enabled_ = enabled; }
    /// Cap the dual simplex iterations of each node LP solve; negative (the
    /// default) leaves the LP's own limit. A node whose LP stops on the cap
    /// is dropped unexplored, and the result then reports it (#283): its
    /// bound enters best_bound and the status is not Optimal unless the gap
    /// closes. Mainly for tests that need a node LP without a verdict.
    void setNodeLpIterationLimit(Int limit) { node_lp_iter_limit_ = limit; }
    void setSearchProfile(SearchProfile profile) { search_profile_ = profile; }
    /// Pin the serial queue policy independently of the search profile.
    /// Pass nullopt for the profile schedule. DepthFirst is not a supported override.
    void setSerialNodePolicyOverride(std::optional<NodePolicy> policy);
    void setRestartsEnabled(bool enabled) { restarts_enabled_ = enabled; }
    /// Opt in to spacing enabled serial restart triggers. The minimum starts
    /// at the stagnation interval and doubles after each restart.
    void setGeometricRestartsEnabled(bool enabled) { geometric_restarts_enabled_ = enabled; }
    /// Opt in to exact-safe serial restarts on recent learned conflicts.
    /// Evidence decays by 0.995 per node, gains one per nonempty published
    /// conflict, and resets on any restart.
    /// Zero disables this trigger. Stable search and the queue-size floor still apply.
    void setConflictRestartThreshold(Int threshold) {
        conflict_restart_threshold_ = std::max<Int>(0, threshold);
    }
    void setRestartControls(Int stagnation_nodes, Int keep_nodes) {
        restart_stagnation_nodes_ = std::max<Int>(8, stagnation_nodes);
        restart_keep_nodes_ = std::max<Int>(2, keep_nodes);
    }
    /// Configure the serial plunge (dive) phase.
    /// @param max_depth      Maximum number of consecutive dives after a branch.
    ///                       0 disables plunging entirely (the default).
    /// @param bound_quotient Relative bound-degradation quotient: a dive is
    ///                       declined when the child LP bound exceeds the
    ///                       queue's best bound by more than
    ///                       bound_quotient * max(1, |best bound|).
    void setPlungeControls(Int max_depth, Real bound_quotient) {
        plunge_max_depth_ = std::max<Int>(0, max_depth);
        plunge_bound_quotient_ = std::max<Real>(0.0, bound_quotient);
    }
    [[nodiscard]] Int getPlungeMaxDepth() const { return plunge_max_depth_; }
    [[nodiscard]] Real getPlungeBoundQuotient() const { return plunge_bound_quotient_; }
    /// Check child bounds for a row propagation contradiction before solving the LP.
    void setPreLpPropagationEnabled(bool enabled) { pre_lp_propagation_enabled_ = enabled; }
    /// Apply row bounds at depth-scheduled tree presolve nodes of all-discrete
    /// models. Deterministic batches require setDeterministicTreePresolve.
    /// Off by default.
    void setPreLpPropagationBoundsEnabled(bool enabled) {
        pre_lp_propagation_bounds_enabled_ = enabled;
    }
    void setTreePresolveEnabled(bool enabled) { tree_presolve_enabled_ = enabled; }
    /// Allow in-tree presolve in deterministic TBB batches when tree presolve is enabled.
    /// This includes depth-gated pre-LP bounds when enabled separately.
    void setDeterministicTreePresolve(bool enabled) { deterministic_tree_presolve_ = enabled; }
    void setTreePresolveAutoTuning(bool enabled) {
        tree_presolve_auto_tuning_enabled_ = enabled;
        if (loaded_) {
            refreshTreePresolveProfile();
        }
    }
    void setTreeCutsEnabled(bool enabled) { tree_cuts_enabled_ = enabled; }
    /// Objective propagation (#117): propagate c^T x <= incumbent as a row in
    /// node propagation, and re-propagate the root domain against it whenever
    /// a new incumbent lowers the cutoff. Off by default until benchmarked.
    void setObjectivePropagationEnabled(bool enabled) { objective_propagation_enabled_ = enabled; }
    /// Objective integrality (#261): before the tree search, detect whether
    /// every solution's objective lies on a lattice offset + k s (see
    /// ObjectiveLattice), and if so prune nodes, and cut off in objective
    /// propagation, at incumbent - s (plus a margin) instead of incumbent -
    /// 1e-6. Off by default until benchmarked.
    void setObjectiveIntegralityEnabled(bool enabled) { objective_integrality_enabled_ = enabled; }
    /// The lattice the last solve detected; undetected when the option is off
    /// or the objective is not integral. It describes the solver's
    /// internal model, not the one loaded: the presolved minimization (a
    /// maximization negated) with the root's fixings folded into the offset.
    [[nodiscard]] const ObjectiveLattice& getObjectiveLattice() const { return objective_lattice_; }
    /// Called with every solution submitted as an incumbent (from a
    /// heuristic, an integral node LP or the root), whether or not it improves
    /// on the pool's best, and with the solution presolve finds when it
    /// removes every column; postsolved to the loaded model's columns, with
    /// the name of its source. Each reported point holds the model as loaded
    /// to the incumbent gate: a point that fails it is reported repaired
    /// with its integers fixed, or not at all (#245). For validity audits.
    using IncumbentObserver =
        std::function<void(const std::vector<Real>& x, const std::string& source)>;
    void setIncumbentObserver(IncumbentObserver observer) {
        incumbent_observer_ = std::move(observer);
    }
    /// Tighten separated cuts' coefficients against the global bounds before
    /// they are scored and pooled (SeparatorManager::setStrengthenCuts).
    /// "Global" includes root reduced-cost fixings, which hold relative to the
    /// incumbent, as they do for everything else that reads problem_.
    void setCutStrengthening(bool enabled) { cut_strengthening_ = enabled; }
    [[nodiscard]] bool getCutStrengthening() const { return cut_strengthening_; }
    void setTreePresolveControls(Int max_depth, Int min_frac, Int depth_frequency) {
        tree_presolve_max_depth_ = std::max<Int>(1, max_depth);
        tree_presolve_min_frac_ = std::max<Int>(1, min_frac);
        tree_presolve_depth_frequency_ = std::max<Int>(1, depth_frequency);
    }
    const MipLpStats& getLpStats() const { return lp_stats_; }
    const MipCutStats& getCutStats() const { return cut_stats_; }
    /// Per-family separation statistics accumulated by the last root cut loop.
    /// Empty until `solve()` has run a root cutting-plane pass.
    [[nodiscard]] const CutSeparationStats& getRootCutFamilyStats() const {
        return root_cut_family_stats_;
    }
    const MipConflictStats& getConflictStats() const { return conflict_stats_; }
    const MipPreRootStats& getPreRootStats() const { return pre_root_stats_; }
    [[nodiscard]] bool hasLpLightCapability() const;
    const MipSearchStats& getSearchStats() const { return search_stats_; }
    /// Work counters include all serial tree attempts in the last solve. A
    /// policy is reported once the overridden tree selects or prunes a node.
    const MipSerialNodePolicyStats& getSerialNodePolicyStats() const {
        return serial_node_policy_stats_;
    }
    /// Indexed by loaded-model column (after feature linearization, before presolve).
    /// Empty when no serial tree was entered or measurement was disabled.
    const std::vector<MipBranchInferenceStats>& getBranchInferenceStats() const {
        return branch_inference_stats_;
    }
    /// Coverage from the last solve attempt.
    const BranchInferenceCoverageStats& getBranchInferenceCoverageStats() const {
        return branch_inference_coverage_stats_;
    }
    /// Last serial solve attempt, indexed by original column before presolve.
    const std::vector<MipBranchCutoffStats>& getBranchCutoffStats() const {
        return branch_cutoff_stats_;
    }
    const BranchCutoffComparisonStats& getBranchCutoffComparisonStats() const {
        return branch_cutoff_comparison_stats_;
    }

    const BranchDualDegeneracyStats& getBranchDualDegeneracyStats() const {
        return branch_dual_degeneracy_stats_;
    }
    const MipTreePresolveStats& getTreePresolveStats() const { return tree_presolve_stats_; }
    const RcFixingStats& getRcFixingStats() const { return rc_fixing_stats_; }
    [[nodiscard]] ObjPropStats getObjPropStats() const {
        ObjPropStats out;
        out.node_tightenings = obj_prop_node_tightenings_.load(std::memory_order_relaxed);
        out.node_infeasible = obj_prop_node_infeasible_.load(std::memory_order_relaxed);
        out.repropagations = obj_prop_repropagations_.load(std::memory_order_relaxed);
        out.global_tightenings = objective_cutoff_bounds_.numTightenings();
        return out;
    }
    [[nodiscard]] bool isTreePresolveAutoTuningEnabled() const {
        return tree_presolve_auto_tuning_enabled_;
    }
    [[nodiscard]] bool isTreePresolveBinaryLiteProfileActive() const {
        return tree_presolve_binary_lite_profile_active_;
    }
    const MipSymmetryStats& getSymmetryStats() const { return symmetry_stats_; }
    const MipExactRefinementStats& getExactRefinementStats() const {
        return exact_refinement_stats_;
    }
    const BranchingTelemetry& getBranchingStats() const { return branching_stats_; }
    void setProbingEnabled(bool enabled) { probing_enabled_ = enabled; }
    void setProbingMaxRounds(Int rounds) { probing_max_rounds_ = std::max<Int>(1, rounds); }
    void setProbingTimeLimit(double seconds) { probing_time_limit_ = std::max(0.1, seconds); }
    void setProbingMaxProbesPerRound(Int probes) {
        probing_max_probes_per_round_ = std::max<Int>(1, probes);
    }
    /// Replace the legacy root probing pass with one bounded, checked pass.
    void setSystematicProbingEnabled(bool enabled) { systematic_probing_enabled_ = enabled; }
    void setSystematicProbingMaxPairs(Int pairs) {
        systematic_probing_max_pairs_ = std::max<Int>(1, pairs);
    }
    void setSystematicProbingRowsPerBranch(Int rows) {
        systematic_probing_rows_per_branch_ = std::max<Int>(1, rows);
    }
    const MipProbingStats& getProbingStats() const { return probing_stats_; }

    /// Search a separate propagation-only tree over transformed binary columns.
    void setRootRapidLearningEnabled(bool enabled) { root_rapid_learning_enabled_ = enabled; }
    void setRootRapidLearningMaxNodes(Int nodes) {
        root_rapid_learning_max_nodes_ = std::max<Int>(0, nodes);
    }
    void setRootRapidLearningRowBudget(Int rows) {
        root_rapid_learning_row_budget_ = std::max<Int>(0, rows);
    }
    const MipRapidLearningStats& getRapidLearningStats() const { return rapid_learning_stats_; }
    /// Enable the root conflict graph / clique table. When on, cliques drive
    /// node domain propagation and maximal-clique separation at the root.
    void setCliqueTableEnabled(bool enabled) { clique_table_enabled_ = enabled; }
    /// Skip building the conflict graph / clique table above this many
    /// binary columns. Both builds are superlinear in the binary count, so
    /// this bounds the root cost the way the probing and cut budgets do.
    void setCliqueMaxBinaries(Int value) { clique_max_binaries_ = std::max<Int>(0, value); }
    [[nodiscard]] Int getCliqueMaxBinaries() const { return clique_max_binaries_; }
    [[nodiscard]] bool isCliqueTableEnabled() const { return clique_table_enabled_; }
    /// Clique stats, with the node-attachment counter folded in from the
    /// atomic the tree threads increment.
    [[nodiscard]] MipCliqueStats getCliqueStats() const {
        MipCliqueStats out = clique_stats_;
        out.node_attachments = clique_node_attachments_.load(std::memory_order_relaxed);
        return out;
    }
    const CliqueTable& getCliqueTable() const { return clique_table_; }
    const ImplicationGraph& getImplicationGraph() const { return implication_graph_; }
    const VariableBoundStore& getVariableBoundStore() const { return vb_store_; }

private:
    void refreshTreePresolveProfile();
    [[nodiscard]] MipTreePresolveStats treePresolveStatsSnapshot() const;
    void mergeTreePresolveStatsDelta(const MipTreePresolveStats& delta);
    /// Merge a node-local reduced-cost fixing delta into the solver totals.
    /// Node workers accumulate into a stack-local RcFixingStats and flush once
    /// per node, so the hot path stays lock-free.
    void mergeRcFixingStatsDelta(const RcFixingStats& delta);

    struct NodeWorkStats {
        double bound_apply_seconds = 0.0;
        double basis_set_seconds = 0.0;
        double lp_solve_seconds = 0.0;
        Int nodes_solved = 0;
        Int warm_starts = 0;
        Int cold_starts = 0;
        Int pre_lp_propagation_calls = 0;
        Int pre_lp_propagation_pruned = 0;
        Int pre_lp_propagation_fixings = 0;
        std::uint64_t probing_implication_fixings = 0;
        std::uint64_t probing_implication_cutoffs = 0;
        Sos2PropagationStats sos2_propagation;
    };

    void mergeNodeWorkStats(const NodeWorkStats& stats);
    bool applyProbingImplications(DualSimplexSolver& lp, std::vector<Real>& lower,
                                  std::vector<Real>& upper, std::vector<Index>& touched,
                                  NodeWorkStats& stats) const;

    struct NodeScratch {
        std::vector<Int> var_epoch;
        std::vector<Index> var_pos;
        Int current_epoch = 1;
        std::vector<Index> vars;
        std::vector<Real> node_lb;
        std::vector<Real> node_ub;
        std::vector<Index> row_starts;
        std::vector<Index> row_col_indices;
        std::vector<Real> row_values;
        std::vector<Real> row_lower;
        std::vector<Real> row_upper;
        std::vector<Cut> kept_cuts;
        std::vector<Index> rc_tightened;
        std::unique_ptr<DomainPropagator> pre_lp_propagator;
        std::vector<MipBranchInferenceStats>* branch_inference = nullptr;
        std::span<MipBranchInferenceStats> branch_inference_history;
        std::span<MipBranchCutoffStats> branch_cutoff_history;
        BranchDualDegeneracyStats* branch_dual_degeneracy = nullptr;
        std::span<const Index> branch_history_col_mapping;
        std::vector<std::pair<Real, Real>> propagation_input_bounds;
        LpProblem* conflict_global_rows = nullptr;
        const LpProblem* reconvergence_model = nullptr;
        CutPool reconvergence_pool;
    };

    /// The solve proper, on a minimization: solve() negates a maximization
    /// into one before calling it and converts the result back.
    MipResult solveMinimization();
    MipResult confirmNativeSos2Unbounded(MipResult relaxation);
    /// One solve of the minimization; solveMinimization() checks its answer.
    MipResult solveMinimizationOnce();

    /// Run cutting plane rounds at the root node.
    Int runCuttingPlanes(DualSimplexSolver& lp, Int& total_lp_iters, double& total_work,
                         LpProblem* certificate_problem, Real cutoff);

    /// Serial branch-and-bound loop.
    void solveSerial(DualSimplexSolver& lp, NodeQueue& queue, Int& nodes_explored,
                     Int& total_lp_iters, double& total_work, HeuristicRuntime& heuristic_runtime,
                     SolutionPool& solution_pool, Real& incumbent, std::vector<Real>& best_solution,
                     Real root_bound, const std::function<double()>& elapsed,
                     std::span<const Index> branch_history_col_mapping,
                     std::span<MipBranchInferenceStats> branch_inference_history = {},
                     LpProblem* conflict_global_rows = nullptr,
                     const LpProblem* reconvergence_model = nullptr);

    /// Parallel branch-and-bound loop using TBB.
    void solveParallel(const DualSimplexSolver& root_lp, NodeQueue& queue, Int& nodes_explored,
                       Int& total_lp_iters, double& total_work, HeuristicRuntime& heuristic_runtime,
                       SolutionPool& solution_pool, Real& incumbent,
                       std::vector<Real>& best_solution, Real root_bound,
                       const std::function<double()>& elapsed);

    // Check if all integer variables are integral in the given solution.
    bool isFeasibleMip(const std::vector<Real>& primals) const;
    bool isFeasibleLp(const std::vector<Real>& primals) const;
    // The gate every incumbent passes (#240), heuristic_detail::isFeasiblePoint.
    bool isAcceptableIncumbent(std::span<const Real> x) const;
    // An integral node LP point as an incumbent: the point itself when it
    // passes the gate, else the LP completion of its integer values when that
    // passes, else nothing. The completion's work is added to work_units.
    // The completion LP stops by `deadline` (#351), and its status goes to
    // `*completion_status` when given (Optimal when no LP was needed).
    std::optional<HeuristicSolution> integralNodeIncumbent(std::span<const Real> x, Real node_obj,
                                                           double& work_units, Deadline deadline,
                                                           Status* completion_status) const;
    /// integralNodeIncumbent() against `model`, the gate being
    /// heuristic_detail::isFeasiblePoint on it.
    std::optional<HeuristicSolution> completeIncumbent(const LpProblem& model,
                                                       std::span<const Real> x, Real node_obj,
                                                       double& work_units, Deadline deadline = {},
                                                       Status* completion_status = nullptr) const;

    // Compute optimality gap.
    Real computeGap(Real incumbent, Real best_bound) const;

    // Log a progress line.
    /// True when diving into a child whose LP bound is child_bound would degrade
    /// the queue's best bound by more than the configured plunge quotient.
    [[nodiscard]] bool plungeBoundDegraded(Real child_bound, const NodeQueue& queue) const;

    void logProgress(Int nodes, Int open, Int lp_iters, Real incumbent, Real best_bound,
                     double elapsed, bool new_incumbent = false, Int int_inf = -1) const;

    struct DeferredBranching {
        std::optional<TreePseudocostSample> tree_sample;
        BranchingTelemetry telemetry;
        std::vector<StrongBranchPseudocostSample> probe_samples;
    };

    struct DeferredTreePresolve {
        const MipTreePresolveStats& snapshot;
        MipTreePresolveStats& delta;
    };

    /// Process a single node. Returns true if children were created.
    ///
    /// The node's LP outcome is a pseudocost sample for the branching that
    /// created it. Deterministic batch tasks store the sample in
    /// deferred_branching for ordered application after the batch. With batch
    /// branching enabled, selection and estimates also read the stable shared
    /// rule, and probe samples are deferred alongside the tree sample.
    /// With deferred_tree_presolve, use the batch's gate snapshot and collect
    /// this task's presolve stats for the coordinator to apply in order.
    bool processNode(DualSimplexSolver& lp, BnbNode& node, Real incumbent_snapshot,
                     std::vector<BnbNode>& children_out, Real& node_obj_out,
                     std::vector<Real>& node_primals_out, Int& node_iters_out,
                     double& node_work_out, std::vector<Real>& current_lower,
                     std::vector<Real>& current_upper, std::vector<Index>& touched_vars,
                     NodeScratch& node_scratch, NodeWorkStats& node_stats, Int& int_inf_out,
                     Int strong_branch_budget_override = -1,
                     DeferredBranching* deferred_branching = nullptr,
                     const DeferredTreePresolve* deferred_tree_presolve = nullptr);
    /// Re-solve from a cold basis a node LP that ended without a verdict;
    /// returns `first` unchanged when it has one (#283), or when the time
    /// limit stopped it: no time is left to retry in (#351).
    LpResult resolveNodeLpWithoutVerdict(DualSimplexSolver& lp, const LpResult& first,
                                         Int& node_iters_out, double& node_work_out);
    /// Record a node left unexplored for want of an LP verdict, with a valid
    /// lower bound for its subtree (#283); `status` is the LP's, and
    /// TimeLimit counts it as stopped by the limit (#351).
    void recordNodeDroppedWithoutVerdict(Real bound, Status status);
    void ageConflictPool();
    void learnConflictFromNode(const std::vector<BranchDecision>& bound_changes, bool lp_infeasible,
                               const DualSimplexSolver* lp = nullptr,
                               const LpProblem* global_rows = nullptr,
                               std::span<const Index> local_rows = {},
                               std::optional<BoundReasonScope> scope = std::nullopt);
    void learnPropagationConflict(const std::vector<BranchDecision>& path,
                                  const DomainPropagator& dp, NodeScratch& scratch,
                                  std::optional<BoundReasonScope> fallback_scope,
                                  bool learn_nogood = true);
    bool learnLpConflictProof(const std::vector<BranchDecision>& path, const DualSimplexSolver& lp,
                              const LpProblem& global_rows, std::span<const Index> local_rows,
                              Real cutoff, Status status);
    [[nodiscard]] bool lpConflictProofsActive() const {
        return conflict_lp_proofs_enabled_ && conflicts_enabled_ && num_threads_ <= 1;
    }
    [[nodiscard]] bool reconvergenceCutsActive() const {
        return reconvergence_cuts_enabled_ && cuts_enabled_ &&
               cut_effort_mode_ != CutEffortMode::Off && conflicts_enabled_ && num_threads_ <= 1 &&
               model_continuous_vars_ == 0;
    }
    [[nodiscard]] bool boundConflictLearningActive() const {
        return bound_conflict_learning_enabled_ && conflicts_enabled_ && num_threads_ <= 1 &&
               model_continuous_vars_ == 0;
    }
    bool propagateBoundConflicts(DualSimplexSolver& lp, Real cutoff, std::vector<Real>& lower,
                                 std::vector<Real>& upper, std::vector<Index>& touched);
    bool isConflictTriggered(const NodeScratch& node_scratch);
    Index selectConflictAwareBranchVariable(std::span<const Real> primals,
                                            std::span<const Real> current_lower,
                                            std::span<const Real> current_upper,
                                            Index default_var) const;
    HeuristicRuntimeConfig makeHeuristicRuntimeConfig() const;

    /// Build the root conflict graph and clique table from problem_.
    /// No-op when the feature is off or the problem has fewer than two
    /// binaries; populates clique_stats_ either way.
    void buildCliqueTable();

    /// True when node propagation should consult the clique table.
    [[nodiscard]] bool cliquePropagationActive() const {
        return clique_table_enabled_ && clique_table_.numCliques() > 0;
    }

    /// Hand a node's DomainPropagator the objective cutoff row, when objective
    /// propagation is on and there is an incumbent to cut off against.
    void attachObjectiveCutoff(DomainPropagator& dp, Real incumbent) const {
        if (objective_propagator_.loaded() && incumbent < kInf) {
            dp.setObjectiveCutoff(&objective_propagator_, objectivePropagationCutoff(incumbent));
        }
    }

    /// A node whose LP bound reaches this holds nothing better than the
    /// incumbent: incumbent - 1e-6, or tighter on an objective lattice (#261).
    [[nodiscard]] Real pruneBound(Real incumbent) const {
        return objective_lattice_.pruneBound(incumbent);
    }

    /// The cutoff handed to the objective propagator, which prunes at
    /// cutoff - ObjectivePropagator::kPruneTol: the incumbent itself without
    /// a lattice, so that it prunes where pruneBound does.
    [[nodiscard]] Real objectivePropagationCutoff(Real incumbent) const {
        if (!objective_lattice_.detected()) {
            return incumbent;
        }
        return pruneBound(incumbent) + ObjectivePropagator::kPruneTol;
    }

    /// dp.propagate(), counting what the objective row did in it.
    bool propagateNode(DomainPropagator& dp);

    /// Run root probing and populate implication graph / variable bounds.
    /// Returns true if probing proved the problem infeasible.
    [[nodiscard]] bool runRootProbing();
    [[nodiscard]] bool runRootRapidLearning(SolutionPool& solutions, const LpProblem* loaded,
                                            const Presolver* postsolve);

    [[nodiscard]] bool enforceSymmetryBounds(std::vector<Real>& lower, std::vector<Real>& upper,
                                             std::vector<Index>* tightened_vars = nullptr,
                                             double* work_units = nullptr) const;

    using ConflictLiteral = BranchDecision;

    struct ConflictClause {
        std::vector<ConflictLiteral> literals;
        Int age = 0;
        Int hits = 0;
        bool ray_shortened = false;
        bool rapid_learning = false;
        std::uint64_t watch_id = 0;
        std::optional<BoundReasonScope> bound_scope;
        Int lbd = 0;

        /// Checked ray shortening proves the exact path endpoints.
        [[nodiscard]] Real tolerance() const { return ray_shortened ? 0.0 : 1e-9; }
    };
    ConflictClause acquireConflictClause();
    void recycleConflictClause(ConflictClause clause);
    enum class ConflictSource {
        BoundInfeasible,
        LpInfeasible,
        LpCutoff,
        RootProbe,
        RootRapidLearning
    };
    void publishConflict(ConflictClause clause, ConflictSource source);
    void recordConflictCutoff(ConflictClause& clause);

    // Problem data.
    LpProblem problem_;
    bool loaded_ = false;
    // False when an integer column of the loaded model has no integer value
    // between its bounds.
    bool integer_bounds_ok_ = true;
    // -1 while a maximization is solved as its negation, so logs show objective
    // values in the model's own sense.
    Real log_obj_sign_ = 1.0;

    // Parameters.
    bool sos2_adjacent_intervals_ = false;
    bool sos2_native_branching_ = false;
    bool sos2_propagation_enabled_ = false;
    Int node_limit_ = 1000000;
    double time_limit_ = 3600.0;
    // Start of the current solveMinimizationOnce(): with time_limit_, the one
    // deadline every stage of the solve runs against (#336).
    std::chrono::steady_clock::time_point solve_start_ = std::chrono::steady_clock::now();
    // Seconds left before the deadline, at least 0.
    [[nodiscard]] double timeLeft() const;
    Real gap_tol_ = 1e-4;
    bool verbose_ = true;
    bool presolve_ = true;
    PresolveOptions presolve_options_{};
    Int num_threads_ = 1;

    // Cutting plane parameters.
    Int max_cut_rounds_ = 20;
    bool root_cut_recheck_ = false;
    bool root_cut_propagation_enabled_ = false;
    bool root_cut_inference_enabled_ = false;
    bool root_cutoff_feedback_enabled_ = false;
    Int root_cut_propagation_row_budget_ = 0;
    Int max_cuts_per_round_ = 50;
    bool cuts_enabled_ = true;
    CutFamilyConfig cut_family_config_{};
    CutEffortMode cut_effort_mode_ = CutEffortMode::Auto;
    double cut_per_node_work_budget_ = 2.5e5;
    double cut_per_round_work_budget_ = 5.0e4;
    double cut_global_work_budget_ = 1.0e6;
    RootLpPolicy root_lp_policy_ = RootLpPolicy::DualDefault;
    bool dual_row_price_actual_work_ = false;
    bool dual_retain_dse_weights_ = false;
    BarrierAlgorithm barrier_algorithm_ = BarrierAlgorithm::Auto;
    BarrierOrdering barrier_ordering_ = BarrierOrdering::Auto;
    BarrierDualize barrier_dualize_ = BarrierDualize::Off;
    ParallelMode parallel_mode_ = ParallelMode::Deterministic;
    bool persistent_tree_workers_ = false;
    bool deterministic_batch_branching_ = false;
    uint64_t heuristic_seed_ = 1;
    Int root_heuristic_max_int_inf_ = kRootHeuristicMaxIntInf;
    Int root_heuristic_max_int_vars_ = kRootHeuristicMaxIntVars;
    Int root_feasjump_max_restarts_ = HeuristicRuntimeConfig::kDefaultRootFeasibilityJumpRestarts;
    bool pre_root_lp_free_enabled_ = false;
    bool pre_root_lp_free_early_stop_ = true;
    Int pre_root_lp_free_max_rounds_ = 24;
    double pre_root_lp_free_work_budget_ = 5.0e4;
    bool pre_root_lp_light_enabled_ = false;
    bool pre_root_portfolio_enabled_ = true;
    bool conflicts_enabled_ = true;
    bool conflict_activity_branching_enabled_ = false;
    bool conflict_watches_enabled_ = false;
    bool conflict_bench_stats_ = false;
    bool bound_reasons_enabled_ = false;
    bool bound_conflict_learning_enabled_ = false;
    bool reconvergence_cuts_enabled_ = false;
    bool conflict_ray_shortening_enabled_ = false;
    bool conflict_lp_proofs_enabled_ = false;
    bool strong_branching_enabled_ = true;
    bool tree_pseudocosts_enabled_ = true;
    bool directional_inference_enabled_ = false;
    bool inference_branch_coverage_enabled_ = false;
    bool inference_branch_score_enabled_ = false;
    bool branch_cutoff_stats_enabled_ = false;
    bool branch_dual_degeneracy_enabled_ = false;
    BranchCutoffScoreMode branch_cutoff_score_mode_ = BranchCutoffScoreMode::Off;
    Int conflict_max_pool_size_ = 512;
    Int conflict_max_age_ = 64;
    SearchProfile search_profile_ = SearchProfile::Default;
    std::optional<NodePolicy> serial_node_policy_override_;
    bool restarts_enabled_ = false;
    bool geometric_restarts_enabled_ = false;
    Int restart_stagnation_nodes_ = 96;
    Int restart_keep_nodes_ = 32;
    Int conflict_restart_threshold_ = 0;
    // Plunging is off by default: enabling it changes the serial node order.
    Int plunge_max_depth_ = 0;
    Real plunge_bound_quotient_ = 0.05;
    bool pre_lp_propagation_enabled_ = false;
    bool pre_lp_propagation_bounds_enabled_ = false;
    // Serial in-tree presolve is enabled by default because it materially
    // improves hard MIP search on benchmark instances.
    bool tree_presolve_enabled_ = true;
    bool deterministic_tree_presolve_ = false;
    bool tree_presolve_auto_tuning_enabled_ = true;
    Int tree_presolve_max_depth_ = 24;
    Int tree_presolve_min_frac_ = 8;
    Int tree_presolve_depth_frequency_ = 3;
    bool tree_presolve_binary_lite_profile_active_ = false;
    Int model_binary_vars_ = 0;
    Int model_general_integer_vars_ = 0;
    Int model_continuous_vars_ = 0;
    // Serial in-tree cuts are disabled by default while correctness hardening
    // is in progress for benchmark regressions.
    bool tree_cuts_enabled_ = false;
    // Off until a benchmark decides: on, it changes the default search.
    bool cut_strengthening_ = false;
    MipLpStats lp_stats_{};
    MipCutStats cut_stats_{};
    CutSeparationStats root_cut_family_stats_{};
    MipConflictStats conflict_stats_{};
    MipPreRootStats pre_root_stats_{};
    MipSearchStats search_stats_{};
    MipSerialNodePolicyStats serial_node_policy_stats_{};
    std::vector<MipBranchInferenceStats> branch_inference_stats_;
    BranchInferenceCoverageStats branch_inference_coverage_stats_;
    std::vector<MipBranchCutoffStats> branch_cutoff_stats_;
    BranchCutoffComparisonStats branch_cutoff_comparison_stats_;
    BranchDualDegeneracyStats branch_dual_degeneracy_stats_;
    MipTreePresolveStats tree_presolve_stats_{};
    MipSymmetryStats symmetry_stats_{};
    RcFixingStats rc_fixing_stats_{};
    ReducedCostFixer rc_fixer_;
    bool objective_propagation_enabled_ = false;
    bool objective_integrality_enabled_ = false;
    ObjectiveLattice objective_lattice_;
    IncumbentObserver incumbent_observer_;
    ObjectivePropagator objective_propagator_;
    ObjectiveCutoffBounds objective_cutoff_bounds_;
    // Node workers bump these concurrently.
    std::atomic<Int> obj_prop_node_tightenings_{0};
    std::atomic<Int> obj_prop_node_infeasible_{0};
    std::atomic<Int> obj_prop_repropagations_{0};
    ExactRefinementMode exact_refinement_mode_ = ExactRefinementMode::Off;
    bool exact_refinement_rational_check_ = false;
    Real exact_refinement_warning_tol_ = 1e-7;
    Real exact_refinement_certificate_tol_ = 1e-8;
    Real exact_refinement_rational_scale_ = 1.0e6;
    Int exact_refinement_max_rounds_ = 2;
    Int exact_refinement_repair_passes_ = 2;
    MipExactRefinementStats exact_refinement_stats_{};
    std::vector<ConflictClause> conflict_pool_{};
    std::vector<ConflictClause> conflict_clause_pool_{};
    std::unique_ptr<ConflictWatchIndex> conflict_watches_;
    std::size_t conflict_clause_pool_limit_ = 256;
    std::size_t conflict_clause_capacity_limit_ = 4096;
    std::vector<Real> conflict_scores_{};
    Real conflict_restart_activity_ = 0.0;
    std::unordered_map<Int, Index> sibling_branch_cache_{};
    ReliabilityBranching branching_rule_;
    BranchingTelemetry branching_stats_{};
    std::mutex branching_mutex_;
    mutable std::mutex tree_presolve_stats_mutex_;
    mutable std::mutex rc_fixing_stats_mutex_;
    // Guards the no-verdict fields of lp_stats_, which node workers update.
    std::mutex no_verdict_mutex_;
    Int node_lp_iter_limit_ = -1;
    SymmetryManager symmetry_manager_;
    bool symmetry_enabled_ = true;

    // Probing infrastructure.
    bool probing_enabled_ = false;
    bool systematic_probing_enabled_ = false;
    Int systematic_probing_max_pairs_ = 128;
    Int systematic_probing_rows_per_branch_ = 10000;
    Int probing_max_rounds_ = 3;
    double probing_time_limit_ = 30.0;
    Int probing_max_probes_per_round_ = ProbingConfig{}.max_probes_per_round;
    ImplicationGraph implication_graph_;
    VariableBoundStore vb_store_;
    MipProbingStats probing_stats_{};
    bool root_rapid_learning_enabled_ = false;
    Int root_rapid_learning_max_nodes_ = 128;
    Int root_rapid_learning_row_budget_ = 20000;
    MipRapidLearningStats rapid_learning_stats_{};
    bool rapid_learning_clauses_active_ = false;

    // Conflict graph / clique table, rebuilt at the root of every solve().
    bool clique_table_enabled_ = false;
    Int clique_max_binaries_ = 2000;
    // Counts node DomainPropagators handed the clique table. Node processing
    // is multi-threaded, so this is atomic and folded into clique_stats_ when
    // the solve reports.
    std::atomic<Int> clique_node_attachments_{0};
    ConflictGraph conflict_graph_;
    CliqueTable clique_table_;
    MipCliqueStats clique_stats_{};

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
    static constexpr Int kRootFeasJumpMaxIterations = 500;
    static constexpr Int kRootCrossoverSubproblemIterLimit = 40;
    static constexpr Int kRootCrossoverMinFixedVars = 4;
    static constexpr Int kRootProximitySubproblemIterLimit = 40;
    static constexpr Int kRootProximityMinBinaryVars = 4;
    static constexpr Int kRootUncoverSubproblemIterLimit = 40;
    static constexpr Int kRootUncoverMinFreeVars = 2;
    static constexpr Int kRootReducedCostSubproblemIterLimit = 40;
    static constexpr Real kRootReducedCostRcThreshold = 0.5;
    static constexpr Int kRootReducedCostMinFixedVars = 4;
    static constexpr Real kHeurBudgetMaxWorkShare = 0.20;
    static constexpr Int kHeurBudgetMaxFrequencyScale = 8;
    static constexpr Int kTreePresolveBinaryLiteMaxCols = 512;
    static constexpr Int kTreePresolveBinaryLiteMaxRows = 512;
    static constexpr Int kTreePresolveBinaryLiteMaxDepth = 8;
    static constexpr Int kTreePresolveBinaryLiteMinFrac = 12;
    static constexpr Int kTreePresolveBinaryLiteDepthFrequency = 6;
};

}  // namespace mipx
