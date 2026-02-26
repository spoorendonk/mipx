#include "mipx/mip_solver.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <functional>
#include <numeric>
#include <thread>
#include <unordered_map>

#ifdef __linux__
#include <unistd.h>
#endif

#include "mipx/cut_pool.h"
#include "mipx/barrier.h"
#include "mipx/heuristics.h"
#include "mipx/pdlp.h"

#ifdef MIPX_HAS_TBB
#include <tbb/parallel_for.h>
#include <tbb/task_group.h>
#endif

namespace mipx {

namespace {

/// Get physical core count (linux only, falls back to hardware_concurrency).
unsigned getPhysicalCores() {
#ifdef __linux__
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    if (n > 0) return static_cast<unsigned>(n);
#endif
    return std::thread::hardware_concurrency();
}

/// Count binary, general-integer, and continuous variables.
/// An integer variable with bounds [0,1] is reported as binary.
void countVarTypes(const LpProblem& problem,
                   Int& n_binary, Int& n_integer, Int& n_continuous) {
    n_binary = n_integer = n_continuous = 0;
    for (Index j = 0; j < problem.num_cols; ++j) {
        if (problem.col_type[j] == VarType::Continuous) {
            ++n_continuous;
        } else {
            bool is_binary = (problem.col_type[j] == VarType::Binary) ||
                             (problem.col_lower[j] >= -1e-12 &&
                              problem.col_lower[j] <= 1e-12 &&
                              problem.col_upper[j] >= 1.0 - 1e-12 &&
                              problem.col_upper[j] <= 1.0 + 1e-12);
            if (is_binary)
                ++n_binary;
            else
                ++n_integer;
        }
    }
}

const char* rootPolicyName(RootLpPolicy p) {
    switch (p) {
        case RootLpPolicy::DualDefault: return "dual";
        case RootLpPolicy::BarrierRoot: return "barrier";
        case RootLpPolicy::PdlpRoot: return "pdlp";
        case RootLpPolicy::ConcurrentRootExperimental: return "concurrent";
        default: return "dual";
    }
}

const char* cutEffortName(CutEffortMode mode) {
    switch (mode) {
        case CutEffortMode::Off: return "off";
        case CutEffortMode::Conservative: return "conservative";
        case CutEffortMode::Aggressive: return "aggressive";
        case CutEffortMode::Auto: return "auto";
        default: return "auto";
    }
}

Real sparseCosineSimilarity(std::span<const Index> ind_a, std::span<const Real> val_a,
                            std::span<const Index> ind_b, std::span<const Real> val_b) {
    Real dot = 0.0;
    Real norm_a = 0.0;
    Real norm_b = 0.0;

    for (Real v : val_a) norm_a += v * v;
    for (Real v : val_b) norm_b += v * v;
    if (norm_a <= 1e-30 || norm_b <= 1e-30) return 0.0;

    Index ia = 0;
    Index ib = 0;
    while (ia < static_cast<Index>(ind_a.size()) &&
           ib < static_cast<Index>(ind_b.size())) {
        if (ind_a[ia] == ind_b[ib]) {
            dot += val_a[ia] * val_b[ib];
            ++ia;
            ++ib;
        } else if (ind_a[ia] < ind_b[ib]) {
            ++ia;
        } else {
            ++ib;
        }
    }

    return std::abs(dot) / (std::sqrt(norm_a) * std::sqrt(norm_b));
}

Real averageOrthogonality(const std::vector<const Cut*>& cuts) {
    if (cuts.size() < 2) return 1.0;
    const std::size_t pair_cap = 32;
    std::size_t pairs = 0;
    Real sim_sum = 0.0;
    for (std::size_t i = 0; i < cuts.size() && pairs < pair_cap; ++i) {
        for (std::size_t j = i + 1; j < cuts.size() && pairs < pair_cap; ++j) {
            sim_sum += sparseCosineSimilarity(cuts[i]->indices, cuts[i]->values,
                                              cuts[j]->indices, cuts[j]->values);
            ++pairs;
        }
    }
    if (pairs == 0) return 1.0;
    const Real avg_sim = sim_sum / static_cast<Real>(pairs);
    return std::clamp<Real>(1.0 - avg_sim, 0.0, 1.0);
}

}  // namespace

void MipSolver::load(const LpProblem& problem) {
    problem_ = problem;
    loaded_ = true;
}

HeuristicRuntimeConfig MipSolver::makeHeuristicRuntimeConfig() const {
    HeuristicRuntimeConfig config;
    config.mode = heuristic_mode_;
    config.seed = heuristic_seed_;
    config.rins_node_frequency = kRinsNodeFrequency;
    config.rins_subproblem_iter_limit = kRinsSubproblemIterLimit;
    config.rins_agreement_tol = kRinsAgreementTol;
    config.rins_max_int_inf_for_run = kRinsMaxIntInfForRun;
    config.rins_min_fixed_vars = kRinsMinFixedVars;
    config.rins_min_fixed_rate = kRinsMinFixedRate;
    config.rins_max_relative_gap_for_run = kRinsMaxRelativeGapForRun;
    config.root_max_int_inf = kRootHeuristicMaxIntInf;
    config.root_max_int_vars = kRootHeuristicMaxIntVars;
    config.root_feaspump_max_iter = kRootFeasPumpMaxIter;
    config.root_feaspump_subproblem_iter_limit = kRootFeasPumpSubproblemIterLimit;
    config.root_auxobj_subproblem_iter_limit = kRootAuxObjSubproblemIterLimit;
    config.root_auxobj_min_active_integer_vars = kRootAuxObjMinActiveIntegerVars;
    config.root_zeroobj_subproblem_iter_limit = kRootZeroObjSubproblemIterLimit;
    config.root_rens_subproblem_iter_limit = kRootRensSubproblemIterLimit;
    config.root_rens_min_fixed_vars = kRootRensMinFixedVars;
    config.root_rens_min_fixed_rate = kRootRensMinFixedRate;
    config.root_local_branching_subproblem_iter_limit =
        kRootLocalBranchingSubproblemIterLimit;
    config.root_local_branching_neighborhood_small =
        kRootLocalBranchingNeighborhoodSmall;
    config.root_local_branching_neighborhood_medium =
        kRootLocalBranchingNeighborhoodMedium;
    config.root_local_branching_neighborhood_large =
        kRootLocalBranchingNeighborhoodLarge;
    config.root_local_branching_min_binary_vars = kRootLocalBranchingMinBinaryVars;
    config.budget_max_work_share = kHeurBudgetMaxWorkShare;
    config.budget_max_frequency_scale = kHeurBudgetMaxFrequencyScale;
    return config;
}

bool MipSolver::isFeasibleMip(const std::vector<Real>& primals) const {
    for (Index j = 0; j < problem_.num_cols; ++j) {
        if (problem_.col_type[j] == VarType::Continuous) continue;
        if (!isIntegral(primals[j], kIntTol)) return false;
    }
    return true;
}

bool MipSolver::isFeasibleLp(const std::vector<Real>& primals) const {
    if (static_cast<Index>(primals.size()) != problem_.num_cols) return false;
    const Real tol = 1e-6;
    for (Index j = 0; j < problem_.num_cols; ++j) {
        if (std::isfinite(problem_.col_lower[j]) && primals[j] < problem_.col_lower[j] - tol) {
            return false;
        }
        if (std::isfinite(problem_.col_upper[j]) && primals[j] > problem_.col_upper[j] + tol) {
            return false;
        }
    }

    if (problem_.num_rows == 0) return true;
    std::vector<Real> activity(static_cast<size_t>(problem_.num_rows), 0.0);
    problem_.matrix.multiply(primals, activity);
    for (Index i = 0; i < problem_.num_rows; ++i) {
        if (std::isfinite(problem_.row_lower[i]) && activity[i] < problem_.row_lower[i] - tol) {
            return false;
        }
        if (std::isfinite(problem_.row_upper[i]) && activity[i] > problem_.row_upper[i] + tol) {
            return false;
        }
    }
    return true;
}

Real MipSolver::computeGap(Real incumbent, Real best_bound) const {
    if (std::abs(incumbent) < 1e-10) {
        return std::abs(incumbent - best_bound);
    }
    return std::abs(incumbent - best_bound) / std::max(1.0, std::abs(incumbent));
}

void MipSolver::logProgress(Int nodes, Int open, Int lp_iters,
                             Real incumbent, Real best_bound,
                             double elapsed, bool new_incumbent,
                             Int int_inf) const {
    if (!verbose_) return;

    const char* prefix = new_incumbent ? " *" : "  ";
    Int lpit_per_node = (nodes > 0) ? lp_iters / nodes : 0;

    char int_inf_buf[16];
    if (int_inf >= 0)
        std::snprintf(int_inf_buf, sizeof(int_inf_buf), "%6d", int_inf);
    else
        std::snprintf(int_inf_buf, sizeof(int_inf_buf), "%6s", "-");

    if (incumbent < kInf) {
        Real gap = computeGap(incumbent, best_bound) * 100.0;
        log_.log("%s%8d  %8d  %6d  %s  %14.6e  %14.6e  %7.2f%%  %5.1fs\n",
                 prefix, nodes, open, lpit_per_node, int_inf_buf,
                 best_bound, incumbent, gap, elapsed);
    } else {
        log_.log("%s%8d  %8d  %6d  %s  %14.6e  %14s  %7s  %5.1fs\n",
                 prefix, nodes, open, lpit_per_node, int_inf_buf,
                 best_bound, "-", "-", elapsed);
    }
}

bool MipSolver::processNode(DualSimplexSolver& lp, BnbNode& node,
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
                             Int& int_inf_out) {
    children_out.clear();
    node_iters_out = 0;
    node_work_out = 0.0;
    int_inf_out = -1;
    ++node_stats.nodes_solved;

    // Skip if pruned by bound (might have changed since pop).
    if (incumbent_snapshot < kInf && node.lp_bound >= incumbent_snapshot - 1e-6) {
        return false;
    }

    // Restore variables touched by the previous node to root bounds.
    auto t0 = std::chrono::steady_clock::now();
    for (Index j : touched_vars) {
        if (current_lower[j] != problem_.col_lower[j] ||
            current_upper[j] != problem_.col_upper[j]) {
            lp.setColBounds(j, problem_.col_lower[j], problem_.col_upper[j]);
            current_lower[j] = problem_.col_lower[j];
            current_upper[j] = problem_.col_upper[j];
        }
    }
    touched_vars.clear();

    // Apply this node's bound changes using per-variable aggregation.
    std::unordered_map<Index, Index> var_pos;
    std::vector<Index> vars;
    std::vector<Real> node_lb;
    std::vector<Real> node_ub;
    vars.reserve(node.bound_changes.size());
    node_lb.reserve(node.bound_changes.size());
    node_ub.reserve(node.bound_changes.size());

    for (const auto& bc : node.bound_changes) {
        auto [it, inserted] = var_pos.emplace(bc.variable, static_cast<Index>(vars.size()));
        Index pos = it->second;
        if (inserted) {
            vars.push_back(bc.variable);
            node_lb.push_back(problem_.col_lower[bc.variable]);
            node_ub.push_back(problem_.col_upper[bc.variable]);
        }
        if (bc.is_upper) {
            node_ub[pos] = std::min(node_ub[pos], bc.bound);
        } else {
            node_lb[pos] = std::max(node_lb[pos], bc.bound);
        }
    }

    for (Index p = 0; p < static_cast<Index>(vars.size()); ++p) {
        Index j = vars[p];
        if (node_lb[p] > node_ub[p] + 1e-12) {
            node_stats.bound_apply_seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();
            return false;
        }
        if (current_lower[j] != node_lb[p] || current_upper[j] != node_ub[p]) {
            lp.setColBounds(j, node_lb[p], node_ub[p]);
            current_lower[j] = node_lb[p];
            current_upper[j] = node_ub[p];
        }
        touched_vars.push_back(j);
    }
    node_stats.bound_apply_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();

    // Set basis for warm-start.
    t0 = std::chrono::steady_clock::now();
    if (!node.basis.empty()) {
        ++node_stats.warm_starts;
        lp.setBasis(node.basis);
    } else {
        ++node_stats.cold_starts;
    }
    node_stats.basis_set_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();

    // Solve node LP.
    t0 = std::chrono::steady_clock::now();
    auto node_result = lp.solve();
    node_stats.lp_solve_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    node_iters_out = node_result.iterations;
    node_work_out = node_result.work_units;

    if (node_result.status == Status::Infeasible) {
        return false;
    }
    if (node_result.status != Status::Optimal) {
        return false;
    }

    // Pruned by bound (check again with current incumbent).
    if (incumbent_snapshot < kInf && node_result.objective >= incumbent_snapshot - 1e-6) {
        return false;
    }

    node_obj_out = node_result.objective;
    node_primals_out = lp.getPrimalValues();

    // Count integer infeasibilities (fractional integer variables).
    Int frac_count = 0;
    for (Index j = 0; j < problem_.num_cols; ++j) {
        if (problem_.col_type[j] == VarType::Continuous) continue;
        if (!isIntegral(node_primals_out[j], kIntTol)) ++frac_count;
    }
    int_inf_out = frac_count;

    // Check integer feasibility.
    if (frac_count == 0) {
        return false;  // Caller handles incumbent update.
    }

    // Branch with reliability branching (strong-branch bootstrap + pseudocosts).
    Index branch_var = -1;
    {
        std::lock_guard<std::mutex> lock(branching_mutex_);
        auto selection = branching_rule_.select(lp, problem_,
                                                node_primals_out,
                                                current_lower,
                                                current_upper,
                                                node_result.objective,
                                                false,
                                                branching_stats_);
        branch_var = selection.variable;
    }
    if (branch_var < 0) {
        return false;
    }

    BnbNode solved_node = std::move(node);
    solved_node.lp_bound = node_result.objective;
    solved_node.basis = lp.getBasis();

    auto [left, right] = createChildren(solved_node, branch_var,
                                        node_primals_out[branch_var]);
    left.lp_bound = node_result.objective;
    right.lp_bound = node_result.objective;

    children_out.push_back(std::move(left));
    children_out.push_back(std::move(right));
    return true;
}

void MipSolver::solveSerial(DualSimplexSolver& lp, NodeQueue& queue,
                             Int& nodes_explored, Int& total_lp_iters,
                             double& total_work,
                             HeuristicRuntime& heuristic_runtime,
                             SolutionPool& solution_pool,
                             Real& incumbent, std::vector<Real>& best_solution,
                             Real root_bound,
                             const std::function<double()>& elapsed) {
    std::vector<Real> current_lower = problem_.col_lower;
    std::vector<Real> current_upper = problem_.col_upper;
    std::vector<Index> touched_vars;
    double last_log_time = -kLogInterval;  // ensure first node is logged
    touched_vars.reserve(problem_.num_cols);
    NodeWorkStats node_stats;
    Int rins_calls = 0;
    Int rins_solves = 0;
    Int rins_found = 0;
    Int rins_skip_no_inc = 0;
    Int rins_skip_few_fix = 0;
    Int rins_fixed_sum = 0;

    while (!queue.empty()) {
        if (nodes_explored >= node_limit_) {
            if (verbose_) log_.log("Node limit reached.\n");
            break;
        }
        if (elapsed() >= time_limit_) {
            if (verbose_) log_.log("Time limit reached.\n");
            break;
        }

        if (incumbent < kInf) {
            queue.prune(incumbent - 1e-6);
            if (queue.empty()) break;
        }

        Real best_bound = queue.bestBound();
        if (incumbent < kInf && computeGap(incumbent, best_bound) < gap_tol_) {
            if (verbose_) log_.log("Gap tolerance reached.\n");
            break;
        }

        BnbNode node = queue.pop();
        ++nodes_explored;

        std::vector<BnbNode> children;
        Real node_obj = 0.0;
        std::vector<Real> node_primals;
        Int node_iters = 0;
        double node_work = 0.0;
        Int node_int_inf = -1;

        bool branched = processNode(lp, node, incumbent,
                                    children, node_obj, node_primals, node_iters,
                                    node_work,
                                    current_lower, current_upper, touched_vars,
                                    node_stats, node_int_inf);
        total_lp_iters += node_iters;
        total_work += node_work;

        WorkerHeuristicContext heur_ctx{
            .problem = problem_,
            .lp = lp,
            .primals = node_primals,
            .node_count = nodes_explored,
            .int_inf = node_int_inf,
            .node_objective = node_obj,
            .incumbent = incumbent,
            .incumbent_values = std::span<const Real>(best_solution.data(),
                                                      best_solution.size()),
            .thread_id = 0,
            .total_work_units = total_work,
        };
        auto heur_outcome = heuristic_runtime.runTreeWorker(heur_ctx);
        if (heur_outcome.attempted) {
            ++rins_calls;
            if (heur_outcome.executed_solve) ++rins_solves;
            if (heur_outcome.skipped_no_incumbent) ++rins_skip_no_inc;
            if (heur_outcome.skipped_few_fixes) ++rins_skip_few_fix;
            rins_fixed_sum += heur_outcome.fixed_count;
        }
        total_lp_iters += heur_outcome.lp_iterations;
        total_work += heur_outcome.work_units;

        if (heur_outcome.improved &&
            heur_outcome.solution.has_value() &&
            heur_outcome.solution->objective < incumbent) {
            incumbent = heur_outcome.solution->objective;
            best_solution = std::move(heur_outcome.solution->values);
            solution_pool.submit({best_solution, incumbent}, "rins", 0);
            ++rins_found;
            logProgress(nodes_explored, queue.size(), total_lp_iters,
                        incumbent,
                        queue.empty() ? incumbent : queue.bestBound(),
                        elapsed(), true, node_int_inf);
            queue.prune(incumbent - 1e-6);
        }

        // Log progress periodically (time-based).
        double now = elapsed();
        if (verbose_ && (now - last_log_time >= kLogInterval || nodes_explored <= 1)) {
            logProgress(nodes_explored, queue.size(), total_lp_iters,
                       incumbent,
                       queue.empty() ? (incumbent < kInf ? incumbent : root_bound) : queue.bestBound(),
                       now, false, node_int_inf);
            last_log_time = now;
        }

        if (!branched && !node_primals.empty() && node_int_inf == 0) {
            if (node_obj < incumbent) {
                incumbent = node_obj;
                best_solution = node_primals;
                solution_pool.submit({best_solution, incumbent}, "tree_integral", 0);
                logProgress(nodes_explored, queue.size(), total_lp_iters,
                           incumbent,
                           queue.empty() ? incumbent : queue.bestBound(),
                           elapsed(), true, 0);
                queue.prune(incumbent - 1e-6);
            }
        }

        for (auto& child : children) {
            queue.push(std::move(child));
        }
    }

    lp_stats_.node_bound_apply_seconds += node_stats.bound_apply_seconds;
    lp_stats_.node_basis_set_seconds += node_stats.basis_set_seconds;
    lp_stats_.node_lp_solve_seconds += node_stats.lp_solve_seconds;
    lp_stats_.nodes_solved += node_stats.nodes_solved;
    lp_stats_.warm_starts += node_stats.warm_starts;
    lp_stats_.cold_starts += node_stats.cold_starts;

    if (verbose_ && rins_calls > 0) {
        double avg_fixed = static_cast<double>(rins_fixed_sum) /
                           static_cast<double>(rins_calls);
        log_.log("RINS(serial): calls=%d solves=%d found=%d avg_fixed=%.1f "
                 "skip_no_inc=%d skip_few_fix=%d\n",
                 rins_calls, rins_solves, rins_found, avg_fixed,
                 rins_skip_no_inc, rins_skip_few_fix);
    }
}

#ifdef MIPX_HAS_TBB
void MipSolver::solveParallel(const DualSimplexSolver& root_lp, NodeQueue& queue,
                               Int& nodes_explored, Int& total_lp_iters,
                               double& total_work,
                               const HeuristicRuntimeConfig& runtime_config,
                               SolutionPool& solution_pool,
                               Real& incumbent, std::vector<Real>& best_solution,
                               Real root_bound,
                               const std::function<double()>& elapsed) {
    // Shared state protected by mutexes.
    std::mutex queue_mutex;
    std::mutex incumbent_mutex;
    std::atomic<Int> atomic_nodes{nodes_explored};
    std::atomic<Int> atomic_lp_iters{total_lp_iters};
    std::atomic<double> atomic_work{total_work};
    std::atomic<bool> should_stop{false};
    std::mutex stats_mutex;
    double last_log_time = -kLogInterval;  // ensure first log fires
    Int rins_calls = 0;
    Int rins_solves = 0;
    Int rins_found = 0;
    Int rins_skip_no_inc = 0;
    Int rins_skip_few_fix = 0;
    Int rins_fixed_sum = 0;

    // Each thread needs its own LP solver instance.
    Int actual_threads = std::min(num_threads_, static_cast<Int>(std::thread::hardware_concurrency()));
    if (actual_threads < 1) actual_threads = 1;

    if (verbose_) {
        log_.log("Parallel tree search with %d threads\n", actual_threads);
    }

    // Worker function.
    auto worker = [&](Int thread_id) {
        HeuristicRuntime heuristic_runtime(runtime_config);

        // Create thread-local LP solver by loading problem fresh.
        DualSimplexSolver local_lp;
        local_lp.load(problem_);
        local_lp.setVerbose(false);
        // Warm-start from root basis.
        local_lp.setBasis(root_lp.getBasis());
        std::vector<Real> current_lower = problem_.col_lower;
        std::vector<Real> current_upper = problem_.col_upper;
        std::vector<Index> touched_vars;
        touched_vars.reserve(problem_.num_cols);
        NodeWorkStats local_node_stats;
        Int local_rins_calls = 0;
        Int local_rins_solves = 0;
        Int local_rins_found = 0;
        Int local_rins_skip_no_inc = 0;
        Int local_rins_skip_few_fix = 0;
        Int local_rins_fixed_sum = 0;

        while (!should_stop.load(std::memory_order_relaxed)) {
            // Check limits.
            if (atomic_nodes.load(std::memory_order_relaxed) >= node_limit_) {
                should_stop.store(true, std::memory_order_relaxed);
                break;
            }
            if (elapsed() >= time_limit_) {
                should_stop.store(true, std::memory_order_relaxed);
                break;
            }

            // Get a node from the shared queue.
            BnbNode node;
            {
                std::lock_guard<std::mutex> lock(queue_mutex);
                if (queue.empty()) break;

                Real inc = incumbent;  // Read under potential race, but safe enough.
                if (inc < kInf) {
                    queue.prune(inc - 1e-6);
                    if (queue.empty()) break;
                }

                Real best_bound = queue.bestBound();
                if (inc < kInf && computeGap(inc, best_bound) < gap_tol_) {
                    should_stop.store(true, std::memory_order_relaxed);
                    break;
                }

                node = queue.pop();
            }

            Int node_num = atomic_nodes.fetch_add(1, std::memory_order_relaxed) + 1;

            // Get current incumbent for pruning.
            Real inc_snapshot;
            {
                std::lock_guard<std::mutex> lock(incumbent_mutex);
                inc_snapshot = incumbent;
            }

            // Process node.
            std::vector<BnbNode> children;
            Real node_obj = 0.0;
            std::vector<Real> node_primals;
            Int node_iters = 0;
            double node_work = 0.0;
            Int node_int_inf = -1;

            bool branched = processNode(local_lp, node, inc_snapshot,
                                        children, node_obj, node_primals, node_iters,
                                        node_work,
                                        current_lower, current_upper, touched_vars,
                                        local_node_stats, node_int_inf);
            atomic_lp_iters.fetch_add(node_iters, std::memory_order_relaxed);
            // Atomically add work (relaxed is fine for accumulation).
            auto old_w = atomic_work.load(std::memory_order_relaxed);
            while (!atomic_work.compare_exchange_weak(old_w, old_w + node_work,
                                                       std::memory_order_relaxed)) {}

            bool found_incumbent = false;
            std::vector<Real> incumbent_values;
            Real heur_incumbent = kInf;
            {
                std::lock_guard<std::mutex> lock(incumbent_mutex);
                if (incumbent < kInf && !best_solution.empty()) {
                    heur_incumbent = incumbent;
                    incumbent_values = best_solution;
                }
            }
            if (incumbent_values.empty()) {
                auto pooled = solution_pool.bestSolution();
                if (pooled.has_value()) {
                    heur_incumbent = pooled->objective;
                    incumbent_values = std::move(pooled->values);
                }
            }

            WorkerHeuristicContext heur_ctx{
                .problem = problem_,
                .lp = local_lp,
                .primals = node_primals,
                .node_count = node_num,
                .int_inf = node_int_inf,
                .node_objective = node_obj,
                .incumbent = heur_incumbent,
                .incumbent_values = std::span<const Real>(incumbent_values.data(),
                                                          incumbent_values.size()),
                .thread_id = thread_id,
                .total_work_units = atomic_work.load(std::memory_order_relaxed),
            };
            auto heur_outcome = heuristic_runtime.runTreeWorker(heur_ctx);
            if (heur_outcome.attempted) {
                ++local_rins_calls;
                atomic_lp_iters.fetch_add(heur_outcome.lp_iterations,
                                          std::memory_order_relaxed);
                auto old_w2 = atomic_work.load(std::memory_order_relaxed);
                while (!atomic_work.compare_exchange_weak(
                    old_w2, old_w2 + heur_outcome.work_units,
                    std::memory_order_relaxed)) {}

                if (heur_outcome.executed_solve) ++local_rins_solves;
                if (heur_outcome.skipped_no_incumbent) ++local_rins_skip_no_inc;
                if (heur_outcome.skipped_few_fixes) ++local_rins_skip_few_fix;
                local_rins_fixed_sum += heur_outcome.fixed_count;

                if (heur_outcome.improved && heur_outcome.solution.has_value()) {
                    bool accepted = false;
                    std::lock_guard<std::mutex> lock(incumbent_mutex);
                    if (heur_outcome.solution->objective < incumbent) {
                        incumbent = heur_outcome.solution->objective;
                        best_solution = std::move(heur_outcome.solution->values);
                        solution_pool.submit({best_solution, incumbent}, "rins", thread_id);
                        found_incumbent = true;
                        accepted = true;
                    }
                    if (accepted) {
                        ++local_rins_found;
                    }
                }
            }

            // Check for new incumbent.
            if (!branched && !node_primals.empty() && node_int_inf == 0) {
                std::lock_guard<std::mutex> lock(incumbent_mutex);
                if (node_obj < incumbent) {
                    incumbent = node_obj;
                    best_solution = node_primals;
                    solution_pool.submit({best_solution, incumbent},
                                         "tree_integral", thread_id);
                    found_incumbent = true;
                }
            }

            // Push children to shared queue.
            if (!children.empty()) {
                std::lock_guard<std::mutex> lock(queue_mutex);
                for (auto& child : children) {
                    queue.push(std::move(child));
                }
            }

            // Log progress: new incumbent or periodic heartbeat (time-based).
            double now = elapsed();
            if (verbose_ && (found_incumbent || (now - last_log_time >= kLogInterval))) {
                last_log_time = now;
                std::lock_guard<std::mutex> lock(queue_mutex);
                logProgress(atomic_nodes.load(), queue.size(),
                           atomic_lp_iters.load(),
                           incumbent,
                           queue.empty() ? (incumbent < kInf ? incumbent : root_bound) : queue.bestBound(),
                           elapsed(), found_incumbent, node_int_inf);
            }
        }

        std::lock_guard<std::mutex> lock(stats_mutex);
        lp_stats_.node_bound_apply_seconds += local_node_stats.bound_apply_seconds;
        lp_stats_.node_basis_set_seconds += local_node_stats.basis_set_seconds;
        lp_stats_.node_lp_solve_seconds += local_node_stats.lp_solve_seconds;
        lp_stats_.nodes_solved += local_node_stats.nodes_solved;
        lp_stats_.warm_starts += local_node_stats.warm_starts;
        lp_stats_.cold_starts += local_node_stats.cold_starts;
        rins_calls += local_rins_calls;
        rins_solves += local_rins_solves;
        rins_found += local_rins_found;
        rins_skip_no_inc += local_rins_skip_no_inc;
        rins_skip_few_fix += local_rins_skip_few_fix;
        rins_fixed_sum += local_rins_fixed_sum;
    };

    // Launch worker threads.
    tbb::task_group tg;
    for (Int t = 0; t < actual_threads; ++t) {
        tg.run([&worker, t]() { worker(t); });
    }
    tg.wait();

    // Sync back.
    nodes_explored = atomic_nodes.load();
    total_lp_iters = atomic_lp_iters.load();
    total_work = atomic_work.load();

    if (verbose_) {
        if (rins_calls > 0) {
            double avg_fixed = static_cast<double>(rins_fixed_sum) /
                               static_cast<double>(rins_calls);
            log_.log("RINS(parallel): calls=%d solves=%d found=%d avg_fixed=%.1f "
                     "skip_no_inc=%d skip_few_fix=%d\n",
                     rins_calls, rins_solves, rins_found, avg_fixed,
                     rins_skip_no_inc, rins_skip_few_fix);
        }
        if (nodes_explored >= node_limit_) log_.log("Node limit reached.\n");
        else if (elapsed() >= time_limit_) log_.log("Time limit reached.\n");
    }
}
#else
void MipSolver::solveParallel(const DualSimplexSolver& /*root_lp*/, NodeQueue& queue,
                               Int& nodes_explored, Int& total_lp_iters,
                               double& total_work,
                               const HeuristicRuntimeConfig& runtime_config,
                               SolutionPool& solution_pool,
                               Real& incumbent, std::vector<Real>& best_solution,
                               Real root_bound,
                               const std::function<double()>& elapsed) {
    // Fallback to serial when TBB is not available.
    if (verbose_) {
        log_.log("TBB not available, falling back to serial.\n");
    }
    // Need a mutable LP solver for serial path.
    DualSimplexSolver lp;
    lp.load(problem_);
    auto root_result = lp.solve();
    total_lp_iters += root_result.iterations;
    total_work += root_result.work_units;
    HeuristicRuntime heuristic_runtime(runtime_config);
    solveSerial(lp, queue, nodes_explored, total_lp_iters,
                total_work, heuristic_runtime, solution_pool, incumbent,
                best_solution, root_bound, elapsed);
}
#endif

MipResult MipSolver::solve() {
    if (!loaded_) return {};
    lp_stats_ = {};
    lp_stats_.root_policy = root_lp_policy_;

    auto start_time = std::chrono::steady_clock::now();
    auto elapsed = [&]() -> double {
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(now - start_time).count();
    };

    // Print banner before presolve (uses original problem dimensions).
    if (verbose_ && problem_.hasIntegers()) {
        log_.log("mipx v0.3\n");

        // Thread count and platform capabilities.
        unsigned logical = std::thread::hardware_concurrency();
        unsigned physical = getPhysicalCores();
        const char* tbb_str = "";
        const char* simd_str = "";
#ifdef MIPX_HAS_TBB
        tbb_str = ", TBB";
#endif
#ifdef __AVX512F__
        simd_str = ", AVX-512";
#elif defined(__AVX2__)
        simd_str = ", AVX2";
#elif defined(__AVX__)
        simd_str = ", AVX";
#elif defined(__SSE4_2__)
        simd_str = ", SSE4.2";
#endif
        log_.log("Thread count: %u physical cores, %u logical processors, using up to %d threads%s%s\n",
                 physical, logical, num_threads_, tbb_str, simd_str);

        // Variable type breakdown.
        Int n_binary = 0, n_integer = 0, n_continuous = 0;
        countVarTypes(problem_, n_binary, n_integer, n_continuous);

        // Build type string.
        char type_buf[128] = "";
        if (n_binary > 0 || n_integer > 0) {
            char* p = type_buf;
            p += std::snprintf(p, sizeof(type_buf), " (");
            bool first = true;
            if (n_binary > 0) { p += std::snprintf(p, sizeof(type_buf) - (p - type_buf), "%d binary", n_binary); first = false; }
            if (n_integer > 0) { if (!first) p += std::snprintf(p, sizeof(type_buf) - (p - type_buf), ", "); p += std::snprintf(p, sizeof(type_buf) - (p - type_buf), "%d integer", n_integer); first = false; }
            if (n_continuous > 0) { if (!first) p += std::snprintf(p, sizeof(type_buf) - (p - type_buf), ", "); p += std::snprintf(p, sizeof(type_buf) - (p - type_buf), "%d continuous", n_continuous); }
            std::snprintf(p, sizeof(type_buf) - (p - type_buf), ")");
        }
        log_.log("Solving MIP with:\n  %d rows, %d cols%s, %d nonzeros\n",
                 problem_.num_rows, problem_.num_cols, type_buf,
                 problem_.matrix.numNonzeros());

        // Settings line: only non-default values.
        char settings[256] = "";
        char* sp = settings;
        if (time_limit_ != 3600.0) sp += std::snprintf(sp, sizeof(settings) - (sp - settings), "time=%.0fs ", time_limit_);
        if (gap_tol_ != 1e-4) sp += std::snprintf(sp, sizeof(settings) - (sp - settings), "gap=%.2f%% ", gap_tol_ * 100.0);
        if (node_limit_ != 1000000) sp += std::snprintf(sp, sizeof(settings) - (sp - settings), "nodes=%d ", node_limit_);
        if (!presolve_) sp += std::snprintf(sp, sizeof(settings) - (sp - settings), "presolve=off ");
        if (!cuts_enabled_) sp += std::snprintf(sp, sizeof(settings) - (sp - settings), "cuts=off ");
        if (max_cut_rounds_ != 20) sp += std::snprintf(sp, sizeof(settings) - (sp - settings), "cut_rounds=%d ", max_cut_rounds_);
        if (cut_effort_mode_ != CutEffortMode::Auto) {
            sp += std::snprintf(sp, sizeof(settings) - (sp - settings),
                                "cut_mode=%s ", cutEffortName(cut_effort_mode_));
        }
        if (root_lp_policy_ != RootLpPolicy::DualDefault) {
            sp += std::snprintf(sp, sizeof(settings) - (sp - settings),
                                "root_lp=%s ", rootPolicyName(root_lp_policy_));
        }
        if (heuristic_mode_ != HeuristicRuntimeMode::Deterministic) {
            sp += std::snprintf(sp, sizeof(settings) - (sp - settings),
                                "heur_mode=opportunistic ");
        }
        if (heuristic_seed_ != 1) {
            sp += std::snprintf(sp, sizeof(settings) - (sp - settings),
                                "heur_seed=%llu ",
                                static_cast<unsigned long long>(heuristic_seed_));
        }
        if (sp > settings) {
            // Trim trailing space.
            if (sp > settings && *(sp - 1) == ' ') *(sp - 1) = '\0';
            log_.log("  Settings  : %s\n", settings);
        }
        log_.log("\n");
    }

    // Presolve.
    Presolver presolver;
    LpProblem working_problem;
    bool did_presolve = false;

    if (presolve_) {
        working_problem = presolver.presolve(problem_);
        const auto& stats = presolver.stats();
        if (stats.vars_removed > 0 || stats.rows_removed > 0) {
            did_presolve = true;
            if (verbose_) {
                log_.log("Presolve: %d vars removed, %d rows removed, "
                         "%d bounds tightened, %d rounds (%d changed), %.3fs "
                         "[rules: implied=%d abt=%d dual=%d empty_col=%d dup_row=%d] "
                         "[examined: %d rows, %d cols]\n\n",
                         stats.vars_removed, stats.rows_removed,
                         stats.bounds_tightened, stats.rounds,
                         stats.rounds_with_changes, stats.time_seconds,
                         stats.implied_equation_changes,
                         stats.activity_bound_tightening_changes,
                         stats.dual_fixing_changes,
                         stats.empty_col_changes,
                         stats.duplicate_row_changes,
                         stats.rows_examined, stats.cols_examined);
            }
        } else {
            working_problem = problem_;
        }
    } else {
        working_problem = problem_;
    }

    if (presolver.isInfeasible()) {
        MipResult result;
        result.status = Status::Infeasible;
        result.nodes = 0;
        result.lp_iterations = 0;
        result.time_seconds = elapsed();
        if (verbose_) log_.log("Presolve detected infeasibility.\n");
        return result;
    }

    if (working_problem.num_cols == 0) {
        MipResult result;
        result.status = Status::Optimal;
        result.objective = working_problem.obj_offset;
        result.best_bound = working_problem.obj_offset;
        result.gap = 0.0;
        result.nodes = 0;
        result.lp_iterations = 0;
        result.time_seconds = elapsed();
        if (did_presolve) {
            result.solution = presolver.postsolve({});
        }
        return result;
    }

    LpProblem original_problem;
    if (did_presolve) {
        original_problem = std::move(problem_);
        problem_ = std::move(working_problem);
    }

    auto applyPostsolve = [&](MipResult& result) {
        if (did_presolve) {
            if (result.status == Status::Optimal || !result.solution.empty()) {
                result.objective += problem_.obj_offset;
                result.best_bound += problem_.obj_offset;
                result.solution = presolver.postsolve(result.solution);
            } else {
                result.objective += problem_.obj_offset;
                if (result.best_bound != kInf && result.best_bound != -kInf) {
                    result.best_bound += problem_.obj_offset;
                }
            }
            problem_ = std::move(original_problem);
        }
    };

    // If no integers, solve as LP directly.
    if (!problem_.hasIntegers()) {
        if (verbose_) {
            log_.log("mipx v0.3\n");
            unsigned logical = std::thread::hardware_concurrency();
            const char* tbb_str = "";
#ifdef MIPX_HAS_TBB
            tbb_str = ", TBB";
#endif
            log_.log("Thread count: %u logical processors, using up to 1 thread%s\n",
                     logical, tbb_str);
            log_.log("Solving LP with:\n  %d rows, %d cols, %d nonzeros\n\n",
                     problem_.num_rows, problem_.num_cols,
                     problem_.matrix.numNonzeros());
        }

        DualSimplexSolver lp;
        lp.load(problem_);
        lp.setVerbose(verbose_);
        auto lr = lp.solve();

        MipResult result;
        result.status = lr.status;
        result.objective = lr.objective;
        result.best_bound = lr.objective;
        result.gap = 0.0;
        result.nodes = 0;
        result.lp_iterations = lr.iterations;
        result.work_units = lr.work_units;
        result.time_seconds = elapsed();
        if (lr.status == Status::Optimal) {
            result.solution = lp.getPrimalValues();
            if (verbose_) {
                std::fflush(stdout);
                log_.log("\nOptimal: %.10e (%d iterations)\n",
                         result.objective, lr.iterations);
            }
        }
        applyPostsolve(result);
        return result;
    }

    branching_rule_.reset(problem_.num_cols);
    branching_stats_ = {};

    // Solve root LP.
    DualSimplexSolver lp;
    lp.load(problem_);
    lp.setVerbose(false);

    bool root_used_dual = true;
    std::vector<Real> root_primals;
    std::vector<BasisStatus> root_basis;

    auto t0 = std::chrono::steady_clock::now();
    LpResult root_result;
    if (root_lp_policy_ == RootLpPolicy::BarrierRoot) {
        root_used_dual = false;
        BarrierSolver barrier;
        BarrierOptions bopts;
        bopts.verbose = verbose_;
        bopts.use_gpu = barrier_use_gpu_;
        bopts.gpu_min_rows = barrier_gpu_min_rows_;
        bopts.gpu_min_nnz = barrier_gpu_min_nnz_;
        barrier.setOptions(bopts);
        barrier.load(problem_);
        root_result = barrier.solve();
        root_primals = barrier.getPrimalValues();
        if (verbose_) {
            log_.log("Root barrier mode%s.\n", barrier.usedGpu() ? " (GPU backend)" : "");
        }
    } else if (root_lp_policy_ == RootLpPolicy::PdlpRoot) {
        root_used_dual = false;
        PdlpSolver pdlp;
        PdlpOptions popts;
        popts.verbose = verbose_;
        popts.use_gpu = pdlp_use_gpu_;
        popts.gpu_min_rows = pdlp_gpu_min_rows_;
        popts.gpu_min_nnz = pdlp_gpu_min_nnz_;
        pdlp.setOptions(popts);
        pdlp.load(problem_);
        root_result = pdlp.solve();
        root_primals = pdlp.getPrimalValues();
        if (verbose_) {
            log_.log("Root PDLP mode%s.\n", pdlp.usedGpu() ? " (GPU backend)" : "");
        }
    } else {
        if (root_lp_policy_ == RootLpPolicy::ConcurrentRootExperimental && verbose_) {
            log_.log("Root concurrent mode requested; PDLP/barrier/dual racing is "
                     "not yet enabled. Falling back to dual simplex.\n");
        }
        lp.setVerbose(verbose_);
        root_result = lp.solve();
        root_primals = lp.getPrimalValues();
        root_basis = lp.getBasis();
    }
    lp_stats_.root_lp_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();

    Int total_lp_iters = root_result.iterations;
    double total_work = root_result.work_units;

    if (root_result.status == Status::Infeasible) {
        if (verbose_) log_.log("Root LP infeasible.\n");
        MipResult result;
        result.status = Status::Infeasible;
        result.nodes = 1;
        result.lp_iterations = total_lp_iters;
        result.work_units = total_work;
        result.time_seconds = elapsed();
        return result;
    }

    if (root_result.status == Status::Unbounded) {
        if (verbose_) log_.log("Root LP unbounded.\n");
        MipResult result;
        result.status = Status::Unbounded;
        result.nodes = 1;
        result.lp_iterations = total_lp_iters;
        result.work_units = total_work;
        result.time_seconds = elapsed();
        return result;
    }

    if (root_result.status != Status::Optimal) {
        if (verbose_) log_.log("Root LP failed: status %d\n", static_cast<int>(root_result.status));
        MipResult result;
        result.status = Status::Error;
        result.nodes = 1;
        result.lp_iterations = total_lp_iters;
        result.work_units = total_work;
        result.time_seconds = elapsed();
        return result;
    }

    Real root_bound = root_result.objective;

    if (verbose_) {
        std::fflush(stdout);  // flush LP solver's printf output before Logger write()
        log_.log("Root LP (%s): obj = %.10e, %d iters\n",
                 rootPolicyName(root_lp_policy_), root_bound, root_result.iterations);
    }

    // Dual simplex supports reliable warm-starts for node LPs only after at least one solve.
    // In non-dual root mode, run one simplex solve to initialize basis state for the tree.
    if (!root_used_dual) {
        auto sync = lp.solve();
        total_lp_iters += sync.iterations;
        total_work += sync.work_units;
        if (sync.status == Status::Infeasible) {
            if (verbose_) {
                log_.log("Root sync LP infeasible after non-dual root solve.\n");
            }
            MipResult result;
            result.status = Status::Infeasible;
            result.nodes = 1;
            result.lp_iterations = total_lp_iters;
            result.work_units = total_work;
            result.time_seconds = elapsed();
            return result;
        }
        if (sync.status == Status::Unbounded) {
            if (verbose_) {
                log_.log("Root sync LP unbounded after non-dual root solve.\n");
            }
            MipResult result;
            result.status = Status::Unbounded;
            result.nodes = 1;
            result.lp_iterations = total_lp_iters;
            result.work_units = total_work;
            result.time_seconds = elapsed();
            return result;
        }
        if (sync.status != Status::Optimal) {
            if (verbose_) {
                log_.log("Root sync LP failed: status %d\n", static_cast<int>(sync.status));
            }
            MipResult result;
            result.status = Status::Error;
            result.nodes = 1;
            result.lp_iterations = total_lp_iters;
            result.work_units = total_work;
            result.time_seconds = elapsed();
            return result;
        }
        root_bound = sync.objective;
        root_primals = lp.getPrimalValues();
        root_basis = lp.getBasis();
    }

    // Run cutting planes at root (suppress LP iteration logs — we log per round).
    lp.setVerbose(false);
    if (cuts_enabled_ && problem_.hasIntegers()) {
        if (root_used_dual || !root_basis.empty()) {
            t0 = std::chrono::steady_clock::now();
            Int cuts_added = runCuttingPlanes(lp, total_lp_iters, total_work);
            lp_stats_.root_cut_lp_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();
            if (cuts_added > 0) {
                root_bound = lp.getObjective();
                root_primals = lp.getPrimalValues();
                root_basis = lp.getBasis();
            }
        }
    }

    // Check if root solution is integer feasible.
    root_basis = lp.getBasis();
    Real incumbent = kInf;
    std::vector<Real> best_solution;
    HeuristicRuntimeConfig runtime_config = makeHeuristicRuntimeConfig();
    SolutionPool solution_pool(problem_.sense);
    HeuristicRuntime root_runtime(runtime_config);

    if (isFeasibleLp(root_primals) && isFeasibleMip(root_primals)) {
        incumbent = root_bound;
        best_solution = root_primals;
        solution_pool.submit({best_solution, incumbent}, "root_lp", 0);
        if (verbose_) log_.log("Root solution is integer feasible!\n");
        MipResult result;
        result.status = Status::Optimal;
        result.objective = incumbent;
        result.best_bound = root_bound;
        result.gap = 0.0;
        result.nodes = 1;
        result.lp_iterations = total_lp_iters;
        result.work_units = total_work;
        result.time_seconds = elapsed();
        result.solution = std::move(best_solution);
        return result;
    }

    // Root heuristic runtime.
    Int root_int_inf = 0;
    Int root_int_vars = 0;
    for (Index j = 0; j < problem_.num_cols; ++j) {
        if (problem_.col_type[j] == VarType::Continuous) continue;
        ++root_int_vars;
        if (!isIntegral(root_primals[j], kIntTol)) ++root_int_inf;
    }
    const RootHeuristicContext root_ctx{
        .problem = problem_,
        .lp = lp,
        .primals = root_primals,
        .root_int_inf = root_int_inf,
        .root_int_vars = root_int_vars,
        .node_count = 0,
        .thread_id = 0,
        .total_work_units = total_work,
        .solution_pool = &solution_pool,
    };
    const auto root_heur_outcome =
        root_runtime.runRootPortfolio(root_ctx, incumbent, best_solution);
    total_lp_iters += root_heur_outcome.lp_iterations;
    total_work += root_heur_outcome.work_units;

    const bool root_basis_dirty = root_heur_outcome.basis_dirty;
    if (root_basis_dirty && !root_basis.empty()) {
        lp.setBasis(root_basis);
    }

    // Suppress LP iteration logs during tree search.
    lp.setVerbose(false);

    // Branch-and-bound.
    if (verbose_) {
        log_.log("\n%10s  %8s  %6s  %6s  %14s  %14s  %7s  %5s\n",
                 "Nodes", "Active", "LPit/n", "IntInf",
                 "BestBound", "BestSolution", "Gap", "Time");
    }

    NodeQueue queue(NodePolicy::BestFirst);

    // Create root children.
    BnbNode root_node;
    root_node.id = 0;
    root_node.depth = 0;
    root_node.lp_bound = root_bound;
    root_node.basis = root_basis;

    Index branch_var = -1;
    {
        std::lock_guard<std::mutex> lock(branching_mutex_);
        auto selection = branching_rule_.select(lp, problem_,
                                                root_primals,
                                                problem_.col_lower,
                                                problem_.col_upper,
                                                root_bound,
                                                true,
                                                branching_stats_);
        branch_var = selection.variable;
    }

    if (branch_var >= 0) {
        auto [left, right] = createChildren(root_node, branch_var,
                                            root_primals[branch_var]);
        left.lp_bound = root_bound;
        right.lp_bound = root_bound;
        queue.push(std::move(left));
        queue.push(std::move(right));
    }

    Int nodes_explored = 1;  // root counts

    // Choose serial or parallel tree search.
    bool use_parallel = false;
#ifdef MIPX_HAS_TBB
    use_parallel = (num_threads_ > 1);
#endif

    if (use_parallel) {
        solveParallel(lp, queue, nodes_explored, total_lp_iters,
                      total_work, runtime_config, solution_pool,
                      incumbent, best_solution, root_bound, elapsed);
    } else {
        solveSerial(lp, queue, nodes_explored, total_lp_iters,
                    total_work, root_runtime, solution_pool, incumbent,
                    best_solution, root_bound, elapsed);
    }

    if (best_solution.empty()) {
        auto pooled = solution_pool.bestSolution();
        if (pooled.has_value()) {
            incumbent = pooled->objective;
            best_solution = std::move(pooled->values);
        }
    }

    // Build result.
    MipResult result;
    result.lp_iterations = total_lp_iters;
    result.work_units = total_work;
    result.nodes = nodes_explored;
    result.time_seconds = elapsed();

    if (incumbent < kInf) {
        result.objective = incumbent;
        result.solution = std::move(best_solution);
        Real final_bound = queue.empty() ? incumbent : queue.bestBound();
        result.best_bound = final_bound;
        result.gap = computeGap(incumbent, final_bound);

        if (queue.empty() || result.gap < gap_tol_) {
            result.status = Status::Optimal;
        } else if (nodes_explored >= node_limit_) {
            result.status = Status::NodeLimit;
        } else {
            result.status = Status::TimeLimit;
        }
    } else {
        if (queue.empty()) {
            result.status = Status::Infeasible;
        } else if (nodes_explored >= node_limit_) {
            result.status = Status::NodeLimit;
        } else {
            result.status = Status::TimeLimit;
        }
        result.best_bound = queue.empty() ? kInf : queue.bestBound();
    }

    applyPostsolve(result);

    if (verbose_) {
        if (branching_stats_.selections > 0) {
            const double avg_probe_iters =
                branching_stats_.strong_branch_probes > 0
                    ? static_cast<double>(branching_stats_.strong_branch_probe_iters) /
                          static_cast<double>(branching_stats_.strong_branch_probes)
                    : 0.0;
            const double avg_probe_work =
                branching_stats_.strong_branch_probes > 0
                    ? branching_stats_.strong_branch_probe_work_units /
                          static_cast<double>(branching_stats_.strong_branch_probes)
                    : 0.0;
            const double pseudocost_hit_rate =
                branching_stats_.pseudocost_uses > 0
                    ? 100.0 * static_cast<double>(branching_stats_.pseudocost_hits) /
                          static_cast<double>(branching_stats_.pseudocost_uses)
                    : 0.0;
            log_.log("Branching: strong_calls=%d probes=%d avg_probe_lp_it=%.1f "
                     "avg_probe_work=%.2f pseudocost_uses=%d hit_rate=%.1f%%\n",
                     branching_stats_.strong_branch_calls,
                     branching_stats_.strong_branch_probes,
                     avg_probe_iters,
                     avg_probe_work,
                     branching_stats_.pseudocost_uses,
                     pseudocost_hit_rate);
        }
        char node_buf[16], iter_buf[16];
        Logger::formatCount(result.nodes, node_buf, sizeof(node_buf));
        Logger::formatCount(result.lp_iterations, iter_buf, sizeof(iter_buf));
        log_.log("\nExplored %s nodes, %s LP iterations, %.1fs\n",
                 node_buf, iter_buf, result.time_seconds);
        if (result.status == Status::Optimal) {
            log_.log("Optimal: %.10e\n", result.objective);
        } else if (incumbent < kInf) {
            log_.log("Best solution: %.10e (gap %.2f%%)\n",
                     result.objective, result.gap * 100.0);
        }
    }

    return result;
}

Int MipSolver::runCuttingPlanes(DualSimplexSolver& lp, Int& total_lp_iters, double& total_work) {
    if (cut_effort_mode_ == CutEffortMode::Off) return 0;

    CutPool pool;
    SeparatorManager separators;
    CutSeparationStats total_family_stats;
    CutManager cut_manager;
    cut_manager.setMode(cut_effort_mode_);
    cut_manager.setBaseLimits(max_cut_rounds_, max_cuts_per_round_);
    cut_manager.setBudgets(cut_per_node_work_budget_,
                           cut_per_round_work_budget_,
                           cut_global_work_budget_);
    cut_manager.resetNodeState(true, 0);
    cut_manager.setFamilyEnabled(CutFamily::Gomory, cut_family_config_.gomory);
    cut_manager.setFamilyEnabled(CutFamily::Mir, cut_family_config_.mir);
    cut_manager.setFamilyEnabled(CutFamily::Cover, cut_family_config_.cover);
    cut_manager.setFamilyEnabled(CutFamily::ImpliedBound, cut_family_config_.implied_bound);
    cut_manager.setFamilyEnabled(CutFamily::Clique, cut_family_config_.clique);
    cut_manager.setFamilyEnabled(CutFamily::ZeroHalf, cut_family_config_.zero_half);
    cut_manager.setFamilyEnabled(CutFamily::Mixing, cut_family_config_.mixing);

    auto userFamilyEnabled = [&](CutFamily family) -> bool {
        switch (family) {
            case CutFamily::Gomory: return cut_family_config_.gomory;
            case CutFamily::Mir: return cut_family_config_.mir;
            case CutFamily::Cover: return cut_family_config_.cover;
            case CutFamily::ImpliedBound: return cut_family_config_.implied_bound;
            case CutFamily::Clique: return cut_family_config_.clique;
            case CutFamily::ZeroHalf: return cut_family_config_.zero_half;
            case CutFamily::Mixing: return cut_family_config_.mixing;
            case CutFamily::Unknown:
            case CutFamily::Count:
            default: return false;
        }
    };

    Int total_cuts = 0;
    Int rounds_done = 0;
    Real start_obj = lp.getObjective();
    Int stagnation_rounds = 0;
    double cut_node_work = 0.0;

    for (Int round = 0; round < max_cut_rounds_; ++round) {
        const auto policy = cut_manager.beginRound(
            round, true, 0, cut_node_work, total_work);
        if (!policy.run) break;

        CutFamilyConfig round_config{};
        round_config.gomory = policy.family_enabled[static_cast<std::size_t>(CutFamily::Gomory)] &&
                              userFamilyEnabled(CutFamily::Gomory);
        round_config.mir = policy.family_enabled[static_cast<std::size_t>(CutFamily::Mir)] &&
                           userFamilyEnabled(CutFamily::Mir);
        round_config.cover = policy.family_enabled[static_cast<std::size_t>(CutFamily::Cover)] &&
                             userFamilyEnabled(CutFamily::Cover);
        round_config.implied_bound =
            policy.family_enabled[static_cast<std::size_t>(CutFamily::ImpliedBound)] &&
            userFamilyEnabled(CutFamily::ImpliedBound);
        round_config.clique = policy.family_enabled[static_cast<std::size_t>(CutFamily::Clique)] &&
                              userFamilyEnabled(CutFamily::Clique);
        round_config.zero_half =
            policy.family_enabled[static_cast<std::size_t>(CutFamily::ZeroHalf)] &&
            userFamilyEnabled(CutFamily::ZeroHalf);
        round_config.mixing = policy.family_enabled[static_cast<std::size_t>(CutFamily::Mixing)] &&
                              userFamilyEnabled(CutFamily::Mixing);
        separators.setConfig(round_config);
        separators.setMaxCutsPerFamily(std::max<Int>(1, policy.max_cuts_per_round));

        auto primals = lp.getPrimalValues();

        if (isFeasibleMip(primals)) break;

        Real prev_obj = lp.getObjective();

        CutSeparationStats round_family_stats;
        Int new_cuts = separators.separate(lp, problem_, primals, pool, round_family_stats);

        if (new_cuts == 0) break;

        auto top_indices = pool.topByEfficacy(policy.max_cuts_per_round);

        std::vector<Index> starts;
        std::vector<Index> col_indices;
        std::vector<Real> values;
        std::vector<Real> lower;
        std::vector<Real> upper;
        std::vector<const Cut*> selected_cuts;
        std::array<Int, static_cast<std::size_t>(CutFamily::Count)> selected_by_family{};
        selected_by_family.fill(0);
        Int selected_total = 0;

        for (Index idx : top_indices) {
            const auto& cut = pool[idx];
            if (cut.age > 0) continue;
            const auto fi = static_cast<std::size_t>(cut.family);
            if (!policy.family_enabled[fi]) continue;
            if (policy.per_family_cap[fi] > 0 &&
                selected_by_family[fi] >= policy.per_family_cap[fi]) {
                continue;
            }

            starts.push_back(static_cast<Index>(col_indices.size()));
            for (Index k = 0; k < static_cast<Index>(cut.indices.size()); ++k) {
                col_indices.push_back(cut.indices[k]);
                values.push_back(cut.values[k]);
            }
            lower.push_back(cut.lower);
            upper.push_back(cut.upper);
            selected_cuts.push_back(&cut);
            selected_by_family[fi] += 1;
            ++selected_total;
        }

        if (lower.empty()) break;

        Index cuts_this_round = static_cast<Index>(lower.size());

        lp.addRows(starts, col_indices, values, lower, upper);

        auto result = lp.solve();
        total_lp_iters += result.iterations;
        total_work += result.work_units;
        cut_node_work += result.work_units;

        if (result.status != Status::Optimal) break;

        total_cuts += cuts_this_round;
        rounds_done = round + 1;
        Real new_obj = result.objective;
        Real improvement = std::abs(new_obj - prev_obj);
        const Real orthogonality = averageOrthogonality(selected_cuts);
        const double separation_seconds = std::accumulate(
            round_family_stats.families.begin(),
            round_family_stats.families.end(),
            0.0,
            [](double acc, const CutFamilyStats& s) {
                return acc + s.time_seconds;
            });

        for (std::size_t fi = 0; fi < total_family_stats.families.size(); ++fi) {
            auto& dst = total_family_stats.families[fi];
            auto& src = round_family_stats.families[fi];
            if (selected_total > 0 && selected_by_family[fi] > 0) {
                src.lp_delta += improvement * static_cast<Real>(selected_by_family[fi]) /
                                static_cast<Real>(selected_total);
            }
            dst.attempted += src.attempted;
            dst.generated += src.generated;
            dst.accepted += src.accepted;
            dst.efficacy_sum += src.efficacy_sum;
            dst.lp_delta += src.lp_delta;
            dst.time_seconds += src.time_seconds;
        }
        cut_manager.recordRound(round_family_stats, selected_by_family,
                                improvement, orthogonality,
                                separation_seconds, result.work_units,
                                true, 0);

        auto new_primals = lp.getPrimalValues();
        pool.ageAll(new_primals);
        pool.purge(10);

        if (verbose_) {
            log_.log("  cut-round %d: selected=%d lp_delta=%.3e ortho=%.3f "
                     "reopt_work=%.1f policy={%s}\n",
                     round + 1, selected_total, improvement, orthogonality,
                     result.work_units, cut_manager.summarizeState().c_str());
        }

        if (result.work_units > cut_per_round_work_budget_) break;
        if (improvement < kCutImprovementTol) {
            ++stagnation_rounds;
            if (stagnation_rounds >= 2) break;
        } else {
            stagnation_rounds = 0;
        }
        if (cut_node_work >= cut_per_node_work_budget_) break;
        if (total_work >= cut_global_work_budget_) break;
    }

    if (verbose_ && total_cuts > 0) {
        Real end_obj = lp.getObjective();
        log_.log("Cutting planes (%s): %d rounds, %d cuts, obj %.10e -> %.10e\n",
                 cutEffortName(cut_effort_mode_), rounds_done, total_cuts,
                 start_obj, end_obj);
        for (std::size_t fi = 0; fi < total_family_stats.families.size(); ++fi) {
            const auto family = static_cast<CutFamily>(fi);
            if (family == CutFamily::Unknown || family == CutFamily::Count) continue;
            const auto& s = total_family_stats.families[fi];
            if (s.attempted == 0 && s.generated == 0 && s.accepted == 0) continue;
            const Real avg_eff = (s.accepted > 0)
                ? s.efficacy_sum / static_cast<Real>(s.accepted)
                : 0.0;
            log_.log("  cuts[%s]: attempted=%d generated=%d accepted=%d "
                     "avg_eff=%.3e lp_delta=%.3e time=%.3fs\n",
                     cutFamilyName(family), s.attempted, s.generated, s.accepted,
                     avg_eff, s.lp_delta, s.time_seconds);
        }
        for (std::size_t fi = 0; fi < cut_manager.kpis().size(); ++fi) {
            const auto family = static_cast<CutFamily>(fi);
            if (family == CutFamily::Unknown || family == CutFamily::Count) continue;
            const auto& kpi = cut_manager.kpis()[fi];
            if (kpi.attempted == 0) continue;
            log_.log("  policy[%s]: enabled=%d roi=%.3e ortho=%.3f "
                     "rejected=%d demotions=%d promotions=%d\n",
                     cutFamilyName(family),
                     kpi.enabled ? 1 : 0,
                     kpi.roi_ema,
                     kpi.orthogonality_ema,
                     kpi.rejected,
                     kpi.demotions,
                     kpi.promotions);
        }
    }

    return total_cuts;
}

}  // namespace mipx
