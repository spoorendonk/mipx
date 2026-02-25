#include "mipx/mip_solver.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <print>
#include <thread>
#include <unordered_map>

#include "mipx/cut_pool.h"
#include "mipx/gomory.h"

#ifdef MIPX_HAS_TBB
#include <tbb/parallel_for.h>
#include <tbb/task_group.h>
#endif

namespace mipx {

void MipSolver::load(const LpProblem& problem) {
    problem_ = problem;
    loaded_ = true;
}

bool MipSolver::isFeasibleMip(const std::vector<Real>& primals) const {
    for (Index j = 0; j < problem_.num_cols; ++j) {
        if (problem_.col_type[j] == VarType::Continuous) continue;
        if (!isIntegral(primals[j], kIntTol)) return false;
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
                             double elapsed) const {
    if (!verbose_) return;

    if (incumbent < kInf) {
        Real gap = computeGap(incumbent, best_bound) * 100.0;
        std::println("  {:>8d}  {:>8d}  {:>10d}  {:>14.6e}  {:>14.6e}  {:>7.2f}%  {:>7.1f}s",
                     nodes, open, lp_iters, incumbent, best_bound, gap, elapsed);
    } else {
        std::println("  {:>8d}  {:>8d}  {:>10d}  {:>14s}  {:>14.6e}  {:>7s}  {:>7.1f}s",
                     nodes, open, lp_iters, "-", best_bound, "-", elapsed);
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
                             NodeWorkStats& node_stats) {
    children_out.clear();
    node_iters_out = 0;
    node_work_out = 0.0;
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

    // Check integer feasibility.
    if (isFeasibleMip(node_primals_out)) {
        return false;  // Caller handles incumbent update.
    }

    // Branch.
    MostFractionalBranching branching;
    Index branch_var = branching.select(node_primals_out, problem_.col_type,
                                        problem_.col_lower, problem_.col_upper);
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
                             Real& incumbent, std::vector<Real>& best_solution,
                             Real root_bound,
                             const std::function<double()>& elapsed) {
    std::vector<Real> current_lower = problem_.col_lower;
    std::vector<Real> current_upper = problem_.col_upper;
    std::vector<Index> touched_vars;
    touched_vars.reserve(problem_.num_cols);
    NodeWorkStats node_stats;

    while (!queue.empty()) {
        if (nodes_explored >= node_limit_) {
            if (verbose_) std::println("Node limit reached.");
            break;
        }
        if (elapsed() >= time_limit_) {
            if (verbose_) std::println("Time limit reached.");
            break;
        }

        if (incumbent < kInf) {
            queue.prune(incumbent - 1e-6);
            if (queue.empty()) break;
        }

        Real best_bound = queue.bestBound();
        if (incumbent < kInf && computeGap(incumbent, best_bound) < gap_tol_) {
            if (verbose_) std::println("Gap tolerance reached.");
            break;
        }

        BnbNode node = queue.pop();
        ++nodes_explored;

        std::vector<BnbNode> children;
        Real node_obj = 0.0;
        std::vector<Real> node_primals;
        Int node_iters = 0;
        double node_work = 0.0;

        bool branched = processNode(lp, node, incumbent,
                                    children, node_obj, node_primals, node_iters,
                                    node_work,
                                    current_lower, current_upper, touched_vars,
                                    node_stats);
        total_lp_iters += node_iters;
        total_work += node_work;

        // Log progress periodically.
        if (verbose_ && (nodes_explored % kLogFrequency == 0 || nodes_explored <= 10)) {
            logProgress(nodes_explored, queue.size(), total_lp_iters,
                       incumbent,
                       queue.empty() ? (incumbent < kInf ? incumbent : root_bound) : queue.bestBound(),
                       elapsed());
        }

        if (!branched && !node_primals.empty() && isFeasibleMip(node_primals)) {
            if (node_obj < incumbent) {
                incumbent = node_obj;
                best_solution = node_primals;
                if (verbose_) {
                    std::println("  * New incumbent: {:.10e} (node {}, depth {})",
                                 incumbent, nodes_explored, node.depth);
                }
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
}

#ifdef MIPX_HAS_TBB
void MipSolver::solveParallel(const DualSimplexSolver& root_lp, NodeQueue& queue,
                               Int& nodes_explored, Int& total_lp_iters,
                               double& total_work,
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

    // Each thread needs its own LP solver instance.
    Int actual_threads = std::min(num_threads_, static_cast<Int>(std::thread::hardware_concurrency()));
    if (actual_threads < 1) actual_threads = 1;

    if (verbose_) {
        std::println("Parallel tree search with {} threads", actual_threads);
    }

    // Worker function.
    auto worker = [&](Int /*thread_id*/) {
        // Create thread-local LP solver by loading problem fresh.
        DualSimplexSolver local_lp;
        local_lp.load(problem_);
        // Warm-start from root basis.
        local_lp.setBasis(root_lp.getBasis());
        std::vector<Real> current_lower = problem_.col_lower;
        std::vector<Real> current_upper = problem_.col_upper;
        std::vector<Index> touched_vars;
        touched_vars.reserve(problem_.num_cols);
        NodeWorkStats local_node_stats;

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

            bool branched = processNode(local_lp, node, inc_snapshot,
                                        children, node_obj, node_primals, node_iters,
                                        node_work,
                                        current_lower, current_upper, touched_vars,
                                        local_node_stats);
            atomic_lp_iters.fetch_add(node_iters, std::memory_order_relaxed);
            // Atomically add work (relaxed is fine for accumulation).
            auto old_w = atomic_work.load(std::memory_order_relaxed);
            while (!atomic_work.compare_exchange_weak(old_w, old_w + node_work,
                                                       std::memory_order_relaxed)) {}

            // Check for new incumbent.
            if (!branched && !node_primals.empty() && isFeasibleMip(node_primals)) {
                std::lock_guard<std::mutex> lock(incumbent_mutex);
                if (node_obj < incumbent) {
                    incumbent = node_obj;
                    best_solution = node_primals;
                }
            }

            // Push children to shared queue.
            if (!children.empty()) {
                std::lock_guard<std::mutex> lock(queue_mutex);
                for (auto& child : children) {
                    queue.push(std::move(child));
                }
            }

            // Log progress periodically (only from one thread to avoid garbled output).
            if (verbose_ && (node_num % kLogFrequency == 0)) {
                std::lock_guard<std::mutex> lock(queue_mutex);
                logProgress(atomic_nodes.load(), queue.size(),
                           atomic_lp_iters.load(),
                           incumbent,
                           queue.empty() ? (incumbent < kInf ? incumbent : root_bound) : queue.bestBound(),
                           elapsed());
            }
        }

        std::lock_guard<std::mutex> lock(stats_mutex);
        lp_stats_.node_bound_apply_seconds += local_node_stats.bound_apply_seconds;
        lp_stats_.node_basis_set_seconds += local_node_stats.basis_set_seconds;
        lp_stats_.node_lp_solve_seconds += local_node_stats.lp_solve_seconds;
        lp_stats_.nodes_solved += local_node_stats.nodes_solved;
        lp_stats_.warm_starts += local_node_stats.warm_starts;
        lp_stats_.cold_starts += local_node_stats.cold_starts;
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
        if (nodes_explored >= node_limit_) std::println("Node limit reached.");
        else if (elapsed() >= time_limit_) std::println("Time limit reached.");
    }
}
#else
void MipSolver::solveParallel(const DualSimplexSolver& /*root_lp*/, NodeQueue& queue,
                               Int& nodes_explored, Int& total_lp_iters,
                               double& total_work,
                               Real& incumbent, std::vector<Real>& best_solution,
                               Real root_bound,
                               const std::function<double()>& elapsed) {
    // Fallback to serial when TBB is not available.
    if (verbose_) {
        std::println("TBB not available, falling back to serial.");
    }
    // Need a mutable LP solver for serial path.
    DualSimplexSolver lp;
    lp.load(problem_);
    auto root_result = lp.solve();
    total_lp_iters += root_result.iterations;
    total_work += root_result.work_units;
    solveSerial(lp, queue, nodes_explored, total_lp_iters,
                total_work, incumbent, best_solution, root_bound, elapsed);
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
                std::println("Presolve: {} vars removed, {} rows removed, "
                             "{} bounds tightened, {} rounds",
                             stats.vars_removed, stats.rows_removed,
                             stats.bounds_tightened, stats.rounds);
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
        if (verbose_) std::println("Presolve detected infeasibility.");
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
        DualSimplexSolver lp;
        lp.load(problem_);
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
        }
        applyPostsolve(result);
        return result;
    }

    if (verbose_) {
        std::println("mipx MIP solver");
        std::println("Problem: {} ({} rows, {} cols, {} nonzeros)",
                     problem_.name, problem_.num_rows, problem_.num_cols,
                     problem_.matrix.numNonzeros());
        std::println("");
    }

    // Solve root LP.
    DualSimplexSolver lp;
    lp.load(problem_);
    if (root_lp_policy_ == RootLpPolicy::ConcurrentRootExperimental && verbose_) {
        std::println("Root concurrent mode requested, but alternate LP backend "
                     "is not integrated. Falling back to dual simplex.");
    }
    auto t0 = std::chrono::steady_clock::now();
    auto root_result = lp.solve();
    lp_stats_.root_lp_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();

    Int total_lp_iters = root_result.iterations;
    double total_work = root_result.work_units;

    if (root_result.status == Status::Infeasible) {
        if (verbose_) std::println("Root LP infeasible.");
        MipResult result;
        result.status = Status::Infeasible;
        result.nodes = 1;
        result.lp_iterations = total_lp_iters;
        result.work_units = total_work;
        result.time_seconds = elapsed();
        return result;
    }

    if (root_result.status == Status::Unbounded) {
        if (verbose_) std::println("Root LP unbounded.");
        MipResult result;
        result.status = Status::Unbounded;
        result.nodes = 1;
        result.lp_iterations = total_lp_iters;
        result.work_units = total_work;
        result.time_seconds = elapsed();
        return result;
    }

    if (root_result.status != Status::Optimal) {
        if (verbose_) std::println("Root LP failed: status {}", static_cast<int>(root_result.status));
        MipResult result;
        result.status = Status::Error;
        result.nodes = 1;
        result.lp_iterations = total_lp_iters;
        result.work_units = total_work;
        result.time_seconds = elapsed();
        return result;
    }

    Real root_bound = root_result.objective;
    auto root_primals = lp.getPrimalValues();

    if (verbose_) {
        std::println("Root LP: obj = {:.10e}, iters = {}", root_bound, root_result.iterations);
    }

    // Run cutting planes at root.
    if (cuts_enabled_ && problem_.hasIntegers()) {
        t0 = std::chrono::steady_clock::now();
        Int cuts_added = runCuttingPlanes(lp, total_lp_iters, total_work);
        lp_stats_.root_cut_lp_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        if (cuts_added > 0) {
            root_bound = lp.getObjective();
            root_primals = lp.getPrimalValues();
            if (verbose_) {
                std::println("After cuts: obj = {:.10e}, {} cuts added",
                             root_bound, cuts_added);
            }
        }
    }

    // Check if root solution is integer feasible.
    Real incumbent = kInf;
    std::vector<Real> best_solution;

    if (isFeasibleMip(root_primals)) {
        incumbent = root_bound;
        best_solution = root_primals;
        if (verbose_) std::println("Root solution is integer feasible!");
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

    // Branch-and-bound.
    if (verbose_) {
        std::println("");
        std::println("  {:>8s}  {:>8s}  {:>10s}  {:>14s}  {:>14s}  {:>7s}  {:>7s}",
                     "Nodes", "Open", "LP iters", "Incumbent", "Best bound", "Gap", "Time");
    }

    NodeQueue queue(NodePolicy::BestFirst);
    MostFractionalBranching branching;

    // Create root children.
    auto root_basis = lp.getBasis();
    BnbNode root_node;
    root_node.id = 0;
    root_node.depth = 0;
    root_node.lp_bound = root_bound;
    root_node.basis = root_basis;

    Index branch_var = branching.select(
        root_primals, problem_.col_type,
        problem_.col_lower, problem_.col_upper);

    if (branch_var >= 0) {
        auto [left, right] = createChildren(root_node, branch_var,
                                            root_primals[branch_var]);
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
                      total_work, incumbent, best_solution, root_bound, elapsed);
    } else {
        solveSerial(lp, queue, nodes_explored, total_lp_iters,
                    total_work, incumbent, best_solution, root_bound, elapsed);
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
        std::println("");
        std::println("Explored {} nodes, {} LP iterations, {:.2f} work, {:.1f}s",
                     result.nodes, result.lp_iterations, result.work_units, result.time_seconds);
        if (result.status == Status::Optimal) {
            std::println("Optimal: {:.10e}", result.objective);
        } else if (incumbent < kInf) {
            std::println("Best solution: {:.10e} (gap {:.2f}%)",
                         result.objective, result.gap * 100.0);
        }
    }

    return result;
}

Int MipSolver::runCuttingPlanes(DualSimplexSolver& lp, Int& total_lp_iters, double& total_work) {
    CutPool pool;
    GomorySeparator gomory;
    gomory.setMaxCuts(max_cuts_per_round_);

    Int total_cuts = 0;

    for (Int round = 0; round < max_cut_rounds_; ++round) {
        auto primals = lp.getPrimalValues();

        if (isFeasibleMip(primals)) break;

        Real prev_obj = lp.getObjective();

        Int new_cuts = gomory.separate(lp, problem_, primals, pool);

        if (new_cuts == 0) {
            if (verbose_) {
                std::println("  Cut round {}: no cuts generated, stopping.", round + 1);
            }
            break;
        }

        auto top_indices = pool.topByEfficacy(max_cuts_per_round_);

        std::vector<Index> starts;
        std::vector<Index> col_indices;
        std::vector<Real> values;
        std::vector<Real> lower;
        std::vector<Real> upper;

        for (Index idx : top_indices) {
            const auto& cut = pool[idx];
            if (cut.age > 0) continue;

            starts.push_back(static_cast<Index>(col_indices.size()));
            for (Index k = 0; k < static_cast<Index>(cut.indices.size()); ++k) {
                col_indices.push_back(cut.indices[k]);
                values.push_back(cut.values[k]);
            }
            lower.push_back(cut.lower);
            upper.push_back(cut.upper);
        }

        if (lower.empty()) break;

        Index cuts_this_round = static_cast<Index>(lower.size());

        lp.addRows(starts, col_indices, values, lower, upper);

        auto result = lp.solve();
        total_lp_iters += result.iterations;
        total_work += result.work_units;

        if (result.status != Status::Optimal) {
            if (verbose_) {
                std::println("  Cut round {}: LP solve failed after adding cuts.", round + 1);
            }
            break;
        }

        total_cuts += cuts_this_round;
        Real new_obj = result.objective;
        Real improvement = std::abs(new_obj - prev_obj);

        if (verbose_) {
            std::println("  Cut round {}: {} cuts, obj {:.10e} -> {:.10e} (improve {:.2e})",
                         round + 1, cuts_this_round, prev_obj, new_obj, improvement);
        }

        auto new_primals = lp.getPrimalValues();
        pool.ageAll(new_primals);
        pool.purge(10);

        if (improvement < kCutImprovementTol) {
            if (verbose_) {
                std::println("  Improvement below tolerance, stopping cut rounds.");
            }
            break;
        }
    }

    return total_cuts;
}

}  // namespace mipx
