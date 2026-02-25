#include "mipx/mip_solver.h"

#include <algorithm>
#include <cmath>
#include <print>

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

MipResult MipSolver::solve() {
    if (!loaded_) return {};

    auto start_time = std::chrono::steady_clock::now();
    auto elapsed = [&]() {
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(now - start_time).count();
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
        result.time_seconds = elapsed();
        if (lr.status == Status::Optimal) {
            result.solution = lp.getPrimalValues();
        }
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
    auto root_result = lp.solve();

    Int total_lp_iters = root_result.iterations;

    if (root_result.status == Status::Infeasible) {
        if (verbose_) std::println("Root LP infeasible.");
        MipResult result;
        result.status = Status::Infeasible;
        result.nodes = 1;
        result.lp_iterations = total_lp_iters;
        result.time_seconds = elapsed();
        return result;
    }

    if (root_result.status == Status::Unbounded) {
        if (verbose_) std::println("Root LP unbounded.");
        MipResult result;
        result.status = Status::Unbounded;
        result.nodes = 1;
        result.lp_iterations = total_lp_iters;
        result.time_seconds = elapsed();
        return result;
    }

    if (root_result.status != Status::Optimal) {
        if (verbose_) std::println("Root LP failed: status {}", static_cast<int>(root_result.status));
        MipResult result;
        result.status = Status::Error;
        result.nodes = 1;
        result.lp_iterations = total_lp_iters;
        result.time_seconds = elapsed();
        return result;
    }

    Real root_bound = root_result.objective;
    auto root_primals = lp.getPrimalValues();

    if (verbose_) {
        std::println("Root LP: obj = {:.10e}, iters = {}", root_bound, root_result.iterations);
    }

    // Check if root solution is integer feasible.
    Real incumbent = kInf;
    std::vector<Real> best_solution;

    if (isFeasibleMip(root_primals)) {
        incumbent = root_result.objective;
        best_solution = root_primals;
        if (verbose_) std::println("Root solution is integer feasible!");
        MipResult result;
        result.status = Status::Optimal;
        result.objective = incumbent;
        result.best_bound = root_bound;
        result.gap = 0.0;
        result.nodes = 1;
        result.lp_iterations = total_lp_iters;
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

    while (!queue.empty()) {
        // Check limits.
        if (nodes_explored >= node_limit_) {
            if (verbose_) std::println("Node limit reached.");
            break;
        }
        if (elapsed() >= time_limit_) {
            if (verbose_) std::println("Time limit reached.");
            break;
        }

        // Prune by bound.
        if (incumbent < kInf) {
            queue.prune(incumbent - 1e-6);
            if (queue.empty()) break;
        }

        // Check gap.
        Real best_bound = queue.bestBound();
        if (incumbent < kInf && computeGap(incumbent, best_bound) < gap_tol_) {
            if (verbose_) std::println("Gap tolerance reached.");
            break;
        }

        BnbNode node = queue.pop();
        ++nodes_explored;

        // Skip if pruned by bound (might have changed since push).
        if (incumbent < kInf && node.lp_bound >= incumbent - 1e-6) {
            continue;
        }

        // Apply bound changes from root.
        // Reset LP bounds to original, then apply accumulated changes.
        for (Index j = 0; j < problem_.num_cols; ++j) {
            lp.setColBounds(j, problem_.col_lower[j], problem_.col_upper[j]);
        }
        for (const auto& bc : node.bound_changes) {
            if (bc.is_upper) {
                // Upper bound change: x_j <= bound
                Real lb = problem_.col_lower[bc.variable];
                // Find current lower from earlier changes.
                for (const auto& prev : node.bound_changes) {
                    if (prev.variable == bc.variable && !prev.is_upper) {
                        lb = prev.bound;
                    }
                }
                lp.setColBounds(bc.variable, lb, bc.bound);
            } else {
                // Lower bound change: x_j >= bound
                Real ub = problem_.col_upper[bc.variable];
                // Find current upper from earlier changes.
                for (const auto& prev : node.bound_changes) {
                    if (prev.variable == bc.variable && prev.is_upper) {
                        ub = prev.bound;
                    }
                }
                lp.setColBounds(bc.variable, bc.bound, ub);
            }
        }

        // Set basis for warm-start.
        if (!node.basis.empty()) {
            lp.setBasis(node.basis);
        }

        // Solve node LP.
        auto node_result = lp.solve();
        total_lp_iters += node_result.iterations;

        // Log progress periodically.
        if (verbose_ && (nodes_explored % kLogFrequency == 0 || nodes_explored <= 10)) {
            logProgress(nodes_explored, queue.size(), total_lp_iters,
                       incumbent, queue.empty() ? (incumbent < kInf ? incumbent : root_bound) : queue.bestBound(),
                       elapsed());
        }

        if (node_result.status == Status::Infeasible) {
            continue;  // Pruned by infeasibility.
        }

        if (node_result.status != Status::Optimal) {
            continue;  // Error or limit — skip.
        }

        // Pruned by bound.
        if (incumbent < kInf && node_result.objective >= incumbent - 1e-6) {
            continue;
        }

        auto primals = lp.getPrimalValues();

        // Check integer feasibility.
        if (isFeasibleMip(primals)) {
            if (node_result.objective < incumbent) {
                incumbent = node_result.objective;
                best_solution = primals;
                if (verbose_) {
                    std::println("  * New incumbent: {:.10e} (node {}, depth {})",
                                 incumbent, nodes_explored, node.depth);
                }
                // Prune queue with new incumbent.
                queue.prune(incumbent - 1e-6);
            }
            continue;
        }

        // Branch.
        branch_var = branching.select(primals, problem_.col_type,
                                      problem_.col_lower, problem_.col_upper);
        if (branch_var < 0) {
            continue;  // All integers integral? Shouldn't happen if isFeasibleMip was false.
        }

        BnbNode solved_node = std::move(node);
        solved_node.lp_bound = node_result.objective;
        solved_node.basis = lp.getBasis();

        auto [left, right] = createChildren(solved_node, branch_var,
                                            primals[branch_var]);
        left.lp_bound = node_result.objective;   // Inherit parent bound as estimate.
        right.lp_bound = node_result.objective;

        queue.push(std::move(left));
        queue.push(std::move(right));
    }

    // Build result.
    MipResult result;
    result.lp_iterations = total_lp_iters;
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
        // No feasible solution found.
        if (queue.empty()) {
            result.status = Status::Infeasible;
        } else if (nodes_explored >= node_limit_) {
            result.status = Status::NodeLimit;
        } else {
            result.status = Status::TimeLimit;
        }
        result.best_bound = queue.empty() ? kInf : queue.bestBound();
    }

    if (verbose_) {
        std::println("");
        std::println("Explored {} nodes, {} LP iterations, {:.1f}s",
                     result.nodes, result.lp_iterations, result.time_seconds);
        if (result.status == Status::Optimal) {
            std::println("Optimal: {:.10e}", result.objective);
        } else if (incumbent < kInf) {
            std::println("Best solution: {:.10e} (gap {:.2f}%)",
                         result.objective, result.gap * 100.0);
        }
    }

    return result;
}

}  // namespace mipx
