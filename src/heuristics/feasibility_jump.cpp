#include "mipx/dual_simplex.h"
#include "mipx/heuristics.h"

#include <cmath>
#include <vector>

namespace mipx {

namespace {

constexpr Real kFeasTol = 1e-6;
constexpr Int kTabooTenure = 5;

bool isIntegerVar(VarType t) {
    return t != VarType::Continuous;
}

Real computeObjective(const LpProblem& problem, std::span<const Real> values) {
    Real obj = problem.obj_offset;
    for (Index j = 0; j < problem.num_cols; ++j) {
        obj += problem.obj[j] * values[j];
    }
    return obj;
}

Real clampToBounds(Real val, Real lb, Real ub) {
    return std::max(lb, std::min(val, ub));
}

/// Compute weighted violation for a single row given its activity.
Real rowViolation(Real act, Real lb, Real ub) {
    if (act < lb - kFeasTol) {
        return lb - act;
    }
    if (act > ub + kFeasTol) {
        return act - ub;
    }
    return 0.0;
}

}  // namespace

std::optional<HeuristicSolution> FeasibilityJumpHeuristic::run(
    const LpProblem& problem, [[maybe_unused]] DualSimplexSolver& lp, std::span<const Real> primals,
    Real incumbent) {
    const Index n = problem.num_cols;
    const Index m = problem.num_rows;
    if (n == 0) {
        return std::nullopt;
    }

    // Collect integer variable indices.
    std::vector<Index> int_vars;
    int_vars.reserve(n);
    for (Index j = 0; j < n; ++j) {
        if (isIntegerVar(problem.col_type[j])) {
            int_vars.push_back(j);
        }
    }
    if (int_vars.empty()) {
        return std::nullopt;
    }

    // The problem keeps its original sense; betterObjective(problem.sense, ...)
    // handles acceptance directly. obj_sign flips the objective term used for
    // tiebreaking so "more negative is better" holds for both senses.
    const Real obj_sign = (problem.sense == Sense::Maximize) ? -1.0 : 1.0;

    std::vector<Real> x(primals.begin(), primals.end());

    // Round integer variables to start.
    for (Index j : int_vars) {
        x[j] = std::round(primals[j]);
        x[j] = clampToBounds(x[j], problem.col_lower[j], problem.col_upper[j]);
    }

    std::optional<HeuristicSolution> best;

    for (Int restart = 0; restart < max_restarts_; ++restart) {
        // Constraint weights (Berthold-Salvagnin weight escalation).
        std::vector<Real> weight(m, 1.0);

        // Taboo: iteration at which each variable's taboo expires.
        std::vector<Int> taboo_until(n, -1);

        // Compute row activities.
        std::vector<Real> activity(m, 0.0);
        for (Index i = 0; i < m; ++i) {
            auto row = problem.matrix.row(i);
            for (Index k = 0; k < row.size(); ++k) {
                activity[i] += row.values[k] * x[row.indices[k]];
            }
        }

        for (Int iter = 0; iter < max_iterations_; ++iter) {
            // Check total (unweighted) violation for feasibility.
            Real total_violation = 0.0;
            for (Index i = 0; i < m; ++i) {
                total_violation +=
                    rowViolation(activity[i], problem.row_lower[i], problem.row_upper[i]);
            }

            if (total_violation <= kFeasTol) {
                // Feasible! Check if we beat incumbent.
                Real obj = computeObjective(problem, x);
                if (betterObjective(problem.sense, obj, incumbent)) {
                    if (!best || betterObjective(problem.sense, obj, best->objective)) {
                        best = HeuristicSolution{x, obj};
                        incumbent = obj;
                    }
                }
                break;
            }

            // Find the best move among integer variables.
            // Also track the best "jump" variable (highest weighted score,
            // regardless of whether the move improves).
            Index best_var = -1;
            Real best_delta_score = 0.0;  // Must be negative to count as improving.
            Real best_obj_change = 0.0;
            Real best_new_val = 0.0;

            // Jump candidate: variable with highest weighted score (for
            // diversification when no improving move exists).
            Index jump_var = -1;
            Real jump_score = -1.0;
            Real jump_new_val = 0.0;

            for (Index j : int_vars) {
                Real current = x[j];
                auto col = problem.matrix.col(j);

                // Weighted score of this variable: sum of weights of the
                // violated constraints it participates in. Independent of the
                // move direction, so computed once per variable.
                Real var_score = 0.0;
                for (Index k = 0; k < col.size(); ++k) {
                    Index row_idx = col.indices[k];
                    if (rowViolation(activity[row_idx], problem.row_lower[row_idx],
                                     problem.row_upper[row_idx]) > 0.0) {
                        var_score += weight[row_idx];
                    }
                }

                for (Real delta_val : {-1.0, 1.0}) {
                    Real new_val = current + delta_val;
                    if (problem.col_type[j] == VarType::Binary) {
                        new_val = (new_val >= 0.5) ? 1.0 : 0.0;
                    }
                    new_val = clampToBounds(new_val, problem.col_lower[j], problem.col_upper[j]);
                    if (std::abs(new_val - current) < 1e-12) {
                        continue;
                    }

                    // Compute change in weighted violation incrementally.
                    Real delta_weighted = 0.0;
                    Real change = new_val - current;
                    for (Index k = 0; k < col.size(); ++k) {
                        Index row_idx = col.indices[k];
                        Real old_act = activity[row_idx];
                        Real new_act = old_act + col.values[k] * change;

                        Real old_viol = rowViolation(old_act, problem.row_lower[row_idx],
                                                     problem.row_upper[row_idx]);
                        Real new_viol = rowViolation(new_act, problem.row_lower[row_idx],
                                                     problem.row_upper[row_idx]);

                        delta_weighted += weight[row_idx] * (new_viol - old_viol);
                    }

                    // Objective change for tiebreaking.
                    Real obj_change = obj_sign * problem.obj[j] * change;

                    // Track best jump candidate (highest var_score, ignoring
                    // taboo — jumps bypass taboo).
                    if (var_score > jump_score) {
                        jump_score = var_score;
                        jump_var = j;
                        jump_new_val = new_val;
                    }

                    // For normal move selection, skip taboo variables.
                    if (taboo_until[j] > iter) {
                        continue;
                    }

                    // Is this an improving move?
                    bool is_better = false;
                    if (delta_weighted < best_delta_score - 1e-12) {
                        is_better = true;
                    } else if (std::abs(delta_weighted - best_delta_score) <= 1e-12 &&
                               delta_weighted < -1e-12) {
                        // Equal improvement — prefer better objective.
                        if (obj_change < best_obj_change - 1e-12) {
                            is_better = true;
                        }
                    }

                    if (is_better) {
                        best_delta_score = delta_weighted;
                        best_obj_change = obj_change;
                        best_var = j;
                        best_new_val = new_val;
                    }
                }
            }

            // Decide: improving move or jump.
            Index move_var;
            Real move_new_val;

            if (best_var >= 0 && best_delta_score < -1e-12) {
                // We have an improving move.
                move_var = best_var;
                move_new_val = best_new_val;
            } else if (jump_var >= 0) {
                // No improving move — jump to diversify.
                move_var = jump_var;
                move_new_val = jump_new_val;
            } else {
                // No move possible at all.
                break;
            }

            // Apply the move.
            Real change = move_new_val - x[move_var];
            auto col = problem.matrix.col(move_var);
            for (Index k = 0; k < col.size(); ++k) {
                activity[col.indices[k]] += col.values[k] * change;
            }
            x[move_var] = move_new_val;

            // Add to taboo list.
            taboo_until[move_var] = iter + kTabooTenure;

            // Weight escalation: increment weights of all still-violated
            // constraints.
            for (Index i = 0; i < m; ++i) {
                if (rowViolation(activity[i], problem.row_lower[i], problem.row_upper[i]) > 0.0) {
                    weight[i] += 1.0;
                }
            }
        }

        // For subsequent restarts, perturb the solution by flipping some variables.
        if (restart + 1 < max_restarts_) {
            for (Index j : int_vars) {
                // Deterministic perturbation based on restart number and variable index.
                if (((restart * 7 + j * 13) % 5) == 0) {
                    Real new_val = x[j];
                    if (problem.col_type[j] == VarType::Binary) {
                        new_val = 1.0 - x[j];
                    } else {
                        new_val = x[j] + ((restart % 2 == 0) ? 1.0 : -1.0);
                    }
                    x[j] = clampToBounds(new_val, problem.col_lower[j], problem.col_upper[j]);
                    if (problem.col_type[j] == VarType::Integer) {
                        x[j] = std::round(x[j]);
                    }
                }
            }
        }
    }

    return best;
}

}  // namespace mipx
