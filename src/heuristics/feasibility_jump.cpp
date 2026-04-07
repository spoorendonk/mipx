#include "mipx/heuristics.h"

#include <cmath>
#include <vector>

#include "mipx/dual_simplex.h"

namespace mipx {

namespace {

constexpr Real kFeasTol = 1e-6;
constexpr Real kObjectiveTol = 1e-6;

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

}  // namespace

std::optional<HeuristicSolution> FeasibilityJumpHeuristic::run(
    const LpProblem& problem,
    [[maybe_unused]] DualSimplexSolver& lp,
    std::span<const Real> primals,
    Real incumbent) {

    const Index n = problem.num_cols;
    if (n == 0) return std::nullopt;

    // Collect integer variable indices.
    std::vector<Index> int_vars;
    int_vars.reserve(n);
    for (Index j = 0; j < n; ++j) {
        if (isIntegerVar(problem.col_type[j])) int_vars.push_back(j);
    }
    if (int_vars.empty()) return std::nullopt;

    std::vector<Real> x(primals.begin(), primals.end());

    // Round integer variables to start.
    for (Index j : int_vars) {
        x[j] = std::round(primals[j]);
        x[j] = clampToBounds(x[j], problem.col_lower[j], problem.col_upper[j]);
    }

    std::optional<HeuristicSolution> best;

    for (Int restart = 0; restart < max_restarts_; ++restart) {
        // Compute row activities.
        std::vector<Real> activity(problem.num_rows, 0.0);
        for (Index i = 0; i < problem.num_rows; ++i) {
            auto row = problem.matrix.row(i);
            for (Index k = 0; k < row.size(); ++k) {
                activity[i] += row.values[k] * x[row.indices[k]];
            }
        }

        for (Int iter = 0; iter < max_iterations_; ++iter) {
            // Compute row violations.
            Real total_violation = 0.0;
            for (Index i = 0; i < problem.num_rows; ++i) {
                if (activity[i] < problem.row_lower[i] - kFeasTol) {
                    total_violation += problem.row_lower[i] - activity[i];
                }
                if (activity[i] > problem.row_upper[i] + kFeasTol) {
                    total_violation += activity[i] - problem.row_upper[i];
                }
            }

            if (total_violation <= kFeasTol) {
                // Feasible! Check if we beat incumbent.
                Real obj = computeObjective(problem, x);
                if (obj < incumbent - kObjectiveTol) {
                    if (!best || obj < best->objective - kObjectiveTol) {
                        best = HeuristicSolution{x, obj};
                        incumbent = obj;
                    }
                }
                break;
            }

            // Find the best single-variable flip that reduces violation most.
            Index best_var = -1;
            Real best_delta = 0.0;
            Real best_new_val = 0.0;

            for (Index j : int_vars) {
                Real current = x[j];

                // Try +1 and -1 moves.
                for (Real delta_val : {-1.0, 1.0}) {
                    Real new_val = current + delta_val;
                    if (problem.col_type[j] == VarType::Binary) {
                        new_val = (new_val >= 0.5) ? 1.0 : 0.0;
                    }
                    new_val = clampToBounds(new_val, problem.col_lower[j],
                                            problem.col_upper[j]);
                    if (std::abs(new_val - current) < 1e-12) continue;

                    // Compute change in violation incrementally using column j.
                    Real delta_violation = 0.0;
                    auto col = problem.matrix.col(j);
                    Real change = new_val - current;
                    for (Index k = 0; k < col.size(); ++k) {
                        Index row_idx = col.indices[k];
                        Real old_act = activity[row_idx];
                        Real new_act = old_act + col.values[k] * change;

                        // Old violation contribution.
                        Real old_viol = 0.0;
                        if (old_act < problem.row_lower[row_idx] - kFeasTol) {
                            old_viol = problem.row_lower[row_idx] - old_act;
                        }
                        if (old_act > problem.row_upper[row_idx] + kFeasTol) {
                            old_viol = old_act - problem.row_upper[row_idx];
                        }

                        // New violation contribution.
                        Real new_viol = 0.0;
                        if (new_act < problem.row_lower[row_idx] - kFeasTol) {
                            new_viol = problem.row_lower[row_idx] - new_act;
                        }
                        if (new_act > problem.row_upper[row_idx] + kFeasTol) {
                            new_viol = new_act - problem.row_upper[row_idx];
                        }

                        delta_violation += new_viol - old_viol;
                    }

                    if (delta_violation < best_delta - 1e-12) {
                        best_delta = delta_violation;
                        best_var = j;
                        best_new_val = new_val;
                    }
                }
            }

            if (best_var < 0 || best_delta >= -1e-12) {
                // No improving move found; break from this restart.
                break;
            }

            // Apply the best move.
            Real change = best_new_val - x[best_var];
            auto col = problem.matrix.col(best_var);
            for (Index k = 0; k < col.size(); ++k) {
                activity[col.indices[k]] += col.values[k] * change;
            }
            x[best_var] = best_new_val;
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
                    x[j] = clampToBounds(new_val, problem.col_lower[j],
                                          problem.col_upper[j]);
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
