#include "mipx/heuristics.h"

#include <algorithm>
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

// ---------------------------------------------------------------------------
// 1-opt local search
// ---------------------------------------------------------------------------

std::optional<HeuristicSolution> OneOptHeuristic::run(
    const LpProblem& problem,
    [[maybe_unused]] DualSimplexSolver& lp,
    std::span<const Real> primals,
    Real incumbent) {
    return run(problem, primals, incumbent);
}

std::optional<HeuristicSolution> OneOptHeuristic::run(
    const LpProblem& problem,
    std::span<const Real> incumbent_values,
    Real incumbent) {

    const Index n = problem.num_cols;
    if (incumbent == kInf || static_cast<Index>(incumbent_values.size()) != n) {
        return std::nullopt;
    }

    std::vector<Real> x(incumbent_values.begin(), incumbent_values.end());

    // Pre-compute row activities.
    std::vector<Real> activity(problem.num_rows, 0.0);
    for (Index i = 0; i < problem.num_rows; ++i) {
        auto row = problem.matrix.row(i);
        for (Index k = 0; k < row.size(); ++k) {
            activity[i] += row.values[k] * x[row.indices[k]];
        }
    }

    bool improved = false;

    // Try +1/-1 moves for each integer variable.
    for (Index j = 0; j < n; ++j) {
        if (!isIntegerVar(problem.col_type[j])) continue;

        for (Real delta : {-1.0, 1.0}) {
            Real new_val = x[j] + delta;
            if (problem.col_type[j] == VarType::Binary) {
                new_val = (new_val >= 0.5) ? 1.0 : 0.0;
            }
            new_val = clampToBounds(new_val, problem.col_lower[j], problem.col_upper[j]);
            Real change = new_val - x[j];
            if (std::abs(change) < 1e-12) continue;

            // Check objective improvement.
            Real obj_change = problem.obj[j] * change;
            if (obj_change >= -kObjectiveTol) continue;  // Not improving.

            // Check feasibility incrementally.
            auto col = problem.matrix.col(j);
            bool feasible = true;
            for (Index k = 0; k < col.size(); ++k) {
                Index row_idx = col.indices[k];
                Real new_act = activity[row_idx] + col.values[k] * change;
                if (new_act < problem.row_lower[row_idx] - kFeasTol ||
                    new_act > problem.row_upper[row_idx] + kFeasTol) {
                    feasible = false;
                    break;
                }
            }

            if (feasible) {
                // Apply the move.
                for (Index k = 0; k < col.size(); ++k) {
                    activity[col.indices[k]] += col.values[k] * change;
                }
                x[j] = new_val;
                improved = true;
                break;  // Move to next variable.
            }
        }
    }

    if (!improved) return std::nullopt;

    Real obj = computeObjective(problem, x);
    if (obj >= incumbent - kObjectiveTol) return std::nullopt;

    return HeuristicSolution{std::move(x), obj};
}

// ---------------------------------------------------------------------------
// 2-opt local search
// ---------------------------------------------------------------------------

std::optional<HeuristicSolution> TwoOptHeuristic::run(
    const LpProblem& problem,
    [[maybe_unused]] DualSimplexSolver& lp,
    std::span<const Real> primals,
    Real incumbent) {
    return run(problem, primals, incumbent);
}

std::optional<HeuristicSolution> TwoOptHeuristic::run(
    const LpProblem& problem,
    std::span<const Real> incumbent_values,
    Real incumbent) {

    const Index n = problem.num_cols;
    if (incumbent == kInf || static_cast<Index>(incumbent_values.size()) != n) {
        return std::nullopt;
    }

    // Collect integer variables.
    std::vector<Index> int_vars;
    int_vars.reserve(n);
    for (Index j = 0; j < n; ++j) {
        if (isIntegerVar(problem.col_type[j])) int_vars.push_back(j);
    }
    if (static_cast<Int>(int_vars.size()) < 2) return std::nullopt;

    std::vector<Real> x(incumbent_values.begin(), incumbent_values.end());

    // Pre-compute row activities.
    std::vector<Real> activity(problem.num_rows, 0.0);
    for (Index i = 0; i < problem.num_rows; ++i) {
        auto row = problem.matrix.row(i);
        for (Index k = 0; k < row.size(); ++k) {
            activity[i] += row.values[k] * x[row.indices[k]];
        }
    }

    bool improved = false;
    Int pairs_checked = 0;

    // Try swaps of pairs of integer variables (one up, one down).
    for (Index a = 0; a < static_cast<Index>(int_vars.size()) && pairs_checked < max_pairs_; ++a) {
        Index ja = int_vars[a];
        for (Index b = a + 1; b < static_cast<Index>(int_vars.size()) && pairs_checked < max_pairs_; ++b) {
            ++pairs_checked;
            Index jb = int_vars[b];

            // Try all four combinations of +1/-1 moves.
            for (Real da : {-1.0, 1.0}) {
                for (Real db : {-1.0, 1.0}) {
                    Real new_a = x[ja] + da;
                    Real new_b = x[jb] + db;

                    if (problem.col_type[ja] == VarType::Binary) {
                        new_a = (new_a >= 0.5) ? 1.0 : 0.0;
                    }
                    if (problem.col_type[jb] == VarType::Binary) {
                        new_b = (new_b >= 0.5) ? 1.0 : 0.0;
                    }
                    new_a = clampToBounds(new_a, problem.col_lower[ja], problem.col_upper[ja]);
                    new_b = clampToBounds(new_b, problem.col_lower[jb], problem.col_upper[jb]);

                    Real change_a = new_a - x[ja];
                    Real change_b = new_b - x[jb];
                    if (std::abs(change_a) < 1e-12 && std::abs(change_b) < 1e-12) continue;

                    Real obj_change = problem.obj[ja] * change_a +
                                      problem.obj[jb] * change_b;
                    if (obj_change >= -kObjectiveTol) continue;

                    // Check feasibility.
                    // Temporarily apply changes to activity.
                    std::vector<Real> temp_activity = activity;
                    if (std::abs(change_a) > 1e-12) {
                        auto col_a = problem.matrix.col(ja);
                        for (Index k = 0; k < col_a.size(); ++k) {
                            temp_activity[col_a.indices[k]] += col_a.values[k] * change_a;
                        }
                    }
                    if (std::abs(change_b) > 1e-12) {
                        auto col_b = problem.matrix.col(jb);
                        for (Index k = 0; k < col_b.size(); ++k) {
                            temp_activity[col_b.indices[k]] += col_b.values[k] * change_b;
                        }
                    }

                    bool feasible = true;
                    for (Index i = 0; i < problem.num_rows; ++i) {
                        if (temp_activity[i] < problem.row_lower[i] - kFeasTol ||
                            temp_activity[i] > problem.row_upper[i] + kFeasTol) {
                            feasible = false;
                            break;
                        }
                    }

                    if (feasible) {
                        activity = std::move(temp_activity);
                        x[ja] = new_a;
                        x[jb] = new_b;
                        improved = true;
                        goto next_pair;
                    }
                }
            }
            next_pair:;
        }
    }

    if (!improved) return std::nullopt;

    Real obj = computeObjective(problem, x);
    if (obj >= incumbent - kObjectiveTol) return std::nullopt;

    return HeuristicSolution{std::move(x), obj};
}

}  // namespace mipx
