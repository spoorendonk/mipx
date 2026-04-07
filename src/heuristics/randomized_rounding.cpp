#include "mipx/heuristics.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "mipx/dual_simplex.h"

namespace mipx {

namespace {

constexpr Real kFeasTol = 1e-6;
constexpr Real kObjectiveTol = 1e-6;

bool isIntegerVar(VarType t) {
    return t != VarType::Continuous;
}

bool isRowFeasible(const LpProblem& problem, std::span<const Real> values) {
    for (Index i = 0; i < problem.num_rows; ++i) {
        auto row = problem.matrix.row(i);
        Real activity = 0.0;
        for (Index k = 0; k < row.size(); ++k) {
            activity += row.values[k] * values[row.indices[k]];
        }
        if (activity < problem.row_lower[i] - kFeasTol) return false;
        if (activity > problem.row_upper[i] + kFeasTol) return false;
    }
    return true;
}

/// Simple deterministic PRNG for reproducibility without external state.
uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27U)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31U);
}

}  // namespace

std::optional<HeuristicSolution> RandomizedRoundingHeuristic::run(
    const LpProblem& problem,
    [[maybe_unused]] DualSimplexSolver& lp,
    std::span<const Real> primals,
    Real incumbent) {

    const Index n = problem.num_cols;
    std::optional<HeuristicSolution> best;

    uint64_t rng_state = 42;

    for (Int trial = 0; trial < num_trials_; ++trial) {
        std::vector<Real> x(primals.begin(), primals.end());

        // Randomized rounding: for each fractional integer variable,
        // round up with probability = fractionality, down otherwise.
        for (Index j = 0; j < n; ++j) {
            if (!isIntegerVar(problem.col_type[j])) continue;
            Real val = primals[j];
            Real frac = val - std::floor(val);
            if (frac < kFeasTol || frac > 1.0 - kFeasTol) {
                x[j] = std::round(val);
            } else {
                rng_state = splitmix64(rng_state + static_cast<uint64_t>(trial * n + j));
                Real threshold = static_cast<Real>(rng_state & 0xFFFFFFFFULL) /
                                 static_cast<Real>(0xFFFFFFFFULL);
                x[j] = (threshold < frac) ? std::ceil(val) : std::floor(val);
            }
            x[j] = std::max(x[j], problem.col_lower[j]);
            x[j] = std::min(x[j], problem.col_upper[j]);
        }

        // Feasibility repair passes: fix violated constraints by shifting.
        for (Int pass = 0; pass < max_repair_passes_; ++pass) {
            bool changed = false;
            for (Index i = 0; i < problem.num_rows; ++i) {
                auto row = problem.matrix.row(i);
                Real activity = 0.0;
                for (Index k = 0; k < row.size(); ++k) {
                    activity += row.values[k] * x[row.indices[k]];
                }

                Real violation = 0.0;
                bool need_decrease = false;
                if (activity > problem.row_upper[i] + kFeasTol) {
                    violation = activity - problem.row_upper[i];
                    need_decrease = true;
                } else if (activity < problem.row_lower[i] - kFeasTol) {
                    violation = problem.row_lower[i] - activity;
                    need_decrease = false;
                } else {
                    continue;
                }

                // Find the cheapest variable to shift.
                Index best_var = -1;
                Real best_shift = 0.0;
                Real best_cost = kInf;

                for (Index k = 0; k < row.size(); ++k) {
                    Index j = row.indices[k];
                    Real coeff = row.values[k];
                    if (std::abs(coeff) < 1e-12) continue;

                    Real shift = 0.0;
                    if (need_decrease) {
                        if (coeff > 0) {
                            Real avail = x[j] - problem.col_lower[j];
                            shift = -std::min(avail, violation / coeff);
                        } else {
                            Real avail = problem.col_upper[j] - x[j];
                            shift = std::min(avail, violation / (-coeff));
                        }
                    } else {
                        if (coeff > 0) {
                            Real avail = problem.col_upper[j] - x[j];
                            shift = std::min(avail, violation / coeff);
                        } else {
                            Real avail = x[j] - problem.col_lower[j];
                            shift = -std::min(avail, violation / (-coeff));
                        }
                    }

                    if (std::abs(shift) < 1e-12) continue;

                    // For integer vars, round shift.
                    Real new_val = x[j] + shift;
                    if (isIntegerVar(problem.col_type[j])) {
                        new_val = (shift > 0) ? std::ceil(new_val - kFeasTol)
                                              : std::floor(new_val + kFeasTol);
                        new_val = std::max(new_val, problem.col_lower[j]);
                        new_val = std::min(new_val, problem.col_upper[j]);
                        shift = new_val - x[j];
                        if (std::abs(shift) < 1e-12) continue;
                    }

                    Real cost = std::abs(problem.obj[j] * shift);
                    if (cost < best_cost) {
                        best_cost = cost;
                        best_var = j;
                        best_shift = shift;
                    }
                }

                if (best_var >= 0) {
                    x[best_var] += best_shift;
                    x[best_var] = std::max(x[best_var], problem.col_lower[best_var]);
                    x[best_var] = std::min(x[best_var], problem.col_upper[best_var]);
                    if (isIntegerVar(problem.col_type[best_var])) {
                        x[best_var] = std::round(x[best_var]);
                    }
                    changed = true;
                }
            }
            if (!changed) break;
        }

        if (!isRowFeasible(problem, x)) continue;

        // Check integrality.
        bool integral = true;
        for (Index j = 0; j < n; ++j) {
            if (!isIntegerVar(problem.col_type[j])) continue;
            if (std::abs(x[j] - std::round(x[j])) > kFeasTol) {
                integral = false;
                break;
            }
        }
        if (!integral) continue;

        Real obj = problem.obj_offset;
        for (Index j = 0; j < n; ++j) {
            obj += problem.obj[j] * x[j];
        }

        if (obj < incumbent - kObjectiveTol) {
            if (!best || obj < best->objective - kObjectiveTol) {
                best = HeuristicSolution{x, obj};
                incumbent = obj;
            }
        }
    }

    return best;
}

}  // namespace mipx
