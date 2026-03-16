#include "mipx/mip_solver.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <thread>
#include <unordered_map>

#ifdef __linux__
#include <unistd.h>
#endif

#include "mipx/cut_pool.h"
#include "mipx/barrier.h"
#include "mipx/heuristics.h"
#include "mipx/pdlp.h"
#include "mipx/symmetry.h"
#include "mipx/work_units.h"

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

const char* parallelModeName(ParallelMode mode) {
    switch (mode) {
        case ParallelMode::Deterministic: return "deterministic";
        case ParallelMode::Opportunistic: return "opportunistic";
        default: return "deterministic";
    }
}

HeuristicRuntimeMode heuristicRuntimeMode(ParallelMode mode) {
    return mode == ParallelMode::Opportunistic
        ? HeuristicRuntimeMode::Opportunistic
        : HeuristicRuntimeMode::Deterministic;
}

const char* searchProfileName(SearchProfile profile) {
    switch (profile) {
        case SearchProfile::Stable: return "stable";
        case SearchProfile::Default: return "default";
        case SearchProfile::Aggressive: return "aggressive";
        default: return "default";
    }
}

NodePolicy selectSearchPolicy(SearchProfile profile, Int stagnation_nodes) {
    switch (profile) {
        case SearchProfile::Stable:
            return NodePolicy::BestFirst;
        case SearchProfile::Aggressive:
            if (stagnation_nodes >= 24) return NodePolicy::DepthBiased;
            return NodePolicy::BestEstimate;
        case SearchProfile::Default:
        default:
            if (stagnation_nodes >= 64) return NodePolicy::DepthBiased;
            if (stagnation_nodes >= 20) return NodePolicy::BestEstimate;
            return NodePolicy::BestFirst;
    }
}

Int computeStrongBranchBudget(SearchProfile profile,
                              Int node_depth,
                              Int stagnation_nodes,
                              Int restart_stagnation_nodes) {
    Int strong_budget = 8;
    switch (profile) {
        case SearchProfile::Stable: strong_budget = 4; break;
        case SearchProfile::Default: strong_budget = 8; break;
        case SearchProfile::Aggressive: strong_budget = 12; break;
    }
    if (node_depth >= 8) strong_budget -= 2;
    if (stagnation_nodes >= restart_stagnation_nodes / 2) {
        strong_budget += (profile == SearchProfile::Aggressive) ? 4 : 2;
    }
    return std::clamp<Int>(strong_budget, 4, 24);
}

const char* exactRefinementModeName(ExactRefinementMode mode) {
    switch (mode) {
        case ExactRefinementMode::Off: return "off";
        case ExactRefinementMode::Auto: return "auto";
        case ExactRefinementMode::On: return "on";
        default: return "off";
    }
}

constexpr bool kLpLightCompiled =
#ifdef MIPX_HAS_LP_LIGHT
    true;
#else
    false;
#endif

const char* lpLightBackendName() {
    return kLpLightCompiled ? "dual" : "none";
}

uint64_t workUnitTicks(double work_units) {
    if (!std::isfinite(work_units) || work_units <= 0.0) return 0;
    return static_cast<uint64_t>(std::llround(work_units * 1e6));
}

enum class PreRootArm : int {
    FeasJump = 0,
    Fpr = 1,
    LocalMip = 2,
    LpLightFpr = 3,
    LpLightDiving = 4,
};

constexpr int kPreRootArmCount = 5;

constexpr std::size_t preRootArmIndex(PreRootArm arm) {
    return static_cast<std::size_t>(arm);
}

double sampleBeta(double alpha, double beta, std::mt19937_64& rng) {
    const double a = std::max(1e-6, alpha);
    const double b = std::max(1e-6, beta);
    std::gamma_distribution<double> ga(a, 1.0);
    std::gamma_distribution<double> gb(b, 1.0);
    const double x = ga(rng);
    const double y = gb(rng);
    if (x <= 0.0 && y <= 0.0) return 0.5;
    return x / (x + y);
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

Real cutLhs(const Cut& cut, std::span<const Real> primals) {
    Real lhs = 0.0;
    for (Index k = 0; k < static_cast<Index>(cut.indices.size()); ++k) {
        const Index j = cut.indices[k];
        if (j >= 0 && j < static_cast<Index>(primals.size())) {
            lhs += cut.values[k] * primals[j];
        }
    }
    return lhs;
}

Real cutDistanceToBinding(const Cut& cut, std::span<const Real> primals) {
    const Real lhs = cutLhs(cut, primals);
    Real dist = kInf;
    if (cut.lower > -kInf) dist = std::min(dist, std::abs(lhs - cut.lower));
    if (cut.upper < kInf) dist = std::min(dist, std::abs(lhs - cut.upper));
    return dist;
}

class ScopedRowRemoval {
public:
    ScopedRowRemoval(DualSimplexSolver& lp, std::vector<Index>& rows)
        : lp_(lp), rows_(rows) {}
    ~ScopedRowRemoval() {
        if (rows_.empty()) return;
        std::sort(rows_.begin(), rows_.end(), std::greater<Index>());
        lp_.removeRows(rows_);
        rows_.clear();
    }

private:
    DualSimplexSolver& lp_;
    std::vector<Index>& rows_;
};

constexpr Real kLpFreeFeasTol = 1e-6;

bool isDiscreteType(VarType t) {
    return t == VarType::Binary || t == VarType::Integer;
}

Real fallbackLowerBound(Real lb) {
    if (std::isfinite(lb)) return lb;
    return -16.0;
}

Real fallbackUpperBound(Real ub) {
    if (std::isfinite(ub)) return ub;
    return 16.0;
}

Real clampValue(Real v, Real lb, Real ub) {
    const Real lo = fallbackLowerBound(lb);
    const Real hi = std::max(lo, fallbackUpperBound(ub));
    return std::clamp(v, lo, hi);
}

bool betterObjective(Sense sense, Real candidate, Real incumbent) {
    if (!std::isfinite(incumbent) || incumbent >= kInf) return true;
    if (sense == Sense::Minimize) return candidate < incumbent - 1e-9;
    return candidate > incumbent + 1e-9;
}

Real objectiveValue(const LpProblem& problem, std::span<const Real> x) {
    Real obj = problem.obj_offset;
    for (Index j = 0; j < problem.num_cols; ++j) {
        obj += problem.obj[j] * x[j];
    }
    return obj;
}

void canonicalizePoint(const LpProblem& problem, std::vector<Real>& x) {
    if (x.size() != static_cast<std::size_t>(problem.num_cols)) {
        x.assign(problem.num_cols, 0.0);
    }
    for (Index j = 0; j < problem.num_cols; ++j) {
        Real value = clampValue(x[j], problem.col_lower[j], problem.col_upper[j]);
        if (problem.col_type[j] == VarType::Binary) {
            value = (value >= 0.5) ? 1.0 : 0.0;
        } else if (problem.col_type[j] == VarType::Integer) {
            value = std::round(value);
        }
        x[j] = clampValue(value, problem.col_lower[j], problem.col_upper[j]);
    }
}

void projectToBounds(const LpProblem& problem, std::vector<Real>& x) {
    if (x.size() != static_cast<std::size_t>(problem.num_cols)) {
        x.assign(problem.num_cols, 0.0);
    }
    for (Index j = 0; j < problem.num_cols; ++j) {
        x[j] = clampValue(x[j], problem.col_lower[j], problem.col_upper[j]);
    }
}

void randomInitializePoint(const LpProblem& problem,
                           std::span<const Index> discrete_vars,
                           std::mt19937_64& rng,
                           std::vector<Real>& x) {
    x.assign(problem.num_cols, 0.0);
    for (Index j = 0; j < problem.num_cols; ++j) {
        const Real lb = fallbackLowerBound(problem.col_lower[j]);
        const Real ub = std::max(lb, fallbackUpperBound(problem.col_upper[j]));
        if (problem.col_type[j] == VarType::Continuous) {
            x[j] = clampValue(0.5 * (lb + ub), problem.col_lower[j], problem.col_upper[j]);
        } else if (problem.col_type[j] == VarType::Binary) {
            x[j] = std::uniform_int_distribution<int>(0, 1)(rng) ? 1.0 : 0.0;
        } else {
            const int lo = static_cast<int>(std::ceil(lb - 1e-9));
            const int hi = static_cast<int>(std::floor(ub + 1e-9));
            if (lo <= hi) {
                x[j] = static_cast<Real>(std::uniform_int_distribution<int>(lo, hi)(rng));
            } else {
                x[j] = std::round(clampValue(0.0, lb, ub));
            }
        }
    }
    // Add a small deterministic perturbation to diversify restarts.
    for (Index j : discrete_vars) {
        if (problem.col_type[j] == VarType::Binary && std::uniform_int_distribution<int>(0, 7)(rng) == 0) {
            x[j] = 1.0 - x[j];
        }
    }
    canonicalizePoint(problem, x);
}

Real violationScore(const LpProblem& problem, std::span<const Real> x, double* work_units = nullptr) {
    Real violation = 0.0;
    for (Index j = 0; j < problem.num_cols; ++j) {
        if (x[j] < problem.col_lower[j] - kLpFreeFeasTol) {
            violation += problem.col_lower[j] - x[j];
        }
        if (x[j] > problem.col_upper[j] + kLpFreeFeasTol) {
            violation += x[j] - problem.col_upper[j];
        }
        if (problem.col_type[j] != VarType::Continuous) {
            violation += std::abs(x[j] - std::round(x[j]));
        }
    }

    for (Index i = 0; i < problem.num_rows; ++i) {
        const auto row = problem.matrix.row(i);
        Real lhs = 0.0;
        for (Index k = 0; k < row.size(); ++k) {
            lhs += row.values[k] * x[row.indices[k]];
        }
        if (work_units != nullptr) {
            *work_units += static_cast<double>(row.size()) * 1e-6;
        }
        if (lhs < problem.row_lower[i] - kLpFreeFeasTol) {
            violation += problem.row_lower[i] - lhs;
        }
        if (lhs > problem.row_upper[i] + kLpFreeFeasTol) {
            violation += lhs - problem.row_upper[i];
        }
    }
    return violation;
}

bool isFeasiblePoint(const LpProblem& problem, std::span<const Real> x, double* work_units = nullptr) {
    return violationScore(problem, x, work_units) <= 1e-7;
}

void greedyRepair(const LpProblem& problem,
                  std::vector<Real>& x,
                  Int max_passes,
                  double& work_units) {
    for (Int pass = 0; pass < max_passes; ++pass) {
        bool changed = false;
        for (Index i = 0; i < problem.num_rows; ++i) {
            const auto row = problem.matrix.row(i);
            Real lhs = 0.0;
            for (Index k = 0; k < row.size(); ++k) {
                lhs += row.values[k] * x[row.indices[k]];
            }
            work_units += static_cast<double>(row.size()) * 1e-6;

            auto applyAdjustment = [&](bool decrease_lhs, Real magnitude) {
                Index best_var = -1;
                Real best_coeff = 0.0;
                Real best_new_value = 0.0;
                for (Index k = 0; k < row.size(); ++k) {
                    const Index j = row.indices[k];
                    const Real coeff = row.values[k];
                    const bool discrete = isDiscreteType(problem.col_type[j]);
                    const Real lb = fallbackLowerBound(problem.col_lower[j]);
                    const Real ub = std::max(lb, fallbackUpperBound(problem.col_upper[j]));

                    Real candidate = x[j];
                    bool valid = false;
                    if (decrease_lhs) {
                        if (coeff > 1e-12 && x[j] > lb + kLpFreeFeasTol) {
                            const Real step = discrete
                                ? static_cast<Real>(std::max<Int>(1, static_cast<Int>(std::ceil(magnitude / coeff))))
                                : magnitude / coeff;
                            candidate = std::max(lb, x[j] - step);
                            valid = true;
                        } else if (coeff < -1e-12 && x[j] < ub - kLpFreeFeasTol) {
                            const Real step = discrete
                                ? static_cast<Real>(std::max<Int>(1, static_cast<Int>(std::ceil(magnitude / std::abs(coeff)))))
                                : magnitude / std::abs(coeff);
                            candidate = std::min(ub, x[j] + step);
                            valid = true;
                        }
                    } else {
                        if (coeff > 1e-12 && x[j] < ub - kLpFreeFeasTol) {
                            const Real step = discrete
                                ? static_cast<Real>(std::max<Int>(1, static_cast<Int>(std::ceil(magnitude / coeff))))
                                : magnitude / coeff;
                            candidate = std::min(ub, x[j] + step);
                            valid = true;
                        } else if (coeff < -1e-12 && x[j] > lb + kLpFreeFeasTol) {
                            const Real step = discrete
                                ? static_cast<Real>(std::max<Int>(1, static_cast<Int>(std::ceil(magnitude / std::abs(coeff)))))
                                : magnitude / std::abs(coeff);
                            candidate = std::max(lb, x[j] - step);
                            valid = true;
                        }
                    }
                    if (!valid) continue;

                    if (problem.col_type[j] == VarType::Binary) {
                        candidate = (candidate >= 0.5) ? 1.0 : 0.0;
                    } else if (problem.col_type[j] == VarType::Integer) {
                        candidate = std::round(candidate);
                    }
                    candidate = clampValue(candidate, problem.col_lower[j], problem.col_upper[j]);
                    if (std::abs(candidate - x[j]) <= 1e-12) continue;

                    if (std::abs(coeff) > std::abs(best_coeff)) {
                        best_coeff = coeff;
                        best_var = j;
                        best_new_value = candidate;
                    }
                }

                if (best_var >= 0) {
                    x[best_var] = best_new_value;
                    changed = true;
                }
            };

            if (lhs > problem.row_upper[i] + kLpFreeFeasTol && problem.row_upper[i] < kInf) {
                applyAdjustment(true, lhs - problem.row_upper[i]);
            }
            if (lhs < problem.row_lower[i] - kLpFreeFeasTol && problem.row_lower[i] > -kInf) {
                applyAdjustment(false, problem.row_lower[i] - lhs);
            }
        }
        canonicalizePoint(problem, x);
        if (!changed) break;
    }
}

std::optional<HeuristicSolution> runLpFreeFeasJump(
    const LpProblem& problem,
    std::span<const Index> discrete_vars,
    std::mt19937_64& rng,
    Real incumbent,
    double& work_units) {
    if (discrete_vars.empty()) return std::nullopt;

    std::vector<Real> x(problem.num_cols, 0.0);
    for (Int restart = 0; restart < 4; ++restart) {
        randomInitializePoint(problem, discrete_vars, rng, x);
        greedyRepair(problem, x, 6, work_units);
        if (isFeasiblePoint(problem, x, &work_units)) {
            const Real obj = objectiveValue(problem, x);
            if (betterObjective(problem.sense, obj, incumbent)) {
                return HeuristicSolution{.values = x, .objective = obj};
            }
        }

        // Single-variable jump pass.
        for (Index j : discrete_vars) {
            const Real current = x[j];
            Real best_value = current;
            Real best_violation = violationScore(problem, x, &work_units);
            const std::array<Real, 2> candidates = {
                clampValue(current - 1.0, problem.col_lower[j], problem.col_upper[j]),
                clampValue(current + 1.0, problem.col_lower[j], problem.col_upper[j]),
            };
            for (Real c : candidates) {
                if (problem.col_type[j] == VarType::Binary) c = (c >= 0.5) ? 1.0 : 0.0;
                if (problem.col_type[j] == VarType::Integer) c = std::round(c);
                if (std::abs(c - current) <= 1e-12) continue;
                x[j] = c;
                canonicalizePoint(problem, x);
                const Real v = violationScore(problem, x, &work_units);
                if (v + 1e-12 < best_violation) {
                    best_violation = v;
                    best_value = c;
                }
            }
            x[j] = best_value;
            canonicalizePoint(problem, x);
        }
        greedyRepair(problem, x, 4, work_units);
        if (isFeasiblePoint(problem, x, &work_units)) {
            const Real obj = objectiveValue(problem, x);
            if (betterObjective(problem.sense, obj, incumbent)) {
                return HeuristicSolution{.values = x, .objective = obj};
            }
        }
    }
    return std::nullopt;
}

std::optional<HeuristicSolution> runLpFreeFpr(
    const LpProblem& problem,
    std::span<const Index> discrete_vars,
    std::mt19937_64& rng,
    Real incumbent,
    double& work_units) {
    if (discrete_vars.empty()) return std::nullopt;

    std::vector<Real> x(problem.num_cols, 0.0);
    randomInitializePoint(problem, discrete_vars, rng, x);

    std::vector<std::pair<Real, Index>> ranked;
    ranked.reserve(discrete_vars.size());
    for (Index j : discrete_vars) {
        const Real degree = static_cast<Real>(problem.matrix.col(j).size());
        const Real score = std::abs(problem.obj[j]) * (1.0 + degree);
        ranked.push_back({score, j});
        work_units += degree * 1e-6;
    }
    std::sort(ranked.begin(), ranked.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    for (const auto& [score, j] : ranked) {
        (void)score;
        const Real lb = clampValue(problem.col_lower[j], problem.col_lower[j], problem.col_upper[j]);
        const Real ub = clampValue(problem.col_upper[j], problem.col_lower[j], problem.col_upper[j]);
        const bool prefer_upper = (problem.sense == Sense::Minimize)
            ? (problem.obj[j] < 0.0)
            : (problem.obj[j] > 0.0);
        const std::array<Real, 2> candidates = prefer_upper
            ? std::array<Real, 2>{ub, lb}
            : std::array<Real, 2>{lb, ub};
        Real best_choice = x[j];
        Real best_violation = kInf;
        for (Real c : candidates) {
            x[j] = c;
            canonicalizePoint(problem, x);
            greedyRepair(problem, x, 2, work_units);
            const Real v = violationScore(problem, x, &work_units);
            if (v < best_violation) {
                best_violation = v;
                best_choice = x[j];
            }
        }
        x[j] = best_choice;
        canonicalizePoint(problem, x);
    }

    greedyRepair(problem, x, 8, work_units);
    if (!isFeasiblePoint(problem, x, &work_units)) return std::nullopt;
    const Real obj = objectiveValue(problem, x);
    if (!betterObjective(problem.sense, obj, incumbent)) return std::nullopt;
    return HeuristicSolution{.values = std::move(x), .objective = obj};
}

std::optional<HeuristicSolution> runLpFreeLocalMip(
    const LpProblem& problem,
    std::span<const Index> discrete_vars,
    std::mt19937_64& rng,
    Real incumbent,
    std::span<const Real> incumbent_values,
    double& work_units) {
    if (discrete_vars.empty() || incumbent_values.empty()) return std::nullopt;
    std::vector<Real> x(incumbent_values.begin(), incumbent_values.end());
    canonicalizePoint(problem, x);
    if (!isFeasiblePoint(problem, x, &work_units)) {
        greedyRepair(problem, x, 6, work_units);
    }
    if (!isFeasiblePoint(problem, x, &work_units)) return std::nullopt;

    Real best_obj = objectiveValue(problem, x);
    bool improved = false;
    const Int iters = std::max<Int>(24, static_cast<Int>(discrete_vars.size()) * 4);
    for (Int iter = 0; iter < iters; ++iter) {
        const Index pos = std::uniform_int_distribution<Index>(
            0, static_cast<Index>(discrete_vars.size()) - 1)(rng);
        const Index j = discrete_vars[pos];
        const Real current = x[j];
        std::array<Real, 2> candidates = {
            clampValue(current - 1.0, problem.col_lower[j], problem.col_upper[j]),
            clampValue(current + 1.0, problem.col_lower[j], problem.col_upper[j]),
        };
        if (problem.col_type[j] == VarType::Binary) {
            candidates = {1.0 - current, current};
        }

        for (Real c : candidates) {
            std::vector<Real> trial = x;
            trial[j] = c;
            canonicalizePoint(problem, trial);
            greedyRepair(problem, trial, 2, work_units);
            if (!isFeasiblePoint(problem, trial, &work_units)) continue;
            const Real obj = objectiveValue(problem, trial);
            if (betterObjective(problem.sense, obj, best_obj)) {
                x = std::move(trial);
                best_obj = obj;
                improved = true;
            }
        }
    }

    if (!improved || !betterObjective(problem.sense, best_obj, incumbent)) {
        return std::nullopt;
    }
    return HeuristicSolution{.values = std::move(x), .objective = best_obj};
}

struct LpLightGuide {
    std::vector<Real> primals;
    std::vector<Real> reduced_costs;
    Int iterations = 0;
    double work_units = 0.0;
};

std::optional<LpLightGuide> solveLpLightGuide(const LpProblem& problem) {
#ifdef MIPX_HAS_LP_LIGHT
    DualSimplexSolver lp;
    lp.load(problem);
    lp.setVerbose(false);
    const auto result = lp.solve();
    if (result.status != Status::Optimal) return std::nullopt;
    LpLightGuide guide;
    guide.primals = lp.getPrimalValues();
    guide.reduced_costs = lp.getReducedCosts();
    guide.iterations = result.iterations;
    guide.work_units = result.work_units;
    return guide;
#else
    (void)problem;
    return std::nullopt;
#endif
}

std::optional<HeuristicSolution> runLpLightFpr(
    const LpProblem& problem,
    std::span<const Index> discrete_vars,
    std::span<const Real> lp_primals,
    std::span<const Real> reduced_costs,
    std::mt19937_64& rng,
    Real incumbent,
    double& work_units) {
    // Scylla-style LP-light FPR: LP-fractionality ranking + fix/propagate/repair,
    // with an in-repo mipx sub-MIP neighborhood polish step.
    if (discrete_vars.empty() || lp_primals.empty()) return std::nullopt;

    constexpr Real kViolationTol = 1e-6;
    constexpr Real kBoundTol = 1e-9;
    constexpr double kSubMipPolishWorkBudget = 2.5e4;

    auto rowViolation = [&](Index i, Real lhs) -> Real {
        Real v = 0.0;
        if (problem.row_lower[i] > -kInf && lhs < problem.row_lower[i] - kViolationTol) {
            v += problem.row_lower[i] - lhs;
        }
        if (problem.row_upper[i] < kInf && lhs > problem.row_upper[i] + kViolationTol) {
            v += lhs - problem.row_upper[i];
        }
        return v;
    };

    auto fallbackDiscreteValue = [&](Index j,
                                     Real lb,
                                     Real ub,
                                     bool high) -> Real {
        Real v = high ? ub : lb;
        if (problem.col_type[j] == VarType::Binary) {
            v = high ? 1.0 : 0.0;
        } else if (problem.col_type[j] == VarType::Integer) {
            v = std::round(v);
        }
        return clampValue(v, problem.col_lower[j], problem.col_upper[j]);
    };

    double submip_work_used = 0.0;
    auto localImproveWithMipSubproblem =
        [&](const HeuristicSolution& candidate) -> std::optional<HeuristicSolution> {
            if (candidate.values.empty() || discrete_vars.size() < 10) return std::nullopt;
            if (submip_work_used >= kSubMipPolishWorkBudget) return std::nullopt;

            LpProblem sub = problem;
            struct FixRank {
                Real score = 0.0;
                Index var = -1;
            };
            std::vector<FixRank> ranked;
            ranked.reserve(discrete_vars.size());
            for (Index j : discrete_vars) {
                const Real frac = std::abs(lp_primals[j] - std::round(lp_primals[j]));
                const Real rc = (j >= 0 && j < static_cast<Index>(reduced_costs.size()))
                    ? std::abs(reduced_costs[j])
                    : 0.0;
                // Prefer fixing variables that look stable in LP.
                ranked.push_back({(1.0 - frac) + 0.05 * rc, j});
            }
            std::sort(ranked.begin(), ranked.end(),
                      [](const FixRank& a, const FixRank& b) { return a.score > b.score; });

            const Int fix_count = std::max<Int>(
                1, static_cast<Int>(ranked.size() * 7 / 10));
            for (Int k = 0; k < fix_count; ++k) {
                const Index j = ranked[static_cast<std::size_t>(k)].var;
                Real v = candidate.values[j];
                if (problem.col_type[j] == VarType::Binary) {
                    v = (v >= 0.5) ? 1.0 : 0.0;
                } else if (problem.col_type[j] == VarType::Integer) {
                    v = std::round(v);
                }
                v = clampValue(v, problem.col_lower[j], problem.col_upper[j]);
                sub.col_lower[j] = v;
                sub.col_upper[j] = v;
            }

            MipSolver sub_solver;
            sub_solver.setVerbose(false);
            sub_solver.setPresolve(false);
            sub_solver.setCutsEnabled(false);
            sub_solver.setNodeLimit(128);
            sub_solver.setGapTolerance(0.0);
            sub_solver.setNumThreads(1);
            sub_solver.setHeuristicMode(HeuristicRuntimeMode::Deterministic);
            sub_solver.setHeuristicSeed(
                static_cast<uint64_t>(rng() ^ 0x9e3779b97f4a7c15ULL));
            sub_solver.setPreRootLpFreeEnabled(false);
            sub_solver.setPreRootLpLightEnabled(false);
            sub_solver.setPreRootPortfolioEnabled(false);
            sub_solver.load(sub);
            const auto sub_result = sub_solver.solve();
            const double sub_work = std::max(0.0, sub_result.work_units);
            submip_work_used += sub_work;
            work_units += sub_work;

            if (sub_result.solution.empty()) return std::nullopt;
            if (!isFeasiblePoint(problem, sub_result.solution, &work_units)) return std::nullopt;
            const Real obj = objectiveValue(problem, sub_result.solution);
            if (!betterObjective(problem.sense, obj, candidate.objective)) return std::nullopt;
            return HeuristicSolution{.values = sub_result.solution, .objective = obj};
        };

    struct RankedVar {
        Real score = 0.0;
        Index var = -1;
    };

    std::vector<RankedVar> base_ranked;
    base_ranked.reserve(discrete_vars.size());
    for (Index j : discrete_vars) {
        const Real frac = std::abs(lp_primals[j] - std::round(lp_primals[j]));
        const Real rc = (j >= 0 && j < static_cast<Index>(reduced_costs.size()))
            ? std::abs(reduced_costs[j])
            : 0.0;
        const Real degree = static_cast<Real>(problem.matrix.col(j).size());
        base_ranked.push_back({frac * (1.0 + rc + 0.1 * degree), j});
        work_units += (1.0 + degree) * 1e-6;
    }
    std::sort(base_ranked.begin(), base_ranked.end(),
              [](const RankedVar& a, const RankedVar& b) { return a.score > b.score; });

    std::optional<HeuristicSolution> best_candidate;

    const Int attempts = 4;
    const Int repair_budget = std::max<Int>(
        128, static_cast<Int>(12 * discrete_vars.size()));
    for (Int attempt = 0; attempt < attempts; ++attempt) {
        std::vector<Real> x(lp_primals.begin(), lp_primals.end());
        projectToBounds(problem, x);

        std::vector<Real> implied_lb(problem.col_lower.begin(), problem.col_lower.end());
        std::vector<Real> implied_ub(problem.col_upper.begin(), problem.col_upper.end());
        std::vector<char> fixed(problem.num_cols, 0);

        auto chooseFixValue = [&](Index j) -> Real {
            const Real lb = clampValue(implied_lb[j], problem.col_lower[j], problem.col_upper[j]);
            const Real ub = clampValue(implied_ub[j], problem.col_lower[j], problem.col_upper[j]);
            if (ub <= lb + kBoundTol) {
                return fallbackDiscreteValue(j, lb, ub, true);
            }

            const Real hint = clampValue(lp_primals[j], lb, ub);
            Real down = std::floor(hint);
            Real up = std::ceil(hint);
            if (problem.col_type[j] == VarType::Binary) {
                down = 0.0;
                up = 1.0;
            }
            down = clampValue(down, lb, ub);
            up = clampValue(up, lb, ub);
            const bool prefer_up = (problem.sense == Sense::Minimize)
                ? (problem.obj[j] < 0.0)
                : (problem.obj[j] > 0.0);
            Real value = prefer_up ? up : down;
            if (attempt > 0 &&
                std::uniform_int_distribution<int>(0, 9)(rng) == 0) {
                value = prefer_up ? down : up;
            }
            if (problem.col_type[j] == VarType::Integer) value = std::round(value);
            if (problem.col_type[j] == VarType::Binary) value = (value >= 0.5) ? 1.0 : 0.0;
            return clampValue(value, lb, ub);
        };

        auto propagateBounds = [&]() -> bool {
            bool changed = true;
            Int rounds = 0;
            while (changed && rounds < 24) {
                changed = false;
                ++rounds;
                for (Index i = 0; i < problem.num_rows; ++i) {
                    const auto row = problem.matrix.row(i);
                    Real fixed_sum = 0.0;
                    Real min_activity = 0.0;
                    Real max_activity = 0.0;
                    Index unfixed = 0;

                    for (Index k = 0; k < row.size(); ++k) {
                        const Index j = row.indices[k];
                        const Real a = row.values[k];
                        if (fixed[j]) {
                            fixed_sum += a * x[j];
                            continue;
                        }
                        ++unfixed;
                        const Real lb = implied_lb[j];
                        const Real ub = implied_ub[j];
                        if (a >= 0.0) {
                            min_activity += a * lb;
                            max_activity += a * ub;
                        } else {
                            min_activity += a * ub;
                            max_activity += a * lb;
                        }
                    }
                    work_units += static_cast<double>(row.size()) * 1e-6;
                    if (unfixed == 0) continue;

                    const Real row_lb = problem.row_lower[i];
                    const Real row_ub = problem.row_upper[i];
                    for (Index k = 0; k < row.size(); ++k) {
                        const Index j = row.indices[k];
                        const Real a = row.values[k];
                        if (fixed[j] || std::abs(a) <= 1e-15) continue;

                        const Real cur_lb = implied_lb[j];
                        const Real cur_ub = implied_ub[j];
                        Real new_lb = cur_lb;
                        Real new_ub = cur_ub;

                        const Real min_others = min_activity - (a >= 0.0 ? a * cur_lb : a * cur_ub);
                        const Real max_others = max_activity - (a >= 0.0 ? a * cur_ub : a * cur_lb);

                        if (row_ub < kInf) {
                            const Real rhs = row_ub - fixed_sum;
                            const Real b = (rhs - min_others) / a;
                            if (a > 0.0) new_ub = std::min(new_ub, b);
                            else new_lb = std::max(new_lb, b);
                        }
                        if (row_lb > -kInf) {
                            const Real rhs = row_lb - fixed_sum;
                            const Real b = (rhs - max_others) / a;
                            if (a > 0.0) new_lb = std::max(new_lb, b);
                            else new_ub = std::min(new_ub, b);
                        }

                        new_lb = std::max(new_lb, problem.col_lower[j]);
                        new_ub = std::min(new_ub, problem.col_upper[j]);
                        if (isDiscreteType(problem.col_type[j])) {
                            new_lb = std::ceil(new_lb - kViolationTol);
                            new_ub = std::floor(new_ub + kViolationTol);
                        }
                        if (new_lb > new_ub + kViolationTol) return false;
                        if (new_lb > implied_lb[j] + kBoundTol) {
                            implied_lb[j] = new_lb;
                            changed = true;
                        }
                        if (new_ub < implied_ub[j] - kBoundTol) {
                            implied_ub[j] = new_ub;
                            changed = true;
                        }
                        if (!fixed[j] && implied_ub[j] <= implied_lb[j] + kBoundTol) {
                            fixed[j] = 1;
                            x[j] = fallbackDiscreteValue(j, implied_lb[j], implied_ub[j], true);
                            changed = true;
                        }
                    }
                }
            }
            return true;
        };

        std::vector<Index> order;
        order.reserve(base_ranked.size());
        for (const auto& entry : base_ranked) order.push_back(entry.var);
        if (attempt > 0 && order.size() > 4) {
            const std::size_t prefix = std::max<std::size_t>(2, order.size() / 3);
            std::shuffle(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(prefix), rng);
        }

        for (Index j : order) {
            if (fixed[j]) continue;
            const auto saved_lb = implied_lb;
            const auto saved_ub = implied_ub;
            const auto saved_fixed = fixed;
            const auto saved_x = x;

            const Real first = chooseFixValue(j);
            x[j] = first;
            fixed[j] = 1;
            if (!propagateBounds()) {
                implied_lb = saved_lb;
                implied_ub = saved_ub;
                fixed = saved_fixed;
                x = saved_x;
                const Real alt = fallbackDiscreteValue(j, implied_lb[j], implied_ub[j], first <= 0.5);
                x[j] = alt;
                fixed[j] = 1;
                if (!propagateBounds()) {
                    implied_lb = saved_lb;
                    implied_ub = saved_ub;
                    fixed = saved_fixed;
                    x = saved_x;
                }
            }
        }

        for (Index j = 0; j < problem.num_cols; ++j) {
            if (fixed[j]) continue;
            if (problem.col_type[j] == VarType::Continuous) {
                const Real lb = clampValue(implied_lb[j], problem.col_lower[j], problem.col_upper[j]);
                const Real ub = clampValue(implied_ub[j], problem.col_lower[j], problem.col_upper[j]);
                if (problem.obj[j] > 0.0) {
                    x[j] = (problem.sense == Sense::Minimize) ? lb : ub;
                } else if (problem.obj[j] < 0.0) {
                    x[j] = (problem.sense == Sense::Minimize) ? ub : lb;
                } else {
                    x[j] = clampValue(lp_primals[j], lb, ub);
                }
            } else {
                x[j] = chooseFixValue(j);
            }
        }
        canonicalizePoint(problem, x);

        auto runWalksatRepair = [&]() -> bool {
            std::vector<Real> lhs(problem.num_rows, 0.0);
            std::vector<char> violated(problem.num_rows, 0);
            std::vector<Index> violated_rows;
            for (Index i = 0; i < problem.num_rows; ++i) {
                const auto row = problem.matrix.row(i);
                for (Index k = 0; k < row.size(); ++k) {
                    lhs[i] += row.values[k] * x[row.indices[k]];
                }
                work_units += static_cast<double>(row.size()) * 1e-6;
                if (rowViolation(i, lhs[i]) > kViolationTol) {
                    violated[i] = 1;
                    violated_rows.push_back(i);
                }
            }

            for (Int step = 0; step < repair_budget && !violated_rows.empty(); ++step) {
                const Index idx = std::uniform_int_distribution<Index>(
                    0, static_cast<Index>(violated_rows.size()) - 1)(rng);
                const Index row_id = violated_rows[static_cast<std::size_t>(idx)];
                const auto row = problem.matrix.row(row_id);
                if (row.size() == 0) continue;

                Index best_var = -1;
                Real best_value = 0.0;
                Real best_delta = kInf;

                for (Index k = 0; k < row.size(); ++k) {
                    const Index j = row.indices[k];
                    if (!isDiscreteType(problem.col_type[j])) continue;
                    const Real cur = x[j];
                    std::array<Real, 2> candidates{};
                    if (problem.col_type[j] == VarType::Binary) {
                        candidates = {0.0, 1.0};
                    } else if (problem.col_type[j] == VarType::Integer) {
                        const Real down = std::floor(cur);
                        const Real up = std::ceil(cur);
                        if (std::abs(up - down) <= kBoundTol) {
                            // If already integral, try neighboring lattice points.
                            candidates = {down - 1.0, up + 1.0};
                        } else {
                            candidates = {down, up};
                        }
                    } else {
                        candidates = {std::floor(cur), std::ceil(cur)};
                    }
                    for (Real cand : candidates) {
                        if (problem.col_type[j] == VarType::Integer) cand = std::round(cand);
                        if (problem.col_type[j] == VarType::Binary) cand = (cand >= 0.5) ? 1.0 : 0.0;
                        cand = clampValue(cand, implied_lb[j], implied_ub[j]);
                        if (std::abs(cand - cur) <= kBoundTol) continue;

                        const Real delta_x = cand - cur;
                        Real delta_viol = 0.0;
                        const auto col = problem.matrix.col(j);
                        for (Index p = 0; p < col.size(); ++p) {
                            const Index i2 = col.indices[p];
                            const Real old_lhs = lhs[i2];
                            const Real new_lhs = old_lhs + col.values[p] * delta_x;
                            const Real old_v = rowViolation(i2, old_lhs);
                            const Real new_v = rowViolation(i2, new_lhs);
                            delta_viol += (new_v - old_v);
                        }
                        work_units += static_cast<double>(col.size()) * 1e-6;

                        if (delta_viol < best_delta) {
                            best_delta = delta_viol;
                            best_var = j;
                            best_value = cand;
                        }
                    }
                }

                if (best_var < 0) continue;
                const Real delta_x = best_value - x[best_var];
                x[best_var] = best_value;

                const auto col = problem.matrix.col(best_var);
                for (Index p = 0; p < col.size(); ++p) {
                    const Index i2 = col.indices[p];
                    lhs[i2] += col.values[p] * delta_x;
                    const bool is_viol = rowViolation(i2, lhs[i2]) > kViolationTol;
                    if (violated[i2] && !is_viol) {
                        violated[i2] = 0;
                        const auto it = std::find(violated_rows.begin(), violated_rows.end(), i2);
                        if (it != violated_rows.end()) {
                            *it = violated_rows.back();
                            violated_rows.pop_back();
                        }
                    } else if (!violated[i2] && is_viol) {
                        violated[i2] = 1;
                        violated_rows.push_back(i2);
                    }
                }
                work_units += static_cast<double>(col.size()) * 1e-6;
            }

            return violated_rows.empty();
        };

        bool feasible = isFeasiblePoint(problem, x, &work_units);
        if (!feasible) {
            feasible = runWalksatRepair();
            if (feasible) {
                canonicalizePoint(problem, x);
                feasible = isFeasiblePoint(problem, x, &work_units);
            }
        }
        if (!feasible) continue;

        const Real obj = objectiveValue(problem, x);
        if (!betterObjective(problem.sense, obj, incumbent)) continue;

        HeuristicSolution candidate{.values = x, .objective = obj};
        if (const auto polished = localImproveWithMipSubproblem(candidate); polished.has_value()) {
            candidate = *polished;
        }

        if (!best_candidate.has_value() ||
            betterObjective(problem.sense, candidate.objective, best_candidate->objective)) {
            best_candidate = std::move(candidate);
        }
    }

    return best_candidate;
}

std::optional<HeuristicSolution> runLpLightDiving(
    const LpProblem& problem,
    std::span<const Index> discrete_vars,
    std::span<const Real> lp_primals,
    std::span<const Real> reduced_costs,
    Real incumbent,
    double& work_units) {
    if (discrete_vars.empty() || lp_primals.empty()) return std::nullopt;

    std::vector<Real> x(lp_primals.begin(), lp_primals.end());
    projectToBounds(problem, x);

    const Int max_steps = std::max<Int>(12, std::min<Int>(64, static_cast<Int>(discrete_vars.size())));
    for (Int step = 0; step < max_steps; ++step) {
        Index best_var = -1;
        Real best_frac = 0.0;
        for (Index j : discrete_vars) {
            const Real frac = std::abs(x[j] - std::round(x[j]));
            if (frac > best_frac + 1e-12) {
                best_frac = frac;
                best_var = j;
            }
        }
        if (best_var < 0 || best_frac <= 1e-9) break;

        const Real v = x[best_var];
        Real down = std::floor(v);
        Real up = std::ceil(v);
        down = clampValue(down, problem.col_lower[best_var], problem.col_upper[best_var]);
        up = clampValue(up, problem.col_lower[best_var], problem.col_upper[best_var]);
        if (problem.col_type[best_var] == VarType::Binary) {
            down = (down >= 0.5) ? 1.0 : 0.0;
            up = (up >= 0.5) ? 1.0 : 0.0;
        }

        const Real rc = (best_var >= 0 && best_var < static_cast<Index>(reduced_costs.size()))
            ? reduced_costs[best_var]
            : 0.0;
        const bool prefer_up = (problem.sense == Sense::Minimize)
            ? ((problem.obj[best_var] - rc) < 0.0)
            : ((problem.obj[best_var] - rc) > 0.0);
        x[best_var] = prefer_up ? up : down;
        projectToBounds(problem, x);
        work_units += 1e-6;
    }

    canonicalizePoint(problem, x);
    greedyRepair(problem, x, 6, work_units);
    if (!isFeasiblePoint(problem, x, &work_units)) return std::nullopt;
    const Real obj = objectiveValue(problem, x);
    if (!betterObjective(problem.sense, obj, incumbent)) return std::nullopt;
    return HeuristicSolution{.values = std::move(x), .objective = obj};
}

}  // namespace

void MipSolver::load(const LpProblem& problem) {
    if (hasAdvancedModelFeatures(problem)) {
        problem_ = linearizeModelFeatures(problem);
    } else {
        problem_ = problem;
    }
    refreshTreePresolveProfile();
    loaded_ = true;
}

bool MipSolver::hasLpLightCapability() const {
    return kLpLightCompiled;
}

void MipSolver::refreshTreePresolveProfile() {
    countVarTypes(problem_, model_binary_vars_, model_general_integer_vars_,
                  model_continuous_vars_);
    tree_presolve_binary_lite_profile_active_ =
        tree_presolve_auto_tuning_enabled_ &&
        model_binary_vars_ > 0 &&
        model_general_integer_vars_ == 0 &&
        model_continuous_vars_ == 0 &&
        problem_.num_cols <= kTreePresolveBinaryLiteMaxCols &&
        problem_.num_rows <= kTreePresolveBinaryLiteMaxRows;
}

MipTreePresolveStats MipSolver::treePresolveStatsSnapshot() const {
    std::lock_guard<std::mutex> lock(tree_presolve_stats_mutex_);
    return tree_presolve_stats_;
}

void MipSolver::mergeTreePresolveStatsDelta(const MipTreePresolveStats& delta) {
    std::lock_guard<std::mutex> lock(tree_presolve_stats_mutex_);
    tree_presolve_stats_.attempts += delta.attempts;
    tree_presolve_stats_.runs += delta.runs;
    tree_presolve_stats_.skipped += delta.skipped;
    tree_presolve_stats_.infeasible += delta.infeasible;
    tree_presolve_stats_.activity_tightenings += delta.activity_tightenings;
    tree_presolve_stats_.reduced_cost_tightenings += delta.reduced_cost_tightenings;
    tree_presolve_stats_.lp_resolves += delta.lp_resolves;
    tree_presolve_stats_.lp_delta += delta.lp_delta;
}

HeuristicRuntimeConfig MipSolver::makeHeuristicRuntimeConfig() const {
    HeuristicRuntimeConfig config;
    config.mode = heuristicRuntimeMode(parallel_mode_);
    config.seed = heuristic_seed_;
    config.rins_node_frequency = kRinsNodeFrequency;
    config.rins_subproblem_iter_limit = kRinsSubproblemIterLimit;
    config.rins_agreement_tol = kRinsAgreementTol;
    config.rins_max_int_inf_for_run = kRinsMaxIntInfForRun;
    config.rins_min_fixed_vars = kRinsMinFixedVars;
    config.rins_min_fixed_rate = kRinsMinFixedRate;
    config.rins_max_relative_gap_for_run = kRinsMaxRelativeGapForRun;
    config.root_max_int_inf = root_heuristic_max_int_inf_;
    config.root_max_int_vars = root_heuristic_max_int_vars_;
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

MipSolver::ConflictClause MipSolver::acquireConflictClause() {
    if (conflict_clause_pool_.empty()) return {};
    ConflictClause clause = std::move(conflict_clause_pool_.back());
    conflict_clause_pool_.pop_back();
    clause.literals.clear();
    clause.age = 0;
    clause.hits = 0;
    return clause;
}

void MipSolver::recycleConflictClause(ConflictClause clause) {
    clause.literals.clear();
    clause.age = 0;
    clause.hits = 0;
    if (clause.literals.capacity() == 0) return;
    if (clause.literals.capacity() > conflict_clause_capacity_limit_) return;
    if (conflict_clause_pool_.size() >= conflict_clause_pool_limit_) return;
    conflict_clause_pool_.push_back(std::move(clause));
}

void MipSolver::ageConflictPool() {
    if (conflict_pool_.empty()) return;

    for (auto& clause : conflict_pool_) {
        ++clause.age;
    }
    for (auto& score : conflict_scores_) {
        score = std::max<Real>(0.0, score * 0.995);
    }

    Int purged = 0;
    std::vector<ConflictClause> kept;
    kept.reserve(conflict_pool_.size());
    for (auto& clause : conflict_pool_) {
        if (clause.age > conflict_max_age_ || clause.literals.empty()) {
            recycleConflictClause(std::move(clause));
            ++purged;
            continue;
        }
        kept.push_back(std::move(clause));
    }
    conflict_pool_ = std::move(kept);
    conflict_stats_.purged += purged;
}

void MipSolver::learnConflictFromNode(const std::vector<BranchDecision>& bound_changes,
                                      bool lp_infeasible) {
    if (bound_changes.empty()) return;

    ConflictClause clause = acquireConflictClause();
    if (clause.literals.capacity() < bound_changes.size()) {
        clause.literals.reserve(bound_changes.size());
    }
    for (const auto& bc : bound_changes) {
        clause.literals.push_back({bc.variable, bc.bound, bc.is_upper});
    }
    if (clause.literals.empty()) return;

    std::sort(clause.literals.begin(), clause.literals.end(),
              [](const ConflictLiteral& a, const ConflictLiteral& b) {
                  if (a.variable != b.variable) return a.variable < b.variable;
                  if (a.is_upper != b.is_upper) return a.is_upper < b.is_upper;
                  return a.bound < b.bound;
              });

    std::size_t write = 0;
    for (const auto& lit : clause.literals) {
        if (write > 0 &&
            clause.literals[write - 1].variable == lit.variable &&
            clause.literals[write - 1].is_upper == lit.is_upper) {
            if (lit.is_upper) {
                clause.literals[write - 1].bound =
                    std::min(clause.literals[write - 1].bound, lit.bound);
            } else {
                clause.literals[write - 1].bound =
                    std::max(clause.literals[write - 1].bound, lit.bound);
            }
            continue;
        }
        clause.literals[write++] = lit;
    }
    clause.literals.resize(write);

    conflict_stats_.minimized_literals +=
        std::max<Int>(0, static_cast<Int>(bound_changes.size()) -
                             static_cast<Int>(clause.literals.size()));
    conflict_stats_.learned += 1;
    if (lp_infeasible) {
        ++conflict_stats_.lp_infeasible_conflicts;
    } else {
        ++conflict_stats_.bound_infeasible_conflicts;
    }

    for (const auto& lit : clause.literals) {
        if (lit.variable >= 0 &&
            lit.variable < static_cast<Index>(conflict_scores_.size())) {
            conflict_scores_[lit.variable] += 1.0;
        }
    }

    conflict_pool_.push_back(std::move(clause));
    while (static_cast<Int>(conflict_pool_.size()) > conflict_max_pool_size_) {
        auto victim = std::max_element(
            conflict_pool_.begin(), conflict_pool_.end(),
            [](const ConflictClause& a, const ConflictClause& b) {
                if (a.age != b.age) return a.age < b.age;
                return a.hits < b.hits;
            });
        if (victim == conflict_pool_.end()) break;
        recycleConflictClause(std::move(*victim));
        conflict_pool_.erase(victim);
        ++conflict_stats_.purged;
    }
}

bool MipSolver::isConflictTriggered(const NodeScratch& node_scratch) {
    if (conflict_pool_.empty()) return false;

    for (auto& clause : conflict_pool_) {
        bool triggered = true;
        for (const auto& lit : clause.literals) {
            if (lit.variable < 0 ||
                lit.variable >= static_cast<Index>(node_scratch.var_epoch.size())) {
                triggered = false;
                break;
            }
            if (node_scratch.var_epoch[lit.variable] != node_scratch.current_epoch) {
                triggered = false;
                break;
            }
            const Index p = node_scratch.var_pos[lit.variable];
            if (lit.is_upper) {
                if (node_scratch.node_ub[p] > lit.bound + 1e-9) {
                    triggered = false;
                    break;
                }
            } else {
                if (node_scratch.node_lb[p] < lit.bound - 1e-9) {
                    triggered = false;
                    break;
                }
            }
        }

        if (triggered) {
            ++conflict_stats_.reused;
            ++conflict_stats_.pruned;
            clause.age = 0;
            ++clause.hits;
            for (const auto& lit : clause.literals) {
                if (lit.variable >= 0 &&
                    lit.variable < static_cast<Index>(conflict_scores_.size())) {
                    conflict_scores_[lit.variable] += 0.25;
                }
            }
            return true;
        }
    }
    return false;
}

Index MipSolver::selectConflictAwareBranchVariable(
    std::span<const Real> primals,
    std::span<const Real> current_lower,
    std::span<const Real> current_upper,
    Index default_var) {
    if (default_var < 0 || default_var >= static_cast<Index>(conflict_scores_.size())) {
        return default_var;
    }

    const Real default_score = conflict_scores_[default_var];
    Index best_var = default_var;
    Real best_score = default_score;

    for (Index j = 0; j < problem_.num_cols; ++j) {
        if (problem_.col_type[j] == VarType::Continuous) continue;
        if (j >= static_cast<Index>(primals.size())) continue;
        if (j >= static_cast<Index>(current_lower.size()) ||
            j >= static_cast<Index>(current_upper.size())) {
            continue;
        }
        if (current_lower[j] >= current_upper[j] - 1e-9) continue;
        if (isIntegral(primals[j], kIntTol)) continue;
        const Real score = (j < static_cast<Index>(conflict_scores_.size()))
                               ? conflict_scores_[j]
                               : 0.0;
        if (score > best_score + 1e-12 ||
            (std::abs(score - best_score) <= 1e-12 && j < best_var)) {
            best_var = j;
            best_score = score;
        }
    }

    if (best_var != default_var && best_score > default_score + 0.25) {
        ++conflict_stats_.branch_score_overrides;
        return best_var;
    }
    return default_var;
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

bool MipSolver::enforceSymmetryBounds(std::vector<Real>& lower,
                                      std::vector<Real>& upper,
                                      std::vector<Index>* tightened_vars,
                                      double* work_units) const {
    if (!symmetry_enabled_) return true;
    const auto& fixes = symmetry_manager_.orbitalFixes();
    if (fixes.empty()) return true;

    if (work_units != nullptr) {
        *work_units += static_cast<double>(fixes.size());
    }

    std::vector<unsigned char> seen;
    if (tightened_vars != nullptr) {
        seen.assign(lower.size(), 0);
        tightened_vars->clear();
    }

    bool changed = true;
    const Index cols = static_cast<Index>(lower.size());
    while (changed) {
        changed = false;
        for (const auto& fix : fixes) {
            if (fix.canonical < 0 || fix.canonical >= cols ||
                fix.variable < 0 || fix.variable >= cols) {
                continue;
            }
            const Real new_canon_lower = std::max(lower[fix.canonical],
                                                  lower[fix.variable]);
            if (new_canon_lower > upper[fix.canonical] + 1e-12) {
                return false;
            }
            if (new_canon_lower > lower[fix.canonical]) {
                lower[fix.canonical] = new_canon_lower;
                changed = true;
                if (tightened_vars != nullptr &&
                    !seen[static_cast<std::size_t>(fix.canonical)]) {
                    tightened_vars->push_back(fix.canonical);
                    seen[static_cast<std::size_t>(fix.canonical)] = 1;
                }
            }
            const Real new_var_upper = std::min(upper[fix.variable],
                                                upper[fix.canonical]);
            if (new_var_upper < lower[fix.variable] - 1e-12) {
                return false;
            }
            if (new_var_upper < upper[fix.variable]) {
                upper[fix.variable] = new_var_upper;
                changed = true;
                if (tightened_vars != nullptr &&
                    !seen[static_cast<std::size_t>(fix.variable)]) {
                    tightened_vars->push_back(fix.variable);
                    seen[static_cast<std::size_t>(fix.variable)] = 1;
                }
            }
        }
        if (work_units != nullptr && changed) {
            *work_units += static_cast<double>(fixes.size());
        }
    }
    return true;
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
                             NodeScratch& node_scratch,
                             NodeWorkStats& node_stats,
                             Int& int_inf_out,
                             Int strong_branch_budget_override) {
    children_out.clear();
    node_iters_out = 0;
    node_work_out = 0.0;
    int_inf_out = -1;
    ++node_stats.nodes_solved;
    const bool use_conflicts = conflicts_enabled_ && num_threads_ <= 1;
    if (use_conflicts) ageConflictPool();
    std::vector<Index> temp_local_rows;
    ScopedRowRemoval local_row_guard(lp, temp_local_rows);

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
    std::vector<Index> symmetry_tightened;
    if (!enforceSymmetryBounds(current_lower, current_upper,
                               &symmetry_tightened, &node_work_out)) {
        if (use_conflicts) {
            learnConflictFromNode(node.bound_changes, false);
        }
        node_stats.bound_apply_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        return false;
    }
    for (Index j : symmetry_tightened) {
        lp.setColBounds(j, current_lower[j], current_upper[j]);
        touched_vars.push_back(j);
    }

    // Apply this node's bound changes using deterministic reusable scratch maps.
    if (node_scratch.var_epoch.size() != static_cast<std::size_t>(problem_.num_cols)) {
        node_scratch.var_epoch.assign(static_cast<std::size_t>(problem_.num_cols), 0);
        node_scratch.var_pos.assign(static_cast<std::size_t>(problem_.num_cols), -1);
        node_scratch.current_epoch = 1;
    }
    if (node_scratch.current_epoch == std::numeric_limits<Int>::max()) {
        std::fill(node_scratch.var_epoch.begin(), node_scratch.var_epoch.end(), 0);
        node_scratch.current_epoch = 1;
    } else {
        ++node_scratch.current_epoch;
    }
    const Int epoch = node_scratch.current_epoch;

    auto& vars = node_scratch.vars;
    auto& node_lb = node_scratch.node_lb;
    auto& node_ub = node_scratch.node_ub;
    vars.clear();
    node_lb.clear();
    node_ub.clear();
    vars.reserve(node.bound_changes.size());
    node_lb.reserve(node.bound_changes.size());
    node_ub.reserve(node.bound_changes.size());

    for (const auto& bc : node.bound_changes) {
        if (bc.variable < 0 || bc.variable >= problem_.num_cols) continue;
        Index pos = -1;
        if (node_scratch.var_epoch[bc.variable] != epoch) {
            node_scratch.var_epoch[bc.variable] = epoch;
            pos = static_cast<Index>(vars.size());
            node_scratch.var_pos[bc.variable] = pos;
            vars.push_back(bc.variable);
            node_lb.push_back(problem_.col_lower[bc.variable]);
            node_ub.push_back(problem_.col_upper[bc.variable]);
        } else {
            pos = node_scratch.var_pos[bc.variable];
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
            if (use_conflicts) {
                learnConflictFromNode(node.bound_changes, false);
            }
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
    symmetry_tightened.clear();
    if (!enforceSymmetryBounds(current_lower, current_upper,
                               &symmetry_tightened, &node_work_out)) {
        if (use_conflicts) {
            learnConflictFromNode(node.bound_changes, false);
        }
        node_stats.bound_apply_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        return false;
    }
    for (Index j : symmetry_tightened) {
        lp.setColBounds(j, current_lower[j], current_upper[j]);
        touched_vars.push_back(j);
    }
    node_stats.bound_apply_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    if (use_conflicts && isConflictTriggered(node_scratch)) {
        return false;
    }

    // Set basis for warm-start.
    t0 = std::chrono::steady_clock::now();
    if (!node.basis.empty() && node.basis_rows == lp.numRows()) {
        ++node_stats.warm_starts;
        lp.setBasis(node.basis);
    } else {
        ++node_stats.cold_starts;
    }
    node_stats.basis_set_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();

    if (!node.local_cuts.empty()) {
        auto& starts = node_scratch.row_starts;
        auto& col_indices = node_scratch.row_col_indices;
        auto& values = node_scratch.row_values;
        auto& lower = node_scratch.row_lower;
        auto& upper = node_scratch.row_upper;
        starts.clear();
        col_indices.clear();
        values.clear();
        lower.clear();
        upper.clear();
        starts.reserve(node.local_cuts.size());
        lower.reserve(node.local_cuts.size());
        upper.reserve(node.local_cuts.size());
        for (const auto& cut : node.local_cuts) {
            starts.push_back(static_cast<Index>(col_indices.size()));
            for (Index k = 0; k < static_cast<Index>(cut.indices.size()); ++k) {
                col_indices.push_back(cut.indices[k]);
                values.push_back(cut.values[k]);
            }
            lower.push_back(cut.lower);
            upper.push_back(cut.upper);
        }
        const Index base_row = lp.numRows();
        lp.addRows(starts, col_indices, values, lower, upper);
        for (Index i = 0; i < static_cast<Index>(lower.size()); ++i) {
            temp_local_rows.push_back(base_row + i);
        }
    }

    // Solve node LP.
    t0 = std::chrono::steady_clock::now();
    auto node_result = lp.solve();
    node_stats.lp_solve_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    node_iters_out = node_result.iterations;
    node_work_out += node_result.work_units;

    if (node_result.status == Status::Infeasible) {
        if (use_conflicts) {
            learnConflictFromNode(node.bound_changes, true);
        }
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

    Int tree_presolve_max_depth = tree_presolve_max_depth_;
    Int tree_presolve_min_frac = tree_presolve_min_frac_;
    Int tree_presolve_depth_frequency = tree_presolve_depth_frequency_;
    bool tree_presolve_frac_gate_enabled = true;
    if (tree_presolve_binary_lite_profile_active_) {
        tree_presolve_max_depth = std::min<Int>(
            tree_presolve_max_depth, kTreePresolveBinaryLiteMaxDepth);
        tree_presolve_min_frac = std::max<Int>(
            tree_presolve_min_frac, kTreePresolveBinaryLiteMinFrac);
        tree_presolve_depth_frequency = std::max<Int>(
            tree_presolve_depth_frequency,
            kTreePresolveBinaryLiteDepthFrequency);
        tree_presolve_frac_gate_enabled = false;
    }

    const bool tree_presolve_model_supported = (model_continuous_vars_ == 0);

    // In-tree presolve: currently limited to all-discrete models. Mixed
    // continuous MIPs need additional hardening before row-activity tightening
    // is trusted in-tree.
    if (tree_presolve_enabled_ &&
        tree_presolve_model_supported &&
        (num_threads_ <= 1 || parallel_mode_ == ParallelMode::Opportunistic) &&
        node.depth > 0 && node.depth <= tree_presolve_max_depth) {
        const auto tree_presolve_snapshot = treePresolveStatsSnapshot();
        MipTreePresolveStats tree_presolve_delta{};
        tree_presolve_delta.attempts = 1;
        bool tree_presolve_stats_flushed = false;
        auto flushTreePresolveStats = [&]() {
            if (tree_presolve_stats_flushed) return;
            mergeTreePresolveStatsDelta(tree_presolve_delta);
            tree_presolve_stats_flushed = true;
        };
        const bool depth_gate = (node.depth % tree_presolve_depth_frequency == 0);
        const bool frac_gate = tree_presolve_frac_gate_enabled &&
                               (frac_count >= tree_presolve_min_frac);

        bool benefit_skip = false;
        const Int benefit_skip_runs =
            tree_presolve_binary_lite_profile_active_ ? 4 : 6;
        if (tree_presolve_snapshot.runs >= benefit_skip_runs) {
            const Real avg_tight = static_cast<Real>(
                tree_presolve_snapshot.activity_tightenings +
                tree_presolve_snapshot.reduced_cost_tightenings) /
                static_cast<Real>(std::max<Int>(1, tree_presolve_snapshot.runs));
            const bool binary_lite_cold = tree_presolve_binary_lite_profile_active_ &&
                                          avg_tight < 4.0;
            if (binary_lite_cold ||
                (avg_tight < 0.5 && frac_count < tree_presolve_min_frac * 2 &&
                 !depth_gate)) {
                benefit_skip = true;
            }
        }

        if ((depth_gate || frac_gate) && !benefit_skip) {
            DomainPropagator dp;
            dp.load(problem_);
            for (Index j = 0; j < problem_.num_cols; ++j) {
                if (current_lower[j] > problem_.col_lower[j] + 1e-9 ||
                    current_upper[j] < problem_.col_upper[j] - 1e-9) {
                    dp.setBound(j, current_lower[j], current_upper[j]);
                }
            }

            if (!dp.propagate()) {
                ++tree_presolve_delta.infeasible;
                flushTreePresolveStats();
                if (use_conflicts) {
                    learnConflictFromNode(node.bound_changes, false);
                }
                return false;
            }

            Int activity_tight = 0;
            Int rc_tight = 0;
            auto applyTightening = [&](Index j, Real new_lb, Real new_ub, bool from_rc) {
                const Real lb = std::max(current_lower[j], new_lb);
                const Real ub = std::min(current_upper[j], new_ub);
                if (lb > ub + 1e-12) {
                    ++tree_presolve_delta.infeasible;
                    return false;
                }
                if (lb > current_lower[j] + 1e-9 || ub < current_upper[j] - 1e-9) {
                    lp.setColBounds(j, lb, ub);
                    current_lower[j] = lb;
                    current_upper[j] = ub;
                    touched_vars.push_back(j);
                    if (from_rc) {
                        ++rc_tight;
                    } else {
                        ++activity_tight;
                    }
                }
                return true;
            };

            for (Index j = 0; j < problem_.num_cols; ++j) {
                if (!applyTightening(j, dp.getLower(j), dp.getUpper(j), false)) {
                    if (use_conflicts) {
                        learnConflictFromNode(node.bound_changes, false);
                    }
                    return false;
                }
            }

            symmetry_tightened.clear();
            if (!enforceSymmetryBounds(current_lower, current_upper,
                                       &symmetry_tightened, &node_work_out)) {
                ++tree_presolve_delta.infeasible;
                flushTreePresolveStats();
                if (use_conflicts) {
                    learnConflictFromNode(node.bound_changes, false);
                }
                return false;
            }
            for (Index j : symmetry_tightened) {
                lp.setColBounds(j, current_lower[j], current_upper[j]);
                touched_vars.push_back(j);
                ++activity_tight;
            }

            if (incumbent_snapshot < kInf && node_obj_out < incumbent_snapshot - 1e-6) {
                const auto reduced = lp.getReducedCosts();
                if (static_cast<Index>(reduced.size()) >= problem_.num_cols) {
                    const Real gap = incumbent_snapshot - node_obj_out;
                    for (Index j = 0; j < problem_.num_cols; ++j) {
                        if (!std::isfinite(current_lower[j]) || !std::isfinite(current_upper[j])) {
                            continue;
                        }
                        if (current_lower[j] >= current_upper[j] - 1e-9) continue;

                        const Real rc = reduced[j];
                        if (rc > 1e-7 &&
                            std::abs(node_primals_out[j] - current_lower[j]) <= 1e-6) {
                            const Real new_ub = current_lower[j] + gap / rc;
                            if (!applyTightening(j, current_lower[j], new_ub, true)) {
                                if (use_conflicts) {
                                    learnConflictFromNode(node.bound_changes, false);
                                }
                                return false;
                            }
                        } else if (rc < -1e-7 &&
                                   std::abs(node_primals_out[j] - current_upper[j]) <= 1e-6) {
                            const Real new_lb = current_upper[j] + gap / rc;  // rc is negative
                            if (!applyTightening(j, new_lb, current_upper[j], true)) {
                                if (use_conflicts) {
                                    learnConflictFromNode(node.bound_changes, false);
                                }
                                return false;
                            }
                        }
                    }
                    symmetry_tightened.clear();
                    if (!enforceSymmetryBounds(current_lower, current_upper,
                                               &symmetry_tightened,
                                               &node_work_out)) {
                        ++tree_presolve_delta.infeasible;
                        flushTreePresolveStats();
                        if (use_conflicts) {
                            learnConflictFromNode(node.bound_changes, false);
                        }
                        return false;
                    }
                    for (Index j : symmetry_tightened) {
                        lp.setColBounds(j, current_lower[j], current_upper[j]);
                        touched_vars.push_back(j);
                        ++rc_tight;
                    }
                }
            }

            ++tree_presolve_delta.runs;
            tree_presolve_delta.activity_tightenings += activity_tight;
            tree_presolve_delta.reduced_cost_tightenings += rc_tight;
            const Int total_tight = activity_tight + rc_tight;
            if (total_tight > 0) {
                const Real prev_obj = node_obj_out;
                const auto refresh = lp.solve();
                node_iters_out += refresh.iterations;
                node_work_out += refresh.work_units;
                ++tree_presolve_delta.lp_resolves;

                if (refresh.status == Status::Infeasible) {
                    ++tree_presolve_delta.infeasible;
                    flushTreePresolveStats();
                    if (use_conflicts) {
                        learnConflictFromNode(node.bound_changes, true);
                    }
                    return false;
                }
                if (refresh.status != Status::Optimal) {
                    flushTreePresolveStats();
                    return false;
                }
                node_obj_out = refresh.objective;
                node_primals_out = lp.getPrimalValues();
                tree_presolve_delta.lp_delta += std::abs(prev_obj - node_obj_out);
                if (incumbent_snapshot < kInf &&
                    node_obj_out >= incumbent_snapshot - 1e-6) {
                    flushTreePresolveStats();
                    return false;
                }

                frac_count = 0;
                for (Index j = 0; j < problem_.num_cols; ++j) {
                    if (problem_.col_type[j] == VarType::Continuous) continue;
                    if (!isIntegral(node_primals_out[j], kIntTol)) ++frac_count;
                }
                int_inf_out = frac_count;
            }
            flushTreePresolveStats();
        } else {
            ++tree_presolve_delta.skipped;
            flushTreePresolveStats();
        }
    }

    std::vector<Cut> node_local_cuts = std::move(node.local_cuts);
    if (!node_local_cuts.empty()) {
        auto& kept = node_scratch.kept_cuts;
        kept.clear();
        kept.reserve(node_local_cuts.size());
        for (auto& cut : node_local_cuts) {
            const Real dist = cutDistanceToBinding(cut, node_primals_out);
            if (dist <= 1e-4) {
                cut.age = 0;
                ++cut.activity;
            } else {
                ++cut.age;
            }

            if (cut.age > 6) {
                if (dist <= 1e-2) {
                    cut.age = 0;
                    ++cut_stats_.tree_cuts_revived;
                    kept.push_back(std::move(cut));
                } else {
                    ++cut_stats_.tree_cuts_purged;
                }
            } else {
                kept.push_back(std::move(cut));
            }
        }
        node_local_cuts = std::move(kept);
    }

    // In-tree cut management (serial path only): depth-gated rounds and
    // local-vs-global cut handling.
    bool tree_cut_ran = false;
    if (tree_cuts_enabled_ &&
        cuts_enabled_ && cut_effort_mode_ != CutEffortMode::Off &&
        num_threads_ <= 1 && node.depth > 0 && frac_count > 0 &&
        problem_.num_cols <= 64) {
        const bool aggressive = (node.depth <= 2);
        const bool conservative = (node.depth <= 8);
        const bool run_tree_rounds = aggressive || (conservative && (node.depth % 2 == 0));
        if (run_tree_rounds) {
            SeparatorManager tree_separators;
            tree_separators.setConfig(cut_family_config_);
            const Int max_tree_cuts = aggressive ? 8 : 4;
            tree_separators.setMaxCutsPerFamily(max_tree_cuts);
            const Int max_rounds = aggressive ? 2 : 1;

            for (Int round = 0; round < max_rounds; ++round) {
                CutPool tree_pool;
                CutSeparationStats tree_stats;
                const Int generated = tree_separators.separate(
                    lp, problem_, node_primals_out, tree_pool, tree_stats);
                if (generated == 0) break;

                auto top_indices = tree_pool.topByEfficacy(max_tree_cuts);
                auto& starts = node_scratch.row_starts;
                auto& col_indices = node_scratch.row_col_indices;
                auto& values = node_scratch.row_values;
                auto& lower = node_scratch.row_lower;
                auto& upper = node_scratch.row_upper;
                starts.clear();
                col_indices.clear();
                values.clear();
                lower.clear();
                upper.clear();
                Int global_rows = 0;

                for (Index idx : top_indices) {
                    const auto& cut = tree_pool[idx];
                    const bool promote_global =
                        (node.depth <= 2 && cut.efficacy >= 0.15);

                    starts.push_back(static_cast<Index>(col_indices.size()));
                    for (Index k = 0; k < static_cast<Index>(cut.indices.size()); ++k) {
                        col_indices.push_back(cut.indices[k]);
                        values.push_back(cut.values[k]);
                    }
                    lower.push_back(cut.lower);
                    upper.push_back(cut.upper);

                    if (promote_global) {
                        ++global_rows;
                        ++cut_stats_.tree_cuts_global;
                    } else {
                        Cut local_cut = cut;
                        local_cut.local = true;
                        local_cut.age = 0;
                        local_cut.activity = 0;
                        node_local_cuts.push_back(std::move(local_cut));
                        ++cut_stats_.tree_cuts_local;
                    }
                }

                if (lower.empty()) break;

                const Index base_row = lp.numRows();
                lp.addRows(starts, col_indices, values, lower, upper);
                for (Index i = global_rows; i < static_cast<Index>(lower.size()); ++i) {
                    temp_local_rows.push_back(base_row + i);
                }

                const Real prev_obj = node_obj_out;
                auto tree_result = lp.solve();
                node_iters_out += tree_result.iterations;
                node_work_out += tree_result.work_units;
                if (tree_result.status != Status::Optimal) break;

                node_obj_out = tree_result.objective;
                node_primals_out = lp.getPrimalValues();
                const Real improvement = std::abs(node_obj_out - prev_obj);
                cut_stats_.tree_lp_delta += improvement;
                ++cut_stats_.tree_rounds;
                tree_cut_ran = true;
                if (improvement < kCutImprovementTol) break;
            }
        } else {
            ++cut_stats_.tree_nodes_skipped;
        }
    } else if (node.depth > 0 && frac_count > 0) {
        ++cut_stats_.tree_nodes_skipped;
    }
    if (tree_cut_ran) ++cut_stats_.tree_nodes_with_cuts;

    // Check integer feasibility after any in-tree separation.
    if (isFeasibleMip(node_primals_out)) {
        int_inf_out = 0;
        return false;  // Caller handles incumbent update.
    }
    frac_count = 0;
    for (Index j = 0; j < problem_.num_cols; ++j) {
        if (problem_.col_type[j] == VarType::Continuous) continue;
        if (!isIntegral(node_primals_out[j], kIntTol)) ++frac_count;
    }
    int_inf_out = frac_count;

    // Branch with reliability branching (strong-branch bootstrap + pseudocosts),
    // with optional sibling branch-variable reuse.
    Index branch_var = -1;
    const bool use_sibling_cache = (num_threads_ <= 1);
    if (use_sibling_cache && node.parent_id >= 0) {
        auto it = sibling_branch_cache_.find(node.parent_id);
        if (it != sibling_branch_cache_.end()) {
            const Index cached = it->second;
            if (cached >= 0 && cached < problem_.num_cols &&
                problem_.col_type[cached] != VarType::Continuous &&
                current_lower[cached] < current_upper[cached] - 1e-9 &&
                !isIntegral(node_primals_out[cached], kIntTol)) {
                branch_var = cached;
                ++search_stats_.sibling_cache_hits;
            } else {
                ++search_stats_.sibling_cache_misses;
            }
        }
    }

    if (branch_var < 0) {
        {
            std::lock_guard<std::mutex> lock(branching_mutex_);
            const Int saved_budget = branching_rule_.strongBranchProbeBudget();
            if (strong_branch_budget_override >= 2 &&
                strong_branch_budget_override != saved_budget) {
                branching_rule_.setStrongBranchProbeBudget(
                    strong_branch_budget_override);
            }
            auto selection = branching_rule_.select(lp, problem_,
                                                    node_primals_out,
                                                    current_lower,
                                                    current_upper,
                                                    node_result.objective,
                                                    false,
                                                    branching_stats_);
            if (strong_branch_budget_override >= 2 &&
                strong_branch_budget_override != saved_budget) {
                branching_rule_.setStrongBranchProbeBudget(saved_budget);
            }
            branch_var = selection.variable;
        }
        if (use_conflicts && branch_var >= 0) {
            branch_var = selectConflictAwareBranchVariable(
                node_primals_out, current_lower, current_upper, branch_var);
        }
    }
    if (branch_var < 0) {
        return false;
    }

    BnbNode solved_node = std::move(node);
    solved_node.lp_bound = node_obj_out;
    solved_node.estimate = node_obj_out + 0.05 * static_cast<Real>(std::max<Int>(0, frac_count));
    solved_node.basis_rows = lp.numRows() - static_cast<Index>(temp_local_rows.size());
    if (temp_local_rows.empty()) {
        solved_node.basis = lp.getBasis();
    } else {
        solved_node.basis.clear();
    }
    solved_node.local_cuts = std::move(node_local_cuts);

    auto [left, right] = createChildren(std::move(solved_node), branch_var,
                                        node_primals_out[branch_var]);
    left.lp_bound = node_obj_out;
    right.lp_bound = node_obj_out;
    left.estimate = solved_node.estimate;
    right.estimate = solved_node.estimate;

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
    std::vector<Index> root_symmetry_tightened;
    double root_symmetry_work = 0.0;
    if (!enforceSymmetryBounds(current_lower, current_upper,
                               &root_symmetry_tightened,
                               &root_symmetry_work)) {
        if (verbose_) {
            log_.log("Symmetry propagation detected infeasible root bounds.\n");
        }
        return;
    }
    for (Index j : root_symmetry_tightened) {
        lp.setColBounds(j, current_lower[j], current_upper[j]);
    }
    total_work += root_symmetry_work;
    std::vector<Index> touched_vars;
    double last_log_time = -kLogInterval;  // ensure first node is logged
    uint64_t det_log_next_tick = 1000000;
    touched_vars.reserve(problem_.num_cols);
    NodeWorkStats node_stats;
    NodeScratch node_scratch;
    Int rins_calls = 0;
    Int rins_solves = 0;
    Int rins_found = 0;
    Int rins_skip_no_inc = 0;
    Int rins_skip_few_fix = 0;
    Int rins_fixed_sum = 0;
    Int nodes_since_incumbent = 0;

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

        const NodePolicy target_policy =
            selectSearchPolicy(search_profile_, nodes_since_incumbent);
        if (queue.policy() != target_policy) {
            queue.setPolicy(target_policy);
            ++search_stats_.policy_switches;
        }

        const bool allow_restarts =
            (search_profile_ == SearchProfile::Aggressive) || restarts_enabled_;
        if (allow_restarts && search_profile_ != SearchProfile::Stable &&
            nodes_since_incumbent >= restart_stagnation_nodes_ &&
            queue.size() > restart_keep_nodes_) {
            // Exact-safe restart: reset the queue policy without discarding any
            // unexplored nodes. Dropping queued nodes would turn restart into a
            // heuristic truncation and can produce invalid "optimal" claims.
            queue.setPolicy(NodePolicy::BestFirst);
            ++search_stats_.restarts;
            nodes_since_incumbent = 0;
            if (queue.empty()) break;
        }

        BnbNode node = queue.pop();
        const Int strong_budget = computeStrongBranchBudget(
            search_profile_, node.depth, nodes_since_incumbent,
            restart_stagnation_nodes_);
        ++search_stats_.strong_budget_updates;
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
                                    node_scratch,
                                    node_stats, node_int_inf,
                                    strong_budget);
        total_lp_iters += node_iters;
        total_work += node_work;
        bool improved_this_node = false;

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
            const double log_elapsed =
                (parallel_mode_ == ParallelMode::Deterministic)
                    ? static_cast<double>(workUnitTicks(total_work)) * 1e-6
                    : elapsed();
            logProgress(nodes_explored, queue.size(), total_lp_iters,
                        incumbent,
                        queue.empty() ? incumbent : queue.bestBound(),
                        log_elapsed, true, node_int_inf);
            queue.prune(incumbent - 1e-6);
            nodes_since_incumbent = 0;
            improved_this_node = true;
        }

        if (verbose_) {
            if (parallel_mode_ == ParallelMode::Deterministic) {
                const uint64_t cur_ticks = workUnitTicks(total_work);
                if (cur_ticks >= det_log_next_tick || nodes_explored <= 1) {
                    while (cur_ticks >= det_log_next_tick) {
                        det_log_next_tick += 1000000;
                    }
                    logProgress(nodes_explored, queue.size(), total_lp_iters,
                               incumbent,
                               queue.empty() ? (incumbent < kInf ? incumbent : root_bound)
                                             : queue.bestBound(),
                               static_cast<double>(cur_ticks) * 1e-6,
                               false, node_int_inf);
                }
            } else {
                // Log progress periodically (time-based).
                double now = elapsed();
                if (now - last_log_time >= kLogInterval || nodes_explored <= 1) {
                    logProgress(nodes_explored, queue.size(), total_lp_iters,
                               incumbent,
                               queue.empty() ? (incumbent < kInf ? incumbent : root_bound)
                                             : queue.bestBound(),
                               now, false, node_int_inf);
                    last_log_time = now;
                }
            }
        }

        if (!branched && !node_primals.empty() && node_int_inf == 0) {
            if (node_obj < incumbent) {
                incumbent = node_obj;
                best_solution = node_primals;
                solution_pool.submit({best_solution, incumbent}, "tree_integral", 0);
                const double log_elapsed =
                    (parallel_mode_ == ParallelMode::Deterministic)
                        ? static_cast<double>(workUnitTicks(total_work)) * 1e-6
                        : elapsed();
                logProgress(nodes_explored, queue.size(), total_lp_iters,
                           incumbent,
                           queue.empty() ? incumbent : queue.bestBound(),
                           log_elapsed, true, 0);
                queue.prune(incumbent - 1e-6);
                nodes_since_incumbent = 0;
                improved_this_node = true;
            }
        }

        if (branched && !children.empty()) {
            sibling_branch_cache_[node.id] = children.front().branch.variable;
            if (static_cast<Int>(sibling_branch_cache_.size()) > 4096) {
                sibling_branch_cache_.clear();
            }
        }
        for (auto& child : children) {
            queue.push(std::move(child));
        }

        if (!improved_this_node) {
            ++nodes_since_incumbent;
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
                               HeuristicRuntime& heuristic_runtime,
                               SolutionPool& solution_pool,
                               Real& incumbent, std::vector<Real>& best_solution,
                               Real root_bound,
                               const std::function<double()>& elapsed) {
    // Shared state protected by mutexes.
    std::mutex queue_mutex;
    std::mutex incumbent_mutex;
    std::atomic<Int> atomic_nodes{nodes_explored};
    std::atomic<Int> atomic_lp_iters{total_lp_iters};
    AtomicWorkUnits atomic_work;
    atomic_work.count(workUnitTicks(total_work));
    std::atomic<bool> should_stop{false};
    constexpr uint64_t kDetHeurGateQuantumTicks = 50000;  // 5e-2 deterministic work units
    constexpr uint64_t kDetLogQuantumTicks = 1000000;     // 1.0 deterministic work units
    std::atomic<uint64_t> det_heur_next_tick{kDetHeurGateQuantumTicks};
    std::mutex stats_mutex;
    double last_log_time = -kLogInterval;  // ensure first log fires
    Int rins_calls = 0;
    Int rins_solves = 0;
    Int rins_found = 0;
    Int rins_skip_no_inc = 0;
    Int rins_skip_few_fix = 0;
    Int rins_fixed_sum = 0;

    // Each thread needs its own LP solver instance.
    Int requested_threads = num_threads_;
    Int actual_threads =
        std::min(requested_threads, static_cast<Int>(std::thread::hardware_concurrency()));
    if (actual_threads < 1) actual_threads = 1;

    if (parallel_mode_ == ParallelMode::Deterministic) {
        struct DeterministicWorkerContext {
            DualSimplexSolver lp;
            std::vector<Real> current_lower;
            std::vector<Real> current_upper;
            std::vector<Index> touched_vars;
            NodeScratch node_scratch;
            NodeWorkStats node_stats;
            Int rins_calls = 0;
            Int rins_solves = 0;
            Int rins_found = 0;
            Int rins_skip_no_inc = 0;
            Int rins_skip_few_fix = 0;
            Int rins_fixed_sum = 0;
        };

        struct DeterministicTask {
            Int worker_id = 0;
            Int node_num = 0;
            Int strong_branch_budget = -1;
            BnbNode node;
            bool branched = false;
            std::vector<BnbNode> children;
            Real node_obj = 0.0;
            std::vector<Real> node_primals;
            Int node_iters = 0;
            double node_work = 0.0;
            Int node_int_inf = -1;
        };

        if (verbose_) {
            log_.log("Parallel tree search with %d threads (deterministic)\n",
                     actual_threads);
        }

        std::vector<DeterministicWorkerContext> workers(
            static_cast<std::size_t>(actual_threads));
        std::vector<Int> active_workers;
        active_workers.reserve(static_cast<std::size_t>(actual_threads));
        for (Int t = 0; t < actual_threads; ++t) {
            auto& worker_ctx = workers[static_cast<std::size_t>(t)];
            worker_ctx.lp.load(problem_);
            worker_ctx.lp.setVerbose(false);
            worker_ctx.lp.setBasis(root_lp.getBasis());
            worker_ctx.current_lower = problem_.col_lower;
            worker_ctx.current_upper = problem_.col_upper;
            worker_ctx.touched_vars.reserve(
                static_cast<std::size_t>(problem_.num_cols));
            std::vector<Index> root_symmetry_tightened;
            double root_symmetry_work = 0.0;
            if (!enforceSymmetryBounds(worker_ctx.current_lower,
                                       worker_ctx.current_upper,
                                       &root_symmetry_tightened,
                                       &root_symmetry_work)) {
                continue;
            }
            for (Index j : root_symmetry_tightened) {
                worker_ctx.lp.setColBounds(j,
                                           worker_ctx.current_lower[j],
                                           worker_ctx.current_upper[j]);
            }
            total_work += root_symmetry_work;
            active_workers.push_back(t);
        }
        if (active_workers.empty()) return;

        uint64_t det_heur_next_tick = kDetHeurGateQuantumTicks;
        uint64_t det_log_next_tick = kDetLogQuantumTicks;
        Int nodes_since_incumbent = 0;
        while (true) {
            if (nodes_explored >= node_limit_) break;
            if (elapsed() >= time_limit_) break;
            if (queue.empty()) break;

            if (incumbent < kInf) {
                queue.prune(incumbent - 1e-6);
                if (queue.empty()) break;
            }
            const Real best_bound = queue.bestBound();
            if (incumbent < kInf && computeGap(incumbent, best_bound) < gap_tol_) {
                break;
            }

            const NodePolicy target_policy =
                selectSearchPolicy(search_profile_, nodes_since_incumbent);
            if (queue.policy() != target_policy) {
                queue.setPolicy(target_policy);
                ++search_stats_.policy_switches;
            }

            const bool allow_restarts =
                (search_profile_ == SearchProfile::Aggressive) || restarts_enabled_;
            if (allow_restarts && search_profile_ != SearchProfile::Stable &&
                nodes_since_incumbent >= restart_stagnation_nodes_ &&
                queue.size() > restart_keep_nodes_) {
                // Keep deterministic threaded restart exact-safe as well.
                queue.setPolicy(NodePolicy::BestFirst);
                ++search_stats_.restarts;
                nodes_since_incumbent = 0;
                if (queue.empty()) break;
            }

            const Int remaining_nodes = std::max<Int>(0, node_limit_ - nodes_explored);
            const Int batch_size = std::min<Int>(
                remaining_nodes,
                std::min<Int>(queue.size(),
                              static_cast<Int>(active_workers.size())));
            if (batch_size <= 0) break;
            const Real batch_incumbent = incumbent;

            std::vector<DeterministicTask> tasks(
                static_cast<std::size_t>(batch_size));
            for (Int i = 0; i < batch_size; ++i) {
                auto& task = tasks[static_cast<std::size_t>(i)];
                task.worker_id = active_workers[static_cast<std::size_t>(i)];
                task.node_num = ++nodes_explored;
                task.node = queue.pop();
                task.strong_branch_budget = computeStrongBranchBudget(
                    search_profile_,
                    task.node.depth,
                    nodes_since_incumbent + i,
                    restart_stagnation_nodes_);
                ++search_stats_.strong_budget_updates;
            }

            tbb::parallel_for(Int(0), batch_size, [&](Int i) {
                auto& task = tasks[static_cast<std::size_t>(i)];
                auto& worker_ctx =
                    workers[static_cast<std::size_t>(task.worker_id)];
                task.branched = processNode(worker_ctx.lp,
                                            task.node,
                                            batch_incumbent,
                                            task.children,
                                            task.node_obj,
                                            task.node_primals,
                                            task.node_iters,
                                            task.node_work,
                                            worker_ctx.current_lower,
                                            worker_ctx.current_upper,
                                            worker_ctx.touched_vars,
                                            worker_ctx.node_scratch,
                                            worker_ctx.node_stats,
                                            task.node_int_inf,
                                            task.strong_branch_budget);
            });

            for (Int i = 0; i < batch_size; ++i) {
                auto& task = tasks[static_cast<std::size_t>(i)];
                auto& worker_ctx =
                    workers[static_cast<std::size_t>(task.worker_id)];
                total_lp_iters += task.node_iters;
                total_work += task.node_work;

                bool found_incumbent = false;
                std::vector<Real> incumbent_values;
                Real heur_incumbent = kInf;
                if (incumbent < kInf && !best_solution.empty()) {
                    heur_incumbent = incumbent;
                    incumbent_values = best_solution;
                } else {
                    auto pooled = solution_pool.bestSolution();
                    if (pooled.has_value()) {
                        heur_incumbent = pooled->objective;
                        incumbent_values = std::move(pooled->values);
                    }
                }

                bool run_tree_heuristic = false;
                double heuristic_work_view = total_work;
                const uint64_t cur_ticks = workUnitTicks(total_work);
                if (cur_ticks >= det_heur_next_tick) {
                    run_tree_heuristic = true;
                    heuristic_work_view =
                        static_cast<double>(det_heur_next_tick) * 1e-6;
                    det_heur_next_tick += kDetHeurGateQuantumTicks;
                }

                WorkerHeuristicOutcome heur_outcome;
                if (run_tree_heuristic) {
                    WorkerHeuristicContext heur_ctx{
                        .problem = problem_,
                        .lp = worker_ctx.lp,
                        .primals = task.node_primals,
                        .node_count = task.node_num,
                        .int_inf = task.node_int_inf,
                        .node_objective = task.node_obj,
                        .incumbent = heur_incumbent,
                        .incumbent_values = std::span<const Real>(
                            incumbent_values.data(), incumbent_values.size()),
                        .thread_id = task.worker_id,
                        .total_work_units = heuristic_work_view,
                    };
                    heur_outcome = heuristic_runtime.runTreeWorker(heur_ctx);
                }
                if (heur_outcome.attempted) {
                    ++worker_ctx.rins_calls;
                    total_lp_iters += heur_outcome.lp_iterations;
                    total_work += heur_outcome.work_units;
                    if (heur_outcome.executed_solve) ++worker_ctx.rins_solves;
                    if (heur_outcome.skipped_no_incumbent) {
                        ++worker_ctx.rins_skip_no_inc;
                    }
                    if (heur_outcome.skipped_few_fixes) {
                        ++worker_ctx.rins_skip_few_fix;
                    }
                    worker_ctx.rins_fixed_sum += heur_outcome.fixed_count;

                    if (heur_outcome.improved &&
                        heur_outcome.solution.has_value() &&
                        heur_outcome.solution->objective < incumbent) {
                        incumbent = heur_outcome.solution->objective;
                        best_solution = std::move(heur_outcome.solution->values);
                        solution_pool.submit({best_solution, incumbent},
                                             "rins",
                                             task.worker_id);
                        found_incumbent = true;
                        ++worker_ctx.rins_found;
                    }
                }

                if (!task.branched && !task.node_primals.empty() &&
                    task.node_int_inf == 0) {
                    if (task.node_obj < incumbent) {
                        incumbent = task.node_obj;
                        best_solution = task.node_primals;
                        solution_pool.submit({best_solution, incumbent},
                                             "tree_integral",
                                             task.worker_id);
                        found_incumbent = true;
                    }
                }

                if (!task.children.empty()) {
                    for (auto& child : task.children) {
                        queue.push(std::move(child));
                    }
                }

                if (verbose_) {
                    const uint64_t cur_ticks = workUnitTicks(total_work);
                    bool emit_log = found_incumbent;
                    if (!emit_log && cur_ticks >= det_log_next_tick) {
                        emit_log = true;
                        while (cur_ticks >= det_log_next_tick) {
                            det_log_next_tick += kDetLogQuantumTicks;
                        }
                    }
                    if (emit_log) {
                        const Real log_best_bound = queue.empty()
                            ? (incumbent < kInf ? incumbent : root_bound)
                            : queue.bestBound();
                        logProgress(nodes_explored, queue.size(), total_lp_iters,
                                    incumbent, log_best_bound,
                                    static_cast<double>(cur_ticks) * 1e-6,
                                    found_incumbent, task.node_int_inf);
                    }
                }

                if (found_incumbent) {
                    nodes_since_incumbent = 0;
                } else {
                    ++nodes_since_incumbent;
                }
            }
        }

        for (const auto& worker_ctx : workers) {
            lp_stats_.node_bound_apply_seconds +=
                worker_ctx.node_stats.bound_apply_seconds;
            lp_stats_.node_basis_set_seconds +=
                worker_ctx.node_stats.basis_set_seconds;
            lp_stats_.node_lp_solve_seconds +=
                worker_ctx.node_stats.lp_solve_seconds;
            lp_stats_.nodes_solved += worker_ctx.node_stats.nodes_solved;
            lp_stats_.warm_starts += worker_ctx.node_stats.warm_starts;
            lp_stats_.cold_starts += worker_ctx.node_stats.cold_starts;
            rins_calls += worker_ctx.rins_calls;
            rins_solves += worker_ctx.rins_solves;
            rins_found += worker_ctx.rins_found;
            rins_skip_no_inc += worker_ctx.rins_skip_no_inc;
            rins_skip_few_fix += worker_ctx.rins_skip_few_fix;
            rins_fixed_sum += worker_ctx.rins_fixed_sum;
        }

        if (verbose_) {
            if (rins_calls > 0) {
                const double avg_fixed = static_cast<double>(rins_fixed_sum) /
                                         static_cast<double>(rins_calls);
                log_.log("RINS(parallel): calls=%d solves=%d found=%d avg_fixed=%.1f "
                         "skip_no_inc=%d skip_few_fix=%d\n",
                         rins_calls, rins_solves, rins_found, avg_fixed,
                         rins_skip_no_inc, rins_skip_few_fix);
            }
            if (nodes_explored >= node_limit_) log_.log("Node limit reached.\n");
            else if (elapsed() >= time_limit_) log_.log("Time limit reached.\n");
        }
        return;
    }

    if (verbose_) {
        log_.log("Parallel tree search with %d threads\n", actual_threads);
    }

    // Worker function.
    auto worker = [&](Int thread_id) {
        // Create thread-local LP solver by loading problem fresh.
        DualSimplexSolver local_lp;
        local_lp.load(problem_);
        local_lp.setVerbose(false);
        // Warm-start from root basis.
        local_lp.setBasis(root_lp.getBasis());
        std::vector<Real> current_lower = problem_.col_lower;
        std::vector<Real> current_upper = problem_.col_upper;
        std::vector<Index> root_symmetry_tightened;
        double root_symmetry_work = 0.0;
        if (!enforceSymmetryBounds(current_lower, current_upper,
                                   &root_symmetry_tightened,
                                   &root_symmetry_work)) {
            return;
        }
        for (Index j : root_symmetry_tightened) {
            local_lp.setColBounds(j, current_lower[j], current_upper[j]);
        }
        atomic_work.count(workUnitTicks(root_symmetry_work));
        std::vector<Index> touched_vars;
        touched_vars.reserve(problem_.num_cols);
        NodeWorkStats local_node_stats;
        NodeScratch local_node_scratch;
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
                Real inc = kInf;
                {
                    std::lock_guard<std::mutex> incumbent_lock(incumbent_mutex);
                    inc = incumbent;
                }
                std::lock_guard<std::mutex> lock(queue_mutex);
                if (queue.empty()) break;

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
                                        local_node_scratch,
                                        local_node_stats, node_int_inf);
            atomic_lp_iters.fetch_add(node_iters, std::memory_order_relaxed);
            atomic_work.count(workUnitTicks(node_work));

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

            bool run_tree_heuristic = true;
            double heuristic_work_view = atomic_work.units();
            if (parallel_mode_ == ParallelMode::Deterministic) {
                run_tree_heuristic = false;
                const uint64_t cur_ticks = workUnitTicks(heuristic_work_view);
                uint64_t gate_tick = det_heur_next_tick.load(std::memory_order_relaxed);
                while (cur_ticks >= gate_tick) {
                    if (det_heur_next_tick.compare_exchange_weak(
                            gate_tick,
                            gate_tick + kDetHeurGateQuantumTicks,
                            std::memory_order_relaxed)) {
                        run_tree_heuristic = true;
                        heuristic_work_view = static_cast<double>(gate_tick) * 1e-6;
                        break;
                    }
                }
            }

            WorkerHeuristicOutcome heur_outcome;
            if (run_tree_heuristic) {
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
                    .total_work_units = heuristic_work_view,
                };
                heur_outcome = heuristic_runtime.runTreeWorker(heur_ctx);
            }
            if (heur_outcome.attempted) {
                ++local_rins_calls;
                atomic_lp_iters.fetch_add(heur_outcome.lp_iterations,
                                          std::memory_order_relaxed);
                if (parallel_mode_ == ParallelMode::Deterministic) {
                    // Keep deterministic-mode heuristic gate cadence invariant to
                    // sub-MIP solve variability across worker interleavings.
                    atomic_work.count(kDetHeurGateQuantumTicks);
                } else {
                    atomic_work.count(workUnitTicks(heur_outcome.work_units));
                }

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
                Real log_incumbent = kInf;
                {
                    std::lock_guard<std::mutex> lock(incumbent_mutex);
                    log_incumbent = incumbent;
                }
                Int open_nodes = 0;
                Real best_bound = root_bound;
                {
                    std::lock_guard<std::mutex> lock(queue_mutex);
                    open_nodes = queue.size();
                    best_bound = queue.empty()
                        ? (log_incumbent < kInf ? log_incumbent : root_bound)
                        : queue.bestBound();
                }
                logProgress(atomic_nodes.load(), open_nodes,
                           atomic_lp_iters.load(),
                           log_incumbent,
                           best_bound,
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
    total_work = atomic_work.units();

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
                               HeuristicRuntime& heuristic_runtime,
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
    solveSerial(lp, queue, nodes_explored, total_lp_iters,
                total_work, heuristic_runtime, solution_pool, incumbent,
                best_solution, root_bound, elapsed);
}
#endif

MipResult MipSolver::solve() {
    if (!loaded_) return {};
    lp_stats_ = {};
    cut_stats_ = {};
    conflict_stats_ = {};
    pre_root_stats_ = {};
    search_stats_ = {};
    tree_presolve_stats_ = {};
    symmetry_stats_ = {};
    exact_refinement_stats_ = {};
    for (auto& clause : conflict_pool_) {
        recycleConflictClause(std::move(clause));
    }
    conflict_pool_.clear();
    conflict_scores_.assign(problem_.num_cols, 0.0);
    sibling_branch_cache_.clear();
    lp_stats_.root_policy = root_lp_policy_;
    exact_refinement_stats_.mode = exact_refinement_mode_;
    exact_refinement_stats_.rational_verification_enabled =
        exact_refinement_rational_check_;

    auto start_time = std::chrono::steady_clock::now();
    auto elapsed = [&]() -> double {
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(now - start_time).count();
    };
    auto canonicalReportedWork = [&](double work_units) -> double {
        return work_units;
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

        // Build type string.
        char type_buf[128] = "";
        if (model_binary_vars_ > 0 || model_general_integer_vars_ > 0) {
            char* p = type_buf;
            p += std::snprintf(p, sizeof(type_buf), " (");
            bool first = true;
            if (model_binary_vars_ > 0) {
                p += std::snprintf(p, sizeof(type_buf) - (p - type_buf),
                                   "%d binary", model_binary_vars_);
                first = false;
            }
            if (model_general_integer_vars_ > 0) {
                if (!first) {
                    p += std::snprintf(p, sizeof(type_buf) - (p - type_buf),
                                       ", ");
                }
                p += std::snprintf(p, sizeof(type_buf) - (p - type_buf),
                                   "%d integer", model_general_integer_vars_);
                first = false;
            }
            if (model_continuous_vars_ > 0) {
                if (!first) {
                    p += std::snprintf(p, sizeof(type_buf) - (p - type_buf),
                                       ", ");
                }
                p += std::snprintf(p, sizeof(type_buf) - (p - type_buf),
                                   "%d continuous", model_continuous_vars_);
            }
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
        if (presolve_options_.enable_forcing_rows) {
            sp += std::snprintf(sp, sizeof(settings) - (sp - settings),
                                "presolve_forcing=on ");
        }
        if (presolve_options_.enable_dual_fixing) {
            sp += std::snprintf(sp, sizeof(settings) - (sp - settings),
                                "presolve_dual_fixing=on ");
        }
        if (presolve_options_.enable_coefficient_tightening) {
            sp += std::snprintf(sp, sizeof(settings) - (sp - settings),
                                "presolve_coeff_tightening=on ");
        }
        if (!tree_presolve_enabled_) {
            sp += std::snprintf(sp, sizeof(settings) - (sp - settings), "tree_presolve=off ");
        } else if (tree_presolve_binary_lite_profile_active_) {
            sp += std::snprintf(sp, sizeof(settings) - (sp - settings),
                                "tree_presolve_profile=binary-lite ");
        }
        if (!tree_cuts_enabled_) {
            sp += std::snprintf(sp, sizeof(settings) - (sp - settings), "tree_cuts=off ");
        }
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
        if (parallel_mode_ != ParallelMode::Deterministic) {
            sp += std::snprintf(sp, sizeof(settings) - (sp - settings),
                                "parallel_mode=opportunistic ");
        }
        if (heuristic_seed_ != 1) {
            sp += std::snprintf(sp, sizeof(settings) - (sp - settings),
                                "heur_seed=%llu ",
                                static_cast<unsigned long long>(heuristic_seed_));
        }
        if (search_profile_ != SearchProfile::Default) {
            sp += std::snprintf(sp, sizeof(settings) - (sp - settings),
                                "search=%s ", searchProfileName(search_profile_));
        }
        if (pre_root_lp_free_enabled_) {
            sp += std::snprintf(sp, sizeof(settings) - (sp - settings),
                                "pre_root_lpfree=on pre_root_work=%.0f pre_root_rounds=%d ",
                                pre_root_lp_free_work_budget_,
                                pre_root_lp_free_max_rounds_);
        }
        if (pre_root_lp_light_enabled_) {
            sp += std::snprintf(sp, sizeof(settings) - (sp - settings),
                                "pre_root_lplight=on lplight_backend=%s ",
                                lpLightBackendName());
        }
        if (!pre_root_portfolio_enabled_) {
            sp += std::snprintf(sp, sizeof(settings) - (sp - settings),
                                "pre_root_sched=fixed ");
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
    presolver.setOptions(presolve_options_);
    LpProblem working_problem;
    bool did_presolve = false;
    double symmetry_pre_work = 0.0;
    Index symmetry_cuts_added = 0;

    if (presolve_) {
        working_problem = presolver.presolve(problem_);
        const auto& stats = presolver.stats();
        if (stats.vars_removed > 0 || stats.rows_removed > 0) {
            did_presolve = true;
            if (verbose_) {
                log_.log("Presolve: %d vars removed, %d rows removed, "
                         "%d bounds tightened, %d rounds (%d changed), %.3fs "
                         "[rules: forcing=%d implied=%d abt=%d dual=%d coeff=%d "
                         "doubleton=%d empty_col=%d dup_row=%d par_row=%d] "
                         "[examined: %d rows, %d cols]\n\n",
                         stats.vars_removed, stats.rows_removed,
                         stats.bounds_tightened, stats.rounds,
                         stats.rounds_with_changes, stats.time_seconds,
                         stats.forcing_row_changes,
                         stats.implied_equation_changes,
                         stats.activity_bound_tightening_changes,
                         stats.dual_fixing_changes,
                         stats.coeff_tightening_changes,
                         stats.doubleton_eq_changes,
                         stats.empty_col_changes,
                         stats.duplicate_row_changes,
                         stats.parallel_row_changes,
                         stats.rows_examined, stats.cols_examined);
            }
        } else {
            working_problem = problem_;
        }
    } else {
        working_problem = problem_;
    }

    if (symmetry_enabled_) {
        symmetry_manager_.detect(working_problem);
        const double detect_work = symmetry_manager_.detectWorkUnits();
        symmetry_pre_work += detect_work;
        symmetry_stats_.detect_work_units = detect_work;
        const SymmetryManager* ptr = symmetry_manager_.hasSymmetry()
            ? &symmetry_manager_
            : nullptr;
        symmetry_stats_.orbits =
            static_cast<Int>(symmetry_manager_.orbits().size());
        branching_rule_.setSymmetryManager(ptr);
        if (ptr) {
            symmetry_cuts_added = symmetry_manager_.addSymmetryCuts(working_problem);
            symmetry_stats_.cuts_added = symmetry_cuts_added;
            const double cut_work = symmetry_manager_.cutWorkUnits();
            symmetry_pre_work += cut_work;
            symmetry_stats_.cut_work_units = cut_work;
            if (verbose_) {
                const auto& orbits = symmetry_manager_.orbits();
                log_.log("Symmetry detected %zu orbit%s (canonical branching enforced, "
                         "%d symmetry cut%s added).\n",
                         orbits.size(), orbits.size() == 1 ? "" : "s",
                         symmetry_cuts_added, symmetry_cuts_added == 1 ? "" : "s");
            }
        }
    } else {
        branching_rule_.setSymmetryManager(nullptr);
    }

    if (presolver.isInfeasible()) {
        MipResult result;
        result.status = Status::Infeasible;
        result.nodes = 0;
        result.lp_iterations = 0;
        result.work_units = canonicalReportedWork(symmetry_pre_work);
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
        result.work_units = canonicalReportedWork(symmetry_pre_work);
        result.time_seconds = elapsed();
        if (did_presolve) {
            result.solution = presolver.postsolve({});
        }
        return result;
    }

    const bool using_transformed_problem = did_presolve || symmetry_cuts_added > 0;
    symmetry_stats_.cuts_applied = (symmetry_cuts_added > 0) &&
                                   using_transformed_problem;
    LpProblem original_problem;
    if (using_transformed_problem) {
        original_problem = std::move(problem_);
        problem_ = std::move(working_problem);
        refreshTreePresolveProfile();
    }
    auto restoreProblem = [&]() {
        if (!using_transformed_problem) return;
        problem_ = std::move(original_problem);
        refreshTreePresolveProfile();
    };

    auto applyPostsolve = [&](MipResult& result) {
        if (did_presolve) {
            // LP/MIP subsolvers already report values in the transformed space
            // including transformed obj_offset. Postsolve reconstructs primal
            // variables only; objective/bound offsets must not be added again.
            if (result.status == Status::Optimal || !result.solution.empty()) {
                result.solution = presolver.postsolve(result.solution);
            }
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
        result.work_units = canonicalReportedWork(lr.work_units + symmetry_pre_work);
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
        restoreProblem();
        return result;
    }

    branching_rule_.reset(problem_.num_cols);
    branching_stats_ = {};
    Real incumbent = kInf;
    std::vector<Real> best_solution;
    Int total_lp_iters = 0;
    double total_work = symmetry_pre_work;

    std::vector<Index> discrete_vars;
    discrete_vars.reserve(problem_.num_cols);
    for (Index j = 0; j < problem_.num_cols; ++j) {
        if (isDiscreteType(problem_.col_type[j])) discrete_vars.push_back(j);
    }

    const bool pre_root_stage_enabled =
        (pre_root_lp_free_enabled_ || pre_root_lp_light_enabled_) && !discrete_vars.empty();
    if (pre_root_stage_enabled) {
        pre_root_stats_.enabled = true;
        pre_root_stats_.lp_light_enabled = pre_root_lp_light_enabled_;
        pre_root_stats_.lp_light_available = pre_root_lp_light_enabled_ && hasLpLightCapability();

        std::optional<LpLightGuide> lp_light_guide;
        if (pre_root_stats_.lp_light_available) {
            lp_light_guide = solveLpLightGuide(problem_);
            if (lp_light_guide.has_value()) {
                pre_root_stats_.lp_light_lp_solves = 1;
                pre_root_stats_.lp_light_lp_iterations = lp_light_guide->iterations;
                pre_root_stats_.lp_light_lp_work = lp_light_guide->work_units;
                total_lp_iters += lp_light_guide->iterations;
                total_work += lp_light_guide->work_units;
            }
        }
        const bool run_lp_free_arms = pre_root_lp_free_enabled_;
        const bool run_lp_light_arms = lp_light_guide.has_value();
        std::vector<PreRootArm> available_arms;
        if (run_lp_free_arms) {
            available_arms.push_back(PreRootArm::FeasJump);
            available_arms.push_back(PreRootArm::Fpr);
            available_arms.push_back(PreRootArm::LocalMip);
        }
        if (run_lp_light_arms) {
            available_arms.push_back(PreRootArm::LpLightFpr);
            available_arms.push_back(PreRootArm::LpLightDiving);
        }
        const Int hw_threads = static_cast<Int>(
            std::max<unsigned>(1, std::thread::hardware_concurrency()));
        const Int stage_threads = std::max<Int>(1, std::min(num_threads_, hw_threads));
        const bool adaptive_portfolio_requested =
            pre_root_portfolio_enabled_ && available_arms.size() > 1;
        const bool force_fixed_for_determinism =
            (parallel_mode_ == ParallelMode::Deterministic) && (stage_threads > 1);
        pre_root_stats_.portfolio_enabled =
            adaptive_portfolio_requested && !force_fixed_for_determinism;
        const auto lp_light_primals = run_lp_light_arms
            ? std::span<const Real>(lp_light_guide->primals.data(), lp_light_guide->primals.size())
            : std::span<const Real>{};
        const auto lp_light_reduced_costs = run_lp_light_arms
            ? std::span<const Real>(lp_light_guide->reduced_costs.data(), lp_light_guide->reduced_costs.size())
            : std::span<const Real>{};

        if (verbose_ && pre_root_lp_light_enabled_ && !pre_root_stats_.lp_light_available) {
            log_.log("Pre-root LP-light requested but unavailable (backend=%s).\n",
                     lpLightBackendName());
        }
        if (verbose_ && pre_root_stats_.lp_light_available && !run_lp_light_arms) {
            log_.log("Pre-root LP-light guide solve unavailable for this model; "
                     "continuing with LP-free arms only.\n");
        }
        if (verbose_ && force_fixed_for_determinism && adaptive_portfolio_requested) {
            log_.log("Pre-root adaptive portfolio disabled in deterministic multi-thread mode; "
                     "using fixed schedule.\n");
        }
        if (!run_lp_free_arms && !run_lp_light_arms) {
            if (verbose_) {
                log_.log("Pre-root stage enabled but no usable arms are available.\n");
            }
        } else {
        class AdaptivePortfolioState {
        public:
            AdaptivePortfolioState(bool enabled,
                                   HeuristicRuntimeMode mode,
                                   std::span<const PreRootArm> arms)
                : enabled_(enabled), mode_(mode), arms_(arms.begin(), arms.end()) {
                alpha_.fill(1.0);
                beta_.fill(1.0);
                reward_sum_.fill(0.0);
                pulls_.fill(0);
                improvements_.fill(0);
            }

            PreRootArm select(Int round, std::mt19937_64& rng, bool local_mip_allowed) {
                std::lock_guard<std::mutex> lock(mutex_);
                ++epochs_;
                if (arms_.empty()) return PreRootArm::FeasJump;
                std::array<PreRootArm, kPreRootArmCount> eligible{};
                std::size_t eligible_count = 0;
                for (PreRootArm arm : arms_) {
                    if (!local_mip_allowed && arm == PreRootArm::LocalMip) continue;
                    eligible[eligible_count++] = arm;
                }
                if (eligible_count == 0) return PreRootArm::FeasJump;
                if (!enabled_) {
                    if (mode_ == HeuristicRuntimeMode::Deterministic) {
                        const std::size_t idx =
                            static_cast<std::size_t>(std::max<Int>(0, round)) % eligible_count;
                        ++pulls_[preRootArmIndex(eligible[idx])];
                        return eligible[idx];
                    }
                    const std::size_t idx = std::uniform_int_distribution<std::size_t>(
                        0, eligible_count - 1)(rng);
                    ++pulls_[preRootArmIndex(eligible[idx])];
                    return eligible[idx];
                }

                // Warm-up: make one pull from each available arm before sampling.
                for (std::size_t i = 0; i < eligible_count; ++i) {
                    const PreRootArm arm = eligible[i];
                    if (pulls_[preRootArmIndex(arm)] == 0) {
                        ++pulls_[preRootArmIndex(arm)];
                        return arm;
                    }
                }

                if (mode_ == HeuristicRuntimeMode::Deterministic) {
                    // Deterministic adaptive choice: pick highest posterior mean with stable tie-break.
                    PreRootArm best_arm = eligible[0];
                    double best_score = -1.0;
                    std::size_t best_idx = preRootArmIndex(best_arm);
                    for (std::size_t i = 0; i < eligible_count; ++i) {
                        const PreRootArm arm = eligible[i];
                        const std::size_t idx = preRootArmIndex(arm);
                        const double denom = alpha_[idx] + beta_[idx];
                        const double score = (denom > 0.0) ? (alpha_[idx] / denom) : 0.0;
                        if (score > best_score + 1e-12 ||
                            (std::abs(score - best_score) <= 1e-12 && idx < best_idx)) {
                            best_score = score;
                            best_arm = arm;
                            best_idx = idx;
                        }
                    }
                    ++pulls_[preRootArmIndex(best_arm)];
                    return best_arm;
                }

                PreRootArm best_arm = eligible[0];
                double best_score = -1.0;
                for (std::size_t i = 0; i < eligible_count; ++i) {
                    const PreRootArm arm = eligible[i];
                    const std::size_t idx = preRootArmIndex(arm);
                    const double score = sampleBeta(alpha_[idx], beta_[idx], rng);
                    if (score > best_score) {
                        best_score = score;
                        best_arm = arm;
                    }
                }
                ++pulls_[preRootArmIndex(best_arm)];
                return best_arm;
            }

            void update(PreRootArm arm, double reward, bool improved) {
                std::lock_guard<std::mutex> lock(mutex_);
                if (!enabled_ && mode_ == HeuristicRuntimeMode::Deterministic) {
                    return;
                }
                const std::size_t idx = preRootArmIndex(arm);
                const double clipped = std::clamp(reward, 0.0, 1.0);
                reward_sum_[idx] += clipped;
                alpha_[idx] += clipped;
                beta_[idx] += (1.0 - clipped);

                if (improved) {
                    ++improvements_[idx];
                    ++wins_total_;
                    effort_scale_ = std::min(1.75, effort_scale_ * 1.08);
                } else if (clipped <= 1e-9) {
                    ++stagnant_total_;
                    effort_scale_ = std::max(0.35, effort_scale_ * 0.94);
                }
            }

            [[nodiscard]] double currentBudget(double base_budget) const {
                std::lock_guard<std::mutex> lock(mutex_);
                return std::max(1.0, base_budget * effort_scale_);
            }
            [[nodiscard]] Int epochs() const {
                std::lock_guard<std::mutex> lock(mutex_);
                return epochs_;
            }
            [[nodiscard]] Int winsTotal() const {
                std::lock_guard<std::mutex> lock(mutex_);
                return wins_total_;
            }
            [[nodiscard]] Int stagnantTotal() const {
                std::lock_guard<std::mutex> lock(mutex_);
                return stagnant_total_;
            }
            [[nodiscard]] double effortScale() const {
                std::lock_guard<std::mutex> lock(mutex_);
                return effort_scale_;
            }
            [[nodiscard]] Int improvements(PreRootArm arm) const {
                std::lock_guard<std::mutex> lock(mutex_);
                return improvements_[preRootArmIndex(arm)];
            }
            [[nodiscard]] double rewardSum(PreRootArm arm) const {
                std::lock_guard<std::mutex> lock(mutex_);
                return reward_sum_[preRootArmIndex(arm)];
            }

        private:
            bool enabled_ = false;
            HeuristicRuntimeMode mode_ = HeuristicRuntimeMode::Deterministic;
            std::vector<PreRootArm> arms_;
            mutable std::mutex mutex_;
            std::array<double, kPreRootArmCount> alpha_{};
            std::array<double, kPreRootArmCount> beta_{};
            std::array<double, kPreRootArmCount> reward_sum_{};
            std::array<Int, kPreRootArmCount> pulls_{};
            std::array<Int, kPreRootArmCount> improvements_{};
            Int epochs_ = 0;
            Int wins_total_ = 0;
            Int stagnant_total_ = 0;
            double effort_scale_ = 1.0;
        };

        AdaptivePortfolioState portfolio(
            pre_root_stats_.portfolio_enabled,
            heuristicRuntimeMode(parallel_mode_),
            available_arms);
        std::vector<LpProblem> stage_thread_problems;
        if (stage_threads > 1) {
            stage_thread_problems.reserve(static_cast<std::size_t>(stage_threads));
            for (Int t = 0; t < stage_threads; ++t) {
                stage_thread_problems.push_back(problem_);
            }
        }
        auto stageProblemForThread = [&](Int thread_id) -> const LpProblem& {
            if (stage_thread_problems.empty()) return problem_;
            return stage_thread_problems[static_cast<std::size_t>(thread_id)];
        };
        std::atomic<Int> next_round{0};
        std::atomic<bool> should_stop{false};
        std::atomic<Int> calls{0};
        std::atomic<Int> improvements{0};
        std::atomic<Int> feasible_found{0};
        std::atomic<Int> early_stops{0};
        std::atomic<Int> fj_calls{0};
        std::atomic<Int> fpr_calls{0};
        std::atomic<Int> local_mip_calls{0};
        std::atomic<Int> lp_light_calls{0};
        std::atomic<Int> lp_light_fpr_calls{0};
        std::atomic<Int> lp_light_diving_calls{0};
        AtomicWorkUnits stage_work;
        std::atomic<double> first_feasible_seconds{kInf};
        constexpr uint64_t kNoFirstFeasibleWorkTick =
            std::numeric_limits<uint64_t>::max();
        std::atomic<uint64_t> first_feasible_work_tick{kNoFirstFeasibleWorkTick};
        SolutionPool pre_root_pool(problem_.sense);
        Real deterministic_stage_incumbent = pre_root_pool.bestObjective();
        std::vector<Real> deterministic_incumbent_values;
        const auto stage_start = std::chrono::steady_clock::now();
        auto stageElapsed = [&]() -> double {
            return std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - stage_start).count();
        };
        auto addStageWork = [&](double add) {
            stage_work.count(workUnitTicks(add));
        };
        auto roundSeed = [&](Int round) -> uint64_t {
            uint64_t x = heuristic_seed_ ^
                (0x9e3779b97f4a7c15ULL * static_cast<uint64_t>(std::max<Int>(0, round) + 1));
            x += 0x9e3779b97f4a7c15ULL;
            x = (x ^ (x >> 30U)) * 0xbf58476d1ce4e5b9ULL;
            x = (x ^ (x >> 27U)) * 0x94d049bb133111ebULL;
            x = x ^ (x >> 31U);
            return x;
        };

        auto countArmCall = [&](PreRootArm arm) {
            switch (arm) {
                case PreRootArm::FeasJump: ++fj_calls; break;
                case PreRootArm::Fpr: ++fpr_calls; break;
                case PreRootArm::LocalMip: ++local_mip_calls; break;
                case PreRootArm::LpLightFpr:
                    ++lp_light_calls;
                    ++lp_light_fpr_calls;
                    break;
                case PreRootArm::LpLightDiving:
                    ++lp_light_calls;
                    ++lp_light_diving_calls;
                    break;
            }
        };

        struct PreRootRoundResult {
            Int round = -1;
            Int thread_id = 0;
            PreRootArm arm = PreRootArm::FeasJump;
            std::optional<HeuristicSolution> candidate;
            double local_work = 0.0;
        };

        auto runPreRootRound =
            [&](Int round,
                Int thread_id,
                std::mt19937_64& round_rng,
                Real deterministic_incumbent_for_round,
                std::span<const Real> deterministic_incumbent_span) -> PreRootRoundResult {
            PreRootRoundResult out;
            out.round = round;
            out.thread_id = thread_id;
            const LpProblem& round_problem = stageProblemForThread(thread_id);

            const Real incumbent_for_round =
                (parallel_mode_ == ParallelMode::Deterministic)
                    ? deterministic_incumbent_for_round
                    : pre_root_pool.bestObjective();
            std::optional<HeuristicSolution> incumbent_sol;
            std::span<const Real> incumbent_span_for_round = deterministic_incumbent_span;
            if (parallel_mode_ != ParallelMode::Deterministic) {
                incumbent_sol = pre_root_pool.bestSolution();
                if (incumbent_sol.has_value()) {
                    incumbent_span_for_round = std::span<const Real>(
                        incumbent_sol->values.data(),
                        incumbent_sol->values.size());
                } else {
                    incumbent_span_for_round = std::span<const Real>{};
                }
            }
            const bool has_incumbent =
                std::isfinite(incumbent_for_round) && incumbent_for_round < kInf &&
                !incumbent_span_for_round.empty();
            out.arm = portfolio.select(round, round_rng, has_incumbent);

            switch (out.arm) {
                case PreRootArm::FeasJump: {
                    out.candidate = runLpFreeFeasJump(round_problem, discrete_vars, round_rng,
                                                      incumbent_for_round,
                                                      out.local_work);
                    break;
                }
                case PreRootArm::Fpr: {
                    out.candidate = runLpFreeFpr(round_problem, discrete_vars, round_rng,
                                                 incumbent_for_round,
                                                 out.local_work);
                    break;
                }
                case PreRootArm::LocalMip: {
                    const std::span<const Real> incumbent_span = incumbent_span_for_round;
                    if (incumbent_span.empty()) break;
                    out.candidate = runLpFreeLocalMip(round_problem, discrete_vars, round_rng,
                                                      incumbent_for_round,
                                                      incumbent_span, out.local_work);
                    break;
                }
                case PreRootArm::LpLightFpr: {
                    if (!run_lp_light_arms) break;
                    out.candidate = runLpLightFpr(round_problem, discrete_vars,
                                                  lp_light_primals, lp_light_reduced_costs,
                                                  round_rng,
                                                  incumbent_for_round,
                                                  out.local_work);
                    break;
                }
                case PreRootArm::LpLightDiving: {
                    if (!run_lp_light_arms) break;
                    out.candidate = runLpLightDiving(round_problem, discrete_vars,
                                                     lp_light_primals, lp_light_reduced_costs,
                                                     incumbent_for_round,
                                                     out.local_work);
                    break;
                }
            }
            return out;
        };

        auto commitPreRootRound = [&](const PreRootRoundResult& out) {
            ++calls;
            countArmCall(out.arm);
            addStageWork(out.local_work);

            const Real incumbent_before = pre_root_pool.bestObjective();
            const bool had_incumbent_before =
                std::isfinite(incumbent_before) && incumbent_before < kInf;
            bool accepted = false;
            if (out.candidate.has_value() &&
                pre_root_pool.submit(*out.candidate, "pre_root_stage", out.thread_id)) {
                accepted = true;
                ++improvements;
                ++feasible_found;
                if (parallel_mode_ == ParallelMode::Deterministic) {
                    deterministic_stage_incumbent = out.candidate->objective;
                    deterministic_incumbent_values = out.candidate->values;
                    const uint64_t seen_tick = stage_work.ticks();
                    uint64_t old_tick =
                        first_feasible_work_tick.load(std::memory_order_relaxed);
                    while (seen_tick < old_tick &&
                           !first_feasible_work_tick.compare_exchange_weak(
                               old_tick, seen_tick, std::memory_order_relaxed)) {}
                } else {
                    const double seen = stageElapsed();
                    auto old_first = first_feasible_seconds.load(std::memory_order_relaxed);
                    while (seen < old_first &&
                           !first_feasible_seconds.compare_exchange_weak(
                               old_first, seen, std::memory_order_relaxed)) {}
                }
                if (pre_root_lp_free_early_stop_) {
                    should_stop.store(true, std::memory_order_relaxed);
                    ++early_stops;
                }
            }

            double reward = 0.0;
            if (accepted && !had_incumbent_before) reward = 1.0;
            else if (accepted) reward = 0.8;
            else if (out.candidate.has_value()) reward = 0.2;
            portfolio.update(out.arm, reward, accepted);
        };

        if (parallel_mode_ == ParallelMode::Deterministic && stage_threads > 1) {
            const uint64_t budget_ticks =
                workUnitTicks(portfolio.currentBudget(pre_root_lp_free_work_budget_));
            Int next_round_value = 0;
            while (!should_stop.load(std::memory_order_relaxed)) {
                if (next_round_value >= pre_root_lp_free_max_rounds_) break;
                if (stage_work.ticks() >= budget_ticks) break;

                const Int batch_size = std::min<Int>(
                    stage_threads, pre_root_lp_free_max_rounds_ - next_round_value);
                std::vector<PreRootRoundResult> batch(
                    static_cast<std::size_t>(batch_size));
                const Real batch_incumbent = deterministic_stage_incumbent;
                const std::span<const Real> batch_incumbent_span(
                    deterministic_incumbent_values.data(),
                    deterministic_incumbent_values.size());
                std::vector<std::thread> batch_threads;
                batch_threads.reserve(static_cast<std::size_t>(batch_size));
                for (Int t = 0; t < batch_size; ++t) {
                    const Int round = next_round_value + t;
                    batch_threads.emplace_back([&, t, round]() {
                        std::mt19937_64 round_rng(roundSeed(round));
                        batch[static_cast<std::size_t>(t)] =
                            runPreRootRound(round, t, round_rng,
                                            batch_incumbent,
                                            batch_incumbent_span);
                    });
                }
                for (auto& w : batch_threads) w.join();

                next_round_value += batch_size;
                next_round.store(next_round_value, std::memory_order_relaxed);

                for (Int t = 0; t < batch_size; ++t) {
                    commitPreRootRound(batch[static_cast<std::size_t>(t)]);
                    if (should_stop.load(std::memory_order_relaxed)) break;
                    if (stage_work.ticks() >= budget_ticks) break;
                }
            }
        } else {
            auto worker = [&](Int thread_id) {
                std::mt19937_64 thread_rng(
                    heuristic_seed_ ^
                    (0x9e3779b97f4a7c15ULL * static_cast<uint64_t>(thread_id + 1)));
                while (!should_stop.load(std::memory_order_relaxed)) {
                    if (stage_work.units() >=
                        portfolio.currentBudget(pre_root_lp_free_work_budget_)) {
                        break;
                    }
                    const Int round = next_round.fetch_add(1, std::memory_order_relaxed);
                    if (round >= pre_root_lp_free_max_rounds_) break;

                    std::mt19937_64 round_rng(
                        parallel_mode_ == ParallelMode::Deterministic
                            ? roundSeed(round)
                            : thread_rng());
                    const std::span<const Real> incumbent_span(
                        deterministic_incumbent_values.data(),
                        deterministic_incumbent_values.size());
                    const auto out = runPreRootRound(round, thread_id, round_rng,
                                                     deterministic_stage_incumbent,
                                                     incumbent_span);
                    commitPreRootRound(out);
                }
            };

            std::vector<std::thread> workers;
            workers.reserve(stage_threads);
            for (Int t = 0; t < stage_threads; ++t) {
                workers.emplace_back(worker, t);
            }
            for (auto& w : workers) w.join();
        }

        if (auto best = pre_root_pool.bestSolution(); best.has_value()) {
            incumbent = best->objective;
            best_solution = std::move(best->values);
        }
        pre_root_stats_.calls = calls.load(std::memory_order_relaxed);
        pre_root_stats_.rounds = pre_root_stats_.calls;
        pre_root_stats_.improvements = improvements.load(std::memory_order_relaxed);
        pre_root_stats_.feasible_found = feasible_found.load(std::memory_order_relaxed);
        pre_root_stats_.early_stops = early_stops.load(std::memory_order_relaxed);
        pre_root_stats_.fj_calls = fj_calls.load(std::memory_order_relaxed);
        pre_root_stats_.fpr_calls = fpr_calls.load(std::memory_order_relaxed);
        pre_root_stats_.local_mip_calls = local_mip_calls.load(std::memory_order_relaxed);
        pre_root_stats_.lp_light_calls = lp_light_calls.load(std::memory_order_relaxed);
        pre_root_stats_.lp_light_fpr_calls = lp_light_fpr_calls.load(std::memory_order_relaxed);
        pre_root_stats_.lp_light_diving_calls =
            lp_light_diving_calls.load(std::memory_order_relaxed);
        pre_root_stats_.portfolio_epochs = portfolio.epochs();
        pre_root_stats_.portfolio_wins = portfolio.winsTotal();
        pre_root_stats_.portfolio_stagnant = portfolio.stagnantTotal();
        pre_root_stats_.effort_scale_final = portfolio.effortScale();
        pre_root_stats_.fj_improvements = portfolio.improvements(PreRootArm::FeasJump);
        pre_root_stats_.fpr_improvements = portfolio.improvements(PreRootArm::Fpr);
        pre_root_stats_.local_mip_improvements = portfolio.improvements(PreRootArm::LocalMip);
        pre_root_stats_.lp_light_fpr_improvements =
            portfolio.improvements(PreRootArm::LpLightFpr);
        pre_root_stats_.lp_light_diving_improvements =
            portfolio.improvements(PreRootArm::LpLightDiving);
        pre_root_stats_.fj_reward = portfolio.rewardSum(PreRootArm::FeasJump);
        pre_root_stats_.fpr_reward = portfolio.rewardSum(PreRootArm::Fpr);
        pre_root_stats_.local_mip_reward = portfolio.rewardSum(PreRootArm::LocalMip);
        pre_root_stats_.lp_light_fpr_reward = portfolio.rewardSum(PreRootArm::LpLightFpr);
        pre_root_stats_.lp_light_diving_reward =
            portfolio.rewardSum(PreRootArm::LpLightDiving);
        pre_root_stats_.work_units = stage_work.units();
        if (parallel_mode_ == ParallelMode::Deterministic) {
            pre_root_stats_.time_seconds = pre_root_stats_.work_units;
            const uint64_t first_tick =
                first_feasible_work_tick.load(std::memory_order_relaxed);
            pre_root_stats_.time_to_first_feasible =
                (first_tick == kNoFirstFeasibleWorkTick)
                    ? kInf
                    : static_cast<double>(first_tick) * 1e-6;
        } else {
            pre_root_stats_.time_seconds = stageElapsed();
            pre_root_stats_.time_to_first_feasible =
                first_feasible_seconds.load(std::memory_order_relaxed);
        }
        pre_root_stats_.incumbent_at_root = incumbent;
        total_work += pre_root_stats_.work_units;

        if (verbose_) {
            if (std::isfinite(pre_root_stats_.time_to_first_feasible)) {
                log_.log(
                    "Pre-root stage (%s, %d thread%s): calls=%d fj=%d fpr=%d local=%d "
                    "lpfpr=%d lpdiv=%d "
                    "found=%d best=%.10e work=%.3f time=%.3fs first=%.3fs\n",
                    parallelModeName(parallel_mode_),
                    stage_threads,
                    stage_threads == 1 ? "" : "s",
                    pre_root_stats_.calls,
                    pre_root_stats_.fj_calls,
                    pre_root_stats_.fpr_calls,
                    pre_root_stats_.local_mip_calls,
                    pre_root_stats_.lp_light_fpr_calls,
                    pre_root_stats_.lp_light_diving_calls,
                    pre_root_stats_.feasible_found,
                    pre_root_stats_.incumbent_at_root,
                    pre_root_stats_.work_units,
                    pre_root_stats_.time_seconds,
                    pre_root_stats_.time_to_first_feasible);
            } else {
                log_.log(
                    "Pre-root stage (%s, %d thread%s): calls=%d fj=%d fpr=%d local=%d "
                    "lpfpr=%d lpdiv=%d "
                    "found=%d best=%.10e work=%.3f time=%.3fs first=n/a\n",
                    parallelModeName(parallel_mode_),
                    stage_threads,
                    stage_threads == 1 ? "" : "s",
                    pre_root_stats_.calls,
                    pre_root_stats_.fj_calls,
                    pre_root_stats_.fpr_calls,
                    pre_root_stats_.local_mip_calls,
                    pre_root_stats_.lp_light_fpr_calls,
                    pre_root_stats_.lp_light_diving_calls,
                    pre_root_stats_.feasible_found,
                    pre_root_stats_.incumbent_at_root,
                    pre_root_stats_.work_units,
                    pre_root_stats_.time_seconds);
            }
            if (pre_root_stats_.lp_light_lp_solves > 0) {
                log_.log(
                    "Pre-root LP-light guide (%s): solves=%d iters=%d work=%.3f\n",
                    lpLightBackendName(),
                    pre_root_stats_.lp_light_lp_solves,
                    pre_root_stats_.lp_light_lp_iterations,
                    pre_root_stats_.lp_light_lp_work);
            }
            log_.log(
                "Pre-root portfolio: mode=%s epochs=%d wins=%d stagnant=%d effort=%.2f "
                "reward[fj=%.2f fpr=%.2f local=%.2f lpfpr=%.2f lpdiv=%.2f] "
                "impr[fj=%d fpr=%d local=%d lpfpr=%d lpdiv=%d]\n",
                pre_root_stats_.portfolio_enabled ? "adaptive" : "fixed",
                pre_root_stats_.portfolio_epochs,
                pre_root_stats_.portfolio_wins,
                pre_root_stats_.portfolio_stagnant,
                pre_root_stats_.effort_scale_final,
                pre_root_stats_.fj_reward,
                pre_root_stats_.fpr_reward,
                pre_root_stats_.local_mip_reward,
                pre_root_stats_.lp_light_fpr_reward,
                pre_root_stats_.lp_light_diving_reward,
                pre_root_stats_.fj_improvements,
                pre_root_stats_.fpr_improvements,
                pre_root_stats_.local_mip_improvements,
                pre_root_stats_.lp_light_fpr_improvements,
                pre_root_stats_.lp_light_diving_improvements);
        }
    }
    }

    // Solve root LP.
    DualSimplexSolver lp;
    lp.load(problem_);
    lp.setVerbose(false);

    enum class RootBackend : int {
        Dual = 0,
        Barrier = 1,
        Pdlp = 2,
    };

    struct RootCandidateResult {
        RootBackend backend = RootBackend::Dual;
        LpResult lp_result{};
        std::vector<Real> primals;
        std::vector<BasisStatus> basis;
        bool used_gpu = false;
        double seconds = 0.0;
    };

    auto backendName = [](RootBackend backend) {
        switch (backend) {
            case RootBackend::Dual: return "dual";
            case RootBackend::Barrier: return "barrier";
            case RootBackend::Pdlp: return "pdlp";
            default: return "dual";
        }
    };

    auto backendWarmStartRank = [](RootBackend backend) {
        switch (backend) {
            case RootBackend::Dual: return 0;
            case RootBackend::Barrier: return 1;
            case RootBackend::Pdlp: return 2;
            default: return 3;
        }
    };

    auto betterBound = [&](Real candidate, Real incumbent) {
        if (!std::isfinite(incumbent)) return true;
        if (problem_.sense == Sense::Minimize) return candidate < incumbent - 1e-7;
        return candidate > incumbent + 1e-7;
    };

    auto comparableBound = [](Real a, Real b) {
        const Real scale = std::max<Real>({1.0, std::abs(a), std::abs(b)});
        return std::abs(a - b) <= 1e-7 * scale;
    };

    auto runRootBackend = [&](RootBackend backend,
                              const std::atomic<bool>* stop_flag) -> RootCandidateResult {
        RootCandidateResult out;
        out.backend = backend;

        if (stop_flag != nullptr && stop_flag->load(std::memory_order_relaxed)) {
            out.lp_result.status = Status::IterLimit;
            return out;
        }

        const auto started = std::chrono::steady_clock::now();
        if (backend == RootBackend::Dual) {
            DualSimplexSolver solver;
            solver.load(problem_);
            solver.setVerbose(false);
            auto opts = solver.getOptions();
            opts.stop_flag = stop_flag;
            solver.setOptions(opts);
            out.lp_result = solver.solve();
            out.primals = solver.getPrimalValues();
            out.basis = solver.getBasis();
        } else if (backend == RootBackend::Barrier) {
            BarrierSolver solver;
            BarrierOptions opts;
            opts.verbose = false;
            opts.algorithm = barrier_algorithm_;
            opts.stop_flag = stop_flag;
            solver.setOptions(opts);
            solver.load(problem_);
            out.lp_result = solver.solve();
            out.primals = solver.getPrimalValues();
            out.used_gpu = solver.usedGpu();
        } else {
            PdlpSolver solver;
            PdlpOptions opts;
            opts.verbose = false;
            opts.use_gpu = (barrier_algorithm_ == BarrierAlgorithm::Auto ||
                           barrier_algorithm_ == BarrierAlgorithm::GpuCholesky ||
                           barrier_algorithm_ == BarrierAlgorithm::GpuAugmented);
            opts.stop_flag = stop_flag;
            solver.setOptions(opts);
            solver.load(problem_);
            out.lp_result = solver.solve();
            out.primals = solver.getPrimalValues();
            out.used_gpu = solver.usedGpu();
        }
        out.seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        return out;
    };

    bool root_used_dual = true;
    std::vector<Real> root_primals;
    std::vector<BasisStatus> root_basis;
    const char* root_backend_used = rootPolicyName(root_lp_policy_);

    auto t0 = std::chrono::steady_clock::now();
    LpResult root_result;
    Int root_lp_iters_total = 0;
    double root_lp_work_total = 0.0;
    if (root_lp_policy_ == RootLpPolicy::BarrierRoot) {
        root_used_dual = false;
        const auto result = runRootBackend(RootBackend::Barrier, nullptr);
        root_result = result.lp_result;
        root_primals = result.primals;
        root_lp_iters_total = root_result.iterations;
        root_lp_work_total = root_result.work_units;
        root_backend_used = "barrier";
        if (verbose_) {
            log_.log("Root barrier mode%s.\n", result.used_gpu ? " (GPU backend)" : "");
        }
    } else if (root_lp_policy_ == RootLpPolicy::PdlpRoot) {
        root_used_dual = false;
        const auto result = runRootBackend(RootBackend::Pdlp, nullptr);
        root_result = result.lp_result;
        root_primals = result.primals;
        root_lp_iters_total = root_result.iterations;
        root_lp_work_total = root_result.work_units;
        root_backend_used = "pdlp";
        if (verbose_) {
            log_.log("Root PDLP mode%s.\n", result.used_gpu ? " (GPU backend)" : "");
        }
    } else if (root_lp_policy_ == RootLpPolicy::ConcurrentRootExperimental) {
        root_used_dual = false;
        lp_stats_.root_race_runs = 1;
        std::vector<RootCandidateResult> race_results;
        race_results.reserve(3);

        if (parallel_mode_ == ParallelMode::Deterministic) {
            race_results.push_back(runRootBackend(RootBackend::Dual, nullptr));
            race_results.push_back(runRootBackend(RootBackend::Barrier, nullptr));
            race_results.push_back(runRootBackend(RootBackend::Pdlp, nullptr));
        } else {
            std::atomic<bool> race_stop{false};
            std::mutex race_mutex;
            auto run_and_store = [&](RootBackend backend) {
                auto result = runRootBackend(backend, &race_stop);
                bool mark_stop = false;
                {
                    std::lock_guard<std::mutex> lock(race_mutex);
                    race_results.push_back(result);
                    if (result.lp_result.status == Status::Optimal &&
                        !race_stop.load(std::memory_order_relaxed)) {
                        mark_stop = true;
                    }
                }
                if (mark_stop) {
                    race_stop.store(true, std::memory_order_relaxed);
                }
            };

            std::thread dual_thread(run_and_store, RootBackend::Dual);
            std::thread barrier_thread(run_and_store, RootBackend::Barrier);
            std::thread pdlp_thread(run_and_store, RootBackend::Pdlp);
            dual_thread.join();
            barrier_thread.join();
            pdlp_thread.join();
        }

        lp_stats_.root_race_candidates = static_cast<Int>(race_results.size());
        for (const auto& candidate : race_results) {
            root_lp_iters_total += candidate.lp_result.iterations;
            root_lp_work_total += candidate.lp_result.work_units;
            if (candidate.lp_result.status == Status::IterLimit) {
                ++lp_stats_.root_race_cancelled;
            }
        }

        std::optional<std::size_t> winner;
        Real best_obj = (problem_.sense == Sense::Minimize) ? kInf : -kInf;
        for (std::size_t i = 0; i < race_results.size(); ++i) {
            const auto& c = race_results[i];
            if (c.lp_result.status != Status::Optimal) continue;
            if (!winner.has_value() || betterBound(c.lp_result.objective, best_obj)) {
                winner = i;
                best_obj = c.lp_result.objective;
                continue;
            }
            if (!comparableBound(c.lp_result.objective, best_obj)) continue;
            const auto& w = race_results[*winner];
            if (c.seconds + 1e-12 < w.seconds) {
                winner = i;
            } else if (std::abs(c.seconds - w.seconds) <= 1e-12 &&
                       backendWarmStartRank(c.backend) < backendWarmStartRank(w.backend)) {
                winner = i;
            }
        }

        if (!winner.has_value()) {
            for (std::size_t i = 0; i < race_results.size(); ++i) {
                if (race_results[i].lp_result.status == Status::Infeasible ||
                    race_results[i].lp_result.status == Status::Unbounded) {
                    winner = i;
                    break;
                }
            }
        }

        if (!winner.has_value()) {
            // Safety fallback if all race arms terminate without a usable root result.
            auto dual_fallback = runRootBackend(RootBackend::Dual, nullptr);
            race_results.push_back(dual_fallback);
            winner = race_results.size() - 1;
            root_lp_iters_total += dual_fallback.lp_result.iterations;
            root_lp_work_total += dual_fallback.lp_result.work_units;
        }

        const auto& selected = race_results[*winner];
        root_result = selected.lp_result;
        root_primals = selected.primals;
        root_basis = selected.basis;
        lp_stats_.root_race_winner_seconds = selected.seconds;
        root_backend_used = backendName(selected.backend);
        switch (selected.backend) {
            case RootBackend::Dual: ++lp_stats_.root_race_dual_wins; break;
            case RootBackend::Barrier: ++lp_stats_.root_race_barrier_wins; break;
            case RootBackend::Pdlp: ++lp_stats_.root_race_pdlp_wins; break;
            default: break;
        }

        if (verbose_) {
            for (const auto& candidate : race_results) {
                log_.log("Root race arm=%s status=%d obj=%.10e iters=%d time=%.4fs%s\n",
                         backendName(candidate.backend),
                         static_cast<int>(candidate.lp_result.status),
                         candidate.lp_result.objective,
                         candidate.lp_result.iterations,
                         candidate.seconds,
                         candidate.used_gpu ? " gpu" : "");
            }
            log_.log("Root race winner=%s time=%.4fs\n",
                     root_backend_used,
                     lp_stats_.root_race_winner_seconds);
        }
    } else {
        lp.setVerbose(verbose_);
        root_result = lp.solve();
        root_primals = lp.getPrimalValues();
        root_basis = lp.getBasis();
        root_lp_iters_total = root_result.iterations;
        root_lp_work_total = root_result.work_units;
        root_backend_used = "dual";
    }
    lp_stats_.root_lp_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();

    total_lp_iters += root_lp_iters_total;
    total_work += root_lp_work_total;

    if (root_result.status == Status::Infeasible) {
        if (verbose_) log_.log("Root LP infeasible.\n");
        MipResult result;
        result.status = Status::Infeasible;
        result.nodes = 1;
        result.lp_iterations = total_lp_iters;
        result.work_units = canonicalReportedWork(total_work);
        result.time_seconds = elapsed();
        restoreProblem();
        return result;
    }

    if (root_result.status == Status::Unbounded) {
        if (verbose_) log_.log("Root LP unbounded.\n");
        MipResult result;
        result.status = Status::Unbounded;
        result.nodes = 1;
        result.lp_iterations = total_lp_iters;
        result.work_units = canonicalReportedWork(total_work);
        result.time_seconds = elapsed();
        restoreProblem();
        return result;
    }

    if (root_result.status != Status::Optimal) {
        if (verbose_) log_.log("Root LP failed: status %d\n", static_cast<int>(root_result.status));
        MipResult result;
        result.status = Status::Error;
        result.nodes = 1;
        result.lp_iterations = total_lp_iters;
        result.work_units = canonicalReportedWork(total_work);
        result.time_seconds = elapsed();
        restoreProblem();
        return result;
    }

    Real root_bound = root_result.objective;

    if (verbose_) {
        std::fflush(stdout);  // flush LP solver's printf output before Logger write()
        log_.log("Root LP (%s -> %s): obj = %.10e, %d iters\n",
                 rootPolicyName(root_lp_policy_), root_backend_used, root_bound, root_result.iterations);
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
            result.work_units = canonicalReportedWork(total_work);
            result.time_seconds = elapsed();
            restoreProblem();
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
            result.work_units = canonicalReportedWork(total_work);
            result.time_seconds = elapsed();
            restoreProblem();
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
            result.work_units = canonicalReportedWork(total_work);
            result.time_seconds = elapsed();
            restoreProblem();
            return result;
        }
        root_bound = sync.objective;
        root_primals = lp.getPrimalValues();
        root_basis = lp.getBasis();
    }

    // Run cutting planes at root (suppress LP iteration logs — we log per round).
    LpProblem root_certificate_problem;
    LpProblem* root_certificate_problem_ptr = nullptr;
    if (exact_refinement_mode_ != ExactRefinementMode::Off) {
        root_certificate_problem = problem_;
        root_certificate_problem_ptr = &root_certificate_problem;
    }
    lp.setVerbose(false);
    if (cuts_enabled_ && problem_.hasIntegers()) {
        if (root_used_dual || !root_basis.empty()) {
            t0 = std::chrono::steady_clock::now();
            Int cuts_added =
                runCuttingPlanes(lp, total_lp_iters, total_work, root_certificate_problem_ptr);
            lp_stats_.root_cut_lp_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();
            if (cuts_added > 0) {
                root_bound = lp.getObjective();
                root_primals = lp.getPrimalValues();
                root_basis = lp.getBasis();
            }
        }
    }

    auto objectiveMismatchAllowed = [&](const LpCertificateMetrics& metrics,
                                        Real reported_obj,
                                        Real tol) {
        const Real scale =
            std::max<Real>({1.0, std::abs(reported_obj), std::abs(metrics.recomputed_objective)});
        return metrics.objective_mismatch <= tol * scale;
    };
    auto certificatePassed = [&](const LpCertificateMetrics& metrics,
                                 Real reported_obj) {
        if (metrics.max_row_violation > exact_refinement_certificate_tol_) return false;
        if (metrics.max_col_violation > exact_refinement_certificate_tol_) return false;
        if (!objectiveMismatchAllowed(metrics, reported_obj,
                                      exact_refinement_certificate_tol_)) {
            return false;
        }
        if (exact_refinement_rational_check_ &&
            (!metrics.rational_supported || !metrics.rational_ok)) {
            return false;
        }
        return true;
    };

    if (exact_refinement_mode_ != ExactRefinementMode::Off) {
        const LpProblem& certificate_problem =
            (root_certificate_problem_ptr != nullptr) ? *root_certificate_problem_ptr : problem_;
        double eval_work_units = 0.0;
        auto baseline_metrics = evaluateLpCertificate(
            certificate_problem, root_primals, root_bound,
            exact_refinement_rational_check_,
            exact_refinement_certificate_tol_,
            exact_refinement_rational_scale_,
            &eval_work_units);
        exact_refinement_stats_.max_row_violation_before = baseline_metrics.max_row_violation;
        exact_refinement_stats_.max_col_violation_before = baseline_metrics.max_col_violation;
        exact_refinement_stats_.objective_mismatch_before = baseline_metrics.objective_mismatch;

        const bool warning_trigger =
            baseline_metrics.max_row_violation > exact_refinement_warning_tol_ ||
            baseline_metrics.max_col_violation > exact_refinement_warning_tol_ ||
            !objectiveMismatchAllowed(baseline_metrics, root_bound,
                                      exact_refinement_warning_tol_) ||
            (exact_refinement_rational_check_ &&
             (!baseline_metrics.rational_supported || !baseline_metrics.rational_ok));

        const bool trigger_refinement =
            (exact_refinement_mode_ == ExactRefinementMode::On) ||
            (exact_refinement_mode_ == ExactRefinementMode::Auto && warning_trigger);
        exact_refinement_stats_.triggered = trigger_refinement;

        auto refined_primals = root_primals;
        Real refined_objective = root_bound;
        auto refined_metrics = baseline_metrics;

        if (trigger_refinement) {
            for (Int round = 0; round < exact_refinement_max_rounds_; ++round) {
                ++exact_refinement_stats_.rounds;

                iterativePrimalRepair(certificate_problem, refined_primals,
                                      exact_refinement_certificate_tol_,
                                      exact_refinement_repair_passes_,
                                      &eval_work_units);
                exact_refinement_stats_.repair_passes +=
                    exact_refinement_repair_passes_;

                refined_metrics = evaluateLpCertificate(
                    certificate_problem, refined_primals, refined_objective,
                    exact_refinement_rational_check_,
                    exact_refinement_certificate_tol_,
                    exact_refinement_rational_scale_,
                    &eval_work_units);
                if (certificatePassed(refined_metrics, refined_objective)) {
                    break;
                }

                auto refine_result = lp.solve();
                ++exact_refinement_stats_.resolve_calls;
                exact_refinement_stats_.resolve_iterations +=
                    refine_result.iterations;
                exact_refinement_stats_.resolve_work_units +=
                    refine_result.work_units;
                total_lp_iters += refine_result.iterations;
                total_work += refine_result.work_units;
                if (refine_result.status != Status::Optimal) {
                    break;
                }

                refined_objective = refine_result.objective;
                refined_primals = lp.getPrimalValues();
                refined_metrics = evaluateLpCertificate(
                    certificate_problem, refined_primals, refined_objective,
                    exact_refinement_rational_check_,
                    exact_refinement_certificate_tol_,
                    exact_refinement_rational_scale_,
                    &eval_work_units);
                if (certificatePassed(refined_metrics, refined_objective)) {
                    break;
                }
            }
        }

        if (trigger_refinement &&
            objectiveMismatchAllowed(refined_metrics, refined_objective,
                                     exact_refinement_certificate_tol_) &&
            refined_metrics.max_row_violation <=
                baseline_metrics.max_row_violation + exact_refinement_certificate_tol_ &&
            refined_metrics.max_col_violation <=
                baseline_metrics.max_col_violation + exact_refinement_certificate_tol_) {
            root_primals = refined_primals;
            root_bound = refined_objective;
        }

        exact_refinement_stats_.max_row_violation_after = refined_metrics.max_row_violation;
        exact_refinement_stats_.max_col_violation_after = refined_metrics.max_col_violation;
        exact_refinement_stats_.objective_mismatch_after = refined_metrics.objective_mismatch;
        exact_refinement_stats_.rows_evaluated = refined_metrics.rows_evaluated;
        exact_refinement_stats_.cols_evaluated = refined_metrics.cols_evaluated;
        exact_refinement_stats_.rational_supported = refined_metrics.rational_supported;
        exact_refinement_stats_.rational_certificate_passed =
            (!exact_refinement_rational_check_) ||
            (refined_metrics.rational_supported && refined_metrics.rational_ok);
        exact_refinement_stats_.certificate_passed =
            certificatePassed(refined_metrics, refined_objective);
        exact_refinement_stats_.evaluation_work_units = eval_work_units;
        total_work += eval_work_units;

        if (verbose_) {
            log_.log(
                "Exact refinement: mode=%s triggered=%s rounds=%d resolves=%d "
                "row_violation=%.3e->%.3e col_violation=%.3e->%.3e "
                "obj_mismatch=%.3e->%.3e cert=%s rational=%s\n",
                exactRefinementModeName(exact_refinement_mode_),
                trigger_refinement ? "yes" : "no",
                exact_refinement_stats_.rounds,
                exact_refinement_stats_.resolve_calls,
                exact_refinement_stats_.max_row_violation_before,
                exact_refinement_stats_.max_row_violation_after,
                exact_refinement_stats_.max_col_violation_before,
                exact_refinement_stats_.max_col_violation_after,
                exact_refinement_stats_.objective_mismatch_before,
                exact_refinement_stats_.objective_mismatch_after,
                exact_refinement_stats_.certificate_passed ? "ok" : "warn",
                exact_refinement_stats_.rational_certificate_passed ? "ok" : "warn");
        }
    }

    // Check if root solution is integer feasible.
    root_basis = lp.getBasis();
    HeuristicRuntimeConfig runtime_config = makeHeuristicRuntimeConfig();
    SolutionPool solution_pool(problem_.sense);
    HeuristicRuntime root_runtime(runtime_config);
    if (incumbent < kInf && !best_solution.empty()) {
        solution_pool.submit({best_solution, incumbent}, "pre_root_stage", 0);
    }

    if (isFeasibleLp(root_primals) && isFeasibleMip(root_primals)) {
        incumbent = root_bound;
        best_solution = root_primals;
        solution_pool.submit({best_solution, incumbent}, "root_lp", 0);
        if (verbose_) log_.log("Root solution is integer feasible!\n");
        root_runtime.finish();
        MipResult result;
        result.status = Status::Optimal;
        result.objective = incumbent;
        result.best_bound = root_bound;
        result.gap = 0.0;
        result.nodes = 1;
        result.lp_iterations = total_lp_iters;
        result.work_units = canonicalReportedWork(total_work);
        result.time_seconds = elapsed();
        result.solution = std::move(best_solution);
        applyPostsolve(result);
        restoreProblem();
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
    if (verbose_) {
        log_.log(
            "Root heuristics: int_inf=%d/%d int_vars=%d/%d calls=%d wins=%d "
            "round=%d/%d aux=%d/%d zero=%d/%d "
            "fp=%d/%d rens=%d/%d rins=%d/%d lb=%d/%d work=%.3f\n",
            root_int_inf,
            runtime_config.root_max_int_inf,
            root_int_vars,
            runtime_config.root_max_int_vars,
            root_heur_outcome.calls,
            root_heur_outcome.improvements,
            root_heur_outcome.rounding_calls,
            root_heur_outcome.rounding_improvements,
            root_heur_outcome.auxobj_calls,
            root_heur_outcome.auxobj_improvements,
            root_heur_outcome.zeroobj_calls,
            root_heur_outcome.zeroobj_improvements,
            root_heur_outcome.feaspump_calls,
            root_heur_outcome.feaspump_improvements,
            root_heur_outcome.rens_calls,
            root_heur_outcome.rens_improvements,
            root_heur_outcome.rins_calls,
            root_heur_outcome.rins_improvements,
            root_heur_outcome.localbranching_calls,
            root_heur_outcome.localbranching_improvements,
            root_heur_outcome.work_units);
    }

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

    NodePolicy initial_policy = NodePolicy::BestFirst;
    if (search_profile_ == SearchProfile::Aggressive) {
        initial_policy = NodePolicy::BestEstimate;
    }
    NodeQueue queue(initial_policy);

    // Create root children.
    BnbNode root_node;
    root_node.id = 0;
    root_node.depth = 0;
    root_node.lp_bound = root_bound;
    root_node.estimate = root_bound;
    root_node.basis = root_basis;
    root_node.basis_rows = lp.numRows();

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
        auto [left, right] = createChildren(std::move(root_node), branch_var,
                                            root_primals[branch_var]);
        left.lp_bound = root_bound;
        right.lp_bound = root_bound;
        left.estimate = root_bound + 0.05 * static_cast<Real>(std::max<Int>(0, root_int_inf));
        right.estimate = left.estimate;
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
                      total_work, root_runtime, solution_pool,
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

    root_runtime.finish();

    // Build result.
    MipResult result;
    result.lp_iterations = total_lp_iters;
    result.work_units = canonicalReportedWork(total_work);
    result.nodes = nodes_explored;
    result.time_seconds = elapsed();

    if (incumbent < kInf) {
        result.objective = incumbent;
        result.solution = std::move(best_solution);
        Real final_bound = queue.empty() ? incumbent : queue.bestBound();
        result.best_bound = final_bound;
        result.gap = computeGap(incumbent, final_bound);
        result.gap_limit_reached = !queue.empty() && result.gap < gap_tol_;

        if (queue.empty() || result.gap_limit_reached) {
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
    restoreProblem();

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
        if (cut_stats_.root_rounds > 0 || cut_stats_.tree_rounds > 0) {
            log_.log("Cuts: root_rounds=%d root_added=%d tree_nodes=%d "
                     "tree_skipped=%d tree_rounds=%d tree_local=%d tree_global=%d "
                     "tree_purged=%d tree_revived=%d tree_lp_delta=%.3e\n",
                     cut_stats_.root_rounds,
                     cut_stats_.root_cuts_added,
                     cut_stats_.tree_nodes_with_cuts,
                     cut_stats_.tree_nodes_skipped,
                     cut_stats_.tree_rounds,
                     cut_stats_.tree_cuts_local,
                     cut_stats_.tree_cuts_global,
                     cut_stats_.tree_cuts_purged,
                     cut_stats_.tree_cuts_revived,
                     cut_stats_.tree_lp_delta);
        }
        if (conflict_stats_.learned > 0 || conflict_stats_.reused > 0) {
            log_.log("Conflicts: learned=%d reused=%d pruned=%d purged=%d "
                     "min_literals=%d src_lp=%d src_bound=%d branch_overrides=%d\n",
                     conflict_stats_.learned,
                     conflict_stats_.reused,
                     conflict_stats_.pruned,
                     conflict_stats_.purged,
                     conflict_stats_.minimized_literals,
                     conflict_stats_.lp_infeasible_conflicts,
                     conflict_stats_.bound_infeasible_conflicts,
                     conflict_stats_.branch_score_overrides);
        }
        if (search_stats_.policy_switches > 0 || search_stats_.restarts > 0 ||
            search_stats_.sibling_cache_hits > 0) {
            log_.log("Search: profile=%s switches=%d restarts=%d dropped=%d "
                     "sibling_hits=%d sibling_misses=%d strong_budget_updates=%d\n",
                     searchProfileName(search_profile_),
                     search_stats_.policy_switches,
                     search_stats_.restarts,
                     search_stats_.restart_nodes_dropped,
                     search_stats_.sibling_cache_hits,
                     search_stats_.sibling_cache_misses,
                     search_stats_.strong_budget_updates);
        }
        if (tree_presolve_stats_.attempts > 0) {
            log_.log("TreePresolve: attempts=%d runs=%d skipped=%d infeasible=%d "
                     "tight_activity=%d tight_rc=%d resolves=%d lp_delta=%.3e\n",
                     tree_presolve_stats_.attempts,
                     tree_presolve_stats_.runs,
                     tree_presolve_stats_.skipped,
                     tree_presolve_stats_.infeasible,
                     tree_presolve_stats_.activity_tightenings,
                     tree_presolve_stats_.reduced_cost_tightenings,
                     tree_presolve_stats_.lp_resolves,
                     tree_presolve_stats_.lp_delta);
        }
        char node_buf[16], iter_buf[16];
        Logger::formatCount(result.nodes, node_buf, sizeof(node_buf));
        Logger::formatCount(result.lp_iterations, iter_buf, sizeof(iter_buf));
        if (parallel_mode_ == ParallelMode::Deterministic) {
            log_.log("\nExplored %s nodes, %s LP iterations, %.1f work\n",
                     node_buf, iter_buf, result.work_units);
        } else {
            log_.log("\nExplored %s nodes, %s LP iterations, %.1fs\n",
                     node_buf, iter_buf, result.time_seconds);
        }
        if (result.status == Status::Optimal) {
            if (result.gap_limit_reached) {
                log_.log("Gap-limited: %.10e (gap %.2f%%)\n",
                         result.objective, result.gap * 100.0);
            } else {
                log_.log("Optimal: %.10e\n", result.objective);
            }
        } else if (incumbent < kInf) {
            log_.log("Best solution: %.10e (gap %.2f%%)\n",
                     result.objective, result.gap * 100.0);
        }
    }

    return result;
}

Int MipSolver::runCuttingPlanes(DualSimplexSolver& lp, Int& total_lp_iters, double& total_work,
                                LpProblem* certificate_problem) {
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

        if (certificate_problem != nullptr) {
            for (Index r = 0; r < cuts_this_round; ++r) {
                const Index start = starts[r];
                const Index end =
                    (r + 1 < cuts_this_round) ? starts[r + 1] :
                                                static_cast<Index>(col_indices.size());
                const auto idx_span = std::span<const Index>(
                    col_indices.data() + start,
                    static_cast<std::size_t>(end - start));
                const auto val_span = std::span<const Real>(
                    values.data() + start,
                    static_cast<std::size_t>(end - start));
                certificate_problem->matrix.addRow(idx_span, val_span);
                certificate_problem->row_lower.push_back(lower[r]);
                certificate_problem->row_upper.push_back(upper[r]);
                certificate_problem->row_names.push_back("");
                const double mirror_work =
                    static_cast<double>((end - start) + 1) * 1e-6;
                total_work += mirror_work;
                cut_node_work += mirror_work;
            }
            certificate_problem->num_rows = certificate_problem->matrix.numRows();
        }

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

    cut_stats_.root_rounds += rounds_done;
    cut_stats_.root_cuts_added += total_cuts;

    return total_cuts;
}

}  // namespace mipx
