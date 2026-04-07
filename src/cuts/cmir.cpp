#include "mipx/separators.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "mipx/lp_problem.h"

namespace mipx {

namespace {

constexpr Real kIntTol = 1e-6;
constexpr Real kCoeffTol = 1e-10;

/// Apply MIR rounding to a single-row relaxation with complementation choices.
/// Given: sum_j a_j x_j <= b  (all integer vars have 0/1 complementation flag)
/// Complementing variable j means substituting x_j' = u_j - x_j.
/// The MIR inequality is:
///   sum_{j: f_j <= f0} f_j x_j + sum_{j: f_j > f0} (1-f_j)/(1-f0) * a_j_int x_j
///   + sum_{continuous} max(a_j, 0)/f0 x_j + ... <= floor(b)
/// where f0 = b - floor(b), f_j = a_j - floor(a_j).
struct CmirResult {
    std::vector<Index> indices;
    std::vector<Real> values;
    Real rhs = 0.0;
    Real violation = 0.0;
    bool valid = false;
};

CmirResult applyCmir(const std::vector<Index>& row_indices,
                     const std::vector<Real>& row_values,
                     Real row_rhs,
                     const LpProblem& problem,
                     std::span<const Real> primals,
                     const std::vector<bool>& complement) {
    CmirResult result;

    // Build the modified row after complementation.
    // For complemented integer var j: x_j' = u_j - x_j, so a_j x_j = a_j(u_j - x_j') = -a_j x_j' + a_j u_j
    // New coefficient: -a_j, rhs adjusted by -a_j * u_j
    Real mod_rhs = row_rhs;
    std::vector<Index> mod_indices;
    std::vector<Real> mod_values;
    mod_indices.reserve(row_indices.size());
    mod_values.reserve(row_indices.size());

    for (std::size_t k = 0; k < row_indices.size(); ++k) {
        const Index j = row_indices[k];
        Real a = row_values[k];

        if (complement[k] && problem.col_type[j] != VarType::Continuous) {
            const Real ub = problem.col_upper[j];
            if (!std::isfinite(ub)) {
                result.valid = false;
                return result;
            }
            mod_rhs -= a * ub;
            a = -a;
        }

        mod_indices.push_back(j);
        mod_values.push_back(a);
    }

    // Apply MIR rounding.
    const Real f0 = mod_rhs - std::floor(mod_rhs);
    if (f0 < kIntTol || f0 > 1.0 - kIntTol) {
        result.valid = false;
        return result;
    }

    Real cut_rhs = std::floor(mod_rhs);
    std::vector<Real> cut_values(mod_indices.size(), 0.0);

    for (std::size_t k = 0; k < mod_indices.size(); ++k) {
        const Index j = mod_indices[k];
        const Real a = mod_values[k];

        if (problem.col_type[j] == VarType::Continuous) {
            // Continuous: coefficient is max(a, 0) / f0 for >= side,
            // but for <= cut: if a > 0, coeff = a/f0; if a < 0, coeff = a/(f0-1)
            if (a >= 0.0) {
                cut_values[k] = a / f0;
            } else {
                cut_values[k] = a / (f0 - 1.0);
            }
        } else {
            // Integer: apply MIR formula
            Real fj = a - std::floor(a);
            if (fj < 0.0) fj += 1.0;
            if (fj > 1.0 - kCoeffTol) fj = 0.0;

            if (fj <= f0 + kCoeffTol) {
                cut_values[k] = std::floor(a);
            } else {
                cut_values[k] = std::floor(a) + (fj - f0) / (1.0 - f0);
            }
        }
    }

    // Undo complementation in the cut coefficients.
    for (std::size_t k = 0; k < mod_indices.size(); ++k) {
        if (complement[k] && problem.col_type[mod_indices[k]] != VarType::Continuous) {
            const Real ub = problem.col_upper[mod_indices[k]];
            cut_rhs -= cut_values[k] * ub;
            cut_values[k] = -cut_values[k];
        }
    }

    // Build sparse cut (drop near-zeros).
    for (std::size_t k = 0; k < mod_indices.size(); ++k) {
        if (std::abs(cut_values[k]) > kCoeffTol) {
            result.indices.push_back(mod_indices[k]);
            result.values.push_back(cut_values[k]);
        }
    }

    if (result.indices.empty()) {
        result.valid = false;
        return result;
    }

    result.rhs = cut_rhs;

    // Compute violation: sum a_j x_j should be <= rhs; violation = lhs - rhs.
    Real lhs = 0.0;
    for (std::size_t k = 0; k < result.indices.size(); ++k) {
        const Index j = result.indices[k];
        if (j >= 0 && j < static_cast<Index>(primals.size())) {
            lhs += result.values[k] * primals[j];
        }
    }
    result.violation = lhs - result.rhs;
    result.valid = true;
    return result;
}

}  // namespace

Int SeparatorManager::separateCmir(DualSimplexSolver& lp,
                                   const LpProblem& problem,
                                   std::span<const Real> primals,
                                   CutPool& pool,
                                   CutFamilyStats& stats) {
    (void)lp;  // LP solver not needed for row-based CMIR.
    Int accepted = 0;

    for (Index i = 0; i < problem.num_rows && accepted < max_cuts_per_family_; ++i) {
        if (problem.row_upper[i] >= kInf) continue;
        const Real rhs = problem.row_upper[i];
        if (!std::isfinite(rhs)) continue;

        auto row = problem.matrix.row(i);
        if (row.size() < 2) continue;

        // Check row has at least one integer variable.
        bool has_integer = false;
        std::vector<Index> row_indices;
        std::vector<Real> row_values;
        row_indices.reserve(static_cast<std::size_t>(row.size()));
        row_values.reserve(static_cast<std::size_t>(row.size()));

        for (Index p = 0; p < row.size(); ++p) {
            const Index j = row.indices[p];
            const Real a = row.values[p];
            if (std::abs(a) < kCoeffTol) continue;
            if (problem.col_type[j] != VarType::Continuous) has_integer = true;
            row_indices.push_back(j);
            row_values.push_back(a);
        }
        if (!has_integer) continue;
        ++stats.attempted;

        // Try different complementation patterns.
        // Optimal complementation: complement variable j if x_j > (l_j + u_j)/2.
        // Also try no complementation as a baseline.
        Real best_violation = min_violation_;
        CmirResult best_result;

        auto tryComplementation = [&](const std::vector<bool>& comp) {
            auto res = applyCmir(row_indices, row_values, rhs, problem, primals, comp);
            if (res.valid && res.violation > best_violation) {
                best_violation = res.violation;
                best_result = std::move(res);
            }
        };

        // No complementation.
        std::vector<bool> no_comp(row_indices.size(), false);
        tryComplementation(no_comp);

        // Optimal single-variable complementation: complement vars closer to upper bound.
        std::vector<bool> opt_comp(row_indices.size(), false);
        for (std::size_t k = 0; k < row_indices.size(); ++k) {
            const Index j = row_indices[k];
            if (problem.col_type[j] == VarType::Continuous) continue;
            const Real lb = problem.col_lower[j];
            const Real ub = problem.col_upper[j];
            if (!std::isfinite(ub)) continue;
            const Real mid = 0.5 * (lb + ub);
            if (j < static_cast<Index>(primals.size()) && primals[j] > mid) {
                opt_comp[k] = true;
            }
        }
        tryComplementation(opt_comp);

        // Greedy complementation: for each integer variable, try flipping.
        std::vector<bool> greedy_comp = opt_comp;
        for (std::size_t k = 0; k < row_indices.size() && accepted < max_cuts_per_family_; ++k) {
            const Index j = row_indices[k];
            if (problem.col_type[j] == VarType::Continuous) continue;
            if (!std::isfinite(problem.col_upper[j])) continue;
            greedy_comp[k] = !greedy_comp[k];
            auto res = applyCmir(row_indices, row_values, rhs, problem, primals, greedy_comp);
            if (res.valid && res.violation > best_violation) {
                best_violation = res.violation;
                best_result = std::move(res);
            } else {
                greedy_comp[k] = !greedy_comp[k];  // Revert.
            }
        }

        if (!best_result.valid) continue;

        // Sort indices for the cut.
        std::vector<std::pair<Index, Real>> sorted_cut;
        sorted_cut.reserve(best_result.indices.size());
        for (std::size_t k = 0; k < best_result.indices.size(); ++k) {
            sorted_cut.emplace_back(best_result.indices[k], best_result.values[k]);
        }
        std::sort(sorted_cut.begin(), sorted_cut.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });

        Cut cut;
        cut.family = CutFamily::Cmir;
        cut.lower = -kInf;
        cut.upper = best_result.rhs;
        for (const auto& [idx, val] : sorted_cut) {
            cut.indices.push_back(idx);
            cut.values.push_back(val);
        }

        ++stats.generated;
        Real norm_sq = 0.0;
        for (Real v : cut.values) norm_sq += v * v;
        if (norm_sq < kCoeffTol) continue;
        cut.efficacy = best_violation / std::sqrt(norm_sq);
        if (!std::isfinite(cut.efficacy) || cut.efficacy <= 0.0) continue;

        if (pool.addCut(std::move(cut))) {
            ++stats.accepted;
            stats.efficacy_sum += best_violation / std::sqrt(norm_sq);
            ++accepted;
        }
    }

    return accepted;
}

Int SeparatorManager::separateStrongCg(DualSimplexSolver& lp,
                                       const LpProblem& problem,
                                       std::span<const Real> primals,
                                       CutPool& pool,
                                       CutFamilyStats& stats) {
    (void)lp;  // LP solver not needed for row-based Strong CG.
    // Strong Chvatal-Gomory cuts.
    // For each row: sum a_j x_j <= b
    // CG cut: sum floor(a_j) x_j <= floor(b) for integer variables.
    // Strong CG: multiply row by a scalar t before rounding.
    // We try several multipliers t to find the best violation.
    Int accepted = 0;

    const std::vector<Real> multipliers = {0.5, 1.0, 2.0, 3.0, 0.25, 0.75, 1.5};

    for (Index i = 0; i < problem.num_rows && accepted < max_cuts_per_family_; ++i) {
        if (problem.row_upper[i] >= kInf) continue;
        const Real rhs = problem.row_upper[i];
        if (!std::isfinite(rhs)) continue;

        auto row = problem.matrix.row(i);
        if (row.size() < 1) continue;

        bool has_integer = false;
        bool valid = true;
        for (Index p = 0; p < row.size(); ++p) {
            const Index j = row.indices[p];
            if (problem.col_type[j] != VarType::Continuous) has_integer = true;
            if (problem.col_lower[j] < -1e-12) {
                valid = false;
                break;
            }
        }
        if (!has_integer || !valid) continue;
        ++stats.attempted;

        Real best_violation = min_violation_;
        Cut best_cut;
        bool found = false;

        for (Real t : multipliers) {
            const Real scaled_rhs = t * rhs;
            const Real floored_rhs = std::floor(scaled_rhs + 1e-9);
            if (floored_rhs + 1e-9 >= scaled_rhs) continue;  // No rounding.

            Cut cut;
            cut.family = CutFamily::StrongCg;
            cut.lower = -kInf;
            cut.upper = floored_rhs;
            bool changed = (floored_rhs + 1e-9 < scaled_rhs);

            for (Index p = 0; p < row.size(); ++p) {
                const Index j = row.indices[p];
                const Real a = row.values[p];
                const Real scaled_a = t * a;
                if (scaled_a < -kCoeffTol) {
                    // Negative coefficient on nonneg var: skip this multiplier.
                    changed = false;
                    break;
                }
                if (problem.col_type[j] == VarType::Continuous) {
                    // Keep continuous coefficients as-is (they bound the cut).
                    if (scaled_a > kCoeffTol) {
                        cut.indices.push_back(j);
                        cut.values.push_back(scaled_a);
                    }
                } else {
                    const Real rounded = std::floor(scaled_a + 1e-9);
                    if (rounded <= 0.0) continue;
                    if (rounded + 1e-9 < scaled_a) changed = true;
                    cut.indices.push_back(j);
                    cut.values.push_back(rounded);
                }
            }

            if (!changed || cut.indices.empty()) continue;

            // Compute violation.
            Real lhs = 0.0;
            for (Index k = 0; k < static_cast<Index>(cut.indices.size()); ++k) {
                const Index j = cut.indices[k];
                if (j >= 0 && j < static_cast<Index>(primals.size())) {
                    lhs += cut.values[k] * primals[j];
                }
            }
            const Real violation = lhs - cut.upper;
            if (violation > best_violation) {
                Real norm_sq = 0.0;
                for (Real v : cut.values) norm_sq += v * v;
                if (norm_sq < kCoeffTol) continue;
                cut.efficacy = violation / std::sqrt(norm_sq);
                if (std::isfinite(cut.efficacy) && cut.efficacy > 0.0) {
                    best_violation = violation;
                    best_cut = std::move(cut);
                    found = true;
                }
            }
        }

        if (!found) continue;
        ++stats.generated;
        if (pool.addCut(std::move(best_cut))) {
            ++stats.accepted;
            stats.efficacy_sum += best_violation;
            ++accepted;
        }
    }

    return accepted;
}

}  // namespace mipx
