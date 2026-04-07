#include "mipx/separators.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include "mipx/lp_problem.h"

namespace mipx {

namespace {

constexpr Real kCoeffTol = 1e-10;

/// Sequence-independent lifting (Gu-Nemhauser-Savelsbergh).
/// Given a minimal cover C with sum_{j in C} a_j > b (knapsack capacity),
/// for each variable j NOT in C, compute the lifting coefficient alpha_j
/// such that: sum_{j in C} x_j + sum_{j not in C} alpha_j x_j <= |C| - 1
///
/// The lifting coefficient for variable j is:
///   alpha_j = |C| - 1 - max{ sum_{i in C} x_i : sum_{i in C} a_i x_i <= b - a_j, x binary }
/// which equals |C| - 1 minus the optimal value of a small binary knapsack.
///
/// For efficiency, we solve the knapsack greedily (good enough for most instances).
Int solveSmallKnapsack(const std::vector<Real>& weights,
                       Real capacity) {
    if (capacity < 0.0) return 0;
    // Greedy: take items in order of increasing weight.
    std::vector<std::size_t> order(weights.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::sort(order.begin(), order.end(),
              [&](std::size_t a, std::size_t b) { return weights[a] < weights[b]; });

    Int count = 0;
    Real remaining = capacity;
    for (std::size_t idx : order) {
        if (weights[idx] <= remaining + 1e-9) {
            remaining -= weights[idx];
            ++count;
        }
    }
    return count;
}

}  // namespace

Int SeparatorManager::separateLiftedCover(const LpProblem& problem,
                                          std::span<const Real> primals,
                                          CutPool& pool,
                                          CutFamilyStats& stats) {
    struct Item {
        Index var;
        Real coeff;
        Real primal;
    };

    Int accepted = 0;

    for (Index i = 0; i < problem.num_rows && accepted < max_cuts_per_family_; ++i) {
        if (problem.row_upper[i] >= kInf) continue;
        const Real rhs = problem.row_upper[i];
        if (!std::isfinite(rhs) || rhs <= 0.0) continue;

        auto row = problem.matrix.row(i);

        // Collect binary variables with positive coefficients (knapsack form).
        std::vector<Item> items;
        std::vector<Item> non_cover_items;
        items.reserve(static_cast<std::size_t>(row.size()));
        bool valid = true;

        for (Index p = 0; p < row.size(); ++p) {
            const Index j = row.indices[p];
            const Real a = row.values[p];
            if (a < 0.0 || problem.col_lower[j] < -1e-12) {
                valid = false;
                break;
            }
            if (problem.col_type[j] == VarType::Binary && a > 1e-9) {
                const Real x = (j < static_cast<Index>(primals.size()))
                    ? primals[j] : 0.0;
                items.push_back({j, a, x});
            }
        }
        if (!valid || items.size() < 2) continue;
        ++stats.attempted;

        // Sort by decreasing LP value * coefficient (greedy cover selection).
        std::sort(items.begin(), items.end(), [](const Item& a, const Item& b) {
            const Real wa = a.primal * a.coeff;
            const Real wb = b.primal * b.coeff;
            if (std::abs(wa - wb) > 1e-12) return wa > wb;
            return a.coeff > b.coeff;
        });

        // Find minimal cover: greedily add items until sum > rhs.
        std::vector<Item> cover;
        Real cover_sum = 0.0;
        std::vector<bool> in_cover(items.size(), false);

        for (std::size_t k = 0; k < items.size(); ++k) {
            cover.push_back(items[k]);
            in_cover[k] = true;
            cover_sum += items[k].coeff;
            if (cover_sum > rhs + 1e-9) break;
        }
        if (cover_sum <= rhs + 1e-9) continue;

        // Minimize cover: remove items that are not needed.
        for (auto it = cover.begin(); it != cover.end(); ) {
            if (cover.size() <= 2) break;
            if (cover_sum - it->coeff > rhs + 1e-9) {
                cover_sum -= it->coeff;
                it = cover.erase(it);
            } else {
                ++it;
            }
        }
        if (cover_sum <= rhs + 1e-9) continue;

        // Collect non-cover items for lifting.
        non_cover_items.clear();
        for (std::size_t k = 0; k < items.size(); ++k) {
            bool found = false;
            for (const auto& ci : cover) {
                if (ci.var == items[k].var) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                non_cover_items.push_back(items[k]);
            }
        }

        // Sequence-independent lifting.
        // Base cover inequality: sum_{j in C} x_j <= |C| - 1.
        const Int cover_size = static_cast<Int>(cover.size());
        const Real base_rhs = static_cast<Real>(cover_size) - 1.0;

        // Compute lifting coefficients for non-cover variables.
        std::vector<Real> cover_weights;
        cover_weights.reserve(cover.size());
        for (const auto& ci : cover) {
            cover_weights.push_back(ci.coeff);
        }

        // Build the cut.
        std::vector<std::pair<Index, Real>> cut_entries;
        cut_entries.reserve(items.size());

        // Cover variables get coefficient 1.
        for (const auto& ci : cover) {
            cut_entries.emplace_back(ci.var, 1.0);
        }

        // Lift non-cover variables.
        for (const auto& nc : non_cover_items) {
            const Real residual_cap = rhs - nc.coeff;
            const Int knapsack_opt = solveSmallKnapsack(cover_weights, residual_cap);
            const Real alpha = base_rhs - static_cast<Real>(knapsack_opt);
            if (alpha > kCoeffTol) {
                cut_entries.emplace_back(nc.var, alpha);
            }
        }

        if (cut_entries.size() <= 1) continue;

        // Sort by variable index.
        std::sort(cut_entries.begin(), cut_entries.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });

        Cut cut;
        cut.family = CutFamily::LiftedCover;
        cut.lower = -kInf;
        cut.upper = base_rhs;
        for (const auto& [idx, val] : cut_entries) {
            cut.indices.push_back(idx);
            cut.values.push_back(val);
        }

        // Compute violation and efficacy.
        Real lhs = 0.0;
        for (Index k = 0; k < static_cast<Index>(cut.indices.size()); ++k) {
            const Index j = cut.indices[k];
            if (j >= 0 && j < static_cast<Index>(primals.size())) {
                lhs += cut.values[k] * primals[j];
            }
        }
        const Real violation = lhs - cut.upper;
        if (violation < min_violation_) continue;

        Real norm_sq = 0.0;
        for (Real v : cut.values) norm_sq += v * v;
        if (norm_sq < kCoeffTol) continue;
        cut.efficacy = violation / std::sqrt(norm_sq);
        if (!std::isfinite(cut.efficacy) || cut.efficacy <= 0.0) continue;

        ++stats.generated;
        if (pool.addCut(std::move(cut))) {
            ++stats.accepted;
            stats.efficacy_sum += violation / std::sqrt(norm_sq);
            ++accepted;
        }
    }

    return accepted;
}

}  // namespace mipx
