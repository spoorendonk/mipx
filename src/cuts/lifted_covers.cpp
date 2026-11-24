#include "mipx/lp_problem.h"
#include "mipx/separators.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace mipx {

namespace {

/// Sequential up-lifting of a cover inequality sum_{j in C} x_j <= |C| - 1
/// for a knapsack row sum_j a_j x_j <= b over 0/1 columns (Padberg 1975;
/// Wolsey, Integer Programming, 1998, section 9.4). Variables are lifted one
/// at a time, and each alpha_j is |C| - 1 minus the largest value of the
/// inequality built so far over the points that still fit with x_j = 1:
///   alpha_j = |C| - 1 - max{ sum_i v_i x_i : sum_i a_i x_i <= b - a_j },
/// the sum over C (value 1 each) and the variables lifted before j (value
/// alpha_i). Giving every variable the coefficient it would get if lifted
/// first, as this file did before #259, is not valid: it cut off feasible
/// points.
///
/// Values are integers in [0, |C| - 1] on every feasible point, so the max is
/// a 0/1 knapsack over values: min_weight_[v] is the least weight of a subset
/// worth at least v, updated as each lifted variable joins.
class SequentialLifter {
public:
    explicit SequentialLifter(Int base_rhs)
        : base_rhs_(base_rhs), min_weight_(static_cast<std::size_t>(base_rhs) + 1, kInf) {
        min_weight_[0] = 0.0;
    }

    /// Add an item of integer value @p value and weight @p weight.
    void add(Int value, Real weight) {
        if (value <= 0) {
            return;
        }
        // Descending, so each item is used at most once.
        for (Int v = base_rhs_; v >= 0; --v) {
            if (at(v) >= kInf) {
                continue;
            }
            const Int target = std::min(base_rhs_, v + value);
            at(target) = std::min(at(target), at(v) + weight);
        }
        // Worth at least v is also worth at least v - 1.
        for (Int v = base_rhs_; v > 0; --v) {
            at(v - 1) = std::min(at(v - 1), at(v));
        }
    }

    /// The largest value of a subset of the items added so far whose weight
    /// fits in @p capacity; -1 when not even the empty set fits.
    [[nodiscard]] Int maxValue(Real capacity) const {
        // The tolerance admits more subsets, so it can only lower a
        // coefficient, never make the cut invalid.
        for (Int v = base_rhs_; v >= 0; --v) {
            if (at(v) <= capacity + 1e-9) {
                return v;
            }
        }
        return -1;
    }

private:
    [[nodiscard]] Real& at(Int v) { return min_weight_[static_cast<std::size_t>(v)]; }
    [[nodiscard]] Real at(Int v) const { return min_weight_[static_cast<std::size_t>(v)]; }

    Int base_rhs_;
    std::vector<Real> min_weight_;
};

}  // namespace

Int SeparatorManager::separateLiftedCover(const LpProblem& problem, std::span<const Real> primals,
                                          CutPool& pool, CutFamilyStats& stats) {
    struct Item {
        Index var;
        Real coeff;
        Real primal;
    };

    Int accepted = 0;

    for (Index i = 0; i < problem.num_rows && accepted < max_cuts_per_family_ && !pastDeadline();
         ++i) {
        if (problem.row_upper[i] >= kInf)
            continue;
        const Real rhs = problem.row_upper[i];
        if (!std::isfinite(rhs) || rhs <= 0.0)
            continue;

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
            if (isBinaryColumn(problem, j) && a > 1e-9) {
                const Real x = (j < static_cast<Index>(primals.size())) ? primals[j] : 0.0;
                items.push_back({j, a, x});
            }
        }
        if (!valid || items.size() < 2)
            continue;
        ++stats.attempted;

        // Sort by decreasing LP value * coefficient (greedy cover selection).
        std::sort(items.begin(), items.end(), [](const Item& a, const Item& b) {
            const Real wa = a.primal * a.coeff;
            const Real wb = b.primal * b.coeff;
            if (std::abs(wa - wb) > 1e-12)
                return wa > wb;
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
            if (cover_sum > rhs + 1e-9)
                break;
        }
        if (cover_sum <= rhs + 1e-9)
            continue;

        // Minimize cover: remove items that are not needed.
        for (auto it = cover.begin(); it != cover.end();) {
            if (cover.size() <= 2)
                break;
            if (cover_sum - it->coeff > rhs + 1e-9) {
                cover_sum -= it->coeff;
                it = cover.erase(it);
            } else {
                ++it;
            }
        }
        if (cover_sum <= rhs + 1e-9)
            continue;

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

        // Base cover inequality: sum_{j in C} x_j <= |C| - 1, lifted in the
        // order of non_cover_items (largest LP value times coefficient first,
        // so the variables that can make the cut violated are lifted while
        // their coefficients can still be large).
        const Int cover_size = static_cast<Int>(cover.size());
        const Real base_rhs = static_cast<Real>(cover_size) - 1.0;

        SequentialLifter lifter(cover_size - 1);
        std::vector<std::pair<Index, Real>> cut_entries;
        cut_entries.reserve(items.size());
        for (const auto& ci : cover) {
            lifter.add(1, ci.coeff);
            cut_entries.emplace_back(ci.var, 1.0);
        }
        for (const auto& nc : non_cover_items) {
            const Int best = lifter.maxValue(rhs - nc.coeff);
            // best < 0: x_j = 1 alone overfills the row, so any coefficient
            // would be valid; leave it at 0 rather than lean on that.
            if (best < 0) {
                continue;
            }
            const Int alpha = (cover_size - 1) - best;
            if (alpha > 0) {
                lifter.add(alpha, nc.coeff);
                cut_entries.emplace_back(nc.var, static_cast<Real>(alpha));
            }
        }

        if (cut_entries.size() <= 1)
            continue;

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

        if (addViolatedCut(std::move(cut), primals, pool, stats)) {
            ++accepted;
        }
    }

    return accepted;
}

}  // namespace mipx
