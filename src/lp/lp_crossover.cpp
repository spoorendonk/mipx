#include "lp_crossover.h"

#include "mipx/dual_simplex.h"
#include "rank_revealing_lu.h"

#include <algorithm>
#include <cmath>

namespace mipx {

namespace {

struct CrossoverBound {
    BasisStatus status;
    Real distance;
};

CrossoverBound nearestCrossoverBound(Real value, Real lower, Real upper) {
    if (lower == upper) {
        return {BasisStatus::Fixed, 0.0};
    }
    if (!std::isfinite(lower) && !std::isfinite(upper)) {
        return {BasisStatus::Free, kInf};
    }
    const Real from_lower = value - lower;
    const Real from_upper = upper - value;
    if (from_lower <= from_upper) {
        return {BasisStatus::AtLower, from_lower / std::max(1.0, std::abs(lower))};
    }
    return {BasisStatus::AtUpper, from_upper / std::max(1.0, std::abs(upper))};
}

}  // namespace

std::optional<std::vector<BasisStatus>> crossoverBasis(const LpProblem& lp,
                                                       std::span<const Real> primal,
                                                       std::span<const Real> reduced_costs,
                                                       const Deadline& deadline) {
    constexpr Real kBoundTol = 1e-6;
    constexpr Real kDualBoundTol = 1e-3;
    constexpr Real kReducedCostTol = 1e-6;
    const Index n = lp.num_cols;
    const Index m = lp.num_rows;
    std::vector<BasisStatus> basis(static_cast<size_t>(n + m));
    std::vector<Real> activity(static_cast<size_t>(m));
    lp.matrix.multiply(primal, activity);

    // Candidate columns in CSC, with a map back to [A | -I]. Use +1 for
    // logicals here: the pivot rows and column rank are invariant to sign.
    std::vector<Index> starts{0}, rows, vars;
    std::vector<Real> values;
    const auto finishColumn = [&](Index var) {
        vars.push_back(var);
        starts.push_back(static_cast<Index>(rows.size()));
    };
    for (Index i = 0; i < m; ++i) {
        if (i % 256 == 0 && deadline.passed()) {
            return std::nullopt;
        }
        const auto bound = nearestCrossoverBound(activity[i], lp.row_lower[i], lp.row_upper[i]);
        basis[n + i] = bound.status;
        // Slack on an inactive row should stay basic. Unit pivots reserve
        // those rows before elimination of the structural columns.
        if (bound.distance > kBoundTol) {
            rows.push_back(i);
            values.push_back(1.0);
            finishColumn(n + i);
        }
    }
    for (Index j = 0; j < n; ++j) {
        if (deadline.passed()) {
            return std::nullopt;
        }
        const auto bound = nearestCrossoverBound(primal[j], lp.col_lower[j], lp.col_upper[j]);
        basis[j] = bound.status;
        // A positive IPM gap can leave bound-active columns slightly in the
        // interior. For a one-sided column, the reduced cost is its bound
        // multiplier (up to sign). Measure it against the column's cost
        // so a small relative dual residual does not drive this screen.
        const bool one_sided = std::isfinite(lp.col_lower[j]) != std::isfinite(lp.col_upper[j]);
        const bool dual_at_bound =
            one_sided && bound.distance <= kDualBoundTol &&
            std::abs(reduced_costs[j]) > kReducedCostTol * std::max(1.0, std::abs(lp.obj[j]));
        if (bound.distance <= kBoundTol || dual_at_bound) {
            continue;
        }
        const auto col = lp.matrix.col(j);
        // Raw CSR can contain repeated entries or zeros. The CSC view
        // groups each row's entries, so sum them as the simplex does.
        for (Index k = 0; k < col.size();) {
            const Index row = col.indices[k];
            Real value = 0.0;
            do {
                value += col.values[k++];
            } while (k < col.size() && col.indices[k] == row);
            if (value != 0.0) {
                rows.push_back(row);
                values.push_back(value);
            }
        }
        finishColumn(j);
    }

    if (deadline.passed()) {
        return std::nullopt;
    }
    RankRevealingLU rank;
    rank.setup(m, starts, rows, values);
    if (rank.build(deadline) == RankRevealingLU::kBuildKernelReturnTimeout) {
        return std::nullopt;
    }
    const auto rejected = rank.colWithNoPivot();
    auto next_rejected = rejected.begin();
    for (Index j = 0; j < static_cast<Index>(vars.size()); ++j) {
        if (next_rejected != rejected.end() && *next_rejected == j) {
            ++next_rejected;
        } else {
            basis[vars[j]] = BasisStatus::Basic;
        }
    }
    // The pivoted columns are independent on the pivot rows. The logical
    // of each unmatched real row completes that submatrix to a square basis.
    for (Index i : rank.rowWithNoPivot()) {
        if (i < m) {
            basis[n + i] = BasisStatus::Basic;
        }
    }
    return basis;
}

LpResult solveCrossoverBasis(DualSimplexSolver& simplex, std::span<const BasisStatus> basis) {
    simplex.setBasis(basis);
    return simplex.solve();
}

}  // namespace mipx
