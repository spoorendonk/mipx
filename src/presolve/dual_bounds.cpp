#include "dual_bounds.h"

#include "presolve_matrix.h"

#include <algorithm>
#include <cmath>

namespace mipx {

namespace {

// Presolve's activities take a value this large as infinite (kHugeBound in
// presolve.cpp, #300).
constexpr Real kHuge = 1e20;
// HiGHS's kHighsTiny: an implied bound so large that this fraction of it
// exceeds the tolerance is dropped.
constexpr Real kTiny = 1e-14;

Real lowerOrInf(Real v) {
    return v <= -kHuge ? -kInf : v;
}
Real upperOrInf(Real v) {
    return v >= kHuge ? kInf : v;
}

/// Add coef * bound exactly, its rounding error as a second term, as
/// HiGHS's HighsCDouble product does.
void addProduct(CompensatedSum& sum, Real coef, Real bound) {
    const Real p = coef * bound;
    sum.add(p);
    sum.add(std::fma(coef, bound, -p));
}

/// One bound's term in an activity sum: counted infinite or added.
void addTerm(CompensatedSum& sum, Index& num_inf, Real coef, Real bound) {
    if (std::isinf(bound)) {
        ++num_inf;
    } else {
        addProduct(sum, coef, bound);
    }
}

/// The sum less one term; -inf (or +inf for an upper sum) while another
/// term is unbounded (HighsLinearSumBounds::residual).
Real residualOf(CompensatedSum sum, Index num_inf, Real coef, Real bound, bool lower_sum) {
    if (std::isinf(bound)) {
        --num_inf;
    } else {
        addProduct(sum, -coef, bound);
    }
    if (num_inf != 0) {
        return lower_sum ? -kInf : kInf;
    }
    return sum.value();
}

/// The sum plus `offset`, rounded once (HighsLinearSumBounds::getSumLower).
Real withOffset(CompensatedSum sum, Real offset) {
    sum.add(offset);
    return sum.value();
}

}  // namespace

void DualBounds::compute(const LpProblem& lp, const PresolveMatrix& m,
                         const std::vector<bool>& col_removed,
                         const std::vector<bool>& row_removed) {
    const auto n = static_cast<size_t>(lp.num_cols);
    const auto nrows = static_cast<size_t>(lp.num_rows);
    const Real sense = lp.sense == Sense::Maximize ? -1.0 : 1.0;

    col_lower_.assign(n, -kInf);
    col_upper_.assign(n, kInf);
    cost_.assign(n, 0.0);
    col_size_.assign(n, 0);
    for (size_t j = 0; j < n; ++j) {
        col_lower_[j] = lowerOrInf(lp.col_lower[j]);
        col_upper_[j] = upperOrInf(lp.col_upper[j]);
        cost_[j] = sense * lp.obj[j];
        if (col_removed[j]) {
            continue;
        }
        for (const auto [row, a] : m.col(static_cast<Index>(j))) {
            if (!row_removed[row] && a != 0.0) {
                ++col_size_[j];
            }
        }
    }

    // HiGHS's rowDualLower/Upper: y_i <= 0 without a lower side, y_i >= 0
    // without an upper side, so a free row's dual is 0.
    row_lower_.assign(nrows, -kInf);
    row_upper_.assign(nrows, kInf);
    row_dual_lower_.assign(nrows, -kInf);
    row_dual_upper_.assign(nrows, kInf);
    for (size_t i = 0; i < nrows; ++i) {
        row_lower_[i] = lowerOrInf(lp.row_lower[i]);
        row_upper_[i] = upperOrInf(lp.row_upper[i]);
        if (row_lower_[i] == -kInf) {
            row_dual_upper_[i] = 0.0;
        }
        if (row_upper_[i] == kInf) {
            row_dual_lower_[i] = 0.0;
        }
    }
    impl_row_dual_lower_.assign(nrows, -kInf);
    impl_row_dual_upper_.assign(nrows, kInf);
    row_dual_lower_source_.assign(nrows, -1);
    row_dual_upper_source_.assign(nrows, -1);
    impl_col_lower_.assign(n, -kInf);
    impl_col_upper_.assign(n, kInf);
    col_lower_source_.assign(n, -1);
    col_upper_source_.assign(n, -1);

    // The ...Orig sums read the row sides' dual bounds only, so they hold
    // through the alternation below.
    computeDualSums(m, col_removed, row_removed);

    // Alternate the primal and the dual implied bounds until no row becomes
    // an implied equation. Each round starts from the last one's bounds and
    // only tightens them by more than the threshold, as HiGHS's updates do,
    // so the implied equations only grow and the loop ends (within
    // 2 * rows + 1 rounds, the last one only confirming; one or two on
    // Netlib). Each deduction rests on
    // bounds derived before it, never on itself.
    auto impliedEquations = [&] {
        Index count = 0;
        for (size_t i = 0; i < nrows; ++i) {
            if (!row_removed[i]) {
                const auto r = static_cast<Index>(i);
                count +=
                    (isImpliedEquationAtLower(r) ? 1 : 0) + (isImpliedEquationAtUpper(r) ? 1 : 0);
            }
        }
        return count;
    };
    Index equations = 0;
    for (;;) {
        computeImpliedColBounds(lp, m, col_removed, row_removed);
        computeImpliedRowDualBounds(m, col_removed, row_removed);
        const Index now = impliedEquations();
        if (now == equations) {
            break;
        }
        equations = now;
    }

    computeDualSums(m, col_removed, row_removed);
}

void DualBounds::computeImpliedColBounds(const LpProblem& lp, const PresolveMatrix& m,
                                         const std::vector<bool>& col_removed,
                                         const std::vector<bool>& row_removed) {
    const Real threshold = 1000.0 * kPrimalFeasTol;
    for (Index i = 0; i < lp.num_rows; ++i) {
        if (row_removed[i]) {
            continue;
        }
        // HiGHS's checkUpdateColImpliedBounds: an implied equation's side
        // stands for both.
        const Real my_lower = isImpliedEquationAtUpper(i) ? row_upper_[i] : row_lower_[i];
        const Real my_upper = isImpliedEquationAtLower(i) ? row_lower_[i] : row_upper_[i];

        // The row's activity bounds from the columns' own bounds (HiGHS's
        // impliedRowBounds ...Orig sums).
        CompensatedSum min_act;
        CompensatedSum max_act;
        Index min_inf = 0;
        Index max_inf = 0;
        for (const auto [j, a] : m.row(i)) {
            if (col_removed[j] || a == 0.0) {
                continue;
            }
            addTerm(min_act, min_inf, a, a > 0.0 ? col_lower_[j] : col_upper_[j]);
            addTerm(max_act, max_inf, a, a > 0.0 ? col_upper_[j] : col_lower_[j]);
        }
        if (!((my_lower != -kInf && max_inf <= 1) || (my_upper != kInf && min_inf <= 1))) {
            continue;
        }

        auto check = [&](Index j, Real a, Real side, Real residual, Real direction) {
            if (direction * residual <= -kInf) {
                return;
            }
            const Real implied = (side - residual) / a;
            if (std::abs(implied) * kTiny > kPrimalFeasTol) {
                return;
            }
            if (direction * a > 0.0) {
                if (implied < impl_col_upper_[j] - threshold) {
                    impl_col_upper_[j] = implied;
                    col_upper_source_[j] = i;
                }
            } else if (implied > impl_col_lower_[j] + threshold) {
                impl_col_lower_[j] = implied;
                col_lower_source_[j] = i;
            }
        };
        for (const auto [j, a] : m.row(i)) {
            if (col_removed[j] || a == 0.0) {
                continue;
            }
            if (my_upper != kInf) {
                check(
                    j, a, my_upper,
                    residualOf(min_act, min_inf, a, a > 0.0 ? col_lower_[j] : col_upper_[j], true),
                    1.0);
            }
            if (my_lower != -kInf) {
                check(
                    j, a, my_lower,
                    residualOf(max_act, max_inf, a, a > 0.0 ? col_upper_[j] : col_lower_[j], false),
                    -1.0);
            }
        }
    }
}

void DualBounds::computeImpliedRowDualBounds(const PresolveMatrix& m,
                                             const std::vector<bool>& col_removed,
                                             const std::vector<bool>& row_removed) {
    const Real threshold = 1000.0 * kDualFeasTol;
    const auto n = static_cast<Index>(cost_.size());
    for (Index j = 0; j < n; ++j) {
        if (col_removed[j]) {
            continue;
        }
        // HiGHS's checkUpdateRowDualImpliedBounds.
        const Real dual_lower = isStrictlyImpliedForDual(j, true) ? cost_[j] : -kInf;
        const Real dual_upper = isStrictlyImpliedForDual(j, false) ? cost_[j] : kInf;
        if (!((dual_lower != -kInf && numInfSumUpperOrig(j) <= 1) ||
              (dual_upper != kInf && numInfSumLowerOrig(j) <= 1))) {
            continue;
        }

        auto check = [&](Index i, Real a, Real side, Real residual, Real direction) {
            if (direction * residual <= -kInf) {
                return;
            }
            const Real implied = (side - residual) / a;
            if (std::abs(implied) * kTiny > kDualFeasTol) {
                return;
            }
            if (direction * a > 0.0) {
                if (implied < impl_row_dual_upper_[i] - threshold) {
                    impl_row_dual_upper_[i] = implied;
                    row_dual_upper_source_[i] = j;
                }
            } else if (implied > impl_row_dual_lower_[i] + threshold) {
                impl_row_dual_lower_[i] = implied;
                row_dual_lower_source_[i] = j;
            }
        };
        for (const auto [i, a] : m.col(j)) {
            if (row_removed[i] || a == 0.0) {
                continue;
            }
            if (dual_upper != kInf) {
                check(i, a, dual_upper, residualSumLowerOrig(j, i, a), 1.0);
            }
            if (dual_lower != -kInf) {
                check(i, a, dual_lower, residualSumUpperOrig(j, i, a), -1.0);
            }
        }
    }
}

void DualBounds::computeDualSums(const PresolveMatrix& m, const std::vector<bool>& col_removed,
                                 const std::vector<bool>& row_removed) {
    const auto n = static_cast<Index>(cost_.size());
    sums_.assign(static_cast<size_t>(n), Sum{});
    for (Index j = 0; j < n; ++j) {
        if (col_removed[j]) {
            continue;
        }
        Sum& s = sums_[j];
        for (const auto [i, a] : m.col(j)) {
            if (row_removed[i] || a == 0.0) {
                continue;
            }
            const Real lo = row_dual_lower_[i];
            const Real up = row_dual_upper_[i];
            addTerm(s.lower_orig, s.lower_orig_inf, a, a > 0.0 ? lo : up);
            addTerm(s.upper_orig, s.upper_orig_inf, a, a > 0.0 ? up : lo);
            const Real impl_lo = implVarLower(j, i);
            const Real impl_up = implVarUpper(j, i);
            addTerm(s.lower, s.lower_inf, a, a > 0.0 ? impl_lo : impl_up);
            addTerm(s.upper, s.upper_inf, a, a > 0.0 ? impl_up : impl_lo);
        }
    }
}

Real DualBounds::implVarLower(Index col, Index row) const {
    return row_dual_lower_source_[row] == col
               ? row_dual_lower_[row]
               : std::max(impl_row_dual_lower_[row], row_dual_lower_[row]);
}

Real DualBounds::implVarUpper(Index col, Index row) const {
    return row_dual_upper_source_[row] == col
               ? row_dual_upper_[row]
               : std::min(impl_row_dual_upper_[row], row_dual_upper_[row]);
}

std::vector<Index> DualBounds::rowsSourcedBy(Index col) const {
    std::vector<Index> rows;
    for (size_t i = 0; i < row_dual_lower_source_.size(); ++i) {
        if (row_dual_lower_source_[i] == col || row_dual_upper_source_[i] == col) {
            rows.push_back(static_cast<Index>(i));
        }
    }
    return rows;
}

bool DualBounds::isImpliedEquationAtLower(Index row) const {
    return row_lower_[row] != -kInf && impl_row_dual_lower_[row] > kDualFeasTol;
}

bool DualBounds::isImpliedEquationAtUpper(Index row) const {
    return row_upper_[row] != kInf && impl_row_dual_upper_[row] < -kDualFeasTol;
}

Index DualBounds::numImpliedRowDualBounds() const {
    Index count = 0;
    for (size_t i = 0; i < impl_row_dual_lower_.size(); ++i) {
        count += impl_row_dual_lower_[i] != -kInf ? 1 : 0;
        count += impl_row_dual_upper_[i] != kInf ? 1 : 0;
    }
    return count;
}

bool DualBounds::isStrictlyImpliedForDual(Index col, bool lower) const {
    // HiGHS's isStrictlyImpliedForDual: a singleton column's bound need only
    // be implied within the tolerance.
    const Real margin = col_size_[col] != 1 ? kPrimalFeasTol : -kPrimalFeasTol;
    if (lower) {
        return col_lower_[col] == -kInf || impl_col_lower_[col] > col_lower_[col] + margin;
    }
    return col_upper_[col] == kInf || impl_col_upper_[col] < col_upper_[col] - margin;
}

Real DualBounds::sumLower(Index col, Real offset) const {
    const Sum& s = sums_[col];
    return s.lower_inf == 0 ? withOffset(s.lower, offset) : -kInf;
}

Real DualBounds::sumUpper(Index col, Real offset) const {
    const Sum& s = sums_[col];
    return s.upper_inf == 0 ? withOffset(s.upper, offset) : kInf;
}

Real DualBounds::sumLowerOrig(Index col, Real offset) const {
    const Sum& s = sums_[col];
    return s.lower_orig_inf == 0 ? withOffset(s.lower_orig, offset) : -kInf;
}

Real DualBounds::sumUpperOrig(Index col, Real offset) const {
    const Sum& s = sums_[col];
    return s.upper_orig_inf == 0 ? withOffset(s.upper_orig, offset) : kInf;
}

Real DualBounds::residualSumLower(Index col, Index row, Real a) const {
    const Sum& s = sums_[col];
    return residualOf(s.lower, s.lower_inf, a,
                      a > 0.0 ? implVarLower(col, row) : implVarUpper(col, row), true);
}

Real DualBounds::residualSumUpper(Index col, Index row, Real a) const {
    const Sum& s = sums_[col];
    return residualOf(s.upper, s.upper_inf, a,
                      a > 0.0 ? implVarUpper(col, row) : implVarLower(col, row), false);
}

Real DualBounds::residualSumLowerOrig(Index col, Index row, Real a) const {
    const Sum& s = sums_[col];
    return residualOf(s.lower_orig, s.lower_orig_inf, a,
                      a > 0.0 ? row_dual_lower_[row] : row_dual_upper_[row], true);
}

Real DualBounds::residualSumUpperOrig(Index col, Index row, Real a) const {
    const Sum& s = sums_[col];
    return residualOf(s.upper_orig, s.upper_orig_inf, a,
                      a > 0.0 ? row_dual_upper_[row] : row_dual_lower_[row], false);
}

}  // namespace mipx
