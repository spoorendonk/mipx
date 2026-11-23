// Translated from HiGHS 1.16: highs/presolve/HPresolve.cpp
// (checkUpdateColImpliedBounds, updateColImpliedBounds,
// checkUpdateRowDualImpliedBounds, updateRowDualImpliedBounds,
// isStrictlyImpliedForDual, isImpliedEquationAtLower/Upper) and
// highs/util/HighsLinearSumBounds.h, HighsLinearSumBounds.cpp
// (getImplVarLower/Upper, the residual and sum queries). The fixpoint driver
// that recomputes them from the presolve state is mipx's own. HiGHS's
// licence:
//
// MIT License
//
// Copyright (c) 2026 HiGHS
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include "compensated_sum.h"
#include "mipx/lp_problem.h"

#include <vector>

namespace mipx {

class PresolveMatrix;

/// The bounds HiGHS's dominated-column detection reads (#375, part 1 of
/// #361; HiGHS 1.16.0 HPresolve), for an LP read as a minimisation, with the
/// row duals y_i and each column's dual constraint c_j - sum_i a_ij y_i:
///
/// - row-dual bounds from the row sides: a row with no lower side has
///   y_i <= 0, one with no upper side y_i >= 0 (HiGHS's rowDualLower/Upper);
/// - primal implied column bounds from the rows, each with the row it came
///   from (implColLower/Upper, colLowerSource/UpperSource), which tell
///   whether a column bound is strictly implied and so never active;
/// - implied row-dual bounds from the columns' dual constraints, each with
///   the column it came from (implRowDualLower/Upper, rowDualLowerSource/
///   UpperSource, updateRowDualImpliedBounds);
/// - the dual activity sums per column, sum_i a_ij y_i over its active rows
///   (HiGHS's impliedDualRowBounds, a HighsLinearSumBounds): with the
///   implied row-dual bounds, except one the column itself is the source of,
///   and the ...Orig sums with the row sides' bounds only.
///
/// HiGHS keeps these up to date event by event. mipx's passes keep no such
/// store, so compute() derives them afresh from the presolve state as it
/// stands: a source column fixed, removed or changed counts on the next call.
/// The primal and dual implied bounds are coupled (a row whose dual is
/// implied nonzero is an implied equation, which tightens the side the
/// primal bounds use; a primal bound strictly implied gives the column's
/// dual constraint a side), so compute() alternates the two until the
/// implied equations stop growing, the fixpoint HiGHS's propagation reaches.
/// Each implied bound comes from the ...Orig residuals, as in HiGHS, and so
/// does not depend on the other implied bounds of its kind.
class DualBounds {
public:
    /// HiGHS's default primal and dual feasibility tolerances.
    static constexpr Real kPrimalFeasTol = 1e-7;
    static constexpr Real kDualFeasTol = 1e-7;

    /// Derive every bound and sum from the active rows and columns of `m`,
    /// with `lp`'s bounds, sides, costs and sense. Values at or beyond 1e20
    /// in size count as infinite, as presolve's activities take them.
    void compute(const LpProblem& lp, const PresolveMatrix& m, const std::vector<bool>& col_removed,
                 const std::vector<bool>& row_removed);

    // Row duals.
    [[nodiscard]] Real rowDualLower(Index row) const { return row_dual_lower_[row]; }
    [[nodiscard]] Real rowDualUpper(Index row) const { return row_dual_upper_[row]; }
    [[nodiscard]] Real implRowDualLower(Index row) const { return impl_row_dual_lower_[row]; }
    [[nodiscard]] Real implRowDualUpper(Index row) const { return impl_row_dual_upper_[row]; }
    /// The column an implied row-dual bound came from, -1 for none.
    [[nodiscard]] Index rowDualLowerSource(Index row) const { return row_dual_lower_source_[row]; }
    [[nodiscard]] Index rowDualUpperSource(Index row) const { return row_dual_upper_source_[row]; }
    /// Rows with an implied dual bound from `col` (HiGHS's
    /// implRowDualSourceByCol), in row order.
    [[nodiscard]] std::vector<Index> rowsSourcedBy(Index col) const;
    [[nodiscard]] bool isImpliedEquationAtLower(Index row) const;
    [[nodiscard]] bool isImpliedEquationAtUpper(Index row) const;
    /// Number of finite implied row-dual bounds, lower and upper counted
    /// apart.
    [[nodiscard]] Index numImpliedRowDualBounds() const;

    // Primal implied column bounds.
    [[nodiscard]] Real implColLower(Index col) const { return impl_col_lower_[col]; }
    [[nodiscard]] Real implColUpper(Index col) const { return impl_col_upper_[col]; }
    /// The row an implied column bound came from, -1 for none.
    [[nodiscard]] Index colLowerSource(Index col) const { return col_lower_source_[col]; }
    [[nodiscard]] Index colUpperSource(Index col) const { return col_upper_source_[col]; }
    /// Whether the column's lower (upper) bound is strictly implied by the
    /// rows, so never active: its reduced cost cannot be positive
    /// (negative), and its dual constraint has the side
    /// sum_i a_ij y_i >= (<=) c_j. A column in one row needs the bound
    /// implied only within the primal tolerance.
    [[nodiscard]] bool isStrictlyImpliedForDual(Index col, bool lower) const;

    // Dual activity sums of a column, plus `offset`; -inf (+inf) while a
    // term is unbounded.
    [[nodiscard]] Real sumLower(Index col, Real offset = 0.0) const;
    [[nodiscard]] Real sumUpper(Index col, Real offset = 0.0) const;
    [[nodiscard]] Real sumLowerOrig(Index col, Real offset = 0.0) const;
    [[nodiscard]] Real sumUpperOrig(Index col, Real offset = 0.0) const;
    [[nodiscard]] Index numInfSumLower(Index col) const { return sums_[col].lower_inf; }
    [[nodiscard]] Index numInfSumUpper(Index col) const { return sums_[col].upper_inf; }
    [[nodiscard]] Index numInfSumLowerOrig(Index col) const { return sums_[col].lower_orig_inf; }
    [[nodiscard]] Index numInfSumUpperOrig(Index col) const { return sums_[col].upper_orig_inf; }
    /// The sums less row `row`'s term, `a` being its coefficient in `col`.
    [[nodiscard]] Real residualSumLower(Index col, Index row, Real a) const;
    [[nodiscard]] Real residualSumUpper(Index col, Index row, Real a) const;
    [[nodiscard]] Real residualSumLowerOrig(Index col, Index row, Real a) const;
    [[nodiscard]] Real residualSumUpperOrig(Index col, Index row, Real a) const;

private:
    struct Sum {
        CompensatedSum lower;
        CompensatedSum upper;
        CompensatedSum lower_orig;
        CompensatedSum upper_orig;
        Index lower_inf = 0;
        Index upper_inf = 0;
        Index lower_orig_inf = 0;
        Index upper_orig_inf = 0;
    };

    void computeImpliedColBounds(const LpProblem& lp, const PresolveMatrix& m,
                                 const std::vector<bool>& col_removed,
                                 const std::vector<bool>& row_removed);
    void computeImpliedRowDualBounds(const PresolveMatrix& m, const std::vector<bool>& col_removed,
                                     const std::vector<bool>& row_removed);
    void computeDualSums(const PresolveMatrix& m, const std::vector<bool>& col_removed,
                         const std::vector<bool>& row_removed);
    /// A row dual's bounds as column `col`'s sum sees them: the implied one
    /// unless `col` is its source (HighsLinearSumBounds::getImplVarLower).
    [[nodiscard]] Real implVarLower(Index col, Index row) const;
    [[nodiscard]] Real implVarUpper(Index col, Index row) const;

    // Model data the predicates read, copied at compute().
    std::vector<Real> col_lower_;
    std::vector<Real> col_upper_;
    std::vector<Real> cost_;
    std::vector<Real> row_lower_;
    std::vector<Real> row_upper_;
    std::vector<Index> col_size_;

    std::vector<Real> row_dual_lower_;
    std::vector<Real> row_dual_upper_;
    std::vector<Real> impl_row_dual_lower_;
    std::vector<Real> impl_row_dual_upper_;
    std::vector<Index> row_dual_lower_source_;
    std::vector<Index> row_dual_upper_source_;

    std::vector<Real> impl_col_lower_;
    std::vector<Real> impl_col_upper_;
    std::vector<Index> col_lower_source_;
    std::vector<Index> col_upper_source_;

    std::vector<Sum> sums_;
};

}  // namespace mipx
