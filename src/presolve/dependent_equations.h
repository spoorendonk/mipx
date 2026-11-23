// Translated from HiGHS 1.16: highs/presolve/HPresolveDependent.cpp
// (HPresolve::removeDependentEquations). HiGHS's licence:
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

#include "mipx/core.h"

#include <vector>

namespace mipx {

struct LpProblem;
class PresolveMatrix;

/// What findDependentEquations found.
struct DependentEquations {
    /// The equations to drop, in the factorization's order.
    std::vector<Index> rows;
    /// Columns left without a pivot that were fictitious (HiGHS's
    /// "fictitious rows skipped"): nothing to drop for them. Always 0, as in
    /// HiGHS, kept to match it: build() returns the count of real unpivoted
    /// columns, which varWithNoPivot() lists before the -1s.
    Index fictitious = 0;
    /// The factorization ran out of time; `rows` is then empty.
    bool timed_out = false;
};

/// HiGHS's removeDependentEquations (#393), up to the removal. Each active
/// equation (row_lower == row_upper exactly, as HiGHS's isEquation) is a
/// column of a matrix with one row per LP column and one more, num_cols, for
/// the rhs: its nonzeros on active columns, and row_lower in the rhs row when
/// nonzero. The columns come ordered as HiGHS's equations set, by that size
/// and then by row. Two known deviations, neither a correctness matter.
/// mipx reads the set from the current state; HiGHS builds it when it sets
/// up its matrix or shrinks the problem (which it does right before this
/// pass once half the rows or columns are gone) and in between only deletes,
/// adds cut rows and re-sorts some rows whose size changed, so it misses LP
/// rows that became equations through bound changes, and some sizes are
/// stale. And each equation's entries go in by LP column index, where HiGHS
/// walks its row tree in pre-order; the Markowitz search may then break ties
/// differently. A column the rank-revealing LU leaves without a pivot is a
/// combination of columns with one, rhs included, so its equation holds
/// wherever theirs do. When LP rows are combinations of others but their rhs
/// is not, the rhs row takes a pivot and the span keeps the inconsistency,
/// so whatever is dropped, the LP solve still finds it infeasible: HiGHS
/// reports no infeasibility here. `time_limit` is the factorization's, in
/// seconds.
DependentEquations findDependentEquations(const LpProblem& lp, const PresolveMatrix& work,
                                          const std::vector<bool>& col_removed,
                                          const std::vector<bool>& row_removed, double time_limit);

}  // namespace mipx
