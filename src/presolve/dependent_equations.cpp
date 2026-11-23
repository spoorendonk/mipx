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

#include "dependent_equations.h"

#include "../lp/rank_revealing_lu.h"
#include "mipx/lp_problem.h"
#include "presolve_matrix.h"

#include <algorithm>
#include <utility>

namespace mipx {

DependentEquations findDependentEquations(const LpProblem& lp, const PresolveMatrix& work,
                                          const std::vector<bool>& col_removed,
                                          const std::vector<bool>& row_removed, double time_limit) {
    DependentEquations result;

    // (size, row): HiGHS's std::set of equations, the size its rowsize, the
    // row's nonzeros on active columns.
    std::vector<std::pair<Index, Index>> equations;
    for (Index i = 0; i < lp.num_rows; ++i) {
        if (row_removed[i] || lp.row_lower[i] != lp.row_upper[i]) {
            continue;
        }
        Index size = 0;
        for (const auto [j, a] : work.row(i)) {
            if (!col_removed[j] && a != 0.0) {
                ++size;
            }
        }
        equations.emplace_back(size, i);
    }
    if (equations.empty()) {
        return result;
    }
    std::sort(equations.begin(), equations.end());

    std::vector<Index> start;
    std::vector<Index> index;
    std::vector<Real> value;
    start.reserve(equations.size() + 1);
    start.push_back(0);
    for (const auto& [size, i] : equations) {
        // The working matrix keeps each row sorted by column.
        for (const auto [j, a] : work.row(i)) {
            if (!col_removed[j] && a != 0.0) {
                index.push_back(j);
                value.push_back(a);
            }
        }
        if (lp.row_lower[i] != 0.0) {
            index.push_back(lp.num_cols);
            value.push_back(lp.row_lower[i]);
        }
        start.push_back(static_cast<Index>(index.size()));
    }

    RankRevealingLU lu;
    lu.setup(lp.num_cols + 1, start, index, value);
    lu.setTimeLimit(time_limit);
    const Index build_return = lu.build();
    if (build_return == RankRevealingLU::kBuildKernelReturnTimeout) {
        result.timed_out = true;
        return result;
    }
    // As HiGHS: the first build_return entries, a negative one fictitious.
    const auto no_pivot = lu.varWithNoPivot();
    for (Index k = 0; k < build_return; ++k) {
        if (no_pivot[k] < 0) {
            ++result.fictitious;
        } else {
            result.rows.push_back(equations[no_pivot[k]].second);
        }
    }
    return result;
}

}  // namespace mipx
