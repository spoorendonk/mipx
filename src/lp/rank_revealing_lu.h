// Translated from HiGHS 1.16: highs/util/HFactor.cpp, highs/util/HFactor.h
// and highs/util/HFactorConst.h. HiGHS's licence:
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

#include <chrono>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace mipx {

/// Rank-revealing sparse LU of a rectangular matrix (#392, part 1 of #391):
/// HiGHS 1.16's HFactor::build() elimination, as HPresolve's
/// removeDependentEquations uses it to find dependent equations. The matrix
/// has num_row rows and num_basic columns, either more or fewer than the
/// rows. build() eliminates as HFactor does (unit and singleton passes, then
/// Markowitz search over count lists with threshold pivoting) and reports
/// which columns got no pivot.
///
/// The columns are HFactor's basic variables with basic_index the identity,
/// all structural (HPresolve passes 0..num_basic-1 over a matrix of exactly
/// those columns). Only the elimination is kept: no L or U factor is stored,
/// so there are no solves or updates.
class RankRevealingLU {
public:
    // Pivot threshold limits and default (HFactorConst.h).
    static constexpr Real kMinPivotThreshold = 8e-4;
    static constexpr Real kDefaultPivotThreshold = 0.1;
    static constexpr Real kMaxPivotThreshold = 0.5;
    // Minimum absolute pivot limits and default (HFactorConst.h).
    static constexpr Real kMinPivotTolerance = 0;
    static constexpr Real kDefaultPivotTolerance = 1e-10;
    static constexpr Real kMaxPivotTolerance = 1.0;
    /// build()'s return when the kernel bails out on the time limit
    /// (HFactor.h).
    static constexpr Index kBuildKernelReturnTimeout = -1;

    /// HFactor::setupGeneral: the matrix as CSC, column j's entries at
    /// positions start[j]..start[j+1]-1 of `index` (rows in [0, num_row),
    /// no row twice in a column) and `value`. The spans are not copied and
    /// must outlive build(). The thresholds are clamped to their limits, as
    /// HFactor does, and the time limit is reset to none. Throws
    /// std::invalid_argument on inconsistent sizes or a bad row index.
    void setup(Index num_row, std::span<const Index> start, std::span<const Index> index,
               std::span<const Real> value, Real pivot_threshold = kDefaultPivotThreshold,
               Real pivot_tolerance = kDefaultPivotTolerance);

    /// HFactor::setTimeLimit: seconds of wall clock build() may take; < 0
    /// for none (the default). The kernel checks its expected finishing time
    /// against it every so many pivots and bails out when it would be over.
    void setTimeLimit(double time_limit);

    /// HFactor::build up to buildMarkSingC. Returns kBuildKernelReturnTimeout
    /// on a timeout, else the rank deficiency of the columns: with
    /// num_basic < num_row, rankDeficiency() less the num_row - num_basic
    /// rows no column could have covered. Throws std::logic_error before
    /// setup().
    [[nodiscard]] Index build();

    [[nodiscard]] Index numRow() const { return num_row_; }
    [[nodiscard]] Index numBasic() const { return num_basic_; }

    /// HFactor::rank_deficiency after build(): the matrix's, counting the
    /// num_row - num_basic missing columns of an incomplete one, or the
    /// excess num_basic - num_row columns of an over-complete one. 0 after a
    /// timeout.
    [[nodiscard]] Index rankDeficiency() const { return rank_deficiency_; }
    /// HFactor::var_with_no_pivot, rankDeficiency() entries: the column each
    /// row in rowWithNoPivot() was left to, -1 for a fictitious column
    /// (num_basic <= its index < num_row, filling out an incomplete matrix).
    /// The real ones come first. Empty when the matrix has full rank.
    [[nodiscard]] std::span<const Index> varWithNoPivot() const { return var_with_no_pivot_; }
    /// HFactor::row_with_no_pivot: the rows with no pivot, ascending, then
    /// any fictitious rows num_row.. of an over-complete matrix.
    [[nodiscard]] std::span<const Index> rowWithNoPivot() const { return row_with_no_pivot_; }
    /// HFactor::col_with_no_pivot: the columns with no pivot, ascending,
    /// then the fictitious columns num_basic..num_row-1 of an incomplete
    /// matrix.
    [[nodiscard]] std::span<const Index> colWithNoPivot() const { return col_with_no_pivot_; }

    /// The kernel's bounds on the total build time, seconds, from its timeout
    /// checks (HFactor::min_time_bound_, max_time_bound_; HPresolve logs
    /// them). inf and 0 when no check ran.
    [[nodiscard]] double minTimeBound() const { return min_time_bound_; }
    [[nodiscard]] double maxTimeBound() const { return max_time_bound_; }

private:
    using Clock = std::chrono::steady_clock;

    void buildSimple();
    Index buildKernel();
    void buildHandleRankDeficiency();
    void buildMarkSingC();
    void zeroCol(Index j_col, Real pivot_multiplier);
    /// HighsTimer::read on build()'s timer: seconds since build() started.
    [[nodiscard]] double readTimer() const;

    // HFactor.h's local in-line functions.
    void colInsert(Index i_col, Index i_row, Real value) {
        const Index iput = mc_start_[i_col] + mc_count_a_[i_col]++;
        mc_index_[iput] = i_row;
        mc_value_[iput] = value;
    }
    void colStoreN(Index i_col, Index i_row, Real value) {
        const Index iput = mc_start_[i_col] + mc_space_[i_col] - (++mc_count_n_[i_col]);
        mc_index_[iput] = i_row;
        mc_value_[iput] = value;
    }
    void colFixMax(Index i_col);
    Real colDelete(Index i_col, Index i_row) {
        Index idel = mc_start_[i_col];
        const Index imov = idel + (--mc_count_a_[i_col]);
        while (mc_index_[idel] != i_row) {
            ++idel;
        }
        const Real pivot_multiplier = mc_value_[idel];
        mc_index_[idel] = mc_index_[imov];
        mc_value_[idel] = mc_value_[imov];
        return pivot_multiplier;
    }
    void rowInsert(Index i_col, Index i_row) {
        const Index iput = mr_start_[i_row] + mr_count_[i_row]++;
        mr_index_[iput] = i_col;
    }
    void rowDelete(Index i_col, Index i_row) {
        Index idel = mr_start_[i_row];
        const Index imov = idel + (--mr_count_[i_row]);
        while (mr_index_[idel] != i_col) {
            ++idel;
        }
        mr_index_[idel] = mr_index_[imov];
    }
    void clinkAdd(Index index, Index count) {
        const Index mover = col_link_first_[count];
        col_link_last_[index] = -2 - count;
        col_link_next_[index] = mover;
        col_link_first_[count] = index;
        if (mover >= 0) {
            col_link_last_[mover] = index;
        }
    }
    void clinkDel(Index index) {
        const Index xlast = col_link_last_[index];
        const Index xnext = col_link_next_[index];
        if (xlast >= 0) {
            col_link_next_[xlast] = xnext;
        } else {
            col_link_first_[-xlast - 2] = xnext;
        }
        if (xnext >= 0) {
            col_link_last_[xnext] = xlast;
        }
    }
    void rlinkAdd(Index index, Index count) {
        const Index mover = row_link_first_[count];
        row_link_last_[index] = -2 - count;
        row_link_next_[index] = mover;
        row_link_first_[count] = index;
        if (mover >= 0) {
            row_link_last_[mover] = index;
        }
    }
    void rlinkDel(Index index) {
        const Index xlast = row_link_last_[index];
        const Index xnext = row_link_next_[index];
        if (xlast >= 0) {
            row_link_next_[xlast] = xnext;
        } else {
            row_link_first_[-xlast - 2] = xnext;
        }
        if (xnext >= 0) {
            row_link_last_[xnext] = xlast;
        }
    }

    // Problem size and matrix.
    Index num_row_ = 0;
    Index num_basic_ = 0;
    std::span<const Index> a_start_;
    std::span<const Index> a_index_;
    std::span<const Real> a_value_;
    Real pivot_threshold_ = kDefaultPivotThreshold;
    Real pivot_tolerance_ = kDefaultPivotTolerance;
    double time_limit_ = std::numeric_limits<double>::infinity();
    Clock::time_point build_start_{};

    // Results.
    Index rank_deficiency_ = 0;
    std::vector<Index> row_with_no_pivot_;
    std::vector<Index> col_with_no_pivot_;
    std::vector<Index> var_with_no_pivot_;
    double min_time_bound_ = std::numeric_limits<double>::infinity();
    double max_time_bound_ = 0;

    // Kernel statistics the timeout model reads.
    Index kernel_dim_ = 0;
    Index num_active_nz_ = 0;

    // Working buffer.
    Index nwork_ = 0;
    std::vector<Index> iwork_;

    // Basis matrix.
    std::vector<Index> b_start_;
    std::vector<Index> b_index_;
    std::vector<Real> b_value_;

    // Permutation: the row each column pivoted on, -1 for none.
    std::vector<Index> permute_;

    // Kernel matrix, column-wise: per column the active entries from
    // mc_start_, the non-active ones (in pivoted rows) at the end of its
    // mc_space_.
    std::vector<Index> mc_start_;
    std::vector<Index> mc_count_a_;
    std::vector<Index> mc_count_n_;
    std::vector<Index> mc_space_;
    std::vector<Index> mc_index_;
    std::vector<Real> mc_value_;
    std::vector<Real> mc_min_pivot_;

    // Kernel matrix, row-wise pattern.
    std::vector<Index> mr_start_;
    std::vector<Index> mr_count_;
    std::vector<Index> mr_space_;
    std::vector<Index> mr_count_before_;
    std::vector<Index> mr_index_;

    // Kernel column buffer.
    std::vector<Index> mwz_column_index_;
    std::vector<std::uint8_t> mwz_column_mark_;
    std::vector<Real> mwz_column_array_;

    // Count link lists.
    std::vector<Index> col_link_first_;
    std::vector<Index> col_link_next_;
    std::vector<Index> col_link_last_;
    std::vector<Index> row_link_first_;
    std::vector<Index> row_link_next_;
    std::vector<Index> row_link_last_;
};

}  // namespace mipx
