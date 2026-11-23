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

#include "rank_revealing_lu.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <string>

namespace mipx {

namespace {

// HConst.h's kHighsTiny: an eliminated value below it in size cancels.
constexpr Real kHighsTiny = 1e-14;
constexpr double kHighsInf = std::numeric_limits<double>::infinity();
// HFactorConst.h: the Markowitz arrays' initial capacity, as multiples of
// the basis matrix limit size. Only where a kernel resize moves the data
// depends on them, and that the timeout model leaves out of its average.
constexpr Index kMCExtraEntriesMultiplier = 2;
constexpr Index kMRExtraEntriesMultiplier = 2;
// HFactor::buildKernel's literal 8: the Markowitz search stops after this
// many columns and rows once it has a candidate.
constexpr Index kMarkowitzSearchLimit = 8;

}  // namespace

void RankRevealingLU::setup(Index num_row, std::span<const Index> start,
                            std::span<const Index> index, std::span<const Real> value,
                            Real pivot_threshold, Real pivot_tolerance) {
    if (num_row < 0) {
        throw std::invalid_argument("RankRevealingLU: negative row count");
    }
    if (start.empty()) {
        throw std::invalid_argument("RankRevealingLU: start needs num_basic + 1 entries");
    }
    const auto num_basic = static_cast<Index>(start.size() - 1);
    if (start[0] < 0 || static_cast<std::size_t>(start[num_basic]) > index.size() ||
        index.size() != value.size()) {
        throw std::invalid_argument("RankRevealingLU: index and value do not match start");
    }
    // No row twice in a column: colDelete and rowDelete find an entry by
    // its index, so a duplicate would corrupt the kernel.
    std::vector<Index> last_col(static_cast<std::size_t>(num_row), -1);
    for (Index j = 0; j < num_basic; ++j) {
        if (start[j + 1] < start[j]) {
            throw std::invalid_argument("RankRevealingLU: start is not monotone");
        }
        for (Index k = start[j]; k < start[j + 1]; ++k) {
            const Index i = index[k];
            if (i < 0 || i >= num_row) {
                throw std::invalid_argument("RankRevealingLU: row index " + std::to_string(i) +
                                            " out of range in column " + std::to_string(j));
            }
            if (last_col[i] == j) {
                throw std::invalid_argument("RankRevealingLU: row " + std::to_string(i) +
                                            " twice in column " + std::to_string(j));
            }
            last_col[i] = j;
        }
    }

    num_row_ = num_row;
    num_basic_ = num_basic;
    a_start_ = start;
    a_index_ = index;
    a_value_ = value;
    pivot_threshold_ = std::max(kMinPivotThreshold, std::min(pivot_threshold, kMaxPivotThreshold));
    pivot_tolerance_ = std::max(kMinPivotTolerance, std::min(pivot_tolerance, kMaxPivotTolerance));
    time_limit_ = kHighsInf;

    // setupGeneral's basis matrix limit size. Every column of the matrix is
    // basic here, so its count-sorted sum takes all of them: the nonzeros
    // plus the larger dimension.
    const Index b_max_dim = std::max(num_row_, num_basic_);
    const Index basis_matrix_limit_size = (start[num_basic] - start[0]) + b_max_dim;
    mc_start_.resize(num_basic_);
    mc_count_a_.resize(num_basic_);
    mc_count_n_.resize(num_basic_);
    mc_space_.resize(num_basic_);
    mc_min_pivot_.resize(num_basic_);
    mc_index_.reserve(static_cast<std::size_t>(basis_matrix_limit_size) *
                      kMCExtraEntriesMultiplier);
    mc_value_.reserve(static_cast<std::size_t>(basis_matrix_limit_size) *
                      kMCExtraEntriesMultiplier);

    mr_start_.resize(num_row_);
    mr_count_.resize(num_row_);
    mr_space_.resize(num_row_);
    mr_count_before_.resize(num_row_);
    mr_index_.reserve(static_cast<std::size_t>(basis_matrix_limit_size) *
                      kMRExtraEntriesMultiplier);

    mwz_column_mark_.assign(num_row_, 0);
    mwz_column_index_.resize(num_row_);
    mwz_column_array_.assign(num_row_, 0.0);

    col_link_next_.resize(num_basic_);
    col_link_last_.resize(num_basic_);
    row_link_next_.resize(num_row_);
    row_link_last_.resize(num_row_);
}

void RankRevealingLU::setTimeLimit(double time_limit) {
    time_limit_ = kHighsInf;
    if (time_limit < 0) {
        return;
    }
    time_limit_ = time_limit;
}

double RankRevealingLU::readTimer() const {
    return std::chrono::duration<double>(Clock::now() - build_start_).count();
}

void RankRevealingLU::colFixMax(Index i_col) {
    Real max_value = 0;
    for (Index k = mc_start_[i_col]; k < mc_start_[i_col] + mc_count_a_[i_col]; ++k) {
        max_value = std::max(max_value, std::fabs(mc_value_[k]));
    }
    // As HFactor: not max_value * pivot_threshold alone, which may fall below
    // pivot_tolerance and so admit a pivot the tolerance then defers while a
    // larger entry would have done (HiGHS #3352).
    mc_min_pivot_[i_col] = std::max(max_value * pivot_threshold_, pivot_tolerance_);
}

Index RankRevealingLU::build() {
    // HFactor's build timer, read by the kernel's timeout check.
    build_start_ = Clock::now();
    rank_deficiency_ = 0;
    row_with_no_pivot_.clear();
    col_with_no_pivot_.clear();
    var_with_no_pivot_.clear();

    buildSimple();
    const Index build_kernel_return = buildKernel();
    // kBuildKernelReturnTimeout: the time limit is reached. Otherwise the
    // rank deficiency of the columns; with num_basic < num_row the rows they
    // leave uncovered are found by continuing as if a full set of columns
    // were rank deficient.
    if (build_kernel_return == kBuildKernelReturnTimeout) {
        return kBuildKernelReturnTimeout;
    }
    rank_deficiency_ = build_kernel_return;
    const bool incomplete_basis = num_basic_ < num_row_;
    if (rank_deficiency_ != 0 || incomplete_basis) {
        buildHandleRankDeficiency();
        buildMarkSingC();
    }
    if (incomplete_basis) {
        return rank_deficiency_ - (num_row_ - num_basic_);
    }
    return rank_deficiency_;
}

// HFactor::buildSimple's three stages (unit columns, singleton passes, the
// kernel's set-up) kept in one function, as HiGHS has them, so the port reads
// against its reference stage by stage.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void RankRevealingLU::buildSimple() {
    // Unpermuted (rank deficient) columns keep -1.
    permute_.assign(num_basic_, -1);

    // 1. Prepare the basis matrix and deal with unit columns.
    Index b_count_x = 0;
    std::fill(mr_count_before_.begin(), mr_count_before_.end(), 0);
    nwork_ = 0;
    // iwork_ collects the nwork_ non-unit columns: they form the kernel.
    iwork_.assign(num_basic_ + 1, 0);
    b_start_.assign(num_basic_ + 1, 0);
    b_index_.resize(static_cast<std::size_t>(a_start_[num_basic_] - a_start_[0]));
    b_value_.resize(b_index_.size());
    for (Index i_col = 0; i_col < num_basic_; ++i_col) {
        Index i_row = -1;
        const Index start = a_start_[i_col];
        const Index count = a_start_[i_col + 1] - start;
        // A unit column pivots at once unless its row already has a pivot
        // (mr_count_before_ negative); then it goes to the kernel, so that
        // any rank deficiency is found as late as possible.
        bool ok_unit_col = false;
        if (count == 1) {
            ok_unit_col = a_value_[start] == 1 && mr_count_before_[a_index_[start]] >= 0;
        }
        if (ok_unit_col) {
            i_row = a_index_[start];
        } else {
            for (Index k = start; k < start + count; ++k) {
                mr_count_before_[a_index_[k]]++;
                b_index_[b_count_x] = a_index_[k];
                b_value_[b_count_x++] = a_value_[k];
            }
            iwork_[nwork_++] = i_col;
        }
        if (i_row >= 0) {
            permute_[i_col] = i_row;
            // Negative enough that the accumulated count can never reach 0.
            mr_count_before_[i_row] = -num_basic_;
        }
        b_start_[i_col + 1] = b_count_x;
    }

    // 2. Search for and deal with singletons.
    while (nwork_ > 0) {
        const Index nwork_last = nwork_;
        nwork_ = 0;
        for (Index i = 0; i < nwork_last; ++i) {
            const Index i_col = iwork_[i];
            const Index start = b_start_[i_col];
            const Index end = b_start_[i_col + 1];
            Index pivot_k = -1;
            bool found_row_singleton = false;
            Index count = 0;

            // 2.1 Search for a singleton.
            for (Index k = start; k < end; ++k) {
                const Index i_row = b_index_[k];
                if (mr_count_before_[i_row] == 1) {
                    pivot_k = k;
                    found_row_singleton = true;
                    break;
                }
                if (mr_count_before_[i_row] > 1) {
                    pivot_k = k;
                    count++;
                }
            }

            if (found_row_singleton) {
                // 2.2 Row singleton: the column's other entries leave their
                // rows (to L if active, U if not; neither is stored here).
                for (Index k = start; k < end; ++k) {
                    if (k != pivot_k) {
                        mr_count_before_[b_index_[k]]--;
                    }
                }
                const Index i_row = b_index_[pivot_k];
                mr_count_before_[i_row] = 0;
                permute_[i_col] = i_row;
            } else if (count == 1) {
                // 2.3 Column singleton: its other entries are all in rows
                // already pivoted (to U).
                const Index i_row = b_index_[pivot_k];
                mr_count_before_[i_row] = 0;
                permute_[i_col] = i_row;
            } else {
                iwork_[nwork_++] = i_col;
            }
        }
        // No singleton found in the last pass.
        if (nwork_last == nwork_) {
            break;
        }
    }

    // 3. Prepare the kernel parts.
    // 3.1 Row links, row matrix spaces.
    row_link_first_.assign(num_basic_ + 1, -1);
    std::fill(mr_count_.begin(), mr_count_.end(), 0);
    Index mr_count_x = 0;
    // The active nonzeros, updated as values are eliminated and fill-in or
    // cancellation occur.
    num_active_nz_ = 0;
    for (Index i_row = 0; i_row < num_row_; ++i_row) {
        const Index count = mr_count_before_[i_row];
        if (count > 0) {
            mr_start_[i_row] = mr_count_x;
            mr_space_[i_row] = count * 2;
            mr_count_x += count * 2;
            rlinkAdd(i_row, count);
            num_active_nz_ += count;
        }
    }
    mr_index_.resize(mr_count_x);

    // 3.2 Column links, kernel matrix. HFactor sizes col_link_first to
    // num_row + 1; at least 2 here, so that the kernel's read of
    // col_link_first[1] stays in range for a matrix with no rows.
    col_link_first_.assign(std::max(num_row_, Index{1}) + 1, -1);
    mc_index_.clear();
    mc_value_.clear();
    std::fill(mc_count_a_.begin(), mc_count_a_.end(), 0);
    std::fill(mc_count_n_.begin(), mc_count_n_.end(), 0);
    Index mc_count_x = 0;
    for (Index i = 0; i < nwork_; ++i) {
        const Index i_col = iwork_[i];
        mc_start_[i_col] = mc_count_x;
        mc_space_[i_col] = (b_start_[i_col + 1] - b_start_[i_col]) * 2;
        mc_count_x += mc_space_[i_col];
        mc_index_.resize(mc_count_x);
        mc_value_.resize(mc_count_x);
        for (Index k = b_start_[i_col]; k < b_start_[i_col + 1]; ++k) {
            const Index i_row = b_index_[k];
            const Real value = b_value_[k];
            if (mr_count_before_[i_row] > 0) {
                // In the active part of the kernel.
                colInsert(i_col, i_row, value);
                rowInsert(i_col, i_row);
            } else {
                // Above the active part of the kernel.
                colStoreN(i_col, i_row, value);
            }
        }
        colFixMax(i_col);
        clinkAdd(i_col, mc_count_a_[i_col]);
    }
    kernel_dim_ = nwork_;
    min_time_bound_ = kHighsInf;
    max_time_bound_ = 0;
}

// HFactor::buildKernel, 'n-work' pivoting on the kernel. One long function,
// as HiGHS's: the timeout model, the pivot search and the elimination share
// its running state, and splitting them would obscure the correspondence.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
Index RankRevealingLU::buildKernel() {
    Index search_k = 0;

    // With a finite time limit the kernel can bail out (HiGHS uses one only
    // for the dependent equations rule in presolve). The bail-out is
    // non-deterministic, as HiGHS's is.
    const bool check_for_timeout = time_limit_ < kHighsInf;
    // A deferred pivot increases nwork_ to force another loop, so count them
    // to know how many iterations might be needed.
    Index num_defer_pivot = 0;
    // Initial timer frequency: may be reduced if iterations get slow.
    const Index max_timer_frequency = 1000;
    Index timer_frequency = 100;
    // Average iteration time.
    double previous_iteration_time = readTimer();
    double average_iteration_time = 0;
    // Average rate of change of the active nonzeros, to predict when they
    // reach zero.
    Index previous_num_active_nz = num_active_nz_;
    double average_num_active_nz_change_rate = 0;
    // An iteration that moved a vector to resize it is abnormally slow and
    // is left out of the average.
    bool resize_data_shift = false;
    // Running average parameters.
    double mu0 = 0;
    double mu1 = 0;

    auto resize_index = [&](std::vector<Index>& i_vector, Index to_size) {
        const Index* from_p = i_vector.data();
        i_vector.resize(to_size);
        if (i_vector.data() != from_p) {
            resize_data_shift = true;
        }
    };
    auto resize_real = [&](std::vector<Real>& d_vector, Index to_size) {
        const Real* from_p = d_vector.data();
        d_vector.resize(to_size);
        if (d_vector.data() != from_p) {
            resize_data_shift = true;
        }
    };
    auto running_average_mu = [&]() {
        mu0 = (1.0 * timer_frequency) / (10.0 * max_timer_frequency);
        assert(mu0 <= 0.2);
        mu1 = 1.0 - mu0;
    };

    running_average_mu();
    while (nwork_-- > 0) {
        // Return if the current or expected time exceeds the limit.
        if (check_for_timeout && search_k > 0 && search_k % timer_frequency == 0) {
            // The current iteration time, into the running average unless a
            // resize moved data.
            const double current_time = readTimer();
            const double time_difference = current_time - previous_iteration_time;
            previous_iteration_time = current_time;
            const double iteration_time = time_difference / (1.0 * timer_frequency);
            if (!resize_data_shift) {
                average_iteration_time = mu1 * average_iteration_time + mu0 * iteration_time;
                if (timer_frequency > 1 && time_difference > time_limit_ / 1e1) {
                    timer_frequency = std::max(Index{1}, timer_frequency / 10);
                    running_average_mu();
                }
            }
            const double num_active_nz_change_rate =
                static_cast<double>(num_active_nz_ - previous_num_active_nz) /
                static_cast<double>(timer_frequency);
            previous_num_active_nz = num_active_nz_;
            average_num_active_nz_change_rate =
                mu1 * average_num_active_nz_change_rate + mu0 * num_active_nz_change_rate;

            // Iterations left: the bound, or fewer from the average rate of
            // change of the active nonzeros (if negative).
            Index iterations_left = kernel_dim_ - search_k + num_defer_pivot + 1;
            if (average_num_active_nz_change_rate < -1) {
                const auto active_nz_iterations_left =
                    static_cast<Index>(-num_active_nz_ / average_num_active_nz_change_rate);
                iterations_left = std::min(active_nz_iterations_left, iterations_left);
            }
            const double remaining_time_bound = average_iteration_time * iterations_left;
            const double total_time_bound = current_time + remaining_time_bound;
            min_time_bound_ = std::min(total_time_bound, min_time_bound_);
            max_time_bound_ = std::max(total_time_bound, max_time_bound_);
            if (current_time > time_limit_ || total_time_bound > time_limit_) {
                return kBuildKernelReturnTimeout;
            }
            resize_data_shift = false;
        }

        // 1. Search for the pivot.
        Index j_col_pivot = -1;
        Index i_row_pivot = -1;
        // 1.1 Search merits.
        const Index search_limit = std::min(nwork_, kMarkowitzSearchLimit);
        Index search_count = 0;
        const double merit_limit = 1.0 * num_basic_ * num_row_;
        double merit_pivot = merit_limit;
        search_k++;

        // 1.2 Local singletons.
        bool found_pivot = false;
        if (col_link_first_[1] != -1) {
            j_col_pivot = col_link_first_[1];
            i_row_pivot = mc_index_[mc_start_[j_col_pivot]];
            found_pivot = true;
        }
        if (!found_pivot && row_link_first_[1] != -1) {
            i_row_pivot = row_link_first_[1];
            j_col_pivot = mr_index_[mr_start_[i_row_pivot]];
            found_pivot = true;
        }

        // 1.3 Major search loop. A row count can exceed num_row when
        // num_basic > num_row.
        const Index max_count = std::max(num_row_, num_basic_);
        for (Index count = 2; !found_pivot && count <= max_count; ++count) {
            // A column count cannot exceed the number of rows.
            if (count <= num_row_) {
                // 1.3.1 Columns.
                for (Index j = col_link_first_[count]; j != -1; j = col_link_next_[j]) {
                    const Real min_pivot = mc_min_pivot_[j];
                    const Index start = mc_start_[j];
                    const Index end = start + mc_count_a_[j];
                    for (Index k = start; k < end; ++k) {
                        if (std::fabs(mc_value_[k]) >= min_pivot) {
                            const Index i = mc_index_[k];
                            const Index row_count = mr_count_[i];
                            const double merit_local = 1.0 * (count - 1) * (row_count - 1);
                            if (merit_pivot > merit_local) {
                                merit_pivot = merit_local;
                                j_col_pivot = j;
                                i_row_pivot = i;
                                found_pivot = found_pivot || (row_count < count);
                            }
                        }
                    }
                    if (search_count++ >= search_limit && merit_pivot < merit_limit) {
                        found_pivot = true;
                    }
                    if (found_pivot) {
                        break;
                    }
                }
            }

            // A row count cannot exceed the number of columns.
            if (count <= num_basic_) {
                // 1.3.2 Rows.
                for (Index i = row_link_first_[count]; i != -1; i = row_link_next_[i]) {
                    const Index start = mr_start_[i];
                    const Index end = start + mr_count_[i];
                    for (Index k = start; k < end; ++k) {
                        const Index j = mr_index_[k];
                        const Index column_count = mc_count_a_[j];
                        const double merit_local = 1.0 * (count - 1) * (column_count - 1);
                        if (merit_local < merit_pivot) {
                            Index ifind = mc_start_[j];
                            while (mc_index_[ifind] != i) {
                                ++ifind;
                            }
                            if (std::fabs(mc_value_[ifind]) >= mc_min_pivot_[j]) {
                                merit_pivot = merit_local;
                                j_col_pivot = j;
                                i_row_pivot = i;
                                found_pivot = found_pivot || (column_count <= count);
                            }
                        }
                    }
                    if (search_count++ >= search_limit && merit_pivot < merit_limit) {
                        found_pivot = true;
                    }
                    if (found_pivot) {
                        break;
                    }
                }
            }
        }

        // 1.4 Nothing found: singular.
        if (i_row_pivot < 0) {
            assert(j_col_pivot < 0);
            assert(!found_pivot);
            rank_deficiency_ = nwork_ + 1;
            return rank_deficiency_;
        }

        // 2. Eliminate the other entries by the pivot.
        // 2.1 Delete the pivot from both copies and from the count lists. One
        // active nonzero is lost.
        num_active_nz_--;
        const Real pivot_multiplier = colDelete(j_col_pivot, i_row_pivot);
        rowDelete(j_col_pivot, i_row_pivot);
        clinkDel(j_col_pivot);
        rlinkDel(i_row_pivot);
        if (std::fabs(pivot_multiplier) < pivot_tolerance_) {
            // Singular, but defer since other valid pivots may exist.
            num_defer_pivot++;
            if (mr_count_[i_row_pivot] == 0) {
                // A singleton row: the entry is zeroed and nothing more, since
                // the pivotal column may have other valid entries.
                clinkAdd(j_col_pivot, mc_count_a_[j_col_pivot]);
            } else {
                // Otherwise the column's other entries are smaller than the
                // pivot, so zero the column.
                zeroCol(j_col_pivot, pivot_multiplier);
                rlinkAdd(i_row_pivot, mr_count_[i_row_pivot]);
            }
            // No pivot found, so nwork_ goes back up.
            nwork_++;
            continue;
        }
        permute_[j_col_pivot] = i_row_pivot;

        // 2.2 The active pivot column (HFactor stores it to L).
        const Index start_a = mc_start_[j_col_pivot];
        const Index end_a = start_a + mc_count_a_[j_col_pivot];
        Index mwz_column_count = 0;
        for (Index k = start_a; k < end_a; ++k) {
            const Index i_row = mc_index_[k];
            const Real value = mc_value_[k] / pivot_multiplier;
            mwz_column_index_[mwz_column_count++] = i_row;
            mwz_column_array_[i_row] = value;
            mwz_column_mark_[i_row] = 1;
            mr_count_before_[i_row] = mr_count_[i_row];
            rowDelete(j_col_pivot, i_row);
        }
        // One active entry is lost for each entry in the pivotal column.
        num_active_nz_ -= (end_a - start_a);
        // 2.3 The non-active pivot column would go to U: not stored here.

        // 2.4 Loop over the pivot row to eliminate the other columns.
        const Index row_start = mr_start_[i_row_pivot];
        const Index row_end = row_start + mr_count_[i_row_pivot];
        for (Index row_k = row_start; row_k < row_end; ++row_k) {
            // 2.4.1 This column.
            const Index i_col = mr_index_[row_k];
            const Index my_count = mc_count_a_[i_col];
            const Index my_start = mc_start_[i_col];
            const Index my_end = my_start + my_count - 1;
            const Real my_pivot = colDelete(i_col, i_row_pivot);
            colStoreN(i_col, i_row_pivot, my_pivot);
            // One active entry is lost for each entry in the pivotal row.
            num_active_nz_--;

            // 2.4.2 Elimination on the overlapping part.
            Index n_fillin = mwz_column_count;
            Index n_cancel = 0;
            for (Index my_k = my_start; my_k < my_end; ++my_k) {
                const Index i_row = mc_index_[my_k];
                Real value = mc_value_[my_k];
                if (mwz_column_mark_[i_row] != 0) {
                    mwz_column_mark_[i_row] = 0;
                    n_fillin--;
                    value -= my_pivot * mwz_column_array_[i_row];
                    if (std::fabs(value) < kHighsTiny) {
                        value = 0;
                        n_cancel++;
                    }
                    mc_value_[my_k] = value;
                }
            }
            // One active entry is lost for each cancellation.
            num_active_nz_ -= n_cancel;

            // 2.4.3 Remove cancellation gaps.
            if (n_cancel > 0) {
                Index new_end = my_start;
                for (Index my_k = my_start; my_k < my_end; ++my_k) {
                    if (mc_value_[my_k] != 0) {
                        mc_index_[new_end] = mc_index_[my_k];
                        mc_value_[new_end++] = mc_value_[my_k];
                    } else {
                        rowDelete(i_col, mc_index_[my_k]);
                    }
                }
                mc_count_a_[i_col] = new_end - my_start;
            }

            // 2.4.4 Insert fill-in.
            if (n_fillin > 0) {
                // 2.4.4.1 Column size: p1&2 active, p3&4 non-active, p5 new
                // p1, p7 new p3.
                if (mc_count_a_[i_col] + mc_count_n_[i_col] + n_fillin > mc_space_[i_col]) {
                    const Index p1 = mc_start_[i_col];
                    const Index p2 = p1 + mc_count_a_[i_col];
                    const Index p3 = p1 + mc_space_[i_col] - mc_count_n_[i_col];
                    const Index p4 = p1 + mc_space_[i_col];
                    mc_space_[i_col] += std::max(mc_space_[i_col], n_fillin);
                    const auto p5 = static_cast<Index>(mc_index_.size());
                    mc_start_[i_col] = p5;
                    const Index p7 = p5 + mc_space_[i_col] - mc_count_n_[i_col];
                    resize_index(mc_index_, p5 + mc_space_[i_col]);
                    resize_real(mc_value_, p5 + mc_space_[i_col]);
                    std::copy(mc_index_.begin() + p1, mc_index_.begin() + p2,
                              mc_index_.begin() + p5);
                    std::copy(mc_value_.begin() + p1, mc_value_.begin() + p2,
                              mc_value_.begin() + p5);
                    std::copy(mc_index_.begin() + p3, mc_index_.begin() + p4,
                              mc_index_.begin() + p7);
                    std::copy(mc_value_.begin() + p3, mc_value_.begin() + p4,
                              mc_value_.begin() + p7);
                }

                // 2.4.4.2 Fill into the column copy. One active entry is
                // gained for each fill-in.
                for (Index i = 0; i < mwz_column_count; ++i) {
                    const Index i_row = mwz_column_index_[i];
                    if (mwz_column_mark_[i_row] != 0) {
                        colInsert(i_col, i_row, -my_pivot * mwz_column_array_[i_row]);
                        num_active_nz_++;
                    }
                }

                // 2.4.4.3 Fill into the row copy.
                for (Index i = 0; i < mwz_column_count; ++i) {
                    const Index i_row = mwz_column_index_[i];
                    if (mwz_column_mark_[i_row] != 0) {
                        // Expand the row space.
                        if (mr_count_[i_row] == mr_space_[i_row]) {
                            const Index p1 = mr_start_[i_row];
                            const Index p2 = p1 + mr_count_[i_row];
                            const auto p3 = static_cast<Index>(mr_index_.size());
                            mr_start_[i_row] = p3;
                            mr_space_[i_row] *= 2;
                            resize_index(mr_index_, p3 + mr_space_[i_row]);
                            std::copy(mr_index_.begin() + p1, mr_index_.begin() + p2,
                                      mr_index_.begin() + p3);
                        }
                        rowInsert(i_col, i_row);
                    }
                }
            }

            // 2.4.5 Reset the pivot column mark.
            for (Index i = 0; i < mwz_column_count; ++i) {
                mwz_column_mark_[mwz_column_index_[i]] = 1;
            }

            // 2.4.6 Fix the max value and the link list.
            colFixMax(i_col);
            if (my_count != mc_count_a_[i_col]) {
                clinkDel(i_col);
                clinkAdd(i_col, mc_count_a_[i_col]);
            }
        }

        // 2.5 Clear the pivot column buffer.
        for (Index i = 0; i < mwz_column_count; ++i) {
            mwz_column_mark_[mwz_column_index_[i]] = 0;
        }

        // 2.6 Correct the row links of the remaining active part.
        for (Index i = start_a; i < end_a; ++i) {
            const Index i_row = mc_index_[i];
            if (mr_count_before_[i_row] != mr_count_[i_row]) {
                rlinkDel(i_row);
                rlinkAdd(i_row, mr_count_[i_row]);
            }
        }
    }
    rank_deficiency_ = 0;
    return rank_deficiency_;
}

void RankRevealingLU::zeroCol(Index j_col, [[maybe_unused]] Real pivot_multiplier) {
    const Index a_count = mc_count_a_[j_col];
    const Index a_start = mc_start_[j_col];
    const Index a_end = a_start + a_count;
    for (Index i_el = a_start; i_el < a_end; ++i_el) {
        [[maybe_unused]] const Real abs_value = std::fabs(mc_value_[i_el]);
        const Index i_row = mc_index_[i_el];
        // Remove the column from this row and relink the row at its new
        // count.
        rowDelete(j_col, i_row);
        rlinkDel(i_row);
        rlinkAdd(i_row, mr_count_[i_row]);
        // Only used once |pivot_multiplier| < pivot_tolerance found the
        // column singular, so no entry exceeds the pivot.
        assert(abs_value <= std::fabs(pivot_multiplier));
    }
    clinkDel(j_col);
    mc_count_a_[j_col] = 0;
    mc_count_n_[j_col] = 0;
}

void RankRevealingLU::buildHandleRankDeficiency() {
    // iwork_ is now workspace: iwork_[row] is the column pivoted on it, or
    // -1, the -1's marking the rows with no pivot. On entry rank_deficiency_
    // is the columns'; an incomplete matrix adds the rows no column covers.
    if (num_basic_ < num_row_) {
        rank_deficiency_ += num_row_ - num_basic_;
    }
    row_with_no_pivot_.resize(rank_deficiency_);
    col_with_no_pivot_.resize(rank_deficiency_);
    Index lc_rank_deficiency = 0;
    if (num_basic_ < num_row_) {
        iwork_.resize(num_row_);
    } else if (num_basic_ > num_row_) {
        iwork_.resize(num_basic_);
    }
    for (Index i = 0; i < num_row_; ++i) {
        iwork_[i] = -1;
    }
    for (Index i = 0; i < num_basic_; ++i) {
        const Index perm_i = permute_[i];
        if (perm_i >= 0) {
            // HFactor stores basic_index[i]: the identity here.
            iwork_[perm_i] = i;
        } else {
            col_with_no_pivot_[lc_rank_deficiency++] = i;
        }
    }
    if (num_basic_ < num_row_) {
        // Complete permute_ and col_with_no_pivot_ with fictitious columns.
        permute_.resize(num_row_);
        for (Index i = num_basic_; i < num_row_; ++i) {
            col_with_no_pivot_[lc_rank_deficiency++] = i;
            permute_[i] = -1;
        }
    }
    assert(lc_rank_deficiency == rank_deficiency_);
    lc_rank_deficiency = 0;
    for (Index i = 0; i < num_row_; ++i) {
        if (iwork_[i] < 0) {
            // A row with no pivot, marked by the negation of one more than
            // its rank deficiency counter.
            row_with_no_pivot_[lc_rank_deficiency] = i;
            iwork_[i] = -(lc_rank_deficiency + 1);
            lc_rank_deficiency++;
        }
    }
    if (num_row_ < num_basic_) {
        // Fictitious rows for the excess columns, so that permute_ becomes a
        // permutation of all of them.
        for (Index i = num_row_; i < num_basic_; ++i) {
            row_with_no_pivot_[lc_rank_deficiency] = i;
            iwork_[i] = -(lc_rank_deficiency + 1);
            lc_rank_deficiency++;
        }
    }
    assert(lc_rank_deficiency == rank_deficiency_);
    // Complete the permutation with the rows with no pivot (HFactor also
    // puts a unit pivot into L and U for each true one).
    for (Index k = 0; k < rank_deficiency_; ++k) {
        const Index i_row = row_with_no_pivot_[k];
        const Index i_col = col_with_no_pivot_[k];
        assert(permute_[i_col] == -1);
        permute_[i_col] = i_row;
    }
}

void RankRevealingLU::buildMarkSingC() {
    // HFactor records each column with no pivot and replaces it in
    // basic_index by the logical of the row left to it. With basic_index the
    // identity, the column is its own variable; a fictitious column of an
    // incomplete matrix has none, -1.
    var_with_no_pivot_.resize(rank_deficiency_);
    for (Index k = 0; k < rank_deficiency_; ++k) {
        const Index asm_col = col_with_no_pivot_[k];
        if (asm_col < num_basic_) {
            var_with_no_pivot_[k] = asm_col;
        } else if (num_basic_ < num_row_) {
            var_with_no_pivot_[k] = -1;
        }
    }
}

}  // namespace mipx
