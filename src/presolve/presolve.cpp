// Parts of this file are translated from HiGHS 1.16; the rest is mipx's own.
// The translated parts:
//
// - Presolver::dualFixing, its single-equation dual fixing: the lambdas
//   checkColumn, computeActivity, handleSingleEquation and fixCols of
//   HPresolve::dualFixing in highs/presolve/HPresolveDualFixing.cpp.
// - Presolver::aggregateImpliedFreeColumns, its candidate order, pivot choice,
//   fill-in limit and give-up rule: HPresolve::aggregator in
//   highs/presolve/HPresolveAggregator.cpp. The implied-free test and the
//   substitution are mipx's own.
// - Presolver::removeDominatedColumns, the dominated and forcing column tests:
//   HPresolve::detectDominatedCol in highs/presolve/HPresolve.cpp.
//
// HiGHS's licence:
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

#include "mipx/presolve.h"

#include "../heuristics/common.h"
#include "compensated_sum.h"
#include "dependent_equations.h"
#include "dual_bounds.h"
#include "mipx/implication_graph.h"
#include "mipx/probing.h"
#include "mipx/variable_bounds.h"
#include "presolve_matrix.h"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace mipx {

namespace {

// A finite bound this large (an MPS file's 1e30 for "none") counts as
// infinite where activities are summed: added in, it swallows every smaller
// term, and taking it back out leaves 0 where the rest of the row was (#300).
constexpr Real kHugeBound = 1e20;

inline void markRowsTouchingCol(const PresolveMatrix& m, Index col,
                                const std::vector<bool>& row_removed,
                                std::vector<uint8_t>& dirty_rows) {
    for (const auto [row, a] : m.col(col)) {
        if (!row_removed[row]) {
            dirty_rows[row] = 1;
        }
    }
}

inline void markColsInRow(const PresolveMatrix& m, Index row, const std::vector<bool>& col_removed,
                          std::vector<uint8_t>& dirty_cols) {
    for (const auto [col, a] : m.row(row)) {
        if (!col_removed[col]) {
            dirty_cols[col] = 1;
        }
    }
}

inline uint64_t coeffBits(Real v) {
    static_assert(sizeof(uint64_t) == sizeof(Real));
    uint64_t bits = 0;
    std::memcpy(&bits, &v, sizeof(bits));
    return bits;
}

inline uint64_t mixHash(uint64_t h, uint64_t x) {
    // SplitMix64-style mix for sparse-pattern hashing.
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x ^= (x >> 31);
    return h ^ (x + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2));
}

inline uint64_t rowPatternHash(const PresolveMatrix& m, Index row,
                               const std::vector<bool>& col_removed) {
    uint64_t h = 1469598103934665603ULL;
    Index active = 0;
    for (const auto [col, a] : m.row(row)) {
        if (col_removed[col]) {
            continue;
        }
        h = mixHash(h, static_cast<uint64_t>(col));
        h = mixHash(h, coeffBits(a));
        ++active;
    }
    return mixHash(h, static_cast<uint64_t>(active));
}

inline uint64_t rowSupportHash(const PresolveMatrix& m, Index row,
                               const std::vector<bool>& col_removed) {
    uint64_t h = 1469598103934665603ULL;
    Index active = 0;
    for (const auto [col, a] : m.row(row)) {
        if (col_removed[col]) {
            continue;
        }
        h = mixHash(h, static_cast<uint64_t>(col));
        ++active;
    }
    return mixHash(h, static_cast<uint64_t>(active));
}

/// Steps `it` past entries of removed lines.
template <typename It>
void skipRemoved(It& it, const It& end, const std::vector<bool>& removed) {
    while (it != end && removed[(*it).index]) {
        ++it;
    }
}

inline bool rowsHaveSamePattern(const PresolveMatrix& m, Index r1, Index r2,
                                const std::vector<bool>& col_removed, Real tol) {
    const auto a = m.row(r1);
    const auto b = m.row(r2);
    auto i = a.begin();
    auto j = b.begin();
    while (true) {
        skipRemoved(i, a.end(), col_removed);
        skipRemoved(j, b.end(), col_removed);
        const bool done_i = (i == a.end());
        const bool done_j = (j == b.end());
        if (done_i || done_j) {
            return done_i && done_j;
        }
        const auto [col_i, ai] = *i;
        const auto [col_j, bj] = *j;
        if (col_i != col_j) {
            return false;
        }
        if (std::abs(ai - bj) > tol) {
            return false;
        }
        ++i;
        ++j;
    }
}

inline bool rowsHaveScaledPattern(const PresolveMatrix& m, Index r1, Index r2,
                                  const std::vector<bool>& col_removed, Real tol, Real& scale_out) {
    const auto a = m.row(r1);
    const auto b = m.row(r2);
    auto i = a.begin();
    auto j = b.begin();
    bool have_scale = false;
    Real scale = 0.0;
    while (true) {
        skipRemoved(i, a.end(), col_removed);
        skipRemoved(j, b.end(), col_removed);
        const bool done_i = (i == a.end());
        const bool done_j = (j == b.end());
        if (done_i || done_j) {
            if (done_i && done_j && have_scale) {
                scale_out = scale;
                return true;
            }
            return false;
        }
        const auto [col_i, ai] = *i;
        const auto [col_j, bj] = *j;
        if (col_i != col_j) {
            return false;
        }
        if (std::abs(ai) <= tol && std::abs(bj) <= tol) {
            ++i;
            ++j;
            continue;
        }
        if (std::abs(ai) <= tol || std::abs(bj) <= tol) {
            return false;
        }
        if (!have_scale) {
            scale = bj / ai;
            if (std::abs(scale) <= tol) {
                return false;
            }
            have_scale = true;
        } else {
            const Real expected = scale * ai;
            if (std::abs(expected - bj) >
                tol * (1.0 + std::max(std::abs(expected), std::abs(bj)))) {
                return false;
            }
        }
        ++i;
        ++j;
    }
}

inline bool rowIntervalSubsumes(Real keep_lb, Real keep_ub, Real rem_lb, Real rem_ub, Real tol) {
    // [keep_lb, keep_ub] subset [rem_lb, rem_ub] => row(rem) redundant.
    bool lb_ok = std::isinf(rem_lb) || keep_lb >= rem_lb - tol;
    bool ub_ok = std::isinf(rem_ub) || keep_ub <= rem_ub + tol;
    return lb_ok && ub_ok;
}

inline void removeActiveColumn(const PresolveMatrix& m, Index col, std::vector<bool>& col_removed,
                               const std::vector<bool>& row_removed,
                               std::vector<Index>& row_active_nnz,
                               std::vector<Index>& col_active_nnz,
                               std::vector<uint8_t>& next_dirty_rows,
                               std::vector<uint8_t>& next_dirty_cols) {
    if (col_removed[col]) {
        return;
    }
    col_removed[col] = true;
    col_active_nnz[col] = 0;
    next_dirty_cols[col] = 1;

    for (const auto [row, a] : m.col(col)) {
        if (row_removed[row]) {
            continue;
        }
        if (row_active_nnz[row] > 0) {
            --row_active_nnz[row];
        }
        next_dirty_rows[row] = 1;
    }
}

inline void removeActiveRow(const PresolveMatrix& m, Index row,
                            const std::vector<bool>& col_removed, std::vector<bool>& row_removed,
                            std::vector<Index>& row_active_nnz, std::vector<Index>& col_active_nnz,
                            std::vector<uint8_t>& next_dirty_rows,
                            std::vector<uint8_t>& next_dirty_cols) {
    if (row_removed[row]) {
        return;
    }
    row_removed[row] = true;
    row_active_nnz[row] = 0;
    next_dirty_rows[row] = 1;

    for (const auto [col, a] : m.row(row)) {
        if (col_removed[col]) {
            continue;
        }
        if (col_active_nnz[col] > 0) {
            --col_active_nnz[col];
        }
        next_dirty_cols[col] = 1;
    }
}

/// Rounding tolerance for an integer column's bound implied by a row through
/// coefficient a. The row holds a * x to within kRowTol, which leaves x free
/// to within kRowTol / |a|, and the division scales the rhs and residual
/// rounding up by the same factor: with a = 1e-7 a rhs error of 1e-15 put a
/// bound 3.5e-8 above an integer, and rounding at 1e-8 moved it a whole unit
/// and emptied the column (#235). Rounding a candidate at this scale before
/// it is compared can only keep more integer points. A row that is dropped
/// afterwards must still hold at the rounded bound to within kRowTol, so it
/// rounds at no more than kRowTol / |a|; a row that stays in the model also
/// gets the propagator's 1e-6, for residuals inexact at that size.
Real integerRoundTol(Real a, bool row_dropped) {
    constexpr Real kRowTol = 1e-8;
    constexpr Real kIntRoundTol = 1e-6;
    return std::max(row_dropped ? kRowTol : kIntRoundTol, kRowTol / std::abs(a));
}

// x_col for a zero-cost singleton slack (#359): the x in [col_lower,
// col_upper] with row_lower <= a x + R <= row_upper, R the row's other terms
// as postsolved. A bound where one fits, so the point stays a vertex, as
// HiGHS's undo picks one; else the row side that fits. A reduced point that
// left R a hair outside the widened sides has no such x: the bound nearest
// the row's range then misses the row by that hair and no more.
Real zeroCostSingletonValue(const PostsolveZeroCostSingleton& op, const std::vector<Real>& full) {
    CompensatedSum rest;
    for (std::size_t k = 0; k < op.row_cols.size(); ++k) {
        rest.add(op.row_coeffs[k] * full[op.row_cols[k]]);
    }
    const Real a = op.coeff;
    // a x in [row_lower - R, row_upper - R], both finite.
    const Real from_lower = -rest.without(op.row_lower) / a;
    const Real from_upper = -rest.without(op.row_upper) / a;
    const Real x_lo = a > 0.0 ? from_lower : from_upper;
    const Real x_hi = a > 0.0 ? from_upper : from_lower;
    const Real l = op.col_lower;
    const Real u = op.col_upper;
    const bool l_finite = std::abs(l) < kHugeBound;
    const bool u_finite = std::abs(u) < kHugeBound;
    if (l_finite && x_hi < l) {
        return l;
    }
    if (u_finite && x_lo > u) {
        return u;
    }
    if (l_finite && x_lo <= l) {
        return l;
    }
    if (u_finite && x_hi >= u) {
        return u;
    }
    return x_lo;  // Inside (l, u): the row at its lower side over a x.
}

// x_col for a forcing column (#383), as HiGHS's ForcingColumn::undo: from
// the other bound, the nearest value in the column's free direction that
// puts every removed row at or past its side, R the row's other terms as
// postsolved. The row the value comes from is then at its side.
Real forcingColValue(const PostsolveForcingCol& op, const std::vector<Real>& full) {
    bool limited = std::abs(op.other_bound) < kHugeBound;
    Real x = op.other_bound;
    for (const auto& row : op.rows) {
        if (std::abs(row.rhs) >= kHugeBound) {
            continue;  // No limit from an infinite side.
        }
        CompensatedSum rest;
        for (std::size_t k = 0; k < row.cols.size(); ++k) {
            rest.add(row.coeffs[k] * full[row.cols[k]]);
        }
        const Real from_row = -rest.without(row.rhs) / row.coeff;
        x = !limited               ? from_row
            : op.at_infinite_upper ? std::max(x, from_row)
                                   : std::min(x, from_row);
        limited = true;
    }
    // No finite limit and no finite other bound: every value holds the rows.
    // A limit is kept whatever its size, as a tiny coefficient may need it.
    return limited ? x : 0.0;
}

}  // namespace

// =============================================================================
// PostsolveStack
// =============================================================================

void PostsolveStack::push(PostsolveOp op) {
    ops_.push_back(std::move(op));
}

void PostsolveStack::clear() {
    ops_.clear();
}

std::vector<Real> PostsolveStack::postsolve(const std::vector<Real>& presolved_solution,
                                            const std::vector<Index>& col_mapping,
                                            Index orig_num_cols) const {
    constexpr Real kTol = 1e-8;

    // Start with a full-size solution initialized to zero.
    std::vector<Real> full(orig_num_cols, 0.0);

    // Copy presolved solution values into their original positions.
    for (Index j = 0; j < static_cast<Index>(presolved_solution.size()); ++j) {
        if (j < static_cast<Index>(col_mapping.size())) {
            full[col_mapping[j]] = presolved_solution[j];
        }
    }

    // Apply postsolve operations in reverse order.
    for (auto it = ops_.rbegin(); it != ops_.rend(); ++it) {
        std::visit(
            [&](const auto& op) {
                using T = std::decay_t<decltype(op)>;

                if constexpr (std::is_same_v<T, PostsolveFixVariable>) {
                    full[op.orig_col] = op.value;
                } else if constexpr (std::is_same_v<T, PostsolveSingletonRow>) {
                    // Nothing to do: the row was redundant after bound tightening.
                } else if constexpr (std::is_same_v<T, PostsolveSingletonCol>) {
                    // The variable was substituted out of its single constraint.
                    // We need to compute its value from the constraint.
                    // The constraint is: row_lower <= a * x_col + (rest) <= row_upper.
                    // rest = sum of other variables in the row (already set).
                    // We don't store the other coefficients, so we set the variable
                    // to satisfy the constraint as tightly as possible.
                    //
                    // For a singleton column, x_col only appeared in one row, so
                    // its value is determined by the constraint and bounds.
                    // Set it to be within bounds and satisfy the constraint.
                    //
                    // Since we removed the variable and adjusted the row to account
                    // for it, we need to back-compute. The simplest correct approach:
                    // set it to 0 if feasible, otherwise to the nearest bound.
                    // But actually, the presolver adjusted the row bounds when removing
                    // the column, so the constraint is satisfied by the remaining vars.
                    // We just need to set this variable's value to satisfy its bounds
                    // and any objective contribution.
                    //
                    // For correctness: the value was set based on objective and bounds
                    // during presolve. We'll use the value that was implicitly chosen.
                    // Actually, singleton column removal fixes the variable at a bound.
                    // Let's use the value stored during presolve.
                    // (The presolve sets the var to a specific value and stores it.)
                    // The value is already set by PostsolveFixVariable that accompanies this.
                    // This op mainly records the row/col relationship for reference.
                    // But if we need to compute from scratch:
                    Real val = full[op.orig_col];
                    // Clamp to bounds.
                    val = std::max(val, op.col_lower);
                    val = std::min(val, op.col_upper);
                    full[op.orig_col] = val;
                } else if constexpr (std::is_same_v<T, PostsolveForcingRow>) {
                    for (const auto& fv : op.fixed_vars) {
                        full[fv.orig_col] = fv.value;
                    }
                } else if constexpr (std::is_same_v<T, PostsolveDominatedRow>) {
                    // Nothing to do: the row was redundant.
                } else if constexpr (std::is_same_v<T, PostsolveCoeffTightening>) {
                    // Nothing to do for variable values.
                    // The solution is still valid.
                } else if constexpr (std::is_same_v<T, PostsolveDoubletonEquality>) {
                    const Real denom = op.a_eliminated;
                    if (std::abs(denom) <= kTol) {
                        return;
                    }
                    const Real x_keep = full[op.kept_col];
                    Real x_elim = (op.rhs - op.a_kept * x_keep) / denom;
                    if (!std::isinf(op.eliminated_lower)) {
                        x_elim = std::max(x_elim, op.eliminated_lower);
                    }
                    if (!std::isinf(op.eliminated_upper)) {
                        x_elim = std::min(x_elim, op.eliminated_upper);
                    }
                    full[op.eliminated_col] = x_elim;
                } else if constexpr (std::is_same_v<T, PostsolveParallelCol>) {
                    // y = x_kept + scale * x_removed: pick x_removed within
                    // its bounds that leaves x_kept within its own.
                    const Real y = full[op.kept_col];
                    const Real s = op.scale;
                    Real lo = op.removed_lower;
                    Real hi = op.removed_upper;
                    // x_kept = y - s * x_removed in [kept_lower, kept_upper].
                    const Real from_upper = (y - op.kept_upper) / s;
                    const Real from_lower = (y - op.kept_lower) / s;
                    if (s > 0.0) {
                        lo = std::max(lo, from_upper);
                        hi = std::min(hi, from_lower);
                    } else {
                        lo = std::max(lo, from_lower);
                        hi = std::min(hi, from_upper);
                    }
                    // A finite end, so the point is a vertex where it can be;
                    // rounding may cross lo and hi by a hair, lo then wins.
                    // A huge bound (1e30 from MPS) counts as infinite: at it,
                    // y - s * x loses y to rounding.
                    auto usable = [](Real v) { return std::abs(v) < kHugeBound; };
                    Real x = std::max(std::min(0.0, hi), lo);
                    if (usable(lo)) {
                        x = lo;
                    } else if (usable(hi)) {
                        x = hi;
                    }
                    x = std::max(std::min(x, op.removed_upper), op.removed_lower);
                    full[op.removed_col] = x;
                    full[op.kept_col] = y - s * x;
                } else if constexpr (std::is_same_v<T, PostsolveFreeColSubstitution>) {
                    // The row held at rhs; x_col's bounds were implied, so no clamp.
                    CompensatedSum rest;
                    for (std::size_t k = 0; k < op.row_cols.size(); ++k) {
                        rest.add(op.row_coeffs[k] * full[op.row_cols[k]]);
                    }
                    // rest.without(rhs) is rest - rhs, rounded once.
                    full[op.orig_col] = -rest.without(op.rhs) / op.coeff;
                } else if constexpr (std::is_same_v<T, PostsolveZeroCostSingleton>) {
                    full[op.orig_col] = zeroCostSingletonValue(op, full);
                } else if constexpr (std::is_same_v<T, PostsolveForcingCol>) {
                    full[op.orig_col] = forcingColValue(op, full);
                }
            },
            *it);
    }

    return full;
}

// =============================================================================
// Presolver — individual reductions
// =============================================================================

Index Presolver::removeFixedVariables(LpProblem& lp, std::vector<bool>& col_removed,
                                      std::vector<bool>& row_removed,
                                      std::vector<Index>& row_active_nnz,
                                      std::vector<Index>& col_active_nnz,
                                      const std::vector<Index>& dirty_cols,
                                      std::vector<uint8_t>& next_dirty_rows,
                                      std::vector<uint8_t>& next_dirty_cols) {
    Index changes = 0;

    for (Index j : dirty_cols) {
        if (col_removed[j]) {
            continue;
        }
        ++stats_.cols_examined;
        const Real width = lp.col_upper[j] - lp.col_lower[j];
        if (std::abs(width) > kTol) {
            continue;
        }

        // Pinning x_j at its lower bound displaces it by up to `width`, and the
        // loop below moves that displacement into every live row's bounds,
        // scaled by a_kj. A width test against a bare kTol bounds the first
        // perturbation but not the second: x0 in [1, 1 + 8e-9] with a
        // coefficient of 1e3 shifts that row's rhs by 8e-6, enough to turn a
        // row satisfiable only near x0's upper bound into an infeasibility
        // (issue #215). This is the fixed-column twin of the forcing-row
        // defect fixed in #210, and gets the same treatment: bound the
        // displacement the reduction authorizes, folding in the largest
        // coefficient over live rows. The factor is clamped at 1, so this only
        // rejects columns the plain test accepted. An exact point (width 0)
        // always passes.
        //
        // Crossed bounds (lb > ub, by at most kTol) are fixed regardless:
        // left in the model they make it infeasible outright, which is worse
        // than any displacement fixing them introduces.
        if (width > 0.0) {
            Real amplification = 1.0;
            for (const auto [row, a] : work_->col(j)) {
                if (row_removed[row]) {
                    continue;
                }
                amplification = std::max(amplification, std::abs(a));
            }
            if (width * amplification > kTol) {
                continue;
            }
        }

        // Variable is fixed.
        Real value = lp.col_lower[j];
        removeActiveColumn(*work_, j, col_removed, row_removed, row_active_nnz, col_active_nnz,
                           next_dirty_rows, next_dirty_cols);
        ++changes;
        ++stats_.vars_removed;

        // Adjust constraint bounds: for each row containing this variable,
        // subtract the fixed contribution.
        for (const auto [row, a] : work_->col(j)) {
            if (row_removed[row]) {
                continue;
            }
            shiftRowSides(lp, row, a * value);
            next_dirty_rows[row] = 1;
        }

        // Adjust objective offset.
        lp.obj_offset += lp.obj[j] * value;

        postsolve_stack_.push(PostsolveFixVariable{j, value});
    }

    return changes;
}

Index Presolver::removeSingletonRows(LpProblem& lp, std::vector<bool>& col_removed,
                                     std::vector<bool>& row_removed,
                                     std::vector<Index>& row_active_nnz,
                                     std::vector<Index>& col_active_nnz,
                                     const std::vector<Index>& dirty_rows,
                                     std::vector<uint8_t>& next_dirty_rows,
                                     std::vector<uint8_t>& next_dirty_cols) {
    Index changes = 0;

    for (Index i : dirty_rows) {
        if (row_removed[i]) {
            continue;
        }
        ++stats_.rows_examined;

        // Fast path from maintained active nonzero counts.
        Index count = row_active_nnz[i];
        Index singleton_col = -1;
        Real singleton_coeff = 0.0;

        if (count == 0) {
            // Empty row. Infeasible past its slack; dropped only when it
            // holds within the feasibility tolerance, since postsolve never
            // checks it again. In between it stays, and the LP decides (#236).
            const Real excess = std::max(lp.row_lower[i], -lp.row_upper[i]);
            if (excess > rowTol(i)) {
                infeasible_ = true;
                return changes;
            }
            if (excess > std::min(rowTol(i), kDropTol)) {
                continue;
            }
            removeActiveRow(*work_, i, col_removed, row_removed, row_active_nnz, col_active_nnz,
                            next_dirty_rows, next_dirty_cols);
            ++changes;
            ++stats_.rows_removed;
            postsolve_stack_.push(PostsolveDominatedRow{i});
            continue;
        }

        if (count != 1) {
            continue;
        }

        for (const auto [col, a] : work_->row(i)) {
            if (!col_removed[col]) {
                singleton_col = col;
                singleton_coeff = a;
                break;
            }
        }

        // Singleton row: row_lower <= a * x_j <= row_upper.
        // This implies bounds on x_j.
        Real new_lb = -kInf;
        Real new_ub = kInf;

        if (singleton_coeff > 0) {
            if (!std::isinf(lp.row_lower[i])) {
                new_lb = lp.row_lower[i] / singleton_coeff;
            }
            if (!std::isinf(lp.row_upper[i])) {
                new_ub = lp.row_upper[i] / singleton_coeff;
            }
        } else {
            // Negative coefficient flips bounds.
            if (!std::isinf(lp.row_upper[i])) {
                new_lb = lp.row_upper[i] / singleton_coeff;
            }
            if (!std::isinf(lp.row_lower[i])) {
                new_ub = lp.row_lower[i] / singleton_coeff;
            }
        }

        // The gap guard below reads the bounds as the row implies them.
        const Real implied_lb = new_lb;
        const Real implied_ub = new_ub;
        if (lp.col_type[singleton_col] != VarType::Continuous) {
            const Real round_tol = integerRoundTol(singleton_coeff, true);
            new_lb = std::ceil(new_lb - round_tol);
            new_ub = std::floor(new_ub + round_tol);
        }

        bool tightened = false;
        if (new_lb > lp.col_lower[singleton_col] + kTol) {
            lp.col_lower[singleton_col] = new_lb;
            tightened = true;
            ++stats_.bounds_tightened;
            markRowsTouchingCol(*work_, singleton_col, row_removed, next_dirty_rows);
            next_dirty_cols[singleton_col] = 1;
        }
        if (new_ub < lp.col_upper[singleton_col] - kTol) {
            lp.col_upper[singleton_col] = new_ub;
            tightened = true;
            ++stats_.bounds_tightened;
            markRowsTouchingCol(*work_, singleton_col, row_removed, next_dirty_rows);
            next_dirty_cols[singleton_col] = 1;
        }

        // Integer rounding.
        if (lp.col_type[singleton_col] != VarType::Continuous) {
            Real rounded_lb = std::ceil(lp.col_lower[singleton_col] - kTol);
            Real rounded_ub = std::floor(lp.col_upper[singleton_col] + kTol);
            if (rounded_lb > lp.col_lower[singleton_col] + kTol) {
                lp.col_lower[singleton_col] = rounded_lb;
                tightened = true;
                markRowsTouchingCol(*work_, singleton_col, row_removed, next_dirty_rows);
                next_dirty_cols[singleton_col] = 1;
            }
            if (rounded_ub < lp.col_upper[singleton_col] - kTol) {
                lp.col_upper[singleton_col] = rounded_ub;
                tightened = true;
                markRowsTouchingCol(*work_, singleton_col, row_removed, next_dirty_rows);
                next_dirty_cols[singleton_col] = 1;
            }
        }

        // Dropping the row leaves the column's bounds as its only record, but a
        // tightening of at most kTol in column space was not applied above, and
        // the row can still exclude a gap of up to |a| * kTol of activity: with
        // a = 1e6, x in [0, 1] and row 1e6 * x >= 5e-3, the implied bound 5e-9
        // is skipped and postsolve may return x = 0, missing the row entirely.
        // Keep such a row; it is the same displacement-versus-gap distinction
        // as the fixed-column guard (issue #215).
        {
            const Real a_abs = std::abs(singleton_coeff);
            const bool lb_gap = implied_lb > lp.col_lower[singleton_col] &&
                                a_abs * (implied_lb - lp.col_lower[singleton_col]) > kTol;
            const bool ub_gap = implied_ub < lp.col_upper[singleton_col] &&
                                a_abs * (lp.col_upper[singleton_col] - implied_ub) > kTol;
            if (lb_gap || ub_gap) {
                if (lp.col_lower[singleton_col] > lp.col_upper[singleton_col] + kTol) {
                    infeasible_ = true;
                    return changes;
                }
                if (tightened) {
                    ++changes;
                }
                continue;
            }
        }

        // The row-implied range can contradict the column's own bounds, or the
        // range another singleton row on the same column already imposed, and
        // integer rounding can empty a fractional range outright. This row is
        // about to be discarded, and a later pass in the same round may remove
        // the column before the end-of-round crossed-bound scan runs -- that
        // scan only looks at columns still present -- so the contradiction has
        // to be reported here or it is lost entirely.
        if (lp.col_lower[singleton_col] > lp.col_upper[singleton_col] + kTol) {
            infeasible_ = true;
            return changes;
        }

        removeActiveRow(*work_, i, col_removed, row_removed, row_active_nnz, col_active_nnz,
                        next_dirty_rows, next_dirty_cols);
        ++changes;
        ++stats_.rows_removed;
        if (tightened) {
            ++changes;
        }

        postsolve_stack_.push(PostsolveSingletonRow{i});
    }

    return changes;
}

Index Presolver::removeSingletonCols(LpProblem& lp, std::vector<bool>& col_removed,
                                     std::vector<bool>& row_removed,
                                     std::vector<Index>& row_active_nnz,
                                     std::vector<Index>& col_active_nnz,
                                     const std::vector<Index>& dirty_cols,
                                     std::vector<uint8_t>& next_dirty_rows,
                                     std::vector<uint8_t>& next_dirty_cols) {
    Index changes = 0;

    for (Index j : dirty_cols) {
        if (col_removed[j]) {
            continue;
        }
        ++stats_.cols_examined;

        // Fixing a column with crossed bounds at one of them would bury the
        // contradiction in the postsolve stack, so report it instead -- the
        // same guard removeEmptyColumns already applies.
        if (lp.col_lower[j] > lp.col_upper[j] + kTol) {
            infeasible_ = true;
            return changes;
        }

        Index count = col_active_nnz[j];
        Index singleton_row = -1;
        Real singleton_coeff = 0.0;

        if (count != 1) {
            continue;
        }
        for (const auto [row, a] : work_->col(j)) {
            if (!row_removed[row]) {
                singleton_row = row;
                singleton_coeff = a;
                break;
            }
        }

        // Singleton column: variable appears in only one active constraint.
        // We only eliminate by bound fixing when we can prove objective-optimality
        // and preserve feasibility for remaining row variables.
        Real obj = lp.obj[j];

        bool lb_finite = !std::isinf(lp.col_lower[j]);
        bool ub_finite = !std::isinf(lp.col_upper[j]);

        if (!lb_finite && !ub_finite) {
            continue;  // Free variable — skip.
        }

        // Count remaining variables in the row (excluding this one).
        Index remaining_in_row = 0;
        for (const auto [col, a] : work_->row(singleton_row)) {
            if (col != j && !col_removed[col]) {
                ++remaining_in_row;
            }
        }

        auto objective_prefers_lower = [&]() -> bool {
            if (lp.sense == Sense::Minimize) {
                return obj > kTol;
            }
            return obj < -kTol;
        };
        auto objective_prefers_upper = [&]() -> bool {
            if (lp.sense == Sense::Minimize) {
                return obj < -kTol;
            }
            return obj > kTol;
        };

        bool fix_at_lower = false;
        bool has_fix = false;

        if (remaining_in_row > 0) {
            const bool has_row_lower = !std::isinf(lp.row_lower[singleton_row]);
            const bool has_row_upper = !std::isinf(lp.row_upper[singleton_row]);

            // Ranged/equality rows cannot be safely relaxed by a one-sided fix.
            if (has_row_lower && has_row_upper) {
                continue;
            }

            bool required_lower = false;
            if (has_row_upper) {
                // <= row: only the activity-minimizing fix is safe for remaining vars.
                required_lower = singleton_coeff > 0.0;
            } else if (has_row_lower) {
                // >= row: only the activity-maximizing fix is safe.
                required_lower = singleton_coeff < 0.0;
            } else {
                // No active row bounds; choose purely by objective.
                if (objective_prefers_lower()) {
                    required_lower = true;
                } else if (objective_prefers_upper()) {
                    required_lower = false;
                } else {
                    required_lower = lb_finite;
                }
            }

            // Respect objective direction when it is nonzero.
            if (std::abs(obj) > kTol) {
                bool objective_lower = objective_prefers_lower();
                if (objective_lower != required_lower) {
                    continue;
                }
            }

            if (required_lower) {
                if (!lb_finite) {
                    continue;
                }
                fix_at_lower = true;
            } else {
                if (!ub_finite) {
                    continue;
                }
                fix_at_lower = false;
            }
            has_fix = true;
        } else {
            // Row contains only this variable. The objective picks which bound
            // to take, but the row still restricts it: earlier reductions fold
            // the values of the columns they remove into the row bounds, so
            // what is left of the row is a real constraint on this variable.
            if (objective_prefers_lower()) {
                if (!lb_finite) {
                    continue;
                }
                fix_at_lower = true;
            } else if (objective_prefers_upper()) {
                if (!ub_finite) {
                    continue;
                }
                fix_at_lower = false;
            } else if (lb_finite) {
                fix_at_lower = true;
            } else if (ub_finite) {
                fix_at_lower = false;
            } else {
                continue;
            }
            has_fix = true;
        }
        if (!has_fix) {
            continue;
        }

        const Real fix_value = fix_at_lower ? lp.col_lower[j] : lp.col_upper[j];

        if (remaining_in_row == 0) {
            // Only this variable is left in the row, so the row reduces to a
            // bound on it. Take the fix only when the chosen value satisfies
            // that bound. Singleton-row tightening narrows the column to the
            // row-implied range on a later pass, after which the same fix is
            // valid -- so skipping here costs the reduction nothing.
            const Real activity = singleton_coeff * fix_value;
            if (!std::isinf(lp.row_lower[singleton_row]) &&
                activity < lp.row_lower[singleton_row] - kTol) {
                continue;
            }
            if (!std::isinf(lp.row_upper[singleton_row]) &&
                activity > lp.row_upper[singleton_row] + kTol) {
                continue;
            }
        }

        removeActiveColumn(*work_, j, col_removed, row_removed, row_active_nnz, col_active_nnz,
                           next_dirty_rows, next_dirty_cols);
        ++changes;
        ++stats_.vars_removed;

        // Adjust row bounds.
        shiftRowSides(lp, singleton_row, singleton_coeff * fix_value);

        // Adjust objective.
        lp.obj_offset += obj * fix_value;

        postsolve_stack_.push(PostsolveSingletonCol{
            j, singleton_row, singleton_coeff, obj, lp.row_lower[singleton_row],
            lp.row_upper[singleton_row], lp.col_type[j], lp.col_lower[j], lp.col_upper[j]});
        postsolve_stack_.push(PostsolveFixVariable{j, fix_value});
    }

    return changes;
}

// A continuous column x_j whose one row i keeps it inside its bounds for every
// value the row's other terms R can take is implied free: x_j = (t - R) / a
// over the row then holds the row at t and x_j in its bounds, so the row and
// the column both go and c_j x_j moves onto the other columns (HiGHS's free
// column substitution, #324). t is the row's side when it is an equation;
// on an inequality it is the side x_j's cost pushes the row to, since the
// bounds never stop x_j short of it, or any finite side at zero cost.
Index Presolver::substituteFreeColSingletons(LpProblem& lp, std::vector<bool>& col_removed,
                                             std::vector<bool>& row_removed,
                                             std::vector<Index>& row_active_nnz,
                                             std::vector<Index>& col_active_nnz,
                                             const std::vector<Index>& dirty_cols,
                                             std::vector<uint8_t>& next_dirty_rows,
                                             std::vector<uint8_t>& next_dirty_cols) {
    // A pivot below this fraction of the row's largest coefficient would
    // scale the row's rounding up into x_j and the transferred costs.
    constexpr Real kPivotRatio = 1e-2;
    // How far x_j's implied range may reach past a bound and still count as
    // within it. Absolute: postsolve does not clamp, so this is this pass's
    // share of how far x_j may come out beyond its bound, and the reduced
    // model gains that sliver of the row (relative to a bound of 1000 it was
    // 1e-5). The other columns' own errors still reach x_j, times a_k / a.
    constexpr Real kFreeTol = 1e-9;
    Index changes = 0;

    for (Index j : dirty_cols) {
        if (col_removed[j] || col_active_nnz[j] != 1 || lp.col_type[j] != VarType::Continuous) {
            continue;
        }
        ++stats_.cols_examined;

        Index i = -1;
        Real a = 0.0;
        for (const auto [row, value] : work_->col(j)) {
            if (!row_removed[row]) {
                i = row;
                a = value;
                break;
            }
        }
        if (i < 0 || a == 0.0 || !std::isfinite(a)) {
            continue;
        }
        const bool has_lower = std::abs(lp.row_lower[i]) < kHugeBound;
        const bool has_upper = std::abs(lp.row_upper[i]) < kHugeBound;
        // Crossed sides make the model infeasible, which holding the row at
        // one side and dropping it would hide; later passes report it.
        if ((!has_lower && !has_upper) ||
            (has_lower && has_upper && lp.row_lower[i] > lp.row_upper[i])) {
            continue;
        }

        // Activity range of the row's other terms, huge bounds infinite.
        CompensatedSum r_min;
        CompensatedSum r_max;
        Index inf_min = 0;
        Index inf_max = 0;
        Real max_abs = std::abs(a);
        std::vector<Index> row_cols;
        std::vector<Real> row_coeffs;
        for (const auto [col, ak] : work_->row(i)) {
            if (col == j || col_removed[col]) {
                continue;
            }
            if (ak == 0.0) {
                continue;
            }
            row_cols.push_back(col);
            row_coeffs.push_back(ak);
            max_abs = std::max(max_abs, std::abs(ak));
            const Real at_min = ak > 0.0 ? lp.col_lower[col] : lp.col_upper[col];
            const Real at_max = ak > 0.0 ? lp.col_upper[col] : lp.col_lower[col];
            if (std::abs(at_min) >= kHugeBound) {
                ++inf_min;
            } else {
                r_min.add(ak * at_min);
            }
            if (std::abs(at_max) >= kHugeBound) {
                ++inf_max;
            } else {
                r_max.add(ak * at_max);
            }
        }
        if (std::abs(a) < kPivotRatio * max_abs) {
            continue;
        }

        // x_j's range from the row: a x_j in [L - R_max, U - R_min].
        const Real ax_lo = has_lower && inf_max == 0 ? lp.row_lower[i] - r_max.value() : -kInf;
        const Real ax_hi = has_upper && inf_min == 0 ? lp.row_upper[i] - r_min.value() : kInf;
        const Real implied_lo = a > 0.0 ? ax_lo / a : ax_hi / a;
        const Real implied_hi = a > 0.0 ? ax_hi / a : ax_lo / a;
        const Real lj = lp.col_lower[j];
        const Real uj = lp.col_upper[j];
        const bool lower_implied =
            lj <= -kHugeBound || (std::isfinite(implied_lo) && implied_lo >= lj - kFreeTol);
        const bool upper_implied =
            uj >= kHugeBound || (std::isfinite(implied_hi) && implied_hi <= uj + kFreeTol);
        if (!lower_implied || !upper_implied) {
            continue;
        }

        // The side the row is held at.
        const Real cj = lp.obj[j];
        Real t = 0.0;
        if (has_lower && has_upper && lp.row_lower[i] == lp.row_upper[i]) {
            t = lp.row_upper[i];
        } else {
            const Real d = lp.sense == Sense::Minimize ? cj : -cj;
            bool to_lower = has_lower;
            if (d != 0.0) {
                // x_j goes down when d > 0, which takes a x_j + R down when a > 0.
                to_lower = (d > 0.0) == (a > 0.0);
            }
            if (to_lower ? !has_lower : !has_upper) {
                continue;  // Unbounded in x_j's direction; leave it to the solver.
            }
            t = to_lower ? lp.row_lower[i] : lp.row_upper[i];
        }

        if (cj != 0.0) {
            lp.obj_offset += cj * t / a;
            for (std::size_t k = 0; k < row_cols.size(); ++k) {
                lp.obj[row_cols[k]] -= cj * row_coeffs[k] / a;
            }
            lp.obj[j] = 0.0;
        }

        removeActiveColumn(*work_, j, col_removed, row_removed, row_active_nnz, col_active_nnz,
                           next_dirty_rows, next_dirty_cols);
        removeActiveRow(*work_, i, col_removed, row_removed, row_active_nnz, col_active_nnz,
                        next_dirty_rows, next_dirty_cols);

        ++stats_.vars_removed;
        ++stats_.rows_removed;
        ++changes;
        postsolve_stack_.push(
            PostsolveFreeColSubstitution{j, i, a, t, std::move(row_cols), std::move(row_coeffs)});
    }

    return changes;
}

// HiGHS's zero-cost singleton (HPresolve::zeroCostSingleton, #359): a
// continuous column x_j with cost 0 and one active row i, lo <= a x_j + R <= up
// with both sides finite (an equation included), is the row's slack. Over R
// alone the row is lo - max(a l, a u) <= R <= up - min(a l, a u): every R in
// that range has an x_j in [l, u] that puts the row back in [lo, up], and no
// R outside it has one. x_j goes and the row keeps those sides; an infinite
// bound of x_j makes its side infinite. As x_j costs nothing, which x_j
// postsolve picks does not change the objective. A one-sided row is left to
// dual fixing, as in HiGHS.
Index Presolver::removeZeroCostSingletons(LpProblem& lp, std::vector<bool>& col_removed,
                                          std::vector<bool>& row_removed,
                                          std::vector<Index>& row_active_nnz,
                                          std::vector<Index>& col_active_nnz,
                                          const std::vector<Index>& dirty_cols,
                                          std::vector<uint8_t>& next_dirty_rows,
                                          std::vector<uint8_t>& next_dirty_cols) {
    Index changes = 0;

    for (Index j : dirty_cols) {
        if (col_removed[j] || col_active_nnz[j] != 1 || lp.col_type[j] != VarType::Continuous ||
            lp.obj[j] != 0.0) {
            continue;
        }
        ++stats_.cols_examined;
        const Real l = lp.col_lower[j];
        const Real u = lp.col_upper[j];
        if (l > u) {
            continue;  // Crossed bounds: removeSingletonCols reports them.
        }

        Index i = -1;
        Real a = 0.0;
        for (const auto [row, value] : work_->col(j)) {
            if (!row_removed[row]) {
                i = row;
                a = value;
                break;
            }
        }
        if (i < 0 || a == 0.0 || !std::isfinite(a)) {
            continue;
        }
        const Real lo = lp.row_lower[i];
        const Real up = lp.row_upper[i];
        if (std::abs(lo) >= kHugeBound || std::abs(up) >= kHugeBound || lo > up) {
            continue;  // One-sided, free, or crossed (infeasible, reported elsewhere).
        }

        // a x_j over [l, u], a huge bound infinite.
        const bool l_finite = std::abs(l) < kHugeBound;
        const bool u_finite = std::abs(u) < kHugeBound;
        const Real at_l = a * l;
        const Real at_u = a * u;
        const bool max_finite = a > 0.0 ? u_finite : l_finite;
        const bool min_finite = a > 0.0 ? l_finite : u_finite;
        const Real ax_max = a > 0.0 ? at_u : at_l;
        const Real ax_min = a > 0.0 ? at_l : at_u;
        // A lower side at or below -kHugeBound, or an upper side at or above
        // it, is infinite, as the passes that test sides with std::isinf
        // take it. A side huge the other way (R >= 1e20, or <= -1e20) is a
        // row no point of sane size meets: the column is left to the passes
        // that report that, rather than the side dropped or kept at 1e20.
        const bool lo_kept = max_finite && lo - ax_max > -kHugeBound;
        const bool up_kept = min_finite && up - ax_min < kHugeBound;
        if ((lo_kept && lo - ax_max >= kHugeBound) || (up_kept && up - ax_min <= -kHugeBound)) {
            continue;
        }
        const Real new_lo = lo_kept ? lo - ax_max : -kInf;
        const Real new_up = up_kept ? up - ax_min : kInf;

        std::vector<Index> row_cols;
        std::vector<Real> row_coeffs;
        for (const auto [col, ak] : work_->row(i)) {
            if (col == j || col_removed[col] || ak == 0.0) {
                continue;
            }
            row_cols.push_back(col);
            row_coeffs.push_back(ak);
        }

        removeActiveColumn(*work_, j, col_removed, row_removed, row_active_nnz, col_active_nnz,
                           next_dirty_rows, next_dirty_cols);
        lp.row_lower[i] = new_lo;
        lp.row_upper[i] = new_up;
        // The sides kept finite took a x_j's extremes in, as shiftRowSides
        // folds a term: the row's tolerance scales with them.
        if (lo_kept) {
            row_scale_[i] = std::max(row_scale_[i], std::abs(ax_max));
        }
        if (up_kept) {
            row_scale_[i] = std::max(row_scale_[i], std::abs(ax_min));
        }
        next_dirty_rows[i] = 1;
        // A side gone infinite leaves the row one-sided, which singleton
        // columns and dual fixing take: the row's columns are looked at again.
        for (const Index col : row_cols) {
            next_dirty_cols[col] = 1;
        }
        ++stats_.vars_removed;
        ++changes;
        postsolve_stack_.push(PostsolveZeroCostSingleton{
            j, i, a, lo, up, new_lo, new_up, l, u, std::move(row_cols), std::move(row_coeffs)});
    }

    return changes;
}

Index Presolver::aggregateDoubletonEqualities(LpProblem& lp, std::vector<bool>& col_removed,
                                              std::vector<bool>& row_removed,
                                              std::vector<Index>& row_active_nnz,
                                              std::vector<Index>& col_active_nnz,
                                              const std::vector<Index>& dirty_rows,
                                              std::vector<uint8_t>& next_dirty_rows,
                                              std::vector<uint8_t>& next_dirty_cols) {
    Index changes = 0;

    // An entry, explicit zero included; none means substituting would fill in.
    auto findRowCoeff = [&](Index row, Index col, Real& coeff) -> bool {
        if (!work_->contains(row, col)) {
            return false;
        }
        coeff = work_->coeff(row, col);
        return true;
    };

    for (Index i : dirty_rows) {
        if (row_removed[i]) {
            continue;
        }
        ++stats_.rows_examined;
        if (row_active_nnz[i] != 2) {
            continue;
        }
        if (std::isinf(lp.row_lower[i]) || std::isinf(lp.row_upper[i])) {
            continue;
        }
        if (std::abs(lp.row_upper[i] - lp.row_lower[i]) > kTol) {
            continue;
        }

        std::array<Index, 2> row_cols = {-1, -1};
        std::array<Real, 2> row_coeffs = {0.0, 0.0};
        Index n_active = 0;
        for (const auto [col, a] : work_->row(i)) {
            if (col_removed[col]) {
                continue;
            }
            if (std::abs(a) <= kTol) {
                continue;
            }
            if (n_active >= 2) {
                n_active = 3;
                break;
            }
            row_cols[n_active] = col;
            row_coeffs[n_active] = a;
            ++n_active;
        }
        if (n_active != 2) {
            continue;
        }

        const Index c0 = row_cols[0];
        const Index c1 = row_cols[1];
        const Real a0 = row_coeffs[0];
        const Real a1 = row_coeffs[1];
        if (lp.col_type[c0] != VarType::Continuous || lp.col_type[c1] != VarType::Continuous) {
            continue;
        }

        auto candidateFeasible = [&](Index elim_col, Index keep_col, Real a_elim, Real a_keep,
                                     Index& score) -> bool {
            if (std::abs(a_elim) <= kTol || std::abs(a_keep) <= kTol) {
                return false;
            }
            score = col_active_nnz[elim_col];
            const Real ratio = a_keep / a_elim;
            // The eliminated column's bounds live on as bounds of the kept
            // one, projected through 1/ratio. A ratio too small to project
            // through would drop them, and postsolve would then clamp the
            // eliminated value off its equality (#244).
            const bool elim_bounded =
                !std::isinf(lp.col_lower[elim_col]) || !std::isinf(lp.col_upper[elim_col]);
            if (elim_bounded && std::abs(ratio) <= kTol) {
                return false;
            }

            for (const auto [row, a_row_elim] : work_->col(elim_col)) {
                if (row_removed[row] || row == i) {
                    continue;
                }

                Real a_row_keep = 0.0;
                if (!findRowCoeff(row, keep_col, a_row_keep)) {
                    if (!options_.enable_doubleton_fill_in) {
                        return false;  // Would create fill-in.
                    }
                    // x_k enters this row as -a_row_elim * ratio, held to the
                    // rule a cancelling entry is held to below.
                    if (a_row_elim != 0.0 && std::abs(a_row_elim * ratio) <= 10.0 * kTol) {
                        return false;
                    }
                    continue;
                }
                // A zero or tiny x_k entry is as good as none: with fill-in it
                // is held to the rule on new_keep below, as an absent one is.
                if (std::abs(a_row_keep) <= kTol && !options_.enable_doubleton_fill_in) {
                    return false;
                }
                const Real new_keep = a_row_keep - a_row_elim * ratio;
                if (std::abs(new_keep) <= 10.0 * kTol) {
                    return false;
                }
            }
            return true;
        };

        bool have_choice = false;
        Index best_elim = -1;
        Index best_keep = -1;
        Real best_a_elim = 0.0;
        Real best_a_keep = 0.0;
        Index best_score = 0;

        for (int pass = 0; pass < 2; ++pass) {
            const Index elim_col = (pass == 0) ? c0 : c1;
            const Index keep_col = (pass == 0) ? c1 : c0;
            const Real a_elim = (pass == 0) ? a0 : a1;
            const Real a_keep = (pass == 0) ? a1 : a0;
            Index score = 0;
            if (!candidateFeasible(elim_col, keep_col, a_elim, a_keep, score)) {
                continue;
            }
            if (!have_choice || score < best_score) {
                have_choice = true;
                best_elim = elim_col;
                best_keep = keep_col;
                best_a_elim = a_elim;
                best_a_keep = a_keep;
                best_score = score;
            }
        }
        if (!have_choice) {
            continue;
        }

        const Real rhs = lp.row_upper[i];
        const Real alpha = rhs / best_a_elim;
        const Real ratio = best_a_keep / best_a_elim;
        const Real beta = -ratio;
        if (!std::isfinite(alpha) || !std::isfinite(beta) || !std::isfinite(ratio)) {
            continue;
        }
        const Real elim_lb = lp.col_lower[best_elim];
        const Real elim_ub = lp.col_upper[best_elim];

        // With fill-in, a row of x_e without x_k gets x_k: at most |col_e| - 1
        // inserts against the |col_e| entries of x_e and the x_k of row i
        // that go, so one substitution shrinks the active matrix by at least
        // two entries and needs no cap on what it adds. (The working
        // matrix's pool grows by the inserts: removal is only logical.)
        struct RowUpdate {
            Index row = -1;
            Real shift = 0.0;
            Real new_keep = 0.0;
            bool write = true;    // false: x_e's entry is an explicit zero
            bool insert = false;  // x_k is not yet an entry of this row
        };
        std::vector<RowUpdate> updates;
        bool valid_updates = true;

        // Substitute eliminated variable into objective: x_e = alpha + beta*x_k.
        for (const auto [row, a_row_elim] : work_->col(best_elim)) {
            if (row_removed[row] || row == i) {
                continue;
            }

            const Real shift = a_row_elim * alpha;
            if (!std::isinf(lp.row_lower[row]) && !std::isfinite(shift)) {
                valid_updates = false;
                break;
            }
            if (!std::isinf(lp.row_upper[row]) && !std::isfinite(shift)) {
                valid_updates = false;
                break;
            }

            Real a_row_keep = 0.0;
            const bool has_keep = findRowCoeff(row, best_keep, a_row_keep);
            if (!has_keep && !options_.enable_doubleton_fill_in) {
                valid_updates = false;
                break;
            }
            const Real new_keep = a_row_keep - a_row_elim * ratio;
            if (!std::isfinite(new_keep)) {
                valid_updates = false;
                break;
            }
            const bool write = has_keep || a_row_elim != 0.0;
            updates.push_back({row, shift, new_keep, write, !has_keep && write});
        }
        if (!valid_updates) {
            continue;
        }

        lp.obj_offset += lp.obj[best_elim] * alpha;
        lp.obj[best_keep] += lp.obj[best_elim] * beta;
        lp.obj[best_elim] = 0.0;
        next_dirty_cols[best_keep] = 1;

        for (const auto& update : updates) {
            shiftRowSides(lp, update.row, update.shift);
            // Exact, however small the change: on a row of small coefficients
            // a change below kTol is still the whole substitution (#244:
            // 1.875e-6 became 1.883e-6).
            if (update.write) {
                // Safe to insert: the walk of col(best_elim) is over.
                work_->set(update.row, best_keep, update.new_keep);
            }
            if (update.insert) {
                ++row_active_nnz[update.row];
                ++col_active_nnz[best_keep];
                ++stats_.doubleton_fill_in;
            }
            next_dirty_rows[update.row] = 1;
            markColsInRow(*work_, update.row, col_removed, next_dirty_cols);
        }

        // Project eliminated variable bounds into kept variable bounds. Any
        // tighter projection is taken, however little: skipping one within
        // kTol let the reduced LP put x_k where x_e = alpha + beta x_k left
        // its bounds, and postsolve's clamp then moved x_e off the row; on
        // the #257 model x1 <= 1 was skipped against x1 <= 1 + 5e-9, and the
        // row missed by 5e-5 (#325).
        bool lower_projected = false;
        bool upper_projected = false;
        auto tightenKeptBound = [&](Real bound, bool eliminated_lower) {
            if (std::isinf(bound) || std::abs(beta) <= kTol) {
                return;
            }
            const Real implied = (bound - alpha) / beta;
            // x_e >= bound bounds x_k from below when beta > 0.
            if ((beta > 0.0) == eliminated_lower) {
                if (implied > lp.col_lower[best_keep]) {
                    lp.col_lower[best_keep] = implied;
                    lower_projected = true;
                    ++stats_.bounds_tightened;
                }
            } else if (implied < lp.col_upper[best_keep]) {
                lp.col_upper[best_keep] = implied;
                upper_projected = true;
                ++stats_.bounds_tightened;
            }
        };
        tightenKeptBound(elim_lb, true);
        tightenKeptBound(elim_ub, false);

        if (lp.col_lower[best_keep] > lp.col_upper[best_keep] + kTol) {
            infeasible_ = true;
            return changes;
        }
        // Bounds crossed within kTol fix x_k, uncrossed so the reduced LP does
        // not read them as infeasible. The projected side wins: it keeps x_e
        // within its bounds, and so on the row, while x_k misses its own
        // bound by at most kTol. Its own side would put x_e off by up to
        // kTol |beta| (5e-5 on the #257 model with x1 >= 1 + 5e-9).
        if (lp.col_lower[best_keep] > lp.col_upper[best_keep]) {
            if (upper_projected && !lower_projected) {
                lp.col_lower[best_keep] = lp.col_upper[best_keep];
            } else {
                lp.col_upper[best_keep] = lp.col_lower[best_keep];
            }
        }
        if (lower_projected || upper_projected) {
            markRowsTouchingCol(*work_, best_keep, row_removed, next_dirty_rows);
            next_dirty_cols[best_keep] = 1;
        }

        removeActiveColumn(*work_, best_elim, col_removed, row_removed, row_active_nnz,
                           col_active_nnz, next_dirty_rows, next_dirty_cols);
        removeActiveRow(*work_, i, col_removed, row_removed, row_active_nnz, col_active_nnz,
                        next_dirty_rows, next_dirty_cols);

        ++stats_.vars_removed;
        ++stats_.rows_removed;
        ++changes;

        postsolve_stack_.push(PostsolveDoubletonEquality{best_elim, best_keep, best_a_elim,
                                                         best_a_keep, rhs, elim_lb, elim_ub});
    }

    return changes;
}

// HiGHS 1.16.0's aggregator (HPresolve::aggregator, HPresolveAggregator.cpp);
// the rules taken, and where they differ, are written on #348. An equation
// a x_j + sum_k a_k x_k = t with x_j continuous and implied free gives x_j =
// (t - sum_k a_k x_k) / a; every other row r of x_j gets -(a_rj / a) times
// the equation added (filling in where r lacks an x_k) and its sides moved
// by a_rj t / a, the objective takes x_j's cost over the same way, and the
// equation and x_j go. Postsolve takes x_j back from the equation.
//
// Implied free: each finite bound of x_j follows from some active row's
// activity over its other columns, row i included. That stays sound with
// row i gone: postsolve puts x_j on row i, each changed row r holds exactly
// when the original r does, so the rows that implied x_j's bounds still hold.
Index Presolver::aggregateImpliedFreeColumns(LpProblem& lp, std::vector<bool>& col_removed,
                                             std::vector<bool>& row_removed,
                                             std::vector<Index>& row_active_nnz,
                                             std::vector<Index>& col_active_nnz,
                                             std::vector<uint8_t>& next_dirty_rows,
                                             std::vector<uint8_t>& next_dirty_cols) {
    // HiGHS's presolve_pivot_threshold and presolve_substitution_maxfillin,
    // and the fill refusals in a row that end its pass.
    constexpr Real kPivotThreshold = 1e-2;
    constexpr Index kMaxFillIn = 10;
    constexpr Index kMaxFillFailures = 3;
    // How far an implied bound may reach past x_j's own and still count as
    // within it, as substituteFreeColSingletons holds it.
    constexpr Real kFreeTol = 1e-9;
    // A changed or inserted entry must come out above this, as a doubleton
    // substitution's must (#343); HiGHS drops one at or below 1e-9 instead.
    constexpr Real kMinEntry = 10.0 * kTol;
    Index changes = 0;

    // Each row's activity over its active columns, huge bounds infinite,
    // summed on first use and again once a substitution changes the row.
    // Column bounds do not change in this pass.
    struct Activity {
        CompensatedSum min;
        CompensatedSum max;
        Index min_inf = 0;
        Index max_inf = 0;
        bool valid = false;
    };
    std::vector<Activity> activity(lp.num_rows);
    auto rowActivity = [&](Index r) -> const Activity& {
        Activity& act = activity[r];
        if (!act.valid) {
            act = Activity{};
            for (const auto [col, a] : work_->row(r)) {
                if (col_removed[col] || a == 0.0) {
                    continue;
                }
                const Real at_min = a > 0.0 ? lp.col_lower[col] : lp.col_upper[col];
                const Real at_max = a > 0.0 ? lp.col_upper[col] : lp.col_lower[col];
                if (std::abs(at_min) >= kHugeBound) {
                    ++act.min_inf;
                } else {
                    act.min.add(a * at_min);
                }
                if (std::abs(at_max) >= kHugeBound) {
                    ++act.max_inf;
                } else {
                    act.max.add(a * at_max);
                }
            }
            act.valid = true;
        }
        return act;
    };

    auto impliedFree = [&](Index j) -> bool {
        const Real lj = lp.col_lower[j];
        const Real uj = lp.col_upper[j];
        bool lower_implied = lj <= -kHugeBound;
        bool upper_implied = uj >= kHugeBound;
        for (const auto [r, a] : work_->col(j)) {
            if (lower_implied && upper_implied) {
                break;
            }
            if (row_removed[r] || a == 0.0) {
                continue;
            }
            const Activity& act = rowActivity(r);
            // The row's other terms: x_j's own taken back out.
            const Real j_at_min = a > 0.0 ? lj : uj;
            const Real j_at_max = a > 0.0 ? uj : lj;
            Index inf_min = act.min_inf;
            Index inf_max = act.max_inf;
            Real r_min = 0.0;
            Real r_max = 0.0;
            if (std::abs(j_at_min) >= kHugeBound) {
                --inf_min;
                r_min = act.min.value();
            } else {
                r_min = act.min.without(a * j_at_min);
            }
            if (std::abs(j_at_max) >= kHugeBound) {
                --inf_max;
                r_max = act.max.value();
            } else {
                r_max = act.max.without(a * j_at_max);
            }
            // a x_j in [L - R_max, U - R_min].
            const bool has_lower = std::abs(lp.row_lower[r]) < kHugeBound;
            const bool has_upper = std::abs(lp.row_upper[r]) < kHugeBound;
            const Real ax_lo = has_lower && inf_max == 0 ? lp.row_lower[r] - r_max : -kInf;
            const Real ax_hi = has_upper && inf_min == 0 ? lp.row_upper[r] - r_min : kInf;
            const Real lo = a > 0.0 ? ax_lo / a : ax_hi / a;
            const Real hi = a > 0.0 ? ax_hi / a : ax_lo / a;
            if (std::isfinite(lo) && lo >= lj - kFreeTol) {
                lower_implied = true;
            }
            if (std::isfinite(hi) && hi <= uj + kFreeTol) {
                upper_implied = true;
            }
        }
        return lower_implied && upper_implied;
    };

    // Held at its upper side, as a doubleton equation is.
    auto isEquation = [&](Index i) {
        return !row_removed[i] && std::abs(lp.row_lower[i]) < kHugeBound &&
               std::abs(lp.row_upper[i]) < kHugeBound &&
               std::abs(lp.row_upper[i] - lp.row_lower[i]) <= kTol;
    };

    struct Candidate {
        Index row;
        Index col;
        uint64_t hash;
    };
    std::vector<Candidate> candidates;
    std::vector<int8_t> free_status(lp.num_cols, -1);
    for (Index i = 0; i < lp.num_rows; ++i) {
        if (!isEquation(i)) {
            continue;
        }
        for (const auto [j, a] : work_->row(i)) {
            if (col_removed[j] || std::abs(a) <= kTol || lp.col_type[j] != VarType::Continuous) {
                continue;
            }
            if (free_status[j] < 0) {
                free_status[j] = impliedFree(j) ? 1 : 0;
            }
            if (free_status[j] != 0) {
                const uint64_t h =
                    mixHash(mixHash(0, static_cast<uint64_t>(i)), static_cast<uint64_t>(j));
                candidates.push_back({i, j, h});
            }
        }
    }

    // HiGHS's order: a row or column of length 2 first, then the smallest
    // product of the lengths, then the smallest of the two; ties by hash.
    std::sort(candidates.begin(), candidates.end(), [&](const Candidate& c1, const Candidate& c2) {
        const Index min1 = std::min(row_active_nnz[c1.row], col_active_nnz[c1.col]);
        const Index min2 = std::min(row_active_nnz[c2.row], col_active_nnz[c2.col]);
        if ((min1 == 2) != (min2 == 2)) {
            return min1 == 2;
        }
        const int64_t prod1 = int64_t{row_active_nnz[c1.row]} * col_active_nnz[c1.col];
        const int64_t prod2 = int64_t{row_active_nnz[c2.row]} * col_active_nnz[c2.col];
        if (prod1 != prod2) {
            return prod1 < prod2;
        }
        if (min1 != min2) {
            return min1 < min2;
        }
        return std::tie(c1.hash, c1.row, c1.col) < std::tie(c2.hash, c2.row, c2.col);
    });

    struct Update {
        Index row;
        Index col;
        Real value;
        bool insert;
    };
    std::vector<Index> row_cols;
    std::vector<Real> row_coeffs;
    std::vector<Index> other_rows;
    std::vector<Real> other_coeffs;
    std::vector<Real> shifts;
    std::vector<Update> updates;
    Index fill_failures = 0;

    for (const Candidate& cand : candidates) {
        const Index i = cand.row;
        const Index j = cand.col;
        if (!isEquation(i) || col_removed[j] || !work_->contains(i, j)) {
            continue;
        }
        const Real a = work_->coeff(i, j);
        if (std::abs(a) <= kTol || !std::isfinite(a) || !impliedFree(j)) {
            continue;
        }
        const Index row_size = row_active_nnz[i];
        const Index col_size = col_active_nnz[j];

        row_cols.clear();
        row_coeffs.clear();
        Real row_max = std::abs(a);
        for (const auto [k, ak] : work_->row(i)) {
            if (k == j || col_removed[k] || ak == 0.0) {
                continue;
            }
            row_cols.push_back(k);
            row_coeffs.push_back(ak);
            row_max = std::max(row_max, std::abs(ak));
        }
        other_rows.clear();
        other_coeffs.clear();
        Real col_max = std::abs(a);
        for (const auto [r, ar] : work_->col(j)) {
            if (r == i || row_removed[r]) {
                continue;
            }
            col_max = std::max(col_max, std::abs(ar));
            // An explicit zero needs no row added; x_j's removal drops it.
            if (ar != 0.0) {
                other_rows.push_back(r);
                other_coeffs.push_back(ar);
            }
        }

        // A row or column of length 2 cannot fill in badly: always taken.
        if (row_size != 2 && col_size != 2) {
            // The pivot against the shorter line's largest entry, and failing
            // that against the longer one's.
            const bool row_shorter = row_size < col_size;
            if (std::abs(a) < kPivotThreshold * (row_shorter ? row_max : col_max) &&
                std::abs(a) < kPivotThreshold * (row_shorter ? col_max : row_max)) {
                continue;
            }
            Index fill = -(row_size + col_size - 1);
            for (const Index r : other_rows) {
                for (const Index k : row_cols) {
                    if (!work_->contains(r, k)) {
                        ++fill;
                    }
                }
                if (fill > kMaxFillIn) {
                    break;
                }
            }
            if (fill > kMaxFillIn) {
                // Too much fill again and again: the rows have grown too
                // dense for substitutions.
                if (++fill_failures == kMaxFillFailures) {
                    break;
                }
                continue;
            }
        }

        const Real t = lp.row_upper[i];
        shifts.clear();
        updates.clear();
        bool valid = true;
        for (std::size_t m = 0; m < other_rows.size() && valid; ++m) {
            const Index r = other_rows[m];
            const Real scale = -other_coeffs[m] / a;
            const Real shift = other_coeffs[m] * t / a;
            if (!std::isfinite(scale) || !std::isfinite(shift)) {
                valid = false;
                break;
            }
            shifts.push_back(shift);
            for (std::size_t n = 0; n < row_cols.size(); ++n) {
                const Index k = row_cols[n];
                const bool has = work_->contains(r, k);
                const Real value = (has ? work_->coeff(r, k) : 0.0) + scale * row_coeffs[n];
                if (!std::isfinite(value) || std::abs(value) <= kMinEntry) {
                    valid = false;
                    break;
                }
                updates.push_back({r, k, value, !has});
            }
        }
        if (!valid) {
            continue;
        }
        // Only a substitution ends a run of fill refusals (as in HiGHS, where
        // nothing refuses a pair after its fill test).
        if (row_size != 2 && col_size != 2) {
            fill_failures = 0;
        }

        const Real cj = lp.obj[j];
        if (cj != 0.0) {
            lp.obj_offset += cj * t / a;
            for (std::size_t n = 0; n < row_cols.size(); ++n) {
                lp.obj[row_cols[n]] -= cj * row_coeffs[n] / a;
            }
            lp.obj[j] = 0.0;
        }
        for (std::size_t m = 0; m < other_rows.size(); ++m) {
            shiftRowSides(lp, other_rows[m], shifts[m]);
        }
        // Safe to insert: no walk of these rows or columns is open.
        for (const Update& u : updates) {
            work_->set(u.row, u.col, u.value);
            if (u.insert) {
                ++row_active_nnz[u.row];
                ++col_active_nnz[u.col];
                ++stats_.aggregator_fill_in;
            }
        }
        for (const Index r : other_rows) {
            activity[r].valid = false;
            next_dirty_rows[r] = 1;
            markColsInRow(*work_, r, col_removed, next_dirty_cols);
        }

        removeActiveColumn(*work_, j, col_removed, row_removed, row_active_nnz, col_active_nnz,
                           next_dirty_rows, next_dirty_cols);
        removeActiveRow(*work_, i, col_removed, row_removed, row_active_nnz, col_active_nnz,
                        next_dirty_rows, next_dirty_cols);

        ++stats_.vars_removed;
        ++stats_.rows_removed;
        ++changes;
        postsolve_stack_.push(
            PostsolveFreeColSubstitution{j, i, a, t, std::move(row_cols), std::move(row_coeffs)});
        row_cols.clear();
        row_coeffs.clear();
    }

    return changes;
}

void Presolver::shiftRowSides(LpProblem& lp, Index row, Real shift) {
    if (!std::isinf(lp.row_lower[row])) {
        lp.row_lower[row] -= shift;
    }
    if (!std::isinf(lp.row_upper[row])) {
        lp.row_upper[row] -= shift;
    }
    row_scale_[row] = std::max(row_scale_[row], std::abs(shift));
}

Index Presolver::removeForcingRows(LpProblem& lp, std::vector<bool>& col_removed,
                                   std::vector<bool>& row_removed,
                                   std::vector<Index>& row_active_nnz,
                                   std::vector<Index>& col_active_nnz,
                                   const std::vector<Index>& dirty_rows,
                                   std::vector<uint8_t>& next_dirty_rows,
                                   std::vector<uint8_t>& next_dirty_cols) {
    Index changes = 0;

    for (Index i : dirty_rows) {
        if (row_removed[i]) {
            continue;
        }
        ++stats_.rows_examined;

        const auto rv = work_->row(i);

        // Compute min/max activity for this row.
        Real act_min = 0.0;
        Real act_max = 0.0;
        bool has_inf_min = false;
        bool has_inf_max = false;
        // The row's magnitude: its sides, the terms folded into them, and
        // its largest finite term.
        Real scale = row_scale_[i];

        for (const auto [j, a] : rv) {
            if (col_removed[j]) {
                continue;
            }

            Real lb = lp.col_lower[j];
            Real ub = lp.col_upper[j];

            if (!std::isinf(lb)) {
                scale = std::max(scale, std::abs(a * lb));
            }
            if (!std::isinf(ub)) {
                scale = std::max(scale, std::abs(a * ub));
            }
            if (a > 0) {
                if (std::isinf(lb)) {
                    has_inf_min = true;
                } else {
                    act_min += a * lb;
                }
                if (std::isinf(ub)) {
                    has_inf_max = true;
                } else {
                    act_max += a * ub;
                }
            } else {
                if (std::isinf(ub)) {
                    has_inf_min = true;
                } else {
                    act_min += a * ub;
                }
                if (std::isinf(lb)) {
                    has_inf_max = true;
                } else {
                    act_max += a * lb;
                }
            }
        }

        // Infeasible only past kTol relative to the row's magnitude, as in
        // DomainPropagator: with terms near 1e9 the activity sums round by
        // more than an absolute kTol, and a model whose only point lies
        // exactly on the row read as infeasible (#236).
        const Real row_tol = kTol * scale;
        if (!has_inf_min && !std::isinf(lp.row_upper[i]) && act_min > lp.row_upper[i] + row_tol) {
            infeasible_ = true;
            return changes;
        }
        if (!has_inf_max && !std::isinf(lp.row_lower[i]) && act_max < lp.row_lower[i] - row_tol) {
            infeasible_ = true;
            return changes;
        }

        // A row is forcing when one of its extreme activities meets the opposite
        // row bound: every variable is then pinned at the bound that attains
        // that activity. near_* is that test at the plain primal tolerance --
        // the shape the code had before issue #210.
        const bool near_force_to_min =
            !has_inf_min && !std::isinf(lp.row_upper[i]) &&
            std::abs(act_min - lp.row_upper[i]) <= kTol &&
            (std::isinf(lp.row_lower[i]) || act_min >= lp.row_lower[i] - kTol);
        const bool near_force_to_max =
            !has_inf_max && !std::isinf(lp.row_lower[i]) &&
            std::abs(act_max - lp.row_lower[i]) <= kTol &&
            (std::isinf(lp.row_upper[i]) || act_max <= lp.row_upper[i] + kTol);

        if (!near_force_to_min && !near_force_to_max) {
            continue;
        }

        // That test compares in *activity* units, but what it licenses is a
        // fixing in *variable* units, and the two are related by 1/|a_ij|. With
        // a gap g between the extreme activity and the row bound, variable j can
        // still move d_j = g/|a_ij| off the bound that attains that activity and
        // keep the row satisfied. Pinning it there perturbs the model twice
        // over: by d_j in j's own value, and by |a_kj| * d_j in the right-hand
        // side of every other row k that contains j, since the fixing loop below
        // shifts those bounds by a_kj * fix_value. Both perturbations have to
        // stay inside the primal tolerance, which bounds the gap at
        //
        //     g <= kTol * |a_ij| / max(1, max_k |a_kj|)   for every j.
        //
        // Testing g against a bare kTol was the defect behind Netlib bore3d
        // (issue #210): activity bound tightening had shrunk a column's upper
        // bound to 1.7e-8 without reaching its true value of 0, leaving an
        // equality row with rhs 0 at min activity -9.8e-11 -- inside kTol, so
        // "forcing". Fixing that column at 1.7e-8 instead of 0 shifted a second
        // equality row's rhs to -1.7e-8, over kTol, and presolve reported the
        // instance infeasible. The gap was within tolerance; the displacement it
        // authorized, amplified by a coefficient of 5.7e-3, was not.
        //
        // The ratio is clamped at 1 so forcing_tol <= kTol: this only ever
        // rejects rows the plain kTol test accepted, never the reverse. It is
        // computed here rather than with the activities because it needs a
        // column scan, and only rows that already look forcing reach this point.
        //
        // The bound covers the columns the fixing loop below actually pins,
        // i.e. those with |a_ij| > kTol. A column whose coefficient in this row
        // is at or below kTol is not pinned, as its displacement g/|a_ij| is
        // not small, and removing the row would then lose the constraint it
        // places on that column (#244: x0 >= 37500 through a coefficient of
        // -5e-9, every term of the row that small). Such a row stays.
        bool unpinned_column = false;
        for (const auto [j, a] : rv) {
            if (!col_removed[j] && a != 0.0 && std::abs(a) <= kTol &&
                lp.col_lower[j] != lp.col_upper[j]) {
                unpinned_column = true;
                break;
            }
        }
        if (unpinned_column) {
            continue;
        }
        Real forcing_tol = kTol;
        for (const auto [j, a] : rv) {
            if (col_removed[j]) {
                continue;
            }
            if (std::abs(a) <= kTol) {
                continue;
            }
            // Largest coefficient this column carries in any other live row --
            // the factor by which its displacement reaches that row's bounds.
            Real amplification = 1.0;
            for (const auto [other_row, other_a] : work_->col(j)) {
                if (other_row == i || row_removed[other_row]) {
                    continue;
                }
                amplification = std::max(amplification, std::abs(other_a));
            }
            forcing_tol = std::min(forcing_tol, kTol * std::abs(a) / amplification);
        }

        bool force_to_min = near_force_to_min && std::abs(act_min - lp.row_upper[i]) <= forcing_tol;
        bool force_to_max = near_force_to_max && std::abs(act_max - lp.row_lower[i]) <= forcing_tol;

        if (near_force_to_min && near_force_to_max) {
            bool consistent = true;
            for (const auto [j, a] : rv) {
                if (col_removed[j]) {
                    continue;
                }
                if (std::abs(a) <= kTol) {
                    continue;
                }
                const Real min_fix = (a > 0.0) ? lp.col_lower[j] : lp.col_upper[j];
                const Real max_fix = (a > 0.0) ? lp.col_upper[j] : lp.col_lower[j];
                if (std::isinf(min_fix) || std::isinf(max_fix) ||
                    std::abs(min_fix - max_fix) > kTol) {
                    consistent = false;
                    break;
                }
            }
            if (!consistent) {
                continue;
            }
            // The row looks forcing from both sides, so the two directions
            // disagree about which bound each variable is pinned at. `consistent`
            // only bounds that disagreement by kTol, which is exactly the
            // displacement the tolerance above exists to rule out -- so resolve
            // it the way this code always has, to the min side, and take the
            // reduction only when the min side passes on its own. Silently
            // falling through to the max side would fix every variable at the
            // opposite bound, up to kTol away from what was chosen before.
            if (!force_to_min) {
                continue;
            }
            force_to_max = false;
        }

        if (!force_to_min && !force_to_max) {
            continue;
        }

        std::vector<std::pair<Index, Real>> fixings;
        fixings.reserve(static_cast<size_t>(row_active_nnz[i]));
        bool can_fix_all = true;
        for (const auto [j, a] : rv) {
            if (col_removed[j]) {
                continue;
            }
            if (std::abs(a) <= kTol) {
                continue;
            }

            const Real fix_value = force_to_min ? ((a > 0.0) ? lp.col_lower[j] : lp.col_upper[j])
                                                : ((a > 0.0) ? lp.col_upper[j] : lp.col_lower[j]);
            if (std::isinf(fix_value)) {
                can_fix_all = false;
                break;
            }
            fixings.emplace_back(j, fix_value);
        }
        if (!can_fix_all) {
            continue;
        }

        PostsolveForcingRow postsolve_op;
        postsolve_op.orig_row = i;

        for (const auto& [j, fix_value] : fixings) {
            removeActiveColumn(*work_, j, col_removed, row_removed, row_active_nnz, col_active_nnz,
                               next_dirty_rows, next_dirty_cols);
            ++stats_.vars_removed;
            lp.obj_offset += lp.obj[j] * fix_value;

            for (const auto [other_row, a] : work_->col(j)) {
                if (row_removed[other_row] || other_row == i) {
                    continue;
                }
                shiftRowSides(lp, other_row, a * fix_value);
                next_dirty_rows[other_row] = 1;
            }

            postsolve_op.fixed_vars.push_back({j, fix_value});
            postsolve_stack_.push(PostsolveFixVariable{j, fix_value});
        }

        removeActiveRow(*work_, i, col_removed, row_removed, row_active_nnz, col_active_nnz,
                        next_dirty_rows, next_dirty_cols);
        ++stats_.rows_removed;
        ++changes;
        postsolve_stack_.push(std::move(postsolve_op));
    }

    return changes;
}

Index Presolver::removeDominatedRows(LpProblem& lp, std::vector<bool>& col_removed,
                                     std::vector<bool>& row_removed,
                                     std::vector<Index>& row_active_nnz,
                                     std::vector<Index>& col_active_nnz,
                                     const std::vector<Index>& dirty_rows,
                                     std::vector<uint8_t>& next_dirty_rows,
                                     std::vector<uint8_t>& next_dirty_cols) {
    Index changes = 0;

    for (Index i : dirty_rows) {
        if (row_removed[i]) {
            continue;
        }
        ++stats_.rows_examined;

        const auto rv = work_->row(i);

        // Compute activity bounds.
        Real act_min = 0.0;
        Real act_max = 0.0;
        bool has_inf_min = false;
        bool has_inf_max = false;

        for (const auto [j, a] : rv) {
            if (col_removed[j]) {
                continue;
            }

            Real lb = lp.col_lower[j];
            Real ub = lp.col_upper[j];

            if (a > 0) {
                if (std::isinf(lb)) {
                    has_inf_min = true;
                } else {
                    act_min += a * lb;
                }
                if (std::isinf(ub)) {
                    has_inf_max = true;
                } else {
                    act_max += a * ub;
                }
            } else {
                if (std::isinf(ub)) {
                    has_inf_min = true;
                } else {
                    act_min += a * ub;
                }
                if (std::isinf(lb)) {
                    has_inf_max = true;
                } else {
                    act_max += a * lb;
                }
            }
        }

        // Row is redundant if activity bounds are within row bounds.
        bool lb_redundant =
            std::isinf(lp.row_lower[i]) || (!has_inf_min && act_min >= lp.row_lower[i] - kTol);
        bool ub_redundant =
            std::isinf(lp.row_upper[i]) || (!has_inf_max && act_max <= lp.row_upper[i] + kTol);

        if (lb_redundant && ub_redundant) {
            removeActiveRow(*work_, i, col_removed, row_removed, row_active_nnz, col_active_nnz,
                            next_dirty_rows, next_dirty_cols);
            ++changes;
            ++stats_.rows_removed;
            postsolve_stack_.push(PostsolveDominatedRow{i});
        }
    }

    return changes;
}

// Dependent equations (#393), found as HiGHS finds them
// (dependent_equations.h), go as redundant rows: nothing to restore in
// postsolve, and the op keeps the row for a later dual pass.
Index Presolver::removeDependentEquations(LpProblem& lp, std::vector<bool>& col_removed,
                                          std::vector<bool>& row_removed,
                                          std::vector<Index>& row_active_nnz,
                                          std::vector<Index>& col_active_nnz,
                                          std::vector<uint8_t>& next_dirty_rows,
                                          std::vector<uint8_t>& next_dirty_cols) {
    const DependentEquations found = findDependentEquations(
        lp, *work_, col_removed, row_removed, options_.dependent_equations_time_limit);
    stats_.dependent_equation_fictitious += found.fictitious;
    if (found.timed_out) {
        ++stats_.dependent_equation_timeouts;
    }
    Index changes = 0;
    for (const Index i : found.rows) {
        removeActiveRow(*work_, i, col_removed, row_removed, row_active_nnz, col_active_nnz,
                        next_dirty_rows, next_dirty_cols);
        ++changes;
        ++stats_.rows_removed;
        postsolve_stack_.push(PostsolveDominatedRow{i});
    }
    return changes;
}

Index Presolver::detectImpliedEquations(LpProblem& lp, std::vector<bool>& col_removed,
                                        std::vector<bool>& row_removed,
                                        const std::vector<Index>& dirty_rows,
                                        std::vector<uint8_t>& next_dirty_rows,
                                        std::vector<uint8_t>& next_dirty_cols) {
    Index changes = 0;

    for (Index i : dirty_rows) {
        if (row_removed[i]) {
            continue;
        }
        ++stats_.rows_examined;
        if (std::abs(lp.row_upper[i] - lp.row_lower[i]) <= kTol) {
            continue;
        }

        const auto rv = work_->row(i);
        Real act_min = 0.0;
        Real act_max = 0.0;
        bool has_inf_min = false;
        bool has_inf_max = false;

        for (const auto [j, a] : rv) {
            if (col_removed[j]) {
                continue;
            }

            Real lb = lp.col_lower[j];
            Real ub = lp.col_upper[j];
            if (a > 0) {
                if (std::isinf(lb)) {
                    has_inf_min = true;
                } else {
                    act_min += a * lb;
                }
                if (std::isinf(ub)) {
                    has_inf_max = true;
                } else {
                    act_max += a * ub;
                }
            } else {
                if (std::isinf(ub)) {
                    has_inf_min = true;
                } else {
                    act_min += a * ub;
                }
                if (std::isinf(lb)) {
                    has_inf_max = true;
                } else {
                    act_max += a * lb;
                }
            }
        }

        if (!std::isinf(lp.row_upper[i]) && !has_inf_min &&
            std::abs(act_min - lp.row_upper[i]) <= kTol) {
            lp.row_lower[i] = lp.row_upper[i];
            ++changes;
            next_dirty_rows[i] = 1;
            markColsInRow(*work_, i, col_removed, next_dirty_cols);
        } else if (!std::isinf(lp.row_lower[i]) && !has_inf_max &&
                   std::abs(act_max - lp.row_lower[i]) <= kTol) {
            lp.row_upper[i] = lp.row_lower[i];
            ++changes;
            next_dirty_rows[i] = 1;
            markColsInRow(*work_, i, col_removed, next_dirty_cols);
        }
    }

    return changes;
}

Index Presolver::activityBoundTightening(LpProblem& lp, std::vector<bool>& col_removed,
                                         std::vector<bool>& row_removed,
                                         const std::vector<Index>& dirty_rows,
                                         std::vector<uint8_t>& next_dirty_rows,
                                         std::vector<uint8_t>& next_dirty_cols) {
    Index changes = 0;

    for (Index i : dirty_rows) {
        if (row_removed[i]) {
            continue;
        }
        ++stats_.rows_examined;

        const auto rv = work_->row(i);
        // Compensated, so a term much larger than the rest of the row (a
        // bound of 1e17, say) is taken back out exactly below.
        CompensatedSum act_min;
        CompensatedSum act_max;
        Index inf_min_count = 0;
        Index inf_max_count = 0;
        Index inf_min_col = -1;
        Index inf_max_col = -1;
        // a * x at the bound that gives the row its min (at_min) or max
        // activity; infinite when that bound is huge (kHugeBound), so the
        // residual activities below never subtract a 1e30 term back out.
        auto contrib = [&](Real a, Index j, bool at_min) {
            const Real b = (a > 0) == at_min ? lp.col_lower[j] : lp.col_upper[j];
            if (std::abs(b) >= kHugeBound) {
                return (a > 0) == (b > 0) ? kInf : -kInf;
            }
            return a * b;
        };

        for (const auto [j, a] : rv) {
            if (col_removed[j]) {
                continue;
            }
            if (a == 0.0) {
                continue;  // 0 * 1e30 would otherwise count as infinite.
            }

            Real contrib_min = contrib(a, j, true);
            Real contrib_max = contrib(a, j, false);

            if (std::isinf(contrib_min)) {
                ++inf_min_count;
                inf_min_col = j;
            } else {
                act_min.add(contrib_min);
            }
            if (std::isinf(contrib_max)) {
                ++inf_max_count;
                inf_max_col = j;
            } else {
                act_max.add(contrib_max);
            }
        }

        for (const auto [j, a] : rv) {
            if (col_removed[j]) {
                continue;
            }
            if (a == 0.0) {
                continue;
            }

            bool res_min_finite = false;
            bool res_max_finite = false;
            Real res_min = -kInf;
            Real res_max = kInf;

            if (inf_min_count == 0) {
                res_min_finite = true;
                res_min = act_min.without(contrib(a, j, true));
            } else if (inf_min_count == 1 && inf_min_col == j) {
                res_min_finite = true;
                res_min = act_min.value();
            }

            if (inf_max_count == 0) {
                res_max_finite = true;
                res_max = act_max.without(contrib(a, j, false));
            } else if (inf_max_count == 1 && inf_max_col == j) {
                res_max_finite = true;
                res_max = act_max.value();
            }

            // Huge sides count as infinite too, and a huge derived bound is
            // dropped: either would only put a 1e30-sized finite bound where
            // an infinite one was. Dropping a bound is always valid.
            const bool has_row_lower = std::abs(lp.row_lower[i]) < kHugeBound;
            const bool has_row_upper = std::abs(lp.row_upper[i]) < kHugeBound;
            Real new_ub = kInf;
            if (a > 0 && has_row_upper && res_min_finite) {
                new_ub = (lp.row_upper[i] - res_min) / a;
            } else if (a < 0 && has_row_lower && res_max_finite) {
                new_ub = (lp.row_lower[i] - res_max) / a;
            }
            Real new_lb = -kInf;
            if (a > 0 && has_row_lower && res_max_finite) {
                new_lb = (lp.row_lower[i] - res_max) / a;
            } else if (a < 0 && has_row_upper && res_min_finite) {
                new_lb = (lp.row_upper[i] - res_min) / a;
            }
            if (std::abs(new_ub) >= kHugeBound) {
                new_ub = kInf;
            }
            if (std::abs(new_lb) >= kHugeBound) {
                new_lb = -kInf;
            }
            if (lp.col_type[j] != VarType::Continuous) {
                const Real round_tol = integerRoundTol(a, false);
                new_ub = std::floor(new_ub + round_tol);
                new_lb = std::ceil(new_lb - round_tol);
            }

            bool tightened = false;
            if (new_ub < lp.col_upper[j] - kTol) {
                lp.col_upper[j] = new_ub;
                tightened = true;
            }
            if (new_lb > lp.col_lower[j] + kTol) {
                lp.col_lower[j] = new_lb;
                tightened = true;
            }

            if (tightened && lp.col_type[j] != VarType::Continuous) {
                lp.col_lower[j] = std::ceil(lp.col_lower[j] - kTol);
                lp.col_upper[j] = std::floor(lp.col_upper[j] + kTol);
            }

            if (lp.col_lower[j] > lp.col_upper[j] + kTol) {
                infeasible_ = true;
                return changes;
            }

            if (tightened) {
                ++changes;
                ++stats_.bounds_tightened;
                markRowsTouchingCol(*work_, j, row_removed, next_dirty_rows);
                next_dirty_cols[j] = 1;
            }
        }
    }

    return changes;
}

Index Presolver::dualFixing(LpProblem& lp, std::vector<bool>& col_removed,
                            std::vector<bool>& row_removed, std::vector<Index>& row_active_nnz,
                            std::vector<Index>& col_active_nnz,
                            const std::vector<Index>& dirty_cols,
                            std::vector<uint8_t>& next_dirty_rows,
                            std::vector<uint8_t>& next_dirty_cols) {
    Index changes = 0;

    // Each row's activity over its active columns, huge bounds infinite,
    // summed on first use and updated as fixes take columns out of the row
    // (the extended and single-equation cases), allocated on first use. No other
    // bound changes in this pass.
    struct Activity {
        CompensatedSum min;
        CompensatedSum max;
        Index min_inf = 0;
        Index max_inf = 0;
        bool valid = false;
    };
    std::vector<Activity> activity;
    auto rowActivity = [&](Index r) -> const Activity& {
        if (activity.empty()) {
            activity.resize(static_cast<size_t>(lp.num_rows));
        }
        Activity& act = activity[r];
        if (!act.valid) {
            act = Activity{};
            for (const auto [col, a] : work_->row(r)) {
                if (col_removed[col] || a == 0.0) {
                    continue;
                }
                const Real at_min = a > 0.0 ? lp.col_lower[col] : lp.col_upper[col];
                const Real at_max = a > 0.0 ? lp.col_upper[col] : lp.col_lower[col];
                if (std::abs(at_min) >= kHugeBound) {
                    ++act.min_inf;
                } else {
                    act.min.add(a * at_min);
                }
                if (std::abs(at_max) >= kHugeBound) {
                    ++act.max_inf;
                } else {
                    act.max.add(a * at_max);
                }
            }
            act.valid = true;
        }
        return act;
    };
    // Whether every row that locks x_j against moving down (upward: up)
    // holds at x_j = value for any values of its other columns within their
    // bounds: HiGHS's worst-case bound (computeWorstCaseLowerBound) at or
    // past that bound.
    auto holdsForAnyRest = [&](Index j, Real value, bool upward) -> bool {
        const Real lj = lp.col_lower[j];
        const Real uj = lp.col_upper[j];
        for (const auto [r, a] : work_->col(j)) {
            if (row_removed[r] || a == 0.0) {
                continue;
            }
            const bool lower_locks = !std::isinf(lp.row_lower[r]) && ((a > 0.0) != upward);
            const bool upper_locks = !std::isinf(lp.row_upper[r]) && ((a < 0.0) != upward);
            if (!lower_locks && !upper_locks) {
                continue;
            }
            const Activity& act = rowActivity(r);
            if (lower_locks) {
                // The row's least activity over its other columns.
                const Real j_at_min = a > 0.0 ? lj : uj;
                Index inf = act.min_inf;
                Real rest = 0.0;
                if (std::abs(j_at_min) >= kHugeBound) {
                    --inf;
                    rest = act.min.value();
                } else {
                    rest = act.min.without(a * j_at_min);
                }
                if (inf > 0 || rest + a * value < lp.row_lower[r]) {
                    return false;
                }
            }
            if (upper_locks) {
                const Real j_at_max = a > 0.0 ? uj : lj;
                Index inf = act.max_inf;
                Real rest = 0.0;
                if (std::abs(j_at_max) >= kHugeBound) {
                    --inf;
                    rest = act.max.value();
                } else {
                    rest = act.max.without(a * j_at_max);
                }
                if (inf > 0 || rest + a * value > lp.row_upper[r]) {
                    return false;
                }
            }
        }
        return true;
    };

    // Equations the single-equation case has checked (#367), allocated on
    // first use; a fix clears the flag of each row it touches.
    std::vector<uint8_t> equation_checked;
    // Fixes x_k at value, its term moved into its rows' sides.
    auto fixColumn = [&](Index k, Real value) {
        removeActiveColumn(*work_, k, col_removed, row_removed, row_active_nnz, col_active_nnz,
                           next_dirty_rows, next_dirty_cols);
        ++changes;
        ++stats_.vars_removed;
        lp.obj_offset += lp.obj[k] * value;
        postsolve_stack_.push(PostsolveFixVariable{k, value});

        for (const auto [row, a] : work_->col(k)) {
            if (row_removed[row]) {
                continue;
            }
            shiftRowSides(lp, row, a * value);
            next_dirty_rows[row] = 1;
            if (!equation_checked.empty()) {
                // The row lost a column: HiGHS checks it again (markChangedRow).
                equation_checked[row] = 0;
            }
            if (!activity.empty() && activity[row].valid && a != 0.0) {
                // x_k's term leaves the row's activity, its value the sides
                // (shiftRowSides): updated in place, as a dense row re-summed
                // after each of its columns is fixed costs O(n^2).
                Activity& act = activity[row];
                const Real at_min = a > 0.0 ? lp.col_lower[k] : lp.col_upper[k];
                const Real at_max = a > 0.0 ? lp.col_upper[k] : lp.col_lower[k];
                if (std::abs(at_min) >= kHugeBound) {
                    --act.min_inf;
                } else {
                    act.min.add(-a * at_min);
                }
                if (std::abs(at_max) >= kHugeBound) {
                    --act.max_inf;
                } else {
                    act.max.add(-a * at_max);
                }
            }
        }
    };

    // Dual fixing's single-equation extension (#367), HiGHS 1.16.0's
    // checkColumn and handleSingleEquation (HPresolveDualFixing.cpp;
    // Achterberg et al., "Presolve Reductions in Mixed Integer Programming",
    // INFORMS JoC 32(2), section 6.1). As in HiGHS, an equation's flag is set
    // after its check, so the fixes that check makes (which clear their rows'
    // flags in fixColumn, as HiGHS's markChangedRow) leave it checked, and a
    // fix from another equation's check clears it. The flags last one call,
    // where HiGHS's last until a change, so an equation can be checked again
    // in a later call with no change between; each fix is valid whenever it
    // is found, so that can only change how many columns are fixed.
    // Whether row r holds for any values of its columns within their bounds:
    // HiGHS's isRedundant, but by the columns' bounds only, where HiGHS's
    // implied row bounds also take implied column bounds, so fewer rows count
    // as redundant here. The tests use kTol where HiGHS uses primal_feastol;
    // both differences can only leave a column unfixed.
    auto isRedundantRow = [&](Index r) {
        const Activity& act = rowActivity(r);
        const bool lower_holds = std::isinf(lp.row_lower[r]) ||
                                 (act.min_inf == 0 && act.min.value() >= lp.row_lower[r] - kTol);
        const bool upper_holds = std::isinf(lp.row_upper[r]) ||
                                 (act.max_inf == 0 && act.max.value() <= lp.row_upper[r] + kTol);
        return lower_holds && upper_holds;
    };
    // Whether x_k, its coefficient in the equation's sign flipped by
    // direction, is pushed down by its cost and only locked upward by its
    // other rows: then it would lower the equation's activity (direction =
    // sign of its coefficient there; S+) or raise it (the other sign; S-) if
    // the equation let it.
    auto checkColumn = [&](Index k, int direction, Index equation) {
        const Real cost = lp.sense == Sense::Minimize ? lp.obj[k] : -lp.obj[k];
        if (direction * cost < 0.0) {
            return false;
        }
        for (const auto [r, a] : work_->col(k)) {
            if (r == equation || row_removed[r] || a == 0.0 || isRedundantRow(r)) {
                continue;
            }
            // Another ranged row (or equation) locks x_k both ways.
            if (!std::isinf(lp.row_lower[r]) && !std::isinf(lp.row_upper[r])) {
                return false;
            }
            const int row_direction =
                std::isinf(lp.row_lower[r]) && !std::isinf(lp.row_upper[r]) ? 1 : -1;
            if (static_cast<Real>(direction * row_direction) * a < 0.0) {
                return false;
            }
        }
        return true;
    };
    struct EquationEntry {
        Index col;
        Real val;
        int mark;  // 1: S+, -1: S-, 0: neither.
    };
    std::vector<EquationEntry> equation_entries;
    auto handleSingleEquation = [&](Index equation) {
        equation_entries.clear();
        Index num_plus = 0;
        Index num_minus = 0;
        for (const auto [k, a] : work_->row(equation)) {
            if (col_removed[k] || a == 0.0) {
                continue;
            }
            const int col_direction = a > 0.0 ? 1 : -1;
            int mark = 0;
            if (checkColumn(k, col_direction, equation)) {
                ++num_plus;
                mark = 1;
            } else if (checkColumn(k, -col_direction, equation)) {
                ++num_minus;
                mark = -1;
            }
            equation_entries.push_back({k, a, mark});
        }
        if (num_plus == 0 && num_minus == 0) {
            return;
        }
        // T+: the equation's supremum over the columns outside S+ (and
        // integral ones), S^C+: its infimum over the continuous S+; T- and
        // S^C- the other way round.
        CompensatedSum t_plus;
        CompensatedSum t_minus;
        CompensatedSum sc_plus;
        CompensatedSum sc_minus;
        bool t_plus_finite = true;
        bool t_minus_finite = true;
        bool sc_plus_finite = true;
        bool sc_minus_finite = true;
        const auto addActivity = [&](const EquationEntry& e, CompensatedSum& sum, bool& finite,
                                     int direction) {
            const Real bound = direction * e.val > 0.0 ? lp.col_upper[e.col] : lp.col_lower[e.col];
            finite = finite && std::abs(bound) < kHugeBound;
            if (finite) {
                sum.add(e.val * bound);
            }
        };
        for (const EquationEntry& e : equation_entries) {
            const bool integral = lp.col_type[e.col] != VarType::Continuous;
            if (e.mark <= 0 || integral) {
                addActivity(e, t_plus, t_plus_finite, 1);
            } else {
                addActivity(e, sc_plus, sc_plus_finite, -1);
            }
            if (e.mark >= 0 || integral) {
                addActivity(e, t_minus, t_minus_finite, -1);
            } else {
                addActivity(e, sc_minus, sc_minus_finite, 1);
            }
            if ((num_minus == 0 || !t_plus_finite || !sc_plus_finite) &&
                (num_plus == 0 || !t_minus_finite || !sc_minus_finite)) {
                break;
            }
        }
        // Fix all of S- at the bound that raises the equation's activity, or
        // all of S+ at the one that lowers it (HiGHS's fixCols).
        int direction = 0;
        const Real rhs = lp.row_lower[equation];
        if (num_minus > 0 && t_plus_finite && sc_plus_finite &&
            t_plus.value() + sc_plus.value() <= rhs + kTol) {
            direction = 1;
        } else if (num_plus > 0 && t_minus_finite && sc_minus_finite &&
                   t_minus.value() + sc_minus.value() >= rhs - kTol) {
            direction = -1;
        }
        if (direction == 0) {
            return;
        }
        for (const EquationEntry& e : equation_entries) {
            if (direction * e.mark >= 0) {
                continue;
            }
            // Same guard as the column loop: report a crossed bound rather
            // than bury it in the postsolve stack.
            if (lp.col_lower[e.col] > lp.col_upper[e.col] + kTol) {
                infeasible_ = true;
                return;
            }
            // As the extended case's fixes, these can leave a tight row's
            // dual giving x_k a reduced cost of the wrong sign: a dual
            // postsolve (#323) must recompute it.
            fixColumn(e.col, direction * e.val > 0.0 ? lp.col_upper[e.col] : lp.col_lower[e.col]);
            ++stats_.single_equation_dual_fixing_changes;
        }
    };

    for (Index j : dirty_cols) {
        if (col_removed[j]) {
            continue;
        }
        ++stats_.cols_examined;

        // Same guard as removeSingletonCols: a lock-based fix would bury a
        // crossed bound in the postsolve stack instead of reporting it.
        if (lp.col_lower[j] > lp.col_upper[j] + kTol) {
            infeasible_ = true;
            return changes;
        }

        Index up_locks = 0;
        Index down_locks = 0;
        // The last row locking x_j each way, for the single-equation case.
        Index up_lock_row = -1;
        Index down_lock_row = -1;
        Index active_nnz = col_active_nnz[j];
        if (active_nnz == 0) {
            continue;  // Handle empty columns in removeEmptyColumns().
        }
        const auto cv = work_->col(j);
        for (const auto [row, a] : cv) {
            if (row_removed[row]) {
                continue;
            }
            bool has_upper = !std::isinf(lp.row_upper[row]);
            bool has_lower = !std::isinf(lp.row_lower[row]);

            if (has_upper) {
                if (a > 0) {
                    ++up_locks;
                    up_lock_row = row;
                }
                if (a < 0) {
                    ++down_locks;
                    down_lock_row = row;
                }
            }
            if (has_lower) {
                if (a < 0) {
                    ++up_locks;
                    up_lock_row = row;
                }
                if (a > 0) {
                    ++down_locks;
                    down_lock_row = row;
                }
            }
        }

        bool prefer_down;
        if (lp.sense == Sense::Minimize) {
            prefer_down = (lp.obj[j] >= 0.0);
        } else {
            prefer_down = (lp.obj[j] <= 0.0);
        }

        bool fixed = false;
        Real fix_value = 0.0;
        if (prefer_down && down_locks == 0 && !std::isinf(lp.col_lower[j])) {
            fixed = true;
            fix_value = lp.col_lower[j];
        } else if (!prefer_down && up_locks == 0 && !std::isinf(lp.col_upper[j])) {
            fixed = true;
            fix_value = lp.col_upper[j];
        }
        // HiGHS's dualFixing skips a column fixed by its bounds.
        if (!fixed && options_.enable_single_equation_dual_fixing &&
            lp.col_lower[j] != lp.col_upper[j]) {
            // HiGHS's trigger: its locks count the cost as one, from no row,
            // so a single down (up) lock is one row's with the cost not
            // locking that way too, and a column with no lock either way is
            // the cases above. The fixes may take x_j itself.
            const Real cost = lp.sense == Sense::Minimize ? lp.obj[j] : -lp.obj[j];
            const Index highs_down = down_locks + (cost < 0.0 ? 1 : 0);
            const Index highs_up = up_locks + (cost > 0.0 ? 1 : 0);
            if (highs_down > 0 && highs_up > 0) {
                const auto isEquation = [&](Index r) {
                    return r >= 0 && lp.row_lower[r] == lp.row_upper[r];
                };
                Index equation = -1;
                if (highs_down == 1 && down_locks == 1 && isEquation(down_lock_row)) {
                    equation = down_lock_row;
                } else if (highs_up == 1 && up_locks == 1 && isEquation(up_lock_row)) {
                    equation = up_lock_row;
                }
                if (equation >= 0) {
                    if (equation_checked.empty()) {
                        equation_checked.resize(static_cast<size_t>(lp.num_rows), 0);
                    }
                    if (equation_checked[equation] == 0) {
                        handleSingleEquation(equation);
                        if (infeasible_) {
                            return changes;
                        }
                        equation_checked[equation] = 1;
                        if (col_removed[j]) {
                            continue;
                        }
                    }
                }
            }
        }
        if (!fixed && options_.enable_extended_dual_fixing) {
            // HiGHS's dualFixing (#366), its bound strengthening applied as a
            // fix: a column whose cost does not push it up (cost >= 0 when
            // minimizing) and whose every down-locking row holds at x_j = l_j
            // whatever values its other columns take is optimal at l_j.
            // Moving it down there keeps those rows by that test, the others
            // because they do not lock it downward, and does not raise the
            // objective. The same upward. A zero-cost column with no up locks
            // is the case with no row to test; prefer_down picked the lower
            // side for it and missed it.
            //
            // With x_j at that bound the test is the row's whole least (most)
            // activity, so each locking side is redundant. A tested side can
            // still be tight at the reduced optimum, though: its dual may then
            // leave x_j a reduced cost of the wrong sign at the bound (the old
            // rule's cannot). A dual postsolve (#323) must recompute d_j for
            // these fixes, as HiGHS's FixedCol::undo does.
            const Real cost = lp.sense == Sense::Minimize ? lp.obj[j] : -lp.obj[j];
            if (cost >= 0.0 && lp.col_lower[j] > -kHugeBound &&
                holdsForAnyRest(j, lp.col_lower[j], false)) {
                fixed = true;
                fix_value = lp.col_lower[j];
            } else if (cost <= 0.0 && lp.col_upper[j] < kHugeBound &&
                       holdsForAnyRest(j, lp.col_upper[j], true)) {
                fixed = true;
                fix_value = lp.col_upper[j];
            }
            if (fixed) {
                ++stats_.extended_dual_fixing_changes;
            }
        }
        if (fixed) {
            fixColumn(j, fix_value);
        }
    }

    return changes;
}

// Dominated columns, HiGHS 1.16.0's detectDominatedCol (HPresolve.cpp; #376,
// part 2 of #361), on an LP. With the row duals' implied bounds (#375), the
// reduced cost d_j = c_j - sum_i a_ij y_i of x_j lies in
// [c_j - sumUpper_j, c_j - sumLower_j] in every optimal dual. A lower bound
// above the dual feasibility tolerance puts x_j at its lower bound in every
// optimum (strongly dominated); one at or above -tolerance leaves an optimum
// there (weakly dominated). The same upward. Either way x_j is fixed at
// that bound when it is finite. HiGHS calls it as it presolves each column,
// before dual fixing, and here it is one pass before dual fixing.
//
// The bounds are derived once per pass (DualBounds recomputes; HiGHS keeps
// them by events). Fixing x_k drops its dual constraint, and so every
// implied row-dual bound x_k was the source of, which HiGHS resets
// (resetRowDualImpliedBoundsDerivedFromCol): a column with such a bound in
// one of its rows waits for the next pass, which derives them again, so no
// two columns are each fixed through a bound from the other. Every other
// bound still holds: a fix keeps the optimal primal set or shrinks it, so
// what the other columns' bounds rest on holds there too.
//
// The wait matters within tolerance. Say x_k is fixed at its lower bound,
// so c_k - max(sum_i a_ik y_i) >= -tol, the max over the bounds its sums
// read (the side, in a row whose bound k sourced). A bound k sourced on y_r
// comes from one side of its dual constraint with its other rows at their
// extremes over the sides: sum_i a_ik y_i <= c_k (x_k unbounded above), or
// >= c_k (x_k's lower bound strictly implied). Beside a strong fix the
// first is no tighter than r's side and the second crosses bounds (a dual
// infeasible LP). A weak fix leaves room within tol: y_r can then be
// tighter than its side by up to about tol/|a_kr|, enough for another
// column's fix to rest on. Only the effective bound, the tighter of the
// implied one and the side, lies within the sides, so the guard compares
// the implied bound against the side.
//
// A strongly dominated column at an infinite bound makes the LP dual
// infeasible, which presolve has no status for; it is left to the solver.
// As for #366's fixes, a dual postsolve (#323) must recompute d_j for these
// fixes: the reduced LP's duals need not give d_j its sign at the bound.
//
// The column bounds are the presolved model's, including those activity
// bound tightening wrote in from the rows. So a column that was half-free
// can be weakly fixed at an implied bound where HiGHS, which keeps implied
// bounds apart, sees an infinite one and tries a forcing column. Both are
// valid (the tightened LP has the same optima), but the counts differ from
// HiGHS's.
//
// A forcing column (#383) is weakly dominated toward an infinite bound
// with d_j = 0 by the row sides alone: every row of x_j is one-sided, and
// x_j moving toward that bound meets each side. x_j and its rows go, and
// postsolve moves x_j just far enough. The LP left is the original's
// projection, with the same optima, so the bounds read for later columns
// still hold and none of them waits for it. One exception within tol: a
// cost on the dual-infeasible side (c_j < 0 going up) sources row-dual
// bounds that cross the sides, and a later column in those rows can read
// them, though the rows are gone (HiGHS re-sums on row removal). The LP is
// then unbounded in exact arithmetic, as the comment on removeForcing
// below says.
Index Presolver::removeDominatedColumns(LpProblem& lp, std::vector<bool>& col_removed,
                                        std::vector<bool>& row_removed,
                                        std::vector<Index>& row_active_nnz,
                                        std::vector<Index>& col_active_nnz,
                                        std::vector<uint8_t>& next_dirty_rows,
                                        std::vector<uint8_t>& next_dirty_cols) {
    if (infeasible_) {
        return 0;
    }
    DualBounds bounds;
    bounds.compute(lp, *work_, col_removed, row_removed);
    const Real sense = lp.sense == Sense::Maximize ? -1.0 : 1.0;
    constexpr Real tol = DualBounds::kDualFeasTol;

    Index changes = 0;
    // Columns this pass fixed, allocated on the first fix.
    std::vector<uint8_t> fixed_here;
    auto sourcedByFixed = [&](Index j) {
        for (const auto [row, a] : work_->col(j)) {
            if (row_removed[row] || a == 0.0) {
                continue;
            }
            // Only a bound tighter than the row's side counts: the sums read
            // the side otherwise, so dropping the bound changes nothing.
            const Index lo = bounds.rowDualLowerSource(row);
            const Index up = bounds.rowDualUpperSource(row);
            if (lo >= 0 && fixed_here[lo] != 0 &&
                bounds.implRowDualLower(row) > bounds.rowDualLower(row)) {
                return true;
            }
            if (up >= 0 && fixed_here[up] != 0 &&
                bounds.implRowDualUpper(row) < bounds.rowDualUpper(row)) {
                return true;
            }
        }
        return false;
    };

    // HiGHS's forcing-column test: x_j's bound in `direction` is infinite,
    // so dual feasibility bounds d_j on the other side by 0, and by the
    // row sides alone (the ...Orig sums) d_j is bounded by 0 on this side
    // too, within tol over the column's dynamism.
    auto isForcing = [&](Index j, Real cost, Real direction) {
        Real min_abs = kInf;
        Real max_abs = 0.0;
        for (const auto [row, a] : work_->col(j)) {
            if (!row_removed[row] && a != 0.0) {
                min_abs = std::min(min_abs, std::abs(a));
                max_abs = std::max(max_abs, std::abs(a));
            }
        }
        const Real dynamism = max_abs / min_abs;
        const Real bound =
            direction > 0.0 ? -bounds.sumUpperOrig(j, -cost) : -bounds.sumLowerOrig(j, -cost);
        return std::abs(bound) <= std::max(tol / dynamism, std::numeric_limits<Real>::epsilon());
    };
    // Removes x_j with every row it is in. Each row records the side that
    // x_j meets by moving toward its infinite bound (the upper where
    // direction * a > 0, direction > 0 for an infinite lower bound), and its
    // other terms, for postsolve. The rows' other sides are infinite, or
    // the row duals would be free and the ...Orig sums infinite. x_j's cost
    // is within the test's tolerance of 0, and is dropped, as in HiGHS:
    // c_j x_j is not carried into the offset, so the postsolved objective
    // may differ from the reduced one by it, and an LP unbounded only
    // through such a cost presolves to a bounded one.
    auto removeForcing = [&](Index j, Real direction) {
        PostsolveForcingCol op{
            j, direction > 0.0 ? lp.col_upper[j] : lp.col_lower[j], direction < 0.0, {}};
        for (const auto [row, a] : work_->col(j)) {
            if (row_removed[row] || a == 0.0) {
                continue;
            }
            PostsolveForcingCol::Row r{
                row, a, direction * a > 0.0 ? lp.row_upper[row] : lp.row_lower[row], {}, {}};
            for (const auto [col, b] : work_->row(row)) {
                if (col != j && !col_removed[col] && b != 0.0) {
                    r.cols.push_back(col);
                    r.coeffs.push_back(b);
                }
            }
            op.rows.push_back(std::move(r));
        }
        removeActiveColumn(*work_, j, col_removed, row_removed, row_active_nnz, col_active_nnz,
                           next_dirty_rows, next_dirty_cols);
        for (const auto& r : op.rows) {
            removeActiveRow(*work_, r.orig_row, col_removed, row_removed, row_active_nnz,
                            col_active_nnz, next_dirty_rows, next_dirty_cols);
        }
        ++stats_.vars_removed;
        stats_.rows_removed += static_cast<Index>(op.rows.size());
        ++stats_.weakly_dominated_col_changes;
        ++stats_.forcing_col_changes;
        postsolve_stack_.push(std::move(op));
    };

    for (Index j = 0; j < lp.num_cols; ++j) {
        if (col_removed[j] || col_active_nnz[j] == 0 || lp.col_lower[j] == lp.col_upper[j]) {
            continue;
        }
        if (!fixed_here.empty() && sourcedByFixed(j)) {
            continue;
        }
        const Real cost = sense * lp.obj[j];
        const Real dual_lower = -bounds.sumUpper(j, -cost);
        const Real dual_upper = -bounds.sumLower(j, -cost);
        const Real lower = lp.col_lower[j];
        const Real upper = lp.col_upper[j];
        const bool lower_finite = lower > -kHugeBound;
        const bool upper_finite = upper < kHugeBound;

        // HiGHS's order: strong downward, strong upward, weak downward, weak
        // upward; the first that applies with a finite bound fixes x_j. A
        // weak one at an infinite bound may make x_j a forcing column.
        Real value = 0.0;
        bool fix = false;
        bool weak = false;
        Real forcing = 0.0;  // The forcing column's direction, 0 for none.
        if (dual_lower > tol) {
            fix = lower_finite;
            value = lower;
        } else if (dual_upper < -tol) {
            fix = upper_finite;
            value = upper;
        } else {
            if (dual_lower >= -tol) {
                if (lower_finite) {
                    fix = weak = true;
                    value = lower;
                } else if (isForcing(j, cost, 1.0)) {
                    forcing = 1.0;
                }
            }
            if (!fix && forcing == 0.0 && dual_upper <= tol) {
                if (upper_finite) {
                    fix = weak = true;
                    value = upper;
                } else if (isForcing(j, cost, -1.0)) {
                    forcing = -1.0;
                }
            }
        }
        if (forcing != 0.0) {
            removeForcing(j, forcing);
            ++changes;
            continue;
        }
        if (!fix) {
            continue;
        }

        removeActiveColumn(*work_, j, col_removed, row_removed, row_active_nnz, col_active_nnz,
                           next_dirty_rows, next_dirty_cols);
        ++changes;
        ++stats_.vars_removed;
        if (weak) {
            ++stats_.weakly_dominated_col_changes;
        }
        lp.obj_offset += lp.obj[j] * value;
        postsolve_stack_.push(PostsolveFixVariable{j, value});
        for (const auto [row, a] : work_->col(j)) {
            if (!row_removed[row]) {
                shiftRowSides(lp, row, a * value);
            }
        }
        if (fixed_here.empty()) {
            fixed_here.resize(static_cast<size_t>(lp.num_cols), 0);
        }
        fixed_here[j] = 1;
    }
    return changes;
}

Index Presolver::removeEmptyColumns(LpProblem& lp, std::vector<bool>& col_removed,
                                    std::vector<bool>& row_removed,
                                    std::vector<Index>& row_active_nnz,
                                    std::vector<Index>& col_active_nnz,
                                    const std::vector<Index>& dirty_cols,
                                    std::vector<uint8_t>& next_dirty_rows,
                                    std::vector<uint8_t>& next_dirty_cols) {
    Index changes = 0;

    for (Index j : dirty_cols) {
        if (col_removed[j]) {
            continue;
        }
        ++stats_.cols_examined;
        if (lp.col_lower[j] > lp.col_upper[j] + kTol) {
            infeasible_ = true;
            return changes;
        }

        if (col_active_nnz[j] != 0) {
            continue;
        }

        bool lb_finite = !std::isinf(lp.col_lower[j]);
        bool ub_finite = !std::isinf(lp.col_upper[j]);
        Real fix_value = 0.0;
        if (lp.sense == Sense::Minimize) {
            if (lp.obj[j] > kTol) {
                if (!lb_finite) {
                    continue;  // Keep variable; objective can be unbounded below.
                }
                fix_value = lp.col_lower[j];
            } else if (lp.obj[j] < -kTol) {
                if (!ub_finite) {
                    continue;  // Keep variable; objective can be unbounded below.
                }
                fix_value = lp.col_upper[j];
            } else {
                if (!lb_finite && !ub_finite) {
                    fix_value = 0.0;
                } else {
                    fix_value = lb_finite ? lp.col_lower[j] : lp.col_upper[j];
                }
            }
        } else {
            if (lp.obj[j] > kTol) {
                if (!ub_finite) {
                    continue;  // Keep variable; objective can be unbounded above.
                }
                fix_value = lp.col_upper[j];
            } else if (lp.obj[j] < -kTol) {
                if (!lb_finite) {
                    continue;  // Keep variable; objective can be unbounded above.
                }
                fix_value = lp.col_lower[j];
            } else {
                if (!lb_finite && !ub_finite) {
                    fix_value = 0.0;
                } else {
                    fix_value = lb_finite ? lp.col_lower[j] : lp.col_upper[j];
                }
            }
        }

        removeActiveColumn(*work_, j, col_removed, row_removed, row_active_nnz, col_active_nnz,
                           next_dirty_rows, next_dirty_cols);
        ++changes;
        ++stats_.vars_removed;
        lp.obj_offset += lp.obj[j] * fix_value;
        postsolve_stack_.push(PostsolveFixVariable{j, fix_value});
    }

    return changes;
}

Index Presolver::removeDuplicateRows(LpProblem& lp, std::vector<bool>& col_removed,
                                     std::vector<bool>& row_removed,
                                     std::vector<Index>& row_active_nnz,
                                     std::vector<Index>& col_active_nnz,
                                     const std::vector<Index>& dirty_rows,
                                     std::vector<uint8_t>& next_dirty_rows,
                                     std::vector<uint8_t>& next_dirty_cols) {
    Index changes = 0;
    std::unordered_map<uint64_t, std::vector<Index>> buckets;
    buckets.reserve(static_cast<size_t>(lp.num_rows));

    for (Index i : dirty_rows) {
        if (row_removed[i]) {
            continue;
        }
        ++stats_.rows_examined;
        uint64_t h = rowPatternHash(*work_, i, col_removed);
        buckets[h].push_back(i);
    }

    for (const auto& bucket : buckets) {
        const auto& rows = bucket.second;
        if (rows.size() < 2) {
            continue;
        }

        for (size_t a = 0; a < rows.size(); ++a) {
            Index i = rows[a];
            if (row_removed[i]) {
                continue;
            }
            for (size_t b = a + 1; b < rows.size(); ++b) {
                Index j = rows[b];
                if (row_removed[j]) {
                    continue;
                }
                if (!rowsHaveSamePattern(*work_, i, j, col_removed, kTol)) {
                    continue;
                }

                bool i_subsumes_j = rowIntervalSubsumes(lp.row_lower[i], lp.row_upper[i],
                                                        lp.row_lower[j], lp.row_upper[j], kTol);
                bool j_subsumes_i = rowIntervalSubsumes(lp.row_lower[j], lp.row_upper[j],
                                                        lp.row_lower[i], lp.row_upper[i], kTol);

                if (i_subsumes_j && !j_subsumes_i) {
                    removeActiveRow(*work_, j, col_removed, row_removed, row_active_nnz,
                                    col_active_nnz, next_dirty_rows, next_dirty_cols);
                    ++changes;
                    ++stats_.rows_removed;
                    postsolve_stack_.push(PostsolveDominatedRow{j});
                } else if (j_subsumes_i && !i_subsumes_j) {
                    removeActiveRow(*work_, i, col_removed, row_removed, row_active_nnz,
                                    col_active_nnz, next_dirty_rows, next_dirty_cols);
                    ++changes;
                    ++stats_.rows_removed;
                    postsolve_stack_.push(PostsolveDominatedRow{i});
                    break;
                } else if (i_subsumes_j && j_subsumes_i) {
                    // Exact duplicate row intervals: drop the later one.
                    removeActiveRow(*work_, j, col_removed, row_removed, row_active_nnz,
                                    col_active_nnz, next_dirty_rows, next_dirty_cols);
                    ++changes;
                    ++stats_.rows_removed;
                    postsolve_stack_.push(PostsolveDominatedRow{j});
                }
            }
        }
    }

    return changes;
}

Index Presolver::removeParallelRows(LpProblem& lp, std::vector<bool>& col_removed,
                                    std::vector<bool>& row_removed,
                                    std::vector<Index>& row_active_nnz,
                                    std::vector<Index>& col_active_nnz,
                                    const std::vector<Index>& dirty_rows,
                                    std::vector<uint8_t>& next_dirty_rows,
                                    std::vector<uint8_t>& next_dirty_cols) {
    Index changes = 0;
    std::unordered_map<uint64_t, std::vector<Index>> buckets;
    buckets.reserve(static_cast<size_t>(lp.num_rows));

    for (Index i : dirty_rows) {
        if (row_removed[i]) {
            continue;
        }
        ++stats_.rows_examined;
        const uint64_t h = rowSupportHash(*work_, i, col_removed);
        buckets[h].push_back(i);
    }

    auto scaleInterval = [](Real lower, Real upper, Real scale) -> std::pair<Real, Real> {
        if (scale > 0.0) {
            return {lower * scale, upper * scale};
        }
        return {upper * scale, lower * scale};
    };

    for (const auto& bucket : buckets) {
        const auto& rows = bucket.second;
        if (rows.size() < 2) {
            continue;
        }
        for (size_t a = 0; a < rows.size(); ++a) {
            const Index i = rows[a];
            if (row_removed[i]) {
                continue;
            }
            for (size_t b = a + 1; b < rows.size(); ++b) {
                const Index j = rows[b];
                if (row_removed[j]) {
                    continue;
                }

                Real scale = 0.0;
                if (!rowsHaveScaledPattern(*work_, i, j, col_removed, kTol, scale)) {
                    continue;
                }
                if (std::abs(scale) <= kTol) {
                    continue;
                }
                if (!std::isfinite(scale)) {
                    continue;
                }

                const auto [i_to_j_lb, i_to_j_ub] =
                    scaleInterval(lp.row_lower[i], lp.row_upper[i], scale);
                const auto [j_to_i_lb, j_to_i_ub] =
                    scaleInterval(lp.row_lower[j], lp.row_upper[j], 1.0 / scale);

                const bool i_subsumes_j = rowIntervalSubsumes(i_to_j_lb, i_to_j_ub, lp.row_lower[j],
                                                              lp.row_upper[j], kTol);
                const bool j_subsumes_i = rowIntervalSubsumes(j_to_i_lb, j_to_i_ub, lp.row_lower[i],
                                                              lp.row_upper[i], kTol);

                if (i_subsumes_j && !j_subsumes_i) {
                    removeActiveRow(*work_, j, col_removed, row_removed, row_active_nnz,
                                    col_active_nnz, next_dirty_rows, next_dirty_cols);
                    ++changes;
                    ++stats_.rows_removed;
                    postsolve_stack_.push(PostsolveDominatedRow{j});
                } else if (j_subsumes_i && !i_subsumes_j) {
                    removeActiveRow(*work_, i, col_removed, row_removed, row_active_nnz,
                                    col_active_nnz, next_dirty_rows, next_dirty_cols);
                    ++changes;
                    ++stats_.rows_removed;
                    postsolve_stack_.push(PostsolveDominatedRow{i});
                    break;
                } else if (i_subsumes_j && j_subsumes_i) {
                    removeActiveRow(*work_, j, col_removed, row_removed, row_active_nnz,
                                    col_active_nnz, next_dirty_rows, next_dirty_cols);
                    ++changes;
                    ++stats_.rows_removed;
                    postsolve_stack_.push(PostsolveDominatedRow{j});
                }
            }
        }
    }

    return changes;
}

namespace {

/// Hash of a column's active rows and of its coefficients divided by the
/// first active one, so parallel columns share a bucket and columns that
/// only share a support (all of a single row's, say) mostly do not. The
/// quotients are hashed to their top 30 mantissa bits: a pair whose
/// rounding straddles a step is missed, never merged wrongly, as every
/// pair is checked again.
/// `first_out` receives the first active coefficient.
uint64_t colPatternHash(const PresolveMatrix& m, Index col, const std::vector<bool>& row_removed,
                        Real& first_out) {
    uint64_t h = 1469598103934665603ULL;
    Index active = 0;
    first_out = 0.0;
    for (const auto [row, a] : m.col(col)) {
        if (row_removed[row]) {
            continue;
        }
        if (active == 0) {
            first_out = a;
        }
        h = mixHash(h, static_cast<uint64_t>(row));
        if (first_out != 0.0) {
            // Sign, exponent and the top 30 of the 52 mantissa bits.
            constexpr uint64_t kKeep = ~((uint64_t{1} << 22) - 1);
            h = mixHash(h, std::bit_cast<uint64_t>(a / first_out) & kKeep);
        }
        ++active;
    }
    return mixHash(h, static_cast<uint64_t>(active));
}

/// Whether A_c2 = scale * A_c1 over the active rows, to a relative 1e-12: a
/// shift between the two columns then moves no row activity by more than
/// rounding.
bool colsHaveScaledPattern(const PresolveMatrix& m, Index c1, Index c2,
                           const std::vector<bool>& row_removed, Real& scale_out) {
    constexpr Real kRelTol = 1e-12;
    const auto a = m.col(c1);
    const auto b = m.col(c2);
    auto p = a.begin();
    auto q = b.begin();
    bool have_scale = false;
    Real scale = 0.0;
    while (true) {
        skipRemoved(p, a.end(), row_removed);
        skipRemoved(q, b.end(), row_removed);
        const bool done_a = p == a.end();
        const bool done_b = q == b.end();
        if (done_a || done_b) {
            if (done_a && done_b && have_scale) {
                scale_out = scale;
                return true;
            }
            return false;
        }
        const auto [row, va] = *p;
        const auto [row_b, vb] = *q;
        if (row != row_b) {
            return false;
        }
        if (va == 0.0 || vb == 0.0) {
            return false;
        }
        if (!have_scale) {
            scale = vb / va;
            if (!std::isfinite(scale) || scale == 0.0) {
                return false;
            }
            have_scale = true;
        } else if (std::abs(scale * va - vb) >
                   kRelTol * std::max(std::abs(scale * va), std::abs(vb))) {
            return false;
        }
        ++p;
        ++q;
    }
}

}  // namespace

Index Presolver::removeParallelCols(LpProblem& lp, std::vector<bool>& col_removed,
                                    const std::vector<bool>& row_removed,
                                    std::vector<Index>& row_active_nnz,
                                    std::vector<Index>& col_active_nnz,
                                    const std::vector<Index>& dirty_cols,
                                    std::vector<uint8_t>& next_dirty_rows,
                                    std::vector<uint8_t>& next_dirty_cols) {
    // Parallel columns j share a pattern u: A_j = a_j * u, with a_j the first
    // active coefficient. In terms of z_j = a_j * x_j each adds z_j to the
    // same activity u * sum(z) at a cost of r_j = c_j / a_j per unit
    // (minimisation), so moving d units of z from `rem` to `keep` keeps every
    // row and changes the cost by d * (r_keep - r_rem). Hence:
    //  - when some column's z is unbounded above, the cheapest such one
    //    (least r) can take over any column of r at least its own: that
    //    column's z goes to its lower end, a fixing; the mirror holds for z
    //    unbounded below, the dearest one, and the upper end;
    //  - columns of equal r merge into y = x_keep + s * x_rem
    //    (s = a_rem / a_keep), whose bounds are the sum of theirs.
    // One absorber per direction and a sort for the merges keep a bucket
    // linear (plus the sort) in its size. HiGHS's parallel-columns rule
    // (detectParallelRowsAndCols) does both reductions. Continuous columns
    // only: an integer one could neither absorb a fractional move nor be
    // split back.
    Index changes = 0;
    const Real sense = lp.sense == Sense::Maximize ? -1.0 : 1.0;
    // A finite bound this large (an MPS file's 1e30 for "none") is not fixed
    // at: shifting the rows by it would swamp their other terms.
    // Every candidate is hashed, but a bucket is looked at only when one of
    // its columns is dirty: a clean column's partner may be dirty, e.g. a
    // bound of the dirty one turned finite since the pair was last seen.
    std::vector<uint8_t> dirty(static_cast<size_t>(lp.num_cols), 0);
    for (Index j : dirty_cols) {
        dirty[j] = 1;
    }
    struct Member {
        uint64_t hash;
        Index col;
        Real a;  ///< First active coefficient.
        Real r;  ///< Cost per unit of a * x.
    };
    bool any_dirty = false;
    // Sorted by hash into buckets: cheaper than a map of vectors when most
    // buckets hold one column.
    std::vector<Member> all;
    for (Index j = 0; j < lp.num_cols; ++j) {
        if (col_removed[j] || col_active_nnz[j] == 0 || lp.col_type[j] != VarType::Continuous ||
            lp.col_lower[j] > lp.col_upper[j]) {
            continue;
        }
        Real a = 0.0;
        const uint64_t h = colPatternHash(*work_, j, row_removed, a);
        if (a == 0.0) {
            continue;
        }
        if (dirty[j] != 0) {
            ++stats_.cols_examined;
            any_dirty = true;
        }
        all.push_back({h, j, a, sense * lp.obj[j] / a});
    }
    if (!any_dirty) {
        return 0;
    }
    std::sort(all.begin(), all.end(), [](const Member& p, const Member& q) {
        return p.hash != q.hash ? p.hash < q.hash : p.col < q.col;
    });

    auto fixAt = [&](Index col, Real value) {
        removeActiveColumn(*work_, col, col_removed, row_removed, row_active_nnz, col_active_nnz,
                           next_dirty_rows, next_dirty_cols);
        lp.obj_offset += lp.obj[col] * value;
        postsolve_stack_.push(PostsolveFixVariable{col, value});
        for (const auto [row, a] : work_->col(col)) {
            if (!row_removed[row]) {
                shiftRowSides(lp, row, a * value);
            }
        }
        ++changes;
        ++stats_.vars_removed;
    };
    // c_rem - s * c_keep for A_rem = s * A_keep, the cost of moving x_rem up
    // by one with x_keep making up for it, and the tolerance it is read to.
    // False when the two are not parallel after all (a hash collision).
    auto pairDiff = [&](Index keep, Index rem, Real& s, Real& diff, Real& tol) -> bool {
        if (!colsHaveScaledPattern(*work_, keep, rem, row_removed, s)) {
            return false;
        }
        const Real c_keep = sense * lp.obj[keep];
        const Real c_rem = sense * lp.obj[rem];
        diff = c_rem - s * c_keep;
        tol = 1e-12 * std::max({1.0, std::abs(c_rem), std::abs(s * c_keep)});
        return true;
    };
    // The x bound at which a * x is at its low (high) end.
    auto zLowBound = [&](const Member& m) {
        return m.a > 0.0 ? lp.col_lower[m.col] : lp.col_upper[m.col];
    };
    auto zHighBound = [&](const Member& m) {
        return m.a > 0.0 ? lp.col_upper[m.col] : lp.col_lower[m.col];
    };
    auto fixable = [](Real v) { return std::isfinite(v) && std::abs(v) < kHugeBound; };

    for (size_t begin = 0, end = 0; begin < all.size(); begin = end) {
        end = begin + 1;
        while (end < all.size() && all[end].hash == all[begin].hash) {
            ++end;
        }
        const std::span<const Member> members(all.data() + begin, end - begin);
        if (members.size() < 2 ||
            std::none_of(members.begin(), members.end(),
                         [&](const Member& m) { return dirty[m.col] != 0; })) {
            continue;
        }
        // z unbounded above: the least r takes over every column of r at
        // least its own, which goes to its low end. `up` is false for the
        // mirror: z unbounded below, the greatest r, the high end.
        for (const bool up : {true, false}) {
            const Member* absorber = nullptr;
            for (const auto& m : members) {
                // a * x unbounded in this pass's direction: x in the same
                // one for a > 0, in the other for a < 0.
                const bool x_up = (m.a > 0.0) == up;
                const bool unbounded =
                    x_up ? std::isinf(lp.col_upper[m.col]) && lp.col_upper[m.col] > 0.0
                         : std::isinf(lp.col_lower[m.col]) && lp.col_lower[m.col] < 0.0;
                if (!col_removed[m.col] && unbounded &&
                    (absorber == nullptr || (up ? m.r < absorber->r : m.r > absorber->r))) {
                    absorber = &m;
                }
            }
            if (absorber == nullptr) {
                continue;
            }
            for (const auto& m : members) {
                const Real z_end = up ? zLowBound(m) : zHighBound(m);
                Real s = 0.0;
                Real diff = 0.0;
                Real tol = 0.0;
                if (m.col == absorber->col || col_removed[m.col] || !fixable(z_end) ||
                    !pairDiff(absorber->col, m.col, s, diff, tol)) {
                    continue;
                }
                // r_m - r_absorber = diff / a_m: at least zero to go low,
                // at most zero to go high.
                const bool cheaper_to_move = (m.a > 0.0) == up ? diff >= -tol : diff <= tol;
                if (cheaper_to_move) {
                    fixAt(m.col, z_end);
                }
            }
        }
        // Equal r: merge into the first of each run, in order of r.
        std::vector<Member> left;
        for (const auto& m : members) {
            if (!col_removed[m.col]) {
                left.push_back(m);
            }
        }
        std::sort(left.begin(), left.end(), [](const Member& p, const Member& q) {
            return p.r != q.r ? p.r < q.r : p.col < q.col;
        });
        size_t head = 0;
        for (size_t i = 1; i < left.size(); ++i) {
            const Index keep = left[head].col;
            const Index rem = left[i].col;
            Real s = 0.0;
            Real diff = 0.0;
            Real tol = 0.0;
            if (!pairDiff(keep, rem, s, diff, tol)) {
                continue;
            }
            if (std::abs(diff) > tol) {
                head = i;
                continue;
            }
            const Real kl = lp.col_lower[keep];
            const Real ku = lp.col_upper[keep];
            const Real rl = lp.col_lower[rem];
            const Real ru = lp.col_upper[rem];
            postsolve_stack_.push(PostsolveParallelCol{keep, rem, s, kl, ku, rl, ru});
            lp.col_lower[keep] = s > 0.0 ? kl + s * rl : kl + s * ru;
            lp.col_upper[keep] = s > 0.0 ? ku + s * ru : ku + s * rl;
            removeActiveColumn(*work_, rem, col_removed, row_removed, row_active_nnz,
                               col_active_nnz, next_dirty_rows, next_dirty_cols);
            next_dirty_cols[keep] = 1;
            ++changes;
            ++stats_.vars_removed;
        }
    }
    return changes;
}

// =============================================================================
// Presolver — root probing
// =============================================================================

namespace {

/// Returns true when the column behaves as a 0/1 variable.
inline bool isBinaryCol(const LpProblem& lp, Index col) {
    if (lp.col_type[col] == VarType::Continuous) {
        return false;
    }
    return lp.col_lower[col] > -1e-8 && lp.col_lower[col] < 1e-8 &&
           lp.col_upper[col] > 1.0 - 1e-8 && lp.col_upper[col] < 1.0 + 1e-8;
}

}  // namespace

Index Presolver::probingPass(LpProblem& lp, const std::vector<bool>& col_removed,
                             const std::vector<bool>& row_removed,
                             std::vector<uint8_t>& next_dirty_rows,
                             std::vector<uint8_t>& next_dirty_cols) {
    // Probing view of the current reduced problem, in the same row/column
    // indexing as `lp`: removed rows are made vacuous, removed columns are
    // detached (their contribution has already been folded into the row
    // bounds) and coefficient overrides are materialized. This is exactly what
    // buildReducedProblem() produces, minus the reindexing.
    LpProblem view;
    view.sense = lp.sense;
    view.num_cols = lp.num_cols;
    view.num_rows = lp.num_rows;
    view.obj.assign(static_cast<std::size_t>(lp.num_cols), 0.0);
    view.col_lower = lp.col_lower;
    view.col_upper = lp.col_upper;
    view.col_type = lp.col_type;
    view.row_lower = lp.row_lower;
    view.row_upper = lp.row_upper;

    for (Index j = 0; j < lp.num_cols; ++j) {
        if (!col_removed[j]) {
            continue;
        }
        view.col_lower[j] = 0.0;
        view.col_upper[j] = 0.0;
    }
    for (Index i = 0; i < lp.num_rows; ++i) {
        if (!row_removed[i]) {
            continue;
        }
        view.row_lower[i] = -kInf;
        view.row_upper[i] = kInf;
    }

    std::vector<Triplet> triplets;
    for (Index i = 0; i < lp.num_rows; ++i) {
        if (row_removed[i]) {
            continue;
        }
        for (const auto [j, a] : work_->row(i)) {
            if (col_removed[j]) {
                continue;
            }
            triplets.push_back({i, j, a});
        }
    }
    view.matrix = SparseMatrix(lp.num_rows, lp.num_cols, std::move(triplets));

    // Probing runs once per presolve call, so the budget is per call, not per
    // round.
    ProbingConfig config;
    config.max_rounds = options_.probing_max_rounds;
    config.max_probes_per_round = options_.probing_max_probes_per_round;
    config.time_limit = options_.probing_time_limit;
    // Equivalence and domination output has no consumer in presolve yet.
    config.detect_equivalences = false;
    config.detect_dominated = false;
    config.learn_vubs = true;
    config.learn_implications = true;

    ImplicationGraph graph;
    VariableBoundStore vb_store;
    ProbingEngine engine;
    const ProbingStats probing_stats = engine.probe(view, graph, vb_store, config);

    // Both branches of some binary propagate to infeasibility: propagation only
    // ever tightens valid bounds, so the model itself is infeasible.
    if (probing_stats.infeasible) {
        infeasible_ = true;
        return 0;
    }

    Index changes = 0;

    // Probing fixings ride the ordinary fixed-variable path: tightening the
    // column bounds makes removeFixedVariables() remove the column and record
    // the PostsolveFixVariable operation in the next round.
    for (const auto& [var, value] : engine.fixings()) {
        if (var < 0 || var >= lp.num_cols) {
            continue;
        }
        if (col_removed[var]) {
            continue;
        }
        if (value < lp.col_lower[var] - kTol || value > lp.col_upper[var] + kTol) {
            infeasible_ = true;
            return changes;
        }
        if (lp.col_upper[var] - lp.col_lower[var] <= kTol) {
            continue;  // Already fixed.
        }

        lp.col_lower[var] = value;
        lp.col_upper[var] = value;
        next_dirty_cols[var] = 1;
        markRowsTouchingCol(*work_, var, row_removed, next_dirty_rows);
        ++changes;
        ++stats_.probing_fixings;
    }

    changes += strengthenCoefficientsFromVarBounds(lp, vb_store, col_removed, row_removed,
                                                   next_dirty_rows, next_dirty_cols);

    return changes;
}

Index Presolver::strengthenCoefficientsFromVarBounds(LpProblem& lp,
                                                     const VariableBoundStore& vb_store,
                                                     const std::vector<bool>& col_removed,
                                                     const std::vector<bool>& row_removed,
                                                     std::vector<uint8_t>& next_dirty_rows,
                                                     std::vector<uint8_t>& next_dirty_cols) {
    if (vb_store.numVUBs() == 0 && vb_store.numVLBs() == 0) {
        return 0;
    }

    Index changes = 0;
    std::vector<Index> row_cols;
    std::vector<Real> row_vals;

    for (Index i = 0; i < lp.num_rows; ++i) {
        if (row_removed[i]) {
            continue;
        }
        // One-sided <= rows only: the reduction rewrites a coefficient, which
        // would also move the other side of a ranged or equality row.
        if (std::isinf(lp.row_upper[i])) {
            continue;
        }
        if (!std::isinf(lp.row_lower[i])) {
            continue;
        }

        row_cols.clear();
        row_vals.clear();
        for (const auto [j, a] : work_->row(i)) {
            if (col_removed[j]) {
                continue;
            }
            row_cols.push_back(j);
            row_vals.push_back(a);
        }
        if (row_cols.size() < 2) {
            continue;
        }

        Real rhs = lp.row_upper[i];
        for (std::size_t k = 0; k < row_cols.size(); ++k) {
            const Index j = row_cols[k];
            if (!isBinaryCol(lp, j)) {
                continue;
            }

            const auto res = vb_store.strengthenCoefficient(j, row_vals[k], rhs, row_cols, row_vals,
                                                            lp.col_lower, lp.col_upper);
            if (!res.strengthened) {
                continue;
            }

            const Real new_rhs = rhs + res.rhs_delta;
            if (!std::isfinite(new_rhs) || !std::isfinite(res.new_coeff)) {
                continue;
            }

            postsolve_stack_.push(
                PostsolveCoeffTightening{i, j, row_vals[k], res.new_coeff, rhs, new_rhs});

            work_->set(i, j, res.new_coeff);
            row_vals[k] = res.new_coeff;
            rhs = new_rhs;
            lp.row_upper[i] = new_rhs;

            markRowsTouchingCol(*work_, j, row_removed, next_dirty_rows);
            markColsInRow(*work_, i, col_removed, next_dirty_cols);
            ++changes;
            ++stats_.coeffs_tightened;
            ++stats_.probing_coeff_strengthenings;
        }
    }

    return changes;
}

Index Presolver::tightenCoefficients(LpProblem& lp, std::vector<bool>& col_removed,
                                     std::vector<bool>& row_removed,
                                     const std::vector<Index>& dirty_rows,
                                     std::vector<uint8_t>& next_dirty_rows,
                                     std::vector<uint8_t>& next_dirty_cols) {
    Index changes = 0;

    for (Index i : dirty_rows) {
        if (row_removed[i]) {
            continue;
        }

        auto tightenOneSided = [&](bool use_upper_side) {
            Real rhs = 0.0;
            // One-sided rows only: the reduction rewrites a coefficient, which
            // would also move the other side of a ranged or equality row.
            if (use_upper_side) {
                if (std::isinf(lp.row_upper[i]) || !std::isinf(lp.row_lower[i])) {
                    return;
                }
                rhs = lp.row_upper[i];
            } else {
                if (!std::isinf(lp.row_upper[i])) {
                    return;
                }
                if (std::isinf(lp.row_lower[i])) {
                    return;
                }
                rhs = -lp.row_lower[i];
            }

            // Savelsbergh's coefficient tightening, in transformed <= form
            // a_j x_j + rest <= b. Split an integer column at its upper bound:
            // if the row is slack by d > 0 whenever x_j <= u_j - 1, the row
            // (a_j - d) x_j + rest <= b - d u_j cuts off the same integer points
            // at x_j = u_j and stays redundant below it. A negative coefficient
            // is the mirror image at the lower bound. Same arithmetic and
            // margin as strengthenCut in the cut code.
            constexpr Real kMarginRel = 1e-9;
            constexpr Real kMarginAbs = 1e-9;
            constexpr Real kMinReductionRel = 1e-6;
            constexpr Real kIntBoundTol = 1e-6;
            constexpr Real kRedundantTol = 1e-6;
            // The split needs integral bounds; an integer column's integer
            // points lie within its rounded-in bounds anyway.
            auto lowerOf = [&](Index j) {
                const Real lb = lp.col_lower[j];
                return lp.col_type[j] == VarType::Continuous ? lb : std::ceil(lb - kIntBoundTol);
            };
            auto upperOf = [&](Index j) {
                const Real ub = lp.col_upper[j];
                return lp.col_type[j] == VarType::Continuous ? ub : std::floor(ub + kIntBoundTol);
            };

            Real total_max = 0.0;
            Real magnitude = std::abs(rhs);
            for (const auto [j, a_orig] : work_->row(i)) {
                if (col_removed[j]) {
                    continue;
                }

                const Real a = use_upper_side ? a_orig : -a_orig;
                const Real contrib = (a > 0.0) ? a * upperOf(j) : a * lowerOf(j);
                if (!std::isfinite(contrib)) {
                    return;
                }
                total_max += contrib;
                magnitude += std::abs(contrib);
            }
            // A row the bounds already satisfy is redundant, and every
            // reduction would shrink its coefficients towards zero. Otherwise
            // b < total_max, so each d below stays short of its coefficient.
            // A row redundant only within the feasibility tolerance is left
            // alone too: tightening it would scale that violation up to a
            // full one on coefficients near zero.
            if (total_max <= rhs + kRedundantTol) {
                return;
            }
            const Real margin = kMarginRel * magnitude + kMarginAbs;

            // Each reduction lowers total_max and the rhs by the same amount,
            // so the slack the later columns see is unchanged.
            // Changing a value while walking the row is fine; nothing here
            // inserts into it.
            for (const auto [j, a_orig] : work_->row(i)) {
                if (col_removed[j]) {
                    continue;
                }
                if (lp.col_type[j] == VarType::Continuous) {
                    continue;
                }
                const Real lb = lowerOf(j);
                const Real ub = upperOf(j);
                if (ub - lb < 0.5) {
                    continue;  // Fixed: no x_j strictly inside the bounds.
                }

                const Real a = use_upper_side ? a_orig : -a_orig;

                Real new_a = a;
                Real new_rhs = rhs;
                if (a > 0.0) {
                    // Largest activity with x_j <= ub - 1, against the rhs.
                    const Real rest = total_max - a * ub;
                    const Real d = rhs - a * (ub - 1.0) - rest - margin;
                    if (d <= kMinReductionRel * a) {
                        continue;
                    }
                    new_a = a - d;
                    new_rhs = rhs - d * ub;
                } else if (a < 0.0) {
                    // Largest activity with x_j >= lb + 1, against the rhs.
                    const Real rest = total_max - a * lb;
                    const Real d = rhs - a * (lb + 1.0) - rest - margin;
                    if (d <= kMinReductionRel * -a) {
                        continue;
                    }
                    new_a = a + d;
                    new_rhs = rhs + d * lb;
                } else {
                    continue;
                }

                const Real new_a_orig = use_upper_side ? new_a : -new_a;
                if (!std::isfinite(new_rhs) || !std::isfinite(new_a_orig)) {
                    continue;
                }
                const Real old_rhs_bound = use_upper_side ? lp.row_upper[i] : lp.row_lower[i];
                const Real new_rhs_bound = use_upper_side ? new_rhs : -new_rhs;

                postsolve_stack_.push(PostsolveCoeffTightening{i, j, a_orig, new_a_orig,
                                                               old_rhs_bound, new_rhs_bound});

                total_max += new_rhs - rhs;
                rhs = new_rhs;
                if (use_upper_side) {
                    lp.row_upper[i] = new_rhs;
                } else {
                    lp.row_lower[i] = -new_rhs;
                }

                work_->set(i, j, new_a_orig);
                // Coefficient changes must be visible to later passes.
                markRowsTouchingCol(*work_, j, row_removed, next_dirty_rows);
                markColsInRow(*work_, i, col_removed, next_dirty_cols);
                ++changes;
                ++stats_.coeffs_tightened;
            }
        };

        tightenOneSided(true);
        tightenOneSided(false);
    }

    return changes;
}

// =============================================================================
// Presolver — main entry point
// =============================================================================

LpProblem Presolver::presolve(const LpProblem& problem) {
    // The reductions read every column as an interval and every constraint as
    // a row, so a semi column's {0} and SOS or indicator constraints would be
    // dropped silently (#231). MipSolver linearizes these before presolve.
    if (hasAdvancedModelFeatures(problem)) {
        throw std::invalid_argument(
            "Presolver: model has semi columns, SOS or indicator constraints; "
            "linearize it with linearizeModelFeatures first");
    }
    auto t0 = std::chrono::steady_clock::now();

    // Reset state so one Presolver instance can be reused safely.
    postsolve_stack_.clear();
    col_mapping_.clear();
    orig_num_cols_ = 0;
    stats_ = PresolveStats{};
    infeasible_ = false;
    probing_done_ = false;

    // Work on a copy.
    LpProblem lp = problem;
    orig_num_cols_ = lp.num_cols;
    // lp.matrix stays the input's; the working matrix carries every change.
    PresolveMatrix work(lp.matrix);
    struct WorkScope {
        PresolveMatrix*& slot;
        ~WorkScope() { slot = nullptr; }
    } work_scope{work_};
    work_ = &work;

    // Round integer bounds in before any reduction reasons from them, so a
    // column fixed at a bound is fixed at an integer.
    infeasible_ = !roundIntegerBounds(lp);

    row_scale_.assign(lp.num_rows, 1.0);
    for (Index i = 0; i < lp.num_rows; ++i) {
        for (const Real side : {lp.row_lower[i], lp.row_upper[i]}) {
            if (!std::isinf(side)) {
                row_scale_[i] = std::max(row_scale_[i], std::abs(side));
            }
        }
    }

    std::vector<bool> col_removed(lp.num_cols, false);
    std::vector<bool> row_removed(lp.num_rows, false);
    std::vector<Index> row_active_nnz(lp.num_rows, 0);
    std::vector<Index> col_active_nnz(lp.num_cols, 0);
    for (Index i = 0; i < lp.num_rows; ++i) {
        row_active_nnz[i] = work.rowSize(i);
    }
    for (Index j = 0; j < lp.num_cols; ++j) {
        col_active_nnz[j] = work.colSize(j);
    }
    std::vector<uint8_t> dirty_rows(lp.num_rows, 1);
    std::vector<uint8_t> dirty_cols(lp.num_cols, 1);

    // Run one reduction, adding its wall time to `seconds`.
    const auto timed = [](Real& seconds, auto&& reduce) {
        const auto start = std::chrono::steady_clock::now();
        const Index changes = reduce();
        seconds += std::chrono::duration<Real>(std::chrono::steady_clock::now() - start).count();
        return changes;
    };

    const bool lp_only = std::none_of(lp.col_type.begin(), lp.col_type.end(),
                                      [](VarType t) { return t != VarType::Continuous; });
    // Dominated columns read LP duals, so they run on an LP only (#376).
    const bool dominated_columns = options_.enable_dominated_columns && lp_only;
    // HiGHS runs dependent equations on an LP only, too.
    bool dependent_equations_done = false;

    // Iterative reduction loop.
    for (Index round = 0; round < max_rounds_ && !infeasible_; ++round) {
        Index total_changes = 0;
        std::vector<Index> dirty_row_list;
        std::vector<Index> dirty_col_list;
        dirty_row_list.reserve(lp.num_rows);
        dirty_col_list.reserve(lp.num_cols);
        for (Index i = 0; i < lp.num_rows; ++i) {
            if (dirty_rows[i]) {
                dirty_row_list.push_back(i);
            }
        }
        for (Index j = 0; j < lp.num_cols; ++j) {
            if (dirty_cols[j]) {
                dirty_col_list.push_back(j);
            }
        }
        if (dirty_row_list.empty() && dirty_col_list.empty()) {
            break;
        }

        std::vector<uint8_t> next_dirty_rows(lp.num_rows, 0);
        std::vector<uint8_t> next_dirty_cols(lp.num_cols, 0);

        Index ch = timed(stats_.fixed_var_seconds, [&] {
            return removeFixedVariables(lp, col_removed, row_removed, row_active_nnz,
                                        col_active_nnz, dirty_col_list, next_dirty_rows,
                                        next_dirty_cols);
        });
        stats_.fixed_var_changes += ch;
        total_changes += ch;

        ch = timed(stats_.singleton_row_seconds, [&] {
            return removeSingletonRows(lp, col_removed, row_removed, row_active_nnz, col_active_nnz,
                                       dirty_row_list, next_dirty_rows, next_dirty_cols);
        });
        stats_.singleton_row_changes += ch;
        total_changes += ch;

        ch = timed(stats_.singleton_col_seconds, [&] {
            return removeSingletonCols(lp, col_removed, row_removed, row_active_nnz, col_active_nnz,
                                       dirty_col_list, next_dirty_rows, next_dirty_cols);
        });
        stats_.singleton_col_changes += ch;
        total_changes += ch;

        if (options_.enable_free_col_substitution) {
            ch = timed(stats_.free_col_substitution_seconds, [&] {
                return substituteFreeColSingletons(lp, col_removed, row_removed, row_active_nnz,
                                                   col_active_nnz, dirty_col_list, next_dirty_rows,
                                                   next_dirty_cols);
            });
            stats_.free_col_substitution_changes += ch;
            total_changes += ch;
        }

        // After free column substitution, as in HiGHS's singletonCol: an
        // implied-free slack takes its row with it there.
        if (options_.enable_zero_cost_singletons) {
            ch = timed(stats_.zero_cost_singleton_seconds, [&] {
                return removeZeroCostSingletons(lp, col_removed, row_removed, row_active_nnz,
                                                col_active_nnz, dirty_col_list, next_dirty_rows,
                                                next_dirty_cols);
            });
            stats_.zero_cost_singleton_changes += ch;
            total_changes += ch;
        }

        if (options_.enable_doubleton_aggregation) {
            ch = timed(stats_.doubleton_eq_seconds, [&] {
                return aggregateDoubletonEqualities(lp, col_removed, row_removed, row_active_nnz,
                                                    col_active_nnz, dirty_row_list, next_dirty_rows,
                                                    next_dirty_cols);
            });
            stats_.doubleton_eq_changes += ch;
            total_changes += ch;
        }

        if (options_.enable_forcing_rows) {
            ch = timed(stats_.forcing_row_seconds, [&] {
                return removeForcingRows(lp, col_removed, row_removed, row_active_nnz,
                                         col_active_nnz, dirty_row_list, next_dirty_rows,
                                         next_dirty_cols);
            });
            stats_.forcing_row_changes += ch;
            total_changes += ch;
        }

        ch = timed(stats_.dominated_row_seconds, [&] {
            return removeDominatedRows(lp, col_removed, row_removed, row_active_nnz, col_active_nnz,
                                       dirty_row_list, next_dirty_rows, next_dirty_cols);
        });
        stats_.dominated_row_changes += ch;
        total_changes += ch;

        ch = timed(stats_.duplicate_row_seconds, [&] {
            return removeDuplicateRows(lp, col_removed, row_removed, row_active_nnz, col_active_nnz,
                                       dirty_row_list, next_dirty_rows, next_dirty_cols);
        });
        stats_.duplicate_row_changes += ch;
        total_changes += ch;

        if (options_.enable_parallel_rows) {
            ch = timed(stats_.parallel_row_seconds, [&] {
                return removeParallelRows(lp, col_removed, row_removed, row_active_nnz,
                                          col_active_nnz, dirty_row_list, next_dirty_rows,
                                          next_dirty_cols);
            });
            stats_.parallel_row_changes += ch;
            total_changes += ch;
        }

        if (options_.enable_parallel_cols) {
            ch = timed(stats_.parallel_col_seconds, [&] {
                return removeParallelCols(lp, col_removed, row_removed, row_active_nnz,
                                          col_active_nnz, dirty_col_list, next_dirty_rows,
                                          next_dirty_cols);
            });
            stats_.parallel_col_changes += ch;
            total_changes += ch;
        }

        ch = timed(stats_.implied_equation_seconds, [&] {
            return detectImpliedEquations(lp, col_removed, row_removed, dirty_row_list,
                                          next_dirty_rows, next_dirty_cols);
        });
        stats_.implied_equation_changes += ch;
        total_changes += ch;

        ch = timed(stats_.activity_bound_tightening_seconds, [&] {
            return activityBoundTightening(lp, col_removed, row_removed, dirty_row_list,
                                           next_dirty_rows, next_dirty_cols);
        });
        stats_.activity_bound_tightening_changes += ch;
        total_changes += ch;

        if (dominated_columns) {
            ch = timed(stats_.dominated_col_seconds, [&] {
                return removeDominatedColumns(lp, col_removed, row_removed, row_active_nnz,
                                              col_active_nnz, next_dirty_rows, next_dirty_cols);
            });
            stats_.dominated_col_changes += ch;
            total_changes += ch;
        }

        if (options_.enable_dual_fixing) {
            ch = timed(stats_.dual_fixing_seconds, [&] {
                return dualFixing(lp, col_removed, row_removed, row_active_nnz, col_active_nnz,
                                  dirty_col_list, next_dirty_rows, next_dirty_cols);
            });
            stats_.dual_fixing_changes += ch;
            total_changes += ch;
        }

        ch = timed(stats_.empty_col_seconds, [&] {
            return removeEmptyColumns(lp, col_removed, row_removed, row_active_nnz, col_active_nnz,
                                      dirty_col_list, next_dirty_rows, next_dirty_cols);
        });
        stats_.empty_col_changes += ch;
        total_changes += ch;

        if (options_.enable_coefficient_tightening) {
            ch = timed(stats_.coeff_tightening_seconds, [&] {
                return tightenCoefficients(lp, col_removed, row_removed, dirty_row_list,
                                           next_dirty_rows, next_dirty_cols);
            });
            stats_.coeff_tightening_changes += ch;
            total_changes += ch;
        }

        // The aggregator runs where HiGHS runs it, once the other reductions
        // have reached their fixpoint, and the loop goes on if it changes
        // anything; and in the last round, so it runs where the rounds run
        // out first.
        if (options_.enable_aggregator && (total_changes == 0 || round + 1 == max_rounds_)) {
            ch = timed(stats_.aggregator_seconds, [&] {
                return aggregateImpliedFreeColumns(lp, col_removed, row_removed, row_active_nnz,
                                                   col_active_nnz, next_dirty_rows,
                                                   next_dirty_cols);
            });
            stats_.aggregator_changes += ch;
            total_changes += ch;
        }

        // Probing is the most expensive pass, so it runs once, on the first
        // round where the cheap reductions have reached their fixpoint. Any
        // fixing it finds re-triggers the loop.
        if (options_.enable_probing && !probing_done_ && total_changes == 0) {
            probing_done_ = true;
            ch = timed(stats_.probing_seconds, [&] {
                return probingPass(lp, col_removed, row_removed, next_dirty_rows, next_dirty_cols);
            });
            stats_.probing_changes += ch;
            total_changes += ch;
            if (infeasible_) {
                ++stats_.rounds;
                if (total_changes > 0) {
                    ++stats_.rounds_with_changes;
                }
                break;
            }
        }

        // Dependent equations run where HiGHS runs them, on an LP, once the
        // loop's other reductions, the aggregator and probing included, have
        // stalled (or the rounds run out), and once. The loop goes on only
        // if they shrink the problem by more than HiGHS's 0.05%
        // (problemSizeReduction, a percentage of the rows here, as only rows
        // go).
        if (options_.enable_dependent_equations && lp_only && !dependent_equations_done &&
            (total_changes == 0 || round + 1 == max_rounds_)) {
            dependent_equations_done = true;
            const Index active_rows =
                static_cast<Index>(std::count(row_removed.begin(), row_removed.end(), false));
            ch = timed(stats_.dependent_equation_seconds, [&] {
                return removeDependentEquations(lp, col_removed, row_removed, row_active_nnz,
                                                col_active_nnz, next_dirty_rows, next_dirty_cols);
            });
            stats_.dependent_equation_changes += ch;
            if (100.0 * static_cast<Real>(ch) > 0.05 * static_cast<Real>(active_rows)) {
                total_changes += ch;
            }
        }

        ++stats_.rounds;
        if (total_changes > 0) {
            ++stats_.rounds_with_changes;
        }

        // Check for infeasibility: any non-removed variable with lb > ub.
        for (Index j = 0; j < lp.num_cols; ++j) {
            if (!col_removed[j] && lp.col_lower[j] > lp.col_upper[j] + kTol) {
                infeasible_ = true;
                break;
            }
        }
        if (infeasible_) {
            break;
        }

        // Check for infeasible empty rows.
        for (Index i = 0; i < lp.num_rows; ++i) {
            if (row_removed[i]) {
                continue;
            }
            if (row_active_nnz[i] == 0) {
                // Empty row: check bounds.
                if (lp.row_lower[i] > rowTol(i) || lp.row_upper[i] < -rowTol(i)) {
                    infeasible_ = true;
                    break;
                }
            }
        }
        if (infeasible_) {
            break;
        }

        if (total_changes == 0) {
            break;
        }

        dirty_rows.swap(next_dirty_rows);
        dirty_cols.swap(next_dirty_cols);

        // Keep progress even when only removals happened.
        bool has_dirty = false;
        for (uint8_t f : dirty_rows) {
            if (f) {
                has_dirty = true;
                break;
            }
        }
        if (!has_dirty) {
            for (uint8_t f : dirty_cols) {
                if (f) {
                    has_dirty = true;
                    break;
                }
            }
        }
        if (!has_dirty) {
            break;
        }
    }

    if (!infeasible_ && std::none_of(lp.col_type.begin(), lp.col_type.end(),
                                     [](VarType t) { return t != VarType::Continuous; })) {
        if (options_.enable_dominated_columns) {
            stats_.implied_row_dual_bounds = timed(stats_.dual_bounds_seconds, [&] {
                DualBounds bounds;
                bounds.compute(lp, *work_, col_removed, row_removed);
                return bounds.numImpliedRowDualBounds();
            });
        }
        stats_.implied_bounds_relaxed = timed(stats_.implied_bounds_relaxed_seconds, [&] {
            return relaxImpliedBounds(lp, problem, col_removed, row_removed);
        });
    }

    stats_.time_seconds =
        std::chrono::duration<Real>(std::chrono::steady_clock::now() - t0).count();

    return buildReducedProblem(lp, col_removed, row_removed);
}

// Activity bound tightening and singleton rows write the bounds they derive
// into the LP, so a half-bounded column comes out boxed. The barrier gives
// every boxed column a row of its own and factors a dense m x m matrix, so on
// ship08l the 2706 bounds activity tightening made explicit took its standard
// form from 680 rows to 3197 and its solve from 0.6 s to 150 s (#279).
//
// A tightened bound that a remaining row implies, given the other columns'
// current bounds, cuts nothing off the reduced LP, so putting the original
// bound back leaves its feasible set, and so postsolve, as they were (up to
// the check's 1e-9 relative tolerance, far inside the solvers'). Each
// column is put back against the bounds as they stand after the earlier
// ones, never against bounds this pass has already loosened: two columns
// whose tightened bounds each follow from the other's would otherwise both
// lose theirs. Integer columns keep their bounds, and the caller runs this
// only on an LP: a MIP's search and cuts reason from the tightened bounds.
Index Presolver::relaxImpliedBounds(LpProblem& lp, const LpProblem& original,
                                    const std::vector<bool>& col_removed,
                                    const std::vector<bool>& row_removed) {
    // One of a row's two activities over its active columns, the minimum or
    // the maximum: the count of infinite terms and the finite terms' running
    // sum, with what bounds the sum's rounding. `added` counts the products
    // added and `magnitude` sums their absolute values, removals included:
    // a running sum of n rounded products, whatever their signs and order,
    // is off by at most gamma_n * magnitude, gamma_n = n*u/(1-n*u) with unit
    // roundoff u = eps/2 (the inner-product bound, Higham, Accuracy and Stability of Numerical
    // Algorithms, 2nd ed., section 3.1). #311: a column's check then costs
    // O(1) per row; the full rescan runs only when that bound leaves the
    // check undecided.
    struct Activity {
        Real finite = 0.0;
        Real magnitude = 0.0;
        Index added = 0;
        Index inf = 0;
    };
    struct RowActivity {
        Activity min;
        Activity max;
    };
    std::vector<RowActivity> act(lp.num_rows);
    // A finite bound this large (an MPS file's 1e30 for "none") counts as
    // infinite: as a finite term it would swamp the row's other terms, and
    // the rounding could imply a bound the row does not. Counting it as
    // infinite only keeps more bounds.
    auto addProduct = [](Activity& s, Real a, Real b, int sign) {
        if (std::abs(b) >= kHugeBound) {
            s.inf += sign;
            return;
        }
        const Real p = sign * a * b;
        s.finite += p;
        s.magnitude += std::abs(p);
        ++s.added;
    };
    // Adds (sign +1) or removes (sign -1) the term a*x of a row's activities
    // for x in [lb, ub].
    auto addTerm = [&](RowActivity& r, Real a, Real lb, Real ub, int sign) {
        if (a == 0.0) {
            return;
        }
        addProduct(r.min, a, a > 0.0 ? lb : ub, sign);
        addProduct(r.max, a, a > 0.0 ? ub : lb, sign);
    };
    for (Index i = 0; i < lp.num_rows; ++i) {
        if (row_removed[i]) {
            continue;
        }
        for (const auto [j, a] : work_->row(i)) {
            if (!col_removed[j]) {
                addTerm(act[i], a, lp.col_lower[j], lp.col_upper[j], 1);
            }
        }
    }

    // Whether a remaining row bounds column j by `bound` or tighter from
    // above (upper) or below, the column's own term taken out, from a fresh
    // scan of the row without column j, with the scan's rounding error
    // counted against it. The running activities settle nearly every check;
    // this decides the ones they leave within their error bound. Their bound
    // counts the column's own term, the rescan's does not: taking a large
    // term out of a running sum rounds away the small ones beside it (x <=
    // 1e12 in a row leaves its other terms good to about 1e-4).
    constexpr Real kEps = std::numeric_limits<Real>::epsilon();
    auto rowImplies = [&](Index i, Index j, Real a, bool use_upper_side, bool upper, Real bound,
                          Real tol) {
        const Real side = use_upper_side ? lp.row_upper[i] : lp.row_lower[i];
        Real rest = 0.0;
        Real magnitude = std::abs(side);
        Index terms = 0;
        for (const auto [jj, aa] : work_->row(i)) {
            if (jj == j || col_removed[jj]) {
                continue;
            }
            if (aa == 0.0) {
                continue;
            }
            // The end of column jj that gives the row's minimum activity
            // (upper side) or maximum (lower side).
            const Real b = (aa > 0.0) == use_upper_side ? lp.col_lower[jj] : lp.col_upper[jj];
            if (std::abs(b) >= kHugeBound) {
                return false;
            }
            rest += aa * b;
            magnitude += std::abs(aa * b);
            ++terms;
        }
        const Real value = (side - rest) / a;
        const Real err = (static_cast<Real>(terms) + 2.0) * kEps * magnitude / std::abs(a) +
                         2.0 * kEps * std::abs(value);
        return upper ? value + err <= bound + tol : value - err >= bound - tol;
    };

    // The relative bound misses a product that underflows; this much per
    // product covers it.
    constexpr Real kTiny = std::numeric_limits<Real>::min();
    auto implied = [&](Index j, bool upper, Real bound) {
        const Real tol = 1e-9 * std::max<Real>(1.0, std::abs(bound));
        for (const auto [i, a] : work_->col(j)) {
            if (row_removed[i]) {
                continue;
            }
            if (a == 0.0) {
                continue;
            }
            // a*x <= row_upper - rest_min and a*x >= row_lower - rest_max;
            // dividing by a < 0 swaps which side bounds x from above.
            const bool use_upper_side = upper == (a > 0.0);
            const Real side = use_upper_side ? lp.row_upper[i] : lp.row_lower[i];
            if (std::abs(side) >= kHugeBound) {
                continue;
            }
            // Column j's own term in that activity, taken back out.
            Activity rest = use_upper_side ? act[i].min : act[i].max;
            const Real own = (a > 0.0) == use_upper_side ? lp.col_lower[j] : lp.col_upper[j];
            addProduct(rest, a, own, -1);
            if (rest.inf > 0) {
                continue;
            }
            // The side is one more term, side - rest one more addition. err
            // bounds the sum's rounding over |a| and the division's; 2*n*eps
            // is 4*n*u, room for gamma_n over n*u and the rounding of
            // magnitude itself.
            const Real n = static_cast<Real>(rest.added + 1);
            const Real magnitude = rest.magnitude + std::abs(side);
            const Real value = (side - rest.finite) / a;
            const Real err = 2.0 * kEps * std::abs(value) +
                             2.0 * n * kEps * magnitude / std::abs(a) + n * kTiny / std::abs(a);
            if (upper ? value + err <= bound + tol : value - err >= bound - tol) {
                return true;
            }
            // Within err of the test: the rescan decides, so no bound it
            // would put back is kept.
            if ((upper ? value - err <= bound + tol : value + err >= bound - tol) &&
                rowImplies(i, j, a, use_upper_side, upper, bound, tol)) {
                return true;
            }
        }
        return false;
    };

    Index relaxed = 0;
    for (Index j = 0; j < lp.num_cols; ++j) {
        if (col_removed[j]) {
            continue;
        }
        for (const bool upper : {true, false}) {
            const Real current = upper ? lp.col_upper[j] : lp.col_lower[j];
            const Real orig = upper ? original.col_upper[j] : original.col_lower[j];
            const bool tightened = upper ? current < orig : current > orig;
            if (!tightened || !implied(j, upper, current)) {
                continue;
            }
            (upper ? lp.col_upper[j] : lp.col_lower[j]) = orig;
            for (const auto [i, a] : work_->col(j)) {
                if (row_removed[i] || a == 0.0) {
                    continue;
                }
                // Only the activity that reads the changed bound moves: the
                // maximum for an upper bound with a > 0 or a lower with a < 0.
                Activity& s = upper == (a > 0.0) ? act[i].max : act[i].min;
                addProduct(s, a, current, -1);
                addProduct(s, a, orig, 1);
            }
            ++relaxed;
        }
    }
    return relaxed;
}

LpProblem Presolver::buildReducedProblem(const LpProblem& lp, const std::vector<bool>& col_removed,
                                         const std::vector<bool>& row_removed) {
    // Build column mapping: new index -> original index.
    col_mapping_.clear();
    std::vector<Index> orig_to_new(lp.num_cols, -1);

    for (Index j = 0; j < lp.num_cols; ++j) {
        if (!col_removed[j]) {
            orig_to_new[j] = static_cast<Index>(col_mapping_.size());
            col_mapping_.push_back(j);
        }
    }

    // Build row mapping.
    std::vector<Index> row_mapping;  // new -> orig
    std::vector<Index> orig_row_to_new(lp.num_rows, -1);
    for (Index i = 0; i < lp.num_rows; ++i) {
        if (!row_removed[i]) {
            orig_row_to_new[i] = static_cast<Index>(row_mapping.size());
            row_mapping.push_back(i);
        }
    }

    Index new_cols = static_cast<Index>(col_mapping_.size());
    Index new_rows = static_cast<Index>(row_mapping.size());

    LpProblem result;
    result.name = lp.name;
    result.sense = lp.sense;
    result.obj_offset = lp.obj_offset;
    result.num_cols = new_cols;
    result.num_rows = new_rows;

    // Copy column data.
    result.obj.resize(new_cols);
    result.col_lower.resize(new_cols);
    result.col_upper.resize(new_cols);
    result.col_type.resize(new_cols);
    result.col_names.resize(new_cols);

    for (Index jj = 0; jj < new_cols; ++jj) {
        Index j = col_mapping_[jj];
        result.obj[jj] = lp.obj[j];
        result.col_lower[jj] = lp.col_lower[j];
        result.col_upper[jj] = lp.col_upper[j];
        result.col_type[jj] = lp.col_type[j];
        if (j < static_cast<Index>(lp.col_names.size())) {
            result.col_names[jj] = lp.col_names[j];
        }
    }

    // Copy row data.
    result.row_lower.resize(new_rows);
    result.row_upper.resize(new_rows);
    result.row_names.resize(new_rows);

    for (Index ii = 0; ii < new_rows; ++ii) {
        Index i = row_mapping[ii];
        result.row_lower[ii] = lp.row_lower[i];
        result.row_upper[ii] = lp.row_upper[i];
        if (i < static_cast<Index>(lp.row_names.size())) {
            result.row_names[ii] = lp.row_names[i];
        }
    }

    // Build new matrix.
    std::vector<Triplet> triplets;
    for (Index ii = 0; ii < new_rows; ++ii) {
        Index orig_row = row_mapping[ii];
        for (const auto [orig_col, value] : work_->row(orig_row)) {
            if (col_removed[orig_col]) {
                continue;
            }
            triplets.push_back({ii, orig_to_new[orig_col], value});
        }
    }

    result.matrix = SparseMatrix(new_rows, new_cols, std::move(triplets));

    return result;
}

std::vector<Real> Presolver::postsolve(const std::vector<Real>& presolved_solution) const {
    return postsolve_stack_.postsolve(presolved_solution, col_mapping_, orig_num_cols_);
}

bool pointHoldsModel(const LpProblem& model, std::span<const Real> x) {
    return static_cast<Index>(x.size()) == model.num_cols &&
           heuristic_detail::isFeasiblePoint(model, x);
}

Real pointViolation(const LpProblem& model, std::span<const Real> x) {
    if (static_cast<Index>(x.size()) != model.num_cols) {
        return kInf;
    }
    Real worst = 0.0;
    for (Index j = 0; j < model.num_cols; ++j) {
        if (!std::isfinite(x[j])) {
            return kInf;
        }
        worst = std::max({worst, model.col_lower[j] - x[j], x[j] - model.col_upper[j]});
    }
    for (Index i = 0; i < model.num_rows; ++i) {
        const auto row = model.matrix.row(i);
        Real activity = 0.0;
        for (Index k = 0; k < row.size(); ++k) {
            activity += row.values[k] * x[row.indices[k]];
        }
        worst = std::max({worst, model.row_lower[i] - activity, activity - model.row_upper[i]});
    }
    return worst;
}

}  // namespace mipx
