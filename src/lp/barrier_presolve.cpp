#include "mipx/barrier_presolve.h"

#include <cmath>
#include <utility>

namespace mipx {

namespace {

inline bool isFinite(Real v) {
    return std::isfinite(v);
}

}  // namespace

// ============================================================================
// Individual reductions
// ============================================================================

Index BarrierPresolver::removeFixedVariables(LpProblem& lp, std::vector<bool>& col_removed,
                                             std::vector<bool>& row_removed,
                                             std::vector<Index>& row_nnz,
                                             std::vector<Index>& col_nnz) {
    Index changes = 0;
    for (Index j = 0; j < lp.num_cols; ++j) {
        if (col_removed[j]) {
            continue;
        }
        Real lb = lp.col_lower[j];
        Real ub = lp.col_upper[j];
        if (!isFinite(lb) || !isFinite(ub)) {
            continue;
        }
        if (ub < lb - kTol) {
            infeasible_ = true;
            return changes;
        }
        if (ub - lb > kTol) {
            continue;
        }

        // Fixed variable: x_j = lb (or midpoint).
        Real val = 0.5 * (lb + ub);

        // Record postsolve op with row info for dual reconstruction.
        BpFixVariable op;
        op.orig_col = j;
        op.value = val;
        op.obj_coeff = lp.obj[j];

        auto cv = lp.matrix.col(j);
        for (Index k = 0; k < cv.size(); ++k) {
            Index row = cv.indices[k];
            if (row_removed[row]) {
                continue;
            }
            Real coeff = cv.values[k];
            op.row_indices.push_back(row);
            op.row_coeffs.push_back(coeff);
        }
        ops_.push_back(std::move(op));

        // Adjust RHS of affected rows.
        for (Index k = 0; k < cv.size(); ++k) {
            Index row = cv.indices[k];
            if (row_removed[row]) {
                continue;
            }
            Real coeff = cv.values[k];
            if (isFinite(lp.row_lower[row])) {
                lp.row_lower[row] -= coeff * val;
            }
            if (isFinite(lp.row_upper[row])) {
                lp.row_upper[row] -= coeff * val;
            }
            --row_nnz[row];
        }

        // Adjust objective offset.
        obj_offset_delta_ += lp.obj[j] * val;

        col_removed[j] = true;
        col_nnz[j] = 0;
        ++changes;
    }

    stats_.fixed_vars += changes;
    return changes;
}

Index BarrierPresolver::removeSingletonRows(LpProblem& lp, std::vector<bool>& col_removed,
                                            std::vector<bool>& row_removed,
                                            std::vector<Index>& row_nnz,
                                            std::vector<Index>& col_nnz) {
    Index changes = 0;
    for (Index i = 0; i < lp.num_rows; ++i) {
        if (row_removed[i]) {
            continue;
        }
        if (row_nnz[i] != 1) {
            continue;
        }

        // Find the single active column.
        auto rv = lp.matrix.row(i);
        Index col = -1;
        Real coeff = 0.0;
        for (Index k = 0; k < rv.size(); ++k) {
            if (!col_removed[rv.indices[k]]) {
                col = rv.indices[k];
                coeff = rv.values[k];
                break;
            }
        }
        if (col < 0 || std::abs(coeff) < kTol) {
            continue;
        }

        // Derive implied bounds on x_col from the row.
        Real implied_lb = -kInf;
        Real implied_ub = kInf;
        if (coeff > 0) {
            if (isFinite(lp.row_lower[i])) {
                implied_lb = lp.row_lower[i] / coeff;
            }
            if (isFinite(lp.row_upper[i])) {
                implied_ub = lp.row_upper[i] / coeff;
            }
        } else {
            if (isFinite(lp.row_upper[i])) {
                implied_lb = lp.row_upper[i] / coeff;
            }
            if (isFinite(lp.row_lower[i])) {
                implied_ub = lp.row_lower[i] / coeff;
            }
        }

        // Tighten variable bounds.
        Real new_lb = lp.col_lower[col];
        Real new_ub = lp.col_upper[col];
        if (isFinite(implied_lb) && implied_lb > new_lb + kTol) {
            new_lb = implied_lb;
        }
        if (isFinite(implied_ub) && implied_ub < new_ub - kTol) {
            new_ub = implied_ub;
        }

        if (new_lb > new_ub + kTol) {
            infeasible_ = true;
            return changes;
        }

        // Record postsolve op.
        BpSingletonRow op;
        op.orig_row = i;
        op.orig_col = col;
        op.coeff = coeff;
        op.row_lower = lp.row_lower[i];
        op.row_upper = lp.row_upper[i];
        ops_.push_back(std::move(op));

        lp.col_lower[col] = new_lb;
        lp.col_upper[col] = new_ub;

        // Remove the row.
        row_removed[i] = true;
        --col_nnz[col];
        row_nnz[i] = 0;
        ++changes;
    }

    stats_.singleton_rows += changes;
    return changes;
}

Index BarrierPresolver::removeSingletonCols(LpProblem& lp, std::vector<bool>& col_removed,
                                            std::vector<bool>& row_removed,
                                            std::vector<Index>& row_nnz,
                                            std::vector<Index>& col_nnz) {
    Index changes = 0;
    for (Index j = 0; j < lp.num_cols; ++j) {
        if (col_removed[j]) {
            continue;
        }
        if (col_nnz[j] != 1) {
            continue;
        }

        // Find the single active row.
        auto cv = lp.matrix.col(j);
        Index row = -1;
        Real coeff = 0.0;
        for (Index k = 0; k < cv.size(); ++k) {
            if (!row_removed[cv.indices[k]]) {
                row = cv.indices[k];
                coeff = cv.values[k];
                break;
            }
        }
        if (row < 0 || std::abs(coeff) < kTol) {
            continue;
        }

        // Skip free variables -- handled by removeFreeColSingletons.
        if (!isFinite(lp.col_lower[j]) && !isFinite(lp.col_upper[j])) {
            continue;
        }

        // Only eliminate singleton columns with zero objective cost.
        // Columns with nonzero cost that aren't forced to a bound should remain
        // in the reduced problem so the barrier optimizer can determine their value.
        Real obj_coeff = lp.obj[j];
        if (std::abs(obj_coeff) > kTol) {
            continue;
        }

        Real lb = lp.col_lower[j];
        Real ub = lp.col_upper[j];

        // Record postsolve op before modifying bounds. Capture the row's other
        // active columns so postsolve can recover x_j from the row activity
        // (pinning to a bound generally violates equality/range rows).
        BpSingletonCol op;
        op.orig_col = j;
        op.orig_row = row;
        op.coeff = coeff;
        op.obj_coeff = obj_coeff;
        op.col_lower = lb;
        op.col_upper = ub;
        op.row_lower = lp.row_lower[row];
        op.row_upper = lp.row_upper[row];
        auto row_view = lp.matrix.row(row);
        for (Index k = 0; k < row_view.size(); ++k) {
            Index other = row_view.indices[k];
            if (other == j || col_removed[other]) {
                continue;
            }
            op.row_col_indices.push_back(other);
            op.row_col_coeffs.push_back(row_view.values[k]);
        }
        ops_.push_back(std::move(op));

        // Adjust row bounds to absorb x_j.
        // row: a*x_j + (rest) in [lo, hi]  =>  (rest) in [lo - a*ub, hi - a*lb] if a > 0
        //                                       (rest) in [lo - a*lb, hi - a*ub] if a < 0
        Real contrib_lo, contrib_hi;
        if (coeff > 0) {
            contrib_lo = isFinite(lb) ? coeff * lb : -kInf;
            contrib_hi = isFinite(ub) ? coeff * ub : kInf;
        } else {
            contrib_lo = isFinite(ub) ? coeff * ub : -kInf;
            contrib_hi = isFinite(lb) ? coeff * lb : kInf;
        }

        Real new_row_lower = lp.row_lower[row];
        Real new_row_upper = lp.row_upper[row];
        if (isFinite(new_row_lower) && isFinite(contrib_hi)) {
            new_row_lower -= contrib_hi;
        } else if (isFinite(new_row_lower)) {
            new_row_lower = -kInf;
        }

        if (isFinite(new_row_upper) && isFinite(contrib_lo)) {
            new_row_upper -= contrib_lo;
        } else if (isFinite(new_row_upper)) {
            new_row_upper = kInf;
        }

        lp.row_lower[row] = new_row_lower;
        lp.row_upper[row] = new_row_upper;

        col_removed[j] = true;
        col_nnz[j] = 0;
        --row_nnz[row];
        ++changes;
    }

    stats_.singleton_cols += changes;
    return changes;
}

Index BarrierPresolver::removeFreeColSingletons(LpProblem& lp, std::vector<bool>& col_removed,
                                                std::vector<bool>& row_removed,
                                                std::vector<Index>& row_nnz,
                                                std::vector<Index>& col_nnz) {
    Index changes = 0;
    for (Index j = 0; j < lp.num_cols; ++j) {
        if (col_removed[j]) {
            continue;
        }
        if (col_nnz[j] != 1) {
            continue;
        }
        if (isFinite(lp.col_lower[j]) || isFinite(lp.col_upper[j])) {
            continue;
        }

        // Free variable in a single constraint -> eliminate both.
        auto cv = lp.matrix.col(j);
        Index row = -1;
        Real coeff = 0.0;
        for (Index k = 0; k < cv.size(); ++k) {
            if (!row_removed[cv.indices[k]]) {
                row = cv.indices[k];
                coeff = cv.values[k];
                break;
            }
        }
        if (row < 0 || std::abs(coeff) < kTol) {
            continue;
        }

        // Record postsolve op.
        BpFreeColSingleton op;
        op.orig_col = j;
        op.orig_row = row;
        op.coeff = coeff;
        op.obj_coeff = lp.obj[j];
        op.row_lower = lp.row_lower[row];
        op.row_upper = lp.row_upper[row];

        auto rv = lp.matrix.row(row);
        for (Index k = 0; k < rv.size(); ++k) {
            if (rv.indices[k] != j && !col_removed[rv.indices[k]]) {
                op.row_col_indices.push_back(rv.indices[k]);
                op.row_col_coeffs.push_back(rv.values[k]);
            }
        }
        ops_.push_back(std::move(op));

        // Remove the column.
        col_removed[j] = true;
        col_nnz[j] = 0;

        // Remove the row and update col_nnz for all other columns in the row.
        row_removed[row] = true;
        for (Index k = 0; k < rv.size(); ++k) {
            if (!col_removed[rv.indices[k]]) {
                --col_nnz[rv.indices[k]];
            }
        }
        row_nnz[row] = 0;
        ++changes;
    }

    stats_.free_col_singletons += changes;
    return changes;
}

Index BarrierPresolver::removeEmptyRowsCols(LpProblem& lp, std::vector<bool>& col_removed,
                                            std::vector<bool>& row_removed,
                                            std::vector<Index>& row_nnz,
                                            std::vector<Index>& col_nnz) {
    Index changes = 0;

    // Empty rows: 0 in [lo, hi] -- check feasibility.
    for (Index i = 0; i < lp.num_rows; ++i) {
        if (row_removed[i]) {
            continue;
        }
        if (row_nnz[i] != 0) {
            continue;
        }

        Real lo = lp.row_lower[i];
        Real hi = lp.row_upper[i];
        if ((isFinite(lo) && lo > kTol) || (isFinite(hi) && hi < -kTol)) {
            infeasible_ = true;
            return changes;
        }

        BpEmptyRow op;
        op.orig_row = i;
        op.row_lower = lo;
        op.row_upper = hi;
        ops_.push_back(std::move(op));

        row_removed[i] = true;
        ++changes;
        ++stats_.empty_rows;
    }

    // Empty columns: no constraint involvement.
    for (Index j = 0; j < lp.num_cols; ++j) {
        if (col_removed[j]) {
            continue;
        }
        if (col_nnz[j] != 0) {
            continue;
        }

        // Fix at optimal bound.
        Real lb = lp.col_lower[j];
        Real ub = lp.col_upper[j];
        Real obj = lp.obj[j];
        Real sense_sign = (lp.sense == Sense::Minimize) ? 1.0 : -1.0;

        Real val;
        if (sense_sign * obj > 0) {
            if (!isFinite(lb)) {
                // Unbounded.
                // For barrier presolve on continuous relaxation, just skip.
                continue;
            }
            val = lb;
        } else if (sense_sign * obj < 0) {
            if (!isFinite(ub)) {
                continue;
            }
            val = ub;
        } else {
            val = isFinite(lb) ? lb : (isFinite(ub) ? ub : 0.0);
        }

        BpEmptyCol op;
        op.orig_col = j;
        op.obj_coeff = obj;
        op.col_lower = lb;
        op.col_upper = ub;
        op.value = val;
        ops_.push_back(std::move(op));

        obj_offset_delta_ += obj * val;
        col_removed[j] = true;
        col_nnz[j] = 0;
        ++changes;
        ++stats_.empty_cols;
    }

    return changes;
}

Index BarrierPresolver::tightenImpliedBounds(LpProblem& lp, const std::vector<bool>& col_removed,
                                             const std::vector<bool>& row_removed,
                                             const std::vector<Index>& row_nnz) {
    Index changes = 0;
    for (Index i = 0; i < lp.num_rows; ++i) {
        if (row_removed[i]) {
            continue;
        }
        if (row_nnz[i] < 2) {
            continue;
        }

        auto rv = lp.matrix.row(i);

        // Compute activity bounds: min/max of sum a_j * x_j.
        Real act_min = 0.0;
        Real act_max = 0.0;
        bool act_min_finite = true;
        bool act_max_finite = true;

        for (Index k = 0; k < rv.size(); ++k) {
            Index j = rv.indices[k];
            if (col_removed[j]) {
                continue;
            }
            Real a = rv.values[k];
            Real lb = lp.col_lower[j];
            Real ub = lp.col_upper[j];

            if (a > 0) {
                if (isFinite(lb)) {
                    act_min += a * lb;
                } else {
                    act_min_finite = false;
                }
                if (isFinite(ub)) {
                    act_max += a * ub;
                } else {
                    act_max_finite = false;
                }
            } else {
                if (isFinite(ub)) {
                    act_min += a * ub;
                } else {
                    act_min_finite = false;
                }
                if (isFinite(lb)) {
                    act_max += a * lb;
                } else {
                    act_max_finite = false;
                }
            }

            if (!act_min_finite && !act_max_finite) {
                break;
            }
        }

        // For each variable with one-sided infinite contribution, tighten.
        for (Index k = 0; k < rv.size(); ++k) {
            Index j = rv.indices[k];
            if (col_removed[j]) {
                continue;
            }
            Real a = rv.values[k];
            if (std::abs(a) < kTol) {
                continue;
            }
            Real lb = lp.col_lower[j];
            Real ub = lp.col_upper[j];

            // Tighten upper bound from row_upper:
            // sum_{k!=j} a_k x_k + a_j x_j <= row_upper
            // x_j <= (row_upper - min_{k!=j} a_k x_k) / a_j  (if a_j > 0)
            if (isFinite(lp.row_upper[i]) && act_min_finite && a > 0 && isFinite(lb)) {
                Real rest_min = act_min - a * lb;
                Real implied_ub = (lp.row_upper[i] - rest_min) / a;
                if (implied_ub < ub - kTol) {
                    lp.col_upper[j] = implied_ub;
                    ++changes;
                }
            }
            if (isFinite(lp.row_upper[i]) && act_max_finite && a < 0 && isFinite(ub)) {
                Real rest_max = act_max - a * ub;
                Real implied_lb = (lp.row_upper[i] - rest_max) / a;
                if (implied_lb > lb + kTol) {
                    lp.col_lower[j] = implied_lb;
                    ++changes;
                }
            }
            // Tighten from row_lower:
            if (isFinite(lp.row_lower[i]) && act_max_finite && a > 0 && isFinite(ub)) {
                Real rest_max = act_max - a * ub;
                Real implied_lb = (lp.row_lower[i] - rest_max) / a;
                if (implied_lb > lb + kTol) {
                    lp.col_lower[j] = implied_lb;
                    ++changes;
                }
            }
            if (isFinite(lp.row_lower[i]) && act_min_finite && a < 0 && isFinite(lb)) {
                Real rest_min = act_min - a * lb;
                Real implied_ub = (lp.row_lower[i] - rest_min) / a;
                if (implied_ub < ub - kTol) {
                    lp.col_upper[j] = implied_ub;
                    ++changes;
                }
            }
        }
    }

    stats_.implied_bounds_tightened += changes;
    return changes;
}

Index BarrierPresolver::removeRedundantRows(LpProblem& lp, const std::vector<bool>& col_removed,
                                            std::vector<bool>& row_removed,
                                            std::vector<Index>& row_nnz,
                                            std::vector<Index>& col_nnz) {
    Index changes = 0;
    for (Index i = 0; i < lp.num_rows; ++i) {
        if (row_removed[i]) {
            continue;
        }
        if (row_nnz[i] == 0) {
            continue;
        }

        auto rv = lp.matrix.row(i);

        // Compute activity bounds.
        Real act_min = 0.0;
        Real act_max = 0.0;
        bool act_min_finite = true;
        bool act_max_finite = true;

        for (Index k = 0; k < rv.size(); ++k) {
            Index j = rv.indices[k];
            if (col_removed[j]) {
                continue;
            }
            Real a = rv.values[k];
            Real lb = lp.col_lower[j];
            Real ub = lp.col_upper[j];

            if (a > 0) {
                if (isFinite(lb)) {
                    act_min += a * lb;
                } else {
                    act_min_finite = false;
                }
                if (isFinite(ub)) {
                    act_max += a * ub;
                } else {
                    act_max_finite = false;
                }
            } else {
                if (isFinite(ub)) {
                    act_min += a * ub;
                } else {
                    act_min_finite = false;
                }
                if (isFinite(lb)) {
                    act_max += a * lb;
                } else {
                    act_max_finite = false;
                }
            }
        }

        // Check if the row is redundant.
        bool lb_redundant =
            !isFinite(lp.row_lower[i]) || (act_min_finite && act_min >= lp.row_lower[i] - kTol);
        bool ub_redundant =
            !isFinite(lp.row_upper[i]) || (act_max_finite && act_max <= lp.row_upper[i] + kTol);

        if (lb_redundant && ub_redundant) {
            BpEmptyRow op;
            op.orig_row = i;
            op.row_lower = lp.row_lower[i];
            op.row_upper = lp.row_upper[i];
            ops_.push_back(std::move(op));

            // Update col_nnz.
            for (Index k = 0; k < rv.size(); ++k) {
                if (!col_removed[rv.indices[k]]) {
                    --col_nnz[rv.indices[k]];
                }
            }
            row_removed[i] = true;
            row_nnz[i] = 0;
            ++changes;
        }
    }

    stats_.redundant_rows += changes;
    return changes;
}

// ============================================================================
// Build reduced problem
// ============================================================================

LpProblem BarrierPresolver::buildReducedProblem(const LpProblem& lp,
                                                const std::vector<bool>& col_removed,
                                                const std::vector<bool>& row_removed) {
    // Build column mapping.
    col_mapping_.clear();
    std::vector<Index> col_new_idx(static_cast<size_t>(lp.num_cols), -1);
    for (Index j = 0; j < lp.num_cols; ++j) {
        if (!col_removed[j]) {
            col_new_idx[j] = static_cast<Index>(col_mapping_.size());
            col_mapping_.push_back(j);
        }
    }

    // Build row mapping.
    row_mapping_.clear();
    std::vector<Index> row_new_idx(static_cast<size_t>(lp.num_rows), -1);
    for (Index i = 0; i < lp.num_rows; ++i) {
        if (!row_removed[i]) {
            row_new_idx[i] = static_cast<Index>(row_mapping_.size());
            row_mapping_.push_back(i);
        }
    }

    Index new_ncols = static_cast<Index>(col_mapping_.size());
    Index new_nrows = static_cast<Index>(row_mapping_.size());

    LpProblem reduced;
    reduced.name = lp.name;
    reduced.sense = lp.sense;
    reduced.num_cols = new_ncols;
    reduced.num_rows = new_nrows;
    reduced.obj_offset = lp.obj_offset + obj_offset_delta_;

    reduced.obj.resize(static_cast<size_t>(new_ncols));
    reduced.col_lower.resize(static_cast<size_t>(new_ncols));
    reduced.col_upper.resize(static_cast<size_t>(new_ncols));
    reduced.col_type.resize(static_cast<size_t>(new_ncols));
    reduced.col_names.resize(static_cast<size_t>(new_ncols));

    for (Index jj = 0; jj < new_ncols; ++jj) {
        Index j = col_mapping_[jj];
        reduced.obj[jj] = lp.obj[j];
        reduced.col_lower[jj] = lp.col_lower[j];
        reduced.col_upper[jj] = lp.col_upper[j];
        reduced.col_type[jj] = lp.col_type[j];
        if (j < static_cast<Index>(lp.col_names.size())) {
            reduced.col_names[jj] = lp.col_names[j];
        }
    }

    reduced.row_lower.resize(static_cast<size_t>(new_nrows));
    reduced.row_upper.resize(static_cast<size_t>(new_nrows));
    reduced.row_names.resize(static_cast<size_t>(new_nrows));

    std::vector<Triplet> trips;
    for (Index ii = 0; ii < new_nrows; ++ii) {
        Index i = row_mapping_[ii];
        reduced.row_lower[ii] = lp.row_lower[i];
        reduced.row_upper[ii] = lp.row_upper[i];
        if (i < static_cast<Index>(lp.row_names.size())) {
            reduced.row_names[ii] = lp.row_names[i];
        }

        auto rv = lp.matrix.row(i);
        for (Index k = 0; k < rv.size(); ++k) {
            Index j = rv.indices[k];
            if (col_removed[j]) {
                continue;
            }
            trips.push_back({ii, col_new_idx[j], rv.values[k]});
        }
    }

    reduced.matrix = SparseMatrix(new_nrows, new_ncols, std::move(trips));
    return reduced;
}

// ============================================================================
// Main presolve entry point
// ============================================================================

LpProblem BarrierPresolver::presolve(const LpProblem& problem) {
    ops_.clear();
    col_mapping_.clear();
    row_mapping_.clear();
    obj_offset_delta_ = 0.0;
    stats_ = {};
    infeasible_ = false;

    orig_num_cols_ = problem.num_cols;
    orig_num_rows_ = problem.num_rows;
    stats_.orig_rows = problem.num_rows;
    stats_.orig_cols = problem.num_cols;
    stats_.orig_nnz = problem.matrix.numNonzeros();

    // Work on a mutable copy.
    LpProblem lp = problem;

    std::vector<bool> col_removed(static_cast<size_t>(lp.num_cols), false);
    std::vector<bool> row_removed(static_cast<size_t>(lp.num_rows), false);

    // Compute initial nnz counts.
    std::vector<Index> row_nnz(static_cast<size_t>(lp.num_rows), 0);
    std::vector<Index> col_nnz(static_cast<size_t>(lp.num_cols), 0);
    for (Index i = 0; i < lp.num_rows; ++i) {
        auto rv = lp.matrix.row(i);
        row_nnz[i] = rv.size();
        for (Index k = 0; k < rv.size(); ++k) {
            ++col_nnz[rv.indices[k]];
        }
    }

    // Iterate reductions until no more changes.
    constexpr Index kMaxRounds = 20;
    for (Index round = 0; round < kMaxRounds; ++round) {
        Index changes = 0;

        changes += removeFixedVariables(lp, col_removed, row_removed, row_nnz, col_nnz);
        if (infeasible_) {
            break;
        }

        changes += removeEmptyRowsCols(lp, col_removed, row_removed, row_nnz, col_nnz);
        if (infeasible_) {
            break;
        }

        changes += removeSingletonRows(lp, col_removed, row_removed, row_nnz, col_nnz);
        if (infeasible_) {
            break;
        }

        changes += removeFreeColSingletons(lp, col_removed, row_removed, row_nnz, col_nnz);
        if (infeasible_) {
            break;
        }

        changes += removeSingletonCols(lp, col_removed, row_removed, row_nnz, col_nnz);
        if (infeasible_) {
            break;
        }

        changes += removeEmptyRowsCols(lp, col_removed, row_removed, row_nnz, col_nnz);
        if (infeasible_) {
            break;
        }

        changes += tightenImpliedBounds(lp, col_removed, row_removed, row_nnz);
        if (infeasible_) {
            break;
        }

        changes += removeRedundantRows(lp, col_removed, row_removed, row_nnz, col_nnz);
        if (infeasible_) {
            break;
        }

        changes += removeEmptyRowsCols(lp, col_removed, row_removed, row_nnz, col_nnz);
        if (infeasible_) {
            break;
        }

        if (changes == 0) {
            break;
        }
    }

    LpProblem reduced = buildReducedProblem(lp, col_removed, row_removed);
    stats_.reduced_rows = reduced.num_rows;
    stats_.reduced_cols = reduced.num_cols;
    stats_.reduced_nnz = reduced.matrix.numNonzeros();
    return reduced;
}

// ============================================================================
// Postsolve
// ============================================================================

void BarrierPresolver::postsolve(const std::vector<Real>& primal_reduced,
                                 const std::vector<Real>& dual_reduced,
                                 const std::vector<Real>& rc_reduced,
                                 std::vector<Real>& primal_orig, std::vector<Real>& dual_orig,
                                 std::vector<Real>& rc_orig) const {
    // Initialize original-space vectors.
    primal_orig.assign(static_cast<size_t>(orig_num_cols_), 0.0);
    dual_orig.assign(static_cast<size_t>(orig_num_rows_), 0.0);
    rc_orig.assign(static_cast<size_t>(orig_num_cols_), 0.0);

    // Map reduced solution to original indices.
    for (Index jj = 0; jj < static_cast<Index>(col_mapping_.size()); ++jj) {
        Index j = col_mapping_[jj];
        primal_orig[j] = primal_reduced[jj];
        if (jj < static_cast<Index>(rc_reduced.size())) {
            rc_orig[j] = rc_reduced[jj];
        }
    }
    for (Index ii = 0; ii < static_cast<Index>(row_mapping_.size()); ++ii) {
        Index i = row_mapping_[ii];
        if (ii < static_cast<Index>(dual_reduced.size())) {
            dual_orig[i] = dual_reduced[ii];
        }
    }

    // Apply postsolve ops in reverse order.
    for (auto it = ops_.rbegin(); it != ops_.rend(); ++it) {
        std::visit(
            [&](const auto& op) {
                using T = std::decay_t<decltype(op)>;

                if constexpr (std::is_same_v<T, BpFixVariable>) {
                    primal_orig[op.orig_col] = op.value;
                    // Reduced cost for a fixed variable: c_j - sum_i a_{ij} * y_i.
                    Real rc = op.obj_coeff;
                    for (Index k = 0; k < static_cast<Index>(op.row_indices.size()); ++k) {
                        rc -= op.row_coeffs[k] * dual_orig[op.row_indices[k]];
                    }
                    rc_orig[op.orig_col] = rc;

                } else if constexpr (std::is_same_v<T, BpSingletonRow>) {
                    // The row had a single nonzero: a * x_col = rhs (roughly).
                    // The dual for this row can be inferred from reduced cost balance.
                    // y_row = (c_col - rc_col - sum_{other rows touching col} a_{row,col} * y_row)
                    // / a Since this is barrier (approximate), set dual to 0.
                    dual_orig[op.orig_row] = 0.0;

                } else if constexpr (std::is_same_v<T, BpSingletonCol>) {
                    // x_j had zero cost and was in a single row, which was kept
                    // (its bounds relaxed to absorb x_j). x_j must now take a
                    // value that satisfies that row given the other columns, not
                    // merely sit at a bound. Recover the feasible x_j interval
                    // from the row activity and column bounds; any point in it is
                    // optimal (zero cost), so prefer the one nearest zero.
                    Real rest = 0.0;
                    for (Index k = 0; k < static_cast<Index>(op.row_col_indices.size()); ++k) {
                        rest += op.row_col_coeffs[k] * primal_orig[op.row_col_indices[k]];
                    }
                    Real target_lo = -kInf;
                    Real target_hi = kInf;
                    if (op.coeff > 0) {
                        if (isFinite(op.row_lower)) {
                            target_lo = (op.row_lower - rest) / op.coeff;
                        }
                        if (isFinite(op.row_upper)) {
                            target_hi = (op.row_upper - rest) / op.coeff;
                        }
                    } else {
                        if (isFinite(op.row_upper)) {
                            target_lo = (op.row_upper - rest) / op.coeff;
                        }
                        if (isFinite(op.row_lower)) {
                            target_hi = (op.row_lower - rest) / op.coeff;
                        }
                    }
                    Real lo = std::max(target_lo, isFinite(op.col_lower) ? op.col_lower : -kInf);
                    Real hi = std::min(target_hi, isFinite(op.col_upper) ? op.col_upper : kInf);
                    Real val;
                    if (lo <= hi) {
                        val = std::clamp(0.0, lo, hi);
                    } else {
                        // Empty interval from numerical slack: fall back to a bound.
                        val = isFinite(op.col_lower)
                                  ? op.col_lower
                                  : (isFinite(op.col_upper) ? op.col_upper : 0.0);
                    }
                    primal_orig[op.orig_col] = val;
                    rc_orig[op.orig_col] = 0.0;

                } else if constexpr (std::is_same_v<T, BpFreeColSingleton>) {
                    // Free variable was used to make a constraint feasible.
                    // Recover x_j from the row: a_j * x_j = rhs - sum_{k!=j} a_k * x_k.
                    // Use midpoint of row bounds as target.
                    Real rhs_target;
                    if (isFinite(op.row_lower) && isFinite(op.row_upper)) {
                        rhs_target = 0.5 * (op.row_lower + op.row_upper);
                    } else if (isFinite(op.row_lower)) {
                        rhs_target = op.row_lower;
                    } else if (isFinite(op.row_upper)) {
                        rhs_target = op.row_upper;
                    } else {
                        rhs_target = 0.0;
                    }

                    Real rest = 0.0;
                    for (Index k = 0; k < static_cast<Index>(op.row_col_indices.size()); ++k) {
                        rest += op.row_col_coeffs[k] * primal_orig[op.row_col_indices[k]];
                    }
                    primal_orig[op.orig_col] = (rhs_target - rest) / op.coeff;

                    // Dual for the eliminated row: from the free variable's optimality,
                    // c_j = a_j * y_row  =>  y_row = c_j / a_j.
                    dual_orig[op.orig_row] = op.obj_coeff / op.coeff;
                    rc_orig[op.orig_col] = 0.0;

                } else if constexpr (std::is_same_v<T, BpEmptyRow>) {
                    dual_orig[op.orig_row] = 0.0;

                } else if constexpr (std::is_same_v<T, BpEmptyCol>) {
                    // Reuse the sense-aware value chosen by presolve; recomputing
                    // from the raw objective sign here would pick the wrong bound
                    // for Maximize problems.
                    primal_orig[op.orig_col] = op.value;
                    rc_orig[op.orig_col] = op.obj_coeff;
                }
            },
            *it);
    }
}

}  // namespace mipx
