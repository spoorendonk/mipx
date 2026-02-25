#include "mipx/domain.h"

#include <algorithm>
#include <cmath>

namespace mipx {

void DomainPropagator::load(const LpProblem& problem) {
    num_cols_ = problem.num_cols;
    num_rows_ = problem.num_rows;
    matrix_ = problem.matrix;
    row_lower_ = problem.row_lower;
    row_upper_ = problem.row_upper;
    col_type_ = problem.col_type;
    lower_ = problem.col_lower;
    upper_ = problem.col_upper;

    changes_.clear();
    checkpoint_stack_.clear();

    row_in_queue_.assign(num_rows_, false);
    queue_.clear();
}

void DomainPropagator::setBound(Index col, Real lower, Real upper) {
    recordChange(col);
    bool changed = false;

    if (lower > lower_[col] + kTol) {
        lower_[col] = lower;
        changed = true;
    }
    if (upper < upper_[col] - kTol) {
        upper_[col] = upper;
        changed = true;
    }

    // Round for integer variables.
    if (col_type_[col] != VarType::Continuous) {
        Real rounded_lb = std::ceil(lower_[col] - kTol);
        Real rounded_ub = std::floor(upper_[col] + kTol);
        if (rounded_lb > lower_[col] + kTol) {
            lower_[col] = rounded_lb;
            changed = true;
        }
        if (rounded_ub < upper_[col] - kTol) {
            upper_[col] = rounded_ub;
            changed = true;
        }
    }

    if (changed) {
        enqueueColumn(col);
    }
}

Real DomainPropagator::getLower(Index col) const { return lower_[col]; }

Real DomainPropagator::getUpper(Index col) const { return upper_[col]; }

bool DomainPropagator::propagate() {
    // If queue is empty, seed with all rows.
    if (queue_.empty()) {
        for (Index i = 0; i < num_rows_; ++i) {
            queue_.push_back(i);
            row_in_queue_[i] = true;
        }
    }

    while (!queue_.empty()) {
        Index row = queue_.back();
        queue_.pop_back();
        row_in_queue_[row] = false;

        if (!propagateRow(row)) {
            return false;
        }
    }
    return true;
}

void DomainPropagator::pushCheckpoint() {
    checkpoint_stack_.push_back(static_cast<Index>(changes_.size()));
}

void DomainPropagator::popCheckpoint() {
    Index target = checkpoint_stack_.back();
    checkpoint_stack_.pop_back();

    // Undo changes in reverse order.
    while (static_cast<Index>(changes_.size()) > target) {
        auto& c = changes_.back();
        lower_[c.col] = c.old_lower;
        upper_[c.col] = c.old_upper;
        changes_.pop_back();
    }

    // Clear propagation queue since bounds have been restored.
    queue_.clear();
    std::fill(row_in_queue_.begin(), row_in_queue_.end(), false);
}

Int DomainPropagator::numTightened() const {
    Index base = checkpoint_stack_.empty()
                     ? 0
                     : checkpoint_stack_.back();
    return static_cast<Int>(changes_.size()) - base;
}

bool DomainPropagator::propagateRow(Index row) {
    auto rv = matrix_.row(row);
    Real rl = row_lower_[row];
    Real ru = row_upper_[row];

    // Compute activity bounds for the row.
    // Track finite sums and count of infinite contributions separately.
    Real act_min = 0.0;
    Real act_max = 0.0;
    Index inf_min_count = 0;
    Index inf_max_count = 0;
    Index inf_min_col = -1;  // Valid only when inf_min_count == 1.
    Index inf_max_col = -1;

    for (Index k = 0; k < rv.size(); ++k) {
        Index j = rv.indices[k];
        Real a = rv.values[k];
        Real lb = lower_[j];
        Real ub = upper_[j];

        Real contrib_min = (a > 0) ? a * lb : a * ub;
        Real contrib_max = (a > 0) ? a * ub : a * lb;

        if (std::isinf(contrib_min)) {
            ++inf_min_count;
            inf_min_col = j;
        } else {
            act_min += contrib_min;
        }

        if (std::isinf(contrib_max)) {
            ++inf_max_count;
            inf_max_col = j;
        } else {
            act_max += contrib_max;
        }
    }

    // Check infeasibility using the full activity bounds.
    if (inf_min_count == 0 && act_min > ru + kTol) return false;
    if (inf_max_count == 0 && act_max < rl - kTol) return false;

    // If both sides have 2+ infinite contributions, no tightening is possible.
    if (inf_min_count >= 2 && inf_max_count >= 2) return true;

    // Try to tighten each variable's bounds.
    for (Index k = 0; k < rv.size(); ++k) {
        Index j = rv.indices[k];
        Real a = rv.values[k];

        // Compute residual activity bounds excluding variable j.
        Real res_min;
        bool res_min_finite;
        if (inf_min_count == 0) {
            res_min_finite = true;
            Real contrib_min_j = (a > 0) ? a * lower_[j] : a * upper_[j];
            res_min = act_min - contrib_min_j;
        } else if (inf_min_count == 1 && inf_min_col == j) {
            // j was the only infinite contributor; residual is finite.
            res_min_finite = true;
            res_min = act_min;
        } else {
            // 2+ infinite contributions, or 1 but from another variable.
            res_min_finite = false;
            res_min = -kInf;
        }

        Real res_max;
        bool res_max_finite;
        if (inf_max_count == 0) {
            res_max_finite = true;
            Real contrib_max_j = (a > 0) ? a * upper_[j] : a * lower_[j];
            res_max = act_max - contrib_max_j;
        } else if (inf_max_count == 1 && inf_max_col == j) {
            res_max_finite = true;
            res_max = act_max;
        } else {
            res_max_finite = false;
            res_max = kInf;
        }

        bool changed = false;

        // Derive new upper bound.
        if (a > 0 && !std::isinf(ru) && res_min_finite) {
            Real new_ub = (ru - res_min) / a;
            if (new_ub < upper_[j] - kTol) {
                recordChange(j);
                upper_[j] = new_ub;
                changed = true;
            }
        } else if (a < 0 && !std::isinf(rl) && res_max_finite) {
            Real new_ub = (rl - res_max) / a;
            if (new_ub < upper_[j] - kTol) {
                recordChange(j);
                upper_[j] = new_ub;
                changed = true;
            }
        }

        // Derive new lower bound.
        if (a > 0 && !std::isinf(rl) && res_max_finite) {
            Real new_lb = (rl - res_max) / a;
            if (new_lb > lower_[j] + kTol) {
                recordChange(j);
                lower_[j] = new_lb;
                changed = true;
            }
        } else if (a < 0 && !std::isinf(ru) && res_min_finite) {
            Real new_lb = (ru - res_min) / a;
            if (new_lb > lower_[j] + kTol) {
                recordChange(j);
                lower_[j] = new_lb;
                changed = true;
            }
        }

        // Integer rounding.
        if (changed && col_type_[j] != VarType::Continuous) {
            Real rounded_lb = std::ceil(lower_[j] - kTol);
            Real rounded_ub = std::floor(upper_[j] + kTol);
            if (rounded_lb > lower_[j] + kTol) {
                recordChange(j);
                lower_[j] = rounded_lb;
            }
            if (rounded_ub < upper_[j] - kTol) {
                recordChange(j);
                upper_[j] = rounded_ub;
            }
        }

        // Check infeasibility.
        if (lower_[j] > upper_[j] + kTol) {
            return false;
        }

        if (changed) {
            enqueueColumn(j);
        }
    }

    return true;
}

void DomainPropagator::recordChange(Index col) {
    changes_.push_back({col, lower_[col], upper_[col]});
}

void DomainPropagator::enqueueColumn(Index col) {
    auto cv = matrix_.col(col);
    for (Index k = 0; k < cv.size(); ++k) {
        Index row = cv.indices[k];
        if (!row_in_queue_[row]) {
            row_in_queue_[row] = true;
            queue_.push_back(row);
        }
    }
}

}  // namespace mipx
