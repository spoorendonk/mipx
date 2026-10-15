#include "mipx/objective_propagator.h"

#include "mipx/domain.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace mipx {

void ObjectivePropagator::load(const LpProblem& problem) {
    obj_cols_.clear();
    obj_vals_.clear();
    for (Index j = 0; j < problem.num_cols; ++j) {
        if (problem.obj[j] != 0.0) {
            obj_cols_.push_back(j);
            obj_vals_.push_back(problem.obj[j]);
        }
    }
    obj_offset_ = problem.obj_offset;
    col_type_ = problem.col_type;
    loaded_ = true;
}

// The activity-bound reasoning of DomainPropagator::propagateRow for one row,
// kept in one piece so the two can be read side by side.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
bool ObjectivePropagator::propagate(Real cutoff, std::span<const Real> lower,
                                    std::span<const Real> upper,
                                    std::vector<BoundUpdate>& updates) const {
    updates.clear();
    if (!loaded_ || !std::isfinite(cutoff)) {
        return true;
    }
    const Real rhs = cutoff - obj_offset_;
    const Index n = static_cast<Index>(obj_cols_.size());

    // Min activity of sum_j c_j x_j, with the infinite contributions counted
    // apart as in DomainPropagator::propagateRow.
    Real act_min = 0.0;
    Index inf_count = 0;
    Index inf_col = -1;
    for (Index k = 0; k < n; ++k) {
        const Index j = obj_cols_[k];
        const Real c = obj_vals_[k];
        const Real contrib = (c > 0) ? c * lower[j] : c * upper[j];
        if (std::isinf(contrib)) {
            ++inf_count;
            inf_col = j;
        } else {
            act_min += contrib;
        }
    }

    if (inf_count == 0 && act_min >= rhs - kPruneTol) {
        return false;
    }
    if (inf_count >= 2) {
        return true;
    }

    for (Index k = 0; k < n; ++k) {
        const Index j = obj_cols_[k];
        const Real c = obj_vals_[k];
        Real res_min;
        if (inf_count == 0) {
            res_min = act_min - ((c > 0) ? c * lower[j] : c * upper[j]);
        } else if (inf_col == j) {
            res_min = act_min;
        } else {
            continue;
        }

        Real new_lower = lower[j];
        Real new_upper = upper[j];
        bool changed = false;
        if (c > 0) {
            const Real ub = (rhs - res_min) / c;
            if (ub < new_upper - kTol) {
                new_upper = ub;
                changed = true;
            }
        } else {
            const Real lb = (rhs - res_min) / c;
            if (lb > new_lower + kTol) {
                new_lower = lb;
                changed = true;
            }
        }
        if (!changed) {
            continue;
        }
        if (col_type_[j] != VarType::Continuous) {
            new_lower = std::ceil(new_lower - kTol);
            new_upper = std::floor(new_upper + kTol);
        }
        if (new_lower > new_upper + kTol) {
            return false;
        }
        if (new_lower > lower[j] + kTol || new_upper < upper[j] - kTol) {
            updates.push_back({j, new_lower, new_upper});
        }
    }
    return true;
}

void ObjectiveCutoffBounds::load(const LpProblem& root, const ObjectivePropagator* objective,
                                 const ImplicationGraph* implications, const CliqueTable* cliques) {
    root_ = root;
    objective_ = objective;
    implications_ = implications;
    cliques_ = cliques;
    std::lock_guard<std::mutex> lock(mutex_);
    snapshot_ = std::make_shared<const Snapshot>();
}

void ObjectiveCutoffBounds::reset() {
    root_ = LpProblem{};
    objective_ = nullptr;
    implications_ = nullptr;
    cliques_ = nullptr;
    std::lock_guard<std::mutex> lock(mutex_);
    snapshot_.reset();
}

std::shared_ptr<const ObjectiveCutoffBounds::Snapshot> ObjectiveCutoffBounds::current() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return snapshot_;
}

bool ObjectiveCutoffBounds::refresh(Real cutoff) {
    if (objective_ == nullptr || !std::isfinite(cutoff)) {
        return false;
    }
    // Cheap test first, so workers do not queue behind a running refresh for a
    // cutoff that is not new.
    if (const auto seen = current();
        seen == nullptr || seen->infeasible || cutoff >= seen->cutoff) {
        return false;
    }
    std::lock_guard<std::mutex> refresh_lock(refresh_mutex_);
    const auto old = current();
    if (old == nullptr || old->infeasible || cutoff >= old->cutoff) {
        return false;
    }

    DomainPropagator dp;
    dp.load(root_);
    if (implications_ != nullptr) {
        dp.setImplicationGraph(implications_);
    }
    if (cliques_ != nullptr) {
        dp.setCliqueTable(cliques_);
    }
    dp.setObjectiveCutoff(objective_, cutoff);

    auto next = std::make_shared<Snapshot>();
    next->cutoff = cutoff;
    if (!dp.propagate()) {
        next->infeasible = true;
    } else {
        for (Index j = 0; j < root_.num_cols; ++j) {
            Real lb = dp.getLower(j);
            Real ub = dp.getUpper(j);
            // The propagator accepts a continuous column crossed by up to its
            // tolerance; replayed on every node as is, that would prune them
            // all. The swapped interval still holds every feasible value up to
            // rounding, where a single point might leave a row unsatisfiable.
            if (lb > ub) {
                std::swap(lb, ub);
            }
            if (lb > root_.col_lower[j] || ub < root_.col_upper[j]) {
                next->cols.push_back(j);
                next->lower.push_back(lb);
                next->upper.push_back(ub);
            }
        }
    }
    std::lock_guard<std::mutex> lock(mutex_);
    snapshot_ = std::move(next);
    return true;
}

bool ObjectiveCutoffBounds::enforce(std::vector<Real>& lower, std::vector<Real>& upper,
                                    std::vector<Index>& tightened) const {
    const auto snap = current();
    if (snap == nullptr) {
        return true;
    }
    if (snap->infeasible) {
        return false;
    }
    for (std::size_t k = 0; k < snap->cols.size(); ++k) {
        const Index j = snap->cols[k];
        Real lb = std::max(lower[j], snap->lower[k]);
        Real ub = std::min(upper[j], snap->upper[k]);
        if (lb > ub + kCrossTol) {
            return false;
        }
        if (lb > ub) {
            std::swap(lb, ub);
        }
        if (lb > lower[j] || ub < upper[j]) {
            lower[j] = lb;
            upper[j] = ub;
            tightened.push_back(j);
        }
    }
    return true;
}

Int ObjectiveCutoffBounds::numTightenings() const {
    const auto snap = current();
    return snap == nullptr ? 0 : static_cast<Int>(snap->cols.size());
}

}  // namespace mipx
