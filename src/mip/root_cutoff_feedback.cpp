#include "root_cutoff_feedback.h"

#include "mipx/domain.h"

#include <algorithm>
#include <numeric>
#include <utility>

namespace mipx {

LpResult backOutRootCutRound(DualSimplexSolver& lp, Index keep_rows,
                             std::span<const BasisStatus> basis, LpProblem* certificate,
                             std::span<const RootCutBound> staged_bounds) {
    for (const auto& bound : staged_bounds) {
        lp.setColBounds(bound.col, certificate->col_lower[bound.col],
                        certificate->col_upper[bound.col]);
    }

    std::vector<Index> removed(static_cast<std::size_t>(lp.numRows() - keep_rows));
    std::iota(removed.begin(), removed.end(), keep_rows);
    lp.removeRows(removed);
    lp.setBasis(basis);
    const auto restored = lp.solve();

    if (certificate != nullptr) {
        const Index mirror_keep =
            certificate->matrix.numRows() - static_cast<Index>(removed.size());
        while (certificate->matrix.numRows() > mirror_keep) {
            certificate->matrix.removeRow(certificate->matrix.numRows() - 1);
            certificate->row_lower.pop_back();
            certificate->row_upper.pop_back();
            certificate->row_names.pop_back();
        }
        certificate->num_rows = certificate->matrix.numRows();
    }
    return restored;
}

LpResult tryScopedCutBounds(DualSimplexSolver& lp, std::span<const RootCutBound> staged_bounds) {
    auto trial = lp;
    for (const auto& bound : staged_bounds) {
        trial.setColBounds(bound.col, bound.lower, bound.upper);
    }
    const auto result = trial.solve();
    if (result.status == Status::Optimal) {
        lp = std::move(trial);
    }
    return result;
}

RootCutoffFeedback::RootCutoffFeedback(const LpProblem& problem, Real cutoff) : cutoff_(cutoff) {
    objective_.load(problem);
    rc_fixer_.load(problem);
}

void RootCutoffFeedback::strengthenCutoff(Real cutoff) {
    cutoff_ = std::min(cutoff_, cutoff);
}

LpResult RootCutoffFeedback::update(const LpProblem& rows, const DualSimplexSolver& lp,
                                    Int& row_visits_left, Deadline deadline, MipCutStats& stats,
                                    double work_left) {
    const LpResult unchanged{Status::Optimal, lp.getObjective(), 0};
    if (work_left <= 0.0 || deadline.passed() || unchanged.objective >= cutoff_) {
        primals_.clear();
        return unchanged;
    }

    std::vector<Real> lp_lower(rows.num_cols);
    std::vector<Real> lp_upper(rows.num_cols);
    for (Index col = 0; col < rows.num_cols; ++col) {
        lp.getColBounds(col, lp_lower[col], lp_upper[col]);
    }
    auto lower = lp_lower;
    auto upper = lp_upper;

    // Price the endpoints of the LP that supplied the duals. Intersect with
    // the previously retained cutoff domain only after pricing those bounds.
    std::vector<Index> rc_columns;
    RcFixingStats rc_stats;
    const bool rc_feasible =
        rc_fixer_.applyLocalFixing(lp.getReducedCosts(), lp.getPrimalValues(), lp.getObjective(),
                                   cutoff_, lower, upper, rc_columns, rc_stats);
    for (const Index col : rc_columns) {
        stats.root_cutoff_reduced_cost_tightenings +=
            lower_.empty() || lower[col] > std::max(lp_lower[col], lower_[col]) ||
            upper[col] < std::min(lp_upper[col], upper_[col]);
    }
    for (Index col = 0; col < static_cast<Index>(lower_.size()); ++col) {
        lower[col] = std::max(lower[col], lower_[col]);
        upper[col] = std::min(upper[col], upper_[col]);
        if (lower[col] > upper[col]) {
            primals_.clear();
            return unchanged;
        }
    }

    std::vector<ObjectivePropagator::BoundUpdate> updates;
    if (!rc_feasible || !objective_.propagate(cutoff_, lower, upper, updates)) {
        primals_.clear();
        return unchanged;
    }
    stats.root_cutoff_objective_tightenings += static_cast<Int>(updates.size());
    for (const auto& update : updates) {
        lower[update.col] = update.new_lower;
        upper[update.col] = update.new_upper;
    }

    if (lower == lp_lower && upper == lp_upper) {
        lower_.clear();
        upper_.clear();
        primals_.clear();
        return unchanged;
    }

    if (row_visits_left > 0) {
        DomainPropagator propagator;
        propagator.load(rows);
        // Changed endpoints enqueue their incident rows, including installed
        // cuts, and deductions from those rows in turn wake other rows.
        for (Index col = 0; col < rows.num_cols; ++col) {
            propagator.setBound(col, lower[col], upper[col]);
        }
        if (!propagator.propagateWithRowBudget(row_visits_left)) {
            primals_.clear();
            return unchanged;
        }
        for (Index col = 0; col < rows.num_cols; ++col) {
            const Real lb = std::max(lower[col], propagator.getLower(col));
            const Real ub = std::min(upper[col], propagator.getUpper(col));
            if (lb > ub) {
                primals_.clear();
                return unchanged;
            }
            stats.root_cutoff_row_tightenings += (lb > lower[col]) + (ub < upper[col]);
            lower[col] = lb;
            upper[col] = ub;
        }
    }

    return resolve(lp, lp_lower, lp_upper, std::move(lower), std::move(upper), deadline, stats);
}

LpResult RootCutoffFeedback::updateScheduled(const LpProblem& rows, const DualSimplexSolver& lp,
                                             Int& visits_left, DomainPropagationWork& work,
                                             PropagationScheduler& history, Deadline deadline,
                                             MipCutStats& stats, double work_left,
                                             const RootCutoffInference& inference) {
    LpResult unchanged{Status::Optimal, lp.getObjective(), 0};
    const double setup_work =
        1e-6 * (static_cast<double>(rows.matrix.numNonzeros()) + rows.num_rows + rows.num_cols);
    if (work_left <= setup_work || deadline.passed() || unchanged.objective >= cutoff_) {
        primals_.clear();
        return unchanged;
    }
    if (visits_left <= 0) {
        const auto before = history.stats();
        const Int scan_work = cutoffScanWork(rows.num_cols);
        history.enqueue(PropagationProducer::Objective, scan_work);
        history.enqueue(PropagationProducer::ReducedCost, scan_work);
        if (!lower_.empty() || inference.first_new_row.value_or(rows.num_rows) < rows.num_rows) {
            history.enqueue(PropagationProducer::Rows);
        }
        history.finish(PropagationExit::Budget);
        stats.root_propagation_schedule.mergeDelta(history.stats(), before);
        primals_.clear();
        return unchanged;
    }
    unchanged.work_units = setup_work;

    std::vector<Real> lp_lower(rows.num_cols);
    std::vector<Real> lp_upper(rows.num_cols);
    DomainPropagator domain;
    domain.load(rows);
    domain.setObjectiveCutoff(&objective_, cutoff_);
    domain.setCliqueTable(inference.cliques);
    domain.setImplicationGraph(inference.implications);
    for (Index col = 0; col < rows.num_cols; ++col) {
        lp.getColBounds(col, lp_lower[col], lp_upper[col]);
        const Real lb = lower_.empty() ? lp_lower[col] : std::max(lp_lower[col], lower_[col]);
        const Real ub = upper_.empty() ? lp_upper[col] : std::min(lp_upper[col], upper_[col]);
        if (lb > ub) {
            primals_.clear();
            return unchanged;
        }
        if (lb > domain.getLower(col) || ub < domain.getUpper(col)) {
            // A prior pass may have stopped with these wake-ups pending.
            domain.setBound(col, lb, ub);
        }
    }

    const ScheduledReducedCostInput reduced_cost{lp, rc_fixer_, cutoff_};
    const auto previous = work;
    const bool feasible =
        domain.propagateScheduled(visits_left, work, stats.root_propagation_schedule,
                                  {.sweep_rows = false,
                                   .first_new_row = inference.first_new_row,
                                   .deadline = &deadline,
                                   .reduced_cost = &reduced_cost,
                                   .history = &history});
    stats.root_cutoff_objective_tightenings +=
        static_cast<Int>(work.objective.tightenings - previous.objective.tightenings);
    stats.root_cutoff_reduced_cost_tightenings +=
        static_cast<Int>(work.reduced_cost.tightenings - previous.reduced_cost.tightenings);
    stats.root_cutoff_row_tightenings +=
        static_cast<Int>(work.rows.tightenings - previous.rows.tightenings);
    if (!feasible) {
        primals_.clear();
        return unchanged;
    }

    std::vector<Real> lower(rows.num_cols);
    std::vector<Real> upper(rows.num_cols);
    for (Index col = 0; col < rows.num_cols; ++col) {
        lower[col] = domain.getLower(col);
        upper[col] = domain.getUpper(col);
        if (lower[col] > upper[col]) {
            primals_.clear();
            return unchanged;
        }
    }
    if (lower == lp_lower && upper == lp_upper) {
        lower_.clear();
        upper_.clear();
        primals_.clear();
        return unchanged;
    }
    auto result =
        resolve(lp, lp_lower, lp_upper, std::move(lower), std::move(upper), deadline, stats);
    result.work_units += setup_work;
    return result;
}

LpResult RootCutoffFeedback::resolve(const DualSimplexSolver& lp, std::span<const Real> lp_lower,
                                     std::span<const Real> lp_upper, std::vector<Real> lower,
                                     std::vector<Real> upper, Deadline deadline,
                                     MipCutStats& stats) {
    auto scoped_lp = lp;
    auto options = scoped_lp.getOptions();
    options.deadline = deadline;
    scoped_lp.setOptions(options);
    for (Index col = 0; col < lp.numCols(); ++col) {
        if (lower[col] > lp_lower[col] || upper[col] < lp_upper[col]) {
            scoped_lp.setColBounds(col, lower[col], upper[col]);
        }
    }
    const auto result = scoped_lp.solve();
    if (result.status != Status::Optimal) {
        ++stats.root_cutoff_resolves_backed_out;
        return result;
    }
    lower_ = std::move(lower);
    upper_ = std::move(upper);
    primals_ = scoped_lp.getPrimalValues();
    ++stats.root_cutoff_feedbacks;
    return result;
}

}  // namespace mipx
