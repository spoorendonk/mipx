#include "mipx/domain.h"

#include "mipx/clique_table.h"
#include "mipx/implication_graph.h"
#include "mipx/objective_propagator.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>

namespace mipx {

void DomainPropagator::load(const LpProblem& problem, bool record_reasons, Index global_rows) {
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
    if (record_reasons) {
        if (!reasons_) {
            reasons_.emplace();
        }
        reasons_->load(lower_, upper_);
    } else {
        reasons_.reset();
    }
    global_rows_ = global_rows < 0 ? num_rows_ : global_rows;
    reason_level_ = 0;

    row_in_queue_.assign(num_rows_, false);
    queue_.clear();

    objective_tightenings_ = 0;
    objective_infeasible_ = false;
}

void DomainPropagator::setObjectiveCutoff(const ObjectivePropagator* objective, Real cutoff) {
    objective_ = objective;
    objective_cutoff_ = cutoff;
}

void DomainPropagator::setCliqueTable(const CliqueTable* table, BoundReasonScope scope) {
    clique_table_ = table;
    clique_scope_ = scope;
}

void DomainPropagator::setBound(Index col, Real lower, Real upper,
                                std::span<const BranchDecision> path) {
    setBound(col, lower, upper, path, false, reason_level_);
}

void DomainPropagator::setBranchBound(Index col, bool is_upper, Real value, Int level) {
    setBound(col, is_upper ? -kInf : value, is_upper ? value : kInf, {}, true, level);
}

void DomainPropagator::recordInputBound(BoundEndpoint endpoint,
                                        std::span<const BranchDecision> path, bool branch,
                                        Int level) {
    if (!reasons_) {
        return;
    }
    if (branch) {
        reasons_->record(endpoint, BoundReasonSource::Branch, {}, -1, {}, level);
        return;
    }
    const auto previous = reasons_->current(endpoint.col, endpoint.is_upper);
    Real recorded = reasons_->reason(previous).endpoint.value;
    for (const auto& decision : path) {
        if (decision.level <= 0 || decision.variable != endpoint.col ||
            decision.is_upper != endpoint.is_upper) {
            continue;
        }
        const bool entailed =
            endpoint.is_upper ? endpoint.value <= decision.bound : endpoint.value >= decision.bound;
        const bool tightens =
            endpoint.is_upper ? decision.bound < recorded - kTol : decision.bound > recorded + kTol;
        if (entailed && tightens) {
            reasons_->record({endpoint.col, endpoint.is_upper, decision.bound},
                             BoundReasonSource::Branch, {}, -1, {}, decision.level);
            recorded = decision.bound;
        }
    }
    if (recorded != endpoint.value) {
        reasons_->record(endpoint, BoundReasonSource::Opaque, {}, -1, {.node_local = true}, level);
    }
}

void DomainPropagator::setBound(Index col, Real lower, Real upper,
                                std::span<const BranchDecision> path, bool branch, Int level) {
    Real old_lower = lower_[col];
    Real old_upper = upper_[col];
    bool changed = false;

    if (lower > lower_[col] + kTol) {
        lower_[col] = lower;
        changed = true;
        recordInputBound({col, false, lower}, path, branch, level);
    }
    if (upper < upper_[col] - kTol) {
        upper_[col] = upper;
        changed = true;
        recordInputBound({col, true, upper}, path, branch, level);
    }

    // Round for integer variables.
    if (col_type_[col] != VarType::Continuous) {
        const Real rounded_lb = std::ceil(lower_[col] - kIntRoundTol);
        const Real rounded_ub = std::floor(upper_[col] + kIntRoundTol);
        if (rounded_lb > lower_[col] + kTol) {
            lower_[col] = rounded_lb;
            recordRounding(col, false, rounded_lb);
            changed = true;
        }
        if (rounded_ub < upper_[col] - kTol) {
            upper_[col] = rounded_ub;
            recordRounding(col, true, rounded_ub);
            changed = true;
        }
    }

    if (changed) {
        changes_.push_back({col, old_lower, old_upper});
        enqueueColumn(col);
        if (reasons_) {
            // Retain the two endpoints even when this column has no rows.
            (void)checkCrossing(col);
        }
    }
}

Real DomainPropagator::getLower(Index col) const {
    return lower_[col];
}

Real DomainPropagator::getUpper(Index col) const {
    return upper_[col];
}

bool DomainPropagator::propagate(Int max_iterations) {
    if (max_iterations <= 0) {
        const int64_t budget = int64_t{kMaxRowRounds} * std::max<Index>(num_rows_, 1);
        max_iterations =
            static_cast<Int>(std::min<int64_t>(budget, std::numeric_limits<Int>::max()));
    }
    return propagateWithRowBudget(max_iterations);
}

bool DomainPropagator::propagateWithRowBudget(Int& row_visits_left) {
    objective_infeasible_ = false;
    if (row_visits_left <= 0) {
        return true;
    }
    const bool feasible = propagate(row_visits_left, 0);
    if (!feasible && reasons_ && !reasons_->contradiction()) {
        reasons_->fail(
            BoundReasonSource::Opaque, {}, -1,
            {.node_local = true, .cutoff = objective_infeasible_ ? objective_cutoff_ : kInf});
    }
    return feasible;
}

bool DomainPropagator::propagate(Int& row_visits_left, Int objective_passes) {
    // If queue is empty, seed with all rows.
    if (queue_.empty()) {
        for (Index i = 0; i < num_rows_; ++i) {
            queue_.push_back(i);
            row_in_queue_[i] = true;
        }
    }

    while (!queue_.empty()) {
        if (row_visits_left == 0) {
            return true;
        }

        Index row = queue_.back();
        queue_.pop_back();
        row_in_queue_[row] = false;

        --row_visits_left;
        if (!propagateRow(row)) {
            return false;
        }
    }

    // Run clique propagation after row-based propagation converges.
    if (clique_table_ != nullptr && queue_.empty()) {
        if (!propagateCliques()) {
            return false;
        }
        // If clique propagation changed bounds, re-run row propagation.
        if (!queue_.empty()) {
            return row_visits_left == 0 || propagate(row_visits_left, objective_passes);
        }
    }

    // The objective row last: it is dense, so it runs once per fixpoint of
    // the sparse rows rather than whenever one of its columns moves.
    if (objective_ != nullptr && queue_.empty() && objective_passes < kMaxObjectivePasses) {
        if (!propagateObjective()) {
            return false;
        }
        if (!queue_.empty()) {
            return row_visits_left == 0 || propagate(row_visits_left, objective_passes + 1);
        }
    }

    return true;
}

bool DomainPropagator::propagateObjective() {
    std::vector<ObjectivePropagator::BoundUpdate> updates;
    ObjectivePropagator::BoundUpdate crossing;
    const auto objective_row = objective_->row();
    const auto min_reasons = activityReasons(objective_row, true);
    const BoundReasonScope scope{.cutoff = objective_cutoff_};
    const auto record = [&](const ObjectivePropagator::BoundUpdate& upd, bool is_upper, Real value,
                            bool applied) -> BoundReasonId {
        if (!reasons_) {
            return kNoBoundReason;
        }
        if (is_upper != upd.cutoff_is_upper) {
            return recordRounding(upd.col, is_upper, value, applied);
        }
        const auto term = std::ranges::lower_bound(objective_row.indices, upd.col) -
                          objective_row.indices.begin();
        return reasons_->record({upd.col, is_upper, value}, BoundReasonSource::Objective,
                                {min_reasons, static_cast<std::size_t>(term)}, -1, scope, 0,
                                applied);
    };
    const bool feasible = objective_->propagate(objective_cutoff_, lower_, upper_, updates,
                                                reasons_ ? &crossing : nullptr);
    if (!feasible) {
        objective_infeasible_ = true;
        if (!reasons_) {
            return false;
        }
        if (crossing.col < 0) {
            reasons_->fail(BoundReasonSource::Objective, min_reasons, -1, scope);
        } else {
            const Index j = crossing.col;
            const auto lower = crossing.new_lower == lower_[j]
                                   ? reasons_->current(j, false)
                                   : record(crossing, false, crossing.new_lower, false);
            const auto upper = crossing.new_upper == upper_[j]
                                   ? reasons_->current(j, true)
                                   : record(crossing, true, crossing.new_upper, false);
            reasons_->fail(BoundReasonSource::Crossing, {lower, upper}, j, scope);
        }
        return false;
    }
    for (const auto& upd : updates) {
        const Index j = upd.col;
        recordChange(j);
        if (upd.new_lower > lower_[j] + kTol) {
            lower_[j] = upd.new_lower;
            record(upd, false, upd.new_lower, true);
            ++objective_tightenings_;
        }
        if (upd.new_upper < upper_[j] - kTol) {
            upper_[j] = upd.new_upper;
            record(upd, true, upd.new_upper, true);
            ++objective_tightenings_;
        }
        if (reasons_) {
            (void)checkCrossing(j);
        }
        enqueueColumn(j);
        if (implication_graph_ != nullptr && col_type_[j] != VarType::Continuous &&
            lower_[j] > upper_[j] - kTol) {
            if (!propagateImplications(j)) {
                return false;
            }
        }
    }
    return true;
}

bool DomainPropagator::propagateCliques() {
    if (clique_table_ == nullptr) {
        return true;
    }

    std::vector<CliqueTable::BoundUpdate> updates;
    CliqueTable::PropagationFailure failure;
    const bool feasible =
        clique_table_->propagate(lower_, upper_, updates, reasons_ ? &failure : nullptr);
    // Take all antecedents from the input of this batch. An earlier update
    // can otherwise hide the endpoint that triggered a later clique.
    const auto literalReason = [&](const Literal& literal, bool value) {
        return reasons_->current(literal.var, value == literal.complemented);
    };
    const auto supportsFor = [&](Index clique, Index one, Index exclude) {
        const auto& literals = clique_table_->clique(clique).literals;
        std::vector<BoundReasonId> supports;
        if (one >= 0) {
            supports.push_back(literalReason(literals[one], true));
        } else {
            for (Index k = 0; k < static_cast<Index>(literals.size()); ++k) {
                const auto& literal = literals[k];
                if (k != exclude && literal.var >= 0 && literal.var < num_cols_) {
                    supports.push_back(literalReason(literal, false));
                }
            }
        }
        return supports;
    };
    const auto record = [&](const CliqueTable::BoundUpdate& upd,
                            const BoundReasonSupports& supports, bool is_upper, Real value,
                            bool applied) {
        return reasons_->record({upd.col, is_upper, value}, BoundReasonSource::Clique, supports,
                                upd.clique, clique_scope_,
                                clique_scope_.node_local ? reason_level_ : 0, applied);
    };
    if (!feasible) {
        if (!reasons_) {
            return false;
        }
        if (failure.crossing.col < 0) {
            auto supports = supportsFor(failure.clique, failure.fixed_one, -1);
            if (failure.other_one >= 0) {
                supports.push_back(literalReason(
                    clique_table_->clique(failure.clique).literals[failure.other_one], true));
            }
            reasons_->fail(BoundReasonSource::Clique, std::move(supports), failure.clique,
                           clique_scope_);
            return false;
        }
        const auto& upd = failure.crossing;
        const Index j = upd.col;
        const BoundReasonSupports supports = supportsFor(upd.clique, upd.fixed_one, upd.target);
        const auto lower = upd.new_lower == lower_[j]
                               ? reasons_->current(j, false)
                               : record(upd, supports, false, upd.new_lower, false);
        const auto upper = upd.new_upper == upper_[j]
                               ? reasons_->current(j, true)
                               : record(upd, supports, true, upd.new_upper, false);
        reasons_->fail(BoundReasonSource::Crossing, {lower, upper}, j);
        return false;
    }
    std::vector<BoundReasonSupports> batch_supports;
    if (reasons_) {
        batch_supports.reserve(updates.size());
        for (const auto& upd : updates) {
            batch_supports.push_back(supportsFor(upd.clique, upd.fixed_one, upd.target));
        }
    }

    for (std::size_t i = 0; i < updates.size(); ++i) {
        const auto& upd = updates[i];
        if (upd.col < 0 || upd.col >= num_cols_) {
            continue;
        }

        bool changed = false;
        bool recorded = false;
        if (upd.new_lower > lower_[upd.col] + kTol) {
            if (!recorded) {
                recordChange(upd.col);
                recorded = true;
            }
            lower_[upd.col] = upd.new_lower;
            if (reasons_) {
                record(upd, batch_supports[i], false, upd.new_lower, true);
            }
            changed = true;
        }
        if (upd.new_upper < upper_[upd.col] - kTol) {
            if (!recorded) {
                recordChange(upd.col);
                recorded = true;
            }
            upper_[upd.col] = upd.new_upper;
            if (reasons_) {
                record(upd, batch_supports[i], true, upd.new_upper, true);
            }
            changed = true;
        }

        if (!checkCrossing(upd.col)) {
            return false;
        }

        if (changed) {
            enqueueColumn(upd.col);
        }
    }

    return true;
}

void DomainPropagator::pushCheckpoint() {
    checkpoint_stack_.push_back({static_cast<Index>(changes_.size()), reason_level_});
    if (reasons_) {
        reasons_->pushCheckpoint();
    }
}

void DomainPropagator::popCheckpoint() {
    const auto [target, level] = checkpoint_stack_.back();
    checkpoint_stack_.pop_back();
    reason_level_ = level;

    // Undo changes in reverse order.
    while (static_cast<Index>(changes_.size()) > target) {
        auto& c = changes_.back();
        lower_[c.col] = c.old_lower;
        upper_[c.col] = c.old_upper;
        changes_.pop_back();
    }
    if (reasons_) {
        reasons_->popCheckpoint();
    }

    // Clear propagation queue since bounds have been restored.
    for (Index row : queue_) {
        row_in_queue_[row] = false;
    }
    queue_.clear();
}

Int DomainPropagator::numTightened() const {
    Index base = checkpoint_stack_.empty() ? 0 : checkpoint_stack_.back().changes;
    return static_cast<Int>(changes_.size()) - base;
}

void DomainPropagator::collectTightenedColumns(std::vector<Index>& columns) const {
    const auto base = checkpoint_stack_.empty() ? 0 : checkpoint_stack_.back().changes;
    columns.clear();
    columns.reserve(changes_.size() - base);
    for (std::size_t i = base; i < changes_.size(); ++i) {
        columns.push_back(changes_[i].col);
    }
    std::ranges::sort(columns);
    const auto duplicate_tail = std::ranges::unique(columns);
    columns.erase(duplicate_tail.begin(), duplicate_tail.end());
}

bool DomainPropagator::checkCrossing(Index col) {
    if (!(lower_[col] > upper_[col] + kTol)) {
        return true;
    }
    if (reasons_) {
        reasons_->fail(BoundReasonSource::Crossing,
                       {reasons_->current(col, false), reasons_->current(col, true)}, col);
    }
    return false;
}

BoundReasonId DomainPropagator::recordRounding(Index col, bool is_upper, Real value, bool applied) {
    return reasons_ ? reasons_->record({col, is_upper, value}, BoundReasonSource::IntegerRounding,
                                       {reasons_->current(col, is_upper)}, -1, {}, 0, applied)
                    : kNoBoundReason;
}

BoundReasonSupports::Snapshot DomainPropagator::activityReasons(SparseVectorView row,
                                                                bool minimum) const {
    if (!reasons_) {
        return {};
    }
    auto result = std::make_shared<std::vector<BoundReasonId>>();
    result->reserve(row.indices.size());
    for (Index k = 0; k < row.size(); ++k) {
        const auto id = row.values[k] == 0
                            ? kNoBoundReason
                            : reasons_->current(row.indices[k], (row.values[k] < 0) == minimum);
        result->push_back(id);
    }
    return result;
}

bool DomainPropagator::propagateRow(Index row) {
    auto rv = matrix_.row(row);
    Real rl = row_lower_[row];
    Real ru = row_upper_[row];
    // Save the endpoints that contributed to each activity sum. If an
    // earlier implication tightens the target itself, subtracting its live
    // contribution weakens the residual deduction from the saved endpoints.
    const auto min_reasons = activityReasons(rv, true);
    const auto max_reasons = activityReasons(rv, false);
    const BoundReasonScope scope{.node_local = row >= global_rows_};
    const Int level = scope.node_local ? reason_level_ : 0;

    // Compute activity bounds for the row.
    // Track finite sums and count of infinite contributions separately.
    Real act_min = 0.0;
    Real act_max = 0.0;
    Index inf_min_count = 0;
    Index inf_max_count = 0;
    Index inf_min_col = -1;  // Valid only when inf_min_count == 1.
    Index inf_max_col = -1;
    // The row's magnitude: its finite sides and its largest finite term.
    Real scale = 1.0;
    if (!std::isinf(rl)) {
        scale = std::max(scale, std::abs(rl));
    }
    if (!std::isinf(ru)) {
        scale = std::max(scale, std::abs(ru));
    }

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
            scale = std::max(scale, std::abs(contrib_min));
        }

        if (std::isinf(contrib_max)) {
            ++inf_max_count;
            inf_max_col = j;
        } else {
            act_max += contrib_max;
            scale = std::max(scale, std::abs(contrib_max));
        }
    }

    // The row holds to kTol relative to its magnitude (#236). An absolute
    // kTol is below the rounding of the activity sums once terms reach about
    // 1e8, and a point exactly on the row read as infeasible. The derived
    // bounds below take the same slack, so a term smaller than it (a tiny
    // coefficient) bounds nothing instead of amplifying the sums' rounding
    // by 1/|a| into a crossing.
    const Real row_tol = kTol * scale;
    if (inf_min_count == 0 && act_min > ru + row_tol) {
        if (reasons_) {
            reasons_->fail(BoundReasonSource::Row, min_reasons, row, scope);
        }
        return false;
    }
    if (inf_max_count == 0 && act_max < rl - row_tol) {
        if (reasons_) {
            reasons_->fail(BoundReasonSource::Row, max_reasons, row, scope);
        }
        return false;
    }

    // If both sides have 2+ infinite contributions, no tightening is possible.
    if (inf_min_count >= 2 && inf_max_count >= 2) {
        return true;
    }

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

        // Derive new bounds. A derived bound carries the rounding of the row
        // sums and the inexactness of the other columns' bounds, both scaled
        // by the coefficients, so on an integer column it can miss an integer
        // by far more than kTol (#235: 1.9e-7 with terms near 1e7), and
        // rounding at kTol turned that into a whole-unit crossing. Integer
        // candidates round at the LP's feasibility scale before they are
        // compared, so a bound only ever tightens to an integer.
        const bool integral = col_type_[j] != VarType::Continuous;
        Real new_ub = kInf;
        if (a > 0 && !std::isinf(ru) && res_min_finite) {
            new_ub = (ru + row_tol - res_min) / a;
        } else if (a < 0 && !std::isinf(rl) && res_max_finite) {
            new_ub = (rl - row_tol - res_max) / a;
        }
        Real new_lb = -kInf;
        if (a > 0 && !std::isinf(rl) && res_max_finite) {
            new_lb = (rl - row_tol - res_max) / a;
        } else if (a < 0 && !std::isinf(ru) && res_min_finite) {
            new_lb = (ru + row_tol - res_min) / a;
        }
        if (integral) {
            new_ub = std::floor(new_ub + kIntRoundTol);
            new_lb = std::ceil(new_lb - kIntRoundTol);
        }

        bool changed = false;
        if (new_ub < upper_[j] - kTol) {
            recordChange(j);
            upper_[j] = new_ub;
            if (reasons_) {
                reasons_->record({j, true, new_ub}, BoundReasonSource::Row,
                                 {a > 0 ? min_reasons : max_reasons, static_cast<std::size_t>(k)},
                                 row, scope, level);
            }
            changed = true;
        }
        if (new_lb > lower_[j] + kTol) {
            recordChange(j);
            lower_[j] = new_lb;
            if (reasons_) {
                reasons_->record({j, false, new_lb}, BoundReasonSource::Row,
                                 {a > 0 ? max_reasons : min_reasons, static_cast<std::size_t>(k)},
                                 row, scope, level);
            }
            changed = true;
        }
        // The other bound may still be fractional (a model bound).
        if (changed && integral) {
            const Real rounded_lb = std::ceil(lower_[j] - kIntRoundTol);
            const Real rounded_ub = std::floor(upper_[j] + kIntRoundTol);
            if (rounded_lb > lower_[j] + kTol) {
                recordChange(j);
                lower_[j] = rounded_lb;
                recordRounding(j, false, rounded_lb);
            }
            if (rounded_ub < upper_[j] - kTol) {
                recordChange(j);
                upper_[j] = rounded_ub;
                recordRounding(j, true, rounded_ub);
            }
        }

        // Check infeasibility.
        if (!checkCrossing(j)) {
            return false;
        }

        if (changed) {
            enqueueColumn(j);
            // Propagate binary implications when a variable becomes fixed.
            if (implication_graph_ != nullptr && col_type_[j] != VarType::Continuous &&
                lower_[j] > upper_[j] - kTol) {
                if (!propagateImplications(j)) {
                    return false;
                }
            }
        }
    }

    return true;
}

bool DomainPropagator::propagateImplications(Index col) {
    if (implication_graph_ == nullptr) {
        return true;
    }

    std::vector<Index> worklist;
    worklist.push_back(col);

    while (!worklist.empty()) {
        Index cur = worklist.back();
        worklist.pop_back();

        if (!implication_graph_->isBinary(cur)) {
            continue;
        }

        // Determine the fixed value.
        bool fixed_val = (lower_[cur] > 0.5);
        const auto support = reasons_ ? reasons_->current(cur, !fixed_val) : kNoBoundReason;

        const auto& imps = implication_graph_->implications(cur, fixed_val);
        for (const auto& imp : imps) {
            Index j = imp.to_var;
            if (j < 0 || j >= num_cols_) {
                continue;
            }

            Real new_lb = imp.to_val ? 1.0 : lower_[j];
            Real new_ub = imp.to_val ? upper_[j] : 0.0;

            bool bound_changed = false;
            bool recorded = false;
            if (new_lb > lower_[j] + kTol) {
                if (!recorded) {
                    recordChange(j);
                    recorded = true;
                }
                lower_[j] = new_lb;
                if (reasons_) {
                    reasons_->record({j, false, new_lb}, BoundReasonSource::Implication, {support},
                                     cur, implication_scope_,
                                     implication_scope_.node_local ? reason_level_ : 0);
                }
                bound_changed = true;
            }
            if (new_ub < upper_[j] - kTol) {
                if (!recorded) {
                    recordChange(j);
                    recorded = true;
                }
                upper_[j] = new_ub;
                if (reasons_) {
                    reasons_->record({j, true, new_ub}, BoundReasonSource::Implication, {support},
                                     cur, implication_scope_,
                                     implication_scope_.node_local ? reason_level_ : 0);
                }
                bound_changed = true;
            }

            if (!checkCrossing(j)) {
                return false;
            }

            if (bound_changed) {
                enqueueColumn(j);
                // If this variable also became fixed, add to worklist.
                if (lower_[j] > upper_[j] - kTol) {
                    worklist.push_back(j);
                }
            }
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
