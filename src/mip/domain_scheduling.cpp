#include "mipx/clique_table.h"
#include "mipx/deadline.h"
#include "mipx/domain.h"
#include "mipx/implication_graph.h"

#include <algorithm>
#include <deque>
#include <utility>

namespace mipx {

bool DomainPropagator::propagateScheduled(Int& visits_left, DomainPropagationWork& work,
                                          PropagationScheduleStats& stats,
                                          const ScheduledPropagationInputs& inputs) {
    using Producer = PropagationProducer;
    const auto stopped = [&] { return inputs.deadline != nullptr && inputs.deadline->passed(); };
    std::uint32_t polled = 0;
    bool deadline_reached = false;
    const auto poll = [&] {
        if (!deadline_reached && (polled++ & 255U) == 0) {
            deadline_reached = stopped();
        }
        return deadline_reached;
    };
    const Index first = inputs.first_new_row.value_or(num_rows_);
    Index next_sweep = !inputs.first_new_row && inputs.sweep_rows ? num_rows_ : 0;
    const Index num_cliques = clique_table_ != nullptr ? clique_table_->numCliques() : 0;
    PropagationScheduler scheduler;
    // A setup stop reports requested producer scans. Some inputs have not
    // been inspected yet, so these counts are an upper bound on eligibility.
    const auto stopDuringSetup = [&](PropagationExit exit) {
        if (first < num_rows_ || next_sweep > 0 || !queue_.empty()) {
            scheduler.enqueue(Producer::Rows);
        }
        if (!inputs.first_new_row && num_cliques > 0) {
            scheduler.enqueue(Producer::Cliques);
        }
        if (!inputs.first_new_row && implication_graph_ != nullptr &&
            implication_graph_->numImplications() > 0) {
            scheduler.enqueue(Producer::Implications);
        }
        if (!inputs.inactive.empty()) {
            scheduler.enqueue(Producer::InactiveCuts);
        }
        scheduler.finish(exit);
        stats.merge(scheduler.stats());
        return true;
    };
    if (stopped()) {
        return stopDuringSetup(PropagationExit::Deadline);
    }
    if (visits_left <= 0) {
        return stopDuringSetup(PropagationExit::Budget);
    }
    for (Index row = std::max<Index>(0, first); row < num_rows_ && !poll(); ++row) {
        if (!row_in_queue_[row]) {
            row_in_queue_[row] = true;
            queue_.push_back(row);
        }
    }

    if (stopped()) {
        return stopDuringSetup(PropagationExit::Deadline);
    }
    std::deque<Index> cliques;
    std::vector<bool> clique_pending(num_cliques);
    const auto enqueueClique = [&](Index clique) {
        if (!clique_pending[clique]) {
            clique_pending[clique] = true;
            cliques.push_back(clique);
        }
    };
    if (!inputs.first_new_row && num_cliques > 0) {
        const Index start = std::clamp(inputs.first_clique, Index{0}, num_cliques);
        for (Index i = start; i < num_cliques && !poll(); ++i) {
            enqueueClique(i);
        }
        for (Index i = 0; i < start && !poll(); ++i) {
            enqueueClique(i);
        }
    }

    if (stopped()) {
        return stopDuringSetup(PropagationExit::Deadline);
    }
    struct PendingEdges {
        Index from;
        std::span<const BinaryImplication> edges;
    };
    std::deque<PendingEdges> implications;
    std::vector<bool> implication_seen(implication_graph_ != nullptr ? num_cols_ : 0);
    const auto enqueueImplications = [&](Index col) {
        if (implication_graph_ == nullptr || col < 0 || col >= num_cols_ || implication_seen[col] ||
            !implication_graph_->isBinary(col) || lower_[col] <= upper_[col] - kTol) {
            return;
        }
        implication_seen[col] = true;
        const auto& edges = implication_graph_->implications(col, lower_[col] > 0.5);
        if (!edges.empty()) {
            implications.push_back({col, edges});
        }
    };
    if (!inputs.first_new_row && implication_graph_ != nullptr) {
        for (const Index col : implication_graph_->binaryVars()) {
            if (poll()) {
                break;
            }
            enqueueImplications(col);
        }
    }

    if (stopped()) {
        return stopDuringSetup(PropagationExit::Deadline);
    }
    std::deque<std::size_t> inactive;
    std::vector<bool> inactive_pending(inputs.inactive.size());
    std::vector<std::vector<std::size_t>> inactive_by_col(inputs.inactive.empty() ? 0 : num_cols_);
    const auto enqueueInactive = [&](std::size_t cut) {
        if (!inactive_pending[cut]) {
            inactive_pending[cut] = true;
            inactive.push_back(cut);
        }
    };
    for (std::size_t i = 0; i < inputs.inactive.size() && !poll(); ++i) {
        const auto& cut = *inputs.inactive[i];
        if (cut.isUnconditionalGlobal()) {
            enqueueInactive(i);
            for (Index col : cut.indices) {
                if (poll()) {
                    return stopDuringSetup(PropagationExit::Deadline);
                }
                inactive_by_col[col].push_back(i);
            }
        }
    }

    if (stopped()) {
        return stopDuringSetup(PropagationExit::Deadline);
    }
    std::vector<bool> swept_rows(next_sweep);
    const auto enqueueReadyProducers = [&] {
        while (next_sweep > 0 && swept_rows[next_sweep - 1]) {
            --next_sweep;
        }
        if (!queue_.empty() || next_sweep > 0) {
            scheduler.enqueue(Producer::Rows);
        }
        if (!implications.empty()) {
            scheduler.enqueue(Producer::Implications);
        }
        if (!cliques.empty()) {
            scheduler.enqueue(Producer::Cliques);
        }
        if (!inactive.empty()) {
            scheduler.enqueue(Producer::InactiveCuts);
        }
    };
    enqueueReadyProducers();

    Int clique_visit = 0;
    DomainPropagationWork clique_work;
    InferencePass clique_pass{.shared_visits_left = clique_visit,
                              .inference_visits_left = 0,
                              .work = clique_work,
                              .next_clique = 0,
                              .cliques_left = 0,
                              .seen = {},
                              .next_unvisited_row = 0,
                              .visited_rows = {},
                              .incident_cliques = std::vector<Index>(1),
                              .follow_implications = false};
    // Compare the first saved version of each changed column with its final
    // endpoints. Multiple writes in one visit yield just one count per side.
    std::vector<std::size_t> first_change(num_cols_, kNoBoundReason);
    std::vector<Index> changed_cols;
    bool feasible = true;
    PropagationExit exit;
    while (true) {
        if (stopped()) {
            exit = PropagationExit::Deadline;
            break;
        }
        const auto producer = scheduler.next(visits_left);
        if (!producer) {
            exit = scheduler.pending() ? PropagationExit::Budget : PropagationExit::Empty;
            break;
        }
        const auto before = changes_.size();
        PropagationWork* metric;
        Index visited_clique = -1;
        switch (*producer) {
            case Producer::Rows: {
                Index row;
                if (queue_.empty()) {
                    row = --next_sweep;
                } else {
                    row = queue_.back();
                    queue_.pop_back();
                    row_in_queue_[row] = false;
                }
                if (!swept_rows.empty()) {
                    swept_rows[row] = true;
                }
                feasible = propagateActivity(matrix_.row(row), row_lower_[row], row_upper_[row],
                                             row, nullptr, nullptr, inputs.deadline, false);
                if (stopped() && feasible && !row_in_queue_[row]) {
                    row_in_queue_[row] = true;
                    queue_.push_back(row);
                }
                metric = &work.rows;
                break;
            }
            case Producer::Cliques: {
                const Index clique = cliques.front();
                visited_clique = clique;
                cliques.pop_front();
                clique_pending[clique] = false;
                clique_visit = clique_pass.inference_visits_left = clique_pass.cliques_left = 1;
                clique_pass.next_clique = 0;
                clique_pass.incident_cliques->front() = clique;
                feasible = propagateCliques(&clique_pass);
                metric = &work.cliques;
                break;
            }
            case Producer::Implications: {
                auto& pending = implications.front();
                feasible = propagateImplication(pending.from, pending.edges.front(), nullptr);
                pending.edges = pending.edges.subspan(1);
                if (pending.edges.empty()) {
                    implications.pop_front();
                }
                metric = &work.implications;
                break;
            }
            case Producer::InactiveCuts: {
                const auto index = inactive.front();
                inactive.pop_front();
                inactive_pending[index] = false;
                const auto& cut = *inputs.inactive[index];
                feasible = propagateActivity({cut.indices, cut.values}, cut.lower, cut.upper, -1,
                                             nullptr, nullptr, inputs.deadline, false);
                if (stopped() && feasible) {
                    enqueueInactive(index);
                }
                metric = &work.inactive_cuts;
                break;
            }
            case Producer::Count:
                std::unreachable();
        }

        for (auto i = before; i < changes_.size(); ++i) {
            const Index col = changes_[i].col;
            if (first_change[col] == kNoBoundReason) {
                first_change[col] = i;
                changed_cols.push_back(col);
            }
        }
        std::uint64_t endpoints = 0;
        for (const Index col : changed_cols) {
            const auto& old = changes_[first_change[col]];
            endpoints += (lower_[col] > old.old_lower) + (upper_[col] < old.old_upper);
            first_change[col] = kNoBoundReason;
            if (poll()) {
                continue;
            }
            if (num_cliques > 0) {
                for (const bool complemented : {false, true}) {
                    for (const Index clique : clique_table_->cliquesOf({col, complemented})) {
                        if (poll()) {
                            break;
                        }
                        if (clique != visited_clique) {
                            enqueueClique(clique);
                        }
                    }
                }
            }
            enqueueImplications(col);
            if (!inactive_by_col.empty()) {
                for (const auto cut : inactive_by_col[col]) {
                    if (poll()) {
                        break;
                    }
                    enqueueInactive(cut);
                }
            }
        }
        changed_cols.clear();
        // As in the row-budget path, a visit that started is charged when a
        // deadline stops it partway through.
        --visits_left;
        ++metric->visits;
        metric->tightenings += endpoints;
        scheduler.record(*producer, 1, endpoints);
        if (!feasible) {
            scheduler.enqueue(*producer);
            exit = PropagationExit::Contradiction;
            break;
        }
        enqueueReadyProducers();
    }
    scheduler.finish(exit);
    stats.merge(scheduler.stats());
    return feasible;
}

}  // namespace mipx
