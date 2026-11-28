#pragma once

#include "mipx/bnb_node.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mipx {

/// Indexes the conditions of live path no-goods. The owner keeps each
/// literal buffer valid and unchanged until it removes the clause.
class ConflictWatchIndex {
public:
    using Id = std::uint64_t;

    struct QueryResult {
        std::optional<Id> triggered;
        std::uint64_t candidates = 0;
        std::uint64_t body_scans = 0;
        std::uint64_t literal_checks = 0;
    };

    /// Register a nonempty clause in insertion order.
    Id add(std::span<const BranchDecision> literals, Real tolerance) {
        const Id id = next_id_++;
        const std::size_t second = literals.size() > 1 ? 1 : 0;
        clauses_.emplace(id, Clause{literals, {0, second}, tolerance});
        by_column_[literals.front().variable].insert(id);
        if (literals[second].variable != literals.front().variable) {
            by_column_[literals[second].variable].insert(id);
        }
        return id;
    }

    void remove(Id id) {
        const auto it = clauses_.find(id);
        const Clause& clause = it->second;
        const Index first = clause.literals[clause.watches[0]].variable;
        const Index second = clause.literals[clause.watches[1]].variable;
        removeWatch(first, id);
        if (second != first) {
            removeWatch(second, id);
        }
        clauses_.erase(it);
    }

    /// Paths are independent snapshots. A triggered clause must have its
    /// watched columns explicitly present, regardless of earlier paths or
    /// why a condition was chosen as a watch.
    template <typename Entails>
    QueryResult findTriggered(std::span<const Index> path_columns, Entails entails) {
        candidates_.clear();
        for (const Index col : path_columns) {
            if (const auto found = by_column_.find(col); found != by_column_.end()) {
                candidates_.insert(candidates_.end(), found->second.begin(), found->second.end());
            }
        }
        // Snapshot the candidates before moving any watches. Preserve the
        // scan's choice when several clauses trigger on this path.
        std::sort(candidates_.begin(), candidates_.end());
        candidates_.erase(std::unique(candidates_.begin(), candidates_.end()), candidates_.end());
        QueryResult result;
        for (const Id id : candidates_) {
            ++result.candidates;
            Clause& clause = clauses_.at(id);
            const auto entailed = [&](std::size_t i) {
                ++result.literal_checks;
                return entails(clause.literals[i], clause.tolerance);
            };
            const auto [first, second] = clause.watches;
            if (!entailed(first) || (second != first && !entailed(second))) {
                continue;
            }
            if (clause.literals.size() > 2) {
                ++result.body_scans;
                for (std::size_t i = 0; i < clause.literals.size(); ++i) {
                    if (i != first && i != second && !entailed(i)) {
                        moveFirstWatch(id, clause, i);
                        break;
                    }
                }
                if (clause.watches[0] != first) {
                    continue;
                }
            }
            result.triggered = id;
            break;
        }
        return result;
    }

private:
    struct Clause {
        std::span<const BranchDecision> literals;
        std::array<std::size_t, 2> watches;
        Real tolerance;
    };

    void removeWatch(Index col, Id id) {
        const auto it = by_column_.find(col);
        it->second.erase(id);
        if (it->second.empty()) {
            by_column_.erase(it);
        }
    }

    void moveFirstWatch(Id id, Clause& clause, std::size_t replacement) {
        const Index old = clause.literals[clause.watches[0]].variable;
        const Index kept = clause.literals[clause.watches[1]].variable;
        const Index next = clause.literals[replacement].variable;
        if (old != next) {
            if (next != kept) {
                by_column_[next].insert(id);
            }
            if (old != kept) {
                removeWatch(old, id);
            }
        }
        clause.watches[0] = replacement;
    }

    Id next_id_ = 0;
    std::unordered_map<Id, Clause> clauses_;
    std::unordered_map<Index, std::unordered_set<Id>> by_column_;
    std::vector<Id> candidates_;
};

}  // namespace mipx
