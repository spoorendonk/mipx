#pragma once

#include "mipx/core.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <optional>

namespace mipx {

// Declaration order breaks ties and tries unmeasured sparse producers first.
enum class PropagationProducer {
    Implications,
    Cliques,
    Rows,
    InactiveCuts,
    Objective,
    ReducedCost,
    Count
};
enum class PropagationExit { Empty, Budget, Deadline, Contradiction, Count };

/// One unit per block of up to 16 columns in an objective or price scan.
[[nodiscard]] constexpr Int cutoffScanWork(Index columns) {
    constexpr Int kColumnsPerWorkUnit = 16;
    return std::max<Int>(1, columns / kColumnsPerWorkUnit + (columns % kColumnsPerWorkUnit != 0));
}

struct PropagationProducerStats {
    std::uint64_t calls = 0;
    std::uint64_t work = 0;
    // An endpoint is counted once per call, even when it changes twice.
    std::uint64_t endpoints = 0;
    std::uint64_t deferrals = 0;
    std::array<std::uint64_t, static_cast<std::size_t>(PropagationExit::Count)> exits{};

    bool operator==(const PropagationProducerStats&) const = default;
};

struct PropagationScheduleStats {
    static constexpr auto kCount = static_cast<std::size_t>(PropagationProducer::Count);
    std::array<PropagationProducerStats, kCount> producers{};

    void merge(const PropagationScheduleStats& sample) {
        mergeDelta(sample, PropagationScheduleStats{});
    }

    void mergeDelta(const PropagationScheduleStats& sample,
                    const PropagationScheduleStats& before) {
        for (std::size_t i = 0; i < producers.size(); ++i) {
            auto& total = producers[i];
            const auto& after = sample.producers[i];
            const auto& prior = before.producers[i];
            total.calls += after.calls - prior.calls;
            total.work += after.work - prior.work;
            total.endpoints += after.endpoints - prior.endpoints;
            total.deferrals += after.deferrals - prior.deferrals;
            for (std::size_t j = 0; j < total.exits.size(); ++j) {
                total.exits[j] += after.exits[j] - prior.exits[j];
            }
        }
    }
};

class PropagationScheduler {
public:
    void enqueue(PropagationProducer producer, Int minimum_work = 1) {
        const auto i = static_cast<std::size_t>(producer);
        pending_[i] = minimum_work;
        participated_[i] = true;
    }

    [[nodiscard]] bool pending() const {
        for (const auto work : pending_) {
            if (work > 0) {
                return true;
            }
        }
        return false;
    }

    [[nodiscard]] std::optional<PropagationProducer> next(Int work_left) {
        std::size_t best = pending_.size();
        auto next_probe = std::numeric_limits<std::uint64_t>::max();
        for (std::size_t i = 0; i < pending_.size(); ++i) {
            if (pending_[i] <= 0 || pending_[i] > work_left) {
                continue;
            }
            if (resume_[i] > epoch_ + 1) {
                next_probe = std::min(next_probe, resume_[i]);
            } else if (best == pending_.size() || precedes(i, best)) {
                best = i;
            }
        }
        if (best == pending_.size()) {
            if (next_probe == std::numeric_limits<std::uint64_t>::max()) {
                return std::nullopt;
            }
            // All affordable producers are deferred. Advance the idle epochs
            // without inventing work or dropping any pending input.
            epoch_ = next_probe - 1;
            return next(work_left);
        }
        ++epoch_;
        pending_[best] = 0;
        resume_[best] = 0;
        return static_cast<PropagationProducer>(best);
    }

    void record(PropagationProducer producer, std::uint64_t work, std::uint64_t endpoints) {
        const auto i = static_cast<std::size_t>(producer);
        auto& stat = stats_.producers[i];
        ++stat.calls;
        stat.work += work;
        stat.endpoints += endpoints;
        if (endpoints > 0) {
            zero_calls_[i] = 0;
        } else if (++zero_calls_[i] == 3) {
            ++stat.deferrals;
            resume_[i] = epoch_ + 4;
            // A reprobe starts a new three-call measurement window.
            zero_calls_[i] = 0;
        }
    }

    void finish(PropagationExit exit) {
        for (std::size_t i = 0; i < pending_.size(); ++i) {
            const bool count = exit == PropagationExit::Empty ? participated_[i] : pending_[i] > 0;
            stats_.producers[i].exits[static_cast<std::size_t>(exit)] += count;
        }
        pending_.fill(0);
        participated_.fill(false);
    }

    [[nodiscard]] const PropagationScheduleStats& stats() const { return stats_; }

private:
    [[nodiscard]] bool precedes(std::size_t lhs, std::size_t rhs) const {
        // A due, affordable probe gets a sample even while a productive
        // producer remains ready. Producer order breaks ties between probes.
        if (resume_[lhs] != 0 || resume_[rhs] != 0) {
            return resume_[lhs] != 0 && resume_[rhs] == 0;
        }
        return higherYield(lhs, rhs);
    }

    [[nodiscard]] bool higherYield(std::size_t lhs, std::size_t rhs) const {
        const auto& a = stats_.producers[lhs];
        const auto& b = stats_.producers[rhs];
        if (a.work == 0 || b.work == 0) {
            return a.work == 0 && b.work > 0;
        }
        return static_cast<long double>(a.endpoints) * b.work >
               static_cast<long double>(b.endpoints) * a.work;
    }

    PropagationScheduleStats stats_;
    std::array<Int, PropagationScheduleStats::kCount> pending_{};
    std::array<bool, PropagationScheduleStats::kCount> participated_{};
    std::array<Int, PropagationScheduleStats::kCount> zero_calls_{};
    std::array<std::uint64_t, PropagationScheduleStats::kCount> resume_{};
    std::uint64_t epoch_ = 0;
};

}  // namespace mipx
