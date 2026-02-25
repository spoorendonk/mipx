#pragma once

#include <atomic>
#include <cstdint>

namespace mipx {

/// Deterministic work counter for platform-independent performance measurement.
///
/// Counts weighted units of work (nonzeros touched, candidates scanned, etc.)
/// instead of wall-clock time. The count is deterministic in serial mode and
/// approximately reproducible in parallel mode.
///
/// Overhead: one integer addition per instrumented operation.
class WorkUnits {
public:
    WorkUnits() = default;

    /// Add work units (e.g., number of nonzeros processed).
    void count(uint64_t work) { ticks_ += work; }

    /// Get raw tick count.
    [[nodiscard]] uint64_t ticks() const { return ticks_; }

    /// Get work in normalized units (roughly calibrated to seconds on modern hardware).
    [[nodiscard]] double units() const { return static_cast<double>(ticks_) * 1e-6; }

    /// Reset counter.
    void reset() { ticks_ = 0; }

    /// Accumulate from another counter.
    void add(const WorkUnits& other) { ticks_ += other.ticks_; }

private:
    uint64_t ticks_ = 0;
};

/// Thread-safe variant for parallel tree search.
class AtomicWorkUnits {
public:
    AtomicWorkUnits() = default;

    /// Atomically add work units.
    void count(uint64_t work) { ticks_.fetch_add(work, std::memory_order_relaxed); }

    /// Get raw tick count.
    [[nodiscard]] uint64_t ticks() const { return ticks_.load(std::memory_order_relaxed); }

    /// Get work in normalized units.
    [[nodiscard]] double units() const { return static_cast<double>(ticks()) * 1e-6; }

    /// Accumulate from a non-atomic counter.
    void add(const WorkUnits& other) { ticks_.fetch_add(other.ticks(), std::memory_order_relaxed); }

private:
    std::atomic<uint64_t> ticks_{0};
};

}  // namespace mipx
