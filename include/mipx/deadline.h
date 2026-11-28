#pragma once

#include <atomic>
#include <chrono>
#include <optional>

namespace mipx {

/// A wall-clock point a solve must stop by; none by default. Checked inside
/// long loops (the barrier's iterations and factorizations, #275), so it is
/// one comparison when unset and one steady_clock read when set.
class Deadline {
public:
    using Clock = std::chrono::steady_clock;

    Deadline() = default;
    /// seconds < 0 (or NaN): none; 0 or more: that long from now. Above
    /// kNever (an infinite limit included) also none: the clock's duration
    /// would overflow.
    [[nodiscard]] static Deadline after(double seconds,
                                        std::optional<Clock::time_point> start = std::nullopt) {
        constexpr double kNever = 1e9;  // ~32 years
        Deadline d;
        if (seconds >= 0.0 && seconds <= kNever) {
            d.set_ = true;
            d.at_ = (start ? *start : Clock::now()) + std::chrono::duration_cast<Clock::duration>(
                                                          std::chrono::duration<double>(seconds));
        }
        return d;
    }

    /// Keep this stop flag and use the earlier of the two wall-clock limits.
    [[nodiscard]] Deadline limitBy(const Deadline& other) const {
        Deadline d = *this;
        if (other.set_ && (!d.set_ || other.at_ < d.at_)) {
            d.set_ = true;
            d.at_ = other.at_;
        }
        return d;
    }

    /// This deadline, passed also once `*flag` is true: a caller's stop
    /// request (the MIP root race's), which the long loops then end on too.
    /// The flag must outlive the copy; nullptr adds none.
    [[nodiscard]] Deadline orStop(const std::atomic<bool>* flag) const {
        Deadline d = *this;
        d.stop_ = flag;
        return d;
    }

    /// Whether a time is set; a stop flag alone is not one.
    [[nodiscard]] bool isSet() const { return set_; }
    /// The time has passed, or the stop flag is raised.
    [[nodiscard]] bool passed() const { return (set_ && Clock::now() >= at_) || stopped(); }
    [[nodiscard]] bool passed(Clock::time_point now) const {
        return (set_ && now >= at_) || stopped();
    }
    /// The stop flag is raised, whatever the time.
    [[nodiscard]] bool stopped() const {
        return stop_ != nullptr && stop_->load(std::memory_order_relaxed);
    }
    /// Seconds left, at least 0; -1 when none is set.
    [[nodiscard]] double remainingSeconds() const {
        if (!set_) {
            return -1.0;
        }
        const double left = std::chrono::duration<double>(at_ - Clock::now()).count();
        return left > 0.0 ? left : 0.0;
    }

private:
    bool set_ = false;
    Clock::time_point at_{};
    const std::atomic<bool>* stop_ = nullptr;
};

}  // namespace mipx
