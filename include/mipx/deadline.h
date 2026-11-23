#pragma once

#include <chrono>

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
    [[nodiscard]] static Deadline after(double seconds) {
        constexpr double kNever = 1e9;  // ~32 years
        Deadline d;
        if (seconds >= 0.0 && seconds <= kNever) {
            d.set_ = true;
            d.at_ = Clock::now() + std::chrono::duration_cast<Clock::duration>(
                                       std::chrono::duration<double>(seconds));
        }
        return d;
    }

    [[nodiscard]] bool isSet() const { return set_; }
    [[nodiscard]] bool passed() const { return set_ && Clock::now() >= at_; }
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
};

}  // namespace mipx
