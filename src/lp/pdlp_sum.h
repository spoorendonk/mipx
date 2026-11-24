#pragma once

#include "mipx/core.h"

#include <cmath>

namespace mipx::pdlp_detail {

// Neumaier's variant of Kahan summation also recovers the rounding error
// when the new term is larger than the running sum.
class ConvergenceSum {
public:
    explicit ConvergenceSum(Real initial = 0.0) : sum_(initial) {}

    void add(Real term) {
        const Real next = sum_ + term;
        correction_ +=
            std::abs(sum_) >= std::abs(term) ? (sum_ - next) + term : (term - next) + sum_;
        sum_ = next;
    }

    void merge(const ConvergenceSum& other) {
        add(other.sum_);
        if (std::isfinite(other.sum_))
            add(other.correction_);
    }

    [[nodiscard]] Real value() const {
        // Preserve the ordinary sum's infinities and NaNs if compensation
        // encountered an indeterminate subtraction.
        return std::isfinite(sum_) ? sum_ + correction_ : sum_;
    }

    [[nodiscard]] Real scaledWithOffset(Real scale, Real offset) const {
        if (!std::isfinite(sum_) || !std::isfinite(scale))
            return offset + value() * scale;
        // Apply the offset before the correction is rounded into the total.
        return std::fma(correction_, scale, std::fma(sum_, scale, offset));
    }

private:
    Real sum_;
    Real correction_ = 0.0;
};

}  // namespace mipx::pdlp_detail
