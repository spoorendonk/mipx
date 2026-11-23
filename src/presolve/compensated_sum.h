#pragma once

#include "mipx/lp_problem.h"

namespace mipx {

/// A sum kept as hi + lo with the rounding error of every addition in lo
/// (Knuth's TwoSum), as HiGHS keeps activities (HighsCDouble). A term much
/// larger than the rest, e.g. 1e17 next to 0.5, can then be taken back out
/// and leave the rest, where a plain sum leaves 0 (#300).
struct CompensatedSum {
    Real hi = 0.0;
    Real lo = 0.0;

    void add(Real x) {
        const Real s = hi + x;
        const Real bb = s - hi;
        lo += (hi - (s - bb)) + (x - bb);
        hi = s;
    }
    /// The sum less x, rounded once.
    [[nodiscard]] Real without(Real x) const {
        const Real s = hi - x;
        const Real bb = s - hi;
        const Real err = (hi - (s - bb)) + (-x - bb);
        return s + (err + lo);
    }
    [[nodiscard]] Real value() const { return hi + lo; }
};

}  // namespace mipx
