#include "mipx/objective_integrality.h"

#include "mipx/lp_problem.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

namespace mipx {

namespace {

constexpr std::int64_t kMaxDenominator = 1000;
constexpr std::int64_t kMaxScale = 1000000;
constexpr Real kRationalTol = 1e-9;
// The pruning tolerance every node test used before #261.
constexpr Real kStandardPruneTol = 1e-6;
// The margin left above incumbent - s for the LP bound's own rounding: a share
// of s, or a relative one on large objectives, whichever is larger.
constexpr Real kLatticeMargin = 1e-3;
constexpr Real kRelativeMargin = 1e-7;
// Coefficients beyond this are integers or rejected: m * c must stay exact.
constexpr Real kMaxCoefficient = 1e9;

bool isIntegralType(VarType type) {
    return type == VarType::Integer || type == VarType::Binary || type == VarType::SemiInteger;
}

// The denominator of the first continued-fraction convergent of x > 0 within
// kRationalTol of it, or 0 when none has a denominator up to kMaxDenominator.
std::int64_t rationalDenominator(Real x) {
    const Real tol = kRationalTol * std::max(1.0, x);
    if (std::abs(x - std::round(x)) <= tol) {
        return 1;
    }
    if (x > kMaxCoefficient) {
        return 0;
    }
    // h/k runs through the convergents, seeded with h_-2/k_-2 = 0/1 and
    // h_-1/k_-1 = 1/0.
    std::int64_t h_prev = 0;
    std::int64_t h = 1;
    std::int64_t k_prev = 1;
    std::int64_t k = 0;
    Real r = x;
    while (true) {
        const auto a = static_cast<std::int64_t>(std::floor(r));
        const std::int64_t h_next = a * h + h_prev;
        const std::int64_t k_next = a * k + k_prev;
        if (k_next > kMaxDenominator) {
            return 0;
        }
        if (std::abs(x - static_cast<Real>(h_next) / static_cast<Real>(k_next)) <= tol) {
            return k_next;
        }
        h_prev = h;
        h = h_next;
        k_prev = k;
        k = k_next;
        const Real frac = r - static_cast<Real>(a);
        if (frac <= 0.0) {
            return 0;
        }
        r = 1.0 / frac;
        // From here k >= 1, so a partial quotient past the limit takes the
        // next denominator past it too; stopping here keeps a * h exact.
        if (r > static_cast<Real>(kMaxDenominator) + 1.0) {
            return 0;
        }
    }
}

}  // namespace

Real ObjectiveLattice::pruneBound(Real incumbent) const {
    const Real standard = incumbent - kStandardPruneTol;
    if (!detected() || !std::isfinite(incumbent)) {
        return standard;
    }
    // Every solution the standard test keeps is below `standard`, so on the
    // lattice it is at or below the last lattice point there. The incumbent
    // need not be on the lattice itself: a root fixing may exclude it.
    const Real below = offset + (std::floor((standard - offset) / scale) * scale);
    const Real margin =
        std::max(kLatticeMargin * scale, kRelativeMargin * std::max(1.0, std::abs(below)));
    return std::min(standard, below + margin);
}

ObjectiveLattice detectObjectiveLattice(const LpProblem& problem) {
    Real offset = problem.obj_offset;
    std::vector<Real> coefs;
    for (Index j = 0; j < problem.num_cols; ++j) {
        const Real c = problem.obj[j];
        if (c == 0.0) {
            continue;
        }
        const VarType type = problem.col_type[j];
        // A semi-continuous column is 0 or anywhere in [l, u], fixed or not.
        if (type == VarType::SemiContinuous) {
            return {};
        }
        if (type != VarType::SemiInteger && problem.col_lower[j] == problem.col_upper[j]) {
            offset += c * problem.col_lower[j];
            continue;
        }
        if (!isIntegralType(type)) {
            return {};
        }
        coefs.push_back(std::abs(c));
    }
    if (coefs.empty() || !std::isfinite(offset)) {
        return {};
    }

    std::int64_t m = 1;
    for (const Real c : coefs) {
        const std::int64_t q = rationalDenominator(c);
        if (q == 0) {
            return {};
        }
        m = std::lcm(m, q);
        if (m > kMaxScale) {
            return {};
        }
    }
    const Real scale_m = static_cast<Real>(m);
    std::int64_t g = 0;
    for (const Real c : coefs) {
        const Real mc = scale_m * c;
        if (mc > 9.0e15) {
            return {};
        }
        const std::int64_t k = std::llround(mc);
        if (std::abs(mc - static_cast<Real>(k)) > kRationalTol * scale_m * std::max(1.0, c)) {
            return {};
        }
        g = std::gcd(g, k);
    }
    if (g == 0) {
        return {};
    }
    ObjectiveLattice lattice;
    lattice.scale = static_cast<Real>(g) / scale_m;
    lattice.offset = offset;
    return lattice;
}

}  // namespace mipx
