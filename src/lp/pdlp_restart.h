#pragma once

#include "mipx/pdlp.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>

namespace mipx::pdlp_detail {

struct ProjectedErrors {
    Real gap = std::numeric_limits<Real>::infinity();
    Real kkt = std::numeric_limits<Real>::infinity();

    static ProjectedErrors measure(Real primal, Real dual, Real pobj, Real dobj) {
        // Scale both objectives and the unit term before either the difference
        // or the sum of magnitudes can overflow.
        const Real safe_magnitude = std::numeric_limits<Real>::max() / 2.0;
        const Real unit = std::max(std::abs(pobj), std::abs(dobj)) > safe_magnitude ? 0.5 : 1.0;
        pobj *= unit;
        dobj *= unit;
        const Real gap = std::abs(pobj - dobj);
        const Real normalized_gap = gap / (unit + std::abs(pobj));
        const Real relative_gap = gap / (unit + std::abs(pobj) + std::abs(dobj));
        const Real kkt = std::isfinite(primal) && std::isfinite(dual) && std::isfinite(relative_gap)
                             ? std::max({primal, dual, relative_gap})
                             : std::numeric_limits<Real>::infinity();
        return {normalized_gap, kkt};
    }
};

class RestartPolicy {
public:
    RestartPolicy(const PdlpOptions& options, PdlpRestartStats& stats)
        : options_(options), stats_(stats) {}

    [[nodiscard]] bool needsProjectedErrors() const {
        return enabled(options_.restart_gap_decay) || enabled(options_.restart_kkt_decay) ||
               options_.adaptive_eval_frequency;
    }

    bool restart(bool first_batch, Int inner_count, Int phase_count, Real fpe, Real initial_fpe,
                 Real last_trial_fpe, ProjectedErrors errors, Int total_count) {
        if (first_batch)
            return record(stats_.first_batch, "first batch", errors, total_count);
        if (initial_fpe > 0.0 && fpe <= options_.restart_sufficient_decay * initial_fpe)
            return record(stats_.sufficient_fpe, "sufficient FPE decay", errors, total_count);
        if (initial_fpe > 0.0 && fpe <= options_.restart_necessary_decay * initial_fpe &&
            fpe > last_trial_fpe)
            return record(stats_.necessary_fpe, "necessary FPE decay", errors, total_count);
        if (inner_count >=
            static_cast<Int>(options_.restart_artificial_fraction * static_cast<Real>(phase_count)))
            return record(stats_.artificial, "artificial", errors, total_count);
        if (decayed(errors.gap, reference_.gap, options_.restart_gap_decay))
            return record(stats_.gap, "duality gap decay", errors, total_count);
        if (decayed(errors.kkt, reference_.kkt, options_.restart_kkt_decay))
            return record(stats_.kkt, "KKT error decay", errors, total_count);
        return false;
    }

private:
    static bool enabled(Real fraction) { return fraction > 0.0 && fraction < 1.0; }

    static bool decayed(Real value, Real initial, Real fraction) {
        return enabled(fraction) && initial > 0.0 && std::isfinite(initial) &&
               value <= fraction * initial;
    }

    bool record(Int& count, const char* rule, ProjectedErrors errors, Int total_count) {
        // The last projected check that triggered a restart is the reference.
        reference_ = errors;
        ++count;
        if (options_.verbose)
            std::printf("PDLP restart at %d (%s, count=%d)\n", total_count, rule, count);
        return true;
    }

    const PdlpOptions& options_;
    PdlpRestartStats& stats_;
    ProjectedErrors reference_;
};

class EvaluationCadence {
public:
    explicit EvaluationCadence(const PdlpOptions& options)
        : adaptive_(options.adaptive_eval_frequency),
          interval_(adaptive_ ? std::clamp(options.termination_eval_frequency, kMin, kMax)
                              : std::max<Int>(1, options.termination_eval_frequency)) {}

    [[nodiscard]] Int interval() const { return interval_; }

    void observe(Real kkt) {
        if (!adaptive_)
            return;
        ++checks_;
        if (checks_ > 1) {
            if (kkt >= previous_kkt_)
                interval_ = std::max(kMin, interval_ / 2);
            else if (kkt <= 0.8 * previous_kkt_)
                interval_ = std::min(kMax, interval_ * 2);
        }
        previous_kkt_ = kkt;
    }

    // Preserve the fixed-mode certificate and logging schedule, even for a
    // partial last batch. In adaptive mode count evaluations instead.
    [[nodiscard]] bool periodicCheck(Int phase_count) const {
        if (adaptive_)
            return checks_ % 10 == 0;
        return phase_count % (10 * static_cast<std::int64_t>(interval_)) == 0;
    }

private:
    static constexpr Int kMin = 50;
    static constexpr Int kMax = 1000;

    bool adaptive_;
    Int interval_;
    Int checks_ = 0;
    Real previous_kkt_ = std::numeric_limits<Real>::infinity();
};

}  // namespace mipx::pdlp_detail
