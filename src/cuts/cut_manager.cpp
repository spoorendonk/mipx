#include "mipx/cut_manager.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>

namespace mipx {

void CutManager::setBaseLimits(Int max_rounds, Int max_cuts_per_round) {
    base_max_rounds_ = std::max<Int>(1, max_rounds);
    base_max_cuts_per_round_ = std::max<Int>(1, max_cuts_per_round);
}

void CutManager::setBudgets(double per_node_budget, double per_round_budget,
                            double global_budget) {
    per_node_work_budget_ = std::max(1.0, per_node_budget);
    per_round_work_budget_ = std::max(1.0, per_round_budget);
    global_work_budget_ = std::max(1.0, global_budget);
}

void CutManager::resetNodeState(bool is_root, Index depth) {
    node_is_root_ = is_root;
    node_depth_ = depth;
    node_rounds_used_ = 0;
}

CutRoundPolicy CutManager::beginRound(Int round, bool is_root, Index depth,
                                      double node_cut_work_used,
                                      double global_cut_work_used) const {
    CutRoundPolicy policy;
    policy.family_enabled.fill(false);
    policy.per_family_cap.fill(0);

    if (mode_ == CutEffortMode::Off) return policy;
    if (node_cut_work_used >= per_node_work_budget_) return policy;
    if (global_cut_work_used >= global_work_budget_) return policy;

    Int max_rounds = base_max_rounds_;
    Int max_cuts = base_max_cuts_per_round_;
    switch (mode_) {
        case CutEffortMode::Conservative:
            max_rounds = is_root ? std::min<Int>(8, base_max_rounds_) : 1;
            max_cuts = std::max<Int>(4, base_max_cuts_per_round_ / 4);
            break;
        case CutEffortMode::Aggressive:
            max_rounds = is_root ? std::max<Int>(24, base_max_rounds_)
                                 : std::max<Int>(2, base_max_rounds_ / 3);
            max_cuts = std::max<Int>(16, base_max_cuts_per_round_ * 2);
            break;
        case CutEffortMode::Auto:
            max_rounds = is_root ? std::min<Int>(6, base_max_rounds_)
                                 : std::max<Int>(1, base_max_rounds_ / 4);
            max_cuts = is_root ? std::max<Int>(8, base_max_cuts_per_round_ / 2)
                               : std::max<Int>(4, base_max_cuts_per_round_ / 4);
            break;
        case CutEffortMode::Off:
        default:
            break;
    }

    if (round >= max_rounds) return policy;

    policy.run = true;
    policy.max_cuts_per_round = max_cuts;

    Int enabled_families = 0;
    for (std::size_t fi = 0; fi < family_kpis_.size(); ++fi) {
        const CutFamily family = static_cast<CutFamily>(fi);
        if (family == CutFamily::Unknown || family == CutFamily::Count) continue;

        bool enabled = false;
        Int cap = 0;
        switch (mode_) {
            case CutEffortMode::Conservative:
                enabled = conservativeFamilyEnabled(family, is_root, depth);
                cap = std::max<Int>(1, max_cuts / 4);
                break;
            case CutEffortMode::Aggressive:
                enabled = true;
                cap = std::max<Int>(2, max_cuts);
                break;
            case CutEffortMode::Auto: {
                enabled = family_kpis_[fi].enabled;
                if (!is_root) {
                    if (depth > 8 &&
                        family != CutFamily::Gomory &&
                        family != CutFamily::Mir) {
                        enabled = false;
                    }
                    if (depth > 16 && family != CutFamily::Gomory) {
                        enabled = false;
                    }
                }
                if (!enabled && (round % 5 == static_cast<Int>(fi % 5))) {
                    // Periodic probe rounds let throttled families prove ROI.
                    enabled = true;
                    cap = 1;
                } else {
                    cap = family_kpis_[fi].enabled
                        ? std::max<Int>(1, max_cuts / 3)
                        : 1;
                }
                break;
            }
            case CutEffortMode::Off:
            default:
                break;
        }

        policy.family_enabled[fi] = enabled;
        policy.per_family_cap[fi] = enabled ? cap : 0;
        if (enabled) ++enabled_families;
    }

    if (enabled_families == 0) {
        policy.run = false;
    }
    return policy;
}

void CutManager::recordRound(
    const CutSeparationStats& separation_stats,
    const std::array<Int, static_cast<std::size_t>(CutFamily::Count)>&
        selected_by_family,
    Real lp_delta,
    Real orthogonality,
    double separation_seconds,
    double reopt_work,
    bool is_root,
    Index depth) {
    ++node_rounds_used_;
    const Int total_selected = std::accumulate(
        selected_by_family.begin(), selected_by_family.end(), 0);

    for (std::size_t fi = 0; fi < family_kpis_.size(); ++fi) {
        const CutFamily family = static_cast<CutFamily>(fi);
        if (family == CutFamily::Unknown || family == CutFamily::Count) continue;

        auto& kpi = family_kpis_[fi];
        const auto& round = separation_stats.families[fi];
        const Int selected = selected_by_family[fi];
        const Int rejected = std::max<Int>(0, round.generated - selected);

        kpi.attempted += round.attempted;
        kpi.generated += round.generated;
        kpi.accepted += selected;
        kpi.rejected += rejected;
        kpi.separation_seconds += round.time_seconds;

        const Real avg_eff = selected > 0
            ? round.efficacy_sum / static_cast<Real>(selected)
            : 0.0;
        kpi.efficacy_ema = kEmaAlpha * avg_eff +
                           (1.0 - kEmaAlpha) * kpi.efficacy_ema;

        Real fam_delta = 0.0;
        double fam_reopt_work = 0.0;
        if (total_selected > 0 && selected > 0) {
            const Real share = static_cast<Real>(selected) /
                               static_cast<Real>(total_selected);
            fam_delta = lp_delta * share;
            fam_reopt_work = reopt_work * static_cast<double>(share);
        }
        kpi.last_round_lp_delta = fam_delta;
        kpi.reopt_work += fam_reopt_work;

        const double fam_sep_time = separation_seconds +
                                    round.time_seconds;
        const Real roi = fam_delta /
            static_cast<Real>(fam_reopt_work + 1000.0 * fam_sep_time + 1e-9);
        kpi.roi_ema = kEmaAlpha * roi + (1.0 - kEmaAlpha) * kpi.roi_ema;

        if (selected > 0) {
            kpi.orthogonality_ema = kEmaAlpha * orthogonality +
                (1.0 - kEmaAlpha) * kpi.orthogonality_ema;
        }
    }

    if (mode_ == CutEffortMode::Auto) {
        applyAutoAdjustments(is_root, depth);
    }
}

bool CutManager::familyEnabled(CutFamily family) const {
    if (family == CutFamily::Unknown || family == CutFamily::Count) return false;
    return family_kpis_[familyIndex(family)].enabled;
}

void CutManager::setFamilyEnabled(CutFamily family, bool enabled) {
    if (family == CutFamily::Unknown || family == CutFamily::Count) return;
    family_kpis_[familyIndex(family)].enabled = enabled;
}

void CutManager::applyAutoAdjustments(bool is_root, Index depth) {
    for (std::size_t fi = 0; fi < family_kpis_.size(); ++fi) {
        const CutFamily family = static_cast<CutFamily>(fi);
        if (family == CutFamily::Unknown || family == CutFamily::Count) continue;
        if (family == CutFamily::Gomory) continue;  // keep fallback always available

        auto& kpi = family_kpis_[fi];
        const bool deep_tree_penalty =
            !is_root && depth > 8 &&
            family != CutFamily::Mir;

        if (kpi.enabled) {
            if ((kpi.attempted >= 3 && kpi.roi_ema < kDemoteRoi) ||
                deep_tree_penalty) {
                kpi.enabled = false;
                ++kpi.demotions;
            }
        } else {
            if (kpi.roi_ema > kPromoteRoi &&
                (kpi.generated - kpi.rejected) > 0) {
                kpi.enabled = true;
                ++kpi.promotions;
            }
        }
    }

    // Safety valve: at least one family must remain enabled.
    bool any_enabled = false;
    for (std::size_t fi = 1; fi + 1 < family_kpis_.size(); ++fi) {
        any_enabled = any_enabled || family_kpis_[fi].enabled;
    }
    if (!any_enabled) {
        family_kpis_[familyIndex(CutFamily::Gomory)].enabled = true;
    }
}

bool CutManager::conservativeFamilyEnabled(CutFamily family,
                                           bool is_root,
                                           Index depth) const {
    if (family == CutFamily::Gomory || family == CutFamily::Mir) return true;
    if (is_root && family == CutFamily::Cover) return true;
    if (!is_root && depth <= 2 && family == CutFamily::Cover) return true;
    return false;
}

std::string CutManager::summarizeState() const {
    std::ostringstream oss;
    oss << "mode=";
    switch (mode_) {
        case CutEffortMode::Off: oss << "off"; break;
        case CutEffortMode::Conservative: oss << "conservative"; break;
        case CutEffortMode::Aggressive: oss << "aggressive"; break;
        case CutEffortMode::Auto: oss << "auto"; break;
    }
    for (std::size_t fi = 1; fi + 1 < family_kpis_.size(); ++fi) {
        const auto family = static_cast<CutFamily>(fi);
        const auto& kpi = family_kpis_[fi];
        oss << " " << cutFamilyName(family)
            << "=" << (kpi.enabled ? "on" : "off")
            << "(roi=" << kpi.roi_ema << ",acc=" << kpi.accepted << ")";
    }
    return oss.str();
}

}  // namespace mipx
