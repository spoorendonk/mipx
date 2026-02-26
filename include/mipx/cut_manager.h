#pragma once

#include <array>
#include <string>

#include "mipx/core.h"
#include "mipx/cut_pool.h"
#include "mipx/separators.h"

namespace mipx {

enum class CutEffortMode {
    Off,
    Conservative,
    Aggressive,
    Auto,
};

struct CutFamilyKpi {
    Int attempted = 0;
    Int generated = 0;
    Int accepted = 0;
    Int rejected = 0;
    Real efficacy_ema = 0.0;
    Real orthogonality_ema = 1.0;
    Real roi_ema = 0.0;
    Real last_round_lp_delta = 0.0;
    double separation_seconds = 0.0;
    double reopt_work = 0.0;
    Int demotions = 0;
    Int promotions = 0;
    bool enabled = true;
};

struct CutRoundPolicy {
    bool run = false;
    Int max_cuts_per_round = 0;
    std::array<bool, static_cast<std::size_t>(CutFamily::Count)> family_enabled{};
    std::array<Int, static_cast<std::size_t>(CutFamily::Count)> per_family_cap{};
};

class CutManager {
public:
    void setMode(CutEffortMode mode) { mode_ = mode; }
    [[nodiscard]] CutEffortMode mode() const { return mode_; }

    void setBaseLimits(Int max_rounds, Int max_cuts_per_round);
    void setBudgets(double per_node_budget, double per_round_budget,
                    double global_budget);

    void resetNodeState(bool is_root, Index depth);
    [[nodiscard]] CutRoundPolicy beginRound(Int round,
                                            bool is_root,
                                            Index depth,
                                            double node_cut_work_used,
                                            double global_cut_work_used) const;

    void recordRound(const CutSeparationStats& separation_stats,
                     const std::array<Int, static_cast<std::size_t>(CutFamily::Count)>&
                         selected_by_family,
                     Real lp_delta,
                     Real orthogonality,
                     double separation_seconds,
                     double reopt_work,
                     bool is_root,
                     Index depth);

    [[nodiscard]] bool familyEnabled(CutFamily family) const;
    void setFamilyEnabled(CutFamily family, bool enabled);
    [[nodiscard]] const std::array<CutFamilyKpi,
                                   static_cast<std::size_t>(CutFamily::Count)>&
    kpis() const {
        return family_kpis_;
    }
    [[nodiscard]] std::string summarizeState() const;

private:
    [[nodiscard]] static std::size_t familyIndex(CutFamily family) {
        return static_cast<std::size_t>(family);
    }
    void applyAutoAdjustments(bool is_root, Index depth);
    [[nodiscard]] bool conservativeFamilyEnabled(CutFamily family,
                                                 bool is_root,
                                                 Index depth) const;

    CutEffortMode mode_ = CutEffortMode::Auto;
    Int base_max_rounds_ = 20;
    Int base_max_cuts_per_round_ = 50;
    double per_node_work_budget_ = 2.5e5;
    double per_round_work_budget_ = 5.0e4;
    double global_work_budget_ = 1.0e6;
    Int node_rounds_used_ = 0;
    bool node_is_root_ = true;
    Index node_depth_ = 0;

    std::array<CutFamilyKpi, static_cast<std::size_t>(CutFamily::Count)> family_kpis_{};
    static constexpr Real kEmaAlpha = 0.35;
    static constexpr Real kDemoteRoi = 1e-7;
    static constexpr Real kPromoteRoi = 2e-6;
};

}  // namespace mipx
