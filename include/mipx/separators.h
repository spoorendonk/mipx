#pragma once

#include <array>
#include <span>

#include "mipx/core.h"
#include "mipx/cut_pool.h"
#include "mipx/dual_simplex.h"
#include "mipx/gomory.h"
#include "mipx/lp_problem.h"

namespace mipx {

struct CutFamilyStats {
    Int attempted = 0;
    Int generated = 0;
    Int accepted = 0;
    Real efficacy_sum = 0.0;
    Real lp_delta = 0.0;
    double time_seconds = 0.0;
};

struct CutSeparationStats {
    std::array<CutFamilyStats, static_cast<std::size_t>(CutFamily::Count)> families{};

    [[nodiscard]] CutFamilyStats& at(CutFamily family) {
        return families[static_cast<std::size_t>(family)];
    }
    [[nodiscard]] const CutFamilyStats& at(CutFamily family) const {
        return families[static_cast<std::size_t>(family)];
    }
};

struct CutFamilyConfig {
    bool gomory = true;
    bool mir = true;
    bool cover = true;
    bool implied_bound = true;
    bool clique = true;
    bool zero_half = true;
    bool mixing = true;
};

class SeparatorManager {
public:
    void setConfig(const CutFamilyConfig& config) { config_ = config; }
    [[nodiscard]] const CutFamilyConfig& config() const { return config_; }
    void setMaxCutsPerFamily(Int value) { max_cuts_per_family_ = std::max<Int>(1, value); }
    void setMinViolation(Real value) { min_violation_ = std::max<Real>(1e-8, value); }

    Int separate(DualSimplexSolver& lp,
                 const LpProblem& problem,
                 std::span<const Real> primals,
                 CutPool& pool,
                 CutSeparationStats& stats);

private:
    Int separateGomory(DualSimplexSolver& lp,
                       const LpProblem& problem,
                       std::span<const Real> primals,
                       CutPool& pool,
                       CutFamilyStats& stats);
    Int separateMir(const LpProblem& problem,
                    std::span<const Real> primals,
                    CutPool& pool,
                    CutFamilyStats& stats);
    Int separateCover(const LpProblem& problem,
                      std::span<const Real> primals,
                      CutPool& pool,
                      CutFamilyStats& stats);
    Int separateImpliedBound(const LpProblem& problem,
                             std::span<const Real> primals,
                             CutPool& pool,
                             CutFamilyStats& stats);
    Int separateClique(const LpProblem& problem,
                       std::span<const Real> primals,
                       CutPool& pool,
                       CutFamilyStats& stats);
    Int separateZeroHalf(const LpProblem& problem,
                         std::span<const Real> primals,
                         CutPool& pool,
                         CutFamilyStats& stats);
    Int separateMixing(const LpProblem& problem,
                       std::span<const Real> primals,
                       CutPool& pool,
                       CutFamilyStats& stats);

    [[nodiscard]] bool isEnabled(CutFamily family) const;

    CutFamilyConfig config_{};
    GomorySeparator gomory_{};
    Int max_cuts_per_family_ = 50;
    Real min_violation_ = 1e-5;
};

}  // namespace mipx
