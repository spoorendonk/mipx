#pragma once

#include <cstddef>
#include <vector>

#include "mipx/lp_problem.h"

namespace mipx {

struct OrbitalFix {
    Index variable = -1;
    Index canonical = -1;
};

class SymmetryManager {
public:
    SymmetryManager() = default;

    void detect(const LpProblem& problem);
    [[nodiscard]] bool hasSymmetry() const;
    [[nodiscard]] Index canonical(Index var) const;
    [[nodiscard]] bool isCanonical(Index var) const;
    [[nodiscard]] const std::vector<std::vector<Index>>& orbits() const;
    [[nodiscard]] const std::vector<OrbitalFix>& orbitalFixes() const;
    [[nodiscard]] double detectWorkUnits() const;
    [[nodiscard]] double cutWorkUnits() const;

    [[nodiscard]] Index addSymmetryCuts(LpProblem& problem);

private:
    std::vector<std::vector<Index>> orbits_;
    std::vector<Index> canonical_;
    std::vector<OrbitalFix> orbital_fixes_;
    double detect_work_units_ = 0.0;
    double cut_work_units_ = 0.0;
};

}  // namespace mipx
