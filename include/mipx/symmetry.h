#pragma once

#include <cstddef>
#include <vector>

#include "mipx/automorphism.h"
#include "mipx/lp_problem.h"
#include "mipx/orbital_fixing.h"
#include "mipx/schreier_sims.h"
#include "mipx/symbreak.h"

namespace mipx {

struct OrbitalFix {
    Index variable = -1;
    Index canonical = -1;
};

class SymmetryManager {
public:
    SymmetryManager() = default;

    /// Detect symmetries using column-signature hashing (fast, approximate).
    void detect(const LpProblem& problem);

    /// Detect symmetries using graph automorphism (slower, exact).
    /// Falls back to column-signature if the incidence graph is too large.
    void detectFull(const LpProblem& problem);

    [[nodiscard]] bool hasSymmetry() const;
    [[nodiscard]] Index canonical(Index var) const;
    [[nodiscard]] bool isCanonical(Index var) const;
    [[nodiscard]] const std::vector<std::vector<Index>>& orbits() const;
    [[nodiscard]] const std::vector<OrbitalFix>& orbitalFixes() const;
    [[nodiscard]] double detectWorkUnits() const;
    [[nodiscard]] double cutWorkUnits() const;

    /// Add symmetry-breaking cuts (legacy: lex-ordering from orbital fixes).
    [[nodiscard]] Index addSymmetryCuts(LpProblem& problem);

    /// Add advanced symmetry-breaking constraints (symresack, orbitope, lex-leader).
    [[nodiscard]] Index addSymbreakConstraints(LpProblem& problem);

    /// Aggregate symmetric constraints in presolve.
    [[nodiscard]] Index aggregateConstraints(LpProblem& problem);

    /// Get the Schreier-Sims structure for orbit computation at nodes.
    [[nodiscard]] const SchreierSims& schreierSims() const { return schreier_sims_; }

    /// Get the automorphism generators.
    [[nodiscard]] const std::vector<Permutation>& generators() const {
        return automorphism_result_.generators;
    }

    /// Apply orbital fixing at a tree node.
    /// @param col_lower    Current lower bounds (may be modified).
    /// @param col_upper    Current upper bounds (may be modified).
    /// @param col_type     Variable types.
    /// @param fixed_vars   Variables fixed at this node.
    /// @param num_cols     Number of columns.
    /// @return Orbital fixing result with tightenings.
    [[nodiscard]] OrbitalFixingResult applyOrbitalFixing(
        std::vector<Real>& col_lower,
        std::vector<Real>& col_upper,
        const std::vector<VarType>& col_type,
        const std::vector<Index>& fixed_vars,
        Index num_cols) const;

    /// Check if a node can be pruned by isomorphism.
    [[nodiscard]] bool canPruneByIsomorphism(
        const std::vector<Real>& col_lower,
        const std::vector<Real>& col_upper) const;

    /// Record a node configuration as explored (for isomorphism pruning).
    void recordExplored(const std::vector<Real>& col_lower,
                        const std::vector<Real>& col_upper);

    /// Get the isomorphism pruner statistics.
    [[nodiscard]] Index numIsomorphismPrunes() const;

    /// Whether full automorphism detection was used (vs column-signature).
    [[nodiscard]] bool usedFullDetection() const { return used_full_detection_; }

    /// Get the number of automorphism generators found.
    [[nodiscard]] Index numGenerators() const {
        return static_cast<Index>(automorphism_result_.generators.size());
    }

private:
    /// Column-signature detection (original fast path).
    void detectByColumnSignature(const LpProblem& problem);

    /// Graph automorphism detection (new, slower, more complete).
    void detectByAutomorphism(const LpProblem& problem);

    /// Build canonical mapping and orbital fixes from orbits.
    void buildCanonicalMapping();

    std::vector<std::vector<Index>> orbits_;
    std::vector<Index> canonical_;
    std::vector<OrbitalFix> orbital_fixes_;
    AutomorphismResult automorphism_result_;
    SchreierSims schreier_sims_;
    OrbitalFixer orbital_fixer_;
    mutable IsomorphismPruner isomorphism_pruner_;
    SymbreakGenerator symbreak_generator_;
    double detect_work_units_ = 0.0;
    double cut_work_units_ = 0.0;
    bool used_full_detection_ = false;

    /// Max incidence graph size for full automorphism detection.
    static constexpr Index kMaxGraphVertices = 50000;
};

}  // namespace mipx
