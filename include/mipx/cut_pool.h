#pragma once

#include "mipx/core.h"

#include <cmath>
#include <limits>
#include <span>
#include <vector>

namespace mipx {

inline constexpr Real kDefaultCutMinViolation = 1e-5;

enum class CutFamily : Int {
    Unknown = 0,
    Gomory,
    Mir,
    Cover,
    ImpliedBound,
    Clique,
    ZeroHalf,
    Mixing,
    Cmir,
    StrongCg,
    LiftedCover,
    ModK,
    IntersectionCut,
    MultiRow,
    FlowCover,
    Count,
};

[[nodiscard]] const char* cutFamilyName(CutFamily family);

/// A single cut: sparse row with bounds (row_lower <= a^T x <= row_upper).
struct Cut {
    std::vector<Index> indices;
    std::vector<Real> values;
    Real lower = -std::numeric_limits<Real>::infinity();
    Real upper = std::numeric_limits<Real>::infinity();
    CutFamily family = CutFamily::Unknown;
    bool local = false;
    Real efficacy = 0.0;  // violation / ||a||
    Int age = 0;          // rounds since last active (binding)
    Int activity = 0;     // number of rounds where cut was active
    /// Finite when validity requires an incumbent cutoff at least this strong.
    /// Root cutoff feedback currently guides only unconditional separation.
    Real valid_cutoff = std::numeric_limits<Real>::infinity();

    [[nodiscard]] bool requiresCutoff() const {
        return valid_cutoff != std::numeric_limits<Real>::infinity();
    }
    [[nodiscard]] bool isUnconditionalGlobal() const { return !local && !requiresCutoff(); }
};

/// Unscaled violation of either bound at the given LP point.
[[nodiscard]] Real cutViolation(const Cut& cut, std::span<const Real> primals);

/// Screen a cut for numerical safety.
///
/// CutPool::addCut applies it to every cut it is offered, so no cut that fails
/// it can enter a pool, whichever separator built it. Callers may screen
/// earlier as well, to skip work on a doomed cut or to count rejections, but
/// the guarantee does not rest on them. A cut is rejected when it is empty, its index and
/// value arrays disagree in length, a bound is infinite in the direction that
/// bounds it (a lower of +inf or an upper of -inf), its indices are negative
/// or not strictly increasing, any coefficient is non-finite or no larger
/// than 1e-12 in magnitude, its squared norm falls outside [1e-12, 1e16], the
/// largest coefficient magnitude exceeds 1e6, or the largest/smallest
/// coefficient ratio exceeds 1e8.
///
/// Note the bound test does not reject a NaN bound: a one-sided cut carries
/// +/-kInf on its open side, and the predicate distinguishes those from a
/// bound infinite the wrong way by comparing against kInf, which NaN fails on
/// both sides. No current separator can produce one -- Gomory rejects a
/// non-finite rhs before the screen, and the manager families build bounds by
/// flooring finite values -- so this is a gap in the screen, not a live hole.
[[nodiscard]] bool isNumericallySafeCut(const Cut& cut);

/// Pool of cutting planes with efficacy ranking and parallelism filtering.
class CutPool {
public:
    CutPool() = default;

    /// Add a cut to the pool. Returns true if accepted: the cut passes
    /// isNumericallySafeCut, meets the minimum efficacy, and is not too
    /// parallel to a pooled cut of at least its efficacy.
    bool addCut(Cut cut);

    /// Age all cuts by one round. A cut near binding at either point stays active.
    void ageAll(std::span<const Real> primals, Real active_tol = 1e-4,
                std::span<const Real> guide = {});

    /// Remove cuts older than the given threshold.
    void purge(Int age_threshold = 10);
    /// Also preserve the next position in an inactive-cut scan.
    void purge(Int age_threshold, Index* next);

    /// Return indices of the top-k cuts by efficacy.
    [[nodiscard]] std::vector<Index> topByEfficacy(Index k) const;

    /// Rank uninstalled cuts by their best efficacy across the given points.
    [[nodiscard]] std::vector<Index> violatedByEfficacy(std::span<const Real> primals,
                                                        Real min_violation,
                                                        std::span<const Real> guide = {}) const;

    /// Record that these cuts are now rows of the LP.
    void markInstalled(std::span<const Index> indices);

    /// Invalidate entries without shifting indices. purge() removes them.
    void invalidate(std::span<const Index> indices);

    /// Cycle through insertion order for propagation, excluding rows selected
    /// for this batch. next, if supplied, resumes the scan on the next call.
    /// The returned pointers remain valid until the pool is mutated.
    [[nodiscard]] std::vector<const Cut*> inactiveGlobalCuts(Int limit,
                                                             std::span<const Index> selected = {},
                                                             Index* next = nullptr) const;

    /// Number of cuts in the pool.
    [[nodiscard]] Index size() const { return static_cast<Index>(cuts_.size()); }

    /// Access a cut by index.
    [[nodiscard]] const Cut& operator[](Index i) const { return cuts_[i].cut; }

    /// Clear all cuts.
    void clear() { cuts_.clear(); }

    /// Set the parallelism threshold (cosine similarity). Default 0.9.
    void setParallelismThreshold(Real t) { parallelism_threshold_ = t; }

    /// Set the minimum efficacy for a cut to be accepted.
    void setMinEfficacy(Real e) { min_efficacy_ = e; }

private:
    /// Compute the cosine similarity between two sparse vectors.
    [[nodiscard]] static Real cosineSimilarity(std::span<const Index> ind_a,
                                               std::span<const Real> val_a,
                                               std::span<const Index> ind_b,
                                               std::span<const Real> val_b);

    struct Entry {
        Cut cut;
        bool installed = false;
        bool invalidated = false;
    };
    std::vector<Entry> cuts_;
    Real parallelism_threshold_ = 0.9;
    Real min_efficacy_ = 1e-5;
};

}  // namespace mipx
