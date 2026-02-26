#pragma once

#include <cmath>
#include <limits>
#include <span>
#include <vector>

#include "mipx/core.h"

namespace mipx {

enum class CutFamily : Int {
    Unknown = 0,
    Gomory,
    Mir,
    Cover,
    ImpliedBound,
    Clique,
    ZeroHalf,
    Mixing,
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
    Real efficacy = 0.0;   // violation / ||a||
    Int age = 0;           // rounds since last active (binding)
    Int activity = 0;      // number of rounds where cut was active
};

/// Pool of cutting planes with efficacy ranking and parallelism filtering.
class CutPool {
public:
    CutPool() = default;

    /// Add a cut to the pool. Returns true if accepted (not too parallel).
    bool addCut(Cut cut);

    /// Age all cuts by one round. Mark active cuts (those within tol of binding).
    void ageAll(std::span<const Real> primals, Real active_tol = 1e-4);

    /// Remove cuts older than the given threshold.
    void purge(Int age_threshold = 10);

    /// Return indices of the top-k cuts by efficacy.
    [[nodiscard]] std::vector<Index> topByEfficacy(Index k) const;

    /// Number of cuts in the pool.
    [[nodiscard]] Index size() const { return static_cast<Index>(cuts_.size()); }

    /// Access a cut by index.
    [[nodiscard]] const Cut& operator[](Index i) const { return cuts_[i]; }

    /// Clear all cuts.
    void clear() { cuts_.clear(); }

    /// Set the parallelism threshold (cosine similarity). Default 0.9.
    void setParallelismThreshold(Real t) { parallelism_threshold_ = t; }

    /// Set the minimum efficacy for a cut to be accepted.
    void setMinEfficacy(Real e) { min_efficacy_ = e; }

private:
    /// Compute the cosine similarity between two sparse vectors.
    [[nodiscard]] static Real cosineSimilarity(
        std::span<const Index> ind_a, std::span<const Real> val_a,
        std::span<const Index> ind_b, std::span<const Real> val_b);

    std::vector<Cut> cuts_;
    Real parallelism_threshold_ = 0.9;
    Real min_efficacy_ = 1e-5;
};

}  // namespace mipx
