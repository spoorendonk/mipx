#include "mipx/cut_pool.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace mipx {

const char* cutFamilyName(CutFamily family) {
    switch (family) {
        case CutFamily::Gomory: return "gomory";
        case CutFamily::Mir: return "mir";
        case CutFamily::Cover: return "cover";
        case CutFamily::ImpliedBound: return "implbd";
        case CutFamily::Clique: return "clique";
        case CutFamily::ZeroHalf: return "zerohalf";
        case CutFamily::Mixing: return "mixing";
        case CutFamily::Unknown:
        case CutFamily::Count:
        default: return "unknown";
    }
}

Real CutPool::cosineSimilarity(
    std::span<const Index> ind_a, std::span<const Real> val_a,
    std::span<const Index> ind_b, std::span<const Real> val_b) {
    // Compute dot product and norms using merge of two sorted sparse vectors.
    Real dot = 0.0;
    Real norm_a = 0.0;
    Real norm_b = 0.0;

    for (auto v : val_a) norm_a += v * v;
    for (auto v : val_b) norm_b += v * v;

    if (norm_a < 1e-30 || norm_b < 1e-30) return 0.0;

    Index ia = 0, ib = 0;
    Index na = static_cast<Index>(ind_a.size());
    Index nb = static_cast<Index>(ind_b.size());

    while (ia < na && ib < nb) {
        if (ind_a[ia] == ind_b[ib]) {
            dot += val_a[ia] * val_b[ib];
            ++ia;
            ++ib;
        } else if (ind_a[ia] < ind_b[ib]) {
            ++ia;
        } else {
            ++ib;
        }
    }

    return std::abs(dot) / (std::sqrt(norm_a) * std::sqrt(norm_b));
}

bool CutPool::addCut(Cut cut) {
    // Check minimum efficacy.
    if (cut.efficacy < min_efficacy_) return false;

    // Check parallelism against existing cuts.
    for (const auto& existing : cuts_) {
        Real sim = cosineSimilarity(
            cut.indices, cut.values,
            existing.indices, existing.values);
        if (sim > parallelism_threshold_) {
            // Too parallel — only keep if strictly better efficacy.
            if (cut.efficacy <= existing.efficacy) return false;
        }
    }

    cuts_.push_back(std::move(cut));
    return true;
}

void CutPool::ageAll(std::span<const Real> primals, Real active_tol) {
    for (auto& cut : cuts_) {
        // Compute activity: a^T x.
        Real activity = 0.0;
        for (Index k = 0; k < static_cast<Index>(cut.indices.size()); ++k) {
            Index j = cut.indices[k];
            if (j < static_cast<Index>(primals.size())) {
                activity += cut.values[k] * primals[j];
            }
        }

        // Check if the cut is active (binding).
        bool active = false;
        if (cut.upper < std::numeric_limits<Real>::infinity()) {
            if (std::abs(activity - cut.upper) <= active_tol) active = true;
        }
        if (cut.lower > -std::numeric_limits<Real>::infinity()) {
            if (std::abs(activity - cut.lower) <= active_tol) active = true;
        }

        if (active) {
            cut.age = 0;
        } else {
            ++cut.age;
        }
    }
}

void CutPool::purge(Int age_threshold) {
    std::erase_if(cuts_, [age_threshold](const Cut& c) {
        return c.age > age_threshold;
    });
}

std::vector<Index> CutPool::topByEfficacy(Index k) const {
    std::vector<Index> indices(static_cast<std::size_t>(size()));
    std::iota(indices.begin(), indices.end(), 0);

    // Partial sort to get top-k.
    Index n = std::min(k, size());
    std::partial_sort(indices.begin(), indices.begin() + n, indices.end(),
        [this](Index a, Index b) {
            return cuts_[a].efficacy > cuts_[b].efficacy;
        });

    indices.resize(static_cast<std::size_t>(n));
    return indices;
}

}  // namespace mipx
