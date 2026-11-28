#include "mipx/cut_pool.h"

#include "mipx/lp_problem.h"  // kInf

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>

namespace mipx {

const char* cutFamilyName(CutFamily family) {
    switch (family) {
        case CutFamily::Gomory:
            return "gomory";
        case CutFamily::Mir:
            return "mir";
        case CutFamily::Cover:
            return "cover";
        case CutFamily::ImpliedBound:
            return "implbd";
        case CutFamily::Clique:
            return "clique";
        case CutFamily::ZeroHalf:
            return "zerohalf";
        case CutFamily::Mixing:
            return "mixing";
        case CutFamily::Cmir:
            return "cmir";
        case CutFamily::StrongCg:
            return "strongcg";
        case CutFamily::LiftedCover:
            return "liftcov";
        case CutFamily::ModK:
            return "modk";
        case CutFamily::IntersectionCut:
            return "intersect";
        case CutFamily::MultiRow:
            return "multirow";
        case CutFamily::FlowCover:
            return "flowcover";
        case CutFamily::Unknown:
        case CutFamily::Count:
        default:
            return "unknown";
    }
}

namespace {

bool hasFiniteBounds(const Cut& cut) {
    if (!std::isfinite(cut.lower) && cut.lower > -kInf) {
        return false;
    }
    if (!std::isfinite(cut.upper) && cut.upper < kInf) {
        return false;
    }
    return true;
}

Real cutLhs(const Cut& cut, std::span<const Real> primals) {
    Real lhs = 0.0;
    for (std::size_t k = 0; k < cut.indices.size(); ++k) {
        const Index j = cut.indices[k];
        if (j >= 0 && j < static_cast<Index>(primals.size())) {
            lhs += cut.values[k] * primals[j];
        }
    }
    return lhs;
}

}  // namespace

Real cutViolation(const Cut& cut, std::span<const Real> primals) {
    const Real lhs = cutLhs(cut, primals);
    Real violation = 0.0;
    if (cut.lower > -kInf) {
        violation = std::max(violation, cut.lower - lhs);
    }
    if (cut.upper < kInf) {
        violation = std::max(violation, lhs - cut.upper);
    }
    return violation;
}

bool isNumericallySafeCut(const Cut& cut) {
    if (cut.indices.empty()) {
        return false;
    }
    if (cut.indices.size() != cut.values.size()) {
        return false;
    }
    if (!hasFiniteBounds(cut)) {
        return false;
    }

    Real norm_sq = 0.0;
    Real max_abs = 0.0;
    Real min_abs = std::numeric_limits<Real>::infinity();
    Index prev = -1;

    for (Index k = 0; k < static_cast<Index>(cut.indices.size()); ++k) {
        const Index idx = cut.indices[k];
        const Real val = cut.values[k];
        if (idx < 0 || idx <= prev) {
            return false;
        }
        if (!std::isfinite(val)) {
            return false;
        }
        const Real abs_v = std::abs(val);
        if (abs_v <= 1e-12) {
            return false;
        }
        norm_sq += val * val;
        max_abs = std::max(max_abs, abs_v);
        min_abs = std::min(min_abs, abs_v);
        prev = idx;
    }

    if (norm_sq < 1e-12 || norm_sq > 1e16) {
        return false;
    }
    if (max_abs > 1e6) {
        return false;
    }
    if (min_abs < std::numeric_limits<Real>::infinity() && max_abs / min_abs > 1e8) {
        return false;
    }
    return true;
}

Real CutPool::cosineSimilarity(std::span<const Index> ind_a, std::span<const Real> val_a,
                               std::span<const Index> ind_b, std::span<const Real> val_b) {
    // Compute dot product and norms using merge of two sorted sparse vectors.
    Real dot = 0.0;
    Real norm_a = 0.0;
    Real norm_b = 0.0;

    for (auto v : val_a) {
        norm_a += v * v;
    }
    for (auto v : val_b) {
        norm_b += v * v;
    }

    if (norm_a < 1e-30 || norm_b < 1e-30) {
        return 0.0;
    }

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
    // The pool screens every cut itself, so no path into it can bypass
    // isNumericallySafeCut (issue #212).
    if (!isNumericallySafeCut(cut)) {
        return false;
    }

    // Check minimum efficacy.
    if (cut.efficacy < min_efficacy_) {
        return false;
    }

    // Check parallelism against existing cuts.
    for (const auto& entry : cuts_) {
        if (entry.invalidated || entry.cut.valid_cutoff != cut.valid_cutoff) {
            continue;
        }
        const auto& existing = entry.cut;
        Real sim = cosineSimilarity(cut.indices, cut.values, existing.indices, existing.values);
        if (sim > parallelism_threshold_) {
            // Too parallel — only keep if strictly better efficacy.
            if (cut.efficacy <= existing.efficacy) {
                return false;
            }
        }
    }

    cuts_.push_back({std::move(cut)});
    return true;
}

void CutPool::ageAll(std::span<const Real> primals, Real active_tol, std::span<const Real> guide) {
    for (auto& entry : cuts_) {
        if (entry.invalidated) {
            continue;
        }
        auto& cut = entry.cut;
        auto nearBinding = [&](std::span<const Real> point) {
            const Real activity = cutLhs(cut, point);
            return (cut.upper < kInf && std::abs(activity - cut.upper) <= active_tol) ||
                   (cut.lower > -kInf && std::abs(activity - cut.lower) <= active_tol);
        };
        if (nearBinding(primals) || (!guide.empty() && nearBinding(guide))) {
            cut.age = 0;
        } else {
            ++cut.age;
        }
    }
}

void CutPool::purge(Int age_threshold) {
    purge(age_threshold, nullptr);
}

void CutPool::purge(Int age_threshold, Index* next) {
    const auto expired = [age_threshold](const Entry& entry) {
        return entry.invalidated || entry.cut.age > age_threshold;
    };
    if (next != nullptr) {
        const Index before = std::clamp(*next, Index{0}, size());
        *next = before -
                static_cast<Index>(std::count_if(cuts_.begin(), cuts_.begin() + before, expired));
    }
    std::erase_if(cuts_, expired);
}

std::vector<Index> CutPool::topByEfficacy(Index k) const {
    std::vector<Index> indices;
    indices.reserve(cuts_.size());
    for (Index i = 0; i < size(); ++i) {
        if (!cuts_[i].invalidated && !cuts_[i].cut.requiresCutoff()) {
            indices.push_back(i);
        }
    }

    // Partial sort to get top-k.
    const Index n = std::clamp(k, Index{0}, static_cast<Index>(indices.size()));
    std::partial_sort(
        indices.begin(), indices.begin() + n, indices.end(),
        [this](Index a, Index b) { return cuts_[a].cut.efficacy > cuts_[b].cut.efficacy; });

    indices.resize(static_cast<std::size_t>(n));
    return indices;
}

std::vector<Index> CutPool::uninstalledByEfficacy() const {
    std::vector<Index> indices;
    for (Index i = 0; i < size(); ++i) {
        const auto& [cut, installed, invalidated] = cuts_[i];
        if (!installed && !invalidated && cut.isUnconditionalGlobal() &&
            std::isfinite(cut.efficacy)) {
            indices.push_back(i);
        }
    }
    std::ranges::sort(indices, [this](Index a, Index b) {
        const Real a_efficacy = cuts_[a].cut.efficacy;
        const Real b_efficacy = cuts_[b].cut.efficacy;
        return a_efficacy == b_efficacy ? a < b : a_efficacy > b_efficacy;
    });
    return indices;
}

std::vector<Index> CutPool::violatedByEfficacy(std::span<const Real> primals, Real min_violation,
                                               std::span<const Real> guide) const {
    std::vector<std::pair<Index, Real>> ranked;
    for (Index i = 0; i < size(); ++i) {
        const auto& [cut, installed, invalidated] = cuts_[i];
        if (installed || invalidated || cut.requiresCutoff()) {
            continue;
        }
        const Real violation = guide.empty()
                                   ? cutViolation(cut, primals)
                                   : std::max(cutViolation(cut, primals), cutViolation(cut, guide));
        if (violation < min_violation) {
            continue;
        }
        const Real norm_sq =
            std::inner_product(cut.values.begin(), cut.values.end(), cut.values.begin(), 0.0);
        const Real efficacy = violation / std::sqrt(norm_sq);
        if (std::isfinite(efficacy) && efficacy >= min_efficacy_) {
            ranked.emplace_back(i, efficacy);
        }
    }
    std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
        if (a.second != b.second) {
            return a.second > b.second;
        }
        return a.first < b.first;
    });
    std::vector<Index> indices;
    indices.reserve(ranked.size());
    for (const auto& [index, efficacy] : ranked) {
        indices.push_back(index);
    }
    return indices;
}

void CutPool::markInstalled(std::span<const Index> indices) {
    for (const Index index : indices) {
        cuts_[index].installed = true;
    }
}

void CutPool::invalidate(std::span<const Index> indices) {
    for (const Index index : indices) {
        cuts_[index].invalidated = true;
    }
}

std::vector<const Cut*> CutPool::inactiveGlobalCuts(Int limit, std::span<const Index> selected,
                                                    Index* next) const {
    std::vector<const Cut*> candidates;
    const Index count = size();
    Index i = next != nullptr && *next >= 0 && *next < count ? *next : 0;
    for (Index scanned = 0; scanned < count && static_cast<Int>(candidates.size()) < limit;
         ++scanned) {
        const auto& [cut, installed, invalidated] = cuts_[i];
        if (!installed && !invalidated && cut.isUnconditionalGlobal() &&
            std::ranges::find(selected, i) == selected.end()) {
            candidates.push_back(&cut);
        }
        i = i + 1 == count ? 0 : i + 1;
    }
    if (next != nullptr) {
        *next = i;
    }
    return candidates;
}

}  // namespace mipx
