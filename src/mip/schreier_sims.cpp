#include "mipx/schreier_sims.h"

#include <algorithm>
#include <cassert>
#include <numeric>
#include <queue>
#include <unordered_set>

namespace mipx {

// ---------------------------------------------------------------------------
// Level construction: orbit + transversal via BFS
// ---------------------------------------------------------------------------

SchreierSims::Level SchreierSims::computeOrbitAndTransversal(
    Index point, const std::vector<Permutation>& generators, Index n) {
    Level level;
    level.base_point = point;

    // BFS to compute orbit and transversal.
    Permutation identity(static_cast<std::size_t>(n));
    std::iota(identity.begin(), identity.end(), 0);

    level.transversal[point] = identity;
    level.orbit.push_back(point);

    std::queue<Index> queue;
    queue.push(point);

    while (!queue.empty()) {
        Index current = queue.front();
        queue.pop();

        for (const auto& gen : generators) {
            Index image = gen[current];
            if (level.transversal.find(image) == level.transversal.end()) {
                // new orbit element: transversal[image] = gen * transversal[current]
                level.transversal[image] = composePermutations(
                    level.transversal[current], gen);
                level.orbit.push_back(image);
                queue.push(image);
            }
        }
    }

    std::sort(level.orbit.begin(), level.orbit.end());
    return level;
}

// ---------------------------------------------------------------------------
// Sift: push a permutation through the BSGS levels
// ---------------------------------------------------------------------------

Permutation SchreierSims::sift(const Permutation& perm) const {
    Permutation g = perm;
    for (const auto& level : levels_) {
        Index beta = g[level.base_point];
        auto it = level.transversal.find(beta);
        if (it == level.transversal.end()) {
            return g;  // not in group at this level
        }
        // g = g * inverse(transversal[beta])
        Permutation inv = inversePermutation(it->second);
        g = composePermutations(g, inv);
    }
    return g;
}

// ---------------------------------------------------------------------------
// Build BSGS
// ---------------------------------------------------------------------------

void SchreierSims::build(const std::vector<Permutation>& generators, Index n,
                          Index num_tracked_points) {
    n_ = n;
    num_tracked_ = (num_tracked_points < 0) ? n : num_tracked_points;
    levels_.clear();
    base_.clear();
    strong_generators_ = generators;
    built_ = false;
    work_units_ = 0.0;

    if (generators.empty() || n == 0) {
        built_ = true;
        return;
    }

    // Choose initial base: points moved by generators, preferring variable vertices.
    std::vector<bool> moved(static_cast<std::size_t>(n), false);
    for (const auto& gen : generators) {
        for (Index i = 0; i < n && i < static_cast<Index>(gen.size()); ++i) {
            if (gen[i] != i) moved[i] = true;
        }
    }

    // First add moved variable vertices, then others.
    for (Index i = 0; i < num_tracked_; ++i) {
        if (moved[i]) base_.push_back(i);
    }
    for (Index i = num_tracked_; i < n; ++i) {
        if (moved[i]) base_.push_back(i);
    }

    // Build levels using Schreier-Sims algorithm.
    // Level i uses generators that fix base points 0..i-1.
    std::vector<Permutation> current_gens = generators;

    for (Index base_idx = 0; base_idx < static_cast<Index>(base_.size()); ++base_idx) {
        if (current_gens.empty()) break;

        Index bp = base_[base_idx];
        Level level = computeOrbitAndTransversal(bp, current_gens, n);
        work_units_ += static_cast<double>(level.orbit.size() * current_gens.size());

        // Compute Schreier generators for the stabilizer.
        // For each orbit element o and generator g:
        //   schreier_gen = transversal[g(o)]^{-1} * g * transversal[o]
        std::vector<Permutation> stab_gens;
        for (Index o : level.orbit) {
            for (const auto& gen : current_gens) {
                Index image = gen[o];
                auto it = level.transversal.find(image);
                if (it == level.transversal.end()) continue;

                Permutation schreier = composePermutations(
                    level.transversal[o], gen);
                Permutation inv_t = inversePermutation(it->second);
                schreier = composePermutations(schreier, inv_t);

                work_units_ += static_cast<double>(n);

                if (!isIdentity(schreier)) {
                    // Check if this is genuinely new (not already in stab_gens).
                    // Simple duplicate check: test if it's already present.
                    bool duplicate = false;
                    for (const auto& existing : stab_gens) {
                        if (existing == schreier) { duplicate = true; break; }
                    }
                    if (!duplicate) {
                        stab_gens.push_back(std::move(schreier));
                        // Limit stabilizer generators to avoid explosion.
                        if (stab_gens.size() > 64) break;
                    }
                }
            }
            if (stab_gens.size() > 64) break;
        }

        level.stabilizer_generators = stab_gens;
        levels_.push_back(std::move(level));
        current_gens = std::move(stab_gens);
    }

    // Verification pass: check all strong generators sift to identity.
    // If any doesn't, add it and rebuild affected levels.
    // NOTE: We must collect new generators before adding them, because
    // addGenerator modifies strong_generators_ (invalidating iterators).
    static constexpr int kMaxVerificationRounds = 3;
    for (int round = 0; round < kMaxVerificationRounds; ++round) {
        std::vector<Permutation> new_gens;
        for (std::size_t gi = 0; gi < strong_generators_.size(); ++gi) {
            Permutation residual = sift(strong_generators_[gi]);
            if (!isIdentity(residual)) {
                new_gens.push_back(std::move(residual));
            }
        }
        if (new_gens.empty()) break;
        for (auto& g : new_gens) {
            addGenerator(g);
        }
    }

    built_ = true;
}

void SchreierSims::addGenerator(const Permutation& perm) {
    strong_generators_.push_back(perm);
    // Rebuild all levels with the extended generator set.
    // This is a simplified approach; a production implementation
    // would only rebuild affected levels.
    std::vector<Permutation> current_gens = strong_generators_;
    levels_.clear();

    for (Index base_idx = 0; base_idx < static_cast<Index>(base_.size()); ++base_idx) {
        if (current_gens.empty()) break;

        Index bp = base_[base_idx];
        Level level = computeOrbitAndTransversal(bp, current_gens, n_);
        work_units_ += static_cast<double>(level.orbit.size() * current_gens.size());

        std::vector<Permutation> stab_gens;
        for (Index o : level.orbit) {
            for (const auto& gen : current_gens) {
                Index image = gen[o];
                auto it = level.transversal.find(image);
                if (it == level.transversal.end()) continue;

                Permutation schreier = composePermutations(
                    level.transversal[o], gen);
                Permutation inv_t = inversePermutation(it->second);
                schreier = composePermutations(schreier, inv_t);
                work_units_ += static_cast<double>(n_);

                if (!isIdentity(schreier)) {
                    bool duplicate = false;
                    for (const auto& existing : stab_gens) {
                        if (existing == schreier) { duplicate = true; break; }
                    }
                    if (!duplicate) {
                        stab_gens.push_back(std::move(schreier));
                        if (stab_gens.size() > 64) break;
                    }
                }
            }
            if (stab_gens.size() > 64) break;
        }

        level.stabilizer_generators = stab_gens;
        levels_.push_back(std::move(level));
        current_gens = std::move(stab_gens);
    }
}

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------

bool SchreierSims::isMember(const Permutation& perm) const {
    if (!built_) return false;
    Permutation residual = sift(perm);
    return isIdentity(residual);
}

std::vector<Index> SchreierSims::orbit(Index point) const {
    if (!built_ || levels_.empty()) return {point};

    // Use all strong generators to compute the orbit via BFS.
    std::unordered_set<Index> visited;
    visited.insert(point);
    std::queue<Index> queue;
    queue.push(point);

    while (!queue.empty()) {
        Index current = queue.front();
        queue.pop();
        for (const auto& gen : strong_generators_) {
            if (current >= static_cast<Index>(gen.size())) continue;
            Index image = gen[current];
            if (visited.insert(image).second) {
                queue.push(image);
            }
        }
    }

    std::vector<Index> result(visited.begin(), visited.end());
    std::sort(result.begin(), result.end());
    return result;
}

std::vector<Index> SchreierSims::orbitUnderStabilizer(
    Index point, const std::vector<Index>& fixed_points) const {
    if (!built_) return {point};

    // Collect generators that fix all fixed_points.
    std::vector<const Permutation*> stab_gens;
    for (const auto& gen : strong_generators_) {
        bool fixes_all = true;
        for (Index fp : fixed_points) {
            if (fp >= 0 && fp < static_cast<Index>(gen.size()) && gen[fp] != fp) {
                fixes_all = false;
                break;
            }
        }
        if (fixes_all) {
            stab_gens.push_back(&gen);
        }
    }

    // BFS with stabilizer generators.
    std::unordered_set<Index> visited;
    visited.insert(point);
    std::queue<Index> queue;
    queue.push(point);

    while (!queue.empty()) {
        Index current = queue.front();
        queue.pop();
        for (const auto* gen : stab_gens) {
            if (current >= static_cast<Index>(gen->size())) continue;
            Index image = (*gen)[current];
            if (visited.insert(image).second) {
                queue.push(image);
            }
        }
    }

    std::vector<Index> result(visited.begin(), visited.end());
    std::sort(result.begin(), result.end());
    return result;
}

std::vector<std::vector<Index>> SchreierSims::allOrbits() const {
    if (!built_) return {};
    return computeVariableOrbits(strong_generators_,
                                  (num_tracked_ >= 0) ? num_tracked_ : n_);
}

}  // namespace mipx
