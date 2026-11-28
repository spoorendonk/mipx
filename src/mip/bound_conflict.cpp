#include "mipx/bound_conflict.h"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <map>
#include <set>
#include <unordered_set>
#include <utility>

namespace mipx {
namespace {

using Side = std::pair<Index, bool>;
using Frontier = std::map<Side, BoundReasonId>;

Side side(const BoundEndpoint& endpoint) {
    return {endpoint.col, endpoint.is_upper};
}

bool entails(Real bound, bool is_upper, Real condition) {
    return is_upper ? bound <= condition : bound >= condition;
}

BranchDecision literal(const BoundReason& reason) {
    const auto [col, is_upper, value] = reason.endpoint;
    return {col, value, is_upper, reason.level};
}

// Resolving pivot against its reason subsumes the conflict when the other
// conditions imply every premise. Follow reasons for premises which are not
// already entailed, so the test also finds redundancy through a chain.
bool subsumed(const BoundReasonTrail& trail, const Frontier& frontier, BoundReasonId pivot) {
    const auto& reason = trail.reason(pivot);
    if (reason.source == BoundReasonSource::Branch) {
        return false;
    }
    std::vector<BoundReasonId> pending;
    for (const auto support : reason.supports.ids()) {
        pending.push_back(support);
    }
    std::unordered_set<BoundReasonId> seen;
    while (!pending.empty()) {
        const auto id = pending.back();
        pending.pop_back();
        const auto& premise = trail.reason(id);
        if (premise.source == BoundReasonSource::Model || !seen.insert(id).second) {
            continue;
        }
        const auto& endpoint = premise.endpoint;
        const auto found = frontier.find(side(endpoint));
        if (found != frontier.end() && found->second != pivot &&
            entails(trail.reason(found->second).endpoint.value, endpoint.is_upper,
                    endpoint.value)) {
            continue;
        }
        if (premise.source == BoundReasonSource::Branch) {
            return false;
        }
        for (const auto support : premise.supports.ids()) {
            pending.push_back(support);
        }
    }
    return true;
}

}  // namespace

std::optional<LearnedBoundConflict> analyzeBoundConflict(const BoundReasonTrail& trail) {
    const auto& contradiction = trail.contradiction();
    // The trail has already combined scope over all ancestors of the failure.
    if (!contradiction || contradiction->scope.node_local ||
        contradiction->source == BoundReasonSource::Opaque) {
        return std::nullopt;
    }
    return analyzeBoundConflict(trail, contradiction->supports, contradiction->scope);
}

std::optional<LearnedBoundConflict> analyzeBoundConflict(const BoundReasonTrail& trail,
                                                         const BoundReasonSupports& supports,
                                                         BoundReasonScope scope) {
    for (const auto id : supports.ids()) {
        const auto& reason = trail.reason(id);
        if (reason.scope.node_local || reason.source == BoundReasonSource::Opaque) {
            return std::nullopt;
        }
        scope.cutoff = std::min(scope.cutoff, reason.scope.cutoff);
    }
    if (scope.node_local) {
        return std::nullopt;
    }
    Frontier frontier;
    std::set<std::pair<Int, BoundReasonId>> by_level;
    const auto add = [&](BoundReasonId id) {
        const auto& reason = trail.reason(id);
        if (reason.source == BoundReasonSource::Model) {
            return;
        }
        const auto& endpoint = reason.endpoint;
        const auto [pos, inserted] = frontier.try_emplace(side(endpoint), id);
        if (!inserted) {
            const auto& old = trail.reason(pos->second);
            if (entails(old.endpoint.value, endpoint.is_upper, endpoint.value)) {
                return;
            }
            by_level.erase({old.level, pos->second});
            pos->second = id;
        }
        by_level.emplace(reason.level, id);
    };
    for (const auto id : supports.ids()) {
        add(id);
    }

    BoundReasonId uip = kNoBoundReason;
    while (!by_level.empty()) {
        const auto latest = std::prev(by_level.end());
        const auto [level, id] = *latest;
        if (level <= 0) {
            break;
        }
        if (latest == by_level.begin() || std::prev(latest)->first != level) {
            uip = id;
            break;
        }
        const auto& reason = trail.reason(id);
        if (reason.source == BoundReasonSource::Branch) {
            // Several independent decisions can share a depth (e.g. SOS).
            return std::nullopt;
        }
        frontier.erase(side(reason.endpoint));
        by_level.erase(latest);
        for (const auto premise : reason.supports.ids()) {
            add(premise);
        }
    }
    if (uip == kNoBoundReason) {
        return std::nullopt;
    }

    LearnedBoundConflict result;
    result.scope = scope;
    // This also discharges deductions from model bounds alone at level zero.
    // A bound imported as a decision at level zero still needs a premise.
    for (auto it = frontier.begin(); it != frontier.end();) {
        const auto id = it->second;
        if (id != uip && subsumed(trail, frontier, id)) {
            result.removed_literals.push_back(literal(trail.reason(id)));
            it = frontier.erase(it);
        } else {
            ++it;
        }
    }
    std::set<Int> levels;
    result.literals.reserve(frontier.size());
    for (const auto& entry : frontier) {
        const auto condition = literal(trail.reason(entry.second));
        if (!std::isfinite(condition.bound)) {
            return std::nullopt;
        }
        result.literals.push_back(condition);
        levels.insert(condition.level);
    }
    result.lbd = static_cast<Int>(levels.size());
    return result;
}

std::optional<BoundConflictInference> evaluateBoundConflict(
    std::span<const BranchDecision> literals, BoundReasonScope scope, std::span<const Real> lower,
    std::span<const Real> upper, std::span<const VarType> types, Real cutoff) {
    if (scope.node_local || !(cutoff <= scope.cutoff)) {
        return std::nullopt;
    }
    std::size_t open = literals.size();
    for (std::size_t i = 0; i < literals.size(); ++i) {
        const auto& literal = literals[i];
        const auto col = literal.variable;
        const auto bound = literal.bound;
        const auto is_upper = literal.is_upper;
        if (!entails(is_upper ? upper[col] : lower[col], is_upper, bound)) {
            if (open != literals.size() || (is_upper ? lower[col] > bound : upper[col] < bound)) {
                return std::nullopt;
            }
            open = i;
        }
    }
    if (open == literals.size()) {
        return BoundConflictInference{std::nullopt, literals, {}, scope};
    }
    const auto& literal = literals[open];
    const auto type = types[literal.variable];
    if (type != VarType::Binary && type != VarType::Integer && type != VarType::SemiInteger) {
        return std::nullopt;
    }
    const Real threshold = literal.is_upper ? std::floor(literal.bound) : std::ceil(literal.bound);
    const Real complement = threshold + (literal.is_upper ? 1.0 : -1.0);
    if (std::abs(complement - threshold) != 1.0 ||
        entails(literal.is_upper ? lower[literal.variable] : upper[literal.variable],
                !literal.is_upper, complement)) {
        return std::nullopt;
    }
    return BoundConflictInference{BoundEndpoint{literal.variable, !literal.is_upper, complement},
                                  literals.first(open), literals.subspan(open + 1), scope};
}

}  // namespace mipx
