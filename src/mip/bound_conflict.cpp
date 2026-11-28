#include "mipx/bound_conflict.h"

#include "mipx/domain.h"

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

struct ConflictFrontier {
    const BoundReasonTrail& trail;
    Frontier conditions;
    std::set<std::pair<Int, BoundReasonId>> by_level;

    void add(BoundReasonId id) {
        const auto& reason = trail.reason(id);
        if (reason.source == BoundReasonSource::Model) {
            return;
        }
        const auto& endpoint = reason.endpoint;
        const auto [pos, inserted] = conditions.try_emplace(side(endpoint), id);
        if (!inserted) {
            const auto& old = trail.reason(pos->second);
            if (entails(old.endpoint.value, endpoint.is_upper, endpoint.value)) {
                return;
            }
            by_level.erase({old.level, pos->second});
            pos->second = id;
        }
        by_level.emplace(reason.level, id);
    }

    void addSupports(const BoundReasonSupports& supports) {
        for (const auto id : supports.ids()) {
            add(id);
        }
    }

    bool resolveLatest() {
        const auto latest = std::prev(by_level.end());
        const auto& reason = trail.reason(latest->second);
        if (reason.source == BoundReasonSource::Branch) {
            return false;
        }
        conditions.erase(side(reason.endpoint));
        by_level.erase(latest);
        addSupports(reason.supports);
        return true;
    }

    [[nodiscard]] bool uniqueAtHighestLevel() const {
        return by_level.size() == 1 ||
               std::prev(by_level.end(), 2)->first < by_level.rbegin()->first;
    }
};

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

bool binaryFixing(const BranchDecision& condition) {
    return condition.bound == (condition.is_upper ? 0.0 : 1.0);
}

// Check one side of a row using the saved activity endpoints. Each fused
// multiply-add is rounded outwards, so cancellation cannot manufacture a
// violation. An overflow or an infinite endpoint leaves this proof unchecked.
bool checksRowSide(const LpProblem& model, const BoundReasonTrail& trail, Index row,
                   const BoundReasonSupports& supports, std::optional<BoundEndpoint> opposite,
                   bool minimum) {
    const auto sparse = model.matrix.row(row);
    auto ids = supports.ids();
    auto next = ids.begin();
    Real activity = 0.0;
    for (Index k = 0; k < sparse.size(); ++k) {
        const auto col = sparse.indices[k];
        const Real coefficient = sparse.values[k];
        if (coefficient == 0.0) {
            continue;
        }
        BoundEndpoint endpoint;
        if (opposite && opposite->col == col) {
            endpoint = *opposite;
        } else {
            if (next == ids.end()) {
                return false;
            }
            endpoint = trail.reason(*next).endpoint;
            ++next;
        }
        if (endpoint.col != col || endpoint.is_upper != ((coefficient < 0.0) == minimum)) {
            return false;
        }
        const Real sum = std::fma(coefficient, endpoint.value, activity);
        if (!std::isfinite(sum)) {
            return false;
        }
        activity = std::nextafter(sum, minimum ? -kInf : kInf);
    }
    return next == ids.end() &&
           (minimum ? activity > model.row_upper[row] : activity < model.row_lower[row]);
}

bool checksRow(const LpProblem& model, const BoundReasonTrail& trail, Index row,
               const BoundReasonSupports& supports, std::optional<BoundEndpoint> opposite = {}) {
    if (row < 0 || row >= model.num_rows) {
        return false;
    }
    return checksRowSide(model, trail, row, supports, opposite, true) ||
           checksRowSide(model, trail, row, supports, opposite, false);
}

bool checksDeduction(const LpProblem& model, const BoundReasonTrail& trail, const BoundReason& step,
                     std::span<const BranchDecision> premises) {
    const auto& endpoint = step.endpoint;
    switch (step.source) {
        case BoundReasonSource::Model:
            return endpoint.value == (endpoint.is_upper ? model.col_upper[endpoint.col]
                                                        : model.col_lower[endpoint.col]);
        case BoundReasonSource::Branch:
            return std::ranges::any_of(premises, [&](const BranchDecision& premise) {
                return premise.variable == endpoint.col && premise.is_upper == endpoint.is_upper &&
                       premise.bound == endpoint.value;
            });
        case BoundReasonSource::Row: {
            auto opposite = endpoint;
            opposite.is_upper = !endpoint.is_upper;
            if (model.col_type[endpoint.col] != VarType::Continuous) {
                if (endpoint.value != std::floor(endpoint.value)) {
                    return false;
                }
                const Real delta = endpoint.is_upper ? 1.0 : -1.0;
                opposite.value += delta;
                if (opposite.value - endpoint.value != delta) {
                    return false;
                }
            }
            return checksRow(model, trail, step.source_index, step.supports, opposite);
        }
        case BoundReasonSource::IntegerRounding: {
            if (model.col_type[endpoint.col] == VarType::Continuous) {
                return false;
            }
            auto ids = step.supports.ids();
            auto id = ids.begin();
            if (id == ids.end()) {
                return false;
            }
            const auto& premise = trail.reason(*id++).endpoint;
            const Real rounded =
                endpoint.is_upper ? std::floor(premise.value) : std::ceil(premise.value);
            return id == ids.end() && premise.col == endpoint.col &&
                   premise.is_upper == endpoint.is_upper &&
                   entails(rounded, endpoint.is_upper, endpoint.value);
        }
        default:
            return false;
    }
}

bool checksContradiction(const LpProblem& model, const BoundReasonTrail& trail,
                         std::span<const BranchDecision> premises) {
    const auto& failure = trail.contradiction();
    if (!failure || failure->scope.node_local || failure->scope.cutoff != kInf) {
        return false;
    }
    std::vector<BoundReasonId> pending;
    for (const auto id : failure->supports.ids()) {
        pending.push_back(id);
    }
    if (failure->source == BoundReasonSource::Crossing && pending.size() == 2) {
        const auto& first = trail.reason(pending[0]).endpoint;
        const auto& second = trail.reason(pending[1]).endpoint;
        const Real lower = first.is_upper ? second.value : first.value;
        const Real upper = first.is_upper ? first.value : second.value;
        if (first.col != second.col || first.is_upper == second.is_upper || !(lower > upper)) {
            return false;
        }
    } else if (failure->source != BoundReasonSource::Row ||
               !checksRow(model, trail, failure->source_index, failure->supports)) {
        return false;
    }
    std::unordered_set<BoundReasonId> checked;
    while (!pending.empty()) {
        const auto id = pending.back();
        pending.pop_back();
        if (checked.insert(id).second) {
            const auto& step = trail.reason(id);
            if (step.scope.node_local || step.scope.cutoff != kInf ||
                !checksDeduction(model, trail, step, premises)) {
                return false;
            }
            for (const auto support : step.supports.ids()) {
                pending.push_back(support);
            }
        }
    }
    return true;
}

std::vector<BranchDecision> reconvergenceConditions(const BoundReasonTrail& trail,
                                                    BoundReasonId target) {
    const auto& reason = trail.reason(target);
    auto complement = literal(reason);
    if (!binaryFixing(complement)) {
        return {};
    }
    complement.is_upper = !complement.is_upper;
    complement.bound = 1.0 - complement.bound;

    // Replace the conflict sink by the first UIP. Look for a second UIP that
    // reaches it along multiple paths at that level.
    ConflictFrontier frontier{trail, {}, {}};
    frontier.addSupports(reason.supports);
    bool saw_fork = false;
    while (!frontier.by_level.empty() && frontier.by_level.rbegin()->first == reason.level) {
        const bool unique = frontier.uniqueAtHighestLevel();
        if (unique && saw_fork) {
            const auto earlier_uip = frontier.by_level.rbegin()->second;
            std::vector<BranchDecision> forbidden{complement};
            for (auto it = frontier.conditions.begin(); it != frontier.conditions.end();) {
                const auto id = it->second;
                if (id != earlier_uip && subsumed(trail, frontier.conditions, id)) {
                    it = frontier.conditions.erase(it);
                } else {
                    forbidden.push_back(literal(trail.reason(id)));
                    ++it;
                }
            }
            return forbidden;
        }
        saw_fork |= !unique;
        if (!frontier.resolveLatest()) {
            break;
        }
    }
    return {};
}

std::optional<LearnedBoundConflict> analyzeConflict(const BoundReasonTrail& trail,
                                                    const BoundReasonSupports& supports,
                                                    BoundReasonScope scope,
                                                    bool find_reconvergence) {
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
    ConflictFrontier frontier{trail, {}, {}};
    frontier.addSupports(supports);

    BoundReasonId uip = kNoBoundReason;
    while (!frontier.by_level.empty()) {
        const auto [level, id] = *frontier.by_level.rbegin();
        if (level <= 0) {
            break;
        }
        if (frontier.uniqueAtHighestLevel()) {
            uip = id;
            break;
        }
        if (!frontier.resolveLatest()) {
            // Several independent decisions can share a depth (e.g. SOS).
            return std::nullopt;
        }
    }
    if (uip == kNoBoundReason) {
        return std::nullopt;
    }

    LearnedBoundConflict result;
    result.scope = scope;
    if (find_reconvergence && result.scope.cutoff == kInf) {
        result.reconvergence = reconvergenceConditions(trail, uip);
    }
    // This also discharges deductions from model bounds alone at level zero.
    // A bound imported as a decision at level zero still needs a premise.
    for (auto it = frontier.conditions.begin(); it != frontier.conditions.end();) {
        const auto id = it->second;
        if (id != uip && subsumed(trail, frontier.conditions, id)) {
            result.removed_literals.push_back(literal(trail.reason(id)));
            it = frontier.conditions.erase(it);
        } else {
            ++it;
        }
    }
    std::set<Int> levels;
    result.literals.reserve(frontier.conditions.size());
    for (const auto& entry : frontier.conditions) {
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

}  // namespace

bool checkGlobalBoundContradiction(const LpProblem& model, const BoundReasonTrail& trail,
                                   std::span<const BranchDecision> premises) {
    if (std::ranges::any_of(model.col_type, [](VarType type) {
            return type == VarType::SemiContinuous || type == VarType::SemiInteger;
        })) {
        return false;
    }
    return checksContradiction(model, trail, premises);
}

std::optional<LearnedBoundConflict> analyzeBoundConflict(const BoundReasonTrail& trail,
                                                         bool find_reconvergence) {
    const auto& contradiction = trail.contradiction();
    if (!contradiction || contradiction->source == BoundReasonSource::Opaque) {
        return std::nullopt;
    }
    return analyzeConflict(trail, contradiction->supports, contradiction->scope,
                           find_reconvergence);
}

std::optional<LearnedBoundConflict> analyzeBoundConflict(const BoundReasonTrail& trail,
                                                         const BoundReasonSupports& supports,
                                                         BoundReasonScope scope) {
    return analyzeConflict(trail, supports, scope, false);
}

std::optional<Cut> checkedReconvergenceCut(std::span<const BranchDecision> forbidden,
                                           const LpProblem& global_model) {
    if (forbidden.empty() || std::ranges::any_of(global_model.col_type, [](VarType type) {
            return type == VarType::SemiContinuous || type == VarType::SemiInteger;
        })) {
        return std::nullopt;
    }
    std::vector<BranchDecision> sorted(forbidden.begin(), forbidden.end());
    std::ranges::sort(sorted, {}, &BranchDecision::variable);
    Cut cut;
    cut.upper = -1.0;
    for (const auto& condition : sorted) {
        const Index col = condition.variable;
        if (col < 0 || col >= global_model.num_cols || !binaryFixing(condition) ||
            !isBinaryColumn(global_model, col) || global_model.col_lower[col] < 0.0 ||
            global_model.col_upper[col] > 1.0 ||
            (!cut.indices.empty() && cut.indices.back() == col)) {
            return std::nullopt;
        }
        // A satisfied forbidden literal contributes x if it fixes x=1,
        // and 1-x if it fixes x=0. At least one literal must be false.
        cut.indices.push_back(col);
        cut.values.push_back(condition.is_upper ? -1.0 : 1.0);
        cut.upper += condition.is_upper ? 0.0 : 1.0;
    }

    // Recheck after subsumption under the candidate clause alone. The
    // proof trail may use rows or bounds valid only for a subtree or cutoff.
    DomainPropagator verifier;
    verifier.load(global_model, true);
    for (const auto& condition : sorted) {
        verifier.setBranchBound(condition.variable, condition.is_upper, condition.bound, 1);
    }
    // The ordinary propagation pass discovers the certificate. Check its
    // deductions separately before accepting the resulting inequality.
    const auto& replay = *verifier.boundReasons();
    if (!replay.contradiction() && verifier.propagate()) {
        return std::nullopt;
    }
    if (!checksContradiction(global_model, replay, sorted)) {
        return std::nullopt;
    }
    return cut;
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
