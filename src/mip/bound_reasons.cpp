#include "mipx/bound_reasons.h"

#include <algorithm>
#include <utility>

namespace mipx {

namespace {

void inheritScope(BoundReasonScope& scope, const BoundReasonScope& dependency) {
    scope.node_local |= dependency.node_local;
    scope.cutoff = std::min(scope.cutoff, dependency.cutoff);
}

}  // namespace

BoundReasonSupports::BoundReasonSupports(std::initializer_list<BoundReasonId> supports)
    : BoundReasonSupports(std::vector<BoundReasonId>(supports)) {}

BoundReasonSupports::BoundReasonSupports(std::vector<BoundReasonId> supports)
    : snapshot_(std::make_shared<const std::vector<BoundReasonId>>(std::move(supports))) {}

BoundReasonSupports::BoundReasonSupports(Snapshot snapshot, std::size_t excluded)
    : snapshot_(std::move(snapshot)), excluded_(excluded) {}

void BoundReasonTrail::load(std::span<const Real> lower, std::span<const Real> upper) {
    current_.assign(lower.size(), {kNoBoundReason, kNoBoundReason});
    reasons_.clear();
    contradiction_.reset();
    checkpoints_.clear();
    for (Index col = 0; col < static_cast<Index>(lower.size()); ++col) {
        record({col, false, lower[col]}, BoundReasonSource::Model);
        record({col, true, upper[col]}, BoundReasonSource::Model);
    }
}

void BoundReasonTrail::pushCheckpoint() {
    checkpoints_.push_back({reasons_.size(), contradiction_});
}

void BoundReasonTrail::popCheckpoint() {
    auto& checkpoint = checkpoints_.back();
    while (reasons_.size() > checkpoint.size) {
        const auto& entry = reasons_.back();
        auto& current_id = current_[entry.endpoint.col][entry.endpoint.is_upper];
        if (current_id == reasons_.size() - 1) {
            current_id = entry.previous;
        }
        reasons_.pop_back();
    }
    contradiction_ = std::move(checkpoint.contradiction);
    checkpoints_.pop_back();
}

BoundReasonId BoundReasonTrail::record(BoundEndpoint endpoint, BoundReasonSource source,
                                       BoundReasonSupports supports, Index source_index,
                                       BoundReasonScope scope, Int level, bool applied) {
    for (const auto support : supports.ids()) {
        inheritScope(scope, reasons_[support].scope);
        level = std::max(level, reasons_[support].level);
    }
    auto& current_id = current_[endpoint.col][endpoint.is_upper];
    const auto id = reasons_.size();
    reasons_.push_back(
        {endpoint, source, source_index, level, scope, std::move(supports), current_id});
    if (applied) {
        current_id = id;
    }
    return id;
}

void BoundReasonTrail::fail(BoundReasonSource source, BoundReasonSupports supports,
                            Index source_index, BoundReasonScope scope) {
    for (const auto support : supports.ids()) {
        inheritScope(scope, reasons_[support].scope);
    }
    contradiction_ = BoundContradiction{source, source_index, scope, std::move(supports)};
}

}  // namespace mipx
