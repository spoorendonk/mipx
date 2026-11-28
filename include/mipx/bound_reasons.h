#pragma once

#include "mipx/core.h"
#include "mipx/lp_problem.h"

#include <array>
#include <cstddef>
#include <initializer_list>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <vector>

namespace mipx {

struct BoundEndpoint {
    Index col;
    bool is_upper;
    Real value;
};

enum class BoundReasonSource {
    Model,
    Branch,
    Opaque,
    IntegerRounding,
    Row,
    Implication,
    Clique,
    Objective,
    Crossing
};

// Local rows restrict reuse to the current node. Cutoff proofs also require
// a cutoff at least as strong, even when the rows they use are all global.
struct BoundReasonScope {
    bool node_local = false;
    Real cutoff = kInf;
};

using BoundReasonId = std::size_t;
inline constexpr BoundReasonId kNoBoundReason = static_cast<BoundReasonId>(-1);

// All deductions from one activity sum can share its snapshot. Each skips
// the target term when taking the residual, and zero terms use kNoBoundReason.
class BoundReasonSupports {
public:
    using Snapshot = std::shared_ptr<const std::vector<BoundReasonId>>;

    BoundReasonSupports() = default;
    BoundReasonSupports(std::initializer_list<BoundReasonId> supports);
    BoundReasonSupports(std::vector<BoundReasonId> supports);
    BoundReasonSupports(Snapshot snapshot, std::size_t excluded = kNoBoundReason);

    [[nodiscard]] auto ids() const {
        const auto snapshot = snapshot_ ? std::span<const BoundReasonId>{*snapshot_}
                                        : std::span<const BoundReasonId>{};
        return std::views::iota(std::size_t{0}, snapshot.size()) |
               std::views::filter([snapshot, excluded = excluded_](std::size_t k) {
                   return k != excluded && snapshot[k] != kNoBoundReason;
               }) |
               std::views::transform([snapshot](std::size_t k) { return snapshot[k]; });
    }

private:
    Snapshot snapshot_;
    std::size_t excluded_ = kNoBoundReason;
};

struct BoundReason {
    BoundEndpoint endpoint;
    BoundReasonSource source;
    Index source_index;
    Int level;
    BoundReasonScope scope;
    BoundReasonSupports supports;
    BoundReasonId previous;
};

struct BoundContradiction {
    BoundReasonSource source;
    Index source_index;
    BoundReasonScope scope;
    BoundReasonSupports supports;
};

// IDs name immutable snapshots of endpoints. They remain valid until the
// checkpoint that created them is popped or the domain is loaded again.
class BoundReasonTrail {
public:
    [[nodiscard]] BoundReasonId current(Index col, bool is_upper) const {
        return current_[col][is_upper];
    }
    [[nodiscard]] const BoundReason& reason(BoundReasonId id) const { return reasons_[id]; }
    [[nodiscard]] const std::optional<BoundContradiction>& contradiction() const {
        return contradiction_;
    }

private:
    friend class DomainPropagator;

    void load(std::span<const Real> lower, std::span<const Real> upper);
    void pushCheckpoint();
    void popCheckpoint();
    BoundReasonId record(BoundEndpoint endpoint, BoundReasonSource source,
                         BoundReasonSupports supports = {}, Index source_index = -1,
                         BoundReasonScope scope = {}, Int level = 0, bool applied = true);
    void fail(BoundReasonSource source, BoundReasonSupports supports, Index source_index = -1,
              BoundReasonScope scope = {});

    struct Checkpoint {
        std::size_t size;
        std::optional<BoundContradiction> contradiction;
    };

    std::vector<std::array<BoundReasonId, 2>> current_;
    std::vector<BoundReason> reasons_;
    std::optional<BoundContradiction> contradiction_;
    std::vector<Checkpoint> checkpoints_;
};

}  // namespace mipx
