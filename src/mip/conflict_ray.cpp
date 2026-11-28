#include "conflict_ray.h"

#include "mipx/bnb_node.h"
#include "mipx/lp_problem.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace mipx {
namespace {

using Wide = long double;
constexpr Wide kWideInf = std::numeric_limits<Wide>::infinity();

Wide down(Wide value) {
    return std::nextafter(value, -kWideInf);
}
Wide up(Wide value) {
    return std::nextafter(value, kWideInf);
}

struct Interval {
    Wide lower = 0.0L;
    Wide upper = 0.0L;

    void addProduct(Real a, Real b) {
        if (a != 0.0 && b != 0.0) {
            const Wide value = static_cast<Wide>(a) * b;
            lower = down(lower + down(value));
            upper = up(upper + up(value));
        }
    }
};

// A lower bound on d*x over both the enclosure of d and the column box.
// An uncertain sign cannot price a free side, even if the nominal sum is zero.
Wide minimumProduct(Interval coefficient, Real lower, Real upper) {
    if (!std::isfinite(coefficient.lower) || !std::isfinite(coefficient.upper)) {
        return -kWideInf;
    }
    Wide result = kWideInf;
    if (coefficient.upper > 0.0L) {
        if (!std::isfinite(lower)) {
            return -kWideInf;
        }
        const Wide d = lower < 0.0 ? coefficient.upper : std::max(0.0L, coefficient.lower);
        result = down(d * lower);
    }
    if (coefficient.lower < 0.0L) {
        if (!std::isfinite(upper)) {
            return -kWideInf;
        }
        const Wide d = upper > 0.0 ? coefficient.lower : std::min(0.0L, coefficient.upper);
        result = std::min(result, down(d * upper));
    }
    return result == kWideInf ? 0.0L : result;
}

bool shortenConflictWithProof(const LpProblem& global_rows, std::span<const Real> weights,
                              std::span<const Index> local_rows, std::span<const Real> global_lower,
                              std::span<const Real> global_upper, std::vector<BranchDecision>& path,
                              std::optional<Real> cutoff) {
    const auto cols = static_cast<std::size_t>(global_rows.num_cols);
    if ((!cutoff && weights.empty()) ||
        weights.size() != static_cast<std::size_t>(global_rows.num_rows) + local_rows.size() ||
        global_lower.size() != cols || global_upper.size() != cols || path.empty()) {
        return false;
    }

    // Each row contributes y*A*x <= y*side. In an objective proof add
    // c*x <= cutoff - offset, so separation of the box proves a lower bound
    // on the original objective. Enclose coefficients left by cancellation.
    std::vector<Interval> coefficients(cols);
    Interval beta;
    Wide row_magnitude = 0.0L;
    if (cutoff) {
        if (global_rows.sense != Sense::Minimize || global_rows.obj.size() != cols ||
            !std::isfinite(*cutoff) || !std::isfinite(global_rows.obj_offset)) {
            return false;
        }
        for (std::size_t j = 0; j < cols; ++j) {
            coefficients[j] = {global_rows.obj[j], global_rows.obj[j]};
        }
        const Wide rhs = static_cast<Wide>(*cutoff) - global_rows.obj_offset;
        beta = {down(rhs), up(rhs)};
        row_magnitude = std::abs(rhs);
    }
    std::size_t local = 0;
    Index row = 0;
    for (Index i = 0; i < static_cast<Index>(weights.size()); ++i) {
        const Real y = cutoff ? -weights[i] : weights[i];
        if (local < local_rows.size() && i == local_rows[local]) {
            if (y != 0.0) {
                return false;
            }
            ++local;
            continue;
        }
        if (row >= global_rows.num_rows || !std::isfinite(y)) {
            return false;
        }
        const Index global_row = row++;
        if (y == 0.0) {
            continue;
        }
        const Real side =
            y > 0.0 ? global_rows.row_upper[global_row] : global_rows.row_lower[global_row];
        if (!std::isfinite(side)) {
            return false;
        }
        beta.addProduct(y, side);
        row_magnitude += std::abs(static_cast<Wide>(y) * side);
        const auto entries = global_rows.matrix.row(global_row);
        for (Index k = 0; k < entries.size(); ++k) {
            coefficients[entries.indices[k]].addProduct(y, entries.values[k]);
        }
    }
    if (!std::isfinite(beta.upper) || local != local_rows.size()) {
        return false;
    }

    struct PathColumn {
        Index index;
        Real lower;
        Real upper;
    };
    std::vector<PathColumn> columns;
    std::vector<std::size_t> positions;
    columns.reserve(path.size());
    positions.reserve(path.size());
    for (std::size_t i = 0; i < path.size(); ++i) {
        const auto& literal = path[i];
        const Index j = literal.variable;
        if (j < 0 || static_cast<std::size_t>(j) >= cols || !std::isfinite(literal.bound)) {
            return false;
        }
        if (i > 0 && (path[i - 1].variable > j ||
                      (path[i - 1].variable == j && path[i - 1].is_upper >= literal.is_upper))) {
            return false;
        }
        if (columns.empty() || columns.back().index != j) {
            columns.push_back({j, global_lower[j], global_upper[j]});
        }
        positions.push_back(columns.size() - 1);
        auto& column = columns.back();
        if (literal.is_upper) {
            column.upper = std::min(column.upper, literal.bound);
        } else {
            column.lower = std::max(column.lower, literal.bound);
        }
    }

    auto accumulate = [&](Index j, Real lower, Real upper, Wide& alpha, Wide& magnitude) {
        if (!(lower <= upper)) {
            return false;
        }
        const auto& coefficient = coefficients[j];
        if (coefficient.lower != 0.0L || coefficient.upper != 0.0L) {
            const Wide term = minimumProduct(coefficient, lower, upper);
            if (!std::isfinite(term)) {
                return false;
            }
            alpha = down(alpha + term);
            magnitude += std::abs(term);
        }
        return true;
    };

    // The minimum on columns absent from the path is the same for every
    // trial. Compute it once even when the rest of the model is large.
    Wide global_alpha = 0.0L;
    Wide global_magnitude = row_magnitude;
    auto path_column = columns.begin();
    for (Index j = 0; j < global_rows.num_cols; ++j) {
        if (path_column != columns.end() && path_column->index == j) {
            ++path_column;
        } else if (!accumulate(j, global_lower[j], global_upper[j], global_alpha,
                               global_magnitude)) {
            return false;
        }
    }
    auto proves = [&]() {
        Wide alpha = global_alpha;
        Wide magnitude = global_magnitude;
        for (const auto& column : columns) {
            if (!accumulate(column.index, column.lower, column.upper, alpha, magnitude)) {
                return false;
            }
        }
        // The ray API can return a marginal certificate within simplex
        // tolerances. Require a positive separation with room to spare.
        return std::isfinite(alpha) && alpha - beta.upper > 1e-9L * magnitude;
    };

    if (!proves()) {
        return false;
    }
    std::size_t kept = 0;
    for (std::size_t i = 0; i < path.size(); ++i) {
        const auto& literal = path[i];
        const Index j = literal.variable;
        auto& column = columns[positions[i]];
        Real& bound = literal.is_upper ? column.upper : column.lower;
        const Real saved = bound;
        bound = literal.is_upper ? global_upper[j] : global_lower[j];
        if ((kept == 0 && i + 1 == path.size()) || !proves()) {
            bound = saved;
            path[kept++] = literal;
        }
    }
    path.resize(kept);
    return true;
}

}  // namespace

bool shortenConflictWithRay(const LpProblem& global_rows, std::span<const Real> ray,
                            std::span<const Index> local_rows, std::span<const Real> global_lower,
                            std::span<const Real> global_upper, std::vector<BranchDecision>& path) {
    return shortenConflictWithProof(global_rows, ray, local_rows, global_lower, global_upper, path,
                                    std::nullopt);
}

bool shortenConflictWithCutoff(const LpProblem& global_rows, std::span<const Real> dual,
                               std::span<const Index> local_rows, Real cutoff,
                               std::vector<BranchDecision>& path) {
    return shortenConflictWithProof(global_rows, dual, local_rows, global_rows.col_lower,
                                    global_rows.col_upper, path, cutoff);
}

std::optional<LearnedBoundConflict> checkCombinedLpConflict(
    const LpProblem& global_rows, std::span<const Real> weights, std::span<const Index> local_rows,
    const BoundReasonTrail& trail, BoundReasonScope scope, bool objective_proof) {
    std::vector<BranchDecision> bounds;
    for (Index col = 0; col < global_rows.num_cols; ++col) {
        for (const bool is_upper : {false, true}) {
            const auto id = trail.current(col, is_upper);
            if (id == kNoBoundReason) {
                continue;
            }
            const auto& reason = trail.reason(id);
            if (reason.source == BoundReasonSource::Model || reason.scope.node_local ||
                reason.source == BoundReasonSource::Opaque ||
                !(scope.cutoff <= reason.scope.cutoff)) {
                continue;
            }
            bounds.push_back({col, reason.endpoint.value, is_upper, reason.level});
        }
    }
    const auto cutoff = objective_proof ? std::optional<Real>{scope.cutoff} : std::nullopt;
    if (!shortenConflictWithProof(global_rows, weights, local_rows, global_rows.col_lower,
                                  global_rows.col_upper, bounds, cutoff)) {
        return std::nullopt;
    }

    std::vector<BoundReasonId> supports;
    bool uses_propagation = false;
    for (const auto& bound : bounds) {
        const auto id = trail.current(bound.variable, bound.is_upper);
        uses_propagation |= trail.reason(id).source != BoundReasonSource::Branch;
        supports.push_back(id);
    }
    if (!uses_propagation) {
        return std::nullopt;
    }
    return analyzeBoundConflict(trail, BoundReasonSupports(std::move(supports)), scope);
}

}  // namespace mipx
