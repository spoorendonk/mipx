#include "mipx/bound_conflict.h"
#include "mipx/deadline.h"
#include "mipx/mip_solver.h"

#include <cmath>
#include <map>
#include <set>
#include <utility>

namespace mipx {
namespace {

constexpr Int kNodesPerBlock = 8;
constexpr Int kEmptyBlocksToStop = 2;

// Use the sign of the exact addition residual (FastTwoSum) to widen only
// when necessary. Exactly representable activities can certify equalities.
Real directedAdd(Real a, Real b, bool upper) {
    if (std::abs(a) < std::abs(b)) {
        std::swap(a, b);
    }
    const Real sum = a + b;
    if (!std::isfinite(sum)) {
        return sum;
    }
    const Real error = b - (sum - a);
    return (upper ? error > 0.0 : error < 0.0) ? std::nextafter(sum, upper ? kInf : -kInf) : sum;
}

struct ActivityInterval {
    Real lower = 0.0;
    Real upper = 0.0;

    bool addProduct(Real coefficient, Real value) {
        const Real product = coefficient * value;
        if (!std::isfinite(product)) {
            return false;
        }
        Real lo = product;
        Real hi = product;
        if (coefficient != 0.0 && value != 0.0 && std::abs(coefficient) != 1.0 &&
            std::abs(value) != 1.0) {
            // Near underflow the product residual can round to zero. Widen
            // both sides where a product of two doubles can have bits below
            // the smallest subnormal (2^-1074).
            const bool residual_may_underflow = std::abs(product) < 0x1p-968;
            const Real error = std::fma(coefficient, value, -product);
            if (residual_may_underflow || error < 0.0) {
                lo = std::nextafter(product, -kInf);
            }
            if (residual_may_underflow || error > 0.0) {
                hi = std::nextafter(product, kInf);
            }
        }
        lower = directedAdd(lower, lo, false);
        upper = directedAdd(upper, hi, true);
        return std::isfinite(lower) && std::isfinite(upper);
    }
};

// LP tolerances alone cannot certify that a constructed point supplies a
// cutoff: a small violation can buy a large apparent objective gain.
std::optional<Real> checkedPointObjective(const LpProblem& model, std::span<const Real> point,
                                          BoundProofBudget& budget) {
    if (point.size() != static_cast<std::size_t>(model.num_cols) ||
        !std::isfinite(model.obj_offset)) {
        return std::nullopt;
    }
    ActivityInterval objective{model.obj_offset, model.obj_offset};
    for (Index col = 0; col < model.num_cols; ++col) {
        const Real value = point[col];
        if (!std::isfinite(value) || value < model.col_lower[col] || value > model.col_upper[col] ||
            (model.col_type[col] != VarType::Continuous && std::trunc(value) != value) ||
            !objective.addProduct(model.obj[col], value)) {
            return std::nullopt;
        }
    }
    for (const auto& members : model.native_sos2) {
        const auto [first, last] = sos2Support(model, members, point);
        if (last - first > 1) {
            return std::nullopt;
        }
    }
    for (Index row = 0; row < model.num_rows; ++row) {
        if (!budget.debitRow()) {
            return std::nullopt;
        }
        ActivityInterval activity;
        const auto sparse = model.matrix.row(row);
        for (Index k = 0; k < sparse.size(); ++k) {
            if (!activity.addProduct(sparse.values[k], point[sparse.indices[k]])) {
                return std::nullopt;
            }
        }
        if (activity.lower < model.row_lower[row] || activity.upper > model.row_upper[row]) {
            return std::nullopt;
        }
    }
    return objective.upper;
}

std::vector<Real> candidatePoint(const LpProblem& model, const DomainPropagator& domain) {
    std::vector<Real> point(model.num_cols);
    for (Index col = 0; col < model.num_cols; ++col) {
        Real lower = domain.getLower(col);
        Real upper = domain.getUpper(col);
        if (model.col_type[col] != VarType::Continuous) {
            lower = std::ceil(lower);
            upper = std::floor(upper);
        }
        if (lower > upper) {
            return {};
        }
        const Real preferred = model.obj[col] < 0.0 ? upper : lower;
        point[col] = std::isfinite(preferred) ? preferred : std::clamp(0.0, lower, upper);
    }
    return point;
}

}  // namespace

bool MipSolver::runRootRapidLearning(SolutionPool& solutions, const LpProblem* loaded,
                                     const Presolver* postsolve) {
    auto& stats = rapid_learning_stats_;
    const auto deadline = Deadline::after(timeLeft());
    if (deadline.passed()) {
        stats.stop = RapidLearningStop::Deadline;
        return false;
    }
    if (root_rapid_learning_max_nodes_ == 0 || root_rapid_learning_row_budget_ == 0) {
        stats.stop = root_rapid_learning_max_nodes_ == 0 ? RapidLearningStop::NodeLimit
                                                         : RapidLearningStop::RowLimit;
        return false;
    }

    DomainPropagator domain;
    // The proof checker sees the same global rows and root bounds. Keep
    // graph-, clique- and cutoff-derived deductions outside this domain.
    domain.load(problem_, true);
    const auto& trail = *domain.boundReasons();
    std::vector<Index> binaries;
    for (Index col = 0; col < problem_.num_cols; ++col) {
        if (isBinaryColumn(problem_, col) && problem_.col_lower[col] == 0.0 &&
            problem_.col_upper[col] == 1.0) {
            binaries.push_back(col);
        }
    }

    using BoundSide = std::pair<Index, bool>;
    Int rows_left = root_rapid_learning_row_budget_;
    BoundProofBudget budget{rows_left, deadline};
    std::map<BoundSide, Real> imported_bounds;
    std::set<std::vector<BoundSide>> imported_conflicts;
    const auto import_bound = [&](BoundEndpoint bound) {
        const auto [col, upper, value] = bound;
        const Real model_value = upper ? problem_.col_upper[col] : problem_.col_lower[col];
        const auto [it, inserted] = imported_bounds.try_emplace({col, upper}, model_value);
        if (upper ? value < it->second : value > it->second) {
            it->second = value;
            ++stats.bounds_imported;
        } else if (inserted) {
            imported_bounds.erase(it);
        }
    };
    const auto import_root_bounds = [&]() {
        std::vector<Index> tightened;
        domain.collectTightenedColumns(tightened);
        for (const Index col : tightened) {
            for (const bool upper : {false, true}) {
                const Real value = upper ? domain.getUpper(col) : domain.getLower(col);
                if (value == (upper ? problem_.col_upper[col] : problem_.col_lower[col])) {
                    continue;
                }
                if (checkGlobalBoundDeduction(problem_, trail, trail.current(col, upper),
                                              &budget)) {
                    import_bound({col, upper, value});
                } else if (budget.interrupted) {
                    return;
                } else {
                    ++stats.proofs_rejected;
                }
            }
        }
    };
    const auto import_point = [&]() {
        auto point = candidatePoint(problem_, domain);
        auto objective = checkedPointObjective(problem_, point, budget);
        std::vector<Real> mapped;
        if (objective && loaded != nullptr) {
            if (postsolve != nullptr) {
                mapped = postsolve->postsolve(point);
            }
            const auto loaded_objective =
                checkedPointObjective(*loaded, postsolve != nullptr ? mapped : point, budget);
            if (loaded_objective) {
                // Both objective representations include their own offset.
                *objective = std::max(*objective, *loaded_objective);
            } else {
                objective.reset();
            }
        }
        if (budget.interrupted || deadline.passed()) {
            return;
        }
        if (!objective) {
            ++stats.incumbents_rejected;
            return;
        }
        if (*objective >= solutions.bestObjective()) {
            return;
        }
        if (incumbent_observer_ && postsolve == nullptr) {
            mapped = point;
        }
        if (solutions.submit({std::move(point), *objective}, "root_rapid_learning", 0)) {
            ++stats.incumbents_imported;
            if (incumbent_observer_) {
                incumbent_observer_(mapped, "root_rapid_learning");
            }
        }
    };

    std::vector<BranchDecision> path;
    const auto prefer_upper = [&](Index col) { return problem_.obj[col] >= 0.0; };
    const auto enter = [&](Index col, bool upper) {
        domain.pushCheckpoint();
        const Int level = static_cast<Int>(path.size()) + 1;
        path.push_back({col, upper ? 0.0 : 1.0, upper, level});
        domain.setReasonLevel(level);
        domain.setBranchBound(col, upper, path.back().bound, level);
    };
    const auto advance = [&]() {
        while (!path.empty()) {
            const auto branch = path.back();
            path.pop_back();
            domain.popCheckpoint();
            if (branch.is_upper == prefer_upper(branch.variable)) {
                enter(branch.variable, !branch.is_upper);
                return true;
            }
        }
        return false;
    };
    const auto import_conflict = [&]() {
        auto proof = checkedGlobalBoundConflict(problem_, trail, path, &budget);
        if (!proof) {
            if (!budget.interrupted) {
                ++stats.proofs_rejected;
            }
            return false;
        }
        if (proof->empty()) {
            return true;
        }
        if (proof->size() == 1) {
            const auto& branch = proof->front();
            import_bound({branch.variable, !branch.is_upper, 1.0 - branch.bound});
        } else if (conflicts_enabled_ && num_threads_ <= 1) {
            // All literals are binary branch endpoints. Canonicalize without
            // the search level, which can differ across two proofs.
            std::ranges::sort(*proof, {}, &BranchDecision::variable);
            std::vector<BoundSide> key;
            key.reserve(proof->size());
            for (const auto& branch : *proof) {
                key.emplace_back(branch.variable, branch.is_upper);
            }
            if (!imported_conflicts.insert(std::move(key)).second) {
                ++stats.duplicate_conflicts;
                return false;
            }
            auto clause = acquireConflictClause();
            clause.literals = std::move(*proof);
            clause.lbd = static_cast<Int>(clause.literals.size());
            clause.bound_scope = BoundReasonScope{};
            publishConflict(std::move(clause), ConflictSource::RootRapidLearning);
            ++stats.conflicts_imported;
        }
        return false;
    };

    const auto out_of_work = [&]() {
        if (deadline.passed()) {
            stats.stop = RapidLearningStop::Deadline;
            return true;
        }
        if (rows_left == 0) {
            stats.stop = RapidLearningStop::RowLimit;
            return true;
        }
        return false;
    };
    std::uint64_t last_block_yield = 0;
    for (;;) {
        ++stats.nodes;
        const bool feasible =
            !trail.contradiction() && domain.propagateWithRowBudget(rows_left, &deadline);
        if (deadline.passed()) {
            stats.stop = RapidLearningStop::Deadline;
            break;
        }
        if (!feasible) {
            if (import_conflict()) {
                stats.stop = RapidLearningStop::Infeasible;
                break;
            }
        } else if (path.empty()) {
            import_root_bounds();
        }
        if (out_of_work()) {
            break;
        }

        auto split = binaries.end();
        if (feasible) {
            split = std::ranges::find_if(binaries, [&](Index col) {
                return domain.getLower(col) <= 0.0 && domain.getUpper(col) >= 1.0;
            });
            if (split == binaries.end()) {
                import_point();
                if (out_of_work()) {
                    break;
                }
            }
        }
        if (stats.nodes % kNodesPerBlock == 0) {
            const auto facts =
                stats.bounds_imported + stats.conflicts_imported + stats.incumbents_imported;
            stats.empty_blocks = facts == last_block_yield ? stats.empty_blocks + 1 : 0;
            last_block_yield = facts;
            if (stats.empty_blocks == kEmptyBlocksToStop) {
                stats.stop = RapidLearningStop::NoYield;
                break;
            }
        }
        if (stats.nodes == root_rapid_learning_max_nodes_) {
            stats.stop = RapidLearningStop::NodeLimit;
            break;
        }
        if (split != binaries.end()) {
            enter(*split, prefer_upper(*split));
        } else if (!advance()) {
            stats.stop = RapidLearningStop::Exhausted;
            break;
        }
    }
    stats.row_visits = root_rapid_learning_row_budget_ - rows_left;
    if (stats.stop == RapidLearningStop::Infeasible) {
        return true;
    }

    // Keep verification on the original root domain throughout the search.
    // Only now can the main solver install the checked bounds.
    for (const auto& [side, value] : imported_bounds) {
        const auto [col, upper] = side;
        (upper ? problem_.col_upper[col] : problem_.col_lower[col]) = value;
        if (problem_.col_lower[col] > problem_.col_upper[col]) {
            stats.stop = RapidLearningStop::Infeasible;
            return true;
        }
    }
    return false;
}

}  // namespace mipx
