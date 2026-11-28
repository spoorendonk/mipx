#include "mipx/bound_conflict.h"
#include "mipx/deadline.h"
#include "mipx/mip_solver.h"

#include <array>
#include <cmath>
#include <deque>
#include <iterator>
#include <set>
#include <tuple>

namespace mipx {
namespace {

struct SplitCandidate {
    Index col = -1;
    Real value = 0.0;
    Real fractionality = 0.0;
};

std::array<SplitCandidate, 2> lookaheadCandidates(const LpProblem& problem,
                                                  std::span<const Real> primal,
                                                  const Deadline& deadline) {
    std::array<SplitCandidate, 2> best;
    for (Index col = 0; col < problem.num_cols; ++col) {
        if (col % 256 == 0 && deadline.passed()) {
            break;
        }
        const VarType type = problem.col_type[col];
        const Real value = primal[col];
        if ((type != VarType::Binary && type != VarType::Integer) || !std::isfinite(value) ||
            isIntegral(value)) {
            continue;
        }
        const SplitCandidate candidate{col, value, fractionality(value)};
        // Scanning in column order keeps the lower index on a tie.
        if (candidate.fractionality > best[0].fractionality) {
            best[1] = best[0];
            best[0] = candidate;
        } else if (candidate.fractionality > best[1].fractionality) {
            best[1] = candidate;
        }
    }
    return best;
}

struct LookaheadParent {
    std::vector<BranchDecision> path;
    std::array<SplitCandidate, 2> candidates;
    std::vector<BasisStatus> basis;
    Real objective;
};

struct ColumnBounds {
    Index col;
    Real lower;
    Real upper;
};

enum class Propagation { Unchanged, Feasible, Infeasible };

using LiteralKey = std::tuple<Index, bool, Real>;

LiteralKey literalKey(const BranchDecision& branch) {
    return {branch.variable, branch.is_upper, branch.bound};
}

// Assemble the path's bounds without any propagation results. Keep the
// lower and upper bounds together when a path splits the same column twice.
std::vector<ColumnBounds> pathBounds(const LpProblem& problem,
                                     std::span<const BranchDecision> path) {
    std::vector<ColumnBounds> bounds;
    bounds.reserve(path.size());
    for (const auto& branch : path) {
        const Index col = branch.variable;
        auto it = std::ranges::find(bounds, col, &ColumnBounds::col);
        if (it == bounds.end()) {
            bounds.push_back({col, problem.col_lower[col], problem.col_upper[col]});
            it = std::prev(bounds.end());
        }
        if (branch.is_upper) {
            it->upper = std::min(it->upper, branch.bound);
        } else {
            it->lower = std::max(it->lower, branch.bound);
        }
    }
    return bounds;
}

}  // namespace

void MipSolver::runRootLpLookahead(const DualSimplexOptions& root_options,
                                   std::span<const Real> root_primals,
                                   std::span<const BasisStatus> root_basis, Real root_objective,
                                   Real incumbent) {
    auto& stats = root_lp_lookahead_stats_;
    const auto deadline = Deadline::after(timeLeft());
    Int work_left = root_lp_lookahead_work_;
    stats.stop = RootLpLookaheadStop::Exhausted;
    const auto stopped = [&]() {
        if (deadline.passed()) {
            stats.stop = RootLpLookaheadStop::Deadline;
        } else if (work_left == 0) {
            stats.stop = RootLpLookaheadStop::WorkLimit;
        }
        return stats.stop != RootLpLookaheadStop::Exhausted;
    };
    if (stopped()) {
        return;
    }
    const auto root_candidates = lookaheadCandidates(problem_, root_primals, deadline);
    if (stopped() || root_candidates.front().col < 0) {
        return;
    }

    DomainPropagator domain;
    domain.load(problem_, true);
    const auto& trail = *domain.boundReasons();
    DualSimplexSolver lp;
    lp.setVerbose(false);
    auto options = root_options;
    options.max_solve_seconds = -1.0;
    options.deadline = deadline;
    lp.setOptions(options);

    const auto propagate = [&](const BranchDecision& branch) {
        domain.setReasonLevel(branch.level);
        const Int changes = domain.numTightened();
        domain.setBranchBound(branch.variable, branch.is_upper, branch.bound, branch.level);
        if (trail.contradiction()) {
            return Propagation::Infeasible;
        }
        // An already entailed bound wakes no rows. Calling propagate here
        // would sweep unrelated rows and attribute their deductions to it.
        if (domain.numTightened() == changes) {
            return Propagation::Unchanged;
        }
        const Int before = work_left;
        const bool feasible = domain.propagateWithRowBudget(work_left, &deadline);
        stats.row_visits += before - work_left;
        return feasible ? Propagation::Feasible : Propagation::Infeasible;
    };
    std::set<std::vector<LiteralKey>> published;
    const auto check_conflict = [&](std::span<const BranchDecision> path) {
        if (!conflicts_enabled_ || num_threads_ > 1) {
            return;
        }
        const Int before = work_left;
        BoundProofBudget budget{work_left, deadline};
        auto proof = checkedGlobalBoundConflict(problem_, trail, path, &budget);
        stats.row_visits += before - work_left;
        if (budget.stopped()) {
            stopped();
            return;
        }
        if (!proof) {
            ++stats.proofs_rejected;
            return;
        }
        if (proof->empty()) {
            return;
        }
        std::vector<LiteralKey> key;
        key.reserve(proof->size());
        for (const auto& literal : *proof) {
            key.push_back(literalKey(literal));
        }
        std::ranges::sort(key);
        const bool known = std::ranges::any_of(conflict_pool_, [&](const ConflictClause& clause) {
            return clause.bound_scope && !clause.bound_scope->node_local &&
                   clause.bound_scope->cutoff == kInf && clause.literals.size() == key.size() &&
                   std::ranges::is_permutation(clause.literals, key, {}, literalKey);
        });
        if (known || !published.insert(std::move(key)).second) {
            ++stats.duplicate_conflicts;
            return;
        }
        auto clause = acquireConflictClause();
        clause.lbd = static_cast<Int>(proof->size());
        clause.literals = std::move(*proof);
        clause.bound_scope = BoundReasonScope{};
        publishConflict(std::move(clause), ConflictSource::RootLpLookahead);
        ++stats.conflicts_imported;
    };

    std::deque<LookaheadParent> frontier;
    frontier.push_back(
        {{}, root_candidates, {root_basis.begin(), root_basis.end()}, root_objective});
    std::vector<Index> tightened;
    const auto probe = [&](const LookaheadParent& parent, const SplitCandidate& candidate,
                           const BranchDecision& branch, const std::vector<BranchDecision>& path) {
        auto& sample = stats.samples.emplace_back();
        sample.branch = branch;
        const auto propagation = propagate(branch);
        if (deadline.passed() || (propagation != Propagation::Infeasible && work_left == 0)) {
            sample.inference.skipped = 1;
            stopped();
            return;
        }
        if (propagation == Propagation::Infeasible) {
            sample.inference.cutoffs = 1;
            check_conflict(path);
            return;
        }
        if (propagation == Propagation::Feasible) {
            domain.collectTightenedColumns(tightened);
            sample.inference.samples = 1;
            sample.inference.tightened_columns = static_cast<std::uint64_t>(std::ranges::count_if(
                tightened, [&](Index col) { return col != branch.variable; }));
        } else {
            sample.inference.skipped = 1;
        }

        const auto bounds = pathBounds(problem_, path);
        if (std::ranges::any_of(bounds, [](const auto& col) { return col.lower > col.upper; }) ||
            stopped()) {
            return;
        }
        if (stats.lp_solves == 0 && !lp.loadUntil(problem_, deadline)) {
            stopped();
            return;
        }
        for (const auto& col : bounds) {
            lp.setColBounds(col.col, col.lower, col.upper);
        }
        lp.setBasis(parent.basis);
        lp.setIterationLimit(--work_left);
        const auto result = lp.solve();
        ++stats.lp_solves;
        stats.lp_iterations += result.iterations;
        stats.lp_work_units += result.work_units;
        work_left -= result.iterations;
        sample.lp_status = result.status;
        const bool cutoff = std::isfinite(incumbent) && result.objective >= pruneBound(incumbent);
        const TreePseudocostSample gain{branch,        candidate.value,  parent.objective,
                                        result.status, result.objective, cutoff};
        if (deadline.passed() || result.status == Status::IterLimit) {
            stopped();
        } else {
            sample.gain_per_unit = gain.gainPerUnit();
        }
        if (sample.gain_per_unit) {
            branching_rule_.updatePseudoCost(branch.variable, !branch.is_upper,
                                             *sample.gain_per_unit);
        }
        if (sample.gain_per_unit && branch.level < root_lp_lookahead_depth_ && !stopped()) {
            const auto candidates = lookaheadCandidates(problem_, lp.getPrimalValues(), deadline);
            if (!stopped() && candidates.front().col >= 0) {
                frontier.push_back(
                    {path, {candidates.front(), {}}, lp.getBasis(), result.objective});
            }
        }
        for (const auto& col : bounds) {
            lp.setColBounds(col.col, problem_.col_lower[col.col], problem_.col_upper[col.col]);
        }
    };

    while (!frontier.empty() && !stopped()) {
        auto parent = std::move(frontier.front());
        frontier.pop_front();
        // Reconstruct a parent's deductions so a descendant's first-pass
        // inference counts only its own changes. Replay visits are charged.
        domain.pushCheckpoint();
        const bool ready = std::ranges::all_of(parent.path, [&](const auto& ancestor) {
            return !stopped() && propagate(ancestor) != Propagation::Infeasible;
        });
        if (ready) {
            auto path = parent.path;
            const Int level = static_cast<Int>(path.size()) + 1;
            for (const auto& candidate : parent.candidates) {
                if (candidate.col < 0 || stopped()) {
                    break;
                }
                for (const bool down : {true, false}) {
                    if (stopped()) {
                        break;
                    }
                    const BranchDecision branch{
                        candidate.col,
                        down ? std::floor(candidate.value) : std::ceil(candidate.value), down,
                        level};
                    path.push_back(branch);
                    domain.pushCheckpoint();
                    probe(parent, candidate, branch, path);
                    domain.popCheckpoint();
                    path.pop_back();
                }
            }
        }
        domain.popCheckpoint();
    }
}

}  // namespace mipx
