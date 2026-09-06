#include "common.h"
#include "mipx/clique_table.h"
#include "mipx/conflict_graph.h"
#include "mipx/dual_simplex.h"
#include "mipx/heuristics.h"

#include <cmath>
#include <utility>
#include <vector>

namespace mipx {

using namespace heuristic_detail;

namespace {

/// A clique member is only forced to 1 when the LP already leans that way.
constexpr Real kCliqueActivationThreshold = 0.5;

/// Per-column decision taken by the clique pass. Columns left at kUndecided are
/// rounded to nearest afterwards.
constexpr int kUndecided = -1;

/// LP value of a literal: x_j for a plain literal, 1 - x_j for a complemented
/// one. A clique states that these values sum to at most one.
Real literalValue(const Literal& lit, std::span<const Real> primals) {
    return lit.complemented ? 1.0 - primals[lit.var] : primals[lit.var];
}

bool literalIsOne(const Literal& lit, const std::vector<int>& assign) {
    const int val = assign[lit.var];
    return lit.complemented ? (val == 0) : (val == 1);
}

void setLiteral(const Literal& lit, bool one, std::vector<int>& assign) {
    assign[lit.var] = (lit.complemented == one) ? 0 : 1;
}

/// Binaries whose bounds already pin them are decided before the clique pass,
/// so the pass never proposes a value the bounds forbid.
void assignBoundFixedBinaries(const LpProblem& problem, std::vector<int>& assign) {
    for (Index col = 0; col < problem.num_cols; ++col) {
        if (problem.col_type[col] != VarType::Binary) {
            continue;
        }
        if (problem.col_lower[col] > kCliqueActivationThreshold) {
            assign[col] = 1;
        } else if (problem.col_upper[col] < kCliqueActivationThreshold) {
            assign[col] = 0;
        }
    }
}

/// Decide one clique. Every member left undecided on entry is decided here, so
/// no clique member ever reaches the round-to-nearest fallback. Returns false
/// when two members are already pinned to 1 and no rounding can satisfy the
/// clique.
bool assignOneClique(const Clique& clique, std::span<const Real> primals,
                     std::vector<int>& assign) {
    Int ones = 0;
    for (const Literal& lit : clique.literals) {
        if (assign[lit.var] != kUndecided && literalIsOne(lit, assign)) {
            ++ones;
        }
    }
    if (ones >= 2) {
        return false;
    }

    if (ones == 1) {
        for (const Literal& lit : clique.literals) {
            if (assign[lit.var] == kUndecided) {
                setLiteral(lit, false, assign);
            }
        }
        return true;
    }

    // No member is 1 yet: the free literal closest to 1 in the LP is the only
    // candidate, everything else in the clique goes to 0.
    const Literal* best = nullptr;
    Real best_value = -kInf;
    for (const Literal& lit : clique.literals) {
        if (assign[lit.var] != kUndecided) {
            continue;
        }
        const Real lit_value = literalValue(lit, primals);
        if (lit_value > best_value) {
            best_value = lit_value;
            best = &lit;
        }
    }
    if (best == nullptr) {
        // Every member was already decided by an overlapping clique. For a
        // plain clique that is fine; an equality clique still needs exactly
        // one member at 1, and if none survived, this assignment cannot be
        // completed. Today isRowFeasible would also catch it, but only
        // because every equality clique currently comes from a real "= 1"
        // row -- do not depend on that.
        return !clique.is_equality;
    }

    // An equality clique needs exactly one member at 1; a plain clique only
    // takes the candidate when the LP already leans that way.
    const bool activate = clique.is_equality || best_value >= kCliqueActivationThreshold;
    for (const Literal& lit : clique.literals) {
        if (assign[lit.var] == kUndecided) {
            setLiteral(lit, activate && (&lit == best), assign);
        }
    }
    return true;
}

/// Turn a clique assignment into a full candidate point: decided columns take
/// their assigned value, undecided integers round to nearest inside the
/// integral hull of their bounds, continuous columns keep the LP value.
/// Returns false when some integer column has no integral value in its bounds.
bool buildCandidate(const LpProblem& problem, std::span<const Real> primals,
                    const std::vector<int>& assign, std::vector<Real>& values) {
    values.assign(primals.begin(), primals.begin() + problem.num_cols);
    for (Index col = 0; col < problem.num_cols; ++col) {
        if (assign[col] != kUndecided) {
            values[col] = static_cast<Real>(assign[col]);
            continue;
        }
        if (!isIntegerVar(problem.col_type[col])) {
            continue;
        }
        const Real lower = std::isfinite(problem.col_lower[col])
                               ? std::ceil(problem.col_lower[col] - kFeasTol)
                               : problem.col_lower[col];
        const Real upper = std::isfinite(problem.col_upper[col])
                               ? std::floor(problem.col_upper[col] + kFeasTol)
                               : problem.col_upper[col];
        if (lower > upper) {
            return false;
        }
        values[col] = clampToBounds(std::round(values[col]), lower, upper);
    }
    return true;
}

bool isBoundFeasibleAndIntegral(const LpProblem& problem, std::span<const Real> values) {
    for (Index col = 0; col < problem.num_cols; ++col) {
        if (values[col] < problem.col_lower[col] - kFeasTol) {
            return false;
        }
        if (values[col] > problem.col_upper[col] + kFeasTol) {
            return false;
        }
        if (isIntegerVar(problem.col_type[col]) && !isIntegral(values[col])) {
            return false;
        }
    }
    return true;
}

Index countBinaries(const LpProblem& problem) {
    Index count = 0;
    for (Index col = 0; col < problem.num_cols; ++col) {
        if (problem.col_type[col] == VarType::Binary) {
            ++count;
        }
    }
    return count;
}

}  // namespace

std::optional<HeuristicSolution> CliqueRoundingHeuristic::run(
    const LpProblem& problem, [[maybe_unused]] DualSimplexSolver& solver,
    std::span<const Real> primals, Real incumbent) {
    last_clique_count_ = 0;
    skipped_too_large_ = false;

    const Index num_cols = problem.num_cols;
    if (std::cmp_less(primals.size(), num_cols)) {
        return std::nullopt;
    }
    const Index num_binaries = countBinaries(problem);
    if (num_binaries < 2) {
        return std::nullopt;
    }
    // ConflictGraph::build is O(k^2) per row with a linear-scan dedup, and the
    // clique merge is quadratic in the clique count, so a large set-packing
    // model costs tens of seconds and hundreds of megabytes here. Cap it the
    // way the root clique build is capped rather than stalling the portfolio.
    if (num_binaries > max_binaries_) {
        skipped_too_large_ = true;
        return std::nullopt;
    }

    ConflictGraph graph;
    graph.build(problem);
    CliqueTable table;
    table.build(problem, graph);
    last_clique_count_ = static_cast<Int>(table.numCliques());
    if (table.numCliques() == 0) {
        return std::nullopt;
    }

    std::vector<int> assign(static_cast<std::size_t>(num_cols), kUndecided);
    assignBoundFixedBinaries(problem, assign);
    for (Index idx = 0; idx < table.numCliques(); ++idx) {
        if (!assignOneClique(table.clique(idx), primals, assign)) {
            return std::nullopt;
        }
    }

    std::vector<Real> values;
    if (!buildCandidate(problem, primals, assign, values)) {
        return std::nullopt;
    }
    if (!isBoundFeasibleAndIntegral(problem, values)) {
        return std::nullopt;
    }
    if (!isRowFeasible(problem, values)) {
        return std::nullopt;
    }

    const Real objective = computeObjective(problem, values);
    if (!betterObjective(problem.sense, objective, incumbent)) {
        return std::nullopt;
    }

    return HeuristicSolution{std::move(values), objective};
}

}  // namespace mipx
