#include "mipx/branching.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <optional>

namespace mipx {

namespace {

/// Integrality tolerance used when deciding which variables contribute to the
/// pseudocost node estimate. Matches the branching/LP integrality tolerance.
constexpr Real kPseudocostEstimateIntTol = 1e-6;
constexpr Real kNearTieRelativeGap = 0.01;

}  // namespace

Index MostFractionalBranching::select(std::span<const Real> primal_values,
                                      std::span<const VarType> var_types,
                                      std::span<const Real> col_lower,
                                      std::span<const Real> col_upper) const {
    Index best = -1;
    Real best_score = -1.0;  // Best = closest to 0.5.

    for (Index j = 0; j < static_cast<Index>(var_types.size()); ++j) {
        if (var_types[j] == VarType::Continuous)
            continue;
        // Skip fixed variables.
        if (col_lower[j] == col_upper[j])
            continue;

        Real val = primal_values[j];
        if (isIntegral(val))
            continue;

        Real frac = fractionality(val);
        // Score: closer to 0.5 is better. Use 0.5 - |frac - 0.5|.
        Real score = 0.5 - std::abs(frac - 0.5);
        if (score > best_score) {
            best_score = score;
            best = j;
        }
    }
    return best;
}

Index FirstFractionalBranching::select(std::span<const Real> primal_values,
                                       std::span<const VarType> var_types,
                                       std::span<const Real> col_lower,
                                       std::span<const Real> col_upper) const {
    for (Index j = 0; j < static_cast<Index>(var_types.size()); ++j) {
        if (var_types[j] == VarType::Continuous)
            continue;
        if (col_lower[j] == col_upper[j])
            continue;

        if (!isIntegral(primal_values[j])) {
            return j;
        }
    }
    return -1;
}

void ReliabilityBranching::reset(Index num_cols) {
    pseudocosts_.assign(static_cast<std::size_t>(std::max<Index>(0, num_cols)), PseudoCost{});
}

bool ReliabilityBranching::isReliable(Index var) const {
    if (!inRange(var))
        return false;
    const auto& pc = pseudocosts_[var];
    return pc.up_count >= reliability_threshold_ && pc.down_count >= reliability_threshold_;
}

Real ReliabilityBranching::upPseudoCost(Index var) const {
    if (!inRange(var))
        return pseudocost_fallback_;
    const auto& pc = pseudocosts_[var];
    if (pc.up_count <= 0)
        return pseudocost_fallback_;
    return std::max<Real>(pc.up, 1e-8);
}

Real ReliabilityBranching::downPseudoCost(Index var) const {
    if (!inRange(var))
        return pseudocost_fallback_;
    const auto& pc = pseudocosts_[var];
    if (pc.down_count <= 0)
        return pseudocost_fallback_;
    return std::max<Real>(pc.down, 1e-8);
}

Int ReliabilityBranching::upReliability(Index var) const {
    if (!inRange(var))
        return 0;
    return pseudocosts_[var].up_count;
}

Int ReliabilityBranching::downReliability(Index var) const {
    if (!inRange(var))
        return 0;
    return pseudocosts_[var].down_count;
}

void ReliabilityBranching::updatePseudoCost(Index var, bool up_direction, Real gain_per_unit) {
    if (!inRange(var) || !std::isfinite(gain_per_unit))
        return;
    gain_per_unit = std::max<Real>(gain_per_unit, 0.0);

    auto& pc = pseudocosts_[var];
    if (up_direction) {
        const Real weighted = pc.up * static_cast<Real>(pc.up_count);
        ++pc.up_count;
        pc.up = (weighted + gain_per_unit) / static_cast<Real>(pc.up_count);
    } else {
        const Real weighted = pc.down * static_cast<Real>(pc.down_count);
        ++pc.down_count;
        pc.down = (weighted + gain_per_unit) / static_cast<Real>(pc.down_count);
    }
}

std::optional<Real> TreePseudocostSample::gainPerUnit() const {
    if (status != Status::Optimal || cutoff || !std::isfinite(parent_objective) ||
        !std::isfinite(objective) || !std::isfinite(branch_value)) {
        return std::nullopt;
    }
    // is_upper means the child tightened x_j's upper bound: the down branch.
    const Real dist = branch.is_upper ? branch_value - branch.bound : branch.bound - branch_value;
    if (!std::isfinite(dist) || !(dist > 1e-8)) {
        return std::nullopt;
    }
    const Real gain = std::max<Real>(0.0, objective - parent_objective) / dist;
    return std::isfinite(gain) ? std::optional{gain} : std::nullopt;
}

bool ReliabilityBranching::updateFromChild(const TreePseudocostSample& sample,
                                           BranchingTelemetry& telemetry) {
    const Index var = sample.branch.variable;
    if (!inRange(var) || !std::isfinite(sample.parent_objective)) {
        return false;  // Root, or a node not made by createChildren().
    }
    if (sample.status != Status::Optimal || !std::isfinite(sample.objective) || sample.cutoff) {
        ++telemetry.pseudocost_tree_skipped;
        return false;
    }
    const auto gain = sample.gainPerUnit();
    if (!gain) {
        return false;
    }
    updatePseudoCost(var, !sample.branch.is_upper, *gain);
    ++telemetry.pseudocost_updates_tree;
    return true;
}

Real ReliabilityBranching::safeUpCost(Index var) const {
    return upPseudoCost(var);
}

Real ReliabilityBranching::safeDownCost(Index var) const {
    return downPseudoCost(var);
}

std::partial_ordering ReliabilityBranching::RankScore::operator<=>(const RankScore& other) const {
    if (exponent != other.exponent) {
        return exponent <=> other.exponent;
    }
    return mantissa <=> other.mantissa;
}

ReliabilityBranching::RankScore ReliabilityBranching::RankScore::weightedBy(Real factor) const {
    if (exponent == kScalarExponent) {
        return {mantissa * factor};
    }
    if (mantissa == 0.0) {
        return *this;
    }
    int shift = 0;
    const Real weighted = std::frexp(mantissa * factor, &shift);
    return {weighted, exponent + shift};
}

Real ReliabilityBranching::RankScore::log1p() const {
    if (exponent == kScalarExponent) {
        return std::log1p(std::max<Real>(0.0, mantissa));
    }
    if (mantissa == 0.0) {
        return 0.0;
    }
    if (exponent <= std::numeric_limits<Real>::max_exponent) {
        return std::log1p(std::ldexp(mantissa, exponent));
    }
    // log(1 + x) = log(x) + log(1 + 1/x), without representing x.
    return std::log(mantissa) + exponent * std::numbers::ln2_v<Real> +
           std::log1p(std::ldexp(1.0 / mantissa, -exponent));
}

Real ReliabilityBranching::RankScore::gapFrom(RankScore winner) const {
    if (*this == winner) {
        return 0.0;
    }
    const Real aligned = exponent == winner.exponent || mantissa == 0.0
                             ? mantissa
                             : std::ldexp(mantissa, exponent - winner.exponent);
    return (winner.mantissa - aligned) / winner.mantissa;
}

ReliabilityBranching::RankScore ReliabilityBranching::directionalScore(BranchScoreMode mode,
                                                                       Real down_gain,
                                                                       Real up_gain) {
    if (mode == BranchScoreMode::Product) {
        if (down_gain == 0.0 || up_gain == 0.0) {
            return {0.0, RankScore::kZeroProductExponent};
        }
        int down_exp = 0;
        int up_exp = 0;
        int shift = 0;
        const Real down = std::frexp(down_gain, &down_exp);
        const Real up = std::frexp(up_gain, &up_exp);
        const Real mantissa = std::frexp(down * up, &shift);
        return {mantissa, down_exp + up_exp + shift};
    }
    const Real larger_gain = std::max(down_gain, up_gain);
    if (mode == BranchScoreMode::Balanced) {
        return {0.8 * std::min(down_gain, up_gain) + 0.2 * larger_gain};
    }
    return {larger_gain};
}

void BranchInferenceCoverageStats::recordArbitration(const BranchingSelection& selection,
                                                     Index chosen, Index baseline_chosen) {
    if (selection.inference_baseline >= 0 && chosen != baseline_chosen) {
        ++(selection.used_reliable_pseudocost ? reliable : fallback).surviving_changes;
    }
}

void BranchCutoffComparisonStats::recordArbitration(const BranchingSelection& selection,
                                                    Index chosen, Index baseline_chosen) {
    if (selection.cutoff_baseline >= 0 && chosen != baseline_chosen) {
        ++(selection.used_reliable_pseudocost ? reliable : fallback).surviving_changes;
    }
}

void ReliabilityBranching::compareInference(const BranchInferenceCoverage& coverage,
                                            std::span<const Candidate> candidates,
                                            RankScore winner_score, BranchingSelection& selection) {
    if (!winner_score.isFinite()) {
        return;
    }
    const Index winner = selection.variable;
    const bool reliable = selection.used_reliable_pseudocost;
    auto& path = reliable ? coverage.stats.reliable : coverage.stats.fallback;
    ++path.decisions;
    const auto weakerMean = [&](Index variable) -> std::optional<Real> {
        const auto columns = coverage.original_columns;
        const auto& sample = coverage.history[columns.empty() ? variable : columns[variable]];
        if (sample.down.samples < 4 || sample.up.samples < 4) {
            return std::nullopt;
        }
        return std::min(
            static_cast<Real>(sample.down.tightened_columns) /
                static_cast<Real>(sample.down.samples),
            static_cast<Real>(sample.up.tightened_columns) / static_cast<Real>(sample.up.samples));
    };

    enum class Evidence { None, Close, Ready, Favored };
    std::array<Evidence, InferenceCoveragePath::kGapPercents.size()> evidence{};
    const auto winner_mean = weakerMean(winner);
    const Candidate* best = nullptr;
    Real best_mean = winner_mean.value_or(0.0);
    for (const auto& candidate : candidates) {
        if (candidate.var == winner || (reliable && !candidate.reliable) ||
            !candidate.pseudo_score.isFinite()) {
            continue;
        }
        const Real gap = candidate.pseudo_score.gapFrom(winner_score);
        if (gap > InferenceCoveragePath::kGapPercents.back() / 100.0) {
            continue;
        }
        Evidence observed = Evidence::Close;
        if (winner_mean) {
            if (const auto challenger_mean = weakerMean(candidate.var)) {
                observed = *challenger_mean > *winner_mean ? Evidence::Favored : Evidence::Ready;
                const bool best_tie = best != nullptr && *challenger_mean == best_mean &&
                                      candidate.winsHistoryTie(*best, false);
                if (coverage.select_by_inference && gap <= kNearTieRelativeGap &&
                    (*challenger_mean > best_mean || best_tie)) {
                    best = &candidate;
                    best_mean = *challenger_mean;
                }
            }
        }
        for (std::size_t i = 0; i < evidence.size(); ++i) {
            if (gap <= InferenceCoveragePath::kGapPercents[i] / 100.0) {
                evidence[i] = std::max(evidence[i], observed);
            }
        }
    }
    for (std::size_t i = 0; i < evidence.size(); ++i) {
        path.bands[i].close += evidence[i] >= Evidence::Close;
        path.bands[i].ready += evidence[i] >= Evidence::Ready;
        path.bands[i].favored += evidence[i] == Evidence::Favored;
    }
    if (best != nullptr) {
        selection.inference_baseline = winner;
        selection.variable = best->var;
        ++path.selection_changes;
    }
}

Index ReliabilityBranching::compareCutoffs(const BranchCutoffComparison& comparison,
                                           std::span<const Candidate> candidates,
                                           RankScore winner_score,
                                           const BranchingSelection& selection) {
    if (!winner_score.isFinite()) {
        return -1;
    }
    const Index winner = selection.variable;
    const bool reliable = selection.used_reliable_pseudocost;
    auto& path = reliable ? comparison.stats.reliable : comparison.stats.fallback;
    ++path.decisions;
    const auto weakerRate = [&](Index variable) -> std::optional<Real> {
        const auto columns = comparison.original_columns;
        if (!columns.empty() && static_cast<std::size_t>(variable) >= columns.size()) {
            return std::nullopt;
        }
        const Index original = columns.empty() ? variable : columns[variable];
        if (original < 0 || static_cast<std::size_t>(original) >= comparison.history.size()) {
            return std::nullopt;
        }
        const auto& sample = comparison.history[original];
        if (sample.down.qualified_attempts < 4 || sample.up.qualified_attempts < 4) {
            return std::nullopt;
        }
        const auto rate = [](const MipDirectionalCutoffStats& direction) {
            return (static_cast<Real>(direction.cutoffs) + 1.0) /
                   (static_cast<Real>(direction.qualified_attempts) + 2.0);
        };
        return std::min(rate(sample.down), rate(sample.up));
    };

    const auto winner_rate = weakerRate(winner);
    if (!winner_rate) {
        return -1;
    }
    const Candidate* best = nullptr;
    Real best_rate = *winner_rate;
    const RankScore minimum_score = winner_score.weightedBy(1.0 - kNearTieRelativeGap);
    for (const auto& candidate : candidates) {
        if (candidate.var == winner || (reliable && !candidate.reliable) ||
            !candidate.pseudo_score.isFinite() || candidate.pseudo_score < minimum_score) {
            continue;
        }
        const auto rate = weakerRate(candidate.var);
        if (!rate) {
            continue;
        }
        ++path.ready_pairs;
        const bool best_tie =
            best != nullptr && *rate == best_rate && candidate.winsHistoryTie(*best, true);
        if (*rate > best_rate || best_tie) {
            best = &candidate;
            best_rate = *rate;
        }
    }
    path.favored_decisions += best != nullptr;
    return best == nullptr ? -1 : best->var;
}

BranchingSelection ReliabilityBranching::select(DualSimplexSolver& lp, const LpProblem& problem,
                                                std::span<const Real> primal_values,
                                                std::span<const Real> col_lower,
                                                std::span<const Real> col_upper,
                                                Real node_objective, bool force_strong_branch,
                                                BranchingTelemetry& telemetry) {
    return selectWithConflictActivity(lp, problem, primal_values, col_lower, col_upper,
                                      node_objective, force_strong_branch, telemetry, {});
}

BranchingSelection ReliabilityBranching::selectWithConflictActivity(
    DualSimplexSolver& lp, const LpProblem& problem, std::span<const Real> primal_values,
    std::span<const Real> col_lower, std::span<const Real> col_upper, Real node_objective,
    bool force_strong_branch, BranchingTelemetry& telemetry,
    std::span<const Real> conflict_activity) {
    return selectWithInferenceCoverage(lp, problem, primal_values, col_lower, col_upper,
                                       node_objective, force_strong_branch, telemetry,
                                       conflict_activity, nullptr);
}

BranchingSelection ReliabilityBranching::selectWithInferenceCoverage(
    DualSimplexSolver& lp, const LpProblem& problem, std::span<const Real> primal_values,
    std::span<const Real> col_lower, std::span<const Real> col_upper, Real node_objective,
    bool force_strong_branch, BranchingTelemetry& telemetry,
    std::span<const Real> conflict_activity, const BranchInferenceCoverage* coverage) {
    return selectWithHistories(lp, problem, primal_values, col_lower, col_upper, node_objective,
                               force_strong_branch, telemetry, conflict_activity, {coverage});
}

BranchingSelection ReliabilityBranching::selectWithHistories(
    DualSimplexSolver& lp, const LpProblem& problem, std::span<const Real> primal_values,
    std::span<const Real> col_lower, std::span<const Real> col_upper, Real node_objective,
    bool force_strong_branch, BranchingTelemetry& telemetry,
    std::span<const Real> conflict_activity, BranchSelectionHistories histories) {
    return selectWithProbeSink(
        lp, problem, primal_values, col_lower, col_upper, node_objective, force_strong_branch,
        strong_branch_probe_budget_, telemetry, conflict_activity, histories,
        [&](const StrongBranchPseudocostSample& sample) {
            updatePseudoCost(sample.variable, sample.up_direction, sample.gain_per_unit);
        });
}

BranchingSelection ReliabilityBranching::selectDeferred(
    DualSimplexSolver& lp, const LpProblem& problem, std::span<const Real> primal_values,
    std::span<const Real> col_lower, std::span<const Real> col_upper, Real node_objective,
    Int probe_budget, BranchingTelemetry& telemetry,
    std::vector<StrongBranchPseudocostSample>& samples) const {
    return selectWithProbeSink(
        lp, problem, primal_values, col_lower, col_upper, node_objective, false, probe_budget,
        telemetry, {}, {},
        [&](const StrongBranchPseudocostSample& sample) { samples.push_back(sample); });
}

void BranchDualDegeneracyStats::record(std::optional<LpDualDegeneracy> measurement) {
    if (!measurement || measurement->eligible == 0) {
        ++unavailable;
        return;
    }
    const auto columns = static_cast<std::uint64_t>(measurement->eligible);
    const auto zeros = static_cast<std::uint64_t>(measurement->zero_reduced_cost);
    eligible += columns;
    zero_reduced_cost += zeros;
    ++histogram[10 * zeros / columns];
}

template <typename RecordProbe>
BranchingSelection ReliabilityBranching::selectWithProbeSink(
    DualSimplexSolver& lp, const LpProblem& problem, std::span<const Real> primal_values,
    std::span<const Real> col_lower, std::span<const Real> col_upper, Real node_objective,
    bool force_strong_branch, Int probe_budget, BranchingTelemetry& telemetry,
    std::span<const Real> conflict_activity, BranchSelectionHistories histories,
    RecordProbe record_probe) const {
    BranchingSelection selection;
    ++telemetry.selections;

    std::vector<Candidate> candidates;
    candidates.reserve(static_cast<std::size_t>(problem.num_cols));
    const bool score_conflicts =
        conflict_activity.size() == static_cast<std::size_t>(problem.num_cols);

    for (Index j = 0; j < problem.num_cols; ++j) {
        if (problem.col_type[j] == VarType::Continuous)
            continue;
        if (col_lower[j] >= col_upper[j] - 1e-12)
            continue;

        const Real value = primal_values[j];
        if (isIntegral(value))
            continue;

        const Real floor_v = std::floor(value);
        const Real ceil_v = std::ceil(value);
        const Real down_dist = std::max<Real>(0.0, value - floor_v);
        const Real up_dist = std::max<Real>(0.0, ceil_v - value);
        if (down_dist <= 1e-8 || up_dist <= 1e-8)
            continue;

        Candidate c;
        c.var = j;
        c.value = value;
        c.down_dist = down_dist;
        c.up_dist = up_dist;
        c.reliable = isReliable(j);
        c.down_gain = down_dist * safeDownCost(j);
        c.up_gain = up_dist * safeUpCost(j);
        candidates.push_back(c);
    }

    if (candidates.empty())
        return selection;

    if (histories.dual_degeneracy != nullptr) {
        histories.dual_degeneracy->record(lp.measureDualDegeneracy());
    }

    if (symmetry_manager_ != nullptr) {
        std::vector<Candidate> filtered;
        filtered.reserve(candidates.size());
        for (const auto& c : candidates) {
            if (isCanonicalCandidate(c.var, primal_values)) {
                filtered.push_back(c);
            }
        }
        if (!filtered.empty()) {
            candidates.swap(filtered);
        }
    }

    auto mode = score_mode_;
    if (mode == BranchScoreMode::Product && std::ranges::any_of(candidates, [](const Candidate& c) {
            return !std::isfinite(c.down_gain) || !std::isfinite(c.up_gain);
        })) {
        mode = BranchScoreMode::LargerGain;
    }

    const auto makeShortlist = [&] {
        for (auto& c : candidates) {
            c.objective_score = directionalScore(mode, c.down_gain, c.up_gain);
            c.pseudo_score = c.objective_score;
            if (score_conflicts) {
                // Bound the conflict boost so LP gains remain relevant.
                const Real activity = std::max<Real>(0.0, conflict_activity[c.var]);
                c.pseudo_score = c.pseudo_score.weightedBy(2.0 - 1.0 / (1.0 + activity));
            }
            c.prefilter_score =
                0.5 - std::abs(fractionality(c.value) - 0.5) + 0.15 * c.pseudo_score.log1p();
        }
        std::ranges::sort(candidates, [](const Candidate& a, const Candidate& b) {
            if (a.prefilter_score != b.prefilter_score) {
                return a.prefilter_score > b.prefilter_score;
            }
            return a.var < b.var;
        });
        return std::span<const Candidate>(candidates)
            .first(std::min(candidates.size(),
                            static_cast<std::size_t>(strong_branch_max_candidates_)));
    };
    auto shortlist = makeShortlist();
    const auto bestReliable = [&]() -> const Candidate* {
        const Candidate* best = nullptr;
        for (const auto& c : shortlist) {
            const RankScore best_score =
                best ? best->pseudo_score : RankScore{-kInf, c.pseudo_score.exponent};
            if (c.reliable && c.pseudo_score > best_score) {
                best = &c;
            }
        }
        return best;
    };

    const bool may_retry = mode == BranchScoreMode::Product && probe_budget >= 2;
    const Candidate* best_reliable = bestReliable();

    struct CachedProbe {
        Index variable;
        bool up;
        LpResult result;
    };
    std::vector<CachedProbe> cached_probes;
    std::vector<StrongBranchPseudocostSample> pending_samples;
    Int fresh_probes_remaining = probe_budget;
    Int distinct_probed = 0;
    bool started_probing = false;
    const auto strongWinner = [&]() -> std::optional<Index> {
        if ((!force_strong_branch && best_reliable != nullptr) || probe_budget < 2) {
            return -1;
        }
        telemetry.strong_branch_calls += !started_probing;
        started_probing = true;

        const auto saved_basis = lp.getBasis();
        const Int old_limit = lp.getIterationLimit();
        lp.setIterationLimit(strong_branch_iter_limit_);

        const auto cachedResult = [&](Index variable, bool up) -> const LpResult* {
            const auto cached = std::ranges::find_if(cached_probes, [&](const CachedProbe& p) {
                return p.variable == variable && p.up == up;
            });
            return cached == cached_probes.end() ? nullptr : &cached->result;
        };
        const auto probeDirection = [&](const Candidate& c, bool up, Real lower, Real upper,
                                        const LpResult* cached) {
            if (cached != nullptr) {
                return *cached;
            }
            --fresh_probes_remaining;
            lp.setBasis(saved_basis);
            lp.setColBounds(c.var, lower, upper);
            const auto probe = lp.solve();
            ++telemetry.strong_branch_probes;
            telemetry.strong_branch_probe_iters += probe.iterations;
            telemetry.strong_branch_probe_work_units += probe.work_units;
            lp.setColBounds(c.var, col_lower[c.var], col_upper[c.var]);
            if (mode == BranchScoreMode::Product) {
                cached_probes.push_back({c.var, up, probe});
            }
            return probe;
        };
        const auto probeGain = [&](const LpResult& probe, Real predicted) {
            if (probe.status == Status::Infeasible) {
                return kStrongInfeasibleGain;
            }
            // Keep iteration-limited probes eligible using their partial
            // gain, as in the original reliability rule (#222).
            if (probe.status == Status::Optimal || probe.status == Status::IterLimit) {
                const Real gain = probe.objective - node_objective;
                return mode == BranchScoreMode::Product && !std::isfinite(gain)
                           ? gain
                           : std::max<Real>(0.0, gain);
            }
            return predicted;
        };
        const bool replay = may_retry && mode != BranchScoreMode::Product;
        const auto record = [&](const Candidate& c, bool up, const LpResult& probe, Real gain) {
            const StrongBranchPseudocostSample sample{c.var, up,
                                                      gain / (up ? c.up_dist : c.down_dist)};
            if (score_mode_ != BranchScoreMode::Product) {
                record_probe(sample);
                ++telemetry.pseudocost_updates_strong;
                return;
            }
            const bool observed = probe.status == Status::Infeasible ||
                                  (probe.status == Status::Optimal &&
                                   std::isfinite(probe.objective - node_objective));
            if (observed && std::isfinite(sample.gain_per_unit) &&
                !(replay && std::ranges::any_of(pending_samples, [&](const auto& previous) {
                      return previous.variable == c.var && previous.up_direction == up;
                  }))) {
                pending_samples.push_back(sample);
            }
        };

        std::optional<Index> winner = -1;
        std::optional<RankScore> best_score;
        const auto consider = [&](Index variable, RankScore score) {
            if (score > best_score.value_or(RankScore{-kInf, score.exponent})) {
                best_score = score;
                winner = variable;
            }
        };
        Int remaining = probe_budget;
        for (const auto& c : shortlist) {
            if (remaining < 2)
                break;
            if (!force_strong_branch && c.reliable)
                continue;

            const Real orig_lb = col_lower[c.var];
            const Real orig_ub = col_upper[c.var];
            const Real down_ub = std::floor(c.value);
            const Real up_lb = std::ceil(c.value);
            const bool solve_down = down_ub >= orig_lb - 1e-12;
            const bool solve_up = up_lb <= orig_ub + 1e-12;
            const LpResult* cached_down = replay ? cachedResult(c.var, false) : nullptr;
            const LpResult* cached_up = replay ? cachedResult(c.var, true) : nullptr;
            const Int fresh_needed =
                (solve_down && cached_down == nullptr) + (solve_up && cached_up == nullptr);
            const bool new_candidate =
                fresh_needed > 0 && cached_down == nullptr && cached_up == nullptr;
            if (replay && (fresh_needed > fresh_probes_remaining ||
                           (new_candidate && distinct_probed >= strong_branch_max_candidates_))) {
                continue;
            }
            distinct_probed += new_candidate;

            const auto down = solve_down ? probeDirection(c, false, orig_lb, down_ub, cached_down)
                                         : LpResult{Status::Infeasible};
            if (down.status == Status::TimeLimit) {
                break;
            }
            const Real down_gain = probeGain(down, c.down_gain);
            if (mode == BranchScoreMode::Product && !std::isfinite(down_gain)) {
                winner.reset();
                break;
            }
            record(c, false, down, down_gain);
            --remaining;

            const auto up = solve_up ? probeDirection(c, true, up_lb, orig_ub, cached_up)
                                     : LpResult{Status::Infeasible};
            if (up.status == Status::TimeLimit) {
                // The down sample stands. The unobserved up gain is zero
                // when an already strong down gain makes this pair eligible.
                // In product mode this competes as a zero product.
                if (down_gain >= kStrongInfeasibleGain) {
                    consider(c.var, directionalScore(mode, down_gain, 0.0));
                }
                break;
            }
            const Real up_gain = probeGain(up, c.up_gain);
            if (mode == BranchScoreMode::Product && !std::isfinite(up_gain)) {
                winner.reset();
                break;
            }
            record(c, true, up, up_gain);
            --remaining;
            consider(c.var, directionalScore(mode, down_gain, up_gain));
        }

        lp.setIterationLimit(old_limit);
        lp.setBasis(saved_basis);
        return winner;
    };

    auto best_strong = strongWinner();
    if (!best_strong) {
        // Rebuild the shortlist before ranking. Reuse matching outcomes
        // and share the physical solve and candidate caps across both passes.
        mode = BranchScoreMode::LargerGain;
        shortlist = makeShortlist();
        best_reliable = bestReliable();
        best_strong = strongWinner();
    }
    // The ranking can discard a candidate, but a completed direction still
    // supplies one valid observation from the LP that was actually solved.
    for (const auto& sample : pending_samples) {
        record_probe(sample);
        ++telemetry.pseudocost_updates_strong;
    }
    if (best_strong.value_or(-1) >= 0) {
        selection.variable = *best_strong;
        return selection;
    }

    const auto observeHistories = [&](RankScore winner_score) {
        const auto* cutoff = histories.cutoff;
        const bool score_cutoffs = cutoff != nullptr && cutoff->select_by_cutoff;
        const Index cutoff_candidate =
            cutoff != nullptr ? compareCutoffs(*cutoff, shortlist, winner_score, selection) : -1;
        if (histories.inference != nullptr) {
            auto inference = *histories.inference;
            inference.select_by_inference &= !score_cutoffs;
            compareInference(inference, shortlist, winner_score, selection);
        }
        if (score_cutoffs && cutoff_candidate >= 0) {
            selection.cutoff_baseline = selection.variable;
            selection.variable = cutoff_candidate;
            auto& stats = cutoff->stats;
            ++(selection.used_reliable_pseudocost ? stats.reliable : stats.fallback)
                  .selection_changes;
        }
    };
    if (best_reliable != nullptr) {
        selection.variable = best_reliable->var;
        selection.used_pseudocost = true;
        selection.used_reliable_pseudocost = true;
        observeHistories(best_reliable->pseudo_score);
        ++telemetry.pseudocost_uses;
        ++telemetry.pseudocost_hits;
        return selection;
    }

    const auto best_it = std::max_element(shortlist.begin(), shortlist.end(),
                                          [](const Candidate& a, const Candidate& b) {
                                              if (a.pseudo_score != b.pseudo_score) {
                                                  return a.pseudo_score < b.pseudo_score;
                                              }
                                              return a.var > b.var;
                                          });

    selection.variable = (best_it == shortlist.end()) ? -1 : best_it->var;
    if (selection.variable >= 0) {
        selection.used_pseudocost = true;
        observeHistories(best_it->pseudo_score);
        ++telemetry.pseudocost_uses;
    }
    return selection;
}

Real pseudocostNodeEstimate(const ReliabilityBranching& rule, const LpProblem& problem,
                            std::span<const Real> primal_values, Real lp_bound) {
    if (!std::isfinite(lp_bound)) {
        return lp_bound;
    }
    // Bound by col_type too: this is a public free function, so callers are
    // not guaranteed to hold MipSolver's num_cols == col_type.size() invariant.
    const Index n = std::min<Index>({problem.num_cols, static_cast<Index>(primal_values.size()),
                                     static_cast<Index>(problem.col_type.size())});
    Real penalty = 0.0;
    for (Index j = 0; j < n; ++j) {
        if (problem.col_type[j] == VarType::Continuous) {
            continue;
        }
        const Real value = primal_values[j];
        if (!std::isfinite(value)) {
            continue;
        }
        const Real frac = value - std::floor(value);
        if (frac <= kPseudocostEstimateIntTol || frac >= 1.0 - kPseudocostEstimateIntTol) {
            continue;
        }
        const Real down_gain = frac * rule.downPseudoCost(j);
        const Real up_gain = (1.0 - frac) * rule.upPseudoCost(j);
        const Real gain = std::min(down_gain, up_gain);
        if (std::isfinite(gain) && gain > 0.0) {
            penalty += gain;
        }
    }
    return lp_bound + penalty;
}

namespace {

std::pair<BnbNode, BnbNode> initializeChildren(BnbNode parent) {
    BnbNode left;
    left.parent_id = parent.id;
    left.depth = parent.depth + 1;
    left.basis = parent.basis;
    left.estimate = parent.estimate;
    left.basis_rows = parent.basis_rows;
    left.parent_objective = parent.lp_bound;
    left.bound_changes = parent.bound_changes;
    left.local_cuts = parent.local_cuts;

    BnbNode right;
    right.parent_id = parent.id;
    right.depth = parent.depth + 1;
    right.basis = std::move(parent.basis);
    right.estimate = parent.estimate;
    right.basis_rows = parent.basis_rows;
    right.parent_objective = parent.lp_bound;
    right.bound_changes = std::move(parent.bound_changes);
    right.local_cuts = std::move(parent.local_cuts);

    return {std::move(left), std::move(right)};
}

}  // namespace

std::pair<BnbNode, BnbNode> createChildren(BnbNode parent, Index branch_var, Real branch_val) {
    auto [left, right] = initializeChildren(std::move(parent));
    left.branch = {branch_var, std::floor(branch_val), true, left.depth};
    right.branch = {branch_var, std::ceil(branch_val), false, right.depth};
    left.branch_value = right.branch_value = branch_val;
    left.bound_changes.push_back(left.branch);
    right.bound_changes.push_back(right.branch);
    return {std::move(left), std::move(right)};
}

std::vector<BnbNode> createSos2Children(BnbNode parent, std::span<const Index> members,
                                        std::size_t split, std::span<const Real> lower,
                                        std::span<const Real> upper) {
    const Real bound = parent.lp_bound;
    auto [left, right] = initializeChildren(std::move(parent));
    std::vector<BnbNode> children;
    children.reserve(2);
    const auto addSide = [&](BnbNode& child, std::span<const Index> excluded) {
        for (Index j : excluded) {
            if (lower[j] > 0.0 || upper[j] < 0.0) {
                return;
            }
            // Record both ends even when the node already has a zero bound:
            // temporary local tightenings need not carry over to its children.
            child.bound_changes.push_back({j, 0.0, true, child.depth});
            child.bound_changes.push_back({j, 0.0, false, child.depth});
        }
        child.lp_bound = bound;
        children.push_back(std::move(child));
    };
    addSide(left, members.subspan(split + 1));
    addSide(right, members.first(split));
    return children;
}

}  // namespace mipx
