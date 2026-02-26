#include "mipx/heuristic_runtime.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>

namespace mipx {

namespace {

constexpr Real kObjectiveTol = 1e-6;

constexpr std::array<RestartStrategy, 6> kRestartStrategies = {
    RestartStrategy::Uniform,
    RestartStrategy::TournamentCrossover,
    RestartStrategy::Perturbation,
    RestartStrategy::PolarityGuided,
    RestartStrategy::ActivityGuided,
    RestartStrategy::DistanceGuided,
};

}  // namespace

bool SolutionPool::better(Sense sense, Real lhs, Real rhs) {
    if (sense == Sense::Minimize) {
        return lhs < rhs - kObjectiveTol;
    }
    return lhs > rhs + kObjectiveTol;
}

bool SolutionPool::submit(HeuristicSolution solution,
                          const char* source,
                          Int thread_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!best_ || better(sense_, solution.objective, best_->solution.objective)) {
        best_ = Entry{std::move(solution), source != nullptr ? source : "", thread_id};
        ++update_count_;
        return true;
    }
    return false;
}

std::optional<HeuristicSolution> SolutionPool::bestSolution() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!best_) return std::nullopt;
    return best_->solution;
}

bool SolutionPool::hasSolution() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return best_.has_value();
}

Real SolutionPool::bestObjective() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!best_) {
        return (sense_ == Sense::Minimize) ? kInf : -kInf;
    }
    return best_->solution.objective;
}

uint64_t SolutionPool::updateCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return update_count_;
}

uint64_t RestartStrategyEngine::splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27U)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31U);
}

RestartStrategy RestartStrategyEngine::choose(HeuristicRuntimeMode mode,
                                              Int epoch,
                                              Int stagnation_epochs) const {
    const uint64_t size = static_cast<uint64_t>(kRestartStrategies.size());
    if (mode == HeuristicRuntimeMode::Deterministic) {
        const uint64_t idx = static_cast<uint64_t>(
            std::max<Int>(0, epoch + stagnation_epochs)) % size;
        return kRestartStrategies[idx];
    }

    const uint64_t h = splitmix64(
        seed_ ^
        (static_cast<uint64_t>(std::max<Int>(0, epoch)) * 0x9e3779b97f4a7c15ULL) ^
        (static_cast<uint64_t>(std::max<Int>(0, stagnation_epochs)) * 0xbf58476d1ce4e5b9ULL));
    return kRestartStrategies[h % size];
}

const char* RestartStrategyEngine::name(RestartStrategy s) {
    switch (s) {
        case RestartStrategy::Uniform: return "uniform";
        case RestartStrategy::TournamentCrossover: return "tournament";
        case RestartStrategy::Perturbation: return "perturbation";
        case RestartStrategy::PolarityGuided: return "polarity";
        case RestartStrategy::ActivityGuided: return "activity";
        case RestartStrategy::DistanceGuided: return "distance";
        default: return "uniform";
    }
}

HeuristicRuntime::HeuristicRuntime(const HeuristicRuntimeConfig& config)
    : config_(config), restart_engine_(config.seed) {
    resetForSolve();
}

void HeuristicRuntime::resetForSolve() {
    budget_.setBaseTreeFrequency(config_.rins_node_frequency);
    budget_.setMaxFrequencyScale(config_.budget_max_frequency_scale);
    budget_.setMaxWorkShare(config_.budget_max_work_share);
    restart_engine_.setSeed(config_.seed);
    stats_ = {};
    stagnation_epochs_ = 0;
    emitted_root_header_ = false;
    emitted_worker_header_ = false;
}

void HeuristicRuntime::finish() {
    if (callback_ != nullptr) callback_->onFinish(stats_);
}

bool HeuristicRuntime::hasIncumbent(Real incumbent) {
    return std::isfinite(incumbent) && incumbent < kInf;
}

Real HeuristicRuntime::relativeGap(Real incumbent, Real bound) {
    if (!std::isfinite(incumbent) || !std::isfinite(bound)) return kInf;
    if (std::abs(incumbent) < 1e-10) {
        return std::abs(incumbent - bound);
    }
    return std::abs(incumbent - bound) / std::max(1.0, std::abs(incumbent));
}

bool HeuristicRuntime::isImprovement(Sense sense, Real candidate, Real incumbent) {
    if (!hasIncumbent(incumbent)) return true;
    if (sense == Sense::Minimize) {
        return candidate < incumbent - kObjectiveTol;
    }
    return candidate > incumbent + kObjectiveTol;
}

bool HeuristicRuntime::allowRootCall(double total_work_units) const {
    if (config_.mode == HeuristicRuntimeMode::Opportunistic) {
        return true;
    }
    return budget_.allowRootHeuristic(total_work_units);
}

bool HeuristicRuntime::allowTreeCall(Int node_count, double total_work_units) const {
    if (config_.mode == HeuristicRuntimeMode::Deterministic) {
        return budget_.allowTreeHeuristic(node_count, total_work_units);
    }
    if (!budget_.allowRootHeuristic(total_work_units)) return false;
    const Int freq = std::max<Int>(1, config_.rins_node_frequency / 2);
    return node_count % freq == 0;
}

void HeuristicRuntime::recordCall(Int node_count, double work_units, bool improved) {
    budget_.recordHeuristicCall(node_count, work_units, improved);
    ++stats_.calls;
    if (improved) {
        ++stats_.improvements;
        stagnation_epochs_ = 0;
    } else {
        ++stagnation_epochs_;
    }
    stats_.work_units += std::max(0.0, work_units);
}

RootHeuristicOutcome HeuristicRuntime::runRootPortfolio(
    const RootHeuristicContext& ctx,
    Real& incumbent,
    std::vector<Real>& best_solution) {
    if (!emitted_root_header_ && callback_ != nullptr) {
        callback_->onHeader("root", config_.mode);
        emitted_root_header_ = true;
    }

    RootHeuristicOutcome out;

    auto accept = [&](const char* source,
                      std::optional<HeuristicSolution>& candidate) -> bool {
        if (!candidate.has_value()) return false;
        if (!isImprovement(ctx.problem.sense, candidate->objective, incumbent)) return false;
        incumbent = candidate->objective;
        best_solution = std::move(candidate->values);
        ++out.improvements;
        if (ctx.solution_pool != nullptr) {
            ctx.solution_pool->submit({best_solution, incumbent}, source, ctx.thread_id);
        }
        if (callback_ != nullptr) {
            callback_->onIncumbent(source, incumbent, ctx.node_count, ctx.thread_id);
        }
        return true;
    };

    const bool run_root_lp_heuristics =
        (ctx.root_int_inf > 0) &&
        (ctx.root_int_inf <= config_.root_max_int_inf) &&
        (ctx.root_int_vars <= config_.root_max_int_vars);

    {
        ++out.calls;
        ++stats_.calls;
        RoundingHeuristic rounding;
        auto hsol = rounding.run(ctx.problem, ctx.lp, ctx.primals, incumbent);
        if (accept("rounding", hsol)) {
            ++stats_.improvements;
        }
    }
    if (callback_ != nullptr) callback_->onHeartbeat(ctx.node_count, stats_);

    if (run_root_lp_heuristics && incumbent == kInf &&
        allowRootCall(ctx.total_work_units + out.work_units)) {
        ++out.calls;
        AuxObjectiveHeuristic auxobj;
        auxobj.setSubproblemIterLimit(config_.root_auxobj_subproblem_iter_limit);
        auxobj.setMinActiveIntegerVars(config_.root_auxobj_min_active_integer_vars);
        auto hsol = auxobj.run(ctx.problem, ctx.lp, ctx.primals, incumbent);
        out.basis_dirty = out.basis_dirty || auxobj.lastExecutedSolve();
        out.lp_iterations += auxobj.lastLpIterations();
        out.work_units += auxobj.lastWorkUnits();
        stats_.lp_iterations += auxobj.lastLpIterations();
        const bool improved = accept("auxobj", hsol);
        recordCall(0, auxobj.lastWorkUnits(), improved);
        if (callback_ != nullptr) callback_->onHeartbeat(ctx.node_count, stats_);
    }

    if (run_root_lp_heuristics && incumbent == kInf &&
        allowRootCall(ctx.total_work_units + out.work_units)) {
        ++out.calls;
        ZeroObjectiveHeuristic zeroobj;
        zeroobj.setSubproblemIterLimit(config_.root_zeroobj_subproblem_iter_limit);
        auto hsol = zeroobj.run(ctx.problem, ctx.lp, ctx.primals, incumbent);
        out.basis_dirty = out.basis_dirty || zeroobj.lastExecutedSolve();
        out.lp_iterations += zeroobj.lastLpIterations();
        out.work_units += zeroobj.lastWorkUnits();
        stats_.lp_iterations += zeroobj.lastLpIterations();
        const bool improved = accept("zeroobj", hsol);
        recordCall(0, zeroobj.lastWorkUnits(), improved);
        if (callback_ != nullptr) callback_->onHeartbeat(ctx.node_count, stats_);
    }

    if (run_root_lp_heuristics && incumbent == kInf &&
        allowRootCall(ctx.total_work_units + out.work_units)) {
        ++out.calls;
        FeasibilityPumpHeuristic feaspump;
        feaspump.setMaxIterations(config_.root_feaspump_max_iter);
        feaspump.setSubproblemIterLimit(config_.root_feaspump_subproblem_iter_limit);
        auto hsol = feaspump.run(ctx.problem, ctx.lp, ctx.primals, incumbent);
        out.basis_dirty = true;
        out.lp_iterations += feaspump.lastLpIterations();
        out.work_units += feaspump.lastWorkUnits();
        stats_.lp_iterations += feaspump.lastLpIterations();
        const bool improved = accept("feaspump", hsol);
        recordCall(0, feaspump.lastWorkUnits(), improved);
        if (callback_ != nullptr) callback_->onHeartbeat(ctx.node_count, stats_);
    }

    if (run_root_lp_heuristics && incumbent == kInf &&
        allowRootCall(ctx.total_work_units + out.work_units)) {
        ++out.calls;
        RensHeuristic rens;
        rens.setSubproblemIterLimit(config_.root_rens_subproblem_iter_limit);
        rens.setMinFixedVars(config_.root_rens_min_fixed_vars);
        rens.setMinFixedRate(config_.root_rens_min_fixed_rate);
        auto hsol = rens.run(ctx.problem, ctx.lp, ctx.primals, incumbent);
        out.basis_dirty = true;
        out.lp_iterations += rens.lastLpIterations();
        out.work_units += rens.lastWorkUnits();
        stats_.lp_iterations += rens.lastLpIterations();
        const bool improved = accept("rens", hsol);
        recordCall(0, rens.lastWorkUnits(), improved);
        if (callback_ != nullptr) callback_->onHeartbeat(ctx.node_count, stats_);
    }

    if (run_root_lp_heuristics &&
        hasIncumbent(incumbent) && !best_solution.empty() &&
        ctx.root_int_inf > 0 && ctx.root_int_inf <= config_.rins_max_int_inf_for_run &&
        allowRootCall(ctx.total_work_units + out.work_units)) {
        ++out.calls;
        RinsHeuristic rins;
        rins.setSubproblemIterLimit(config_.rins_subproblem_iter_limit);
        rins.setAgreementTol(config_.rins_agreement_tol);
        rins.setMinFixedVars(config_.rins_min_fixed_vars);
        rins.setMinFixedRate(config_.rins_min_fixed_rate);
        auto hsol = rins.run(ctx.problem, ctx.lp, ctx.primals, incumbent, best_solution);
        out.basis_dirty = true;
        out.lp_iterations += rins.lastLpIterations();
        out.work_units += rins.lastWorkUnits();
        stats_.lp_iterations += rins.lastLpIterations();
        const bool improved = accept("rins", hsol);
        recordCall(0, rins.lastWorkUnits(), improved);
        if (callback_ != nullptr) callback_->onHeartbeat(ctx.node_count, stats_);
    }

    if (run_root_lp_heuristics &&
        hasIncumbent(incumbent) && !best_solution.empty() &&
        ctx.root_int_inf > 0 &&
        allowRootCall(ctx.total_work_units + out.work_units)) {
        constexpr std::array<Int, 3> kNeighborhoods = {0, 1, 2};
        for (Int k : kNeighborhoods) {
            if (!allowRootCall(ctx.total_work_units + out.work_units)) break;
            ++out.calls;

            Int radius = config_.root_local_branching_neighborhood_small;
            if (k == 1) radius = config_.root_local_branching_neighborhood_medium;
            if (k == 2) radius = config_.root_local_branching_neighborhood_large;

            LocalBranchingHeuristic local_branching;
            local_branching.setSubproblemIterLimit(
                config_.root_local_branching_subproblem_iter_limit);
            local_branching.setNeighborhoodSize(radius);
            local_branching.setMinBinaryVars(config_.root_local_branching_min_binary_vars);
            auto hsol = local_branching.run(ctx.problem, ctx.lp, ctx.primals,
                                            incumbent, best_solution);
            out.basis_dirty = true;
            out.lp_iterations += local_branching.lastLpIterations();
            out.work_units += local_branching.lastWorkUnits();
            stats_.lp_iterations += local_branching.lastLpIterations();
            const bool improved = accept("localbranching", hsol);
            recordCall(0, local_branching.lastWorkUnits(), improved);
            if (callback_ != nullptr) callback_->onHeartbeat(ctx.node_count, stats_);
        }
    }

    return out;
}

WorkerHeuristicOutcome HeuristicRuntime::runTreeWorker(
    const WorkerHeuristicContext& ctx) {
    if (!emitted_worker_header_ && callback_ != nullptr) {
        callback_->onHeader("worker", config_.mode);
        emitted_worker_header_ = true;
    }

    WorkerHeuristicOutcome out;

    if (ctx.int_inf <= 0 || ctx.int_inf > config_.rins_max_int_inf_for_run) {
        return out;
    }
    if (!hasIncumbent(ctx.incumbent)) {
        return out;
    }
    if (relativeGap(ctx.incumbent, ctx.node_objective) >
        config_.rins_max_relative_gap_for_run) {
        return out;
    }
    if (static_cast<Index>(ctx.incumbent_values.size()) != ctx.problem.num_cols) {
        out.skipped_no_incumbent = true;
        return out;
    }
    if (!allowTreeCall(ctx.node_count, ctx.total_work_units)) {
        return out;
    }

    out.attempted = true;

    RinsHeuristic rins;
    rins.setSubproblemIterLimit(config_.rins_subproblem_iter_limit);
    rins.setAgreementTol(config_.rins_agreement_tol);
    rins.setMinFixedVars(config_.rins_min_fixed_vars);
    rins.setMinFixedRate(config_.rins_min_fixed_rate);

    if (config_.mode == HeuristicRuntimeMode::Opportunistic) {
        const RestartStrategy strategy = restart_engine_.choose(
            config_.mode, stats_.calls, stagnation_epochs_);
        out.restart_strategy = strategy;
        switch (strategy) {
            case RestartStrategy::Uniform:
                break;
            case RestartStrategy::TournamentCrossover:
                rins.setMinFixedRate(std::min<Real>(0.5, config_.rins_min_fixed_rate * 1.5));
                break;
            case RestartStrategy::Perturbation:
                rins.setAgreementTol(config_.rins_agreement_tol * 2.0);
                break;
            case RestartStrategy::PolarityGuided:
                rins.setMinFixedVars(std::max<Int>(4, config_.rins_min_fixed_vars - 2));
                break;
            case RestartStrategy::ActivityGuided:
                rins.setSubproblemIterLimit(config_.rins_subproblem_iter_limit + 16);
                break;
            case RestartStrategy::DistanceGuided:
                rins.setMinFixedRate(std::min<Real>(0.5, config_.rins_min_fixed_rate * 1.25));
                break;
        }
    }

    auto hsol = rins.run(ctx.problem, ctx.lp, ctx.primals,
                         ctx.incumbent, ctx.incumbent_values);
    out.executed_solve = rins.lastExecutedSolve();
    out.skipped_no_incumbent = rins.lastSkippedNoIncumbent();
    out.skipped_few_fixes = rins.lastSkippedFewFixes();
    out.fixed_count = rins.lastFixedCount();
    out.lp_iterations = rins.lastLpIterations();
    out.work_units = rins.lastWorkUnits();
    stats_.lp_iterations += rins.lastLpIterations();

    bool improved = false;
    if (hsol.has_value() &&
        isImprovement(ctx.problem.sense, hsol->objective, ctx.incumbent)) {
        out.improved = true;
        out.solution = std::move(hsol);
        improved = true;
        if (callback_ != nullptr) {
            callback_->onIncumbent("rins", out.solution->objective,
                                   ctx.node_count, ctx.thread_id);
        }
    }

    recordCall(ctx.node_count, out.work_units, improved);
    if (callback_ != nullptr) callback_->onHeartbeat(ctx.node_count, stats_);
    return out;
}

}  // namespace mipx
