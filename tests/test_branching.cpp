#include "mipx/branching.h"
#include "mipx/dual_simplex.h"
#include "mipx/symmetry.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using namespace mipx;

TEST_CASE("MostFractional selects variable closest to 0.5", "[branching]") {
    // x0 = 1.1 (frac 0.1), x1 = 2.5 (frac 0.5), x2 = 3.3 (frac 0.3)
    std::vector<Real> vals = {1.1, 2.5, 3.3};
    std::vector<VarType> types = {VarType::Integer, VarType::Integer, VarType::Integer};
    std::vector<Real> lb = {0, 0, 0};
    std::vector<Real> ub = {10, 10, 10};

    MostFractionalBranching rule;
    Index selected = rule.select(vals, types, lb, ub);
    REQUIRE(selected == 1);
}

TEST_CASE("FirstFractional selects first fractional variable", "[branching]") {
    // x0 integral, x1 fractional, x2 fractional
    std::vector<Real> vals = {2.0, 1.7, 3.3};
    std::vector<VarType> types = {VarType::Integer, VarType::Integer, VarType::Integer};
    std::vector<Real> lb = {0, 0, 0};
    std::vector<Real> ub = {10, 10, 10};

    FirstFractionalBranching rule;
    Index selected = rule.select(vals, types, lb, ub);
    REQUIRE(selected == 1);
}

TEST_CASE("All integral returns -1", "[branching]") {
    std::vector<Real> vals = {1.0, 2.0, 3.0};
    std::vector<VarType> types = {VarType::Integer, VarType::Integer, VarType::Integer};
    std::vector<Real> lb = {0, 0, 0};
    std::vector<Real> ub = {10, 10, 10};

    MostFractionalBranching mf;
    REQUIRE(mf.select(vals, types, lb, ub) == -1);

    FirstFractionalBranching ff;
    REQUIRE(ff.select(vals, types, lb, ub) == -1);
}

TEST_CASE("Continuous variables are skipped", "[branching]") {
    // x0 continuous fractional, x1 integer fractional
    std::vector<Real> vals = {1.5, 2.5};
    std::vector<VarType> types = {VarType::Continuous, VarType::Integer};
    std::vector<Real> lb = {0, 0};
    std::vector<Real> ub = {10, 10};

    FirstFractionalBranching rule;
    REQUIRE(rule.select(vals, types, lb, ub) == 1);
}

TEST_CASE("Binary variables handled correctly", "[branching]") {
    // x0 = 0.3 binary, x1 = 0.7 binary
    std::vector<Real> vals = {0.3, 0.7};
    std::vector<VarType> types = {VarType::Binary, VarType::Binary};
    std::vector<Real> lb = {0, 0};
    std::vector<Real> ub = {1, 1};

    MostFractionalBranching rule;
    Index selected = rule.select(vals, types, lb, ub);
    // Both are fractional binary variables; either is a valid selection.
    REQUIRE((selected == 0 || selected == 1));
}

TEST_CASE("Fixed variables are never selected", "[branching]") {
    // x0 fixed (lb==ub), x1 fractional
    std::vector<Real> vals = {2.5, 1.5};
    std::vector<VarType> types = {VarType::Integer, VarType::Integer};
    std::vector<Real> lb = {2.5, 0};
    std::vector<Real> ub = {2.5, 10};

    FirstFractionalBranching rule;
    REQUIRE(rule.select(vals, types, lb, ub) == 1);
}

TEST_CASE("createChildren produces correct child nodes", "[branching]") {
    BnbNode parent;
    parent.id = 5;
    parent.depth = 3;
    parent.basis = {BasisStatus::Basic, BasisStatus::AtLower};

    auto [left, right] = createChildren(parent, 0, 2.7);

    // Left child: x_0 <= 2 (upper bound).
    REQUIRE(left.parent_id == 5);
    REQUIRE(left.depth == 4);
    REQUIRE(left.branch.variable == 0);
    REQUIRE(left.branch.bound == 2.0);
    REQUIRE(left.branch.is_upper == true);
    REQUIRE(left.basis.size() == 2);
    REQUIRE(left.bound_changes.size() == 1);
    REQUIRE(left.bound_changes[0].level == left.depth);

    // Right child: x_0 >= 3 (lower bound).
    REQUIRE(right.parent_id == 5);
    REQUIRE(right.depth == 4);
    REQUIRE(right.branch.variable == 0);
    REQUIRE(right.branch.bound == 3.0);
    REQUIRE(right.branch.is_upper == false);
    REQUIRE(right.basis.size() == 2);
    REQUIRE(right.bound_changes.size() == 1);
    REQUIRE(right.bound_changes[0].level == right.depth);
}

TEST_CASE("createChildren accumulates bound changes", "[branching]") {
    BnbNode parent;
    parent.id = 2;
    parent.depth = 1;
    parent.bound_changes = {{0, 1.0, false, parent.depth}};

    auto [left, right] = createChildren(parent, 1, 3.5);

    REQUIRE(left.bound_changes.size() == 2);
    REQUIRE(left.bound_changes[0].variable == 0);
    REQUIRE(left.bound_changes[0].level == 1);
    REQUIRE(left.bound_changes[1].variable == 1);
    REQUIRE(left.bound_changes[1].level == 2);

    REQUIRE(right.bound_changes.size() == 2);
}

TEST_CASE("createChildren propagates local subtree cuts", "[branching]") {
    BnbNode parent;
    Cut local_cut;
    local_cut.indices = {0};
    local_cut.values = {1.0};
    local_cut.upper = 2.0;
    local_cut.local = true;
    parent.local_cuts.push_back(local_cut);

    auto [left, right] = createChildren(parent, 0, 1.4);
    REQUIRE(left.local_cuts.size() == 1);
    REQUIRE(right.local_cuts.size() == 1);
    CHECK(left.local_cuts[0].local);
    CHECK(right.local_cuts[0].local);
}

TEST_CASE("isIntegral helper", "[branching]") {
    REQUIRE(isIntegral(3.0));
    REQUIRE(isIntegral(3.0 + 1e-8));
    REQUIRE(isIntegral(3.0 - 1e-8));
    REQUIRE_FALSE(isIntegral(3.5));
    REQUIRE_FALSE(isIntegral(3.1));
    // Custom tolerance.
    REQUIRE(isIntegral(3.01, 0.02));
    REQUIRE_FALSE(isIntegral(3.01, 0.005));
}

TEST_CASE("fractionality helper", "[branching]") {
    REQUIRE(fractionality(3.0) < 1e-12);
    REQUIRE_THAT(fractionality(3.5), Catch::Matchers::WithinAbs(0.5, 1e-12));
    REQUIRE_THAT(fractionality(3.3), Catch::Matchers::WithinAbs(0.3, 1e-12));
    REQUIRE_THAT(fractionality(3.7), Catch::Matchers::WithinAbs(0.3, 1e-12));
}

TEST_CASE("ReliabilityBranching updates pseudocost reliability counters", "[branching]") {
    ReliabilityBranching branching;
    branching.reset(3);
    branching.setReliabilityThreshold(2);

    branching.updatePseudoCost(1, false, 2.0);
    branching.updatePseudoCost(1, true, 6.0);
    CHECK_FALSE(branching.isReliable(1));
    CHECK(branching.downReliability(1) == 1);
    CHECK(branching.upReliability(1) == 1);

    branching.updatePseudoCost(1, false, 4.0);
    branching.updatePseudoCost(1, true, 2.0);
    CHECK(branching.isReliable(1));
    CHECK(branching.downReliability(1) == 2);
    CHECK(branching.upReliability(1) == 2);
    CHECK_THAT(branching.downPseudoCost(1), Catch::Matchers::WithinAbs(3.0, 1e-12));
    CHECK_THAT(branching.upPseudoCost(1), Catch::Matchers::WithinAbs(4.0, 1e-12));
}

TEST_CASE("Inference coverage only compares candidates on the winning pseudocost path",
          "[branching]") {
    LpProblem lp;
    lp.num_cols = 6;
    lp.col_type.assign(lp.num_cols, VarType::Binary);
    lp.col_lower.assign(lp.num_cols, 0.0);
    lp.col_upper.assign(lp.num_cols, 1.0);
    const std::vector<Real> primals(lp.num_cols, 0.5);
    DualSimplexSolver solver;
    ReliabilityBranching branching;
    branching.reset(lp.num_cols);
    branching.setStrongBranchProbeBudget(0);
    branching.setReliabilityThreshold(1);
    branching.setPseudocostFallback(200.0);
    const std::array<Real, 5> costs = {200.0, 200.0, 198.5, 192.0, 184.0};
    for (Index j = 0; j < static_cast<Index>(costs.size()); ++j) {
        branching.updatePseudoCost(j, false, costs[j]);
        branching.updatePseudoCost(j, true, costs[j]);
    }

    const std::array<Index, 6> mapping = {6, 2, 4, 1, 5, 3};
    std::vector<MipBranchInferenceStats> history(7);
    for (const Index original : mapping) {
        history[original] = {{.samples = 4, .tightened_columns = 8},
                             {.samples = 4, .tightened_columns = 40}};
    }
    history[mapping[1]].up.samples = 3;               // Exact tie without a ready challenger.
    history[mapping[2]].up.tightened_columns = 100;   // Weaker mean unchanged.
    history[mapping[3]].down.tightened_columns = 12;  // Better weaker mean at 4%.
    history[mapping[4]].down.tightened_columns = 36;  // Better but outside 5%.
    history[mapping[5]].down.tightened_columns = 24;  // Ready, but unreliable pseudocost.
    BranchingTelemetry telemetry;
    BranchInferenceCoverageStats stats;
    const BranchInferenceCoverage coverage{history, mapping, stats};

    const auto select = [&](const BranchInferenceCoverage* observation) {
        return branching.selectWithInferenceCoverage(solver, lp, primals, lp.col_lower,
                                                     lp.col_upper, 0.0, false, telemetry, {},
                                                     observation);
    };

    CHECK(select(nullptr).variable == 0);
    CHECK(stats.reliable.decisions == 0);
    CHECK(stats.fallback.decisions == 0);
    const auto chosen = select(&coverage);
    REQUIRE(chosen.variable == 0);
    CHECK(chosen.used_reliable_pseudocost);
    CHECK(stats.reliable.decisions == 1);
    CHECK(stats.fallback.decisions == 0);
    const auto& bands = stats.reliable.bands;
    CHECK(bands[0].close == 1);
    CHECK(bands[0].ready == 0);
    CHECK(bands[0].favored == 0);
    CHECK(bands[1].close == 1);
    CHECK(bands[1].ready == 1);
    CHECK(bands[1].favored == 0);
    CHECK(bands[2].close == 1);
    CHECK(bands[2].ready == 1);
    CHECK(bands[2].favored == 1);

    branching.setReliabilityThreshold(2);
    CHECK(select(&coverage).variable == 0);
    CHECK(stats.reliable.decisions == 1);
    CHECK(stats.fallback.decisions == 1);
    for (const auto& band : stats.fallback.bands) {
        CHECK(band.close == 1);
        CHECK(band.ready == 1);
        CHECK(band.favored == 1);
    }
    history[mapping[0]].down.samples = 3;  // Winner readiness is required too.
    CHECK(select(&coverage).variable == 0);
    CHECK(stats.fallback.decisions == 2);
    for (const auto& band : stats.fallback.bands) {
        CHECK(band.close == 2);
        CHECK(band.ready == 1);
        CHECK(band.favored == 1);
    }
}

TEST_CASE("Inference picks a ready near tie only when the weaker direction improves",
          "[branching]") {
    const Real scale = GENERATE(1e-8, 1.0, 1e150);
    const Index reliability_threshold = GENERATE(1, 2);
    LpProblem lp;
    lp.num_cols = 6;
    lp.col_type.assign(lp.num_cols, VarType::Binary);
    lp.col_lower.assign(lp.num_cols, 0.0);
    lp.col_upper.assign(lp.num_cols, 1.0);
    const std::array<Real, 6> primals{0.5, 0.5, 0.125, 0.25, 0.5, 0.5};
    DualSimplexSolver solver;
    ReliabilityBranching branching;
    branching.reset(lp.num_cols);
    branching.setStrongBranchProbeBudget(0);
    branching.setReliabilityThreshold(reliability_threshold);
    const std::array<Real, 6> costs{200.0, 199.0, 199.5, 199.5, 197.0, 200.0};
    for (Index j = 0; j < lp.num_cols; ++j) {
        // Columns 2 and 3 have the same predicted score. The prefilter visits 1, 3, 2.
        const Real down_multiplier = j == 2 ? 4.0 : (j == 3 ? 2.0 : 1.0);
        branching.updatePseudoCost(j, false, costs[j] * scale * down_multiplier);
        branching.updatePseudoCost(j, true, costs[j] * scale * (j == 2 || j == 3 ? 0.5 : 1.0));
    }

    std::array<MipBranchInferenceStats, 6> history{};
    for (auto& entry : history) {
        entry = {{.samples = 4, .tightened_columns = 12}, {.samples = 4, .tightened_columns = 40}};
    }
    history[0].down.tightened_columns = 8;
    history[4].down.tightened_columns = 36;  // Outside 1%.
    history[5].up.samples = 3;               // Incomplete challenger.
    BranchingTelemetry telemetry;
    BranchInferenceCoverageStats stats;
    const BranchInferenceCoverage coverage{history, {}, stats, true};
    const auto select = [&](const BranchInferenceCoverage* observation) {
        return branching.selectWithInferenceCoverage(solver, lp, primals, lp.col_lower,
                                                     lp.col_upper, 0.0, false, telemetry, {},
                                                     observation);
    };
    auto& path = reliability_threshold == 1 ? stats.reliable : stats.fallback;
    REQUIRE(select(nullptr).variable == 0);
    const auto preferred = select(&coverage);
    REQUIRE(preferred.variable == 2);
    REQUIRE(preferred.inference_baseline == 0);
    REQUIRE(path.selection_changes == 1);
    stats.recordArbitration(preferred, 1, 1);  // The override replaces the proposal.
    stats.recordArbitration(preferred, 2, 2);  // The baseline would choose it anyway.
    CHECK(path.surviving_changes == 0);
    stats.recordArbitration(preferred, 2, 0);
    CHECK(path.surviving_changes == 1);
    stats.recordArbitration(preferred, 1, 0);  // The override takes a third column.
    CHECK(path.surviving_changes == 2);

    history[2].down.samples = 3;
    CHECK(select(&coverage).variable == 3);
    history[0].down.tightened_columns = 12;
    CHECK(select(&coverage).variable == 0);  // No strictly better ready challenger.
    history[0].down.samples = 3;
    history[3].down.tightened_columns = 24;
    CHECK(select(&coverage).variable == 0);  // Incomplete baseline.
    CHECK(path.selection_changes == 2);
}

TEST_CASE("Smoothed weaker child cutoff rates rank ready pseudocost near ties", "[branching]") {
    const Index reliability_threshold = GENERATE(1, 2);
    const Real scale = GENERATE(0.01, 1.0);
    LpProblem lp;
    lp.num_cols = 6;
    lp.col_type.assign(lp.num_cols, VarType::Binary);
    lp.col_lower.assign(lp.num_cols, 0.0);
    lp.col_upper.assign(lp.num_cols, 1.0);
    const std::vector<Real> primals(lp.num_cols, 0.5);
    DualSimplexSolver solver;
    ReliabilityBranching branching;
    branching.reset(lp.num_cols);
    branching.setStrongBranchProbeBudget(0);
    branching.setReliabilityThreshold(reliability_threshold);
    // Activity puts column 1 ahead of columns 2 and 3, which sit at the 1% boundary.
    const std::array<Real, 6> costs{200.0, 133.2, 198.0, 198.0, 196.0, 200.0};
    const std::array<Real, 6> activity{0.0, 1.0, 0.0, 0.0, 0.0, 0.0};
    for (Index j = 0; j < lp.num_cols; ++j) {
        branching.updatePseudoCost(j, false, costs[j] * scale);
        branching.updatePseudoCost(j, true, costs[j] * scale);
    }

    const std::array<Index, 6> columns{4, 5, 0, 2, 1, 3};
    std::array<MipBranchCutoffStats, 6> history;
    history[columns[0]] = {{8, 0}, {8, 8}};     // Weaker rate 1/10.
    history[columns[1]] = {{4, 0}, {4, 4}};     // Weaker rate 1/6, equal raw rate.
    history[columns[2]] = {{20, 20}, {12, 0}};  // Better pooled rate, weaker rate 1/14.
    history[columns[3]] = {{4, 4}, {3, 3}};     // Insufficient up exposure.
    history[columns[4]] = {{4, 4}, {4, 4}};     // Outside 1%.
    history[columns[5]] = history[columns[0]];
    BranchingTelemetry telemetry;
    BranchCutoffComparisonStats stats;
    BranchCutoffComparison cutoff{history, columns, stats};
    const auto select = [&] {
        return branching.selectWithHistories(solver, lp, primals, lp.col_lower, lp.col_upper, 0.0,
                                             false, telemetry, activity, {.cutoff = &cutoff});
    };
    const auto& path = reliability_threshold == 1 ? stats.reliable : stats.fallback;
    CHECK(select().variable == 0);
    REQUIRE(path.ready_pairs == 3);
    REQUIRE(path.favored_decisions == 1);
    CHECK(path.selection_changes == 0);

    cutoff.select_by_cutoff = true;
    const auto preferred = select();
    REQUIRE(preferred.variable == 1);
    REQUIRE(preferred.cutoff_baseline == 0);
    CHECK(path.selection_changes == 1);
    stats.recordArbitration(preferred, 1, 1);
    CHECK(path.surviving_changes == 0);
    stats.recordArbitration(preferred, 1, 0);
    CHECK(path.surviving_changes == 1);

    history[columns[2]] = history[columns[1]];
    history[columns[3]] = history[columns[1]];
    CHECK(select().variable == 2);  // Equal rate, then objective score and column index.
    history[columns[0]].up.qualified_attempts = 3;
    history[columns[0]].up.cutoffs = 3;
    CHECK(select().variable == 0);
    CHECK(path.selection_changes == 2);

    history[columns[0]].up = {8, 8};
    cutoff.original_columns = std::span(columns).first(1);
    CHECK(select().variable == 0);
    auto invalid_mapping = columns;
    invalid_mapping[0] = static_cast<Index>(history.size());
    cutoff.original_columns = invalid_mapping;
    CHECK(select().variable == 0);
    CHECK(path.selection_changes == 2);
}

TEST_CASE("ReliabilityBranching gives sibling probes the same reliability view", "[branching]") {
    LpProblem lp;
    lp.num_cols = 3;
    lp.obj = {0.0, 0.0, 1.0};
    lp.col_lower = {0.0, 0.0, 0.0};
    lp.col_upper = {1.0, 1.0, kInf};
    lp.col_type = {VarType::Binary, VarType::Binary, VarType::Continuous};
    lp.num_rows = 4;
    // Both siblings have x = 0.5 at optimum. Fixing x costs 3 when z = 0,
    // and 1 when z = 1 because the steeper rows are then slack.
    lp.row_lower = {1.0, -1.0, 3.0, -3.0};
    lp.row_upper.assign(4, kInf);
    lp.matrix = SparseMatrix(4, 3,
                             std::vector<Triplet>{{0, 0, 2.0},
                                                  {0, 2, 1.0},
                                                  {1, 0, -2.0},
                                                  {1, 2, 1.0},
                                                  {2, 0, 6.0},
                                                  {2, 1, 3.0},
                                                  {2, 2, 1.0},
                                                  {3, 0, -6.0},
                                                  {3, 1, 3.0},
                                                  {3, 2, 1.0}});

    ReliabilityBranching branching;
    branching.reset(lp.num_cols);
    for (Int n = 0; n < 3; ++n) {
        branching.updatePseudoCost(0, false, 1.0);
        branching.updatePseudoCost(0, true, 1.0);
    }
    REQUIRE_FALSE(branching.isReliable(0));

    std::vector<StrongBranchPseudocostSample> samples;
    for (const Real fixed : {0.0, 1.0}) {
        auto sibling = lp;
        sibling.col_lower[1] = sibling.col_upper[1] = fixed;
        DualSimplexSolver solver;
        solver.setVerbose(false);
        solver.load(sibling);
        const auto result = solver.solve();
        REQUIRE(result.status == Status::Optimal);
        const auto primals = solver.getPrimalValues();
        REQUIRE_THAT(primals[0], Catch::Matchers::WithinAbs(0.5, 1e-12));
        BranchingTelemetry telemetry;
        const auto selection =
            branching.selectDeferred(solver, sibling, primals, sibling.col_lower, sibling.col_upper,
                                     result.objective, 2, telemetry, samples);
        CHECK(selection.variable == 0);
        CHECK(telemetry.strong_branch_probes == 2);
        CHECK(telemetry.pseudocost_updates_strong == 2);
        CHECK(telemetry.pseudocost_updates_tree == 0);
        CHECK_FALSE(branching.isReliable(0));
        CHECK(branching.upReliability(0) == 3);
        CHECK(branching.downReliability(0) == 3);
    }
    REQUIRE(samples.size() == 4);
    for (const auto& sample : samples) {
        REQUIRE(sample.variable == 0);
        branching.updatePseudoCost(sample.variable, sample.up_direction, sample.gain_per_unit);
    }
    CHECK(branching.isReliable(0));
    CHECK(branching.upReliability(0) == 5);
    CHECK(branching.downReliability(0) == 5);
    CHECK_THAT(branching.upPseudoCost(0), Catch::Matchers::WithinAbs(2.2, 1e-12));
    CHECK_THAT(branching.downPseudoCost(0), Catch::Matchers::WithinAbs(2.2, 1e-12));
}

TEST_CASE("Balanced branch scoring prefers gains in both pseudocost directions", "[branching]") {
    LpProblem lp;
    lp.num_cols = 2;
    lp.col_type = {VarType::Binary, VarType::Binary};
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {1.0, 1.0};
    const std::vector<Real> primals = {0.5, 0.5};
    DualSimplexSolver solver;
    ReliabilityBranching branching;
    branching.reset(lp.num_cols);
    branching.setReliabilityThreshold(1);
    branching.updatePseudoCost(0, false, 2.0);
    branching.updatePseudoCost(0, true, 28.0);
    branching.updatePseudoCost(1, false, 8.0);
    branching.updatePseudoCost(1, true, 8.0);

    BranchingTelemetry telemetry;
    CHECK(branching.select(solver, lp, primals, lp.col_lower, lp.col_upper, 0.0, false, telemetry)
              .variable == 0);
    branching.setBalancedScore(true);
    CHECK(branching.select(solver, lp, primals, lp.col_lower, lp.col_upper, 0.0, false, telemetry)
              .variable == 1);
    CHECK(telemetry.pseudocost_hits == 2);
    CHECK(telemetry.strong_branch_probes == 0);
}

TEST_CASE("Conflict activity changes pseudocost ranking without changing observations",
          "[branching][conflicts]") {
    LpProblem lp;
    lp.num_cols = 2;
    lp.col_type = {VarType::Binary, VarType::Binary};
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {1.0, 1.0};
    const std::vector<Real> primals = {0.5, 0.5};
    DualSimplexSolver solver;
    ReliabilityBranching branching;
    branching.reset(lp.num_cols);
    branching.setStrongBranchProbeBudget(0);
    const Int reliability_threshold = GENERATE(1, 2);
    branching.setReliabilityThreshold(reliability_threshold);
    for (const bool up : {false, true}) {
        branching.updatePseudoCost(0, up, 4.0);
        branching.updatePseudoCost(1, up, 3.0);
    }

    BranchingTelemetry telemetry;
    const auto select = [&](std::span<const Real> activity) {
        return branching.selectWithConflictActivity(solver, lp, primals, lp.col_lower, lp.col_upper,
                                                    0.0, false, telemetry, activity);
    };
    REQUIRE(select({}).variable == 0);
    const std::vector<Real> activity = {0.0, 1.0};
    const auto selected = select(activity);
    CHECK(selected.variable == 1);
    CHECK(selected.used_pseudocost);
    CHECK(selected.used_reliable_pseudocost == (reliability_threshold == 1));
    const std::vector<Real> low_activity = {0.0, 0.25};
    CHECK(select(low_activity).variable == 0);

    CHECK(branching.upPseudoCost(0) == 4.0);
    CHECK(branching.downPseudoCost(0) == 4.0);
    CHECK(branching.upPseudoCost(1) == 3.0);
    CHECK(branching.downPseudoCost(1) == 3.0);
    for (Index col = 0; col < lp.num_cols; ++col) {
        CHECK(branching.upReliability(col) == 1);
        CHECK(branching.downReliability(col) == 1);
    }
    CHECK(telemetry.pseudocost_updates_strong == 0);
    CHECK(telemetry.pseudocost_updates_tree == 0);
}

TEST_CASE("Balanced branch scoring prefers gains in both strong probes", "[branching]") {
    LpProblem lp;
    lp.sense = Sense::Minimize;
    lp.num_cols = 4;
    lp.obj = {0.0, 0.0, 1.0, 1.0};
    lp.col_type = {VarType::Binary, VarType::Binary, VarType::Continuous, VarType::Continuous};
    lp.col_lower = {0.0, 0.0, 0.0, 0.0};
    lp.col_upper = {1.0, 1.0, kInf, kInf};
    lp.num_rows = 4;
    // The two epigraph variables drive x and y to 0.5 at the root.
    // Fixing x costs 1 down and 14 up. Fixing y costs 4 in either direction.
    lp.row_lower = {1.0, -14.0, 4.0, -4.0};
    lp.row_upper.assign(4, kInf);
    lp.matrix = SparseMatrix(4, 4,
                             std::vector<Triplet>{{0, 0, 2.0},
                                                  {0, 2, 1.0},
                                                  {1, 0, -28.0},
                                                  {1, 2, 1.0},
                                                  {2, 1, 8.0},
                                                  {2, 3, 1.0},
                                                  {3, 1, -8.0},
                                                  {3, 3, 1.0}});

    DualSimplexSolver solver;
    solver.setVerbose(false);
    solver.load(lp);
    const auto root = solver.solve();
    REQUIRE(root.status == Status::Optimal);
    const auto primals = solver.getPrimalValues();
    REQUIRE_THAT(primals[0], Catch::Matchers::WithinAbs(0.5, 1e-8));
    REQUIRE_THAT(primals[1], Catch::Matchers::WithinAbs(0.5, 1e-8));

    ReliabilityBranching branching;
    branching.reset(lp.num_cols);
    BranchingTelemetry telemetry;
    CHECK(branching
              .select(solver, lp, primals, lp.col_lower, lp.col_upper, root.objective, true,
                      telemetry)
              .variable == 0);
    branching.reset(lp.num_cols);
    branching.setBalancedScore(true);
    CHECK(branching
              .select(solver, lp, primals, lp.col_lower, lp.col_upper, root.objective, true,
                      telemetry)
              .variable == 1);
    CHECK(telemetry.strong_branch_probes == 8);
}

TEST_CASE("Balanced branch scoring retains a strong down gain when the up probe stops",
          "[branching][time_limit]") {
    class InterruptedProbes : public DualSimplexSolver {
    public:
        LpResult solve() override { return results_.at(next_++); }

    private:
        std::size_t next_ = 0;
        const std::vector<LpResult> results_ = {{Status::Optimal, 10.0},
                                                {Status::Optimal, 10.0},
                                                {Status::Infeasible},
                                                {Status::TimeLimit}};
    };

    LpProblem lp;
    lp.num_cols = 2;
    lp.obj = {0.0, 0.0};
    lp.col_type = {VarType::Binary, VarType::Binary};
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {1.0, 1.0};
    lp.num_rows = 1;
    lp.row_lower = {-kInf};
    lp.row_upper = {1.0};
    lp.matrix = SparseMatrix(1, 2, std::vector<Triplet>{{0, 0, 1.0}, {0, 1, 1.0}});

    InterruptedProbes solver;
    solver.setVerbose(false);
    solver.load(lp);
    REQUIRE(solver.DualSimplexSolver::solve().status == Status::Optimal);

    ReliabilityBranching branching;
    branching.reset(lp.num_cols);
    branching.setBalancedScore(true);
    BranchingTelemetry telemetry;
    const std::vector<Real> primals = {0.5, 0.5};
    CHECK(branching.select(solver, lp, primals, lp.col_lower, lp.col_upper, 0.0, true, telemetry)
              .variable == 1);
    CHECK(telemetry.strong_branch_probes == 4);
    CHECK(branching.downReliability(1) == 1);
    CHECK(branching.upReliability(1) == 0);
}

TEST_CASE("ReliabilityBranching learns pseudocosts from solved children", "[branching]") {
    // Issue #199: a child's LP gain over its parent is a pseudocost sample.
    // No strong branching happens here at all, so any change in the
    // pseudocosts comes from the tree update alone.
    ReliabilityBranching branching;
    branching.reset(3);
    BranchingTelemetry telemetry;

    BnbNode parent;
    parent.id = 7;
    parent.lp_bound = 10.0;
    auto [down, up] = createChildren(std::move(parent), 1, 2.25);
    REQUIRE(down.branch.is_upper);
    CHECK(down.parent_objective == 10.0);
    CHECK(down.branch_value == 2.25);

    // Down child: x1 <= 2, distance 0.25, gain 0.5 -> 2 per unit.
    CHECK(branching.updateFromChild(TreePseudocostSample::fromChild(down, Status::Optimal, 10.5),
                                    telemetry));
    // Up child: x1 >= 3, distance 0.75, gain 3 -> 4 per unit.
    CHECK(branching.updateFromChild(TreePseudocostSample::fromChild(up, Status::Optimal, 13.0),
                                    telemetry));

    CHECK(branching.downReliability(1) == 1);
    CHECK(branching.upReliability(1) == 1);
    CHECK_THAT(branching.downPseudoCost(1), Catch::Matchers::WithinAbs(2.0, 1e-12));
    CHECK_THAT(branching.upPseudoCost(1), Catch::Matchers::WithinAbs(4.0, 1e-12));
    CHECK(telemetry.pseudocost_updates_tree == 2);
    CHECK(telemetry.pseudocost_updates_strong == 0);
    CHECK(telemetry.pseudocost_tree_skipped == 0);
}

TEST_CASE("ReliabilityBranching records nothing from children without a gain", "[branching]") {
    ReliabilityBranching branching;
    branching.reset(3);
    BranchingTelemetry telemetry;

    BnbNode parent;
    parent.lp_bound = 10.0;
    auto [down, up] = createChildren(std::move(parent), 1, 2.25);

    // An infeasible child and one whose LP stopped early carry no gain.
    CHECK_FALSE(branching.updateFromChild(
        TreePseudocostSample::fromChild(down, Status::Infeasible, 0.0), telemetry));
    CHECK_FALSE(branching.updateFromChild(
        TreePseudocostSample::fromChild(up, Status::IterLimit, 11.0), telemetry));
    // A child the incumbent cuts off measures the cutoff, not the branching.
    CHECK_FALSE(branching.updateFromChild(
        TreePseudocostSample::fromChild(up, Status::Optimal, 50.0, true), telemetry));
    // The root has no parent and no branching to credit.
    CHECK_FALSE(branching.updateFromChild(
        TreePseudocostSample::fromChild(BnbNode{}, Status::Optimal, 11.0), telemetry));

    CHECK(branching.downReliability(1) == 0);
    CHECK(branching.upReliability(1) == 0);
    CHECK(telemetry.pseudocost_updates_tree == 0);
    CHECK(telemetry.pseudocost_tree_skipped == 3);
}

TEST_CASE("SymmetryManager adds canonical symmetry cuts", "[symmetry]") {
    LpProblem lp;
    lp.name = "symmetry";
    lp.sense = Sense::Minimize;
    lp.num_cols = 2;
    lp.obj = {1.0, 1.0};
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {1.0, 1.0};
    lp.col_type = {VarType::Binary, VarType::Binary};
    lp.col_names = {"x", "y"};

    lp.num_rows = 1;
    lp.row_lower = {-kInf};
    lp.row_upper = {1.0};
    lp.row_names = {"sum"};
    std::vector<Triplet> trips = {
        {0, 0, 1.0},
        {0, 1, 1.0},
    };
    lp.matrix = SparseMatrix(1, 2, std::move(trips));

    SymmetryManager manager;
    manager.detect(lp);
    REQUIRE(manager.hasSymmetry());
    REQUIRE(!manager.orbitalFixes().empty());

    LpProblem working = lp;
    const Index root_rows = working.num_rows;
    const Index cuts = manager.addSymmetryCuts(working);
    CHECK(cuts == static_cast<Index>(manager.orbitalFixes().size()));
    CHECK(working.num_rows == root_rows + cuts);

    const Index sym_row = working.num_rows - 1;
    CHECK(working.row_lower[sym_row] == -kInf);
    CHECK(working.row_upper[sym_row] == 0.0);
    CHECK(working.matrix.coeff(sym_row, 0) == -1.0);
    CHECK(working.matrix.coeff(sym_row, 1) == 1.0);
    CHECK(manager.detectWorkUnits() > 0.0);
    CHECK(manager.cutWorkUnits() > 0.0);
}

TEST_CASE("SymmetryManager orbit and cut order is deterministic", "[symmetry]") {
    LpProblem lp;
    lp.name = "symmetry_order";
    lp.sense = Sense::Minimize;
    lp.num_cols = 4;
    lp.obj = {1.0, 1.0, 2.0, 2.0};
    lp.col_lower = {0.0, 0.0, 0.0, 0.0};
    lp.col_upper = {1.0, 1.0, 1.0, 1.0};
    lp.col_type = {VarType::Binary, VarType::Binary, VarType::Binary, VarType::Binary};
    lp.col_names = {"x0", "x1", "x2", "x3"};

    lp.num_rows = 2;
    lp.row_lower = {-kInf, -kInf};
    lp.row_upper = {1.0, 1.0};
    lp.row_names = {"r0", "r1"};
    std::vector<Triplet> trips = {
        {0, 0, 1.0},
        {0, 1, 1.0},
        {1, 2, 1.0},
        {1, 3, 1.0},
    };
    lp.matrix = SparseMatrix(2, 4, std::move(trips));

    SymmetryManager manager;
    manager.detect(lp);
    REQUIRE(manager.hasSymmetry());
    REQUIRE(manager.orbits().size() == 2);
    CHECK(manager.orbits()[0].size() == 2);
    CHECK(manager.orbits()[0][0] == 0);
    CHECK(manager.orbits()[0][1] == 1);
    CHECK(manager.orbits()[1].size() == 2);
    CHECK(manager.orbits()[1][0] == 2);
    CHECK(manager.orbits()[1][1] == 3);

    LpProblem working = lp;
    working.row_names = {"r0"};
    const Index cuts = manager.addSymmetryCuts(working);
    CHECK(cuts == 2);
    CHECK(working.row_names.size() == static_cast<std::size_t>(working.num_rows));
    CHECK(working.row_names[2] == "sym_cut_0_1");
    CHECK(working.row_names[3] == "sym_cut_2_3");
    CHECK(manager.cutWorkUnits() == 4.0);
}

TEST_CASE("Symmetry detection enforces canonical branching", "[branching][symmetry]") {
    LpProblem lp;
    lp.name = "symmetry";
    lp.sense = Sense::Minimize;
    lp.num_cols = 2;
    lp.obj = {1.0, 1.0};
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {1.0, 1.0};
    lp.col_type = {VarType::Binary, VarType::Binary};
    lp.col_names = {"x", "y"};

    lp.num_rows = 1;
    lp.row_lower = {-kInf};
    lp.row_upper = {1.0};
    lp.row_names = {"sum"};
    std::vector<Triplet> trips = {
        {0, 0, 1.0},
        {0, 1, 1.0},
    };
    lp.matrix = SparseMatrix(1, 2, std::move(trips));

    SymmetryManager manager;
    manager.detect(lp);
    REQUIRE(manager.hasSymmetry());
    CHECK(manager.canonical(0) == 0);
    CHECK(manager.canonical(1) == 0);

    ReliabilityBranching branching;
    branching.reset(lp.num_cols);
    branching.setSymmetryManager(&manager);
    branching.setReliabilityThreshold(1);
    branching.updatePseudoCost(0, false, 1.0);
    branching.updatePseudoCost(0, true, 1.0);
    DualSimplexSolver solver;
    solver.load(lp);
    solver.setVerbose(false);
    solver.solve();

    std::vector<Real> primal = {0.5, 0.5};
    std::vector<Real> lb = lp.col_lower;
    std::vector<Real> ub = lp.col_upper;
    BranchingTelemetry telemetry;
    auto selection = branching.select(solver, lp, primal, lb, ub, 0.0, false, telemetry);
    CHECK(selection.variable == 0);
}

// #351: once the MIP's deadline has passed, the first strong-branching probe
// stops on TimeLimit; the rule probes no more, learns no pseudocost from the
// cut-short probe, and still picks a variable, by pseudocosts.
TEST_CASE("ReliabilityBranching stops strong branching at a passed deadline",
          "[branching][time_limit]") {
    LpProblem lp;
    lp.name = "deadline";
    lp.sense = Sense::Minimize;
    lp.num_cols = 2;
    lp.obj = {-1.0, -1.0};
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {1.0, 1.0};
    lp.col_type = {VarType::Binary, VarType::Binary};
    lp.col_names = {"x", "y"};
    lp.num_rows = 1;
    lp.row_lower = {-kInf};
    lp.row_upper = {1.0};
    lp.row_names = {"sum"};
    lp.matrix = SparseMatrix(1, 2, std::vector<Triplet>{{0, 0, 1.0}, {0, 1, 1.0}});

    DualSimplexSolver solver;
    solver.setVerbose(false);
    solver.load(lp);
    REQUIRE(solver.solve().status == Status::Optimal);
    auto opts = solver.getOptions();
    opts.deadline = Deadline::after(0.0);
    solver.setOptions(opts);

    ReliabilityBranching branching;
    branching.reset(lp.num_cols);
    std::vector<Real> primal = {0.5, 0.5};
    std::vector<Real> lb = lp.col_lower;
    std::vector<Real> ub = lp.col_upper;
    BranchingTelemetry telemetry;
    const auto selection = branching.select(solver, lp, primal, lb, ub, -1.0, false, telemetry);

    CHECK(telemetry.strong_branch_calls == 1);
    CHECK(telemetry.strong_branch_probes == 1);
    CHECK(telemetry.pseudocost_updates_strong == 0);
    CHECK(branching.downReliability(0) == 0);
    CHECK(branching.upReliability(0) == 0);
    CHECK(selection.variable >= 0);
    CHECK(selection.used_pseudocost);
}

TEST_CASE("pseudocostNodeEstimate matches the hand-computed sum", "[branching][search]") {
    LpProblem problem;
    problem.num_cols = 3;
    problem.col_type = {VarType::Integer, VarType::Binary, VarType::Continuous};
    problem.col_lower = {0.0, 0.0, 0.0};
    problem.col_upper = {10.0, 1.0, 10.0};

    ReliabilityBranching rule;
    rule.reset(problem.num_cols);
    rule.setPseudocostFallback(3.0);
    rule.updatePseudoCost(0, /*up_direction=*/false, 2.0);  // down cost of x0 = 2
    rule.updatePseudoCost(1, /*up_direction=*/true, 4.0);   // up cost of x1 = 4

    // x0 = 1.25: min(0.25 * 2.0, 0.75 * 3.0) = 0.5   (up unobserved -> fallback)
    // x1 = 0.50: min(0.50 * 3.0, 0.50 * 4.0) = 1.5   (down unobserved -> fallback)
    // x2 continuous -> skipped.
    const std::vector<Real> primals = {1.25, 0.5, 0.75};
    const Real estimate = pseudocostNodeEstimate(rule, problem, primals, 10.0);
    CHECK_THAT(estimate, Catch::Matchers::WithinAbs(12.0, 1e-12));
}

TEST_CASE("pseudocostNodeEstimate is the LP bound without fractional variables",
          "[branching][search]") {
    LpProblem problem;
    problem.num_cols = 2;
    problem.col_type = {VarType::Integer, VarType::Continuous};
    problem.col_lower = {0.0, 0.0};
    problem.col_upper = {10.0, 10.0};

    ReliabilityBranching rule;
    rule.reset(problem.num_cols);

    const std::vector<Real> primals = {3.0, 1.5};
    CHECK_THAT(pseudocostNodeEstimate(rule, problem, primals, -7.5),
               Catch::Matchers::WithinAbs(-7.5, 1e-12));

    // A non-finite LP bound is passed through untouched (no NaN arithmetic).
    const std::vector<Real> fractional = {3.5, 1.5};
    CHECK(pseudocostNodeEstimate(rule, problem, fractional, -kInf) == -kInf);
}
