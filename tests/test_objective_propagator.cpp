#include "mipx/domain.h"
#include "mipx/mip_solver.h"
#include "mipx/objective_propagator.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using namespace mipx;
using Catch::Matchers::WithinAbs;

namespace {

/// min obj^T x over a box, with no rows unless the caller adds them.
LpProblem makeObjectiveProblem(std::vector<Real> obj, std::vector<VarType> col_type,
                               std::vector<Real> col_lower, std::vector<Real> col_upper) {
    LpProblem prob;
    prob.num_cols = static_cast<Index>(obj.size());
    prob.num_rows = 0;
    prob.sense = Sense::Minimize;
    prob.obj = std::move(obj);
    prob.col_type = std::move(col_type);
    prob.col_lower = std::move(col_lower);
    prob.col_upper = std::move(col_upper);
    prob.matrix = SparseMatrix(0, prob.num_cols, {});
    return prob;
}

/// min 10x + y, x and y binary.
LpProblem tenXPlusY() {
    return makeObjectiveProblem({10.0, 1.0}, {VarType::Binary, VarType::Binary}, {0.0, 0.0},
                                {1.0, 1.0});
}

/// A bounded general-integer knapsack (the reduced-cost fixing test model)
/// that branches a few dozen times with root cuts off.
LpProblem integerKnapsack() {
    constexpr Index n = 10;
    LpProblem lp;
    lp.name = "objprop_knapsack";
    lp.sense = Sense::Minimize;
    lp.num_cols = n;
    lp.obj = {-15.0, -14.0, -17.0, -17.0, -2.0, -17.0, -14.0, -8.0, -14.0, -13.0};
    lp.col_lower.assign(n, 0.0);
    lp.col_upper.assign(n, 5.0);
    lp.col_type.assign(n, VarType::Integer);
    lp.num_rows = 1;
    lp.row_lower = {-kInf};
    lp.row_upper = {82.0};
    const Real w[n] = {5.0, 5.0, 4.0, 9.0, 7.0, 5.0, 3.0, 5.0, 3.0, 1.0};
    std::vector<Triplet> trips;
    for (Index j = 0; j < n; ++j) {
        trips.push_back({0, j, w[j]});
    }
    lp.matrix = SparseMatrix(1, n, std::move(trips));
    return lp;
}

}  // namespace

TEST_CASE("ObjectivePropagator: the objective row alone forces a bound", "[objprop]") {
    const LpProblem lp = tenXPlusY();
    // 10x + y <= 5 leaves room for x = 0.5 at most, so x = 0 once rounded.
    ObjectivePropagator objective;
    objective.load(lp);
    std::vector<ObjectivePropagator::BoundUpdate> updates;
    REQUIRE(objective.propagate(5.0, lp.col_lower, lp.col_upper, updates));
    REQUIRE(updates.size() == 1);
    CHECK(updates[0].col == 0);
    CHECK(updates[0].new_lower == 0.0);
    CHECK(updates[0].new_upper == 0.0);

    SECTION("through DomainPropagator, and not without the objective attached") {
        DomainPropagator with;
        with.load(lp);
        with.setObjectiveCutoff(&objective, 5.0);
        REQUIRE(with.propagate());
        CHECK(with.getUpper(0) == 0.0);
        CHECK(with.getUpper(1) == 1.0);
        CHECK(with.numObjectiveTightenings() == 1);

        DomainPropagator without;
        without.load(lp);
        REQUIRE(without.propagate());
        CHECK(without.getUpper(0) == 1.0);
    }
}

TEST_CASE("ObjectivePropagator: min activity at the cutoff is infeasible", "[objprop]") {
    LpProblem lp = tenXPlusY();
    lp.col_lower[0] = 1.0;  // min activity 10
    ObjectivePropagator objective;
    objective.load(lp);
    std::vector<ObjectivePropagator::BoundUpdate> updates;
    CHECK_FALSE(objective.propagate(10.0, lp.col_lower, lp.col_upper, updates));
    CHECK(updates.empty());
    CHECK_FALSE(objective.propagate(9.0, lp.col_lower, lp.col_upper, updates));
    // Just above: still feasible, y is forced to 0 instead.
    REQUIRE(objective.propagate(10.5, lp.col_lower, lp.col_upper, updates));
    REQUIRE(updates.size() == 1);
    CHECK(updates[0].col == 1);
    CHECK(updates[0].new_upper == 0.0);

    DomainPropagator dp;
    dp.load(lp);
    dp.setObjectiveCutoff(&objective, 10.0);
    CHECK_FALSE(dp.propagate());
    CHECK(dp.objectiveInfeasible());
    CHECK(dp.numObjectiveTightenings() == 0);
}

TEST_CASE("ObjectivePropagator: offset and negative coefficients", "[objprop]") {
    // min 3 - 10x + y + 2z, z continuous in [0, 4].
    LpProblem lp = makeObjectiveProblem({-10.0, 1.0, 2.0},
                                        {VarType::Integer, VarType::Binary, VarType::Continuous},
                                        {0.0, 0.0, 0.0}, {2.0, 1.0, 4.0});
    lp.obj_offset = 3.0;
    ObjectivePropagator objective;
    objective.load(lp);
    std::vector<ObjectivePropagator::BoundUpdate> updates;
    // -10x + y + 2z <= -12: x >= 1.2 -> 2, and with x at most 2 the residual
    // is -20, leaving y + 2z <= 8: z stays at 4, y at 1.
    REQUIRE(objective.propagate(-9.0, lp.col_lower, lp.col_upper, updates));
    REQUIRE(updates.size() == 1);
    CHECK(updates[0].col == 0);
    CHECK(updates[0].new_lower == 2.0);
    CHECK(updates[0].new_upper == 2.0);

    // With x fixed at 2, cutoff -14 leaves y + 2z <= 3: z <= 1.5, y untouched.
    std::vector<Real> lower = {2.0, 0.0, 0.0};
    std::vector<Real> upper = {2.0, 1.0, 4.0};
    REQUIRE(objective.propagate(-14.0, lower, upper, updates));
    REQUIRE(updates.size() == 1);
    CHECK(updates[0].col == 2);
    CHECK_THAT(updates[0].new_upper, WithinAbs(1.5, 1e-12));
}

TEST_CASE("ObjectivePropagator: an infinite contribution limits what is tightened", "[objprop]") {
    // min x + y, y free below: only y is bounded by the row, x gets nothing.
    LpProblem lp = makeObjectiveProblem({1.0, 1.0}, {VarType::Continuous, VarType::Continuous},
                                        {0.0, -kInf}, {4.0, kInf});
    ObjectivePropagator objective;
    objective.load(lp);
    std::vector<ObjectivePropagator::BoundUpdate> updates;
    REQUIRE(objective.propagate(3.0, lp.col_lower, lp.col_upper, updates));
    REQUIRE(updates.size() == 1);
    CHECK(updates[0].col == 1);
    CHECK_THAT(updates[0].new_upper, WithinAbs(3.0, 1e-12));
    // No cutoff, nothing to do.
    REQUIRE(objective.propagate(kInf, lp.col_lower, lp.col_upper, updates));
    CHECK(updates.empty());
}

TEST_CASE("ObjectivePropagator: tightenings feed back into the rows", "[objprop]") {
    // min 10x + y + 0z with x + z >= 1: the cutoff forces x = 0, the row then z = 1.
    LpProblem lp =
        makeObjectiveProblem({10.0, 1.0, 0.0}, {VarType::Binary, VarType::Binary, VarType::Binary},
                             {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0});
    lp.num_rows = 1;
    lp.row_lower = {1.0};
    lp.row_upper = {kInf};
    lp.matrix = SparseMatrix(1, 3, {{0, 0, 1.0}, {0, 2, 1.0}});
    ObjectivePropagator objective;
    objective.load(lp);
    DomainPropagator dp;
    dp.load(lp);
    dp.setObjectiveCutoff(&objective, 5.0);
    REQUIRE(dp.propagate());
    CHECK(dp.getUpper(0) == 0.0);
    CHECK(dp.getLower(2) == 1.0);
}

TEST_CASE("ObjectiveCutoffBounds: re-propagates only on a lower cutoff", "[objprop]") {
    const LpProblem lp = tenXPlusY();
    ObjectivePropagator objective;
    objective.load(lp);
    ObjectiveCutoffBounds bounds;
    bounds.load(lp, &objective, nullptr, nullptr);

    std::vector<Real> lower = lp.col_lower;
    std::vector<Real> upper = lp.col_upper;
    std::vector<Index> tightened;
    REQUIRE(bounds.enforce(lower, upper, tightened));
    CHECK(tightened.empty());

    CHECK_FALSE(bounds.refresh(kInf));
    CHECK(bounds.refresh(20.0));
    CHECK(bounds.numTightenings() == 0);
    CHECK_FALSE(bounds.refresh(20.0));
    CHECK(bounds.refresh(5.0));
    CHECK(bounds.numTightenings() == 1);
    CHECK_FALSE(bounds.refresh(7.0));

    REQUIRE(bounds.enforce(lower, upper, tightened));
    REQUIRE(tightened.size() == 1);
    CHECK(tightened[0] == 0);
    CHECK(upper[0] == 0.0);

    // At or below the best the root can reach, nothing improves on it.
    CHECK(bounds.refresh(0.0));
    lower = lp.col_lower;
    upper = lp.col_upper;
    tightened.clear();
    CHECK_FALSE(bounds.enforce(lower, upper, tightened));
}

TEST_CASE("MipSolver: objective propagation tightens and keeps the optimum", "[mip][objprop]") {
    const LpProblem lp = integerKnapsack();

    MipSolver off;
    off.setVerbose(false);
    off.setNumThreads(1);
    off.setGapTolerance(0.0);
    off.setCutsEnabled(false);
    off.load(lp);
    const auto off_result = off.solve();
    REQUIRE(off_result.status == Status::Optimal);
    const ObjPropStats off_stats = off.getObjPropStats();
    CHECK(off_stats.node_tightenings == 0);
    CHECK(off_stats.repropagations == 0);

    MipSolver on;
    on.setVerbose(false);
    on.setNumThreads(1);
    on.setGapTolerance(0.0);
    on.setCutsEnabled(false);
    on.setObjectivePropagationEnabled(true);
    on.load(lp);
    const auto on_result = on.solve();
    REQUIRE(on_result.status == Status::Optimal);
    CHECK_THAT(on_result.objective, WithinAbs(off_result.objective, 1e-6));
    const ObjPropStats stats = on.getObjPropStats();
    CHECK(stats.repropagations > 0);
    CHECK(stats.node_tightenings > 0);
    CHECK(stats.global_tightenings > 0);
}
