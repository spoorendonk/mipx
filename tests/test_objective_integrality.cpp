#include "mipx/mip_solver.h"
#include "mipx/objective_integrality.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>

using namespace mipx;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

/// min obj^T x over a box, with no rows.
LpProblem boxProblem(std::vector<Real> obj, std::vector<VarType> col_type,
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

LpProblem integerBox(std::vector<Real> obj) {
    const auto n = obj.size();
    return boxProblem(std::move(obj), std::vector<VarType>(n, VarType::Integer),
                      std::vector<Real>(n, 0.0), std::vector<Real>(n, 5.0));
}

/// The bounded general-integer knapsack of the objective propagation tests,
/// its objective times `scale`. With cuts off it branches a few dozen times.
LpProblem integerKnapsack(Real scale) {
    constexpr Index n = 10;
    LpProblem lp;
    lp.name = "objint_knapsack";
    lp.sense = Sense::Minimize;
    lp.num_cols = n;
    lp.obj = {-15.0, -14.0, -17.0, -17.0, -2.0, -17.0, -14.0, -8.0, -14.0, -13.0};
    for (Real& c : lp.obj) {
        c *= scale;
    }
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

MipResult solveKnapsack(Real scale, bool integrality, bool propagation, MipSolver& solver) {
    solver.setVerbose(false);
    solver.setNumThreads(1);
    solver.setGapTolerance(0.0);
    solver.setCutsEnabled(false);
    solver.setObjectiveIntegralityEnabled(integrality);
    solver.setObjectivePropagationEnabled(propagation);
    solver.load(integerKnapsack(scale));
    return solver.solve();
}

/// Vertex cover of an odd cycle of length `n`: min sum x, x_i + x_{i+1} >= 1.
/// The LP sets every x to 1/2, (n+1)/2 - 1/2 below the integer optimum, so
/// every node bound within 1 of it is pruned by the lattice and not without.
LpProblem oddCycleCover(Index n) {
    LpProblem lp;
    lp.name = "objint_odd_cycle";
    lp.sense = Sense::Minimize;
    lp.num_cols = n;
    lp.obj.assign(n, 1.0);
    lp.col_lower.assign(n, 0.0);
    lp.col_upper.assign(n, 1.0);
    lp.col_type.assign(n, VarType::Integer);
    lp.num_rows = n;
    lp.row_lower.assign(n, 1.0);
    lp.row_upper.assign(n, kInf);
    std::vector<Triplet> trips;
    for (Index i = 0; i < n; ++i) {
        trips.push_back({i, i, 1.0});
        trips.push_back({i, (i + 1) % n, 1.0});
    }
    lp.matrix = SparseMatrix(n, n, std::move(trips));
    return lp;
}

MipResult solveOddCycle(bool integrality, MipSolver& solver) {
    solver.setVerbose(false);
    solver.setNumThreads(1);
    solver.setGapTolerance(0.0);
    solver.setCutsEnabled(false);
    solver.setObjectiveIntegralityEnabled(integrality);
    solver.load(oddCycleCover(7));
    return solver.solve();
}

}  // namespace

TEST_CASE("ObjectiveLattice: integer coefficients give their gcd", "[objint]") {
    LpProblem lp = integerBox({2.0, -4.0, 6.0});
    lp.obj_offset = 0.5;
    const ObjectiveLattice lattice = detectObjectiveLattice(lp);
    REQUIRE(lattice.detected());
    CHECK_THAT(lattice.scale, WithinAbs(2.0, 1e-12));
    CHECK_THAT(lattice.offset, WithinAbs(0.5, 1e-12));
}

TEST_CASE("ObjectiveLattice: fractional coefficients scale to a common lattice", "[objint]") {
    const ObjectiveLattice halves = detectObjectiveLattice(integerBox({0.5, 1.5, 3.0}));
    REQUIRE(halves.detected());
    CHECK_THAT(halves.scale, WithinAbs(0.5, 1e-12));

    const ObjectiveLattice thirds = detectObjectiveLattice(integerBox({1.0 / 3.0, 2.0 / 3.0}));
    REQUIRE(thirds.detected());
    CHECK_THAT(thirds.scale, WithinRel(1.0 / 3.0, 1e-12));

    // 0.3 and 0.25: denominators 10 and 4, lcm 20, gcd(6, 5) = 1.
    const ObjectiveLattice mixed = detectObjectiveLattice(integerBox({0.3, 0.25}));
    REQUIRE(mixed.detected());
    CHECK_THAT(mixed.scale, WithinRel(0.05, 1e-12));
}

TEST_CASE("ObjectiveLattice: no lattice without a rational common scale", "[objint]") {
    CHECK_FALSE(detectObjectiveLattice(integerBox({1.0, std::sqrt(2.0)})).detected());
    CHECK_FALSE(detectObjectiveLattice(integerBox({0.0, 0.0})).detected());
    // Denominators 997, 991 and 983 are each allowed; their lcm is past 1e6.
    CHECK_FALSE(
        detectObjectiveLattice(integerBox({1.0 / 997.0, 1.0 / 991.0, 1.0 / 983.0})).detected());
    // A denominator past 1000.
    CHECK_FALSE(detectObjectiveLattice(integerBox({1.0 / 1009.0})).detected());
}

TEST_CASE("ObjectiveLattice: continuous columns", "[objint]") {
    // An unfixed continuous column with a cost rules the lattice out.
    CHECK_FALSE(
        detectObjectiveLattice(
            boxProblem({1.0, 1.0}, {VarType::Integer, VarType::Continuous}, {0.0, 0.0}, {5.0, 5.0}))
            .detected());
    // One fixed by its bounds is a constant, folded into the offset.
    const ObjectiveLattice fixed = detectObjectiveLattice(
        boxProblem({1.0, 0.7}, {VarType::Integer, VarType::Continuous}, {0.0, 0.3}, {5.0, 0.3}));
    REQUIRE(fixed.detected());
    CHECK_THAT(fixed.scale, WithinAbs(1.0, 1e-12));
    CHECK_THAT(fixed.offset, WithinAbs(0.21, 1e-12));
    // A continuous column without a cost does not matter.
    CHECK(detectObjectiveLattice(boxProblem({1.0, 0.0}, {VarType::Integer, VarType::Continuous},
                                            {0.0, 0.0}, {5.0, 5.0}))
              .detected());
    // A semi-continuous one is 0 or its fixed value, so not a constant.
    CHECK_FALSE(
        detectObjectiveLattice(boxProblem({1.0, 0.7}, {VarType::Integer, VarType::SemiContinuous},
                                          {0.0, 0.3}, {5.0, 0.3}))
            .detected());
}

TEST_CASE("ObjectiveLattice: pruneBound", "[objint]") {
    const ObjectiveLattice none;
    CHECK(none.pruneBound(10.0) == 10.0 - 1e-6);

    ObjectiveLattice unit;
    unit.scale = 1.0;
    unit.offset = 0.0;
    // On the lattice: one step down, plus the margin.
    CHECK_THAT(unit.pruneBound(10.0), WithinAbs(9.001, 1e-9));
    CHECK_THAT(unit.pruneBound(-10.0), WithinAbs(-10.999, 1e-9));
    // Off it, the last lattice point below the incumbent still counts as an
    // improvement and stays below the bound.
    CHECK_THAT(unit.pruneBound(10.4), WithinAbs(10.001, 1e-9));
    // An incumbent a hair above a lattice point keeps that point too.
    CHECK(unit.pruneBound(10.0 + 1e-5) > 10.0);
    // Where the relative margin swallows the step, the standard bound.
    CHECK(unit.pruneBound(1e9) == 1e9 - 1e-6);
    CHECK(unit.pruneBound(kInf) == kInf);

    ObjectiveLattice shifted;
    shifted.scale = 0.5;
    shifted.offset = 0.21;
    CHECK_THAT(shifted.pruneBound(3.21), WithinAbs(2.7105, 1e-9));
}

TEST_CASE("MipSolver: objective integrality prunes at incumbent - 1 and keeps the optimum",
          "[mip][objint]") {
    MipSolver off;
    const MipResult off_result = solveOddCycle(false, off);
    REQUIRE(off_result.status == Status::Optimal);
    CHECK_FALSE(off.getObjectiveLattice().detected());

    MipSolver on;
    const MipResult on_result = solveOddCycle(true, on);
    REQUIRE(on_result.status == Status::Optimal);
    REQUIRE(on.getObjectiveLattice().detected());
    CHECK_THAT(on.getObjectiveLattice().scale, WithinAbs(1.0, 1e-12));
    CHECK_THAT(on_result.objective, WithinAbs(4.0, 1e-6));
    CHECK_THAT(off_result.objective, WithinAbs(4.0, 1e-6));
    // The tighter cutoff in use: nodes within 1 of the incumbent are pruned.
    CHECK(on_result.nodes < off_result.nodes);
}

TEST_CASE("MipSolver: objective integrality on a scaled objective and with propagation",
          "[mip][objint]") {
    MipSolver off;
    const MipResult off_result = solveKnapsack(0.5, false, true, off);
    REQUIRE(off_result.status == Status::Optimal);

    MipSolver on;
    const MipResult on_result = solveKnapsack(0.5, true, true, on);
    REQUIRE(on_result.status == Status::Optimal);
    REQUIRE(on.getObjectiveLattice().detected());
    CHECK_THAT(on.getObjectiveLattice().scale, WithinAbs(0.5, 1e-12));
    CHECK_THAT(on_result.objective, WithinAbs(off_result.objective, 1e-6));
    CHECK(on_result.nodes <= off_result.nodes);
}
