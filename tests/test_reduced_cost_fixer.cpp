#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "mipx/reduced_cost_fixer.h"

using namespace mipx;
using Catch::Matchers::WithinAbs;

namespace {

LpProblem makeSimpleProblem(Index num_cols,
                            std::vector<VarType> col_type,
                            std::vector<Real> col_lower,
                            std::vector<Real> col_upper) {
    LpProblem prob;
    prob.num_cols = num_cols;
    prob.num_rows = 0;
    prob.sense = Sense::Minimize;
    prob.col_type = std::move(col_type);
    prob.col_lower = std::move(col_lower);
    prob.col_upper = std::move(col_upper);
    prob.obj.assign(num_cols, 1.0);
    return prob;
}

}  // namespace

TEST_CASE("ReducedCostFixer: load and check state", "[rcfixer]") {
    ReducedCostFixer fixer;
    REQUIRE_FALSE(fixer.loaded());

    auto prob = makeSimpleProblem(3,
        {VarType::Binary, VarType::Integer, VarType::Continuous},
        {0.0, 0.0, 0.0}, {1.0, 10.0, 100.0});
    fixer.load(prob);
    REQUIRE(fixer.loaded());
    CHECK(fixer.numGlobalFixings() == 0);
    CHECK_THAT(fixer.globalLower(0), WithinAbs(0.0, 1e-12));
    CHECK_THAT(fixer.globalUpper(0), WithinAbs(1.0, 1e-12));
}

TEST_CASE("ReducedCostFixer: global fixing tightens upper bound with positive RC",
          "[rcfixer]") {
    // Binary variable x with rc = 2.0, at lower bound.
    // gap = incumbent - lp_obj = 10.0 - 5.0 = 5.0
    // new_ub = lb + gap / rc = 0.0 + 5.0 / 2.0 = 2.5
    // For binary: floor(2.5 + 1e-6) = 2 => capped at 1.0 (original ub).
    // Integer variable y with rc = 3.0, at lower bound.
    // new_ub = 0.0 + 5.0 / 3.0 = 1.667 => floor(1.667 + 1e-6) = 1.
    auto prob = makeSimpleProblem(2,
        {VarType::Binary, VarType::Integer},
        {0.0, 0.0}, {1.0, 10.0});
    ReducedCostFixer fixer;
    fixer.load(prob);

    std::vector<Real> rc = {3.0, 3.0};
    std::vector<Real> primals = {0.0, 0.0};
    std::vector<Real> col_lower = {0.0, 0.0};
    std::vector<Real> col_upper = {1.0, 10.0};
    std::vector<Index> tightened;

    bool ok = fixer.applyGlobalFixing(rc, primals, 5.0, 10.0,
                                       col_lower, col_upper, tightened);
    REQUIRE(ok);
    // x: new_ub = 0 + 5/3 = 1.667, floor(1.667 + 1e-6) = 1.0, same as original.
    CHECK_THAT(col_upper[0], WithinAbs(1.0, 1e-9));
    // y: new_ub = 0 + 5/3 = 1.667, floor(1.667 + 1e-6) = 1.0, tightened from 10.
    CHECK_THAT(col_upper[1], WithinAbs(1.0, 1e-9));
    CHECK(tightened.size() >= 1);  // At least y was tightened.
    CHECK(fixer.stats().root_global_tightenings > 0);
}

TEST_CASE("ReducedCostFixer: global fixing tightens lower bound with negative RC",
          "[rcfixer]") {
    // Variable x in [0, 10], rc = -4.0 (at upper bound).
    // gap = 20.0 - 8.0 = 12.0
    // new_lb = ub + gap / rc = 10.0 + 12.0 / (-4.0) = 10.0 - 3.0 = 7.0
    // For integer: ceil(7.0 - 1e-6) = 7.
    auto prob = makeSimpleProblem(1,
        {VarType::Integer},
        {0.0}, {10.0});
    ReducedCostFixer fixer;
    fixer.load(prob);

    std::vector<Real> rc = {-4.0};
    std::vector<Real> primals = {10.0};
    std::vector<Real> col_lower = {0.0};
    std::vector<Real> col_upper = {10.0};
    std::vector<Index> tightened;

    bool ok = fixer.applyGlobalFixing(rc, primals, 8.0, 20.0,
                                       col_lower, col_upper, tightened);
    REQUIRE(ok);
    CHECK_THAT(col_lower[0], WithinAbs(7.0, 1e-9));
    CHECK(col_upper[0] == 10.0);
    REQUIRE(tightened.size() == 1);
    CHECK(tightened[0] == 0);
}

TEST_CASE("ReducedCostFixer: no fixing when gap is zero", "[rcfixer]") {
    auto prob = makeSimpleProblem(1,
        {VarType::Integer},
        {0.0}, {10.0});
    ReducedCostFixer fixer;
    fixer.load(prob);

    std::vector<Real> rc = {5.0};
    std::vector<Real> primals = {0.0};
    std::vector<Real> col_lower = {0.0};
    std::vector<Real> col_upper = {10.0};
    std::vector<Index> tightened;

    // incumbent == lp_objective => gap = 0 => no tightening.
    bool ok = fixer.applyGlobalFixing(rc, primals, 10.0, 10.0,
                                       col_lower, col_upper, tightened);
    REQUIRE(ok);
    CHECK(tightened.empty());
    CHECK(col_upper[0] == 10.0);
}

TEST_CASE("ReducedCostFixer: no fixing when no incumbent", "[rcfixer]") {
    auto prob = makeSimpleProblem(1,
        {VarType::Integer},
        {0.0}, {10.0});
    ReducedCostFixer fixer;
    fixer.load(prob);

    std::vector<Real> rc = {5.0};
    std::vector<Real> primals = {0.0};
    std::vector<Real> col_lower = {0.0};
    std::vector<Real> col_upper = {10.0};
    std::vector<Index> tightened;

    bool ok = fixer.applyGlobalFixing(rc, primals, 5.0, kInf,
                                       col_lower, col_upper, tightened);
    REQUIRE(ok);
    CHECK(tightened.empty());
}

TEST_CASE("ReducedCostFixer: local fixing at tree node", "[rcfixer]") {
    auto prob = makeSimpleProblem(2,
        {VarType::Integer, VarType::Continuous},
        {0.0, 0.0}, {10.0, 100.0});
    ReducedCostFixer fixer;
    fixer.load(prob);

    // Integer var with rc=6 => new_ub = 0 + 3/6 = 0.5, floor(0.5+1e-6) = 0 => fixed.
    // Continuous var with rc=2 => new_ub = 0 + 3/2 = 1.5, tightened from 100.
    std::vector<Real> rc = {6.0, 2.0};
    std::vector<Real> primals = {0.0, 0.0};
    std::vector<Real> col_lower = {0.0, 0.0};
    std::vector<Real> col_upper = {10.0, 100.0};
    std::vector<Index> tightened;

    bool ok = fixer.applyLocalFixing(rc, primals, 7.0, 10.0,
                                      col_lower, col_upper, tightened);
    REQUIRE(ok);
    CHECK_THAT(col_upper[0], WithinAbs(0.0, 1e-9));  // integer, fixed at 0
    CHECK_THAT(col_upper[1], WithinAbs(1.5, 1e-9));   // continuous, tightened
    CHECK(tightened.size() == 2);
    CHECK(fixer.stats().tree_local_fixings == 1);
    CHECK(fixer.stats().tree_local_tightenings == 1);
}

TEST_CASE("ReducedCostFixer: enforce global fixings at node", "[rcfixer]") {
    auto prob = makeSimpleProblem(2,
        {VarType::Integer, VarType::Integer},
        {0.0, 0.0}, {10.0, 10.0});
    ReducedCostFixer fixer;
    fixer.load(prob);

    // Create global fixings first.
    std::vector<Real> rc = {5.0, 5.0};
    std::vector<Real> primals = {0.0, 0.0};
    std::vector<Real> col_lower = {0.0, 0.0};
    std::vector<Real> col_upper = {10.0, 10.0};
    std::vector<Index> tightened;

    // gap = 20 - 5 = 15, new_ub = 0 + 15/5 = 3, floor(3 + 1e-6) = 3.
    bool ok = fixer.applyGlobalFixing(rc, primals, 5.0, 20.0,
                                       col_lower, col_upper, tightened);
    REQUIRE(ok);
    CHECK_THAT(col_upper[0], WithinAbs(3.0, 1e-9));
    CHECK_THAT(col_upper[1], WithinAbs(3.0, 1e-9));

    // Now enforce on fresh node bounds (reset to original).
    std::vector<Real> node_lower = {0.0, 0.0};
    std::vector<Real> node_upper = {10.0, 10.0};
    std::vector<Index> node_tightened;
    ok = fixer.enforceGlobalFixings(node_lower, node_upper, node_tightened);
    REQUIRE(ok);
    CHECK_THAT(node_upper[0], WithinAbs(3.0, 1e-9));
    CHECK_THAT(node_upper[1], WithinAbs(3.0, 1e-9));
    CHECK(node_tightened.size() == 2);
}

TEST_CASE("ReducedCostFixer: reset clears state", "[rcfixer]") {
    auto prob = makeSimpleProblem(1,
        {VarType::Integer},
        {0.0}, {10.0});
    ReducedCostFixer fixer;
    fixer.load(prob);

    std::vector<Real> rc = {5.0};
    std::vector<Real> primals = {0.0};
    std::vector<Real> col_lower = {0.0};
    std::vector<Real> col_upper = {10.0};
    std::vector<Index> tightened;
    fixer.applyGlobalFixing(rc, primals, 5.0, 20.0,
                             col_lower, col_upper, tightened);
    CHECK(fixer.numGlobalFixings() > 0);

    fixer.reset();
    CHECK(fixer.numGlobalFixings() == 0);
    CHECK(fixer.stats().root_global_fixings == 0);
    CHECK(fixer.stats().root_global_tightenings == 0);
}

TEST_CASE("ReducedCostFixer: continuous variable tightening", "[rcfixer]") {
    // Continuous variable: no integer rounding.
    auto prob = makeSimpleProblem(1,
        {VarType::Continuous},
        {0.0}, {100.0});
    ReducedCostFixer fixer;
    fixer.load(prob);

    std::vector<Real> rc = {4.0};
    std::vector<Real> primals = {0.0};
    std::vector<Real> col_lower = {0.0};
    std::vector<Real> col_upper = {100.0};
    std::vector<Index> tightened;

    // gap = 10 - 2 = 8, new_ub = 0 + 8/4 = 2.0 (no rounding for continuous).
    bool ok = fixer.applyGlobalFixing(rc, primals, 2.0, 10.0,
                                       col_lower, col_upper, tightened);
    REQUIRE(ok);
    CHECK_THAT(col_upper[0], WithinAbs(2.0, 1e-9));
}

TEST_CASE("ReducedCostFixer: small RC below tolerance is ignored", "[rcfixer]") {
    auto prob = makeSimpleProblem(1,
        {VarType::Integer},
        {0.0}, {10.0});
    ReducedCostFixer fixer;
    fixer.load(prob);

    // RC below kRcTol (1e-7) should not trigger tightening.
    std::vector<Real> rc = {1e-9};
    std::vector<Real> primals = {0.0};
    std::vector<Real> col_lower = {0.0};
    std::vector<Real> col_upper = {10.0};
    std::vector<Index> tightened;

    bool ok = fixer.applyGlobalFixing(rc, primals, 5.0, 20.0,
                                       col_lower, col_upper, tightened);
    REQUIRE(ok);
    CHECK(tightened.empty());
}

TEST_CASE("ReducedCostFixer: fixed variables are skipped", "[rcfixer]") {
    auto prob = makeSimpleProblem(1,
        {VarType::Integer},
        {5.0}, {5.0});  // Already fixed.
    ReducedCostFixer fixer;
    fixer.load(prob);

    std::vector<Real> rc = {10.0};
    std::vector<Real> primals = {5.0};
    std::vector<Real> col_lower = {5.0};
    std::vector<Real> col_upper = {5.0};
    std::vector<Index> tightened;

    bool ok = fixer.applyGlobalFixing(rc, primals, 5.0, 20.0,
                                       col_lower, col_upper, tightened);
    REQUIRE(ok);
    CHECK(tightened.empty());
}
