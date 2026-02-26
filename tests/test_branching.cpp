#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "mipx/branching.h"

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

    // Right child: x_0 >= 3 (lower bound).
    REQUIRE(right.parent_id == 5);
    REQUIRE(right.depth == 4);
    REQUIRE(right.branch.variable == 0);
    REQUIRE(right.branch.bound == 3.0);
    REQUIRE(right.branch.is_upper == false);
    REQUIRE(right.basis.size() == 2);
    REQUIRE(right.bound_changes.size() == 1);
}

TEST_CASE("createChildren accumulates bound changes", "[branching]") {
    BnbNode parent;
    parent.id = 2;
    parent.depth = 1;
    parent.bound_changes = {{0, 1.0, false}};  // existing change

    auto [left, right] = createChildren(parent, 1, 3.5);

    REQUIRE(left.bound_changes.size() == 2);
    REQUIRE(left.bound_changes[0].variable == 0);
    REQUIRE(left.bound_changes[1].variable == 1);

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
