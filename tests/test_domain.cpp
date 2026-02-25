#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "mipx/domain.h"

using namespace mipx;
using Catch::Matchers::WithinAbs;

namespace {

// Helper to build a simple LpProblem.
LpProblem makeProblem(Index num_cols, Index num_rows,
                      std::vector<Triplet> triplets,
                      std::vector<Real> row_lower,
                      std::vector<Real> row_upper,
                      std::vector<Real> col_lower,
                      std::vector<Real> col_upper,
                      std::vector<VarType> col_type = {}) {
    LpProblem prob;
    prob.num_cols = num_cols;
    prob.num_rows = num_rows;
    prob.matrix = SparseMatrix(num_rows, num_cols, std::move(triplets));
    prob.row_lower = std::move(row_lower);
    prob.row_upper = std::move(row_upper);
    prob.col_lower = std::move(col_lower);
    prob.col_upper = std::move(col_upper);
    prob.obj.assign(num_cols, 0.0);
    if (col_type.empty()) {
        prob.col_type.assign(num_cols, VarType::Continuous);
    } else {
        prob.col_type = std::move(col_type);
    }
    return prob;
}

}  // namespace

TEST_CASE("Domain: simple bound tightening", "[domain]") {
    // x + y <= 5, x >= 3, y >= 0
    // Should tighten y <= 2.
    auto prob = makeProblem(
        2, 1,
        {{0, 0, 1.0}, {0, 1, 1.0}},
        {-kInf}, {5.0},
        {3.0, 0.0}, {kInf, kInf});

    DomainPropagator dp;
    dp.load(prob);
    REQUIRE(dp.propagate());
    CHECK_THAT(dp.getUpper(1), WithinAbs(2.0, 1e-7));
    CHECK_THAT(dp.getLower(0), WithinAbs(3.0, 1e-7));
}

TEST_CASE("Domain: integer rounding", "[domain]") {
    // 2x <= 5, x >= 0, x integer => x <= 2
    auto prob = makeProblem(
        1, 1,
        {{0, 0, 2.0}},
        {-kInf}, {5.0},
        {0.0}, {kInf},
        {VarType::Integer});

    DomainPropagator dp;
    dp.load(prob);
    REQUIRE(dp.propagate());
    CHECK_THAT(dp.getUpper(0), WithinAbs(2.0, 1e-7));
}

TEST_CASE("Domain: infeasibility detection", "[domain]") {
    // Constraint with rl > ru is infeasible.
    auto prob = makeProblem(
        1, 1,
        {{0, 0, 1.0}},
        {5.0}, {3.0},
        {0.0}, {kInf});

    DomainPropagator dp;
    dp.load(prob);
    CHECK_FALSE(dp.propagate());
}

TEST_CASE("Domain: infeasibility from bound tightening", "[domain]") {
    // x + y <= 3, x >= 2, y >= 2 => x <= 1 and y <= 1, contradicts.
    auto prob = makeProblem(
        2, 1,
        {{0, 0, 1.0}, {0, 1, 1.0}},
        {-kInf}, {3.0},
        {2.0, 2.0}, {kInf, kInf});

    DomainPropagator dp;
    dp.load(prob);
    CHECK_FALSE(dp.propagate());
}

TEST_CASE("Domain: checkpoint/restore", "[domain]") {
    // x + y <= 10, x >= 0, y >= 0
    auto prob = makeProblem(
        2, 1,
        {{0, 0, 1.0}, {0, 1, 1.0}},
        {-kInf}, {10.0},
        {0.0, 0.0}, {kInf, kInf});

    DomainPropagator dp;
    dp.load(prob);
    REQUIRE(dp.propagate());
    CHECK_THAT(dp.getUpper(0), WithinAbs(10.0, 1e-7));
    CHECK_THAT(dp.getUpper(1), WithinAbs(10.0, 1e-7));

    // Push checkpoint, then tighten x >= 7.
    dp.pushCheckpoint();
    dp.setBound(0, 7.0, dp.getUpper(0));
    REQUIRE(dp.propagate());
    CHECK_THAT(dp.getLower(0), WithinAbs(7.0, 1e-7));
    CHECK_THAT(dp.getUpper(1), WithinAbs(3.0, 1e-7));
    CHECK(dp.numTightened() > 0);

    // Restore — bounds should go back.
    dp.popCheckpoint();
    CHECK_THAT(dp.getLower(0), WithinAbs(0.0, 1e-7));
    CHECK_THAT(dp.getUpper(1), WithinAbs(10.0, 1e-7));
}

TEST_CASE("Domain: chain of implications", "[domain]") {
    // x + y <= 10
    // y + z <= 6
    // x >= 5
    // All non-negative.
    // => y <= 5 (from row 1), z <= 6 (from row 2 with y >= 0).
    auto prob = makeProblem(
        3, 2,
        {{0, 0, 1.0}, {0, 1, 1.0}, {1, 1, 1.0}, {1, 2, 1.0}},
        {-kInf, -kInf}, {10.0, 6.0},
        {5.0, 0.0, 0.0}, {kInf, kInf, kInf});

    DomainPropagator dp;
    dp.load(prob);
    REQUIRE(dp.propagate());
    CHECK_THAT(dp.getUpper(1), WithinAbs(5.0, 1e-7));
    CHECK_THAT(dp.getUpper(2), WithinAbs(6.0, 1e-7));
}

TEST_CASE("Domain: free variables", "[domain]") {
    // x + y <= 10, x free, y free
    // Both free => act_min = -inf (2 inf contributions), act_max = +inf (2 inf).
    // No tightening possible.
    auto prob = makeProblem(
        2, 1,
        {{0, 0, 1.0}, {0, 1, 1.0}},
        {-kInf}, {10.0},
        {-kInf, -kInf}, {kInf, kInf});

    DomainPropagator dp;
    dp.load(prob);
    REQUIRE(dp.propagate());
    CHECK(dp.getUpper(0) == kInf);
    CHECK(dp.getUpper(1) == kInf);
    CHECK(dp.getLower(0) == -kInf);
    CHECK(dp.getLower(1) == -kInf);
}

TEST_CASE("Domain: no tightening when already tight", "[domain]") {
    // x + y <= 10, x in [0, 5], y in [0, 5]
    // Activity max = 10 = row_upper, so no tightening needed.
    auto prob = makeProblem(
        2, 1,
        {{0, 0, 1.0}, {0, 1, 1.0}},
        {-kInf}, {10.0},
        {0.0, 0.0}, {5.0, 5.0});

    DomainPropagator dp;
    dp.load(prob);
    dp.pushCheckpoint();
    REQUIRE(dp.propagate());
    CHECK(dp.numTightened() == 0);
    CHECK_THAT(dp.getUpper(0), WithinAbs(5.0, 1e-7));
    CHECK_THAT(dp.getUpper(1), WithinAbs(5.0, 1e-7));
}

TEST_CASE("Domain: negative coefficients", "[domain]") {
    // -2x + y = 4, x in [0, 10], y in [0, 10]
    // => x <= 3, y >= 4.
    auto prob = makeProblem(
        2, 1,
        {{0, 0, -2.0}, {0, 1, 1.0}},
        {4.0}, {4.0},
        {0.0, 0.0}, {10.0, 10.0});

    DomainPropagator dp;
    dp.load(prob);
    REQUIRE(dp.propagate());
    CHECK_THAT(dp.getUpper(0), WithinAbs(3.0, 1e-7));
    CHECK_THAT(dp.getLower(1), WithinAbs(4.0, 1e-7));
}

TEST_CASE("Domain: binary variable fixing", "[domain]") {
    // 3x + 2y >= 5, x binary, y binary
    // act_max = 5 = rl, so both must be 1.
    auto prob = makeProblem(
        2, 1,
        {{0, 0, 3.0}, {0, 1, 2.0}},
        {5.0}, {kInf},
        {0.0, 0.0}, {1.0, 1.0},
        {VarType::Binary, VarType::Binary});

    DomainPropagator dp;
    dp.load(prob);
    REQUIRE(dp.propagate());
    CHECK_THAT(dp.getLower(0), WithinAbs(1.0, 1e-7));
    CHECK_THAT(dp.getLower(1), WithinAbs(1.0, 1e-7));
}
