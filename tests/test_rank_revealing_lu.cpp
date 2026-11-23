#include "lp/rank_revealing_lu.h"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstddef>
#include <random>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace mipx;

namespace {

using Column = std::vector<std::pair<Index, Real>>;

/// A CSC matrix the factorization's spans point into.
struct Csc {
    Index num_row = 0;
    std::vector<Index> start{0};
    std::vector<Index> index;
    std::vector<Real> value;
};

Csc makeCsc(Index num_row, const std::vector<Column>& cols) {
    Csc a;
    a.num_row = num_row;
    for (const Column& col : cols) {
        for (const auto& [i, v] : col) {
            a.index.push_back(i);
            a.value.push_back(v);
        }
        a.start.push_back(static_cast<Index>(a.index.size()));
    }
    return a;
}

std::vector<Index> toVector(std::span<const Index> s) {
    return {s.begin(), s.end()};
}

/// Rank of the dense matrix by Gaussian elimination with partial pivoting,
/// entries below `tol` times the largest counting as zero.
Index denseRank(Index num_row, const std::vector<Column>& cols, Real tol) {
    const auto n = static_cast<Index>(cols.size());
    std::vector<std::vector<Real>> a(num_row, std::vector<Real>(n, 0.0));
    Real max_abs = 0.0;
    for (Index j = 0; j < n; ++j) {
        for (const auto& [i, v] : cols[j]) {
            a[i][j] = v;
            max_abs = std::max(max_abs, std::fabs(v));
        }
    }
    Index rank = 0;
    for (Index j = 0; j < n && rank < num_row; ++j) {
        Index best = rank;
        for (Index i = rank + 1; i < num_row; ++i) {
            if (std::fabs(a[i][j]) > std::fabs(a[best][j])) {
                best = i;
            }
        }
        if (std::fabs(a[best][j]) <= tol * max_abs) {
            continue;
        }
        std::swap(a[best], a[rank]);
        for (Index i = rank + 1; i < num_row; ++i) {
            const Real f = a[i][j] / a[rank][j];
            for (Index k = j; k < n; ++k) {
                a[i][k] -= f * a[rank][k];
            }
        }
        ++rank;
    }
    return rank;
}

Index factor(RankRevealingLU& lu, const Csc& a) {
    lu.setup(a.num_row, a.start, a.index, a.value);
    return lu.build();
}

}  // namespace

// The expected no-pivot sets below are what HiGHS 1.16's HFactor::build
// returns on the same matrices (checked against HiGHS itself): which column
// of a dependent set goes unpivoted follows from its pivot order.

TEST_CASE("RankRevealingLU: a full-rank square matrix has no deficiency",
          "[lu][rank_revealing_lu]") {
    const Csc a = makeCsc(3, {{{0, 2}, {1, 1}}, {{0, 1}, {1, 3}, {2, 1}}, {{1, 1}, {2, 4}}});
    RankRevealingLU lu;
    CHECK(factor(lu, a) == 0);
    CHECK(lu.rankDeficiency() == 0);
    CHECK(lu.varWithNoPivot().empty());
    CHECK(lu.rowWithNoPivot().empty());
    CHECK(lu.colWithNoPivot().empty());
}

TEST_CASE("RankRevealingLU: full column rank with fewer columns than rows",
          "[lu][rank_revealing_lu]") {
    const Csc a = makeCsc(4, {{{0, 1}, {2, 2}}, {{1, 3}, {2, 1}, {3, 1}}});
    RankRevealingLU lu;
    // The columns' deficiency is 0; the matrix's counts the two rows no
    // column can cover, each left to a fictitious column (-1).
    CHECK(factor(lu, a) == 0);
    CHECK(lu.rankDeficiency() == 2);
    CHECK(toVector(lu.varWithNoPivot()) == std::vector<Index>{-1, -1});
    CHECK(toVector(lu.rowWithNoPivot()) == std::vector<Index>{2, 3});
    CHECK(toVector(lu.colWithNoPivot()) == std::vector<Index>{2, 3});
}

TEST_CASE("RankRevealingLU: more columns than rows leaves the excess unpivoted",
          "[lu][rank_revealing_lu]") {
    const Csc a = makeCsc(2, {{{0, 1}, {1, 1}}, {{0, 1}, {1, -1}}, {{0, 2}}});
    RankRevealingLU lu;
    CHECK(factor(lu, a) == 1);
    CHECK(lu.rankDeficiency() == 1);
    CHECK(toVector(lu.varWithNoPivot()) == std::vector<Index>{1});
    // A fictitious row past the last.
    CHECK(toVector(lu.rowWithNoPivot()) == std::vector<Index>{2});
    CHECK(toVector(lu.colWithNoPivot()) == std::vector<Index>{1});
}

TEST_CASE("RankRevealingLU: a multiple of a unit column has no pivot", "[lu][rank_revealing_lu]") {
    // Column 0 is a unit column and pivots first, so its multiple, column 1,
    // finds its only row taken.
    const Csc a = makeCsc(3, {{{0, 1}}, {{0, 2}}, {{1, 1}, {2, 1}}});
    RankRevealingLU lu;
    CHECK(factor(lu, a) == 1);
    CHECK(toVector(lu.varWithNoPivot()) == std::vector<Index>{1});
    CHECK(toVector(lu.rowWithNoPivot()) == std::vector<Index>{2});
}

TEST_CASE("RankRevealingLU: the Markowitz order decides which dependent column is left",
          "[lu][rank_revealing_lu]") {
    // Column 2 = column 0 + column 1. No singletons, so the kernel searches:
    // it pivots column 1 on row 2, then column 2 on row 0, which cancels
    // column 0 exactly, and column 0 is the one left without a pivot.
    const Csc a = makeCsc(3, {{{0, 1}, {1, 2}}, {{1, 1}, {2, 1}}, {{0, 1}, {1, 3}, {2, 1}}});
    RankRevealingLU lu;
    CHECK(factor(lu, a) == 1);
    CHECK(lu.rankDeficiency() == 1);
    CHECK(toVector(lu.varWithNoPivot()) == std::vector<Index>{0});
    CHECK(toVector(lu.rowWithNoPivot()) == std::vector<Index>{1});
    CHECK(toVector(lu.colWithNoPivot()) == std::vector<Index>{0});
}

TEST_CASE("RankRevealingLU: several dependent columns", "[lu][rank_revealing_lu]") {
    // Rank 3: column 2 = 2 * column 0, column 3 = column 0 - column 1.
    const std::vector<Column> cols = {{{0, 1}, {1, 2}},
                                      {{2, 3}, {3, 1}},
                                      {{0, 2}, {1, 4}},
                                      {{0, 1}, {1, 2}, {2, -3}, {3, -1}},
                                      {{4, 5}}};
    const Csc a = makeCsc(5, cols);
    RankRevealingLU lu;
    CHECK(factor(lu, a) == 2);
    CHECK(denseRank(5, cols, 1e-12) == 3);
    CHECK(toVector(lu.varWithNoPivot()) == std::vector<Index>{0, 2});
    CHECK(toVector(lu.rowWithNoPivot()) == std::vector<Index>{1, 3});
    CHECK(toVector(lu.colWithNoPivot()) == std::vector<Index>{0, 2});
}

TEST_CASE("RankRevealingLU: a dependent column with fewer columns than rows",
          "[lu][rank_revealing_lu]") {
    // Column 1 = 2 * column 0; the real column with no pivot comes first,
    // then the fictitious one.
    const Csc a = makeCsc(4, {{{0, 1}, {1, 1}}, {{0, 2}, {1, 2}}, {{2, 1}, {3, 1}}});
    RankRevealingLU lu;
    CHECK(factor(lu, a) == 1);
    CHECK(lu.rankDeficiency() == 2);
    CHECK(toVector(lu.varWithNoPivot()) == std::vector<Index>{0, -1});
    CHECK(toVector(lu.rowWithNoPivot()) == std::vector<Index>{1, 3});
    CHECK(toVector(lu.colWithNoPivot()) == std::vector<Index>{0, 3});
}

TEST_CASE("RankRevealingLU: a near-combination is dependent within the pivot tolerance only",
          "[lu][rank_revealing_lu]") {
    // Columns (1, 1) and (1, 1 + eps): pivoting column 1 on row 0 leaves
    // -eps in column 0, row 1.
    auto near = [](Real eps) { return makeCsc(2, {{{0, 1}, {1, 1}}, {{0, 1}, {1, 1 + eps}}}); };
    RankRevealingLU lu;

    SECTION("below kHighsTiny the entry cancels") {
        const Csc a = near(1e-15);
        CHECK(factor(lu, a) == 1);
        CHECK(toVector(lu.varWithNoPivot()) == std::vector<Index>{0});
        CHECK(toVector(lu.rowWithNoPivot()) == std::vector<Index>{1});
    }
    SECTION("below the pivot tolerance the pivot is deferred and never taken") {
        const Csc a = near(5e-11);
        CHECK(factor(lu, a) == 1);
        CHECK(toVector(lu.varWithNoPivot()) == std::vector<Index>{0});
        CHECK(toVector(lu.rowWithNoPivot()) == std::vector<Index>{1});
    }
    SECTION("just above the pivot tolerance it is a pivot") {
        const Csc a = near(2e-10);
        CHECK(factor(lu, a) == 0);
        CHECK(lu.varWithNoPivot().empty());
    }
    SECTION("a larger pivot tolerance makes the same column dependent") {
        const Csc a = near(2e-10);
        lu.setup(a.num_row, a.start, a.index, a.value, RankRevealingLU::kDefaultPivotThreshold,
                 1e-9);
        CHECK(lu.build() == 1);
        CHECK(toVector(lu.varWithNoPivot()) == std::vector<Index>{0});
    }
}

TEST_CASE("RankRevealingLU: an empty column has no pivot", "[lu][rank_revealing_lu]") {
    const Csc a = makeCsc(3, {{{0, 2}, {1, 1}}, {}, {{1, 1}, {2, 3}}});
    RankRevealingLU lu;
    CHECK(factor(lu, a) == 1);
    CHECK(toVector(lu.varWithNoPivot()) == std::vector<Index>{1});
    CHECK(toVector(lu.rowWithNoPivot()) == std::vector<Index>{2});
    CHECK(toVector(lu.colWithNoPivot()) == std::vector<Index>{1});
}

TEST_CASE("RankRevealingLU: no columns, or no rows", "[lu][rank_revealing_lu]") {
    RankRevealingLU lu;
    SECTION("no columns: every row left to a fictitious column") {
        const Csc a = makeCsc(3, {});
        CHECK(factor(lu, a) == 0);
        CHECK(lu.rankDeficiency() == 3);
        CHECK(toVector(lu.varWithNoPivot()) == std::vector<Index>{-1, -1, -1});
        CHECK(toVector(lu.rowWithNoPivot()) == std::vector<Index>{0, 1, 2});
    }
    SECTION("no rows: every column dependent") {
        const Csc a = makeCsc(0, {{}, {}});
        CHECK(factor(lu, a) == 2);
        CHECK(toVector(lu.varWithNoPivot()) == std::vector<Index>{0, 1});
        CHECK(toVector(lu.rowWithNoPivot()) == std::vector<Index>{0, 1});
    }
}

TEST_CASE("RankRevealingLU: the deficiency is the rank's on random dependent matrices",
          "[lu][rank_revealing_lu]") {
    // Small integer entries and integer combinations, so the dense rank is
    // unambiguous, and the pivoted columns alone have full column rank.
    std::mt19937 rng(392);
    std::uniform_int_distribution<int> dim(1, 15);
    std::uniform_int_distribution<int> entry(-4, 4);
    std::uniform_int_distribution<int> coin(0, 3);
    for (int trial = 0; trial < 200; ++trial) {
        const Index m = dim(rng);
        const Index n = dim(rng);
        std::vector<Column> cols(n);
        for (Index j = 0; j < n; ++j) {
            if (j >= 2 && coin(rng) == 0) {
                // An integer combination of two earlier columns.
                const Index p = std::uniform_int_distribution<Index>(0, j - 1)(rng);
                const Index q = std::uniform_int_distribution<Index>(0, j - 1)(rng);
                std::vector<Real> dense(m, 0.0);
                const Real cp = entry(rng);
                const Real cq = entry(rng);
                for (const auto& [i, v] : cols[p]) {
                    dense[i] += cp * v;
                }
                for (const auto& [i, v] : cols[q]) {
                    dense[i] += cq * v;
                }
                for (Index i = 0; i < m; ++i) {
                    if (dense[i] != 0.0) {
                        cols[j].emplace_back(i, dense[i]);
                    }
                }
            } else {
                for (Index i = 0; i < m; ++i) {
                    if (coin(rng) == 0) {
                        const int v = entry(rng);
                        if (v != 0) {
                            cols[j].emplace_back(i, v);
                        }
                    }
                }
            }
        }
        const Csc a = makeCsc(m, cols);
        RankRevealingLU lu;
        const Index deficiency = factor(lu, a);
        const Index rank = denseRank(m, cols, 1e-9);
        INFO("trial " << trial << ", " << m << " x " << n);
        REQUIRE(deficiency == n - rank);
        std::vector<Column> pivoted;
        for (Index j = 0; j < n; ++j) {
            const auto none = lu.varWithNoPivot();
            if (std::find(none.begin(), none.end(), j) == none.end()) {
                pivoted.push_back(cols[j]);
            }
        }
        CHECK(denseRank(m, pivoted, 1e-9) == static_cast<Index>(pivoted.size()));
    }
}

TEST_CASE("RankRevealingLU: a tiny time limit times out the kernel", "[lu][rank_revealing_lu]") {
    // A circulant pattern, 3 entries per row and column: no singletons, so
    // all n columns reach the kernel, which first checks the time after 100
    // pivots. By then any positive limit this small has passed.
    const Index n = 500;
    std::mt19937 rng(7);
    std::uniform_real_distribution<Real> val(1.0, 2.0);
    std::vector<Column> cols(n);
    for (Index j = 0; j < n; ++j) {
        cols[j] = {{j, val(rng)}, {(j + 1) % n, val(rng)}, {(j + 3) % n, val(rng)}};
        std::sort(cols[j].begin(), cols[j].end());
    }
    const Csc a = makeCsc(n, cols);
    RankRevealingLU lu;

    lu.setup(a.num_row, a.start, a.index, a.value);
    lu.setTimeLimit(1e-12);
    CHECK(lu.build() == RankRevealingLU::kBuildKernelReturnTimeout);
    CHECK(lu.rankDeficiency() == 0);
    CHECK(lu.varWithNoPivot().empty());
    CHECK(std::isfinite(lu.minTimeBound()));

    // setup() clears the limit; without one, and with a generous one, the
    // same matrix factors to the same result.
    const Index unlimited = factor(lu, a);
    CHECK(unlimited >= 0);
    lu.setup(a.num_row, a.start, a.index, a.value);
    lu.setTimeLimit(1e6);
    CHECK(lu.build() == unlimited);
    CHECK(lu.maxTimeBound() > 0.0);
    // A negative limit is none.
    lu.setTimeLimit(-1.0);
    CHECK(lu.build() == unlimited);
}

TEST_CASE("RankRevealingLU: setup rejects a malformed matrix", "[lu][rank_revealing_lu]") {
    RankRevealingLU lu;
    const std::vector<Index> start{0, 2};
    const std::vector<Real> value{1.0, 2.0};
    const std::vector<Index> out_of_range{0, 3};
    CHECK_THROWS_AS(lu.setup(3, start, out_of_range, value), std::invalid_argument);
    const std::vector<Index> duplicate{1, 1};
    CHECK_THROWS_AS(lu.setup(3, start, duplicate, value), std::invalid_argument);
    const std::vector<Index> short_index{0};
    CHECK_THROWS_AS(lu.setup(3, start, short_index, std::span<const Real>(value).first(1)),
                    std::invalid_argument);
    CHECK_THROWS_AS(lu.setup(3, {}, {}, {}), std::invalid_argument);
}
