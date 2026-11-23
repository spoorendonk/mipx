#include "presolve/presolve_matrix.h"

#include <catch2/catch_test_macros.hpp>
#include <map>
#include <random>
#include <utility>
#include <vector>

using namespace mipx;

namespace {

using Entries = std::map<std::pair<Index, Index>, Real>;

/// A row's live entries, strictly by column, through both forEachInRow and
/// rowIds; likewise columns by row.
Entries rowEntries(const PresolveMatrix& m) {
    Entries out;
    for (Index i = 0; i < m.numRows(); ++i) {
        Index last = -1;
        m.forEachInRow(i, [&](Index j, Real v) {
            CHECK(j > last);  // Sorted, no duplicates.
            last = j;
            out.emplace(std::pair{i, j}, v);
        });
        last = -1;
        for (const Index e : m.rowIds(i)) {
            CHECK(m.entryRow(e) == i);
            CHECK(m.entryCol(e) >= last);  // Dead ids sorted too.
            last = m.entryCol(e);
            if (m.isLive(e)) {
                CHECK(out.at({i, m.entryCol(e)}) == m.entryValue(e));
            }
        }
    }
    return out;
}

Entries colEntries(const PresolveMatrix& m) {
    Entries out;
    for (Index j = 0; j < m.numCols(); ++j) {
        Index last = -1;
        m.forEachInCol(j, [&](Index i, Real v) {
            CHECK(i > last);
            last = i;
            out.emplace(std::pair{i, j}, v);
        });
        last = -1;
        for (const Index e : m.colIds(j)) {
            CHECK(m.entryCol(e) == j);
            CHECK(m.entryRow(e) >= last);
            last = m.entryRow(e);
            if (m.isLive(e)) {
                CHECK(out.at({m.entryRow(e), j}) == m.entryValue(e));
            }
        }
    }
    return out;
}

/// Rows, columns, counts, lookups and the CSR export all agree with `want`.
void requireMatches(const PresolveMatrix& m, const Entries& want) {
    REQUIRE(rowEntries(m) == want);
    REQUIRE(colEntries(m) == want);
    REQUIRE(m.numEntries() == static_cast<Index>(want.size()));
    std::vector<Index> row_size(static_cast<std::size_t>(m.numRows()), 0);
    std::vector<Index> col_size(static_cast<std::size_t>(m.numCols()), 0);
    for (const auto& [ij, v] : want) {
        ++row_size[ij.first];
        ++col_size[ij.second];
    }
    // Every cell, absent ones included, from both lookup sides' view.
    for (Index i = 0; i < m.numRows(); ++i) {
        for (Index j = 0; j < m.numCols(); ++j) {
            const auto it = want.find({i, j});
            CHECK(m.contains(i, j) == (it != want.end()));
            CHECK(m.coeff(i, j) == (it != want.end() ? it->second : 0.0));
        }
    }
    for (Index i = 0; i < m.numRows(); ++i) {
        CHECK(m.rowSize(i) == row_size[i]);
    }
    for (Index j = 0; j < m.numCols(); ++j) {
        CHECK(m.colSize(j) == col_size[j]);
    }
    const SparseMatrix csr = m.toSparseMatrix();
    REQUIRE(csr.numRows() == m.numRows());
    REQUIRE(csr.numCols() == m.numCols());
    REQUIRE(csr.numNonzeros() == static_cast<Index>(want.size()));
    Entries exported;
    for (Index i = 0; i < csr.numRows(); ++i) {
        const auto row = csr.row(i);
        for (Index k = 0; k < row.size(); ++k) {
            if (k > 0) {
                CHECK(row.indices[k - 1] < row.indices[k]);  // Sorted.
            }
            exported[{i, row.indices[k]}] = row.values[k];
        }
    }
    CHECK(exported == want);
}

// [ 1 0 2 ]
// [ 0 3 0 ]
SparseMatrix smallMatrix() {
    return {2, 3, std::vector<Triplet>{{0, 0, 1.0}, {0, 2, 2.0}, {1, 1, 3.0}}};
}

}  // namespace

TEST_CASE("PresolveMatrix: built from CSR, rows and columns agree", "[presolve][matrix]") {
    const PresolveMatrix m(smallMatrix());
    CHECK(m.numRows() == 2);
    CHECK(m.numCols() == 3);
    requireMatches(m, {{{0, 0}, 1.0}, {{0, 2}, 2.0}, {{1, 1}, 3.0}});
    CHECK(m.coeff(1, 0) == 0.0);
    CHECK_FALSE(m.contains(1, 0));
}

TEST_CASE("PresolveMatrix: insert, change and delete", "[presolve][matrix]") {
    PresolveMatrix m(smallMatrix());
    m.set(1, 0, 4.0);  // Fill-in.
    requireMatches(m, {{{0, 0}, 1.0}, {{0, 2}, 2.0}, {{1, 0}, 4.0}, {{1, 1}, 3.0}});
    m.set(0, 2, -5.0);  // Change.
    requireMatches(m, {{{0, 0}, 1.0}, {{0, 2}, -5.0}, {{1, 0}, 4.0}, {{1, 1}, 3.0}});
    CHECK(m.erase(0, 0));
    CHECK_FALSE(m.erase(0, 0));
    requireMatches(m, {{{0, 2}, -5.0}, {{1, 0}, 4.0}, {{1, 1}, 3.0}});
    m.set(1, 1, 0.0);  // A zero stays an entry, as an override of 0 does.
    requireMatches(m, {{{0, 2}, -5.0}, {{1, 0}, 4.0}, {{1, 1}, 0.0}});
    m.set(0, 1, 0.0);  // So does a new one.
    requireMatches(m, {{{0, 1}, 0.0}, {{0, 2}, -5.0}, {{1, 0}, 4.0}, {{1, 1}, 0.0}});
    CHECK(m.erase(1, 1));
    CHECK(m.erase(0, 1));
    m.set(0, 0, 7.0);  // Back after an erase: a new entry, not the dead one.
    requireMatches(m, {{{0, 0}, 7.0}, {{0, 2}, -5.0}, {{1, 0}, 4.0}});
}

TEST_CASE("PresolveMatrix: explicit zeros in the input are entries", "[presolve][matrix]") {
    // CSR arrays as given, a stored 0 at (0, 1).
    const PresolveMatrix m(SparseMatrix(1, 3, {1.0, 0.0, 2.0}, {0, 1, 2}, {0, 3}));
    requireMatches(m, {{{0, 0}, 1.0}, {{0, 1}, 0.0}, {{0, 2}, 2.0}});
}

TEST_CASE("PresolveMatrix: inserts keep rows and columns sorted", "[presolve][matrix]") {
    // Fill-in lands between, before and after existing entries, and next to
    // dead ones of the same position.
    PresolveMatrix m(SparseMatrix(3, 5, std::vector<Triplet>{{1, 1, 1.0}, {1, 3, 3.0}}));
    m.set(1, 2, 2.0);
    m.set(1, 0, 0.5);
    m.set(1, 4, 4.0);
    m.set(0, 2, 5.0);
    m.set(2, 2, 6.0);
    CHECK(m.erase(1, 2));
    m.set(1, 2, 7.0);
    requireMatches(m, {{{0, 2}, 5.0},
                       {{1, 0}, 0.5},
                       {{1, 1}, 1.0},
                       {{1, 2}, 7.0},
                       {{1, 3}, 3.0},
                       {{1, 4}, 4.0},
                       {{2, 2}, 6.0}});
    std::vector<Index> cols;
    m.forEachInRow(1, [&](Index j, Real) { cols.push_back(j); });
    CHECK(cols == std::vector<Index>{0, 1, 2, 3, 4});
}

TEST_CASE("PresolveMatrix: clearing a row or a column", "[presolve][matrix]") {
    PresolveMatrix m(smallMatrix());
    m.set(1, 2, 6.0);
    m.clearCol(2);
    requireMatches(m, {{{0, 0}, 1.0}, {{1, 1}, 3.0}});
    m.clearRow(0);
    requireMatches(m, {{{1, 1}, 3.0}});
    m.set(0, 2, 8.0);  // A cleared row and column take entries again.
    requireMatches(m, {{{0, 2}, 8.0}, {{1, 1}, 3.0}});
}

TEST_CASE("PresolveMatrix: erasing while iterating a row", "[presolve][matrix]") {
    PresolveMatrix m(
        SparseMatrix(1, 4, std::vector<Triplet>{{0, 0, 1.0}, {0, 1, 2.0}, {0, 3, 4.0}}));
    std::vector<Index> seen;
    m.forEachInRow(0, [&](Index j, Real) {
        seen.push_back(j);
        if (j == 0) {
            CHECK(m.erase(0, 1));  // Not yet visited: skipped.
        }
        m.set(0, 3, 9.0);  // Changing a value in place is fine.
    });
    CHECK(seen == std::vector<Index>{0, 3});
    requireMatches(m, {{{0, 0}, 1.0}, {{0, 3}, 9.0}});
}

TEST_CASE("PresolveMatrix: inserting elsewhere while iterating", "[presolve][matrix]") {
    // Each visited entry inserts into another row and column, enough to grow
    // the pool past its reserve, so a reference into it would dangle.
    PresolveMatrix m(
        SparseMatrix(2, 3, std::vector<Triplet>{{0, 0, 1.0}, {0, 1, 2.0}, {0, 2, 3.0}}));
    Entries seen;
    m.forEachInRow(0, [&](const Index& j, const Real& v) {
        m.set(1, j, 10.0 * v);
        seen[{0, j}] = v;
    });
    CHECK(seen == Entries{{{0, 0}, 1.0}, {{0, 1}, 2.0}, {{0, 2}, 3.0}});
    Entries col_seen;
    m.forEachInCol(1, [&](const Index& i, const Real& v) {
        if (i == 0) {
            m.set(0, 0, 5.0);  // Another column of a visited row.
            m.erase(1, 1);     // Not yet visited: skipped.
        }
        col_seen[{i, 1}] = v;
    });
    CHECK(col_seen == Entries{{{0, 1}, 2.0}});
    requireMatches(m,
                   {{{0, 0}, 5.0}, {{0, 1}, 2.0}, {{0, 2}, 3.0}, {{1, 0}, 10.0}, {{1, 2}, 30.0}});
}

TEST_CASE("PresolveMatrix: mixed edits match a dense reference", "[presolve][matrix]") {
    // Many edits on a few rows and columns, so lists fill with dead ids and
    // compact, and lookups search both the row and the column side.
    constexpr Index kRows = 6;
    constexpr Index kCols = 5;
    std::mt19937 rng(12345);
    std::uniform_int_distribution<Index> row_of(0, kRows - 1);
    std::uniform_int_distribution<Index> col_of(0, kCols - 1);
    std::uniform_int_distribution<int> op_of(0, 99);
    std::uniform_int_distribution<int> val_of(-3, 3);

    PresolveMatrix m(SparseMatrix(kRows, kCols));
    Entries want;
    for (int step = 0; step < 5000; ++step) {
        const Index i = row_of(rng);
        const Index j = col_of(rng);
        const int op = op_of(rng);
        if (op < 55) {
            const Real v = static_cast<Real>(val_of(rng));
            m.set(i, j, v);
            want[{i, j}] = v;
        } else if (op < 90) {
            CHECK(m.erase(i, j) == (want.erase({i, j}) == 1));
        } else if (op < 95) {
            m.clearRow(i);
            std::erase_if(want, [i](const auto& e) { return e.first.first == i; });
        } else {
            m.clearCol(j);
            std::erase_if(want, [j](const auto& e) { return e.first.second == j; });
        }
        if (step % 50 == 0) {
            requireMatches(m, want);
        }
    }
    requireMatches(m, want);
}

TEST_CASE("PresolveMatrix: a list of dead ids compacts on insert", "[presolve][matrix]") {
    PresolveMatrix m(SparseMatrix(1, 1));
    for (int k = 0; k < 40; ++k) {
        m.set(0, 0, static_cast<Real>(k + 1));
        CHECK(m.erase(0, 0));
    }
    // Compacted at 8 dead ids each time, so never much longer than that.
    CHECK(m.rowIds(0).size() <= 9);
    CHECK(m.colIds(0).size() <= 9);
    m.set(0, 0, 2.0);
    requireMatches(m, {{{0, 0}, 2.0}});
}
