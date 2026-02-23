#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <vector>

#include "mipx/sparse_matrix.h"

using namespace mipx;
using Catch::Matchers::WithinAbs;

TEST_CASE("SparseMatrix: empty matrix", "[sparse_matrix]") {
    SparseMatrix A(3, 4);
    CHECK(A.numRows() == 3);
    CHECK(A.numCols() == 4);
    CHECK(A.numNonzeros() == 0);

    auto r = A.row(0);
    CHECK(r.size() == 0);

    CHECK(A.coeff(1, 2) == 0.0);
}

TEST_CASE("SparseMatrix: triplet construction", "[sparse_matrix]") {
    // 2x3 matrix:
    // [1  0  2]
    // [0  3  0]
    std::vector<Triplet> trips = {
        {0, 0, 1.0}, {0, 2, 2.0}, {1, 1, 3.0}};
    SparseMatrix A(2, 3, trips);

    CHECK(A.numRows() == 2);
    CHECK(A.numCols() == 3);
    CHECK(A.numNonzeros() == 3);

    CHECK(A.coeff(0, 0) == 1.0);
    CHECK(A.coeff(0, 1) == 0.0);
    CHECK(A.coeff(0, 2) == 2.0);
    CHECK(A.coeff(1, 0) == 0.0);
    CHECK(A.coeff(1, 1) == 3.0);
    CHECK(A.coeff(1, 2) == 0.0);
}

TEST_CASE("SparseMatrix: duplicate summing", "[sparse_matrix]") {
    std::vector<Triplet> trips = {
        {0, 0, 1.0}, {0, 0, 2.0}, {0, 0, 3.0}, {1, 1, 5.0}};
    SparseMatrix A(2, 2, trips);

    CHECK(A.coeff(0, 0) == 6.0);
    CHECK(A.coeff(1, 1) == 5.0);
    CHECK(A.numNonzeros() == 2);
}

TEST_CASE("SparseMatrix: CSR construction", "[sparse_matrix]") {
    // [0  4]
    // [5  6]
    std::vector<Real> vals = {4.0, 5.0, 6.0};
    std::vector<Index> cols = {1, 0, 1};
    std::vector<Index> starts = {0, 1, 3};

    SparseMatrix A(2, 2, std::move(vals), std::move(cols), std::move(starts));
    CHECK(A.coeff(0, 0) == 0.0);
    CHECK(A.coeff(0, 1) == 4.0);
    CHECK(A.coeff(1, 0) == 5.0);
    CHECK(A.coeff(1, 1) == 6.0);
}

TEST_CASE("SparseMatrix: row access", "[sparse_matrix]") {
    std::vector<Triplet> trips = {
        {0, 0, 1.0}, {0, 2, 2.0}, {1, 1, 3.0}, {2, 0, 4.0}, {2, 2, 5.0}};
    SparseMatrix A(3, 3, trips);

    auto r0 = A.row(0);
    CHECK(r0.size() == 2);
    CHECK(r0.indices[0] == 0);
    CHECK(r0.values[0] == 1.0);
    CHECK(r0.indices[1] == 2);
    CHECK(r0.values[1] == 2.0);

    auto r1 = A.row(1);
    CHECK(r1.size() == 1);

    auto r2 = A.row(2);
    CHECK(r2.size() == 2);
}

TEST_CASE("SparseMatrix: column access (CSC)", "[sparse_matrix]") {
    // [1  0  2]
    // [0  3  0]
    // [4  0  5]
    std::vector<Triplet> trips = {
        {0, 0, 1.0}, {0, 2, 2.0}, {1, 1, 3.0}, {2, 0, 4.0}, {2, 2, 5.0}};
    SparseMatrix A(3, 3, trips);

    auto c0 = A.col(0);
    CHECK(c0.size() == 2);
    CHECK(c0.indices[0] == 0);
    CHECK(c0.values[0] == 1.0);
    CHECK(c0.indices[1] == 2);
    CHECK(c0.values[1] == 4.0);

    // Empty column.
    auto c1 = A.col(1);
    CHECK(c1.size() == 1);
    CHECK(c1.indices[0] == 1);
    CHECK(c1.values[0] == 3.0);

    auto c2 = A.col(2);
    CHECK(c2.size() == 2);
}

TEST_CASE("SparseMatrix: SpMV multiply", "[sparse_matrix]") {
    // [1  2]
    // [3  4]
    std::vector<Triplet> trips = {
        {0, 0, 1.0}, {0, 1, 2.0}, {1, 0, 3.0}, {1, 1, 4.0}};
    SparseMatrix A(2, 2, trips);

    std::vector<Real> x = {1.0, 2.0};
    std::vector<Real> y(2);

    A.multiply(x, y);
    CHECK_THAT(y[0], WithinAbs(5.0, 1e-12));
    CHECK_THAT(y[1], WithinAbs(11.0, 1e-12));
}

TEST_CASE("SparseMatrix: SpMV multiplyTranspose", "[sparse_matrix]") {
    // [1  2]
    // [3  4]
    // A^T = [1  3]
    //       [2  4]
    std::vector<Triplet> trips = {
        {0, 0, 1.0}, {0, 1, 2.0}, {1, 0, 3.0}, {1, 1, 4.0}};
    SparseMatrix A(2, 2, trips);

    std::vector<Real> x = {1.0, 2.0};
    std::vector<Real> y(2);

    A.multiplyTranspose(x, y);
    // A^T * [1, 2] = [1*1+3*2, 2*1+4*2] = [7, 10]
    CHECK_THAT(y[0], WithinAbs(7.0, 1e-12));
    CHECK_THAT(y[1], WithinAbs(10.0, 1e-12));
}

TEST_CASE("SparseMatrix: identity SpMV", "[sparse_matrix]") {
    Index n = 5;
    std::vector<Triplet> trips;
    for (Index i = 0; i < n; ++i) {
        trips.push_back({i, i, 1.0});
    }
    SparseMatrix I(n, n, trips);

    std::vector<Real> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<Real> y(n);

    I.multiply(x, y);
    for (Index i = 0; i < n; ++i) {
        CHECK(y[i] == x[i]);
    }

    I.multiplyTranspose(x, y);
    for (Index i = 0; i < n; ++i) {
        CHECK(y[i] == x[i]);
    }
}

TEST_CASE("SparseMatrix: addRow", "[sparse_matrix]") {
    SparseMatrix A(2, 3, std::vector<Triplet>{
                             {0, 0, 1.0}, {0, 2, 2.0}, {1, 1, 3.0}});

    std::vector<Index> indices = {0, 2};
    std::vector<Real> values = {7.0, 8.0};
    A.addRow(indices, values);

    CHECK(A.numRows() == 3);
    CHECK(A.numNonzeros() == 5);
    CHECK(A.coeff(2, 0) == 7.0);
    CHECK(A.coeff(2, 2) == 8.0);
    CHECK(A.coeff(2, 1) == 0.0);
}

TEST_CASE("SparseMatrix: removeRow swap-and-pop", "[sparse_matrix]") {
    // [1  0]
    // [0  2]
    // [3  4]
    std::vector<Triplet> trips = {
        {0, 0, 1.0}, {1, 1, 2.0}, {2, 0, 3.0}, {2, 1, 4.0}};
    SparseMatrix A(3, 2, trips);

    // Remove row 0 — row 2 [3,4] should take its place.
    A.removeRow(0);
    CHECK(A.numRows() == 2);
    CHECK(A.coeff(0, 0) == 3.0);
    CHECK(A.coeff(0, 1) == 4.0);
    CHECK(A.coeff(1, 1) == 2.0);
}

TEST_CASE("SparseMatrix: removeRow last row", "[sparse_matrix]") {
    std::vector<Triplet> trips = {{0, 0, 1.0}, {1, 1, 2.0}};
    SparseMatrix A(2, 2, trips);

    A.removeRow(1);
    CHECK(A.numRows() == 1);
    CHECK(A.coeff(0, 0) == 1.0);
}

TEST_CASE("SparseMatrix: removeRowStable", "[sparse_matrix]") {
    // [1  0]
    // [0  2]
    // [3  4]
    std::vector<Triplet> trips = {
        {0, 0, 1.0}, {1, 1, 2.0}, {2, 0, 3.0}, {2, 1, 4.0}};
    SparseMatrix A(3, 2, trips);

    // Remove row 0 — rows 1, 2 shift up.
    A.removeRowStable(0);
    CHECK(A.numRows() == 2);
    CHECK(A.coeff(0, 1) == 2.0);
    CHECK(A.coeff(1, 0) == 3.0);
    CHECK(A.coeff(1, 1) == 4.0);
}

TEST_CASE("SparseMatrix: CSC invalidation after addRow", "[sparse_matrix]") {
    std::vector<Triplet> trips = {{0, 0, 1.0}, {1, 1, 2.0}};
    SparseMatrix A(2, 2, trips);

    // Access column to trigger CSC build.
    auto c0 = A.col(0);
    CHECK(c0.size() == 1);

    // Add a row — CSC should be invalidated.
    std::vector<Index> idx = {0};
    std::vector<Real> vals = {5.0};
    A.addRow(idx, vals);

    // Col access should rebuild CSC and include new entry.
    auto c0_new = A.col(0);
    CHECK(c0_new.size() == 2);
}

TEST_CASE("SparseMatrix: 1xN matrix", "[sparse_matrix]") {
    std::vector<Triplet> trips = {{0, 0, 1.0}, {0, 3, 2.0}};
    SparseMatrix A(1, 5, trips);

    CHECK(A.numRows() == 1);
    CHECK(A.numCols() == 5);
    CHECK(A.numNonzeros() == 2);

    std::vector<Real> x = {1.0, 0.0, 0.0, 1.0, 0.0};
    std::vector<Real> y(1);
    A.multiply(x, y);
    CHECK_THAT(y[0], WithinAbs(3.0, 1e-12));
}

TEST_CASE("SparseMatrix: Mx1 matrix", "[sparse_matrix]") {
    std::vector<Triplet> trips = {{0, 0, 1.0}, {2, 0, 3.0}};
    SparseMatrix A(3, 1, trips);

    std::vector<Real> x = {2.0};
    std::vector<Real> y(3);
    A.multiply(x, y);
    CHECK_THAT(y[0], WithinAbs(2.0, 1e-12));
    CHECK_THAT(y[1], WithinAbs(0.0, 1e-12));
    CHECK_THAT(y[2], WithinAbs(6.0, 1e-12));
}

TEST_CASE("SparseMatrix: larger matrix SpMV", "[sparse_matrix]") {
    // 4x4 dense-ish matrix verified against dense multiplication.
    // [1  2  0  0]
    // [0  3  4  0]
    // [5  0  6  7]
    // [0  0  0  8]
    std::vector<Triplet> trips = {
        {0, 0, 1.0}, {0, 1, 2.0}, {1, 1, 3.0}, {1, 2, 4.0},
        {2, 0, 5.0}, {2, 2, 6.0}, {2, 3, 7.0}, {3, 3, 8.0}};
    SparseMatrix A(4, 4, trips);

    std::vector<Real> x = {1.0, 2.0, 3.0, 4.0};
    std::vector<Real> y(4);

    A.multiply(x, y);
    // Row 0: 1*1 + 2*2 = 5
    // Row 1: 3*2 + 4*3 = 18
    // Row 2: 5*1 + 6*3 + 7*4 = 51
    // Row 3: 8*4 = 32
    CHECK_THAT(y[0], WithinAbs(5.0, 1e-12));
    CHECK_THAT(y[1], WithinAbs(18.0, 1e-12));
    CHECK_THAT(y[2], WithinAbs(51.0, 1e-12));
    CHECK_THAT(y[3], WithinAbs(32.0, 1e-12));

    std::vector<Real> yt(4);
    A.multiplyTranspose(x, yt);
    // Col 0: 1*1 + 5*3 = 16
    // Col 1: 2*1 + 3*2 = 8
    // Col 2: 4*2 + 6*3 = 26
    // Col 3: 7*3 + 8*4 = 53
    CHECK_THAT(yt[0], WithinAbs(16.0, 1e-12));
    CHECK_THAT(yt[1], WithinAbs(8.0, 1e-12));
    CHECK_THAT(yt[2], WithinAbs(26.0, 1e-12));
    CHECK_THAT(yt[3], WithinAbs(53.0, 1e-12));
}

TEST_CASE("SparseMatrix: raw CSR access", "[sparse_matrix]") {
    std::vector<Triplet> trips = {{0, 0, 1.0}, {0, 2, 2.0}, {1, 1, 3.0}};
    SparseMatrix A(2, 3, trips);

    auto vals = A.csr_values();
    auto cols = A.csr_col_indices();
    auto starts = A.csr_row_starts();

    CHECK(starts.size() == 3);
    CHECK(starts[0] == 0);
    CHECK(starts[1] == 2);
    CHECK(starts[2] == 3);

    CHECK(vals.size() == 3);
    CHECK(cols.size() == 3);
}

TEST_CASE("SparseMatrix: empty column access", "[sparse_matrix]") {
    // Column 1 is empty.
    std::vector<Triplet> trips = {{0, 0, 1.0}, {1, 2, 2.0}};
    SparseMatrix A(2, 3, trips);

    auto c1 = A.col(1);
    CHECK(c1.size() == 0);
}

TEST_CASE("SparseMatrix: removeRow last then addRow", "[sparse_matrix]") {
    std::vector<Triplet> trips = {{0, 0, 1.0}, {1, 1, 2.0}};
    SparseMatrix A(2, 2, trips);

    A.removeRow(1);
    CHECK(A.numRows() == 1);

    // Add a new row — should not include stale data from removed row.
    std::vector<Index> idx = {0};
    std::vector<Real> vals = {9.0};
    A.addRow(idx, vals);

    CHECK(A.numRows() == 2);
    CHECK(A.coeff(1, 0) == 9.0);
    CHECK(A.coeff(1, 1) == 0.0);
    auto r1 = A.row(1);
    CHECK(r1.size() == 1);
}

TEST_CASE("SparseMatrix: triplet construction with row gaps", "[sparse_matrix]") {
    // Only rows 0 and 4 have entries in a 5-row matrix.
    std::vector<Triplet> trips = {{0, 0, 1.0}, {4, 1, 2.0}};
    SparseMatrix A(5, 3, trips);

    CHECK(A.numRows() == 5);
    CHECK(A.numNonzeros() == 2);
    CHECK(A.coeff(0, 0) == 1.0);
    CHECK(A.coeff(4, 1) == 2.0);

    // Empty rows in between.
    for (int r = 1; r <= 3; ++r) {
        CHECK(A.row(r).size() == 0);
    }
}

TEST_CASE("SparseMatrix: addRow unsorted indices", "[sparse_matrix]") {
    SparseMatrix A(0, 3);

    // Add row with unsorted indices — should be sorted internally.
    std::vector<Index> idx = {2, 0};
    std::vector<Real> vals = {5.0, 3.0};
    A.addRow(idx, vals);

    CHECK(A.numRows() == 1);
    auto r = A.row(0);
    CHECK(r.indices[0] == 0);
    CHECK(r.values[0] == 3.0);
    CHECK(r.indices[1] == 2);
    CHECK(r.values[1] == 5.0);
}
