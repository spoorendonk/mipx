#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <vector>

#include "mipx/lu.h"
#include "mipx/sparse_matrix.h"

using namespace mipx;
using Catch::Matchers::WithinAbs;

// Helper: build a dense column from a SparseMatrix column.
static std::vector<Real> denseColumn(const SparseMatrix& A, Index j) {
    std::vector<Real> col(A.numRows(), 0.0);
    auto sv = A.col(j);
    for (Index k = 0; k < sv.size(); ++k) {
        col[sv.indices[k]] = sv.values[k];
    }
    return col;
}

// Helper: dense matrix-vector product y = B*x where B is defined by basis columns.
static std::vector<Real> denseMultiply(const SparseMatrix& A,
                                       std::span<const Index> basis_cols,
                                       std::span<const Real> x) {
    Index m = static_cast<Index>(basis_cols.size());
    std::vector<Real> y(m, 0.0);
    for (Index j = 0; j < m; ++j) {
        auto col = denseColumn(A, basis_cols[j]);
        for (Index i = 0; i < m; ++i) {
            y[i] += col[i] * x[j];
        }
    }
    return y;
}

// Helper: dense B^T*x where B is defined by basis columns.
static std::vector<Real> denseMultiplyTranspose(const SparseMatrix& A,
                                                std::span<const Index> basis_cols,
                                                std::span<const Real> x) {
    Index m = static_cast<Index>(basis_cols.size());
    std::vector<Real> y(m, 0.0);
    for (Index j = 0; j < m; ++j) {
        auto col = denseColumn(A, basis_cols[j]);
        for (Index i = 0; i < m; ++i) {
            y[j] += col[i] * x[i];
        }
    }
    return y;
}

TEST_CASE("SparseLU: identity basis", "[lu]") {
    // 3x3 identity matrix.
    std::vector<Triplet> trips;
    for (Index i = 0; i < 3; ++i) {
        trips.push_back({i, i, 1.0});
    }
    SparseMatrix A(3, 3, trips);

    SparseLU lu;
    std::vector<Index> basis = {0, 1, 2};
    lu.factorize(A, basis);

    CHECK(lu.dimension() == 3);
    CHECK(lu.numUpdates() == 0);

    SECTION("FTRAN returns input unchanged") {
        std::vector<Real> rhs = {1.0, 2.0, 3.0};
        std::vector<Real> expected = rhs;
        lu.ftran(rhs);
        for (Index i = 0; i < 3; ++i) {
            CHECK_THAT(rhs[i], WithinAbs(expected[i], 1e-10));
        }
    }

    SECTION("BTRAN returns input unchanged") {
        std::vector<Real> rhs = {4.0, 5.0, 6.0};
        std::vector<Real> expected = rhs;
        lu.btran(rhs);
        for (Index i = 0; i < 3; ++i) {
            CHECK_THAT(rhs[i], WithinAbs(expected[i], 1e-10));
        }
    }
}

TEST_CASE("SparseLU: small 3x3 system", "[lu]") {
    // B = [[2, 1, 0],
    //      [1, 3, 1],
    //      [0, 1, 2]]
    // Store as a 3x3 matrix with columns 0,1,2.
    std::vector<Triplet> trips = {
        {0, 0, 2.0}, {0, 1, 1.0},
        {1, 0, 1.0}, {1, 1, 3.0}, {1, 2, 1.0},
        {2, 1, 1.0}, {2, 2, 2.0}};
    SparseMatrix A(3, 3, trips);

    SparseLU lu;
    std::vector<Index> basis = {0, 1, 2};
    lu.factorize(A, basis);

    // Solve Bx = b where b = [3, 5, 3].
    // The exact solution: x = [1, 1, 1] since B*[1,1,1] = [3, 5, 3].
    std::vector<Real> rhs = {3.0, 5.0, 3.0};
    lu.ftran(rhs);
    CHECK_THAT(rhs[0], WithinAbs(1.0, 1e-10));
    CHECK_THAT(rhs[1], WithinAbs(1.0, 1e-10));
    CHECK_THAT(rhs[2], WithinAbs(1.0, 1e-10));
}

TEST_CASE("SparseLU: FTRAN round-trip", "[lu]") {
    // B = [[4, 1],
    //      [2, 3]]
    std::vector<Triplet> trips = {
        {0, 0, 4.0}, {0, 1, 1.0},
        {1, 0, 2.0}, {1, 1, 3.0}};
    SparseMatrix A(2, 2, trips);

    SparseLU lu;
    lu.setMaxUpdates(100);
    std::vector<Index> basis = {0, 1};
    lu.factorize(A, basis);

    std::vector<Real> b = {5.0, 7.0};
    std::vector<Real> x = b;
    lu.ftran(x);

    // Verify B*x = b.
    auto Bx = denseMultiply(A, basis, x);
    for (Index i = 0; i < 2; ++i) {
        CHECK_THAT(Bx[i], WithinAbs(b[i], 1e-10));
    }
}

TEST_CASE("SparseLU: BTRAN round-trip", "[lu]") {
    // B = [[4, 1],
    //      [2, 3]]
    std::vector<Triplet> trips = {
        {0, 0, 4.0}, {0, 1, 1.0},
        {1, 0, 2.0}, {1, 1, 3.0}};
    SparseMatrix A(2, 2, trips);

    SparseLU lu;
    lu.setMaxUpdates(100);
    std::vector<Index> basis = {0, 1};
    lu.factorize(A, basis);

    std::vector<Real> c = {3.0, 2.0};
    std::vector<Real> y = c;
    lu.btran(y);

    // Verify B^T * y = c.
    auto Bty = denseMultiplyTranspose(A, basis, y);
    for (Index i = 0; i < 2; ++i) {
        CHECK_THAT(Bty[i], WithinAbs(c[i], 1e-10));
    }
}

TEST_CASE("SparseLU: sparse basis from larger matrix", "[lu]") {
    // 10x15 sparse matrix; select 10 basis columns.
    std::vector<Triplet> trips;
    // Start with a diagonal to ensure nonsingularity.
    for (Index i = 0; i < 10; ++i) {
        trips.push_back({i, i, 2.0 + static_cast<Real>(i)});
    }
    // Add some off-diagonal entries.
    trips.push_back({0, 3, 1.0});
    trips.push_back({1, 5, -0.5});
    trips.push_back({3, 7, 0.3});
    trips.push_back({5, 0, -1.0});
    trips.push_back({7, 2, 0.7});
    trips.push_back({9, 4, -0.2});
    // Add extra columns (10-14) with some entries.
    for (Index j = 10; j < 15; ++j) {
        trips.push_back({j - 10, j, 1.0});
    }

    SparseMatrix A(10, 15, trips);

    SparseLU lu;
    // Use columns 0..9 as basis.
    std::vector<Index> basis = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    lu.factorize(A, basis);
    CHECK(lu.dimension() == 10);

    // FTRAN round-trip.
    std::vector<Real> b = {1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0, -10.0};
    std::vector<Real> x = b;
    lu.ftran(x);
    auto Bx = denseMultiply(A, basis, x);
    for (Index i = 0; i < 10; ++i) {
        CHECK_THAT(Bx[i], WithinAbs(b[i], 1e-10));
    }

    // BTRAN round-trip.
    std::vector<Real> c = {2.0, 4.0, 6.0, 8.0, 10.0, 1.0, 3.0, 5.0, 7.0, 9.0};
    std::vector<Real> y = c;
    lu.btran(y);
    auto Bty = denseMultiplyTranspose(A, basis, y);
    for (Index i = 0; i < 10; ++i) {
        CHECK_THAT(Bty[i], WithinAbs(c[i], 1e-10));
    }
}

TEST_CASE("SparseLU: rank-1 update", "[lu]") {
    // B = [[3, 1],
    //      [1, 4]]
    std::vector<Triplet> trips = {
        {0, 0, 3.0}, {0, 1, 1.0}, {0, 2, 2.0},
        {1, 0, 1.0}, {1, 1, 4.0}, {1, 2, 5.0}};
    SparseMatrix A(2, 3, trips);

    SparseLU lu;
    lu.setMaxUpdates(100);
    std::vector<Index> basis = {0, 1};
    lu.factorize(A, basis);

    // Replace basis column 0 with column 2 of A: [2, 5].
    auto new_col = A.col(2);
    std::vector<Index> new_indices(new_col.indices.begin(), new_col.indices.end());
    std::vector<Real> new_values(new_col.values.begin(), new_col.values.end());

    // First FTRAN the entering column to get its representation in the current basis.
    lu.update(0, new_indices, new_values);

    CHECK(lu.numUpdates() == 1);

    // Now basis is columns {2, 1}: B_new = [[2, 1], [5, 4]].
    // Solve B_new * x = [3, 9] => x = [1, 1] since [2+1, 5+4] = [3, 9].
    std::vector<Real> rhs = {3.0, 9.0};
    lu.ftran(rhs);
    CHECK_THAT(rhs[0], WithinAbs(1.0, 1e-10));
    CHECK_THAT(rhs[1], WithinAbs(1.0, 1e-10));
}

TEST_CASE("SparseLU: updateFromFtranColumn matches update", "[lu]") {
    std::vector<Triplet> trips = {
        {0, 0, 4.0}, {1, 0, 1.0}, {2, 0, 0.5},
        {0, 1, 1.0}, {1, 1, 3.0}, {2, 1, 1.0},
        {0, 2, 0.0}, {1, 2, 2.0}, {2, 2, 5.0},
        {0, 3, 2.0}, {1, 3, 1.0}, {2, 3, 3.0}};
    SparseMatrix A(3, 4, trips);

    std::vector<Index> basis = {0, 1, 2};
    SparseLU lu_ref;
    lu_ref.setMaxUpdates(100);
    lu_ref.factorize(A, basis);

    SparseLU lu_fast;
    lu_fast.setMaxUpdates(100);
    lu_fast.factorize(A, basis);

    auto c = A.col(3);
    std::vector<Index> idx(c.indices.begin(), c.indices.end());
    std::vector<Real> val(c.values.begin(), c.values.end());

    lu_ref.update(1, idx, val);

    std::vector<Real> d(3, 0.0);
    for (Index k = 0; k < c.size(); ++k) {
        d[c.indices[k]] = c.values[k];
    }
    lu_fast.ftran(d);
    lu_fast.updateFromFtranColumn(1, d);

    CHECK(lu_ref.numUpdates() == 1);
    CHECK(lu_fast.numUpdates() == 1);

    std::vector<Real> rhs = {1.0, -2.0, 3.0};
    std::vector<Real> x_ref = rhs;
    std::vector<Real> x_fast = rhs;
    lu_ref.ftran(x_ref);
    lu_fast.ftran(x_fast);
    for (Index i = 0; i < 3; ++i) {
        CHECK_THAT(x_fast[i], WithinAbs(x_ref[i], 1e-10));
    }
}

TEST_CASE("SparseLU: multiple updates", "[lu]") {
    // 3x6 matrix with enough columns for several basis swaps.
    std::vector<Triplet> trips = {
        // col 0: [2, 0, 1]
        {0, 0, 2.0}, {2, 0, 1.0},
        // col 1: [1, 3, 0]
        {0, 1, 1.0}, {1, 1, 3.0},
        // col 2: [0, 1, 4]
        {1, 2, 1.0}, {2, 2, 4.0},
        // col 3: [1, 1, 1]
        {0, 3, 1.0}, {1, 3, 1.0}, {2, 3, 1.0},
        // col 4: [3, 0, 2]
        {0, 4, 3.0}, {2, 4, 2.0},
        // col 5: [0, 2, 3]
        {1, 5, 2.0}, {2, 5, 3.0}};
    SparseMatrix A(3, 6, trips);

    SparseLU lu;
    std::vector<Index> basis = {0, 1, 2};
    lu.factorize(A, basis);

    // Current basis cols: {0, 1, 2}.
    auto verify_ftran = [&](std::vector<Index>& cur_basis) {
        std::vector<Real> b = {1.0, 2.0, 3.0};
        std::vector<Real> x = b;
        lu.ftran(x);
        auto Bx = denseMultiply(A, cur_basis, x);
        for (Index i = 0; i < 3; ++i) {
            CHECK_THAT(Bx[i], WithinAbs(b[i], 1e-10));
        }
    };

    verify_ftran(basis);

    // Update 1: replace position 0 with column 3.
    {
        auto c = A.col(3);
        std::vector<Index> idx(c.indices.begin(), c.indices.end());
        std::vector<Real> val(c.values.begin(), c.values.end());
        lu.update(0, idx, val);
        basis[0] = 3;
        CHECK(lu.numUpdates() == 1);
        verify_ftran(basis);
    }

    // Update 2: replace position 1 with column 4.
    {
        auto c = A.col(4);
        std::vector<Index> idx(c.indices.begin(), c.indices.end());
        std::vector<Real> val(c.values.begin(), c.values.end());
        lu.update(1, idx, val);
        basis[1] = 4;
        CHECK(lu.numUpdates() == 2);
        verify_ftran(basis);
    }

    // Update 3: replace position 2 with column 5.
    {
        auto c = A.col(5);
        std::vector<Index> idx(c.indices.begin(), c.indices.end());
        std::vector<Real> val(c.values.begin(), c.values.end());
        lu.update(2, idx, val);
        basis[2] = 5;
        CHECK(lu.numUpdates() == 3);
        verify_ftran(basis);
    }
}

TEST_CASE("SparseLU: sparse entering-column updates stay consistent across repeated calls",
          "[lu][regression]") {
    // Initial basis columns (0..3) form a nonsingular, coupled matrix so that
    // B^{-1} * a_q can become dense even when a_q is sparse.
    //
    // Extra columns 4..6 are singleton (very sparse) entering columns.
    std::vector<Triplet> trips = {
        // col 0: [2, 1, 0, 0]
        {0, 0, 2.0}, {1, 0, 1.0},
        // col 1: [0, 3, 1, 0]
        {1, 1, 3.0}, {2, 1, 1.0},
        // col 2: [1, 0, 2, 1]
        {0, 2, 1.0}, {2, 2, 2.0}, {3, 2, 1.0},
        // col 3: [0, 1, 0, 2]
        {1, 3, 1.0}, {3, 3, 2.0},
        // col 4: e0
        {0, 4, 1.0},
        // col 5: e3
        {3, 5, 1.0},
        // col 6: e2
        {2, 6, 1.0},
    };
    SparseMatrix A(4, 7, trips);

    SparseLU lu;
    lu.setMaxUpdates(100);
    std::vector<Index> basis = {0, 1, 2, 3};
    lu.factorize(A, basis);

    auto verify_round_trip = [&](const std::vector<Index>& cur_basis) {
        std::vector<Real> b = {1.0, -2.0, 0.5, 3.0};
        std::vector<Real> x = b;
        lu.ftran(x);
        auto Bx = denseMultiply(A, cur_basis, x);
        for (Index i = 0; i < 4; ++i) {
            CHECK_THAT(Bx[i], WithinAbs(b[i], 1e-9));
        }

        std::vector<Real> c = {-1.0, 2.5, 1.25, -0.75};
        std::vector<Real> y = c;
        lu.btran(y);
        auto Bty = denseMultiplyTranspose(A, cur_basis, y);
        for (Index i = 0; i < 4; ++i) {
            CHECK_THAT(Bty[i], WithinAbs(c[i], 1e-9));
        }
    };

    verify_round_trip(basis);

    // Repeated updates with sparse entering columns.
    // This is a regression guard for stale dense-update buffer state.
    {
        auto c = A.col(4);
        std::vector<Index> idx(c.indices.begin(), c.indices.end());
        std::vector<Real> val(c.values.begin(), c.values.end());
        lu.update(2, idx, val);
        basis[2] = 4;
        verify_round_trip(basis);
    }
    {
        auto c = A.col(5);
        std::vector<Index> idx(c.indices.begin(), c.indices.end());
        std::vector<Real> val(c.values.begin(), c.values.end());
        lu.update(0, idx, val);
        basis[0] = 5;
        verify_round_trip(basis);
    }
    {
        auto c = A.col(6);
        std::vector<Index> idx(c.indices.begin(), c.indices.end());
        std::vector<Real> val(c.values.begin(), c.values.end());
        lu.update(1, idx, val);
        basis[1] = 6;
        verify_round_trip(basis);
    }
}

TEST_CASE("SparseLU: refactorization tracking", "[lu]") {
    // Simple 2x2 identity.
    std::vector<Triplet> trips = {{0, 0, 1.0}, {1, 1, 1.0},
                                  {0, 2, 2.0}, {1, 2, 3.0}};
    SparseMatrix A(2, 4, trips);

    SparseLU lu;
    lu.setMaxUpdates(100);
    std::vector<Index> basis = {0, 1};
    lu.factorize(A, basis);

    CHECK(lu.numUpdates() == 0);
    CHECK_FALSE(lu.needsRefactorization());

    // Keep swapping between columns to accumulate updates.
    // With a 100-update limit, after enough updates it should trigger.
    auto col2 = A.col(2);
    std::vector<Index> idx2(col2.indices.begin(), col2.indices.end());
    std::vector<Real> val2(col2.values.begin(), col2.values.end());

    auto col0 = A.col(0);
    std::vector<Index> idx0(col0.indices.begin(), col0.indices.end());
    std::vector<Real> val0(col0.values.begin(), col0.values.end());

    for (Index k = 0; k < 50; ++k) {
        // Swap position 0 between col 0 and col 2.
        if (k % 2 == 0) {
            lu.update(0, idx2, val2);
        } else {
            lu.update(0, idx0, val0);
        }
    }

    CHECK(lu.numUpdates() == 50);

    // Do 50 more to reach 100.
    for (Index k = 0; k < 50; ++k) {
        if (k % 2 == 0) {
            lu.update(0, idx0, val0);
        } else {
            lu.update(0, idx2, val2);
        }
    }

    CHECK(lu.numUpdates() == 100);
    CHECK(lu.needsRefactorization());
}

TEST_CASE("SparseLU: update workspace resets across refactorization", "[lu][regression]") {
    // Basis columns 0..3 form a coupled nonsingular matrix.
    // Extra sparse columns are used for updates before/after refactorization.
    std::vector<Triplet> trips = {
        {0, 0, 2.0}, {1, 0, 1.0},
        {1, 1, 3.0}, {2, 1, 1.0},
        {0, 2, 1.0}, {2, 2, 2.0}, {3, 2, 1.0},
        {1, 3, 1.0}, {3, 3, 2.0},
        {0, 4, 1.0},  // sparse entering col
        {3, 5, 1.0},
        {2, 6, 1.0}};  // sparse entering col
    SparseMatrix A(4, 7, trips);
    std::vector<Index> basis = {0, 1, 2, 3};

    auto make_col = [&](Index col) {
        auto c = A.col(col);
        std::vector<Index> idx(c.indices.begin(), c.indices.end());
        std::vector<Real> val(c.values.begin(), c.values.end());
        return std::pair<std::vector<Index>, std::vector<Real>>{std::move(idx), std::move(val)};
    };

    auto [idx4, val4] = make_col(4);
    auto [idx6, val6] = make_col(6);

    SparseLU lu_reuse;
    lu_reuse.setMaxUpdates(100);
    lu_reuse.factorize(A, basis);
    // Populate update workspace/touch list, then refactorize.
    lu_reuse.update(2, idx4, val4);
    lu_reuse.factorize(A, basis);
    lu_reuse.update(1, idx6, val6);

    SparseLU lu_fresh;
    lu_fresh.setMaxUpdates(100);
    lu_fresh.factorize(A, basis);
    lu_fresh.update(1, idx6, val6);

    CHECK(lu_reuse.numUpdates() == 1);
    CHECK(lu_fresh.numUpdates() == 1);

    std::vector<Real> rhs = {1.0, -2.0, 0.5, 3.0};
    std::vector<Real> x_reuse = rhs;
    std::vector<Real> x_fresh = rhs;
    lu_reuse.ftran(x_reuse);
    lu_fresh.ftran(x_fresh);
    for (Index i = 0; i < 4; ++i) {
        CHECK_THAT(x_reuse[i], WithinAbs(x_fresh[i], 1e-10));
    }

    std::vector<Real> cost = {-1.0, 2.5, 1.25, -0.75};
    std::vector<Real> y_reuse = cost;
    std::vector<Real> y_fresh = cost;
    lu_reuse.btran(y_reuse);
    lu_fresh.btran(y_fresh);
    for (Index i = 0; i < 4; ++i) {
        CHECK_THAT(y_reuse[i], WithinAbs(y_fresh[i], 1e-10));
    }
}

TEST_CASE("SparseLU: configurable update limit", "[lu]") {
    std::vector<Triplet> trips = {{0, 0, 1.0}, {1, 1, 1.0},
                                  {0, 2, 2.0}, {1, 2, 3.0}};
    SparseMatrix A(2, 4, trips);

    SparseLU lu;
    lu.setMaxUpdates(3);
    std::vector<Index> basis = {0, 1};
    lu.factorize(A, basis);

    auto col2 = A.col(2);
    std::vector<Index> idx2(col2.indices.begin(), col2.indices.end());
    std::vector<Real> val2(col2.values.begin(), col2.values.end());

    auto col0 = A.col(0);
    std::vector<Index> idx0(col0.indices.begin(), col0.indices.end());
    std::vector<Real> val0(col0.values.begin(), col0.values.end());

    lu.update(0, idx2, val2);
    CHECK_FALSE(lu.needsRefactorization());
    lu.update(0, idx0, val0);
    CHECK_FALSE(lu.needsRefactorization());
    lu.update(0, idx2, val2);
    CHECK(lu.needsRefactorization());
}

TEST_CASE("SparseLU: diagonal matrix", "[lu]") {
    // Diagonal matrix with varying entries.
    Index n = 5;
    std::vector<Triplet> trips;
    std::vector<Real> diag = {3.0, 1.0, 4.0, 1.5, 2.0};
    for (Index i = 0; i < n; ++i) {
        trips.push_back({i, i, diag[i]});
    }
    SparseMatrix A(n, n, trips);

    SparseLU lu;
    std::vector<Index> basis = {0, 1, 2, 3, 4};
    lu.factorize(A, basis);

    // FTRAN: solving diag(d)*x = b => x[i] = b[i]/d[i].
    std::vector<Real> b = {6.0, 2.0, 8.0, 3.0, 10.0};
    std::vector<Real> x = b;
    lu.ftran(x);
    for (Index i = 0; i < n; ++i) {
        CHECK_THAT(x[i], WithinAbs(b[i] / diag[i], 1e-10));
    }

    // BTRAN: B^T = B for diagonal, same result.
    std::vector<Real> y = b;
    lu.btran(y);
    for (Index i = 0; i < n; ++i) {
        CHECK_THAT(y[i], WithinAbs(b[i] / diag[i], 1e-10));
    }
}

TEST_CASE("SparseLU: permuted identity", "[lu]") {
    // Permuted identity: column j goes to row perm[j].
    // Matrix with columns arranged such that basis forms a permutation.
    // Column 0 = e_2, Column 1 = e_0, Column 2 = e_1
    // So B = [[0, 1, 0],
    //         [0, 0, 1],
    //         [1, 0, 0]]
    std::vector<Triplet> trips = {
        {2, 0, 1.0},
        {0, 1, 1.0},
        {1, 2, 1.0}};
    SparseMatrix A(3, 3, trips);

    SparseLU lu;
    std::vector<Index> basis = {0, 1, 2};
    lu.factorize(A, basis);

    // Solve B*x = [1, 2, 3].
    // B = perm matrix, so x = B^{-1}*b = P^T*b.
    // B*[a,b,c] = [b, c, a], so B^{-1}*[1,2,3] = [3, 1, 2].
    std::vector<Real> rhs = {1.0, 2.0, 3.0};
    lu.ftran(rhs);
    CHECK_THAT(rhs[0], WithinAbs(3.0, 1e-10));
    CHECK_THAT(rhs[1], WithinAbs(1.0, 1e-10));
    CHECK_THAT(rhs[2], WithinAbs(2.0, 1e-10));

    // BTRAN: B^T*y = c => y = B^{-T}*c.
    // B^T = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    // B^T*[a,b,c] = [c, a, b], so B^{-T}*[1,2,3] = [2, 3, 1].
    std::vector<Real> rhs2 = {1.0, 2.0, 3.0};
    lu.btran(rhs2);
    CHECK_THAT(rhs2[0], WithinAbs(2.0, 1e-10));
    CHECK_THAT(rhs2[1], WithinAbs(3.0, 1e-10));
    CHECK_THAT(rhs2[2], WithinAbs(1.0, 1e-10));
}

TEST_CASE("SparseLU: hypersparse solve paths on large sparse rhs", "[lu]") {
    // Structured tridiagonal basis, large enough to trigger hyper-sparse gating.
    constexpr Index n = 300;
    std::vector<Triplet> trips;
    trips.reserve(static_cast<std::size_t>(3 * n));
    for (Index i = 0; i < n; ++i) {
        trips.push_back({i, i, 4.0});
        if (i > 0) trips.push_back({i, i - 1, -1.0});
        if (i + 1 < n) trips.push_back({i, i + 1, 0.5});
    }
    SparseMatrix A(n, n, trips);
    std::vector<Index> basis(n);
    for (Index i = 0; i < n; ++i) basis[i] = i;

    SparseLU lu;
    lu.factorize(A, basis);

    SECTION("FTRAN large sparse rhs round-trip") {
        std::vector<Real> b(n, 0.0);
        b[0] = 3.0;
        b[57] = -1.5;
        b[299] = 2.25;

        std::vector<Real> x = b;
        lu.ftran(x);
        auto Bx = denseMultiply(A, basis, x);
        for (Index i = 0; i < n; ++i) {
            CHECK_THAT(Bx[i], WithinAbs(b[i], 1e-9));
        }
    }

    SECTION("BTRAN large sparse rhs round-trip") {
        std::vector<Real> c(n, 0.0);
        c[4] = 2.0;
        c[120] = -0.75;
        c[298] = 1.0;

        std::vector<Real> y = c;
        lu.btran(y);
        auto Bty = denseMultiplyTranspose(A, basis, y);
        for (Index i = 0; i < n; ++i) {
            CHECK_THAT(Bty[i], WithinAbs(c[i], 1e-9));
        }
    }
}
