#include "mipx/sparse_cholesky.h"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

using namespace mipx;

namespace {

// Lower-triangle pattern (row >= col) of a dense symmetric matrix, by column.
struct Pattern {
    Index n = 0;
    std::vector<Index> col_starts;
    std::vector<Index> rows;
    std::vector<Real> vals;
};

Pattern lowerOf(const std::vector<std::vector<Real>>& m) {
    Pattern p;
    p.n = static_cast<Index>(m.size());
    p.col_starts.push_back(0);
    for (Index j = 0; j < p.n; ++j) {
        for (Index i = j; i < p.n; ++i) {
            if (i == j || m[i][j] != 0.0) {
                p.rows.push_back(i);
                p.vals.push_back(m[i][j]);
            }
        }
        p.col_starts.push_back(static_cast<Index>(p.rows.size()));
    }
    return p;
}

// Random sparse SPD matrix: B·Bᵀ + n·I with B sparse.
std::vector<std::vector<Real>> randomSpd(Index n, double density, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    std::bernoulli_distribution keep(density);
    std::vector<std::vector<Real>> b(n, std::vector<Real>(n, 0.0));
    for (Index i = 0; i < n; ++i) {
        for (Index j = 0; j < n; ++j) {
            if (i == j || keep(rng)) {
                b[i][j] = u(rng);
            }
        }
    }
    std::vector<std::vector<Real>> m(n, std::vector<Real>(n, 0.0));
    for (Index i = 0; i < n; ++i) {
        for (Index j = 0; j < n; ++j) {
            for (Index k = 0; k < n; ++k) {
                m[i][j] += b[i][k] * b[j][k];
            }
        }
        m[i][i] += 0.1;
    }
    return m;
}

std::vector<Index> randomPerm(Index n, unsigned seed) {
    std::vector<Index> perm(n);
    std::iota(perm.begin(), perm.end(), 0);
    std::mt19937 rng(seed);
    std::shuffle(perm.begin(), perm.end(), rng);
    return perm;
}

// max |M·x - b| / (1 + max |b|) after solving with the factor.
double solveResidual(const std::vector<std::vector<Real>>& m, const SparseCholesky& chol,
                     unsigned seed) {
    const auto n = static_cast<Index>(m.size());
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    std::vector<Real> b(n);
    for (auto& v : b) {
        v = u(rng);
    }
    std::vector<Real> x = b;
    chol.solve(x);
    double err = 0.0;
    double bmax = 0.0;
    for (Index i = 0; i < n; ++i) {
        Real r = -b[i];
        for (Index j = 0; j < n; ++j) {
            r += m[i][j] * x[j];
        }
        err = std::max(err, std::abs(r));
        bmax = std::max(bmax, std::abs(b[i]));
    }
    return err / (1.0 + bmax);
}

// Runs the enclosing test on BLAS (when the build found one) or on mipx's own
// loops, restoring BLAS on exit.
struct KernelChoice {
    explicit KernelChoice(bool use_blas) { SparseCholesky::setBlasEnabled(use_blas); }
    ~KernelChoice() { SparseCholesky::setBlasEnabled(true); }
    KernelChoice(const KernelChoice&) = delete;
    KernelChoice& operator=(const KernelChoice&) = delete;
};

// M = A·Aᵀ for an integer A (n_rows × n_cols) with row `dep` = row r1 + row r2.
std::vector<std::vector<Real>> dependentGram(Index n_rows, Index n_cols, double density,
                                             unsigned seed, Index dep, Index r1, Index r2) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> u(-3, 3);
    std::bernoulli_distribution keep(density);
    std::vector<std::vector<Real>> a(n_rows, std::vector<Real>(n_cols, 0.0));
    for (Index i = 0; i < n_rows; ++i) {
        a[i][i % n_cols] = 1.0;
        for (Index k = 0; k < n_cols; ++k) {
            if (keep(rng)) {
                a[i][k] = u(rng);
            }
        }
    }
    a[dep] = a[r1];
    for (Index k = 0; k < n_cols; ++k) {
        a[dep][k] += a[r2][k];
    }
    std::vector<std::vector<Real>> m(n_rows, std::vector<Real>(n_rows, 0.0));
    for (Index i = 0; i < n_rows; ++i) {
        for (Index j = 0; j < n_rows; ++j) {
            for (Index k = 0; k < n_cols; ++k) {
                m[i][j] += a[i][k] * a[j][k];
            }
        }
    }
    return m;
}

// Factors a singular M with one dependent row and checks that a right-hand
// side in M's range is reproduced. Returns the supernode partition.
std::vector<Index> checkOneDependentRow(const std::vector<std::vector<Real>>& m,
                                        std::vector<Index> perm) {
    const auto n = static_cast<Index>(m.size());
    const Pattern p = lowerOf(m);
    SparseCholesky chol;
    REQUIRE(chol.analyze(n, p.col_starts, p.rows, perm));
    SparseCholesky::PivotRule rule;
    rule.dependent_tol = 1e-10;
    REQUIRE(chol.factorize(p.vals, rule));
    CHECK(chol.numDependentPivots() == 1);
    std::mt19937 rng(17);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    std::vector<Real> x_true(n);
    for (auto& v : x_true) {
        v = u(rng);
    }
    std::vector<Real> b(n, 0.0);
    for (Index i = 0; i < n; ++i) {
        for (Index j = 0; j < n; ++j) {
            b[i] += m[i][j] * x_true[j];
        }
    }
    std::vector<Real> x = b;
    chol.solve(x);
    double bmax = 0.0;
    for (const Real v : b) {
        bmax = std::max(bmax, std::abs(v));
    }
    for (Index i = 0; i < n; ++i) {
        Real r = -b[i];
        for (Index j = 0; j < n; ++j) {
            r += m[i][j] * x[j];
        }
        CHECK(std::abs(r) < 1e-8 * (1.0 + bmax));
    }
    return chol.snodeStart();
}

// Index of the supernode that holds permuted column `col`.
std::size_t snodeOf(const std::vector<Index>& snode_start, Index col) {
    const auto it = std::upper_bound(snode_start.begin(), snode_start.end(), col);
    return static_cast<std::size_t>(it - snode_start.begin()) - 1;
}

}  // namespace

TEST_CASE("SparseCholesky solves random sparse SPD systems", "[sparse_cholesky]") {
    const KernelChoice kernels(GENERATE(true, false));
    for (unsigned seed = 1; seed <= 12; ++seed) {
        const Index n = 20 + static_cast<Index>(seed) * 17;
        const double density = seed % 3 == 0 ? 0.2 : 0.02;
        const auto m = randomSpd(n, density, seed);
        const Pattern p = lowerOf(m);
        SparseCholesky chol;
        const auto perm = seed % 2 == 0 ? randomPerm(n, seed) : [&] {
            std::vector<Index> id(n);
            std::iota(id.begin(), id.end(), 0);
            return id;
        }();
        REQUIRE(chol.analyze(n, p.col_starts, p.rows, perm));
        REQUIRE(chol.factorize(p.vals, SparseCholesky::PivotRule{}));
        CHECK(chol.numDependentPivots() == 0);
        CHECK(solveResidual(m, chol, seed + 100) < 1e-9);
    }
}

TEST_CASE("SparseCholesky reuses one analysis for new values", "[sparse_cholesky]") {
    const auto m1 = randomSpd(60, 0.05, 7);
    Pattern p = lowerOf(m1);
    SparseCholesky chol;
    REQUIRE(chol.analyze(p.n, p.col_starts, p.rows, randomPerm(p.n, 3)));
    REQUIRE(chol.factorize(p.vals, SparseCholesky::PivotRule{}));
    CHECK(solveResidual(m1, chol, 1) < 1e-9);
    // Scale the diagonal up: same pattern, new values.
    auto m2 = m1;
    for (Index i = 0; i < p.n; ++i) {
        m2[i][i] *= 3.0;
    }
    const Pattern p2 = lowerOf(m2);
    REQUIRE(p2.rows == p.rows);
    REQUIRE(chol.factorize(p2.vals, SparseCholesky::PivotRule{}));
    CHECK(solveResidual(m2, chol, 2) < 1e-9);
}

TEST_CASE("SparseCholesky accepts entries from either triangle and sums repeats",
          "[sparse_cholesky]") {
    // M = [[4,1,0],[1,5,2],[0,2,6]] given as upper entries, the diagonal split
    // in two repeated entries.
    const std::vector<Index> col_starts = {0, 2, 4, 6};
    const std::vector<Index> rows = {0, 0, 0, 1, 1, 2};
    const std::vector<Real> vals = {2.0, 2.0, 1.0, 5.0, 2.0, 6.0};
    // Column 0: (0,0)=2, (0,0)=2; column 1: (0,1)=1, (1,1)=5; column 2: (1,2)=2, (2,2)=6.
    SparseCholesky chol;
    REQUIRE(chol.analyze(3, col_starts, rows, std::vector<Index>{2, 0, 1}));
    REQUIRE(chol.factorize(vals, SparseCholesky::PivotRule{}));
    const std::vector<std::vector<Real>> m = {{4, 1, 0}, {1, 5, 2}, {0, 2, 6}};
    CHECK(solveResidual(m, chol, 5) < 1e-12);
}

TEST_CASE("SparseCholesky supernodes on known patterns", "[sparse_cholesky]") {
    SECTION("dense matrix is one supernode") {
        const auto m = randomSpd(30, 1.0, 11);
        const Pattern p = lowerOf(m);
        SparseCholesky chol;
        std::vector<Index> id(30);
        std::iota(id.begin(), id.end(), 0);
        REQUIRE(chol.analyze(30, p.col_starts, p.rows, id));
        CHECK(chol.stats().num_supernodes == 1);
        CHECK(chol.stats().nnz_l == 30 * 31 / 2);
    }
    SECTION("tridiagonal matrix has no fill") {
        const Index n = 200;
        std::vector<std::vector<Real>> m(n, std::vector<Real>(n, 0.0));
        for (Index i = 0; i < n; ++i) {
            m[i][i] = 4.0;
            if (i + 1 < n) {
                m[i][i + 1] = m[i + 1][i] = -1.0;
            }
        }
        const Pattern p = lowerOf(m);
        std::vector<Index> id(n);
        std::iota(id.begin(), id.end(), 0);
        SparseCholesky chol;
        REQUIRE(chol.analyze(n, p.col_starts, p.rows, id));
        CHECK(chol.stats().nnz_l == static_cast<std::size_t>(2 * n - 1));
        REQUIRE(chol.factorize(p.vals, SparseCholesky::PivotRule{}));
        CHECK(solveResidual(m, chol, 9) < 1e-12);
    }
    SECTION("arrow matrix ordered hub-last has no fill") {
        const Index n = 100;
        std::vector<std::vector<Real>> m(n, std::vector<Real>(n, 0.0));
        for (Index i = 0; i < n; ++i) {
            m[i][i] = static_cast<Real>(n);
            if (i > 0) {
                m[i][0] = m[0][i] = 1.0;
            }
        }
        const Pattern p = lowerOf(m);
        std::vector<Index> hub_last(n);
        std::iota(hub_last.begin(), hub_last.end(), 1);
        hub_last[n - 1] = 0;  // perm[new] = old: the hub (0) is eliminated last
        SparseCholesky chol;
        REQUIRE(chol.analyze(n, p.col_starts, p.rows, hub_last));
        CHECK(chol.stats().nnz_l == static_cast<std::size_t>(2 * n - 1));
        REQUIRE(chol.factorize(p.vals, SparseCholesky::PivotRule{}));
        CHECK(solveResidual(m, chol, 4) < 1e-12);
    }
    SECTION("block diagonal matrix keeps its blocks apart") {
        const Index nb = 5;
        const Index bs = 12;
        const Index n = nb * bs;
        std::vector<std::vector<Real>> m(n, std::vector<Real>(n, 0.0));
        for (Index b = 0; b < nb; ++b) {
            const auto blk = randomSpd(bs, 1.0, 20 + static_cast<unsigned>(b));
            for (Index i = 0; i < bs; ++i) {
                for (Index j = 0; j < bs; ++j) {
                    m[b * bs + i][b * bs + j] = blk[i][j];
                }
            }
        }
        const Pattern p = lowerOf(m);
        SparseCholesky chol;
        REQUIRE(chol.analyze(n, p.col_starts, p.rows, randomPerm(n, 8)));
        // Each dense block is one supernode, and no update crosses blocks.
        CHECK(chol.stats().num_supernodes == nb);
        CHECK(chol.stats().nnz_l == static_cast<std::size_t>(nb * bs * (bs + 1) / 2));
        REQUIRE(chol.factorize(p.vals, SparseCholesky::PivotRule{}));
        CHECK(solveResidual(m, chol, 3) < 1e-10);
    }
}

TEST_CASE("SparseCholesky loads the BLAS the build found", "[sparse_cholesky]") {
#ifdef MIPX_TEST_EXPECT_BLAS
    CHECK(SparseCholesky::blasAvailable());
#else
    SUCCEED("built without a shared BLAS library");
#endif
}

TEST_CASE("SparseCholesky drops dependent rows with an infinite pivot", "[sparse_cholesky]") {
    // M = A·Aᵀ for A with row 2 = row 0 + row 1: singular, rank 2.
    const std::vector<std::vector<Real>> a = {{1, 0, 2, 0}, {0, 1, 1, 3}, {1, 1, 3, 3}};
    std::vector<std::vector<Real>> m(3, std::vector<Real>(3, 0.0));
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 4; ++k) {
                m[i][j] += a[i][k] * a[j][k];
            }
        }
    }
    const Pattern p = lowerOf(m);
    SparseCholesky chol;
    REQUIRE(chol.analyze(3, p.col_starts, p.rows, std::vector<Index>{0, 1, 2}));
    REQUIRE(chol.factorize(p.vals, SparseCholesky::PivotRule{}));
    CHECK(chol.numDependentPivots() == 1);
    // b in the range of M: the solve reproduces it.
    const std::vector<Real> x_true = {1.0, -2.0, 0.0};
    std::vector<Real> b(3, 0.0);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            b[i] += m[i][j] * x_true[j];
        }
    }
    std::vector<Real> x = b;
    chol.solve(x);
    CHECK(std::abs(x[2]) < 1e-20);
    for (int i = 0; i < 3; ++i) {
        Real r = -b[i];
        for (int j = 0; j < 3; ++j) {
            r += m[i][j] * x[j];
        }
        CHECK(std::abs(r) < 1e-9);
    }
}

TEST_CASE("SparseCholesky factors a supernode wider than one panel", "[sparse_cholesky]") {
    // 150 dense columns: one supernode, three 64-wide panels, so the panel's
    // triangular solve, the rank-k update and the off-diagonal gemm all run.
    const KernelChoice kernels(GENERATE(true, false));
    const auto m = randomSpd(150, 1.0, 31);
    const Pattern p = lowerOf(m);
    SparseCholesky chol;
    REQUIRE(chol.analyze(150, p.col_starts, p.rows, randomPerm(150, 5)));
    CHECK(chol.stats().num_supernodes == 1);
    REQUIRE(chol.factorize(p.vals, SparseCholesky::PivotRule{}));
    CHECK(solveResidual(m, chol, 12) < 1e-9);
}

TEST_CASE("SparseCholesky solves with lower-triangle inter-supernode updates",
          "[sparse_cholesky]") {
    const KernelChoice kernels(GENERATE(true, false));
    constexpr Index block = 32;
    constexpr Index n = 3 * block;
    std::vector<std::vector<Real>> m(n, std::vector<Real>(n, 0.0));
    for (Index i = 0; i < n; ++i) {
        m[i][i] = n;
        for (Index j = 0; j < i; ++j) {
            // The gaps between the updated rows keep the blocks from
            // joining. The first update has a rectangle under its square.
            if (i / block == j / block || i % block < 20) {
                m[i][j] = m[j][i] = -1.0 / (i - j);
            }
        }
    }
    const Pattern p = lowerOf(m);
    std::vector<Index> perm(n);
    std::iota(perm.begin(), perm.end(), 0);
    SparseCholesky chol;
    chol.setLowerTriangleUpdates(true);
    REQUIRE(chol.analyze(n, p.col_starts, p.rows, perm));
    REQUIRE(chol.snodeStart() == std::vector<Index>{0, block, 2 * block, n});
    REQUIRE(chol.factorize(p.vals, SparseCholesky::PivotRule{}));
    CHECK(chol.numDependentPivots() == 0);
    CHECK(solveResidual(m, chol, 12) < 1e-12);
}

TEST_CASE("SparseCholesky drops a dependent row in a later panel", "[sparse_cholesky]") {
    const KernelChoice kernels(GENERATE(true, false));
    // Dense M: a single supernode, the dependent row 90 in its second panel.
    const auto m = dependentGram(100, 300, 1.0, 41, 90, 3, 50);
    std::vector<Index> id(100);
    std::iota(id.begin(), id.end(), 0);
    checkOneDependentRow(m, id);
}

TEST_CASE("SparseCholesky drops a dependent row in a later supernode", "[sparse_cholesky]") {
    const KernelChoice kernels(GENERATE(true, false));
    // Banded A (row i on columns 2i .. 2i+7), so M is banded and splits into
    // many supernodes; row 100 = row 20 + row 21, so its pivot is reached only
    // after the updates of the supernodes between them. M's etree stays a
    // chain (L(j+1, j) != 0 for every j), so the postorder keeps the identity
    // order and permuted column j is row j.
    const Index n_rows = 120;
    const Index n_cols = 2 * n_rows + 8;
    std::mt19937 rng(43);
    std::uniform_int_distribution<int> u(1, 4);
    std::vector<std::vector<Real>> a(n_rows, std::vector<Real>(n_cols, 0.0));
    for (Index i = 0; i < n_rows; ++i) {
        for (Index k = 2 * i; k < 2 * i + 8; ++k) {
            a[i][k] = u(rng);
        }
    }
    for (Index k = 0; k < n_cols; ++k) {
        a[100][k] = a[20][k] + a[21][k];
    }
    std::vector<std::vector<Real>> m(n_rows, std::vector<Real>(n_rows, 0.0));
    for (Index i = 0; i < n_rows; ++i) {
        for (Index j = 0; j < n_rows; ++j) {
            for (Index k = 0; k < n_cols; ++k) {
                m[i][j] += a[i][k] * a[j][k];
            }
        }
    }
    std::vector<Index> id(n_rows);
    std::iota(id.begin(), id.end(), 0);
    const std::vector<Index> snodes = checkOneDependentRow(m, id);
    CHECK(snodeOf(snodes, 100) != snodeOf(snodes, 21));
}

TEST_CASE("SparseCholesky refuses a factor above its memory cap", "[sparse_cholesky]") {
    const auto m = randomSpd(80, 1.0, 2);
    const Pattern p = lowerOf(m);
    std::vector<Index> id(80);
    std::iota(id.begin(), id.end(), 0);
    SparseCholesky chol;
    CHECK_FALSE(chol.analyze(80, p.col_starts, p.rows, id, 1024));
    CHECK_FALSE(chol.error().empty());
}

TEST_CASE("SparseCholesky can reuse a fitting symbolic layout", "[sparse_cholesky]") {
    const auto m = randomSpd(80, 0.02, 19);
    const Pattern p = lowerOf(m);
    const auto perm = randomPerm(p.n, 27);
    SparseCholesky chol;
    REQUIRE(chol.analyzeSymbolic(p.n, p.col_starts, p.rows, perm, 0, {}));
    const auto bytes = chol.factorBytes();
    const auto fill = chol.stats().nnz_l;
    CHECK_FALSE(chol.factorize(p.vals, {}));

    SparseCholesky rejected;
    CHECK_FALSE(rejected.analyzeSymbolic(p.n, p.col_starts, p.rows, perm, bytes - 1, {}));
    CHECK(rejected.stats().nnz_l == fill);
    CHECK(rejected.factorBytes() == bytes);
    CHECK_FALSE(rejected.prepareNumeric(p.col_starts, p.rows, {}));

    SparseCholesky selected = std::move(chol);
    REQUIRE(selected.prepareNumeric(p.col_starts, p.rows, {}));
    CHECK(selected.factorBytes() == bytes);
    CHECK(selected.stats().nnz_l == fill);
    REQUIRE(selected.factorize(p.vals, {}));
    CHECK(solveResidual(m, selected, 17) < 1e-12);
}
