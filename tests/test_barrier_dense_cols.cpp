// Dense columns in the sparse normal-equations backend of the CPU barrier
// (#314): left out of the Cholesky factor and solved with by a Schur
// complement, against the dense normal equations as the reference.

#include "lp/newton_solver.h"
#include "mipx/barrier.h"
#include "mipx/sparse_cholesky.h"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <random>
#include <vector>

using namespace mipx;

namespace {

constexpr Index kRows = 60;
constexpr Index kDense = 3;

// A chain of sparse columns over kRows rows, then kDense columns that cover
// every row. The last row has no sparse entry, so only the dense columns
// cover it; row kDup's sparse part repeats row kDup - 1's, so it depends on
// it in A_s but not in A (the dense columns tell them apart).
constexpr Index kDup = 30;

SparseMatrix buildPlantedMatrix(Index* n_out) {
    std::mt19937 rng(314);
    std::uniform_real_distribution<Real> val(0.5, 2.0);
    std::vector<Triplet> t;
    Index col = 0;
    auto sparseRow = [](Index i) { return i < kRows - 1 && i != kDup; };
    for (Index i = 0; i < kRows - 1; ++i) {
        if (!sparseRow(i)) {
            continue;
        }
        t.push_back({i, col++, val(rng)});  // a singleton, as a slack would be
        if (sparseRow(i + 1)) {
            t.push_back({i, col, val(rng)});
            t.push_back({i + 1, col++, val(rng)});
        }
    }
    // Row kDup: row kDup-1's sparse entries again.
    const size_t sparse_entries = t.size();
    for (size_t p = 0; p < sparse_entries; ++p) {
        if (t[p].row == kDup - 1) {
            t.push_back({kDup, t[p].col, t[p].val});
        }
    }
    for (Index d = 0; d < kDense; ++d) {
        for (Index i = 0; i < kRows; ++i) {
            t.push_back({i, col, val(rng) * (d % 2 == 0 ? 1.0 : -1.0)});
        }
        ++col;
    }
    *n_out = col;
    return SparseMatrix(kRows, col, std::move(t));
}

struct Step {
    std::vector<Real> dz, dy, ds;
};

Step newtonStep(NewtonSolver& solver, Index n, Real theta_spread) {
    std::mt19937 rng(7);
    std::uniform_real_distribution<Real> u(-1.0, 1.0);
    std::vector<Real> z(static_cast<size_t>(n)), s(static_cast<size_t>(n));
    for (Index j = 0; j < n; ++j) {
        z[j] = std::pow(10.0, theta_spread * u(rng));
        s[j] = std::pow(10.0, theta_spread * u(rng));
    }
    std::vector<Real> rp(static_cast<size_t>(kRows)), rd(static_cast<size_t>(n)),
        rc(static_cast<size_t>(n));
    for (auto& v : rp)
        v = u(rng);
    for (auto& v : rd)
        v = u(rng);
    for (auto& v : rc)
        v = u(rng);
    REQUIRE(solver.factorize(z, s, 1e-10));
    Step st;
    st.dz.assign(static_cast<size_t>(n), 0.0);
    st.ds.assign(static_cast<size_t>(n), 0.0);
    st.dy.assign(static_cast<size_t>(kRows), 0.0);
    REQUIRE(solver.solveNewton(rp, rd, rc, st.dz, st.dy, st.ds));
    return st;
}

Real relDiff(const std::vector<Real>& a, const std::vector<Real>& b) {
    Real num = 0.0;
    Real den = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        num = std::max(num, std::abs(a[i] - b[i]));
        den = std::max(den, std::abs(b[i]));
    }
    return num / std::max(den, 1e-300);
}

}  // namespace

TEST_CASE("Sparse NE dense columns: Newton step matches the dense normal equations",
          "[barrier][sparse-ne][dense-cols]") {
    Index n = 0;
    const SparseMatrix A = buildPlantedMatrix(&n);
    const Real spread = GENERATE(0.0, 1.5, 3.0);  // Theta within 10^(+-2 spread)
    BarrierOptions opts;
    opts.verbose = false;
    // No refinement on the full M: the Schur complement solve alone is exact.
    opts.ir_steps = 0;

    auto ref = createCpuCholeskySolver();
    REQUIRE(ref->setup(A, kRows, n, opts));
    const Step want = newtonStep(*ref, n, spread);

    auto sparse = createCpuSparseCholeskySolver();
    REQUIRE(sparse->setup(A, kRows, n, opts));
    CHECK(sparse->factorStats().dense_cols == kDense);
    const Step got = newtonStep(*sparse, n, spread);

    CHECK(relDiff(got.dy, want.dy) < 1e-8);
    CHECK(relDiff(got.dz, want.dz) < 1e-8);
    CHECK(relDiff(got.ds, want.ds) < 1e-8);
}

TEST_CASE("SparseCholesky: a shifted dependent pivot factors M + added e e'",
          "[sparse-cholesky][dense-cols]") {
    // M = [[2,1,1],[1,1,1],[1,1,1]] + reg I: rows 1 and 2 are dependent, so
    // the last pivot cancels to about reg; row 2 may be shifted.
    const Real reg = 1e-10;
    const std::vector<Index> ptr = {0, 3, 5, 6};
    const std::vector<Index> idx = {0, 1, 2, 1, 2, 2};
    const std::vector<Real> val = {2 + reg, 1, 1, 1 + reg, 1, 1 + reg};
    const std::vector<Index> perm = {0, 1, 2};
    SparseCholesky chol;
    REQUIRE(chol.analyze(3, ptr, idx, perm));
    const std::vector<Real> shift = {0.0, 0.0, 5.0};
    SparseCholesky::PivotRule rule;
    rule.diag_offset = reg;
    rule.dependent_shift = shift;
    rule.shift_tol = 1e-6;
    REQUIRE(chol.factorize(val, rule));
    REQUIRE(chol.shiftedPivots().size() == 1);
    const auto sp = chol.shiftedPivots()[0];
    CHECK(sp.row == 2);
    // Dense M + added e2 e2', solved by Cramer's rule against b.
    Real m[3][3] = {{2 + reg, 1, 1}, {1, 1 + reg, 1}, {1, 1, 1 + reg + sp.added}};
    std::vector<Real> b = {1.0, 2.0, 3.0};
    auto det = [](Real a[3][3]) {
        return a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) -
               a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) +
               a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    };
    const Real dm = det(m);
    std::vector<Real> want(3);
    for (int c = 0; c < 3; ++c) {
        Real mc[3][3];
        for (int r = 0; r < 3; ++r) {
            for (int q = 0; q < 3; ++q) {
                mc[r][q] = q == c ? b[r] : m[r][q];
            }
        }
        want[c] = det(mc) / dm;
    }
    chol.solve(b);
    for (int c = 0; c < 3; ++c) {
        CHECK_THAT(b[c], Catch::Matchers::WithinRel(want[c], 1e-6));
    }
}

TEST_CASE("Sparse NE dense columns: leaving them out shrinks the factor",
          "[barrier][sparse-ne][dense-cols]") {
    Index n = 0;
    const SparseMatrix A = buildPlantedMatrix(&n);
    BarrierOptions opts;
    opts.verbose = false;

    auto with = createCpuSparseCholeskySolver();
    REQUIRE(with->setup(A, kRows, n, opts));
    opts.ne_max_dense_cols = 0;
    auto without = createCpuSparseCholeskySolver();
    REQUIRE(without->setup(A, kRows, n, opts));

    CHECK(with->factorStats().dense_cols == kDense);
    CHECK(without->factorStats().dense_cols == 0);
    // Without them M is dense: m(m+1)/2. With them out, L is a chain's.
    CHECK(without->factorStats().nnz_l == static_cast<size_t>(kRows) * (kRows + 1) / 2);
    CHECK(with->factorStats().nnz_l < 3 * static_cast<size_t>(kRows));
}

TEST_CASE("Sparse NE dense columns: the cap keeps the densest",
          "[barrier][sparse-ne][dense-cols]") {
    Index n = 0;
    const SparseMatrix A = buildPlantedMatrix(&n);
    BarrierOptions opts;
    opts.verbose = false;
    opts.ne_max_dense_cols = 2;
    auto solver = createCpuSparseCholeskySolver();
    REQUIRE(solver->setup(A, kRows, n, opts));
    CHECK(solver->factorStats().dense_cols == 2);
}

TEST_CASE("Sparse NE dense columns: LP solves to the dense reference objective",
          "[barrier][sparse-ne][dense-cols]") {
    // min sum c_j x_j + 3 y1 + 2 y2  s.t.  row i: x_i + x_{i+1} + y1 + y2 >= b_i
    // (i < m-1), row m-1: y1 - y2 >= 0.5 -- the last row only the two dense
    // columns y1, y2 cover.
    constexpr Index m = 50;
    LpProblem lp;
    lp.name = "dense_cols";
    lp.sense = Sense::Minimize;
    lp.num_cols = m + 2;
    lp.num_rows = m;
    std::vector<Triplet> t;
    for (Index j = 0; j < m; ++j) {
        lp.obj.push_back(1.0 + 0.1 * static_cast<Real>(j % 7));
    }
    lp.obj.push_back(3.0);
    lp.obj.push_back(2.0);
    lp.col_lower.assign(static_cast<size_t>(m) + 2, 0.0);
    lp.col_upper.assign(static_cast<size_t>(m) + 2, kInf);
    lp.col_type.assign(static_cast<size_t>(m) + 2, VarType::Continuous);
    for (Index i = 0; i + 1 < m; ++i) {
        t.push_back({i, i, 1.0});
        t.push_back({i, i + 1, 1.0});
        t.push_back({i, m, 1.0});
        t.push_back({i, m + 1, 1.0});
        lp.row_lower.push_back(1.0 + static_cast<Real>(i % 5));
        lp.row_upper.push_back(kInf);
    }
    t.push_back({m - 1, m, 1.0});
    t.push_back({m - 1, m + 1, -1.0});
    lp.row_lower.push_back(0.5);
    lp.row_upper.push_back(kInf);
    lp.matrix = SparseMatrix(m, m + 2, std::move(t));

    auto solveWith = [&](BarrierAlgorithm alg, bool presolve) {
        BarrierSolver solver;
        BarrierOptions opts;
        opts.verbose = false;
        opts.algorithm = alg;
        opts.presolve = presolve;
        solver.setOptions(opts);
        solver.load(lp);
        return solver.solve();
    };
    const bool presolve = GENERATE(false, true);
    const auto want = solveWith(BarrierAlgorithm::CpuCholesky, presolve);
    const auto got = solveWith(BarrierAlgorithm::CpuSparseCholesky, presolve);
    REQUIRE(want.status == Status::Optimal);
    REQUIRE(got.status == Status::Optimal);
    CHECK_THAT(got.objective, Catch::Matchers::WithinRel(want.objective, 1e-7) ||
                                  Catch::Matchers::WithinAbs(want.objective, 1e-7));
}
