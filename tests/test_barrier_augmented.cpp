// The augmented-system barrier backend (CpuAugmented) solves the same Newton
// system as the normal equations (#270).

#include "lp/newton_solver.h"
#include "mipx/barrier.h"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cmath>
#include <random>
#include <vector>

using namespace mipx;

namespace {

struct NewtonCase {
    SparseMatrix A;
    std::vector<Real> z, s, rp, rd, rc;
};

// A random m x n system with a full-rank diagonal and about a third of the
// other entries filled, so the augmented factor carries fill-in.
NewtonCase randomCase(Index m, Index n, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<Real> u(0.1, 2.0);
    std::vector<Triplet> t;
    for (Index i = 0; i < m; ++i) {
        for (Index j = 0; j < n; ++j) {
            if (j == i || rng() % 3 == 0) {
                t.push_back({i, j, u(rng)});
            }
        }
    }
    NewtonCase c{SparseMatrix(m, n, t), {}, {}, {}, {}, {}};
    auto fill = [&](std::vector<Real>& v, Index len, Real shift) {
        v.resize(static_cast<std::size_t>(len));
        for (auto& x : v) {
            x = u(rng) - shift;
        }
    };
    fill(c.z, n, 0.0);
    fill(c.s, n, 0.0);
    fill(c.rp, m, 1.0);
    fill(c.rd, n, 1.0);
    fill(c.rc, n, 1.0);
    return c;
}

// Largest violation of the regularized Newton equations both backends solve:
// A dz + reg dy = rp, A' dy + ds = rd, s dz + z ds = rc.
Real newtonResidual(const NewtonCase& c, Real reg, const std::vector<Real>& dz,
                    const std::vector<Real>& dy, const std::vector<Real>& ds) {
    const Index m = c.A.numRows();
    const Index n = c.A.numCols();
    std::vector<Real> adz(m, 0.0), aty(n, 0.0);
    c.A.multiply(dz, adz);
    c.A.multiplyTranspose(dy, aty);
    Real r = 0.0;
    for (Index i = 0; i < m; ++i) {
        r = std::max(r, std::abs(adz[i] + reg * dy[i] - c.rp[i]));
    }
    for (Index j = 0; j < n; ++j) {
        r = std::max(r, std::abs(aty[j] + ds[j] - c.rd[j]));
        r = std::max(r, std::abs(c.s[j] * dz[j] + c.z[j] * ds[j] - c.rc[j]));
    }
    return r;
}

}  // namespace

TEST_CASE("CpuAugmented Newton directions match the normal equations", "[barrier][augmented]") {
    // Two bugs made these directions wrong and the IPM diverge (#270): the
    // solve unpermuted its answer with perm instead of iperm, and the
    // symbolic pattern missed the fill a column inherits from its etree
    // children, so L D L' was not K. Either one alone fails this test.
    const auto ordering = GENERATE(BarrierOrdering::Amd, BarrierOrdering::Nd);
    const auto [m, n] = GENERATE(std::pair<Index, Index>{5, 12}, std::pair<Index, Index>{20, 45},
                                 std::pair<Index, Index>{60, 150});
    const unsigned seed = GENERATE(1U, 2U, 3U);
    const Real reg = GENERATE(1e-10, 1e-8, 1e-6);
    INFO("ordering " << static_cast<int>(ordering) << " m " << m << " n " << n << " seed " << seed
                     << " reg " << reg);
    const NewtonCase c = randomCase(m, n, seed);

    BarrierOptions opts;
    opts.verbose = false;
    opts.ordering = ordering;
    auto ne = createCpuCholeskySolver();
    auto aug = createCpuAugmentedSolver();
    REQUIRE(ne->setup(c.A, m, n, opts));
    REQUIRE(aug->setup(c.A, m, n, opts));
    REQUIRE(ne->factorize(c.z, c.s, reg));
    REQUIRE(aug->factorize(c.z, c.s, reg));

    std::vector<Real> dz1(n), dy1(m), ds1(n), dz2(n), dy2(m), ds2(n);
    REQUIRE(ne->solveNewton(c.rp, c.rd, c.rc, dz1, dy1, ds1));
    REQUIRE(aug->solveNewton(c.rp, c.rd, c.rc, dz2, dy2, ds2));

    // Both answers satisfy the (regularized) Newton equations and agree.
    Real diff = 0.0;
    Real scale = 1.0;
    for (Index i = 0; i < m; ++i) {
        diff = std::max(diff, std::abs(dy1[i] - dy2[i]));
        scale = std::max(scale, std::abs(dy1[i]));
    }
    for (Index j = 0; j < n; ++j) {
        diff = std::max({diff, std::abs(dz1[j] - dz2[j]), std::abs(ds1[j] - ds2[j])});
        scale = std::max({scale, std::abs(dz1[j]), std::abs(ds1[j])});
    }
    CHECK(newtonResidual(c, reg, dz1, dy1, ds1) < 1e-6 * scale);
    CHECK(newtonResidual(c, reg, dz2, dy2, ds2) < 1e-6 * scale);
    CHECK(diff < 1e-6 * scale);
}
