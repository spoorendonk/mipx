#include "mipx/clique_table.h"
#include "mipx/conflict_graph.h"
#include "mipx/cut_pool.h"
#include "mipx/cut_strengthening.h"
#include "mipx/dual_simplex.h"
#include "mipx/gomory.h"
#include "mipx/io.h"
#include "mipx/lp_problem.h"
#include "mipx/mip_solver.h"
#include "mipx/separators.h"
#include "test_paths.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <filesystem>
#include <limits>
#include <random>
#include <span>
#include <vector>

using namespace mipx;
using Catch::Matchers::WithinAbs;

// ---------------------------------------------------------------------------
// CutPool tests
// ---------------------------------------------------------------------------

TEST_CASE("CutPool: add and rank by efficacy", "[cuts]") {
    CutPool pool;

    Cut c1;
    c1.indices = {0, 1};
    c1.values = {1.0, 2.0};
    c1.lower = 3.0;
    c1.efficacy = 0.5;

    Cut c2;
    c2.indices = {0, 2};
    c2.values = {1.0, 1.0};
    c2.lower = 2.0;
    c2.efficacy = 0.8;

    Cut c3;
    c3.indices = {1};
    c3.values = {1.0};
    c3.lower = 1.0;
    c3.efficacy = 0.3;

    REQUIRE(pool.addCut(c1));
    REQUIRE(pool.addCut(c2));
    REQUIRE(pool.addCut(c3));
    CHECK(pool.size() == 3);

    auto top = pool.topByEfficacy(2);
    REQUIRE(top.size() == 2);
    CHECK(pool[top[0]].efficacy > pool[top[1]].efficacy);
    CHECK_THAT(pool[top[0]].efficacy, WithinAbs(0.8, 1e-10));
    CHECK_THAT(pool[top[1]].efficacy, WithinAbs(0.5, 1e-10));
}

TEST_CASE("CutPool: reject low efficacy", "[cuts]") {
    CutPool pool;
    pool.setMinEfficacy(0.1);

    Cut c;
    c.indices = {0};
    c.values = {1.0};
    c.lower = 1.0;
    c.efficacy = 0.01;  // Below threshold.

    CHECK_FALSE(pool.addCut(c));
    CHECK(pool.size() == 0);
}

TEST_CASE("CutPool: parallelism filtering", "[cuts]") {
    CutPool pool;
    pool.setParallelismThreshold(0.95);

    // Cuts are sparse: the pool screens out explicit zero coefficients.
    Cut c1;
    c1.indices = {0};
    c1.values = {1.0};
    c1.lower = 1.0;
    c1.efficacy = 0.5;
    REQUIRE(pool.addCut(c1));

    // Same direction, slightly different magnitude.
    Cut c2;
    c2.indices = {0};
    c2.values = {2.0};
    c2.lower = 2.0;
    c2.efficacy = 0.4;  // Lower efficacy, parallel.
    CHECK_FALSE(pool.addCut(c2));

    // Orthogonal cut: accepted.
    Cut c3;
    c3.indices = {1};
    c3.values = {1.0};
    c3.lower = 1.0;
    c3.efficacy = 0.5;
    CHECK(pool.addCut(c3));
    CHECK(pool.size() == 2);
}

TEST_CASE("CutPool: addCut rejects a numerically unsafe cut itself", "[cuts]") {
    // Issue #212. The screen is the pool's, not its callers', so a cut offered
    // straight to addCut is screened too -- however efficacious it claims to be.
    CutPool pool;

    Cut safe;
    safe.indices = {0, 1};
    safe.values = {1.0, 2.0};
    safe.lower = 1.0;
    safe.efficacy = 0.5;
    REQUIRE(isNumericallySafeCut(safe));

    // Each variant fails a different check of isNumericallySafeCut, and the pool
    // is empty and the efficacy high, so only the screen can reject it.
    Cut huge = safe;
    huge.values = {1.0, 2e6};  // max |coef| above 1e6
    Cut unsorted = safe;
    unsorted.indices = {1, 0};
    Cut zero = safe;
    zero.values = {1.0, 0.0};
    Cut non_finite = safe;
    non_finite.values = {1.0, std::numeric_limits<Real>::quiet_NaN()};
    Cut empty = safe;
    empty.indices.clear();
    empty.values.clear();

    for (const Cut* bad : {&huge, &unsorted, &zero, &non_finite, &empty}) {
        REQUIRE_FALSE(isNumericallySafeCut(*bad));
        Cut offered = *bad;
        offered.efficacy = 10.0;
        CHECK_FALSE(pool.addCut(offered));
    }
    CHECK(pool.size() == 0);

    // The safe original still goes in.
    CHECK(pool.addCut(safe));
    CHECK(pool.size() == 1);
}

TEST_CASE("CutPool: age and purge", "[cuts]") {
    CutPool pool;

    Cut c1;
    c1.indices = {0};
    c1.values = {1.0};
    c1.lower = 5.0;  // x0 >= 5
    c1.efficacy = 0.5;

    Cut c2;
    c2.indices = {1};
    c2.values = {1.0};
    c2.lower = 3.0;  // x1 >= 3
    c2.efficacy = 0.3;

    pool.addCut(c1);
    pool.addCut(c2);

    // Primals where c1 is active (x0 = 5.0), c2 is not (x1 = 10.0).
    std::vector<Real> primals = {5.0, 10.0};

    // Age several times.
    for (int i = 0; i < 11; ++i) {
        pool.ageAll(primals, 0.1);
    }

    // c1 should still be age=0 (active), c2 should be age=11.
    CHECK(pool[0].age == 0);
    CHECK(pool[1].age == 11);

    pool.purge(10);
    CHECK(pool.size() == 1);
    CHECK_THAT(pool[0].efficacy, WithinAbs(0.5, 1e-10));
}

// ---------------------------------------------------------------------------
// Helper: build a MIP where cuts can help
// ---------------------------------------------------------------------------

// min -x - y  s.t. 3x + 2y <= 6, x + 4y <= 4, x, y >= 0, x, y integer
// LP relaxation: x = 16/10, y = 6/10, obj = -2.2
// MIP optimal: x = 2, y = 0, obj = -2  OR  x = 0, y = 1, obj = -1
// Actually x=2, y=0: 3*2+2*0=6<=6, 2+0=2<=4. obj = -2. Good.
// x=1, y=1: 3+2=5<=6, 1+4=5>4. Not feasible.
// x=0, y=1: 0+2=2<=6, 0+4=4<=4. obj = -1.
// x=2, y=0 is optimal.
static LpProblem buildCutTestMip() {
    LpProblem lp;
    lp.name = "cut_test";
    lp.sense = Sense::Minimize;
    lp.num_cols = 2;
    lp.obj = {-1.0, -1.0};
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {kInf, kInf};
    lp.col_type = {VarType::Integer, VarType::Integer};
    lp.col_names = {"x", "y"};

    lp.num_rows = 2;
    lp.row_lower = {-kInf, -kInf};
    lp.row_upper = {6.0, 4.0};
    lp.row_names = {"r1", "r2"};

    std::vector<Triplet> trips = {
        {0, 0, 3.0},
        {0, 1, 2.0},
        {1, 0, 1.0},
        {1, 1, 4.0},
    };
    lp.matrix = SparseMatrix(2, 2, std::move(trips));
    return lp;
}

// A slightly harder problem where cuts should help:
// min -10x - 20y  s.t. x + y <= 5.5, x, y >= 0, integer
// LP opt: x=0, y=5.5, obj=-110
// MIP opt: x=0, y=5, obj=-100  or  x=5, y=0, obj=-50
// Best: y=5, x=0 -> -100.
static LpProblem buildFractionalMip() {
    LpProblem lp;
    lp.name = "fractional_mip";
    lp.sense = Sense::Minimize;
    lp.num_cols = 2;
    lp.obj = {-10.0, -20.0};
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {kInf, kInf};
    lp.col_type = {VarType::Integer, VarType::Integer};
    lp.col_names = {"x", "y"};

    lp.num_rows = 1;
    lp.row_lower = {-kInf};
    lp.row_upper = {5.5};
    lp.row_names = {"cap"};

    std::vector<Triplet> trips = {
        {0, 0, 1.0},
        {0, 1, 1.0},
    };
    lp.matrix = SparseMatrix(1, 2, std::move(trips));
    return lp;
}

// min -x -y -z  s.t. x + y <= 1, y + z <= 1, x + z <= 1, all binary.
// Every pair conflicts, so x + y + z <= 1 is valid, but no single row implies
// it. The LP relaxation optimum is the unique fractional point (.5, .5, .5),
// where each pairwise inequality is tight and therefore not separable.
static LpProblem buildTriangleCliqueMip() {
    LpProblem lp;
    lp.name = "triangle_clique";
    lp.sense = Sense::Minimize;
    lp.num_cols = 3;
    lp.obj = {-1.0, -1.0, -1.0};
    lp.col_lower = {0.0, 0.0, 0.0};
    lp.col_upper = {1.0, 1.0, 1.0};
    lp.col_type = {VarType::Binary, VarType::Binary, VarType::Binary};
    lp.col_names = {"x", "y", "z"};

    lp.num_rows = 3;
    lp.row_lower = {-kInf, -kInf, -kInf};
    lp.row_upper = {1.0, 1.0, 1.0};
    lp.row_names = {"xy", "yz", "xz"};

    std::vector<Triplet> trips = {
        {0, 0, 1.0}, {0, 1, 1.0}, {1, 1, 1.0}, {1, 2, 1.0}, {2, 0, 1.0}, {2, 2, 1.0},
    };
    lp.matrix = SparseMatrix(3, 3, std::move(trips));
    return lp;
}

// min -x1 -x2 -x3  s.t. x1 + x2 + x3 <= 1.5, x binary
// LP optimum is fractional (sum 1.5), good for cover/clique separation tests.
static LpProblem buildBinaryConflictMip() {
    LpProblem lp;
    lp.name = "binary_conflict";
    lp.sense = Sense::Minimize;
    lp.num_cols = 3;
    lp.obj = {-3.0, -2.0, -0.1};
    lp.col_lower = {0.0, 0.0, 0.0};
    lp.col_upper = {1.0, 1.0, 1.0};
    lp.col_type = {VarType::Binary, VarType::Binary, VarType::Binary};
    lp.col_names = {"x1", "x2", "x3"};

    lp.num_rows = 1;
    lp.row_lower = {-kInf};
    lp.row_upper = {1.5};
    lp.row_names = {"cap"};

    std::vector<Triplet> trips = {
        {0, 0, 1.0},
        {0, 1, 1.0},
        {0, 2, 1.0},
    };
    lp.matrix = SparseMatrix(1, 3, std::move(trips));
    return lp;
}

// min -x  s.t. x <= 2.7, x integer
// MIR-style rounding gives x <= 2 at the root LP solution x = 2.7.
static LpProblem buildMirTestMip() {
    LpProblem lp;
    lp.name = "mir_test";
    lp.sense = Sense::Minimize;
    lp.num_cols = 1;
    lp.obj = {-1.0};
    lp.col_lower = {0.0};
    lp.col_upper = {10.0};
    lp.col_type = {VarType::Integer};
    lp.col_names = {"x"};

    lp.num_rows = 1;
    lp.row_lower = {-kInf};
    lp.row_upper = {2.7};
    lp.row_names = {"ub"};

    std::vector<Triplet> trips = {
        {0, 0, 1.0},
    };
    lp.matrix = SparseMatrix(1, 1, std::move(trips));
    return lp;
}

// min -x  s.t. x - 20 y <= 2.000002, x integer in [0, 1000], y continuous in
// [0, 10].
//
// The LP optimum is y = 10 (at its upper bound), x = 202.000002 basic, and the
// row tight. The fractional part of the basic integer variable is f0 = 2e-6,
// barely above GomorySeparator's kIntTol of 1e-6, while the tableau coefficient
// on the continuous nonbasic y is 20. A GMI cut off that row therefore carries
// the coefficient t/f0 = 1e7 on y — an order of magnitude past the safety
// screen's 1e6 limit on the largest coefficient, and a numerically worthless
// restatement of y <= 10.
static LpProblem buildGomoryBlowupMip() {
    LpProblem lp;
    lp.name = "gomory_blowup";
    lp.sense = Sense::Minimize;
    lp.num_cols = 2;
    lp.obj = {-1.0, 0.0};
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {1000.0, 10.0};
    lp.col_type = {VarType::Integer, VarType::Continuous};
    lp.col_names = {"x", "y"};

    lp.num_rows = 1;
    lp.row_lower = {-kInf};
    lp.row_upper = {2.000002};
    lp.row_names = {"link"};

    std::vector<Triplet> trips = {
        {0, 0, 1.0},
        {0, 1, -20.0},
    };
    lp.matrix = SparseMatrix(1, 2, std::move(trips));
    return lp;
}

// A pure-integer MIP that exercises every row sense the GMI substitution has
// to handle: a <= row, a >= row, a ranged row and an equality row. All four
// logicals can end up nonbasic, and each is substituted back with a different
// sign/bound pairing, so a sign error in any one of them shows up as an
// invalid cut in the enumeration test below.
//
// min -2x - 3y - z
//   4x + 5y + 3z <= 13.5        (<=)
//   -x + 2y      >= -1.5        (>=)
//   2x +  y      in [1.5, 4.5]  (ranged)
//    x +  y + 2z  = 4           (equality, integral data)
//   x, y, z integer in [0, 3]
//
// The integer-feasible set is {(1, 1, 1), (0, 2, 1)}; the LP relaxation
// optimum is fractional.
static LpProblem buildMixedSenseMip() {
    LpProblem lp;
    lp.name = "mixed_sense";
    lp.sense = Sense::Minimize;
    lp.num_cols = 3;
    lp.obj = {-2.0, -3.0, -1.0};
    lp.col_lower = {0.0, 0.0, 0.0};
    lp.col_upper = {3.0, 3.0, 3.0};
    lp.col_type = {VarType::Integer, VarType::Integer, VarType::Integer};
    lp.col_names = {"x", "y", "z"};

    lp.num_rows = 4;
    lp.row_lower = {-kInf, -1.5, 1.5, 4.0};
    lp.row_upper = {13.5, kInf, 4.5, 4.0};
    lp.row_names = {"le", "ge", "ranged", "eq"};

    std::vector<Triplet> trips = {
        {0, 0, 4.0}, {0, 1, 5.0}, {0, 2, 3.0}, {1, 0, -1.0}, {1, 1, 2.0},
        {2, 0, 2.0}, {2, 1, 1.0}, {3, 0, 1.0}, {3, 1, 1.0},  {3, 2, 2.0},
    };
    lp.matrix = SparseMatrix(4, 3, std::move(trips));
    return lp;
}

// Check every cut in the pool against every integer point of the box
// [0, limit]^n that is feasible for the model. A cut that fails here removed a
// MIP-feasible point, which includes the integer optimum.
//
// Only for all-integer models: a continuous column cannot be enumerated.
static void checkCutsKeepAllFeasiblePoints(const LpProblem& problem, const CutPool& pool,
                                           Index limit) {
    for (Index j = 0; j < problem.num_cols; ++j) {
        REQUIRE(problem.col_type[j] != VarType::Continuous);
    }
    REQUIRE(pool.size() > 0);

    const Real tol = 1e-7;
    std::vector<Real> point(static_cast<std::size_t>(problem.num_cols), 0.0);
    Index checked = 0;
    while (true) {
        bool feasible = true;
        for (Index j = 0; j < problem.num_cols && feasible; ++j) {
            feasible =
                point[j] >= problem.col_lower[j] - tol && point[j] <= problem.col_upper[j] + tol;
        }
        for (Index i = 0; i < problem.num_rows && feasible; ++i) {
            Real activity = 0.0;
            auto rv = problem.matrix.row(i);
            for (Index p = 0; p < rv.size(); ++p) {
                activity += rv.values[p] * point[rv.indices[p]];
            }
            feasible =
                activity >= problem.row_lower[i] - tol && activity <= problem.row_upper[i] + tol;
        }

        if (feasible) {
            ++checked;
            for (Index ci = 0; ci < pool.size(); ++ci) {
                const Cut& cut = pool[ci];
                Real lhs = 0.0;
                for (Index k = 0; k < static_cast<Index>(cut.indices.size()); ++k) {
                    lhs += cut.values[k] * point[cut.indices[k]];
                }
                INFO("cut " << ci << " lhs " << lhs << " rhs " << cut.lower);
                CHECK(lhs >= cut.lower - 1e-6);
                if (cut.upper < kInf) {
                    CHECK(lhs <= cut.upper + 1e-6);
                }
            }
        }

        Index pos = 0;
        while (pos < problem.num_cols && point[pos] >= static_cast<Real>(limit) - 0.5) {
            point[pos] = 0.0;
            ++pos;
        }
        if (pos == problem.num_cols) {
            break;
        }
        point[pos] += 1.0;
    }
    // A validity sweep that found no feasible point proves nothing.
    REQUIRE(checked > 0);
}

// ---------------------------------------------------------------------------
// Gomory separator tests
// ---------------------------------------------------------------------------

TEST_CASE("Gomory: generate cuts on small MIP", "[cuts][gomory]") {
    auto problem = buildFractionalMip();

    DualSimplexSolver lp;
    lp.load(problem);
    auto result = lp.solve();
    REQUIRE(result.status == Status::Optimal);

    auto primals = lp.getPrimalValues();

    // y should be fractional (5.5).
    bool has_fractional = false;
    for (Index j = 0; j < problem.num_cols; ++j) {
        if (problem.col_type[j] != VarType::Continuous) {
            Real frac = std::abs(primals[j] - std::round(primals[j]));
            if (frac > 1e-6) {
                has_fractional = true;
            }
        }
    }
    CHECK(has_fractional);

    CutPool pool;
    GomorySeparator gomory;
    Int cuts = gomory.separate(lp, problem, primals, pool);

    // At least one cut off the fractional variable. This is the assertion that
    // issue #211 is about: it was CHECK(cuts >= 0), which no behaviour can
    // fail, and it hid a separator that returned zero on every instance.
    CHECK(cuts >= 1);
    CHECK(pool.size() >= 1);

    // The row is used, not skipped: the tableau row of a basic structural
    // variable always carries a nonbasic logical, and substituting it is the
    // whole point of the fix.
    const auto& stats = gomory.stats();
    CHECK(stats.candidate_rows >= 1);
    CHECK(stats.rows_skipped == 0);
    CHECK(stats.accepted == cuts);
    CHECK(stats.local_cuts == 0);

    // Verify any generated cuts are valid (not violated by the LP relaxation
    // optimum... actually they SHOULD be violated, that's the point).
    for (Index i = 0; i < pool.size(); ++i) {
        const auto& cut = pool[i];
        CHECK(cut.efficacy > 0.0);
        CHECK(!cut.indices.empty());

        // Verify the cut is violated by the current LP solution.
        Real lhs = 0.0;
        for (Index k = 0; k < static_cast<Index>(cut.indices.size()); ++k) {
            lhs += cut.values[k] * primals[cut.indices[k]];
        }
        // For a >= cut: lhs should be < rhs (violated).
        if (cut.lower > -kInf) {
            CHECK(lhs < cut.lower + 1e-3);  // violated or nearly so
        }
    }
}

TEST_CASE("Gomory: safety screen rejects a GMI coefficient blow-up", "[cuts][gomory]") {
    // The cut buildGomoryBlowupMip's tableau row produces: y is nonbasic at its
    // upper bound with t = 20, f0 = 2e-6, so gmi_coeff = t / f0 = 1e7, giving
    // -1e7 y >= 1 - 1e7 * 10 — a numerically worthless restatement of y <= 10.
    Cut blown_up;
    blown_up.indices = {1};
    blown_up.values = {-1e7};
    blown_up.lower = 1.0 - 1e7 * 10.0;
    blown_up.family = CutFamily::Gomory;

    // Only the largest-coefficient check trips: the squared norm is 1e14 (limit
    // 1e16) and a single coefficient cannot trip the max/min ratio, so this
    // pins max |coef| > 1e6 specifically rather than passing for some unrelated
    // reason.
    CHECK_FALSE(isNumericallySafeCut(blown_up));

    // The same cut with the coefficient back under the limit is accepted, so
    // the rejection above is the magnitude and not the cut's structure.
    Cut scaled_down = blown_up;
    scaled_down.values = {-9e5};
    scaled_down.lower = 1.0 - 9e5 * 10.0;
    CHECK(isNumericallySafeCut(scaled_down));

    // A second nonbasic at a sane magnitude does not rescue it either, and the
    // max/min ratio here is 1.7e7, still inside the 1e8 limit.
    Cut mixed = blown_up;
    mixed.indices = {0, 1};
    mixed.values = {0.6, -1e7};
    CHECK_FALSE(isNumericallySafeCut(mixed));
}

TEST_CASE("Gomory: blow-up row is never pooled", "[cuts][gomory]") {
    auto problem = buildGomoryBlowupMip();

    DualSimplexSolver lp;
    lp.load(problem);
    auto result = lp.solve();
    REQUIRE(result.status == Status::Optimal);

    auto primals = lp.getPrimalValues();

    // x is basic with a fractional part just above kIntTol, which is what makes
    // the GMI coefficient on y explode.
    const Real f0 = primals[0] - std::floor(primals[0]);
    CHECK(f0 > 1e-6);
    CHECK(f0 < 1e-4);

    CutPool pool;
    GomorySeparator gomory;
    const Int cuts = gomory.separate(lp, problem, primals, pool);

    // Nothing is pooled, and since #211 that is because the built cut is
    // rejected rather than because no cut was built: the row is a candidate
    // and the cut is constructed, which #196 could not demonstrate while the
    // separator emitted nothing at all on any instance.
    CHECK(cuts == 0);
    CHECK(pool.size() == 0);
    const auto& stats = gomory.stats();
    CHECK(stats.candidate_rows == 1);
    CHECK(stats.rows_skipped == 0);
    CHECK(stats.cuts_built == 1);
    CHECK(stats.accepted == 0);

    // What turns it away changed with #216. The 1e7 on y from y's own
    // deviation and the 1e7 from substituting the row logical cancel exactly;
    // the float residual of that sum (~4e-9) used to stay in the cut and trip
    // the max/min ratio screen. It is now dropped and paid for over y's range
    // [0, 10], which leaves the true GMI cut -5e5 x >= rhs, i.e. x <= 202:
    // under every screen limit, and with an efficacy of ~2e-6 the pool refuses
    // it.
    CHECK(stats.rejected_screen == 0);
    CHECK(stats.rejected_pool == 1);

    // Invariant that holds either way and is the point of issue #196: whatever
    // Gomory does pool has passed the shared numerical safety screen.
    for (Index i = 0; i < pool.size(); ++i) {
        CHECK(isNumericallySafeCut(pool[i]));
    }
}

TEST_CASE("Gomory: cuts keep every integer-feasible point", "[cuts][gomory]") {
    // A GMI cut is derived from one tableau row and must still hold at every
    // integer-feasible point of the model, the optimum included. The row-sense
    // mix here is deliberate: <=, >=, ranged and equality rows each hand their
    // logical to the substitution with a different sign and bound.
    auto problem = buildMixedSenseMip();

    DualSimplexSolver lp;
    lp.load(problem);
    auto result = lp.solve();
    REQUIRE(result.status == Status::Optimal);

    auto primals = lp.getPrimalValues();
    CutPool pool;
    GomorySeparator gomory;
    const Int cuts = gomory.separate(lp, problem, primals, pool);
    REQUIRE(cuts >= 1);

    checkCutsKeepAllFeasiblePoints(problem, pool, 3);
}

TEST_CASE("Gomory: cuts on the fractional MIP keep the integer optimum", "[cuts][gomory]") {
    auto problem = buildFractionalMip();

    DualSimplexSolver lp;
    lp.load(problem);
    REQUIRE(lp.solve().status == Status::Optimal);

    auto primals = lp.getPrimalValues();
    CutPool pool;
    GomorySeparator gomory;
    REQUIRE(gomory.separate(lp, problem, primals, pool) >= 1);

    // x + y <= 5.5 with x, y >= 0 integer: every feasible point lies in
    // [0, 5]^2, so the box sweep is exhaustive here.
    checkCutsKeepAllFeasiblePoints(problem, pool, 6);

    // The MIP optimum spelled out, so the sweep cannot go vacuous unnoticed:
    // x = 0, y = 5 is feasible and optimal at objective -100.
    const std::vector<Real> opt = {0.0, 5.0};
    for (Index ci = 0; ci < pool.size(); ++ci) {
        const auto& cut = pool[ci];
        Real lhs = 0.0;
        for (Index k = 0; k < static_cast<Index>(cut.indices.size()); ++k) {
            lhs += cut.values[k] * opt[cut.indices[k]];
        }
        INFO("cut " << ci << " at the integer optimum");
        CHECK(lhs >= cut.lower - 1e-6);
    }
}

TEST_CASE("Gomory: a cut off a tightened bound is marked local", "[cuts][gomory]") {
    // The deviation of a nonbasic column is measured from the bound the LP
    // actually holds it at. When that bound is tighter than the model's, the
    // cut is only valid in the subtree that tightened it, and must say so: the
    // tree cut path promotes cuts to global rows and may not promote this one.
    auto problem = buildFractionalMip();

    DualSimplexSolver lp;
    lp.load(problem);
    REQUIRE(lp.solve().status == Status::Optimal);

    // y is unbounded above in the model; branch it down to 5. It then sits
    // nonbasic at 5 and x = 0.5 becomes the fractional basic integer variable.
    lp.setColBounds(1, 0.0, 5.0);
    REQUIRE(lp.solve().status == Status::Optimal);

    auto primals = lp.getPrimalValues();
    REQUIRE(std::abs(primals[1] - 5.0) < 1e-9);

    CutPool pool;
    GomorySeparator gomory;
    const Int cuts = gomory.separate(lp, problem, primals, pool);
    REQUIRE(cuts >= 1);
    CHECK(gomory.stats().local_cuts == cuts);
    for (Index ci = 0; ci < pool.size(); ++ci) {
        CHECK(pool[ci].local);
    }
}

TEST_CASE("Gomory: cuts off a vouched-for cut row stay global", "[cuts][gomory]") {
    // Rows added after load() are node-local as far as the separator can tell,
    // so it flags cuts built off them -- unless the caller states how many
    // leading rows hold tree-wide, which is what the root cut loop does.
    auto problem = buildMixedSenseMip();

    DualSimplexSolver lp;
    lp.load(problem);
    REQUIRE(lp.solve().status == Status::Optimal);

    // A redundant but globally valid row: 4x + 5y + 3z <= 20.
    const std::vector<Index> starts = {0};
    const std::vector<Index> indices = {0, 1, 2};
    const std::vector<Real> values = {4.0, 5.0, 3.0};
    const std::vector<Real> lower = {-kInf};
    const std::vector<Real> upper = {20.0};
    lp.addRows(starts, indices, values, lower, upper);
    REQUIRE(lp.solve().status == Status::Optimal);

    auto primals = lp.getPrimalValues();

    CutPool vouched_pool;
    GomorySeparator vouched;
    vouched.setGlobalRowCount(lp.numRows());
    const Int vouched_cuts = vouched.separate(lp, problem, primals, vouched_pool);
    REQUIRE(vouched_cuts >= 1);
    CHECK(vouched.stats().local_cuts == 0);
    for (Index ci = 0; ci < vouched_pool.size(); ++ci) {
        CHECK_FALSE(vouched_pool[ci].local);
    }

    // Without the declaration the same cuts are built, only flagged local: the
    // setting is about where a cut may be used, not about which cuts exist.
    CutPool defaulted_pool;
    GomorySeparator defaulted;
    const Int defaulted_cuts = defaulted.separate(lp, problem, primals, defaulted_pool);
    REQUIRE(defaulted_cuts == vouched_cuts);
    REQUIRE(defaulted_pool.size() == vouched_pool.size());
    for (Index ci = 0; ci < vouched_pool.size(); ++ci) {
        REQUIRE(defaulted_pool[ci].indices == vouched_pool[ci].indices);
        for (std::size_t k = 0; k < vouched_pool[ci].values.size(); ++k) {
            CHECK_THAT(defaulted_pool[ci].values[k], WithinAbs(vouched_pool[ci].values[k], 1e-12));
        }
    }

    // Rows inside the prefix can still be named local one by one; that is how
    // the tree cut loop describes local rows interleaved with promoted ones.
    // Every cut leans on some nonbasic logical (a basic structural's tableau
    // row always carries one), so naming every row local makes every cut so.
    std::vector<Index> all_rows(static_cast<std::size_t>(lp.numRows()));
    for (Index r = 0; r < lp.numRows(); ++r) {
        all_rows[static_cast<std::size_t>(r)] = r;
    }
    CutPool listed_pool;
    GomorySeparator listed;
    listed.setGlobalRowCount(lp.numRows());
    listed.setLocalRows(all_rows);
    const Int listed_cuts = listed.separate(lp, problem, primals, listed_pool);
    REQUIRE(listed_cuts == vouched_cuts);
    CHECK(listed.stats().local_cuts == listed_cuts);

    checkCutsKeepAllFeasiblePoints(problem, vouched_pool, 3);
}

TEST_CASE("SeparatorManager: Gomory contributes accepted cuts", "[cuts][gomory]") {
    // End to end through the manager, which is the path the root cut loop
    // takes: the family stats must show Gomory accepting cuts, not just the
    // separator in isolation.
    auto problem = buildMixedSenseMip();

    DualSimplexSolver lp;
    lp.load(problem);
    REQUIRE(lp.solve().status == Status::Optimal);
    auto primals = lp.getPrimalValues();

    CutFamilyConfig only_gomory{};
    only_gomory.mir = false;
    only_gomory.cover = false;
    only_gomory.implied_bound = false;
    only_gomory.clique = false;
    only_gomory.zero_half = false;
    only_gomory.mixing = false;
    only_gomory.cmir = false;
    only_gomory.strong_cg = false;
    only_gomory.lifted_cover = false;
    only_gomory.mod_k = false;
    only_gomory.intersection_cut = false;
    only_gomory.multi_row = false;

    SeparatorManager manager;
    manager.setConfig(only_gomory);
    manager.setGlobalRowCount(lp.numRows());
    CutPool pool;
    CutSeparationStats stats;
    const Int added = manager.separate(lp, problem, primals, pool, stats);

    CHECK(added >= 1);
    CHECK(stats.at(CutFamily::Gomory).accepted == added);
    checkCutsKeepAllFeasiblePoints(problem, pool, 3);
    for (Index ci = 0; ci < pool.size(); ++ci) {
        CHECK(pool[ci].family == CutFamily::Gomory);
        CHECK(isNumericallySafeCut(pool[ci]));
    }
}

TEST_CASE("CutFamilyConfig defaults keep Gomory enabled", "[cuts][gomory]") {
    const CutFamilyConfig config{};
    CHECK(config.gomory);
}

TEST_CASE("Gomory: a dropped coefficient is paid for in the rhs", "[cuts][gomory]") {
    // x0 + 5e-11 x1 - 5e-11 x2 >= 1 over x0 in [0, 1], x1 in [0, 1e12],
    // x2 in [-1e12, 0]. Both tiny terms fall under the drop tolerance, and
    // x1, x2 range wide enough that together they are worth up to 100.
    std::vector<Real> coeff = {1.0, 5e-11, -5e-11};
    const std::vector<Real> lower = {0.0, 0.0, -1e12};
    const std::vector<Real> upper = {1.0, 1e12, 0.0};
    const Real rhs = 1.0;

    // A point of the box that satisfies the original cut only through the
    // tiny terms: 0.5 + 5e-11 * 1e10 - 5e-11 * (-1e10) = 1.5.
    const std::vector<Real> point = {0.5, 1e10, -1e10};
    auto lhs = [&](const std::vector<Real>& c) {
        Real sum = 0.0;
        for (std::size_t j = 0; j < c.size(); ++j) {
            sum += c[j] * point[j];
        }
        return sum;
    };
    REQUIRE(lhs(coeff) >= rhs);

    Int dropped = 0;
    const Real relaxed = dropAndRelax(coeff, rhs, lower, upper, 1e-10, &dropped);
    CHECK(dropped == 2);
    CHECK(coeff[1] == 0.0);
    CHECK(coeff[2] == 0.0);
    CHECK_THAT(relaxed, WithinAbs(1.0 - 50.0 - 50.0, 1e-9));

    // Dropping without the payment (x0 >= 1) would cut the point off; the
    // relaxed cut keeps it.
    CHECK(lhs(coeff) < rhs);
    CHECK(lhs(coeff) >= relaxed);
}

TEST_CASE("Gomory: a tiny coefficient on an unbounded column is kept", "[cuts][gomory]") {
    // No finite bound in the direction the payment needs, so the term cannot
    // be dropped soundly and stays in the cut.
    std::vector<Real> coeff = {1.0, 5e-11, -5e-11};
    const std::vector<Real> lower = {0.0, 0.0, -kInf};
    const std::vector<Real> upper = {1.0, kInf, 0.0};

    Int dropped = 0;
    const Real relaxed = dropAndRelax(coeff, 1.0, lower, upper, 1e-10, &dropped);
    CHECK(dropped == 0);
    CHECK(coeff[1] == 5e-11);
    CHECK(coeff[2] == -5e-11);
    CHECK(relaxed == 1.0);
}

TEST_CASE("Gomory: cancellation residue is paid where bounded, zeroed where not",
          "[cuts][gomory]") {
    // Each coefficient here but the first is what is left of terms summing to
    // the given magnitude. Within 1e-12 of that magnitude it counts as
    // round-off of the sum: paid for over a bounded column like any small
    // term, and zeroed unpaid on an unbounded one -- but only while it is also
    // under the absolute drop tolerance, since a relative residue of a huge
    // sum can be large enough to matter. A residue the same size on a column
    // whose own terms were tiny is genuine and, unbounded, has to stay.
    std::vector<Real> coeff = {1.0, 1e-5, 1e-5, 5e-11, 5e-11};
    const std::vector<Real> magnitude = {1.0, 1e8, 1e8, 1e8, 5e-11};
    const std::vector<Real> lower = {0.0, 0.0, 0.0, 0.0, 0.0};
    const std::vector<Real> upper = {1.0, 10.0, kInf, kInf, kInf};

    Int dropped = 0;
    const Real relaxed = dropAndRelax(coeff, 1.0, lower, upper, 1e-10, &dropped, magnitude, 1e-12);
    CHECK(dropped == 1);
    CHECK_THAT(relaxed, WithinAbs(1.0 - 1e-4, 1e-15));
    CHECK(coeff[1] == 0.0);    // bounded: paid
    CHECK(coeff[2] == 1e-5);   // unbounded, above drop_tol: kept
    CHECK(coeff[3] == 0.0);    // unbounded round-off under drop_tol: zeroed
    CHECK(coeff[4] == 5e-11);  // unbounded and genuine: kept
}

TEST_CASE("SeparatorManager: Gomory family stats carry the separator's counts", "[cuts][gomory]") {
    CutFamilyConfig only_gomory{};
    only_gomory.mir = false;
    only_gomory.cover = false;
    only_gomory.implied_bound = false;
    only_gomory.clique = false;
    only_gomory.zero_half = false;
    only_gomory.mixing = false;
    only_gomory.cmir = false;
    only_gomory.strong_cg = false;
    only_gomory.lifted_cover = false;
    only_gomory.mod_k = false;
    only_gomory.intersection_cut = false;
    only_gomory.multi_row = false;

    SECTION("a row whose cut is rejected") {
        // One candidate row, one cut built, and the pool turns it away:
        // attempted and generated must say so rather than echo the accepted
        // count of zero.
        auto problem = buildGomoryBlowupMip();
        DualSimplexSolver lp;
        lp.load(problem);
        REQUIRE(lp.solve().status == Status::Optimal);
        auto primals = lp.getPrimalValues();

        SeparatorManager manager;
        manager.setConfig(only_gomory);
        CutPool pool;
        CutSeparationStats stats;
        CHECK(manager.separate(lp, problem, primals, pool, stats) == 0);

        const auto& s = stats.at(CutFamily::Gomory);
        CHECK(s.attempted == 1);
        CHECK(s.generated == 1);
        CHECK(s.accepted == 0);
        CHECK(s.efficacy_sum == 0.0);
    }

    SECTION("rows whose cuts are accepted") {
        auto problem = buildMixedSenseMip();
        DualSimplexSolver lp;
        lp.load(problem);
        REQUIRE(lp.solve().status == Status::Optimal);
        auto primals = lp.getPrimalValues();

        SeparatorManager manager;
        manager.setConfig(only_gomory);
        CutPool pool;
        CutSeparationStats stats;
        const Int added = manager.separate(lp, problem, primals, pool, stats);
        REQUIRE(added >= 1);

        // The same separator run directly gives the reference numbers.
        CutPool ref_pool;
        GomorySeparator ref;
        ref.setMinViolation(1e-5);  // SeparatorManager's default
        REQUIRE(ref.separate(lp, problem, primals, ref_pool) == added);
        const auto& g = ref.stats();

        const auto& s = stats.at(CutFamily::Gomory);
        CHECK(s.attempted == g.candidate_rows);
        CHECK(s.generated == g.cuts_built);
        CHECK(s.accepted == added);
        CHECK(s.attempted >= s.generated);
        CHECK(s.generated >= s.accepted);

        Real pooled_efficacy = 0.0;
        for (Index ci = 0; ci < pool.size(); ++ci) {
            pooled_efficacy += pool[ci].efficacy;
        }
        CHECK(s.efficacy_sum > 0.0);
        CHECK_THAT(s.efficacy_sum, WithinAbs(pooled_efficacy, 1e-12));
    }
}

TEST_CASE("SeparatorManager: cover family generates tagged cuts", "[cuts][families]") {
    auto problem = buildBinaryConflictMip();
    DualSimplexSolver lp;
    lp.load(problem);
    auto result = lp.solve();
    REQUIRE(result.status == Status::Optimal);

    CutPool pool;
    SeparatorManager manager;
    CutFamilyConfig config;
    config.gomory = false;
    config.mir = false;
    config.cover = true;
    config.implied_bound = false;
    config.clique = false;
    config.zero_half = false;
    config.mixing = false;
    config.cmir = false;
    config.strong_cg = false;
    config.lifted_cover = false;
    config.mod_k = false;
    config.intersection_cut = false;
    config.multi_row = false;
    manager.setConfig(config);
    manager.setMaxCutsPerFamily(10);

    CutSeparationStats stats;
    Int cuts = manager.separate(lp, problem, lp.getPrimalValues(), pool, stats);
    CHECK(cuts >= 1);
    CHECK(stats.at(CutFamily::Cover).attempted >= 1);
    CHECK(stats.at(CutFamily::Cover).accepted >= 1);
    for (Index i = 0; i < pool.size(); ++i) {
        CHECK(pool[i].family == CutFamily::Cover);
    }
}

TEST_CASE("SeparatorManager: clique family generates conflict cuts", "[cuts][families]") {
    auto problem = buildBinaryConflictMip();
    DualSimplexSolver lp;
    lp.load(problem);
    auto result = lp.solve();
    REQUIRE(result.status == Status::Optimal);

    CutPool pool;
    SeparatorManager manager;
    CutFamilyConfig config;
    config.gomory = false;
    config.mir = false;
    config.cover = false;
    config.implied_bound = false;
    config.clique = true;
    config.zero_half = false;
    config.mixing = false;
    config.cmir = false;
    config.strong_cg = false;
    config.lifted_cover = false;
    config.mod_k = false;
    config.intersection_cut = false;
    config.multi_row = false;
    manager.setConfig(config);
    manager.setMaxCutsPerFamily(10);

    CutSeparationStats stats;
    Int cuts = manager.separate(lp, problem, lp.getPrimalValues(), pool, stats);
    CHECK(cuts >= 1);
    CHECK(stats.at(CutFamily::Clique).attempted >= 1);
    CHECK(stats.at(CutFamily::Clique).accepted >= 1);
    for (Index i = 0; i < pool.size(); ++i) {
        CHECK(pool[i].family == CutFamily::Clique);
    }
}

static CutFamilyConfig cliqueOnlyConfig() {
    CutFamilyConfig config;
    config.gomory = false;
    config.mir = false;
    config.cover = false;
    config.implied_bound = false;
    config.clique = true;
    config.zero_half = false;
    config.mixing = false;
    config.cmir = false;
    config.strong_cg = false;
    config.lifted_cover = false;
    config.mod_k = false;
    config.intersection_cut = false;
    config.multi_row = false;
    return config;
}

TEST_CASE("SeparatorManager: clique table separates maximal clique cuts", "[cuts][clique]") {
    auto problem = buildTriangleCliqueMip();
    DualSimplexSolver lp;
    lp.setVerbose(false);
    lp.load(problem);
    auto result = lp.solve();
    REQUIRE(result.status == Status::Optimal);

    auto primals = lp.getPrimalValues();
    REQUIRE(primals.size() == 3);
    for (Index j = 0; j < 3; ++j) {
        CHECK_THAT(primals[j], WithinAbs(0.5, 1e-9));
    }

    const CutFamilyConfig config = cliqueOnlyConfig();

    // Without a clique table the pairwise row scan only reproduces the rows,
    // which are tight at the LP optimum, so nothing is violated.
    {
        CutPool pool;
        SeparatorManager manager;
        manager.setConfig(config);
        manager.setMaxCutsPerFamily(10);
        CutSeparationStats stats;
        CHECK(manager.separate(lp, problem, primals, pool, stats) == 0);
        CHECK(pool.size() == 0);
    }

    ConflictGraph graph;
    graph.build(problem);
    REQUIRE(graph.numEdges() == 3);

    CliqueTable table;
    table.build(problem, graph);
    REQUIRE(table.numCliques() >= 1);

    CutPool pool;
    SeparatorManager manager;
    manager.setConfig(config);
    manager.setMaxCutsPerFamily(10);
    manager.setCliqueTable(&table);

    CutSeparationStats stats;
    const Int cuts = manager.separate(lp, problem, primals, pool, stats);
    REQUIRE(cuts >= 1);
    CHECK(stats.at(CutFamily::Clique).accepted == cuts);
    CHECK(stats.at(CutFamily::Clique).generated >= 1);
    CHECK(stats.at(CutFamily::Clique).efficacy_sum > 0.0);

    bool found_triangle_cut = false;
    for (Index i = 0; i < pool.size(); ++i) {
        const auto& cut = pool[i];
        CHECK(cut.family == CutFamily::Clique);
        if (cut.indices.size() != 3) {
            continue;
        }
        found_triangle_cut = true;
        CHECK(cut.indices == std::vector<Index>{0, 1, 2});
        for (Real v : cut.values) {
            CHECK_THAT(v, WithinAbs(1.0, 1e-12));
        }
        CHECK_THAT(cut.upper, WithinAbs(1.0, 1e-12));
    }
    CHECK(found_triangle_cut);
}

TEST_CASE("SeparatorManager: MIR family generates rounding cuts", "[cuts][families]") {
    auto problem = buildMirTestMip();
    DualSimplexSolver lp;
    lp.load(problem);
    auto result = lp.solve();
    REQUIRE(result.status == Status::Optimal);

    CutPool pool;
    SeparatorManager manager;
    CutFamilyConfig config;
    config.gomory = false;
    config.mir = true;
    config.cover = false;
    config.implied_bound = false;
    config.clique = false;
    config.zero_half = false;
    config.mixing = false;
    manager.setConfig(config);
    manager.setMaxCutsPerFamily(10);

    CutSeparationStats stats;
    Int cuts = manager.separate(lp, problem, lp.getPrimalValues(), pool, stats);
    CHECK(cuts >= 1);
    CHECK(stats.at(CutFamily::Mir).attempted >= 1);
    CHECK(stats.at(CutFamily::Mir).accepted >= 1);
    for (Index i = 0; i < pool.size(); ++i) {
        CHECK(pool[i].family == CutFamily::Mir);
    }
}

// ---------------------------------------------------------------------------
// Integration tests: MIP solver with and without cuts
// ---------------------------------------------------------------------------

TEST_CASE("MipSolver with cuts: simple MIP", "[cuts][integration]") {
    auto lp = buildCutTestMip();

    MipSolver solver;
    solver.setVerbose(false);
    solver.setCutsEnabled(true);
    solver.setMaxCutRounds(10);
    solver.load(lp);
    auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(-2.0, 1e-6));
}

TEST_CASE("MipSolver with cuts: fractional problem", "[cuts][integration]") {
    auto lp = buildFractionalMip();

    MipSolver solver;
    solver.setVerbose(false);
    solver.setCutsEnabled(true);
    solver.load(lp);
    auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(-100.0, 1e-6));
}

TEST_CASE("MipSolver: cuts disabled gives same result", "[cuts][integration]") {
    auto lp = buildFractionalMip();

    MipSolver solver_no_cuts;
    solver_no_cuts.setVerbose(false);
    solver_no_cuts.setCutsEnabled(false);
    solver_no_cuts.load(lp);
    auto result_no = solver_no_cuts.solve();

    MipSolver solver_cuts;
    solver_cuts.setVerbose(false);
    solver_cuts.setCutsEnabled(true);
    solver_cuts.load(lp);
    auto result_cuts = solver_cuts.solve();

    // Both should find optimal.
    REQUIRE(result_no.status == Status::Optimal);
    REQUIRE(result_cuts.status == Status::Optimal);
    CHECK_THAT(result_no.objective, WithinAbs(result_cuts.objective, 1e-6));
}

TEST_CASE("MipSolver: cut family toggles preserve correctness", "[cuts][integration]") {
    auto lp = buildFractionalMip();

    MipSolver solver_no_families;
    solver_no_families.setVerbose(false);
    solver_no_families.setCutsEnabled(true);
    CutFamilyConfig no_families;
    no_families.gomory = false;
    no_families.mir = false;
    no_families.cover = false;
    no_families.implied_bound = false;
    no_families.clique = false;
    no_families.zero_half = false;
    no_families.mixing = false;
    no_families.cmir = false;
    no_families.strong_cg = false;
    no_families.lifted_cover = false;
    no_families.mod_k = false;
    no_families.intersection_cut = false;
    no_families.multi_row = false;
    solver_no_families.setCutFamilyConfig(no_families);
    solver_no_families.load(lp);
    auto result_no = solver_no_families.solve();

    MipSolver solver_mir_only;
    solver_mir_only.setVerbose(false);
    solver_mir_only.setCutsEnabled(true);
    CutFamilyConfig mir_only;
    mir_only.gomory = false;
    mir_only.mir = true;
    mir_only.cover = false;
    mir_only.implied_bound = false;
    mir_only.clique = false;
    mir_only.zero_half = false;
    mir_only.mixing = false;
    mir_only.cmir = false;
    mir_only.strong_cg = false;
    mir_only.lifted_cover = false;
    mir_only.mod_k = false;
    mir_only.intersection_cut = false;
    mir_only.multi_row = false;
    solver_mir_only.setCutFamilyConfig(mir_only);
    solver_mir_only.load(lp);
    auto result_mir = solver_mir_only.solve();

    REQUIRE(result_no.status == Status::Optimal);
    REQUIRE(result_mir.status == Status::Optimal);
    CHECK_THAT(result_no.objective, WithinAbs(result_mir.objective, 1e-6));
}

TEST_CASE("MipSolver: cut effort off matches cuts-disabled behavior", "[cuts][integration]") {
    auto lp = buildFractionalMip();

    MipSolver solver_off;
    solver_off.setVerbose(false);
    solver_off.setCutsEnabled(true);
    solver_off.setCutEffortMode(CutEffortMode::Off);
    solver_off.load(lp);
    auto result_off = solver_off.solve();

    MipSolver solver_disabled;
    solver_disabled.setVerbose(false);
    solver_disabled.setCutsEnabled(false);
    solver_disabled.load(lp);
    auto result_disabled = solver_disabled.solve();

    REQUIRE(result_off.status == Status::Optimal);
    REQUIRE(result_disabled.status == Status::Optimal);
    CHECK_THAT(result_off.objective, WithinAbs(result_disabled.objective, 1e-6));
}

TEST_CASE("MipSolver with cuts: MIPLIB gt2", "[cuts][miplib]") {
    std::string path = mipx::test::testDataDir() + "/miplib/gt2.mps.gz";
    if (!std::filesystem::exists(path)) {
        SKIP("gt2.mps.gz not found (run download_miplib.sh --small)");
    }

    auto problem = readMps(path);
    REQUIRE(problem.hasIntegers());

    MipSolver solver;
    solver.setVerbose(true);
    solver.setCutsEnabled(true);
    solver.setMaxCutRounds(10);
    solver.setNodeLimit(10000);
    solver.setTimeLimit(60.0);
    solver.load(problem);
    auto result = solver.solve();

    // gt2 optimal: 21166.0
    if (result.status == Status::Optimal) {
        CHECK_THAT(result.objective, WithinAbs(21166.0, 1.0));
    }
    CHECK((result.status == Status::Optimal || result.status == Status::NodeLimit ||
           result.status == Status::TimeLimit));
}

// ---------------------------------------------------------------------------
// CMIR validity.
//
// The MIR inequality is derived over nonnegative variables, treating the
// continuous terms as a nonnegative slack on the right-hand side. Only
// negative continuous coefficients belong in the cut, scaled by 1/(1-f0); a
// positive continuous term can only be relaxed away. Reading positive terms as
// a/f0 (and negative ones with a flipped sign) produces cuts that remove
// integer-feasible points, which the branch-and-bound then reports as an
// optimal solution worse than the true optimum.
// ---------------------------------------------------------------------------

namespace {

// Mixed-integer model with continuous columns of both signs in a <= row whose
// right-hand side has a fractional part, and a known integer-feasible point.
LpProblem buildCmirValidityMip() {
    LpProblem lp;
    lp.name = "cmir_validity";
    lp.sense = Sense::Minimize;
    lp.num_cols = 7;
    lp.num_rows = 5;
    lp.obj = {2.0, -5.0, 5.0, 3.0, 0.0, -5.0, -2.0};
    lp.obj_offset = -2.0;
    lp.col_lower = {0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 3.0};
    lp.col_upper = {1.0, 1.0, 1.0, 1.0, 10.0, 6.0, 8.8};
    lp.col_type = {VarType::Binary,     VarType::Binary,     VarType::Binary,    VarType::Binary,
                   VarType::Continuous, VarType::Continuous, VarType::Continuous};
    lp.col_names = {"c0", "c1", "c2", "c3", "c4", "c5", "c6"};

    lp.row_lower = {-18.0, -kInf, 4.0, -kInf, 15.0};
    lp.row_upper = {kInf, 29.4, 4.0, 3.0, 15.0};
    lp.row_names = {"R0", "R1", "R2", "R3", "R4"};

    std::vector<Triplet> trips = {
        {0, 0, 4.0},  {0, 2, -5.0}, {0, 6, -2.0}, {1, 1, -3.4}, {1, 3, 0.4},  {1, 5, 1.0},
        {1, 6, 3.0},  {2, 1, 2.0},  {2, 2, -3.0}, {2, 3, -3.0}, {2, 5, 1.0},  {3, 0, -4.0},
        {3, 2, -2.0}, {3, 6, 1.0},  {4, 0, -4.0}, {4, 3, -5.0}, {4, 4, -2.0}, {4, 6, 5.0},
    };
    lp.matrix = SparseMatrix(5, 7, std::move(trips));
    return lp;
}

// Feasible for every row of buildCmirValidityMip, integral on the binaries,
// objective -41 (the model's optimum).
const std::vector<Real>& cmirValidityOptimum() {
    static const std::vector<Real> point = {1.0, 1.0, 0.0, 1.0, 5.5, 5.0, 7.0};
    return point;
}

CutFamilyConfig onlyCmirConfig() {
    CutFamilyConfig config;
    config.gomory = false;
    config.mir = false;
    config.cover = false;
    config.implied_bound = false;
    config.clique = false;
    config.zero_half = false;
    config.mixing = false;
    config.cmir = true;
    config.strong_cg = false;
    config.lifted_cover = false;
    config.mod_k = false;
    config.intersection_cut = false;
    config.multi_row = false;
    return config;
}

}  // namespace

TEST_CASE("SeparatorManager: CMIR cuts keep every integer-feasible point", "[cuts][cmir]") {
    auto problem = buildCmirValidityMip();
    const auto& x = cmirValidityOptimum();

    // The reference point really is feasible for the model.
    for (Index i = 0; i < problem.num_rows; ++i) {
        auto row = problem.matrix.row(i);
        Real activity = 0.0;
        for (Index k = 0; k < row.size(); ++k) {
            activity += row.values[k] * x[static_cast<std::size_t>(row.indices[k])];
        }
        INFO("row " << i);
        CHECK(activity <= problem.row_upper[i] + 1e-9);
        CHECK(activity >= problem.row_lower[i] - 1e-9);
    }

    DualSimplexSolver lp;
    lp.load(problem);
    auto result = lp.solve();
    REQUIRE(result.status == Status::Optimal);

    CutPool pool;
    SeparatorManager manager;
    manager.setConfig(onlyCmirConfig());
    manager.setMaxCutsPerFamily(10);

    CutSeparationStats stats;
    manager.separate(lp, problem, lp.getPrimalValues(), pool, stats);

    for (Index c = 0; c < pool.size(); ++c) {
        const Cut& cut = pool[c];
        Real activity = 0.0;
        for (std::size_t k = 0; k < cut.indices.size(); ++k) {
            activity += cut.values[k] * x[static_cast<std::size_t>(cut.indices[k])];
        }
        INFO("cut " << c << " activity " << activity << " in [" << cut.lower << ", " << cut.upper
                    << "]");
        CHECK(activity <= cut.upper + 1e-7);
        CHECK(activity >= cut.lower - 1e-7);
    }
}

TEST_CASE("MipSolver: CMIR cuts do not cut off the optimum", "[cuts][cmir]") {
    auto problem = buildCmirValidityMip();

    MipSolver solver;
    solver.setVerbose(false);
    solver.load(problem);
    auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(-41.0, 1e-6));
}

TEST_CASE("SeparatorManager: CMIR skips rows with negative lower bounds", "[cuts][cmir]") {
    // MIR needs every variable nonnegative after complementation. A free
    // integer column invalidates the rounding argument, so no cut may be
    // produced from a row containing one.
    LpProblem lp;
    lp.name = "cmir_negative_lower";
    lp.sense = Sense::Minimize;
    lp.num_cols = 3;
    lp.num_rows = 1;
    lp.obj = {-1.0, -1.0, -1.0};
    lp.col_lower = {-5.0, 0.0, 0.0};
    lp.col_upper = {5.0, 10.0, 10.0};
    lp.col_type = {VarType::Integer, VarType::Integer, VarType::Continuous};
    lp.col_names = {"z", "y", "s"};
    lp.row_lower = {-kInf};
    lp.row_upper = {7.5};
    lp.row_names = {"R0"};
    std::vector<Triplet> trips = {{0, 0, 1.0}, {0, 1, 2.0}, {0, 2, 1.0}};
    lp.matrix = SparseMatrix(1, 3, std::move(trips));

    DualSimplexSolver solver;
    solver.load(lp);
    auto result = solver.solve();
    REQUIRE(result.status == Status::Optimal);

    CutPool pool;
    SeparatorManager manager;
    manager.setConfig(onlyCmirConfig());
    manager.setMaxCutsPerFamily(10);

    CutSeparationStats stats;
    manager.separate(solver, lp, solver.getPrimalValues(), pool, stats);
    CHECK(pool.size() == 0);
}

// ---------------------------------------------------------------------------
// Strong CG validity.
//
// The Strong CG cut rounds the scaled right-hand side down, which is only
// valid when the scaled left-hand side is integral. Continuous columns never
// are, so they have to be relaxed away (dropped) before the rounding step --
// keeping them makes the cut stronger than the row implies and removes
// integer-feasible points, which branch-and-bound then reports as an optimum
// worse than the true one.
// ---------------------------------------------------------------------------

namespace {

// min -6a -5b -4c -s  s.t.  4a + 3b + 3c + s <= 6.5, a,b,c binary, s in [0,10].
// Optimum -9.5 at b = c = 1, s = 0.5 (weight 6 <= 6.5).
LpProblem buildStrongCgKnapsackMip() {
    LpProblem lp;
    lp.name = "strongcg_knap";
    lp.sense = Sense::Minimize;
    lp.num_cols = 4;
    lp.num_rows = 1;
    lp.obj = {-6.0, -5.0, -4.0, -1.0};
    lp.col_lower = {0.0, 0.0, 0.0, 0.0};
    lp.col_upper = {1.0, 1.0, 1.0, 10.0};
    lp.col_type = {VarType::Binary, VarType::Binary, VarType::Binary, VarType::Continuous};
    lp.col_names = {"a", "b", "c", "s"};
    lp.row_lower = {-kInf};
    lp.row_upper = {6.5};
    lp.row_names = {"R1"};
    std::vector<Triplet> trips = {{0, 0, 4.0}, {0, 1, 3.0}, {0, 2, 3.0}, {0, 3, 1.0}};
    lp.matrix = SparseMatrix(1, 4, std::move(trips));
    return lp;
}

// min -x - y  s.t.  x + y <= 2.5, x integer in [0,10], y continuous in [0,10].
// Optimum -2.5, attained anywhere on x + y = 2.5 with x integral.
LpProblem buildStrongCgMixedMip() {
    LpProblem lp;
    lp.name = "strongcg_mixed";
    lp.sense = Sense::Minimize;
    lp.num_cols = 2;
    lp.num_rows = 1;
    lp.obj = {-1.0, -1.0};
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {10.0, 10.0};
    lp.col_type = {VarType::Integer, VarType::Continuous};
    lp.col_names = {"x", "y"};
    lp.row_lower = {-kInf};
    lp.row_upper = {2.5};
    lp.row_names = {"R1"};
    std::vector<Triplet> trips = {{0, 0, 1.0}, {0, 1, 1.0}};
    lp.matrix = SparseMatrix(1, 2, std::move(trips));
    return lp;
}

// The integer optimum of every `buildNearIntegerMip` model. See below for why
// this particular magnitude.
constexpr Real kNearIntegerOptimum = 100001.0;

// The reproduction from #195, scaled for a family whose separator multiplies
// the row by 1/`divisor`:
//   R1: divisor * (1 - 1e-9) x <= divisor * 100000.99995, x integer in [0, 200000]
//   R2:                 c2 * z <= b2,                     z integer in [0, 1]
// The scaled R1 coefficient is exactly 0.999999999 for every divisor used here,
// so a snapping tolerance floors it to 1 while the scaled right-hand side still
// floors to 100000 -- the invalid cut x <= 100000. x = 100001 is
// integer-feasible and optimal (objective -100001); x = 100002 is not. R2
// carries no near-integer trickery: it exists so that each family still emits a
// *valid* violated cut after the fix, which keeps the per-cut validity check
// non-vacuous.
//
// The magnitude matters. The row is scaled so the LP relaxation sits at
// 100001.00005, a fractionality of 5e-5 -- fifty times MipSolver's integrality
// tolerance. At the issue's original 1001 the LP sits at 1001.0000009, whose
// fractionality is *below* that tolerance, so the solver accepts the root LP
// point as integral and never branches: the objective assertion then passes on
// -1001.0000009 with a tenth of its tolerance to spare, and never proves that x
// came back integral at all.
LpProblem buildNearIntegerMip(const char* name, Real divisor, Real c2, Real b2) {
    LpProblem lp;
    lp.name = name;
    lp.sense = Sense::Minimize;
    lp.num_cols = 2;
    lp.num_rows = 2;
    lp.obj = {-1.0, -0.001};
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {2.0 * kNearIntegerOptimum, 1.0};
    lp.col_type = {VarType::Integer, VarType::Integer};
    lp.col_names = {"x", "z"};
    lp.row_lower = {-kInf, -kInf};
    lp.row_upper = {divisor * 100000.99995, b2};
    lp.row_names = {"R1", "R2"};
    std::vector<Triplet> trips = {{0, 0, divisor * (1.0 - 1e-9)}, {1, 1, c2}};
    lp.matrix = SparseMatrix(2, 2, std::move(trips));
    return lp;
}

// Strong CG's #188 model, in the same shape as the four #195 families: divisor
// 1.0, with R2 `2z <= 1` yielding the valid cut z <= 0 at t = 0.5.
//
// It used to carry optimum 1001, and that magnitude made the solve test weaker
// than it looked: the LP relaxation sits at 1001.0000009, a fractionality below
// MipSolver's integrality tolerance, so the solver accepted the root LP point
// and never branched. An objective-only assertion could not then tell a correct
// 1001 from a regressed 1000.9999995 -- both land within 1e-6 of -1001. Sharing
// the 100001 scale puts the fractionality at 5e-5 and lets the solve pin the
// solution vector, matching what the four #195 families assert.
LpProblem buildStrongCgNearIntegerMip() {
    return buildNearIntegerMip("strongcg_near_integer", 1.0, 2.0, 1.0);
}

// Three binaries under a single row `coeff * (x1 + x2 + x3) <= rhs`, minimizing
// -(x1 + x2 + x3). With `coeff` an exact multiple of the family's divisor no
// coefficient rounds at all, so the whole strength of the cut comes from
// flooring the scaled right-hand side -- the canonical Chvatal-Gomory case. The
// integer optimum is any single 1 (objective -1) whenever rhs < 2 * coeff.
LpProblem buildDivisibleRowMip(const char* name, Real coeff, Real rhs) {
    LpProblem lp;
    lp.name = name;
    lp.sense = Sense::Minimize;
    lp.num_cols = 3;
    lp.num_rows = 1;
    lp.obj = {-1.0, -1.0, -1.0};
    lp.col_lower = {0.0, 0.0, 0.0};
    lp.col_upper = {1.0, 1.0, 1.0};
    lp.col_type = {VarType::Binary, VarType::Binary, VarType::Binary};
    lp.col_names = {"x1", "x2", "x3"};
    lp.row_lower = {-kInf};
    lp.row_upper = {rhs};
    lp.row_names = {"R1"};
    std::vector<Triplet> trips = {{0, 0, coeff}, {0, 1, coeff}, {0, 2, coeff}};
    lp.matrix = SparseMatrix(1, 3, std::move(trips));
    return lp;
}

// Branch-and-cut `problem` with presolve off and only `config`'s family armed,
// so a cut that removes the optimum shows up as a wrong objective rather than
// being masked by another family or by presolve.
MipResult solveWithCutFamilies(const LpProblem& problem, const CutFamilyConfig& config,
                               bool strengthen = false) {
    MipSolver solver;
    solver.setVerbose(false);
    solver.setPresolve(false);
    solver.setCutFamilyConfig(config);
    solver.setCutStrengthening(strengthen);
    solver.load(problem);
    return solver.solve();
}

CutFamilyConfig noCutFamiliesConfig() {
    CutFamilyConfig config;
    config.gomory = false;
    config.mir = false;
    config.cover = false;
    config.implied_bound = false;
    config.clique = false;
    config.zero_half = false;
    config.mixing = false;
    config.cmir = false;
    config.strong_cg = false;
    config.lifted_cover = false;
    config.mod_k = false;
    config.intersection_cut = false;
    config.multi_row = false;
    return config;
}

CutFamilyConfig onlyStrongCgConfig() {
    CutFamilyConfig config = noCutFamiliesConfig();
    config.strong_cg = true;
    return config;
}

CutFamilyConfig onlyMirConfig() {
    CutFamilyConfig config = noCutFamiliesConfig();
    config.mir = true;
    return config;
}

CutFamilyConfig onlyZeroHalfConfig() {
    CutFamilyConfig config = noCutFamiliesConfig();
    config.zero_half = true;
    return config;
}

CutFamilyConfig onlyMixingConfig() {
    CutFamilyConfig config = noCutFamiliesConfig();
    config.mixing = true;
    return config;
}

CutFamilyConfig onlyModKConfig() {
    CutFamilyConfig config = noCutFamiliesConfig();
    config.mod_k = true;
    return config;
}

// Solve a `buildNearIntegerMip` model with one cut family armed and check it
// lands on the true optimum. The objective alone is a weak assertion -- a point
// merely *near* x = 100001 would satisfy it -- so pin the solution vector too,
// which is what actually proves no cut removed the integer optimum.
void checkNearIntegerOptimum(const LpProblem& problem, const CutFamilyConfig& config,
                             bool strengthen = false) {
    const auto result = solveWithCutFamilies(problem, config, strengthen);
    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(-kNearIntegerOptimum, 1e-6));
    REQUIRE(result.solution.size() == 2);
    CHECK_THAT(result.solution[0], WithinAbs(kNearIntegerOptimum, 1e-9));
    CHECK_THAT(result.solution[1], WithinAbs(0.0, 1e-9));
}

// Separate the LP relaxation of `problem` with `config` and check that every
// cut in the pool is satisfied by the integer-feasible point `x`. Returns the
// number of cuts that were checked.
Index separateAndCheckCutsKeep(const LpProblem& problem, const std::vector<Real>& x,
                               const CutFamilyConfig& config, bool strengthen = false) {
    // The reference point really is feasible and integral for the model.
    for (Index j = 0; j < problem.num_cols; ++j) {
        const Real v = x[static_cast<std::size_t>(j)];
        INFO("column " << j);
        CHECK(v >= problem.col_lower[j] - 1e-9);
        CHECK(v <= problem.col_upper[j] + 1e-9);
        if (problem.col_type[j] != VarType::Continuous) {
            CHECK(std::abs(v - std::round(v)) <= 1e-9);
        }
    }
    for (Index i = 0; i < problem.num_rows; ++i) {
        auto row = problem.matrix.row(i);
        Real activity = 0.0;
        for (Index k = 0; k < row.size(); ++k) {
            activity += row.values[k] * x[static_cast<std::size_t>(row.indices[k])];
        }
        INFO("row " << i);
        CHECK(activity <= problem.row_upper[i] + 1e-9);
        CHECK(activity >= problem.row_lower[i] - 1e-9);
    }

    DualSimplexSolver lp;
    lp.load(problem);
    auto result = lp.solve();
    REQUIRE(result.status == Status::Optimal);

    CutPool pool;
    SeparatorManager manager;
    manager.setConfig(config);
    manager.setMaxCutsPerFamily(10);
    manager.setStrengthenCuts(strengthen);

    CutSeparationStats stats;
    manager.separate(lp, problem, lp.getPrimalValues(), pool, stats);

    for (Index c = 0; c < pool.size(); ++c) {
        const Cut& cut = pool[c];
        Real activity = 0.0;
        for (std::size_t k = 0; k < cut.indices.size(); ++k) {
            activity += cut.values[k] * x[static_cast<std::size_t>(cut.indices[k])];
        }
        INFO("cut " << c << " activity " << activity << " in [" << cut.lower << ", " << cut.upper
                    << "]");
        CHECK(activity <= cut.upper + 1e-7);
        CHECK(activity >= cut.lower - 1e-7);
    }
    return pool.size();
}

}  // namespace

TEST_CASE("SeparatorManager: Strong CG cuts keep the knapsack optimum", "[cuts][strongcg]") {
    const auto problem = buildStrongCgKnapsackMip();
    const std::vector<Real> optimum = {0.0, 1.0, 1.0, 0.5};  // objective -9.5

    // The fractional right-hand side makes the row separable, so the check is
    // not passing vacuously on an empty pool.
    CHECK(separateAndCheckCutsKeep(problem, optimum, onlyStrongCgConfig()) > 0);
}

TEST_CASE("SeparatorManager: Strong CG cuts keep the mixed-row optimum", "[cuts][strongcg]") {
    const auto problem = buildStrongCgMixedMip();
    // x = 0, y = 2.5 is integer-feasible and optimal (objective -2.5). Keeping
    // the continuous column produced 0.75y <= 1 from the multiplier t = 0.75,
    // which this point violates by 0.875. Once the column is relaxed away the
    // only integer term left is x, and no multiplier rounds it into a violated
    // cut -- so the pool must be empty rather than hold a valid-but-different
    // cut. If a future change does emit one here, the per-cut checks in the
    // helper must still pass.
    const std::vector<Real> optimum = {0.0, 2.5};

    CHECK(separateAndCheckCutsKeep(problem, optimum, onlyStrongCgConfig()) == 0);
}

TEST_CASE("MipSolver: Strong CG cuts do not cut off the optimum", "[cuts][strongcg]") {
    SECTION("knapsack with a continuous column") {
        MipSolver solver;
        solver.setVerbose(false);
        solver.setPresolve(false);
        solver.setCutFamilyConfig(onlyStrongCgConfig());
        solver.load(buildStrongCgKnapsackMip());
        auto result = solver.solve();

        REQUIRE(result.status == Status::Optimal);
        CHECK_THAT(result.objective, WithinAbs(-9.5, 1e-6));
    }

    SECTION("mixed integer/continuous row") {
        MipSolver solver;
        solver.setVerbose(false);
        solver.setPresolve(false);
        solver.setCutFamilyConfig(onlyStrongCgConfig());
        solver.load(buildStrongCgMixedMip());
        auto result = solver.solve();

        REQUIRE(result.status == Status::Optimal);
        CHECK_THAT(result.objective, WithinAbs(-2.5, 1e-6));
    }
}

TEST_CASE("SeparatorManager: Strong CG does not round a coefficient up", "[cuts][strongcg]") {
    const auto problem = buildStrongCgNearIntegerMip();
    // x = 100001, z = 0 is integer-feasible; the pre-fix cut x <= 100000
    // removes it.
    const std::vector<Real> optimum = {kNearIntegerOptimum, 0.0};

    // Require a non-empty pool so the per-cut validity loop is genuinely
    // exercised: R2 (2z <= 1) still yields the valid cut z <= 0 at t = 0.5.
    // Without this the check would pass on an empty pool if the separator
    // stopped emitting anything at all.
    CHECK(separateAndCheckCutsKeep(problem, optimum, onlyStrongCgConfig()) > 0);
}

TEST_CASE("MipSolver: Strong CG keeps an optimum under a near-integer coefficient",
          "[cuts][strongcg]") {
    checkNearIntegerOptimum(buildStrongCgNearIntegerMip(), onlyStrongCgConfig());
}

// ---------------------------------------------------------------------------
// #195: the other four families that floored a scaled coefficient through a
// snapping tolerance. Each test isolates one family, checks every separated cut
// against the known integer optimum, pins the pool non-empty so that check is
// not vacuous, and re-solves the model end to end.
// ---------------------------------------------------------------------------

TEST_CASE("SeparatorManager: MIR does not round a coefficient up", "[cuts][mir]") {
    // MIR scales the row by 1, so R1 carries the bare 1 - 1e-9 coefficient.
    // R2 (z <= 0.5) floors to z <= 0, which the LP point z = 0.5 violates, so
    // the pool stays non-empty once R1 no longer yields anything.
    const auto problem = buildNearIntegerMip("mir_near_integer", 1.0, 1.0, 0.5);
    const std::vector<Real> optimum = {kNearIntegerOptimum, 0.0};

    SECTION("separated cuts keep the integer optimum") {
        CHECK(separateAndCheckCutsKeep(problem, optimum, onlyMirConfig()) > 0);
    }

    SECTION("solver reports the true optimum") {
        checkNearIntegerOptimum(problem, onlyMirConfig());
    }
}

TEST_CASE("SeparatorManager: zero-half does not round a coefficient up", "[cuts][zerohalf]") {
    // Zero-half scales the row by 1/2, so R1 needs 2 * (1 - 1e-9). R2 stays
    // separable after the fix: 3z <= 1 floors to z <= 0, violated at z = 1/3.
    const auto problem = buildNearIntegerMip("zerohalf_near_integer", 2.0, 3.0, 1.0);
    const std::vector<Real> optimum = {kNearIntegerOptimum, 0.0};

    SECTION("separated cuts keep the integer optimum") {
        CHECK(separateAndCheckCutsKeep(problem, optimum, onlyZeroHalfConfig()) > 0);
    }

    SECTION("solver reports the true optimum") {
        checkNearIntegerOptimum(problem, onlyZeroHalfConfig());
    }
}

TEST_CASE("SeparatorManager: mixing does not round a coefficient up", "[cuts][mixing]") {
    // Mixing scales the row by 1/3, so R1 needs 3 * (1 - 1e-9). R2 stays
    // separable after the fix: 4z <= 1 floors to z <= 0, violated at z = 1/4.
    const auto problem = buildNearIntegerMip("mixing_near_integer", 3.0, 4.0, 1.0);
    const std::vector<Real> optimum = {kNearIntegerOptimum, 0.0};

    SECTION("separated cuts keep the integer optimum") {
        CHECK(separateAndCheckCutsKeep(problem, optimum, onlyMixingConfig()) > 0);
    }

    SECTION("solver reports the true optimum") {
        checkNearIntegerOptimum(problem, onlyMixingConfig());
    }
}

TEST_CASE("SeparatorManager: mod-k does not round a coefficient up", "[cuts][modk]") {
    // Mod-k tries k in {2, 3, 4, 5}; k = 2 is the one that trips on
    // 2 * (1 - 1e-9). R2 stays separable after the fix: 3z <= 1 floors to
    // z <= 0 at k = 2 and k = 3, violated at z = 1/3.
    const auto problem = buildNearIntegerMip("modk_near_integer", 2.0, 3.0, 1.0);
    const std::vector<Real> optimum = {kNearIntegerOptimum, 0.0};

    SECTION("separated cuts keep the integer optimum") {
        CHECK(separateAndCheckCutsKeep(problem, optimum, onlyModKConfig()) > 0);
    }

    SECTION("solver reports the true optimum") {
        checkNearIntegerOptimum(problem, onlyModKConfig());
    }
}

// Rescaling the "did rounding change anything" test in #195 made zero-half and
// mixing stricter, and neither seeds the flag from the right-hand side the way
// MIR and mod-k do. These two pin the case that exposes: a row where no
// coefficient rounds and the whole cut comes from flooring the scaled rhs.

TEST_CASE("SeparatorManager: zero-half still cuts a row only the rhs rounds", "[cuts][zerohalf]") {
    // 2x1 + 2x2 + 2x3 <= 3 over binaries. No coefficient rounds (0.5 * 2 = 1),
    // but the rhs floors from 1.5 to 1, giving x1 + x2 + x3 <= 1 -- the
    // canonical {0,1/2} cut, violated by the LP optimum whose sum is 1.5.
    const auto problem = buildDivisibleRowMip("zerohalf_even_row", 2.0, 3.0);
    const std::vector<Real> optimum = {1.0, 0.0, 0.0};

    SECTION("the cut is still generated and is valid") {
        CHECK(separateAndCheckCutsKeep(problem, optimum, onlyZeroHalfConfig()) > 0);
    }

    SECTION("solver reports the true optimum") {
        const auto result = solveWithCutFamilies(problem, onlyZeroHalfConfig());
        REQUIRE(result.status == Status::Optimal);
        CHECK_THAT(result.objective, WithinAbs(-1.0, 1e-6));
    }
}

TEST_CASE("SeparatorManager: mixing still cuts a row only the rhs rounds", "[cuts][mixing]") {
    // 3x1 + 3x2 + 3x3 <= 4 over binaries. No coefficient rounds (3/3 = 1), but
    // the rhs floors from 4/3 to 1, giving x1 + x2 + x3 <= 1, violated by the
    // LP optimum whose sum is 4/3.
    const auto problem = buildDivisibleRowMip("mixing_multiple_of_three_row", 3.0, 4.0);
    const std::vector<Real> optimum = {1.0, 0.0, 0.0};

    SECTION("the cut is still generated and is valid") {
        CHECK(separateAndCheckCutsKeep(problem, optimum, onlyMixingConfig()) > 0);
    }

    SECTION("solver reports the true optimum") {
        const auto result = solveWithCutFamilies(problem, onlyMixingConfig());
        REQUIRE(result.status == Status::Optimal);
        CHECK_THAT(result.objective, WithinAbs(-1.0, 1e-6));
    }
}

TEST_CASE("MipSolver: knapsack solves to -9.5 with presolve off and cuts on", "[cuts][strongcg]") {
    // The reproduction from the bug report: presolve off, every cut family on.
    MipSolver solver;
    solver.setVerbose(false);
    solver.setPresolve(false);
    solver.setCutsEnabled(true);
    solver.load(buildStrongCgKnapsackMip());
    auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(-9.5, 1e-6));
}

// ---------------------------------------------------------------------------
// Cut family configuration gating (issue #189)
// ---------------------------------------------------------------------------

namespace {

// Every CutFamily value that names a real separator, i.e. the enum range with
// the Unknown sentinel and the Count terminator stripped off. Iterating the
// enum rather than a hand-written list keeps these tests honest when a new
// family is added.
std::vector<CutFamily> allSeparableFamilies() {
    std::vector<CutFamily> families;
    for (Int fi = static_cast<Int>(CutFamily::Unknown) + 1; fi < static_cast<Int>(CutFamily::Count);
         ++fi) {
        families.push_back(static_cast<CutFamily>(fi));
    }
    return families;
}

// Config with every flag set to `enabled`. Built through setCutFamilyEnabled so
// the switch in MipSolver has to handle every family for this to work.
CutFamilyConfig allCutFamilies(bool enabled) {
    MipSolver probe;
    for (CutFamily family : allSeparableFamilies()) {
        probe.setCutFamilyEnabled(family, enabled);
    }
    return probe.getCutFamilyConfig();
}

}  // namespace

TEST_CASE("MipSolver: setCutFamilyEnabled reaches every cut family", "[cuts][families]") {
    // A family setCutFamilyEnabled forgets keeps the CutFamilyConfig default of
    // true, which is exactly the bug: it then separates unconditionally.
    const CutFamilyConfig none = allCutFamilies(false);
    CHECK_FALSE(none.gomory);
    CHECK_FALSE(none.mir);
    CHECK_FALSE(none.cover);
    CHECK_FALSE(none.implied_bound);
    CHECK_FALSE(none.clique);
    CHECK_FALSE(none.zero_half);
    CHECK_FALSE(none.mixing);
    CHECK_FALSE(none.cmir);
    CHECK_FALSE(none.strong_cg);
    CHECK_FALSE(none.lifted_cover);
    CHECK_FALSE(none.mod_k);
    CHECK_FALSE(none.intersection_cut);
    CHECK_FALSE(none.multi_row);

    const CutFamilyConfig all = allCutFamilies(true);
    CHECK(all.gomory);
    CHECK(all.mir);
    CHECK(all.cover);
    CHECK(all.implied_bound);
    CHECK(all.clique);
    CHECK(all.zero_half);
    CHECK(all.mixing);
    CHECK(all.cmir);
    CHECK(all.strong_cg);
    CHECK(all.lifted_cover);
    CHECK(all.mod_k);
    CHECK(all.intersection_cut);
    CHECK(all.multi_row);
}

TEST_CASE("MipSolver: every cut family off adds no root cuts", "[cuts][families][integration]") {
    const auto lp = buildStrongCgKnapsackMip();

    // Baseline with every family on. Cuts must actually be separated here,
    // otherwise the all-off run below would pass for the wrong reason.
    MipSolver baseline;
    baseline.setVerbose(false);
    baseline.setPresolve(false);
    baseline.setCutsEnabled(true);
    baseline.setMaxCutRounds(10);
    baseline.setCutFamilyConfig(allCutFamilies(true));
    baseline.load(lp);
    const auto base_result = baseline.solve();
    REQUIRE(base_result.status == Status::Optimal);
    REQUIRE(baseline.getCutStats().root_cuts_added > 0);

    // Same model, same cut settings, only the family flags differ: cuts stay
    // enabled and the effort mode is untouched, so a zero count can only come
    // from the per-family gating.
    MipSolver all_off;
    all_off.setVerbose(false);
    all_off.setPresolve(false);
    all_off.setCutsEnabled(true);
    all_off.setMaxCutRounds(10);
    all_off.setCutFamilyConfig(allCutFamilies(false));
    all_off.load(lp);
    const auto off_result = all_off.solve();

    REQUIRE(off_result.status == Status::Optimal);
    CHECK(all_off.getCutEffortMode() == baseline.getCutEffortMode());
    CHECK(all_off.getCutStats().root_cuts_added == 0);
    CHECK_THAT(off_result.objective, WithinAbs(base_result.objective, 1e-6));

    // No family was even attempted, while the baseline attempted at least one.
    Int baseline_attempted = 0;
    for (CutFamily family : allSeparableFamilies()) {
        INFO("family " << cutFamilyName(family));
        CHECK(all_off.getRootCutFamilyStats().at(family).attempted == 0);
        CHECK(all_off.getRootCutFamilyStats().at(family).accepted == 0);
        baseline_attempted += baseline.getRootCutFamilyStats().at(family).attempted;
    }
    CHECK(baseline_attempted > 0);
}

TEST_CASE("MipSolver: one cut family at a time silences all the others",
          "[cuts][families][integration]") {
    const auto lp = buildStrongCgKnapsackMip();

    Int solo_accepted_total = 0;
    for (CutFamily solo : allSeparableFamilies()) {
        INFO("solo family " << cutFamilyName(solo));

        MipSolver solver;
        solver.setVerbose(false);
        solver.setPresolve(false);
        solver.setCutsEnabled(true);
        solver.setMaxCutRounds(10);
        solver.setCutFamilyConfig(allCutFamilies(false));
        solver.setCutFamilyEnabled(solo, true);
        solver.load(lp);
        const auto result = solver.solve();

        REQUIRE(result.status == Status::Optimal);
        CHECK_THAT(result.objective, WithinAbs(-9.5, 1e-6));

        const auto& stats = solver.getRootCutFamilyStats();
        for (CutFamily other : allSeparableFamilies()) {
            if (other == solo) {
                continue;
            }
            INFO("silenced family " << cutFamilyName(other));
            CHECK(stats.at(other).attempted == 0);
            CHECK(stats.at(other).generated == 0);
            CHECK(stats.at(other).accepted == 0);
        }
        solo_accepted_total += stats.at(solo).accepted;
    }

    // At least one solo run has to separate something; otherwise every check
    // above would hold vacuously.
    CHECK(solo_accepted_total > 0);
}

TEST_CASE("Gomory: a dropped term is paid over the box the cut is valid in", "[cuts][gomory]") {
    // min -x + 1e-10 z  s.t.  x - 1e-11 z <= 2.2,  x in [0, 10] integer,
    // z in [0, 1e9] continuous. z sits at its lower bound with t_z < 0, so
    // its own GMI term (1e-11 / (1 - f0)) and the one the row logical spreads
    // onto it (1e-11 / f0) add up instead of cancelling: a positive x-space
    // coefficient of ~6e-11, below kCoeffTol, which finalizeCut drops and pays
    // for at z's upper bound.
    LpProblem problem;
    problem.sense = Sense::Minimize;
    problem.num_cols = 2;
    problem.obj = {-1.0, 1e-10};
    problem.col_lower = {0.0, 0.0};
    problem.col_upper = {10.0, 1e9};
    problem.col_type = {VarType::Integer, VarType::Continuous};
    problem.num_rows = 1;
    problem.row_lower = {-kInf};
    problem.row_upper = {2.2};
    problem.matrix = SparseMatrix(1, 2, std::vector<Triplet>{{0, 0, 1.0}, {0, 1, -1e-11}});

    auto separateWithZ = [&](Real z_lower, Real z_upper, bool expect_local) {
        DualSimplexSolver lp;
        lp.load(problem);
        lp.setColBounds(1, z_lower, z_upper);
        REQUIRE(lp.solve().status == Status::Optimal);
        auto primals = lp.getPrimalValues();
        REQUIRE(primals[1] == z_lower);

        CutPool pool;
        GomorySeparator gomory;
        REQUIRE(gomory.separate(lp, problem, primals, pool) == 1);
        CHECK(gomory.stats().relaxed_terms == 1);
        Cut cut = pool[0];
        CHECK(cut.local == expect_local);
        REQUIRE(cut.indices == std::vector<Index>{0});  // z dropped
        return cut;
    };

    // Global: z sits at its model bound, so a node-tightened upper bound does
    // not make the cut local -- and must not change what the drop is paid
    // over. Paying over the node's 1e6 would under-pay for every node where
    // z can reach 1e9.
    const Cut root = separateWithZ(0.0, 1e9, false);
    const Cut node = separateWithZ(0.0, 1e6, false);
    CHECK(node.values[0] == root.values[0]);
    CHECK(node.lower == root.lower);

    // Local: z's lower bound is tightened to 1, so the cut is local, and the
    // drop is paid over the node's range. A tighter node upper bound pays
    // less, giving a strictly stronger cut.
    const Cut wide = separateWithZ(1.0, 1e9, true);
    const Cut narrow = separateWithZ(1.0, 1e6, true);
    CHECK(narrow.values[0] == wide.values[0]);
    CHECK(narrow.lower > wide.lower);

    // And every point of the global cut's box survives it.
    for (Real x = 0.0; x <= 2.0; x += 1.0) {
        for (Real z : {0.0, 1e6, 1e9}) {
            REQUIRE(x - 1e-11 * z <= 2.2);
            CHECK(root.values[0] * x >= root.lower - 1e-9);
        }
    }
}

// ---------------------------------------------------------------------------
// Intersection and multi-row cuts (issue #219)
// ---------------------------------------------------------------------------

namespace {

/// Separate one tableau family through the manager. A negative global row
/// count leaves the manager's default, which vouches for the model's rows.
CutPool separateTableauFamily(CutFamily family, DualSimplexSolver& lp, const LpProblem& problem,
                              Index global_rows = -1, std::vector<Index> local_rows = {}) {
    CutFamilyConfig config = allCutFamilies(false);
    if (family == CutFamily::IntersectionCut) {
        config.intersection_cut = true;
    } else {
        REQUIRE(family == CutFamily::MultiRow);
        config.multi_row = true;
    }
    SeparatorManager manager;
    manager.setConfig(config);
    if (global_rows >= 0) {
        manager.setGlobalRowCount(global_rows);
    }
    manager.setLocalRows(std::move(local_rows));

    const auto primals = lp.getPrimalValues();
    CutPool pool;
    CutSeparationStats stats;
    const Int added = manager.separate(lp, problem, primals, pool, stats);
    CHECK(added == pool.size());
    for (Index ci = 0; ci < pool.size(); ++ci) {
        CHECK(pool[ci].family == family);
        CHECK(isNumericallySafeCut(pool[ci]));
    }
    return pool;
}

Real cutActivity(const Cut& cut, std::span<const Real> point) {
    Real lhs = 0.0;
    for (std::size_t k = 0; k < cut.indices.size(); ++k) {
        lhs += cut.values[k] * point[cut.indices[k]];
    }
    return lhs;
}

}  // namespace

TEST_CASE("CutFamilyConfig defaults leave intersection and multi-row cuts off",
          "[cuts][intersection]") {
    // Before #219 neither family ever built a cut, so on by default they were
    // inert; off keeps the default search what it was until a benchmark says
    // otherwise.
    const CutFamilyConfig config{};
    CHECK_FALSE(config.intersection_cut);
    CHECK_FALSE(config.multi_row);
}

TEST_CASE("Intersection: root cuts off global rows are global and keep every feasible point",
          "[cuts][intersection]") {
    // A basic structural's tableau row always carries a nonbasic logical, so a
    // separator that cannot substitute logicals builds nothing at all.
    for (const auto& problem : {buildFractionalMip(), buildMixedSenseMip()}) {
        INFO(problem.name);
        DualSimplexSolver lp;
        lp.load(problem);
        REQUIRE(lp.solve().status == Status::Optimal);

        const CutPool pool = separateTableauFamily(CutFamily::IntersectionCut, lp, problem);
        REQUIRE(pool.size() >= 1);
        for (Index ci = 0; ci < pool.size(); ++ci) {
            CHECK_FALSE(pool[ci].local);
        }
        checkCutsKeepAllFeasiblePoints(problem, pool, 6);
    }
}

TEST_CASE("Intersection: a cut off a tightened bound is marked local", "[cuts][intersection]") {
    // Branch y <= 2 on max 10x + 20y s.t. x + y <= 5.5: the node LP sits at
    // x = 3.5, y = 2, and the split on x gives x + 2y <= 7, measured from y's
    // node bound. That cut is valid in the subtree and cuts off x = 0, y = 5,
    // the global optimum, outside it. Promoted to a global row it would remove
    // that point from the whole search.
    auto problem = buildFractionalMip();

    DualSimplexSolver lp;
    lp.load(problem);
    REQUIRE(lp.solve().status == Status::Optimal);
    lp.setColBounds(1, 0.0, 2.0);
    REQUIRE(lp.solve().status == Status::Optimal);
    const auto primals = lp.getPrimalValues();
    REQUIRE(std::abs(primals[0] - 3.5) < 1e-9);
    REQUIRE(std::abs(primals[1] - 2.0) < 1e-9);

    const CutPool pool = separateTableauFamily(CutFamily::IntersectionCut, lp, problem);
    REQUIRE(pool.size() >= 1);
    bool cuts_off_global_optimum = false;
    const std::vector<Real> global_opt = {0.0, 5.0};
    for (Index ci = 0; ci < pool.size(); ++ci) {
        CHECK(pool[ci].local);
        cuts_off_global_optimum |= cutActivity(pool[ci], global_opt) < pool[ci].lower - 1e-6;
    }
    CHECK(cuts_off_global_optimum);

    // Valid where it was derived: the subtree's feasible points all survive.
    LpProblem node = problem;
    node.col_upper[1] = 2.0;
    checkCutsKeepAllFeasiblePoints(node, pool, 6);
}

TEST_CASE("Intersection: a cut off a node-local row is marked local", "[cuts][intersection]") {
    // The logical of a row the caller has not vouched for is node-local, so a
    // cut that substitutes it is too; vouched for, the same cut is global.
    auto problem = buildMixedSenseMip();

    DualSimplexSolver lp;
    lp.load(problem);
    REQUIRE(lp.solve().status == Status::Optimal);

    // A redundant but globally valid row: 4x + 5y + 3z <= 20.
    const std::vector<Index> starts = {0};
    const std::vector<Index> indices = {0, 1, 2};
    const std::vector<Real> values = {4.0, 5.0, 3.0};
    const std::vector<Real> lower = {-kInf};
    const std::vector<Real> upper = {20.0};
    lp.addRows(starts, indices, values, lower, upper);
    REQUIRE(lp.solve().status == Status::Optimal);

    const CutPool vouched =
        separateTableauFamily(CutFamily::IntersectionCut, lp, problem, lp.numRows());
    REQUIRE(vouched.size() >= 1);
    for (Index ci = 0; ci < vouched.size(); ++ci) {
        CHECK_FALSE(vouched[ci].local);
    }
    checkCutsKeepAllFeasiblePoints(problem, vouched, 3);

    // Every cut leans on some nonbasic logical, so naming every row local
    // makes every cut so.
    std::vector<Index> all_rows(static_cast<std::size_t>(lp.numRows()));
    for (Index r = 0; r < lp.numRows(); ++r) {
        all_rows[static_cast<std::size_t>(r)] = r;
    }
    const CutPool listed =
        separateTableauFamily(CutFamily::IntersectionCut, lp, problem, lp.numRows(), all_rows);
    REQUIRE(listed.size() == vouched.size());
    for (Index ci = 0; ci < listed.size(); ++ci) {
        CHECK(listed[ci].local);
    }
}

namespace {

// Two copies of buildFractionalMip: min -10x1 - 20y1 - 10x2 - 20y2 with
// x1 + y1 <= 5.5 and x2 + y2 <= 5.5, all integer and nonnegative. The root
// LP has two fractional basic integers, y1 = y2 = 5.5, which a multi-row cut
// needs.
LpProblem buildTwinFractionalMip() {
    LpProblem lp;
    lp.name = "twin_fractional_mip";
    lp.sense = Sense::Minimize;
    lp.num_cols = 4;
    lp.obj = {-10.0, -20.0, -10.0, -20.0};
    lp.col_lower = {0.0, 0.0, 0.0, 0.0};
    lp.col_upper = {kInf, kInf, kInf, kInf};
    lp.col_type = {VarType::Integer, VarType::Integer, VarType::Integer, VarType::Integer};
    lp.col_names = {"x1", "y1", "x2", "y2"};

    lp.num_rows = 2;
    lp.row_lower = {-kInf, -kInf};
    lp.row_upper = {5.5, 5.5};
    lp.row_names = {"cap1", "cap2"};

    std::vector<Triplet> trips = {{0, 0, 1.0}, {0, 1, 1.0}, {1, 2, 1.0}, {1, 3, 1.0}};
    lp.matrix = SparseMatrix(2, 4, std::move(trips));
    return lp;
}

}  // namespace

TEST_CASE("MultiRow: root cuts off global rows are global and keep every feasible point",
          "[cuts][intersection]") {
    auto problem = buildTwinFractionalMip();

    DualSimplexSolver lp;
    lp.load(problem);
    REQUIRE(lp.solve().status == Status::Optimal);

    const CutPool pool = separateTableauFamily(CutFamily::MultiRow, lp, problem);
    REQUIRE(pool.size() >= 1);
    for (Index ci = 0; ci < pool.size(); ++ci) {
        CHECK_FALSE(pool[ci].local);
    }
    checkCutsKeepAllFeasiblePoints(problem, pool, 6);
}

TEST_CASE("MultiRow: a cut off a tightened bound or a node-local row is marked local",
          "[cuts][intersection]") {
    auto problem = buildTwinFractionalMip();

    DualSimplexSolver lp;
    lp.load(problem);
    REQUIRE(lp.solve().status == Status::Optimal);

    SECTION("tightened bound") {
        // Branch y1 <= 2: x1 = 3.5 and y2 = 5.5 are the fractional pair, and
        // the cut leans on y1's node bound.
        lp.setColBounds(1, 0.0, 2.0);
        REQUIRE(lp.solve().status == Status::Optimal);

        const CutPool pool = separateTableauFamily(CutFamily::MultiRow, lp, problem);
        REQUIRE(pool.size() >= 1);
        bool cuts_off_global_point = false;
        // Feasible in the model, outside the y1 <= 2 subtree.
        const std::vector<Real> outside = {0.0, 5.0, 0.0, 5.0};
        for (Index ci = 0; ci < pool.size(); ++ci) {
            CHECK(pool[ci].local);
            cuts_off_global_point |= cutActivity(pool[ci], outside) < pool[ci].lower - 1e-6;
        }
        CHECK(cuts_off_global_point);

        LpProblem node = problem;
        node.col_upper[1] = 2.0;
        checkCutsKeepAllFeasiblePoints(node, pool, 6);
    }

    SECTION("node-local row") {
        const CutPool pool =
            separateTableauFamily(CutFamily::MultiRow, lp, problem, lp.numRows(), {0, 1});
        REQUIRE(pool.size() >= 1);
        for (Index ci = 0; ci < pool.size(); ++ci) {
            CHECK(pool[ci].local);
        }
    }
}

// ---------------------------------------------------------------------------
// #201: coefficient strengthening of separated cuts
// ---------------------------------------------------------------------------

namespace {

Real cutLhs(const Cut& cut, std::span<const Real> x) {
    Real lhs = 0.0;
    for (std::size_t k = 0; k < cut.indices.size(); ++k) {
        lhs += cut.values[k] * x[static_cast<std::size_t>(cut.indices[k])];
    }
    return lhs;
}

bool cutHolds(const Cut& cut, std::span<const Real> x, Real tol) {
    const Real lhs = cutLhs(cut, x);
    return lhs <= cut.upper + tol && lhs >= cut.lower - tol;
}

Real cutCoef(const Cut& cut, Index j) {
    for (std::size_t k = 0; k < cut.indices.size(); ++k) {
        if (cut.indices[k] == j) {
            return cut.values[k];
        }
    }
    return 0.0;
}

// Columns only; strengthenCut reads bounds and types, never rows.
LpProblem columnsOnly(std::vector<Real> lower, std::vector<Real> upper,
                      std::vector<VarType> types) {
    LpProblem lp;
    lp.num_cols = static_cast<Index>(lower.size());
    lp.num_rows = 0;
    lp.obj.assign(lower.size(), 0.0);
    lp.col_lower = std::move(lower);
    lp.col_upper = std::move(upper);
    lp.col_type = std::move(types);
    lp.matrix = SparseMatrix(0, lp.num_cols, {});
    return lp;
}

Cut upperCut(std::vector<Index> idx, std::vector<Real> val, Real rhs) {
    Cut cut;
    cut.indices = std::move(idx);
    cut.values = std::move(val);
    cut.upper = rhs;
    return cut;
}

// Big-M rows over binaries and y, so separated cuts carry coefficients the
// bounds can tighten. With y a general integer the model is all-integer and
// bounded, for the enumeration check; with y continuous the optimum is
// x0 = 1, x1 = x2 = 0, y = 1.5 (objective -4.5).
LpProblem buildBigMMip(bool continuous_y = false) {
    LpProblem lp;
    lp.name = "bigm_strengthen";
    lp.sense = Sense::Minimize;
    lp.num_cols = 4;
    lp.num_rows = 2;
    lp.obj = {-3.0, -1.0, -1.0, -1.0};
    lp.col_lower = {0.0, 0.0, 0.0, 0.0};
    lp.col_upper = {1.0, 1.0, 1.0, 2.0};
    lp.col_type = {VarType::Binary, VarType::Binary, VarType::Binary,
                   continuous_y ? VarType::Continuous : VarType::Integer};
    lp.col_names = {"x0", "x1", "x2", "y"};
    lp.row_lower = {-kInf, -kInf};
    lp.row_upper = {8.5, 3.5};
    lp.row_names = {"R1", "R2"};
    std::vector<Triplet> trips = {{0, 0, 7.0}, {0, 1, 2.0}, {0, 2, 2.0}, {0, 3, 1.0},
                                  {1, 1, 1.0}, {1, 2, 1.0}, {1, 3, 1.0}};
    lp.matrix = SparseMatrix(2, 4, std::move(trips));
    return lp;
}

CutFamilyConfig onlyFamilyConfig(CutFamily family) {
    CutFamilyConfig config = noCutFamiliesConfig();
    switch (family) {
        case CutFamily::Gomory:
            config.gomory = true;
            break;
        case CutFamily::Mir:
            config.mir = true;
            break;
        case CutFamily::Cover:
            config.cover = true;
            break;
        case CutFamily::ImpliedBound:
            config.implied_bound = true;
            break;
        case CutFamily::Clique:
            config.clique = true;
            break;
        case CutFamily::ZeroHalf:
            config.zero_half = true;
            break;
        case CutFamily::Mixing:
            config.mixing = true;
            break;
        case CutFamily::Cmir:
            config.cmir = true;
            break;
        case CutFamily::StrongCg:
            config.strong_cg = true;
            break;
        case CutFamily::LiftedCover:
            config.lifted_cover = true;
            break;
        case CutFamily::ModK:
            config.mod_k = true;
            break;
        case CutFamily::IntersectionCut:
            config.intersection_cut = true;
            break;
        case CutFamily::MultiRow:
            config.multi_row = true;
            break;
        case CutFamily::Unknown:
        case CutFamily::Count:
        default:
            break;
    }
    return config;
}

// Root-LP separation of `problem` with one family armed.
CutPool separateOneFamily(const LpProblem& problem, CutFamily family, bool strengthen,
                          CutSeparationStats& stats) {
    DualSimplexSolver lp;
    lp.load(problem);
    REQUIRE(lp.solve().status == Status::Optimal);
    CutPool pool;
    SeparatorManager manager;
    manager.setConfig(onlyFamilyConfig(family));
    manager.setMaxCutsPerFamily(10);
    manager.setStrengthenCuts(strengthen);
    manager.separate(lp, problem, lp.getPrimalValues(), pool, stats);
    return pool;
}

}  // namespace

TEST_CASE("strengthenCut: lowers a big-M binary coefficient and the rhs", "[cuts][strengthen]") {
    const auto problem =
        columnsOnly({0, 0, 0}, {1, 1, 1}, {VarType::Binary, VarType::Binary, VarType::Binary});
    // 5 x0 + x1 + x2 <= 5: with x0 = 0 the row is slack by 3, so x0's
    // coefficient and the rhs both drop by 3.
    Cut cut = upperCut({0, 1, 2}, {5.0, 1.0, 1.0}, 5.0);
    CHECK(strengthenCut(cut, problem) == 1);
    CHECK_THAT(cutCoef(cut, 0), WithinAbs(2.0, 1e-7));
    CHECK_THAT(cutCoef(cut, 1), WithinAbs(1.0, 0.0));
    CHECK_THAT(cut.upper, WithinAbs(2.0, 1e-7));
    // The margin only ever weakens: never below the exact reduction.
    CHECK(cutCoef(cut, 0) >= 2.0);
    CHECK(cut.upper >= 2.0);
}

TEST_CASE("strengthenCut: a >= cut is tightened through its negation", "[cuts][strengthen]") {
    const auto problem =
        columnsOnly({0, 0, 0}, {1, 1, 1}, {VarType::Binary, VarType::Binary, VarType::Binary});
    Cut cut;
    cut.indices = {0, 1, 2};
    cut.values = {-5.0, -1.0, -1.0};
    cut.lower = -5.0;
    CHECK(strengthenCut(cut, problem) == 1);
    CHECK_THAT(cutCoef(cut, 0), WithinAbs(-2.0, 1e-7));
    CHECK_THAT(cut.lower, WithinAbs(-2.0, 1e-7));
    CHECK(cut.upper == kInf);
}

TEST_CASE("strengthenCut: a negative coefficient is tightened at its lower bound",
          "[cuts][strengthen]") {
    const auto problem =
        columnsOnly({0, 0, 0, 0}, {1, 1, 1, 1},
                    {VarType::Binary, VarType::Binary, VarType::Binary, VarType::Binary});
    // -5 x0 + x1 + x2 + x3 <= 0: with x0 = 1 the row is slack by 2.
    Cut cut = upperCut({0, 1, 2, 3}, {-5.0, 1.0, 1.0, 1.0}, 0.0);
    CHECK(strengthenCut(cut, problem) == 1);
    CHECK_THAT(cutCoef(cut, 0), WithinAbs(-3.0, 1e-7));
    CHECK_THAT(cut.upper, WithinAbs(0.0, 1e-7));
}

TEST_CASE("strengthenCut: a general integer is tightened against its upper bound",
          "[cuts][strengthen]") {
    const auto problem = columnsOnly({0, 0}, {3, 1}, {VarType::Integer, VarType::Continuous});
    // 10 y + z <= 25 with y in {0..3}: y <= 2 leaves slack 4, so 6 y + z <= 13.
    // The LP bound on y drops from 2.5 to 13/6; y = 2, z = 1 still fits.
    Cut cut = upperCut({0, 1}, {10.0, 1.0}, 25.0);
    CHECK(strengthenCut(cut, problem) == 1);
    CHECK_THAT(cutCoef(cut, 0), WithinAbs(6.0, 1e-7));
    CHECK_THAT(cutCoef(cut, 1), WithinAbs(1.0, 0.0));
    CHECK_THAT(cut.upper, WithinAbs(13.0, 1e-6));
}

TEST_CASE("strengthenCut: leaves alone what it cannot tighten", "[cuts][strengthen]") {
    SECTION("an already tight cover") {
        const auto problem =
            columnsOnly({0, 0, 0}, {1, 1, 1}, {VarType::Binary, VarType::Binary, VarType::Binary});
        Cut cut = upperCut({0, 1, 2}, {1.0, 1.0, 1.0}, 2.0);
        const Cut before = cut;
        CHECK(strengthenCut(cut, problem) == 0);
        CHECK(cut.values == before.values);
        CHECK(cut.upper == before.upper);
    }
    SECTION("a cut the bounds already satisfy") {
        // -5 x0 + x1 + x2 <= 2 holds on the whole box: no point to separate,
        // and tightening it would only shrink every coefficient towards zero.
        const auto problem =
            columnsOnly({0, 0, 0}, {1, 1, 1}, {VarType::Binary, VarType::Binary, VarType::Binary});
        Cut cut = upperCut({0, 1, 2}, {-5.0, 1.0, 1.0}, 2.0);
        CHECK(strengthenCut(cut, problem) == 0);
    }
    SECTION("a column unbounded in the direction that raises the activity") {
        const auto problem = columnsOnly({0, 0}, {1, kInf}, {VarType::Binary, VarType::Continuous});
        Cut cut = upperCut({0, 1}, {5.0, 1.0}, 5.0);
        CHECK(strengthenCut(cut, problem) == 0);
    }
    SECTION("a continuous column") {
        const auto problem =
            columnsOnly({0, 0}, {1, 1}, {VarType::Continuous, VarType::Continuous});
        Cut cut = upperCut({0, 1}, {5.0, 1.0}, 5.0);
        CHECK(strengthenCut(cut, problem) == 0);
    }
    SECTION("a two-sided cut") {
        const auto problem =
            columnsOnly({0, 0, 0}, {1, 1, 1}, {VarType::Binary, VarType::Binary, VarType::Binary});
        Cut cut = upperCut({0, 1, 2}, {5.0, 1.0, 1.0}, 5.0);
        cut.lower = 0.0;
        CHECK(strengthenCut(cut, problem) == 0);
    }
    SECTION("an index outside the problem") {
        const auto problem = columnsOnly({0}, {1}, {VarType::Binary});
        Cut cut = upperCut({0, 1}, {5.0, 1.0}, 5.0);
        CHECK(strengthenCut(cut, problem) == 0);
    }
}

TEST_CASE("strengthenCut: random cuts keep every integer point and imply the original",
          "[cuts][strengthen]") {
    // Each random one-sided cut over a few bounded columns is checked on a grid
    // with step 1/4 over the box. At every point integral on the integer
    // columns that satisfies the cut, the strengthened cut must hold too
    // (validity); at every grid point that satisfies the strengthened cut, the
    // original must hold (it is never weaker inside the bounds). The near-
    // integer coefficients are the #195 shape.
    std::mt19937 rng(201);
    const std::vector<Real> coef_pool = {1.0,  2.0,        3.0,           5.0,       7.0,
                                         -1.0, -2.0,       -4.0,          -6.0,      0.5,
                                         2.5,  1.0 - 1e-9, -(1.0 - 1e-9), 3.0 + 1e-9};
    std::uniform_int_distribution<int> pick_coef(0, static_cast<int>(coef_pool.size()) - 1);
    std::uniform_int_distribution<int> pick_type(0, 2);
    std::uniform_int_distribution<int> pick_lb(-2, 1);
    std::uniform_int_distribution<int> pick_width(1, 3);
    std::uniform_real_distribution<Real> unit(0.0, 1.0);

    Int strengthened = 0;
    const int kCuts = 400;
    for (int trial = 0; trial < kCuts; ++trial) {
        const Index n = 2 + trial % 3;
        std::vector<Real> lower(n);
        std::vector<Real> upper(n);
        std::vector<VarType> types(n);
        for (Index j = 0; j < n; ++j) {
            const int t = pick_type(rng);
            if (t == 0) {
                types[j] = VarType::Binary;
                lower[j] = 0.0;
                upper[j] = 1.0;
            } else if (t == 1) {
                types[j] = VarType::Integer;
                lower[j] = pick_lb(rng);
                upper[j] = lower[j] + pick_width(rng);
            } else {
                types[j] = VarType::Continuous;
                lower[j] = -0.5 * pick_width(rng);
                upper[j] = 0.5 * pick_width(rng);
            }
        }
        const auto problem = columnsOnly(lower, upper, types);

        Cut cut;
        Real min_act = 0.0;
        Real max_act = 0.0;
        for (Index j = 0; j < n; ++j) {
            const Real a = coef_pool[pick_coef(rng)];
            cut.indices.push_back(j);
            cut.values.push_back(a);
            min_act += std::min(a * lower[j], a * upper[j]);
            max_act += std::max(a * lower[j], a * upper[j]);
        }
        const Real rhs = min_act + unit(rng) * (max_act - min_act);
        if (trial % 2 == 0) {
            cut.upper = rhs;
        } else {
            cut.lower = rhs;
        }

        Cut strong = cut;
        if (strengthenCut(strong, problem) == 0) {
            continue;
        }
        ++strengthened;

        std::vector<Real> x(n);
        std::vector<int> step(n, 0);
        while (true) {
            bool integral = true;
            for (Index j = 0; j < n; ++j) {
                x[j] = lower[j] + 0.25 * step[j];
                if (types[j] != VarType::Continuous && std::abs(x[j] - std::round(x[j])) > 1e-12) {
                    integral = false;
                }
            }
            INFO("trial " << trial << " lhs " << cutLhs(cut, x) << " strong lhs "
                          << cutLhs(strong, x));
            if (integral && cutHolds(cut, x, 0.0)) {
                CHECK(cutHolds(strong, x, 1e-9));
            }
            if (cutHolds(strong, x, 0.0)) {
                CHECK(cutHolds(cut, x, 1e-9));
            }

            Index pos = 0;
            while (pos < n && lower[pos] + 0.25 * (step[pos] + 1) > upper[pos] + 1e-12) {
                step[pos] = 0;
                ++pos;
            }
            if (pos == n) {
                break;
            }
            ++step[pos];
        }
    }
    // Not vacuous: a good share of the random cuts had something to tighten.
    CHECK(strengthened > kCuts / 10);
}

TEST_CASE("SeparatorManager: strengthened cuts keep every integer-feasible point",
          "[cuts][strengthen]") {
    // Every family, on every all-integer model the [cuts] tests enumerate: each
    // cut pooled with strengthening on must keep every feasible integer point.
    struct Model {
        const char* name;
        LpProblem problem;
        Index limit;
    };
    const std::vector<Model> models = {
        {"cut_test", buildCutTestMip(), 6},
        {"fractional", buildFractionalMip(), 6},
        {"triangle_clique", buildTriangleCliqueMip(), 1},
        {"binary_conflict", buildBinaryConflictMip(), 1},
        {"mir_test", buildMirTestMip(), 6},
        {"bigm", buildBigMMip(), 2},
        {"divisible_row", buildDivisibleRowMip("divisible_row", 2.0, 3.0), 1},
    };

    Int pooled = 0;
    Int strengthened = 0;
    for (const auto& model : models) {
        for (std::size_t fi = 0; fi < static_cast<std::size_t>(CutFamily::Count); ++fi) {
            const auto family = static_cast<CutFamily>(fi);
            if (family == CutFamily::Unknown) {
                continue;
            }
            CutSeparationStats stats;
            const CutPool pool = separateOneFamily(model.problem, family, true, stats);
            strengthened += stats.at(family).strengthened;
            if (pool.size() == 0) {
                continue;
            }
            pooled += pool.size();
            INFO("model " << model.name << " family " << cutFamilyName(family));
            checkCutsKeepAllFeasiblePoints(model.problem, pool, model.limit);
        }
    }
    CHECK(pooled > 0);
    CHECK(strengthened > 0);
}

TEST_CASE("SeparatorManager: strengthening tightens a big-M MIR cut and reports it",
          "[cuts][strengthen]") {
    const auto problem = buildBigMMip();

    // MIR rounds R1 to 7 x0 + 2 x1 + 2 x2 + y <= 8. With x0 = 0 the other terms
    // reach at most 6, so x0's coefficient and the rhs drop by 2.
    auto findBigM = [](const CutPool& pool) -> const Cut* {
        for (Index c = 0; c < pool.size(); ++c) {
            if (cutCoef(pool[c], 0) != 0.0) {
                return &pool[c];
            }
        }
        return nullptr;
    };

    CutSeparationStats off_stats;
    const CutPool off = separateOneFamily(problem, CutFamily::Mir, false, off_stats);
    CutSeparationStats on_stats;
    const CutPool on = separateOneFamily(problem, CutFamily::Mir, true, on_stats);

    const Cut* off_cut = findBigM(off);
    const Cut* on_cut = findBigM(on);
    REQUIRE(off_cut != nullptr);
    REQUIRE(on_cut != nullptr);
    CHECK_THAT(cutCoef(*off_cut, 0), WithinAbs(7.0, 1e-9));
    CHECK_THAT(off_cut->upper, WithinAbs(8.0, 1e-9));
    CHECK_THAT(cutCoef(*on_cut, 0), WithinAbs(5.0, 1e-7));
    CHECK_THAT(on_cut->upper, WithinAbs(6.0, 1e-7));
    CHECK(on_cut->efficacy > 3.0 * off_cut->efficacy);

    const auto& off_mir = off_stats.at(CutFamily::Mir);
    const auto& on_mir = on_stats.at(CutFamily::Mir);
    CHECK(off_mir.strengthened == 0);
    CHECK(off_mir.strengthened_coefs == 0);
    CHECK(off_mir.strengthen_efficacy_gain == 0.0);
    CHECK(on_mir.strengthened == 1);
    CHECK(on_mir.strengthened_coefs == 1);
    CHECK_THAT(on_mir.strengthen_efficacy_gain,
               WithinAbs(on_cut->efficacy - off_cut->efficacy, 1e-7));
    checkCutsKeepAllFeasiblePoints(problem, on, 2);
}

TEST_CASE("SeparatorManager: strengthened cuts keep the mixed-integer optimum",
          "[cuts][strengthen]") {
    // Models with continuous columns cannot be enumerated, so every family's
    // cuts are checked against a known integer-feasible optimum instead.
    const auto bigm = buildBigMMip(true);
    const std::vector<Real> bigm_optimum = {1.0, 0.0, 0.0, 1.5};
    const auto cmir = buildCmirValidityMip();

    Index checked = 0;
    for (std::size_t fi = 0; fi < static_cast<std::size_t>(CutFamily::Count); ++fi) {
        const auto family = static_cast<CutFamily>(fi);
        if (family == CutFamily::Unknown) {
            continue;
        }
        INFO("family " << cutFamilyName(family));
        checked += separateAndCheckCutsKeep(bigm, bigm_optimum, onlyFamilyConfig(family), true);
        checked +=
            separateAndCheckCutsKeep(cmir, cmirValidityOptimum(), onlyFamilyConfig(family), true);
    }
    CHECK(checked > 0);
}

TEST_CASE("SeparatorManager: strengthening keeps the near-integer optimum", "[cuts][strengthen]") {
    // The #195 models: a coefficient of 1 - 1e-9 next to an rhs just under an
    // integer, where rounding the wrong way removes x = 100001.
    const std::vector<Real> optimum = {kNearIntegerOptimum, 0.0};
    struct Case {
        const char* name;
        LpProblem problem;
        CutFamilyConfig config;
    };
    const std::vector<Case> cases = {
        {"strong_cg", buildStrongCgNearIntegerMip(), onlyStrongCgConfig()},
        {"mir", buildNearIntegerMip("mir_near_integer", 1.0, 1.0, 0.5), onlyMirConfig()},
        {"zero_half", buildNearIntegerMip("zerohalf_near_integer", 2.0, 3.0, 1.0),
         onlyZeroHalfConfig()},
        {"mixing", buildNearIntegerMip("mixing_near_integer", 3.0, 4.0, 1.0), onlyMixingConfig()},
        {"mod_k", buildNearIntegerMip("modk_near_integer", 2.0, 3.0, 1.0), onlyModKConfig()},
        {"gomory", buildNearIntegerMip("gomory_near_integer", 1.0, 1.0, 0.5),
         onlyFamilyConfig(CutFamily::Gomory)},
    };
    for (const auto& c : cases) {
        INFO("family " << c.name);
        separateAndCheckCutsKeep(c.problem, optimum, c.config, true);
        checkNearIntegerOptimum(c.problem, c.config, true);
    }
}

TEST_CASE("MipSolver: cut strengthening defaults off and keeps optima when on",
          "[cuts][strengthen][integration]") {
    CHECK_FALSE(MipSolver{}.getCutStrengthening());

    auto solveBoth = [](const LpProblem& problem) {
        CutFamilyConfig all;
        const auto off = solveWithCutFamilies(problem, all, false);
        const auto on = solveWithCutFamilies(problem, all, true);
        REQUIRE(off.status == Status::Optimal);
        REQUIRE(on.status == Status::Optimal);
        CHECK_THAT(on.objective, WithinAbs(off.objective, 1e-6));
    };
    solveBoth(buildBigMMip());
    solveBoth(buildBigMMip(true));
    solveBoth(buildCutTestMip());
    solveBoth(buildFractionalMip());
    solveBoth(buildCmirValidityMip());
    solveBoth(buildStrongCgKnapsackMip());
}
