#include "presolve/dual_bounds.h"
#include "presolve/presolve_matrix.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <vector>

using namespace mipx;
using Catch::Matchers::WithinAbs;

namespace {

struct Model {
    LpProblem lp;
    PresolveMatrix matrix;
    std::vector<bool> col_removed;
    std::vector<bool> row_removed;

    DualBounds compute() const {
        DualBounds b;
        b.compute(lp, matrix, col_removed, row_removed);
        return b;
    }
};

/// rows x cols LP from triplets, every column continuous.
Model makeModel(Index rows, Index cols, const std::vector<Triplet>& entries, std::vector<Real> obj,
                std::vector<Real> col_lower, std::vector<Real> col_upper,
                std::vector<Real> row_lower, std::vector<Real> row_upper,
                Sense sense = Sense::Minimize) {
    LpProblem lp;
    lp.sense = sense;
    lp.num_cols = cols;
    lp.num_rows = rows;
    lp.obj = std::move(obj);
    lp.col_lower = std::move(col_lower);
    lp.col_upper = std::move(col_upper);
    lp.col_type.assign(static_cast<size_t>(cols), VarType::Continuous);
    lp.row_lower = std::move(row_lower);
    lp.row_upper = std::move(row_upper);
    lp.matrix = SparseMatrix(rows, cols, entries);
    PresolveMatrix m(lp.matrix);
    return Model{std::move(lp), std::move(m), std::vector<bool>(static_cast<size_t>(cols), false),
                 std::vector<bool>(static_cast<size_t>(rows), false)};
}

}  // namespace

TEST_CASE("DualBounds: row-dual bounds follow the row sides", "[presolve][dual_bounds]") {
    // Columns x, y in [0, 10]; rows: >= , <=, =, ranged, free.
    const Model m = makeModel(5, 2,
                              {{0, 0, 1.0},
                               {0, 1, 1.0},
                               {1, 0, 1.0},
                               {1, 1, -1.0},
                               {2, 0, 1.0},
                               {2, 1, 1.0},
                               {3, 0, 1.0},
                               {3, 1, 1.0},
                               {4, 0, 1.0},
                               {4, 1, 1.0}},
                              {1.0, 1.0}, {0.0, 0.0}, {10.0, 10.0}, {1.0, -kInf, 3.0, 0.0, -kInf},
                              {kInf, 2.0, 3.0, 5.0, kInf});
    const DualBounds b = m.compute();
    CHECK(b.rowDualLower(0) == 0.0);
    CHECK(b.rowDualUpper(0) == kInf);
    CHECK(b.rowDualLower(1) == -kInf);
    CHECK(b.rowDualUpper(1) == 0.0);
    CHECK(b.rowDualLower(2) == -kInf);
    CHECK(b.rowDualUpper(2) == kInf);
    CHECK(b.rowDualLower(3) == -kInf);
    CHECK(b.rowDualUpper(3) == kInf);
    CHECK(b.rowDualLower(4) == 0.0);
    CHECK(b.rowDualUpper(4) == 0.0);
}

TEST_CASE("DualBounds: a free column bounds the duals of its >= rows from above",
          "[presolve][dual_bounds]") {
    // min 2x + z + w, x free, z, w in [0, 5]; r0: x + z >= 1, r1: x + w >= 1.
    // x's dual constraint is y0 + y1 = 2 with y0, y1 >= 0, so y0, y1 <= 2.
    const Model m =
        makeModel(2, 3, {{0, 0, 1.0}, {0, 1, 1.0}, {1, 0, 1.0}, {1, 2, 1.0}}, {2.0, 1.0, 1.0},
                  {-kInf, 0.0, 0.0}, {kInf, 5.0, 5.0}, {1.0, 1.0}, {kInf, kInf});
    const DualBounds b = m.compute();
    CHECK(b.isStrictlyImpliedForDual(0, true));
    CHECK(b.isStrictlyImpliedForDual(0, false));
    CHECK_FALSE(b.isStrictlyImpliedForDual(1, true));

    CHECK_THAT(b.implRowDualUpper(0), WithinAbs(2.0, 1e-12));
    CHECK_THAT(b.implRowDualUpper(1), WithinAbs(2.0, 1e-12));
    CHECK(b.rowDualUpperSource(0) == 0);
    CHECK(b.rowDualUpperSource(1) == 0);
    CHECK(b.implRowDualLower(0) == -kInf);
    CHECK(b.rowDualLowerSource(0) == -1);
    CHECK(b.rowsSourcedBy(0) == std::vector<Index>{0, 1});
    CHECK(b.numImpliedRowDualBounds() == 2);
    CHECK_FALSE(b.isImpliedEquationAtLower(0));

    // x's own sums leave out the bounds it is the source of.
    CHECK(b.sumLowerOrig(0) == 0.0);
    CHECK(b.numInfSumUpperOrig(0) == 2);
    CHECK(b.sumLower(0) == 0.0);
    CHECK(b.numInfSumUpper(0) == 2);
    CHECK(b.sumUpper(0) == kInf);
    // z's take y0 <= 2 from x.
    CHECK(b.sumUpperOrig(1) == kInf);
    CHECK_THAT(b.sumUpper(1), WithinAbs(2.0, 1e-12));
    CHECK_THAT(b.sumUpper(1, -1.0), WithinAbs(1.0, 1e-12));
    CHECK(b.residualSumUpper(1, 0, 1.0) == 0.0);
    CHECK(b.residualSumUpperOrig(1, 0, 1.0) == 0.0);
    CHECK(b.residualSumLowerOrig(0, 0, 1.0) == 0.0);
    CHECK(b.residualSumUpperOrig(0, 0, 1.0) == kInf);
}

TEST_CASE("DualBounds: a maximisation bounds the duals of its <= rows from below",
          "[presolve][dual_bounds]") {
    // max 2x, i.e. min -2x, x free; r0: x + z <= 4, r1: x + w <= 4, so
    // y0 + y1 = -2 with y0, y1 <= 0: y0, y1 >= -2.
    const Model m =
        makeModel(2, 3, {{0, 0, 1.0}, {0, 1, 1.0}, {1, 0, 1.0}, {1, 2, 1.0}}, {2.0, 0.0, 0.0},
                  {-kInf, 0.0, 0.0}, {kInf, 5.0, 5.0}, {-kInf, -kInf}, {4.0, 4.0}, Sense::Maximize);
    const DualBounds b = m.compute();
    CHECK_THAT(b.implRowDualLower(0), WithinAbs(-2.0, 1e-12));
    CHECK_THAT(b.implRowDualLower(1), WithinAbs(-2.0, 1e-12));
    CHECK(b.rowDualLowerSource(0) == 0);
    CHECK(b.implRowDualUpper(0) == kInf);
    CHECK(b.sumUpperOrig(0) == 0.0);
    CHECK(b.numInfSumLowerOrig(0) == 2);
}

TEST_CASE("DualBounds: negative coefficients swap which dual bound is implied",
          "[presolve][dual_bounds]") {
    // min 2x, x free; r0: -x + z <= 4 (y0 <= 0), r1: x + w >= 1 (y1 >= 0).
    // -y0 + y1 = 2: y0 >= -2, y1 <= 2.
    const Model m =
        makeModel(2, 3, {{0, 0, -1.0}, {0, 1, 1.0}, {1, 0, 1.0}, {1, 2, 1.0}}, {2.0, 0.0, 0.0},
                  {-kInf, 0.0, 0.0}, {kInf, 5.0, 5.0}, {-kInf, 1.0}, {4.0, kInf});
    const DualBounds b = m.compute();
    CHECK_THAT(b.implRowDualLower(0), WithinAbs(-2.0, 1e-12));
    CHECK(b.implRowDualUpper(0) == kInf);
    CHECK_THAT(b.implRowDualUpper(1), WithinAbs(2.0, 1e-12));
    CHECK(b.implRowDualLower(1) == -kInf);
    CHECK(b.sumLowerOrig(0) == 0.0);
    CHECK(b.numInfSumUpperOrig(0) == 2);
}

TEST_CASE("DualBounds: ranged and free rows", "[presolve][dual_bounds]") {
    // min 2x, x free; r0: 0 <= x + z <= 4 (y0 free), r1: x + w >= 1
    // (y1 >= 0), r2: x + v free (y2 = 0). y0 + y1 + y2 = 2: only y0 has its
    // one unbounded term, so only y0 <= 2 follows.
    const Model m = makeModel(
        3, 4, {{0, 0, 1.0}, {0, 1, 1.0}, {1, 0, 1.0}, {1, 2, 1.0}, {2, 0, 1.0}, {2, 3, 1.0}},
        {2.0, 0.0, 0.0, 0.0}, {-kInf, 0.0, 0.0, 0.0}, {kInf, 5.0, 5.0, 5.0}, {0.0, 1.0, -kInf},
        {4.0, kInf, kInf});
    const DualBounds b = m.compute();
    CHECK(b.numInfSumLowerOrig(0) == 1);
    CHECK_THAT(b.implRowDualUpper(0), WithinAbs(2.0, 1e-12));
    CHECK(b.implRowDualLower(0) == -kInf);
    CHECK(b.implRowDualUpper(1) == kInf);
    CHECK(b.implRowDualUpper(2) == kInf);
    CHECK(b.implRowDualLower(2) == -kInf);
    CHECK(b.numImpliedRowDualBounds() == 1);
}

TEST_CASE("DualBounds: primal implied bounds make a bounded column's sides strict",
          "[presolve][dual_bounds]") {
    // min 2x + w, x in [0, 100], z in [1, 5], w in [0, 5]; r0: x - z = 0,
    // r1: x + w >= 1. r0 implies x in [1, 5], so neither bound of x is
    // active: y0 + y1 = 2, and with y1 >= 0, y0 <= 2.
    const Model m =
        makeModel(2, 3, {{0, 0, 1.0}, {0, 1, -1.0}, {1, 0, 1.0}, {1, 2, 1.0}}, {2.0, 0.0, 1.0},
                  {0.0, 1.0, 0.0}, {100.0, 5.0, 5.0}, {0.0, 1.0}, {0.0, kInf});
    const DualBounds b = m.compute();
    CHECK_THAT(b.implColLower(0), WithinAbs(1.0, 1e-12));
    CHECK_THAT(b.implColUpper(0), WithinAbs(5.0, 1e-12));
    CHECK(b.colLowerSource(0) == 0);
    CHECK(b.colUpperSource(0) == 0);
    CHECK(b.isStrictlyImpliedForDual(0, true));
    CHECK(b.isStrictlyImpliedForDual(0, false));
    CHECK_THAT(b.implRowDualUpper(0), WithinAbs(2.0, 1e-12));
    CHECK(b.rowDualUpperSource(0) == 0);

    SECTION("fixing the source column drops the bounds it gave") {
        // As presolve fixes x = 3: x removed, its term moved into the row
        // sides, so r0: -z = -3 and r1: w >= -2.
        Model fixed = m;
        fixed.col_removed[0] = true;
        fixed.lp.row_lower = {-3.0, -2.0};
        fixed.lp.row_upper = {-3.0, kInf};
        const DualBounds f = fixed.compute();
        CHECK(f.rowsSourcedBy(0).empty());
        // z = 3 now lies strictly inside z's [1, 5], so z's dual constraint
        // -y0 = 0 bounds y0 from both sides instead.
        CHECK(f.colLowerSource(1) == 0);
        CHECK(f.colUpperSource(1) == 0);
        CHECK_THAT(f.implColLower(1), WithinAbs(3.0, 1e-12));
        CHECK_THAT(f.implColUpper(1), WithinAbs(3.0, 1e-12));
        CHECK(f.rowDualLowerSource(0) == 1);
        CHECK(f.rowDualUpperSource(0) == 1);
        CHECK_THAT(f.implRowDualLower(0), WithinAbs(0.0, 1e-12));
        CHECK_THAT(f.implRowDualUpper(0), WithinAbs(0.0, 1e-12));
        // r1: w >= -2 implies nothing strict about w in [0, 5].
        CHECK_FALSE(f.isStrictlyImpliedForDual(2, true));
        CHECK(f.rowDualUpperSource(1) == -1);
    }
    SECTION("removing a row drops its term from the dual sums") {
        Model removed = m;
        removed.row_removed[0] = true;
        const DualBounds r = removed.compute();
        CHECK(r.implRowDualUpper(0) == kInf);
        CHECK(r.numInfSumLowerOrig(0) == 0);
        CHECK(r.numInfSumUpperOrig(0) == 1);
        // Only r1 bounds x now, and not strictly: x >= 1 - 5.
        CHECK_THAT(r.implColLower(0), WithinAbs(-4.0, 1e-12));
        CHECK(r.colLowerSource(0) == 1);
        CHECK_FALSE(r.isStrictlyImpliedForDual(0, true));
    }
}

TEST_CASE("DualBounds: an implied equation tightens the primal implied bounds",
          "[presolve][dual_bounds]") {
    // min 2x + z, x free, z in [0, 5]; r0: x + z >= 1. x's dual constraint
    // is y0 = 2 > 0, so r0 holds at equality in every optimum: x + z = 1,
    // which implies x <= 1, a bound r0's lower side alone does not give.
    const Model m = makeModel(1, 2, {{0, 0, 1.0}, {0, 1, 1.0}}, {2.0, 1.0}, {-kInf, 0.0},
                              {kInf, 5.0}, {1.0}, {kInf});
    const DualBounds b = m.compute();
    CHECK_THAT(b.implRowDualLower(0), WithinAbs(2.0, 1e-12));
    CHECK_THAT(b.implRowDualUpper(0), WithinAbs(2.0, 1e-12));
    CHECK(b.isImpliedEquationAtLower(0));
    CHECK_FALSE(b.isImpliedEquationAtUpper(0));
    CHECK_THAT(b.implColUpper(0), WithinAbs(1.0, 1e-12));
    CHECK(b.colUpperSource(0) == 0);
    CHECK_THAT(b.implColLower(0), WithinAbs(-4.0, 1e-12));
}
