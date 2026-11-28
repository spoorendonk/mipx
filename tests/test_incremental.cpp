#include "mipx/dual_simplex.h"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

using namespace mipx;
using Catch::Matchers::WithinAbs;

// ---------------------------------------------------------------------------
// Helper: build LP for testing
// min c^T x  s.t.  Ax <= b,  x >= 0
// ---------------------------------------------------------------------------

static LpProblem buildLP_2var() {
    // min -x - 2y  s.t.  x + y <= 4,  x <= 3,  y <= 3,  x,y >= 0
    // Optimal: x=1, y=3, obj=-7
    LpProblem lp;
    lp.name = "twovars";
    lp.sense = Sense::Minimize;
    lp.num_cols = 2;
    lp.obj = {-1.0, -2.0};
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {kInf, kInf};
    lp.col_type = {VarType::Continuous, VarType::Continuous};
    lp.col_names = {"x", "y"};

    lp.num_rows = 3;
    lp.row_lower = {-kInf, -kInf, -kInf};
    lp.row_upper = {4.0, 3.0, 3.0};
    lp.row_names = {"sum", "ub_x", "ub_y"};

    // Row 0: x + y <= 4
    // Row 1: x <= 3
    // Row 2: y <= 3
    std::vector<Triplet> trips = {
        {0, 0, 1.0},
        {0, 1, 1.0},
        {1, 0, 1.0},
        {2, 1, 1.0},
    };
    lp.matrix = SparseMatrix(3, 2, std::move(trips));
    return lp;
}

// ---------------------------------------------------------------------------
// Test: setColBounds — tighten bound and re-solve
// ---------------------------------------------------------------------------
TEST_CASE("Incremental: setColBounds tighten and re-solve", "[incremental]") {
    auto lp = buildLP_2var();
    DualSimplexSolver solver;
    solver.load(lp);

    // First solve: optimal x=1, y=3, obj=-7
    auto r1 = solver.solve();
    REQUIRE(r1.status == Status::Optimal);
    CHECK_THAT(r1.objective, WithinAbs(-7.0, 1e-6));

    // Tighten y <= 2 (via column bound).
    solver.setColBounds(1, 0.0, 2.0);
    auto r2 = solver.solve();
    REQUIRE(r2.status == Status::Optimal);
    // Now y=2, x=2, obj = -1*2 + -2*2 = -6
    CHECK_THAT(r2.objective, WithinAbs(-6.0, 1e-6));

    // Warm-start should use fewer iterations than cold start.
    // (Not a hard requirement but should generally hold.)
}

// ---------------------------------------------------------------------------
// Test: setColBounds — fix variable
// ---------------------------------------------------------------------------
TEST_CASE("Incremental: setColBounds fix variable", "[incremental]") {
    auto lp = buildLP_2var();
    DualSimplexSolver solver;
    solver.load(lp);

    auto r1 = solver.solve();
    REQUIRE(r1.status == Status::Optimal);

    // Fix x = 0
    solver.setColBounds(0, 0.0, 0.0);
    auto r2 = solver.solve();
    REQUIRE(r2.status == Status::Optimal);
    // y = 3 (limited by y <= 3 and x + y <= 4), obj = -2*3 = -6
    CHECK_THAT(r2.objective, WithinAbs(-6.0, 1e-6));
    auto primals = solver.getPrimalValues();
    CHECK_THAT(primals[0], WithinAbs(0.0, 1e-6));
    CHECK_THAT(primals[1], WithinAbs(3.0, 1e-6));
}

// ---------------------------------------------------------------------------
// Test: setColBounds — make infeasible
// ---------------------------------------------------------------------------
TEST_CASE("Incremental: setColBounds makes LP infeasible", "[incremental]") {
    auto lp = buildLP_2var();
    DualSimplexSolver solver;
    solver.load(lp);

    auto r1 = solver.solve();
    REQUIRE(r1.status == Status::Optimal);

    // Force x >= 5, but x + y <= 4 and y >= 0 => x <= 4. Contradiction.
    solver.setColBounds(0, 5.0, kInf);
    auto r2 = solver.solve();
    CHECK(r2.status == Status::Infeasible);
}

// ---------------------------------------------------------------------------
// Test: setObjective — change objective and re-solve
// ---------------------------------------------------------------------------
TEST_CASE("Incremental: setObjective and re-solve", "[incremental]") {
    auto lp = buildLP_2var();
    DualSimplexSolver solver;
    solver.load(lp);

    auto r1 = solver.solve();
    REQUIRE(r1.status == Status::Optimal);

    // Change objective: min -2x - y (swap weights)
    std::vector<Real> new_obj = {-2.0, -1.0};
    solver.setObjective(new_obj);
    auto r2 = solver.solve();
    REQUIRE(r2.status == Status::Optimal);
    // Now x=3, y=1, obj = -2*3 + -1*1 = -7
    CHECK_THAT(r2.objective, WithinAbs(-7.0, 1e-6));
    auto primals = solver.getPrimalValues();
    CHECK_THAT(primals[0], WithinAbs(3.0, 1e-6));
    CHECK_THAT(primals[1], WithinAbs(1.0, 1e-6));
}

// ---------------------------------------------------------------------------
// Test: addRows — add a constraint and re-solve
// ---------------------------------------------------------------------------
TEST_CASE("Incremental: addRows and re-solve", "[incremental]") {
    auto lp = buildLP_2var();
    DualSimplexSolver solver;
    solver.load(lp);

    auto r1 = solver.solve();
    REQUIRE(r1.status == Status::Optimal);
    CHECK_THAT(r1.objective, WithinAbs(-7.0, 1e-6));

    // Add constraint: x + 2y <= 5
    // This makes the old optimal (x=1, y=3) infeasible (1 + 6 = 7 > 5).
    // New optimal should be on the intersection of x+y<=4, x+2y<=5.
    // Solving: y = 1, x = 3, obj = -1*3 + -2*1 = -5
    // Or: from x+y=4, x+2y=5: y=1, x=3. Check: x<=3 ok, y<=3 ok.
    std::vector<Index> starts = {0};
    std::vector<Index> indices = {0, 1};
    std::vector<Real> values = {1.0, 2.0};
    std::vector<Real> lower = {-kInf};
    std::vector<Real> upper = {5.0};

    solver.addRows(starts, indices, values, lower, upper);
    auto r2 = solver.solve();
    REQUIRE(r2.status == Status::Optimal);
    CHECK_THAT(r2.objective, WithinAbs(-5.0, 1e-6));
}

// ---------------------------------------------------------------------------
// Test: addRows — add infeasible constraint
// ---------------------------------------------------------------------------
TEST_CASE("Incremental: addRows makes LP infeasible", "[incremental]") {
    auto lp = buildLP_2var();
    DualSimplexSolver solver;
    solver.load(lp);

    auto r1 = solver.solve();
    REQUIRE(r1.status == Status::Optimal);

    // Add constraint: x + y >= 10 (but x+y <= 4 already)
    std::vector<Index> starts = {0};
    std::vector<Index> indices = {0, 1};
    std::vector<Real> values = {1.0, 1.0};
    std::vector<Real> lower = {10.0};
    std::vector<Real> upper = {kInf};

    solver.addRows(starts, indices, values, lower, upper);
    auto r2 = solver.solve();
    CHECK(r2.status == Status::Infeasible);
}

// ---------------------------------------------------------------------------
// Test: setBasis — warm-start with saved basis
// ---------------------------------------------------------------------------
TEST_CASE("Incremental: setBasis warm-start", "[incremental]") {
    auto lp = buildLP_2var();
    DualSimplexSolver solver;
    solver.load(lp);

    auto r1 = solver.solve();
    REQUIRE(r1.status == Status::Optimal);

    // Save basis.
    auto basis = solver.getBasis();

    // Create a fresh solver, load same problem, set basis, solve.
    DualSimplexSolver solver2;
    solver2.load(lp);
    solver2.setBasis(basis);
    auto r2 = solver2.solve();
    REQUIRE(r2.status == Status::Optimal);
    CHECK_THAT(r2.objective, WithinAbs(-7.0, 1e-6));
    // Should solve in 0 iterations (already optimal).
    CHECK(r2.iterations <= 1);
}

// ---------------------------------------------------------------------------
// Test: Multiple modifications — bound change + re-solve multiple times
// ---------------------------------------------------------------------------
TEST_CASE("Incremental: multiple bound changes", "[incremental]") {
    auto lp = buildLP_2var();
    DualSimplexSolver solver;
    solver.load(lp);

    auto r1 = solver.solve();
    REQUIRE(r1.status == Status::Optimal);
    CHECK_THAT(r1.objective, WithinAbs(-7.0, 1e-6));

    // Fix x = 2
    solver.setColBounds(0, 2.0, 2.0);
    auto r2 = solver.solve();
    REQUIRE(r2.status == Status::Optimal);
    // y = min(3, 4-2) = 2, obj = -1*2 + -2*2 = -6
    CHECK_THAT(r2.objective, WithinAbs(-6.0, 1e-6));

    // Now fix x = 3
    solver.setColBounds(0, 3.0, 3.0);
    auto r3 = solver.solve();
    REQUIRE(r3.status == Status::Optimal);
    // y = min(3, 4-3) = 1, obj = -1*3 + -2*1 = -5
    CHECK_THAT(r3.objective, WithinAbs(-5.0, 1e-6));

    // Release x back to [0, inf)
    solver.setColBounds(0, 0.0, kInf);
    auto r4 = solver.solve();
    REQUIRE(r4.status == Status::Optimal);
    CHECK_THAT(r4.objective, WithinAbs(-7.0, 1e-6));
}

// ---------------------------------------------------------------------------
// Test: removeRows that cannot preserve the basis, then addRows and re-solve
// ---------------------------------------------------------------------------

// min -x - 2y  s.t.  x + y <= 4,  x <= 3,  y <= 3,  x + 2y <= 9,  x, y >= 0
// Optimal: x = 1, y = 3, obj = -7.
static LpProblem buildLP_2var4row() {
    LpProblem lp;
    lp.name = "twovars4row";
    lp.sense = Sense::Minimize;
    lp.num_cols = 2;
    lp.obj = {-1.0, -2.0};
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {kInf, kInf};
    lp.col_type = {VarType::Continuous, VarType::Continuous};
    lp.col_names = {"x", "y"};

    lp.num_rows = 4;
    lp.row_lower = {-kInf, -kInf, -kInf, -kInf};
    lp.row_upper = {4.0, 3.0, 3.0, 9.0};
    lp.row_names = {"sum", "ub_x", "ub_y", "mix"};

    std::vector<Triplet> trips = {
        {0, 0, 1.0}, {0, 1, 1.0}, {1, 0, 1.0}, {2, 1, 1.0}, {3, 0, 1.0}, {3, 1, 2.0},
    };
    lp.matrix = SparseMatrix(4, 2, std::move(trips));
    return lp;
}

TEST_CASE("Incremental: removeRows without basis preservation leaves no stale state",
          "[incremental]") {
    // removeRows can only remap the basis when every kept row's basic variable
    // survives. Below, a kept row's basic variable is the logical of a row that
    // is being removed, so it cannot, and the fallback has to bring every
    // per-variable vector down to the new size. Leaving them at the old size is
    // not harmless: addRows runs before the next solve(), resizes them back
    // into range and re-asserts has_basis_, and solve() then walks a nonbasic_
    // still naming variables of rows that no longer exist -- reading, and
    // writing, past the end of the current ones. Found through the in-tree cut
    // loop once Gomory started feeding it (issue #211).
    auto lp = buildLP_2var4row();
    DualSimplexSolver solver;
    solver.load(lp);
    REQUIRE(solver.solve().status == Status::Optimal);

    // Variables: 0, 1 structural; 2..5 the logicals of rows 0..3. setBasis
    // hands out basis positions in index order, so this puts the logical of
    // row 0 -- which is removed below -- at the position of kept row 1.
    std::vector<BasisStatus> basis = {
        BasisStatus::AtLower,  // x
        BasisStatus::Basic,    // y
        BasisStatus::Basic,    // logical of row 0  <- removed
        BasisStatus::Basic,    // logical of row 1
        BasisStatus::Basic,    // logical of row 2
        BasisStatus::AtUpper,  // logical of row 3  <- removed
    };
    solver.setBasis(basis);

    const std::vector<Index> remove = {0, 3};
    solver.removeRows(remove);
    REQUIRE(solver.numRows() == 2);

    // A cut arrives before the next solve. This is what resizes the vectors the
    // fallback left oversized, putting stale variable indices back in range.
    const std::vector<Index> starts = {0};
    const std::vector<Index> indices = {0, 1};
    const std::vector<Real> values = {1.0, 1.0};
    const std::vector<Real> row_lower = {-kInf};
    const std::vector<Real> row_upper = {2.0};
    solver.addRows(starts, indices, values, row_lower, row_upper);
    REQUIRE(solver.numRows() == 3);

    // What is left is x <= 3, y <= 3 and x + y <= 2, so min -x - 2y is at
    // y = 2, x = 0 with objective -4.
    auto result = solver.solve();
    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(-4.0, 1e-6));

    auto primals = solver.getPrimalValues();
    REQUIRE(primals.size() == 2);
    CHECK_THAT(primals[1], WithinAbs(2.0, 1e-6));

    // The basis status vector must describe the LP as it stands now.
    CHECK(static_cast<Index>(solver.getBasis().size()) == solver.numCols() + solver.numRows());
}

// ---------------------------------------------------------------------------
// keep_basis_on_row_removal (#519): removeRows keeps the basis by pivoting
// each removed row's nonbasic logical into it first.
// ---------------------------------------------------------------------------

static DualSimplexSolver keepBasisSolver(const LpProblem& lp, bool keep) {
    DualSimplexSolver solver;
    DualSimplexOptions opts;
    opts.keep_basis_on_row_removal = keep;
    solver.setOptions(opts);
    solver.load(lp);
    return solver;
}

// The LP with @p rows (sorted) left out.
static LpProblem withoutRows(const LpProblem& lp, const std::vector<Index>& rows) {
    std::vector<Index> row_map(static_cast<std::size_t>(lp.num_rows), 0);
    Index removed = 0;
    for (Index i = 0; i < lp.num_rows; ++i) {
        const bool drop = std::ranges::binary_search(rows, i);
        row_map[i] = drop ? -1 : i - removed;
        removed += drop ? 1 : 0;
    }
    LpProblem out = lp;
    out.num_rows = lp.num_rows - removed;
    out.row_lower.clear();
    out.row_upper.clear();
    out.row_names.clear();
    std::vector<Triplet> trips;
    for (Index i = 0; i < lp.num_rows; ++i) {
        if (row_map[i] < 0) {
            continue;
        }
        out.row_lower.push_back(lp.row_lower[i]);
        out.row_upper.push_back(lp.row_upper[i]);
        out.row_names.push_back(lp.row_names[i]);
        auto rv = lp.matrix.row(i);
        for (Index k = 0; k < rv.size(); ++k) {
            trips.push_back({row_map[i], rv.indices[k], rv.values[k]});
        }
    }
    out.matrix = SparseMatrix(out.num_rows, lp.num_cols, std::move(trips));
    return out;
}

static Real coldObjective(const LpProblem& lp) {
    DualSimplexSolver solver;
    solver.load(lp);
    auto r = solver.solve();
    REQUIRE(r.status == Status::Optimal);
    return r.objective;
}

TEST_CASE("Incremental: keep_basis_on_row_removal keeps it through binding rows' removal",
          "[incremental]") {
    // The optimal basis at x = 1, y = 3: rows 0 (x + y <= 4) and 2 (y <= 3)
    // bind, so their logicals are nonbasic and removing both rows leaves two
    // basic variables too many. This is the first solve's own basis, so
    // setBasis keeps that solve's positions, and what the positional rule
    // keeps depends on its pivot order. The option instead pivots both
    // removed rows' logicals in first, refactorizing once (counted in the
    // next solve's edits), whatever the positions.
    const std::vector<BasisStatus> basis = {
        BasisStatus::Basic,    // x
        BasisStatus::Basic,    // y
        BasisStatus::AtUpper,  // logical of row 0  <- removed
        BasisStatus::Basic,    // logical of row 1
        BasisStatus::AtUpper,  // logical of row 2  <- removed
        BasisStatus::Basic,    // logical of row 3
    };
    const std::vector<Index> remove = {0, 2};
    auto lp = buildLP_2var4row();

    for (const bool keep : {true, false}) {
        CAPTURE(keep);
        auto solver = keepBasisSolver(lp, keep);
        REQUIRE(solver.solve().status == Status::Optimal);
        solver.setBasis(basis);
        solver.removeRows(remove);
        REQUIRE(solver.numRows() == 2);

        // Left: x <= 3, x + 2y <= 9, so x = 3, y = 3 and objective -9.
        auto result = solver.solve();
        REQUIRE(result.status == Status::Optimal);
        CHECK_FALSE(solver.basisStartStats().cold_start);
        if (keep) {
            CHECK_FALSE(solver.basisStartStats().singular_recovery);
        }
        CHECK((solver.getRefactorStats().edits > 0) == keep);
        CHECK_THAT(result.objective, WithinAbs(-9.0, 1e-6));
        CHECK_THAT(result.objective, WithinAbs(coldObjective(withoutRows(lp, remove)), 1e-6));
    }
}

TEST_CASE("Incremental: keep_basis_on_row_removal keeps it when a logical is basic elsewhere",
          "[incremental]") {
    // The basis of the stale-state test above: row 0's logical is basic at
    // kept row 1's position, which the positional rule cannot remap, and row
    // 3's logical is nonbasic.
    const std::vector<BasisStatus> basis = {
        BasisStatus::AtLower,  // x
        BasisStatus::Basic,    // y
        BasisStatus::Basic,    // logical of row 0  <- removed
        BasisStatus::Basic,    // logical of row 1
        BasisStatus::Basic,    // logical of row 2
        BasisStatus::AtUpper,  // logical of row 3  <- removed
    };
    const std::vector<Index> remove = {0, 3};
    auto lp = buildLP_2var4row();

    for (const bool keep : {true, false}) {
        CAPTURE(keep);
        auto solver = keepBasisSolver(lp, keep);
        REQUIRE(solver.solve().status == Status::Optimal);
        solver.setBasis(basis);
        solver.removeRows(remove);
        REQUIRE(solver.numRows() == 2);

        // Left: x <= 3, y <= 3, so x = y = 3 and objective -9.
        auto result = solver.solve();
        REQUIRE(result.status == Status::Optimal);
        CHECK(solver.basisStartStats().cold_start == !keep);
        CHECK_FALSE(solver.basisStartStats().singular_recovery);
        CHECK_THAT(result.objective, WithinAbs(-9.0, 1e-6));
    }
}

// min c^T x over 0 <= x <= 5 and m rows a_i x <= b_i with b_i > 0, from a
// fixed linear congruential stream, so the LP is feasible (x = 0) and bounded.
static LpProblem buildSeededLP(std::uint32_t seed, Index n, Index m) {
    std::uint32_t s = seed;
    const auto next = [&s](std::uint32_t mod) {
        s = s * 1664525u + 1013904223u;
        return (s >> 8) % mod;
    };
    LpProblem lp;
    lp.name = "seeded";
    lp.sense = Sense::Minimize;
    lp.num_cols = n;
    for (Index j = 0; j < n; ++j) {
        lp.obj.push_back(-1.0 - static_cast<Real>(next(5)));
        lp.col_lower.push_back(0.0);
        lp.col_upper.push_back(5.0);
        lp.col_type.push_back(VarType::Continuous);
        lp.col_names.push_back("x" + std::to_string(j));
    }
    lp.num_rows = m;
    std::vector<Triplet> trips;
    for (Index i = 0; i < m; ++i) {
        for (Index j = 0; j < n; ++j) {
            if (next(3) == 0) {
                continue;
            }
            const Real a = static_cast<Real>(next(6)) - 2.0;
            if (a != 0.0) {
                trips.push_back({i, j, a});
            }
        }
        lp.row_lower.push_back(-kInf);
        lp.row_upper.push_back(3.0 + static_cast<Real>(next(10)));
        lp.row_names.push_back("r" + std::to_string(i));
    }
    lp.matrix = SparseMatrix(m, n, std::move(trips));
    return lp;
}

TEST_CASE("Incremental: keep_basis_on_row_removal stays warm over repeated removals",
          "[incremental]") {
    for (std::uint32_t seed = 1; seed <= 20; ++seed) {
        CAPTURE(seed);
        LpProblem lp = buildSeededLP(seed, 8, 12);
        auto solver = keepBasisSolver(lp, true);
        REQUIRE(solver.solve().status == Status::Optimal);

        // Two rounds: every third row, then the first and last of what is left.
        for (int round = 0; round < 2; ++round) {
            CAPTURE(round);
            std::vector<Index> remove;
            if (round == 0) {
                for (Index i = 0; i < lp.num_rows; i += 3) {
                    remove.push_back(i);
                }
            } else {
                remove = {0, lp.num_rows - 1};
            }
            solver.removeRows(remove);
            lp = withoutRows(lp, remove);
            REQUIRE(solver.numRows() == lp.num_rows);

            auto result = solver.solve();
            REQUIRE(result.status == Status::Optimal);
            CHECK_FALSE(solver.basisStartStats().cold_start);
            CHECK_FALSE(solver.basisStartStats().singular_recovery);
            CHECK_THAT(result.objective, WithinAbs(coldObjective(lp), 1e-6));
        }
    }
}
