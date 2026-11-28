#include "../src/mip/conflict_ray.h"
#include "../src/mip/conflict_watches.h"
#include "mipx/io.h"
#include "mipx/mip_solver.h"
#include "perf/slow_root_set_cover.hpp"
#include "test_paths.hpp"

#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <limits>
#include <utility>

using namespace mipx;
using Catch::Matchers::WithinAbs;
using mipx::test::buildSlowRootSetCover;

// ---------------------------------------------------------------------------
// Helper: build a small MIP for testing
// min -x - 2y  s.t. x + y <= 4, x <= 3, y <= 3, x,y >= 0, x,y integer
// LP relaxation optimal: x=1, y=3, obj=-7
// MIP optimal: x=1, y=3, obj=-7 (same, LP solution is integral)
// ---------------------------------------------------------------------------

static LpProblem buildSimpleMip() {
    LpProblem lp;
    lp.name = "simple_mip";
    lp.sense = Sense::Minimize;
    lp.num_cols = 2;
    lp.obj = {-1.0, -2.0};
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {kInf, kInf};
    lp.col_type = {VarType::Integer, VarType::Integer};
    lp.col_names = {"x", "y"};

    lp.num_rows = 3;
    lp.row_lower = {-kInf, -kInf, -kInf};
    lp.row_upper = {4.0, 3.0, 3.0};
    lp.row_names = {"sum", "ub_x", "ub_y"};

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
// Helper: build a MIP that needs branching
// min -x - 2y  s.t. x + y <= 4.5, x,y >= 0, x,y integer
// LP optimal: x=0, y=4.5, obj=-9
// MIP optimal: x=0, y=4, obj=-8
// (With branching: need to round y down)
// ---------------------------------------------------------------------------

static LpProblem buildBranchingMip() {
    LpProblem lp;
    lp.name = "branching_mip";
    lp.sense = Sense::Minimize;
    lp.num_cols = 2;
    lp.obj = {-1.0, -2.0};
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {kInf, kInf};
    lp.col_type = {VarType::Integer, VarType::Integer};
    lp.col_names = {"x", "y"};

    lp.num_rows = 1;
    lp.row_lower = {-kInf};
    lp.row_upper = {4.5};
    lp.row_names = {"sum"};

    std::vector<Triplet> trips = {
        {0, 0, 1.0},
        {0, 1, 1.0},
    };
    lp.matrix = SparseMatrix(1, 2, std::move(trips));
    return lp;
}

// ---------------------------------------------------------------------------
// Helper: root LP fractional but rounding can provide an incumbent.
// min -x  s.t. x <= 0.49, x >= 0, x integer
// ---------------------------------------------------------------------------

static LpProblem buildRootRoundingMip() {
    LpProblem lp;
    lp.name = "root_rounding_mip";
    lp.sense = Sense::Minimize;
    lp.num_cols = 1;
    lp.obj = {-1.0};
    lp.col_lower = {0.0};
    lp.col_upper = {kInf};
    lp.col_type = {VarType::Integer};
    lp.col_names = {"x"};

    lp.num_rows = 1;
    lp.row_lower = {-kInf};
    lp.row_upper = {0.49};
    lp.row_names = {"ub"};

    std::vector<Triplet> trips = {
        {0, 0, 1.0},
    };
    lp.matrix = SparseMatrix(1, 1, std::move(trips));
    return lp;
}

// ---------------------------------------------------------------------------
// Helper: root LP fractional where plain rounding fails; FP/RENS can still
// provide a feasible incumbent.
// min -x  s.t. x <= 4.6, x >= 0, x integer
// ---------------------------------------------------------------------------

static LpProblem buildRootFractionalHeuristicMip() {
    LpProblem lp;
    lp.name = "root_fractional_heur_mip";
    lp.sense = Sense::Minimize;
    lp.num_cols = 1;
    lp.obj = {-1.0};
    lp.col_lower = {0.0};
    lp.col_upper = {10.0};
    lp.col_type = {VarType::Integer};
    lp.col_names = {"x"};

    lp.num_rows = 1;
    lp.row_lower = {-kInf};
    lp.row_upper = {4.6};
    lp.row_names = {"ub_x"};

    std::vector<Triplet> trips = {
        {0, 0, 1.0},
    };
    lp.matrix = SparseMatrix(1, 1, std::move(trips));
    return lp;
}

// ---------------------------------------------------------------------------
// Helper: LP-light Scylla-FPR probe model with many binaries.
// min -sum_j w_j x_j
// s.t. sum_j x_j <= 5.5, x_j binary
// LP root is fractional, while integer-feasible incumbents are easy to verify.
// ---------------------------------------------------------------------------

static LpProblem buildLpLightScyllaProbeMip() {
    constexpr Index n = 12;
    LpProblem lp;
    lp.name = "lplight_scylla_probe_mip";
    lp.sense = Sense::Minimize;
    lp.num_cols = n;
    lp.obj = {-12.0, -11.0, -10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0};
    lp.col_lower.assign(n, 0.0);
    lp.col_upper.assign(n, 1.0);
    lp.col_type.assign(n, VarType::Binary);
    lp.col_names = {"x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12"};

    lp.num_rows = 1;
    lp.row_lower = {-kInf};
    lp.row_upper = {5.5};
    lp.row_names = {"cap"};

    std::vector<Triplet> trips;
    trips.reserve(n);
    for (Index j = 0; j < n; ++j) {
        trips.push_back({0, j, 1.0});
    }
    lp.matrix = SparseMatrix(1, n, std::move(trips));
    return lp;
}

// ---------------------------------------------------------------------------
// Helper: LP-light integer-repair probe
// min -x
// s.t. x <= 3.7, x integer
// LP root: x=3.7, while integer rounding can violate and requires +-1 repair.
// ---------------------------------------------------------------------------

static LpProblem buildLpLightIntegerRepairProbeMip() {
    LpProblem lp;
    lp.name = "lplight_integer_repair_probe_mip";
    lp.sense = Sense::Minimize;
    lp.num_cols = 1;
    lp.obj = {-1.0};
    lp.col_lower = {0.0};
    lp.col_upper = {10.0};
    lp.col_type = {VarType::Integer};
    lp.col_names = {"x"};

    lp.num_rows = 1;
    lp.row_lower = {-kInf};
    lp.row_upper = {3.7};
    lp.row_names = {"ub_x"};

    std::vector<Triplet> trips = {
        {0, 0, 1.0},
    };
    lp.matrix = SparseMatrix(1, 1, std::move(trips));
    return lp;
}

// ---------------------------------------------------------------------------
// Helper: build infeasible MIP
// x >= 5, x <= 3, x integer
// ---------------------------------------------------------------------------

static LpProblem buildInfeasibleMip() {
    LpProblem lp;
    lp.name = "infeasible_mip";
    lp.sense = Sense::Minimize;
    lp.num_cols = 1;
    lp.obj = {1.0};
    lp.col_lower = {0.0};
    lp.col_upper = {kInf};
    lp.col_type = {VarType::Integer};
    lp.col_names = {"x"};

    // x >= 5 AND x <= 3 via constraints.
    lp.num_rows = 2;
    lp.row_lower = {5.0, -kInf};
    lp.row_upper = {kInf, 3.0};
    lp.row_names = {"lb", "ub"};

    std::vector<Triplet> trips = {
        {0, 0, 1.0},
        {1, 0, 1.0},
    };
    lp.matrix = SparseMatrix(2, 1, std::move(trips));
    return lp;
}

// ---------------------------------------------------------------------------
// Helper: Knapsack MIP
// max 5x1 + 4x2 + 3x3  s.t. 2x1 + 3x2 + 2x3 <= 5, x binary
// = min -5x1 -4x2 -3x3
// Optimal: x1=1, x2=1, x3=0, obj=-9
// (LP relaxation: x1=1, x2=1, x3=0 is already integral for this problem)
// Try another: max 6x1 + 5x2 + 4x3 s.t. 3x1 + 2x2 + 2x3 <= 5, x binary
// LP optimal: x1=1, x2=1, x3=0 gives 3+2=5<=5, obj=11
//             x1=1, x2=0, x3=1 gives 3+2=5<=5, obj=10
// Opt: x1=1, x2=1, x3=0, obj=-11
// ---------------------------------------------------------------------------

static LpProblem buildKnapsackMip() {
    LpProblem lp;
    lp.name = "knapsack";
    lp.sense = Sense::Minimize;
    lp.num_cols = 3;
    lp.obj = {-6.0, -5.0, -4.0};
    lp.col_lower = {0.0, 0.0, 0.0};
    lp.col_upper = {1.0, 1.0, 1.0};
    lp.col_type = {VarType::Binary, VarType::Binary, VarType::Binary};
    lp.col_names = {"x1", "x2", "x3"};

    lp.num_rows = 1;
    lp.row_lower = {-kInf};
    lp.row_upper = {5.0};
    lp.row_names = {"capacity"};

    std::vector<Triplet> trips = {
        {0, 0, 3.0},
        {0, 1, 2.0},
        {0, 2, 2.0},
    };
    lp.matrix = SparseMatrix(1, 3, std::move(trips));
    return lp;
}

// ---------------------------------------------------------------------------
// Helper: fractional nodes beyond root to exercise in-tree cut management.
// min -5x1 -4x2 -3x3  s.t. x1 + x2 + x3 <= 2.5, x integer >= 0
// ---------------------------------------------------------------------------

static LpProblem buildTreeCutMip() {
    LpProblem lp;
    lp.name = "tree_cut_mip";
    lp.sense = Sense::Minimize;
    lp.num_cols = 3;
    lp.obj = {-5.0, -4.0, -3.0};
    lp.col_lower = {0.0, 0.0, 0.0};
    lp.col_upper = {kInf, kInf, kInf};
    lp.col_type = {VarType::Integer, VarType::Integer, VarType::Integer};
    lp.col_names = {"x1", "x2", "x3"};

    lp.num_rows = 1;
    lp.row_lower = {-kInf};
    lp.row_upper = {2.5};
    lp.row_names = {"cap"};

    std::vector<Triplet> trips = {
        {0, 0, 1.0},
        {0, 1, 1.0},
        {0, 2, 1.0},
    };
    lp.matrix = SparseMatrix(1, 3, std::move(trips));
    return lp;
}

// ---------------------------------------------------------------------------
// Helper: regression for presolve objective offset handling.
// min 100*x + y
// s.t. 2 <= y + z <= 100
//      x fixed at 1 (binary), 0 <= y <= 10 (integer), 0 <= z <= 10 (continuous)
// Optimal objective is 100 (x=1, y=0, z=2).
// ---------------------------------------------------------------------------

static LpProblem buildPresolveOffsetRegressionMip() {
    LpProblem lp;
    lp.name = "presolve_offset_regression";
    lp.sense = Sense::Minimize;
    lp.num_cols = 3;
    lp.obj = {100.0, 1.0, 0.0};
    lp.col_lower = {1.0, 0.0, 0.0};
    lp.col_upper = {1.0, 10.0, 10.0};
    lp.col_type = {VarType::Binary, VarType::Integer, VarType::Continuous};
    lp.col_names = {"x", "y", "z"};

    lp.num_rows = 1;
    lp.row_lower = {2.0};
    lp.row_upper = {100.0};
    lp.row_names = {"balance"};

    std::vector<Triplet> trips = {
        {0, 1, 1.0},
        {0, 2, 1.0},
    };
    lp.matrix = SparseMatrix(1, 3, std::move(trips));
    return lp;
}

// ---------------------------------------------------------------------------
// Helper: infeasible binary model with fractional root to exercise conflict
// learning and no-good reuse in the tree.
// x + y = 1.5, x,y binary
// ---------------------------------------------------------------------------

static LpProblem buildConflictLearningMip() {
    LpProblem lp;
    lp.name = "conflict_learning_mip";
    lp.sense = Sense::Minimize;
    lp.num_cols = 2;
    lp.obj = {0.0, 0.0};
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {1.0, 1.0};
    lp.col_type = {VarType::Binary, VarType::Binary};
    lp.col_names = {"x", "y"};

    lp.num_rows = 1;
    lp.row_lower = {1.5};
    lp.row_upper = {1.5};
    lp.row_names = {"eq"};

    std::vector<Triplet> trips = {
        {0, 0, 1.0},
        {0, 1, 1.0},
    };
    lp.matrix = SparseMatrix(1, 2, std::move(trips));
    return lp;
}

// ---------------------------------------------------------------------------
// Helper: infeasible parity-like binary model to trigger search stagnation.
// sum x_i = 2.5, x_i binary.
// ---------------------------------------------------------------------------

static LpProblem buildSearchStagnationMip() {
    constexpr Index n = 10;
    LpProblem lp;
    lp.name = "search_stagnation_mip";
    lp.sense = Sense::Minimize;
    lp.num_cols = n;
    lp.obj.assign(n, 0.0);
    lp.col_lower.assign(n, 0.0);
    lp.col_upper.assign(n, 1.0);
    lp.col_type.assign(n, VarType::Binary);
    lp.col_names = {"x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10"};

    lp.num_rows = 1;
    lp.row_lower = {4.5};
    lp.row_upper = {4.5};
    lp.row_names = {"eq"};

    std::vector<Triplet> trips;
    trips.reserve(n);
    for (Index j = 0; j < n; ++j) {
        trips.push_back({0, j, 1.0});
    }
    lp.matrix = SparseMatrix(1, n, std::move(trips));
    return lp;
}

static LpProblem buildSearchStagnationMipWithEscape() {
    // The parity subproblem is infeasible when escape = 0. At escape = 1,
    // four chosen binaries suffice, with objective one.
    auto lp = buildSearchStagnationMip();
    const Index escape = lp.num_cols++;
    lp.obj.push_back(1.0);
    lp.col_lower.push_back(0.0);
    lp.col_upper.push_back(1.0);
    lp.col_type.push_back(VarType::Binary);
    lp.col_names.emplace_back("escape");
    std::vector<Triplet> row;
    for (Index col = 0; col < escape; ++col) {
        row.push_back({0, col, 1.0});
    }
    row.push_back({0, escape, 0.5});
    lp.matrix = SparseMatrix(1, lp.num_cols, std::move(row));
    return lp;
}

// ---------------------------------------------------------------------------
// Helper: bounded general-integer knapsack whose reduced costs are exploitable
// once an incumbent exists, so that both root global and node-local
// reduced-cost fixing fire. Single constraint, so the true optimum is
// available from a small dynamic program (see rcKnapsackOptimum below).
// ---------------------------------------------------------------------------

static constexpr Index kRcKnapsackCols = 10;
static constexpr Real kRcKnapsackCapacity = 82.0;
static constexpr Real kRcKnapsackVarUpper = 5.0;
static constexpr Real kRcKnapsackObj[kRcKnapsackCols] = {-15.0, -14.0, -17.0, -17.0, -2.0,
                                                         -17.0, -14.0, -8.0,  -14.0, -13.0};
static constexpr Real kRcKnapsackWeight[kRcKnapsackCols] = {5.0, 5.0, 4.0, 9.0, 7.0,
                                                            5.0, 3.0, 5.0, 3.0, 1.0};

static LpProblem buildRcFixingKnapsackMip() {
    LpProblem lp;
    lp.name = "rc_fixing_knapsack";
    lp.sense = Sense::Minimize;
    lp.num_cols = kRcKnapsackCols;
    lp.obj.assign(kRcKnapsackObj, kRcKnapsackObj + kRcKnapsackCols);
    lp.col_lower.assign(kRcKnapsackCols, 0.0);
    lp.col_upper.assign(kRcKnapsackCols, kRcKnapsackVarUpper);
    lp.col_type.assign(kRcKnapsackCols, VarType::Integer);

    lp.num_rows = 1;
    lp.row_lower = {-kInf};
    lp.row_upper = {kRcKnapsackCapacity};
    lp.row_names = {"capacity"};

    std::vector<Triplet> trips;
    trips.reserve(kRcKnapsackCols);
    for (Index j = 0; j < kRcKnapsackCols; ++j) {
        trips.push_back({0, j, kRcKnapsackWeight[j]});
    }
    lp.matrix = SparseMatrix(1, kRcKnapsackCols, std::move(trips));
    return lp;
}

// Independent ground truth for buildRcFixingKnapsackMip: bounded knapsack DP.
static Real rcKnapsackOptimum() {
    const int cap = static_cast<int>(kRcKnapsackCapacity);
    const int mult = static_cast<int>(kRcKnapsackVarUpper);
    std::vector<Real> dp(static_cast<std::size_t>(cap) + 1, 0.0);
    for (Index j = 0; j < kRcKnapsackCols; ++j) {
        const int w = static_cast<int>(kRcKnapsackWeight[j]);
        const Real value = -kRcKnapsackObj[j];
        std::vector<Real> next = dp;
        for (int c = 0; c <= cap; ++c) {
            for (int k = 1; k <= mult; ++k) {
                const int used = k * w;
                if (used > c) {
                    break;
                }
                next[static_cast<std::size_t>(c)] =
                    std::max(next[static_cast<std::size_t>(c)],
                             dp[static_cast<std::size_t>(c - used)] + static_cast<Real>(k) * value);
            }
        }
        dp = std::move(next);
    }
    return -dp[static_cast<std::size_t>(cap)];
}

// ---------------------------------------------------------------------------
// Helper: two-constraint bounded integer program that keeps enough fractional
// variables at shallow depths for the in-tree presolve gates to open, so the
// reduced-cost pass inside the tree-presolve block is exercised too.
// ---------------------------------------------------------------------------

static LpProblem buildRcFixingTreePresolveMip() {
    constexpr Index n = 8;
    LpProblem lp;
    lp.name = "rc_fixing_tree_presolve";
    lp.sense = Sense::Minimize;
    lp.num_cols = n;
    lp.obj = {-17.0, -14.0, -13.0, -11.0, -9.0, -7.0, -5.0, -3.0};
    lp.col_lower.assign(n, 0.0);
    lp.col_upper.assign(n, 8.0);
    lp.col_type.assign(n, VarType::Integer);

    lp.num_rows = 2;
    lp.row_lower = {-kInf, -kInf};
    lp.row_upper = {37.0, 29.0};
    lp.row_names = {"c1", "c2"};

    const Real a[n] = {7.0, 6.0, 6.0, 5.0, 4.0, 3.0, 3.0, 2.0};
    const Real d[n] = {3.0, 5.0, 4.0, 6.0, 2.0, 5.0, 4.0, 3.0};
    std::vector<Triplet> trips;
    trips.reserve(2 * n);
    for (Index j = 0; j < n; ++j) {
        trips.push_back({0, j, a[j]});
        trips.push_back({1, j, d[j]});
    }
    lp.matrix = SparseMatrix(2, n, std::move(trips));
    return lp;
}

// ---------------------------------------------------------------------------
// Helper: symmetric binary pair so symmetry cuts should be generated/applied.
// ---------------------------------------------------------------------------

static LpProblem buildSymmetryProbeMip() {
    LpProblem lp;
    lp.name = "symmetry_probe_mip";
    lp.sense = Sense::Minimize;
    lp.num_cols = 2;
    lp.obj = {-1.0, -1.0};
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {1.0, 1.0};
    lp.col_type = {VarType::Binary, VarType::Binary};
    lp.col_names = {"x0", "x1"};

    lp.num_rows = 1;
    lp.row_lower = {-kInf};
    lp.row_upper = {1.0};
    lp.row_names = {"sum"};

    std::vector<Triplet> trips = {
        {0, 0, 1.0},
        {0, 1, 1.0},
    };
    lp.matrix = SparseMatrix(1, 2, std::move(trips));
    return lp;
}

// ---------------------------------------------------------------------------
// Helpers: the two asymmetric models from issue #185. Both were reported as
// "symmetry removes the optimum", but neither has any orbit: toggling symmetry
// only shifted the work-unit budget, which changed how many root heuristics
// ran and therefore whether root reduced-cost fixing had an incumbent to work
// with. That fixing then read the LP's cached duals, which the heuristic
// portfolio had left pointing at one of its own subproblems.
// ---------------------------------------------------------------------------

// min 4x0 + 0x1 + 3x2 - 4x3 - 4x4 + 2x5 - 4
// R0: -3x0 + 5x1 - 4x2 + 2x3 + 5x4 - 2x5 <= 20
// R1: -4x0 - 5x1               + 5x3 - x5 <=  9
// Optimum by enumeration: -23 at x = (0, 1, 1, 3, 3, 1).
static LpProblem buildIssue185ModelA() {
    LpProblem lp;
    lp.name = "issue185_a";
    lp.sense = Sense::Minimize;
    lp.num_cols = 6;
    lp.obj = {4.0, 0.0, 3.0, -4.0, -4.0, 2.0};
    lp.obj_offset = -4.0;
    lp.col_lower = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    lp.col_upper = {1.0, 1.0, 3.0, 3.0, 3.0, 1.0};
    lp.col_type = {VarType::Integer, VarType::Binary,  VarType::Integer,
                   VarType::Integer, VarType::Integer, VarType::Binary};
    lp.col_names = {"x0", "x1", "x2", "x3", "x4", "x5"};

    lp.num_rows = 2;
    lp.row_lower = {-kInf, -kInf};
    lp.row_upper = {20.0, 9.0};
    lp.row_names = {"R0", "R1"};

    std::vector<Triplet> trips = {
        {0, 0, -3.0}, {0, 1, 5.0},  {0, 2, -4.0}, {0, 3, 2.0}, {0, 4, 5.0},
        {0, 5, -2.0}, {1, 0, -4.0}, {1, 1, -5.0}, {1, 3, 5.0}, {1, 5, -1.0},
    };
    lp.matrix = SparseMatrix(2, 6, std::move(trips));
    return lp;
}

// min -16x0 + 21x1 + 14x2 + 18x3 - x4 - 8x5
// R0: 2x0 + 6x1 + x2 + 3x4 in [27, 28]
// R1: 2x0 - 3x1 + 3x4 + 2x5 <= 8
// R2: -5x0 + 8x1 + 9x2 + 5x3 + 9x4 - 5x5 <= 57
// Optimum by enumeration: -11 at x = (4, 3, 1, 0, 0, 3).
static LpProblem buildIssue185ModelB() {
    LpProblem lp;
    lp.name = "issue185_b";
    lp.sense = Sense::Minimize;
    lp.num_cols = 6;
    lp.obj = {-16.0, 21.0, 14.0, 18.0, -1.0, -8.0};
    lp.col_lower = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    lp.col_upper = {4.0, 4.0, 2.0, 3.0, 2.0, 3.0};
    lp.col_type = std::vector<VarType>(6, VarType::Integer);
    lp.col_names = {"x0", "x1", "x2", "x3", "x4", "x5"};

    lp.num_rows = 3;
    lp.row_lower = {27.0, -kInf, -kInf};
    lp.row_upper = {28.0, 8.0, 57.0};
    lp.row_names = {"R0", "R1", "R2"};

    std::vector<Triplet> trips = {
        {0, 0, 2.0},  {0, 1, 6.0}, {0, 2, 1.0}, {0, 4, 3.0},  {1, 0, 2.0},
        {1, 1, -3.0}, {1, 4, 3.0}, {1, 5, 2.0}, {2, 0, -5.0}, {2, 1, 8.0},
        {2, 2, 9.0},  {2, 3, 5.0}, {2, 4, 9.0}, {2, 5, -5.0},
    };
    lp.matrix = SparseMatrix(3, 6, std::move(trips));
    return lp;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST_CASE("MipSolver: LP with no integers", "[mip]") {
    auto lp = buildSimpleMip();
    // Make it a pure LP.
    lp.col_type = {VarType::Continuous, VarType::Continuous};

    MipSolver solver;
    solver.setVerbose(false);
    solver.load(lp);
    auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(-7.0, 1e-6));
    CHECK(result.nodes == 0);
}

TEST_CASE("MipSolver: integral LP relaxation", "[mip]") {
    auto lp = buildSimpleMip();

    MipSolver solver;
    solver.setVerbose(false);
    solver.load(lp);
    auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(-7.0, 1e-6));
    CHECK(result.nodes == 1);  // Root is already integral.
}

TEST_CASE("MipSolver: needs branching", "[mip]") {
    auto lp = buildBranchingMip();

    MipSolver solver;
    solver.setVerbose(false);
    solver.load(lp);
    auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(-8.0, 1e-6));
    // With cutting planes enabled, the problem may be solved at the root
    // (cuts can close the integrality gap), so nodes >= 1.
    CHECK(result.nodes >= 1);
    REQUIRE(result.solution.size() == 2);
    // x=0, y=4 or x=4, y=0 or other combos with obj=-8.
    // Valid: any (x,y) with x+y<=4, x,y>=0, integer, -x-2y=-8
    // y=4, x=0 gives -8. y=3, x=2 gives -8.
    Real obj = -result.solution[0] - 2.0 * result.solution[1];
    CHECK_THAT(obj, WithinAbs(-8.0, 1e-6));
}

TEST_CASE("MipSolver: reliability branching collects strong-branch telemetry", "[mip][branching]") {
    auto lp = buildBranchingMip();

    MipSolver solver;
    solver.setVerbose(false);
    solver.setCutsEnabled(false);
    solver.load(lp);
    auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    const auto& stats = solver.getBranchingStats();
    CHECK(stats.selections >= 1);
    CHECK(stats.strong_branch_calls >= 1);
    CHECK(stats.strong_branch_probes >= 2);
    CHECK(stats.strong_branch_probe_iters >= 0);
    CHECK(stats.strong_branch_probe_work_units >= 0.0);
}

TEST_CASE("MipSolver: the tree feeds pseudocosts from solved children", "[mip][branching]") {
    // Issue #199: before, only strong-branch probes updated pseudocosts.
    auto lp = buildRcFixingKnapsackMip();

    MipSolver solver;
    solver.setVerbose(false);
    solver.setCutsEnabled(false);
    solver.setPresolve(false);
    solver.load(lp);
    const auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(rcKnapsackOptimum(), 1e-6));
    REQUIRE(result.nodes > 5);
    const auto& stats = solver.getBranchingStats();
    CHECK(stats.pseudocost_updates_tree > 0);
    CHECK(stats.pseudocost_updates_strong > 0);
}

TEST_CASE("MipSolver: without strong branching, pseudocosts come from the tree alone",
          "[mip][branching]") {
    // Issue #199, criterion 2: no strong-branch probe runs, so every
    // pseudocost sample the search collects is a tree update.
    auto lp = buildRcFixingKnapsackMip();

    MipSolver solver;
    solver.setVerbose(false);
    solver.setCutsEnabled(false);
    solver.setPresolve(false);
    solver.setStrongBranchingEnabled(false);
    solver.load(lp);
    const auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(rcKnapsackOptimum(), 1e-6));
    const auto& stats = solver.getBranchingStats();
    CHECK(stats.strong_branch_probes == 0);
    CHECK(stats.pseudocost_updates_strong == 0);
    CHECK(stats.pseudocost_updates_tree > 0);

    // Turning it back on takes effect on the next solve.
    solver.setStrongBranchingEnabled(true);
    solver.load(lp);
    const auto again = solver.solve();
    REQUIRE(again.status == Status::Optimal);
    CHECK(solver.getBranchingStats().strong_branch_probes > 0);
}

TEST_CASE("MipSolver: tree pseudocost updates can be switched off", "[mip][branching]") {
    // Issue #199 landed on by default; the switch restores the search where
    // only strong-branch probes feed pseudocosts, for A/B benchmarking.
    auto lp = buildRcFixingKnapsackMip();
    for (const Int threads : {1, 2}) {
        INFO("threads " << threads);
        MipSolver solver;
        solver.setVerbose(false);
        solver.setCutsEnabled(false);
        solver.setPresolve(false);
        solver.setNumThreads(threads);
        solver.setParallelMode(ParallelMode::Deterministic);
        solver.setTreePseudocostsEnabled(false);
        solver.load(lp);
        const auto result = solver.solve();

        REQUIRE(result.status == Status::Optimal);
        CHECK_THAT(result.objective, WithinAbs(rcKnapsackOptimum(), 1e-6));
        REQUIRE(result.nodes > 5);
        const auto& stats = solver.getBranchingStats();
        CHECK(stats.pseudocost_updates_tree == 0);
        CHECK(stats.pseudocost_tree_skipped == 0);
        CHECK(stats.pseudocost_updates_strong > 0);
    }
}

TEST_CASE("MipSolver: infeasible", "[mip]") {
    auto lp = buildInfeasibleMip();

    MipSolver solver;
    solver.setVerbose(false);
    solver.load(lp);
    auto result = solver.solve();

    CHECK(result.status == Status::Infeasible);
}

TEST_CASE("MipSolver: knapsack", "[mip]") {
    auto lp = buildKnapsackMip();

    MipSolver solver;
    solver.setVerbose(false);
    solver.setNodeLimit(1000);
    solver.load(lp);
    auto result = solver.solve();

    REQUIRE((result.status == Status::Optimal || result.status == Status::NodeLimit));
    if (result.status == Status::Optimal) {
        CHECK_THAT(result.objective, WithinAbs(-11.0, 1e-6));
    }
}

TEST_CASE("MipSolver: node limit", "[mip]") {
    auto lp = buildBranchingMip();

    MipSolver solver;
    solver.setVerbose(false);
    solver.setNodeLimit(1);  // Only solve root.
    solver.load(lp);
    auto result = solver.solve();

    // With node limit 1, we process root but can't explore children.
    CHECK((result.status == Status::NodeLimit || result.status == Status::Optimal));
}

// An incumbent, when one was found, covers every row of the set cover.
static void checkSetCoverIncumbent(const LpProblem& lp, const std::vector<Real>& solution) {
    if (solution.empty()) {
        return;
    }
    for (Index i = 0; i < lp.num_rows; ++i) {
        Real activity = 0.0;
        const auto row = lp.matrix.row(i);
        for (Index p = 0; p < row.size(); ++p) {
            activity += row.values[p] * solution[row.indices[p]];
        }
        CHECK(activity >= 1.0 - 1e-6);
    }
}

// #336: the time limit holds through the root stage. The root LP of this
// model does not finish in 0.2 s, so a 0.2 s limit stops it there,
// and the solve returns TimeLimit within the stated overrun. Wall-clock, so
// [benchmark] keeps it out of the default suite (solver-regression label).
TEST_CASE("MipSolver: time limit stops a root LP that does not finish in time",
          "[mip][benchmark][time_limit]") {
    constexpr double kLimit = 0.2;
    constexpr double kAllowedOverrun = 0.5;

    // The barrier, alone or as a race arm, stops inside its factorization on
    // its own max_solve_seconds (#275), but the steps before its first IPM
    // iteration have no check: on the large model (~40400 standard-form rows)
    // the dense Cholesky fills an m x m matrix of ~13 GB and the sparse one's
    // ordering takes ~7 s. Those two policies get a 10x narrower model, where
    // the time goes into the factorization (#345, #346).
    const auto policy =
        GENERATE(RootLpPolicy::DualDefault, RootLpPolicy::PdlpRoot, RootLpPolicy::BarrierRoot,
                 RootLpPolicy::ConcurrentRootExperimental);
    CAPTURE(static_cast<int>(policy));
    const bool runs_barrier =
        policy == RootLpPolicy::BarrierRoot || policy == RootLpPolicy::ConcurrentRootExperimental;
    const auto lp = buildSlowRootSetCover(400, runs_barrier ? 4000 : 40000, 8);
    MipSolver solver;
    solver.setVerbose(false);
    solver.setNumThreads(1);
    solver.setRootLpPolicy(policy);
    solver.setTimeLimit(kLimit);
    solver.load(lp);
    const auto t0 = std::chrono::steady_clock::now();
    const auto result = solver.solve();
    const double wall =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    CHECK(result.status == Status::TimeLimit);
    CHECK(result.nodes <= 1);
    // A stopped root LP is no bound (minimization: -inf).
    CHECK(result.best_bound == -kInf);
    CHECK(wall <= kLimit + kAllowedOverrun);
    CHECK(result.time_seconds <= kLimit + kAllowedOverrun);
    checkSetCoverIncumbent(lp, result.solution);
}

// #351: the time limit holds through the tree. On this model the root
// finishes well inside the limit and the tree's node and strong-branching
// LPs run past it: before the tree's LPs got the time left, the solve
// overran 8 s by 0.38 s here (32-core box, one thread), after by 0.01 s.
// Wall-clock, so [benchmark] keeps it out of the default suite.
TEST_CASE("MipSolver: time limit stops the tree's LPs", "[mip][benchmark][time_limit]") {
    constexpr double kLimit = 8.0;
    constexpr double kAllowedOverrun = 0.2;
    const auto lp = buildSlowRootSetCover(400, 40000, 8);
    MipSolver solver;
    solver.setVerbose(false);
    solver.setNumThreads(1);
    solver.setTimeLimit(kLimit);
    solver.load(lp);
    const auto t0 = std::chrono::steady_clock::now();
    const auto result = solver.solve();
    const double wall =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    CHECK(result.status == Status::TimeLimit);
    CHECK(result.nodes > 1);  // the limit fell in the tree, not the root
    CHECK(std::isfinite(result.best_bound));
    CHECK(wall <= kLimit + kAllowedOverrun);
    CHECK(result.time_seconds <= kLimit + kAllowedOverrun);
    checkSetCoverIncumbent(lp, result.solution);
}

// #336, deterministic: with no time at all, every root policy, the race
// included, returns TimeLimit with no bound and no node beyond the root.
TEST_CASE("MipSolver: a zero time limit stops in the root with no bound", "[mip][time_limit]") {
    const auto policy =
        GENERATE(RootLpPolicy::DualDefault, RootLpPolicy::PdlpRoot, RootLpPolicy::BarrierRoot,
                 RootLpPolicy::ConcurrentRootExperimental);
    CAPTURE(static_cast<int>(policy));
    const auto lp = buildSlowRootSetCover(100, 1000, 4);
    MipSolver solver;
    solver.setVerbose(false);
    solver.setNumThreads(1);
    solver.setRootLpPolicy(policy);
    solver.setTimeLimit(0.0);
    solver.load(lp);
    const auto result = solver.solve();

    CHECK(result.status == Status::TimeLimit);
    CHECK(result.nodes <= 1);
    CHECK(result.best_bound == -kInf);
    checkSetCoverIncumbent(lp, result.solution);
}

TEST_CASE("MipSolver: a node dropped without an LP verdict leaves the search unproven",
          "[mip][issue283]") {
    // Presolve and cuts would round x + y <= 4.5 to the integral optimum at
    // the root; without them the root LP (-9) must branch to reach -8.
    // Each thread mode reaches nodes through its own caller of processNode.
    const Int threads = GENERATE(1, 2);
    const ParallelMode mode = GENERATE(ParallelMode::Deterministic, ParallelMode::Opportunistic);
    CAPTURE(threads, static_cast<int>(mode));
    auto lp = buildBranchingMip();
    auto configure = [&](MipSolver& solver) {
        solver.setVerbose(false);
        solver.setPresolve(false);
        solver.setCutsEnabled(false);
        solver.setNumThreads(threads);
        solver.setParallelMode(mode);
    };

    SECTION("node LPs stopped on their iteration limit are not pruned as proven") {
        MipSolver solver;
        configure(solver);
        // No pivot allowed: every child LP, warm and then cold, ends on
        // IterLimit, so no node below the root gets a verdict.
        solver.setNodeLpIterationLimit(0);
        solver.load(lp);
        const auto result = solver.solve();

        const auto& stats = solver.getLpStats();
        REQUIRE(stats.nodes_dropped_no_verdict > 0);
        CHECK(stats.node_lp_no_verdict >= stats.nodes_dropped_no_verdict);
        CHECK(stats.node_lp_no_verdict_recovered == 0);
        // The dropped subtrees hold the optimum -8 the root bound -9 allows:
        // the result may not claim a proof, and its bound must admit them.
        CHECK(result.status != Status::Optimal);
        CHECK(result.status != Status::Infeasible);
        CHECK(result.best_bound <= -9.0 + 1e-6);
        if (!result.solution.empty()) {
            CHECK(result.objective >= -8.0 - 1e-6);
            CHECK(result.gap > 0.0);
        }
    }

    SECTION("without the limit the same search proves the optimum") {
        MipSolver solver;
        configure(solver);
        solver.load(lp);
        const auto result = solver.solve();

        CHECK(solver.getLpStats().nodes_dropped_no_verdict == 0);
        REQUIRE(result.status == Status::Optimal);
        CHECK_THAT(result.objective, WithinAbs(-8.0, 1e-6));
    }
}

TEST_CASE("MipSolver: root heuristics provide incumbent before branching", "[mip]") {
    auto lp = buildRootRoundingMip();

    MipSolver solver;
    solver.setVerbose(false);
    solver.setCutsEnabled(false);
    solver.setNodeLimit(1);  // root only
    solver.load(lp);
    auto result = solver.solve();

    CHECK((result.status == Status::NodeLimit || result.status == Status::Optimal));
    REQUIRE(!result.solution.empty());
    CHECK_THAT(result.objective, WithinAbs(0.0, 1e-6));
    CHECK_THAT(result.solution[0], WithinAbs(0.0, 1e-6));
}

TEST_CASE("MipSolver: LP-based root portfolio can bootstrap incumbent", "[mip]") {
    auto lp = buildRootFractionalHeuristicMip();

    MipSolver solver;
    solver.setVerbose(false);
    solver.setCutsEnabled(false);
    solver.setNodeLimit(1);  // root only
    solver.load(lp);
    auto result = solver.solve();

    CHECK((result.status == Status::NodeLimit || result.status == Status::Optimal));
    REQUIRE(!result.solution.empty());
    CHECK(result.solution[0] >= 0.0);
    CHECK(result.solution[0] <= 4.6 + 1e-6);
}

TEST_CASE("MipSolver: in-tree cut telemetry is populated", "[mip][cuts]") {
    auto lp = buildTreeCutMip();

    MipSolver solver;
    solver.setVerbose(false);
    solver.setCutsEnabled(true);
    solver.setMaxCutRounds(0);  // emphasize in-tree cuts over root rounds
    solver.load(lp);
    auto result = solver.solve();

    CHECK((result.status == Status::Optimal || result.status == Status::NodeLimit ||
           result.status == Status::TimeLimit));
    const auto& cut_stats = solver.getCutStats();
    CHECK(cut_stats.tree_nodes_with_cuts + cut_stats.tree_nodes_skipped >= 1);
    CHECK(cut_stats.tree_rounds >= 0);
}

TEST_CASE("MipSolver: symmetry cuts are applied when presolve is off", "[mip][symmetry]") {
    auto lp = buildSymmetryProbeMip();

    MipSolver without_symmetry;
    without_symmetry.setVerbose(false);
    without_symmetry.setCutsEnabled(false);
    without_symmetry.setPresolve(false);
    without_symmetry.setSymmetryEnabled(false);
    without_symmetry.load(lp);
    const auto off = without_symmetry.solve();

    MipSolver with_symmetry;
    with_symmetry.setVerbose(false);
    with_symmetry.setCutsEnabled(false);
    with_symmetry.setPresolve(false);
    with_symmetry.setSymmetryEnabled(true);
    with_symmetry.load(lp);
    const auto on = with_symmetry.solve();

    CHECK((off.status == Status::Optimal || off.status == Status::NodeLimit ||
           off.status == Status::TimeLimit));
    CHECK((on.status == Status::Optimal || on.status == Status::NodeLimit ||
           on.status == Status::TimeLimit));
    CHECK_THAT(on.objective, WithinAbs(off.objective, 1e-9));

    const auto& off_stats = without_symmetry.getSymmetryStats();
    CHECK(off_stats.orbits == 0);
    CHECK(off_stats.cuts_added == 0);
    CHECK_FALSE(off_stats.cuts_applied);

    const auto& on_stats = with_symmetry.getSymmetryStats();
    CHECK(on_stats.orbits == 1);
    CHECK(on_stats.cuts_added == 1);
    CHECK(on_stats.cuts_applied);
    CHECK(on_stats.detect_work_units > 0.0);
    CHECK(on_stats.cut_work_units > 0.0);
}

TEST_CASE("MipSolver: issue 185 model A keeps the optimum at defaults", "[mip][symmetry]") {
    const auto lp = buildIssue185ModelA();

    MipSolver solver;
    solver.setVerbose(false);
    solver.load(lp);
    const auto on = solver.solve();

    REQUIRE(on.status == Status::Optimal);
    CHECK_THAT(on.objective, WithinAbs(-23.0, 1e-6));

    MipSolver without_symmetry;
    without_symmetry.setVerbose(false);
    without_symmetry.setSymmetryEnabled(false);
    without_symmetry.load(lp);
    const auto off = without_symmetry.solve();

    REQUIRE(off.status == Status::Optimal);
    CHECK_THAT(off.objective, WithinAbs(on.objective, 1e-6));
}

TEST_CASE("MipSolver: issue 185 model B keeps the optimum at defaults", "[mip][symmetry]") {
    const auto lp = buildIssue185ModelB();

    MipSolver solver;
    solver.setVerbose(false);
    solver.load(lp);
    const auto on = solver.solve();

    REQUIRE(on.status == Status::Optimal);
    CHECK_THAT(on.objective, WithinAbs(-11.0, 1e-6));

    // The reported point must satisfy R0..R2, not just carry the right value.
    REQUIRE(on.solution.size() == 6);
    const auto activity = [&](Index row) {
        Real sum = 0.0;
        auto r = lp.matrix.row(row);
        for (Index k = 0; k < r.size(); ++k) {
            sum += r.values[k] * on.solution[static_cast<std::size_t>(r.indices[k])];
        }
        return sum;
    };
    for (Index i = 0; i < lp.num_rows; ++i) {
        const Real act = activity(i);
        CHECK(act >= lp.row_lower[static_cast<std::size_t>(i)] - 1e-6);
        CHECK(act <= lp.row_upper[static_cast<std::size_t>(i)] + 1e-6);
    }

    MipSolver without_symmetry;
    without_symmetry.setVerbose(false);
    without_symmetry.setSymmetryEnabled(false);
    without_symmetry.load(lp);
    const auto off = without_symmetry.solve();

    REQUIRE(off.status == Status::Optimal);
    CHECK_THAT(off.objective, WithinAbs(on.objective, 1e-6));
}

TEST_CASE("MipSolver: MIPLIB gt2", "[mip][miplib]") {
    std::string path = mipx::test::testDataDir() + "/miplib/gt2.mps.gz";
    if (!std::filesystem::exists(path)) {
        SKIP("gt2.mps.gz not found (run download_miplib.sh --small)");
    }

    auto problem = readMps(path);
    REQUIRE(problem.hasIntegers());

    MipSolver solver;
    solver.setVerbose(true);
    solver.setNodeLimit(10000);
    solver.setTimeLimit(60.0);
    solver.load(problem);
    auto result = solver.solve();

    // gt2 optimal: 21166.0
    if (result.status == Status::Optimal) {
        CHECK_THAT(result.objective, WithinAbs(21166.0, 1.0));
    }
    // If not solved to optimality, at least it shouldn't crash.
    CHECK((result.status == Status::Optimal || result.status == Status::NodeLimit ||
           result.status == Status::TimeLimit));
}

TEST_CASE("MipSolver: work units are positive", "[mip][work_units]") {
    auto problem = buildBranchingMip();
    MipSolver solver;
    solver.setVerbose(false);
    solver.load(problem);
    auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    CHECK(result.work_units > 0.0);
}

TEST_CASE("MipSolver: actual row-price work is opt-in for root and node LPs",
          "[mip][work_units][row_price]") {
    // min x + 2y subject to x + y >= 4.5, with nonnegative integers.
    // The slack basis is dual feasible, and branching at x = 4.5 makes
    // the down child's warm start primal infeasible again.
    auto problem = buildBranchingMip();
    problem.obj = {1.0, 2.0};
    problem.row_lower = {4.5};
    problem.row_upper = {kInf};

    const auto [threads, mode] =
        GENERATE(table<Int, ParallelMode>({{1, ParallelMode::Deterministic},
                                           {2, ParallelMode::Deterministic},
                                           {2, ParallelMode::Opportunistic}}));
    CAPTURE(threads, static_cast<int>(mode));
    auto solve = [&](Int node_limit, std::optional<bool> actual_work) {
        MipSolver solver;
        solver.setVerbose(false);
        solver.setPresolve(false);
        solver.setCutsEnabled(false);
        solver.setSymmetryEnabled(false);
        solver.setTreePresolveEnabled(false);
        solver.setObjectivePropagationEnabled(false);
        solver.setObjectiveIntegralityEnabled(false);
        solver.setNumThreads(threads);
        solver.setParallelMode(mode);
        solver.setNodeLimit(node_limit);
        if (actual_work.has_value()) {
            solver.setDualRowPriceActualWork(*actual_work);
        }
        solver.load(problem);
        return solver.solve();
    };

    const auto root_default = solve(1, std::nullopt);
    const auto root_flat = solve(1, false);
    const auto root_actual = solve(1, true);
    REQUIRE(root_flat.status == Status::NodeLimit);
    CHECK(root_flat.nodes == 1);
    CHECK(root_default.work_units == root_flat.work_units);
    CHECK(root_actual.status == root_flat.status);
    CHECK(root_actual.lp_iterations == root_flat.lp_iterations);
    CHECK(root_actual.work_units > root_flat.work_units);

    const auto tree_default = solve(10, std::nullopt);
    const auto tree_flat = solve(10, false);
    const auto tree_actual = solve(10, true);
    REQUIRE(tree_flat.status == Status::Optimal);
    REQUIRE(tree_flat.nodes > root_flat.nodes);
    REQUIRE(tree_flat.lp_iterations > root_flat.lp_iterations);
    CHECK(tree_flat.objective == 5.0);
    CHECK(tree_default.work_units == tree_flat.work_units);
    CHECK(tree_actual.status == tree_flat.status);
    CHECK(tree_actual.objective == tree_flat.objective);
    CHECK(tree_actual.lp_iterations == tree_flat.lp_iterations);
    CHECK(tree_actual.nodes == tree_flat.nodes);
    // For one row the actual charge includes the row visit and slack
    // entry that the flat charge omits. Subtract the root contribution.
    CHECK(tree_actual.work_units - root_actual.work_units >
          tree_flat.work_units - root_flat.work_units);
}

TEST_CASE("MipSolver: exact refinement default remains off and non-regressive",
          "[mip][exact_refinement]") {
    auto lp = buildBranchingMip();

    MipSolver solver_default;
    solver_default.setVerbose(false);
    solver_default.setCutsEnabled(false);
    solver_default.load(lp);
    const auto default_result = solver_default.solve();

    MipSolver solver_explicit_off;
    solver_explicit_off.setVerbose(false);
    solver_explicit_off.setCutsEnabled(false);
    solver_explicit_off.setExactRefinementMode(ExactRefinementMode::Off);
    solver_explicit_off.load(lp);
    const auto off_result = solver_explicit_off.solve();

    REQUIRE(default_result.status == Status::Optimal);
    REQUIRE(off_result.status == Status::Optimal);
    CHECK_THAT(default_result.objective, WithinAbs(off_result.objective, 1e-9));
    CHECK_THAT(default_result.work_units, WithinAbs(off_result.work_units, 1e-9));

    const auto& default_stats = solver_default.getExactRefinementStats();
    CHECK(default_stats.mode == ExactRefinementMode::Off);
    CHECK_FALSE(default_stats.triggered);
    CHECK(default_stats.evaluation_work_units == 0.0);
}

TEST_CASE("MipSolver: exact refinement forced mode is deterministic", "[mip][exact_refinement]") {
    auto lp = buildBranchingMip();

    MipSolver solver_a;
    solver_a.setVerbose(false);
    solver_a.setCutsEnabled(false);
    solver_a.setPresolve(false);
    solver_a.setExactRefinementMode(ExactRefinementMode::On);
    solver_a.setExactRefinementRationalCheck(true);
    solver_a.load(lp);
    const auto a = solver_a.solve();

    MipSolver solver_b;
    solver_b.setVerbose(false);
    solver_b.setCutsEnabled(false);
    solver_b.setPresolve(false);
    solver_b.setExactRefinementMode(ExactRefinementMode::On);
    solver_b.setExactRefinementRationalCheck(true);
    solver_b.load(lp);
    const auto b = solver_b.solve();

    REQUIRE(a.status == Status::Optimal);
    REQUIRE(b.status == Status::Optimal);
    CHECK_THAT(a.objective, WithinAbs(b.objective, 1e-9));
    CHECK_THAT(a.work_units, WithinAbs(b.work_units, 1e-9));

    const auto& sa = solver_a.getExactRefinementStats();
    const auto& sb = solver_b.getExactRefinementStats();
    CHECK(sa.mode == ExactRefinementMode::On);
    CHECK(sb.mode == ExactRefinementMode::On);
    CHECK(sa.rational_verification_enabled);
    CHECK(sb.rational_verification_enabled);
    CHECK(sa.triggered);
    CHECK(sb.triggered);
    CHECK(sa.rounds >= 1);
    CHECK(sa.evaluation_work_units > 0.0);
    CHECK_THAT(sa.evaluation_work_units, WithinAbs(sb.evaluation_work_units, 1e-9));
    CHECK(sa.resolve_calls == sb.resolve_calls);
    CHECK(sa.resolve_iterations == sb.resolve_iterations);
}

TEST_CASE("MipSolver: exact refinement auto mode triggers on unsupported rational checks",
          "[mip][exact_refinement]") {
    auto lp = buildBranchingMip();

    MipSolver solver;
    solver.setVerbose(false);
    solver.setCutsEnabled(false);
    solver.setPresolve(false);
    solver.setExactRefinementMode(ExactRefinementMode::Auto);
    solver.setExactRefinementRationalCheck(true);
    solver.setExactRefinementRationalScale(1.0e10);
    solver.load(lp);
    const auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    const auto& stats = solver.getExactRefinementStats();
    CHECK(stats.triggered);
    CHECK_FALSE(stats.rational_supported);
    CHECK_FALSE(stats.rational_certificate_passed);
    CHECK_FALSE(stats.certificate_passed);
}

TEST_CASE("MipSolver: exact refinement evaluates active root LP rows including cuts",
          "[mip][exact_refinement][cuts]") {
    auto lp = buildBranchingMip();

    MipSolver solver;
    solver.setVerbose(false);
    solver.setPresolve(false);
    solver.setSymmetryEnabled(false);
    solver.setCutsEnabled(true);
    solver.setCutEffortMode(CutEffortMode::Aggressive);
    solver.setExactRefinementMode(ExactRefinementMode::On);
    solver.load(lp);
    const auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    const auto& cut_stats = solver.getCutStats();
    REQUIRE(cut_stats.root_cuts_added > 0);

    const auto& stats = solver.getExactRefinementStats();
    CHECK(stats.rows_evaluated == lp.num_rows + cut_stats.root_cuts_added);
    CHECK(stats.cols_evaluated == lp.num_cols);
}

TEST_CASE("MipSolver: exact refinement off mode does not evaluate certificate rows",
          "[mip][exact_refinement][cuts]") {
    auto lp = buildBranchingMip();

    MipSolver solver;
    solver.setVerbose(false);
    solver.setPresolve(false);
    solver.setSymmetryEnabled(false);
    solver.setCutsEnabled(true);
    solver.setCutEffortMode(CutEffortMode::Aggressive);
    solver.setExactRefinementMode(ExactRefinementMode::Off);
    solver.load(lp);
    const auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    const auto& cut_stats = solver.getCutStats();
    REQUIRE(cut_stats.root_cuts_added > 0);

    const auto& stats = solver.getExactRefinementStats();
    CHECK(stats.mode == ExactRefinementMode::Off);
    CHECK_FALSE(stats.triggered);
    CHECK(stats.rows_evaluated == 0);
    CHECK(stats.cols_evaluated == 0);
    CHECK(stats.evaluation_work_units == 0.0);
}

TEST_CASE("MipSolver: root cut propagation carries discrete bounds across cut rounds",
          "[mip][cuts][domain]") {
    // The original rows imply no column bound by activity. The first cut
    // forces x=1, which makes z fractional in the re-solved LP. The next
    // separation then produces a bound fixing z=1 as well.
    LpProblem lp;
    lp.num_cols = 4;
    lp.obj = {1.0, 0.0, 1.0, 0.0};
    lp.col_lower.assign(4, 0.0);
    lp.col_upper.assign(4, 1.0);
    lp.col_type.assign(4, VarType::Binary);
    lp.num_rows = 4;
    lp.row_lower = {1.0, 0.0, -0.5, 0.0};
    lp.row_upper.assign(4, kInf);
    lp.matrix = SparseMatrix(4, 4,
                             {{0, 0, 1.0},
                              {0, 1, 1.0},
                              {1, 0, 1.0},
                              {1, 1, -1.0},
                              {2, 0, -1.0},
                              {2, 2, 1.0},
                              {2, 3, 1.0},
                              {3, 2, 1.0},
                              {3, 3, -1.0}});
    DomainPropagator original_rows;
    original_rows.load(lp);
    REQUIRE(original_rows.propagate());
    CHECK(original_rows.numTightened() == 0);

    MipSolver solver;
    solver.setVerbose(false);
    solver.setPresolve(false);
    solver.setSymmetryEnabled(false);
    solver.setExactRefinementMode(ExactRefinementMode::On);
    solver.load(lp);

    REQUIRE(solver.solve().status == Status::Optimal);
    CHECK(solver.getCutStats().root_cut_bound_tightenings == 0);
    const Int no_propagation_rounds = solver.getCutStats().root_rounds;

    solver.setRootCutPropagationRowBudget(1);
    REQUIRE(solver.solve().status == Status::Optimal);
    CHECK(solver.getCutStats().root_rounds == no_propagation_rounds);
    CHECK(solver.getCutStats().root_cut_propagation_row_visits == 0);
    CHECK(solver.getCutStats().root_cut_propagation_budget_exhaustions == 0);

    solver.setRootCutPropagationEnabled(true);
    for (int repeat = 0; repeat < 2; ++repeat) {
        const auto result = solver.solve();
        REQUIRE(result.status == Status::Optimal);
        CHECK_THAT(result.objective, WithinAbs(2.0, 1e-8));
        const auto& cuts = solver.getCutStats();
        CHECK(cuts.root_rounds == 1);
        CHECK(cuts.root_cut_propagation_row_visits == 1);
        CHECK(cuts.root_cut_propagation_budget_exhaustions == 1);
        CHECK(solver.getExactRefinementStats().certificate_passed);
    }

    solver.setRootCutPropagationRowBudget(0);
    solver.setMaxCutRounds(1);
    REQUIRE(solver.solve().status == Status::Optimal);
    const auto first_round_visits = solver.getCutStats().root_cut_propagation_row_visits;
    REQUIRE(first_round_visits > 1);

    solver.setMaxCutRounds(2);
    for (int repeat = 0; repeat < 2; ++repeat) {
        const auto result = solver.solve();
        REQUIRE(result.status == Status::Optimal);
        CHECK_THAT(result.objective, WithinAbs(2.0, 1e-8));
        REQUIRE(result.solution.size() == 4);
        CHECK_THAT(result.solution[0], WithinAbs(1.0, 1e-8));
        CHECK_THAT(result.solution[2], WithinAbs(1.0, 1e-8));
        const auto& cuts = solver.getCutStats();
        CHECK(cuts.root_rounds == 2);
        CHECK(cuts.root_cut_bound_tightenings == 2);
        CHECK(cuts.root_cut_bound_tightenings_backed_out == 0);
        CHECK(cuts.root_cut_propagation_row_visits > first_round_visits + 1);
        CHECK(cuts.root_cut_propagation_budget_exhaustions == 0);
        const auto& certificate = solver.getExactRefinementStats();
        CHECK(certificate.rows_evaluated == lp.num_rows + cuts.root_cuts_added);
        CHECK(certificate.certificate_passed);
    }

    solver.setRootCutPropagationRowBudget(static_cast<Int>(first_round_visits) + 1);
    const auto result = solver.solve();
    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(2.0, 1e-8));
    CHECK(solver.getCutStats().root_rounds == 2);
    CHECK(solver.getCutStats().root_cut_propagation_row_visits == first_round_visits + 1);
    CHECK(solver.getCutStats().root_cut_propagation_budget_exhaustions == 1);
}

TEST_CASE("MipSolver: a backed-out root cut batch discards its propagated bounds",
          "[mip][cuts][domain]") {
    // An installed x>=1 cut forces x=1 by propagation. The two rows with
    // free columns imply x<=0.75 jointly, which the LP detects but interval
    // propagation on either row cannot detect.
    LpProblem lp;
    lp.num_cols = 4;
    lp.obj = {1.0, 0.0, 0.0, 0.0};
    lp.col_lower = {0.0, 0.0, -kInf, -kInf};
    lp.col_upper = {1.0, 1.0, kInf, kInf};
    lp.col_type = {VarType::Binary, VarType::Binary, VarType::Continuous, VarType::Continuous};
    lp.num_rows = 4;
    lp.row_lower = {1.0, 0.0, -kInf, -kInf};
    lp.row_upper = {kInf, kInf, 0.75, 0.0};
    lp.matrix = SparseMatrix(4, 4,
                             {{0, 0, 1.0},
                              {0, 1, 1.0},
                              {1, 0, 1.0},
                              {1, 1, -1.0},
                              {2, 0, 1.0},
                              {2, 2, 1.0},
                              {2, 3, -1.0},
                              {3, 2, -1.0},
                              {3, 3, 1.0}});

    MipSolver solver;
    solver.setVerbose(false);
    solver.setPresolve(false);
    solver.setSymmetryEnabled(false);
    solver.setRootCutPropagationEnabled(true);
    solver.setExactRefinementMode(ExactRefinementMode::On);
    solver.load(lp);

    for (const Int budget : {0, 1}) {
        CAPTURE(budget);
        solver.setRootCutPropagationRowBudget(budget);
        REQUIRE(solver.solve().status == Status::Infeasible);
        const auto& cuts = solver.getCutStats();
        CHECK(cuts.root_rounds_backed_out == 1);
        CHECK(cuts.root_cuts_added == 0);
        CHECK(cuts.root_cut_bound_tightenings == 0);
        CHECK(cuts.root_cut_bound_tightenings_backed_out == 1);
        if (budget == 1) {
            CHECK(cuts.root_cut_propagation_row_visits == 1);
            CHECK(cuts.root_cut_propagation_budget_exhaustions == 1);
        } else {
            CHECK(cuts.root_cut_propagation_row_visits > 1);
            CHECK(cuts.root_cut_propagation_budget_exhaustions == 0);
        }
        const auto& certificate = solver.getExactRefinementStats();
        CHECK(certificate.rows_evaluated == lp.num_rows);
        CHECK(certificate.max_col_violation_before == 0.0);
        CHECK(certificate.certificate_passed);
    }
}

TEST_CASE("MipSolver: root cutoff feedback needs a validated incumbent", "[mip][cuts][domain]") {
    const bool pre_root = GENERATE(false, true);
    const Int row_budget = GENERATE(0, 1);
    // Two independent fractional optima keep the LP below the incumbent
    // after a cut on either pair. The pricey fifth variable has reduced cost.
    LpProblem model;
    model.num_cols = 5;
    model.obj = {-1.0, 0.0, -1.0, 0.0, -10.0};
    model.obj_offset = 7.0;
    model.sense = Sense::Maximize;
    model.col_lower.assign(5, 0.0);
    model.col_upper.assign(5, 1.0);
    model.col_type.assign(5, VarType::Binary);
    model.num_rows = 4;
    model.row_lower = {1.0, 0.0, 1.0, 0.0};
    model.row_upper.assign(4, kInf);
    model.matrix = SparseMatrix(4, 5,
                                {{0, 0, 1.0},
                                 {0, 1, 1.0},
                                 {1, 0, 1.0},
                                 {1, 1, -1.0},
                                 {2, 2, 1.0},
                                 {2, 3, 1.0},
                                 {3, 2, 1.0},
                                 {3, 3, -1.0}});

    MipSolver solver;
    solver.setVerbose(false);
    solver.setNumThreads(1);
    solver.setPresolve(false);
    solver.setSymmetryEnabled(false);
    solver.setRootCutoffFeedbackEnabled(true);
    solver.setRootCutPropagationRowBudget(row_budget);
    solver.setPreRootLpFreeEnabled(pre_root);
    solver.setMaxCutRounds(2);
    solver.setMaxCutsPerRound(1);
    solver.setCutEffortMode(CutEffortMode::Conservative);
    solver.setRootCutRecheck(true);
    solver.setExactRefinementMode(ExactRefinementMode::On);
    solver.load(model);

    const auto result = solver.solve();
    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(5.0, 1e-8));
    CHECK(std::isfinite(solver.getPreRootStats().incumbent_at_root) == pre_root);
    const auto& stats = solver.getCutStats();
    CHECK(stats.root_rounds == (pre_root && row_budget == 1 ? 1 : 2));
    CHECK(stats.root_cutoff_feedbacks == (pre_root ? 1 : 0));
    CHECK(stats.root_cutoff_reduced_cost_tightenings == (pre_root ? 1 : 0));
    CHECK(stats.root_cutoff_objective_tightenings == 0);
    CHECK(stats.root_cutoff_resolves_backed_out == 0);
    if (!pre_root || row_budget == 1) {
        CHECK(stats.root_cut_propagation_row_visits == (pre_root ? 1 : 0));
        CHECK(stats.root_cut_propagation_budget_exhaustions == (pre_root ? 1 : 0));
    }
    CHECK(solver.getExactRefinementStats().certificate_passed);
}

TEST_CASE("MipSolver: deterministic heuristic mode reproduces with same seed",
          "[mip][heuristics]") {
    auto lp = buildRootFractionalHeuristicMip();

    MipSolver solver_a;
    solver_a.setVerbose(false);
    solver_a.setCutsEnabled(false);
    solver_a.setNodeLimit(1);
    solver_a.setParallelMode(ParallelMode::Deterministic);
    solver_a.setHeuristicSeed(1234);
    solver_a.load(lp);
    auto result_a = solver_a.solve();

    MipSolver solver_b;
    solver_b.setVerbose(false);
    solver_b.setCutsEnabled(false);
    solver_b.setNodeLimit(1);
    solver_b.setParallelMode(ParallelMode::Deterministic);
    solver_b.setHeuristicSeed(1234);
    solver_b.load(lp);
    auto result_b = solver_b.solve();

    REQUIRE((result_a.status == Status::NodeLimit || result_a.status == Status::Optimal));
    REQUIRE((result_b.status == Status::NodeLimit || result_b.status == Status::Optimal));
    CHECK_THAT(result_a.objective, WithinAbs(result_b.objective, 1e-9));
    REQUIRE(result_a.solution.size() == result_b.solution.size());
    for (size_t i = 0; i < result_a.solution.size(); ++i) {
        CHECK_THAT(result_a.solution[i], WithinAbs(result_b.solution[i], 1e-9));
    }
}

TEST_CASE("MipSolver: opportunistic heuristic mode solves", "[mip][heuristics]") {
    auto lp = buildRootFractionalHeuristicMip();

    MipSolver solver;
    solver.setVerbose(false);
    solver.setCutsEnabled(false);
    solver.setNodeLimit(1);
    solver.setParallelMode(ParallelMode::Opportunistic);
    solver.setHeuristicSeed(7);
    solver.load(lp);
    const auto result = solver.solve();

    CHECK((result.status == Status::NodeLimit || result.status == Status::Optimal));
    REQUIRE(!result.solution.empty());
    CHECK(result.solution[0] >= 0.0);
    CHECK(result.solution[0] <= 4.6 + 1e-6);
}

TEST_CASE("MipSolver: pre-root LP-free stage is disabled by default",
          "[mip][heuristics][preroot]") {
    auto lp = buildBranchingMip();

    MipSolver solver;
    solver.setVerbose(false);
    solver.setCutsEnabled(false);
    solver.load(lp);
    auto result = solver.solve();

    CHECK((result.status == Status::Optimal || result.status == Status::NodeLimit ||
           result.status == Status::TimeLimit));
    const auto& stats = solver.getPreRootStats();
    CHECK_FALSE(stats.enabled);
    CHECK(stats.calls == 0);
    CHECK(stats.work_units == 0.0);
    CHECK_FALSE(stats.lp_light_enabled);
    CHECK_FALSE(stats.lp_light_available);
    CHECK(stats.lp_light_calls == 0);
}

TEST_CASE("MipSolver: LP-light capability reflects build configuration",
          "[mip][heuristics][preroot][lplight]") {
    MipSolver solver;
#ifdef MIPX_HAS_LP_LIGHT
    CHECK(solver.hasLpLightCapability());
#else
    CHECK_FALSE(solver.hasLpLightCapability());
#endif
}

TEST_CASE("MipSolver: pre-root LP-free stage can hand off incumbent",
          "[mip][heuristics][preroot]") {
    auto lp = buildRootFractionalHeuristicMip();

    MipSolver solver;
    solver.setVerbose(false);
    solver.setCutsEnabled(false);
    solver.setPresolve(false);
    solver.setNodeLimit(1);
    solver.setHeuristicMode(HeuristicRuntimeMode::Deterministic);
    solver.setHeuristicSeed(11);
    solver.setPreRootLpFreeEnabled(true);
    solver.setPreRootLpFreeWorkBudget(2.0e5);
    solver.setPreRootLpFreeMaxRounds(16);
    solver.setPreRootLpFreeEarlyStop(true);
    solver.load(lp);
    const auto result = solver.solve();

    CHECK((result.status == Status::NodeLimit || result.status == Status::Optimal));
    const auto& stats = solver.getPreRootStats();
    CHECK(stats.enabled);
    CHECK(stats.calls > 0);
    CHECK(stats.work_units > 0.0);
    CHECK(stats.feasible_found >= 1);
    CHECK(stats.lp_light_calls == 0);
    CHECK(std::isfinite(stats.incumbent_at_root));
    REQUIRE(!result.solution.empty());
}

TEST_CASE("MipSolver: pre-root LP-free deterministic mode reproduces with seed",
          "[mip][heuristics][preroot]") {
    auto lp = buildRootFractionalHeuristicMip();

    MipSolver solver_a;
    solver_a.setVerbose(false);
    solver_a.setCutsEnabled(false);
    solver_a.setPresolve(false);
    solver_a.setNodeLimit(1);
    solver_a.setHeuristicMode(HeuristicRuntimeMode::Deterministic);
    solver_a.setHeuristicSeed(77);
    solver_a.setPreRootLpFreeEnabled(true);
    solver_a.setPreRootLpFreeWorkBudget(1.0e5);
    solver_a.setPreRootLpFreeMaxRounds(12);
    solver_a.load(lp);
    const auto a = solver_a.solve();

    MipSolver solver_b;
    solver_b.setVerbose(false);
    solver_b.setCutsEnabled(false);
    solver_b.setPresolve(false);
    solver_b.setNodeLimit(1);
    solver_b.setHeuristicMode(HeuristicRuntimeMode::Deterministic);
    solver_b.setHeuristicSeed(77);
    solver_b.setPreRootLpFreeEnabled(true);
    solver_b.setPreRootLpFreeWorkBudget(1.0e5);
    solver_b.setPreRootLpFreeMaxRounds(12);
    solver_b.load(lp);
    const auto b = solver_b.solve();

    const auto& sa = solver_a.getPreRootStats();
    const auto& sb = solver_b.getPreRootStats();
    CHECK(sa.enabled);
    CHECK(sb.enabled);
    CHECK(sa.calls == sb.calls);
    CHECK_THAT(sa.work_units, WithinAbs(sb.work_units, 1e-9));
    CHECK_THAT(sa.incumbent_at_root, WithinAbs(sb.incumbent_at_root, 1e-9));
    CHECK_THAT(a.objective, WithinAbs(b.objective, 1e-9));
}

TEST_CASE("MipSolver: pre-root LP-light arms can run when enabled",
          "[mip][heuristics][preroot][lplight]") {
    auto lp = buildRootFractionalHeuristicMip();

    MipSolver solver;
    solver.setVerbose(false);
    solver.setCutsEnabled(false);
    solver.setPresolve(false);
    solver.setNodeLimit(1);
    solver.setHeuristicMode(HeuristicRuntimeMode::Deterministic);
    solver.setHeuristicSeed(19);
    solver.setPreRootLpFreeEnabled(false);
    solver.setPreRootLpLightEnabled(true);
    solver.setPreRootLpFreeWorkBudget(1.0e5);
    solver.setPreRootLpFreeMaxRounds(10);
    solver.load(lp);
    const auto result = solver.solve();

    CHECK((result.status == Status::NodeLimit || result.status == Status::Optimal));
    const auto& stats = solver.getPreRootStats();
    CHECK(stats.enabled);
    CHECK(stats.lp_light_enabled);
#ifdef MIPX_HAS_LP_LIGHT
    CHECK(stats.lp_light_available);
    CHECK(stats.lp_light_lp_solves >= 1);
    CHECK(stats.lp_light_calls > 0);
    CHECK(stats.lp_light_fpr_calls + stats.lp_light_diving_calls == stats.lp_light_calls);
#else
    CHECK_FALSE(stats.lp_light_available);
    CHECK(stats.lp_light_calls == 0);
#endif
}

TEST_CASE("MipSolver: pre-root LP-light deterministic mode reproduces with seed",
          "[mip][heuristics][preroot][lplight]") {
    auto lp = buildRootFractionalHeuristicMip();

    MipSolver solver_a;
    solver_a.setVerbose(false);
    solver_a.setCutsEnabled(false);
    solver_a.setPresolve(false);
    solver_a.setNodeLimit(1);
    solver_a.setHeuristicMode(HeuristicRuntimeMode::Deterministic);
    solver_a.setHeuristicSeed(99);
    solver_a.setPreRootLpFreeEnabled(false);
    solver_a.setPreRootLpLightEnabled(true);
    solver_a.setPreRootLpFreeWorkBudget(1.0e5);
    solver_a.setPreRootLpFreeMaxRounds(10);
    solver_a.load(lp);
    const auto a = solver_a.solve();

    MipSolver solver_b;
    solver_b.setVerbose(false);
    solver_b.setCutsEnabled(false);
    solver_b.setPresolve(false);
    solver_b.setNodeLimit(1);
    solver_b.setHeuristicMode(HeuristicRuntimeMode::Deterministic);
    solver_b.setHeuristicSeed(99);
    solver_b.setPreRootLpFreeEnabled(false);
    solver_b.setPreRootLpLightEnabled(true);
    solver_b.setPreRootLpFreeWorkBudget(1.0e5);
    solver_b.setPreRootLpFreeMaxRounds(10);
    solver_b.load(lp);
    const auto b = solver_b.solve();

    const auto& sa = solver_a.getPreRootStats();
    const auto& sb = solver_b.getPreRootStats();
    CHECK(sa.enabled == sb.enabled);
    CHECK(sa.lp_light_enabled == sb.lp_light_enabled);
    CHECK(sa.lp_light_available == sb.lp_light_available);
    CHECK(sa.calls == sb.calls);
    CHECK(sa.lp_light_calls == sb.lp_light_calls);
    CHECK_THAT(sa.work_units, WithinAbs(sb.work_units, 1e-9));
    CHECK_THAT(a.objective, WithinAbs(b.objective, 1e-9));
}

TEST_CASE("MipSolver: pre-root LP-light fixed schedule runs Scylla-FPR arm first",
          "[mip][heuristics][preroot][lplight][scylla]") {
    auto lp = buildLpLightScyllaProbeMip();

    MipSolver solver;
    solver.setVerbose(false);
    solver.setCutsEnabled(false);
    solver.setPresolve(false);
    solver.setNodeLimit(1);
    solver.setHeuristicMode(HeuristicRuntimeMode::Deterministic);
    solver.setHeuristicSeed(2026);
    solver.setPreRootLpFreeEnabled(false);
    solver.setPreRootLpLightEnabled(true);
    solver.setPreRootPortfolioEnabled(false);  // fixed schedule
    solver.setPreRootLpFreeEarlyStop(false);
    solver.setPreRootLpFreeMaxRounds(1);  // single arm call
    solver.setPreRootLpFreeWorkBudget(1.0e9);
    solver.load(lp);
    const auto result = solver.solve();

    CHECK((result.status == Status::NodeLimit || result.status == Status::Optimal));
    const auto& stats = solver.getPreRootStats();
    CHECK(stats.enabled);
    CHECK_FALSE(stats.portfolio_enabled);
#ifdef MIPX_HAS_LP_LIGHT
    CHECK(stats.lp_light_available);
    CHECK(stats.calls == 1);
    CHECK(stats.lp_light_calls == 1);
    CHECK(stats.lp_light_fpr_calls == 1);
    CHECK(stats.lp_light_diving_calls == 0);
    CHECK(stats.work_units > 0.0);
#else
    CHECK_FALSE(stats.lp_light_available);
    CHECK(stats.lp_light_calls == 0);
#endif
}

TEST_CASE("MipSolver: pre-root LP-light Scylla path is deterministic on binary probe",
          "[mip][heuristics][preroot][lplight][scylla]") {
    auto lp = buildLpLightScyllaProbeMip();

    MipSolver solver_a;
    solver_a.setVerbose(false);
    solver_a.setCutsEnabled(false);
    solver_a.setPresolve(false);
    solver_a.setNodeLimit(1);
    solver_a.setHeuristicMode(HeuristicRuntimeMode::Deterministic);
    solver_a.setHeuristicSeed(777);
    solver_a.setPreRootLpFreeEnabled(false);
    solver_a.setPreRootLpLightEnabled(true);
    solver_a.setPreRootPortfolioEnabled(false);
    solver_a.setPreRootLpFreeEarlyStop(false);
    solver_a.setPreRootLpFreeMaxRounds(1);
    solver_a.setPreRootLpFreeWorkBudget(1.0e9);
    solver_a.load(lp);
    const auto a = solver_a.solve();

    MipSolver solver_b;
    solver_b.setVerbose(false);
    solver_b.setCutsEnabled(false);
    solver_b.setPresolve(false);
    solver_b.setNodeLimit(1);
    solver_b.setHeuristicMode(HeuristicRuntimeMode::Deterministic);
    solver_b.setHeuristicSeed(777);
    solver_b.setPreRootLpFreeEnabled(false);
    solver_b.setPreRootLpLightEnabled(true);
    solver_b.setPreRootPortfolioEnabled(false);
    solver_b.setPreRootLpFreeEarlyStop(false);
    solver_b.setPreRootLpFreeMaxRounds(1);
    solver_b.setPreRootLpFreeWorkBudget(1.0e9);
    solver_b.load(lp);
    const auto b = solver_b.solve();

    const auto& sa = solver_a.getPreRootStats();
    const auto& sb = solver_b.getPreRootStats();
    CHECK(sa.calls == sb.calls);
    CHECK(sa.lp_light_calls == sb.lp_light_calls);
    CHECK(sa.lp_light_fpr_calls == sb.lp_light_fpr_calls);
    CHECK(sa.lp_light_diving_calls == sb.lp_light_diving_calls);
    CHECK_THAT(sa.work_units, WithinAbs(sb.work_units, 1e-9));
    CHECK_THAT(a.objective, WithinAbs(b.objective, 1e-9));
}

TEST_CASE("MipSolver: pre-root LP-light Scylla repairs integral-step violations",
          "[mip][heuristics][preroot][lplight][scylla]") {
    auto lp = buildLpLightIntegerRepairProbeMip();

    MipSolver solver;
    solver.setVerbose(false);
    solver.setCutsEnabled(false);
    solver.setPresolve(false);
    solver.setNodeLimit(1);
    solver.setHeuristicMode(HeuristicRuntimeMode::Deterministic);
    solver.setHeuristicSeed(17);
    solver.setPreRootLpFreeEnabled(false);
    solver.setPreRootLpLightEnabled(true);
    solver.setPreRootPortfolioEnabled(false);
    solver.setPreRootLpFreeEarlyStop(false);
    solver.setPreRootLpFreeMaxRounds(1);
    solver.setPreRootLpFreeWorkBudget(1.0e9);
    solver.load(lp);
    const auto result = solver.solve();

    CHECK((result.status == Status::NodeLimit || result.status == Status::Optimal));
    const auto& stats = solver.getPreRootStats();
    CHECK(stats.enabled);
    CHECK_FALSE(stats.portfolio_enabled);
#ifdef MIPX_HAS_LP_LIGHT
    CHECK(stats.lp_light_available);
    CHECK(stats.calls == 1);
    CHECK(stats.lp_light_calls == 1);
    CHECK(stats.lp_light_fpr_calls == 1);
    CHECK(stats.lp_light_diving_calls == 0);
    CHECK(stats.feasible_found >= 1);
    CHECK(stats.improvements >= 1);
    REQUIRE(!result.solution.empty());
    CHECK_THAT(result.solution[0], WithinAbs(3.0, 1e-9));
#else
    CHECK_FALSE(stats.lp_light_available);
    CHECK(stats.lp_light_calls == 0);
#endif
}

TEST_CASE("MipSolver: pre-root fixed schedule can be selected",
          "[mip][heuristics][preroot][portfolio]") {
    auto lp = buildRootFractionalHeuristicMip();

    MipSolver solver;
    solver.setVerbose(false);
    solver.setCutsEnabled(false);
    solver.setPresolve(false);
    solver.setNodeLimit(1);
    solver.setHeuristicMode(HeuristicRuntimeMode::Deterministic);
    solver.setHeuristicSeed(5);
    solver.setPreRootLpFreeEnabled(true);
    solver.setPreRootLpLightEnabled(false);
    solver.setPreRootPortfolioEnabled(false);
    solver.setPreRootLpFreeEarlyStop(false);
    solver.setPreRootLpFreeMaxRounds(6);
    solver.setPreRootLpFreeWorkBudget(1.0e9);
    solver.load(lp);
    (void)solver.solve();

    const auto& stats = solver.getPreRootStats();
    CHECK(stats.enabled);
    CHECK_FALSE(stats.portfolio_enabled);
    CHECK(stats.calls == 6);
    CHECK(stats.feasible_found >= 1);
    CHECK(stats.fj_calls == 2);
    CHECK(stats.fpr_calls == 2);
    CHECK(stats.local_mip_calls == 2);
}

TEST_CASE("MipSolver: pre-root fixed schedule skips LocalMip until an incumbent exists",
          "[mip][heuristics][preroot][portfolio]") {
    auto lp = buildConflictLearningMip();

    MipSolver solver;
    solver.setVerbose(false);
    solver.setCutsEnabled(false);
    solver.setPresolve(false);
    solver.setNodeLimit(1);
    solver.setHeuristicMode(HeuristicRuntimeMode::Deterministic);
    solver.setHeuristicSeed(13);
    solver.setPreRootLpFreeEnabled(true);
    solver.setPreRootLpLightEnabled(false);
    solver.setPreRootPortfolioEnabled(false);
    solver.setPreRootLpFreeEarlyStop(false);
    solver.setPreRootLpFreeMaxRounds(9);
    solver.setPreRootLpFreeWorkBudget(1.0e9);
    solver.load(lp);
    (void)solver.solve();

    const auto& stats = solver.getPreRootStats();
    CHECK(stats.enabled);
    CHECK_FALSE(stats.portfolio_enabled);
    CHECK(stats.calls > 0);
    CHECK(stats.feasible_found == 0);
    CHECK(stats.local_mip_calls == 0);
    CHECK(stats.fj_calls + stats.fpr_calls == stats.calls);
}

TEST_CASE("MipSolver: pre-root adaptive portfolio tracks telemetry",
          "[mip][heuristics][preroot][portfolio]") {
    auto lp = buildRootFractionalHeuristicMip();

    MipSolver solver;
    solver.setVerbose(false);
    solver.setCutsEnabled(false);
    solver.setPresolve(false);
    solver.setNodeLimit(1);
    solver.setHeuristicMode(HeuristicRuntimeMode::Deterministic);
    solver.setHeuristicSeed(23);
    solver.setPreRootLpFreeEnabled(true);
    solver.setPreRootLpLightEnabled(true);
    solver.setPreRootPortfolioEnabled(true);
    solver.setPreRootLpFreeEarlyStop(false);
    solver.setPreRootLpFreeMaxRounds(10);
    solver.setPreRootLpFreeWorkBudget(1.0e6);
    solver.load(lp);
    (void)solver.solve();

    const auto& stats = solver.getPreRootStats();
    CHECK(stats.enabled);
    CHECK((stats.portfolio_enabled || stats.calls <= 1));
    CHECK(stats.portfolio_epochs == stats.calls);
    CHECK(stats.effort_scale_final > 0.0);
    CHECK(stats.fj_calls + stats.fpr_calls + stats.local_mip_calls + stats.lp_light_fpr_calls +
              stats.lp_light_diving_calls ==
          stats.calls);
}

TEST_CASE("MipSolver: pre-root opportunistic fixed schedule enables LocalMip after incumbent",
          "[mip][heuristics][preroot][portfolio]") {
    auto lp = buildRootFractionalHeuristicMip();

    MipSolver solver;
    solver.setVerbose(false);
    solver.setCutsEnabled(false);
    solver.setPresolve(false);
    solver.setNodeLimit(1);
    solver.setNumThreads(1);
    solver.setParallelMode(ParallelMode::Opportunistic);
    solver.setHeuristicSeed(31);
    solver.setPreRootLpFreeEnabled(true);
    solver.setPreRootLpLightEnabled(false);
    solver.setPreRootPortfolioEnabled(false);
    solver.setPreRootLpFreeEarlyStop(false);
    solver.setPreRootLpFreeMaxRounds(96);
    solver.setPreRootLpFreeWorkBudget(1.0e9);
    solver.load(lp);
    (void)solver.solve();

    const auto& stats = solver.getPreRootStats();
    CHECK(stats.enabled);
    CHECK_FALSE(stats.portfolio_enabled);
    CHECK(stats.calls >= 1);
    CHECK(stats.feasible_found >= 1);
    CHECK(stats.local_mip_calls > 0);
    CHECK(stats.fj_calls + stats.fpr_calls + stats.local_mip_calls == stats.calls);
}

TEST_CASE("MipSolver: conflict learning learns and reuses no-goods", "[mip][conflicts]") {
    auto lp = buildConflictLearningMip();

    MipSolver solver;
    solver.setVerbose(false);
    solver.setCutsEnabled(false);
    solver.setPresolve(false);
    solver.setNodeLimit(128);
    solver.load(lp);
    const auto result = solver.solve();

    CHECK(result.status == Status::Infeasible);
    const auto& cstats = solver.getConflictStats();
    CHECK(cstats.learned >= 1);
    CHECK(cstats.lp_infeasible_conflicts >= 1);
    CHECK(cstats.minimized_literals >= 0);
}

TEST_CASE("MipSolver: conflict learning preserves feasible optimum", "[mip][conflicts]") {
    auto lp = buildBranchingMip();

    MipSolver with_conflicts;
    with_conflicts.setVerbose(false);
    with_conflicts.setCutsEnabled(false);
    with_conflicts.setConflictsEnabled(true);
    with_conflicts.load(lp);
    const auto with_result = with_conflicts.solve();

    MipSolver without_conflicts;
    without_conflicts.setVerbose(false);
    without_conflicts.setCutsEnabled(false);
    without_conflicts.setConflictsEnabled(false);
    without_conflicts.load(lp);
    const auto without_result = without_conflicts.solve();

    REQUIRE(with_result.status == Status::Optimal);
    REQUIRE(without_result.status == Status::Optimal);
    CHECK_THAT(with_result.objective, WithinAbs(without_result.objective, 1e-9));
}

TEST_CASE("Conflict watches agree with a scan across sibling paths and eviction",
          "[mip][conflicts]") {
    const std::vector<BranchDecision> interval = {{0, 3, false}, {0, 5, true}, {1, 2, false}};
    const std::vector<BranchDecision> singleton = {{2, 7, true}};
    // Same condition with a strict threshold, as used for ray-shortened clauses.
    const std::vector<BranchDecision> exact = {{2, 7, true}};
    const std::vector<BranchDecision> replacement = {{1, 2, false}};
    const std::vector<BranchDecision> spread = {{0, 3, false}, {1, 2, false}, {2, 7, true}};
    const std::vector<BranchDecision> shared = {{0, 3, false}, {1, 2, false}, {1, 4, true}};
    ConflictWatchIndex watches;
    const auto interval_id = watches.add(interval, 1e-9);
    const auto exact_id = watches.add(exact, 0.0);
    const auto singleton_id = watches.add(singleton, 1e-9);

    struct Entry {
        ConflictWatchIndex::Id id;
        std::span<const BranchDecision> literals;
        Real tolerance;
    };
    std::vector<Entry> active = {
        {interval_id, interval, 1e-9}, {exact_id, exact, 0.0}, {singleton_id, singleton, 1e-9}};
    struct PathBound {
        Index col;
        Real lower;
        Real upper;
    };
    const auto check = [&](std::initializer_list<PathBound> path) {
        std::vector<Index> columns;
        for (const auto& bound : path) {
            columns.push_back(bound.col);
        }
        const auto entails = [&](const BranchDecision& lit, Real tolerance) {
            const auto bound = std::ranges::find(path, lit.variable, &PathBound::col);
            return bound != path.end() && (lit.is_upper ? bound->upper <= lit.bound + tolerance
                                                        : bound->lower >= lit.bound - tolerance);
        };
        const auto expected = std::ranges::find_if(active, [&](const Entry& entry) {
            return std::ranges::all_of(
                entry.literals, [&](const auto& lit) { return entails(lit, entry.tolerance); });
        });
        const auto result = watches.findTriggered(columns, entails);
        CHECK(result.triggered ==
              (expected == active.end() ? std::nullopt : std::optional(expected->id)));
        return result;
    };

    CHECK(check({{3, 0, 1}}).candidates == 0);
    const auto failed_watch = check({{0, 3, 6}, {1, 2, 4}});
    CHECK(failed_watch.body_scans == 0);
    CHECK(failed_watch.literal_checks == 2);
    // Both initial watches share column 0. Move one to the absent column 1,
    // then revisit a sibling whose relaxed upper bound blocks the other.
    CHECK(check({{0, 3, 5}}).body_scans == 1);
    CHECK(check({{0, 3, 6}, {1, 2, 4}}).body_scans == 0);
    CHECK(check({{1, 2, 4}, {0, 3, 5}, {2, 0, 7}}).triggered == interval_id);
    CHECK(check({{2, 0, 7.0 + 5e-10}}).triggered == singleton_id);
    CHECK(check({{2, 0, 7}}).triggered == exact_id);

    watches.remove(interval_id);
    active.erase(active.begin());
    CHECK(check({{0, 3, 5}, {1, 2, 4}}).candidates == 0);
    const auto replacement_id = watches.add(replacement, 1e-9);
    active.push_back({replacement_id, replacement, 1e-9});
    CHECK(check({{0, 3, 5}, {1, 2, 4}}).triggered == replacement_id);
    CHECK(check({{1, 2, 4}, {2, 0, 7}}).triggered == exact_id);

    const auto spread_id = watches.add(spread, 1e-9);
    const auto shared_id = watches.add(shared, 1e-9);
    active.push_back({spread_id, spread, 1e-9});
    active.push_back({shared_id, shared, 1e-9});
    watches.remove(replacement_id);
    std::erase_if(active, [&](const Entry& entry) { return entry.id == replacement_id; });
    // Move the first watch to a new column in spread and to the column of
    // the other watch in shared. Neither clause should still visit column 0.
    CHECK(check({{0, 3, 5}, {1, 2, 5}}).body_scans == 2);
    CHECK(check({{0, 3, 5}}).candidates == 0);
    CHECK(check({{0, 3, 5}, {1, 2, 4}}).triggered == shared_id);
}

TEST_CASE("MipSolver: watched no-goods preserve age purging", "[mip][conflicts]") {
    MipSolver solver;
    solver.setVerbose(false);
    solver.setCutsEnabled(false);
    solver.setPresolve(false);
    solver.setTreePresolveEnabled(false);
    solver.setStrongBranchingEnabled(false);
    solver.setSymmetryEnabled(false);
    solver.load(buildSearchStagnationMip());

    const auto scan = solver.solve();
    const auto stats = solver.getConflictStats();
    REQUIRE(scan.status == Status::Infeasible);
    REQUIRE(stats.purged > 0);

    solver.setConflictWatchesEnabled(true);
    const auto watched = solver.solve();
    const auto& watched_stats = solver.getConflictStats();
    CHECK(watched.status == scan.status);
    CHECK(watched.nodes == scan.nodes);
    CHECK(watched.lp_iterations == scan.lp_iterations);
    CHECK(watched_stats.learned == stats.learned);
    CHECK(watched_stats.reused == stats.reused);
    CHECK(watched_stats.purged == stats.purged);
    CHECK(watched_stats.watch_candidates > 0);
}

TEST_CASE("MipSolver: tracking bound reasons preserves pre-LP search", "[mip][reasons]") {
    MipSolver solver;
    solver.setVerbose(false);
    solver.setCutsEnabled(false);
    solver.setPresolve(false);
    solver.setTreePresolveEnabled(false);
    solver.setStrongBranchingEnabled(false);
    solver.setSymmetryEnabled(false);
    solver.setPreLpPropagationEnabled(true);
    solver.load(buildSearchStagnationMip());
    const auto baseline = solver.solve();
    const auto stats = solver.getLpStats();
    REQUIRE(baseline.status == Status::Infeasible);
    REQUIRE(stats.pre_lp_propagation_calls > 1);

    solver.setBoundReasonsEnabled(true);
    const auto recorded = solver.solve();
    CHECK(recorded.status == baseline.status);
    CHECK(recorded.nodes == baseline.nodes);
    CHECK(recorded.lp_iterations == baseline.lp_iterations);
    CHECK(solver.getLpStats().pre_lp_propagation_calls == stats.pre_lp_propagation_calls);
    CHECK(solver.getLpStats().pre_lp_propagation_pruned == stats.pre_lp_propagation_pruned);
}

TEST_CASE("MipSolver: opt-in bound conflicts learn and tighten serial nodes", "[mip][conflicts]") {
    MipSolver solver;
    solver.setVerbose(false);
    solver.setCutsEnabled(false);
    solver.setPresolve(false);
    solver.setTreePresolveEnabled(false);
    solver.setStrongBranchingEnabled(false);
    solver.setSymmetryEnabled(false);
    solver.load(buildSearchStagnationMip());

    REQUIRE(solver.solve().status == Status::Infeasible);
    CHECK(solver.getConflictStats().bound_conflicts_learned == 0);
    CHECK(solver.getConflictStats().bound_conflict_tightenings == 0);

    solver.setBoundConflictLearningEnabled(true);
    REQUIRE(solver.solve().status == Status::Infeasible);
    const auto learned = solver.getConflictStats();
    REQUIRE(learned.bound_conflicts_learned > 0);
    CHECK(learned.bound_conflict_tightenings > 0);

    solver.setConflictWatchesEnabled(true);
    REQUIRE(solver.solve().status == Status::Infeasible);
    const auto watched = solver.getConflictStats();
    CHECK(watched.bound_conflicts_learned == learned.bound_conflicts_learned);
    CHECK(watched.bound_conflict_tightenings == learned.bound_conflict_tightenings);
    CHECK(watched.bound_conflict_cutoffs == learned.bound_conflict_cutoffs);
    CHECK(watched.pruned == learned.pruned);
}

TEST_CASE("MipSolver: bound conflicts preserve a feasible optimum", "[mip][conflicts]") {
    auto lp = buildSearchStagnationMipWithEscape();

    MipSolver solver;
    solver.setVerbose(false);
    solver.setCutsEnabled(false);
    solver.setPresolve(false);
    solver.setTreePresolveEnabled(false);
    solver.setStrongBranchingEnabled(false);
    solver.setSymmetryEnabled(false);
    solver.load(lp);
    const auto baseline = solver.solve();
    REQUIRE(baseline.status == Status::Optimal);
    CHECK_THAT(baseline.objective, WithinAbs(1.0, 1e-9));

    solver.setBoundConflictLearningEnabled(true);
    const auto learned = solver.solve();
    REQUIRE(learned.status == Status::Optimal);
    CHECK_THAT(learned.objective, WithinAbs(baseline.objective, 1e-9));
    CHECK(solver.getConflictStats().bound_conflict_tightenings > 0);

    lp.col_type.back() = VarType::Continuous;
    solver.load(lp);
    const auto mixed = solver.solve();
    REQUIRE(mixed.status == Status::Optimal);
    CHECK_THAT(mixed.objective, WithinAbs(baseline.objective, 1e-9));
    CHECK(solver.getConflictStats().bound_conflicts_learned == 0);
    CHECK(solver.getLpStats().pre_lp_propagation_calls == 0);
}

TEST_CASE("MipSolver: learned and aged conflict activity changes a tree branch",
          "[mip][conflicts][branching]") {
    enum : Index { r, x, y, w, z, num_cols };
    // ROOT takes r=0.25. The r=0 subtree learns two conflicts on y, one
    // for each endpoint. Then r=w=1 has y=0.5 and the specified fractional x.
    const Real x_fraction = GENERATE(0.167, 0.2);
    LpProblem lp;
    lp.num_cols = num_cols;
    lp.obj = {1.0, 0.1, 0.01, 0.05, 1.5};
    lp.col_lower.assign(num_cols, 0.0);
    lp.col_upper.assign(num_cols, 1.0);
    lp.col_type = {VarType::Binary, VarType::Binary, VarType::Binary, VarType::Binary,
                   VarType::Continuous};
    lp.num_rows = 6;
    lp.row_lower = {0.25, -0.5, -x_fraction, 0.5, -0.5, -kInf};
    lp.row_upper = {kInf, kInf, kInf, kInf, kInf, 0.75};
    lp.matrix = SparseMatrix(6, num_cols,
                             {{0, r, 1.0},
                              {0, z, 1.0},
                              {1, w, 1.0},
                              {1, r, -1.0},
                              {2, x, 1.0},
                              {2, w, -2 * x_fraction},
                              {3, y, 1.0},
                              {3, r, 0.5},
                              {4, y, 1.0},
                              {4, w, -1.0},
                              {5, y, 1.0},
                              {5, r, -0.25}});

    MipSolver solver;
    solver.setVerbose(false);
    solver.setNumThreads(1);
    solver.setCutsEnabled(false);
    solver.setPresolve(false);
    solver.setTreePresolveEnabled(false);
    solver.setStrongBranchingEnabled(false);
    solver.setSymmetryEnabled(false);
    solver.setPlungeControls(1, 1.0);
    solver.setDirectionalInferenceEnabled(true);
    solver.setGapTolerance(0.0);
    solver.setNodeLimit(8);  // Visit only the first child of r=w=1.
    solver.load(lp);

    const auto down_children = [&](Index col) {
        const auto& down = solver.getBranchInferenceStats()[col].down;
        return down.samples + down.cutoffs + down.skipped;
    };
    solver.setConflictsEnabled(false);
    REQUIRE(solver.solve().status == Status::NodeLimit);
    CHECK(down_children(x) == 1);
    CHECK(down_children(y) == 1);
    const auto baseline = solver.getBranchingStats();

    solver.setConflictsEnabled(true);
    solver.setConflictActivityBranchingEnabled(true);
    REQUIRE(solver.solve().status == Status::NodeLimit);
    CHECK(solver.getConflictStats().learned == 4);
    // Two fresh bumps give y a score of 5/6. Aging across the intervening
    // nodes leaves it below 0.833 but above 0.8. Only x's fractional value
    // differs between the two cases, giving it score 0.833 or 0.8.
    const bool y_wins = x_fraction == 0.2;
    CHECK(down_children(x) == (y_wins ? 0 : 1));
    CHECK(down_children(y) == (y_wins ? 2 : 1));
    CHECK(solver.getBranchingStats().pseudocost_updates_tree == baseline.pseudocost_updates_tree);
}

TEST_CASE("MipSolver: watched no-goods preserve serial reuse across solves", "[mip][conflicts]") {
    // Minimizing x first branches on y=0.5. Both y children then take x=0.5.
    // The x>=1 child is LP-infeasible independently of y, so a ray-shortened
    // clause learned on the first side can prune it on the other side.
    LpProblem lp;
    lp.num_cols = 2;
    lp.obj = {0.0, 1.0};
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {1.0, 1.0};
    lp.col_type = {VarType::Binary, VarType::Binary};
    lp.num_rows = 3;
    lp.row_lower = {0.5, -0.5, -kInf};
    lp.row_upper = {kInf, kInf, 0.5};
    lp.matrix =
        SparseMatrix(3, 2, {{0, 0, 1.0}, {0, 1, 1.0}, {1, 0, -1.0}, {1, 1, 1.0}, {2, 1, 1.0}});

    MipSolver solver;
    solver.setVerbose(false);
    solver.setCutsEnabled(false);
    solver.setPresolve(false);
    solver.setTreePresolveEnabled(false);
    solver.setStrongBranchingEnabled(false);
    solver.setConflictRayShorteningEnabled(true);
    solver.setConflictBenchStats(true);
    solver.load(lp);
    const auto scan = solver.solve();
    const auto stats = solver.getConflictStats();
    REQUIRE(scan.status == Status::Infeasible);
    REQUIRE(stats.ray_reused > 0);
    CHECK(stats.watch_candidates == 0);
    CHECK(stats.linear_clause_checks > 0);
    CHECK(stats.linear_literal_checks >= stats.linear_clause_checks);
    REQUIRE(stats.root_bound_before_tree.has_value());
    CHECK_THAT(*stats.root_bound_before_tree, WithinAbs(0.0, 1e-9));

    solver.setConflictWatchesEnabled(true);
    for (int attempt = 0; attempt < 2; ++attempt) {
        const auto watched = solver.solve();
        const auto& watched_stats = solver.getConflictStats();
        CHECK(watched.status == scan.status);
        CHECK(watched.objective == scan.objective);
        CHECK(watched.nodes == scan.nodes);
        CHECK(watched.lp_iterations == scan.lp_iterations);
        CHECK(watched_stats.learned == stats.learned);
        CHECK(watched_stats.reused == stats.reused);
        CHECK(watched_stats.pruned == stats.pruned);
        CHECK(watched_stats.purged == stats.purged);
        CHECK(watched_stats.ray_shortened == stats.ray_shortened);
        CHECK(watched_stats.ray_reused == stats.ray_reused);
        CHECK(watched_stats.watch_candidates > 0);
        CHECK(watched_stats.linear_clause_checks == 0);
        CHECK(watched_stats.linear_literal_checks == 0);
    }

    solver.setConflictsEnabled(false);
    CHECK(solver.solve().status == scan.status);
    CHECK(solver.getConflictStats().learned == 0);
    CHECK(solver.getConflictStats().watch_candidates == 0);
    solver.setConflictsEnabled(true);
    solver.setNumThreads(2);
    CHECK(solver.solve().status == scan.status);
    CHECK(solver.getConflictStats().learned == 0);
    CHECK(solver.getConflictStats().watch_candidates == 0);
}

TEST_CASE("MipSolver: a checked LP ray drops an irrelevant branch bound", "[mip][conflicts]") {
    // Root y=0.5 is the only fractional column. In y=0 the LP takes x=u=0.5.
    // Branching up on x is infeasible independently of y. The y=1 sibling
    // can take x=u=0, which is the global optimum.
    LpProblem lp;
    lp.num_cols = 3;
    lp.obj = {1.0, 0.4, 1.2};
    lp.col_lower = {0.0, 0.0, 0.0};
    lp.col_upper = {1.0, 1.0, 1.0};
    lp.col_type = {VarType::Binary, VarType::Binary, VarType::Continuous};
    lp.col_names = {"y", "x", "u"};
    lp.num_rows = 3;
    lp.row_lower = {0.5, 0.5, -kInf};
    lp.row_upper = {kInf, kInf, 0.5};
    lp.matrix =
        SparseMatrix(3, 3, {{0, 0, 1.0}, {0, 2, 1.0}, {1, 0, 1.0}, {1, 1, 1.0}, {2, 1, 1.0}});

    MipSolver solver;
    solver.setVerbose(false);
    solver.setPresolve(false);
    solver.setCutsEnabled(false);
    solver.setTreePresolveEnabled(false);
    solver.setStrongBranchingEnabled(false);
    solver.setNumThreads(1);
    solver.load(lp);
    REQUIRE(solver.solve().status == Status::Optimal);
    CHECK(solver.getConflictStats().ray_qualified == 0);

    solver.setConflictRayShorteningEnabled(true);
    const auto result = solver.solve();
    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(1.0, 1e-8));
    REQUIRE(result.solution.size() == 3);
    CHECK_THAT(result.solution[0], WithinAbs(1.0, 1e-8));
    CHECK_THAT(result.solution[1], WithinAbs(0.0, 1e-8));
    CHECK_THAT(result.solution[2], WithinAbs(0.0, 1e-8));
    CHECK(solver.getConflictStats().ray_shortened >= 1);
    CHECK(solver.getConflictStats().ray_literals_dropped >= 1);
}

TEST_CASE("MIP conflict ray recheck retains the path if a local row or bound is needed",
          "[mip][conflicts]") {
    LpProblem model;
    model.num_cols = 3;
    model.obj.assign(3, 0.0);
    model.col_lower.assign(3, 0.0);
    model.col_upper.assign(3, 1.0);
    model.col_type.assign(3, VarType::Binary);
    model.num_rows = 1;
    model.row_lower = {0.5};
    model.row_upper = {kInf};
    model.matrix = SparseMatrix(1, 3, {{0, 0, 1.0}, {0, 2, 1.0}});
    std::vector<BranchDecision> path{{0, 0.0, true}, {1, 0.0, true}};

    DualSimplexSolver lp;
    lp.setVerbose(false);
    lp.load(model);
    lp.setColBounds(0, 0.0, 0.0);
    lp.setColBounds(1, 0.0, 0.0);
    std::vector<Index> local_rows;
    if (GENERATE(false, true)) {
        // x=0 cannot satisfy this subtree's cut x>=0.5. Its ray is unusable
        // even though the globally valid row remains in the LP.
        lp.addRows(std::array<Index, 1>{0}, std::array<Index, 1>{0}, std::array<Real, 1>{1.0},
                   std::array<Real, 1>{0.5}, std::array<Real, 1>{kInf});
        local_rows.push_back(1);
    } else {
        // This fixing is absent from the persistent domain and branch path.
        lp.setColBounds(2, 0.0, 0.0);
    }
    REQUIRE(lp.solve().status == Status::Infeasible);
    const auto ray = lp.getDualRay();
    REQUIRE_FALSE(ray.empty());
    CHECK_FALSE(
        shortenConflictWithRay(model, ray, local_rows, model.col_lower, model.col_upper, path));
    REQUIRE(path.size() == 2);
    CHECK(path[0].variable == 0);
    CHECK(path[1].variable == 1);
}

TEST_CASE("A feasible child LP gives a checked objective cutoff clause", "[mip][conflicts]") {
    LpProblem model;
    model.num_cols = 3;
    model.obj = {0.0, 0.0, 1.0};
    model.obj_offset = 19.0;
    model.col_lower = {0.0, 0.0, 0.0};
    model.col_upper = {1.0, 1.0, 4.0};
    model.col_type = {VarType::Binary, VarType::Binary, VarType::Continuous};
    model.num_rows = 1;
    model.row_lower = {-kInf};
    model.row_upper = {0.0};
    model.matrix = SparseMatrix(1, 3, {{0, 0, 8192.0}, {0, 2, -4096.0}});
    constexpr Real cutoff = 20.0;
    const std::vector<BranchDecision> path{{0, 1.0, false, 1}, {1, 0.0, true, 2}};

    DualSimplexSolver lp;
    lp.setVerbose(false);
    lp.load(model);
    lp.setColBounds(0, 1.0, 1.0);
    lp.setColBounds(1, 0.0, 0.0);
    const auto result = lp.solve();
    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(21.0, 1e-8));
    const auto dual = lp.getDualValues();

    auto clause = path;
    REQUIRE(shortenConflictWithCutoff(model, dual, {}, cutoff, clause));
    REQUIRE(clause.size() == 1);
    CHECK(clause.front().variable == 0);
    CHECK(clause.front().bound == 1.0);
    CHECK_FALSE(clause.front().is_upper);
    const auto infer = [&](Real current_cutoff) {
        return evaluateBoundConflict(clause, {.cutoff = cutoff}, model.col_lower, model.col_upper,
                                     model.col_type, current_cutoff);
    };
    const auto unit = infer(19.5);
    REQUIRE(unit);
    REQUIRE(unit->tightening);
    CHECK(unit->tightening->col == 0);
    CHECK(unit->tightening->is_upper);
    CHECK(unit->tightening->value == 0.0);
    CHECK_FALSE(infer(20.5));
    CHECK_FALSE(infer(kInf));

    auto failed = path;
    const Real invalid = GENERATE(1.0, 0.0, std::numeric_limits<Real>::quiet_NaN());
    CHECK_FALSE(shortenConflictWithCutoff(model, std::array{invalid}, {}, cutoff, failed));
    CHECK_FALSE(shortenConflictWithCutoff(model, dual, {}, result.objective, failed));
    CHECK(failed.size() == path.size());

    auto global = model;
    global.num_rows = 0;
    global.row_lower.clear();
    global.row_upper.clear();
    global.matrix = SparseMatrix(0, 3, {});
    CHECK_FALSE(shortenConflictWithCutoff(global, dual, std::array<Index, 1>{0}, cutoff, failed));
}

TEST_CASE("A combined cutoff proof resolves two propagated bounds to their branch",
          "[mip][conflicts]") {
    // x=1 fixes a and b to 0 by integer propagation, while the LP alone
    // permits a=b=0.5. Pricing either unproved endpoint at its global bound
    // would lose the cutoff. z is unrelated.
    LpProblem model;
    model.num_cols = 4;
    model.obj = {0.0, 0.0, -1.0, -1.0};
    model.col_lower.assign(4, 0.0);
    model.col_upper.assign(4, 1.0);
    model.col_type.assign(4, VarType::Binary);
    model.num_rows = 2;
    model.row_lower.assign(2, -kInf);
    model.row_upper.assign(2, 1.0);
    model.matrix = SparseMatrix(2, 4, {{0, 0, 0.5}, {0, 2, 1.0}, {1, 0, 0.5}, {1, 3, 1.0}});
    constexpr Real cutoff = -0.5;
    std::vector<BranchDecision> path{{0, 1.0, false, 1}, {1, 0.0, true, 2}};

    DualSimplexSolver lp;
    lp.setVerbose(false);
    lp.load(model);
    lp.setColBounds(0, 1.0, 1.0);
    lp.setColBounds(1, 0.0, 0.0);
    CHECK_THAT(lp.solve().objective, WithinAbs(-1.0, 1e-8));
    lp.setColBounds(2, 0.0, 0.0);
    lp.setColBounds(3, 0.0, 0.0);
    REQUIRE(lp.solve().status == Status::Optimal);
    CHECK_THAT(lp.getObjective(), WithinAbs(0.0, 1e-8));
    const auto dual = lp.getDualValues();
    CHECK_FALSE(shortenConflictWithCutoff(model, dual, {}, cutoff, path));

    enum class Reasons { Global, Local, Opaque, StrongerCutoff };
    const auto reasons =
        GENERATE(Reasons::Global, Reasons::Local, Reasons::Opaque, Reasons::StrongerCutoff);
    constexpr Real stronger_cutoff = cutoff - 0.25;
    auto replay_model = model;
    CliqueTable clique;
    DomainPropagator replay;
    if (reasons == Reasons::StrongerCutoff) {
        // The LP proof needs b<=0. Disable the row so the replay gets
        // that bound from a cutoff-scoped clique.
        replay_model.row_upper[1] = kInf;
        auto packing = model;
        packing.matrix = SparseMatrix(2, 4, {{1, 0, 1.0}, {1, 3, 1.0}});
        ConflictGraph graph;
        graph.build(packing);
        clique.build(packing, graph);
    }
    replay.load(replay_model, true, reasons == Reasons::Local ? 1 : -1);
    if (reasons == Reasons::StrongerCutoff) {
        replay.setCliqueTable(&clique, {.cutoff = stronger_cutoff});
    }
    for (const auto& decision : path) {
        replay.setBranchBound(decision.variable, decision.is_upper, decision.bound, decision.level);
    }
    if (reasons == Reasons::Opaque) {
        replay.setBound(3, 0.0, 0.0);  // Imported without a reason.
    }
    REQUIRE(replay.propagate());
    const auto learned =
        checkCombinedLpConflict(model, dual, {}, *replay.boundReasons(), {.cutoff = cutoff}, true);
    if (reasons != Reasons::Global) {
        CHECK_FALSE(learned);
        if (reasons == Reasons::StrongerCutoff) {
            const auto& trail = *replay.boundReasons();
            CHECK(trail.reason(trail.current(3, true)).source == BoundReasonSource::Clique);
            const auto stronger =
                checkCombinedLpConflict(model, dual, {}, trail, {.cutoff = stronger_cutoff}, true);
            REQUIRE(stronger);
            CHECK(stronger->scope.cutoff == stronger_cutoff);
        }
        return;
    }
    REQUIRE(learned);
    REQUIRE(learned->literals.size() == 1);
    CHECK(learned->literals[0].variable == 0);
    CHECK(learned->literals[0].bound == 1.0);
    CHECK_FALSE(learned->literals[0].is_upper);
    CHECK(learned->scope.cutoff == cutoff);
    CHECK(learned->lbd == 1);
    REQUIRE(dual.size() == 2);
    const std::array<Index, 1> local_row{2};
    const auto& trail = *replay.boundReasons();
    CHECK(checkCombinedLpConflict(model, std::array{dual[0], dual[1], 0.0}, local_row, trail,
                                  {.cutoff = cutoff}, true));
    CHECK_FALSE(checkCombinedLpConflict(model, std::array{dual[0], dual[1], -1.0}, local_row, trail,
                                        {.cutoff = cutoff}, true));
    CHECK_FALSE(checkCombinedLpConflict(model, std::array{1.0, 0.0}, {}, *replay.boundReasons(),
                                        {.cutoff = cutoff}, true));
}

TEST_CASE("A Farkas conflict can need propagation as well as the LP rows", "[mip][conflicts]") {
    // x=1 fixes a=b=0 by rounding. Neither of the sum rows can detect the
    // resulting contradiction alone: p+q+r >= 2, p+q+r-a-b <= 1.
    LpProblem model;
    model.num_cols = 6;
    model.obj.assign(6, 0.0);
    model.col_lower.assign(6, 0.0);
    model.col_upper.assign(6, 1.0);
    model.col_type.assign(6, VarType::Binary);
    model.num_rows = 4;
    model.row_lower = {-kInf, -kInf, 2.0, -kInf};
    model.row_upper = {1.0, 1.0, kInf, 1.0};
    model.matrix = SparseMatrix(4, 6,
                                {{0, 0, 0.5},
                                 {0, 1, 1.0},
                                 {1, 0, 0.5},
                                 {1, 2, 1.0},
                                 {2, 3, 1.0},
                                 {2, 4, 1.0},
                                 {2, 5, 1.0},
                                 {3, 1, -1.0},
                                 {3, 2, -1.0},
                                 {3, 3, 1.0},
                                 {3, 4, 1.0},
                                 {3, 5, 1.0}});
    DualSimplexSolver lp;
    lp.setVerbose(false);
    lp.load(model);
    lp.setColBounds(0, 1.0, 1.0);
    REQUIRE(lp.solve().status == Status::Optimal);
    lp.setColBounds(1, 0.0, 0.0);
    lp.setColBounds(2, 0.0, 0.0);
    REQUIRE(lp.solve().status == Status::Infeasible);
    const auto ray = lp.getDualRay();
    REQUIRE_FALSE(ray.empty());
    std::vector<BranchDecision> path{{0, 1.0, false, 1}};
    CHECK_FALSE(shortenConflictWithRay(model, ray, {}, model.col_lower, model.col_upper, path));

    DomainPropagator replay;
    replay.load(model, true);
    replay.setBranchBound(0, false, 1.0, 1);
    REQUIRE(replay.propagate());
    const auto learned = checkCombinedLpConflict(model, ray, {}, *replay.boundReasons(), {}, false);
    REQUIRE(learned);
    REQUIRE(learned->literals.size() == 1);
    CHECK(learned->literals.front().variable == 0);
    CHECK(learned->literals.front().bound == 1.0);
    CHECK_FALSE(learned->literals.front().is_upper);
    CHECK(learned->scope.cutoff == kInf);
}

TEST_CASE("MipSolver learns from a feasible child above the incumbent", "[mip][conflicts]") {
    // y >= 4*x - 2 and y >= 1 - 2*x give x=0.5 at the root. The x=0
    // child gives the incumbent, while the x=1 child's feasible LP is worse.
    LpProblem model;
    model.num_cols = 2;
    model.obj = {0.0, -1.0};
    model.obj_offset = 19.0;
    model.sense = Sense::Maximize;
    model.col_lower = {0.0, 0.0};
    model.col_upper = {1.0, 4.0};
    model.col_type = {VarType::Binary, VarType::Continuous};
    model.num_rows = 2;
    model.row_lower = {-2.0, 1.0};
    model.row_upper = {kInf, kInf};
    model.matrix = SparseMatrix(2, 2, {{0, 0, -4.0}, {0, 1, 1.0}, {1, 0, 2.0}, {1, 1, 1.0}});

    MipSolver solver;
    solver.setVerbose(false);
    solver.setPresolve(false);
    solver.setCutsEnabled(false);
    solver.setTreePresolveEnabled(false);
    solver.setStrongBranchingEnabled(false);
    solver.setNumThreads(1);
    solver.load(model);
    const auto baseline = solver.solve();
    REQUIRE(baseline.status == Status::Optimal);
    CHECK_THAT(baseline.objective, WithinAbs(18.0, 1e-8));
    CHECK(solver.getConflictStats().lp_cutoff_proofs_learned == 0);

    solver.setConflictLpProofsEnabled(true);
    solver.setConflictWatchesEnabled(true);
    const auto result = solver.solve();
    REQUIRE(result.status == Status::Optimal);
    CHECK(result.objective == baseline.objective);
    CHECK(solver.getConflictStats().lp_cutoff_proofs_learned == 1);
    CHECK(solver.getConflictStats().lp_infeasible_conflicts == 0);
}

TEST_CASE("MIP conflict rays map a promoted row past an unused local row", "[mip][conflicts]") {
    LpProblem global;
    global.num_cols = 2;
    global.num_rows = 2;
    global.col_lower = {0.0, 0.0};
    global.col_upper = {1.0, 1.0};
    global.row_lower = {-kInf, 0.5};
    global.row_upper = {1.0, kInf};
    global.matrix = SparseMatrix(2, 2, {{0, 1, 1.0}, {1, 0, 1.0}});
    // The second LP row is local. The last row is the promoted x>=0.5.
    const std::array<Real, 3> ray{0.0, 0.0, -1.0};
    const std::array<Index, 1> local_rows{1};
    std::vector<BranchDecision> path{{0, 0.0, true}, {1, 0.0, true}};

    REQUIRE(
        shortenConflictWithRay(global, ray, local_rows, global.col_lower, global.col_upper, path));
    REQUIRE(path.size() == 1);
    CHECK(path.front().variable == 0);
}

TEST_CASE("MipSolver: pre-LP propagation prunes a child and restores sibling bounds",
          "[mip][domain]") {
    // The root LP takes x=0.5, y=0.75, z=0. The x=0 child has a feasible LP
    // with y=0.5, but cannot satisfy integrality. The x=1 sibling is feasible.
    LpProblem lp;
    lp.num_cols = 3;
    lp.obj = {1.0, 0.0, 1.5};
    lp.col_lower = {0.0, 0.0, 0.0};
    lp.col_upper = {1.0, 1.0, 1.0};
    lp.col_type = {VarType::Binary, VarType::Binary,
                   GENERATE(VarType::Binary, VarType::Continuous)};
    lp.num_rows = 2;
    const Real parity_rhs = GENERATE(1.0, 1.0 - 2e-9);
    lp.row_lower = {parity_rhs, 0.5};
    lp.row_upper = {parity_rhs, kInf};
    lp.matrix = SparseMatrix(2, 3, {{0, 0, -1.0}, {0, 1, 2.0}, {1, 0, 1.0}, {1, 2, 1.0}});

    auto down_child = lp;
    down_child.col_upper[0] = 0.0;
    DualSimplexSolver relaxation;
    relaxation.load(down_child);
    REQUIRE(relaxation.solve().status == Status::Optimal);
    CHECK_THAT(relaxation.getPrimalValues()[1], WithinAbs(0.5, 1e-8));

    MipSolver solver;
    solver.setVerbose(false);
    solver.setPresolve(false);
    solver.setCutsEnabled(false);
    solver.setTreePresolveEnabled(false);
    solver.setStrongBranchingEnabled(false);
    solver.setNumThreads(1);
    solver.load(lp);

    const auto baseline = solver.solve();
    REQUIRE(baseline.status == Status::Optimal);
    const auto baseline_stats = solver.getLpStats();
    const auto baseline_branching = solver.getBranchingStats();
    CHECK(baseline_stats.pre_lp_propagation_calls == 0);
    CHECK(baseline_stats.pre_lp_propagation_pruned == 0);

    solver.setPreLpPropagationEnabled(true);
    solver.setDirectionalInferenceEnabled(true);
    for (int repeat = 0; repeat < 2; ++repeat) {
        const auto screened = solver.solve();
        REQUIRE(screened.status == Status::Optimal);
        CHECK_THAT(screened.objective, WithinAbs(1.0, 1e-8));
        REQUIRE(screened.solution.size() == 3);
        CHECK_THAT(screened.solution[0], WithinAbs(1.0, 1e-8));
        CHECK_THAT(screened.solution[1], WithinAbs(1.0, 1e-8));
        CHECK_THAT(screened.solution[2], WithinAbs(0.0, 1e-8));
        const auto& stats = solver.getLpStats();
        CHECK(stats.pre_lp_propagation_calls == 2);
        CHECK(stats.pre_lp_propagation_pruned == 1);
        const auto& inference = solver.getBranchInferenceStats();
        REQUIRE(inference.size() == 3);
        // The root chooses y for branching. Its down side implies x=-1.
        CHECK(inference[1].down.cutoffs == 1);
        CHECK(inference[1].down.samples == 0);
        CHECK(inference[1].down.tightened_columns == 0);
        CHECK(inference[1].down.skipped == 0);
        CHECK(inference[1].up.cutoffs == 0);
        CHECK(inference[1].up.samples == 1);
        CHECK(stats.warm_starts + stats.cold_starts ==
              baseline_stats.warm_starts + baseline_stats.cold_starts - 1);
        CHECK(solver.getConflictStats().bound_infeasible_conflicts == 1);
        const auto& branching = solver.getBranchingStats();
        CHECK(branching.pseudocost_updates_tree + branching.pseudocost_tree_skipped ==
              baseline_branching.pseudocost_updates_tree +
                  baseline_branching.pseudocost_tree_skipped - 1);
    }

    solver.setPreLpPropagationEnabled(false);
    solver.setDirectionalInferenceEnabled(false);
    const auto explicitly_off = solver.solve();
    CHECK(explicitly_off.status == baseline.status);
    CHECK(explicitly_off.objective == baseline.objective);
    CHECK(explicitly_off.nodes == baseline.nodes);
    CHECK(explicitly_off.lp_iterations == baseline.lp_iterations);
    CHECK(solver.getLpStats().pre_lp_propagation_calls == 0);
    CHECK(solver.getBranchInferenceStats().empty());
}

TEST_CASE("MipSolver: directional inference counts distinct columns at the first pass",
          "[mip][domain]") {
    // The root takes x=0.5, y=z=0. Down fixes x=0 and tightens both sides of y.
    // Up fixes x=1 and tightens z from below. Each direction infers one column.
    LpProblem lp;
    lp.num_cols = 3;
    lp.obj = {1.0, 1.5, 0.2};
    lp.col_lower = {0.0, 0.0, 0.0};
    lp.col_upper = {1.0, 1.0, 1.0};
    lp.col_type = {VarType::Binary, VarType::Continuous, VarType::Continuous};
    lp.num_rows = 3;
    lp.row_lower = {0.5, -kInf, -0.5};
    lp.row_upper = {kInf, 0.75, kInf};
    lp.matrix = SparseMatrix(
        3, 3, {{0, 0, 1.0}, {0, 1, 1.0}, {1, 0, -0.5}, {1, 1, 1.0}, {2, 0, -1.0}, {2, 2, 1.0}});

    MipSolver solver;
    solver.setVerbose(false);
    solver.setPresolve(false);
    solver.setCutsEnabled(false);
    solver.setNumThreads(1);
    solver.setPreLpPropagationEnabled(true);
    solver.load(lp);

    const auto baseline = solver.solve();
    REQUIRE(baseline.status == Status::Optimal);
    REQUIRE(baseline.nodes == 3);
    CHECK(solver.getBranchInferenceStats().empty());
    CHECK(solver.getBranchingStats().strong_branch_probes == 2);

    solver.setDirectionalInferenceEnabled(true);
    const auto& inference = solver.getBranchInferenceStats();
    for (const bool coverage : {false, true}) {
        solver.setInferenceBranchCoverageEnabled(coverage);
        const auto measured = solver.solve();
        CHECK(measured.status == baseline.status);
        CHECK(measured.objective == baseline.objective);
        CHECK(measured.solution == baseline.solution);
        CHECK(measured.nodes == baseline.nodes);
        CHECK(measured.lp_iterations == baseline.lp_iterations);
        CHECK(measured.work_units == baseline.work_units);
        REQUIRE(inference.size() == 3);
        for (const auto& direction : {inference[0].down, inference[0].up}) {
            CHECK(direction.samples == 1);
            CHECK(direction.tightened_columns == 1);
            CHECK(direction.cutoffs == 0);
            CHECK(direction.skipped == 0);
        }
        for (Index j : {1, 2}) {
            CHECK(inference[j].down.samples == 0);
            CHECK(inference[j].up.samples == 0);
        }
    }

    solver.setPreLpPropagationEnabled(false);
    const auto skipped = solver.solve();
    REQUIRE(skipped.status == Status::Optimal);
    REQUIRE(skipped.nodes == 3);
    REQUIRE(inference.size() == 3);
    for (const auto& direction : {inference[0].down, inference[0].up}) {
        CHECK(direction.samples == 0);
        CHECK(direction.tightened_columns == 0);
        CHECK(direction.cutoffs == 0);
        CHECK(direction.skipped == 1);
    }

    lp.sos_constraints = {{LpProblem::SosConstraint::Type::Sos2, {0, 1, 2}, {}, "sos2"}};
    solver.setSos2NativeBranching(true);
    solver.setNodeLimit(0);
    solver.load(lp);
    REQUIRE(solver.solve().status == Status::NodeLimit);
    CHECK(solver.getBranchInferenceStats().empty());
}

TEST_CASE("MipSolver: inference coverage observes serial pseudocost decisions only",
          "[mip][branching]") {
    auto lp = buildRcFixingKnapsackMip();

    MipSolver solver;
    solver.setVerbose(false);
    solver.setPresolve(false);
    solver.setSymmetryEnabled(false);
    solver.setCutsEnabled(false);
    solver.setStrongBranchingEnabled(false);
    solver.setPreLpPropagationEnabled(true);
    solver.setNumThreads(1);
    solver.load(lp);

    const auto baseline = solver.solve();
    REQUIRE(baseline.status == Status::Optimal);
    REQUIRE(baseline.nodes > 5);
    const auto& coverage = solver.getBranchInferenceCoverageStats();
    CHECK(coverage.fallback.decisions == 0);
    solver.setInferenceBranchCoverageEnabled(true);
    const auto measured = solver.solve();
    CHECK(measured.status == baseline.status);
    CHECK(measured.objective == baseline.objective);
    CHECK(measured.solution == baseline.solution);
    CHECK(measured.best_bound == baseline.best_bound);
    CHECK(measured.nodes == baseline.nodes);
    CHECK(measured.lp_iterations == baseline.lp_iterations);
    CHECK(measured.work_units == baseline.work_units);
    CHECK(coverage.fallback.decisions > 0);
    CHECK(std::cmp_less(coverage.fallback.decisions, solver.getBranchingStats().selections));
    CHECK(solver.getBranchInferenceStats().empty());

    solver.setNumThreads(2);
    const auto two_threads = solver.solve();
    CHECK(two_threads.status == baseline.status);
    CHECK(two_threads.objective == baseline.objective);
#ifdef MIPX_HAS_TBB
    CHECK(coverage.reliable.decisions == 0);
    CHECK(coverage.fallback.decisions == 0);
#else
    CHECK(coverage.fallback.decisions > 0);
#endif
}

TEST_CASE("MipSolver: pre-LP propagation leaves an inconclusive node LP intact", "[mip][domain]") {
    // x is binary, z continuous. Both children of the root's x=0.5 have an
    // LP solution. Propagation can tighten z in the down child, but that
    // tightening is deliberately not applied to the LP.
    LpProblem lp;
    lp.num_cols = 2;
    lp.obj = {1.0, 1.5};
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {1.0, 1.0};
    lp.col_type = {VarType::Binary, VarType::Continuous};
    lp.num_rows = 1;
    lp.row_lower = {0.5};
    lp.row_upper = {kInf};
    lp.matrix = SparseMatrix(1, 2, {{0, 0, 1.0}, {0, 1, 1.0}});

    MipSolver solver;
    solver.setVerbose(false);
    solver.setPresolve(false);
    solver.setCutsEnabled(false);
    solver.setStrongBranchingEnabled(false);
    solver.load(lp);

    const auto baseline = solver.solve();
    REQUIRE(baseline.status == Status::Optimal);
    CHECK_THAT(baseline.objective, WithinAbs(0.75, 1e-8));
    const auto baseline_stats = solver.getLpStats();

    solver.setPreLpPropagationEnabled(true);
    const auto screened = solver.solve();
    CHECK(screened.status == baseline.status);
    CHECK(screened.objective == baseline.objective);
    CHECK(screened.nodes == baseline.nodes);
    CHECK(screened.lp_iterations == baseline.lp_iterations);
    const auto& stats = solver.getLpStats();
    CHECK(stats.pre_lp_propagation_calls > 0);
    CHECK(stats.pre_lp_propagation_pruned == 0);
    CHECK(stats.warm_starts == baseline_stats.warm_starts);
    CHECK(stats.cold_starts == baseline_stats.cold_starts);
}

TEST_CASE("MipSolver: depth-scheduled row fixing reaches the first LP and stays local",
          "[mip][domain]") {
    // The root LP takes x=0.5, z=0. The down child requires z=1, although
    // its unpropagated LP takes z=0.5. The up child wins with z=0.
    LpProblem lp;
    lp.num_cols = 2;
    lp.obj = {1.0, 1.5};
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {1.0, 1.0};
    lp.col_type = {VarType::Binary, VarType::Binary};
    lp.num_rows = 1;
    lp.row_lower = {0.5};
    lp.row_upper = {kInf};
    lp.matrix = SparseMatrix(1, 2, {{0, 0, 1.0}, {0, 1, 1.0}});

    MipSolver solver;
    solver.setVerbose(false);
    solver.setPresolve(false);
    solver.setCutsEnabled(false);
    solver.setStrongBranchingEnabled(false);
    solver.setTreePresolveAutoTuning(false);
    solver.setTreePresolveControls(1, 1, 1);
    solver.load(lp);

    const auto baseline = solver.solve();
    REQUIRE(baseline.status == Status::Optimal);
    CHECK(solver.getLpStats().pre_lp_propagation_fixings == 0);
    CHECK(solver.getTreePresolveStats().lp_resolves == 1);
    const auto baseline_branching = solver.getBranchingStats();

    solver.setPreLpPropagationBoundsEnabled(true);
    const auto tightened = solver.solve();
    REQUIRE(tightened.status == Status::Optimal);
    CHECK_THAT(tightened.objective, WithinAbs(1.0, 1e-8));
    REQUIRE(tightened.solution.size() == 2);
    CHECK_THAT(tightened.solution[0], WithinAbs(1.0, 1e-8));
    CHECK_THAT(tightened.solution[1], WithinAbs(0.0, 1e-8));
    const auto& stats = solver.getLpStats();
    CHECK(stats.pre_lp_propagation_calls == 2);
    CHECK(stats.pre_lp_propagation_fixings == 1);
    CHECK(stats.warm_starts + stats.cold_starts == 2);
    CHECK(solver.getTreePresolveStats().lp_resolves == 0);
    const auto& branching = solver.getBranchingStats();
    CHECK(branching.pseudocost_updates_tree + branching.pseudocost_tree_skipped ==
          baseline_branching.pseudocost_updates_tree + baseline_branching.pseudocost_tree_skipped -
              1);

    solver.setTreePresolveControls(1, 1, 2);
    REQUIRE(solver.solve().status == Status::Optimal);
    CHECK(solver.getLpStats().pre_lp_propagation_calls == 0);
    CHECK(solver.getLpStats().pre_lp_propagation_fixings == 0);
    CHECK(solver.getTreePresolveStats().lp_resolves == 1);
}

// ---------------------------------------------------------------------------
// Helper: set packing on an odd cycle (5-hole)
// min -x0 -x1 -x2 -x3 -x4  s.t. x_i + x_{i+1 mod 5} <= 1, all binary
// LP relaxation optimum is the fractional point (.5, ..., .5) with value -2.5;
// the MIP optimum is -2 and only branching closes the gap.
// ---------------------------------------------------------------------------

static LpProblem buildOddCycleSetPackingMip() {
    LpProblem lp;
    lp.name = "odd_cycle_set_packing";
    lp.sense = Sense::Minimize;
    lp.num_cols = 5;
    lp.obj.assign(5, -1.0);
    lp.col_lower.assign(5, 0.0);
    lp.col_upper.assign(5, 1.0);
    lp.col_type.assign(5, VarType::Binary);
    lp.col_names = {"x0", "x1", "x2", "x3", "x4"};

    lp.num_rows = 5;
    lp.row_lower.assign(5, -kInf);
    lp.row_upper.assign(5, 1.0);
    lp.row_names = {"c0", "c1", "c2", "c3", "c4"};

    std::vector<Triplet> trips;
    for (Index i = 0; i < 5; ++i) {
        trips.push_back({i, i, 1.0});
        trips.push_back({i, (i + 1) % 5, 1.0});
    }
    lp.matrix = SparseMatrix(5, 5, std::move(trips));
    return lp;
}

// ---------------------------------------------------------------------------
// Helper: single conflict row whose root LP stays fractional
// min -3x1 -2x2 -0.1x3  s.t. x1 + x2 + x3 <= 1.5, all binary
// Any two variables together exceed the capacity, so the MIP optimum is -3,
// while the root LP relaxation sits at x1 = 1, x2 = 0.5.
// ---------------------------------------------------------------------------

static LpProblem buildBinaryConflictRowMip() {
    LpProblem lp;
    lp.name = "binary_conflict_row";
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

TEST_CASE("MipSolver: clique table is built at root and preserves optimum", "[mip][clique]") {
    const auto lp = buildOddCycleSetPackingMip();

    MipSolver with_cliques;
    with_cliques.setVerbose(false);
    with_cliques.setCliqueTableEnabled(true);
    with_cliques.load(lp);
    const auto with_result = with_cliques.solve();

    MipSolver without_cliques;
    without_cliques.setVerbose(false);
    without_cliques.setCliqueTableEnabled(false);
    without_cliques.load(lp);
    const auto without_result = without_cliques.solve();

    REQUIRE(with_result.status == Status::Optimal);
    REQUIRE(without_result.status == Status::Optimal);
    CHECK_THAT(with_result.objective, WithinAbs(without_result.objective, 1e-9));
    CHECK_THAT(with_result.objective, WithinAbs(-2.0, 1e-9));

    const MipCliqueStats stats = with_cliques.getCliqueStats();
    CHECK(stats.enabled);
    CHECK(stats.num_binaries >= 2);
    CHECK(stats.num_conflict_edges >= 1);
    CHECK(stats.num_cliques > 0);
    CHECK(stats.build_time_seconds >= 0.0);

    const MipCliqueStats off_stats = without_cliques.getCliqueStats();
    CHECK_FALSE(off_stats.enabled);
    CHECK(off_stats.node_attachments == 0);
    CHECK_FALSE(off_stats.skipped_too_large);
    CHECK(off_stats.num_binaries == 0);
    CHECK(off_stats.num_cliques == 0);
    CHECK(off_stats.cliques_from_cuts == 0);

    // A second solve() on the same object rebuilds the table from scratch
    // instead of accumulating on top of the first solve's state.
    const auto repeat_result = with_cliques.solve();
    REQUIRE(repeat_result.status == Status::Optimal);
    CHECK_THAT(repeat_result.objective, WithinAbs(with_result.objective, 1e-9));
    const MipCliqueStats repeat_stats = with_cliques.getCliqueStats();
    CHECK(repeat_stats.num_cliques == stats.num_cliques);
    CHECK(repeat_stats.num_conflict_edges == stats.num_conflict_edges);
    CHECK(repeat_stats.cliques_from_cuts == stats.cliques_from_cuts);
}

TEST_CASE("MipSolver: clique table is built over Integer [0, 1] columns", "[mip][clique]") {
    // The same 0/1 model with its columns typed the way MIPLIB's MARKER INTORG
    // integers read in (issue #218): the table must not come out empty.
    auto lp = buildOddCycleSetPackingMip();
    for (Index j = 0; j < lp.num_cols; ++j) {
        REQUIRE(lp.col_lower[j] == 0.0);
        REQUIRE(lp.col_upper[j] == 1.0);
        lp.col_type[j] = VarType::Integer;
    }

    MipSolver solver;
    solver.setVerbose(false);
    solver.setCliqueTableEnabled(true);
    solver.load(lp);
    const auto result = solver.solve();
    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(-2.0, 1e-9));

    const MipCliqueStats stats = solver.getCliqueStats();
    CHECK(stats.num_binaries == lp.num_cols);
    CHECK(stats.num_conflict_edges >= 1);
    CHECK(stats.num_cliques > 0);
}

TEST_CASE("MipSolver: root cuts feed the clique table", "[mip][clique]") {
    const auto lp = buildBinaryConflictRowMip();

    MipSolver solver;
    solver.setVerbose(false);
    // Keep the root LP fractional so the cut loop actually runs.
    solver.setPresolve(false);
    solver.setCutsEnabled(true);
    solver.setCliqueTableEnabled(true);
    solver.setCutFamilyEnabled(CutFamily::Gomory, false);
    solver.setCutFamilyEnabled(CutFamily::Mir, false);
    solver.setCutFamilyEnabled(CutFamily::Cover, false);
    solver.setCutFamilyEnabled(CutFamily::ImpliedBound, false);
    solver.setCutFamilyEnabled(CutFamily::ZeroHalf, false);
    solver.setCutFamilyEnabled(CutFamily::Mixing, false);
    solver.setCutFamilyEnabled(CutFamily::Clique, true);
    solver.load(lp);
    const auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(-3.0, 1e-9));

    const MipCliqueStats stats = solver.getCliqueStats();
    CHECK(stats.enabled);
    CHECK(stats.num_binaries == 3);
    // The single capacity row makes every pair conflict, so the root table
    // already holds the maximal clique, and the root cut loop hands at least
    // one unit-coefficient rhs-1 cut back to it.
    CHECK(stats.num_cliques >= 1);
    CHECK(stats.cliques_from_cuts >= 1);
}

// ---------------------------------------------------------------------------
// Helper: a 12-cycle of set-packing rows (which give the conflict graph real
// edges, hence real cliques) plus an equality row forcing sum(x) = 4.5. The
// equality is integer-infeasible, so the search cannot close at the root and
// must descend far enough for in-tree propagation to run. This is what makes
// the clique table observable at node level.
// ---------------------------------------------------------------------------

static LpProblem buildCliquePropagationMip() {
    constexpr Index n = 12;
    LpProblem lp;
    lp.name = "clique_propagation_mip";
    lp.sense = Sense::Minimize;
    lp.num_cols = n;
    lp.obj.assign(n, -1.0);
    lp.col_lower.assign(n, 0.0);
    lp.col_upper.assign(n, 1.0);
    lp.col_type.assign(n, VarType::Binary);

    lp.num_rows = n + 1;
    lp.row_lower.assign(n, -kInf);
    lp.row_upper.assign(n, 1.0);
    lp.row_lower.push_back(4.5);
    lp.row_upper.push_back(4.5);

    std::vector<Triplet> trips;
    for (Index i = 0; i < n; ++i) {
        trips.push_back({i, i, 1.0});
        trips.push_back({i, (i + 1) % n, 1.0});
    }
    for (Index j = 0; j < n; ++j) {
        trips.push_back({n, j, 1.0});
    }
    lp.matrix = SparseMatrix(n + 1, n, std::move(trips));
    return lp;
}

TEST_CASE("MipSolver: clique table reaches the node propagators", "[mip][clique]") {
    const auto lp = buildCliquePropagationMip();

    MipSolver solver;
    solver.setVerbose(false);
    solver.setCutsEnabled(false);
    solver.setPresolve(false);
    solver.setTreePresolveEnabled(true);
    solver.setCliqueTableEnabled(true);
    solver.setNodeLimit(300);
    solver.load(lp);
    const auto result = solver.solve();

    const MipCliqueStats stats = solver.getCliqueStats();
    REQUIRE(stats.enabled);
    REQUIRE(stats.num_cliques > 0);
    // Criterion 3: without the two setCliqueTable calls in processNode this is
    // zero and every other assertion in the clique tests still passes.
    CHECK(stats.node_attachments > 0);
    CHECK((result.status == Status::Infeasible || result.status == Status::NodeLimit ||
           result.status == Status::Optimal));

    MipSolver off;
    off.setVerbose(false);
    off.setCutsEnabled(false);
    off.setPresolve(false);
    off.setTreePresolveEnabled(true);
    off.setCliqueTableEnabled(false);
    off.setNodeLimit(300);
    off.load(lp);
    const auto off_result = off.solve();
    CHECK(off.getCliqueStats().node_attachments == 0);
    CHECK(off_result.status == result.status);
}

TEST_CASE("MipSolver: stable search profile is reproducible", "[mip][search]") {
    auto lp = buildSearchStagnationMip();

    MipSolver solver_a;
    solver_a.setVerbose(false);
    solver_a.setCutsEnabled(false);
    solver_a.setPresolve(false);
    solver_a.setSearchProfile(SearchProfile::Stable);
    solver_a.setHeuristicMode(HeuristicRuntimeMode::Deterministic);
    solver_a.setHeuristicSeed(42);
    solver_a.setNodeLimit(300);
    solver_a.load(lp);
    const auto a = solver_a.solve();

    MipSolver solver_b;
    solver_b.setVerbose(false);
    solver_b.setCutsEnabled(false);
    solver_b.setPresolve(false);
    solver_b.setSearchProfile(SearchProfile::Stable);
    solver_b.setHeuristicMode(HeuristicRuntimeMode::Deterministic);
    solver_b.setHeuristicSeed(42);
    solver_b.setNodeLimit(300);
    solver_b.load(lp);
    const auto b = solver_b.solve();

    REQUIRE(a.status == b.status);
    CHECK(a.nodes == b.nodes);
    CHECK_THAT(a.work_units, WithinAbs(b.work_units, 1e-9));
}

TEST_CASE("MipSolver: aggressive search profile switches and restarts", "[mip][search]") {
    auto lp = buildSearchStagnationMip();

    MipSolver solver;
    solver.setVerbose(false);
    solver.setCutsEnabled(false);
    solver.setPresolve(false);
    solver.setSearchProfile(SearchProfile::Aggressive);
    solver.setRestartsEnabled(true);
    solver.setRestartControls(8, 2);
    solver.setNodeLimit(300);
    solver.load(lp);
    const auto result = solver.solve();

    CHECK((result.status == Status::Infeasible || result.status == Status::NodeLimit ||
           result.status == Status::TimeLimit));
    const auto& sstats = solver.getSearchStats();
    CHECK(sstats.policy_switches >= 1);
    CHECK(sstats.restarts >= 1);
    CHECK(sstats.restart_nodes_dropped == 0);
    CHECK(sstats.strong_budget_updates >= 1);
    CHECK_FALSE(solver.getSerialNodePolicyStats().policy.has_value());

    solver.setSerialNodePolicyOverride(NodePolicy::DepthBiased);
    const auto pinned = solver.solve();
    const auto& policy_stats = solver.getSerialNodePolicyStats();
    REQUIRE(policy_stats.policy == NodePolicy::DepthBiased);
    CHECK(sstats.policy_switches == 1);
    CHECK(policy_stats.restarts == sstats.restarts);
    CHECK(policy_stats.restarts >= 1);
    CHECK(sstats.restart_nodes_dropped == 0);
    CHECK(policy_stats.nodes_selected == pinned.nodes - 1);
}

TEST_CASE("MipSolver: geometric serial restart spacing starts fresh on every solve",
          "[mip][search]") {
    MipSolver solver;
    solver.setVerbose(false);
    solver.setNumThreads(1);
    solver.setCutsEnabled(false);
    solver.setPresolve(false);
    solver.setTreePresolveEnabled(false);
    solver.setSymmetryEnabled(false);
    solver.setRestartsEnabled(true);
    solver.setSerialNodePolicyOverride(NodePolicy::BestFirst);
    solver.setRestartControls(8, 2);
    solver.load(buildSearchStagnationMip());

    const auto baseline = solver.solve();
    REQUIRE(baseline.status == Status::Infeasible);
    REQUIRE(solver.getSearchStats().restarts > 3);

    solver.setGeometricRestartsEnabled(true);
    // The root counts towards the node limit. A limit at the boundary
    // prevents another loop iteration, so the first restart needs limit 10.
    constexpr std::array<std::pair<Int, Int>, 6> cases{
        {{9, 0}, {10, 1}, {25, 1}, {26, 2}, {57, 2}, {58, 3}}};
    for (const auto& [limit, expected_restarts] : cases) {
        CAPTURE(limit);
        solver.setNodeLimit(limit);
        const auto result = solver.solve();
        REQUIRE(result.status == Status::NodeLimit);
        CHECK(result.nodes == limit);
        CHECK(solver.getSearchStats().restarts == expected_restarts);
        CHECK(solver.getSearchStats().restart_nodes_dropped == 0);
    }

    solver.setNodeLimit(baseline.nodes + 1);
    const auto spaced = solver.solve();
    CHECK(spaced.status == baseline.status);
    CHECK(spaced.nodes == baseline.nodes);
    CHECK(solver.getSearchStats().restarts >= 3);
    CHECK(solver.getSearchStats().restart_nodes_dropped == 0);

    solver.setRestartControls(8, baseline.nodes);
    CHECK(solver.solve().status == baseline.status);
    CHECK(solver.getSearchStats().restarts == 0);

    solver.setRestartControls(8, 2);
    solver.setSearchProfile(SearchProfile::Stable);
    CHECK(solver.solve().status == baseline.status);
    CHECK(solver.getSearchStats().restarts == 0);
}

TEST_CASE("MipSolver: conflict restarts preserve open nodes and learned state",
          "[mip][conflicts][search]") {
    MipSolver solver;
    solver.setVerbose(false);
    solver.setNumThreads(1);
    solver.setCutsEnabled(false);
    solver.setPresolve(false);
    solver.setTreePresolveEnabled(false);
    solver.setSymmetryEnabled(false);
    solver.setBoundConflictLearningEnabled(true);
    solver.setGapTolerance(0.0);
    // Fix the queue order so that a policy reset consumes activity without
    // changing which nodes, clauses, and pseudocost observations we see.
    solver.setSerialNodePolicyOverride(NodePolicy::BestFirst);
    solver.setRestartControls(10000, 2);
    solver.load(buildSearchStagnationMipWithEscape());

    const auto both_off = solver.solve();
    REQUIRE(both_off.status == Status::Optimal);
    CHECK(both_off.objective == 1.0);
    CHECK(solver.getSearchStats().restarts == 0);

    solver.setConflictActivityBranchingEnabled(true);
    const auto baseline = solver.solve();
    const auto conflicts = solver.getConflictStats();
    const auto branching = solver.getBranchingStats();
    REQUIRE(baseline.status == both_off.status);
    CHECK(baseline.objective == both_off.objective);
    REQUIRE(conflicts.bound_conflicts_learned > 2);
    REQUIRE(conflicts.bound_conflict_tightenings > 0);
    REQUIRE(branching.pseudocost_updates_tree > 0);
    REQUIRE(branching.pseudocost_hits > 0);
    CHECK(solver.getSearchStats().restarts == 0);

    solver.setConflictRestartThreshold(2);
    const auto restarted = solver.solve();
    const auto search = solver.getSearchStats();
    REQUIRE(search.conflict_restarts > 1);
    CHECK(search.restarts == search.conflict_restarts);
    // At most one conflict is published per node in this fixture. Two
    // publications with intervening decay cannot reach the threshold of two.
    CHECK(search.conflict_restarts <= conflicts.learned / 3);
    CHECK(search.last_conflict_restart_activity > 2.0);
    CHECK(search.last_conflict_restart_activity < 3.0);
    CHECK(search.restart_nodes_dropped == 0);
    CHECK(restarted.status == baseline.status);
    CHECK(restarted.objective == baseline.objective);
    CHECK(restarted.nodes == baseline.nodes);
    CHECK(restarted.lp_iterations == baseline.lp_iterations);
    CHECK(solver.getConflictStats().bound_conflicts_learned == conflicts.bound_conflicts_learned);
    CHECK(solver.getConflictStats().bound_conflict_tightenings ==
          conflicts.bound_conflict_tightenings);
    CHECK(solver.getConflictStats().bound_conflict_cutoffs == conflicts.bound_conflict_cutoffs);
    CHECK(solver.getBranchingStats().pseudocost_updates_tree == branching.pseudocost_updates_tree);
    CHECK(solver.getBranchingStats().pseudocost_hits == branching.pseudocost_hits);
    CHECK(solver.getBranchingStats().strong_branch_probes == branching.strong_branch_probes);

    // Conflict activity alone fired several times above. A minimum interval
    // longer than the entire search defers all of those restarts.
    REQUIRE(baseline.nodes < 10000);
    solver.setGeometricRestartsEnabled(true);
    const auto deferred = solver.solve();
    CHECK(deferred.status == baseline.status);
    CHECK(deferred.objective == baseline.objective);
    CHECK(deferred.nodes == baseline.nodes);
    CHECK(solver.getSearchStats().restarts == 0);
    CHECK(solver.getConflictStats().bound_conflicts_learned == conflicts.bound_conflicts_learned);

    // Allow only the conflict trigger. Give the fixture time to publish
    // conflicts before checking each side of the first spacing boundary.
    solver.setRestartControls(32, 2);
    solver.setConflictRestartThreshold(1);
    solver.setNodeLimit(33);
    solver.setGeometricRestartsEnabled(false);
    REQUIRE(solver.solve().status == Status::NodeLimit);
    CHECK(solver.getSearchStats().conflict_restarts > 0);
    solver.setGeometricRestartsEnabled(true);
    REQUIRE(solver.solve().status == Status::NodeLimit);
    CHECK(solver.getSearchStats().conflict_restarts == 0);
    solver.setNodeLimit(34);
    REQUIRE(solver.solve().status == Status::NodeLimit);
    CHECK(solver.getSearchStats().conflict_restarts == 1);
    CHECK(solver.getSearchStats().restart_nodes_dropped == 0);

    solver.setGeometricRestartsEnabled(false);
    solver.setRestartControls(10000, 2);
    solver.setConflictRestartThreshold(2);
    solver.setNodeLimit(std::numeric_limits<Int>::max());

    solver.setSerialNodePolicyOverride(std::nullopt);
    const auto scheduled = solver.solve();
    CHECK(scheduled.status == both_off.status);
    CHECK(scheduled.objective == both_off.objective);
    CHECK(solver.getSearchStats().conflict_restarts > 0);

    solver.setSearchProfile(SearchProfile::Stable);
    CHECK(solver.solve().status == Status::Optimal);
    CHECK(solver.getSearchStats().restarts == 0);
    CHECK(solver.getSearchStats().conflict_restarts == 0);
    CHECK(solver.getSearchStats().last_conflict_restart_activity == 0.0);
}

TEST_CASE("MipSolver: serial node policy attributes work and resets on solve", "[mip][search]") {
    const auto policy =
        GENERATE(NodePolicy::BestFirst, NodePolicy::BestEstimate, NodePolicy::DepthBiased);
    MipSolver solver;
    solver.setVerbose(false);
    solver.setCutsEnabled(false);
    solver.setPresolve(false);
    solver.setNumThreads(1);
    solver.setGapTolerance(0.0);
    solver.setSerialNodePolicyOverride(policy);
    solver.load(buildRootRoundingMip());

    for (int repeat = 0; repeat < 2; ++repeat) {
        const auto result = solver.solve();
        REQUIRE(result.status == Status::Optimal);
        CHECK(result.objective == 0.0);
        CHECK(result.nodes == 3);
        const auto& stats = solver.getSerialNodePolicyStats();
        CHECK(stats.policy == policy);
        CHECK(stats.nodes_selected == 2);
        CHECK(stats.nodes_closed == 2);
        CHECK(stats.nodes_pruned_from_queue == 0);
        CHECK(stats.incumbents_accepted == 0);  // The incumbent was available at the root.
        CHECK(stats.restarts == 0);
        CHECK_THAT(stats.proof_bound_improvement, WithinAbs(0.49, 1e-9));
        CHECK(stats.proof_completed);
    }

    solver.setNodeLimit(1);
    const auto limited = solver.solve();
    REQUIRE(limited.status == Status::NodeLimit);
    const auto& limited_stats = solver.getSerialNodePolicyStats();
    CHECK_FALSE(limited_stats.policy.has_value());
    CHECK(limited_stats.nodes_selected == 0);
    CHECK(limited_stats.nodes_closed == 0);
    CHECK(limited_stats.proof_bound_improvement == 0.0);
    CHECK_FALSE(limited_stats.proof_completed);

    solver.setNodeLimit(300);
    solver.setObjectiveIntegralityEnabled(true);
    const auto pruned = solver.solve();
    REQUIRE(pruned.status == Status::Optimal);
    CHECK(pruned.nodes == 1);
    const auto& pruned_stats = solver.getSerialNodePolicyStats();
    CHECK(pruned_stats.policy == policy);
    CHECK(pruned_stats.nodes_selected == 0);
    CHECK(pruned_stats.nodes_closed == 0);
    CHECK(pruned_stats.nodes_pruned_from_queue == 2);
    CHECK_THAT(pruned_stats.proof_bound_improvement, WithinAbs(0.49, 1e-9));
    CHECK(pruned_stats.proof_completed);

    solver.setSerialNodePolicyOverride(std::nullopt);
    solver.setObjectiveIntegralityEnabled(false);
    REQUIRE(solver.solve().status == Status::Optimal);
    CHECK_FALSE(solver.getSerialNodePolicyStats().policy.has_value());
}

namespace {

// Configure a solver the plunge tests share: no cuts, no presolve, so that the
// serial tree search is the only thing closing the gap.
void configurePlungeSolver(MipSolver& solver) {
    solver.setVerbose(false);
    solver.setCutsEnabled(false);
    solver.setPresolve(false);
    solver.setHeuristicMode(HeuristicRuntimeMode::Deterministic);
    solver.setHeuristicSeed(42);
    solver.setNodeLimit(300);
}

Int totalPlungeBacktracks(const MipSearchStats& stats) {
    return stats.plunge_backtracks_infeasible + stats.plunge_backtracks_depth +
           stats.plunge_backtracks_bound;
}

}  // namespace

TEST_CASE("MipSolver: plunging is disabled by default", "[mip][search]") {
    auto lp = buildSearchStagnationMip();

    MipSolver solver;
    configurePlungeSolver(solver);
    solver.load(lp);
    const auto result = solver.solve();

    CHECK((result.status == Status::Infeasible || result.status == Status::NodeLimit ||
           result.status == Status::TimeLimit));
    const auto& stats = solver.getSearchStats();
    CHECK(stats.plunge_nodes == 0);
    CHECK(stats.plunge_backtracks_infeasible == 0);
    CHECK(stats.plunge_backtracks_depth == 0);
    CHECK(stats.plunge_backtracks_bound == 0);
    CHECK(stats.plunge_rounding_calls == 0);
    CHECK(stats.plunge_rounding_improvements == 0);
}

TEST_CASE("MipSolver: plunging dives, backtracks and stays deterministic", "[mip][search]") {
    auto lp = buildSearchStagnationMip();

    MipSolver baseline;
    configurePlungeSolver(baseline);
    baseline.load(lp);
    const auto baseline_result = baseline.solve();

    MipSolver plunged_a;
    configurePlungeSolver(plunged_a);
    plunged_a.setPlungeControls(4, 0.05);
    plunged_a.load(lp);
    const auto a = plunged_a.solve();

    MipSolver plunged_b;
    configurePlungeSolver(plunged_b);
    plunged_b.setPlungeControls(4, 0.05);
    plunged_b.load(lp);
    const auto b = plunged_b.solve();

    CHECK(a.status == baseline_result.status);
    REQUIRE(a.status == b.status);
    CHECK(a.nodes == b.nodes);

    const auto& stats = plunged_a.getSearchStats();
    CHECK(stats.plunge_nodes > 0);
    CHECK(totalPlungeBacktracks(stats) > 0);
    CHECK(stats.plunge_nodes <= a.nodes);
    // No node is lost to plunging: the restart path still drops nothing.
    CHECK(stats.restart_nodes_dropped == 0);
}

TEST_CASE("MipSolver: plunging preserves the optimal objective", "[mip][search]") {
    auto lp = buildRcFixingKnapsackMip();

    MipSolver baseline;
    configurePlungeSolver(baseline);
    baseline.load(lp);
    const auto baseline_result = baseline.solve();

    MipSolver plunged;
    configurePlungeSolver(plunged);
    plunged.setPlungeControls(6, 0.05);
    plunged.load(lp);
    const auto plunged_result = plunged.solve();

    MipSolver plunged_again;
    configurePlungeSolver(plunged_again);
    plunged_again.setPlungeControls(6, 0.05);
    plunged_again.load(lp);
    const auto repeat_result = plunged_again.solve();

    REQUIRE(baseline_result.status == Status::Optimal);
    REQUIRE(plunged_result.status == Status::Optimal);
    CHECK_THAT(plunged_result.objective, WithinAbs(baseline_result.objective, 1e-9));
    CHECK(plunged_result.nodes == repeat_result.nodes);

    const auto& stats = plunged.getSearchStats();
    CHECK(stats.plunge_nodes > 0);
    CHECK(totalPlungeBacktracks(stats) > 0);
    CHECK(baseline.getSearchStats().plunge_nodes == 0);
}

TEST_CASE("MipSolver: plunge depth limit ends the dive", "[mip][search]") {
    auto lp = buildCliquePropagationMip();

    MipSolver solver;
    configurePlungeSolver(solver);
    // A huge bound quotient disables the degradation trigger, so the depth
    // limit is the only thing that can end a dive that keeps branching.
    solver.setPlungeControls(2, 1e9);
    solver.load(lp);
    static_cast<void>(solver.solve());

    const auto& stats = solver.getSearchStats();
    CHECK(stats.plunge_nodes > 0);
    CHECK(stats.plunge_backtracks_depth > 0);
    CHECK(stats.plunge_backtracks_bound == 0);
}

TEST_CASE("MipSolver: plunge bound-degradation trigger is configurable", "[mip][search]") {
    auto lp = buildRcFixingKnapsackMip();

    MipSolver strict;
    configurePlungeSolver(strict);
    // Quotient 0 declines every dive whose child bound is worse than the best
    // bound left in the queue.
    strict.setPlungeControls(8, 0.0);
    strict.load(lp);
    const auto strict_result = strict.solve();

    MipSolver lenient;
    configurePlungeSolver(lenient);
    lenient.setPlungeControls(8, 1e9);
    lenient.load(lp);
    const auto lenient_result = lenient.solve();

    REQUIRE(strict_result.status == Status::Optimal);
    REQUIRE(lenient_result.status == Status::Optimal);
    CHECK_THAT(strict_result.objective, WithinAbs(lenient_result.objective, 1e-9));
    CHECK(strict.getSearchStats().plunge_backtracks_bound > 0);
    CHECK(lenient.getSearchStats().plunge_backtracks_bound == 0);
}

TEST_CASE("MipSolver: plunge leaves run the rounding heuristic", "[mip][search]") {
    auto lp = buildRcFixingKnapsackMip();

    MipSolver solver;
    configurePlungeSolver(solver);
    solver.setPlungeControls(8, 0.0);
    solver.load(lp);
    const auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    const auto& stats = solver.getSearchStats();
    CHECK(stats.plunge_rounding_calls > 0);
    CHECK(stats.plunge_rounding_improvements <= stats.plunge_rounding_calls);
}

// A plunge leaf whose rounded LP solution is actually an improvement, so the
// incumbent-acceptance path (submit + prune + nodes_since_incumbent reset) is
// exercised rather than just the call counter. Found by randomized search:
// asserting only `plunge_rounding_calls > 0` leaves that path untested, and a
// mutation discarding the rounded solution passes the whole suite.
static LpProblem buildPlungeRoundingImprovementMip() {
    LpProblem lp;
    lp.name = "plunge_rounding_improvement";
    lp.sense = Sense::Minimize;
    lp.num_cols = 10;
    lp.obj = {5.0, -8.0, 5.0, 1.0, -7.0, 8.0, 8.0, -3.0, -8.0, 2.0};
    lp.col_lower.assign(10, 0.0);
    lp.col_upper = {3.0, 3.0, 1.0, 1.0, 3.0, 1.0, 3.0, 1.0, 3.0, 1.0};
    lp.col_type = {VarType::Integer, VarType::Integer, VarType::Binary,  VarType::Binary,
                   VarType::Integer, VarType::Binary,  VarType::Integer, VarType::Binary,
                   VarType::Integer, VarType::Binary};

    lp.num_rows = 2;
    lp.row_lower = {-kInf, -kInf};
    lp.row_upper = {6.0, 18.4};
    std::vector<Triplet> trips = {
        {0, 0, 6.0},  {0, 1, -3.0}, {0, 5, -2.0}, {0, 6, -3.0}, {0, 9, 1.0},
        {1, 0, -8.0}, {1, 1, 9.0},  {1, 2, -1.0}, {1, 3, -8.0}, {1, 4, 1.0},
        {1, 7, 4.0},  {1, 8, 8.0},  {1, 9, 7.0},
    };
    lp.matrix = SparseMatrix(2, 10, std::move(trips));
    return lp;
}

TEST_CASE("MipSolver: plunge rounding accepts an improving incumbent", "[mip][search]") {
    const auto lp = buildPlungeRoundingImprovementMip();

    MipSolver solver;
    configurePlungeSolver(solver);
    solver.setPlungeControls(8, 0.0);
    const bool measure_policy = GENERATE(false, true);
    if (measure_policy) {
        solver.setSerialNodePolicyOverride(NodePolicy::BestFirst);
    }
    solver.load(lp);
    const auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    const auto& stats = solver.getSearchStats();
    CHECK(stats.plunge_rounding_calls > 0);
    CHECK(stats.plunge_rounding_improvements > 0);
    if (measure_policy) {
        const auto& policy = solver.getSerialNodePolicyStats();
        CHECK(policy.nodes_selected == result.nodes - 1);
        CHECK(policy.incumbents_accepted >= stats.plunge_rounding_improvements);
    }

    // The accepted incumbent must not corrupt the answer.
    MipSolver baseline;
    baseline.setVerbose(false);
    baseline.setCutsEnabled(false);
    baseline.setPresolve(false);
    baseline.load(lp);
    const auto base_result = baseline.solve();
    REQUIRE(base_result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(base_result.objective, 1e-9));
}

TEST_CASE("MipSolver: in-tree presolve telemetry is populated", "[mip][presolve][tree]") {
    auto lp = buildSearchStagnationMip();

    MipSolver solver;
    solver.setVerbose(false);
    solver.setCutsEnabled(false);
    solver.setPresolve(false);
    solver.setTreePresolveEnabled(true);
    solver.setNodeLimit(300);
    solver.load(lp);
    const auto result = solver.solve();

    CHECK((result.status == Status::Infeasible || result.status == Status::NodeLimit ||
           result.status == Status::TimeLimit));
    const auto& stats = solver.getTreePresolveStats();
    CHECK(stats.attempts >= 1);
    CHECK((stats.runs >= 1 || stats.infeasible >= 1));
    CHECK(stats.activity_tightenings >= 0);
    CHECK(stats.reduced_cost_tightenings >= 0);
}

TEST_CASE("MipSolver: tree presolve auto tuning classifies small pure-binary models",
          "[mip][presolve][tree]") {
    MipSolver pure_binary;
    pure_binary.setVerbose(false);
    pure_binary.load(buildKnapsackMip());
    CHECK(pure_binary.isTreePresolveAutoTuningEnabled());
    CHECK(pure_binary.isTreePresolveBinaryLiteProfileActive());

    pure_binary.setTreePresolveAutoTuning(false);
    CHECK_FALSE(pure_binary.isTreePresolveBinaryLiteProfileActive());

    MipSolver general_integer;
    general_integer.setVerbose(false);
    general_integer.load(buildBranchingMip());
    CHECK_FALSE(general_integer.isTreePresolveBinaryLiteProfileActive());
}

TEST_CASE("MipSolver: in-tree presolve preserves feasible optimum", "[mip][presolve][tree]") {
    auto lp = buildBranchingMip();

    MipSolver with_tree_presolve;
    with_tree_presolve.setVerbose(false);
    with_tree_presolve.setCutsEnabled(false);
    with_tree_presolve.setPresolve(false);
    with_tree_presolve.setTreePresolveEnabled(true);
    with_tree_presolve.setSearchProfile(SearchProfile::Stable);
    with_tree_presolve.load(lp);
    const auto a = with_tree_presolve.solve();

    MipSolver without_tree_presolve;
    without_tree_presolve.setVerbose(false);
    without_tree_presolve.setCutsEnabled(false);
    without_tree_presolve.setPresolve(false);
    without_tree_presolve.setTreePresolveEnabled(false);
    without_tree_presolve.setSearchProfile(SearchProfile::Stable);
    without_tree_presolve.load(lp);
    const auto b = without_tree_presolve.solve();

    REQUIRE(a.status == Status::Optimal);
    REQUIRE(b.status == Status::Optimal);
    CHECK_THAT(a.objective, WithinAbs(b.objective, 1e-9));
}

TEST_CASE("MipSolver: presolve does not double-count objective offset",
          "[mip][presolve][objective]") {
    const auto lp = buildPresolveOffsetRegressionMip();

    auto solve_with = [&](bool presolve) {
        MipSolver solver;
        solver.setVerbose(false);
        solver.setCutsEnabled(false);
        solver.setTreePresolveEnabled(false);
        solver.setSearchProfile(SearchProfile::Stable);
        solver.setNumThreads(1);
        solver.setPresolve(presolve);
        solver.load(lp);
        return solver.solve();
    };

    const auto on = solve_with(true);
    const auto off = solve_with(false);

    REQUIRE(on.status == Status::Optimal);
    REQUIRE(off.status == Status::Optimal);
    REQUIRE(on.solution.size() == 3);
    REQUIRE(off.solution.size() == 3);

    CHECK_THAT(on.objective, WithinAbs(100.0, 1e-6));
    CHECK_THAT(off.objective, WithinAbs(100.0, 1e-6));
    CHECK_THAT(on.objective, WithinAbs(off.objective, 1e-9));
    CHECK_THAT(on.solution[0], WithinAbs(1.0, 1e-9));
    CHECK_THAT(off.solution[0], WithinAbs(1.0, 1e-9));
}

// ---------------------------------------------------------------------------
// Reduced-cost fixing as a tree tool (#124).
// ---------------------------------------------------------------------------

static MipResult solveRcFixing(const LpProblem& lp, bool tree_presolve, Int threads,
                               RcFixingStats& stats_out, MipTreePresolveStats& tp_stats_out) {
    MipSolver solver;
    solver.setVerbose(false);
    solver.setCutsEnabled(false);
    solver.setPresolve(false);
    solver.setTreePresolveEnabled(tree_presolve);
    solver.setNumThreads(threads);
    solver.setSearchProfile(SearchProfile::Stable);
    solver.load(lp);
    const auto result = solver.solve();
    stats_out = solver.getRcFixingStats();
    tp_stats_out = solver.getTreePresolveStats();
    return result;
}

TEST_CASE("MipSolver: reduced-cost fixing reports root and tree phases separately",
          "[mip][rcfixer]") {
    const auto lp = buildRcFixingKnapsackMip();

    RcFixingStats stats{};
    MipTreePresolveStats tp_stats{};
    const auto result = solveRcFixing(lp, /*tree_presolve=*/false, /*threads=*/1, stats, tp_stats);

    REQUIRE(result.status == Status::Optimal);
    // Independent ground truth from the bounded-knapsack DP.
    CHECK_THAT(result.objective, WithinAbs(rcKnapsackOptimum(), 1e-9));

    // Statistics reach the getter without verbose logging enabled, split by
    // phase: root/global and tree/local.
    CHECK((stats.root_global_fixings + stats.root_global_tightenings) > 0);
    CHECK((stats.tree_local_fixings + stats.tree_local_tightenings) > 0);
}

TEST_CASE("MipSolver: node reduced-cost fixing re-triggers domain propagation", "[mip][rcfixer]") {
    const auto lp = buildRcFixingKnapsackMip();

    RcFixingStats stats{};
    MipTreePresolveStats tp_stats{};
    const auto result = solveRcFixing(lp, /*tree_presolve=*/false, /*threads=*/1, stats, tp_stats);

    REQUIRE(result.status == Status::Optimal);
    const Int tree_changes = stats.tree_local_fixings + stats.tree_local_tightenings;
    REQUIRE(tree_changes > 0);
    // One re-propagation per node whose bounds were tightened, and a node can
    // only trigger it after at least one tightening.
    CHECK(stats.propagation_triggers > 0);
    CHECK(stats.propagation_triggers <= tree_changes);
}

TEST_CASE("MipSolver: node reduced-cost fixing runs with in-tree presolve enabled",
          "[mip][rcfixer][tree]") {
    const auto lp = buildRcFixingTreePresolveMip();

    RcFixingStats off_stats{};
    MipTreePresolveStats off_tp{};
    const auto off = solveRcFixing(lp, /*tree_presolve=*/false, /*threads=*/1, off_stats, off_tp);

    RcFixingStats on_stats{};
    MipTreePresolveStats on_tp{};
    const auto on = solveRcFixing(lp, /*tree_presolve=*/true, /*threads=*/1, on_stats, on_tp);

    REQUIRE(off.status == Status::Optimal);
    REQUIRE(on.status == Status::Optimal);
    CHECK_THAT(on.objective, WithinAbs(off.objective, 1e-9));

    // The node-level pass no longer requires tree presolve to have been skipped.
    CHECK((on_stats.tree_local_fixings + on_stats.tree_local_tightenings) > 0);
    CHECK(on_stats.propagation_triggers > 0);
    // The reduced-cost pass inside the tree-presolve block is the same engine,
    // so its counter moves together with the tree-local counters.
    CHECK(on_tp.reduced_cost_tightenings > 0);
}

TEST_CASE("MipSolver: node reduced-cost fixing runs with more than one thread",
          "[mip][rcfixer][parallel]") {
    const auto lp = buildRcFixingKnapsackMip();

    RcFixingStats serial_stats{};
    MipTreePresolveStats serial_tp{};
    const auto serial =
        solveRcFixing(lp, /*tree_presolve=*/true, /*threads=*/1, serial_stats, serial_tp);

    RcFixingStats parallel_stats{};
    MipTreePresolveStats parallel_tp{};
    const auto parallel =
        solveRcFixing(lp, /*tree_presolve=*/true, /*threads=*/4, parallel_stats, parallel_tp);

    REQUIRE(serial.status == Status::Optimal);
    REQUIRE(parallel.status == Status::Optimal);
    CHECK_THAT(parallel.objective, WithinAbs(serial.objective, 1e-9));
    CHECK_THAT(parallel.objective, WithinAbs(rcKnapsackOptimum(), 1e-9));

    CHECK((parallel_stats.tree_local_fixings + parallel_stats.tree_local_tightenings) > 0);
    CHECK(parallel_stats.propagation_triggers > 0);

    // Node-local fixings never feed the global counters, however many threads
    // produced them.
    CHECK(parallel_stats.root_global_fixings == serial_stats.root_global_fixings);
    CHECK(parallel_stats.root_global_tightenings == serial_stats.root_global_tightenings);
}

// Root reduced-cost fixing tightens the loaded problem in place, using bounds
// derived from the incumbent of the solve that is running. Those bounds must
// not survive the call: a second solve on the same object would otherwise
// start inside the first solve's optimality box and can report a worse
// objective as optimal.
TEST_CASE("MipSolver: a second solve is not narrowed by the first solve's RC fixing",
          "[mip][rcfixer]") {
    LpProblem lp;
    lp.name = "rc_fixing_repeated_solve";
    lp.sense = Sense::Minimize;
    lp.num_cols = 5;
    lp.num_rows = 2;
    lp.obj = {-2.0, -4.0, 1.0, 2.0, -1.0};
    lp.col_lower = {0.0, 0.0, 0.0, 0.0, 0.0};
    lp.col_upper = {2.0, 1.0, 4.0, 1.0, 2.0};
    lp.col_type.assign(5, VarType::Integer);
    lp.col_names = {"x0", "x1", "x2", "x3", "x4"};
    lp.row_lower = {-2.0, -kInf};
    lp.row_upper = {1.0, 9.0};
    lp.row_names = {"R0", "R1"};
    std::vector<Triplet> trips = {
        {0, 0, -5.0}, {0, 1, 1.0},  {0, 2, -5.0}, {0, 4, -3.0},
        {1, 0, 4.0},  {1, 1, -3.0}, {1, 4, 5.0},
    };
    lp.matrix = SparseMatrix(2, 5, std::move(trips));

    MipSolver solver;
    solver.setVerbose(false);
    solver.setPresolve(false);
    solver.setSymmetryEnabled(false);
    solver.load(lp);

    auto first = solver.solve();
    REQUIRE(first.status == Status::Optimal);
    CHECK_THAT(first.objective, WithinAbs(-5.0, 1e-9));

    auto second = solver.solve();
    REQUIRE(second.status == first.status);
    CHECK_THAT(second.objective, WithinAbs(first.objective, 1e-9));

    // Re-loading the same model is the other way a caller expects the original
    // bounds back.
    solver.load(lp);
    auto third = solver.solve();
    REQUIRE(third.status == first.status);
    CHECK_THAT(third.objective, WithinAbs(first.objective, 1e-9));
}

// ---------------------------------------------------------------------------
// Issue #186: branch-and-bound reported Status::Optimal with a suboptimal
// objective on small all-integer models.
//
// The cause was root reduced-cost fixing being handed reduced costs that no
// longer belonged to the root LP solution they were priced against: the root
// heuristics run on the same LP object and leave their own dual state behind,
// and restoring the saved basis afterwards does not recompute reduced costs.
// Fixing against stale duals removes bound values that hold the true optimum,
// and the search then proves optimality of a worse point.
//
// Only the second model below exercises that path: the ranged-row model the
// issue reports had already stopped reproducing by the time the cause was
// found, so it stands here as end-to-end coverage of the reported case rather
// than as the guard on root reduced-cost fixing. The second model, from the
// same randomized differential sweep, is the one that regresses to -22 the
// moment the reduced-cost snapshot is removed.
//
// Both models are all-integer, so every acceptance check can be made exactly:
// rows, bounds, integrality and objective.
// ---------------------------------------------------------------------------

namespace {

/// Assert that `solution` is integral, inside the column bounds, satisfies
/// every row of `lp`, and evaluates to `expected_obj`.
void checkIntegerSolution(const LpProblem& lp, const std::vector<Real>& solution,
                          Real expected_obj) {
    REQUIRE(static_cast<Index>(solution.size()) == lp.num_cols);

    Real obj = 0.0;
    for (Index j = 0; j < lp.num_cols; ++j) {
        const Real x = solution[j];
        CHECK_THAT(x, WithinAbs(std::round(x), 1e-6));
        CHECK(x >= lp.col_lower[j] - 1e-6);
        CHECK(x <= lp.col_upper[j] + 1e-6);
        obj += lp.obj[j] * std::round(x);
    }
    CHECK_THAT(obj, WithinAbs(expected_obj, 1e-6));

    for (Index i = 0; i < lp.num_rows; ++i) {
        Real activity = 0.0;
        for (Index j = 0; j < lp.num_cols; ++j) {
            activity += lp.matrix.coeff(i, j) * std::round(solution[j]);
        }
        CHECK(activity >= lp.row_lower[i] - 1e-6);
        CHECK(activity <= lp.row_upper[i] + 1e-6);
    }
}

/// The exact model from issue #186. Optimum is -10 at x = (1, 3, 2, 0).
LpProblem buildIssue186RangedRowMip() {
    LpProblem lp;
    lp.name = "issue186_ranged_rows";
    lp.sense = Sense::Minimize;
    lp.num_cols = 4;
    lp.num_rows = 4;
    lp.obj = {-3.0, -3.0, 1.0, 2.0};
    lp.col_lower = {0.0, 0.0, 0.0, 0.0};
    lp.col_upper = {1.0, 4.0, 4.0, 1.0};
    lp.col_type.assign(4, VarType::Integer);
    lp.col_names = {"x0", "x1", "x2", "x3"};
    lp.row_lower = {-4.0, -7.0, -5.0, -kInf};
    lp.row_upper = {kInf, 11.0, kInf, 6.0};
    lp.row_names = {"R0", "R1", "R2", "R3"};
    std::vector<Triplet> trips = {
        {0, 0, 5.0}, {1, 1, -5.0}, {1, 2, 5.0},  {1, 3, -5.0}, {2, 0, -5.0},
        {2, 2, 2.0}, {2, 3, -1.0}, {3, 0, -3.0}, {3, 1, 3.0},  {3, 3, -5.0},
    };
    lp.matrix = SparseMatrix(4, 4, std::move(trips));
    return lp;
}

/// A second model from the same randomized differential sweep as issue #186.
/// The optimum is -23 at x = (2, 3, 0, 2, 1, 0, 1, 3); the root LP relaxation
/// already attains -23, so the whole answer hangs on the root fixings being
/// derived from that LP's own reduced costs. Against stale duals the solver
/// used to fix x3 to at most 1 and then proved -22 optimal.
LpProblem buildIssue186RootFixingMip() {
    LpProblem lp;
    lp.name = "issue186_root_rc_fixing";
    lp.sense = Sense::Minimize;
    lp.num_cols = 8;
    lp.num_rows = 1;
    lp.obj = {-3.0, -3.0, 2.0, 1.0, 2.0, -3.0, -3.0, -3.0};
    lp.col_lower.assign(8, 0.0);
    lp.col_upper = {2.0, 4.0, 1.0, 2.0, 1.0, 4.0, 1.0, 3.0};
    lp.col_type.assign(8, VarType::Integer);
    lp.col_names = {"x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7"};
    lp.row_lower = {-kInf};
    lp.row_upper = {-7.0};
    lp.row_names = {"R0"};
    std::vector<Triplet> trips = {
        {0, 0, -3.0}, {0, 1, 3.0}, {0, 2, -1.0}, {0, 3, -1.0},
        {0, 4, -5.0}, {0, 5, 4.0}, {0, 7, -1.0},
    };
    lp.matrix = SparseMatrix(1, 8, std::move(trips));
    return lp;
}

/// Turn off every optional component the issue's isolation table covers, so a
/// failure can only be in the core branch-and-bound / node-LP path.
void disableOptionalComponents(MipSolver& solver) {
    solver.setSymmetryEnabled(false);
    solver.setPresolve(false);
    solver.setCutsEnabled(false);
    solver.setTreePresolveEnabled(false);
    solver.setTreeCutsEnabled(false);
    solver.setConflictsEnabled(false);
    solver.setRestartsEnabled(false);
}

/// The exact model from issue #194. R0 forces x2 = 0 while R3 forces x2 = 1,
/// so the model is infeasible; presolve used to tighten both singleton rows
/// into the same column without noticing the crossed bounds, drop them, and
/// report a bogus optimum at x = (1, 4, 0).
LpProblem buildIssue194ConflictingSingletonsMip() {
    LpProblem lp;
    lp.name = "issue194_conflicting_singletons";
    lp.sense = Sense::Minimize;
    lp.num_cols = 3;
    lp.num_rows = 4;
    lp.obj = {5.0, -15.0, -18.0};
    lp.obj_offset = 3.0;
    lp.col_lower = {0.0, 0.0, 0.0};
    lp.col_upper = {1.0, 4.0, 1.0};
    lp.col_type.assign(3, VarType::Integer);
    lp.col_names = {"x0", "x1", "x2"};
    lp.row_lower = {-kInf, -kInf, -kInf, 2.0};
    lp.row_upper = {0.0, -2.0, -1.0, 2.0};
    lp.row_names = {"R0", "R1", "R2", "R3"};
    std::vector<Triplet> trips = {
        {0, 2, 4.0}, {1, 0, -4.0}, {2, 0, -3.0}, {2, 2, -3.0}, {3, 2, 2.0},
    };
    lp.matrix = SparseMatrix(4, 3, std::move(trips));
    return lp;
}

/// Check a reported point against every row bound and against the integrality
/// of every integer column, without pinning an objective value. Unlike
/// checkIntegerSolution, usable where the optimum is not known up front.
void checkPointFeasible(const LpProblem& lp, const std::vector<Real>& solution) {
    REQUIRE(static_cast<Index>(solution.size()) == lp.num_cols);

    for (Index j = 0; j < lp.num_cols; ++j) {
        const Real x = solution[j];
        if (lp.col_type[j] != VarType::Continuous) {
            CHECK_THAT(x, WithinAbs(std::round(x), 1e-6));
        }
        CHECK(x >= lp.col_lower[j] - 1e-6);
        CHECK(x <= lp.col_upper[j] + 1e-6);
    }

    for (Index i = 0; i < lp.num_rows; ++i) {
        Real activity = 0.0;
        for (Index j = 0; j < lp.num_cols; ++j) {
            activity += lp.matrix.coeff(i, j) * solution[j];
        }
        CHECK(activity >= lp.row_lower[i] - 1e-6);
        CHECK(activity <= lp.row_upper[i] + 1e-6);
    }
}
}  // namespace

TEST_CASE("MipSolver: issue 186 ranged-row model is solved to its true optimum",
          "[mip][issue186]") {
    const auto lp = buildIssue186RangedRowMip();

    MipSolver solver;
    solver.setVerbose(false);
    solver.load(lp);
    const auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(-10.0, 1e-6));
    checkIntegerSolution(lp, result.solution, -10.0);
}

TEST_CASE("MipSolver: issue 186 ranged-row model needs no optional component", "[mip][issue186]") {
    const auto lp = buildIssue186RangedRowMip();

    MipSolver solver;
    solver.setVerbose(false);
    disableOptionalComponents(solver);
    solver.load(lp);
    const auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(-10.0, 1e-6));
    checkIntegerSolution(lp, result.solution, -10.0);
}

TEST_CASE("MipSolver: issue 186 root RC fixing keeps the optimum reachable", "[mip][issue186]") {
    const auto lp = buildIssue186RootFixingMip();

    SECTION("default settings") {
        MipSolver solver;
        solver.setVerbose(false);
        solver.load(lp);
        const auto result = solver.solve();

        REQUIRE(result.status == Status::Optimal);
        CHECK_THAT(result.objective, WithinAbs(-23.0, 1e-6));
        checkIntegerSolution(lp, result.solution, -23.0);
    }

    SECTION("optional components disabled") {
        MipSolver solver;
        solver.setVerbose(false);
        disableOptionalComponents(solver);
        solver.load(lp);
        const auto result = solver.solve();

        REQUIRE(result.status == Status::Optimal);
        CHECK_THAT(result.objective, WithinAbs(-23.0, 1e-6));
        checkIntegerSolution(lp, result.solution, -23.0);
    }
}

TEST_CASE("MipSolver: issue 194 conflicting singleton rows stay infeasible",
          "[mip][presolve][issue194]") {
    const auto lp = buildIssue194ConflictingSingletonsMip();

    const auto solveWith = [&](bool presolve) {
        MipSolver solver;
        solver.setVerbose(false);
        solver.setPresolve(presolve);
        solver.load(lp);
        const auto result = solver.solve();
        // A reported optimum has to be a genuine point of the model.
        if (result.status == Status::Optimal) {
            checkPointFeasible(lp, result.solution);
        }
        return result.status;
    };

    const Status with_presolve = solveWith(true);
    const Status without_presolve = solveWith(false);

    CHECK(with_presolve == Status::Infeasible);
    CHECK(without_presolve == Status::Infeasible);
    CHECK(with_presolve == without_presolve);
}
