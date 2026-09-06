#include "mipx/io.h"
#include "mipx/mip_solver.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <filesystem>

using namespace mipx;
using Catch::Matchers::WithinAbs;

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
    std::string path = std::string(TEST_DATA_DIR) + "/miplib/gt2.mps.gz";
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
    solver.load(lp);
    const auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    const auto& stats = solver.getSearchStats();
    CHECK(stats.plunge_rounding_calls > 0);
    CHECK(stats.plunge_rounding_improvements > 0);

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
