#include "mipx/clique_table.h"
#include "mipx/cut_pool.h"

#include <catch2/catch_test_macros.hpp>

using namespace mipx;

namespace {

/// 5 binary vars, row x0+x1+x2 <= 1, row x2+x3+x4 <= 1.
LpProblem makeFiveBinarySetPacking() {
    LpProblem prob;
    prob.num_cols = 5;
    prob.num_rows = 2;
    prob.matrix = SparseMatrix(2, 5,
                               {
                                   {0, 0, 1.0},
                                   {0, 1, 1.0},
                                   {0, 2, 1.0},
                                   {1, 2, 1.0},
                                   {1, 3, 1.0},
                                   {1, 4, 1.0},
                               });
    prob.row_lower = {-kInf, -kInf};
    prob.row_upper = {1.0, 1.0};
    prob.col_lower = {0.0, 0.0, 0.0, 0.0, 0.0};
    prob.col_upper = {1.0, 1.0, 1.0, 1.0, 1.0};
    prob.col_type = {VarType::Binary, VarType::Binary, VarType::Binary, VarType::Binary,
                     VarType::Binary};
    prob.obj = {0.0, 0.0, 0.0, 0.0, 0.0};
    return prob;
}

/// Helper: check if a clique contains a given variable (non-complemented).
bool cliqueContainsVar(const Clique& clq, Index var) {
    for (const auto& lit : clq.literals) {
        if (lit.var == var && !lit.complemented) {
            return true;
        }
    }
    return false;
}

}  // namespace

TEST_CASE("CliqueTable: build from set-packing", "[clique_table]") {
    auto prob = makeFiveBinarySetPacking();
    ConflictGraph graph;
    graph.build(prob);

    CliqueTable table;
    table.build(prob, graph);

    // At least two cliques from the two set-packing rows.
    CHECK(table.numCliques() >= 2);

    // Every clique should have at least 2 literals.
    for (Index i = 0; i < table.numCliques(); ++i) {
        CHECK(table.clique(i).literals.size() >= 2);
    }
}

TEST_CASE("CliqueTable: greedy maximal clique extension", "[clique_table]") {
    auto prob = makeFiveBinarySetPacking();
    ConflictGraph graph;
    graph.build(prob);

    CliqueTable table;
    table.build(prob, graph);

    // The two set-packing rows share x2, and x2 conflicts with x0,x1 and x3,x4.
    // Greedy extension may discover cliques containing x2 along with others.
    // Check that at least one clique contains x2.
    bool found_x2 = false;
    for (Index i = 0; i < table.numCliques(); ++i) {
        if (cliqueContainsVar(table.clique(i), 2)) {
            found_x2 = true;
            break;
        }
    }
    CHECK(found_x2);

    // Verify cliquesOf returns correct results for x2.
    Literal lit_x2 = {2, false};
    auto cliques_of_x2 = table.cliquesOf(lit_x2);
    CHECK(!cliques_of_x2.empty());
    for (Index ci : cliques_of_x2) {
        CHECK(cliqueContainsVar(table.clique(ci), 2));
    }
}

TEST_CASE("CliqueTable: mergeAndSubsume", "[clique_table]") {
    // Build a problem where one clique is a subset of another.
    // Row 0: x0 + x1 + x2 <= 1 (3-clique).
    // Row 1: x0 + x1 <= 1 (2-clique, subset of the 3-clique).
    LpProblem prob;
    prob.num_cols = 3;
    prob.num_rows = 2;
    prob.matrix = SparseMatrix(2, 3,
                               {
                                   {0, 0, 1.0},
                                   {0, 1, 1.0},
                                   {0, 2, 1.0},
                                   {1, 0, 1.0},
                                   {1, 1, 1.0},
                               });
    prob.row_lower = {-kInf, -kInf};
    prob.row_upper = {1.0, 1.0};
    prob.col_lower = {0.0, 0.0, 0.0};
    prob.col_upper = {1.0, 1.0, 1.0};
    prob.col_type = {VarType::Binary, VarType::Binary, VarType::Binary};
    prob.obj = {0.0, 0.0, 0.0};

    ConflictGraph graph;
    graph.build(prob);

    CliqueTable table;
    table.build(prob, graph);

    // After build (which calls mergeAndSubsume internally), the {x0,x1}
    // clique should have been subsumed by the {x0,x1,x2} clique.
    // All remaining cliques should have size >= 2 and the subset should
    // not be a separate entry.

    // Verify no clique is a proper subset of another.
    for (Index i = 0; i < table.numCliques(); ++i) {
        for (Index j = 0; j < table.numCliques(); ++j) {
            if (i == j) {
                continue;
            }
            const auto& ci = table.clique(i);
            const auto& cj = table.clique(j);
            if (ci.literals.size() > cj.literals.size()) {
                continue;
            }
            // ci should not be a subset of cj.
            bool subset = true;
            for (const auto& lit : ci.literals) {
                bool found = false;
                for (const auto& lit2 : cj.literals) {
                    if (lit.var == lit2.var && lit.complemented == lit2.complemented) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    subset = false;
                    break;
                }
            }
            CHECK_FALSE(subset);
        }
    }
}

TEST_CASE("CliqueTable: extractFromCut", "[clique_table]") {
    // Set up a minimal problem with 4 binary vars and one row (to build
    // the conflict graph).
    LpProblem prob;
    prob.num_cols = 4;
    prob.num_rows = 1;
    prob.matrix = SparseMatrix(1, 4,
                               {
                                   {0, 0, 1.0},
                                   {0, 1, 1.0},
                               });
    prob.row_lower = {-kInf};
    prob.row_upper = {1.0};
    prob.col_lower = {0.0, 0.0, 0.0, 0.0};
    prob.col_upper = {1.0, 1.0, 1.0, 1.0};
    prob.col_type = {VarType::Binary, VarType::Binary, VarType::Binary, VarType::Binary};
    prob.obj = {0.0, 0.0, 0.0, 0.0};

    ConflictGraph graph;
    graph.build(prob);

    CliqueTable table;
    table.build(prob, graph);

    Index before = table.numCliques();

    // Create a cut that looks like x2 + x3 <= 1.
    Cut cut;
    cut.indices = {2, 3};
    cut.values = {1.0, 1.0};
    cut.lower = -kInf;
    cut.upper = 1.0;
    cut.family = CutFamily::Clique;

    Int added = table.extractFromCut(cut, prob, graph);
    CHECK(added == 1);
    CHECK(table.numCliques() == before + 1);

    // The new clique should contain x2 and x3.
    const auto& new_clq = table.clique(table.numCliques() - 1);
    CHECK(cliqueContainsVar(new_clq, 2));
    CHECK(cliqueContainsVar(new_clq, 3));
}

TEST_CASE("CliqueTable: separateCliqueCover", "[clique_table]") {
    // 3 binary vars: x0 + x1 + x2 <= 1 (clique).
    // LP solution: x0=0.5, x1=0.5, x2=0.5 -> sum=1.5 > 1.
    LpProblem prob;
    prob.num_cols = 3;
    prob.num_rows = 1;
    prob.matrix = SparseMatrix(1, 3,
                               {
                                   {0, 0, 1.0},
                                   {0, 1, 1.0},
                                   {0, 2, 1.0},
                               });
    prob.row_lower = {-kInf};
    prob.row_upper = {1.0};
    prob.col_lower = {0.0, 0.0, 0.0};
    prob.col_upper = {1.0, 1.0, 1.0};
    prob.col_type = {VarType::Binary, VarType::Binary, VarType::Binary};
    prob.obj = {0.0, 0.0, 0.0};

    ConflictGraph graph;
    graph.build(prob);

    CliqueTable table;
    table.build(prob, graph);

    std::vector<Real> primals = {0.5, 0.5, 0.5};
    CutPool pool;
    pool.setMinEfficacy(0.0);  // Accept any cut.

    Int num_cuts = table.separateCliqueCover(prob, primals, pool);
    CHECK(num_cuts >= 1);
    CHECK(pool.size() >= 1);

    // The generated cut should be violated by the LP solution.
    for (Index i = 0; i < pool.size(); ++i) {
        const auto& c = pool[i];
        CHECK(c.family == CutFamily::Clique);
        // Evaluate: sum of values * primals should exceed upper.
        Real lhs = 0.0;
        for (Index k = 0; k < static_cast<Index>(c.indices.size()); ++k) {
            lhs += c.values[k] * primals[c.indices[k]];
        }
        CHECK(lhs > c.upper);
    }
}

TEST_CASE("CliqueTable: propagate basic", "[clique_table]") {
    // 3 binary vars: x0+x1+x2 <= 1 (clique). Fix x0=1.
    LpProblem prob;
    prob.num_cols = 3;
    prob.num_rows = 1;
    prob.matrix = SparseMatrix(1, 3,
                               {
                                   {0, 0, 1.0},
                                   {0, 1, 1.0},
                                   {0, 2, 1.0},
                               });
    prob.row_lower = {-kInf};
    prob.row_upper = {1.0};
    prob.col_lower = {0.0, 0.0, 0.0};
    prob.col_upper = {1.0, 1.0, 1.0};
    prob.col_type = {VarType::Binary, VarType::Binary, VarType::Binary};
    prob.obj = {0.0, 0.0, 0.0};

    ConflictGraph graph;
    graph.build(prob);

    CliqueTable table;
    table.build(prob, graph);

    // Fix x0 = 1 via bounds.
    std::vector<Real> lower = {1.0, 0.0, 0.0};
    std::vector<Real> upper = {1.0, 1.0, 1.0};

    std::vector<CliqueTable::BoundUpdate> updates;
    bool feasible = table.propagate(lower, upper, updates);
    CHECK(feasible);

    // x1 and x2 should be fixed to 0.
    CHECK(!updates.empty());
    for (const auto& u : updates) {
        CHECK(u.new_upper == 0.0);
    }

    // Check that the updates cover x1 and x2.
    bool has_x1 = false;
    bool has_x2 = false;
    for (const auto& u : updates) {
        if (u.col == 1) {
            has_x1 = true;
        }
        if (u.col == 2) {
            has_x2 = true;
        }
    }
    CHECK(has_x1);
    CHECK(has_x2);
}

TEST_CASE("CliqueTable: propagate equality clique", "[clique_table]") {
    // 3 binary vars: x0+x1+x2 = 1 (equality clique).
    // Fix x0=0 and x1=0. The remaining x2 must be 1.
    LpProblem prob;
    prob.num_cols = 3;
    prob.num_rows = 1;
    prob.matrix = SparseMatrix(1, 3,
                               {
                                   {0, 0, 1.0},
                                   {0, 1, 1.0},
                                   {0, 2, 1.0},
                               });
    prob.row_lower = {1.0};
    prob.row_upper = {1.0};
    prob.col_lower = {0.0, 0.0, 0.0};
    prob.col_upper = {1.0, 1.0, 1.0};
    prob.col_type = {VarType::Binary, VarType::Binary, VarType::Binary};
    prob.obj = {0.0, 0.0, 0.0};

    ConflictGraph graph;
    graph.build(prob);

    CliqueTable table;
    table.build(prob, graph);

    // Verify the equality clique was detected.
    bool found_equality = false;
    for (Index i = 0; i < table.numCliques(); ++i) {
        if (table.clique(i).is_equality) {
            found_equality = true;
            break;
        }
    }
    CHECK(found_equality);

    // Fix x0=0, x1=0, x2 is free.
    std::vector<Real> lower = {0.0, 0.0, 0.0};
    std::vector<Real> upper = {0.0, 0.0, 1.0};

    std::vector<CliqueTable::BoundUpdate> updates;
    bool feasible = table.propagate(lower, upper, updates);
    CHECK(feasible);

    // x2 should be forced to 1.
    bool x2_forced = false;
    for (const auto& u : updates) {
        if (u.col == 2 && u.new_lower == 1.0) {
            x2_forced = true;
        }
    }
    CHECK(x2_forced);
}

TEST_CASE("CliqueTable: propagate infeasibility", "[clique_table]") {
    // 3 binary vars: x0+x1+x2 <= 1 (clique). Fix x0=1, x1=1.
    LpProblem prob;
    prob.num_cols = 3;
    prob.num_rows = 1;
    prob.matrix = SparseMatrix(1, 3,
                               {
                                   {0, 0, 1.0},
                                   {0, 1, 1.0},
                                   {0, 2, 1.0},
                               });
    prob.row_lower = {-kInf};
    prob.row_upper = {1.0};
    prob.col_lower = {0.0, 0.0, 0.0};
    prob.col_upper = {1.0, 1.0, 1.0};
    prob.col_type = {VarType::Binary, VarType::Binary, VarType::Binary};
    prob.obj = {0.0, 0.0, 0.0};

    ConflictGraph graph;
    graph.build(prob);

    CliqueTable table;
    table.build(prob, graph);

    // Fix x0=1 and x1=1 -- infeasible in any clique containing both.
    std::vector<Real> lower = {1.0, 1.0, 0.0};
    std::vector<Real> upper = {1.0, 1.0, 1.0};

    std::vector<CliqueTable::BoundUpdate> updates;
    bool feasible = table.propagate(lower, upper, updates);
    CHECK_FALSE(feasible);
}

TEST_CASE("CliqueTable: findObjectiveCliques", "[clique_table]") {
    // 4 binary vars with positive obj coefficients and mutual conflicts.
    // x0+x1 <= 1, x0+x2 <= 1, x1+x2 <= 1 (complete triangle).
    // x3 has positive obj but no conflicts with x0,x1,x2.
    LpProblem prob;
    prob.num_cols = 4;
    prob.num_rows = 3;
    prob.matrix = SparseMatrix(3, 4,
                               {
                                   {0, 0, 1.0},
                                   {0, 1, 1.0},
                                   {1, 0, 1.0},
                                   {1, 2, 1.0},
                                   {2, 1, 1.0},
                                   {2, 2, 1.0},
                               });
    prob.row_lower = {-kInf, -kInf, -kInf};
    prob.row_upper = {1.0, 1.0, 1.0};
    prob.col_lower = {0.0, 0.0, 0.0, 0.0};
    prob.col_upper = {1.0, 1.0, 1.0, 1.0};
    prob.col_type = {VarType::Binary, VarType::Binary, VarType::Binary, VarType::Binary};
    prob.obj = {3.0, 2.0, 1.0, 0.5};
    prob.sense = Sense::Minimize;

    ConflictGraph graph;
    graph.build(prob);

    CliqueTable table;
    table.build(prob, graph);

    auto obj_cliques = table.findObjectiveCliques(prob, graph);
    CHECK(!obj_cliques.empty());

    // The objective clique should contain a subset of {x0, x1, x2}
    // since they all mutually conflict and have positive obj coefficients.
    // x3 may or may not appear depending on its conflicts.
    for (const auto& clq : obj_cliques) {
        CHECK(clq.literals.size() >= 2);
    }
}

TEST_CASE("CliqueTable: cliquePartition", "[clique_table]") {
    // 5 binary vars: x0+x1+x2 <= 1, x2+x3+x4 <= 1.
    // Fractional LP: all at 0.4.
    auto prob = makeFiveBinarySetPacking();

    ConflictGraph graph;
    graph.build(prob);

    CliqueTable table;
    table.build(prob, graph);

    std::vector<Real> primals = {0.4, 0.4, 0.4, 0.4, 0.4};
    auto partition = table.cliquePartition(primals, prob);

    // At least one partition element with >= 2 fractional variables.
    CHECK(!partition.empty());
    for (const auto& clq : partition) {
        CHECK(clq.literals.size() >= 2);
    }

    // Each variable should appear in at most one partition element.
    std::vector<int> counts(prob.num_cols, 0);
    for (const auto& clq : partition) {
        for (const auto& lit : clq.literals) {
            counts[lit.var]++;
        }
    }
    for (Index j = 0; j < prob.num_cols; ++j) {
        CHECK(counts[j] <= 1);
    }
}

TEST_CASE("CliqueTable: findSubstitutions", "[clique_table]") {
    // 2-clique equality: x0 + x1 = 1 => x1 = 1 - x0.
    LpProblem prob;
    prob.num_cols = 3;
    prob.num_rows = 2;
    prob.matrix = SparseMatrix(2, 3,
                               {
                                   {0, 0, 1.0},
                                   {0, 1, 1.0},
                                   {1, 1, 1.0},
                                   {1, 2, 1.0},
                               });
    prob.row_lower = {1.0, -kInf};
    prob.row_upper = {1.0, 1.0};
    prob.col_lower = {0.0, 0.0, 0.0};
    prob.col_upper = {1.0, 1.0, 1.0};
    prob.col_type = {VarType::Binary, VarType::Binary, VarType::Binary};
    prob.obj = {0.0, 0.0, 0.0};

    ConflictGraph graph;
    graph.build(prob);

    CliqueTable table;
    table.build(prob, graph);

    auto subs = table.findSubstitutions(prob);
    CHECK(!subs.empty());

    // At least one substitution should involve x0 and x1
    // (the 2-clique equality pair).
    bool found = false;
    for (const auto& s : subs) {
        if ((s.eliminated == 1 && s.substitute == 0) || (s.eliminated == 0 && s.substitute == 1)) {
            found = true;
            // In a non-complemented 2-clique equality x0+x1=1,
            // the substitution should be x_eliminated = 1 - x_substitute,
            // so complement should be true.
            CHECK(s.complement);
        }
    }
    CHECK(found);
}
