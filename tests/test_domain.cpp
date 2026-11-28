#include "mipx/bound_conflict.h"
#include "mipx/clique_table.h"
#include "mipx/conflict_graph.h"
#include "mipx/domain.h"
#include "mipx/dual_simplex.h"
#include "mipx/implication_graph.h"
#include "mipx/mip_solver.h"
#include "mipx/objective_propagator.h"

#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using namespace mipx;
using Catch::Matchers::WithinAbs;

namespace {

// Helper to build a simple LpProblem.
LpProblem makeProblem(Index num_cols, Index num_rows, std::vector<Triplet> triplets,
                      std::vector<Real> row_lower, std::vector<Real> row_upper,
                      std::vector<Real> col_lower, std::vector<Real> col_upper,
                      std::vector<VarType> col_type = {}) {
    LpProblem prob;
    prob.num_cols = num_cols;
    prob.num_rows = num_rows;
    prob.matrix = SparseMatrix(num_rows, num_cols, std::move(triplets));
    prob.row_lower = std::move(row_lower);
    prob.row_upper = std::move(row_upper);
    prob.col_lower = std::move(col_lower);
    prob.col_upper = std::move(col_upper);
    prob.obj.assign(num_cols, 0.0);
    if (col_type.empty()) {
        prob.col_type.assign(num_cols, VarType::Continuous);
    } else {
        prob.col_type = std::move(col_type);
    }
    return prob;
}

namespace reconvergence_fixture {

constexpr Index a = 0, b = 1, r = 2, p = 3, q = 4, u = 5, y = 6, z = 7;

LpProblem make(bool branch_fixture) {
    std::vector<Triplet> entries{{0, a, 1},  {0, r, -1}, {1, b, 1},  {1, p, -2}, {2, b, 1},
                                 {2, q, -2}, {3, a, 1},  {3, r, 1},  {3, p, 1},  {3, q, 1},
                                 {3, u, -1}, {4, u, 1},  {4, y, -2}, {5, u, 1},  {5, z, -2},
                                 {6, a, 1},  {6, y, 1},  {6, z, 1}};
    std::vector<Real> row_upper{0, 0, 0, 3, 0, 0, 2};
    const Index cols = branch_fixture ? 10 : 8;
    if (branch_fixture) {
        // Charging for the escape columns gives a and b fractional values in
        // the relaxation, so serial branching can encounter both UIPs.
        entries.insert(entries.end(), {{7, a, 1}, {7, 8, -0.5}, {8, b, 1}, {8, 9, -0.5}});
        row_upper.insert(row_upper.end(), {0.5, 0.5});
    }
    const Index rows = static_cast<Index>(row_upper.size());
    auto model =
        makeProblem(cols, rows, std::move(entries), std::vector<Real>(rows, -kInf),
                    std::move(row_upper), std::vector<Real>(cols, 0), std::vector<Real>(cols, 1),
                    std::vector<VarType>(cols, VarType::Binary));
    model.obj[a] = model.obj[b] = -1;
    model.obj[u] = 1;
    if (branch_fixture) {
        model.obj[8] = model.obj[9] = 0.75;
    }
    return model;
}

}  // namespace reconvergence_fixture

}  // namespace

TEST_CASE("DomainPropagator: explicit row budget charges the contradictory visit", "[domain]") {
    const auto problem = makeProblem(1, 1, {{0, 0, 1}}, {2}, {kInf}, {0}, {1});
    DomainPropagator propagator;
    propagator.load(problem);

    Int row_visits_left = 0;
    REQUIRE(propagator.propagateWithRowBudget(row_visits_left));
    CHECK(row_visits_left == 0);

    row_visits_left = 2;
    REQUIRE_FALSE(propagator.propagateWithRowBudget(row_visits_left));
    CHECK(row_visits_left == 1);
}

TEST_CASE("DomainPropagator: explicit row budget preserves deductions from a partial pass",
          "[domain]") {
    const auto problem =
        makeProblem(2, 2, {{0, 0, 1}, {1, 1, 1}}, {1, 1}, {kInf, kInf}, {0, 0}, {2, 2});
    DomainPropagator propagator;
    propagator.load(problem);

    Int row_visits_left = 1;
    REQUIRE(propagator.propagateWithRowBudget(row_visits_left));
    CHECK(row_visits_left == 0);
    const auto [untouched, tightened] =
        std::minmax({propagator.getLower(0), propagator.getLower(1)});
    CHECK(untouched == 0.0);
    CHECK_THAT(tightened, WithinAbs(1.0, 4 * DomainPropagator::kTol));
}

TEST_CASE("Bound conflicts: first UIP, subsumption and sibling reuse", "[domain][conflicts]") {
    // At level one a fixes r. At level two b fixes u, which forces p and q.
    // The packing row forbids a = r = p = q = 1. Column d is irrelevant.
    // A model row fixes the packing row's slack s before branching.
    constexpr Index a = 0, r = 1, b = 2, u = 3, p = 4, q = 5, d = 6, s = 7;
    const auto prob =
        makeProblem(8, 6,
                    {{0, a, 1},
                     {0, r, -1},
                     {1, b, 1},
                     {1, u, -1},
                     {2, u, 1},
                     {2, p, -1},
                     {3, u, 1},
                     {3, q, -1},
                     {4, a, 1},
                     {4, r, 1},
                     {4, p, 1},
                     {4, q, 1},
                     {4, s, -1},
                     {5, s, 1}},
                    std::vector<Real>(6, -kInf), {0, 0, 0, 0, 3, 0}, std::vector<Real>(8, 0),
                    std::vector<Real>(8, 1), std::vector<VarType>(8, VarType::Binary));
    DomainPropagator dp;
    dp.load(prob, true);
    REQUIRE(dp.propagate());
    REQUIRE(dp.getUpper(s) == 0);
    dp.setBranchBound(a, false, 1, 1);
    REQUIRE(dp.propagate());
    REQUIRE(dp.getLower(r) == 1);
    dp.setBranchBound(d, true, 0, 2);
    dp.setBranchBound(b, false, 1, 2);
    REQUIRE_FALSE(dp.propagate());

    const auto learned = analyzeBoundConflict(*dp.boundReasons());
    REQUIRE(learned);
    CHECK_FALSE(learned->scope.node_local);
    CHECK(learned->scope.cutoff == kInf);
    REQUIRE(learned->removed_literals.size() == 2);
    CHECK(learned->removed_literals.front().variable == r);
    CHECK_FALSE(learned->removed_literals.front().is_upper);
    CHECK(learned->removed_literals.front().bound == 1);
    CHECK(learned->removed_literals.front().level == 1);
    CHECK(learned->removed_literals.back().variable == s);
    CHECK(learned->removed_literals.back().is_upper);
    CHECK(learned->removed_literals.back().bound == 0);
    CHECK(learned->removed_literals.back().level == 0);
    CHECK(learned->lbd == 2);
    REQUIRE(learned->literals.size() == 2);
    const auto& first = learned->literals[0];
    const auto& uip = learned->literals[1];
    CHECK(first.variable == a);
    CHECK(first.bound == 1);
    CHECK_FALSE(first.is_upper);
    CHECK(first.level == 1);
    CHECK(uip.variable == u);
    CHECK(uip.bound == 1);
    CHECK_FALSE(uip.is_upper);
    CHECK(uip.level == 2);

    auto lb = prob.col_lower;
    const auto ub = prob.col_upper;
    lb[a] = 1;
    const auto evaluate = [&] {
        return evaluateBoundConflict(learned->literals, learned->scope, lb, ub, prob.col_type);
    };
    const auto unit = evaluate();
    REQUIRE(unit);
    REQUIRE(unit->tightening);
    CHECK(unit->tightening->col == u);
    CHECK(unit->tightening->is_upper);
    CHECK(unit->tightening->value == 0);
    REQUIRE(unit->antecedents_before.size() == 1);
    CHECK(unit->antecedents_before.front().variable == a);
    CHECK(unit->antecedents_after.empty());

    lb[u] = 1;
    const auto cutoff = evaluate();
    REQUIRE(cutoff);
    CHECK_FALSE(cutoff->tightening);
    CHECK(cutoff->antecedents_before.size() == 2);

    for (unsigned mask = 0; mask < (1U << 8); ++mask) {
        const auto bit = [mask](Index col) { return (mask >> col) & 1U; };
        if (bit(a) <= bit(r) && bit(b) <= bit(u) && bit(u) <= bit(p) && bit(u) <= bit(q) &&
            bit(s) == 0 && bit(a) + bit(r) + bit(p) + bit(q) <= 3 + bit(s)) {
            CAPTURE(mask);
            CHECK_FALSE((bit(a) == 1 && bit(u) == 1));
            if (bit(a) == 1) {
                CHECK(bit(u) <= unit->tightening->value);
            }
        }
    }

    // Importing an endpoint without its decision history makes the proof
    // opaque. Even a later subsumption must not publish it across siblings.
    dp.load(prob, true);
    REQUIRE(dp.propagate());
    dp.setReasonLevel(1);
    dp.setBound(a, 1, 1);
    REQUIRE(dp.propagate());
    dp.setBranchBound(b, false, 1, 2);
    REQUIRE_FALSE(dp.propagate());
    CHECK_FALSE(analyzeBoundConflict(*dp.boundReasons()));
}

TEST_CASE("Bound conflicts: objective scope requires an equally strong cutoff",
          "[domain][conflicts]") {
    auto prob = makeProblem(2, 0, {}, {}, {}, {0, 0}, {1, 1}, std::vector(2, VarType::Binary));
    prob.obj = {1, 1};
    ObjectivePropagator objective;
    objective.load(prob);
    DomainPropagator dp;
    dp.load(prob, true);
    dp.setObjectiveCutoff(&objective, 1.5);
    dp.setBranchBound(0, false, 1, 1);
    dp.setBranchBound(1, false, 1, 2);
    REQUIRE_FALSE(dp.propagate());
    const auto learned = analyzeBoundConflict(*dp.boundReasons());
    REQUIRE(learned);
    CHECK(learned->scope.cutoff == 1.5);
    CHECK_FALSE(learned->scope.node_local);

    const std::vector<Real> lower = {1, 0};
    const auto evaluate = [&](Real active_cutoff) {
        return evaluateBoundConflict(learned->literals, learned->scope, lower, prob.col_upper,
                                     prob.col_type, active_cutoff);
    };
    for (const Real active_cutoff : {1.25, 1.5}) {
        const auto unit = evaluate(active_cutoff);
        REQUIRE(unit);
        REQUIRE(unit->tightening);
        CHECK(unit->tightening->col == 1);
        CHECK(unit->tightening->is_upper);
        CHECK(unit->tightening->value == 0);
        CHECK(unit->scope.cutoff == 1.5);
    }
    CHECK_FALSE(evaluate(2.5));  // Both ones are feasible here.
    CHECK_FALSE(evaluate(kInf));
}

TEST_CASE("Bound conflicts: check and strengthen a binary reconvergence cut",
          "[domain][conflicts]") {
    // b reaches u through p and q. Both a and r support the reason for u,
    // but a already forces r. At the next fork u forces y and z, which
    // conflict when a=1. The two UIPs are b and u.
    using namespace reconvergence_fixture;
    auto model = make(false);

    const auto learn = [&](Index global_rows) {
        DomainPropagator dp;
        dp.load(model, true, global_rows);
        dp.setBranchBound(a, false, 1, 1);
        REQUIRE(dp.propagate());
        REQUIRE(dp.getLower(r) == 1);
        dp.setBranchBound(b, false, 1, 2);
        REQUIRE_FALSE(dp.propagate());
        return analyzeBoundConflict(*dp.boundReasons(), true);
    };
    const auto proof = learn(model.num_rows);
    REQUIRE(proof);
    const auto candidate = checkedReconvergenceCut(proof->reconvergence, model);
    REQUIRE(candidate);
    CutPool pool;
    pool.setMinEfficacy(0.0);
    REQUIRE(pool.addCut(*candidate));
    const auto& cut = pool[0];
    CHECK(cut.indices == std::vector<Index>{a, b, u});
    CHECK(cut.values == std::vector<Real>{1, 1, -1});
    CHECK(cut.upper == 1);
    CHECK(cut.lower == -kInf);
    CHECK_FALSE(cut.local);

    const auto feasible = [&](const std::array<Real, 8>& point) {
        for (Index row = 0; row < model.num_rows; ++row) {
            const auto sparse = model.matrix.row(row);
            Real activity = 0.0;
            for (Index k = 0; k < sparse.size(); ++k) {
                activity += sparse.values[k] * point[sparse.indices[k]];
            }
            if (activity > model.row_upper[row]) {
                return false;
            }
        }
        return true;
    };
    for (unsigned mask = 0; mask < (1U << 8); ++mask) {
        std::array<Real, 8> point{};
        for (Index col = 0; col < model.num_cols; ++col) {
            point[col] = (mask >> col) & 1U;
        }
        if (feasible(point)) {
            CAPTURE(mask);
            CHECK(cutViolation(cut, point) == 0);
        }
    }

    const std::array<Real, 8> fractional{0.5, 1, 0.5, 0.5, 0.5, 0, 0, 0};
    REQUIRE(feasible(fractional));
    // The unstrengthened row is a + r + b - u <= 2.
    CHECK(fractional[a] + fractional[r] + fractional[b] - fractional[u] == 2);
    CHECK(cutViolation(cut, fractional) == 0.5);

    DualSimplexSolver lp;
    lp.setVerbose(false);
    lp.load(model);
    const auto baseline = lp.solve();
    REQUIRE(baseline.status == Status::Optimal);
    CHECK_THAT(baseline.objective, WithinAbs(-2, 1e-9));
    constexpr std::array<Index, 1> starts{0};
    lp.addRows(starts, cut.indices, cut.values, std::span(&cut.lower, 1), std::span(&cut.upper, 1));
    const auto tightened = lp.solve();
    REQUIRE(tightened.status == Status::Optimal);
    CHECK_THAT(tightened.objective, WithinAbs(-1, 1e-9));

    CHECK_FALSE(learn(model.num_rows - 1));
    auto unsupported = proof->reconvergence;
    unsupported.front().bound = -1;
    CHECK_FALSE(checkedReconvergenceCut(unsupported, model));
    model.col_type[b] = VarType::Integer;
    model.col_upper[b] = 2;
    CHECK_FALSE(checkedReconvergenceCut(proof->reconvergence, model));
    model.col_upper[b] = 1;
    model.row_upper[1] = 1;  // b=1 no longer forces p=1.
    CHECK_FALSE(checkedReconvergenceCut(proof->reconvergence, model));
}

TEST_CASE("MipSolver: install a checked reconvergence cut in the serial tree", "[mip][conflicts]") {
    const bool lp_proofs = GENERATE(false, true);
    MipSolver solver;
    solver.setVerbose(false);
    solver.setNumThreads(1);
    solver.setPresolve(false);
    solver.setSymmetryEnabled(false);
    solver.setProbingEnabled(false);
    solver.setStrongBranchingEnabled(false);
    solver.setBalancedBranchScore(true);
    solver.setTreePresolveEnabled(false);
    solver.setMaxCutRounds(0);
    solver.setConflictLpProofsEnabled(lp_proofs);
    solver.load(reconvergence_fixture::make(true));
    const auto baseline = solver.solve();
    REQUIRE(baseline.status == Status::Optimal);
    CHECK_THAT(baseline.objective, WithinAbs(-0.25, 1e-9));
    CHECK(solver.getConflictStats().reconvergence_cuts_added == 0);

    solver.setReconvergenceCutsEnabled(true);
    const auto learned = solver.solve();
    REQUIRE(learned.status == Status::Optimal);
    CHECK_THAT(learned.objective, WithinAbs(baseline.objective, 1e-9));
    CHECK(solver.getLpStats().pre_lp_propagation_calls > 0);
    CHECK(solver.getLpStats().pre_lp_propagation_pruned == 0);
    CHECK(solver.getConflictStats().reconvergence_cuts_added > 0);

    solver.setCutEffortMode(CutEffortMode::Off);
    const auto without_cuts = solver.solve();
    REQUIRE(without_cuts.status == Status::Optimal);
    CHECK_THAT(without_cuts.objective, WithinAbs(baseline.objective, 1e-9));
    CHECK(solver.getConflictStats().reconvergence_cuts_added == 0);
}

TEST_CASE("Bound conflicts: reject reconvergence certificates with overflowing activity",
          "[domain][conflicts]") {
    constexpr Real large = 1e308;
    // (1, 1, 1) is feasible. Summing the positive terms first in double
    // precision makes the apparent minimum activity infinite.
    const auto model =
        makeProblem(3, 1, {{0, 0, large}, {0, 1, large}, {0, 2, -large}}, {-kInf}, {1.1e308},
                    {0, 0, 0}, {1, 1, 1}, std::vector(3, VarType::Binary));
    const std::array<BranchDecision, 3> premises{{{0, 1, false}, {1, 1, false}, {2, 1, false}}};
    CHECK_FALSE(checkedReconvergenceCut(premises, model));
}

TEST_CASE("Bound conflicts: integer complement of a fractional upper endpoint",
          "[domain][conflicts]") {
    const std::vector<BranchDecision> literals = {{0, -1.5, true, 2}, {1, 3, false, 1}};
    const std::vector<Real> lower = {-5, 3}, upper = {4, 7};
    const std::vector<VarType> types = {VarType::Integer, VarType::Integer};
    const auto unit = evaluateBoundConflict(literals, {}, lower, upper, types);
    REQUIRE(unit);
    REQUIRE(unit->tightening);
    CHECK(unit->tightening->col == 0);
    CHECK_FALSE(unit->tightening->is_upper);
    CHECK(unit->tightening->value == -1);
    CHECK(unit->antecedents_before.empty());
    REQUIRE(unit->antecedents_after.size() == 1);
    CHECK(unit->antecedents_after.front().variable == 1);

    CHECK_FALSE(evaluateBoundConflict(literals, {}, std::vector<Real>{-1, 3}, upper, types));
    CHECK_FALSE(evaluateBoundConflict(literals, {}, std::vector<Real>{-5, 2}, upper, types));
    CHECK_FALSE(evaluateBoundConflict(literals, {}, lower, upper,
                                      std::vector{VarType::Continuous, VarType::Integer}));
    CHECK_FALSE(evaluateBoundConflict(literals, {}, lower, upper,
                                      std::vector{VarType::SemiContinuous, VarType::Integer}));
}

TEST_CASE("Bound reasons: row antecedents survive tighter bounds and undo", "[domain][reasons]") {
    // -x + y <= 0. A reason for y's upper bound uses x's upper bound.
    const auto prob = makeProblem(2, 1, {{0, 0, -1}, {0, 1, 1}}, {-kInf}, {0}, {0, 0}, {10, 10},
                                  {VarType::Integer, VarType::Integer});
    DomainPropagator dp;
    dp.load(prob);
    CHECK(dp.boundReasons() == nullptr);
    dp.load(prob, true);
    const auto& trail = *dp.boundReasons();
    dp.setBranchBound(0, true, 4, 1);
    REQUIRE(dp.propagate());
    const auto x4 = trail.current(0, true);
    const auto y4 = trail.current(1, true);
    CHECK(trail.reason(x4).source == BoundReasonSource::Branch);
    CHECK(trail.reason(x4).level == 1);
    CHECK(std::ranges::equal(trail.reason(y4).supports.ids(), std::vector{x4}));

    dp.pushCheckpoint();
    dp.setBranchBound(0, true, 2, 2);
    REQUIRE(dp.propagate());
    const auto x2 = trail.current(0, true);
    const auto y2 = trail.current(1, true);
    const auto& row = trail.reason(y2);
    CHECK(row.endpoint.col == 1);
    CHECK(row.endpoint.is_upper);
    CHECK(row.endpoint.value == 2);
    CHECK(row.source == BoundReasonSource::Row);
    CHECK(row.source_index == 0);
    CHECK(row.level == 2);
    CHECK_FALSE(row.scope.node_local);
    CHECK(row.scope.cutoff == kInf);
    CHECK(std::ranges::equal(row.supports.ids(), std::vector{x2}));
    CHECK(row.previous == y4);
    CHECK(std::ranges::equal(trail.reason(y4).supports.ids(), std::vector{x4}));

    dp.setBranchBound(1, false, 3, 3);
    REQUIRE(trail.contradiction());
    CHECK(trail.contradiction()->source == BoundReasonSource::Crossing);
    CHECK(std::ranges::equal(trail.contradiction()->supports.ids(),
                             std::vector{trail.current(1, false), y2}));
    dp.popCheckpoint();
    CHECK_FALSE(trail.contradiction());
    CHECK(dp.getUpper(1) == 4);
    CHECK(trail.current(0, true) == x4);
    CHECK(trail.current(1, true) == y4);
}

TEST_CASE("Bound reasons: implications retain each edge in a chain", "[domain][reasons]") {
    // a <= x fixes x from the branch a >= 1, then x => !y => z.
    const auto prob = makeProblem(4, 1, {{0, 3, 1}, {0, 0, -1}}, {-kInf}, {0}, {0, 0, 0, 0},
                                  {1, 1, 1, 1}, std::vector(4, VarType::Binary));
    ImplicationGraph graph;
    graph.init(4, {0, 1, 2, 3});
    REQUIRE(graph.addImplication(0, true, 1, false));
    REQUIRE(graph.addImplication(1, false, 2, true));
    DomainPropagator dp;
    dp.load(prob, true);
    dp.setImplicationGraph(&graph);
    dp.setBranchBound(3, false, 1, 1);
    REQUIRE(dp.propagate());
    const auto& trail = *dp.boundReasons();
    const auto x = trail.current(0, false);
    const auto y = trail.current(1, true);
    const auto z = trail.current(2, false);
    CHECK(trail.reason(x).source == BoundReasonSource::Row);
    CHECK(std::ranges::equal(trail.reason(x).supports.ids(), std::vector{trail.current(3, false)}));
    CHECK(trail.reason(y).source == BoundReasonSource::Implication);
    CHECK(trail.reason(y).source_index == 0);
    CHECK(trail.reason(y).endpoint.value == 0);
    CHECK(std::ranges::equal(trail.reason(y).supports.ids(), std::vector{x}));
    CHECK(trail.reason(z).source == BoundReasonSource::Implication);
    CHECK(trail.reason(z).source_index == 1);
    CHECK(trail.reason(z).endpoint.value == 1);
    CHECK(trail.reason(z).level == 1);
    CHECK(std::ranges::equal(trail.reason(z).supports.ids(), std::vector{y}));
}

TEST_CASE("Bounded inference carries an appended clique fixing through an implication and a row",
          "[domain][cuts][clique]") {
    // a = 1, cut a+b <= 1, implication !b => c, row c <= d.
    const auto model = makeProblem(4, 1, {{0, 2, 1}, {0, 3, -1}}, {-kInf}, {0}, {1, 0, 0, 0},
                                   {1, 1, 1, 1}, std::vector(4, VarType::Binary));
    ConflictGraph conflicts;
    conflicts.build(model);
    CliqueTable cliques;
    cliques.build(model, conflicts);
    REQUIRE(cliques.numCliques() == 0);
    const Cut cut{.indices = {0, 1}, .values = {1, 1}, .upper = 1};
    REQUIRE(cliques.extractFromCut(cut, model, conflicts) == 1);
    ImplicationGraph implications;
    implications.init(4, {0, 1, 2, 3});
    REQUIRE(implications.addImplication(1, false, 2, true));

    DomainPropagator dp;
    dp.load(model);
    dp.setCliqueTable(&cliques);
    dp.setImplicationGraph(&implications);
    Int budget = 4;
    DomainPropagationWork work;
    REQUIRE(dp.propagateWithInferenceBudget(budget, work));
    CHECK(dp.getUpper(1) == 0);
    CHECK(dp.getLower(2) == 1);
    CHECK(dp.getLower(3) == 1);
    CHECK(work.cliques.visits == 1);
    CHECK(work.cliques.tightenings == 1);
    CHECK(work.implications.visits == 1);
    CHECK(work.implications.tightenings == 1);
    CHECK(work.rows.visits == 2);
    CHECK(work.rows.tightenings == 1);
    CHECK(budget == 0);

    cliques.truncate(0);
    CHECK(cliques.cliquesOf({0, false}).empty());
    CHECK(cliques.cliquesOf({1, false}).empty());
    dp.load(model);
    budget = 4;
    work = {};
    REQUIRE(dp.propagateWithInferenceBudget(budget, work));
    CHECK(dp.getUpper(1) == 1);
    CHECK(dp.getLower(2) == 0);
    CHECK(dp.getLower(3) == 0);
    CHECK(work.cliques.visits == 0);
    CHECK(work.implications.visits == 0);
    CHECK(work.rows.tightenings == 0);
}

TEST_CASE("Bounded inference uses remaining visits after a late row fixes a binary", "[domain]") {
    const auto model = makeProblem(2, 3, {{0, 0, 1}}, {1, -kInf, -kInf}, {kInf, kInf, kInf}, {0, 0},
                                   {1, 1}, std::vector(2, VarType::Binary));
    ImplicationGraph graph;
    graph.init(2, {0, 1});
    REQUIRE(graph.addImplication(0, true, 1, true));
    DomainPropagator dp;
    dp.load(model);
    dp.setImplicationGraph(&graph);
    DomainPropagationWork work;
    Int budget = 4;
    REQUIRE(dp.propagateWithInferenceBudget(budget, work));
    CHECK(budget == 0);
    CHECK(work.rows.visits == 3);
    CHECK(work.implications.visits == 1);
    CHECK(dp.getLower(1) == 1);
}

TEST_CASE("Bounded inference visits pending cliques before unrelated rows", "[domain][clique]") {
    const bool from_implication = GENERATE(false, true);
    // Inert rows would fill the whole shared budget. The initial clique
    // scan sees only two cliques. Either the third is already actionable,
    // or an implication makes the second clique actionable after that scan.
    const auto model =
        makeProblem(6, 8, {}, std::vector<Real>(8, -kInf), std::vector<Real>(8, kInf),
                    {0, 0, 0, 0, from_implication ? 0.0 : 1.0, 0},
                    {from_implication ? 0.0 : 1.0, 1, 1, 1, 1, 1}, std::vector(6, VarType::Binary));
    ConflictGraph conflicts;
    conflicts.build(model);
    CliqueTable cliques;
    cliques.build(model, conflicts);
    for (Index j : {0, 2, 4}) {
        const Cut cut{.indices = {j, j + 1}, .values = {1, 1}, .upper = 1};
        REQUIRE(cliques.extractFromCut(cut, model, conflicts) == 1);
    }
    ImplicationGraph implications;
    implications.init(6, {0, 1, 2, 3, 4, 5});
    REQUIRE(implications.addImplication(0, false, from_implication ? 2 : 1, true));
    DomainPropagator dp;
    dp.load(model);
    dp.setCliqueTable(&cliques);
    dp.setImplicationGraph(&implications);
    Int budget = 8;
    DomainPropagationWork work;
    REQUIRE(dp.propagateWithInferenceBudget(budget, work));
    CHECK(dp.getUpper(from_implication ? 3 : 5) == 0);
    CHECK(work.implications.visits == (from_implication ? 1 : 0));
    CHECK(work.cliques.tightenings == 1);
    CHECK(work.cliques.visits + work.implications.visits + work.rows.visits == 8);
    CHECK(budget == 0);
}

TEST_CASE("Inactive pool propagation uses satisfied global cuts and wakes active rows",
          "[domain][cuts]") {
    const auto model = makeProblem(7, 3, {{0, 0, 1}, {0, 6, 1}}, {1, -kInf, -kInf},
                                   {kInf, kInf, kInf}, std::vector<Real>(7, 0),
                                   std::vector<Real>(7, 1), std::vector(7, VarType::Binary));
    CutPool pool;
    for (Index j = 0; j < 6; ++j) {
        Cut cut{.indices = {j},
                .values = {1},
                .upper = 0.75,
                .local = j == 1,
                .efficacy = 1,
                .valid_cutoff = j == 2 ? 5.0 : kInf};
        REQUIRE(pool.addCut(std::move(cut)));
    }
    pool.invalidate(std::array<Index, 1>{3});
    pool.markInstalled(std::array<Index, 1>{4});
    const std::array<Index, 1> selected{5};
    const auto candidates = pool.inactiveGlobalCuts(7, selected);
    REQUIRE(candidates.size() == 1);
    CHECK(candidates.front() == &pool[0]);
    const std::array<Real, 7> primals{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5};
    CHECK(cutViolation(*candidates.front(), primals) == 0.0);
    CHECK(pool.inactiveGlobalCuts(0, selected).empty());

    DomainPropagator dp;
    dp.load(model);
    Int budget = 2;
    DomainPropagationWork work;
    REQUIRE(dp.propagateWithInactiveCuts(budget, work, candidates));
    CHECK(budget == 0);
    CHECK(work.inactive_cuts.visits == 1);
    CHECK(work.inactive_cuts.tightenings == 1);
    CHECK(work.rows.visits == 1);
    CHECK(work.rows.tightenings == 1);
    CHECK(dp.getUpper(0) == 0);
    CHECK(dp.getLower(6) == 1);
    for (Index j = 1; j < 6; ++j) {
        CHECK(dp.getUpper(j) == 1);
    }
    Index next = 5;
    pool.purge(10, &next);
    CHECK(pool.size() == 5);
    REQUIRE(next == 4);
    const auto resumed = pool.inactiveGlobalCuts(1, {}, &next);
    REQUIRE(resumed.size() == 1);
    CHECK(resumed.front()->indices.front() == 5);
}

TEST_CASE("Bound reasons: clique records the forcing literal", "[domain][reasons]") {
    const auto binaries = std::vector(3, VarType::Binary);
    const auto packing = makeProblem(3, 1, {{0, 0, 1}, {0, 1, 1}, {0, 2, 1}}, {-kInf}, {1},
                                     {0, 0, 0}, {1, 1, 1}, binaries);
    const auto domain = makeProblem(3, 0, {}, {}, {}, {0, 0, 0}, {1, 1, 1}, binaries);
    ConflictGraph graph;
    graph.build(packing);
    CliqueTable table;
    table.build(packing, graph);
    DomainPropagator dp;
    dp.load(domain, true);
    dp.setCliqueTable(&table);
    dp.setBranchBound(1, false, 1, 1);
    REQUIRE(dp.propagate());
    const auto& trail = *dp.boundReasons();
    const auto& reason = trail.reason(trail.current(0, true));
    CHECK(reason.source == BoundReasonSource::Clique);
    CHECK(reason.endpoint.value == 0);
    CHECK(std::ranges::equal(reason.supports.ids(), std::vector{trail.current(1, false)}));
    CHECK(table.clique(reason.source_index).literals.size() >= 2);

    dp.load(domain, true);
    dp.setBranchBound(0, false, 1, 1);
    dp.setBranchBound(1, false, 1, 2);
    CHECK_FALSE(dp.propagate());
    const auto& crossed = *dp.boundReasons();
    REQUIRE(crossed.contradiction());
    CHECK(crossed.contradiction()->source == BoundReasonSource::Clique);
    CHECK(std::ranges::distance(crossed.contradiction()->supports.ids()) == 2);
    for (auto id : crossed.contradiction()->supports.ids()) {
        CHECK(crossed.reason(id).source == BoundReasonSource::Branch);
        CHECK(crossed.reason(id).endpoint.value == 1);
    }
}

TEST_CASE("Bound reasons: equality clique uses all other false literals", "[domain][reasons]") {
    const auto binaries = std::vector(3, VarType::Binary);
    const auto partition = makeProblem(3, 1, {{0, 0, 1}, {0, 1, 1}, {0, 2, 1}}, {1}, {1}, {0, 0, 0},
                                       {1, 1, 1}, binaries);
    const auto domain = makeProblem(3, 0, {}, {}, {}, {0, 0, 0}, {1, 1, 1}, binaries);
    ConflictGraph graph;
    graph.build(partition);
    CliqueTable table;
    table.build(partition, graph);
    DomainPropagator dp;
    dp.load(domain, true);
    dp.setCliqueTable(&table);
    dp.setBranchBound(0, true, 0, 1);
    dp.setBranchBound(2, true, 0, 2);
    REQUIRE(dp.propagate());
    const auto& trail = *dp.boundReasons();
    const auto& reason = trail.reason(trail.current(1, false));
    CHECK(reason.source == BoundReasonSource::Clique);
    CHECK(reason.endpoint.value == 1);
    CHECK(reason.level == 2);
    CHECK(std::ranges::equal(reason.supports.ids(),
                             std::vector{trail.current(0, true), trail.current(2, true)}));
    CHECK(table.clique(reason.source_index).is_equality);
}

TEST_CASE("Bound reasons: cutoff and local proof scopes compose", "[domain][reasons]") {
    // -x + y <= 1.5 (the objective cutoff), and z <= y (a local row).
    auto prob = makeProblem(3, 1, {{0, 1, -1}, {0, 2, 1}}, {-kInf}, {0}, {0, 0, 0}, {1, 5, 5},
                            {VarType::Binary, VarType::Integer, VarType::Integer});
    prob.obj = {-1, 1, 0};
    ObjectivePropagator objective;
    objective.load(prob);
    DomainPropagator dp;
    dp.load(prob, true, 0);
    dp.setObjectiveCutoff(&objective, 1.5);
    dp.setReasonLevel(2);
    dp.setBranchBound(0, true, 0, 1);
    REQUIRE(dp.propagate());
    const auto& trail = *dp.boundReasons();
    const auto y = trail.current(1, true);
    const auto& cutoff = trail.reason(y);
    CHECK(cutoff.source == BoundReasonSource::Objective);
    CHECK(cutoff.endpoint.value == 1);
    CHECK(cutoff.level == 1);
    CHECK(std::ranges::equal(cutoff.supports.ids(), std::vector{trail.current(0, true)}));
    CHECK_FALSE(cutoff.scope.node_local);
    CHECK(cutoff.scope.cutoff == 1.5);
    const auto& local = trail.reason(trail.current(2, true));
    CHECK(local.source == BoundReasonSource::Row);
    CHECK(local.endpoint.value == 1);
    CHECK(local.level == 2);
    CHECK(std::ranges::equal(local.supports.ids(), std::vector{y}));
    CHECK(local.scope.node_local);
    CHECK(local.scope.cutoff == 1.5);

    dp.load(prob, true, 0);
    dp.setBranchBound(0, true, 0, 1);
    dp.setBranchBound(1, false, 2, 2);
    CHECK_FALSE(dp.propagate());
    REQUIRE(dp.boundReasons()->contradiction());
    const auto& conflict = *dp.boundReasons()->contradiction();
    CHECK(conflict.source == BoundReasonSource::Objective);
    CHECK(conflict.scope.cutoff == 1.5);
    CHECK_FALSE(conflict.scope.node_local);
}

TEST_CASE("Bound reasons: unclassified bounds stay opaque", "[domain][reasons]") {
    const auto prob = makeProblem(1, 0, {}, {}, {}, {0}, {5}, {VarType::Integer});
    DomainPropagator dp;
    dp.load(prob, true);
    dp.setReasonLevel(2);
    dp.setBound(0, 3, 5);
    dp.setBranchBound(0, true, 2, 2);
    const auto& trail = *dp.boundReasons();
    REQUIRE(trail.contradiction());
    const auto& conflict = *trail.contradiction();
    CHECK(conflict.scope.node_local);
    const auto lower = trail.current(0, false);
    CHECK(std::ranges::equal(conflict.supports.ids(), std::vector{lower, trail.current(0, true)}));
    CHECK(trail.reason(lower).source == BoundReasonSource::Opaque);
    CHECK(trail.reason(lower).endpoint.value == 3);
    CHECK(trail.reason(lower).level == 2);
    CHECK(trail.reason(lower).supports.ids().empty());

    dp.load(prob);
    CHECK(dp.boundReasons() == nullptr);
}

TEST_CASE("Bound reasons: importing the path preserves row visit order", "[domain][reasons]") {
    // x0 <= x1 and x2 <= x3. Imported column order differs from path order.
    const auto prob =
        makeProblem(4, 2, {{0, 0, 1}, {0, 1, -1}, {1, 2, 1}, {1, 3, -1}}, {-kInf, -kInf}, {0, 0},
                    {0, 0, 0, 0}, {10, 10, 10, 10}, std::vector(4, VarType::Integer));
    const std::vector<BranchDecision> path{{2, 1, false, 1}, {0, 2, false, 2}, {0, 3, false, 3}};
    DomainPropagator plain;
    plain.load(prob);
    plain.setBound(0, 4, 10);
    plain.setBound(2, 1, 10);
    REQUIRE(plain.propagate(1));

    DomainPropagator traced;
    traced.load(prob, true);
    traced.setReasonLevel(4);
    traced.setBound(0, 4, 10, path);
    traced.setBound(2, 1, 10, path);
    REQUIRE(traced.propagate(1));
    CHECK(traced.numTightened() == plain.numTightened());
    CHECK(traced.getLower(1) == plain.getLower(1));
    CHECK(traced.getLower(3) == plain.getLower(3));
    CHECK(traced.getLower(3) == 1);

    const auto& trail = *traced.boundReasons();
    const auto& opaque = trail.reason(trail.current(0, false));
    CHECK(opaque.source == BoundReasonSource::Opaque);
    CHECK(opaque.level == 4);
    const auto& last_branch = trail.reason(opaque.previous);
    CHECK(last_branch.source == BoundReasonSource::Branch);
    CHECK(last_branch.level == 3);
    CHECK(last_branch.endpoint.value == 3);
    CHECK(trail.reason(last_branch.previous).level == 2);
    CHECK(trail.reason(last_branch.previous).endpoint.value == 2);
}

TEST_CASE("Bound reasons: rejected objective crossing leaves the domain intact",
          "[domain][reasons]") {
    auto prob = makeProblem(1, 0, {}, {}, {}, {0.5}, {2}, {VarType::Integer});
    prob.obj = {1};
    ObjectivePropagator objective;
    objective.load(prob);
    DomainPropagator dp;
    dp.load(prob, true);
    dp.setObjectiveCutoff(&objective, 0.75);
    const auto& trail = *dp.boundReasons();
    const auto lower_before = trail.current(0, false);
    const auto upper_before = trail.current(0, true);
    dp.pushCheckpoint();
    CHECK_FALSE(dp.propagate());
    REQUIRE(trail.contradiction());
    const auto& crossing = *trail.contradiction();
    CHECK(crossing.source == BoundReasonSource::Crossing);
    CHECK(crossing.scope.cutoff == 0.75);
    auto supports = crossing.supports.ids();
    REQUIRE(std::ranges::distance(supports) == 2);
    for (auto id : supports) {
        const auto& reason = trail.reason(id);
        CHECK(reason.previous == (reason.endpoint.is_upper ? upper_before : lower_before));
        CHECK(reason.endpoint.value == (reason.endpoint.is_upper ? 0 : 1));
        CHECK(reason.source == (reason.endpoint.is_upper ? BoundReasonSource::Objective
                                                         : BoundReasonSource::IntegerRounding));
    }
    CHECK(dp.getLower(0) == 0.5);
    CHECK(dp.getUpper(0) == 2);
    CHECK(dp.numTightened() == 0);
    CHECK(trail.current(0, false) == lower_before);
    CHECK(trail.current(0, true) == upper_before);
    dp.popCheckpoint();
    CHECK_FALSE(trail.contradiction());
    CHECK(trail.current(0, false) == lower_before);
    CHECK(trail.current(0, true) == upper_before);
}

TEST_CASE("Bound reasons: rejected clique crossing keeps the original endpoint",
          "[domain][reasons]") {
    const auto binaries = std::vector(2, VarType::Binary);
    const auto packing =
        makeProblem(2, 1, {{0, 0, 1}, {0, 1, 1}}, {-kInf}, {1}, {0, 0}, {1, 1}, binaries);
    const auto domain = makeProblem(2, 0, {}, {}, {}, {0, 0.5}, {1, 1}, binaries);
    ConflictGraph graph;
    graph.build(packing);
    CliqueTable table;
    table.build(packing, graph);
    DomainPropagator dp;
    dp.load(domain, true);
    dp.setCliqueTable(&table);
    dp.setBranchBound(0, false, 1, 1);
    const auto& trail = *dp.boundReasons();
    const auto upper_before = trail.current(1, true);
    dp.pushCheckpoint();
    CHECK_FALSE(dp.propagate());
    REQUIRE(trail.contradiction());
    CHECK(trail.contradiction()->source == BoundReasonSource::Crossing);
    auto supports = trail.contradiction()->supports.ids();
    REQUIRE(std::ranges::distance(supports) == 2);
    auto support = supports.begin();
    CHECK(*support == trail.current(1, false));
    ++support;
    const auto& rejected = trail.reason(*support);
    CHECK(rejected.previous == upper_before);
    CHECK(rejected.source == BoundReasonSource::Clique);
    CHECK(rejected.endpoint.value == 0);
    CHECK(std::ranges::equal(rejected.supports.ids(), std::vector{trail.current(0, false)}));
    CHECK(dp.getUpper(1) == 1);
    CHECK(dp.numTightened() == 0);
    dp.popCheckpoint();
    CHECK_FALSE(trail.contradiction());
    CHECK(trail.current(1, true) == upper_before);
}

TEST_CASE("Domain: simple bound tightening", "[domain]") {
    // x + y <= 5, x >= 3, y >= 0
    // Should tighten y <= 2.
    auto prob =
        makeProblem(2, 1, {{0, 0, 1.0}, {0, 1, 1.0}}, {-kInf}, {5.0}, {3.0, 0.0}, {kInf, kInf});

    DomainPropagator dp;
    dp.load(prob);
    REQUIRE(dp.propagate());
    CHECK_THAT(dp.getUpper(1), WithinAbs(2.0, 1e-7));
    CHECK_THAT(dp.getLower(0), WithinAbs(3.0, 1e-7));
}

TEST_CASE("Domain: integer rounding", "[domain]") {
    // 2x <= 5, x >= 0, x integer => x <= 2
    auto prob = makeProblem(1, 1, {{0, 0, 2.0}}, {-kInf}, {5.0}, {0.0}, {kInf}, {VarType::Integer});

    DomainPropagator dp;
    dp.load(prob);
    REQUIRE(dp.propagate());
    CHECK_THAT(dp.getUpper(0), WithinAbs(2.0, 1e-7));
}

TEST_CASE("Domain: infeasibility detection", "[domain]") {
    // Constraint with rl > ru is infeasible.
    auto prob = makeProblem(1, 1, {{0, 0, 1.0}}, {5.0}, {3.0}, {0.0}, {kInf});

    DomainPropagator dp;
    dp.load(prob);
    CHECK_FALSE(dp.propagate());
}

TEST_CASE("Domain: infeasibility from bound tightening", "[domain]") {
    // x + y <= 3, x >= 2, y >= 2 => x <= 1 and y <= 1, contradicts.
    auto prob =
        makeProblem(2, 1, {{0, 0, 1.0}, {0, 1, 1.0}}, {-kInf}, {3.0}, {2.0, 2.0}, {kInf, kInf});

    DomainPropagator dp;
    dp.load(prob);
    CHECK_FALSE(dp.propagate());
}

TEST_CASE("Domain: checkpoint/restore", "[domain]") {
    // x + y <= 10, x >= 0, y >= 0
    auto prob =
        makeProblem(2, 1, {{0, 0, 1.0}, {0, 1, 1.0}}, {-kInf}, {10.0}, {0.0, 0.0}, {kInf, kInf});

    DomainPropagator dp;
    dp.load(prob);
    REQUIRE(dp.propagate());
    CHECK_THAT(dp.getUpper(0), WithinAbs(10.0, 1e-7));
    CHECK_THAT(dp.getUpper(1), WithinAbs(10.0, 1e-7));

    // Push checkpoint, then tighten x >= 7.
    dp.pushCheckpoint();
    dp.setBound(0, 7.0, dp.getUpper(0));
    REQUIRE(dp.propagate());
    CHECK_THAT(dp.getLower(0), WithinAbs(7.0, 1e-7));
    // A derived bound keeps the row's slack, kTol times its magnitude (#236).
    CHECK_THAT(dp.getUpper(1), WithinAbs(3.0, 2 * DomainPropagator::kTol * 10.0));
    CHECK(dp.getUpper(1) >= 3.0);
    CHECK(dp.numTightened() > 0);

    // Restore — bounds should go back.
    dp.popCheckpoint();
    CHECK_THAT(dp.getLower(0), WithinAbs(0.0, 1e-7));
    CHECK_THAT(dp.getUpper(1), WithinAbs(10.0, 1e-7));
}

TEST_CASE("Domain: chain of implications", "[domain]") {
    // x + y <= 10
    // y + z <= 6
    // x >= 5
    // All non-negative.
    // => y <= 5 (from row 1), z <= 6 (from row 2 with y >= 0).
    auto prob = makeProblem(3, 2, {{0, 0, 1.0}, {0, 1, 1.0}, {1, 1, 1.0}, {1, 2, 1.0}},
                            {-kInf, -kInf}, {10.0, 6.0}, {5.0, 0.0, 0.0}, {kInf, kInf, kInf});

    DomainPropagator dp;
    dp.load(prob);
    REQUIRE(dp.propagate());
    CHECK_THAT(dp.getUpper(1), WithinAbs(5.0, 1e-7));
    CHECK_THAT(dp.getUpper(2), WithinAbs(6.0, 1e-7));
}

TEST_CASE("Domain: free variables", "[domain]") {
    // x + y <= 10, x free, y free
    // Both free => act_min = -inf (2 inf contributions), act_max = +inf (2 inf).
    // No tightening possible.
    auto prob = makeProblem(2, 1, {{0, 0, 1.0}, {0, 1, 1.0}}, {-kInf}, {10.0}, {-kInf, -kInf},
                            {kInf, kInf});

    DomainPropagator dp;
    dp.load(prob);
    REQUIRE(dp.propagate());
    CHECK(dp.getUpper(0) == kInf);
    CHECK(dp.getUpper(1) == kInf);
    CHECK(dp.getLower(0) == -kInf);
    CHECK(dp.getLower(1) == -kInf);
}

TEST_CASE("Domain: no tightening when already tight", "[domain]") {
    // x + y <= 10, x in [0, 5], y in [0, 5]
    // Activity max = 10 = row_upper, so no tightening needed.
    auto prob =
        makeProblem(2, 1, {{0, 0, 1.0}, {0, 1, 1.0}}, {-kInf}, {10.0}, {0.0, 0.0}, {5.0, 5.0});

    DomainPropagator dp;
    dp.load(prob);
    dp.pushCheckpoint();
    REQUIRE(dp.propagate());
    CHECK(dp.numTightened() == 0);
    CHECK_THAT(dp.getUpper(0), WithinAbs(5.0, 1e-7));
    CHECK_THAT(dp.getUpper(1), WithinAbs(5.0, 1e-7));
}

TEST_CASE("Domain: negative coefficients", "[domain]") {
    // -2x + y = 4, x in [0, 10], y in [0, 10]
    // => x <= 3, y >= 4.
    auto prob =
        makeProblem(2, 1, {{0, 0, -2.0}, {0, 1, 1.0}}, {4.0}, {4.0}, {0.0, 0.0}, {10.0, 10.0});

    DomainPropagator dp;
    dp.load(prob);
    REQUIRE(dp.propagate());
    CHECK_THAT(dp.getUpper(0), WithinAbs(3.0, 1e-7));
    CHECK_THAT(dp.getLower(1), WithinAbs(4.0, 1e-7));
}

TEST_CASE("Domain: binary variable fixing", "[domain]") {
    // 3x + 2y >= 5, x binary, y binary
    // act_max = 5 = rl, so both must be 1.
    auto prob = makeProblem(2, 1, {{0, 0, 3.0}, {0, 1, 2.0}}, {5.0}, {kInf}, {0.0, 0.0}, {1.0, 1.0},
                            {VarType::Binary, VarType::Binary});

    DomainPropagator dp;
    dp.load(prob);
    REQUIRE(dp.propagate());
    CHECK_THAT(dp.getLower(0), WithinAbs(1.0, 1e-7));
    CHECK_THAT(dp.getLower(1), WithinAbs(1.0, 1e-7));
}

// A clique that no single row implies: the row x + y + z <= 2 creates no
// pairwise conflict, so row propagation alone learns nothing when x is fixed
// to 1. The conflict graph is given the three edges explicitly, which makes
// {x, y, z} a maximal clique, and clique propagation must then force y = z = 0.
// Without this, DomainPropagator + CliqueTable had no end-to-end coverage.
TEST_CASE("DomainPropagator: clique table infers fixings rows cannot", "[domain][clique]") {
    LpProblem problem;
    problem.name = "loose_row_triangle";
    problem.sense = Sense::Minimize;
    problem.num_cols = 3;
    problem.obj = {-1.0, -1.0, -1.0};
    problem.col_lower = {0.0, 0.0, 0.0};
    problem.col_upper = {1.0, 1.0, 1.0};
    problem.col_type = {VarType::Binary, VarType::Binary, VarType::Binary};
    problem.col_names = {"x", "y", "z"};
    problem.num_rows = 1;
    problem.row_lower = {-kInf};
    problem.row_upper = {2.0};
    problem.row_names = {"loose"};
    problem.matrix = SparseMatrix(1, 3, {{0, 0, 1.0}, {0, 1, 1.0}, {0, 2, 1.0}});

    SECTION("row propagation alone cannot fix y or z") {
        DomainPropagator dp;
        dp.load(problem);
        dp.setBound(0, 1.0, 1.0);  // x branched to 1
        REQUIRE(dp.propagate());
        CHECK_THAT(dp.getUpper(1), WithinAbs(1.0, 1e-7));
        CHECK_THAT(dp.getUpper(2), WithinAbs(1.0, 1e-7));
    }

    SECTION("clique propagation fixes both") {
        ConflictGraph graph;
        graph.build(problem);
        REQUIRE(graph.numEdges() == 0);  // the loose row implies nothing
        graph.addConflict({0, false}, {1, false});
        graph.addConflict({1, false}, {2, false});
        graph.addConflict({0, false}, {2, false});
        REQUIRE(graph.numEdges() == 3);

        CliqueTable table;
        table.build(problem, graph);
        REQUIRE(table.numCliques() >= 1);

        DomainPropagator dp;
        dp.load(problem);
        dp.setCliqueTable(&table);
        dp.setBound(0, 1.0, 1.0);  // x branched to 1
        REQUIRE(dp.propagate());
        CHECK_THAT(dp.getUpper(1), WithinAbs(0.0, 1e-7));
        CHECK_THAT(dp.getUpper(2), WithinAbs(0.0, 1e-7));
        CHECK(dp.numTightened() >= 2);
    }
}

TEST_CASE("Domain: propagation stops on a row that tightens for ever", "[domain]") {
    // x - y = 0.5 over integers x, y >= 0: x >= y + 0.5 rounds x up a unit,
    // y >= x - 0.5 rounds y up a unit, without end (#242).
    auto prob = makeProblem(2, 1, {{0, 0, 1.0}, {0, 1, -1.0}}, {0.5}, {0.5}, {0.0, 0.0},
                            {kInf, kInf}, {VarType::Integer, VarType::Integer});
    DomainPropagator dp;
    dp.load(prob);
    dp.pushCheckpoint();
    CHECK(dp.propagate());
    // Each row propagation tightens both columns at most once (plus rounding).
    CHECK(dp.numTightened() <= 4 * DomainPropagator::kMaxRowRounds);
    CHECK(dp.getLower(0) > 1.0);
    dp.popCheckpoint();
    CHECK(dp.getLower(0) == 0.0);
}

TEST_CASE("Domain: exhausting a row budget does not report a contradiction", "[domain]") {
    // The last row fixes both variables to zero and requeues itself. The
    // conflicting first row is therefore still queued after two visits.
    auto prob =
        makeProblem(2, 2, {{0, 0, 1.0}, {1, 0, 1.0}, {1, 1, 1.0}}, {0.5, -kInf}, {kInf, 0.5},
                    {0.0, 0.0}, {1.0, 1.0}, {VarType::Binary, VarType::Binary});
    DomainPropagator dp;
    dp.load(prob);
    dp.pushCheckpoint();
    CHECK(dp.propagate(prob.num_rows));
    CHECK(dp.getUpper(0) == 0.0);
    CHECK_FALSE(dp.propagate(1));
    dp.popCheckpoint();
    CHECK(dp.getUpper(0) == 1.0);
}
