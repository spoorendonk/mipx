#include "../src/mip/root_cutoff_feedback.h"
#include "mipx/clique_table.h"
#include "mipx/conflict_graph.h"
#include "mipx/implication_graph.h"

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using namespace mipx;
using Catch::Matchers::WithinAbs;

namespace {

LpProblem twoIntegerColumns() {
    LpProblem model;
    model.num_cols = 2;
    model.col_lower = {0.0, 0.0};
    model.col_upper = {3.0, 3.0};
    model.col_type.assign(2, VarType::Integer);
    return model;
}

std::vector<Real> values(std::span<const Real> source) {
    return {source.begin(), source.end()};
}

}  // namespace

TEST_CASE("Root cutoff feedback: objective bounds retain every cutoff-feasible assignment",
          "[mip][cuts][domain]") {
    const bool maximize = GENERATE(false, true);
    const bool scheduled = GENERATE(false, true);
    // The inputs to root feedback are in the internal minimization form.
    // These equal-cost columns have zero reduced cost at the fractional LP optimum.
    auto model = twoIntegerColumns();
    const Real sign = maximize ? -1.0 : 1.0;
    model.obj = {sign, sign};
    model.obj_offset = sign * 7.0;
    model.num_rows = 1;
    model.row_lower = {maximize ? -kInf : 1.5};
    model.row_upper = {maximize ? 4.5 : kInf};
    model.matrix = SparseMatrix(1, 2, {{0, 0, 1.0}, {0, 1, 1.0}});
    const Real cutoff = sign * (maximize ? 11.0 : 9.0);

    DualSimplexSolver lp;
    lp.load(model);
    REQUIRE(lp.solve().status == Status::Optimal);

    Int row_budget = scheduled ? 8 : model.num_rows;
    MipCutStats stats;
    RootCutoffFeedback feedback(model, cutoff);
    if (scheduled) {
        DomainPropagationWork work;
        PropagationScheduler history;
        REQUIRE(feedback.updateScheduled(model, lp, row_budget, work, history, {}, stats).status ==
                Status::Optimal);
        const auto& objective =
            stats.root_propagation_schedule
                .producers[static_cast<std::size_t>(PropagationProducer::Objective)];
        CHECK(objective.work == 2);
        CHECK(objective.endpoints == 2);
    } else {
        REQUIRE(feedback.update(model, lp, row_budget, {}, stats).status == Status::Optimal);
    }
    CHECK(stats.root_cutoff_objective_tightenings == 2);
    CHECK(stats.root_cutoff_reduced_cost_tightenings == 0);
    CHECK(stats.root_cutoff_row_tightenings == 0);
    CHECK(stats.root_cutoff_feedbacks == 1);
    REQUIRE(feedback.lower().size() == 2);
    REQUIRE(feedback.upper().size() == 2);

    Int cutoff_feasible = 0;
    for (Int x = 0; x <= 3; ++x) {
        for (Int y = 0; y <= 3; ++y) {
            CAPTURE(x, y);
            const Real sum = x + y;
            if (sum >= model.row_lower[0] && sum <= model.row_upper[0] &&
                sign * (7.0 + sum) <= cutoff) {
                ++cutoff_feasible;
                CHECK(x >= feedback.lower()[0]);
                CHECK(x <= feedback.upper()[0]);
                CHECK(y >= feedback.lower()[1]);
                CHECK(y <= feedback.upper()[1]);
            }
        }
    }
    CHECK(cutoff_feasible == 3);
    CHECK(feedback.lower()[0] == (maximize ? 1.0 : 0.0));
    CHECK(feedback.upper()[0] == (maximize ? 3.0 : 2.0));
}

TEST_CASE("Root cutoff feedback: reduced costs wake a row with no objective deduction",
          "[mip][cuts][domain]") {
    const bool allow_rows = GENERATE(false, true);
    // The objective has two free columns, so its row has no finite residual.
    // The LP nevertheless prices x at 1 through a >= x. Fixing x to 0 implies
    // y <= 0.25, which fixes integer y to 0.
    LpProblem model;
    model.num_cols = 4;
    model.obj = {0.0, 0.0, 1.0, 1.0};
    model.col_lower = {0.0, 0.0, -kInf, -kInf};
    model.col_upper = {1.0, 1.0, kInf, kInf};
    model.col_type = {VarType::Binary, VarType::Binary, VarType::Continuous, VarType::Continuous};
    model.num_rows = 3;
    model.row_lower = {0.0, 0.0, -0.25};
    model.row_upper.assign(3, kInf);
    model.matrix =
        SparseMatrix(3, 4, {{0, 0, -1.0}, {0, 2, 1.0}, {1, 3, 1.0}, {2, 0, 1.0}, {2, 1, -1.0}});

    DualSimplexSolver lp;
    lp.load(model);
    REQUIRE(lp.solve().status == Status::Optimal);

    Int row_budget = allow_rows ? 1 : 0;
    MipCutStats stats;
    RootCutoffFeedback feedback(model, 0.25);
    REQUIRE(feedback.update(model, lp, row_budget, {}, stats).status == Status::Optimal);
    CHECK(stats.root_cutoff_reduced_cost_tightenings == 1);
    CHECK(stats.root_cutoff_objective_tightenings == 0);
    CHECK(stats.root_cutoff_row_tightenings == (allow_rows ? 1 : 0));
    CHECK(stats.root_cutoff_feedbacks == 1);
    CHECK(row_budget == 0);
    REQUIRE(feedback.upper().size() == 4);
    CHECK(feedback.upper()[0] == 0.0);
    CHECK(feedback.upper()[1] == (allow_rows ? 0.0 : 1.0));
    Real lower, upper;
    lp.getColBounds(0, lower, upper);
    CHECK(upper == 1.0);
}

TEST_CASE("Scheduled cutoff feedback reprices on a fresh LP or a stronger incumbent",
          "[mip][cuts][domain]") {
    const bool stronger_incumbent = GENERATE(false, true);
    // Two free objective columns hide x from objective-row propagation. The
    // new row a >= 1 + x prices x at 1. The incumbent (0, 0, 1.25, 0)
    // satisfies the new row and leaves a gap of 0.25 against the new LP.
    LpProblem model;
    model.num_cols = 4;
    model.obj = {0, 0, 1, 1};
    model.col_lower = {0, 0, -kInf, -kInf};
    model.col_upper = {1, 1, kInf, kInf};
    model.col_type = {VarType::Binary, VarType::Binary, VarType::Continuous, VarType::Continuous};
    model.num_rows = 3;
    model.row_lower = {0, 0, -0.25};
    model.row_upper.assign(3, kInf);
    model.matrix = SparseMatrix(3, 4, {{0, 2, 1}, {1, 3, 1}, {2, 0, 1}, {2, 1, -1}});
    DualSimplexSolver lp;
    lp.load(model);
    const auto installPricingRow = [&] {
        const std::array<Index, 1> starts{0};
        const std::array<Index, 2> columns{0, 2};
        const std::array<Real, 2> coefficients{-1, 1};
        const std::array<Real, 1> low{1};
        const std::array<Real, 1> high{kInf};
        lp.addRows(starts, columns, coefficients, low, high);
        model.matrix.addRow(columns, coefficients);
        model.row_lower.push_back(low[0]);
        model.row_upper.push_back(high[0]);
        ++model.num_rows;
    };
    if (stronger_incumbent) {
        installPricingRow();
    }
    REQUIRE(lp.solve().status == Status::Optimal);
    RootCutoffFeedback feedback(model, stronger_incumbent ? 2.25 : 1.25);
    PropagationScheduler history;
    MipCutStats stats;
    DomainPropagationWork work;
    Int budget = 12;
    REQUIRE(feedback.updateScheduled(model, lp, budget, work, history, {}, stats).status ==
            Status::Optimal);
    CHECK(stats.root_cutoff_feedbacks == 0);
    if (stronger_incumbent) {
        feedback.strengthenCutoff(1.25);
    } else {
        installPricingRow();
        REQUIRE(lp.solve().status == Status::Optimal);
    }
    REQUIRE(lp.getObjective() == 1.0);

    const auto original_point = lp.getPrimalValues();
    const auto original_basis = lp.getBasis();
    constexpr auto rc = static_cast<std::size_t>(PropagationProducer::ReducedCost);
    constexpr auto objective = static_cast<std::size_t>(PropagationProducer::Objective);
    constexpr auto budget_exit = static_cast<std::size_t>(PropagationExit::Budget);
    budget = 1;
    REQUIRE(feedback.updateScheduled(model, lp, budget, work, history, {}, stats).status ==
            Status::Optimal);
    CHECK(budget == 0);
    CHECK(stats.root_propagation_schedule.producers[rc].calls == 1);
    CHECK(stats.root_propagation_schedule.producers[rc].exits[budget_exit] == 1);
    CHECK(stats.root_cutoff_feedbacks == 0);
    CHECK(feedback.lower().empty());

    budget = 12;
    REQUIRE(feedback.updateScheduled(model, lp, budget, work, history, {}, stats).status ==
            Status::Optimal);
    REQUIRE(feedback.upper().size() == 4);
    CHECK(feedback.upper()[0] == 0.0);
    CHECK(feedback.upper()[1] == 0.0);
    CHECK(stats.root_propagation_schedule.producers[rc].calls == 2);
    CHECK(stats.root_propagation_schedule.producers[rc].endpoints == 1);
    // The woken rows provide finite contributions to the objective activity.
    CHECK(stats.root_propagation_schedule.producers[objective].endpoints == 2);
    CHECK(stats.root_cutoff_row_tightenings >= 1);
    CHECK(stats.root_cutoff_feedbacks == 1);
    CHECK(lp.getBasis() == original_basis);
    CHECK(lp.getPrimalValues() == original_point);
    CHECK(lp.getObjective() == 1.0);
    Real lower, upper;
    lp.getColBounds(0, lower, upper);
    CHECK(upper == 1.0);
}

TEST_CASE("Scheduled root cutoff wakes scoped implications and cliques", "[mip][cuts][domain]") {
    // Cutoff x < 0.5 fixes x=0. The original rows imply y=1, then z=0.
    LpProblem model;
    model.num_cols = 3;
    model.num_rows = 2;
    model.obj = {1, 0, 0};
    model.col_lower.assign(3, 0);
    model.col_upper.assign(3, 1);
    model.col_type.assign(3, VarType::Binary);
    model.row_lower = {1, -kInf};
    model.row_upper = {kInf, 1};
    model.matrix = SparseMatrix(2, 3, {{0, 0, 1}, {0, 1, 1}, {1, 1, 1}, {1, 2, 1}});
    ImplicationGraph implications;
    implications.init(3, {0, 1, 2});
    REQUIRE(implications.addImplication(0, false, 1, true));
    ConflictGraph graph;
    graph.build(model);
    CliqueTable cliques;
    cliques.build(model, graph);
    REQUIRE(cliques.numCliques() > 0);

    DualSimplexSolver lp;
    lp.setVerbose(false);
    lp.load(model);
    REQUIRE(lp.solve().status == Status::Optimal);
    RootCutoffFeedback feedback(model, 0.5);
    PropagationScheduler history;
    MipCutStats stats;
    DomainPropagationWork work;
    Int budget = 20;
    REQUIRE(feedback
                .updateScheduled(model, lp, budget, work, history, {}, stats, kInf,
                                 {.first_new_row = model.num_rows,
                                  .cliques = &cliques,
                                  .implications = &implications})
                .status == Status::Optimal);
    CHECK(work.implications.tightenings == 1);
    CHECK(work.cliques.tightenings == 1);
    REQUIRE(feedback.upper().size() == 3);
    CHECK(feedback.upper()[2] == 0.0);
    Real lower;
    Real upper;
    lp.getColBounds(2, lower, upper);
    CHECK(upper == 1.0);
}

TEST_CASE("Root cutoff feedback: a failed re-solve preserves the prior domain and LP",
          "[mip][cuts][domain]") {
    const bool scheduled = GENERATE(false, true);
    auto model = twoIntegerColumns();
    model.obj = {1.0, 0.4};
    model.num_rows = 1;
    model.row_lower = {0.5};
    model.row_upper = {kInf};
    model.row_names = {"original"};
    model.matrix = SparseMatrix(1, 2, {{0, 0, 1.0}, {0, 1, 0.1}});

    DualSimplexSolver lp;
    lp.load(model);
    REQUIRE(lp.solve().status == Status::Optimal);

    Int row_budget = scheduled ? 2 : 0;
    MipCutStats stats;
    // The integer incumbent x = y = 1 attains the cutoff in both rounds.
    RootCutoffFeedback feedback(model, 1.4);
    PropagationScheduler history;
    const auto update = [&] {
        if (scheduled) {
            DomainPropagationWork work;
            return feedback.updateScheduled(model, lp, row_budget, work, history, {}, stats);
        }
        return feedback.update(model, lp, row_budget, {}, stats);
    };
    REQUIRE(update().status == Status::Optimal);
    REQUIRE(stats.root_cutoff_feedbacks == 1);
    const auto prior_lower = values(feedback.lower());
    const auto prior_upper = values(feedback.upper());
    const auto prior_primals = values(feedback.primals());
    const auto prior_basis = lp.getBasis();
    const auto prior_global_point = lp.getPrimalValues();

    // The second LP prefers x = 1.05, beyond the retained cutoff bound x <= 1.
    // Repair to x = 1 and y >= 0.5 needs a pivot in the scoped LP.
    const auto installRound = [&] {
        const std::array<Index, 1> starts{0};
        const std::array<Index, 2> columns{0, 1};
        const std::array<Real, 2> coefficients{1.0, 0.1};
        const std::array<Real, 1> lower{1.05};
        const std::array<Real, 1> upper{kInf};
        lp.addRows(starts, columns, coefficients, lower, upper);
        model.matrix.addRow(columns, coefficients);
        model.row_lower.push_back(lower[0]);
        model.row_upper.push_back(upper[0]);
        model.row_names.emplace_back();
        ++model.num_rows;
        // The new row also implies x >= 0.75, rounded up for integer x.
        lp.setColBounds(0, 1.0, 3.0);
    };
    installRound();
    REQUIRE(lp.solve().status == Status::Optimal);
    const auto global_primals = lp.getPrimalValues();
    REQUIRE_THAT(global_primals[0], WithinAbs(1.05, 1e-8));
    const Int normal_limit = lp.getIterationLimit();
    lp.setIterationLimit(0);
    if (scheduled) {
        // A failed optional cutoff trial keeps the already optimal primary LP.
        const std::array trial_bounds{RootCutBound{0, 1.0, 1.0}};
        const auto primary_basis = lp.getBasis();
        CHECK(tryScopedCutBounds(lp, trial_bounds).status == Status::IterLimit);
        CHECK(lp.getStatus() == Status::Optimal);
        CHECK(lp.getPrimalValues() == global_primals);
        CHECK(lp.getBasis() == primary_basis);
        CHECK(lp.numRows() == 2);
        Real col_lower;
        Real col_upper;
        lp.getColBounds(0, col_lower, col_upper);
        CHECK(col_lower == 1.0);
        CHECK(col_upper == 3.0);
    }
    row_budget = scheduled ? 1 : 0;
    CHECK(update().status == Status::IterLimit);
    CHECK(stats.root_cutoff_resolves_backed_out == 1);
    CHECK(stats.root_cutoff_feedbacks == 1);
    CHECK(values(feedback.lower()) == prior_lower);
    CHECK(values(feedback.upper()) == prior_upper);
    CHECK(values(feedback.primals()) == prior_primals);
    CHECK(lp.getStatus() == Status::Optimal);
    CHECK(lp.getPrimalValues() == global_primals);

    lp.setIterationLimit(normal_limit);
    const std::array staged{RootCutBound{0, 1.0, 3.0}};
    REQUIRE(backOutRootCutRound(lp, 1, prior_basis, &model, staged).status == Status::Optimal);
    CHECK(lp.numRows() == 1);
    CHECK(model.num_rows == 1);
    CHECK(model.row_lower == std::vector<Real>{0.5});
    CHECK(model.row_upper == std::vector<Real>{kInf});
    CHECK(model.row_names == std::vector<std::string>{"original"});
    CHECK(lp.getBasis() == prior_basis);
    CHECK(lp.getPrimalValues() == prior_global_point);
    Real lower, upper;
    lp.getColBounds(0, lower, upper);
    CHECK(lower == 0.0);
    CHECK(upper == 3.0);

    installRound();
    REQUIRE(lp.solve().status == Status::Optimal);
    row_budget = scheduled ? 1 : 0;
    REQUIRE(update().status == Status::Optimal);
    REQUIRE(stats.root_cutoff_feedbacks == 2);
    CHECK_THAT(feedback.primals()[0], WithinAbs(1.0, 1e-8));
    CHECK_THAT(feedback.primals()[1], WithinAbs(scheduled ? 1.0 : 0.5, 1e-8));
}

TEST_CASE("Root cutoff feedback: spent LP work leaves a successful cut batch installed",
          "[mip][cuts][domain]") {
    auto model = twoIntegerColumns();
    model.obj = {1.0, 0.4};
    model.num_rows = 1;
    model.row_lower = {0.5};
    model.row_upper = {kInf};
    model.matrix = SparseMatrix(1, 2, {{0, 0, 1.0}, {0, 1, 0.1}});
    DualSimplexSolver lp;
    lp.load(model);
    REQUIRE(lp.solve().status == Status::Optimal);

    // Rounding the original row's integer consequences gives x >= 1.
    // The incumbent x = y = 1 is still valid and has objective 1.4.
    const std::array<Index, 1> starts{0};
    const std::array<Index, 1> column{0};
    const std::array<Real, 1> coefficient{1.0};
    const std::array<Real, 1> lower{1.0};
    const std::array<Real, 1> upper{kInf};
    lp.addRows(starts, column, coefficient, lower, upper);
    model.matrix.addRow(column, coefficient);
    model.row_lower.push_back(lower[0]);
    model.row_upper.push_back(upper[0]);
    ++model.num_rows;
    const auto cut_solve = lp.solve();
    REQUIRE(cut_solve.status == Status::Optimal);
    REQUIRE(cut_solve.work_units > 0.0);
    const double work_allowance = cut_solve.work_units;

    Int row_budget = model.num_rows;
    MipCutStats stats;
    RootCutoffFeedback feedback(model, 1.4);
    REQUIRE(feedback.update(model, lp, row_budget, {}, stats, work_allowance - cut_solve.work_units)
                .status == Status::Optimal);
    CHECK(stats.root_cutoff_feedbacks == 0);
    CHECK(stats.root_cutoff_reduced_cost_tightenings == 0);
    CHECK(stats.root_cutoff_objective_tightenings == 0);
    CHECK(row_budget == model.num_rows);
    CHECK(lp.numRows() == model.num_rows);
    CHECK_THAT(lp.getObjective(), WithinAbs(1.0, 1e-8));

    REQUIRE(feedback.update(model, lp, row_budget, {}, stats, 1.0).status == Status::Optimal);
    CHECK(stats.root_cutoff_feedbacks == 1);
}
