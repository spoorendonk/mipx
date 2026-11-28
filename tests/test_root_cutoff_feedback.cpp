#include "../src/mip/root_cutoff_feedback.h"

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

    Int row_budget = model.num_rows;
    MipCutStats stats;
    RootCutoffFeedback feedback(model, cutoff);
    REQUIRE(feedback.update(model, lp, row_budget, {}, stats).status == Status::Optimal);
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

TEST_CASE("Root cutoff feedback: a failed re-solve preserves the prior domain and LP",
          "[mip][cuts][domain]") {
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

    Int row_budget = 0;
    MipCutStats stats;
    // The integer incumbent x = y = 1 attains the cutoff in both rounds.
    RootCutoffFeedback feedback(model, 1.4);
    REQUIRE(feedback.update(model, lp, row_budget, {}, stats).status == Status::Optimal);
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
    CHECK(feedback.update(model, lp, row_budget, {}, stats).status == Status::IterLimit);
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
    REQUIRE(feedback.update(model, lp, row_budget, {}, stats).status == Status::Optimal);
    REQUIRE(stats.root_cutoff_feedbacks == 2);
    CHECK_THAT(feedback.primals()[0], WithinAbs(1.0, 1e-8));
    CHECK_THAT(feedback.primals()[1], WithinAbs(0.5, 1e-8));
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
