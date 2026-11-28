#include "mipx/cut_manager.h"
#include "mipx/dual_simplex.h"
#include "mipx/mip_solver.h"
#include "mipx/separators.h"

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using namespace mipx;
using Catch::Matchers::WithinAbs;

namespace {

// MIPLIB exp-1-500-5-5, balance c1 and link c51, with columns ordered as
// x1, x51, x101, x150. INTORG with bounds [0,1] supplies the activation.
LpProblem reducedWitness(Real sign = 1.0) {
    LpProblem problem;
    problem.num_cols = 4;
    problem.num_rows = 2;
    problem.obj = {0.0, 1.0, 1.0, 1.0};
    problem.col_lower = {0.0, 0.0, 0.0, 0.0};
    problem.col_upper = {kInf, 1.0, kInf, kInf};
    problem.col_type = {VarType::Continuous, VarType::Integer, VarType::Continuous,
                        VarType::Continuous};
    problem.row_lower = {sign * 16.0, sign > 0.0 ? -kInf : 0.0};
    problem.row_upper = {sign * 16.0, sign > 0.0 ? 0.0 : kInf};
    problem.matrix = SparseMatrix(
        2, 4, {{0, 0, sign}, {0, 2, -sign}, {0, 3, sign}, {1, 0, sign}, {1, 1, -753.0 * sign}});
    return problem;
}

CutFamilyConfig flowCoverConfig(bool enabled) {
    MipSolver solver;
    for (Int family = 1; family < static_cast<Int>(CutFamily::Count); ++family) {
        solver.setCutFamilyEnabled(static_cast<CutFamily>(family), false);
    }
    solver.setRootFlowCoverEnabled(enabled);
    return solver.getCutFamilyConfig();
}

}  // namespace

TEST_CASE("Signed flow cover cuts the fractional public witness and keeps both activations",
          "[cuts][flowcover]") {
    for (const Real sign : {1.0, -1.0}) {
        CAPTURE(sign);
        const auto problem = reducedWitness(sign);
        DualSimplexSolver lp;
        // These four primal values come from the uncut public root LP.
        const std::array<Real, 4> point{33.0, 33.0 / 753.0, 17.0, 0.0};

        SeparatorManager separator;
        separator.setConfig(flowCoverConfig(true));
        CutPool pool;
        CutSeparationStats stats;
        REQUIRE(separator.separate(lp, problem, point, pool, stats) == 1);
        const Cut& cut = pool[0];
        CHECK(cut.family == CutFamily::FlowCover);
        CHECK_FALSE(cut.local);
        CHECK(cut.indices == std::vector<Index>{0, 1, 2});
        CHECK(cut.values == std::vector<Real>{1.0, -16.0, -1.0});
        CHECK(cut.upper == 0.0);
        CHECK_THAT(cutViolation(cut, point), WithinAbs(15.2988047809, 1e-9));

        // Maximize the cut's left side over the two source rows and sign
        // bounds. A zero optimum proves it for all points at this activation.
        for (const Real activation : {0.0, 1.0}) {
            CAPTURE(activation);
            auto fixed = problem;
            fixed.col_lower[1] = fixed.col_upper[1] = activation;
            fixed.sense = Sense::Maximize;
            fixed.obj = {1.0, -16.0, -1.0, 0.0};
            lp.load(fixed);
            const auto result = lp.solve();
            REQUIRE(result.status == Status::Optimal);
            CHECK_THAT(result.objective, WithinAbs(0.0, 1e-8));
        }
    }
}

TEST_CASE("Signed flow cover needs global sign, activation and explicit row proofs",
          "[cuts][flowcover]") {
    auto problem = reducedWitness();
    SeparatorManager separator;
    std::array<Real, 4> point{16.0, 16.0 / 753.0, 0.0, 0.0};
    SECTION("The dropped positive term can be negative") {
        problem.col_lower[3] = -1.0;
        point = {17.0, 1.0, 0.0, -1.0};
    }
    SECTION("A retained negative term has an unproved sign") {
        problem.col_lower[2] = -1.0;
        point = {0.0, 0.0, -1.0, 15.0};
    }
    SECTION("The activation is continuous even though its bounds are zero and one") {
        problem.col_type[1] = VarType::Continuous;
    }
    SECTION("The integer activation has wider bounds") {
        problem.col_upper[1] = 2.0;
    }
    SECTION("The link allows positive flow at zero activation") {
        problem.row_upper[1] = 1.0;
        point = {1.0, 0.0, 0.0, 15.0};
    }
    SECTION("The link is outside the global row prefix") {
        separator.setGlobalRowCount(1);
    }
    SECTION("The link inside the prefix is local") {
        separator.setLocalRows({1});
    }
    SECTION("The balance inside the prefix is local") {
        separator.setLocalRows({0});
    }

    SECTION("The linked capacity does not exceed the balance bound") {
        // A synthetic point tests the capacity gate: with U=16 the cut is
        // already implied by the link, so no feasible LP point violates it.
        problem.matrix = SparseMatrix(
            2, 4, {{0, 0, 1.0}, {0, 2, -1.0}, {0, 3, 1.0}, {1, 0, 1.0}, {1, 1, -16.0}});
    }

    separator.setConfig(flowCoverConfig(true));
    DualSimplexSolver lp;  // Row-based flow covers need no simplex basis.
    CutPool pool;
    CutSeparationStats stats;
    CHECK(separator.separate(lp, problem, point, pool, stats) == 0);
    CHECK(stats.at(CutFamily::FlowCover).generated == 0);
}

TEST_CASE("MipSolver installs signed flow covers only after opt-in", "[cuts][flowcover]") {
    CHECK_FALSE(MipSolver().getCutFamilyConfig().flow_cover);
    for (const bool enabled : {false, true}) {
        CAPTURE(enabled);
        MipSolver solver;
        solver.setVerbose(false);
        solver.setPresolve(false);
        solver.setCutFamilyConfig(flowCoverConfig(false));
        solver.setCutFamilyEnabled(CutFamily::FlowCover, enabled);
        solver.load(reducedWitness());
        const auto result = solver.solve();
        REQUIRE(result.status == Status::Optimal);
        CHECK_THAT(result.objective, WithinAbs(1.0, 1e-8));
        const auto& stats = solver.getRootCutFamilyStats().at(CutFamily::FlowCover);
        CHECK(stats.generated == (enabled ? 1 : 0));
        CHECK(stats.accepted == (enabled ? 1 : 0));
        CHECK(stats.installed == (enabled ? 1 : 0));
    }
}

TEST_CASE("Signed flow cover policy is root-only and has a family quota", "[cuts][flowcover]") {
    CutManager manager;
    manager.setBaseLimits(6, 48);
    const auto flow_cover = static_cast<std::size_t>(CutFamily::FlowCover);
    for (Int family = 1; family < static_cast<Int>(CutFamily::Count); ++family) {
        manager.setFamilyEnabled(static_cast<CutFamily>(family),
                                 static_cast<std::size_t>(family) == flow_cover);
    }

    const auto root = manager.beginRound(0, true, 0, 0.0, 0.0);
    REQUIRE(root.family_enabled[flow_cover]);
    CHECK(root.per_family_cap[flow_cover] == 8);
    CHECK_FALSE(manager.beginRound(0, false, 1, 0.0, 0.0).family_enabled[flow_cover]);
    manager.recordRound({}, {}, 0.0, 1.0, 0.0, 0.0, true, 0);
    CHECK_FALSE(manager.familyEnabled(CutFamily::Gomory));
    CHECK(manager.summarizeState().contains("flowcover=on"));
}
