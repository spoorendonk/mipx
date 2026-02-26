#include <catch2/catch_test_macros.hpp>

#include "mipx/cut_manager.h"

using namespace mipx;

TEST_CASE("CutManager AUTO demotes low-ROI families", "[cuts][policy]") {
    CutManager manager;
    manager.setMode(CutEffortMode::Auto);
    manager.setBaseLimits(20, 20);
    manager.resetNodeState(true, 0);

    for (Int round = 0; round < 4; ++round) {
        auto policy = manager.beginRound(round, true, 0, 0.0, 0.0);
        REQUIRE(policy.run);

        CutSeparationStats stats;
        auto& cover = stats.at(CutFamily::Cover);
        cover.attempted = 8;
        cover.generated = 4;
        cover.accepted = 4;
        cover.efficacy_sum = 0.04;
        cover.time_seconds = 0.5;

        std::array<Int, static_cast<std::size_t>(CutFamily::Count)> selected{};
        selected.fill(0);
        selected[static_cast<std::size_t>(CutFamily::Cover)] = 4;

        manager.recordRound(stats, selected, 1e-6, 0.1, 0.5, 1e5, true, 0);
    }

    CHECK_FALSE(manager.familyEnabled(CutFamily::Cover));
}

TEST_CASE("CutManager AUTO promotes throttled families on positive ROI", "[cuts][policy]") {
    CutManager manager;
    manager.setMode(CutEffortMode::Auto);
    manager.setBaseLimits(20, 20);
    manager.resetNodeState(true, 0);
    manager.setFamilyEnabled(CutFamily::Clique, false);

    for (Int round = 0; round < 3; ++round) {
        CutSeparationStats stats;
        auto& clique = stats.at(CutFamily::Clique);
        clique.attempted = 4;
        clique.generated = 3;
        clique.accepted = 3;
        clique.efficacy_sum = 3.0;
        clique.time_seconds = 1e-4;

        std::array<Int, static_cast<std::size_t>(CutFamily::Count)> selected{};
        selected.fill(0);
        selected[static_cast<std::size_t>(CutFamily::Clique)] = 3;

        manager.recordRound(stats, selected, 1.0, 0.9, 1e-4, 1.0, true, 0);
    }

    CHECK(manager.familyEnabled(CutFamily::Clique));
}

TEST_CASE("CutManager policy decisions are deterministic", "[cuts][policy]") {
    auto runScenario = []() {
        CutManager manager;
        manager.setMode(CutEffortMode::Auto);
        manager.setBaseLimits(12, 18);
        manager.setBudgets(1000.0, 300.0, 5000.0);
        manager.resetNodeState(true, 0);

        for (Int round = 0; round < 5; ++round) {
            auto policy = manager.beginRound(round, true, 0, 10.0 * round, 20.0 * round);
            if (!policy.run) break;

            CutSeparationStats stats;
            auto& gomory = stats.at(CutFamily::Gomory);
            gomory.attempted = 3;
            gomory.generated = 2;
            gomory.accepted = 2;
            gomory.efficacy_sum = 0.4;
            gomory.time_seconds = 0.001;

            std::array<Int, static_cast<std::size_t>(CutFamily::Count)> selected{};
            selected.fill(0);
            selected[static_cast<std::size_t>(CutFamily::Gomory)] = 2;

            manager.recordRound(stats, selected, 0.01, 0.7, 0.001, 5.0, true, 0);
        }
        return manager.summarizeState();
    };

    const auto a = runScenario();
    const auto b = runScenario();
    CHECK(a == b);
}
