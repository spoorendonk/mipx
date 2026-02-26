#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <array>
#include <cmath>
#include <filesystem>
#include <format>
#include <iostream>
#include <span>
#include <string>
#include <vector>

#include "mipx/dual_simplex.h"
#include "mipx/io.h"
#include "mipx/lp_problem.h"
#include "mipx/mip_solver.h"

using namespace mipx;
using Catch::Matchers::WithinAbs;

namespace fs = std::filesystem;

static std::string testDataDir() { return std::string(TEST_DATA_DIR); }

// ---------------------------------------------------------------------------
// Netlib instance parsing tests — verify we can read all downloaded instances.
// ---------------------------------------------------------------------------

static std::vector<std::string> findInstances(const std::string& dir) {
    std::vector<std::string> paths;
    if (!fs::exists(dir)) return paths;
    for (const auto& entry : fs::directory_iterator(dir)) {
        auto p = entry.path();
        if (p.extension() == ".gz" && p.stem().extension() == ".mps") {
            paths.push_back(p.string());
        } else if (p.extension() == ".mps") {
            paths.push_back(p.string());
        }
    }
    std::ranges::sort(paths);
    return paths;
}

static std::string instanceName(const std::string& path) {
    auto stem = fs::path(path).stem();  // e.g. "afiro.mps" or "afiro"
    if (stem.extension() == ".mps") {
        return stem.stem().string();  // strip .mps from .mps.gz
    }
    return stem.string();
}

static const SoluEntry* findSoluEntry(std::span<const SoluEntry> entries,
                                      const std::string& name) {
    for (const auto& e : entries) {
        if (e.name == name) return &e;
    }
    return nullptr;
}

static Real objectiveTol(Real reference) {
    return std::max<Real>(1e-6, std::abs(reference) * 1e-7);
}

// ---------------------------------------------------------------------------
// Netlib: parse all downloaded instances without error.
// ---------------------------------------------------------------------------
TEST_CASE("Netlib: parse all instances", "[benchmark][netlib]") {
    std::string netlib_dir = testDataDir() + "/netlib";
    auto instances = findInstances(netlib_dir);

    if (instances.empty()) {
        SKIP("No Netlib instances found. Run tests/data/download_netlib.sh");
    }

    // Load the .solu file for validation.
    std::string solu_file = netlib_dir + "/netlib.solu";
    std::vector<SoluEntry> solu;
    if (fs::exists(solu_file)) {
        solu = readSolu(solu_file);
    }

    auto findOpt = [&](const std::string& name) -> const SoluEntry* {
        for (const auto& e : solu) {
            if (e.name == name) return &e;
        }
        return nullptr;
    };

    Index parsed = 0;
    Index failed = 0;

    std::cout << std::format("\n{:<16} {:>6} {:>6} {:>8}  {}\n", "Instance",
                             "Rows", "Cols", "NNZ", "Status");
    std::cout << std::string(60, '-') << "\n";

    for (const auto& path : instances) {
        std::string name = instanceName(path);
        try {
            auto prob = readMps(path);
            parsed++;

            std::string status = "ok";
            auto* entry = findOpt(name);
            if (entry) {
                status = std::format("opt={:.6e}", entry->value);
            }

            std::cout << std::format("{:<16} {:>6} {:>6} {:>8}  {}\n", name,
                                     prob.num_rows, prob.num_cols,
                                     prob.matrix.numNonzeros(), status);
        } catch (const std::exception& e) {
            failed++;
            std::cout << std::format("{:<16}  FAILED: {}\n", name, e.what());
        }
    }

    std::cout << std::format("\nParsed: {}/{}, Failed: {}\n", parsed,
                             instances.size(), failed);

    CHECK(failed == 0);
}

// ---------------------------------------------------------------------------
// MIPLIB: parse all downloaded instances without error.
// ---------------------------------------------------------------------------
TEST_CASE("MIPLIB: parse all instances", "[benchmark][miplib]") {
    std::string miplib_dir = testDataDir() + "/miplib";
    auto instances = findInstances(miplib_dir);

    if (instances.empty()) {
        SKIP("No MIPLIB instances found. Run tests/data/download_miplib.sh");
    }

    // Load the .solu file for validation.
    std::string solu_file = miplib_dir + "/miplib.solu";
    std::vector<SoluEntry> solu;
    if (fs::exists(solu_file)) {
        solu = readSolu(solu_file);
    }

    auto findOpt = [&](const std::string& name) -> const SoluEntry* {
        for (const auto& e : solu) {
            if (e.name == name) return &e;
        }
        return nullptr;
    };

    Index parsed = 0;
    Index failed = 0;
    Index with_integers = 0;

    std::cout << std::format("\n{:<16} {:>6} {:>6} {:>8} {:>5}  {}\n",
                             "Instance", "Rows", "Cols", "NNZ", "IntV",
                             "Status");
    std::cout << std::string(65, '-') << "\n";

    for (const auto& path : instances) {
        std::string name = instanceName(path);
        try {
            auto prob = readMps(path);
            parsed++;

            Index int_vars = 0;
            for (Index j = 0; j < prob.num_cols; ++j) {
                if (prob.col_type[j] != VarType::Continuous) int_vars++;
            }
            if (int_vars > 0) with_integers++;

            std::string status = "ok";
            auto* entry = findOpt(name);
            if (entry) {
                if (entry->is_infeasible) {
                    status = "infeasible";
                } else {
                    status = std::format("opt={:.6e}", entry->value);
                }
            }

            std::cout << std::format("{:<16} {:>6} {:>6} {:>8} {:>5}  {}\n",
                                     name, prob.num_rows, prob.num_cols,
                                     prob.matrix.numNonzeros(), int_vars,
                                     status);
        } catch (const std::exception& e) {
            failed++;
            std::cout << std::format("{:<16}  FAILED: {}\n", name, e.what());
        }
    }

    std::cout << std::format("\nParsed: {}/{}, Failed: {}, With integers: {}\n",
                             parsed, instances.size(), failed, with_integers);

    CHECK(failed == 0);
}

// ---------------------------------------------------------------------------
// Netlib: validate dimensions of known small instances.
// ---------------------------------------------------------------------------
TEST_CASE("Netlib: afiro dimensions", "[benchmark][netlib]") {
    std::string path = testDataDir() + "/netlib/afiro.mps.gz";
    if (!fs::exists(path)) {
        SKIP("afiro not downloaded. Run tests/data/download_netlib.sh --small");
    }

    auto prob = readMps(path);
    CHECK(prob.num_rows == 27);
    CHECK(prob.num_cols == 32);
    CHECK(prob.matrix.numNonzeros() == 83);
    CHECK(prob.sense == Sense::Minimize);
    CHECK_FALSE(prob.hasIntegers());
}

TEST_CASE("Netlib: sc50a dimensions", "[benchmark][netlib]") {
    std::string path = testDataDir() + "/netlib/sc50a.mps.gz";
    if (!fs::exists(path)) {
        SKIP("sc50a not downloaded. Run tests/data/download_netlib.sh --small");
    }

    auto prob = readMps(path);
    CHECK(prob.num_rows == 50);
    CHECK(prob.num_cols == 48);
    CHECK(prob.sense == Sense::Minimize);
}

TEST_CASE("Netlib: blend dimensions", "[benchmark][netlib]") {
    std::string path = testDataDir() + "/netlib/blend.mps.gz";
    if (!fs::exists(path)) {
        SKIP("blend not downloaded. Run tests/data/download_netlib.sh --small");
    }

    auto prob = readMps(path);
    CHECK(prob.num_rows == 74);
    CHECK(prob.num_cols == 83);
    CHECK(prob.sense == Sense::Minimize);
}

// ---------------------------------------------------------------------------
// MIPLIB: validate gt2 has integer variables.
// ---------------------------------------------------------------------------
TEST_CASE("MIPLIB: gt2 has integers", "[benchmark][miplib]") {
    std::string path = testDataDir() + "/miplib/gt2.mps.gz";
    if (!fs::exists(path)) {
        SKIP("gt2 not downloaded. Run tests/data/download_miplib.sh --small");
    }

    auto prob = readMps(path);
    CHECK(prob.hasIntegers());
    CHECK(prob.num_rows > 0);
    CHECK(prob.num_cols > 0);
}

// ---------------------------------------------------------------------------
// MPS round-trip: write and re-read a Netlib instance.
// ---------------------------------------------------------------------------
TEST_CASE("Netlib: MPS round-trip on afiro", "[benchmark][netlib]") {
    std::string path = testDataDir() + "/netlib/afiro.mps.gz";
    if (!fs::exists(path)) {
        SKIP("afiro not downloaded. Run tests/data/download_netlib.sh --small");
    }

    auto orig = readMps(path);

    std::string tmp = testDataDir() + "/afiro_roundtrip.mps";
    writeMps(tmp, orig);

    auto reread = readMps(tmp);

    CHECK(reread.num_cols == orig.num_cols);
    CHECK(reread.num_rows == orig.num_rows);
    CHECK(reread.matrix.numNonzeros() == orig.matrix.numNonzeros());

    for (Index j = 0; j < orig.num_cols; ++j) {
        CHECK_THAT(reread.obj[j], WithinAbs(orig.obj[j], 1e-12));
    }

    fs::remove(tmp);
}

TEST_CASE("Netlib: LP solve objectives match .solu on curated set",
          "[benchmark][netlib][solve]") {
    const std::string netlib_dir = testDataDir() + "/netlib";
    const std::string solu_file = netlib_dir + "/netlib.solu";
    if (!fs::exists(solu_file)) {
        SKIP("netlib.solu not found");
    }

    const auto solu_entries = readSolu(solu_file);
    const std::array<std::string, 3> instances = {"afiro", "blend", "sc50a"};

    bool ran_any = false;
    for (const auto& name : instances) {
        const std::string path = netlib_dir + "/" + name + ".mps.gz";
        if (!fs::exists(path)) continue;

        const auto* entry = findSoluEntry(solu_entries, name);
        if (entry == nullptr || entry->is_infeasible) continue;

        const auto problem = readMps(path);
        REQUIRE_FALSE(problem.hasIntegers());

        DualSimplexSolver solver;
        solver.load(problem);
        const auto result = solver.solve();
        REQUIRE(result.status == Status::Optimal);
        CHECK_THAT(result.objective, WithinAbs(entry->value, objectiveTol(entry->value)));
        CHECK(result.work_units > 0.0);
        ran_any = true;
    }

    if (!ran_any) {
        SKIP("No curated Netlib LP instances available locally for objective check");
    }
}

TEST_CASE("MIPLIB: p0201 objective matches .solu", "[benchmark][miplib][solve]") {
    const std::string miplib_dir = testDataDir() + "/miplib";
    const std::string path = miplib_dir + "/p0201.mps.gz";
    if (!fs::exists(path)) {
        SKIP("p0201 not downloaded. Run tests/data/download_miplib.sh --small");
    }

    const std::string solu_file = miplib_dir + "/miplib.solu";
    if (!fs::exists(solu_file)) {
        SKIP("miplib.solu not found");
    }

    const auto solu_entries = readSolu(solu_file);
    const auto* entry = findSoluEntry(solu_entries, "p0201");
    if (entry == nullptr || entry->is_infeasible) {
        SKIP("No finite p0201 objective in miplib.solu");
    }

    auto problem = readMps(path);
    REQUIRE(problem.hasIntegers());

    MipSolver solver;
    solver.setVerbose(false);
    solver.setTimeLimit(60.0);
    solver.setNodeLimit(300000);
    solver.setGapTolerance(1e-6);
    solver.setHeuristicMode(HeuristicRuntimeMode::Deterministic);
    solver.setHeuristicSeed(1);
    solver.setSearchProfile(SearchProfile::Stable);
    solver.load(problem);
    const auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(entry->value, objectiveTol(entry->value)));
    CHECK(result.work_units > 0.0);
}

TEST_CASE("MIPLIB: p0201 objective matches .solu across root LP policies",
          "[benchmark][miplib][solve][rootpolicy]") {
    const std::string miplib_dir = testDataDir() + "/miplib";
    const std::string path = miplib_dir + "/p0201.mps.gz";
    if (!fs::exists(path)) {
        SKIP("p0201 not downloaded. Run tests/data/download_miplib.sh --small");
    }

    const std::string solu_file = miplib_dir + "/miplib.solu";
    if (!fs::exists(solu_file)) {
        SKIP("miplib.solu not found");
    }

    const auto solu_entries = readSolu(solu_file);
    const auto* entry = findSoluEntry(solu_entries, "p0201");
    if (entry == nullptr || entry->is_infeasible) {
        SKIP("No finite p0201 objective in miplib.solu");
    }

    auto problem = readMps(path);
    REQUIRE(problem.hasIntegers());

    struct RootPolicyCase {
        const char* name;
        RootLpPolicy policy;
    };
    const std::array<RootPolicyCase, 4> root_policies = {{
        {"dual", RootLpPolicy::DualDefault},
        {"barrier", RootLpPolicy::BarrierRoot},
        {"pdlp", RootLpPolicy::PdlpRoot},
        {"concurrent", RootLpPolicy::ConcurrentRootExperimental},
    }};

    for (const auto& root_case : root_policies) {
        MipSolver solver;
        solver.setVerbose(false);
        solver.setTimeLimit(60.0);
        solver.setNodeLimit(300000);
        solver.setGapTolerance(1e-6);
        solver.setHeuristicMode(HeuristicRuntimeMode::Deterministic);
        solver.setHeuristicSeed(11);
        solver.setSearchProfile(SearchProfile::Stable);
        solver.setRootLpPolicy(root_case.policy);
        solver.setBarrierUseGpu(false);
        solver.setPdlpUseGpu(false);
        solver.load(problem);

        INFO(std::format("root_policy={}", root_case.name));
        const auto result = solver.solve();
        REQUIRE(result.status == Status::Optimal);
        CHECK_THAT(result.objective, WithinAbs(entry->value, objectiveTol(entry->value)));
        CHECK(result.work_units > 0.0);

        const auto& lp_stats = solver.getLpStats();
        if (root_case.policy == RootLpPolicy::ConcurrentRootExperimental) {
            CHECK(lp_stats.root_race_runs == 1);
            CHECK(lp_stats.root_race_candidates == 3);
        }
    }
}

TEST_CASE("MIPLIB: p0201 objective matches .solu with pre-root heuristics",
          "[benchmark][miplib][solve][preroot]") {
    const std::string miplib_dir = testDataDir() + "/miplib";
    const std::string path = miplib_dir + "/p0201.mps.gz";
    if (!fs::exists(path)) {
        SKIP("p0201 not downloaded. Run tests/data/download_miplib.sh --small");
    }

    const std::string solu_file = miplib_dir + "/miplib.solu";
    if (!fs::exists(solu_file)) {
        SKIP("miplib.solu not found");
    }

    const auto solu_entries = readSolu(solu_file);
    const auto* entry = findSoluEntry(solu_entries, "p0201");
    if (entry == nullptr || entry->is_infeasible) {
        SKIP("No finite p0201 objective in miplib.solu");
    }

    auto problem = readMps(path);
    REQUIRE(problem.hasIntegers());

    MipSolver solver;
    solver.setVerbose(false);
    solver.setTimeLimit(60.0);
    solver.setNodeLimit(300000);
    solver.setGapTolerance(1e-6);
    solver.setHeuristicMode(HeuristicRuntimeMode::Deterministic);
    solver.setHeuristicSeed(7);
    solver.setSearchProfile(SearchProfile::Stable);
    solver.setPreRootLpFreeEnabled(true);
    solver.setPreRootLpLightEnabled(true);
    solver.setPreRootPortfolioEnabled(true);
    solver.setPreRootLpFreeMaxRounds(8);
    solver.setPreRootLpFreeWorkBudget(2.0e5);
    solver.setPreRootLpFreeEarlyStop(false);
    solver.load(problem);
    const auto result = solver.solve();

    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(entry->value, objectiveTol(entry->value)));
    CHECK(result.work_units > 0.0);
    const auto& stats = solver.getPreRootStats();
    CHECK(stats.enabled);
    CHECK(stats.calls >= 1);
    CHECK(stats.work_units > 0.0);
    CHECK(result.work_units >= stats.work_units);
}
