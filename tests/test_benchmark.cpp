#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <filesystem>
#include <format>
#include <iostream>
#include <string>
#include <vector>

#include "mipx/io.h"
#include "mipx/lp_problem.h"

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
