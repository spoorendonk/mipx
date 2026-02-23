#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <filesystem>
#include <fstream>

#include <zlib.h>

#include "mipx/io.h"
#include "mipx/lp_problem.h"

using namespace mipx;
using Catch::Matchers::WithinAbs;

static std::string testDataDir() {
    return std::string(TEST_DATA_DIR);
}

TEST_CASE("LpProblem: default construction", "[lp_problem]") {
    LpProblem prob;
    CHECK(prob.num_cols == 0);
    CHECK(prob.num_rows == 0);
    CHECK(prob.sense == Sense::Minimize);
    CHECK(prob.obj_offset == 0.0);
    CHECK_FALSE(prob.hasIntegers());
}

TEST_CASE("LpProblem: hasIntegers", "[lp_problem]") {
    LpProblem prob;
    prob.num_cols = 2;
    prob.col_type = {VarType::Continuous, VarType::Continuous};
    CHECK_FALSE(prob.hasIntegers());

    prob.col_type[1] = VarType::Integer;
    CHECK(prob.hasIntegers());

    prob.col_type[1] = VarType::Binary;
    CHECK(prob.hasIntegers());
}

TEST_CASE("MPS reader: tiny.mps", "[io][mps]") {
    auto prob = readMps(testDataDir() + "/tiny.mps");

    CHECK(prob.name == "tiny");
    CHECK(prob.num_cols == 3);
    CHECK(prob.num_rows == 3);

    // Variable names.
    REQUIRE(prob.col_names.size() == 3);
    CHECK(prob.col_names[0] == "x1");
    CHECK(prob.col_names[1] == "x2");
    CHECK(prob.col_names[2] == "x3");

    // Objective.
    CHECK(prob.obj[0] == 1.0);
    CHECK(prob.obj[1] == 3.0);
    CHECK(prob.obj[2] == 2.0);

    // Variable bounds.
    CHECK(prob.col_lower[0] == 0.0);
    CHECK(prob.col_upper[0] == 8.0);
    CHECK(prob.col_lower[1] == 0.0);
    CHECK(prob.col_upper[1] == 1.0);
    CHECK(prob.col_lower[2] == 1.0);
    CHECK(prob.col_upper[2] == 5.0);

    // Variable types.
    CHECK(prob.col_type[0] == VarType::Continuous);
    CHECK(prob.col_type[1] == VarType::Binary);  // BV overrides INTORG
    CHECK(prob.col_type[2] == VarType::Continuous);

    // Constraint names.
    REQUIRE(prob.row_names.size() == 3);
    CHECK(prob.row_names[0] == "c1");
    CHECK(prob.row_names[1] == "c2");
    CHECK(prob.row_names[2] == "c3");

    // Row bounds (converted from sense + rhs).
    // c1: L, rhs=10 → [-inf, 10]
    CHECK(prob.row_lower[0] == -kInf);
    CHECK(prob.row_upper[0] == 10.0);
    // c2: G, rhs=5 → [5, inf]
    CHECK(prob.row_lower[1] == 5.0);
    CHECK(prob.row_upper[1] == kInf);
    // c3: E, rhs=7 → [7, 7]
    CHECK(prob.row_lower[2] == 7.0);
    CHECK(prob.row_upper[2] == 7.0);

    // Matrix coefficients.
    CHECK(prob.matrix.coeff(0, 0) == 2.0);  // c1, x1
    CHECK(prob.matrix.coeff(0, 1) == 1.0);  // c1, x2
    CHECK(prob.matrix.coeff(0, 2) == 0.0);  // c1, x3
    CHECK(prob.matrix.coeff(1, 0) == 1.0);  // c2, x1
    CHECK(prob.matrix.coeff(1, 2) == 1.0);  // c2, x3
    CHECK(prob.matrix.coeff(2, 1) == 1.0);  // c3, x2
    CHECK(prob.matrix.coeff(2, 2) == 1.0);  // c3, x3

    CHECK(prob.hasIntegers());
}

TEST_CASE("MPS writer: round-trip", "[io][mps]") {
    auto orig = readMps(testDataDir() + "/tiny.mps");

    // Write to temp file.
    std::string tmp = testDataDir() + "/tiny_roundtrip.mps";
    writeMps(tmp, orig);

    // Read back.
    auto reread = readMps(tmp);

    CHECK(reread.num_cols == orig.num_cols);
    CHECK(reread.num_rows == orig.num_rows);

    for (Index j = 0; j < orig.num_cols; ++j) {
        CHECK_THAT(reread.obj[j], WithinAbs(orig.obj[j], 1e-12));
        CHECK_THAT(reread.col_lower[j], WithinAbs(orig.col_lower[j], 1e-12));
        if (orig.col_upper[j] < kInf) {
            CHECK_THAT(reread.col_upper[j],
                        WithinAbs(orig.col_upper[j], 1e-12));
        }
    }

    for (Index i = 0; i < orig.num_rows; ++i) {
        if (orig.row_lower[i] > -kInf) {
            CHECK_THAT(reread.row_lower[i],
                        WithinAbs(orig.row_lower[i], 1e-12));
        }
        if (orig.row_upper[i] < kInf) {
            CHECK_THAT(reread.row_upper[i],
                        WithinAbs(orig.row_upper[i], 1e-12));
        }
    }

    // Clean up.
    std::filesystem::remove(tmp);
}

TEST_CASE("LP reader: tiny.lp", "[io][lp]") {
    auto prob = readLp(testDataDir() + "/tiny.lp");

    CHECK(prob.sense == Sense::Minimize);
    CHECK(prob.num_cols == 3);
    CHECK(prob.num_rows == 3);

    // Objective.
    // Find indices by name since LP reader order may differ.
    Index x1 = -1, x2 = -1, x3 = -1;
    for (Index j = 0; j < prob.num_cols; ++j) {
        if (prob.col_names[j] == "x1") x1 = j;
        if (prob.col_names[j] == "x2") x2 = j;
        if (prob.col_names[j] == "x3") x3 = j;
    }
    REQUIRE(x1 >= 0);
    REQUIRE(x2 >= 0);
    REQUIRE(x3 >= 0);

    CHECK(prob.obj[x1] == 1.0);
    CHECK(prob.obj[x2] == 3.0);
    CHECK(prob.obj[x3] == 2.0);

    // Bounds.
    CHECK(prob.col_lower[x1] == 0.0);
    CHECK(prob.col_upper[x1] == 8.0);
    CHECK(prob.col_lower[x3] == 1.0);
    CHECK(prob.col_upper[x3] == 5.0);

    // Integer type.
    CHECK(prob.col_type[x2] == VarType::Integer);

    // Constraints.
    // c1: 2x1 + x2 <= 10
    CHECK(prob.row_upper[0] == 10.0);
    CHECK(prob.row_lower[0] == -kInf);
    // c2: x1 + x3 >= 5
    CHECK(prob.row_lower[1] == 5.0);
    CHECK(prob.row_upper[1] == kInf);
    // c3: x2 + x3 = 7
    CHECK(prob.row_lower[2] == 7.0);
    CHECK(prob.row_upper[2] == 7.0);
}

TEST_CASE("Solu reader: tiny.solu", "[io][solu]") {
    auto entries = readSolu(testDataDir() + "/tiny.solu");

    REQUIRE(entries.size() == 3);

    CHECK(entries[0].name == "tiny");
    CHECK_THAT(entries[0].value, WithinAbs(42.0, 1e-12));
    CHECK_FALSE(entries[0].is_infeasible);

    CHECK(entries[1].name == "infeasible_problem");
    CHECK(entries[1].is_infeasible);

    CHECK(entries[2].name == "another");
    CHECK_THAT(entries[2].value, WithinAbs(123.456, 1e-9));
}

TEST_CASE("MPS reader: missing file throws", "[io][mps]") {
    CHECK_THROWS_AS(readMps("nonexistent.mps"), std::runtime_error);
}

TEST_CASE("LP reader: missing file throws", "[io][lp]") {
    CHECK_THROWS_AS(readLp("nonexistent.lp"), std::runtime_error);
}

TEST_CASE("MPS reader: gzip support", "[io][mps]") {
    // Create a gzipped copy of tiny.mps.
    std::string src = testDataDir() + "/tiny.mps";
    std::string dst = testDataDir() + "/tiny_test.mps.gz";

    // Read original file content.
    std::ifstream in(src, std::ios::binary);
    REQUIRE(in.is_open());
    std::string content((std::istreambuf_iterator<char>(in)),
                        std::istreambuf_iterator<char>());
    in.close();

    // Write gzipped.
    gzFile gz = gzopen(dst.c_str(), "wb");
    REQUIRE(gz != nullptr);
    gzwrite(gz, content.data(), static_cast<unsigned>(content.size()));
    gzclose(gz);

    // Read the gzipped file.
    auto prob = readMps(dst);
    CHECK(prob.name == "tiny");
    CHECK(prob.num_cols == 3);
    CHECK(prob.num_rows == 3);

    // Clean up.
    std::filesystem::remove(dst);
}
