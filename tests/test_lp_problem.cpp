#include "mipx/io.h"
#include "mipx/lp_problem.h"
#include "mipx/mip_solver.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <zlib.h>

using namespace mipx;
using Catch::Matchers::WithinAbs;

static std::string testDataDir() {
    return std::string(TEST_DATA_DIR);
}

static MipResult solveFeatureModel(const LpProblem& lp) {
    MipSolver solver;
    solver.setVerbose(false);
    solver.setPresolve(false);
    solver.setCutsEnabled(false);
    solver.load(lp);
    return solver.solve();
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
            CHECK_THAT(reread.col_upper[j], WithinAbs(orig.col_upper[j], 1e-12));
        }
    }

    for (Index i = 0; i < orig.num_rows; ++i) {
        if (orig.row_lower[i] > -kInf) {
            CHECK_THAT(reread.row_lower[i], WithinAbs(orig.row_lower[i], 1e-12));
        }
        if (orig.row_upper[i] < kInf) {
            CHECK_THAT(reread.row_upper[i], WithinAbs(orig.row_upper[i], 1e-12));
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
        if (prob.col_names[j] == "x1") {
            x1 = j;
        }
        if (prob.col_names[j] == "x2") {
            x2 = j;
        }
        if (prob.col_names[j] == "x3") {
            x3 = j;
        }
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
    std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
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

TEST_CASE("Feature linearization: semi-continuous and semi-integer", "[lp_problem][features]") {
    LpProblem semi_cont;
    semi_cont.name = "semi_cont";
    semi_cont.num_cols = 1;
    semi_cont.obj = {-1.0};
    semi_cont.col_lower = {0.0};
    semi_cont.col_upper = {5.0};
    semi_cont.col_type = {VarType::SemiContinuous};
    semi_cont.col_semi_lower = {2.0};
    semi_cont.col_names = {"x"};
    semi_cont.num_rows = 0;
    semi_cont.matrix = SparseMatrix(0, 1, {});
    const auto semi_cont_result = solveFeatureModel(semi_cont);
    REQUIRE(semi_cont_result.status == Status::Optimal);
    CHECK_THAT(semi_cont_result.objective, WithinAbs(-5.0, 1e-6));
    CHECK(semi_cont_result.work_units > 0.0);

    LpProblem semi_int = semi_cont;
    semi_int.name = "semi_int";
    semi_int.col_type = {VarType::SemiInteger};
    const auto semi_int_result = solveFeatureModel(semi_int);
    REQUIRE(semi_int_result.status == Status::Optimal);
    CHECK_THAT(semi_int_result.objective, WithinAbs(-5.0, 1e-6));
    CHECK(semi_int_result.work_units > 0.0);
}

TEST_CASE("Feature linearization: indicator fallback and MPS round-trip",
          "[lp_problem][features][io]") {
    LpProblem lp;
    lp.name = "indicator_fallback";
    lp.num_cols = 2;
    lp.obj = {-0.5, 1.0};  // x, y
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {10.0, 1.0};
    lp.col_type = {VarType::Continuous, VarType::Binary};
    lp.col_semi_lower = {0.0, 0.0};
    lp.col_names = {"x", "y"};
    lp.num_rows = 0;
    lp.matrix = SparseMatrix(0, 2, {});
    lp.indicator_constraints.push_back({
        .binary_var = 1,
        .active_value = false,
        .indices = {0},
        .values = {1.0},
        .lower = -kInf,
        .upper = 0.0,
        .name = "ind",
    });

    const auto direct = solveFeatureModel(lp);
    REQUIRE(direct.status == Status::Optimal);
    CHECK_THAT(direct.objective, WithinAbs(-4.0, 1e-6));
    CHECK(direct.work_units > 0.0);

    const std::string path = testDataDir() + "/indicator_roundtrip.mps";
    writeMps(path, lp);
    const auto reread = readMps(path);
    const auto reread_result = solveFeatureModel(reread);
    REQUIRE(reread_result.status == Status::Optimal);
    CHECK_THAT(reread_result.objective, WithinAbs(-4.0, 1e-6));
    CHECK(reread_result.work_units > 0.0);
    std::filesystem::remove(path);
}

TEST_CASE("Feature linearization: SOS1 and SOS2", "[lp_problem][features]") {
    LpProblem sos1;
    sos1.name = "sos1_model";
    sos1.num_cols = 2;
    sos1.obj = {-1.0, -1.0};
    sos1.col_lower = {0.0, 0.0};
    sos1.col_upper = {5.0, 5.0};
    sos1.col_type = {VarType::Continuous, VarType::Continuous};
    sos1.col_semi_lower = {0.0, 0.0};
    sos1.col_names = {"x1", "x2"};
    sos1.num_rows = 0;
    sos1.matrix = SparseMatrix(0, 2, {});
    sos1.sos_constraints.push_back({
        .type = LpProblem::SosConstraint::Type::Sos1,
        .vars = {0, 1},
        .weights = {1.0, 2.0},
        .name = "s1",
    });
    const auto sos1_result = solveFeatureModel(sos1);
    REQUIRE(sos1_result.status == Status::Optimal);
    CHECK_THAT(sos1_result.objective, WithinAbs(-5.0, 1e-6));
    CHECK(sos1_result.work_units > 0.0);

    LpProblem sos2;
    sos2.name = "sos2_model";
    sos2.num_cols = 3;
    sos2.obj = {-1.0, 0.0, -1.0};
    sos2.col_lower = {0.0, 0.0, 0.0};
    sos2.col_upper = {5.0, 5.0, 5.0};
    sos2.col_type = {VarType::Continuous, VarType::Continuous, VarType::Continuous};
    sos2.col_semi_lower = {0.0, 0.0, 0.0};
    sos2.col_names = {"x1", "x2", "x3"};
    sos2.num_rows = 0;
    sos2.matrix = SparseMatrix(0, 3, {});
    sos2.sos_constraints.push_back({
        .type = LpProblem::SosConstraint::Type::Sos2,
        .vars = {0, 1, 2},
        .weights = {1.0, 2.0, 3.0},
        .name = "s2",
    });
    const auto sos2_result = solveFeatureModel(sos2);
    REQUIRE(sos2_result.status == Status::Optimal);
    CHECK_THAT(sos2_result.objective, WithinAbs(-5.0, 1e-6));
    CHECK(sos2_result.work_units > 0.0);
}

// ---------------------------------------------------------------------------
// Format dispatch and empty-parse rejection (issue #198).
//
// The readers ignore any line outside a section they know, so feeding one the
// wrong format yields a 0x0 model rather than an error. readModel() picks the
// reader from the extension and turns that silent empty parse into a throw.
// ---------------------------------------------------------------------------

TEST_CASE("detectModelFormat: extension picks the reader", "[io][format]") {
    CHECK(detectModelFormat("model.mps") == ModelFormat::Mps);
    CHECK(detectModelFormat("model.lp") == ModelFormat::Lp);

    // A compression suffix is stripped before the format extension is read,
    // matching what the MPS reader itself decompresses.
    CHECK(detectModelFormat("model.mps.gz") == ModelFormat::Mps);
    CHECK(detectModelFormat("model.mps.bz2") == ModelFormat::Mps);
    CHECK(detectModelFormat("model.lp.gz") == ModelFormat::Lp);

    // Extensions match case-insensitively: case-insensitive filesystems make
    // "Model.LP" an ordinary name.
    CHECK(detectModelFormat("Model.LP") == ModelFormat::Lp);
    CHECK(detectModelFormat("Model.LP.GZ") == ModelFormat::Lp);
    CHECK(detectModelFormat("MODEL.MPS.GZ") == ModelFormat::Mps);

    // Netlib instances often carry no extension at all; MPS stays the default.
    CHECK(detectModelFormat("afiro") == ModelFormat::Mps);
    CHECK(detectModelFormat("/some/dir.lp/afiro") == ModelFormat::Mps);
    CHECK(detectModelFormat("model.txt") == ModelFormat::Mps);
}

TEST_CASE("readModel: dispatches a .lp file to the LP reader", "[io][format]") {
    auto prob = readModel(testDataDir() + "/lp_format.lp");
    CHECK(prob.sense == Sense::Maximize);
    CHECK(prob.num_cols == 2);
    CHECK(prob.num_rows == 1);
}

TEST_CASE("readModel: dispatches a .mps file to the MPS reader", "[io][format]") {
    auto prob = readModel(testDataDir() + "/tiny.mps");
    CHECK(prob.name == "tiny");
    CHECK(prob.num_cols == 3);
    CHECK(prob.num_rows == 3);
}

TEST_CASE("readModel: explicit format overrides the extension", "[io][format]") {
    auto prob = readModel(testDataDir() + "/lp_format.lp", ModelFormat::Lp);
    CHECK(prob.num_cols == 2);

    // The same file forced through the MPS reader is a format error, not a
    // 0x0 model reported as solved.
    CHECK_THROWS_AS(readModel(testDataDir() + "/lp_format.lp", ModelFormat::Mps), ModelFormatError);
}

TEST_CASE("readModel: rejects a malformed MPS file", "[io][format]") {
    const std::string path = testDataDir() + "/malformed.mps";
    REQUIRE_THROWS_AS(readModel(path), ModelFormatError);

    // The message has to name the file and the format problem.
    try {
        readModel(path);
        FAIL("expected readModel to throw");
    } catch (const ModelFormatError& e) {
        const std::string message = e.what();
        CHECK(message.find("malformed.mps") != std::string::npos);
        CHECK(message.find("MPS") != std::string::npos);
    }
}

TEST_CASE("readModel: an uppercase compression suffix is still decompressed", "[io][format]") {
    // Resolving the format past a suffix the reader then refuses to decompress
    // would leave gzip bytes to be parsed as text, so the two must agree on
    // what counts as compressed.
    const std::string src = testDataDir() + "/tiny.mps";
    const std::string dst =
        (std::filesystem::temp_directory_path() / "mipx_upper_suffix_test.MPS.GZ").string();

    std::ifstream in(src, std::ios::binary);
    REQUIRE(in.is_open());
    const std::string content((std::istreambuf_iterator<char>(in)),
                              std::istreambuf_iterator<char>());
    in.close();

    gzFile gz = gzopen(dst.c_str(), "wb");
    REQUIRE(gz != nullptr);
    gzwrite(gz, content.data(), static_cast<unsigned>(content.size()));
    gzclose(gz);

    auto prob = readModel(dst);
    CHECK(prob.name == "tiny");
    CHECK(prob.num_cols == 3);
    CHECK(prob.num_rows == 3);

    std::filesystem::remove(dst);
}

TEST_CASE("readModel: a truncated gzip stream is an error, not a partial model", "[io][format]") {
    // gzread reports EOF and failure the same way, so a damaged archive used to
    // yield whatever decompressed before the break -- a partial model that
    // solves and is reported optimal. That is the silent wrong answer #198 is
    // about, and the format check cannot catch it: the readable prefix contains
    // a real ROWS header, so the file does look like MPS.
    const std::string src = testDataDir() + "/tiny.mps";
    const std::string whole =
        (std::filesystem::temp_directory_path() / "mipx_truncated_whole.mps.gz").string();
    const std::string cut =
        (std::filesystem::temp_directory_path() / "mipx_truncated_cut.mps.gz").string();

    std::ifstream in(src, std::ios::binary);
    REQUIRE(in.is_open());
    const std::string content((std::istreambuf_iterator<char>(in)),
                              std::istreambuf_iterator<char>());
    in.close();

    gzFile gz = gzopen(whole.c_str(), "wb");
    REQUIRE(gz != nullptr);
    gzwrite(gz, content.data(), static_cast<unsigned>(content.size()));
    gzclose(gz);

    // Sanity: the intact archive reads.
    REQUIRE(readModel(whole).num_cols == 3);

    // Now cut it in half and read again.
    std::ifstream packed(whole, std::ios::binary);
    REQUIRE(packed.is_open());
    const std::string bytes((std::istreambuf_iterator<char>(packed)),
                            std::istreambuf_iterator<char>());
    packed.close();
    REQUIRE(bytes.size() > 40);
    {
        std::ofstream out(cut, std::ios::binary | std::ios::trunc);
        REQUIRE(out.is_open());
        out.write(bytes.data(), static_cast<std::streamsize>(bytes.size() / 2));
    }

    CHECK_THROWS_AS(readModel(cut), std::runtime_error);

    std::filesystem::remove(whole);
    std::filesystem::remove(cut);
}

TEST_CASE("readModel: an empty model survives compression", "[io][format]") {
    // "The input file was not empty" cannot be answered from the file size once
    // a compression layer is involved: an empty file gzips to a header's worth
    // of bytes. The reader reports whether it saw any content instead.
    const std::string path =
        (std::filesystem::temp_directory_path() / "mipx_empty_compressed.mps.gz").string();
    gzFile gz = gzopen(path.c_str(), "wb");
    REQUIRE(gz != nullptr);
    gzclose(gz);
    REQUIRE(std::filesystem::file_size(path) > 0);  // non-empty on disk...

    auto prob = readModel(path);  // ...but empty as a model, so no error.
    CHECK(prob.num_rows == 0);
    CHECK(prob.num_cols == 0);

    std::filesystem::remove(path);
}

TEST_CASE("readModel: BOUNDS alone does not identify a format", "[io][format]") {
    // BOUNDS is a section keyword in both MPS and CPLEX LP, and each reader
    // manufactures columns from the other's bound records. So neither an empty
    // parse nor a plausible-looking model size can be trusted here: only a
    // header exclusive to the format settles it.

    // An MPS file forced through the LP reader. Its BOUNDS records invent
    // columns, so the parsed model is not empty -- and still must be refused.
    CHECK_THROWS_AS(readModel(testDataDir() + "/tiny.mps", ModelFormat::Lp), ModelFormatError);

    // The mirror image: an uppercase LP file whose extension says nothing, so
    // it is read as MPS. This is issue #198's silent "Optimal 0.0" all over
    // again if BOUNDS is allowed to vouch for the format.
    CHECK_THROWS_AS(readModel(testDataDir() + "/lp_format_upper.txt"), ModelFormatError);

    // Forced to its real format it reads fine, and is the same model as the
    // lowercase reproducer.
    auto prob = readModel(testDataDir() + "/lp_format_upper.txt", ModelFormat::Lp);
    CHECK(prob.sense == Sense::Maximize);
    CHECK(prob.num_cols == 2);
    CHECK(prob.num_rows == 1);
}

TEST_CASE("readModel: a missing file is an I/O error, not a format error", "[io][format]") {
    // The CLI only offers --format advice for ModelFormatError, so an
    // unreadable file must not be reported as one.
    CHECK_THROWS_AS(readModel("nonexistent_model.mps"), std::runtime_error);
    try {
        readModel("nonexistent_model.mps");
        FAIL("expected readModel to throw");
    } catch (const ModelFormatError&) {
        FAIL("a missing file must not be reported as a format problem");
    } catch (const std::runtime_error&) {
        SUCCEED();
    }
}

TEST_CASE("readModel: a valid MPS file with an empty model is not an error", "[io][format]") {
    auto prob = readModel(testDataDir() + "/empty_model.mps");
    CHECK(prob.num_rows == 0);
    CHECK(prob.num_cols == 0);
}

TEST_CASE("readModel: a zero-byte file is not a format error", "[io][format]") {
    const std::string path =
        (std::filesystem::temp_directory_path() / "mipx_zero_byte_test.mps").string();
    {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        REQUIRE(out.is_open());
    }

    auto prob = readModel(path);
    CHECK(prob.num_rows == 0);
    CHECK(prob.num_cols == 0);

    std::filesystem::remove(path);
}

TEST_CASE("readModel: compressed LP input is refused with a clear error", "[io][format]") {
    // readLp() opens a plain std::ifstream, so it cannot read a gzip stream.
    // Say so rather than letting the binary bytes parse to an empty model.
    const std::string path =
        (std::filesystem::temp_directory_path() / "mipx_compressed_test.lp.gz").string();
    gzFile gz = gzopen(path.c_str(), "wb");
    REQUIRE(gz != nullptr);
    const std::string content = "Minimize\n obj: x\nEnd\n";
    gzwrite(gz, content.data(), static_cast<unsigned>(content.size()));
    gzclose(gz);

    try {
        readModel(path);
        FAIL("expected readModel to throw");
    } catch (const ModelFormatError& e) {
        const std::string message = e.what();
        CHECK(message.find("mipx_compressed_test.lp.gz") != std::string::npos);
        CHECK(message.find("compressed") != std::string::npos);
    }

    std::filesystem::remove(path);
}

TEST_CASE("readMps/readLp: diagnostics report whether the format was seen", "[io][format]") {
    ReadDiagnostics diag;

    readMps(testDataDir() + "/tiny.mps", &diag);
    CHECK(diag.saw_format_section);

    // An LP file run through the MPS reader hits no MPS-exclusive header.
    readMps(testDataDir() + "/lp_format.lp", &diag);
    CHECK_FALSE(diag.saw_format_section);

    // ...not even when it has a BOUNDS section, which MPS also spells BOUNDS.
    readMps(testDataDir() + "/lp_format_upper.txt", &diag);
    CHECK_FALSE(diag.saw_format_section);

    // And the same in reverse: an MPS file has no LP-exclusive header.
    readLp(testDataDir() + "/tiny.mps", &diag);
    CHECK_FALSE(diag.saw_format_section);

    // saw_content answers "was there anything in the file at all", separately
    // from whether any of it was understood.
    readMps(testDataDir() + "/malformed.mps", &diag);
    CHECK(diag.saw_content);
    CHECK_FALSE(diag.saw_format_section);

    readLp(testDataDir() + "/lp_format.lp", &diag);
    CHECK(diag.saw_format_section);
}
