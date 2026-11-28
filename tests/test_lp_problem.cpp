#include "mipx/dual_simplex.h"
#include "mipx/io.h"
#include "mipx/lp_problem.h"
#include "mipx/mip_solver.h"
#include "test_paths.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <utility>
#include <zlib.h>

using namespace mipx;
using Catch::Matchers::WithinAbs;

using mipx::test::testDataDir;

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

TEST_CASE("MPS reader: an objective-row RHS is the negated constant", "[io][mps]") {
    // min x + 2 with x >= 1, written the way e226 writes its constant: as an
    // RHS of -2 on the N row. Read as obj_offset = -RHS, the optimum is 3; the
    // other convention would give -1 (issue #207).
    const std::string path = testDataDir() + "/obj_rhs.mps";
    {
        std::ofstream out(path);
        out << "NAME          OBJRHS\n"
               "ROWS\n"
               " N  COST\n"
               " G  R1\n"
               "COLUMNS\n"
               "    X         COST      1.0          R1        1.0\n"
               "RHS\n"
               "    RHS       COST      -2.0         R1        1.0\n"
               "ENDATA\n";
    }
    auto lp = readMps(path);
    std::filesystem::remove(path);

    CHECK(lp.obj_offset == 2.0);
    REQUIRE(lp.num_rows == 1);
    CHECK(lp.row_lower[0] == 1.0);

    // The writer puts it back on the objective row negated, so a round trip
    // keeps the constant.
    const std::string copy = testDataDir() + "/obj_rhs_roundtrip.mps";
    writeMps(copy, lp);
    auto reread = readMps(copy);
    std::filesystem::remove(copy);
    CHECK(reread.obj_offset == 2.0);
}

TEST_CASE("MPS reader: free MPS data lines may start in column 1", "[io][mps]") {
    // Free MPS allows data lines to start in column 1. Reading them as
    // section headers left a 0 x 0 model that solved to 0 (#267). A column named
    // BOUNDS, an RHS set named RHS and a bound set named BOUNDS are data
    // (more tokens than a header). The SOS section's lines are read as SOS
    // (#254), not taken for bounds, and a member alone on its line is data.
    const std::string path = testDataDir() + "/free_unindented.mps";
    {
        std::ofstream out(path);
        out << "NAME lp\n"
               "OBJSENSE\n"
               "MAX\n"
               "ROWS\n"
               "N obj\n"
               "G r0\n"
               "L r1\n"
               "COLUMNS\n"
               "c0 obj 1 r0 1\n"
               "c1 obj 2 r0 1\n"
               "c1 r1 1\n"
               "BOUNDS obj 3 r1 1\n"
               "RHS\n"
               "RHS r0 3 r1 2\n"
               "BOUNDS\n"
               "UP BOUNDS c0 2.5\n"
               "SOS\n"
               "S1 SOS s1 1\n"
               "c0 1\n"
               "c1\n"
               "ENDATA\n";
    }
    auto lp = readMps(path);
    std::filesystem::remove(path);

    REQUIRE(lp.num_rows == 2);
    REQUIRE(lp.num_cols == 3);
    CHECK(lp.col_names[2] == "BOUNDS");
    CHECK(lp.sense == Sense::Maximize);
    CHECK(lp.obj == std::vector<Real>{1.0, 2.0, 3.0});
    CHECK(lp.row_lower[0] == 3.0);
    CHECK(lp.row_upper[1] == 2.0);
    CHECK(lp.col_upper[0] == 2.5);
    CHECK(lp.col_upper[1] == kInf);
    CHECK(lp.matrix.coeff(1, 1) == 1.0);
    REQUIRE(lp.sos_constraints.size() == 1);
    CHECK(lp.sos_constraints[0].vars == std::vector<Index>{0, 1});
    CHECK(lp.sos_constraints[0].weights == std::vector<Real>{1.0, 2.0});
}

TEST_CASE("MPS writer: round trip keeps every digit and a row named obj", "[io][mps]") {
    // Two things a round trip used to lose: digits past the sixth (the stream
    // default precision), and the objective when a constraint is itself named
    // "obj" -- the reader took the constraint's entries for the objective's.
    LpProblem lp;
    lp.num_cols = 1;
    lp.obj = {0.1234567890123};
    lp.obj_offset = 11.638929066;
    lp.col_lower = {0.0};
    lp.col_upper = {1.0 / 3.0};
    lp.col_type = {VarType::Continuous};
    lp.col_names = {"x"};
    // Rows 1 and 2 are ranged with bounds far enough apart that rebuilding
    // one bound from the other and the range rounds; each has to be written in
    // the form that gives it back exactly. Row 3 is also named like the first
    // fallback objective name, so the uniqueness loop runs twice.
    lp.num_rows = 4;
    lp.row_lower = {1.0 / 7.0, 1.0 / 3.0, 0.1, -kInf};
    lp.row_upper = {kInf, 1e5 + 1.0 / 7.0, 1e16, 5.0};
    lp.row_names = {"obj", "rng1", "rng2", "obj_"};
    lp.matrix = SparseMatrix(
        4, 1, std::vector<Triplet>{{0, 0, 2.0}, {1, 0, 1.0}, {2, 0, 1.0}, {3, 0, 1.0}});

    const std::string path = testDataDir() + "/precision_roundtrip.mps";
    writeMps(path, lp);
    auto reread = readMps(path);
    std::filesystem::remove(path);

    CHECK(reread.obj_offset == lp.obj_offset);
    REQUIRE(reread.num_cols == 1);
    CHECK(reread.obj[0] == lp.obj[0]);
    CHECK(reread.col_upper[0] == lp.col_upper[0]);
    REQUIRE(reread.num_rows == lp.num_rows);
    for (Index i = 0; i < lp.num_rows; ++i) {
        INFO("row " << lp.row_names[i]);
        CHECK(reread.row_names[i] == lp.row_names[i]);
        CHECK(reread.row_lower[i] == lp.row_lower[i]);
        CHECK(reread.row_upper[i] == lp.row_upper[i]);
    }
}

TEST_CASE("MPS reader: OBJSENSE on its own line or on the section line", "[io][mps]") {
    // max x with x <= 4, its sense given in each spelling the readers in use
    // accept (issue #220).
    const auto readWithSense = [](const std::string& sense_lines) {
        const std::string path = testDataDir() + "/objsense.mps";
        {
            std::ofstream out(path);
            out << "NAME          OBJSENSE\n"
                << sense_lines
                << "ROWS\n"
                   " N  COST\n"
                   " L  R1\n"
                   "COLUMNS\n"
                   "    X         COST      1.0          R1        1.0\n"
                   "RHS\n"
                   "    RHS       R1        4.0\n"
                   "ENDATA\n";
        }
        // Removed on every exit, the throwing read included.
        struct RemoveOnExit {
            std::string path;
            ~RemoveOnExit() { std::filesystem::remove(path); }
        } cleanup{path};
        return readMps(path);
    };

    CHECK(readWithSense("").sense == Sense::Minimize);
    CHECK(readWithSense("OBJSENSE\n    MAX\n").sense == Sense::Maximize);
    CHECK(readWithSense("OBJSENSE\n    MAXIMIZE\n").sense == Sense::Maximize);
    CHECK(readWithSense("OBJSENSE MAX\n").sense == Sense::Maximize);
    CHECK(readWithSense("OBJSENSE    MAXIMIZE\n").sense == Sense::Maximize);
    CHECK(readWithSense("OBJSENSE\n    MIN\n").sense == Sense::Minimize);
    CHECK(readWithSense("OBJSENSE MINIMIZE\n").sense == Sense::Minimize);
    CHECK_THROWS_AS(readWithSense("OBJSENSE\n    SIDEWAYS\n"), std::runtime_error);

    // The sense is the model's, not a label: max x over x <= 4 is 4.
    const auto result = solveFeatureModel(readWithSense("OBJSENSE\n    MAX\n"));
    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(4.0, 1e-9));
}

TEST_CASE("MPS writer: round trip keeps a maximisation", "[io][mps]") {
    LpProblem lp;
    lp.sense = Sense::Maximize;
    lp.num_cols = 2;
    lp.obj = {3.0, -1.5};
    lp.obj_offset = 2.25;
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {4.0, kInf};
    lp.col_type = {VarType::Continuous, VarType::Integer};
    lp.col_names = {"x", "y"};
    lp.num_rows = 1;
    lp.row_lower = {-kInf};
    lp.row_upper = {6.0};
    lp.row_names = {"cap"};
    lp.matrix = SparseMatrix(1, 2, std::vector<Triplet>{{0, 0, 1.0}, {0, 1, 1.0}});

    const std::string path = testDataDir() + "/maximize_roundtrip.mps";
    writeMps(path, lp);
    auto reread = readMps(path);
    std::filesystem::remove(path);

    CHECK(reread.sense == Sense::Maximize);
    CHECK(reread.obj == lp.obj);
    CHECK(reread.obj_offset == lp.obj_offset);
    CHECK(reread.col_type == lp.col_type);

    // A minimisation writes no OBJSENSE and still reads back as one.
    lp.sense = Sense::Minimize;
    writeMps(path, lp);
    reread = readMps(path);
    std::filesystem::remove(path);
    CHECK(reread.sense == Sense::Minimize);
    CHECK(reread.obj == lp.obj);
}

TEST_CASE("MPS reader: only the first N row is the objective", "[io][mps]") {
    // A second free row is dropped with its coefficients and RHS; before #220
    // it overwrote the objective and its constant.
    const std::string path = testDataDir() + "/two_free_rows.mps";
    {
        std::ofstream out(path);
        out << "NAME          TWOFREE\n"
               "ROWS\n"
               " N  COST\n"
               " G  R1\n"
               " N  OTHER\n"
               "COLUMNS\n"
               "    X         COST      1.0          R1        1.0\n"
               "    X         OTHER     7.0\n"
               "    Y         OTHER     5.0          R1        1.0\n"
               "RHS\n"
               "    RHS       COST      -2.0         R1        1.0\n"
               "    RHS       OTHER     9.0\n"
               "RANGES\n"
               "    RNG       OTHER     3.0\n"
               "ENDATA\n";
    }
    auto lp = readMps(path);
    std::filesystem::remove(path);

    REQUIRE(lp.num_cols == 2);
    CHECK(lp.obj == std::vector<Real>{1.0, 0.0});
    CHECK(lp.obj_offset == 2.0);
    REQUIRE(lp.num_rows == 1);
    CHECK(lp.row_names[0] == "R1");
    CHECK(lp.row_lower[0] == 1.0);
    CHECK(lp.matrix.numNonzeros() == 2);
}

namespace {

/// The message readMps throws for `file` in tests/data, or "" if it reads.
std::string mpsReadError(const std::string& file) {
    try {
        (void)readMps(testDataDir() + "/" + file);
    } catch (const std::runtime_error& e) {
        return e.what();
    }
    return "";
}

}  // namespace

TEST_CASE("MPS reader: the SOS section becomes SOS constraints", "[io][mps]") {
    // Issue #254: the section was skipped, solving the model without its sets.
    const auto lp = readMps(testDataDir() + "/mps_sos.mps");

    REQUIRE(lp.num_cols == 6);
    REQUIRE(lp.sos_constraints.size() == 2);
    const auto& s1 = lp.sos_constraints[0];
    CHECK(s1.type == LpProblem::SosConstraint::Type::Sos1);
    CHECK(s1.name == "s1");
    CHECK(s1.vars == std::vector<Index>{0, 1, 2});
    CHECK(s1.weights == std::vector<Real>{1.0, 2.0, 3.0});
    const auto& s2 = lp.sos_constraints[1];
    CHECK(s2.type == LpProblem::SosConstraint::Type::Sos2);
    CHECK(s2.name == "s2");
    CHECK(s2.vars == std::vector<Index>{3, 4, 5});
    // Unweighted members take their positions.
    CHECK(s2.weights == std::vector<Real>{1.0, 2.0, 3.0});

    const auto result = solveFeatureModel(lp);
    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(4.0, 1e-6));
}

TEST_CASE("MPS reader: an INDICATORS row becomes an indicator constraint", "[io][mps]") {
    const auto lp = readMps(testDataDir() + "/mps_indicators.mps");

    // Both rows are indicators, so none is left as a linear row.
    CHECK(lp.num_rows == 0);
    CHECK(lp.row_names.empty());
    CHECK(lp.matrix.numNonzeros() == 0);
    REQUIRE(lp.indicator_constraints.size() == 2);
    const auto& rx = lp.indicator_constraints[0];
    CHECK(rx.name == "RX");
    CHECK(rx.binary_var == 2);
    CHECK(rx.active_value);
    CHECK(rx.indices == std::vector<Index>{0});
    CHECK(rx.values == std::vector<Real>{1.0});
    CHECK(rx.lower == -kInf);
    CHECK(rx.upper == 2.0);
    const auto& ry = lp.indicator_constraints[1];
    CHECK(ry.name == "RY");
    CHECK_FALSE(ry.active_value);
    CHECK(ry.indices == std::vector<Index>{1});
    CHECK(ry.upper == 3.0);

    const auto result = solveFeatureModel(lp);
    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(17.0, 1e-6));
}

TEST_CASE("MPS reader: an indicator row leaves the other rows in place", "[io][mps]") {
    const std::string path = (std::filesystem::temp_directory_path() /
                              ("mipx_indicator_middle_" + std::to_string(::getpid()) + ".mps"))
                                 .string();
    {
        std::ofstream out(path);
        out << "NAME IND\nROWS\n N OBJ\n G R0\n L RX\n E R2\nCOLUMNS\n"
               " X OBJ 1 R0 1\n X RX 1 R2 2\n B OBJ 1 R2 1\n"
               "RHS\n RHS R0 1 RX 5\n RHS R2 3\nBOUNDS\n UP BND X 10\n BV BND B\n"
               "INDICATORS\n IF RX B 1\nENDATA\n";
    }
    const auto lp = readMps(path);
    std::filesystem::remove(path);

    REQUIRE(lp.num_rows == 2);
    CHECK(lp.row_names == std::vector<std::string>{"R0", "R2"});
    CHECK(lp.row_lower == std::vector<Real>{1.0, 3.0});
    CHECK(lp.row_upper == std::vector<Real>{kInf, 3.0});
    const auto r2 = lp.matrix.row(1);
    CHECK(std::vector<Index>(r2.indices.begin(), r2.indices.end()) == std::vector<Index>{0, 1});
    CHECK(std::vector<Real>(r2.values.begin(), r2.values.end()) == std::vector<Real>{2.0, 1.0});
    REQUIRE(lp.indicator_constraints.size() == 1);
    CHECK(lp.indicator_constraints[0].upper == 5.0);
}

TEST_CASE("MPS reader: OBJNAME picks the objective among the free rows", "[io][mps]") {
    const auto lp = readMps(testDataDir() + "/mps_objname.mps");

    CHECK(lp.obj == std::vector<Real>{-1.0});
    REQUIRE(lp.num_rows == 1);
    CHECK(lp.row_names[0] == "CAP");
}

TEST_CASE("MPS reader: a section it cannot read is refused by name", "[io][mps]") {
    // Issue #254: these were skipped, solving the linear part as the model.
    for (const auto& [file, section] :
         std::vector<std::pair<std::string, std::string>>{{"mps_quadobj.mps", "QUADOBJ"},
                                                          {"mps_qmatrix.mps", "QMATRIX"},
                                                          {"mps_qcmatrix.mps", "QCMATRIX"},
                                                          {"mps_unknown_section.mps", "FROBNICATE"},
                                                          {"mps_sos_marker.mps", "'SOSORG'"}}) {
        INFO(file);
        const std::string error = mpsReadError(file);
        CHECK(error.find(section) != std::string::npos);
    }
}

TEST_CASE("MPS reader: malformed SOS, INDICATORS and OBJNAME data is refused", "[io][mps]") {
    const auto refusal = [](const std::string& content) {
        const std::string path = (std::filesystem::temp_directory_path() /
                                  ("mipx_bad_extension_" + std::to_string(::getpid()) + ".mps"))
                                     .string();
        {
            std::ofstream out(path, std::ios::trunc);
            out << content;
        }
        std::string error;
        try {
            (void)readMps(path);
        } catch (const std::runtime_error& e) {
            error = e.what();
        }
        std::filesystem::remove(path);
        return error;
    };
    const std::string head =
        "NAME BAD\nROWS\n N OBJ\n L R1\n L R2\nCOLUMNS\n X OBJ 1 R1 1\n B R2 1\n"
        "RHS\n RHS R1 1\nRANGES\n RNG R2 4\nBOUNDS\n BV BND B\n";
    for (const auto& [body, expected] : std::vector<std::pair<std::string, std::string>>{
             {"SOS\n X 1\n", "before any S1 or S2"},
             {"SOS\n S1 s\n NOPE 1\n", "unknown column 'NOPE'"},
             {"INDICATORS\n IF R1 B 2\n", "must be 0 or 1"},
             {"INDICATORS\n IF NOPE B 1\n", "unknown row 'NOPE'"},
             {"INDICATORS\n IF OBJ B 1\n", "free (N) row 'OBJ'"},
             {"INDICATORS\n IF R1 NOPE 1\n", "unknown column 'NOPE'"},
             {"INDICATORS\n IF R1 B\n", "IF row column value"},
             {"INDICATORS\n IF R1 B 1\n IF R1 B 0\n", "names row 'R1' twice"},
             {"INDICATORS\n IF R2 B 1\n", "ranged row"}}) {
        INFO(body);
        CHECK(refusal(head + body + "ENDATA\n").find(expected) != std::string::npos);
    }

    CHECK(refusal("NAME B\nOBJNAME NOPE\nROWS\n N OBJ\nCOLUMNS\n X OBJ 1\nENDATA\n")
              .find("names no free (N) row") != std::string::npos);
    CHECK(refusal("NAME B\nROWS\n N OBJ\nOBJNAME OBJ\nCOLUMNS\n X OBJ 1\nENDATA\n")
              .find("comes after ROWS") != std::string::npos);
    // An extension section before ROWS is refused at the end of an MPS file.
    CHECK(refusal("NAME B\nQUADOBJ\n X X 1\nROWS\n N OBJ\nCOLUMNS\n X OBJ 1\nENDATA\n")
              .find("'QUADOBJ'") != std::string::npos);
    CHECK(refusal("NAME B\nSOS\n S1 s\nROWS\n N OBJ\nCOLUMNS\n X OBJ 1\nENDATA\n")
              .find("'SOS' comes before ROWS") != std::string::npos);

    // A refused header that carries numbers is still a header, not data for
    // the section before it (Gurobi's scenario count, MOSEK's cone fields).
    const std::string model = "NAME B\nROWS\n N OBJ\n L C1\nCOLUMNS\n X OBJ 1 C1 1\nBOUNDS\n";
    CHECK(refusal(model + "SCENARIOS 3\n X OBJ 0\nENDATA\n").find("'SCENARIOS'") !=
          std::string::npos);
    CHECK(refusal(model + "CSECTION k 3e-1 PPOW\n X\nENDATA\n").find("'CSECTION'") !=
          std::string::npos);
    CHECK(refusal("NAME B\nROWS\n N OBJ0 2 1 0 0\n N OBJ1 1 1 0 0\nCOLUMNS\n X OBJ0 1\nENDATA\n")
              .find("multi-objective") != std::string::npos);
}

TEST_CASE("MPS reader: OBJNAME may name a row with a number", "[io][mps]") {
    const std::string path = (std::filesystem::temp_directory_path() /
                              ("mipx_objname_number_" + std::to_string(::getpid()) + ".mps"))
                                 .string();
    {
        std::ofstream out(path);
        out << "NAME B\nOBJNAME 2\nROWS\n N 1\n N 2\nCOLUMNS\n X 1 5 2 7\nENDATA\n";
    }
    const auto lp = readMps(path);
    std::filesystem::remove(path);

    CHECK(lp.obj == std::vector<Real>{7.0});
}

TEST_CASE("readModel: an LP file with an SOS section read as MPS is a format error",
          "[io][format]") {
    const std::string path = (std::filesystem::temp_directory_path() /
                              ("mipx_lp_sos_" + std::to_string(::getpid()) + ".txt"))
                                 .string();
    {
        std::ofstream out(path);
        out << "MAXIMIZE\n obj: x + y\nSUBJECT TO\n c1: x + y <= 3\nSOS\n s1: S1:: x:1 y:2\nEND\n";
    }
    CHECK_THROWS_AS(readModel(path, ModelFormat::Mps), ModelFormatError);
    std::filesystem::remove(path);
}

TEST_CASE("MPS reader: LAZYCONS with a level and OBJSEN are headers", "[io][mps]") {
    const std::string path = (std::filesystem::temp_directory_path() /
                              ("mipx_lazycons_" + std::to_string(::getpid()) + ".mps"))
                                 .string();
    {
        std::ofstream out(path);
        out << "NAME LAZY\nOBJSEN\n    MAX\nROWS\n N OBJ\n L R1\nLAZYCONS 1\n L R2\n"
               "COLUMNS\n X OBJ 1 R1 1\n X R2 1\nRHS\n RHS R1 4 R2 3\nENDATA\n";
    }
    const auto lp = readMps(path);
    std::filesystem::remove(path);

    CHECK(lp.sense == Sense::Maximize);
    CHECK(lp.row_names == std::vector<std::string>{"R1", "R2"});
    CHECK(lp.row_upper == std::vector<Real>{4.0, 3.0});
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

namespace {

/// Write `content` to a temporary .lp file and return its path. The pid keeps
/// concurrent test runs from different checkouts apart.
std::string writeTempLp(const std::string& stem, const std::string& content) {
    const std::string path =
        (std::filesystem::temp_directory_path() / (stem + "_" + std::to_string(::getpid()) + ".lp"))
            .string();
    std::ofstream out(path, std::ios::trunc);
    out << content;
    return path;
}

Index columnNamed(const LpProblem& prob, const std::string& name) {
    for (Index j = 0; j < prob.num_cols; ++j) {
        if (prob.col_names[j] == name) {
            return j;
        }
    }
    return -1;
}

}  // namespace

TEST_CASE("LP reader: every supported bound form parses", "[io][lp]") {
    const std::string path = writeTempLp("mipx_bound_forms",
                                         "Minimize\n"
                                         " obj: a + b + c + d + e + f + g + h + i + j + k + l\n"
                                         "Subject To\n"
                                         " c1: a + b => 1\n"
                                         " c2: a - b =< 2\n"
                                         "Bounds\n"
                                         " a free\n"
                                         " b <= 4\n"
                                         " c >= -2.5\n"
                                         " d = 3\n"
                                         " -1 <= e <= +7\n"
                                         " 2 <= f\n"
                                         " 9 >= g\n"
                                         " 6 = h\n"
                                         " -inf <= i <= 1e3\n"
                                         " j > -infinity\n"
                                         " k =< 5\n"
                                         " l => 1\n"
                                         "End\n");
    const auto prob = readLp(path);
    std::filesystem::remove(path);

    REQUIRE(prob.num_cols == 12);
    // "=>" and "=<" are CPLEX spellings of >= and <=, in constraints too.
    REQUIRE(prob.num_rows == 2);
    CHECK(prob.row_lower[0] == 1.0);
    CHECK(prob.row_upper[0] == kInf);
    CHECK(prob.row_lower[1] == -kInf);
    CHECK(prob.row_upper[1] == 2.0);
    const auto bounds = [&](const std::string& name) {
        const Index j = columnNamed(prob, name);
        REQUIRE(j >= 0);
        return std::pair{prob.col_lower[j], prob.col_upper[j]};
    };
    CHECK(bounds("a") == std::pair{-kInf, kInf});
    CHECK(bounds("b") == std::pair{0.0, 4.0});
    CHECK(bounds("c") == std::pair{-2.5, kInf});
    CHECK(bounds("d") == std::pair{3.0, 3.0});
    CHECK(bounds("e") == std::pair{-1.0, 7.0});
    CHECK(bounds("f") == std::pair{2.0, kInf});
    CHECK(bounds("g") == std::pair{0.0, 9.0});
    CHECK(bounds("h") == std::pair{6.0, 6.0});
    CHECK(bounds("i") == std::pair{-kInf, 1000.0});
    CHECK(bounds("j") == std::pair{-kInf, kInf});
    CHECK(bounds("k") == std::pair{0.0, 5.0});
    CHECK(bounds("l") == std::pair{1.0, kInf});
}

TEST_CASE("LP reader: a malformed bounds line is an error, not a new column", "[io][lp]") {
    // Each line is wrong in a different way; every one used to be accepted,
    // most of them by inventing a column named after the first token.
    const char* const malformed[] = {
        "x1 =~ 3",        // not an operator
        "x1 >== 3",       // not an operator either
        "x1 <= abc",      // value is not a number
        "UP bnd x1 8.0",  // an MPS bound record
        "x1 <= 3 extra",  // trailing junk
        "x1>=3",          // the tokenizer needs whitespace around operators
        "1 <= x1 >= 5",   // mixed double bound
        "3 <= 4",         // no column at all
        "x1 x2 free",     // free needs exactly one name
    };
    for (const char* line : malformed) {
        CAPTURE(line);
        const std::string path = writeTempLp("mipx_bad_bound", std::string("Minimize\n"
                                                                           " obj: x1\n"
                                                                           "Subject To\n"
                                                                           " c1: x1 >= 0\n"
                                                                           "Bounds\n ") +
                                                                   line + "\nEnd\n");
        try {
            readLp(path);
            FAIL("expected readLp to throw");
        } catch (const ModelFormatError& e) {
            const std::string message = e.what();
            CHECK(message.find(path) != std::string::npos);
            CHECK(message.find("line 6") != std::string::npos);
            CHECK(message.find(line) != std::string::npos);
        }
        std::filesystem::remove(path);
    }
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

TEST_CASE("SOS2 interval selectors preserve support and tighten the LP", "[lp_problem][features]") {
    LpProblem lp;
    lp.num_cols = 5;
    lp.obj = {-1.0, 0.0, -0.5, 0.0, -0.25};
    lp.col_lower.assign(5, 0.0);
    lp.col_upper = {1.0, 1.0, 2.0, 1.0, 4.0};
    lp.col_type.assign(5, VarType::Continuous);
    lp.matrix = SparseMatrix(0, lp.num_cols);
    // Weight ties put 3 before 4, so 0, 2 and 4 are mutually nonadjacent.
    lp.sos_constraints.push_back({
        .type = LpProblem::SosConstraint::Type::Sos2,
        .vars = {4, 0, 3, 2, 1},
        .weights = {2.0, 0.0, 2.0, 1.0, 0.0},
        .name = "sos",
    });
    MipSolver solver;
    solver.setVerbose(false);
    solver.setSos2AdjacentIntervals(true);

    SECTION("The interval relaxation excludes fractional nonadjacent activity") {
        auto pairwise = linearizeModelFeatures(lp);
        auto intervals = linearizeModelFeatures(lp, true);
        CHECK(pairwise.num_cols == 10);
        CHECK(pairwise.num_rows == 12);
        CHECK(intervals.num_cols == 9);
        CHECK(intervals.num_rows == 6);
        CHECK(intervals.matrix.numNonzeros() == 17);
        auto relax = [](LpProblem model) {
            model.col_type.assign(model.num_cols, VarType::Continuous);
            DualSimplexSolver simplex;
            simplex.setVerbose(false);
            simplex.load(model);
            auto result = simplex.solve();
            REQUIRE(result.status == Status::Optimal);
            return result.objective;
        };
        CHECK_THAT(relax(pairwise), WithinAbs(-1.5, 1e-6));
        CHECK_THAT(relax(intervals), WithinAbs(-1.0, 1e-6));

        solver.load(lp);
        auto result = solver.solve();
        REQUIRE(result.status == Status::Optimal);
        CHECK_THAT(result.objective, WithinAbs(-1.0, 1e-6));
        CHECK(result.solution.size() == 9);
    }

    SECTION("Two adjacent members can both have positive lower bounds") {
        lp.col_lower[2] = 1.0;
        lp.col_lower[3] = 0.5;
        solver.load(lp);
        auto result = solver.solve();
        REQUIRE(result.status == Status::Optimal);
        CHECK_THAT(result.objective, WithinAbs(-1.0, 1e-6));
        CHECK(result.solution[3] >= 0.5);
        CHECK_THAT(result.solution[4], WithinAbs(0.0, 1e-6));
    }

    SECTION("Forcing two nonadjacent members is infeasible") {
        lp.col_lower[2] = 1.0;
        lp.col_lower[4] = 1.0;
        solver.load(lp);
        CHECK(solver.solve().status == Status::Infeasible);
    }

    SECTION("All members can be zero") {
        lp.obj.assign(lp.num_cols, 1.0);
        solver.load(lp);
        const auto result = solver.solve();
        REQUIRE(result.status == Status::Optimal);
        CHECK_THAT(result.objective, WithinAbs(0.0, 1e-6));
    }

    SECTION("Overlapping unnamed sets get distinct auxiliary names") {
        lp.col_names = {"x0", "x1", "x2", "x3", "x4"};
        lp.sos_constraints.front().name.clear();
        lp.sos_constraints.front().vars = {0, 1, 2};
        lp.sos_constraints.front().weights.clear();
        lp.sos_constraints.push_back({
            .type = LpProblem::SosConstraint::Type::Sos2,
            .vars = {0, 3, 4},
            .weights = {},
            .name = {},
        });
        auto expanded = linearizeModelFeatures(lp, true);
        std::ranges::sort(expanded.col_names);
        std::ranges::sort(expanded.row_names);
        CHECK(std::ranges::adjacent_find(expanded.col_names) == expanded.col_names.end());
        CHECK(std::ranges::adjacent_find(expanded.row_names) == expanded.row_names.end());
    }

    SECTION("Sets with one or two members need no auxiliary rows or columns") {
        lp.sos_constraints.front().weights.clear();
        for (const auto& vars : {std::vector<Index>{4}, std::vector<Index>{4, 0}}) {
            CAPTURE(vars.size());
            lp.sos_constraints.front().vars = vars;
            const auto intervals = linearizeModelFeatures(lp, true);
            CHECK(intervals.num_rows == 0);
            CHECK(intervals.num_cols == lp.num_cols);
            solver.load(lp);
            auto result = solver.solve();
            REQUIRE(result.status == Status::Optimal);
            CHECK_THAT(result.objective, WithinAbs(-3.0, 1e-6));
        }
    }
}

TEST_CASE("Native SOS2 matches an adjacent-window oracle with overlapping sets",
          "[lp_problem][features]") {
    LpProblem lp;
    lp.num_cols = 7;
    lp.obj = {0.0, -4.0, 0.0, -7.0, -0.5, -2.5, -1.2};
    lp.col_lower = {2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    lp.col_upper = {2.0, 1.0, 0.0, 1.0, 1.0, 1.0, 2.0};
    lp.col_type.assign(lp.num_cols, VarType::Continuous);
    lp.col_type[5] = VarType::Binary;
    lp.col_type[6] = VarType::Integer;
    lp.num_rows = 1;
    lp.row_lower = {-kInf};
    lp.row_upper = {3.0};
    lp.matrix = SparseMatrix(1, lp.num_cols, {{0, 3, 1.0}, {0, 6, 2.0}});
    const auto type = LpProblem::SosConstraint::Type::Sos2;
    lp.sos_constraints = {{type, {3, 2, 1}, {2.0, 1.0, 0.0}, "first"},
                          {type, {4, 5, 3}, {1.0, 2.0, 1.0}, "second"}};
    // The second set breaks equal weights by original column number.
    constexpr std::array ordered = {std::array{1, 2, 3}, std::array{3, 4, 5}};

    SECTION("Fixed zero in the middle still separates the endpoints") {}
    SECTION("One branch would zero a member with a positive lower bound") {
        lp.col_lower[1] = 0.5;
    }
    SECTION("Both branches would zero a member with a positive lower bound") {
        lp.col_lower[1] = 0.5;
        lp.col_lower[3] = 0.5;
    }

    Real expected = kInf;
    for (Index exclude_first : {ordered[0].front(), ordered[0].back()}) {
        for (Index exclude_second : {ordered[1].front(), ordered[1].back()}) {
            if (lp.col_lower[exclude_first] > 0.0 || lp.col_lower[exclude_second] > 0.0) {
                continue;
            }
            auto window = lp;
            window.sos_constraints.clear();
            window.col_upper[exclude_first] = window.col_upper[exclude_second] = 0.0;
            const auto result = solveFeatureModel(window);
            if (result.status == Status::Optimal) {
                expected = std::min(expected, result.objective);
            } else {
                CHECK(result.status == Status::Infeasible);
            }
        }
    }
    const auto legacy = solveFeatureModel(lp);
    CHECK(legacy.status == (expected == kInf ? Status::Infeasible : Status::Optimal));
    if (expected != kInf) {
        CHECK_THAT(legacy.objective, WithinAbs(expected, 1e-7));
    }

    const auto holds_support = [&](std::span<const Real> x) {
        return std::ranges::all_of(ordered, [&](const auto& set) {
            return std::abs(x[set.front()]) <= 1e-7 || std::abs(x[set.back()]) <= 1e-7;
        });
    };
    for (Int threads : {1, 3}) {
        for (const auto mode : {ParallelMode::Deterministic, ParallelMode::Opportunistic}) {
            CAPTURE(threads, static_cast<int>(mode));
            MipSolver solver;
            solver.setVerbose(false);
            solver.setGapTolerance(0.0);
            solver.setNumThreads(threads);
            solver.setParallelMode(mode);
            solver.setSos2NativeBranching(true);
            solver.setSos2AdjacentIntervals(true);
            std::atomic<bool> bad_incumbent{false};
            std::atomic<int> observed{0};
            solver.setIncumbentObserver([&](auto x, const auto&) {
                ++observed;
                if (std::cmp_not_equal(x.size(), lp.num_cols) || !holds_support(x)) {
                    bad_incumbent = true;
                }
            });
            solver.load(lp);
            const auto exact = solver.solve();
            CHECK_FALSE(bad_incumbent.load());
            if (expected == kInf) {
                CHECK(exact.status == Status::Infeasible);
                CHECK(exact.solution.empty());
                CHECK(observed == 0);
            } else {
                REQUIRE(exact.status == Status::Optimal);
                CHECK_THAT(exact.objective, WithinAbs(expected, 1e-7));
                REQUIRE(exact.solution.size() == 7);
                CHECK(holds_support(exact.solution));
                CHECK_THAT(exact.solution[0], WithinAbs(2.0, 1e-7));
                CHECK(observed > 0);
            }

            solver.setNodeLimit(2);
            const auto limited = solver.solve();
            CHECK(limited.nodes <= 2);
            CHECK(limited.status != Status::Error);
            CHECK_FALSE(bad_incumbent.load());
            if (!limited.solution.empty()) {
                CHECK(holds_support(limited.solution));
            }
        }
    }
}

TEST_CASE("An unbounded SOS2 relaxation must have SOS2-feasible support",
          "[lp_problem][features]") {
    LpProblem lp;
    lp.num_cols = 4;
    lp.obj = {0.0, 0.0, 0.0, -1.0};
    lp.col_lower = {0.5, 0.0, 0.0, 0.0};
    lp.col_upper = {1.0, 1.0, 1.0, kInf};
    lp.col_type.assign(lp.num_cols, VarType::Continuous);
    lp.matrix = SparseMatrix(0, lp.num_cols);
    lp.sos_constraints = {{LpProblem::SosConstraint::Type::Sos2, {0, 1, 2}, {}, "sos2"}};
    Status expected = Status::Unbounded;
    SECTION("A feasible support can follow the ray") {}
    SECTION("A second nonadjacent required member rules out the ray") {
        lp.col_lower[2] = 0.5;
        expected = Status::Infeasible;
    }
    MipSolver solver;
    solver.setVerbose(false);
    solver.setSos2NativeBranching(true);
    solver.load(lp);
    solver.setNodeLimit(0);
    CHECK(solver.solve().status == Status::NodeLimit);
    solver.setNodeLimit(1);
    CHECK(solver.solve().status == Status::NodeLimit);
    solver.setNodeLimit(100);
    CHECK(solver.solve().status == expected);
    solver.setTimeLimit(0.0);
    const auto stopped = solver.solve();
    CHECK(stopped.status == Status::TimeLimit);
    CHECK(stopped.nodes == 0);
    CHECK(stopped.solution.empty());
}

TEST_CASE("Small nonzero SOS2 members can change the optimum", "[lp_problem][features]") {
    LpProblem lp;
    lp.num_cols = 3;
    lp.obj = {-2e7, 0.0, -2e7};
    lp.col_lower.assign(3, 0.0);
    lp.col_upper = {5e-8, 0.0, 5e-8};
    lp.col_type.assign(3, VarType::Continuous);
    lp.matrix = SparseMatrix(0, lp.num_cols);
    lp.sos_constraints = {{LpProblem::SosConstraint::Type::Sos2, {0, 1, 2}, {}, "sos2"}};

    MipSolver solver;
    solver.setVerbose(false);
    solver.setSos2NativeBranching(true);
    solver.load(lp);
    const auto result = solver.solve();
    REQUIRE(result.status == Status::Optimal);
    REQUIRE(result.solution.size() == 3);
    CHECK_THAT(result.objective, WithinAbs(-1.0, 1e-8));
    CHECK((result.solution.front() == 0.0 || result.solution.back() == 0.0));
}

TEST_CASE("MPS export preserves SOS2 constraints in a lowered native model",
          "[lp_problem][features][io]") {
    LpProblem lp;
    lp.name = "sos2";
    lp.sense = Sense::Maximize;
    lp.num_cols = 4;
    lp.obj = {1.0, 0.0, 0.0, 0.5};
    lp.col_lower.assign(4, 0.0);
    lp.col_upper.assign(4, 1.0);
    lp.col_type.assign(4, VarType::Continuous);
    lp.col_names = {"S2"};
    lp.col_type[2] = VarType::Binary;
    lp.col_upper[2] = 0.0;
    lp.num_rows = 1;
    lp.row_lower = {-kInf};
    lp.row_upper = {1.0};
    lp.matrix = SparseMatrix(1, lp.num_cols, {{0, 1, 1.0}});
    lp.sos_constraints = {
        {LpProblem::SosConstraint::Type::Sos2, {3, 1, 0, 2}, {4.0, 2.0, 1.0, 3.0}, "sos2"}};
    auto expected = Status::Optimal;
    SECTION("Weighted order and missing names survive the round trip") {}
    SECTION("Generated names avoid collisions with supplied names") {
        lp.col_names = {"C1"};
        lp.row_names = {"R1"};
        lp.num_rows = 2;
        lp.row_lower.push_back(-kInf);
        lp.row_upper.push_back(1.0);
        lp.matrix = SparseMatrix(2, lp.num_cols, {{0, 1, 1.0}, {1, 2, 1.0}});
    }
    SECTION("Nonadjacent signed bounds remain infeasible") {
        lp.col_lower[0] = lp.col_upper[0] = -5e-10;
        lp.col_lower[3] = lp.col_upper[3] = -5e-10;
        expected = Status::Infeasible;
    }
    const auto native = linearizeModelFeatures(lp, Sos2Formulation::Native);
    const auto path = std::filesystem::temp_directory_path() /
                      ("mipx_native_sos2_" + std::to_string(::getpid()) + ".mps");
    writeMps(path.string(), native);
    const auto reread = readMps(path.string());
    std::filesystem::remove(path);
    if (lp.num_rows == 2) {
        CHECK(reread.col_names == std::vector<std::string>{"C1", "C1_", "C2", "C3"});
        CHECK(reread.row_names == std::vector<std::string>{"R1", "R1_"});
    } else {
        CHECK(reread.col_names == std::vector<std::string>{"S2", "C1", "C2", "C3"});
        CHECK(reread.row_names == std::vector<std::string>{"R0"});
    }
    CHECK(reread.col_lower == lp.col_lower);
    CHECK(reread.col_upper == lp.col_upper);
    CHECK(reread.col_type == lp.col_type);
    REQUIRE(reread.sos_constraints.size() == 1);
    CHECK(reread.sos_constraints.front().vars == std::vector<Index>{0, 1, 2, 3});
    MipSolver solver;
    solver.setVerbose(false);
    solver.setSos2NativeBranching(true);
    solver.load(reread);
    const auto result = solver.solve();
    REQUIRE(result.status == expected);
    if (expected == Status::Optimal) {
        CHECK_THAT(result.objective, WithinAbs(1.0, 1e-7));
    }
}

TEST_CASE("Presolve stopped at its deadline returns an identity postsolve",
          "[lp_problem][features]") {
    LpProblem lp;
    lp.num_cols = 1;
    lp.obj = {1.0};
    lp.col_lower = {2.0};
    lp.col_upper = {2.0};
    lp.col_type = {VarType::Continuous};
    lp.matrix = SparseMatrix(0, lp.num_cols);
    Presolver presolver;
    presolver.setDeadline(Deadline::after(0.0));
    const auto stopped = presolver.presolve(lp);
    CHECK(stopped.num_cols == 1);
    CHECK_FALSE(presolver.isInfeasible());
    CHECK(presolver.postsolve({2.0}) == std::vector<Real>{2.0});
}

TEST_CASE("Native SOS2 propagation closes overlapping forced windows", "[lp_problem][features]") {
    LpProblem lp;
    lp.sense = Sense::Maximize;
    lp.num_cols = 7;
    lp.obj.assign(7, 1.0);
    lp.col_lower = {1e-10, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0};
    lp.col_upper = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0};
    lp.col_type.assign(7, VarType::Continuous);
    lp.matrix = SparseMatrix(0, lp.num_cols);
    const auto type = LpProblem::SosConstraint::Type::Sos2;
    const std::vector<std::vector<Index>> sets{{0, 1, 2, 3}, {2, 6, 3, 4, 5}};
    lp.sos_constraints = {{type, sets[0], {}, "a"}, {type, sets[1], {}, "b"}};
    SECTION("One forced member restricts each set") {}
    SECTION("Two adjacent forced members restrict a set to their pair") {
        lp.col_lower[1] = 0.5;
    }
    SECTION("A tiny positive lower bound can contradict a distant forced member") {
        lp.col_lower[3] = 1e-10;
    }

    Real expected = -kInf;
    for (std::size_t a = 0; a + 1 < sets[0].size(); ++a) {
        for (std::size_t b = 0; b + 1 < sets[1].size(); ++b) {
            auto window = lp;
            window.sos_constraints.clear();
            for (std::size_t i = 0; i < sets.size(); ++i) {
                const auto first = i == 0 ? a : b;
                for (std::size_t p = 0; p < sets[i].size(); ++p) {
                    if (p < first || p > first + 1) {
                        window.col_upper[sets[i][p]] = 0.0;
                    }
                }
            }
            if (std::ranges::equal(window.col_lower, window.col_upper, std::less_equal{})) {
                const auto oracle = solveFeatureModel(window);
                REQUIRE(oracle.status == Status::Optimal);
                expected = std::max(expected, oracle.objective);
            }
        }
    }
    MipSolver solver;
    solver.setVerbose(false);
    solver.setPresolve(false);
    solver.setGapTolerance(0.0);
    solver.setSos2NativeBranching(true);
    solver.load(lp);
    const auto branching = solver.solve();
    CHECK(solver.getLpStats().root_sos2_propagation.member_visits == 0);
    solver.setSos2Propagation(true);
    const auto propagated = solver.solve();
    const auto stats = solver.getLpStats().root_sos2_propagation;
    REQUIRE(branching.status == propagated.status);
    if (expected == -kInf) {
        CHECK(propagated.status == Status::Infeasible);
        CHECK(stats.prunes == 1);
        CHECK(propagated.lp_iterations == 0);
    } else {
        REQUIRE(propagated.status == Status::Optimal);
        CHECK(stats.fixings == 2);
        CHECK(stats.prunes == 0);
        CHECK_THAT(propagated.objective, WithinAbs(expected, 1e-7));
        CHECK_THAT(branching.objective, WithinAbs(expected, 1e-7));
        REQUIRE(propagated.solution.size() == 7);
        CHECK(std::ranges::equal(propagated.solution, branching.solution,
                                 [](Real lhs, Real rhs) { return std::abs(lhs - rhs) <= 1e-9; }));
        CHECK(propagated.solution[2] == 0.0);
        CHECK(propagated.solution[3] == 0.0);
        CHECK(propagated.solution[6] == 0.0);
    }
}

TEST_CASE("Native SOS2 propagation uses node-local integer branch bounds",
          "[lp_problem][features]") {
    LpProblem lp;
    lp.sense = Sense::Maximize;
    // Columns b, a, middle, c, t_b, t_a. The root branches on b=0.5.
    // At b=0 the LP takes a=0.75, t_a=3 and c=0. The serial plunge
    // visits a=1 first and zeros c. Its a=0 sibling needs c=3 to win.
    lp.num_cols = 6;
    lp.obj = {-10.0, -0.125, 0.0, 1.0, 100.0, 10.0};
    lp.col_lower.assign(6, 0.0);
    lp.col_upper = {1.0, 1.0, 0.0, 3.0, 1.0, 3.0};
    lp.col_type = {VarType::Binary,     VarType::Binary,     VarType::Continuous,
                   VarType::Continuous, VarType::Continuous, VarType::Continuous};
    lp.num_rows = 6;
    lp.row_lower.assign(6, -kInf);
    lp.row_upper = {0.0, 2.0, 0.0, 12.0, 3.0, 3.0};
    lp.matrix = SparseMatrix(6, lp.num_cols,
                             {{0, 4, 1.0},
                              {0, 0, -2.0},
                              {1, 4, 1.0},
                              {1, 0, 2.0},
                              {2, 5, 1.0},
                              {2, 1, -4.0},
                              {3, 5, 1.0},
                              {3, 1, 12.0},
                              {4, 5, 1.0},
                              {4, 4, 3.0},
                              {5, 5, 1.0},
                              {5, 3, 1.0}});
    lp.sos_constraints = {{LpProblem::SosConstraint::Type::Sos2, {1, 2, 3}, {}, "test"}};

    for (Int threads : {1, 2}) {
        for (auto mode : {ParallelMode::Deterministic, ParallelMode::Opportunistic}) {
            MipSolver solver;
            solver.setVerbose(false);
            solver.setPresolve(false);
            solver.setCutsEnabled(false);
            solver.setStrongBranchingEnabled(false);
            solver.setRootHeuristicThresholds(0, 0);
            solver.setPlungeControls(2, 100.0);
            solver.setGapTolerance(0.0);
            solver.setNumThreads(threads);
            solver.setParallelMode(mode);
            solver.setSos2NativeBranching(true);
            solver.load(lp);
            const auto baseline = solver.solve();
            REQUIRE(baseline.status == Status::Optimal);
            CHECK_THAT(baseline.objective, WithinAbs(3.0, 1e-7));
            CHECK(solver.getLpStats().tree_sos2_propagation.member_visits == 0);
            solver.setSos2Propagation(true);
            const auto propagated = solver.solve();
            REQUIRE(propagated.status == Status::Optimal);
            CHECK_THAT(propagated.objective, WithinAbs(3.0, 1e-7));
            CHECK(
                std::ranges::equal(propagated.solution, baseline.solution,
                                   [](Real lhs, Real rhs) { return std::abs(lhs - rhs) <= 1e-9; }));
            CHECK(propagated.solution[1] == 0.0);
            CHECK_THAT(propagated.solution[3], WithinAbs(3.0, 1e-7));
            if (threads == 1) {
                CHECK(solver.getSearchStats().plunge_nodes > 0);
            }
            CHECK(solver.getLpStats().root_sos2_propagation.fixings == 0);
            CHECK(solver.getLpStats().tree_sos2_propagation.fixings > 0);
        }
    }
}

TEST_CASE("Native SOS2 prunes support forced by pre-LP row bounds", "[lp_problem][features]") {
    LpProblem lp;
    lp.sense = Sense::Maximize;
    lp.num_cols = 5;
    lp.obj = {0.0, -1.0, 0.0, -1.0, 10.0};
    lp.col_lower.assign(5, 0.0);
    lp.col_upper = {1.0, 1.0, 0.0, 1.0, 1.0};
    lp.col_type.assign(5, VarType::Binary);
    // Branching up from b=0.5 forces both nonadjacent members through the rows.
    lp.num_rows = 4;
    lp.row_lower.assign(4, -kInf);
    lp.row_upper = {0.0, 2.0, 1.0, 1.0};
    lp.matrix = SparseMatrix(4, lp.num_cols,
                             {{0, 4, 1.0},
                              {0, 0, -2.0},
                              {1, 4, 1.0},
                              {1, 0, 2.0},
                              {2, 0, 2.0},
                              {2, 1, -1.0},
                              {3, 0, 2.0},
                              {3, 3, -1.0}});
    lp.sos_constraints = {{LpProblem::SosConstraint::Type::Sos2, {1, 2, 3}, {}, "test"}};
    MipSolver solver;
    solver.setVerbose(false);
    solver.setPresolve(false);
    solver.setCutsEnabled(false);
    solver.setStrongBranchingEnabled(false);
    solver.setRootHeuristicThresholds(0, 0);
    solver.setTreePresolveAutoTuning(false);
    solver.setTreePresolveControls(2, 1, 1);
    solver.setPreLpPropagationBoundsEnabled(true);
    solver.setSos2NativeBranching(true);
    solver.setSos2Propagation(true);
    solver.load(lp);
    const auto result = solver.solve();
    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(result.objective, WithinAbs(0.0, 1e-7));
    CHECK(solver.getLpStats().pre_lp_propagation_fixings >= 2);
    CHECK(solver.getLpStats().tree_sos2_propagation.prunes == 1);
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

    // An MPS file forced through the LP reader. Its BOUNDS records used to
    // invent columns, so the parsed model was not empty; the LP reader now
    // refuses them as malformed bound lines, and the file must be refused
    // either way.
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

    // And the same in reverse: an MPS file has no LP-exclusive header. (One
    // with bound records does not get this far: its BOUNDS lines are not LP
    // bounds, and readLp refuses them.)
    const std::string no_bounds_mps =
        (std::filesystem::temp_directory_path() /
         ("mipx_no_bounds_test_" + std::to_string(::getpid()) + ".mps"))
            .string();
    {
        std::ofstream out(no_bounds_mps, std::ios::trunc);
        out << "NAME nobounds\nROWS\n N obj\n L c1\nCOLUMNS\n x obj 1 c1 1\n"
               "RHS\n rhs c1 4\nENDATA\n";
    }
    readLp(no_bounds_mps, &diag);
    std::filesystem::remove(no_bounds_mps);
    CHECK(diag.saw_content);
    CHECK_FALSE(diag.saw_format_section);
    CHECK_THROWS_AS(readLp(testDataDir() + "/tiny.mps"), ModelFormatError);
    CHECK_THROWS_AS(readLp(testDataDir() + "/empty_model.mps"), ModelFormatError);

    // saw_content answers "was there anything in the file at all", separately
    // from whether any of it was understood.
    readMps(testDataDir() + "/malformed.mps", &diag);
    CHECK(diag.saw_content);
    CHECK_FALSE(diag.saw_format_section);

    readLp(testDataDir() + "/lp_format.lp", &diag);
    CHECK(diag.saw_format_section);
}
