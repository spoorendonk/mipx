// Issue #259: MIPLIB declares its 0/1 variables as MARKER INTORG integers with
// UP 1, which the MPS reader types Integer, not Binary. Everything that asks
// "is this column binary" must treat such a column as a BV one, so the same
// knapsack read both ways must separate the same cuts and break the same
// symmetry.

#include "mipx/cut_pool.h"
#include "mipx/dual_simplex.h"
#include "mipx/io.h"
#include "mipx/lp_problem.h"
#include "mipx/separators.h"
#include "mipx/symmetry.h"
#include "test_paths.hpp"

#include <catch2/catch_test_macros.hpp>
#include <string>
#include <vector>

using namespace mipx;
using mipx::test::testDataDir;

namespace {

// min -5 (x1 + x2 + x3 + x4)  s.t.  3 (x1 + x2 + x3 + x4) <= 7, x binary:
// four interchangeable items, any three of which are a cover.
LpProblem readKnapsack(const std::string& file) {
    return readMps(testDataDir() + "/" + file);
}

CutFamilyConfig onlyFamily(CutFamily family) {
    CutFamilyConfig config;
    config.gomory = false;
    config.mir = false;
    config.cover = family == CutFamily::Cover;
    config.implied_bound = false;
    config.clique = false;
    config.zero_half = false;
    config.mixing = false;
    config.cmir = false;
    config.strong_cg = false;
    config.lifted_cover = family == CutFamily::LiftedCover;
    config.mod_k = false;
    config.intersection_cut = false;
    config.multi_row = false;
    return config;
}

struct Separated {
    Int accepted = 0;
    std::vector<std::vector<Index>> supports;
};

Separated separateAtRoot(const LpProblem& problem, CutFamily family) {
    DualSimplexSolver lp;
    lp.setVerbose(false);
    lp.load(problem);
    REQUIRE(lp.solve().status == Status::Optimal);

    SeparatorManager manager;
    manager.setConfig(onlyFamily(family));
    manager.setMaxCutsPerFamily(10);
    CutPool pool;
    CutSeparationStats stats;
    // An optimal vertex of the LP, chosen so that the cover {x1, x2, x3} the
    // separators try first is violated; the solver's own vertex depends on
    // its pivoting, which may leave x4 the fractional one.
    const std::vector<Real> vertex = {1.0, 1.0, 1.0 / 3.0, 0.0};
    manager.separate(lp, problem, vertex, pool, stats);

    Separated out;
    out.accepted = stats.at(family).accepted;
    for (Index i = 0; i < pool.size(); ++i) {
        CHECK(pool[i].family == family);
        out.supports.push_back(pool[i].indices);
    }
    return out;
}

}  // namespace

TEST_CASE("Binary columns: the MARKER knapsack reads as Integer over [0, 1]", "[binary_columns]") {
    const auto marker = readKnapsack("knapsack4_marker.mps");
    const auto bv = readKnapsack("knapsack4_bv.mps");
    REQUIRE(marker.num_cols == 4);
    REQUIRE(bv.num_cols == 4);
    for (Index j = 0; j < 4; ++j) {
        CHECK(marker.col_type[j] == VarType::Integer);
        CHECK(bv.col_type[j] == VarType::Binary);
        CHECK(isBinaryColumn(marker, j));
        CHECK(isBinaryColumn(bv, j));
        CHECK(binaryAwareType(marker, j) == VarType::Binary);
    }
}

TEST_CASE("Binary columns: cover cuts fire on the MARKER knapsack as on the BV one",
          "[binary_columns][cuts]") {
    for (const CutFamily family : {CutFamily::Cover, CutFamily::LiftedCover}) {
        CAPTURE(static_cast<int>(family));
        const auto bv = separateAtRoot(readKnapsack("knapsack4_bv.mps"), family);
        const auto marker = separateAtRoot(readKnapsack("knapsack4_marker.mps"), family);
        CHECK(bv.accepted >= 1);
        CHECK(marker.accepted == bv.accepted);
        CHECK(marker.supports == bv.supports);
    }
}

TEST_CASE("Binary columns: symmetry breaking on the MARKER knapsack matches the BV one",
          "[binary_columns][symmetry]") {
    const auto bv = readKnapsack("knapsack4_bv.mps");
    const auto marker = readKnapsack("knapsack4_marker.mps");

    SymmetryManager bv_sym;
    bv_sym.detectFull(bv);
    SymmetryManager marker_sym;
    marker_sym.detectFull(marker);
    REQUIRE(bv_sym.hasSymmetry());
    CHECK(marker_sym.orbits() == bv_sym.orbits());
    REQUIRE(bv_sym.numGenerators() > 0);
    CHECK(marker_sym.numGenerators() == bv_sym.numGenerators());

    // Binary generators get symresacks and the four-column orbit an
    // orbitope; read as general integers they would get lex-leaders only.
    LpProblem bv_work = bv;
    LpProblem marker_work = marker;
    const Index bv_added = bv_sym.addSymbreakConstraints(bv_work);
    const Index marker_added = marker_sym.addSymbreakConstraints(marker_work);
    CHECK(bv_added > 0);
    CHECK(marker_added == bv_added);
    CHECK(marker_work.row_names == bv_work.row_names);
    bool has_orbitope = false;
    for (const auto& name : bv_work.row_names) {
        has_orbitope = has_orbitope || name.starts_with("orbitope_");
    }
    CHECK(has_orbitope);
}

TEST_CASE("Binary columns: BV and MARKER binaries share a symmetry orbit",
          "[binary_columns][symmetry]") {
    // Two of the items typed Binary, two Integer [0, 1]: the same 0/1
    // variables, so detection must not split them by declared type.
    auto mixed = readKnapsack("knapsack4_marker.mps");
    mixed.col_type[0] = VarType::Binary;
    mixed.col_type[2] = VarType::Binary;

    SymmetryManager by_signature;
    by_signature.detect(mixed);
    REQUIRE(by_signature.orbits().size() == 1);
    CHECK(by_signature.orbits()[0] == std::vector<Index>{0, 1, 2, 3});

    SymmetryManager full;
    full.detectFull(mixed);
    REQUIRE(full.orbits().size() == 1);
    CHECK(full.orbits()[0] == std::vector<Index>{0, 1, 2, 3});
}
