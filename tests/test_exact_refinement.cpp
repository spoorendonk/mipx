#include <catch2/catch_test_macros.hpp>

#include "mipx/exact_refinement.h"

using namespace mipx;

namespace {

LpProblem buildRefinementProbeLp() {
    LpProblem lp;
    lp.name = "exact_refine_probe";
    lp.sense = Sense::Minimize;
    lp.num_cols = 2;
    lp.obj = {1.0, 1.0};
    lp.col_lower = {0.0, 0.0};
    lp.col_upper = {1.0, 1.0};
    lp.col_type = {VarType::Continuous, VarType::Continuous};

    lp.num_rows = 1;
    lp.row_lower = {-kInf};
    lp.row_upper = {1.0};
    std::vector<Triplet> trips = {
        {0, 0, 1.0},
        {0, 1, 1.0},
    };
    lp.matrix = SparseMatrix(1, 2, std::move(trips));
    return lp;
}

}  // namespace

TEST_CASE("Exact refinement metrics detect row violations", "[exact_refinement]") {
    const auto lp = buildRefinementProbeLp();
    const std::vector<Real> x = {0.8, 0.7};
    const auto metrics = evaluateLpCertificate(
        lp, x, 1.5, false, 1e-8, 1e6, nullptr);

    CHECK(metrics.max_row_violation > 0.49);
    CHECK(metrics.max_col_violation < 1e-12);
}

TEST_CASE("Exact refinement primal repair reduces violation", "[exact_refinement]") {
    const auto lp = buildRefinementProbeLp();
    std::vector<Real> x = {0.8, 0.7};
    const auto before = evaluateLpCertificate(
        lp, x, 1.5, false, 1e-8, 1e6, nullptr);

    iterativePrimalRepair(lp, x, 1e-9, 4, nullptr);
    const auto after = evaluateLpCertificate(
        lp, x, x[0] + x[1], false, 1e-8, 1e6, nullptr);

    CHECK(after.max_row_violation <= before.max_row_violation + 1e-12);
    CHECK(after.max_row_violation < 1e-6);
}

TEST_CASE("Exact refinement scaled-rational verification can certify clean points",
          "[exact_refinement]") {
    const auto lp = buildRefinementProbeLp();
    const std::vector<Real> x = {0.25, 0.75};
    const auto metrics = evaluateLpCertificate(
        lp, x, 1.0, true, 1e-9, 1e6, nullptr);

    CHECK(metrics.rational_supported);
    CHECK(metrics.rational_ok);
    CHECK(metrics.objective_mismatch < 1e-12);
}

TEST_CASE("Exact refinement scaled-rational verification reports unsupported scale",
          "[exact_refinement]") {
    const auto lp = buildRefinementProbeLp();
    const std::vector<Real> x = {0.25, 0.75};
    const auto metrics = evaluateLpCertificate(
        lp, x, 1.0, true, 1e-9, 1e10, nullptr);

    CHECK_FALSE(metrics.rational_supported);
    CHECK_FALSE(metrics.rational_ok);
}
