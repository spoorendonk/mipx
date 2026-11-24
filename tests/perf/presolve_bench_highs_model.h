#pragma once

#include "mipx/lp_problem.h"
#include "presolve_bench_common.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <Highs.h>

namespace mipx::presolve_bench {

static_assert(sizeof(HighsInt) == sizeof(Index), "HiGHS built with 64-bit HighsInt");

// Widen HiGHS's limits so it takes the LP mipx read, as in mipx-lpseq-replay-highs.
// Refuse any coefficient HiGHS would still drop.
inline constexpr double kHighsSmallestEntry = 1e-12;

inline bool widenHighsLimits(Highs& highs) {
    return highs.setOptionValue("infinite_bound", kHighsInf) == HighsStatus::kOk &&
           highs.setOptionValue("infinite_cost", kHighsInf) == HighsStatus::kOk &&
           highs.setOptionValue("large_matrix_value", kHighsInf) == HighsStatus::kOk &&
           highs.setOptionValue("small_matrix_value", kHighsSmallestEntry) == HighsStatus::kOk;
}

inline HighsLp toHighs(const LpProblem& lp) {
    HighsLp h;
    h.num_col_ = lp.num_cols;
    h.num_row_ = lp.num_rows;
    h.sense_ = lp.sense == Sense::Maximize ? ObjSense::kMaximize : ObjSense::kMinimize;
    h.offset_ = lp.obj_offset;
    h.col_cost_ = lp.obj;
    h.col_lower_ = lp.col_lower;
    h.col_upper_ = lp.col_upper;
    h.row_lower_ = lp.row_lower;
    h.row_upper_ = lp.row_upper;
    // The bench uses an LP: readOrExit has already relaxed integrality.
    h.a_matrix_.format_ = MatrixFormat::kRowwise;
    h.a_matrix_.num_col_ = lp.num_cols;
    h.a_matrix_.num_row_ = lp.num_rows;
    const auto starts = lp.matrix.csr_row_starts();
    const auto cols = lp.matrix.csr_col_indices();
    const auto values = lp.matrix.csr_values();
    h.a_matrix_.start_.assign(starts.begin(), starts.end());
    h.a_matrix_.index_.assign(cols.begin(), cols.end());
    h.a_matrix_.value_.assign(values.begin(), values.end());
    if (h.a_matrix_.start_.empty()) {
        h.a_matrix_.start_.assign(1, 0);
    }
    for (const double v : h.a_matrix_.value_) {
        if (v != 0.0 && std::abs(v) <= kHighsSmallestEntry) {
            std::fprintf(stderr, "Error: a coefficient below 1e-12, which HiGHS drops\n");
            std::exit(kExitUsage);
        }
    }
    return h;
}

}  // namespace mipx::presolve_bench
