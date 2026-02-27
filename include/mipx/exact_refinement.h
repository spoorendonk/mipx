#pragma once

#include <span>
#include <vector>

#include "mipx/core.h"
#include "mipx/lp_problem.h"

namespace mipx {

struct LpCertificateMetrics {
    Int rows_evaluated = 0;
    Int cols_evaluated = 0;
    Real max_row_violation = 0.0;
    Real max_col_violation = 0.0;
    Real objective_mismatch = 0.0;
    Real recomputed_objective = 0.0;
    bool rational_supported = true;
    bool rational_ok = true;
};

/// Evaluate primal/certificate quality in long-double arithmetic.
LpCertificateMetrics evaluateLpCertificate(const LpProblem& problem,
                                           std::span<const Real> primals,
                                           Real reported_objective,
                                           bool rational_check,
                                           Real certificate_tol,
                                           Real rational_scale,
                                           double* work_units = nullptr);

/// Deterministic row/bound polishing pass for numerically noisy primals.
void iterativePrimalRepair(const LpProblem& problem,
                           std::vector<Real>& primals,
                           Real feasibility_tol,
                           Int max_passes,
                           double* work_units = nullptr);

}  // namespace mipx
