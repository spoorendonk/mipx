#include "mipx/separators.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "mipx/lp_problem.h"

namespace mipx {

namespace {

constexpr Real kCoeffTol = 1e-10;

}  // namespace

/// Mod-k separation: generalize zero-half cuts to arbitrary k.
/// For a row sum a_j x_j <= b with integer variables:
///   Scale by 1/k, round down coefficients: floor(a_j/k) x_j <= floor(b/k).
/// This generalizes zero-half (k=2) to k=3,4,5.
Int SeparatorManager::separateModK(const LpProblem& problem,
                                   std::span<const Real> primals,
                                   CutPool& pool,
                                   CutFamilyStats& stats) {
    Int accepted = 0;
    const std::vector<Int> k_values = {2, 3, 4, 5};

    for (Index i = 0; i < problem.num_rows && accepted < max_cuts_per_family_; ++i) {
        if (problem.row_upper[i] >= kInf) continue;
        const Real rhs = problem.row_upper[i];
        if (!std::isfinite(rhs) || rhs <= 0.0) continue;

        auto row = problem.matrix.row(i);
        if (row.size() < 1) continue;

        // Check nonnegativity.
        bool nonneg = true;
        bool has_integer = false;
        for (Index p = 0; p < row.size(); ++p) {
            const Index j = row.indices[p];
            if (row.values[p] < 0.0 || problem.col_lower[j] < -1e-12) {
                nonneg = false;
                break;
            }
            if (problem.col_type[j] != VarType::Continuous) has_integer = true;
        }
        if (!nonneg || !has_integer) continue;

        for (Int k : k_values) {
            if (accepted >= max_cuts_per_family_) break;
            ++stats.attempted;

            const Real inv_k = 1.0 / static_cast<Real>(k);
            const Real floored_rhs = std::floor(inv_k * rhs + 1e-9);

            Cut cut;
            cut.family = CutFamily::ModK;
            cut.lower = -kInf;
            cut.upper = floored_rhs;
            bool changed = (floored_rhs + 1e-9 < inv_k * rhs);

            for (Index p = 0; p < row.size(); ++p) {
                const Index j = row.indices[p];
                const Real a = row.values[p];
                if (problem.col_type[j] == VarType::Continuous) continue;
                const Real rounded = std::floor(inv_k * a + 1e-9);
                if (rounded <= 0.0) continue;
                if (rounded + 1e-9 < inv_k * a) changed = true;
                cut.indices.push_back(j);
                cut.values.push_back(rounded);
            }

            if (!changed || cut.indices.empty()) continue;

            // Compute violation.
            Real lhs = 0.0;
            for (Index idx = 0; idx < static_cast<Index>(cut.indices.size()); ++idx) {
                const Index j = cut.indices[idx];
                if (j >= 0 && j < static_cast<Index>(primals.size())) {
                    lhs += cut.values[idx] * primals[j];
                }
            }
            const Real violation = lhs - cut.upper;
            if (violation < min_violation_) continue;

            Real norm_sq = 0.0;
            for (Real v : cut.values) norm_sq += v * v;
            if (norm_sq < kCoeffTol) continue;
            cut.efficacy = violation / std::sqrt(norm_sq);
            if (!std::isfinite(cut.efficacy) || cut.efficacy <= 0.0) continue;

            ++stats.generated;
            if (pool.addCut(std::move(cut))) {
                ++stats.accepted;
                stats.efficacy_sum += violation / std::sqrt(norm_sq);
                ++accepted;
            }
        }
    }

    return accepted;
}

}  // namespace mipx
