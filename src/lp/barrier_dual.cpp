#include "mipx/barrier.h"

#include <cmath>
#include <utility>

namespace mipx {

std::optional<BarrierSolver::DualForm> BarrierSolver::buildDualForm(const LpProblem& problem) {
    constexpr Real kBoundTol = 1e-12;
    const Real sense = problem.sense == Sense::Minimize ? 1.0 : -1.0;
    DualForm form;
    form.offset = -sense * problem.obj_offset;
    form.primals.resize(static_cast<size_t>(problem.num_cols));
    form.row_prices.resize(static_cast<size_t>(problem.num_rows));
    std::vector<Triplet> trips;
    auto addCol = [&](Real cost) {
        const auto col = static_cast<Index>(form.cost.size());
        form.cost.push_back(cost);
        return col;
    };

    // In the minimization primal, write x = shift + sign*w. For a free x,
    // w is free and sign = 1. Eliminating fixed x leaves no dual constraint.
    // Each remaining variable gives a dual equation A'y - t + s = c in the
    // w coordinates. Here t prices a finite upper bound on w and s is the
    // slack for a nonnegative w. Neither is present for a free w.
    for (Index j = 0; j < problem.num_cols; ++j) {
        const Real lower = problem.col_lower[j];
        const Real upper = problem.col_upper[j];
        if (lower > upper + kBoundTol) {
            return std::nullopt;
        }
        const bool has_lower = std::isfinite(lower);
        const bool has_upper = std::isfinite(upper);
        const Real sign = !has_lower && has_upper ? -1.0 : 1.0;
        auto& primal = form.primals[j];
        primal.offset = has_lower ? lower : (has_upper ? upper : 0.0);
        const Real cost = sense * problem.obj[j];
        form.offset -= cost * primal.offset;
        if (has_lower && has_upper && upper <= lower) {
            continue;
        }

        const auto row = static_cast<Index>(form.rhs.size());
        form.rhs.push_back(sign * cost);
        // The IPM minimizes the negative of the dual objective, so its
        // equality multiplier is -w. Map back after unscaling that multiplier.
        primal.col_a = row;
        primal.coeff_a = -sign;
        if (has_lower && has_upper) {
            trips.push_back({row, addCol(upper - lower), -1.0});
        }
        if (has_lower || has_upper) {
            trips.push_back({row, addCol(0.0), 1.0});
        }
    }

    // A lower side contributes +p and an upper side -q to y. For a
    // proper range these are two nonnegative multipliers with distinct
    // costs; for an equality they split a single unrestricted multiplier.
    for (Index i = 0; i < problem.num_rows; ++i) {
        const Real lower = problem.row_lower[i];
        const Real upper = problem.row_upper[i];
        const bool has_lower = std::isfinite(lower);
        const bool has_upper = std::isfinite(upper);
        if (!has_lower && !has_upper) {
            continue;
        }
        if (lower > upper + kBoundTol) {
            return std::nullopt;
        }

        const auto row = problem.matrix.row(i);
        Real shift = 0.0;
        bool constant = true;
        for (Index k = 0; k < row.size(); ++k) {
            const auto& primal = form.primals[row.indices[k]];
            shift += row.values[k] * primal.offset;
            constant &= primal.col_a < 0 || row.values[k] == 0.0;
        }
        if (constant) {
            if (shift < lower - 1e-9 || shift > upper + 1e-9) {
                return std::nullopt;
            }
            continue;
        }

        auto& price = form.row_prices[i];
        const bool equality = has_lower && has_upper && upper - lower <= kBoundTol;
        if (has_lower) {
            price.col_a = addCol(shift - (equality ? upper : lower));
            price.coeff_a = sense;
        }
        if (has_upper) {
            price.col_b = addCol(upper - shift);
            price.coeff_b = -sense;
        }
        for (Index k = 0; k < row.size(); ++k) {
            const auto& primal = form.primals[row.indices[k]];
            if (primal.col_a < 0 || row.values[k] == 0.0) {
                continue;
            }
            const Real coeff = -primal.coeff_a * row.values[k];
            if (has_lower) {
                trips.push_back({primal.col_a, price.col_a, coeff});
            }
            if (has_upper) {
                trips.push_back({primal.col_a, price.col_b, -coeff});
            }
        }
    }

    form.matrix = SparseMatrix(static_cast<Index>(form.rhs.size()),
                               static_cast<Index>(form.cost.size()), std::move(trips));
    return form;
}

void BarrierSolver::useDualForm(DualForm form) {
    aeq_ = std::move(form.matrix);
    beq_ = std::move(form.rhs);
    cstd_ = std::move(form.cost);
    ustd_.assign(cstd_.size(), kInf);
    std_obj_offset_ = form.offset;
    col_expr_ = std::move(form.primals);
    dual_row_expr_ = std::move(form.row_prices);
    dualized_ = true;
}

}  // namespace mipx
