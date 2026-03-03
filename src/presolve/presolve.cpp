#include "mipx/presolve.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <unordered_map>
#include <utility>

namespace mipx {

namespace {

inline void markRowsTouchingCol(const LpProblem& lp, Index col,
                                const std::vector<bool>& row_removed,
                                std::vector<uint8_t>& dirty_rows) {
    auto cv = lp.matrix.col(col);
    for (Index k = 0; k < cv.size(); ++k) {
        Index row = cv.indices[k];
        if (!row_removed[row]) dirty_rows[row] = 1;
    }
}

inline void markColsInRow(const LpProblem& lp, Index row,
                          const std::vector<bool>& col_removed,
                          std::vector<uint8_t>& dirty_cols) {
    auto rv = lp.matrix.row(row);
    for (Index k = 0; k < rv.size(); ++k) {
        Index col = rv.indices[k];
        if (!col_removed[col]) dirty_cols[col] = 1;
    }
}

inline uint64_t coeffBits(Real v) {
    static_assert(sizeof(uint64_t) == sizeof(Real));
    uint64_t bits = 0;
    std::memcpy(&bits, &v, sizeof(bits));
    return bits;
}

inline uint64_t mixHash(uint64_t h, uint64_t x) {
    // SplitMix64-style mix for sparse-pattern hashing.
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x ^= (x >> 31);
    return h ^ (x + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2));
}

inline uint64_t coeffKey(Index row, Index col) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(row)) << 32) |
           static_cast<uint32_t>(col);
}

inline Real effectiveCoeff(
    const std::unordered_map<uint64_t, Real>& coeff_overrides,
    Index row, Index col, Real matrix_value) {
    if (coeff_overrides.empty()) return matrix_value;
    const auto it = coeff_overrides.find(coeffKey(row, col));
    return (it != coeff_overrides.end()) ? it->second : matrix_value;
}

inline uint64_t rowPatternHash(const LpProblem& lp, Index row,
                               const std::vector<bool>& col_removed,
                               const std::unordered_map<uint64_t, Real>& coeff_overrides) {
    auto rv = lp.matrix.row(row);
    uint64_t h = 1469598103934665603ULL;
    Index active = 0;
    for (Index k = 0; k < rv.size(); ++k) {
        Index col = rv.indices[k];
        if (col_removed[col]) continue;
        h = mixHash(h, static_cast<uint64_t>(col));
        h = mixHash(h, coeffBits(effectiveCoeff(
            coeff_overrides, row, col, rv.values[k])));
        ++active;
    }
    return mixHash(h, static_cast<uint64_t>(active));
}

inline uint64_t rowSupportHash(const LpProblem& lp, Index row,
                               const std::vector<bool>& col_removed) {
    auto rv = lp.matrix.row(row);
    uint64_t h = 1469598103934665603ULL;
    Index active = 0;
    for (Index k = 0; k < rv.size(); ++k) {
        Index col = rv.indices[k];
        if (col_removed[col]) continue;
        h = mixHash(h, static_cast<uint64_t>(col));
        ++active;
    }
    return mixHash(h, static_cast<uint64_t>(active));
}

inline bool rowsHaveSamePattern(const LpProblem& lp, Index r1, Index r2,
                                const std::vector<bool>& col_removed,
                                const std::unordered_map<uint64_t, Real>& coeff_overrides,
                                Real tol) {
    auto a = lp.matrix.row(r1);
    auto b = lp.matrix.row(r2);
    Index i = 0;
    Index j = 0;
    while (true) {
        while (i < a.size() && col_removed[a.indices[i]]) ++i;
        while (j < b.size() && col_removed[b.indices[j]]) ++j;
        bool done_i = (i >= a.size());
        bool done_j = (j >= b.size());
        if (done_i || done_j) return done_i && done_j;
        if (a.indices[i] != b.indices[j]) return false;
        const Index col = a.indices[i];
        const Real ai = effectiveCoeff(coeff_overrides, r1, col, a.values[i]);
        const Real bj = effectiveCoeff(coeff_overrides, r2, col, b.values[j]);
        if (std::abs(ai - bj) > tol) return false;
        ++i;
        ++j;
    }
}

inline bool rowsHaveScaledPattern(const LpProblem& lp, Index r1, Index r2,
                                  const std::vector<bool>& col_removed,
                                  const std::unordered_map<uint64_t, Real>& coeff_overrides,
                                  Real tol, Real& scale_out) {
    auto a = lp.matrix.row(r1);
    auto b = lp.matrix.row(r2);
    Index i = 0;
    Index j = 0;
    bool have_scale = false;
    Real scale = 0.0;
    while (true) {
        while (i < a.size() && col_removed[a.indices[i]]) ++i;
        while (j < b.size() && col_removed[b.indices[j]]) ++j;
        bool done_i = (i >= a.size());
        bool done_j = (j >= b.size());
        if (done_i || done_j) {
            if (done_i && done_j && have_scale) {
                scale_out = scale;
                return true;
            }
            return false;
        }
        if (a.indices[i] != b.indices[j]) return false;

        const Index col = a.indices[i];
        const Real ai = effectiveCoeff(coeff_overrides, r1, col, a.values[i]);
        const Real bj = effectiveCoeff(coeff_overrides, r2, col, b.values[j]);
        if (std::abs(ai) <= tol && std::abs(bj) <= tol) {
            ++i;
            ++j;
            continue;
        }
        if (std::abs(ai) <= tol || std::abs(bj) <= tol) return false;
        if (!have_scale) {
            scale = bj / ai;
            if (std::abs(scale) <= tol) return false;
            have_scale = true;
        } else {
            const Real expected = scale * ai;
            if (std::abs(expected - bj) > tol * (1.0 + std::max(std::abs(expected), std::abs(bj)))) {
                return false;
            }
        }
        ++i;
        ++j;
    }
}

inline bool rowIntervalSubsumes(Real keep_lb, Real keep_ub,
                                Real rem_lb, Real rem_ub, Real tol) {
    // [keep_lb, keep_ub] subset [rem_lb, rem_ub] => row(rem) redundant.
    bool lb_ok = std::isinf(rem_lb) || keep_lb >= rem_lb - tol;
    bool ub_ok = std::isinf(rem_ub) || keep_ub <= rem_ub + tol;
    return lb_ok && ub_ok;
}

inline void removeActiveColumn(const LpProblem& lp, Index col,
                               std::vector<bool>& col_removed,
                               const std::vector<bool>& row_removed,
                               std::vector<Index>& row_active_nnz,
                               std::vector<Index>& col_active_nnz,
                               std::vector<uint8_t>& next_dirty_rows,
                               std::vector<uint8_t>& next_dirty_cols) {
    if (col_removed[col]) return;
    col_removed[col] = true;
    col_active_nnz[col] = 0;
    next_dirty_cols[col] = 1;

    auto cv = lp.matrix.col(col);
    for (Index k = 0; k < cv.size(); ++k) {
        Index row = cv.indices[k];
        if (row_removed[row]) continue;
        if (row_active_nnz[row] > 0) --row_active_nnz[row];
        next_dirty_rows[row] = 1;
    }
}

inline void removeActiveRow(const LpProblem& lp, Index row,
                            const std::vector<bool>& col_removed,
                            std::vector<bool>& row_removed,
                            std::vector<Index>& row_active_nnz,
                            std::vector<Index>& col_active_nnz,
                            std::vector<uint8_t>& next_dirty_rows,
                            std::vector<uint8_t>& next_dirty_cols) {
    if (row_removed[row]) return;
    row_removed[row] = true;
    row_active_nnz[row] = 0;
    next_dirty_rows[row] = 1;

    auto rv = lp.matrix.row(row);
    for (Index k = 0; k < rv.size(); ++k) {
        Index col = rv.indices[k];
        if (col_removed[col]) continue;
        if (col_active_nnz[col] > 0) --col_active_nnz[col];
        next_dirty_cols[col] = 1;
    }
}

}  // namespace

// =============================================================================
// PostsolveStack
// =============================================================================

void PostsolveStack::push(PostsolveOp op) {
    ops_.push_back(std::move(op));
}

void PostsolveStack::clear() {
    ops_.clear();
}

std::vector<Real> PostsolveStack::postsolve(
    const std::vector<Real>& presolved_solution,
    const std::vector<Index>& col_mapping,
    Index orig_num_cols) const {

    constexpr Real kTol = 1e-8;

    // Start with a full-size solution initialized to zero.
    std::vector<Real> full(orig_num_cols, 0.0);

    // Copy presolved solution values into their original positions.
    for (Index j = 0; j < static_cast<Index>(presolved_solution.size()); ++j) {
        if (j < static_cast<Index>(col_mapping.size())) {
            full[col_mapping[j]] = presolved_solution[j];
        }
    }

    // Apply postsolve operations in reverse order.
    for (auto it = ops_.rbegin(); it != ops_.rend(); ++it) {
        std::visit([&](const auto& op) {
            using T = std::decay_t<decltype(op)>;

            if constexpr (std::is_same_v<T, PostsolveFixVariable>) {
                full[op.orig_col] = op.value;
            }
            else if constexpr (std::is_same_v<T, PostsolveSingletonRow>) {
                // Nothing to do: the row was redundant after bound tightening.
            }
            else if constexpr (std::is_same_v<T, PostsolveSingletonCol>) {
                // The variable was substituted out of its single constraint.
                // We need to compute its value from the constraint.
                // The constraint is: row_lower <= a * x_col + (rest) <= row_upper.
                // rest = sum of other variables in the row (already set).
                // We don't store the other coefficients, so we set the variable
                // to satisfy the constraint as tightly as possible.
                //
                // For a singleton column, x_col only appeared in one row, so
                // its value is determined by the constraint and bounds.
                // Set it to be within bounds and satisfy the constraint.
                //
                // Since we removed the variable and adjusted the row to account
                // for it, we need to back-compute. The simplest correct approach:
                // set it to 0 if feasible, otherwise to the nearest bound.
                // But actually, the presolver adjusted the row bounds when removing
                // the column, so the constraint is satisfied by the remaining vars.
                // We just need to set this variable's value to satisfy its bounds
                // and any objective contribution.
                //
                // For correctness: the value was set based on objective and bounds
                // during presolve. We'll use the value that was implicitly chosen.
                // Actually, singleton column removal fixes the variable at a bound.
                // Let's use the value stored during presolve.
                // (The presolve sets the var to a specific value and stores it.)
                // The value is already set by PostsolveFixVariable that accompanies this.
                // This op mainly records the row/col relationship for reference.
                // But if we need to compute from scratch:
                Real val = full[op.orig_col];
                // Clamp to bounds.
                val = std::max(val, op.col_lower);
                val = std::min(val, op.col_upper);
                full[op.orig_col] = val;
            }
            else if constexpr (std::is_same_v<T, PostsolveForcingRow>) {
                for (const auto& fv : op.fixed_vars) {
                    full[fv.orig_col] = fv.value;
                }
            }
            else if constexpr (std::is_same_v<T, PostsolveDominatedRow>) {
                // Nothing to do: the row was redundant.
            }
            else if constexpr (std::is_same_v<T, PostsolveCoeffTightening>) {
                // Nothing to do for variable values.
                // The solution is still valid.
            }
            else if constexpr (std::is_same_v<T, PostsolveDoubletonEquality>) {
                const Real denom = op.a_eliminated;
                if (std::abs(denom) <= kTol) return;
                const Real x_keep = full[op.kept_col];
                Real x_elim = (op.rhs - op.a_kept * x_keep) / denom;
                if (!std::isinf(op.eliminated_lower)) {
                    x_elim = std::max(x_elim, op.eliminated_lower);
                }
                if (!std::isinf(op.eliminated_upper)) {
                    x_elim = std::min(x_elim, op.eliminated_upper);
                }
                full[op.eliminated_col] = x_elim;
            }
        }, *it);
    }

    return full;
}

// =============================================================================
// Presolver — individual reductions
// =============================================================================

Index Presolver::removeFixedVariables(LpProblem& lp, std::vector<bool>& col_removed,
                                       std::vector<bool>& row_removed,
                                       std::vector<Index>& row_active_nnz,
                                       std::vector<Index>& col_active_nnz,
                                       const std::vector<Index>& dirty_cols,
                                       std::vector<uint8_t>& next_dirty_rows,
                                       std::vector<uint8_t>& next_dirty_cols) {
    Index changes = 0;

    for (Index j : dirty_cols) {
        if (col_removed[j]) continue;
        ++stats_.cols_examined;
        if (std::abs(lp.col_lower[j] - lp.col_upper[j]) > kTol) continue;

        // Variable is fixed.
        Real value = lp.col_lower[j];
        removeActiveColumn(lp, j, col_removed, row_removed, row_active_nnz,
                           col_active_nnz, next_dirty_rows, next_dirty_cols);
        ++changes;
        ++stats_.vars_removed;

        // Adjust constraint bounds: for each row containing this variable,
        // subtract the fixed contribution.
        auto cv = lp.matrix.col(j);
        for (Index k = 0; k < cv.size(); ++k) {
            Index row = cv.indices[k];
            if (row_removed[row]) continue;
            Real a = effectiveCoeff(coeff_overrides_, row, j, cv.values[k]);
            Real shift = a * value;
            if (!std::isinf(lp.row_lower[row])) lp.row_lower[row] -= shift;
            if (!std::isinf(lp.row_upper[row])) lp.row_upper[row] -= shift;
            next_dirty_rows[row] = 1;
        }

        // Adjust objective offset.
        lp.obj_offset += lp.obj[j] * value;

        postsolve_stack_.push(PostsolveFixVariable{j, value});
    }

    return changes;
}

Index Presolver::removeSingletonRows(LpProblem& lp, std::vector<bool>& col_removed,
                                      std::vector<bool>& row_removed,
                                      std::vector<Index>& row_active_nnz,
                                      std::vector<Index>& col_active_nnz,
                                      const std::vector<Index>& dirty_rows,
                                      std::vector<uint8_t>& next_dirty_rows,
                                      std::vector<uint8_t>& next_dirty_cols) {
    Index changes = 0;

    for (Index i : dirty_rows) {
        if (row_removed[i]) continue;
        ++stats_.rows_examined;

        // Fast path from maintained active nonzero counts.
        auto rv = lp.matrix.row(i);
        Index count = row_active_nnz[i];
        Index singleton_col = -1;
        Real singleton_coeff = 0.0;

        if (count == 0) {
            // Empty row. Check feasibility, then remove.
            if (lp.row_lower[i] > kTol || lp.row_upper[i] < -kTol) {
                infeasible_ = true;
                return changes;
            }
            removeActiveRow(lp, i, col_removed, row_removed, row_active_nnz,
                            col_active_nnz, next_dirty_rows, next_dirty_cols);
            ++changes;
            ++stats_.rows_removed;
            postsolve_stack_.push(PostsolveDominatedRow{i});
            continue;
        }

        if (count != 1) continue;

        for (Index k = 0; k < rv.size(); ++k) {
            if (!col_removed[rv.indices[k]]) {
                singleton_col = rv.indices[k];
                singleton_coeff = effectiveCoeff(
                    coeff_overrides_, i, singleton_col, rv.values[k]);
                break;
            }
        }

        // Singleton row: row_lower <= a * x_j <= row_upper.
        // This implies bounds on x_j.
        Real new_lb = -kInf;
        Real new_ub = kInf;

        if (singleton_coeff > 0) {
            if (!std::isinf(lp.row_lower[i]))
                new_lb = lp.row_lower[i] / singleton_coeff;
            if (!std::isinf(lp.row_upper[i]))
                new_ub = lp.row_upper[i] / singleton_coeff;
        } else {
            // Negative coefficient flips bounds.
            if (!std::isinf(lp.row_upper[i]))
                new_lb = lp.row_upper[i] / singleton_coeff;
            if (!std::isinf(lp.row_lower[i]))
                new_ub = lp.row_lower[i] / singleton_coeff;
        }

        bool tightened = false;
        if (new_lb > lp.col_lower[singleton_col] + kTol) {
            lp.col_lower[singleton_col] = new_lb;
            tightened = true;
            ++stats_.bounds_tightened;
            markRowsTouchingCol(lp, singleton_col, row_removed, next_dirty_rows);
            next_dirty_cols[singleton_col] = 1;
        }
        if (new_ub < lp.col_upper[singleton_col] - kTol) {
            lp.col_upper[singleton_col] = new_ub;
            tightened = true;
            ++stats_.bounds_tightened;
            markRowsTouchingCol(lp, singleton_col, row_removed, next_dirty_rows);
            next_dirty_cols[singleton_col] = 1;
        }

        // Integer rounding.
        if (lp.col_type[singleton_col] != VarType::Continuous) {
            Real rounded_lb = std::ceil(lp.col_lower[singleton_col] - kTol);
            Real rounded_ub = std::floor(lp.col_upper[singleton_col] + kTol);
            if (rounded_lb > lp.col_lower[singleton_col] + kTol) {
                lp.col_lower[singleton_col] = rounded_lb;
                tightened = true;
                markRowsTouchingCol(lp, singleton_col, row_removed, next_dirty_rows);
                next_dirty_cols[singleton_col] = 1;
            }
            if (rounded_ub < lp.col_upper[singleton_col] - kTol) {
                lp.col_upper[singleton_col] = rounded_ub;
                tightened = true;
                markRowsTouchingCol(lp, singleton_col, row_removed, next_dirty_rows);
                next_dirty_cols[singleton_col] = 1;
            }
        }

        removeActiveRow(lp, i, col_removed, row_removed, row_active_nnz,
                        col_active_nnz, next_dirty_rows, next_dirty_cols);
        ++changes;
        ++stats_.rows_removed;
        if (tightened) ++changes;

        postsolve_stack_.push(PostsolveSingletonRow{i});
    }

    return changes;
}

Index Presolver::removeSingletonCols(LpProblem& lp, std::vector<bool>& col_removed,
                                      std::vector<bool>& row_removed,
                                      std::vector<Index>& row_active_nnz,
                                      std::vector<Index>& col_active_nnz,
                                      const std::vector<Index>& dirty_cols,
                                      std::vector<uint8_t>& next_dirty_rows,
                                      std::vector<uint8_t>& next_dirty_cols) {
    Index changes = 0;

    for (Index j : dirty_cols) {
        if (col_removed[j]) continue;
        ++stats_.cols_examined;

        auto cv = lp.matrix.col(j);
        Index count = col_active_nnz[j];
        Index singleton_row = -1;
        Real singleton_coeff = 0.0;

        if (count != 1) continue;
        for (Index k = 0; k < cv.size(); ++k) {
            if (!row_removed[cv.indices[k]]) {
                singleton_row = cv.indices[k];
                singleton_coeff = effectiveCoeff(
                    coeff_overrides_, singleton_row, j, cv.values[k]);
                break;
            }
        }

        // Singleton column: variable appears in only one active constraint.
        // We only eliminate by bound fixing when we can prove objective-optimality
        // and preserve feasibility for remaining row variables.
        Real obj = lp.obj[j];

        bool lb_finite = !std::isinf(lp.col_lower[j]);
        bool ub_finite = !std::isinf(lp.col_upper[j]);

        if (!lb_finite && !ub_finite) {
            continue;  // Free variable — skip.
        }

        // Count remaining variables in the row (excluding this one).
        auto rv = lp.matrix.row(singleton_row);
        Index remaining_in_row = 0;
        for (Index k = 0; k < rv.size(); ++k) {
            if (rv.indices[k] != j && !col_removed[rv.indices[k]])
                ++remaining_in_row;
        }

        auto objective_prefers_lower = [&]() -> bool {
            if (lp.sense == Sense::Minimize) return obj > kTol;
            return obj < -kTol;
        };
        auto objective_prefers_upper = [&]() -> bool {
            if (lp.sense == Sense::Minimize) return obj < -kTol;
            return obj > kTol;
        };

        bool fix_at_lower = false;
        bool has_fix = false;

        if (remaining_in_row > 0) {
            const bool has_row_lower = !std::isinf(lp.row_lower[singleton_row]);
            const bool has_row_upper = !std::isinf(lp.row_upper[singleton_row]);

            // Ranged/equality rows cannot be safely relaxed by a one-sided fix.
            if (has_row_lower && has_row_upper) continue;

            bool required_lower = false;
            if (has_row_upper) {
                // <= row: only the activity-minimizing fix is safe for remaining vars.
                required_lower = singleton_coeff > 0.0;
            } else if (has_row_lower) {
                // >= row: only the activity-maximizing fix is safe.
                required_lower = singleton_coeff < 0.0;
            } else {
                // No active row bounds; choose purely by objective.
                if (objective_prefers_lower()) {
                    required_lower = true;
                } else if (objective_prefers_upper()) {
                    required_lower = false;
                } else {
                    required_lower = lb_finite;
                }
            }

            // Respect objective direction when it is nonzero.
            if (std::abs(obj) > kTol) {
                bool objective_lower = objective_prefers_lower();
                if (objective_lower != required_lower) continue;
            }

            if (required_lower) {
                if (!lb_finite) continue;
                fix_at_lower = true;
            } else {
                if (!ub_finite) continue;
                fix_at_lower = false;
            }
            has_fix = true;
        } else {
            // Row contains only this variable; objective-only fixing is safe.
            if (objective_prefers_lower()) {
                if (!lb_finite) continue;
                fix_at_lower = true;
            } else if (objective_prefers_upper()) {
                if (!ub_finite) continue;
                fix_at_lower = false;
            } else if (lb_finite) {
                fix_at_lower = true;
            } else if (ub_finite) {
                fix_at_lower = false;
            } else {
                continue;
            }
            has_fix = true;
        }
        if (!has_fix) continue;

        const Real fix_value = fix_at_lower ? lp.col_lower[j] : lp.col_upper[j];

        removeActiveColumn(lp, j, col_removed, row_removed, row_active_nnz,
                           col_active_nnz, next_dirty_rows, next_dirty_cols);
        ++changes;
        ++stats_.vars_removed;

        // Adjust row bounds.
        Real shift = singleton_coeff * fix_value;
        if (!std::isinf(lp.row_lower[singleton_row]))
            lp.row_lower[singleton_row] -= shift;
        if (!std::isinf(lp.row_upper[singleton_row]))
            lp.row_upper[singleton_row] -= shift;

        // Adjust objective.
        lp.obj_offset += obj * fix_value;

        postsolve_stack_.push(PostsolveSingletonCol{
            j, singleton_row, singleton_coeff, obj,
            lp.row_lower[singleton_row], lp.row_upper[singleton_row],
            lp.col_type[j], lp.col_lower[j], lp.col_upper[j]
        });
        postsolve_stack_.push(PostsolveFixVariable{j, fix_value});
    }

    return changes;
}

Index Presolver::aggregateDoubletonEqualities(
    LpProblem& lp, std::vector<bool>& col_removed, std::vector<bool>& row_removed,
    std::vector<Index>& row_active_nnz, std::vector<Index>& col_active_nnz,
    const std::vector<Index>& dirty_rows, std::vector<uint8_t>& next_dirty_rows,
    std::vector<uint8_t>& next_dirty_cols) {
    Index changes = 0;

    auto findRowCoeff = [&](Index row, Index col, Real& coeff, Real& matrix_coeff) -> bool {
        auto rv = lp.matrix.row(row);
        for (Index k = 0; k < rv.size(); ++k) {
            if (rv.indices[k] != col) continue;
            matrix_coeff = rv.values[k];
            coeff = effectiveCoeff(coeff_overrides_, row, col, rv.values[k]);
            return true;
        }
        return false;
    };

    for (Index i : dirty_rows) {
        if (row_removed[i]) continue;
        ++stats_.rows_examined;
        if (row_active_nnz[i] != 2) continue;
        if (std::isinf(lp.row_lower[i]) || std::isinf(lp.row_upper[i])) continue;
        if (std::abs(lp.row_upper[i] - lp.row_lower[i]) > kTol) continue;

        auto rv = lp.matrix.row(i);
        std::array<Index, 2> row_cols = {-1, -1};
        std::array<Real, 2> row_coeffs = {0.0, 0.0};
        Index n_active = 0;
        for (Index k = 0; k < rv.size(); ++k) {
            const Index col = rv.indices[k];
            if (col_removed[col]) continue;
            const Real a = effectiveCoeff(coeff_overrides_, i, col, rv.values[k]);
            if (std::abs(a) <= kTol) continue;
            if (n_active >= 2) {
                n_active = 3;
                break;
            }
            row_cols[n_active] = col;
            row_coeffs[n_active] = a;
            ++n_active;
        }
        if (n_active != 2) continue;

        const Index c0 = row_cols[0];
        const Index c1 = row_cols[1];
        const Real a0 = row_coeffs[0];
        const Real a1 = row_coeffs[1];
        if (lp.col_type[c0] != VarType::Continuous ||
            lp.col_type[c1] != VarType::Continuous) {
            continue;
        }

        auto candidateFeasible = [&](Index elim_col, Index keep_col, Real a_elim,
                                     Real a_keep, Index& score) -> bool {
            if (std::abs(a_elim) <= kTol || std::abs(a_keep) <= kTol) return false;
            score = col_active_nnz[elim_col];
            const Real ratio = a_keep / a_elim;

            auto cv = lp.matrix.col(elim_col);
            for (Index k = 0; k < cv.size(); ++k) {
                const Index row = cv.indices[k];
                if (row_removed[row] || row == i) continue;

                const Real a_row_elim =
                    effectiveCoeff(coeff_overrides_, row, elim_col, cv.values[k]);
                Real a_row_keep = 0.0;
                [[maybe_unused]] Real matrix_keep = 0.0;
                if (!findRowCoeff(row, keep_col, a_row_keep, matrix_keep)) {
                    return false;  // Would create fill-in.
                }
                if (std::abs(a_row_keep) <= kTol) return false;
                const Real new_keep = a_row_keep - a_row_elim * ratio;
                if (std::abs(new_keep) <= 10.0 * kTol) return false;
            }
            return true;
        };

        bool have_choice = false;
        Index best_elim = -1;
        Index best_keep = -1;
        Real best_a_elim = 0.0;
        Real best_a_keep = 0.0;
        Index best_score = 0;

        for (int pass = 0; pass < 2; ++pass) {
            const Index elim_col = (pass == 0) ? c0 : c1;
            const Index keep_col = (pass == 0) ? c1 : c0;
            const Real a_elim = (pass == 0) ? a0 : a1;
            const Real a_keep = (pass == 0) ? a1 : a0;
            Index score = 0;
            if (!candidateFeasible(elim_col, keep_col, a_elim, a_keep, score)) continue;
            if (!have_choice || score < best_score) {
                have_choice = true;
                best_elim = elim_col;
                best_keep = keep_col;
                best_a_elim = a_elim;
                best_a_keep = a_keep;
                best_score = score;
            }
        }
        if (!have_choice) continue;

        const Real rhs = lp.row_upper[i];
        const Real alpha = rhs / best_a_elim;
        const Real ratio = best_a_keep / best_a_elim;
        const Real beta = -ratio;
        if (!std::isfinite(alpha) || !std::isfinite(beta) || !std::isfinite(ratio)) {
            continue;
        }
        const Real elim_lb = lp.col_lower[best_elim];
        const Real elim_ub = lp.col_upper[best_elim];

        struct RowUpdate {
            Index row = -1;
            Real shift = 0.0;
            Real new_keep = 0.0;
            Real matrix_keep = 0.0;
        };
        std::vector<RowUpdate> updates;
        bool valid_updates = true;

        // Substitute eliminated variable into objective: x_e = alpha + beta*x_k.
        auto cv = lp.matrix.col(best_elim);
        for (Index kk = 0; kk < cv.size(); ++kk) {
            const Index row = cv.indices[kk];
            if (row_removed[row] || row == i) continue;

            const Real a_row_elim =
                effectiveCoeff(coeff_overrides_, row, best_elim, cv.values[kk]);
            const Real shift = a_row_elim * alpha;
            if (!std::isinf(lp.row_lower[row]) && !std::isfinite(shift)) {
                valid_updates = false;
                break;
            }
            if (!std::isinf(lp.row_upper[row]) && !std::isfinite(shift)) {
                valid_updates = false;
                break;
            }

            Real a_row_keep = 0.0;
            Real matrix_keep = 0.0;
            const bool has_keep = findRowCoeff(row, best_keep, a_row_keep, matrix_keep);
            if (!has_keep) {
                valid_updates = false;
                break;
            }
            const Real new_keep = a_row_keep - a_row_elim * ratio;
            if (!std::isfinite(new_keep)) {
                valid_updates = false;
                break;
            }
            updates.push_back({row, shift, new_keep, matrix_keep});
        }
        if (!valid_updates) continue;

        lp.obj_offset += lp.obj[best_elim] * alpha;
        lp.obj[best_keep] += lp.obj[best_elim] * beta;
        lp.obj[best_elim] = 0.0;
        next_dirty_cols[best_keep] = 1;

        for (const auto& update : updates) {
            if (!std::isinf(lp.row_lower[update.row])) lp.row_lower[update.row] -= update.shift;
            if (!std::isinf(lp.row_upper[update.row])) lp.row_upper[update.row] -= update.shift;
            const uint64_t key = coeffKey(update.row, best_keep);
            if (std::abs(update.new_keep - update.matrix_keep) <= kTol) {
                coeff_overrides_.erase(key);
            } else {
                coeff_overrides_[key] = update.new_keep;
            }
            next_dirty_rows[update.row] = 1;
            markColsInRow(lp, update.row, col_removed, next_dirty_cols);
        }

        // Project eliminated variable bounds into kept variable bounds.
        bool keep_bound_tightened = false;
        auto tightenKeptBound = [&](Real bound, bool eliminated_lower) {
            if (std::isinf(bound) || std::abs(beta) <= kTol) return;
            const Real implied = (bound - alpha) / beta;
            if (beta > 0.0) {
                if (eliminated_lower) {
                    if (implied > lp.col_lower[best_keep] + kTol) {
                        lp.col_lower[best_keep] = implied;
                        keep_bound_tightened = true;
                        ++stats_.bounds_tightened;
                    }
                } else {
                    if (implied < lp.col_upper[best_keep] - kTol) {
                        lp.col_upper[best_keep] = implied;
                        keep_bound_tightened = true;
                        ++stats_.bounds_tightened;
                    }
                }
            } else {
                if (eliminated_lower) {
                    if (implied < lp.col_upper[best_keep] - kTol) {
                        lp.col_upper[best_keep] = implied;
                        keep_bound_tightened = true;
                        ++stats_.bounds_tightened;
                    }
                } else {
                    if (implied > lp.col_lower[best_keep] + kTol) {
                        lp.col_lower[best_keep] = implied;
                        keep_bound_tightened = true;
                        ++stats_.bounds_tightened;
                    }
                }
            }
        };
        tightenKeptBound(elim_lb, true);
        tightenKeptBound(elim_ub, false);

        if (lp.col_lower[best_keep] > lp.col_upper[best_keep] + kTol) {
            infeasible_ = true;
            return changes;
        }
        if (keep_bound_tightened) {
            markRowsTouchingCol(lp, best_keep, row_removed, next_dirty_rows);
            next_dirty_cols[best_keep] = 1;
        }

        removeActiveColumn(lp, best_elim, col_removed, row_removed, row_active_nnz,
                           col_active_nnz, next_dirty_rows, next_dirty_cols);
        removeActiveRow(lp, i, col_removed, row_removed, row_active_nnz,
                        col_active_nnz, next_dirty_rows, next_dirty_cols);

        // Clean up stale coeff_overrides_ entries for the eliminated column.
        std::erase_if(coeff_overrides_, [best_elim](const auto& entry) {
            return static_cast<Index>(entry.first & 0xFFFFFFFF) == best_elim;
        });

        ++stats_.vars_removed;
        ++stats_.rows_removed;
        ++changes;

        postsolve_stack_.push(PostsolveDoubletonEquality{
            best_elim, best_keep, best_a_elim, best_a_keep, rhs, elim_lb, elim_ub});
    }

    return changes;
}

Index Presolver::removeForcingRows(LpProblem& lp, std::vector<bool>& col_removed,
                                    std::vector<bool>& row_removed,
                                    std::vector<Index>& row_active_nnz,
                                    std::vector<Index>& col_active_nnz,
                                    const std::vector<Index>& dirty_rows,
                                    std::vector<uint8_t>& next_dirty_rows,
                                    std::vector<uint8_t>& next_dirty_cols) {
    Index changes = 0;

    for (Index i : dirty_rows) {
        if (row_removed[i]) continue;
        ++stats_.rows_examined;

        auto rv = lp.matrix.row(i);

        // Compute min/max activity for this row.
        Real act_min = 0.0;
        Real act_max = 0.0;
        bool has_inf_min = false;
        bool has_inf_max = false;

        for (Index k = 0; k < rv.size(); ++k) {
            Index j = rv.indices[k];
            if (col_removed[j]) continue;

            Real a = effectiveCoeff(coeff_overrides_, i, j, rv.values[k]);
            Real lb = lp.col_lower[j];
            Real ub = lp.col_upper[j];

            if (a > 0) {
                if (std::isinf(lb)) has_inf_min = true;
                else act_min += a * lb;
                if (std::isinf(ub)) has_inf_max = true;
                else act_max += a * ub;
            } else {
                if (std::isinf(ub)) has_inf_min = true;
                else act_min += a * ub;
                if (std::isinf(lb)) has_inf_max = true;
                else act_max += a * lb;
            }
        }

        if (!has_inf_min && !std::isinf(lp.row_upper[i]) &&
            act_min > lp.row_upper[i] + kTol) {
            infeasible_ = true;
            return changes;
        }
        if (!has_inf_max && !std::isinf(lp.row_lower[i]) &&
            act_max < lp.row_lower[i] - kTol) {
            infeasible_ = true;
            return changes;
        }

        // <= row is forcing when even the minimum activity is at the upper limit.
        bool force_to_min = !has_inf_min && !std::isinf(lp.row_upper[i]) &&
                            std::abs(act_min - lp.row_upper[i]) <= kTol &&
                            (std::isinf(lp.row_lower[i]) ||
                             act_min >= lp.row_lower[i] - kTol);

        // >= row is forcing when even the maximum activity is at the lower limit.
        bool force_to_max = !has_inf_max && !std::isinf(lp.row_lower[i]) &&
                            std::abs(act_max - lp.row_lower[i]) <= kTol &&
                            (std::isinf(lp.row_upper[i]) ||
                             act_max <= lp.row_upper[i] + kTol);

        if (force_to_min && force_to_max) {
            bool consistent = true;
            for (Index k = 0; k < rv.size(); ++k) {
                const Index j = rv.indices[k];
                if (col_removed[j]) continue;
                const Real a = effectiveCoeff(
                    coeff_overrides_, i, j, rv.values[k]);
                if (std::abs(a) <= kTol) continue;
                const Real min_fix = (a > 0.0) ? lp.col_lower[j] : lp.col_upper[j];
                const Real max_fix = (a > 0.0) ? lp.col_upper[j] : lp.col_lower[j];
                if (std::isinf(min_fix) || std::isinf(max_fix) ||
                    std::abs(min_fix - max_fix) > kTol) {
                    consistent = false;
                    break;
                }
            }
            if (!consistent) continue;
            force_to_max = false;
        }

        if (!force_to_min && !force_to_max) continue;

        std::vector<std::pair<Index, Real>> fixings;
        fixings.reserve(static_cast<size_t>(row_active_nnz[i]));
        bool can_fix_all = true;
        for (Index k = 0; k < rv.size(); ++k) {
            const Index j = rv.indices[k];
            if (col_removed[j]) continue;
            const Real a = effectiveCoeff(coeff_overrides_, i, j, rv.values[k]);
            if (std::abs(a) <= kTol) continue;

            const Real fix_value = force_to_min
                ? ((a > 0.0) ? lp.col_lower[j] : lp.col_upper[j])
                : ((a > 0.0) ? lp.col_upper[j] : lp.col_lower[j]);
            if (std::isinf(fix_value)) {
                can_fix_all = false;
                break;
            }
            fixings.emplace_back(j, fix_value);
        }
        if (!can_fix_all) continue;

        PostsolveForcingRow postsolve_op;
        postsolve_op.orig_row = i;

        for (const auto& [j, fix_value] : fixings) {
            removeActiveColumn(lp, j, col_removed, row_removed, row_active_nnz,
                               col_active_nnz, next_dirty_rows, next_dirty_cols);
            ++stats_.vars_removed;
            lp.obj_offset += lp.obj[j] * fix_value;

            auto cv = lp.matrix.col(j);
            for (Index kk = 0; kk < cv.size(); ++kk) {
                Index other_row = cv.indices[kk];
                if (row_removed[other_row] || other_row == i) continue;
                const Real a = effectiveCoeff(
                    coeff_overrides_, other_row, j, cv.values[kk]);
                Real shift = a * fix_value;
                if (!std::isinf(lp.row_lower[other_row])) {
                    lp.row_lower[other_row] -= shift;
                }
                if (!std::isinf(lp.row_upper[other_row])) {
                    lp.row_upper[other_row] -= shift;
                }
                next_dirty_rows[other_row] = 1;
            }

            postsolve_op.fixed_vars.push_back({j, fix_value});
            postsolve_stack_.push(PostsolveFixVariable{j, fix_value});
        }

        removeActiveRow(lp, i, col_removed, row_removed, row_active_nnz,
                        col_active_nnz, next_dirty_rows, next_dirty_cols);
        ++stats_.rows_removed;
        ++changes;
        postsolve_stack_.push(std::move(postsolve_op));
    }

    return changes;
}

Index Presolver::removeDominatedRows(LpProblem& lp, std::vector<bool>& col_removed,
                                      std::vector<bool>& row_removed,
                                      std::vector<Index>& row_active_nnz,
                                      std::vector<Index>& col_active_nnz,
                                      const std::vector<Index>& dirty_rows,
                                      std::vector<uint8_t>& next_dirty_rows,
                                      std::vector<uint8_t>& next_dirty_cols) {
    Index changes = 0;

    for (Index i : dirty_rows) {
        if (row_removed[i]) continue;
        ++stats_.rows_examined;

        auto rv = lp.matrix.row(i);

        // Compute activity bounds.
        Real act_min = 0.0;
        Real act_max = 0.0;
        bool has_inf_min = false;
        bool has_inf_max = false;

        for (Index k = 0; k < rv.size(); ++k) {
            Index j = rv.indices[k];
            if (col_removed[j]) continue;

            Real a = effectiveCoeff(coeff_overrides_, i, j, rv.values[k]);
            Real lb = lp.col_lower[j];
            Real ub = lp.col_upper[j];

            if (a > 0) {
                if (std::isinf(lb)) has_inf_min = true;
                else act_min += a * lb;
                if (std::isinf(ub)) has_inf_max = true;
                else act_max += a * ub;
            } else {
                if (std::isinf(ub)) has_inf_min = true;
                else act_min += a * ub;
                if (std::isinf(lb)) has_inf_max = true;
                else act_max += a * lb;
            }
        }

        // Row is redundant if activity bounds are within row bounds.
        bool lb_redundant = std::isinf(lp.row_lower[i]) ||
                             (!has_inf_min && act_min >= lp.row_lower[i] - kTol);
        bool ub_redundant = std::isinf(lp.row_upper[i]) ||
                             (!has_inf_max && act_max <= lp.row_upper[i] + kTol);

        if (lb_redundant && ub_redundant) {
            removeActiveRow(lp, i, col_removed, row_removed, row_active_nnz,
                            col_active_nnz, next_dirty_rows, next_dirty_cols);
            ++changes;
            ++stats_.rows_removed;
            postsolve_stack_.push(PostsolveDominatedRow{i});
        }
    }

    return changes;
}

Index Presolver::detectImpliedEquations(LpProblem& lp, std::vector<bool>& col_removed,
                                         std::vector<bool>& row_removed,
                                         const std::vector<Index>& dirty_rows,
                                         std::vector<uint8_t>& next_dirty_rows,
                                         std::vector<uint8_t>& next_dirty_cols) {
    Index changes = 0;

    for (Index i : dirty_rows) {
        if (row_removed[i]) continue;
        ++stats_.rows_examined;
        if (std::abs(lp.row_upper[i] - lp.row_lower[i]) <= kTol) continue;

        auto rv = lp.matrix.row(i);
        Real act_min = 0.0;
        Real act_max = 0.0;
        bool has_inf_min = false;
        bool has_inf_max = false;

        for (Index k = 0; k < rv.size(); ++k) {
            Index j = rv.indices[k];
            if (col_removed[j]) continue;

            Real a = effectiveCoeff(coeff_overrides_, i, j, rv.values[k]);
            Real lb = lp.col_lower[j];
            Real ub = lp.col_upper[j];
            if (a > 0) {
                if (std::isinf(lb)) has_inf_min = true;
                else act_min += a * lb;
                if (std::isinf(ub)) has_inf_max = true;
                else act_max += a * ub;
            } else {
                if (std::isinf(ub)) has_inf_min = true;
                else act_min += a * ub;
                if (std::isinf(lb)) has_inf_max = true;
                else act_max += a * lb;
            }
        }

        if (!std::isinf(lp.row_upper[i]) && !has_inf_min &&
            std::abs(act_min - lp.row_upper[i]) <= kTol) {
            lp.row_lower[i] = lp.row_upper[i];
            ++changes;
            next_dirty_rows[i] = 1;
            markColsInRow(lp, i, col_removed, next_dirty_cols);
        } else if (!std::isinf(lp.row_lower[i]) && !has_inf_max &&
                   std::abs(act_max - lp.row_lower[i]) <= kTol) {
            lp.row_upper[i] = lp.row_lower[i];
            ++changes;
            next_dirty_rows[i] = 1;
            markColsInRow(lp, i, col_removed, next_dirty_cols);
        }
    }

    return changes;
}

Index Presolver::activityBoundTightening(LpProblem& lp, std::vector<bool>& col_removed,
                                          std::vector<bool>& row_removed,
                                          const std::vector<Index>& dirty_rows,
                                          std::vector<uint8_t>& next_dirty_rows,
                                          std::vector<uint8_t>& next_dirty_cols) {
    Index changes = 0;

    for (Index i : dirty_rows) {
        if (row_removed[i]) continue;
        ++stats_.rows_examined;

        auto rv = lp.matrix.row(i);
        Real act_min = 0.0;
        Real act_max = 0.0;
        Index inf_min_count = 0;
        Index inf_max_count = 0;
        Index inf_min_col = -1;
        Index inf_max_col = -1;

        for (Index k = 0; k < rv.size(); ++k) {
            Index j = rv.indices[k];
            if (col_removed[j]) continue;
            Real a = effectiveCoeff(coeff_overrides_, i, j, rv.values[k]);

            Real contrib_min = (a > 0) ? a * lp.col_lower[j] : a * lp.col_upper[j];
            Real contrib_max = (a > 0) ? a * lp.col_upper[j] : a * lp.col_lower[j];

            if (std::isinf(contrib_min)) {
                ++inf_min_count;
                inf_min_col = j;
            } else {
                act_min += contrib_min;
            }
            if (std::isinf(contrib_max)) {
                ++inf_max_count;
                inf_max_col = j;
            } else {
                act_max += contrib_max;
            }
        }

        for (Index k = 0; k < rv.size(); ++k) {
            Index j = rv.indices[k];
            if (col_removed[j]) continue;
            Real a = effectiveCoeff(coeff_overrides_, i, j, rv.values[k]);

            bool res_min_finite = false;
            bool res_max_finite = false;
            Real res_min = -kInf;
            Real res_max = kInf;

            if (inf_min_count == 0) {
                res_min_finite = true;
                Real contrib_min_j = (a > 0) ? a * lp.col_lower[j] : a * lp.col_upper[j];
                res_min = act_min - contrib_min_j;
            } else if (inf_min_count == 1 && inf_min_col == j) {
                res_min_finite = true;
                res_min = act_min;
            }

            if (inf_max_count == 0) {
                res_max_finite = true;
                Real contrib_max_j = (a > 0) ? a * lp.col_upper[j] : a * lp.col_lower[j];
                res_max = act_max - contrib_max_j;
            } else if (inf_max_count == 1 && inf_max_col == j) {
                res_max_finite = true;
                res_max = act_max;
            }

            bool tightened = false;

            if (a > 0 && !std::isinf(lp.row_upper[i]) && res_min_finite) {
                Real new_ub = (lp.row_upper[i] - res_min) / a;
                if (new_ub < lp.col_upper[j] - kTol) {
                    lp.col_upper[j] = new_ub;
                    tightened = true;
                }
            } else if (a < 0 && !std::isinf(lp.row_lower[i]) && res_max_finite) {
                Real new_ub = (lp.row_lower[i] - res_max) / a;
                if (new_ub < lp.col_upper[j] - kTol) {
                    lp.col_upper[j] = new_ub;
                    tightened = true;
                }
            }

            if (a > 0 && !std::isinf(lp.row_lower[i]) && res_max_finite) {
                Real new_lb = (lp.row_lower[i] - res_max) / a;
                if (new_lb > lp.col_lower[j] + kTol) {
                    lp.col_lower[j] = new_lb;
                    tightened = true;
                }
            } else if (a < 0 && !std::isinf(lp.row_upper[i]) && res_min_finite) {
                Real new_lb = (lp.row_upper[i] - res_min) / a;
                if (new_lb > lp.col_lower[j] + kTol) {
                    lp.col_lower[j] = new_lb;
                    tightened = true;
                }
            }

            if (tightened && lp.col_type[j] != VarType::Continuous) {
                lp.col_lower[j] = std::ceil(lp.col_lower[j] - kTol);
                lp.col_upper[j] = std::floor(lp.col_upper[j] + kTol);
            }

            if (lp.col_lower[j] > lp.col_upper[j] + kTol) {
                infeasible_ = true;
                return changes;
            }

            if (tightened) {
                ++changes;
                ++stats_.bounds_tightened;
                markRowsTouchingCol(lp, j, row_removed, next_dirty_rows);
                next_dirty_cols[j] = 1;
            }
        }
    }

    return changes;
}

Index Presolver::dualFixing(LpProblem& lp, std::vector<bool>& col_removed,
                             std::vector<bool>& row_removed,
                             std::vector<Index>& row_active_nnz,
                             std::vector<Index>& col_active_nnz,
                             const std::vector<Index>& dirty_cols,
                             std::vector<uint8_t>& next_dirty_rows,
                             std::vector<uint8_t>& next_dirty_cols) {
    Index changes = 0;

    for (Index j : dirty_cols) {
        if (col_removed[j]) continue;
        ++stats_.cols_examined;

        Index up_locks = 0;
        Index down_locks = 0;
        Index active_nnz = col_active_nnz[j];
        if (active_nnz == 0) continue;  // Handle empty columns in removeEmptyColumns().
        auto cv = lp.matrix.col(j);
        for (Index k = 0; k < cv.size(); ++k) {
            Index row = cv.indices[k];
            if (row_removed[row]) continue;
            Real a = effectiveCoeff(coeff_overrides_, row, j, cv.values[k]);
            bool has_upper = !std::isinf(lp.row_upper[row]);
            bool has_lower = !std::isinf(lp.row_lower[row]);

            if (has_upper) {
                if (a > 0) ++up_locks;
                if (a < 0) ++down_locks;
            }
            if (has_lower) {
                if (a < 0) ++up_locks;
                if (a > 0) ++down_locks;
            }
        }

        bool prefer_down;
        if (lp.sense == Sense::Minimize) {
            prefer_down = (lp.obj[j] >= 0.0);
        } else {
            prefer_down = (lp.obj[j] <= 0.0);
        }

        bool fixed = false;
        Real fix_value = 0.0;
        if (prefer_down && down_locks == 0 && !std::isinf(lp.col_lower[j])) {
            fixed = true;
            fix_value = lp.col_lower[j];
        } else if (!prefer_down && up_locks == 0 && !std::isinf(lp.col_upper[j])) {
            fixed = true;
            fix_value = lp.col_upper[j];
        }
        if (!fixed) continue;

        removeActiveColumn(lp, j, col_removed, row_removed, row_active_nnz,
                           col_active_nnz, next_dirty_rows, next_dirty_cols);
        ++changes;
        ++stats_.vars_removed;
        lp.obj_offset += lp.obj[j] * fix_value;
        postsolve_stack_.push(PostsolveFixVariable{j, fix_value});

        for (Index kk = 0; kk < cv.size(); ++kk) {
            Index row = cv.indices[kk];
            if (row_removed[row]) continue;
            const Real a = effectiveCoeff(
                coeff_overrides_, row, j, cv.values[kk]);
            Real shift = a * fix_value;
            if (!std::isinf(lp.row_lower[row])) lp.row_lower[row] -= shift;
            if (!std::isinf(lp.row_upper[row])) lp.row_upper[row] -= shift;
            next_dirty_rows[row] = 1;
        }
    }

    return changes;
}

Index Presolver::removeEmptyColumns(LpProblem& lp, std::vector<bool>& col_removed,
                                     std::vector<bool>& row_removed,
                                     std::vector<Index>& row_active_nnz,
                                     std::vector<Index>& col_active_nnz,
                                     const std::vector<Index>& dirty_cols,
                                     std::vector<uint8_t>& next_dirty_rows,
                                     std::vector<uint8_t>& next_dirty_cols) {
    Index changes = 0;

    for (Index j : dirty_cols) {
        if (col_removed[j]) continue;
        ++stats_.cols_examined;
        if (lp.col_lower[j] > lp.col_upper[j] + kTol) {
            infeasible_ = true;
            return changes;
        }

        if (col_active_nnz[j] != 0) continue;

        bool lb_finite = !std::isinf(lp.col_lower[j]);
        bool ub_finite = !std::isinf(lp.col_upper[j]);
        Real fix_value = 0.0;
        if (lp.sense == Sense::Minimize) {
            if (lp.obj[j] > kTol) {
                if (!lb_finite) continue;  // Keep variable; objective can be unbounded below.
                fix_value = lp.col_lower[j];
            } else if (lp.obj[j] < -kTol) {
                if (!ub_finite) continue;  // Keep variable; objective can be unbounded below.
                fix_value = lp.col_upper[j];
            } else {
                if (!lb_finite && !ub_finite) {
                    fix_value = 0.0;
                } else {
                    fix_value = lb_finite ? lp.col_lower[j] : lp.col_upper[j];
                }
            }
        } else {
            if (lp.obj[j] > kTol) {
                if (!ub_finite) continue;  // Keep variable; objective can be unbounded above.
                fix_value = lp.col_upper[j];
            } else if (lp.obj[j] < -kTol) {
                if (!lb_finite) continue;  // Keep variable; objective can be unbounded above.
                fix_value = lp.col_lower[j];
            } else {
                if (!lb_finite && !ub_finite) {
                    fix_value = 0.0;
                } else {
                    fix_value = lb_finite ? lp.col_lower[j] : lp.col_upper[j];
                }
            }
        }

        removeActiveColumn(lp, j, col_removed, row_removed, row_active_nnz,
                           col_active_nnz, next_dirty_rows, next_dirty_cols);
        ++changes;
        ++stats_.vars_removed;
        lp.obj_offset += lp.obj[j] * fix_value;
        postsolve_stack_.push(PostsolveFixVariable{j, fix_value});
    }

    return changes;
}

Index Presolver::removeDuplicateRows(LpProblem& lp, std::vector<bool>& col_removed,
                                      std::vector<bool>& row_removed,
                                      std::vector<Index>& row_active_nnz,
                                      std::vector<Index>& col_active_nnz,
                                      const std::vector<Index>& dirty_rows,
                                      std::vector<uint8_t>& next_dirty_rows,
                                      std::vector<uint8_t>& next_dirty_cols) {
    Index changes = 0;
    std::unordered_map<uint64_t, std::vector<Index>> buckets;
    buckets.reserve(static_cast<size_t>(lp.num_rows));

    for (Index i : dirty_rows) {
        if (row_removed[i]) continue;
        ++stats_.rows_examined;
        uint64_t h = rowPatternHash(lp, i, col_removed, coeff_overrides_);
        buckets[h].push_back(i);
    }

    for (const auto& bucket : buckets) {
        const auto& rows = bucket.second;
        if (rows.size() < 2) continue;

        for (size_t a = 0; a < rows.size(); ++a) {
            Index i = rows[a];
            if (row_removed[i]) continue;
            for (size_t b = a + 1; b < rows.size(); ++b) {
                Index j = rows[b];
                if (row_removed[j]) continue;
                if (!rowsHaveSamePattern(lp, i, j, col_removed,
                                         coeff_overrides_, kTol)) continue;

                bool i_subsumes_j = rowIntervalSubsumes(lp.row_lower[i], lp.row_upper[i],
                                                        lp.row_lower[j], lp.row_upper[j], kTol);
                bool j_subsumes_i = rowIntervalSubsumes(lp.row_lower[j], lp.row_upper[j],
                                                        lp.row_lower[i], lp.row_upper[i], kTol);

                if (i_subsumes_j && !j_subsumes_i) {
                    removeActiveRow(lp, j, col_removed, row_removed, row_active_nnz,
                                    col_active_nnz, next_dirty_rows, next_dirty_cols);
                    ++changes;
                    ++stats_.rows_removed;
                    postsolve_stack_.push(PostsolveDominatedRow{j});
                } else if (j_subsumes_i && !i_subsumes_j) {
                    removeActiveRow(lp, i, col_removed, row_removed, row_active_nnz,
                                    col_active_nnz, next_dirty_rows, next_dirty_cols);
                    ++changes;
                    ++stats_.rows_removed;
                    postsolve_stack_.push(PostsolveDominatedRow{i});
                    break;
                } else if (i_subsumes_j && j_subsumes_i) {
                    // Exact duplicate row intervals: drop the later one.
                    removeActiveRow(lp, j, col_removed, row_removed, row_active_nnz,
                                    col_active_nnz, next_dirty_rows, next_dirty_cols);
                    ++changes;
                    ++stats_.rows_removed;
                    postsolve_stack_.push(PostsolveDominatedRow{j});
                }
            }
        }
    }

    return changes;
}

Index Presolver::removeParallelRows(LpProblem& lp, std::vector<bool>& col_removed,
                                     std::vector<bool>& row_removed,
                                     std::vector<Index>& row_active_nnz,
                                     std::vector<Index>& col_active_nnz,
                                     const std::vector<Index>& dirty_rows,
                                     std::vector<uint8_t>& next_dirty_rows,
                                     std::vector<uint8_t>& next_dirty_cols) {
    Index changes = 0;
    std::unordered_map<uint64_t, std::vector<Index>> buckets;
    buckets.reserve(static_cast<size_t>(lp.num_rows));

    for (Index i : dirty_rows) {
        if (row_removed[i]) continue;
        ++stats_.rows_examined;
        const uint64_t h = rowSupportHash(lp, i, col_removed);
        buckets[h].push_back(i);
    }

    auto scaleInterval = [](Real lower, Real upper, Real scale) -> std::pair<Real, Real> {
        if (scale > 0.0) return {lower * scale, upper * scale};
        return {upper * scale, lower * scale};
    };

    for (const auto& bucket : buckets) {
        const auto& rows = bucket.second;
        if (rows.size() < 2) continue;
        for (size_t a = 0; a < rows.size(); ++a) {
            const Index i = rows[a];
            if (row_removed[i]) continue;
            for (size_t b = a + 1; b < rows.size(); ++b) {
                const Index j = rows[b];
                if (row_removed[j]) continue;

                Real scale = 0.0;
                if (!rowsHaveScaledPattern(lp, i, j, col_removed, coeff_overrides_,
                                           kTol, scale)) {
                    continue;
                }
                if (std::abs(scale) <= kTol) continue;
                if (!std::isfinite(scale)) continue;

                const auto [i_to_j_lb, i_to_j_ub] =
                    scaleInterval(lp.row_lower[i], lp.row_upper[i], scale);
                const auto [j_to_i_lb, j_to_i_ub] =
                    scaleInterval(lp.row_lower[j], lp.row_upper[j], 1.0 / scale);

                const bool i_subsumes_j = rowIntervalSubsumes(
                    i_to_j_lb, i_to_j_ub, lp.row_lower[j], lp.row_upper[j], kTol);
                const bool j_subsumes_i = rowIntervalSubsumes(
                    j_to_i_lb, j_to_i_ub, lp.row_lower[i], lp.row_upper[i], kTol);

                if (i_subsumes_j && !j_subsumes_i) {
                    removeActiveRow(lp, j, col_removed, row_removed, row_active_nnz,
                                    col_active_nnz, next_dirty_rows, next_dirty_cols);
                    ++changes;
                    ++stats_.rows_removed;
                    postsolve_stack_.push(PostsolveDominatedRow{j});
                } else if (j_subsumes_i && !i_subsumes_j) {
                    removeActiveRow(lp, i, col_removed, row_removed, row_active_nnz,
                                    col_active_nnz, next_dirty_rows, next_dirty_cols);
                    ++changes;
                    ++stats_.rows_removed;
                    postsolve_stack_.push(PostsolveDominatedRow{i});
                    break;
                } else if (i_subsumes_j && j_subsumes_i) {
                    removeActiveRow(lp, j, col_removed, row_removed, row_active_nnz,
                                    col_active_nnz, next_dirty_rows, next_dirty_cols);
                    ++changes;
                    ++stats_.rows_removed;
                    postsolve_stack_.push(PostsolveDominatedRow{j});
                }
            }
        }
    }

    return changes;
}

Index Presolver::tightenCoefficients(LpProblem& lp, std::vector<bool>& col_removed,
                                      std::vector<bool>& row_removed,
                                      const std::vector<Index>& dirty_rows,
                                      std::vector<uint8_t>& next_dirty_rows,
                                      std::vector<uint8_t>& next_dirty_cols) {
    Index changes = 0;

    for (Index i : dirty_rows) {
        if (row_removed[i]) continue;
        auto rv = lp.matrix.row(i);

        auto tightenOneSided = [&](bool use_upper_side) {
            Real rhs = 0.0;
            if (use_upper_side) {
                if (std::isinf(lp.row_upper[i])) return;
                rhs = lp.row_upper[i];
            } else {
                if (!std::isinf(lp.row_upper[i])) return;  // Keep transformed logic one-sided.
                if (std::isinf(lp.row_lower[i])) return;
                rhs = -lp.row_lower[i];
            }

            // Compute max activity in transformed <= form.
            Real total_max = 0.0;
            bool has_inf = false;
            for (Index k = 0; k < rv.size(); ++k) {
                const Index j = rv.indices[k];
                if (col_removed[j]) continue;

                const Real a_orig = effectiveCoeff(coeff_overrides_, i, j, rv.values[k]);
                const Real a = use_upper_side ? a_orig : -a_orig;
                const Real lb = lp.col_lower[j];
                const Real ub = lp.col_upper[j];
                const Real contrib = (a > 0.0) ? a * ub : a * lb;
                if (std::isinf(contrib)) {
                    has_inf = true;
                    break;
                }
                total_max += contrib;
            }
            if (has_inf) return;

            for (Index k = 0; k < rv.size(); ++k) {
                const Index j = rv.indices[k];
                if (col_removed[j]) continue;
                if (lp.col_type[j] == VarType::Continuous) continue;

                const uint64_t key = coeffKey(i, j);
                const Real a_orig = effectiveCoeff(coeff_overrides_, i, j, rv.values[k]);
                const Real a = use_upper_side ? a_orig : -a_orig;
                if (a <= 0.0) continue;

                const Real ub = lp.col_upper[j];
                const Real lb = lp.col_lower[j];
                if (std::isinf(ub) || lb < -kTol) continue;

                const Real max_without_j = total_max - a * ub;
                const Real max_contrib = rhs - max_without_j;
                if (!(max_contrib < a - kTol && max_contrib > kTol)) continue;

                const Real new_a = max_contrib;
                const Real rhs_adjust = (a - new_a) * lb;
                const Real new_rhs = rhs - rhs_adjust;
                const Real new_a_orig = use_upper_side ? new_a : -new_a;
                if (!std::isfinite(new_rhs) || !std::isfinite(new_a_orig)) continue;
                const Real old_rhs_bound = use_upper_side ? lp.row_upper[i] : lp.row_lower[i];
                const Real new_rhs_bound = use_upper_side ? new_rhs : -new_rhs;

                postsolve_stack_.push(PostsolveCoeffTightening{
                    i, j, a_orig, new_a_orig, old_rhs_bound, new_rhs_bound
                });

                // Update total_max for subsequent variables in this row.
                total_max = total_max - a * ub + new_a * ub;
                rhs = new_rhs;
                if (use_upper_side) lp.row_upper[i] = new_rhs;
                else lp.row_lower[i] = -new_rhs;

                coeff_overrides_[key] = new_a_orig;
                // Coefficient changes must be visible to later passes.
                markRowsTouchingCol(lp, j, row_removed, next_dirty_rows);
                markColsInRow(lp, i, col_removed, next_dirty_cols);
                ++changes;
                ++stats_.coeffs_tightened;
            }
        };

        tightenOneSided(true);
        tightenOneSided(false);
    }

    return changes;
}

// =============================================================================
// Presolver — main entry point
// =============================================================================

LpProblem Presolver::presolve(const LpProblem& problem) {
    auto t0 = std::chrono::steady_clock::now();

    // Reset state so one Presolver instance can be reused safely.
    postsolve_stack_.clear();
    col_mapping_.clear();
    orig_num_cols_ = 0;
    stats_ = PresolveStats{};
    infeasible_ = false;
    coeff_overrides_.clear();

    // Work on a copy.
    LpProblem lp = problem;
    orig_num_cols_ = lp.num_cols;

    std::vector<bool> col_removed(lp.num_cols, false);
    std::vector<bool> row_removed(lp.num_rows, false);
    std::vector<Index> row_active_nnz(lp.num_rows, 0);
    std::vector<Index> col_active_nnz(lp.num_cols, 0);
    for (Index i = 0; i < lp.num_rows; ++i) {
        auto rv = lp.matrix.row(i);
        for (Index k = 0; k < rv.size(); ++k) {
            ++row_active_nnz[i];
            ++col_active_nnz[rv.indices[k]];
        }
    }
    std::vector<uint8_t> dirty_rows(lp.num_rows, 1);
    std::vector<uint8_t> dirty_cols(lp.num_cols, 1);

    // Iterative reduction loop.
    for (Index round = 0; round < max_rounds_; ++round) {
        Index total_changes = 0;
        std::vector<Index> dirty_row_list;
        std::vector<Index> dirty_col_list;
        dirty_row_list.reserve(lp.num_rows);
        dirty_col_list.reserve(lp.num_cols);
        for (Index i = 0; i < lp.num_rows; ++i) {
            if (dirty_rows[i]) dirty_row_list.push_back(i);
        }
        for (Index j = 0; j < lp.num_cols; ++j) {
            if (dirty_cols[j]) dirty_col_list.push_back(j);
        }
        if (dirty_row_list.empty() && dirty_col_list.empty()) break;

        std::vector<uint8_t> next_dirty_rows(lp.num_rows, 0);
        std::vector<uint8_t> next_dirty_cols(lp.num_cols, 0);

        Index ch = removeFixedVariables(lp, col_removed, row_removed,
                                        row_active_nnz, col_active_nnz,
                                        dirty_col_list, next_dirty_rows, next_dirty_cols);
        stats_.fixed_var_changes += ch;
        total_changes += ch;

        ch = removeSingletonRows(lp, col_removed, row_removed,
                                 row_active_nnz, col_active_nnz,
                                 dirty_row_list, next_dirty_rows, next_dirty_cols);
        stats_.singleton_row_changes += ch;
        total_changes += ch;

        ch = removeSingletonCols(lp, col_removed, row_removed,
                                 row_active_nnz, col_active_nnz,
                                 dirty_col_list, next_dirty_rows, next_dirty_cols);
        stats_.singleton_col_changes += ch;
        total_changes += ch;

        if (options_.enable_doubleton_aggregation) {
            ch = aggregateDoubletonEqualities(lp, col_removed, row_removed,
                                              row_active_nnz, col_active_nnz,
                                              dirty_row_list, next_dirty_rows,
                                              next_dirty_cols);
            stats_.doubleton_eq_changes += ch;
            total_changes += ch;
        }

        if (options_.enable_forcing_rows) {
            ch = removeForcingRows(lp, col_removed, row_removed,
                                   row_active_nnz, col_active_nnz,
                                   dirty_row_list, next_dirty_rows, next_dirty_cols);
            stats_.forcing_row_changes += ch;
            total_changes += ch;
        }

        ch = removeDominatedRows(lp, col_removed, row_removed,
                                 row_active_nnz, col_active_nnz,
                                 dirty_row_list, next_dirty_rows, next_dirty_cols);
        stats_.dominated_row_changes += ch;
        total_changes += ch;

        ch = removeDuplicateRows(lp, col_removed, row_removed,
                                 row_active_nnz, col_active_nnz,
                                 dirty_row_list, next_dirty_rows, next_dirty_cols);
        stats_.duplicate_row_changes += ch;
        total_changes += ch;

        if (options_.enable_parallel_rows) {
            ch = removeParallelRows(lp, col_removed, row_removed,
                                    row_active_nnz, col_active_nnz,
                                    dirty_row_list, next_dirty_rows, next_dirty_cols);
            stats_.parallel_row_changes += ch;
            total_changes += ch;
        }

        ch = detectImpliedEquations(lp, col_removed, row_removed,
                                    dirty_row_list, next_dirty_rows, next_dirty_cols);
        stats_.implied_equation_changes += ch;
        total_changes += ch;

        ch = activityBoundTightening(lp, col_removed, row_removed,
                                     dirty_row_list, next_dirty_rows, next_dirty_cols);
        stats_.activity_bound_tightening_changes += ch;
        total_changes += ch;

        if (options_.enable_dual_fixing) {
            ch = dualFixing(lp, col_removed, row_removed,
                            row_active_nnz, col_active_nnz,
                            dirty_col_list, next_dirty_rows, next_dirty_cols);
            stats_.dual_fixing_changes += ch;
            total_changes += ch;
        }

        ch = removeEmptyColumns(lp, col_removed, row_removed,
                                row_active_nnz, col_active_nnz,
                                dirty_col_list, next_dirty_rows, next_dirty_cols);
        stats_.empty_col_changes += ch;
        total_changes += ch;

        if (options_.enable_coefficient_tightening) {
            ch = tightenCoefficients(lp, col_removed, row_removed,
                                     dirty_row_list, next_dirty_rows, next_dirty_cols);
            stats_.coeff_tightening_changes += ch;
            total_changes += ch;
        }

        ++stats_.rounds;
        if (total_changes > 0) ++stats_.rounds_with_changes;

        // Check for infeasibility: any non-removed variable with lb > ub.
        for (Index j = 0; j < lp.num_cols; ++j) {
            if (!col_removed[j] && lp.col_lower[j] > lp.col_upper[j] + kTol) {
                infeasible_ = true;
                break;
            }
        }
        if (infeasible_) break;

        // Check for infeasible empty rows.
        for (Index i = 0; i < lp.num_rows; ++i) {
            if (row_removed[i]) continue;
            if (row_active_nnz[i] == 0) {
                // Empty row: check bounds.
                if (lp.row_lower[i] > kTol || lp.row_upper[i] < -kTol) {
                    infeasible_ = true;
                    break;
                }
            }
        }
        if (infeasible_) break;

        if (total_changes == 0) break;

        dirty_rows.swap(next_dirty_rows);
        dirty_cols.swap(next_dirty_cols);

        // Keep progress even when only removals happened.
        bool has_dirty = false;
        for (uint8_t f : dirty_rows) {
            if (f) {
                has_dirty = true;
                break;
            }
        }
        if (!has_dirty) {
            for (uint8_t f : dirty_cols) {
                if (f) {
                    has_dirty = true;
                    break;
                }
            }
        }
        if (!has_dirty) break;
    }

    stats_.time_seconds =
        std::chrono::duration<Real>(std::chrono::steady_clock::now() - t0).count();

    return buildReducedProblem(lp, col_removed, row_removed);
}

LpProblem Presolver::buildReducedProblem(const LpProblem& lp,
                                           const std::vector<bool>& col_removed,
                                           const std::vector<bool>& row_removed) {
    // Build column mapping: new index -> original index.
    col_mapping_.clear();
    std::vector<Index> orig_to_new(lp.num_cols, -1);

    for (Index j = 0; j < lp.num_cols; ++j) {
        if (!col_removed[j]) {
            orig_to_new[j] = static_cast<Index>(col_mapping_.size());
            col_mapping_.push_back(j);
        }
    }

    // Build row mapping.
    std::vector<Index> row_mapping;  // new -> orig
    std::vector<Index> orig_row_to_new(lp.num_rows, -1);
    for (Index i = 0; i < lp.num_rows; ++i) {
        if (!row_removed[i]) {
            orig_row_to_new[i] = static_cast<Index>(row_mapping.size());
            row_mapping.push_back(i);
        }
    }

    Index new_cols = static_cast<Index>(col_mapping_.size());
    Index new_rows = static_cast<Index>(row_mapping.size());

    LpProblem result;
    result.name = lp.name;
    result.sense = lp.sense;
    result.obj_offset = lp.obj_offset;
    result.num_cols = new_cols;
    result.num_rows = new_rows;

    // Copy column data.
    result.obj.resize(new_cols);
    result.col_lower.resize(new_cols);
    result.col_upper.resize(new_cols);
    result.col_type.resize(new_cols);
    result.col_names.resize(new_cols);

    for (Index jj = 0; jj < new_cols; ++jj) {
        Index j = col_mapping_[jj];
        result.obj[jj] = lp.obj[j];
        result.col_lower[jj] = lp.col_lower[j];
        result.col_upper[jj] = lp.col_upper[j];
        result.col_type[jj] = lp.col_type[j];
        if (j < static_cast<Index>(lp.col_names.size()))
            result.col_names[jj] = lp.col_names[j];
    }

    // Copy row data.
    result.row_lower.resize(new_rows);
    result.row_upper.resize(new_rows);
    result.row_names.resize(new_rows);

    for (Index ii = 0; ii < new_rows; ++ii) {
        Index i = row_mapping[ii];
        result.row_lower[ii] = lp.row_lower[i];
        result.row_upper[ii] = lp.row_upper[i];
        if (i < static_cast<Index>(lp.row_names.size()))
            result.row_names[ii] = lp.row_names[i];
    }

    // Build new matrix.
    std::vector<Triplet> triplets;
    for (Index ii = 0; ii < new_rows; ++ii) {
        Index orig_row = row_mapping[ii];
        auto rv = lp.matrix.row(orig_row);
        for (Index k = 0; k < rv.size(); ++k) {
            Index orig_col = rv.indices[k];
            if (col_removed[orig_col]) continue;
            const Real value = effectiveCoeff(
                coeff_overrides_, orig_row, orig_col, rv.values[k]);
            triplets.push_back({ii, orig_to_new[orig_col], value});
        }
    }

    result.matrix = SparseMatrix(new_rows, new_cols, std::move(triplets));

    return result;
}

std::vector<Real> Presolver::postsolve(
    const std::vector<Real>& presolved_solution) const {
    return postsolve_stack_.postsolve(
        presolved_solution, col_mapping_, orig_num_cols_);
}

}  // namespace mipx
