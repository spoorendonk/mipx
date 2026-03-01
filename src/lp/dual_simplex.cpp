#include "mipx/dual_simplex.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#ifdef MIPX_HAS_TBB
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_sort.h>
#include <tbb/blocked_range.h>
#include <tbb/task_arena.h>
#endif

namespace mipx {

namespace {

inline bool canUseSimd(const DualSimplexOptions& options, Index len) {
    return options.enable_simd_kernels &&
           options.simd_min_length > 0 &&
           len >= options.simd_min_length;
}

inline void axpyInPlace(std::span<Real> dst, std::span<const Real> src, Real alpha,
                        const DualSimplexOptions& options) {
    assert(dst.size() == src.size());
    (void)options;
#if defined(__AVX2__)
    if (canUseSimd(options, static_cast<Index>(dst.size()))) {
        const __m256d a = _mm256_set1_pd(alpha);
        std::size_t i = 0;
        const std::size_t n = dst.size();
        for (; i + 4 <= n; i += 4) {
            __m256d d = _mm256_loadu_pd(dst.data() + i);
            __m256d s = _mm256_loadu_pd(src.data() + i);
#if defined(__FMA__)
            d = _mm256_fmadd_pd(a, s, d);
#else
            d = _mm256_add_pd(d, _mm256_mul_pd(a, s));
#endif
            _mm256_storeu_pd(dst.data() + i, d);
        }
        for (; i < n; ++i) {
            dst[i] += alpha * src[i];
        }
        return;
    }
#endif
    for (std::size_t i = 0; i < dst.size(); ++i) {
        dst[i] += alpha * src[i];
    }
}

inline void negateCopy(std::span<Real> dst, std::span<const Real> src,
                       const DualSimplexOptions& options) {
    assert(dst.size() == src.size());
    (void)options;
#if defined(__AVX2__)
    if (canUseSimd(options, static_cast<Index>(dst.size()))) {
        const __m256d z = _mm256_setzero_pd();
        std::size_t i = 0;
        const std::size_t n = dst.size();
        for (; i + 4 <= n; i += 4) {
            __m256d s = _mm256_loadu_pd(src.data() + i);
            __m256d d = _mm256_sub_pd(z, s);
            _mm256_storeu_pd(dst.data() + i, d);
        }
        for (; i < n; ++i) {
            dst[i] = -src[i];
        }
        return;
    }
#endif
    for (std::size_t i = 0; i < dst.size(); ++i) {
        dst[i] = -src[i];
    }
}

}  // namespace

// ---------------------------------------------------------------------------
//  Load
// ---------------------------------------------------------------------------

void DualSimplexSolver::load(const LpProblem& problem) {
    num_cols_ = problem.num_cols;
    num_rows_ = problem.num_rows;
    sense_ = problem.sense;
    obj_offset_ = problem.obj_offset;

    // Copy objective (negate for maximize).
    obj_ = problem.obj;
    if (sense_ == Sense::Maximize) {
        for (auto& c : obj_) c = -c;
    }

    col_lower_ = problem.col_lower;
    col_upper_ = problem.col_upper;
    row_lower_ = problem.row_lower;
    row_upper_ = problem.row_upper;
    matrix_ = problem.matrix;

    // Apply scaling.
    if (options_.enable_scaling) {
        computeScaling();
        applyScaling();
    } else {
        scaled_ = false;
        row_scale_.assign(num_rows_, 1.0);
        col_scale_.assign(num_cols_, 1.0);
    }

    // Build augmented matrix [A | I].
    buildAugmentedMatrix();

    status_ = Status::Error;
    iterations_ = 0;
    has_basis_ = false;
    nonbasic_pos_.clear();
    bound_perturb_active_ = false;
    bound_perturb_activations_ = 0;
    lower_bound_perturb_.clear();
    upper_bound_perturb_.clear();
    loaded_ = true;
}

// ---------------------------------------------------------------------------
//  Scaling: equilibration
// ---------------------------------------------------------------------------

void DualSimplexSolver::computeScaling() {
    row_scale_.assign(num_rows_, 1.0);
    col_scale_.assign(num_cols_, 1.0);

    // Row scaling: max absolute value in each row.
    for (Index i = 0; i < num_rows_; ++i) {
        auto rv = matrix_.row(i);
        Real maxval = 0.0;
        for (Index k = 0; k < rv.size(); ++k) {
            maxval = std::max(maxval, std::abs(rv.values[k]));
        }
        if (maxval > kZeroTol) {
            row_scale_[i] = 1.0 / maxval;
        }
    }

    // Column scaling: max absolute value in each column (after row scaling).
    for (Index j = 0; j < num_cols_; ++j) {
        auto cv = matrix_.col(j);
        Real maxval = 0.0;
        for (Index k = 0; k < cv.size(); ++k) {
            maxval = std::max(maxval, std::abs(cv.values[k] * row_scale_[cv.indices[k]]));
        }
        if (maxval > kZeroTol) {
            col_scale_[j] = 1.0 / maxval;
        }
    }

    scaled_ = true;
}

void DualSimplexSolver::applyScaling() {
    if (!scaled_) return;

    // Scale the matrix: A' = R * A * C where R = diag(row_scale_), C = diag(col_scale_).
    // We need to rebuild from triplets since we can't modify CSR in-place easily.
    std::vector<Triplet> triplets;
    triplets.reserve(matrix_.numNonzeros());
    for (Index i = 0; i < num_rows_; ++i) {
        auto rv = matrix_.row(i);
        for (Index k = 0; k < rv.size(); ++k) {
            Real val = rv.values[k] * row_scale_[i] * col_scale_[rv.indices[k]];
            if (std::abs(val) > kZeroTol) {
                triplets.push_back({i, rv.indices[k], val});
            }
        }
    }
    matrix_ = SparseMatrix(num_rows_, num_cols_, std::move(triplets));

    // Scale bounds: row bounds *= row_scale, col bounds /= col_scale (since x = C * x').
    for (Index i = 0; i < num_rows_; ++i) {
        if (row_lower_[i] != -kInf) row_lower_[i] *= row_scale_[i];
        if (row_upper_[i] != kInf) row_upper_[i] *= row_scale_[i];
    }
    for (Index j = 0; j < num_cols_; ++j) {
        // x_j = col_scale_[j] * x'_j, so x'_j = x_j / col_scale_[j]
        // bounds: l_j / col_scale_[j] <= x'_j <= u_j / col_scale_[j]
        // But col_scale_ = 1/maxval, so we multiply bounds by maxval = 1/col_scale_.
        // Actually: x = C*x', so l <= x <= u becomes l <= C*x' <= u
        // For component j: l_j <= col_scale_j * x'_j <= u_j
        // So: l_j / col_scale_j <= x'_j <= u_j / col_scale_j
        if (col_lower_[j] != -kInf) col_lower_[j] /= col_scale_[j];
        if (col_upper_[j] != kInf) col_upper_[j] /= col_scale_[j];
        // Objective: c^T x = c^T C x' = (C c)^T x'. So c'_j = c_j * col_scale_j.
        obj_[j] *= col_scale_[j];
    }
}

void DualSimplexSolver::unscaleResults() {
    if (!scaled_) return;

    // Unscale primal values: x_j = col_scale_j * x'_j for structural, row_scale_i * s'_i for slack.
    for (Index j = 0; j < num_cols_; ++j) {
        primal_[j] *= col_scale_[j];
    }
    for (Index i = 0; i < num_rows_; ++i) {
        primal_[num_cols_ + i] /= row_scale_[i];
    }

    // Unscale duals: y'_i was computed for scaled problem. Original y_i = y'_i * row_scale_i.
    for (Index i = 0; i < num_rows_; ++i) {
        dual_[i] *= row_scale_[i];
    }

    // Unscale reduced costs for structural vars.
    for (Index j = 0; j < num_cols_; ++j) {
        reduced_cost_[j] /= col_scale_[j];
    }
}

// ---------------------------------------------------------------------------
//  Build augmented matrix [A | I]
// ---------------------------------------------------------------------------

void DualSimplexSolver::buildAugmentedMatrix() {
    // Build [A | I] as a sparse matrix with num_rows rows and (num_cols + num_rows) cols.
    std::vector<Triplet> triplets;
    triplets.reserve(matrix_.numNonzeros() + num_rows_);

    for (Index i = 0; i < num_rows_; ++i) {
        auto rv = matrix_.row(i);
        for (Index k = 0; k < rv.size(); ++k) {
            triplets.push_back({i, rv.indices[k], rv.values[k]});
        }
        // Identity column for slack.
        triplets.push_back({i, num_cols_ + i, 1.0});
    }

    augmented_ = SparseMatrix(num_rows_, num_cols_ + num_rows_, std::move(triplets));
}

// ---------------------------------------------------------------------------
//  Variable bounds and costs
// ---------------------------------------------------------------------------

Real DualSimplexSolver::varLower(Index k) const {
    assert(k >= 0 && k < numVars());
    const std::size_t idx = static_cast<std::size_t>(k);
    Real lb = (k < num_cols_) ? col_lower_[k] : row_lower_[k - num_cols_];
    if (bound_perturb_active_ && lb != -kInf) {
        assert(lower_bound_perturb_.size() == static_cast<std::size_t>(numVars()));
        lb += lower_bound_perturb_[idx];
    }
    // Slack variable for row (k - num_cols_): s = Ax component.
    // Row i: row_lower_[i] <= a_i^T x <= row_upper_[i]
    // Slack s_i = a_i^T x, so row_lower_[i] <= s_i <= row_upper_[i].
    return lb;
}

Real DualSimplexSolver::varUpper(Index k) const {
    assert(k >= 0 && k < numVars());
    const std::size_t idx = static_cast<std::size_t>(k);
    Real ub = (k < num_cols_) ? col_upper_[k] : row_upper_[k - num_cols_];
    if (bound_perturb_active_ && ub != kInf) {
        assert(upper_bound_perturb_.size() == static_cast<std::size_t>(numVars()));
        ub -= upper_bound_perturb_[idx];
    }
    return ub;
}

Real DualSimplexSolver::varCost(Index k) const {
    if (k < num_cols_) return obj_[k];
    return 0.0;  // slacks have zero cost
}

// ---------------------------------------------------------------------------
//  Initial basis
// ---------------------------------------------------------------------------

void DualSimplexSolver::setupInitialBasis() {
    Index n = numVars();

    basis_.resize(num_rows_);
    nonbasic_.clear();
    nonbasic_.reserve(num_cols_);
    nonbasic_pos_.assign(n, -1);
    basis_pos_.assign(n, -1);
    var_status_.resize(n);
    primal_.resize(n);
    dual_.resize(num_rows_);
    reduced_cost_.resize(n);

    // Initialize Devex weights.
    devex_weights_.assign(num_rows_, 1.0);
    devex_reset_count_ = 0;

    // All slacks basic.
    for (Index i = 0; i < num_rows_; ++i) {
        Index slack = num_cols_ + i;
        basis_[i] = slack;
        basis_pos_[slack] = i;
        var_status_[slack] = BasisStatus::Basic;
    }

    // Structural variables nonbasic.
    // With all-slack basis, y = 0 (slack costs are 0), so reduced_cost_j = c_j.
    // For dual feasibility: rc_j >= 0 if at lower, rc_j <= 0 if at upper.
    for (Index j = 0; j < num_cols_; ++j) {
        Real lb = col_lower_[j];
        Real ub = col_upper_[j];
        Real cj = obj_[j];

        bool has_lower = (lb != -kInf);
        bool has_upper = (ub != kInf);

        if (has_lower && has_upper && std::abs(lb - ub) < kZeroTol) {
            // Fixed.
            var_status_[j] = BasisStatus::Fixed;
            primal_[j] = lb;
        } else if (!has_lower && !has_upper) {
            // Free variable. Place at 0 and apply cost shifting for dual feasibility.
            var_status_[j] = BasisStatus::Free;
            primal_[j] = 0.0;
        } else if (cj >= 0.0 && has_lower) {
            // rc = c_j >= 0, at lower is dual feasible.
            var_status_[j] = BasisStatus::AtLower;
            primal_[j] = lb;
        } else if (cj < 0.0 && has_upper) {
            // rc = c_j < 0, at upper is dual feasible.
            var_status_[j] = BasisStatus::AtUpper;
            primal_[j] = ub;
        } else if (has_lower) {
            // Only lower bound. rc = c_j must be >= 0 for dual feasibility.
            // If c_j < 0, apply cost shifting: shift c_j up to 0.
            var_status_[j] = BasisStatus::AtLower;
            primal_[j] = lb;
        } else {
            // Only upper bound.
            var_status_[j] = BasisStatus::AtUpper;
            primal_[j] = ub;
        }

        nonbasic_.push_back(j);
        nonbasic_pos_[j] = static_cast<Index>(nonbasic_.size() - 1);
    }
}

// ---------------------------------------------------------------------------
//  Compute primals: x_B = B^{-1} * (b - N * x_N)
// ---------------------------------------------------------------------------

void DualSimplexSolver::computePrimals() {
    // Compute rhs = b_eff - sum over nonbasic j: a_j * x_j
    // where b_eff for each row is the slack value contribution.
    // Actually: the system is A_aug * x = 0 style? No.
    // The augmented system: [A | I] * [x; s] has no explicit rhs.
    // The constraint is: A*x + s = something? No.
    // With slacks: row i is row_lower_i <= a_i^T x <= row_upper_i.
    // We define slack s_i = a_i^T x. So a_i^T x - s_i = 0, i.e., [A | -I][x; s] = 0.
    // But we used [A | I] for the augmented matrix. Let me reconsider.
    //
    // The augmented approach: we have variables x_1..x_n, s_1..s_m.
    // The basis equation: B * x_B = -N * x_N + some rhs?
    //
    // Actually, the standard approach: the constraints are encoded as
    // A*x = b has no single rhs. The row bounds define row_lower <= Ax <= row_upper.
    // The slack s_i represents the activity of row i. Constraint: s_i = a_i^T x.
    // Written as: A*x - s = 0, so [A | -I][x; s] = 0.
    // Total: [A | -I]*[x; s] = 0.
    //
    // With augmented = [A | I], we'd have [A|I][x; s] = 2*something... that's wrong.
    //
    // Let me use the standard formulation. The constraint system is:
    //   For each row: a_i^T x = s_i   (slack = activity)
    //   row_lower_i <= s_i <= row_upper_i
    //   col_lower_j <= x_j <= col_upper_j
    //
    // So the "constraint" for the basis is: A*x - s = 0, but we have [A | -I] as the
    // "augmented" matrix. However our augmented_ is [A | I].
    //
    // For the basis matrix approach: we pick m variables from {x_1,...,x_n, s_1,...,s_m}.
    // The equality is: sum over all vars k of a_augmented_col_k * var_k = 0
    //   where for structural vars, a_augmented_col_j = A's column j
    //         for slack s_i, a_augmented_col_{n+i} = -e_i (negative identity)
    //
    // BUT our augmented_ has +I for slacks. So the equation is:
    //   A*x + I*s = 0 would be wrong. We need A*x = s, i.e., A*x - s = 0.
    //
    // Two options: (1) use [A | -I] or (2) keep [A | I] and handle the sign.
    // Since SparseLU::factorize takes the matrix and basis_cols, and extracts those columns,
    // let's just use [A | -I]. Let me rebuild the augmented matrix.
    //
    // Actually wait. Let's think about it differently. With [A | I] and the equation
    // [A | I] * [x; s] = rhs, what is rhs? It's not 0.
    // We have a_i^T x = activity. If s_i is the slack = activity, then:
    //   a_i^T x + 0*s_i = activity = s_i... that's circular.
    //
    // The standard LP formulation is: the constraints A*x = b (equality form).
    // For ranged rows: we introduce slacks and have A*x + s = b_upper (or similar).
    //
    // The simplest correct approach: define the "activity" of row i as:
    //   act_i = a_i^T x
    // And we want row_lower_i <= act_i <= row_upper_i.
    //
    // We can add slack: act_i = a_i^T x, with row_lower_i <= act_i <= row_upper_i.
    // Treat act_i as a variable bounded by [row_lower_i, row_upper_i].
    //
    // The basis system: select m of the (n+m) variables. The linear system is:
    //   B * x_B = -N * x_N
    // where the "augmented" constraint is A_aug * x = 0, with A_aug being the m x (n+m) matrix.
    //
    // For structural column j: the j-th column of A_aug is a_j (the j-th column of A).
    // For slack column (n+i): the column should be -e_i (so that a_i^T x - s_i = 0).
    //
    // So A_aug = [A | -I]. Then A_aug * [x; s] = A*x - s = 0 means s = A*x. Correct.
    //
    // So the B*x_B + N*x_N = 0 decomposition gives x_B = -B^{-1} * N * x_N.
    // Actually: A_aug * x_all = 0 => B * x_B + N * x_N = 0 => x_B = -(B^{-1}) * N * x_N.
    //
    // Wait but B^{-1} * N * x_N involves the full N which is expensive. The standard
    // approach is to compute: x_B = B^{-1} * b where b = -N * x_N.
    //
    // For each nonbasic variable k with value x_k:
    //   b -= a_augmented_col_k * x_k
    //
    // Then x_B = B^{-1} * (-b) actually. Let me be more careful.
    //
    // A_aug * x = 0
    // Sum_{k in B} a_k * x_k + Sum_{k in N} a_k * x_k = 0
    // B * x_B = - Sum_{k in N} a_k * x_k
    // x_B = B^{-1} * (- Sum_{k in N} a_k * x_k)
    //
    // For structural var j nonbasic: a_k = column j of A (the j-th col of augmented_).
    //   Contribution = -a_j * x_j.
    // For slack var (n+i) nonbasic: a_k = -e_i (col n+i of A_aug = [A|-I]).
    //   Contribution = -(-e_i) * x_{n+i} = e_i * x_{n+i}.
    //
    // But we built augmented_ as [A | I], not [A | -I]. So col n+i of augmented_ is +e_i.
    // We need col n+i of A_aug to be -e_i.
    //
    // Fix: rebuild augmented as [A | -I], or handle the sign in the rhs computation.
    // Let me handle the sign:
    //   rhs_i = -Sum_{nonbasic structural j} A_{i,j} * x_j + Sum_{nonbasic slack (n+k)} x_{n+k} * delta_{i,k}
    // and for basis factorization, the slack basis columns also need -e_i.
    //
    // Actually, the simplest correct approach used in practice: don't use [A|I] at all.
    // Instead, the basis is always m columns from the "extended" matrix where:
    //  - Structural columns: columns of A
    //  - Slack columns: -e_i
    //
    // When factorizing, SparseLU::factorize takes a SparseMatrix and column indices.
    // So we need a matrix that has the right columns. If we use [A | -I], then
    // SparseLU::factorize(augmented_neg, basis_cols) works directly.
    //
    // Let me rebuild augmented_ as [A | -I].

    // rhs = - sum_{k in N} a_k * x_k
    // where a_k is column k of A_aug = [A | -I].
    std::vector<Real> rhs(num_rows_, 0.0);

    for (Index k : nonbasic_) {
        Real xk = primal_[k];
        if (std::abs(xk) < kZeroTol) continue;

        if (k < num_cols_) {
            // Structural: a_k = column k of A.
            auto cv = matrix_.col(k);
            for (Index p = 0; p < cv.size(); ++p) {
                rhs[cv.indices[p]] -= cv.values[p] * xk;
            }
        } else {
            // Slack (n+i): a_k = -e_i in [A|-I]. So contribution = -(-e_i)*x_k = e_i * x_k.
            Index i = k - num_cols_;
            rhs[i] += xk;
        }
    }

    // Solve B * x_B = rhs using FTRAN.
    lu_.ftran(rhs);

    // Assign basic variable values.
    for (Index i = 0; i < num_rows_; ++i) {
        primal_[basis_[i]] = rhs[i];
    }
}

// ---------------------------------------------------------------------------
//  Compute duals and reduced costs
// ---------------------------------------------------------------------------

void DualSimplexSolver::computeDuals() {
    // y = B^{-T} * c_B
    // Reduced cost: rc_k = c_k - y^T * a_k
    //   For structural k: rc_k = c_k - y^T * A_col_k
    //   For slack (n+i): rc_{n+i} = 0 - y^T * (-e_i) = y_i

    std::vector<Real> cb(num_rows_, 0.0);
    for (Index i = 0; i < num_rows_; ++i) {
        cb[i] = varCost(basis_[i]);
    }

    // BTRAN: y = B^{-T} * c_B
    lu_.btran(cb);
    dual_ = cb;

    // Reduced costs.
    for (Index k = 0; k < numVars(); ++k) {
        if (var_status_[k] == BasisStatus::Basic) {
            reduced_cost_[k] = 0.0;
        } else if (k < num_cols_) {
            // Structural: rc = c_k - y^T * A_col_k.
            Real rc = obj_[k];
            auto cv = matrix_.col(k);
            for (Index p = 0; p < cv.size(); ++p) {
                rc -= dual_[cv.indices[p]] * cv.values[p];
            }
            reduced_cost_[k] = rc;
        } else {
            // Slack (n+i): rc = 0 - y^T * (-e_i) = y_i.
            reduced_cost_[k] = dual_[k - num_cols_];
        }
    }
}

// ---------------------------------------------------------------------------
//  Refactorize
// ---------------------------------------------------------------------------

void DualSimplexSolver::refactorize() {
    // Build the basis matrix: for each basis position i, the column of [A | -I].
    // We need a SparseMatrix whose column j is the basis_[j] column of [A | -I].
    // SparseLU::factorize(matrix, basis_cols) extracts columns basis_cols[i] from matrix.
    // So we need augmented_ to be [A | -I] and pass basis_ as basis_cols.

    lu_.factorize(augmented_, basis_);
}

// ---------------------------------------------------------------------------
//  Solve
// ---------------------------------------------------------------------------

LpResult DualSimplexSolver::solve() {
    if (!loaded_) {
        return {Status::Error, 0.0, 0};
    }

    // Rebuild augmented matrix as [A | -I] for correct basis representation.
    {
        std::vector<Triplet> triplets;
        triplets.reserve(matrix_.numNonzeros() + num_rows_);
        for (Index i = 0; i < num_rows_; ++i) {
            auto rv = matrix_.row(i);
            for (Index k = 0; k < rv.size(); ++k) {
                triplets.push_back({i, rv.indices[k], rv.values[k]});
            }
            triplets.push_back({i, num_cols_ + i, -1.0});
        }
        augmented_ = SparseMatrix(num_rows_, num_cols_ + num_rows_, std::move(triplets));
    }

    // Reset iteration counter for this solve call.
    iterations_ = 0;
    solve_start_ = std::chrono::steady_clock::now();
    bound_perturb_active_ = false;
    bound_perturb_activations_ = 0;
    lower_bound_perturb_.clear();
    upper_bound_perturb_.clear();

    // Snapshot work at start of solve to compute delta.
    double work_at_start = work_.units();

    auto rowViolation = [&](Real activity, Real lower, Real upper) -> Real {
        if (activity < lower - kPrimalTol) return lower - activity;
        if (activity > upper + kPrimalTol) return activity - upper;
        return 0.0;
    };

    auto runStructuralCrash = [&]() {
        if (!options_.enable_structural_crash || options_.structural_crash_max_swaps <= 0) return;
        struct StructuralCand {
            Index col = -1;
            Index row = -1;
            Real score = 0.0;
        };
        std::vector<StructuralCand> candidates;
        candidates.reserve(static_cast<std::size_t>(num_cols_));
        for (Index j = 0; j < num_cols_; ++j) {
            if (nonbasic_pos_[j] < 0) continue;
            auto cv = matrix_.col(j);
            if (cv.size() <= 0) continue;

            Real sum_abs = 0.0;
            Real best_abs = 0.0;
            Index best_row = -1;
            for (Index p = 0; p < cv.size(); ++p) {
                const Real abs_val = std::abs(cv.values[p]);
                sum_abs += abs_val;
                if (abs_val > best_abs) {
                    best_abs = abs_val;
                    best_row = cv.indices[p];
                }
            }
            if (best_row < 0 || best_abs < options_.structural_crash_min_pivot) continue;
            if (basis_[best_row] < num_cols_) continue;  // keep already-structural rows untouched

            const Real off_abs = sum_abs - best_abs;
            // Prefer near-singleton columns to preserve a stable crash basis.
            if (off_abs > best_abs) continue;

            const Real score = best_abs / (1.0 + off_abs);
            candidates.push_back({j, best_row, score});
        }

        std::sort(candidates.begin(), candidates.end(),
                  [](const StructuralCand& a, const StructuralCand& b) {
                      if (a.score != b.score) return a.score > b.score;
                      if (a.row != b.row) return a.row < b.row;
                      return a.col < b.col;
                  });

        std::vector<uint8_t> used_row(static_cast<std::size_t>(num_rows_), 0);
        Int swaps = 0;
        for (const auto& cand : candidates) {
            if (swaps >= options_.structural_crash_max_swaps) break;
            if (used_row[cand.row] != 0) continue;
            if (nonbasic_pos_[cand.col] < 0) continue;

            const Index leaving = basis_[cand.row];
            if (leaving < num_cols_) continue;  // conservative: only swap out slacks

            const Index nb_pos = nonbasic_pos_[cand.col];
            if (nb_pos < 0) continue;

            basis_[cand.row] = cand.col;
            basis_pos_[cand.col] = cand.row;
            basis_pos_[leaving] = -1;
            var_status_[cand.col] = BasisStatus::Basic;

            const Real lb = varLower(leaving);
            const Real ub = varUpper(leaving);
            if (lb != -kInf && ub != kInf && std::abs(lb - ub) < kZeroTol) {
                var_status_[leaving] = BasisStatus::Fixed;
                primal_[leaving] = lb;
            } else if (lb != -kInf) {
                var_status_[leaving] = BasisStatus::AtLower;
                primal_[leaving] = lb;
            } else if (ub != kInf) {
                var_status_[leaving] = BasisStatus::AtUpper;
                primal_[leaving] = ub;
            } else {
                var_status_[leaving] = BasisStatus::Free;
                primal_[leaving] = 0.0;
            }

            nonbasic_[nb_pos] = leaving;
            nonbasic_pos_[leaving] = nb_pos;
            nonbasic_pos_[cand.col] = -1;
            used_row[cand.row] = 1;
            ++swaps;
        }
    };

    auto runIdiotCrash = [&]() {
        if (!options_.enable_idiot_crash || options_.idiot_crash_passes <= 0) return;

        std::vector<Real> row_activity(num_rows_, 0.0);
        for (Index j = 0; j < num_cols_; ++j) {
            Real xj = primal_[j];
            if (std::abs(xj) <= kZeroTol) continue;
            auto cv = matrix_.col(j);
            for (Index p = 0; p < cv.size(); ++p) {
                row_activity[cv.indices[p]] += cv.values[p] * xj;
            }
        }

        Int flips = 0;
        for (Index pass = 0; pass < options_.idiot_crash_passes; ++pass) {
            bool changed = false;
            for (Index j = 0; j < num_cols_; ++j) {
                if (nonbasic_pos_[j] < 0) continue;
                BasisStatus st = var_status_[j];
                if (st != BasisStatus::AtLower && st != BasisStatus::AtUpper) continue;

                Real lb = col_lower_[j];
                Real ub = col_upper_[j];
                if (lb == -kInf || ub == kInf) continue;
                if (std::abs(ub - lb) <= kZeroTol) continue;

                Real old_x = primal_[j];
                Real new_x = (st == BasisStatus::AtLower) ? ub : lb;
                Real delta = new_x - old_x;
                if (std::abs(delta) <= kZeroTol) continue;

                auto cv = matrix_.col(j);
                Real gain = 0.0;
                for (Index p = 0; p < cv.size(); ++p) {
                    Index i = cv.indices[p];
                    Real old_act = row_activity[i];
                    Real new_act = old_act + cv.values[p] * delta;
                    gain += rowViolation(old_act, row_lower_[i], row_upper_[i]) -
                            rowViolation(new_act, row_lower_[i], row_upper_[i]);
                }

                if (gain <= options_.idiot_crash_min_gain) continue;

                primal_[j] = new_x;
                var_status_[j] =
                    (st == BasisStatus::AtLower) ? BasisStatus::AtUpper : BasisStatus::AtLower;
                for (Index p = 0; p < cv.size(); ++p) {
                    row_activity[cv.indices[p]] += cv.values[p] * delta;
                }
                changed = true;
                ++flips;
                if (options_.idiot_crash_max_flips > 0 &&
                    flips >= options_.idiot_crash_max_flips) {
                    break;
                }
            }

            if (!changed ||
                (options_.idiot_crash_max_flips > 0 &&
                 flips >= options_.idiot_crash_max_flips)) {
                break;
            }
        }
    };

    auto setupCrashBasis = [&]() {
        setupInitialBasis();
        runStructuralCrash();
        runIdiotCrash();
    };

    // Size LU update budget by default from basis dimension to reduce costly
    // full reinversions and update-drift stalls.
    Int lu_update_limit = options_.lu_update_limit;
    if (lu_update_limit <= 0) {
        lu_update_limit = 100;
        if (num_rows_ >= 500 &&
            num_cols_ >= 4 * num_rows_) {
            // Wide LPs benefit from larger update windows when rows are sparse
            // (fewer expensive reinversions) but need tighter windows on denser
            // rows to cap FT apply traffic.
            const Real avg_nnz_per_row =
                static_cast<Real>(matrix_.numNonzeros()) / static_cast<Real>(num_rows_);
            lu_update_limit = (avg_nnz_per_row <= 15.0) ? 200 : 120;
        } else if (num_rows_ >= 1000 &&
                   num_rows_ < 2000 &&
                   num_cols_ >= 2 * num_rows_ &&
                   num_cols_ <= 3 * num_rows_) {
            lu_update_limit = 120;
        }
    }
    lu_.setMaxUpdates(lu_update_limit);
    Real lu_ft_drop_tol = options_.lu_ft_drop_tolerance;
    if (lu_ft_drop_tol <= 0.0) {
        lu_ft_drop_tol = 1e-13;
    }
    lu_.setFtDropTolerance(lu_ft_drop_tol);

    if (!has_basis_) {
        // Cold start: set up crash basis (all-slack + optional crash passes).
        setupCrashBasis();
    }
    // Warm-start: primals are already in internal (scaled) coordinates.
    // No re-scaling needed — we keep primals in internal coords at all times.

    auto recoverFromSingularBasis = [&]() -> bool {
        setupCrashBasis();
        try {
            refactorize();
            return true;
        } catch (const std::runtime_error&) {
            status_ = Status::Error;
            return false;
        }
    };

    // Factorize basis.
    try {
        refactorize();
    } catch (const std::runtime_error&) {
        if (!recoverFromSingularBasis()) {
            has_basis_ = false;
            work_.add(lu_.workUnits());
            lu_.resetWorkUnits();
            return {status_, getObjective(), iterations_, work_.units() - work_at_start};
        }
    }

    // Compute primal and dual values from the basis.
    computePrimals();
    computeDuals();

    // Cost shifting for dual feasibility (Phase 1).
    std::vector<Real> cost_shift(num_cols_, 0.0);
    bool has_cost_shift = false;
    auto clearCostShifts = [&]() {
        if (!has_cost_shift) return;
        for (Index j = 0; j < num_cols_; ++j) {
            if (std::abs(cost_shift[j]) > kZeroTol) {
                obj_[j] -= cost_shift[j];
                cost_shift[j] = 0.0;
            }
        }
        has_cost_shift = false;
    };
    auto applyCostShifts = [&]() {
        clearCostShifts();
        bool any_shift = false;
        for (Index j = 0; j < num_cols_; ++j) {
            if (var_status_[j] == BasisStatus::Basic) continue;
            const Real rc = reduced_cost_[j];
            if (var_status_[j] == BasisStatus::AtLower) {
                if (rc < -kDualTol) {
                    cost_shift[j] = -rc + kDualTol;
                    obj_[j] += cost_shift[j];
                    reduced_cost_[j] = kDualTol;
                    any_shift = true;
                }
            } else if (var_status_[j] == BasisStatus::AtUpper) {
                if (rc > kDualTol) {
                    cost_shift[j] = -rc - kDualTol;
                    obj_[j] += cost_shift[j];
                    reduced_cost_[j] = -kDualTol;
                    any_shift = true;
                }
            } else if (var_status_[j] == BasisStatus::Free) {
                if (std::abs(rc) > kDualTol) {
                    cost_shift[j] = -rc;
                    obj_[j] += cost_shift[j];
                    reduced_cost_[j] = 0.0;
                    any_shift = true;
                }
            }
        }
        has_cost_shift = any_shift;
        if (has_cost_shift) {
            computeDuals();
        }
    };
    applyCostShifts();

    // Work vectors for the iteration.
    std::vector<Real> pivot_row_alpha(numVars(), 0.0);
    std::vector<Real> pivot_col(num_rows_, 0.0);
    std::vector<Real> work(num_rows_, 0.0);
    const bool use_dual_phase_norm_weight = num_rows_ >= 2000;
    // Static entering-score normalization for primal-feasible/dual-infeasible phase:
    // score(k) = rc(k)^2 / col_norm_sq(k), where slacks have norm 1.
    std::vector<Real> dual_phase_col_norm_sq;
    if (use_dual_phase_norm_weight) {
        dual_phase_col_norm_sq.assign(static_cast<std::size_t>(numVars()), 1.0);
        std::fill(dual_phase_col_norm_sq.begin(),
                  dual_phase_col_norm_sq.begin() + static_cast<std::size_t>(num_cols_), 0.0);
        for (Index i = 0; i < num_rows_; ++i) {
            auto rv = matrix_.row(i);
            for (Index p = 0; p < rv.size(); ++p) {
                const Index j = rv.indices[p];
                const Real a = rv.values[p];
                dual_phase_col_norm_sq[static_cast<std::size_t>(j)] += a * a;
            }
        }
        for (Index j = 0; j < num_cols_; ++j) {
            if (dual_phase_col_norm_sq[static_cast<std::size_t>(j)] <= kZeroTol) {
                dual_phase_col_norm_sq[static_cast<std::size_t>(j)] = 1.0;
            }
            dual_phase_col_norm_sq[static_cast<std::size_t>(j)] =
                std::sqrt(dual_phase_col_norm_sq[static_cast<std::size_t>(j)]);
        }
    }
    Index partial_price_offset = 0;
    Int degenerate_pivot_streak = 0;
    Int empty_candidate_retry_streak = 0;
    Int stall_restart_count = 0;
    Int best_pinf_count = std::numeric_limits<Int>::max();
    Int iters_since_pinf_improve = 0;
    const Int primal_feasible_stall_pivots =
        std::max<Int>(1, options_.primal_feasible_adaptive_refactor_stall_pivots);
    const Int primal_feasible_min_updates =
        std::max<Int>(0, options_.primal_feasible_adaptive_refactor_min_updates);
    const Int primal_feasible_progress_window =
        std::max<Int>(0, options_.primal_feasible_dual_progress_window);
    const Int primal_feasible_refactor_cooldown =
        std::max<Int>(0, options_.primal_feasible_refactor_cooldown);
    const Real primal_feasible_progress_rel_tol =
        std::max<Real>(0.0, options_.primal_feasible_dual_progress_improve_rel_tol);
    const Int auto_bfrt_min_cols = std::max<Int>(0, options_.auto_bfrt_min_cols);
    const Real auto_bfrt_min_ratio =
        std::max<Real>(1.0, options_.auto_bfrt_min_col_row_ratio);
    const bool auto_bfrt_wide =
        options_.enable_auto_bfrt_wide &&
        num_rows_ > 0 &&
        num_cols_ >= auto_bfrt_min_cols &&
        static_cast<Real>(num_cols_) >= auto_bfrt_min_ratio * static_cast<Real>(num_rows_);
    const bool primal_progress_gate_enabled = primal_feasible_progress_window > 0;
    Real primal_feasible_dual_progress_reference = kInf;
    Int primal_feasible_dual_stall_pivots = 0;
    Int primal_feasible_pivots_since_refactor = 0;

    auto saturatingIncrement = [](Int& value) {
        if (value < std::numeric_limits<Int>::max()) {
            ++value;
        }
    };
    auto resetPrimalFeasibleProgress = [&]() {
        primal_feasible_dual_progress_reference = kInf;
        primal_feasible_dual_stall_pivots = 0;
    };

    auto clearBoundPerturbation = [&]() {
        if (!bound_perturb_active_) return;
        bound_perturb_active_ = false;
        std::fill(lower_bound_perturb_.begin(), lower_bound_perturb_.end(), 0.0);
        std::fill(upper_bound_perturb_.begin(), upper_bound_perturb_.end(), 0.0);
    };

    auto activateBoundPerturbation = [&]() -> bool {
        if (!options_.enable_bound_perturbation) return false;
        if (bound_perturb_active_) return false;
        if (options_.bound_perturbation_max_activations >= 0 &&
            bound_perturb_activations_ >= options_.bound_perturbation_max_activations) {
            return false;
        }
        if (options_.bound_perturbation_magnitude <= 0.0) return false;

        const Index n = numVars();
        lower_bound_perturb_.assign(static_cast<std::size_t>(n), 0.0);
        upper_bound_perturb_.assign(static_cast<std::size_t>(n), 0.0);

        for (Index k = 0; k < n; ++k) {
            const Real lb = (k < num_cols_) ? col_lower_[k] : row_lower_[k - num_cols_];
            const Real ub = (k < num_cols_) ? col_upper_[k] : row_upper_[k - num_cols_];
            if (lb == -kInf && ub == kInf) continue;

            const uint32_t hash = static_cast<uint32_t>(k + 1) * 2246822519u;
            const Real weight = 1.0 + static_cast<Real>((hash >> 24) & 0xff) / 512.0;
            const Real scale = std::max<Real>(
                1.0, std::max((lb == -kInf) ? 0.0 : std::abs(lb),
                              (ub == kInf) ? 0.0 : std::abs(ub)));
            Real eps = options_.bound_perturbation_magnitude * weight * scale;
            if (eps <= kZeroTol) continue;

            if (lb != -kInf && ub != kInf) {
                const Real width = ub - lb;
                if (width <= 4.0 * kPrimalTol) continue;
                eps = std::min(eps, 0.25 * width);
                lower_bound_perturb_[k] = eps;
                upper_bound_perturb_[k] = eps;
            } else if (lb != -kInf) {
                lower_bound_perturb_[k] = eps;
            } else if (ub != kInf) {
                upper_bound_perturb_[k] = eps;
            }
        }

        bound_perturb_active_ = true;
        ++bound_perturb_activations_;
        return true;
    };

    // Log header.
    if (verbose_) std::printf("%-7s  %9s  %20s  %11s  %9s\n",
                              "Method", "Iteration", "Objective", "Primal.NInf", "Time");

#ifdef MIPX_HAS_TBB
    auto sipThreadGatePass = [&]() -> bool {
        Int need = std::max<Int>(1, options_.sip_parallel_min_threads);
        return static_cast<Int>(tbb::this_task_arena::max_concurrency()) >= need;
    };
#endif

    // Main dual simplex loop.
    while (iterations_ < iter_limit_) {
        if (options_.max_solve_seconds >= 0.0) {
            double elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - solve_start_).count();
            if (elapsed >= options_.max_solve_seconds) {
                status_ = Status::TimeLimit;
                break;
            }
        }
        if (options_.stop_flag != nullptr &&
            options_.stop_flag->load(std::memory_order_relaxed)) {
            status_ = Status::IterLimit;
            break;
        }

        if (bound_perturb_active_) {
            const std::size_t expected = static_cast<std::size_t>(numVars());
            if (lower_bound_perturb_.size() != expected ||
                upper_bound_perturb_.size() != expected) {
                status_ = Status::Error;
                break;
            }
            if (degenerate_pivot_streak == 0) {
                clearBoundPerturbation();
            }
        }
        if (!bound_perturb_active_ &&
            options_.enable_bound_perturbation &&
            options_.bound_perturbation_stall_pivots > 0 &&
            degenerate_pivot_streak >= options_.bound_perturbation_stall_pivots) {
            activateBoundPerturbation();
        }

        const bool restart_budget_ok =
            (options_.stall_restart_max_restarts < 0) ||
            (stall_restart_count < options_.stall_restart_max_restarts);
        if (options_.enable_stall_restart &&
            options_.stall_restart_pivots > 0 &&
            degenerate_pivot_streak >= options_.stall_restart_pivots &&
            restart_budget_ok) {
            clearBoundPerturbation();
            clearCostShifts();
            setupCrashBasis();
            try {
                refactorize();
            } catch (const std::runtime_error&) {
                if (!recoverFromSingularBasis()) {
                    break;
                }
            }
            computePrimals();
            computeDuals();
            applyCostShifts();

            ++stall_restart_count;
            degenerate_pivot_streak = 0;
            empty_candidate_retry_streak = 0;
            partial_price_offset = 0;
            best_pinf_count = std::numeric_limits<Int>::max();
            iters_since_pinf_improve = 0;
            resetPrimalFeasibleProgress();
            primal_feasible_pivots_since_refactor = 0;
            ++iterations_;
            continue;
        }

        // Log every kLogFrequency iterations.
        if (iterations_ % kLogFrequency == 0) {
            // Compute current objective.
            Real obj_val = 0.0;
            for (Index k = 0; k < numVars(); ++k) {
                obj_val += varCost(k) * primal_[k];
            }

            // Count primal infeasibilities (number of infeasible rows).
            Int pinf_count = 0;
            for (Index i = 0; i < num_rows_; ++i) {
                Index k = basis_[i];
                Real xk = primal_[k];
                Real lb = varLower(k);
                Real ub = varUpper(k);
                if (xk < lb - kPrimalTol || xk > ub + kPrimalTol) ++pinf_count;
            }
            double solve_elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - solve_start_).count();
            if (verbose_) std::printf("%-7s  %9d  %20.10e  %11d  %7.2fs\n",
                                      "Dual", iterations_, obj_val, pinf_count, solve_elapsed);
        }

        #ifdef MIPX_HAS_TBB
        bool sip_numerical_gate = true;
        if (options_.sip_parallel_disable_on_stall &&
            options_.sip_parallel_stall_pivots > 0 &&
            degenerate_pivot_streak >= options_.sip_parallel_stall_pivots) {
            sip_numerical_gate = false;
        }
        #endif

        bool use_dual_perturbation =
            options_.enable_dual_perturbation &&
            options_.dual_perturbation_stall_pivots > 0 &&
            degenerate_pivot_streak >= options_.dual_perturbation_stall_pivots;
        auto reducedCostForPricing = [&](Index k) -> Real {
            Real rc = reduced_cost_[k];
            if (!use_dual_perturbation) return rc;

            const uint32_t hash = static_cast<uint32_t>(k + 1) * 2654435761u;
            const Real weight = 1.0 + static_cast<Real>((hash >> 22) & 0x3ff) / 1024.0;
            const Real eps = options_.dual_perturbation_magnitude * weight;
            const BasisStatus st = var_status_[k];
            if (st == BasisStatus::AtLower) return rc + eps;
            if (st == BasisStatus::AtUpper) return rc - eps;
            if (st == BasisStatus::Free) return rc + ((k & 1) ? eps : -eps);
            return rc;
        };
        const Real* lower_perturb =
            bound_perturb_active_ ? lower_bound_perturb_.data() : nullptr;
        const Real* upper_perturb =
            bound_perturb_active_ ? upper_bound_perturb_.data() : nullptr;
        auto lowerBoundFast = [&](Index k) -> Real {
            Real lb = (k < num_cols_) ? col_lower_[k] : row_lower_[k - num_cols_];
            if (lower_perturb != nullptr) {
                lb += lower_perturb[static_cast<std::size_t>(k)];
            }
            return lb;
        };
        auto upperBoundFast = [&](Index k) -> Real {
            Real ub = (k < num_cols_) ? col_upper_[k] : row_upper_[k - num_cols_];
            if (upper_perturb != nullptr) {
                ub -= upper_perturb[static_cast<std::size_t>(k)];
            }
            return ub;
        };

        // ---- CHUZR: Find leaving variable (Devex pricing) ----
        work_.count(static_cast<uint64_t>(num_rows_));  // pricing scan
        Index leaving_row = -1;
        Real max_score = 0.0;
        Int current_pinf_count = 0;

        auto scoreRow = [&](Index i) -> Real {
            Index k = basis_[i];
            Real xk = primal_[k];
            Real lb = lowerBoundFast(k);
            Real ub = upperBoundFast(k);

            Real viol = 0.0;
            if (xk < lb - kPrimalTol) viol = lb - xk;
            else if (xk > ub + kPrimalTol) viol = xk - ub;

            if (viol > kPrimalTol) {
                return viol * viol / devex_weights_[i];
            }
            return 0.0;
        };

#ifdef MIPX_HAS_TBB
        if (options_.enable_sip_parallel_chuzr &&
            num_rows_ >= options_.sip_parallel_min_rows &&
            options_.sip_parallel_row_grain > 0 &&
            sipThreadGatePass() &&
            sip_numerical_gate) {
            struct ChuzrBest {
                Index row = -1;
                Real score = 0.0;
                Int pinf = 0;
            };
            auto better = [](const ChuzrBest& a, const ChuzrBest& b) {
                if (a.row < 0) return false;
                if (b.row < 0) return true;
                if (a.score > b.score) return true;
                if (a.score < b.score) return false;
                return a.row < b.row;  // deterministic tie-break
            };

            tbb::enumerable_thread_specific<ChuzrBest> locals;
            const Index grain = std::max<Index>(1, options_.sip_parallel_row_grain);
            tbb::parallel_for(tbb::blocked_range<Index>(0, num_rows_, grain),
                              [&](const tbb::blocked_range<Index>& range) {
                ChuzrBest local_best;
                Int local_pinf = 0;
                for (Index i = range.begin(); i < range.end(); ++i) {
                    Real score = scoreRow(i);
                    if (score > 0.0) {
                        ++local_pinf;
                        ChuzrBest cand{i, score};
                        if (better(cand, local_best)) {
                            local_best = cand;
                        }
                    }
                }
                if (local_best.row >= 0) {
                    local_best.pinf = local_pinf;
                    auto& slot = locals.local();
                    if (better(local_best, slot)) {
                        local_best.pinf += slot.pinf;
                        slot = local_best;
                    } else {
                        slot.pinf += local_pinf;
                    }
                }
            });

            for (const auto& local : locals) {
                current_pinf_count += local.pinf;
                if (local.score > max_score ||
                    (local.score == max_score && local.row >= 0 &&
                     (leaving_row < 0 || local.row < leaving_row))) {
                    max_score = local.score;
                    leaving_row = local.row;
                }
            }
        } else
#endif
        {
            for (Index i = 0; i < num_rows_; ++i) {
                Real score = scoreRow(i);
                if (score > 0.0) {
                    ++current_pinf_count;
                }
                if (score > max_score) {
                    max_score = score;
                    leaving_row = i;
                }
            }
        }

        if (current_pinf_count < best_pinf_count) {
            best_pinf_count = current_pinf_count;
            iters_since_pinf_improve = 0;
        } else {
            ++iters_since_pinf_improve;
        }

        if (leaving_row < 0) {
            // All basic variables are primal feasible.

            if (has_cost_shift) {
                // Phase 1 complete: remove cost shifts.
                clearCostShifts();
                computeDuals();
                resetPrimalFeasibleProgress();

                // Flip dual-infeasible nonbasic variables to their other bound.
                bool flipped = false;
                for (Index k : nonbasic_) {
                    Real rc = reduced_cost_[k];
                    BasisStatus st = var_status_[k];
                    if (st == BasisStatus::AtLower && rc < -kDualTol &&
                        varUpper(k) != kInf) {
                        primal_[k] = varUpper(k);
                        var_status_[k] = BasisStatus::AtUpper;
                        flipped = true;
                    } else if (st == BasisStatus::AtUpper && rc > kDualTol &&
                               varLower(k) != -kInf) {
                        primal_[k] = varLower(k);
                        var_status_[k] = BasisStatus::AtLower;
                        flipped = true;
                    }
                }
                if (flipped) {
                    computePrimals();
                }
                continue;  // Flipping may have created primal infeasibility.
            }

            // No cost shifts active. Check dual feasibility.
            Index entering_p = -1;
            Real worst_rc = 0.0;
            #ifdef MIPX_HAS_TBB
            Index nnb = static_cast<Index>(nonbasic_.size());
            #endif

#ifdef MIPX_HAS_TBB
            if (options_.enable_sip_parallel_dual_scan &&
                nnb >= options_.sip_parallel_min_nonbasic &&
                options_.sip_parallel_grain > 0 &&
                sipThreadGatePass() &&
                sip_numerical_gate) {
                struct DualScanBest {
                    Index var = -1;
                    Index pos = -1;
                    Real rc = 0.0;
                    Real score = 0.0;
                };
                auto better = [](const DualScanBest& a, const DualScanBest& b) {
                    if (a.var < 0) return false;
                    if (b.var < 0) return true;
                    if (a.score > b.score) return true;
                    if (a.score < b.score) return false;
                    return a.pos < b.pos;  // deterministic tie-break
                };

                tbb::enumerable_thread_specific<DualScanBest> locals;
                const Index grain = std::max<Index>(1, options_.sip_parallel_grain);
                tbb::parallel_for(tbb::blocked_range<Index>(0, nnb, grain),
                                  [&](const tbb::blocked_range<Index>& range) {
                    DualScanBest local;
                    for (Index pos = range.begin(); pos < range.end(); ++pos) {
                        Index k = nonbasic_[pos];
                        Real rc = reducedCostForPricing(k);
                        BasisStatus st = var_status_[k];
                        bool dual_inf =
                            (st == BasisStatus::AtLower && rc < -kDualTol) ||
                            (st == BasisStatus::AtUpper && rc > kDualTol) ||
                            (st == BasisStatus::Free && std::abs(rc) > kDualTol);
                        if (!dual_inf) continue;
                        const Real score = use_dual_phase_norm_weight
                            ? (rc * rc) / dual_phase_col_norm_sq[static_cast<std::size_t>(k)]
                            : std::abs(rc);
                        DualScanBest cand{k, pos, rc, score};
                        if (better(cand, local)) {
                            local = cand;
                        }
                    }
                    if (local.var >= 0) {
                        auto& slot = locals.local();
                        if (better(local, slot)) {
                            slot = local;
                        }
                    }
                });

                DualScanBest best;
                for (const auto& local : locals) {
                    if (better(local, best)) {
                        best = local;
                    }
                }
                if (best.var >= 0) {
                    entering_p = best.var;
                    worst_rc = best.rc;
                }
            } else
#endif
            {
                Real best_entering_score = -1.0;
                for (Index k : nonbasic_) {
                    Real rc = reducedCostForPricing(k);
                    BasisStatus st = var_status_[k];
                    bool dual_inf =
                        (st == BasisStatus::AtLower && rc < -kDualTol) ||
                        (st == BasisStatus::AtUpper && rc > kDualTol) ||
                        (st == BasisStatus::Free && std::abs(rc) > kDualTol);
                    if (!dual_inf) continue;
                    const Real score = use_dual_phase_norm_weight
                        ? (rc * rc) / dual_phase_col_norm_sq[static_cast<std::size_t>(k)]
                        : std::abs(rc);
                    if (score > best_entering_score) {
                        best_entering_score = score;
                        worst_rc = rc;
                        entering_p = k;
                    }
                }
            }

            if (entering_p < 0) {
                if (bound_perturb_active_) {
                    // Re-validate optimality against original (unperturbed) bounds.
                    clearBoundPerturbation();
                    degenerate_pivot_streak = 0;
                    resetPrimalFeasibleProgress();
                    continue;
                }
                status_ = Status::Optimal;
                break;
            }

            const Real worst_dual_inf_abs = std::abs(worst_rc);
            if (primal_progress_gate_enabled) {
                if (primal_feasible_dual_progress_reference == kInf) {
                    primal_feasible_dual_progress_reference = worst_dual_inf_abs;
                    primal_feasible_dual_stall_pivots = 0;
                } else {
                    const Real target = primal_feasible_dual_progress_reference *
                                        (1.0 - primal_feasible_progress_rel_tol);
                    if (worst_dual_inf_abs + kDualTol <= target) {
                        primal_feasible_dual_progress_reference = worst_dual_inf_abs;
                        primal_feasible_dual_stall_pivots = 0;
                    } else {
                        if (worst_dual_inf_abs < primal_feasible_dual_progress_reference) {
                            primal_feasible_dual_progress_reference = worst_dual_inf_abs;
                        }
                        saturatingIncrement(primal_feasible_dual_stall_pivots);
                    }
                }
            }

            // Primal feasible but not dual feasible: primal simplex pivot.

            // FTRAN: compute pivot column eta = B^{-1} * a_entering.
            std::fill(pivot_col.begin(), pivot_col.end(), 0.0);
            if (entering_p < num_cols_) {
                auto cv = matrix_.col(entering_p);
                for (Index p = 0; p < cv.size(); ++p) {
                    pivot_col[cv.indices[p]] = cv.values[p];
                }
            } else {
                pivot_col[entering_p - num_cols_] = -1.0;
            }
            lu_.ftran(pivot_col);

            // Direction: entering moves to improve objective.
            Real delta_dir;
            if (var_status_[entering_p] == BasisStatus::AtLower) {
                delta_dir = 1.0;  // increase (rc < 0)
            } else if (var_status_[entering_p] == BasisStatus::AtUpper) {
                delta_dir = -1.0;  // decrease (rc > 0)
            } else {
                delta_dir = (worst_rc < 0) ? 1.0 : -1.0;
            }

            // Primal ratio test: find leaving variable.
            Index leaving_row_p = -1;
            Real min_step = kInf;
            constexpr Real kPrimalHarrisTol = 1e-10;
            Real best_leaving_pivot_abs = 0.0;
            for (Index i = 0; i < num_rows_; ++i) {
                Real f = pivot_col[i] * delta_dir;
                // x_B[i]_new = x_B[i]_old - f * step
                if (std::abs(f) < kPivotTol) continue;

                Index bvar = basis_[i];
                Real lb = varLower(bvar);
                Real ub = varUpper(bvar);
                Real xval = primal_[bvar];

                Real step;
                if (f > 0) {
                    // x_B[i] decreases: limited by lower bound.
                    step = (lb == -kInf) ? kInf : (xval - lb) / f;
                } else {
                    // x_B[i] increases: limited by upper bound.
                    step = (ub == kInf) ? kInf : (ub - xval) / (-f);
                }

                if (step < -kPrimalTol) continue;
                const Real abs_f = std::abs(f);
                if (step + kPrimalHarrisTol < min_step) {
                    min_step = step;
                    leaving_row_p = i;
                    best_leaving_pivot_abs = abs_f;
                } else if (step <= min_step + kPrimalHarrisTol &&
                           abs_f > best_leaving_pivot_abs) {
                    leaving_row_p = i;
                    best_leaving_pivot_abs = abs_f;
                }
            }

            // Check entering variable's opposite bound.
            Real entering_bound_step = kInf;
            if (delta_dir > 0 && varUpper(entering_p) != kInf) {
                entering_bound_step =
                    varUpper(entering_p) - primal_[entering_p];
            } else if (delta_dir < 0 && varLower(entering_p) != -kInf) {
                entering_bound_step =
                    primal_[entering_p] - varLower(entering_p);
            }

            if (min_step == kInf && entering_bound_step == kInf) {
                status_ = Status::Unbounded;
                break;
            }

            Real step = std::min(min_step, entering_bound_step);
            if (step < 0) step = 0;
            Real delta_entering = delta_dir * step;
            const bool primal_step_degenerate =
                std::abs(step) <= options_.adaptive_refactor_degenerate_pivot_tol;

            // Update primals.
            for (Index i = 0; i < num_rows_; ++i) {
                primal_[basis_[i]] -= pivot_col[i] * delta_entering;
            }
            primal_[entering_p] += delta_entering;

            if (entering_bound_step <= min_step) {
                // Entering variable hits its opposite bound. No basis change.
                if (delta_dir > 0) {
                    primal_[entering_p] = varUpper(entering_p);
                    var_status_[entering_p] = BasisStatus::AtUpper;
                } else {
                    primal_[entering_p] = varLower(entering_p);
                    var_status_[entering_p] = BasisStatus::AtLower;
                }
                if (primal_step_degenerate) {
                    ++degenerate_pivot_streak;
                } else {
                    degenerate_pivot_streak = 0;
                }
                saturatingIncrement(primal_feasible_pivots_since_refactor);
                ++iterations_;
                continue;
            }

            // Basis swap: entering enters, leaving exits.
            Index leaving_var_p = basis_[leaving_row_p];
            Real pivot_elem = pivot_col[leaving_row_p];

            // BTRAN for dual update.
            std::fill(work.begin(), work.end(), 0.0);
            work[leaving_row_p] = 1.0;
            lu_.btran(work);

            // Compute pivot row alphas (row-wise).
            work_.count(static_cast<uint64_t>(matrix_.numNonzeros()));  // alpha computation
            std::fill(pivot_row_alpha.begin(), pivot_row_alpha.end(), 0.0);
            for (Index i = 0; i < num_rows_; ++i) {
                Real rho_i = work[i];
                if (std::abs(rho_i) < kZeroTol) continue;
                auto rv = matrix_.row(i);
                for (Index pp = 0; pp < rv.size(); ++pp) {
                    pivot_row_alpha[rv.indices[pp]] += rho_i * rv.values[pp];
                }
            }
            negateCopy(std::span<Real>(pivot_row_alpha.data() + num_cols_,
                                       static_cast<std::size_t>(num_rows_)),
                       std::span<const Real>(work.data(),
                                             static_cast<std::size_t>(num_rows_)),
                       options_);

            // Update reduced costs.
            Real theta_d_p = reduced_cost_[entering_p] / pivot_elem;
            for (Index k : nonbasic_) {
                if (k == entering_p) continue;
                if (std::abs(pivot_row_alpha[k]) > kZeroTol) {
                    reduced_cost_[k] -= theta_d_p * pivot_row_alpha[k];
                }
            }
            reduced_cost_[entering_p] = 0.0;
            reduced_cost_[leaving_var_p] = -theta_d_p;

            // Update duals.
            axpyInPlace(std::span<Real>(dual_.data(), static_cast<std::size_t>(num_rows_)),
                        std::span<const Real>(work.data(),
                                              static_cast<std::size_t>(num_rows_)),
                        theta_d_p, options_);

            // Determine leaving variable status.
            Real leaving_lb = varLower(leaving_var_p);
            Real leaving_ub = varUpper(leaving_var_p);
            BasisStatus leaving_st;
            if (leaving_lb != -kInf && leaving_ub != kInf &&
                std::abs(leaving_lb - leaving_ub) < kZeroTol) {
                leaving_st = BasisStatus::Fixed;
            } else if (pivot_col[leaving_row_p] * delta_dir > 0) {
                // x_B decreased -> hit lower bound.
                primal_[leaving_var_p] = leaving_lb;
                leaving_st = BasisStatus::AtLower;
            } else {
                primal_[leaving_var_p] = leaving_ub;
                leaving_st = BasisStatus::AtUpper;
            }

            // Swap basis.
            basis_[leaving_row_p] = entering_p;
            basis_pos_[entering_p] = leaving_row_p;
            basis_pos_[leaving_var_p] = -1;
            var_status_[entering_p] = BasisStatus::Basic;
            var_status_[leaving_var_p] = leaving_st;

            Index nb_pos = nonbasic_pos_[entering_p];
            if (nb_pos >= 0) {
                nonbasic_[nb_pos] = leaving_var_p;
                nonbasic_pos_[leaving_var_p] = nb_pos;
                nonbasic_pos_[entering_p] = -1;
            }

            // Reuse transformed pivot column and avoid a second FTRAN in LU update.
            lu_.updateFromFtranColumn(
                leaving_row_p,
                std::span<const Real>(pivot_col.data(),
                                      static_cast<std::size_t>(num_rows_)));

            if (primal_step_degenerate) {
                ++degenerate_pivot_streak;
            } else {
                degenerate_pivot_streak = 0;
            }
            saturatingIncrement(primal_feasible_pivots_since_refactor);

            bool should_refactorize = lu_.needsRefactorization();
            const bool primal_progress_stalled =
                !primal_progress_gate_enabled ||
                primal_feasible_dual_stall_pivots >= primal_feasible_progress_window;
            const bool primal_cooldown_elapsed =
                primal_feasible_pivots_since_refactor >= primal_feasible_refactor_cooldown;
            if (!should_refactorize &&
                options_.enable_adaptive_refactorization &&
                degenerate_pivot_streak >= primal_feasible_stall_pivots &&
                lu_.numUpdates() >= primal_feasible_min_updates &&
                primal_progress_stalled &&
                primal_cooldown_elapsed) {
                should_refactorize = true;
            }

            if (should_refactorize) {
                try {
                    refactorize();
                } catch (const std::runtime_error&) {
                    if (!recoverFromSingularBasis()) {
                        break;
                    }
                }
                computePrimals();
                computeDuals();
                degenerate_pivot_streak = 0;
                resetPrimalFeasibleProgress();
                primal_feasible_pivots_since_refactor = 0;
                std::fill(devex_weights_.begin(), devex_weights_.end(), 1.0);
                devex_reset_count_ = 0;
            }

            ++iterations_;
            continue;
        }

        Index leaving_var = basis_[leaving_row];
        Real leaving_val = primal_[leaving_var];
        Real leaving_lb = lowerBoundFast(leaving_var);
        Real leaving_ub = upperBoundFast(leaving_var);

        // Direction: +1 if below lower bound, -1 if above upper bound.
        Real sigma = (leaving_val < leaving_lb) ? 1.0 : -1.0;

        // ---- BTRAN: Compute pivot row representation ----
        // y = B^{-T} * e_{leaving_row}
        std::fill(work.begin(), work.end(), 0.0);
        work[leaving_row] = 1.0;
        lu_.btran(work);

        // Compute alpha_j = rho^T * a_j for all nonbasic j.
        // Row-wise: alpha = rho^T * A, then alpha_{n+i} = -rho[i] for slacks.
        // This is much faster than column-wise when rho is sparse.
        work_.count(static_cast<uint64_t>(matrix_.numNonzeros()));  // alpha computation
        std::fill(pivot_row_alpha.begin(), pivot_row_alpha.end(), 0.0);
        for (Index i = 0; i < num_rows_; ++i) {
            Real rho_i = work[i];
            if (std::abs(rho_i) < kZeroTol) continue;
            auto rv = matrix_.row(i);
            for (Index p = 0; p < rv.size(); ++p) {
                pivot_row_alpha[rv.indices[p]] += rho_i * rv.values[p];
            }
        }
        // Slack alpha values: alpha_{n+i} = rho^T * (-e_i) = -rho[i].
        negateCopy(std::span<Real>(pivot_row_alpha.data() + num_cols_,
                                   static_cast<std::size_t>(num_rows_)),
                   std::span<const Real>(work.data(),
                                         static_cast<std::size_t>(num_rows_)),
                   options_);

        // ---- CHUZC: Dual ratio test with BFRT (Bound Flipping) ----
        Index entering_var = -1;
        Real harris_tol = 1e-6;

        auto isEligible = [&](Index k, Real alpha) -> bool {
            if (std::abs(alpha) < kPivotTol) return false;
            BasisStatus st = var_status_[k];
            if (st == BasisStatus::Fixed) return false;
            if (st == BasisStatus::Free) return true;
            if (sigma > 0.0) {
                if (st == BasisStatus::AtLower && alpha < -kPivotTol) return true;
                if (st == BasisStatus::AtUpper && alpha > kPivotTol) return true;
            } else {
                if (st == BasisStatus::AtLower && alpha > kPivotTol) return true;
                if (st == BasisStatus::AtUpper && alpha < -kPivotTol) return true;
            }
            return false;
        };

        // Collect eligible candidates with dual ratios and bound gaps.
        struct BfrtCand {
            Index var;
            Index nonbasic_pos;
            Real ratio;
            Real alpha;
            Real gap;  // ub - lb, kInf if no finite opposite bound
        };
        static thread_local std::vector<BfrtCand> bfrt_cands;
        bfrt_cands.clear();

        bool has_flips = false;
        Real target = (sigma > 0.0) ? leaving_lb : leaving_ub;

        Index nnb = static_cast<Index>(nonbasic_.size());
        if (bfrt_cands.capacity() < static_cast<std::size_t>(nnb)) {
            bfrt_cands.reserve(static_cast<std::size_t>(nnb));
        }
        auto appendCandidate = [&](Index pos, std::vector<BfrtCand>& out) {
            Index k = nonbasic_[pos];
            Real alpha = pivot_row_alpha[k];
            if (!isEligible(k, alpha)) return;
            Real ratio = std::abs(reducedCostForPricing(k) / alpha);
            Real lb = lowerBoundFast(k);
            Real ub = upperBoundFast(k);
            Real gap = (lb != -kInf && ub != kInf) ? (ub - lb) : kInf;
            out.push_back({k, pos, ratio, alpha, gap});
        };

        auto collectCandidatesSerial = [&](Index offset, Index count) {
            if (count <= 0) return;
            Index first_end = std::min<Index>(nnb, offset + count);
            for (Index pos = offset; pos < first_end; ++pos) {
                appendCandidate(pos, bfrt_cands);
            }
            Index wrapped = count - (first_end - offset);
            for (Index pos = 0; pos < wrapped; ++pos) {
                appendCandidate(pos, bfrt_cands);
            }
        };

        auto collectCandidates = [&](Index offset, Index count) {
#ifdef MIPX_HAS_TBB
            if (options_.enable_sip_parallel_candidates &&
                count >= options_.sip_parallel_min_nonbasic &&
                options_.sip_parallel_grain > 0 &&
                sipThreadGatePass() &&
                sip_numerical_gate) {
                tbb::enumerable_thread_specific<std::vector<BfrtCand>> locals;
                const Index grain = std::max<Index>(1, options_.sip_parallel_grain);
                tbb::parallel_for(tbb::blocked_range<Index>(0, count, grain),
                                  [&](const tbb::blocked_range<Index>& range) {
                    auto& local = locals.local();
                    local.reserve(local.size() +
                        static_cast<std::size_t>((range.end() - range.begin()) / 4 + 8));
                    for (Index t = range.begin(); t < range.end(); ++t) {
                        Index pos = offset + t;
                        if (pos >= nnb) pos -= nnb;
                        appendCandidate(pos, local);
                    }
                });

                for (auto& local : locals) {
                    bfrt_cands.insert(bfrt_cands.end(), local.begin(), local.end());
                }
                return;
            }
#endif
            collectCandidatesSerial(offset, count);
        };

        bool did_partial_scan = false;
        if (options_.enable_partial_pricing &&
            nnb > options_.partial_pricing_chunk_min &&
            options_.partial_pricing_full_scan_freq > 0 &&
            (iterations_ % options_.partial_pricing_full_scan_freq) != 0) {
            Index chunk = std::max<Index>(options_.partial_pricing_chunk_min, nnb / 8);
            chunk = std::min(chunk, nnb);
            collectCandidates(partial_price_offset, chunk);
            partial_price_offset = (partial_price_offset + chunk) % nnb;
            did_partial_scan = true;
        } else {
            collectCandidates(0, nnb);
        }

        // Fallback to full scan if partial scan found no eligible entering candidates.
        if (did_partial_scan && bfrt_cands.empty()) {
            collectCandidates(0, nnb);
        }

        if (bfrt_cands.empty()) {
            // No eligible entering variable for this leaving row can be caused
            // by numerical drift, stale factorization, or partial pricing
            // artifacts. Refactorize once before declaring infeasibility.
            bool any_primal_violation = false;
            for (Index i = 0; i < num_rows_; ++i) {
                Index k = basis_[i];
                Real xk = primal_[k];
                if (xk < lowerBoundFast(k) - kPrimalTol ||
                    xk > upperBoundFast(k) + kPrimalTol) {
                    any_primal_violation = true;
                    break;
                }
            }
            if (!any_primal_violation) {
                status_ = Status::Infeasible;
                break;
            }
            bool dual_feasible = true;
            for (Index k : nonbasic_) {
                const Real rc = reduced_cost_[k];
                const BasisStatus st = var_status_[k];
                const bool dual_inf =
                    (st == BasisStatus::AtLower && rc < -kDualTol) ||
                    (st == BasisStatus::AtUpper && rc > kDualTol) ||
                    (st == BasisStatus::Free && std::abs(rc) > kDualTol);
                if (dual_inf) {
                    dual_feasible = false;
                    break;
                }
            }
            if (dual_feasible) {
                status_ = Status::Infeasible;
                break;
            }
            if (empty_candidate_retry_streak >= 4) {
                status_ = Status::Error;
                break;
            }

            try {
                refactorize();
            } catch (const std::runtime_error&) {
                if (!recoverFromSingularBasis()) {
                    break;
                }
            }
            computePrimals();
            computeDuals();
            degenerate_pivot_streak = 0;
            ++empty_candidate_retry_streak;
            ++iterations_;
            continue;
        }
        empty_candidate_retry_streak = 0;

        auto cand_ratio_pos_less = [](const BfrtCand& a, const BfrtCand& b) {
            if (a.ratio != b.ratio) return a.ratio < b.ratio;
            if (a.nonbasic_pos != b.nonbasic_pos) return a.nonbasic_pos < b.nonbasic_pos;
            return a.var < b.var;
        };

        bool has_finite_gap_cand = false;
        Index best_ci = -1;
        for (Index ci = 0; ci < static_cast<Index>(bfrt_cands.size()); ++ci) {
            const auto& c = bfrt_cands[ci];
            if (c.gap != kInf) has_finite_gap_cand = true;
            if (best_ci < 0 || cand_ratio_pos_less(c, bfrt_cands[best_ci])) {
                best_ci = ci;
            }
        }

        bool adaptive_bfrt_gate = true;
        if (options_.enable_adaptive_bfrt) {
            const bool high_primal_infeasibility =
                current_pinf_count > options_.adaptive_bfrt_max_pinf;
            const bool stalled_pinf_progress =
                options_.adaptive_bfrt_progress_window > 0 &&
                iters_since_pinf_improve >= options_.adaptive_bfrt_progress_window;
            adaptive_bfrt_gate = high_primal_infeasibility || stalled_pinf_progress;
        }
        const bool bfrt_enabled = options_.enable_bfrt || auto_bfrt_wide;
        const bool use_bfrt =
            has_finite_gap_cand &&
            bfrt_enabled &&
            adaptive_bfrt_gate;

        if (!use_bfrt) {
            // Common LP fast path: all candidates are effectively unflippable
            // (one-sided bounds), so BFRT reduces to min-ratio selection.
            Real enter_ratio = bfrt_cands[best_ci].ratio;
            Real harris_threshold = enter_ratio + harris_tol;
            entering_var = bfrt_cands[best_ci].var;
            Real best_alpha_val = std::abs(bfrt_cands[best_ci].alpha);
            Index best_pos = bfrt_cands[best_ci].nonbasic_pos;

            for (Index ci = 0; ci < static_cast<Index>(bfrt_cands.size()); ++ci) {
                const auto& c = bfrt_cands[ci];
                if (c.ratio > harris_threshold) continue;
                Real abs_alpha = std::abs(c.alpha);
                if (abs_alpha > best_alpha_val ||
                    (abs_alpha == best_alpha_val && c.nonbasic_pos < best_pos)) {
                    best_alpha_val = abs_alpha;
                    best_pos = c.nonbasic_pos;
                    entering_var = c.var;
                }
            }
        } else {
            // Sort by ratio for BFRT sweep.
            // Keep legacy ordering when SIP scan is disabled to avoid behavior drift.
            if (!options_.enable_sip_parallel_candidates) {
                std::sort(bfrt_cands.begin(), bfrt_cands.end(),
                    [](const BfrtCand& a, const BfrtCand& b) {
                        return a.ratio < b.ratio;
                    });
            } else {
                auto cmp = [](const BfrtCand& a, const BfrtCand& b) {
                    if (a.ratio != b.ratio) return a.ratio < b.ratio;
                    if (a.nonbasic_pos != b.nonbasic_pos) {
                        return a.nonbasic_pos < b.nonbasic_pos;
                    }
                    return a.var < b.var;
                };
#ifdef MIPX_HAS_TBB
                if (options_.enable_sip_parallel_candidate_sort &&
                    static_cast<Index>(bfrt_cands.size()) >= options_.sip_parallel_sort_min_candidates &&
                    sipThreadGatePass() &&
                    sip_numerical_gate) {
                    tbb::parallel_sort(bfrt_cands.begin(), bfrt_cands.end(), cmp);
                } else
#endif
                {
                    std::sort(bfrt_cands.begin(), bfrt_cands.end(), cmp);
                }
            }

            // BFRT sweep: flip bounded variables until the leaving variable's
            // infeasibility can be resolved by the next candidate entering.
            Real needed = std::abs(leaving_val - target);
            Real accumulated = 0.0;
            Index entering_ci = -1;

            for (Index ci = 0; ci < static_cast<Index>(bfrt_cands.size()); ++ci) {
                auto& c = bfrt_cands[ci];

                if (c.gap == kInf) {
                    entering_ci = ci;
                    break;
                }

                Real contribution = std::abs(c.alpha) * c.gap;
                if (accumulated + contribution >= needed) {
                    entering_ci = ci;
                    break;
                }

                // Flip this variable to its opposite bound.
                if (var_status_[c.var] == BasisStatus::AtLower) {
                    primal_[c.var] = upperBoundFast(c.var);
                    var_status_[c.var] = BasisStatus::AtUpper;
                } else {
                    primal_[c.var] = lowerBoundFast(c.var);
                    var_status_[c.var] = BasisStatus::AtLower;
                }
                has_flips = true;
                accumulated += contribution;
            }

            if (entering_ci < 0) {
                entering_ci = static_cast<Index>(bfrt_cands.size()) - 1;
            }

            // Harris refinement: among candidates near the entering ratio,
            // pick the one with largest |alpha| for numerical stability.
            Real enter_ratio = bfrt_cands[entering_ci].ratio;
            Real harris_threshold = enter_ratio + harris_tol;
            entering_var = bfrt_cands[entering_ci].var;
            Real best_alpha_val = std::abs(bfrt_cands[entering_ci].alpha);

            for (Index ci = entering_ci + 1;
                 ci < static_cast<Index>(bfrt_cands.size()); ++ci) {
                if (bfrt_cands[ci].ratio > harris_threshold) break;
                if (std::abs(bfrt_cands[ci].alpha) > best_alpha_val) {
                    best_alpha_val = std::abs(bfrt_cands[ci].alpha);
                    entering_var = bfrt_cands[ci].var;
                }
            }
        }

        Real alpha_entering = pivot_row_alpha[entering_var];

        // ---- FTRAN: Compute pivot column ----
        // aq = B^{-1} * a_entering (use matrix_ directly, not augmented_)
        std::fill(pivot_col.begin(), pivot_col.end(), 0.0);
        if (entering_var < num_cols_) {
            auto cv = matrix_.col(entering_var);
            for (Index p = 0; p < cv.size(); ++p) {
                pivot_col[cv.indices[p]] = cv.values[p];
            }
        } else {
            pivot_col[entering_var - num_cols_] = -1.0;
        }
        lu_.ftran(pivot_col);

        Real pivot_element = pivot_col[leaving_row];
        if (std::abs(pivot_element) < kPivotTol) {
            // Numerically degenerate pivot. Refactorize and retry.
            try {
                refactorize();
            } catch (const std::runtime_error&) {
                if (!recoverFromSingularBasis()) {
                    break;
                }
            }
            computePrimals();
            computeDuals();
            degenerate_pivot_streak = 0;
            continue;
        }

        // ---- UPDATE ----
        // Dual step size: delta_dual = sigma * (bound violation) / alpha_entering? No.
        // The dual step: theta_d = -rc_entering / alpha_entering (to drive rc_entering to 0
        // in the correct direction). Actually:
        // theta_d = best_ratio (the dual ratio test result).
        // But we need the sign right.
        //
        // Standard dual simplex update:
        // Primal step for leaving var:
        //   If below lower: leaving moves to lower bound. Step = (lower - val) / pivot_element.
        //   But pivot_element = (B^{-1} a_entering)[leaving_row].
        //
        // Let's use the direct approach:
        // theta_p = primal step = (target - leaving_val) / pivot_element
        // where target is the bound we're moving toward.
        Real theta_p = (leaving_val - target) / pivot_element;

        // Update primal values of basic variables.
        for (Index i = 0; i < num_rows_; ++i) {
            if (i != leaving_row) {
                primal_[basis_[i]] -= theta_p * pivot_col[i];
            }
        }
        primal_[leaving_var] = target;  // leaving goes to its bound

        // Entering variable primal: x_entering changes by theta_p.
        Real old_entering_val = primal_[entering_var];
        primal_[entering_var] = old_entering_val + theta_p;

        // If BFRT flipped variables, recompute primals after the basis swap.

        // Update reduced costs.
        Real theta_d = reduced_cost_[entering_var] / alpha_entering;
        for (Index k : nonbasic_) {
            if (k == entering_var) continue;
            Real alpha_k = pivot_row_alpha[k];
            if (std::abs(alpha_k) > kZeroTol) {
                reduced_cost_[k] -= theta_d * alpha_k;
            }
        }
        reduced_cost_[entering_var] = 0.0;  // entering becomes basic

        // Update dual values.
        axpyInPlace(std::span<Real>(dual_.data(), static_cast<std::size_t>(num_rows_)),
                    std::span<const Real>(work.data(),
                                          static_cast<std::size_t>(num_rows_)),
                    theta_d, options_);  // work still holds btran result

        // Reduced cost of leaving var (now nonbasic).
        // The leaving var was basic (rc=0). Its column through B^{-1} gives e_r.
        // So alpha_leaving_self = rho^T * a_leaving = 1.
        // rc_leaving_new = 0 - theta_d * 1 = -theta_d.
        reduced_cost_[leaving_var] = -theta_d;

        // ---- Swap basis ----
        // Entering goes into basis at leaving_row position.
        // Leaving goes nonbasic.
        basis_[leaving_row] = entering_var;
        basis_pos_[entering_var] = leaving_row;
        basis_pos_[leaving_var] = -1;

        var_status_[entering_var] = BasisStatus::Basic;

        // Set leaving variable status based on which bound it moved to.
        if (sigma > 0.0) {
            // Moved to lower bound.
            Real lb = varLower(leaving_var);
            Real ub = varUpper(leaving_var);
            if (lb != -kInf && ub != kInf && std::abs(lb - ub) < kZeroTol) {
                var_status_[leaving_var] = BasisStatus::Fixed;
            } else {
                var_status_[leaving_var] = BasisStatus::AtLower;
            }
        } else {
            Real lb = varLower(leaving_var);
            Real ub = varUpper(leaving_var);
            if (lb != -kInf && ub != kInf && std::abs(lb - ub) < kZeroTol) {
                var_status_[leaving_var] = BasisStatus::Fixed;
            } else {
                var_status_[leaving_var] = BasisStatus::AtUpper;
            }
        }

        // Update nonbasic list.
        // Remove entering_var, add leaving_var.
        Index nb_pos = nonbasic_pos_[entering_var];
        if (nb_pos >= 0) {
            nonbasic_[nb_pos] = leaving_var;
            nonbasic_pos_[leaving_var] = nb_pos;
            nonbasic_pos_[entering_var] = -1;
        }

        // ---- Update Devex weights ----
        {
            Real pivot_sq = pivot_col[leaving_row] * pivot_col[leaving_row];
            for (Index i = 0; i < num_rows_; ++i) {
                if (i == leaving_row) continue;
                Real ratio = pivot_col[i] / pivot_col[leaving_row];
                devex_weights_[i] = std::max(kDualTol,
                    devex_weights_[i] + ratio * ratio * devex_weights_[leaving_row]);
            }
            devex_weights_[leaving_row] = 1.0 / pivot_sq;
        }
        ++devex_reset_count_;
        if (devex_reset_count_ >= kDevexResetFreq) {
            std::fill(devex_weights_.begin(), devex_weights_.end(), 1.0);
            devex_reset_count_ = 0;
        }

        // ---- LU update ----
        // Reuse transformed pivot column and avoid a second FTRAN in LU update.
        lu_.updateFromFtranColumn(
            leaving_row,
            std::span<const Real>(pivot_col.data(),
                                  static_cast<std::size_t>(num_rows_)));

        if (std::abs(theta_p) <= options_.adaptive_refactor_degenerate_pivot_tol) {
            ++degenerate_pivot_streak;
        } else {
            degenerate_pivot_streak = 0;
        }

        // Check if refactorization needed.
        bool should_refactorize = lu_.needsRefactorization();
        if (!should_refactorize &&
            options_.enable_adaptive_refactorization &&
            degenerate_pivot_streak >= options_.adaptive_refactor_stall_pivots &&
            lu_.numUpdates() >= options_.adaptive_refactor_min_updates) {
            should_refactorize = true;
        }

        if (should_refactorize) {
            try {
                refactorize();
            } catch (const std::runtime_error&) {
                if (!recoverFromSingularBasis()) {
                    break;
                }
            }
            computePrimals();
            computeDuals();
            degenerate_pivot_streak = 0;
            resetPrimalFeasibleProgress();
            primal_feasible_pivots_since_refactor = 0;

            // Reset Devex weights after refactorization.
            std::fill(devex_weights_.begin(), devex_weights_.end(), 1.0);
            devex_reset_count_ = 0;

            // Re-apply cost shifts if still active.
            if (has_cost_shift) {
                computeDuals();
            }
        } else if (has_flips) {
            // BFRT flipped nonbasic variables, so the incremental primal
            // update is inexact. Recompute from scratch (one extra FTRAN).
            computePrimals();
        }

        ++iterations_;
    }

    if (iterations_ >= iter_limit_ &&
        status_ != Status::Optimal &&
        status_ != Status::Infeasible &&
        status_ != Status::TimeLimit) {
        status_ = Status::IterLimit;
    }

    clearBoundPerturbation();
    clearCostShifts();

    // Preserve warm-start basis only when solve did not end in internal error.
    has_basis_ = (status_ != Status::Error);

    // Final iteration log (only if it wouldn't duplicate a periodic line).
    if (verbose_ && (iterations_ % kLogFrequency) != 0) {
        Real obj_val = 0.0;
        for (Index k = 0; k < numVars(); ++k) {
            obj_val += varCost(k) * primal_[k];
        }
        Int pinf_count = 0;
        for (Index i = 0; i < num_rows_; ++i) {
            Index k = basis_[i];
            Real xk = primal_[k];
            Real lb = varLower(k);
            Real ub = varUpper(k);
            if (xk < lb - kPrimalTol || xk > ub + kPrimalTol) ++pinf_count;
        }
        double solve_elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - solve_start_).count();
        std::printf("%-7s  %9d  %20.10e  %11d  %7.2fs\n",
                    "Dual", iterations_, obj_val, pinf_count, solve_elapsed);
    }

    // Note: primals are kept in internal (scaled) coordinates.
    // Getters (getPrimalValues, getObjective, etc.) unscale on-the-fly.

    // Collect LU work accumulated during this solve.
    work_.add(lu_.workUnits());
    lu_.resetWorkUnits();

    return {status_, getObjective(), iterations_, work_.units() - work_at_start};
}

// ---------------------------------------------------------------------------
//  Getters
// ---------------------------------------------------------------------------

Real DualSimplexSolver::getObjective() const {
    // Primals are in internal (scaled) coordinates.
    // obj_internal[j] = obj_original[j] * col_scale[j] (negated if maximize).
    // primal_internal[j] = primal_external[j] / col_scale[j].
    // So obj_internal[j] * primal_internal[j] = obj_original[j] * primal_external[j]
    //   (with negation if maximize).
    Real obj = obj_offset_;
    for (Index j = 0; j < num_cols_; ++j) {
        obj += obj_[j] * primal_[j];
    }
    if (sense_ == Sense::Maximize) {
        obj = -obj;
    }
    return obj;
}

std::vector<Real> DualSimplexSolver::getPrimalValues() const {
    // Unscale: external = internal * col_scale.
    std::vector<Real> result(num_cols_);
    for (Index j = 0; j < num_cols_; ++j) {
        result[j] = primal_[j] * (scaled_ ? col_scale_[j] : 1.0);
    }
    return result;
}

std::vector<Real> DualSimplexSolver::getDualValues() const {
    // Unscale: dual_external[i] = dual_internal[i] * row_scale[i].
    std::vector<Real> result(num_rows_);
    for (Index i = 0; i < num_rows_; ++i) {
        result[i] = dual_[i] * (scaled_ ? row_scale_[i] : 1.0);
        if (sense_ == Sense::Maximize) result[i] = -result[i];
    }
    return result;
}

std::vector<Real> DualSimplexSolver::getReducedCosts() const {
    // Unscale: rc_external[j] = rc_internal[j] / col_scale[j].
    std::vector<Real> result(num_cols_);
    for (Index j = 0; j < num_cols_; ++j) {
        result[j] = reduced_cost_[j] / (scaled_ ? col_scale_[j] : 1.0);
        if (sense_ == Sense::Maximize) result[j] = -result[j];
    }
    return result;
}

std::vector<BasisStatus> DualSimplexSolver::getBasis() const {
    return std::vector<BasisStatus>(var_status_.begin(),
                                     var_status_.begin() + num_cols_ + num_rows_);
}

void DualSimplexSolver::setBasis(std::span<const BasisStatus> basis) {
    // External basis: first num_cols are structural, next num_rows are slacks.
    Index n = numVars();
    if (static_cast<Index>(basis.size()) != n) return;

    // Fast path: identical basis status vector already installed.
    if (has_basis_ && static_cast<Index>(var_status_.size()) == n &&
        std::equal(basis.begin(), basis.end(), var_status_.begin())) {
        return;
    }

    // Ensure solution vectors are allocated.
    var_status_.resize(n);
    primal_.resize(n, 0.0);
    dual_.resize(num_rows_, 0.0);
    reduced_cost_.resize(n, 0.0);
    devex_weights_.assign(num_rows_, 1.0);
    devex_reset_count_ = 0;

    basis_.clear();
    nonbasic_.clear();
    nonbasic_pos_.assign(n, -1);
    basis_pos_.assign(n, -1);

    for (Index k = 0; k < n; ++k) {
        var_status_[k] = basis[k];
        if (basis[k] == BasisStatus::Basic) {
            basis_pos_[k] = static_cast<Index>(basis_.size());
            basis_.push_back(k);
        } else {
            nonbasic_.push_back(k);
            nonbasic_pos_[k] = static_cast<Index>(nonbasic_.size() - 1);
            if (basis[k] == BasisStatus::AtLower || basis[k] == BasisStatus::Fixed) {
                primal_[k] = varLower(k);
            } else if (basis[k] == BasisStatus::AtUpper) {
                primal_[k] = varUpper(k);
            } else {
                primal_[k] = 0.0;
            }
        }
    }
    has_basis_ = true;
}

// ---------------------------------------------------------------------------
//  Incremental modifications (Step 8)
// ---------------------------------------------------------------------------

void DualSimplexSolver::setColBounds(Index col, Real lower, Real upper) {
    if (!loaded_ || col < 0 || col >= num_cols_) return;

    // Apply scaling: internal bounds = external / col_scale.
    Real scale = scaled_ ? col_scale_[col] : 1.0;
    Real int_lower = (lower == -kInf) ? -kInf : lower / scale;
    Real int_upper = (upper == kInf) ? kInf : upper / scale;

    col_lower_[col] = int_lower;
    col_upper_[col] = int_upper;

    // If we have a basis, update the nonbasic variable's status/value.
    if (has_basis_) {
        BasisStatus st = var_status_[col];
        if (st != BasisStatus::Basic) {
            if (int_lower != -kInf && int_upper != kInf &&
                std::abs(int_lower - int_upper) < kZeroTol) {
                var_status_[col] = BasisStatus::Fixed;
                primal_[col] = int_lower;
            } else if (st == BasisStatus::AtLower || st == BasisStatus::Fixed) {
                if (int_lower != -kInf) {
                    primal_[col] = int_lower;
                    var_status_[col] = BasisStatus::AtLower;
                } else if (int_upper != kInf) {
                    primal_[col] = int_upper;
                    var_status_[col] = BasisStatus::AtUpper;
                } else {
                    primal_[col] = 0.0;
                    var_status_[col] = BasisStatus::Free;
                }
            } else if (st == BasisStatus::AtUpper) {
                if (int_upper != kInf) {
                    primal_[col] = int_upper;
                } else if (int_lower != -kInf) {
                    primal_[col] = int_lower;
                    var_status_[col] = BasisStatus::AtLower;
                } else {
                    primal_[col] = 0.0;
                    var_status_[col] = BasisStatus::Free;
                }
            }
        }
        // If basic, the value may now violate bounds — dual simplex handles this.
    }
}

void DualSimplexSolver::getColBounds(Index col, Real& lower, Real& upper) const {
    if (!loaded_ || col < 0 || col >= num_cols_) {
        lower = -kInf;
        upper = kInf;
        return;
    }
    Real scale = scaled_ ? col_scale_[col] : 1.0;
    lower = (col_lower_[col] == -kInf) ? -kInf : col_lower_[col] * scale;
    upper = (col_upper_[col] == kInf) ? kInf : col_upper_[col] * scale;
}

void DualSimplexSolver::setObjective(std::span<const Real> obj) {
    if (!loaded_ || static_cast<Index>(obj.size()) != num_cols_) return;

    for (Index j = 0; j < num_cols_; ++j) {
        Real c = obj[j];
        if (sense_ == Sense::Maximize) c = -c;
        if (scaled_) c *= col_scale_[j];
        obj_[j] = c;
    }
}

void DualSimplexSolver::addRows(
    std::span<const Index> starts,
    std::span<const Index> indices,
    std::span<const Real> values,
    std::span<const Real> lower,
    std::span<const Real> upper) {
    if (!loaded_) return;

    Index num_new = static_cast<Index>(lower.size());
    if (num_new == 0) return;

    Index old_rows = num_rows_;
    Index old_vars = numVars();
    num_rows_ += num_new;

    // Extend row bounds.
    for (Index i = 0; i < num_new; ++i) {
        row_lower_.push_back(lower[i]);
        row_upper_.push_back(upper[i]);
    }

    // Extend scaling factors (new rows get scale = 1.0).
    if (scaled_) {
        row_scale_.resize(num_rows_, 1.0);
    }

    // Build new constraint matrix with added rows.
    std::vector<Triplet> triplets;
    triplets.reserve(matrix_.numNonzeros() + static_cast<Index>(values.size()));

    for (Index i = 0; i < old_rows; ++i) {
        auto rv = matrix_.row(i);
        for (Index k = 0; k < rv.size(); ++k) {
            triplets.push_back({i, rv.indices[k], rv.values[k]});
        }
    }

    for (Index i = 0; i < num_new; ++i) {
        Index row = old_rows + i;
        Index start = starts[i];
        Index end = (i + 1 < num_new) ? starts[i + 1]
                                      : static_cast<Index>(values.size());
        for (Index p = start; p < end; ++p) {
            // Apply column scaling to match internal matrix representation.
            // The input is in external space. Internal: a'_j = a_j * rs_i * cs_j.
            // For new rows, rs_i = 1.0, so internal = a_j * cs_j.
            Real val = values[p];
            if (scaled_ && indices[p] < num_cols_) {
                val *= col_scale_[indices[p]];
            }
            if (std::abs(val) > kZeroTol) {
                triplets.push_back({row, indices[p], val});
            }
        }
    }
    matrix_ = SparseMatrix(num_rows_, num_cols_, std::move(triplets));

    // Extend basis: new slack variables enter as basic.
    Index new_total = numVars();
    basis_pos_.resize(new_total, -1);
    nonbasic_pos_.resize(new_total, -1);
    var_status_.resize(new_total);
    primal_.resize(new_total);
    reduced_cost_.resize(new_total, 0.0);
    dual_.resize(num_rows_, 0.0);
    devex_weights_.resize(num_rows_, 1.0);

    for (Index i = 0; i < num_new; ++i) {
        Index slack = old_vars + i;
        basis_.push_back(slack);
        basis_pos_[slack] = old_rows + i;
        var_status_[slack] = BasisStatus::Basic;

        // Slack value = activity of new row in internal coords.
        // The new row in internal matrix: sum_j (a_j * cs_j) * x'_j.
        // This equals sum_j a_j * cs_j * (x_j / cs_j) = sum_j a_j * x_j = external activity.
        // But primal_[indices[p]] is in internal coords (x'_j), so:
        // activity = sum_j (a_j * cs_j) * x'_j.
        Real activity = 0.0;
        Index start = starts[i];
        Index end = (i + 1 < num_new) ? starts[i + 1]
                                      : static_cast<Index>(values.size());
        for (Index p = start; p < end; ++p) {
            Real val = values[p];
            if (scaled_ && indices[p] < num_cols_) {
                val *= col_scale_[indices[p]];
            }
            activity += val * primal_[indices[p]];
        }
        primal_[slack] = activity;
    }

    has_basis_ = true;
}

void DualSimplexSolver::removeRows(std::span<const Index> rows) {
    if (!loaded_ || rows.empty()) return;

    Index old_rows = num_rows_;
    Index old_vars = numVars();

    std::vector<Index> sorted_rows(rows.begin(), rows.end());
    std::sort(sorted_rows.begin(), sorted_rows.end());

    // Build row mapping: old row -> new row (-1 if removed).
    std::vector<Index> row_map(num_rows_, 0);
    Index removed = 0;
    Index ri = 0;
    for (Index i = 0; i < num_rows_; ++i) {
        if (ri < static_cast<Index>(sorted_rows.size()) &&
            sorted_rows[ri] == i) {
            row_map[i] = -1;
            ++removed;
            ++ri;
        } else {
            row_map[i] = i - removed;
        }
    }

    Index new_rows = num_rows_ - removed;

    // Rebuild matrix without removed rows.
    std::vector<Triplet> triplets;
    triplets.reserve(matrix_.numNonzeros());
    for (Index i = 0; i < num_rows_; ++i) {
        if (row_map[i] < 0) continue;
        auto rv = matrix_.row(i);
        for (Index k = 0; k < rv.size(); ++k) {
            triplets.push_back({row_map[i], rv.indices[k], rv.values[k]});
        }
    }

    // Compact row bounds and scaling.
    std::vector<Real> new_row_lower, new_row_upper;
    std::vector<Real> new_row_scale;
    new_row_lower.reserve(new_rows);
    new_row_upper.reserve(new_rows);
    if (scaled_) new_row_scale.reserve(new_rows);

    for (Index i = 0; i < num_rows_; ++i) {
        if (row_map[i] < 0) continue;
        new_row_lower.push_back(row_lower_[i]);
        new_row_upper.push_back(row_upper_[i]);
        if (scaled_) new_row_scale.push_back(row_scale_[i]);
    }

    num_rows_ = new_rows;
    row_lower_ = std::move(new_row_lower);
    row_upper_ = std::move(new_row_upper);
    if (scaled_) row_scale_ = std::move(new_row_scale);
    matrix_ = SparseMatrix(num_rows_, num_cols_, std::move(triplets));

    // Try to preserve basis via slack index remapping.
    if (has_basis_) {
        // old var index -> new var index (-1 if removed)
        std::vector<Index> var_map(old_vars, -1);
        for (Index j = 0; j < num_cols_; ++j) var_map[j] = j;
        for (Index i = 0; i < old_rows; ++i) {
            Index old_slack = num_cols_ + i;
            if (row_map[i] >= 0) {
                var_map[old_slack] = num_cols_ + row_map[i];
            }
        }

        bool can_preserve = true;
        std::vector<Index> new_basis(static_cast<std::size_t>(new_rows), -1);
        if (static_cast<Index>(basis_.size()) != old_rows) {
            can_preserve = false;
        } else {
            for (Index old_row = 0; old_row < old_rows; ++old_row) {
                Index new_row = row_map[old_row];
                if (new_row < 0) continue;
                Index old_var = basis_[old_row];
                if (old_var < 0 || old_var >= old_vars) {
                    can_preserve = false;
                    break;
                }
                Index new_var = var_map[old_var];
                if (new_var < 0) {
                    can_preserve = false;
                    break;
                }
                new_basis[new_row] = new_var;
            }
            for (Index i = 0; i < new_rows; ++i) {
                if (new_basis[i] < 0) {
                    can_preserve = false;
                    break;
                }
            }
        }

        if (can_preserve) {
            Index new_vars = numVars();
            std::vector<BasisStatus> new_var_status(static_cast<std::size_t>(new_vars),
                                                    BasisStatus::AtLower);
            std::vector<Real> new_primal(static_cast<std::size_t>(new_vars), 0.0);
            std::vector<Real> new_reduced(static_cast<std::size_t>(new_vars), 0.0);
            std::vector<Index> new_basis_pos(static_cast<std::size_t>(new_vars), -1);
            std::vector<Index> new_nonbasic;
            new_nonbasic.reserve(static_cast<std::size_t>(new_vars - new_rows));

            // Transfer mapped solution/status values.
            for (Index old_var = 0; old_var < old_vars; ++old_var) {
                Index new_var = var_map[old_var];
                if (new_var < 0) continue;
                if (old_var < static_cast<Index>(var_status_.size())) {
                    new_var_status[new_var] = var_status_[old_var];
                }
                if (old_var < static_cast<Index>(primal_.size())) {
                    new_primal[new_var] = primal_[old_var];
                }
                if (old_var < static_cast<Index>(reduced_cost_.size())) {
                    new_reduced[new_var] = reduced_cost_[old_var];
                }
            }

            // Enforce basis rows and rebuild basis_pos.
            for (Index i = 0; i < new_rows; ++i) {
                Index v = new_basis[i];
                new_basis_pos[v] = i;
                new_var_status[v] = BasisStatus::Basic;
                new_reduced[v] = 0.0;
            }

            // Rebuild nonbasic list and sanitize statuses.
            for (Index v = 0; v < new_vars; ++v) {
                if (new_basis_pos[v] >= 0) continue;
                BasisStatus st = new_var_status[v];
                if (st == BasisStatus::Basic) {
                    Real lb = varLower(v);
                    Real ub = varUpper(v);
                    if (lb != -kInf && ub != kInf && std::abs(lb - ub) < kZeroTol) {
                        st = BasisStatus::Fixed;
                        new_primal[v] = lb;
                    } else if (lb != -kInf) {
                        st = BasisStatus::AtLower;
                        new_primal[v] = lb;
                    } else if (ub != kInf) {
                        st = BasisStatus::AtUpper;
                        new_primal[v] = ub;
                    } else {
                        st = BasisStatus::Free;
                        new_primal[v] = 0.0;
                    }
                    new_var_status[v] = st;
                }
                new_nonbasic.push_back(v);
            }

            basis_ = std::move(new_basis);
            basis_pos_ = std::move(new_basis_pos);
            var_status_ = std::move(new_var_status);
            primal_ = std::move(new_primal);
            reduced_cost_ = std::move(new_reduced);
            nonbasic_ = std::move(new_nonbasic);
            nonbasic_pos_.assign(static_cast<std::size_t>(new_vars), -1);
            for (Index pos = 0; pos < static_cast<Index>(nonbasic_.size()); ++pos) {
                nonbasic_pos_[nonbasic_[pos]] = pos;
            }
            dual_.assign(static_cast<std::size_t>(num_rows_), 0.0);
            devex_weights_.assign(static_cast<std::size_t>(num_rows_), 1.0);
            devex_reset_count_ = 0;
            has_basis_ = true;
            return;
        }
    }

    // Could not preserve basis safely.
    has_basis_ = false;
    nonbasic_pos_.clear();
}

// ---------------------------------------------------------------------------
//  getTableauRow — for cut generation
// ---------------------------------------------------------------------------

void DualSimplexSolver::getTableauRow(Index basis_pos,
                                       std::vector<Real>& tableau_row) {
    Index nv = numVars();
    tableau_row.assign(static_cast<std::size_t>(nv), 0.0);

    // BTRAN: compute rho = B^{-T} * e_{basis_pos}.
    std::vector<Real> rho(static_cast<std::size_t>(num_rows_), 0.0);
    rho[basis_pos] = 1.0;
    lu_.btran(rho);

    // Compute tableau row: alpha_j = rho^T * a_j for each variable j.
    // Structural: alpha_j = rho^T * A_col_j.
    // Slack (n+i): alpha_{n+i} = rho^T * (-e_i) = -rho[i].

    // Row-wise computation (faster when rho is dense):
    for (Index i = 0; i < num_rows_; ++i) {
        Real rho_i = rho[i];
        if (std::abs(rho_i) < kZeroTol) continue;
        auto rv = matrix_.row(i);
        for (Index p = 0; p < rv.size(); ++p) {
            tableau_row[rv.indices[p]] += rho_i * rv.values[p];
        }
    }

    // Slack entries.
    for (Index i = 0; i < num_rows_; ++i) {
        tableau_row[num_cols_ + i] = -rho[i];
    }

    // Convert to external (unscaled) coordinates.
    // Internal: x'_bvar + sum_j alpha_j^int * x'_j = b^int
    // External: x_bvar + sum_j alpha_j^ext * x_j = b^ext
    //
    // Relations: x'_j = x_j / cs_j (structural), s'_k = s_k * rs_k (slack)
    // x'_bvar = x_bvar / cs_bvar (assuming bvar is structural)
    //
    // Substituting into internal equation and multiplying by cs_bvar:
    //   x_bvar + sum_j (alpha_j^int * cs_bvar / cs_j) * x_j
    //          + sum_k (alpha_{n+k}^int * cs_bvar / (1/rs_k)) * ... = b^ext
    //
    // More carefully for slacks: s'_k = s_k * rs_k, so x'_{n+k} = s_k * rs_k
    //   alpha_{n+k}^int * s'_k = alpha_{n+k}^int * rs_k * s_k
    //   = alpha_{n+k}^int * cs_bvar * rs_k * s_k / cs_bvar
    // After multiplying by cs_bvar:
    //   alpha_{n+k}^ext * s_k = alpha_{n+k}^int * cs_bvar * rs_k * s_k
    // So alpha_{n+k}^ext = alpha_{n+k}^int * cs_bvar * rs_k
    if (scaled_) {
        Index bvar = basis_[basis_pos];
        Real cs_bvar = (bvar < num_cols_) ? col_scale_[bvar] : 1.0;
        // For slacks as basic variable: cs_bvar should be 1/rs_k
        // Actually, the basic variable could be a slack. In that case,
        // x'_{n+k} = s_k * rs_k, so the "scale" for the basic slack is rs_k.
        // The equation becomes: s'_k + sum = b^int
        // In external: s_k * rs_k + sum = b^int
        // s_k + (1/rs_k) * sum = b^int / rs_k = b^ext
        // So the "multiplier" is 1/rs_k for a slack basic variable.
        if (bvar >= num_cols_) {
            Index ridx = bvar - num_cols_;
            cs_bvar = 1.0 / row_scale_[ridx];
        }

        for (Index j = 0; j < num_cols_; ++j) {
            tableau_row[j] *= cs_bvar / col_scale_[j];
        }
        for (Index k = 0; k < num_rows_; ++k) {
            tableau_row[num_cols_ + k] *= cs_bvar * row_scale_[k];
        }
    }
}

}  // namespace mipx
