#pragma once

#include "mipx/core.h"
#include "mipx/cut_pool.h"
#include "mipx/dual_simplex.h"
#include "mipx/lp_problem.h"

#include <algorithm>
#include <cmath>
#include <span>
#include <vector>

/// Shared machinery for cuts read off a simplex tableau row: Gomory,
/// intersection and multi-row cuts.
///
/// Each of them works over the deviation form of one or more tableau rows,
///
///   x_i + sum_k t_k * delta_k = b,   delta_k >= 0,
///
/// derives a cut sum_k coeff_k * delta_k >= 1 by a family-specific rule, and
/// substitutes the deviations back into structural columns. The substitution,
/// and with it the question of where the cut is valid, is the same for every
/// family and lives here: a structural deviation is x_k - l_k or u_k - x_k
/// measured from the bound the LP holds the column at, and the deviation of
/// the logical of row r is a_r^T x - lower_r or upper_r - a_r^T x. A cut is
/// local when a column's bound is tighter than the model's, or when it
/// substitutes the logical of a row that is not valid tree-wide.
namespace mipx::tableau_detail {

/// Tolerance for calling model data (a bound, a matrix entry) integral.
/// Tighter than the integrality tolerance used on solution values: this
/// decides whether a deviation variable may be treated as integer in the GMI
/// formula, and a wrong answer there produces an invalid cut.
constexpr Real kDataIntTol = 1e-9;

/// Relative size, against the magnitude of the terms summed into it, below
/// which a cut coefficient is cancellation round-off (a few thousand ulps).
constexpr Real kCancellationTol = 1e-12;

inline bool isIntegralValue(Real v, Real tol) {
    return std::abs(v - std::round(v)) <= tol;
}

/// True when a column is constrained to integer values. Continuous is the
/// obvious exclusion; SemiContinuous is the trap, because it is not
/// VarType::Continuous yet ranges over {0} U [l, u]. Treating it as integral
/// would let a basic semi-continuous value at 2.7 open a tableau row and give
/// a nonbasic one the integer coefficient formula, both invalid. MipSolver
/// linearizes these away before separation, but the separators are public API.
inline bool isIntegerValued(VarType t) {
    return t == VarType::Integer || t == VarType::Binary || t == VarType::SemiInteger;
}

/// Scratch buffers for the row of a substituted logical, reused across rows.
struct RowScratch {
    std::vector<Index> indices;
    std::vector<Real> values;
};

/// Which LP rows a cut may lean on and stay valid tree-wide.
struct RowProvenance {
    Index global_rows = 0;                   // rows at or past this are local
    std::span<const Index> local_rows = {};  // sorted; local inside the prefix

    [[nodiscard]] bool isLocal(Index row) const {
        return row >= global_rows || std::binary_search(local_rows.begin(), local_rows.end(), row);
    }
};

/// The provenance of the LP's rows as a caller declared it. A negative
/// @p global_row_count means "only the rows of @p problem", the safe reading
/// when the caller says nothing; both readings are clamped to the rows the LP
/// holds, since a problem carrying more rows than that would otherwise make
/// every logical look global.
RowProvenance makeRowProvenance(const DualSimplexSolver& lp, const LpProblem& problem,
                                Index global_row_count, std::span<const Index> local_rows);

/// Column bounds a dropped coefficient is paid for over.
struct ColumnBox {
    std::span<const Real> global_lower;
    std::span<const Real> global_upper;
    std::vector<Real> lp_lower;
    std::vector<Real> lp_upper;
};

ColumnBox makeColumnBox(const DualSimplexSolver& lp, const LpProblem& problem);

/// One cut under construction: sum_j coeff[j] * x_j >= rhs, plus whether it
/// stayed global.
struct CutBuilder {
    std::vector<Real> coeff;
    std::vector<Real> magnitude;  // sum of |term| that went into each coeff
    Real rhs = 1.0;
    bool local = false;

    void reset(Index num_cols) {
        coeff.assign(static_cast<std::size_t>(num_cols), 0.0);
        magnitude.assign(static_cast<std::size_t>(num_cols), 0.0);
        rhs = 1.0;
        local = false;
    }

    void add(Index j, Real term) {
        coeff[static_cast<std::size_t>(j)] += term;
        magnitude[static_cast<std::size_t>(j)] += std::abs(term);
    }
};

/// How the deviation of one nonbasic variable reads in x.
struct Deviation {
    /// +1 at a lower bound (delta = v - bound), -1 at an upper bound
    /// (delta = bound - v). A tableau entry alpha becomes t = sign * alpha.
    Real sign = 1.0;
    Real bound = 0.0;
    /// delta is integer at every integer-feasible point.
    bool integral = false;
    /// The derivation leans on a bound or row that holds only in this subtree.
    bool local = false;
};

/// Describe the deviation of nonbasic variable @p k (a structural column when
/// k < num_cols, else the logical of row k - num_cols). For a logical, the
/// row is left in @p scratch for applyDeviation. Returns false when delta_k
/// has no finite reference bound or is not sign-constrained, which makes the
/// whole tableau row unusable.
bool describeDeviation(const LpProblem& problem, const DualSimplexSolver& lp, Index k,
                       BasisStatus status, const RowProvenance& provenance, RowScratch& scratch,
                       Deviation& dev);

/// Add coeff * delta_k to the cut. Returns false when a substituted row
/// reaches outside the structural columns.
bool applyDeviation(Index k, Index num_cols, const Deviation& dev, Real coeff,
                    const RowScratch& scratch, CutBuilder& builder);

/// Accumulate the term of every nonbasic variable of a tableau row (or of a
/// set of rows read together). @p present(k) says whether variable k appears
/// in the row(s); @p coefficient(k, sign, integral) gives its deviation-space
/// cut coefficient, non-finite to reject the row. A present variable makes
/// the cut local through its bound or row even when its coefficient is zero:
/// the derivation needs delta_k >= 0 for every variable of the row.
/// Returns false when a term cannot be formed, which makes the row unusable.
template <class Present, class Coefficient>
bool accumulateTerms(const LpProblem& problem, const DualSimplexSolver& lp,
                     std::span<const BasisStatus> basis, Present&& present,
                     Coefficient&& coefficient, const RowProvenance& provenance,
                     RowScratch& scratch, CutBuilder& builder) {
    const Index num_cols = problem.num_cols;
    const auto total_vars = static_cast<Index>(basis.size());
    for (Index k = 0; k < total_vars; ++k) {
        if (basis[k] == BasisStatus::Basic || !present(k)) {
            continue;
        }
        Deviation dev;
        if (!describeDeviation(problem, lp, k, basis[k], provenance, scratch, dev)) {
            return false;
        }
        if (dev.local) {
            builder.local = true;
        }
        const Real coeff = coefficient(k, dev.sign, dev.integral);
        if (!std::isfinite(coeff)) {
            return false;
        }
        if (!applyDeviation(k, num_cols, dev, coeff, scratch, builder)) {
            return false;
        }
    }
    return std::isfinite(builder.rhs);
}

/// Turn the accumulated coefficients into a sparse cut. Entries at or below
/// the coefficient tolerance are dropped and paid for in the right-hand side
/// over the column box the cut is valid in: the model's bounds for a global
/// cut, the LP's (node) bounds for a local one, which are never looser.
/// Returns false when nothing of substance is left.
bool finalizeCut(CutBuilder& builder, const ColumnBox& box, Real coeff_tol, Cut& cut, Real& norm_sq,
                 Int& relaxed_terms);

/// A basic integer variable sitting at a fractional value: one candidate row.
struct Candidate {
    Index basis_pos;
    Index col;
    Real frac;
};

/// Basic integer variables with a fractional value, most fractional first.
/// Ordering by distance from 0.5 puts the rows with the strongest cuts (the
/// largest f0-driven violation) at the front of the budget.
std::vector<Candidate> collectCandidates(const DualSimplexSolver& lp, const LpProblem& problem,
                                         std::span<const Real> primals, Real int_tol);

}  // namespace mipx::tableau_detail
