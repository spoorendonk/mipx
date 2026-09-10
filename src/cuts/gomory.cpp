#include "mipx/gomory.h"

#include "mipx/branching.h"
#include "mipx/lp_problem.h"

#include <algorithm>
#include <cmath>
#include <span>
#include <vector>

namespace mipx {

namespace {

/// Tolerance for calling model data (a bound, a matrix entry) integral.
/// Tighter than the integrality tolerance used on solution values: this
/// decides whether a deviation variable may be treated as integer in the GMI
/// formula, and a wrong answer there produces an invalid cut.
constexpr Real kDataIntTol = 1e-9;

bool isIntegralValue(Real v, Real tol) {
    return std::abs(v - std::round(v)) <= tol;
}

/// True when a column is constrained to integer values. Continuous is the
/// obvious exclusion; SemiContinuous is the trap, because it is not
/// VarType::Continuous yet ranges over {0} U [l, u]. Treating it as integral
/// would let a basic semi-continuous value at 2.7 open a Gomory row and give a
/// nonbasic one the integer coefficient formula, both invalid. MipSolver
/// linearizes these away before separation, but the separator is public API.
bool isIntegerValued(VarType t) {
    return t == VarType::Integer || t == VarType::Binary || t == VarType::SemiInteger;
}

/// GMI coefficient of a deviation variable.
///
/// The source row is x_i + sum_k t_k * delta_k = b with delta_k >= 0 and x_i
/// integer; f0 = b - floor(b) is in (0, 1). The Gomory mixed-integer cut is
/// sum_k coeff(t_k) * delta_k >= 1 with
///
///   delta_k integer:  coeff = f_k / f0            if f_k <= f0
///                     coeff = (1 - f_k)/(1 - f0)  otherwise,  f_k = frac(t_k)
///   delta_k real:     coeff = t_k / f0            if t_k > 0
///                     coeff = -t_k / (1 - f0)     if t_k < 0
///
/// Treating an integer deviation as real is valid but weaker, so `integral`
/// may be set only when delta_k really is integer at every feasible point.
Real gmiCoefficient(Real t, bool integral, Real f0) {
    if (integral) {
        Real f = t - std::floor(t);  // in [0, 1) for negative t as well
        // An integral t up to round-off contributes nothing: the term is an
        // integer multiple of an integer deviation. Snapping both sides keeps
        // a 1e-10 tableau residual from entering the cut as a coefficient the
        // max/min ratio screen would then reject the whole cut over.
        if (f > 1.0 - kDataIntTol || f < kDataIntTol) {
            f = 0.0;
        }
        return (f <= f0) ? f / f0 : (1.0 - f) / (1.0 - f0);
    }
    if (t > 0.0) {
        return t / f0;
    }
    return -t / (1.0 - f0);
}

/// True when an LP bound is strictly tighter than the corresponding global
/// bound, i.e. the deviation measured from it is only valid in this subtree.
///
/// The comparison is exact, deliberately. A tolerance here would have to be
/// scaled against something, and the only safe scale is not the bound: getting
/// this wrong marks a locally-derived cut global, and the resulting rhs error
/// is the bound difference times the emitted coefficient, which the safety
/// screen caps at 1e6 rather than at anything proportional to the bound. A
/// bound-magnitude tolerance therefore admits an unsound promotion on exactly
/// the columns where it costs most: a continuous column with a global upper of
/// 1e6 tightened by node propagation to 1e6 - 1e-4 reads as "not local" under
/// a relative 1e-9, and a cut coefficient of that order puts the rhs out by
/// ~1e2. Comparing exactly can only err the other way, marking an untightened
/// bound local, which loses a cut and never validity.
bool boundIsLocal(Real bound, Real global_bound, Real sign) {
    if (!std::isfinite(global_bound)) {
        return true;
    }
    return (sign > 0.0) ? (bound > global_bound) : (bound < global_bound);
}

/// Scratch buffers for the row of a substituted logical, reused across rows.
struct RowScratch {
    std::vector<Index> indices;
    std::vector<Real> values;
};

/// True when the activity a^T x of the given row is integer at every
/// integer-feasible point: integer coefficients on integer columns only.
bool rowActivityIsIntegral(const LpProblem& problem, const RowScratch& row) {
    for (std::size_t p = 0; p < row.indices.size(); ++p) {
        const Index j = row.indices[p];
        if (j < 0 || j >= problem.num_cols) {
            return false;
        }
        if (!isIntegerValued(problem.col_type[j])) {
            return false;
        }
        if (!isIntegralValue(row.values[p], kDataIntTol)) {
            return false;
        }
    }
    return true;
}

/// One GMI term under construction: the cut being accumulated in the form
/// sum_j cut_coeff[j] * x_j >= cut_rhs, plus whether it stayed global.
struct CutBuilder {
    std::vector<Real> coeff;
    Real rhs = 1.0;
    bool local = false;
};

/// Add the term of nonbasic structural column `k` to the cut.
///
/// The deviation is delta = x_k - l_k when the column sits at its lower bound
/// and delta = u_k - x_k when it sits at its upper bound, so the tableau
/// coefficient in deviation form is t = alpha for the former and t = -alpha
/// for the latter. Substituting delta back is the same expression with the
/// sign carried through. Returns false when delta_k has no finite reference
/// bound, which makes the whole row unusable.
bool addStructuralTerm(const LpProblem& problem, const DualSimplexSolver& lp, Index k,
                       BasisStatus status, Real alpha, Real f0, Real coeff_tol,
                       CutBuilder& builder) {
    Real lp_lower = 0.0;
    Real lp_upper = 0.0;
    lp.getColBounds(k, lp_lower, lp_upper);

    Real sign = 1.0;
    Real bound = lp_lower;
    Real global_bound = problem.col_lower[k];
    if (status == BasisStatus::AtUpper) {
        sign = -1.0;
        bound = lp_upper;
        global_bound = problem.col_upper[k];
    } else if (status != BasisStatus::AtLower && status != BasisStatus::Fixed) {
        return false;  // nonbasic free: delta is not sign-constrained
    }
    if (!std::isfinite(bound)) {
        return false;
    }

    // The derivation needs delta_k >= 0 for every column in the row, not only
    // for the ones that end up with a nonzero cut coefficient, so the bound is
    // recorded as local before the coefficient is even looked at -- including
    // for the negligible entry dropped just below.
    if (boundIsLocal(bound, global_bound, sign)) {
        builder.local = true;
    }
    if (std::abs(alpha) < coeff_tol) {
        return true;  // negligible tableau entry: no term, bound still recorded
    }

    const Real t = sign * alpha;
    const bool integral =
        isIntegerValued(problem.col_type[k]) && isIntegralValue(bound, kDataIntTol);
    const Real coeff = gmiCoefficient(t, integral, f0);
    if (std::abs(coeff) < coeff_tol) {
        return true;
    }

    builder.coeff[static_cast<std::size_t>(k)] += sign * coeff;
    builder.rhs += sign * coeff * bound;
    return true;
}

/// Add the term of the nonbasic logical of row `row` to the cut.
///
/// The logical is the row activity s = a^T x bounded by the row's own bounds,
/// so the deviation is s - lower when it sits at the row's lower bound and
/// upper - s when it sits at the row's upper bound. That covers <= rows (only
/// an upper bound, so the logical can only sit there), >= rows, ranged rows
/// (either bound) and equality rows (Fixed, handled as the lower bound, which
/// equals the upper one). Substituting spreads the term over the row's
/// structural columns, which is what makes the cut expressible in x.
/// Returns false when the row cannot be substituted.
bool addLogicalTerm(const LpProblem& problem, const DualSimplexSolver& lp, Index row,
                    BasisStatus status, Real alpha, Real f0, Real coeff_tol, Index global_rows,
                    RowScratch& scratch, CutBuilder& builder) {
    // As for a structural column: delta >= 0 rests on the row holding, so a row
    // the caller has not vouched for makes the cut local whatever coefficient
    // comes out -- including none at all, when the entry below is negligible.
    if (row >= global_rows) {
        builder.local = true;
    }
    if (std::abs(alpha) < coeff_tol) {
        return true;  // negligible tableau entry: no term, row still recorded
    }

    Real row_lower = 0.0;
    Real row_upper = 0.0;
    lp.getRowExternal(row, scratch.indices, scratch.values, row_lower, row_upper);

    Real sign = 1.0;
    Real bound = row_lower;
    if (status == BasisStatus::AtUpper) {
        sign = -1.0;
        bound = row_upper;
    } else if (status != BasisStatus::AtLower && status != BasisStatus::Fixed) {
        return false;  // free row logical: delta is not sign-constrained
    }
    if (!std::isfinite(bound)) {
        return false;
    }

    const Real t = sign * alpha;
    const bool integral =
        isIntegralValue(bound, kDataIntTol) && rowActivityIsIntegral(problem, scratch);
    const Real coeff = gmiCoefficient(t, integral, f0);
    if (std::abs(coeff) < coeff_tol) {
        return true;
    }

    for (std::size_t p = 0; p < scratch.indices.size(); ++p) {
        const Index j = scratch.indices[p];
        if (j < 0 || j >= problem.num_cols) {
            return false;
        }
        builder.coeff[static_cast<std::size_t>(j)] += sign * coeff * scratch.values[p];
    }
    builder.rhs += sign * coeff * bound;
    return true;
}

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
                                         std::span<const Real> primals, Real int_tol) {
    std::vector<Candidate> candidates;
    for (Index j = 0; j < problem.num_cols; ++j) {
        if (!isIntegerValued(problem.col_type[j])) {
            continue;
        }
        const Index bpos = lp.basisPosition(j);
        if (bpos < 0) {
            continue;
        }
        const Real frac = fractionality(primals[j]);
        if (frac > int_tol && frac < 1.0 - int_tol) {
            candidates.push_back({bpos, j, frac});
        }
    }
    std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
        return std::abs(a.frac - 0.5) < std::abs(b.frac - 0.5);
    });
    return candidates;
}

/// Accumulate the GMI term of every nonbasic variable of one tableau row.
/// Returns false when a term cannot be formed, which makes the row unusable.
bool accumulateTerms(const LpProblem& problem, const DualSimplexSolver& lp,
                     std::span<const BasisStatus> basis, std::span<const Real> tab_row, Real f0,
                     Real coeff_tol, Index global_rows, RowScratch& scratch, CutBuilder& builder) {
    const Index num_cols = problem.num_cols;
    const auto total_vars = static_cast<Index>(tab_row.size());
    for (Index k = 0; k < total_vars; ++k) {
        if (basis[k] == BasisStatus::Basic) {
            continue;
        }
        // Only an exactly-zero entry means the variable is absent from the row.
        // A small-but-nonzero one still leans on delta_k >= 0, so it is handed
        // to the term builders, which record the bound and then drop it.
        const Real alpha = tab_row[k];
        if (alpha == 0.0) {
            continue;
        }
        const bool ok =
            (k < num_cols)
                ? addStructuralTerm(problem, lp, k, basis[k], alpha, f0, coeff_tol, builder)
                : addLogicalTerm(problem, lp, k - num_cols, basis[k], alpha, f0, coeff_tol,
                                 global_rows, scratch, builder);
        if (!ok) {
            return false;
        }
    }
    return std::isfinite(builder.rhs);
}

/// Turn the accumulated coefficients into a sparse cut, dropping entries below
/// the coefficient tolerance. Returns false when nothing of substance is left.
bool finalizeCut(const CutBuilder& builder, Index num_cols, Real coeff_tol, Cut& cut,
                 Real& norm_sq) {
    norm_sq = 0.0;
    for (Index j = 0; j < num_cols; ++j) {
        const Real value = builder.coeff[static_cast<std::size_t>(j)];
        if (std::abs(value) > coeff_tol) {
            cut.indices.push_back(j);
            cut.values.push_back(value);
            norm_sq += value * value;
        }
    }
    return !cut.indices.empty() && norm_sq >= coeff_tol;
}

}  // namespace

/// Generate Gomory mixed-integer (GMI) cuts from the simplex tableau.
///
/// Works in external (unscaled) space. The tableau row from getTableauRow is
/// already unscaled, and the cut coefficients are in external space. They are
/// passed to addRows which handles the column scaling internally.
///
/// For a basic integer variable x_i with value b, the tableau row over the
/// nonbasic variables (structural columns and row logicals alike) is
///
///   x_i + sum_k alpha_k * x_k = 0.
///
/// The right-hand side is zero rather than b because each row's own logical is
/// a variable of the system (s_i = a_i^T x, bounded by the row bounds), so the
/// model's right-hand sides sit in the logicals' bounds instead.
///
/// Every nonbasic x_k sits at one of its bounds, so subtracting the same
/// identity at the current point turns it into the deviation form
///
///   x_i + sum_k t_k * delta_k = b,   delta_k >= 0,
///
/// with t_k = alpha_k at a lower bound and t_k = -alpha_k at an upper bound.
/// The GMI cut over the deviations, sum_k coeff_k * delta_k >= 1, is then
/// substituted back into x: a structural deviation is x_k - l_k or u_k - x_k,
/// and the deviation of the logical of row r is a_r^T x - lower_r or
/// upper_r - a_r^T x, since the logical of a row is that row's activity.
///
/// Substituting the logicals is what makes the cut expressible in structural
/// columns. Skipping the rows that contain them instead -- what this separator
/// did before issue #211 -- discards every candidate row there is: row p of
/// B^-1 is nonzero somewhere, and it is zero on every basic logical other than
/// its own, so a row whose basic variable is a structural column always has a
/// nonbasic logical with a nonzero coefficient.
Int GomorySeparator::separate(DualSimplexSolver& lp, const LpProblem& problem,
                              std::span<const Real> primals, CutPool& pool) {
    stats_ = GomoryStats{};

    Int num_cuts = 0;
    const Index num_cols = problem.num_cols;
    // The tableau is indexed by the LP's own columns; a problem that is not the
    // one the LP holds would make every index below mean something else.
    if (lp.numCols() != num_cols || static_cast<Index>(primals.size()) < num_cols) {
        return 0;
    }
    const Index num_rows = lp.numRows();
    const Index total_vars = num_cols + num_rows;

    // Rows the caller has not vouched for are treated as node-local: a cut
    // that substitutes their logical is only valid where they are.
    const Index global_rows =
        (global_row_count_ < 0) ? problem.num_rows : std::min(global_row_count_, num_rows);

    auto basis = lp.getBasis();
    const auto candidates = collectCandidates(lp, problem, primals, kIntTol);
    const auto max_try =
        std::min(static_cast<Index>(candidates.size()), static_cast<Index>(max_cuts_ * 2));

    std::vector<Real> tab_row(static_cast<std::size_t>(total_vars));
    RowScratch scratch;
    CutBuilder builder;

    for (Index ci = 0; ci < max_try && num_cuts < max_cuts_; ++ci) {
        const Real basic_value = primals[candidates[ci].col];
        const Real f0 = basic_value - std::floor(basic_value);
        if (f0 < kIntTol || f0 > 1.0 - kIntTol) {
            continue;
        }
        ++stats_.candidate_rows;

        // The external (unscaled) tableau row of this basic variable.
        lp.getTableauRow(candidates[ci].basis_pos, tab_row);

        builder.coeff.assign(static_cast<std::size_t>(num_cols), 0.0);
        builder.rhs = 1.0;
        builder.local = false;
        if (!accumulateTerms(problem, lp, basis, tab_row, f0, kCoeffTol, global_rows, scratch,
                             builder)) {
            ++stats_.rows_skipped;
            continue;
        }

        Cut cut;
        Real norm_sq = 0.0;
        if (!finalizeCut(builder, num_cols, kCoeffTol, cut, norm_sq)) {
            ++stats_.rejected_empty;
            continue;
        }
        ++stats_.cuts_built;

        cut.lower = builder.rhs;
        cut.upper = kInf;
        cut.family = CutFamily::Gomory;
        cut.local = builder.local;

        // Same numerical safety screen every other family gets through
        // SeparatorManager::addViolatedCut. No check is skipped: the ones the
        // construction above already guarantees (sorted indices, |coef| above
        // kCoeffTol, finite rhs, non-empty support, norm_sq above kCoeffTol)
        // are cheap to re-test, and the ones it does not guarantee -- a maximum
        // coefficient above 1e6, a max/min coefficient ratio above 1e8, and a
        // squared norm above 1e16 -- are exactly why the screen is here. A GMI
        // coefficient is t/f0 (or fj/f0) on a nonbasic column, so a basic
        // variable whose fractional part f0 sits just above kIntTol blows the
        // coefficient up without any of the local checks noticing.
        //
        // Note the screen is stricter than the sparsification in finalizeCut:
        // it drops the whole cut when max|coef| / min|coef| exceeds 1e8, while
        // finalizeCut keeps every coefficient above kCoeffTol = 1e-10. A cut
        // carrying an O(1) coefficient next to a 5e-10 cancellation residual is
        // therefore rejected outright -- and substituting a logical back over a
        // whole row is a ready source of such residuals, so stats()
        // .rejected_screen is worth watching. That is still deliberate: Gomory
        // gets the same screen as everyone else, and a Gomory-only exemption
        // would put the ill-conditioned cut it is meant to catch straight into
        // the LP. The fix, when it comes, is a relative drop tolerance applied
        // to every family before screening (issues #209 and #212), not a
        // carve-out here.
        if (!isNumericallySafeCut(cut)) {
            ++stats_.rejected_screen;
            continue;
        }

        Real lhs = 0.0;
        for (Index k = 0; k < static_cast<Index>(cut.indices.size()); ++k) {
            lhs += cut.values[k] * primals[cut.indices[k]];
        }
        const Real violation = cut.lower - lhs;
        if (violation < min_violation_) {
            ++stats_.rejected_violation;
            continue;
        }
        cut.efficacy = violation / std::sqrt(norm_sq);

        const bool local = cut.local;
        if (!pool.addCut(std::move(cut))) {
            ++stats_.rejected_pool;
            continue;
        }
        ++num_cuts;
        ++stats_.accepted;
        if (local) {
            ++stats_.local_cuts;
        }
    }

    return num_cuts;
}

}  // namespace mipx
