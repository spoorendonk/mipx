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

/// Relative size, against the magnitude of the terms summed into it, below
/// which a cut coefficient is cancellation round-off (a few thousand ulps).
constexpr Real kCancellationTol = 1e-12;

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
        // In [0, 1) for negative t as well. No snap toward 0 or 1: the cut is
        // sum_k c_k delta_k >= 1 with delta_k >= 0, so zeroing a coefficient
        // strengthens it, and on a wide-range column a discarded f/f0 of 1e-4
        // moves the cut by far more than that. A near-integral t instead gives
        // a tiny coefficient here, and finalizeCut drops it and pays for it in
        // the right-hand side (issue #216).
        const Real f = t - std::floor(t);
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

/// Which LP rows a cut may lean on and stay valid tree-wide.
struct RowProvenance {
    Index global_rows = 0;                   // rows at or past this are local
    std::span<const Index> local_rows = {};  // sorted; local inside the prefix

    [[nodiscard]] bool isLocal(Index row) const {
        return row >= global_rows || std::binary_search(local_rows.begin(), local_rows.end(), row);
    }
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

/// Column bounds a dropped coefficient is paid for over.
struct ColumnBox {
    std::span<const Real> global_lower;
    std::span<const Real> global_upper;
    std::vector<Real> lp_lower;
    std::vector<Real> lp_upper;
};

/// One GMI term under construction: the cut being accumulated in the form
/// sum_j cut_coeff[j] * x_j >= cut_rhs, plus whether it stayed global.
struct CutBuilder {
    std::vector<Real> coeff;
    std::vector<Real> magnitude;  // sum of |term| that went into each coeff
    Real rhs = 1.0;
    bool local = false;

    void add(Index j, Real term) {
        coeff[static_cast<std::size_t>(j)] += term;
        magnitude[static_cast<std::size_t>(j)] += std::abs(term);
    }
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
                       BasisStatus status, Real alpha, Real f0, CutBuilder& builder) {
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

    // The derivation needs delta_k >= 0 for every column in the row, whatever
    // coefficient it ends up with, so the bound decides locality here.
    if (boundIsLocal(bound, global_bound, sign)) {
        builder.local = true;
    }

    // A negligible coefficient is not dropped here: dropping a term of a >= cut
    // strengthens it, so it goes into the cut and finalizeCut drops it there
    // against the column's range, relaxing the right-hand side to match.
    const Real t = sign * alpha;
    const bool integral =
        isIntegerValued(problem.col_type[k]) && isIntegralValue(bound, kDataIntTol);
    const Real coeff = gmiCoefficient(t, integral, f0);

    builder.add(k, sign * coeff);
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
                    BasisStatus status, Real alpha, Real f0, const RowProvenance& provenance,
                    RowScratch& scratch, CutBuilder& builder) {
    // As for a structural column: delta >= 0 rests on the row holding, so a row
    // the caller has not vouched for makes the cut local whatever coefficient
    // comes out.
    if (provenance.isLocal(row)) {
        builder.local = true;
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
    if (coeff == 0.0) {
        return true;
    }

    for (std::size_t p = 0; p < scratch.indices.size(); ++p) {
        const Index j = scratch.indices[p];
        if (j < 0 || j >= problem.num_cols) {
            return false;
        }
        builder.add(j, sign * coeff * scratch.values[p]);
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
                     const RowProvenance& provenance, RowScratch& scratch, CutBuilder& builder) {
    const Index num_cols = problem.num_cols;
    const auto total_vars = static_cast<Index>(tab_row.size());
    for (Index k = 0; k < total_vars; ++k) {
        if (basis[k] == BasisStatus::Basic) {
            continue;
        }
        // Only an exactly-zero entry means the variable is absent from the row.
        // A tiny one still leans on delta_k >= 0 and still carries a
        // coefficient, which finalizeCut pays for if it drops it.
        const Real alpha = tab_row[k];
        if (alpha == 0.0) {
            continue;
        }
        const bool ok = (k < num_cols)
                            ? addStructuralTerm(problem, lp, k, basis[k], alpha, f0, builder)
                            : addLogicalTerm(problem, lp, k - num_cols, basis[k], alpha, f0,
                                             provenance, scratch, builder);
        if (!ok) {
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
                 Int& relaxed_terms) {
    const auto num_cols = static_cast<Index>(builder.coeff.size());
    const auto lower = builder.local ? std::span<const Real>(box.lp_lower) : box.global_lower;
    const auto upper = builder.local ? std::span<const Real>(box.lp_upper) : box.global_upper;
    // A column reached both directly and through a substituted logical can
    // cancel to zero in exact arithmetic and leave float round-off behind.
    // Where the column is bounded that residue is paid for like any other
    // small term; where it is not, it is zeroed as the round-off it is, since
    // keeping it would sink the cut at the max/min ratio screen.
    Int dropped = 0;
    builder.rhs = dropAndRelax(builder.coeff, builder.rhs, lower, upper, coeff_tol, &dropped,
                               builder.magnitude, kCancellationTol);
    relaxed_terms += dropped;

    norm_sq = 0.0;
    for (Index j = 0; j < num_cols; ++j) {
        const Real value = builder.coeff[static_cast<std::size_t>(j)];
        if (value != 0.0) {
            cut.indices.push_back(j);
            cut.values.push_back(value);
            norm_sq += value * value;
        }
    }
    return !cut.indices.empty() && norm_sq >= coeff_tol && std::isfinite(builder.rhs);
}

}  // namespace

Real dropAndRelax(std::span<Real> coeff, Real rhs, std::span<const Real> lower,
                  std::span<const Real> upper, Real drop_tol, Int* dropped,
                  std::span<const Real> magnitude, Real cancel_tol) {
    for (std::size_t j = 0; j < coeff.size(); ++j) {
        const Real value = coeff[j];
        if (value == 0.0) {
            continue;
        }
        const Real cancellation = magnitude.empty() ? 0.0 : cancel_tol * magnitude[j];
        if (std::abs(value) > std::max(drop_tol, cancellation)) {
            continue;
        }
        // The term's largest value over the box; lowering rhs by it keeps
        // every point the term could have helped over the line.
        const Real bound = (value > 0.0) ? upper[j] : lower[j];
        if (std::isfinite(bound)) {
            rhs -= value * bound;
            coeff[j] = 0.0;
            if (dropped != nullptr) {
                ++*dropped;
            }
        } else if (std::abs(value) <= std::min(drop_tol, cancellation)) {
            // Round-off of the sum, not cut mass. Capped at drop_tol as well:
            // a sum of 1e13-sized terms (f0 near kIntTol) leaves a relative
            // residue large enough to matter, and that one is kept.
            coeff[j] = 0.0;
        }
    }
    return rhs;
}

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
    // Both readings are clamped to the rows the LP holds: a problem carrying
    // more rows than that would otherwise make every logical look global.
    const Index declared_rows = (global_row_count_ < 0) ? problem.num_rows : global_row_count_;
    const RowProvenance provenance{std::min(declared_rows, num_rows), local_rows_};

    ColumnBox box{problem.col_lower, problem.col_upper, {}, {}};
    box.lp_lower.resize(static_cast<std::size_t>(num_cols));
    box.lp_upper.resize(static_cast<std::size_t>(num_cols));
    for (Index j = 0; j < num_cols; ++j) {
        lp.getColBounds(j, box.lp_lower[j], box.lp_upper[j]);
    }

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
        builder.magnitude.assign(static_cast<std::size_t>(num_cols), 0.0);
        builder.rhs = 1.0;
        builder.local = false;
        if (!accumulateTerms(problem, lp, basis, tab_row, f0, provenance, scratch, builder)) {
            ++stats_.rows_skipped;
            continue;
        }

        Cut cut;
        Real norm_sq = 0.0;
        if (!finalizeCut(builder, box, kCoeffTol, cut, norm_sq, stats_.relaxed_terms)) {
            ++stats_.rejected_empty;
            continue;
        }
        ++stats_.cuts_built;

        cut.lower = builder.rhs;
        cut.upper = kInf;
        cut.family = CutFamily::Gomory;
        cut.local = builder.local;

        // The numerical safety screen CutPool::addCut enforces on every cut,
        // applied here first so a rejection is counted as rejected_screen
        // rather than rejected_pool. What the construction above does not
        // guarantee -- a maximum coefficient above 1e6, a max/min coefficient
        // ratio above 1e8, a squared norm above 1e16 -- is exactly why the
        // screen is here: a GMI coefficient is t/f0 (or fj/f0) on a nonbasic
        // column, so a basic variable whose fractional part f0 sits just above
        // kIntTol blows the coefficient up without any local check noticing.
        //
        // The screen also rejects a cut that keeps a coefficient finalizeCut
        // could not pay to drop: a small genuine term on a column with no
        // finite bound in the needed direction trips the ratio limit. Losing
        // that cut is the price of validity. Gomory gets the same screen as
        // every other family; a Gomory-only exemption would put the
        // ill-conditioned cut it is meant to catch straight into the LP.
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
        const Real efficacy = cut.efficacy;
        if (!pool.addCut(std::move(cut))) {
            ++stats_.rejected_pool;
            continue;
        }
        ++num_cuts;
        ++stats_.accepted;
        stats_.efficacy_sum += efficacy;
        if (local) {
            ++stats_.local_cuts;
        }
    }

    return num_cuts;
}

}  // namespace mipx
