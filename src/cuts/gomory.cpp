#include "mipx/gomory.h"

#include "mipx/lp_problem.h"
#include "tableau_cut.h"

#include <algorithm>
#include <cmath>
#include <span>
#include <vector>

namespace mipx {

namespace {

using tableau_detail::accumulateTerms;
using tableau_detail::collectCandidates;
using tableau_detail::ColumnBox;
using tableau_detail::CutBuilder;
using tableau_detail::finalizeCut;
using tableau_detail::makeColumnBox;
using tableau_detail::makeRowProvenance;
using tableau_detail::RowProvenance;
using tableau_detail::RowScratch;

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
    const RowProvenance provenance = makeRowProvenance(lp, problem, global_row_count_, local_rows_);
    const ColumnBox box = makeColumnBox(lp, problem);

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

        // Only an exactly-zero entry means the variable is absent from the row.
        // A tiny one still leans on delta_k >= 0 and still carries a
        // coefficient, which finalizeCut pays for if it drops it.
        const auto present = [&](Index k) { return tab_row[k] != 0.0; };
        const auto coefficient = [&](Index k, Real sign, bool integral) {
            return gmiCoefficient(sign * tab_row[k], integral, f0);
        };
        builder.reset(num_cols);
        if (!accumulateTerms(problem, lp, basis, present, coefficient, provenance, scratch,
                             builder)) {
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
