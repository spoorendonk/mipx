#include "mipx/lp_problem.h"
#include "mipx/separators.h"
#include "tableau_cut.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace mipx {

namespace {

using tableau_detail::accumulateTerms;
using tableau_detail::Candidate;
using tableau_detail::collectCandidates;
using tableau_detail::ColumnBox;
using tableau_detail::CutBuilder;
using tableau_detail::finalizeCut;
using tableau_detail::makeColumnBox;
using tableau_detail::makeRowProvenance;
using tableau_detail::RowProvenance;
using tableau_detail::RowScratch;

constexpr Real kIntTol = 1e-6;
constexpr Real kCoeffTol = 1e-10;
constexpr Real kRejectRow = std::numeric_limits<Real>::quiet_NaN();

/// Coefficient of a ray, the reciprocal of its step to the boundary of the
/// lattice-free set, or a rejection when that step is within kCoeffTol of
/// zero: the coefficient would blow up. Working with the reciprocal keeps a
/// ray that barely moves the basic variable at a tiny coefficient, which
/// finalizeCut pays for, rather than an overflowing step.
Real rayCoefficient(Real alpha) {
    if (!(alpha < 1.0 / kCoeffTol)) {
        return kRejectRow;
    }
    return alpha;
}

/// Everything a tableau family needs from the LP to turn a row into a cut.
struct TableauContext {
    std::vector<BasisStatus> basis;
    RowProvenance provenance;
    ColumnBox box;
    RowScratch scratch;
    CutBuilder builder;
};

/// Finish the accumulated cut and hand it to the pool. Returns true when the
/// pool took it.
template <class Offer>
bool emitCut(CutFamily family, TableauContext& ctx, Offer&& offer) {
    Cut cut;
    Real norm_sq = 0.0;
    Int relaxed_terms = 0;
    if (!finalizeCut(ctx.builder, ctx.box, kCoeffTol, cut, norm_sq, relaxed_terms)) {
        return false;
    }
    cut.family = family;
    cut.lower = ctx.builder.rhs;
    cut.upper = kInf;
    cut.local = ctx.builder.local;
    return offer(std::move(cut));
}

}  // namespace

/// Intersection cuts from split disjunctions.
///
/// For a basic integer variable x_i with fractional value b:
///   Split disjunction: x_i <= floor(b) OR x_i >= ceil(b).
///
/// The intersection cut is derived from the simplex cone at the current basis.
/// For each nonbasic variable j at its bound:
///   The ray direction is the j-th column of B^{-1} N (the tableau).
///   The intersection of the ray with the split gives a coefficient.
///
/// Cut: sum_j alpha_j * delta_j >= 1
/// where delta_j is the deviation of nonbasic j from its bound,
/// and alpha_j = max(t_j / f0, -t_j / (1-f0)) for the split on x_i.
///
/// This is the GMI formula with every deviation treated as continuous, so it
/// is never stronger than the Gomory cut from the same row.
///
/// The deviations are those of GomorySeparator::separate: the nonbasic row
/// logicals are substituted back into structural columns, every deviation is
/// measured from the bound the LP holds, and a cut leaning on a node-tightened
/// bound or a node-local row is marked local (issue #219).
Int SeparatorManager::separateIntersectionCut(DualSimplexSolver& lp, const LpProblem& problem,
                                              std::span<const Real> primals, CutPool& pool,
                                              CutFamilyStats& stats) {
    const Index num_cols = problem.num_cols;
    // The tableau is indexed by the LP's own columns.
    if (lp.numCols() != num_cols || static_cast<Index>(primals.size()) < num_cols) {
        return 0;
    }
    const Index total_vars = num_cols + lp.numRows();

    TableauContext ctx{lp.getBasis(),
                       makeRowProvenance(lp, problem, global_row_count_, local_rows_),
                       makeColumnBox(lp, problem),
                       {},
                       {}};
    const auto candidates = collectCandidates(lp, problem, primals, kIntTol);
    const Index max_try = std::min(static_cast<Index>(candidates.size()),
                                   static_cast<Index>(max_cuts_per_family_ * 2));

    std::vector<Real> tab_row(static_cast<std::size_t>(total_vars));
    const auto offer = [&](Cut cut) {
        return addViolatedCut(std::move(cut), primals, pool, stats);
    };

    Int accepted = 0;
    for (Index ci = 0; ci < max_try && accepted < max_cuts_per_family_; ++ci) {
        const Candidate& cand = candidates[ci];
        ++stats.attempted;

        lp.getTableauRow(cand.basis_pos, tab_row);

        const Real b = primals[cand.col];
        const Real f0 = b - std::floor(b);
        if (f0 < kIntTol || f0 > 1.0 - kIntTol)
            continue;

        // For the split disjunction x_i <= floor(b) OR x_i >= ceil(b):
        // Ray from basis along nonbasic j: point + lambda * ray_j
        // Intersection with x_i = floor(b): lambda_lo = f0 / t_j  (if t_j > 0)
        // Intersection with x_i = ceil(b):  lambda_hi = -(1-f0) / t_j  (if t_j < 0)
        // Take the closer intersection: alpha_j = 1 / lambda
        const auto present = [&](Index k) { return tab_row[k] != 0.0; };
        const auto coefficient = [&](Index k, Real sign, bool /*integral*/) {
            // Deviation direction: at lower => positive, at upper => negative.
            const Real t_dev = sign * tab_row[k];
            return rayCoefficient((t_dev > 0.0) ? t_dev / f0 : -t_dev / (1.0 - f0));
        };
        ctx.builder.reset(num_cols);
        if (!accumulateTerms(problem, lp, ctx.basis, present, coefficient, ctx.provenance,
                             ctx.scratch, ctx.builder)) {
            continue;
        }
        if (emitCut(CutFamily::IntersectionCut, ctx, offer)) {
            ++accepted;
        }
    }

    return accepted;
}

/// Multi-row cuts from pairs of tableau rows.
///
/// For two basic integer variables x_i, x_k with fractional parts f1, f2,
/// the coefficient of nonbasic j is the reciprocal of the step along its ray
/// to the boundary of the set
///
///   { y : 0 <= y1 <= 1, 0 <= y2 <= 1,
///         (f1 - y1)/(1 - f1) + (f2 - y2)/(1 - f2) <= 1 }
///
/// in the space of the two basic variables' fractional parts: the unit square
/// around (f1, f2) cut by one half-space. Its interior holds no integer point
/// and contains (f1, f2), so the cut is valid. The set lies inside each row's
/// split strip, so every coefficient is at least that row's split
/// coefficient: the cut is dominated by the intersection cut of either row
/// alone. A stronger lattice-free set (a true triangle) would be a separate
/// change; this keeps the existing rule.
///
/// Deviations, substitution and locality are those of the intersection cut.
Int SeparatorManager::separateMultiRow(DualSimplexSolver& lp, const LpProblem& problem,
                                       std::span<const Real> primals, CutPool& pool,
                                       CutFamilyStats& stats) {
    const Index num_cols = problem.num_cols;
    if (lp.numCols() != num_cols || static_cast<Index>(primals.size()) < num_cols) {
        return 0;
    }
    const Index total_vars = num_cols + lp.numRows();

    const auto candidates = collectCandidates(lp, problem, primals, kIntTol);
    if (candidates.size() < 2)
        return 0;

    TableauContext ctx{lp.getBasis(),
                       makeRowProvenance(lp, problem, global_row_count_, local_rows_),
                       makeColumnBox(lp, problem),
                       {},
                       {}};

    // Limit pairs to avoid combinatorial explosion.
    const Index max_cands = std::min(static_cast<Index>(candidates.size()), Index{8});
    std::vector<Real> tab_row1(static_cast<std::size_t>(total_vars));
    std::vector<Real> tab_row2(static_cast<std::size_t>(total_vars));
    const auto offer = [&](Cut cut) {
        return addViolatedCut(std::move(cut), primals, pool, stats);
    };

    Int accepted = 0;
    for (Index c1 = 0; c1 < max_cands && accepted < max_cuts_per_family_; ++c1) {
        for (Index c2 = c1 + 1; c2 < max_cands && accepted < max_cuts_per_family_; ++c2) {
            ++stats.attempted;

            const Candidate& cand1 = candidates[c1];
            const Candidate& cand2 = candidates[c2];

            lp.getTableauRow(cand1.basis_pos, tab_row1);
            lp.getTableauRow(cand2.basis_pos, tab_row2);

            const Real v1 = primals[cand1.col];
            const Real v2 = primals[cand2.col];
            const Real f1 = v1 - std::floor(v1);
            const Real f2 = v2 - std::floor(v2);
            if (f1 < kIntTol || f1 > 1.0 - kIntTol)
                continue;
            if (f2 < kIntTol || f2 > 1.0 - kIntTol)
                continue;

            // For nonbasic j with tableau entries (t1_j, t2_j) in deviation form,
            // the cut coefficient over the set above is:
            //   alpha_j = max(t1_j/f1, t2_j/f2, t1_j/(1-f1) + t2_j/(1-f2),
            //                 -t1_j/(1-f1), -t2_j/(1-f2))
            // (the first facet the ray reaches).
            const auto present = [&](Index k) { return tab_row1[k] != 0.0 || tab_row2[k] != 0.0; };
            const auto coefficient = [&](Index k, Real sign, bool /*integral*/) {
                const Real t1 = sign * tab_row1[k];
                const Real t2 = sign * tab_row2[k];

                // The closest facet the ray reaches: the largest reciprocal step.
                Real alpha = 0.0;
                if (t1 > 0.0) {
                    alpha = std::max(alpha, t1 / f1);
                }
                if (t2 > 0.0) {
                    alpha = std::max(alpha, t2 / f2);
                }
                const Real denom3 = t1 / (1.0 - f1) + t2 / (1.0 - f2);
                if (denom3 > 0.0) {
                    alpha = std::max(alpha, denom3);
                }
                if (t1 < 0.0) {
                    alpha = std::max(alpha, -t1 / (1.0 - f1));
                }
                if (t2 < 0.0) {
                    alpha = std::max(alpha, -t2 / (1.0 - f2));
                }
                return rayCoefficient(alpha);
            };
            ctx.builder.reset(num_cols);
            if (!accumulateTerms(problem, lp, ctx.basis, present, coefficient, ctx.provenance,
                                 ctx.scratch, ctx.builder)) {
                continue;
            }
            if (emitCut(CutFamily::MultiRow, ctx, offer)) {
                ++accepted;
            }
        }
    }

    return accepted;
}

}  // namespace mipx
