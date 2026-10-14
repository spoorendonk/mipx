#include "tableau_cut.h"

#include "mipx/branching.h"
#include "mipx/gomory.h"

namespace mipx::tableau_detail {

namespace {

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

/// The deviation is delta = x_k - l_k when the column sits at its lower bound
/// and delta = u_k - x_k when it sits at its upper bound, measured from the
/// bound the LP holds the column at.
bool describeStructural(const LpProblem& problem, const DualSimplexSolver& lp, Index k,
                        BasisStatus status, Deviation& dev) {
    Real lp_lower = 0.0;
    Real lp_upper = 0.0;
    lp.getColBounds(k, lp_lower, lp_upper);

    Real global_bound = problem.col_lower[k];
    dev.sign = 1.0;
    dev.bound = lp_lower;
    if (status == BasisStatus::AtUpper) {
        dev.sign = -1.0;
        dev.bound = lp_upper;
        global_bound = problem.col_upper[k];
    } else if (status != BasisStatus::AtLower && status != BasisStatus::Fixed) {
        return false;  // nonbasic free: delta is not sign-constrained
    }
    if (!std::isfinite(dev.bound)) {
        return false;
    }
    dev.local = boundIsLocal(dev.bound, global_bound, dev.sign);
    dev.integral = isIntegerValued(problem.col_type[k]) && isIntegralValue(dev.bound, kDataIntTol);
    return true;
}

/// The logical is the row activity s = a^T x bounded by the row's own bounds,
/// so the deviation is s - lower when it sits at the row's lower bound and
/// upper - s when it sits at the row's upper bound. That covers <= rows (only
/// an upper bound, so the logical can only sit there), >= rows, ranged rows
/// (either bound) and equality rows (Fixed, handled as the lower bound, which
/// equals the upper one).
bool describeLogical(const LpProblem& problem, const DualSimplexSolver& lp, Index row,
                     BasisStatus status, const RowProvenance& provenance, RowScratch& scratch,
                     Deviation& dev) {
    // As for a structural column: delta >= 0 rests on the row holding, so a row
    // the caller has not vouched for makes the cut local whatever coefficient
    // comes out.
    dev.local = provenance.isLocal(row);

    Real row_lower = 0.0;
    Real row_upper = 0.0;
    lp.getRowExternal(row, scratch.indices, scratch.values, row_lower, row_upper);

    dev.sign = 1.0;
    dev.bound = row_lower;
    if (status == BasisStatus::AtUpper) {
        dev.sign = -1.0;
        dev.bound = row_upper;
    } else if (status != BasisStatus::AtLower && status != BasisStatus::Fixed) {
        return false;  // free row logical: delta is not sign-constrained
    }
    if (!std::isfinite(dev.bound)) {
        return false;
    }
    dev.integral =
        isIntegralValue(dev.bound, kDataIntTol) && rowActivityIsIntegral(problem, scratch);
    return true;
}

}  // namespace

RowProvenance makeRowProvenance(const DualSimplexSolver& lp, const LpProblem& problem,
                                Index global_row_count, std::span<const Index> local_rows) {
    const Index declared_rows = (global_row_count < 0) ? problem.num_rows : global_row_count;
    return RowProvenance{std::min(declared_rows, lp.numRows()), local_rows};
}

ColumnBox makeColumnBox(const DualSimplexSolver& lp, const LpProblem& problem) {
    const Index num_cols = problem.num_cols;
    ColumnBox box{problem.col_lower, problem.col_upper, {}, {}};
    box.lp_lower.resize(static_cast<std::size_t>(num_cols));
    box.lp_upper.resize(static_cast<std::size_t>(num_cols));
    for (Index j = 0; j < num_cols; ++j) {
        lp.getColBounds(j, box.lp_lower[j], box.lp_upper[j]);
    }
    return box;
}

bool describeDeviation(const LpProblem& problem, const DualSimplexSolver& lp, Index k,
                       BasisStatus status, const RowProvenance& provenance, RowScratch& scratch,
                       Deviation& dev) {
    if (k < problem.num_cols) {
        return describeStructural(problem, lp, k, status, dev);
    }
    return describeLogical(problem, lp, k - problem.num_cols, status, provenance, scratch, dev);
}

bool applyDeviation(Index k, Index num_cols, const Deviation& dev, Real coeff,
                    const RowScratch& scratch, CutBuilder& builder) {
    // A negligible coefficient is not dropped here: dropping a term of a >= cut
    // strengthens it, so it goes into the cut and finalizeCut drops it there
    // against the column's range, relaxing the right-hand side to match.
    if (coeff == 0.0) {
        return true;
    }
    if (k < num_cols) {
        builder.add(k, dev.sign * coeff);
        builder.rhs += dev.sign * coeff * dev.bound;
        return true;
    }
    // Substituting spreads the term over the row's structural columns, which
    // is what makes the cut expressible in x.
    for (std::size_t p = 0; p < scratch.indices.size(); ++p) {
        const Index j = scratch.indices[p];
        if (j < 0 || j >= num_cols) {
            return false;
        }
        builder.add(j, dev.sign * coeff * scratch.values[p]);
    }
    builder.rhs += dev.sign * coeff * dev.bound;
    return true;
}

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

}  // namespace mipx::tableau_detail
