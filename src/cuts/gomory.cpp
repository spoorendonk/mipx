#include "mipx/gomory.h"

#include "mipx/branching.h"
#include "mipx/lp_problem.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace mipx {

namespace {

bool isIntegralValue(Real v, Real tol) {
    return std::abs(v - std::round(v)) <= tol;
}

}  // namespace

/// Generate Gomory mixed-integer rounding cuts from the simplex tableau.
///
/// Works in external (unscaled) space. The tableau row from getTableauRow
/// is already unscaled, and the cut coefficients are in external space.
/// They are passed to addRows which handles the column scaling internally.
///
/// For a basic integer variable x_i with value b:
///   Tableau row: x_i + sum_{j nonbasic} t_j * x_j = b (external coords)
///
/// GMI cut (via deviation variables delta_j >= 0):
///   For at lower: delta_j = x_j - l_j
///   For at upper: delta_j = u_j - x_j
///
/// The cut is:
///   sum over nonbasic j:  gmi_coeff(t_j) * delta_j >= 1

Int GomorySeparator::separate(DualSimplexSolver& lp, const LpProblem& problem,
                              std::span<const Real> primals, CutPool& pool) {
    Int num_cuts = 0;
    Index num_cols = problem.num_cols;
    Index num_rows = lp.numRows();
    Index total_vars = num_cols + num_rows;

    auto basis = lp.getBasis();

    // Collect basic integer variables with fractional values.
    struct Candidate {
        Index basis_pos;
        Index col;
        Real frac;
    };
    std::vector<Candidate> candidates;

    for (Index j = 0; j < num_cols; ++j) {
        if (problem.col_type[j] == VarType::Continuous) {
            continue;
        }
        Index bp = lp.basisPosition(j);
        if (bp < 0) {
            continue;
        }

        Real val = primals[j];
        Real frac = fractionality(val);
        if (frac > kIntTol && frac < 1.0 - kIntTol) {
            candidates.push_back({bp, j, frac});
        }
    }

    // Sort by fractionality closest to 0.5.
    std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
        return std::abs(a.frac - 0.5) < std::abs(b.frac - 0.5);
    });

    Index max_try =
        std::min(static_cast<Index>(candidates.size()), static_cast<Index>(max_cuts_ * 2));

    std::vector<Real> tab_row(static_cast<std::size_t>(total_vars));

    for (Index ci = 0; ci < max_try && num_cuts < max_cuts_; ++ci) {
        const auto& cand = candidates[ci];

        // Get the external (unscaled) tableau row.
        lp.getTableauRow(cand.basis_pos, tab_row);

        Real b = primals[cand.col];
        Real f0 = b - std::floor(b);
        if (f0 < kIntTol || f0 > 1.0 - kIntTol) {
            continue;
        }

        // Safety filter:
        // Generate Gomory cuts only from rows whose nonbasic terms are all
        // structural variables at finite active bounds. We intentionally skip
        // rows with slack nonbasics to avoid invalid substitutions when the LP
        // already contains dynamically added rows/cuts.
        bool supported_row = true;
        for (Index k = 0; k < total_vars; ++k) {
            if (basis[k] == BasisStatus::Basic) {
                continue;
            }
            const Real alpha = tab_row[k];
            if (std::abs(alpha) < kCoeffTol) {
                continue;
            }

            const BasisStatus st = basis[k];
            if (k >= num_cols) {
                supported_row = false;
                break;
            }

            const Real lb = problem.col_lower[k];
            const Real ub = problem.col_upper[k];
            if (st == BasisStatus::AtLower || st == BasisStatus::Fixed) {
                if (!std::isfinite(lb)) {
                    supported_row = false;
                    break;
                }
                if (problem.col_type[k] != VarType::Continuous && !isIntegralValue(lb, kIntTol)) {
                    supported_row = false;
                    break;
                }
            } else if (st == BasisStatus::AtUpper) {
                if (!std::isfinite(ub)) {
                    supported_row = false;
                    break;
                }
                if (problem.col_type[k] != VarType::Continuous && !isIntegralValue(ub, kIntTol)) {
                    supported_row = false;
                    break;
                }
            } else {
                supported_row = false;
                break;
            }
        }
        if (!supported_row) {
            continue;
        }

        // Build the cut in external space: sum_j cut_coeff[j] * x_j >= cut_rhs
        std::vector<Real> cut_coeff(static_cast<std::size_t>(num_cols), 0.0);
        Real cut_rhs = 1.0;
        bool valid = true;

        for (Index k = 0; k < total_vars; ++k) {
            if (basis[k] == BasisStatus::Basic) {
                continue;
            }

            Real alpha = tab_row[k];
            if (std::abs(alpha) < kCoeffTol) {
                continue;
            }

            BasisStatus st = basis[k];

            // Compute the tableau coefficient in deviation form.
            // x_i = b - sum t_j * delta_j
            // At lower: x_j = lb + delta_j => t_j = alpha_j
            // At upper: x_j = ub - delta_j => t_j = -alpha_j
            Real t = alpha;
            if (st == BasisStatus::AtUpper) {
                t = -alpha;
            }

            bool is_integer = (k < num_cols && problem.col_type[k] != VarType::Continuous);

            // GMI coefficient for the deviation.
            Real gmi_coeff = 0.0;
            if (is_integer) {
                Real fj = t - std::floor(t);
                if (fj < 0) {
                    fj += 1.0;
                }
                if (fj > 1.0 - kCoeffTol) {
                    fj = 0.0;
                }

                if (fj <= f0 + kCoeffTol) {
                    gmi_coeff = fj / f0;
                } else {
                    gmi_coeff = (1.0 - fj) / (1.0 - f0);
                }
            } else {
                if (t > kCoeffTol) {
                    gmi_coeff = t / f0;
                } else if (t < -kCoeffTol) {
                    gmi_coeff = -t / (1.0 - f0);
                }
            }

            if (std::abs(gmi_coeff) < kCoeffTol) {
                continue;
            }

            if (k < num_cols) {
                // Structural variable with external bounds.
                Real lb = problem.col_lower[k];
                Real ub = problem.col_upper[k];

                if (st == BasisStatus::AtLower || st == BasisStatus::Fixed) {
                    // delta = x - lb
                    cut_coeff[k] += gmi_coeff;
                    if (lb > -kInf) {
                        cut_rhs += gmi_coeff * lb;
                    }
                } else if (st == BasisStatus::AtUpper) {
                    // delta = ub - x
                    cut_coeff[k] -= gmi_coeff;
                    if (ub < kInf) {
                        cut_rhs -= gmi_coeff * ub;
                    }
                } else {
                    valid = false;
                    break;
                }
            } else {
                // Guarded by supported_row filter above.
                valid = false;
                break;
            }
        }

        if (!valid || !std::isfinite(cut_rhs)) {
            continue;
        }

        // Build sparse cut.
        Cut cut;
        Real norm_sq = 0.0;
        for (Index j = 0; j < num_cols; ++j) {
            if (std::abs(cut_coeff[j]) > kCoeffTol) {
                cut.indices.push_back(j);
                cut.values.push_back(cut_coeff[j]);
                norm_sq += cut_coeff[j] * cut_coeff[j];
            }
        }

        if (cut.indices.empty() || norm_sq < kCoeffTol) {
            continue;
        }

        cut.lower = cut_rhs;
        cut.upper = kInf;
        cut.family = CutFamily::Gomory;

        // Same numerical safety screen every other family gets through
        // SeparatorManager::addViolatedCut. No check is skipped: the ones the
        // construction above already guarantees (sorted indices, |coef| above
        // kCoeffTol, finite rhs, non-empty support, norm_sq above kCoeffTol)
        // are cheap to re-test, and the ones it does not guarantee — a maximum
        // coefficient above 1e6, a max/min coefficient ratio above 1e8, and a
        // squared norm above 1e16 — are exactly why the screen is here. A GMI
        // coefficient is t/f0 (or fj/f0) on a nonbasic column, so a basic
        // variable whose fractional part f0 sits just above kIntTol blows the
        // coefficient up without any of the local checks noticing.
        //
        // Note the screen is stricter than the sparsification just above: it
        // drops the whole cut when max|coef| / min|coef| exceeds 1e8, while the
        // loop keeps every coefficient above kCoeffTol = 1e-10. A cut carrying
        // an O(1) coefficient next to a 5e-10 cancellation residual is
        // therefore rejected outright. That is deliberate — Gomory gets the
        // same screen as everyone else, and a Gomory-only exemption would put
        // the ill-conditioned cut it is meant to catch straight into the LP.
        // If it proves too blunt once GomorySeparator actually emits cuts
        // (issue #211), the fix is a relative drop tolerance applied to every
        // family before screening, not a carve-out here.
        if (!isNumericallySafeCut(cut)) {
            continue;
        }

        // Compute violation: lhs = sum cut_coeff[j] * x_j, violation = rhs - lhs.
        Real lhs = 0.0;
        for (Index k = 0; k < static_cast<Index>(cut.indices.size()); ++k) {
            lhs += cut.values[k] * primals[cut.indices[k]];
        }
        Real violation = cut_rhs - lhs;

        if (violation < min_violation_) {
            continue;
        }

        cut.efficacy = violation / std::sqrt(norm_sq);

        if (pool.addCut(std::move(cut))) {
            ++num_cuts;
        }
    }

    return num_cuts;
}

}  // namespace mipx
