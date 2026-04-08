#include "mipx/separators.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "mipx/branching.h"
#include "mipx/lp_problem.h"

namespace mipx {

namespace {

constexpr Real kIntTol = 1e-6;
constexpr Real kCoeffTol = 1e-10;

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
/// and alpha_j = 1 / max(t_j / f0, -t_j / (1-f0)) for the split on x_i.
Int SeparatorManager::separateIntersectionCut(DualSimplexSolver& lp,
                                              const LpProblem& problem,
                                              std::span<const Real> primals,
                                              CutPool& pool,
                                              CutFamilyStats& stats) {
    Int accepted = 0;
    const Index num_cols = problem.num_cols;
    const Index num_rows = lp.numRows();
    const Index total_vars = num_cols + num_rows;

    auto basis = lp.getBasis();

    // Find fractional basic integer variables.
    struct Candidate {
        Index basis_pos;
        Index col;
        Real frac;
    };
    std::vector<Candidate> candidates;

    for (Index j = 0; j < num_cols; ++j) {
        if (problem.col_type[j] == VarType::Continuous) continue;
        Index bp = lp.basisPosition(j);
        if (bp < 0) continue;
        Real val = primals[j];
        Real frac = fractionality(val);
        if (frac > kIntTol && frac < 1.0 - kIntTol) {
            candidates.push_back({bp, j, frac});
        }
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) {
                  return std::abs(a.frac - 0.5) < std::abs(b.frac - 0.5);
              });

    const Index max_try = std::min(static_cast<Index>(candidates.size()),
                                   static_cast<Index>(max_cuts_per_family_ * 2));

    std::vector<Real> tab_row(static_cast<std::size_t>(total_vars));

    for (Index ci = 0; ci < max_try && accepted < max_cuts_per_family_; ++ci) {
        const auto& cand = candidates[ci];
        ++stats.attempted;

        lp.getTableauRow(cand.basis_pos, tab_row);

        const Real b = primals[cand.col];
        const Real f0 = b - std::floor(b);
        if (f0 < kIntTol || f0 > 1.0 - kIntTol) continue;

        // Only use structural nonbasics at finite bounds.
        bool supported = true;
        for (Index k = 0; k < total_vars; ++k) {
            if (basis[k] == BasisStatus::Basic) continue;
            if (std::abs(tab_row[k]) < kCoeffTol) continue;
            if (k >= num_cols) {
                supported = false;
                break;
            }
            const BasisStatus st = basis[k];
            if (st == BasisStatus::AtLower || st == BasisStatus::Fixed) {
                if (!std::isfinite(problem.col_lower[k])) {
                    supported = false;
                    break;
                }
            } else if (st == BasisStatus::AtUpper) {
                if (!std::isfinite(problem.col_upper[k])) {
                    supported = false;
                    break;
                }
            } else {
                supported = false;
                break;
            }
        }
        if (!supported) continue;

        // Compute intersection cut coefficients.
        // For the split disjunction x_i <= floor(b) OR x_i >= ceil(b):
        // Ray from basis along nonbasic j: point + lambda * ray_j
        // Intersection with x_i = floor(b): lambda_lo = f0 / t_j  (if t_j > 0)
        // Intersection with x_i = ceil(b):  lambda_hi = -(1-f0) / t_j  (if t_j < 0)
        // Take the closer intersection: alpha_j = 1 / min_positive_lambda
        std::vector<Real> cut_coeff(static_cast<std::size_t>(num_cols), 0.0);
        Real cut_rhs = 1.0;
        bool valid = true;

        for (Index k = 0; k < num_cols; ++k) {
            if (basis[k] == BasisStatus::Basic) continue;
            const Real t = tab_row[k];
            if (std::abs(t) < kCoeffTol) continue;

            const BasisStatus st = basis[k];

            // Deviation direction: at lower => positive, at upper => negative.
            Real t_dev = t;
            if (st == BasisStatus::AtUpper) t_dev = -t;

            // Compute the step along the ray to reach the split boundary.
            Real step = 0.0;
            if (t_dev > kCoeffTol) {
                // Ray goes toward floor(b): step to floor = f0 / t_dev.
                step = f0 / t_dev;
            } else if (t_dev < -kCoeffTol) {
                // Ray goes toward ceil(b): step to ceil = (1 - f0) / (-t_dev).
                step = (1.0 - f0) / (-t_dev);
            } else {
                continue;
            }

            if (step < kCoeffTol || !std::isfinite(step)) {
                valid = false;
                break;
            }

            const Real alpha = 1.0 / step;

            // Convert from deviation space to original variable space.
            if (st == BasisStatus::AtLower || st == BasisStatus::Fixed) {
                cut_coeff[k] += alpha;
                const Real lb = problem.col_lower[k];
                if (lb > -kInf) cut_rhs += alpha * lb;
            } else if (st == BasisStatus::AtUpper) {
                cut_coeff[k] -= alpha;
                const Real ub = problem.col_upper[k];
                if (ub < kInf) cut_rhs -= alpha * ub;
            }
        }

        if (!valid || !std::isfinite(cut_rhs)) continue;

        // Build sparse cut.
        Cut cut;
        cut.family = CutFamily::IntersectionCut;
        Real norm_sq = 0.0;

        for (Index j = 0; j < num_cols; ++j) {
            if (std::abs(cut_coeff[j]) > kCoeffTol) {
                cut.indices.push_back(j);
                cut.values.push_back(cut_coeff[j]);
                norm_sq += cut_coeff[j] * cut_coeff[j];
            }
        }

        if (cut.indices.empty() || norm_sq < kCoeffTol) continue;

        cut.lower = cut_rhs;
        cut.upper = kInf;

        // Compute violation.
        Real lhs = 0.0;
        for (Index k = 0; k < static_cast<Index>(cut.indices.size()); ++k) {
            lhs += cut.values[k] * primals[cut.indices[k]];
        }
        const Real violation = cut_rhs - lhs;
        if (violation < min_violation_) continue;

        cut.efficacy = violation / std::sqrt(norm_sq);
        if (!std::isfinite(cut.efficacy) || cut.efficacy <= 0.0) continue;

        ++stats.generated;
        if (pool.addCut(std::move(cut))) {
            ++stats.accepted;
            stats.efficacy_sum += violation / std::sqrt(norm_sq);
            ++accepted;
        }
    }

    return accepted;
}

/// Multi-row cuts: lattice-based relaxation using pairs of tableau rows.
///
/// For two basic integer variables x_i, x_k with fractional values,
/// we use their combined tableau rows to derive a stronger cut than
/// single-row cuts. The approach uses the 2D lattice-free set
/// (specifically, a triangle or cross-polytope) to generate the cut.
///
/// For simplicity, we use the "triangle closure" approach:
/// Given two fractional basic vars with fractionalities f1, f2,
/// the combined cut coefficients for nonbasic j are:
///   alpha_j = max over the two individual GMI coefficients,
///   strengthened by the lattice structure.
Int SeparatorManager::separateMultiRow(DualSimplexSolver& lp,
                                       const LpProblem& problem,
                                       std::span<const Real> primals,
                                       CutPool& pool,
                                       CutFamilyStats& stats) {
    Int accepted = 0;
    const Index num_cols = problem.num_cols;
    const Index num_rows = lp.numRows();
    const Index total_vars = num_cols + num_rows;

    auto basis = lp.getBasis();

    // Find fractional basic integer variables.
    struct Candidate {
        Index basis_pos;
        Index col;
        Real value;
        Real frac;
    };
    std::vector<Candidate> candidates;

    for (Index j = 0; j < num_cols; ++j) {
        if (problem.col_type[j] == VarType::Continuous) continue;
        Index bp = lp.basisPosition(j);
        if (bp < 0) continue;
        Real val = primals[j];
        Real frac = fractionality(val);
        if (frac > kIntTol && frac < 1.0 - kIntTol) {
            candidates.push_back({bp, j, val, frac});
        }
    }

    if (candidates.size() < 2) return 0;

    // Sort by fractionality closest to 0.5.
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) {
                  return std::abs(a.frac - 0.5) < std::abs(b.frac - 0.5);
              });

    // Limit pairs to avoid combinatorial explosion.
    const Index max_cands = std::min(static_cast<Index>(candidates.size()), Index{8});
    std::vector<Real> tab_row1(static_cast<std::size_t>(total_vars));
    std::vector<Real> tab_row2(static_cast<std::size_t>(total_vars));

    for (Index c1 = 0; c1 < max_cands && accepted < max_cuts_per_family_; ++c1) {
        for (Index c2 = c1 + 1; c2 < max_cands && accepted < max_cuts_per_family_; ++c2) {
            ++stats.attempted;

            const auto& cand1 = candidates[c1];
            const auto& cand2 = candidates[c2];

            lp.getTableauRow(cand1.basis_pos, tab_row1);
            lp.getTableauRow(cand2.basis_pos, tab_row2);

            const Real f1 = cand1.value - std::floor(cand1.value);
            const Real f2 = cand2.value - std::floor(cand2.value);
            if (f1 < kIntTol || f1 > 1.0 - kIntTol) continue;
            if (f2 < kIntTol || f2 > 1.0 - kIntTol) continue;

            // Check support: only structural nonbasics at finite bounds.
            bool supported = true;
            for (Index k = 0; k < total_vars; ++k) {
                if (basis[k] == BasisStatus::Basic) continue;
                if (std::abs(tab_row1[k]) < kCoeffTol &&
                    std::abs(tab_row2[k]) < kCoeffTol) continue;
                if (k >= num_cols) {
                    supported = false;
                    break;
                }
                const BasisStatus st = basis[k];
                if (st == BasisStatus::AtLower || st == BasisStatus::Fixed) {
                    if (!std::isfinite(problem.col_lower[k])) {
                        supported = false;
                        break;
                    }
                } else if (st == BasisStatus::AtUpper) {
                    if (!std::isfinite(problem.col_upper[k])) {
                        supported = false;
                        break;
                    }
                } else {
                    supported = false;
                    break;
                }
            }
            if (!supported) continue;

            // Build multi-row cut using the maximal lattice-free triangle.
            // For a 2D relaxation with fractional point (f1, f2),
            // the triangle vertices are: (0,0), (1/f1, 0), (0, 1/f2).
            // For nonbasic j with tableau entries (t1_j, t2_j) in deviation form,
            // the cut coefficient is:
            //   alpha_j = 1 / max(t1_j/f1, t2_j/f2, 1 - t1_j/(1-f1) - t2_j/(1-f2))
            // (choosing the tightest among the three facets).
            std::vector<Real> cut_coeff(static_cast<std::size_t>(num_cols), 0.0);
            Real cut_rhs = 1.0;
            bool valid = true;

            for (Index k = 0; k < num_cols; ++k) {
                if (basis[k] == BasisStatus::Basic) continue;
                Real t1 = tab_row1[k];
                Real t2 = tab_row2[k];
                if (std::abs(t1) < kCoeffTol && std::abs(t2) < kCoeffTol) continue;

                const BasisStatus st = basis[k];
                if (st == BasisStatus::AtUpper) {
                    t1 = -t1;
                    t2 = -t2;
                }

                // Compute step to each facet of the triangle.
                Real min_step = kInf;

                if (t1 > kCoeffTol) {
                    min_step = std::min(min_step, f1 / t1);
                }
                if (t2 > kCoeffTol) {
                    min_step = std::min(min_step, f2 / t2);
                }

                const Real denom3 = t1 / (1.0 - f1) + t2 / (1.0 - f2);
                if (denom3 > kCoeffTol) {
                    min_step = std::min(min_step, 1.0 / denom3);
                }

                if (t1 < -kCoeffTol) {
                    min_step = std::min(min_step, (1.0 - f1) / (-t1));
                }
                if (t2 < -kCoeffTol) {
                    min_step = std::min(min_step, (1.0 - f2) / (-t2));
                }

                if (min_step <= kCoeffTol || min_step >= kInf || !std::isfinite(min_step)) {
                    valid = false;
                    break;
                }

                const Real alpha = 1.0 / min_step;

                if (st == BasisStatus::AtLower || st == BasisStatus::Fixed) {
                    cut_coeff[k] += alpha;
                    const Real lb = problem.col_lower[k];
                    if (lb > -kInf) cut_rhs += alpha * lb;
                } else if (st == BasisStatus::AtUpper) {
                    cut_coeff[k] -= alpha;
                    const Real ub = problem.col_upper[k];
                    if (ub < kInf) cut_rhs -= alpha * ub;
                }
            }

            if (!valid || !std::isfinite(cut_rhs)) continue;

            // Build sparse cut.
            Cut cut;
            cut.family = CutFamily::MultiRow;
            Real norm_sq = 0.0;
            for (Index j = 0; j < num_cols; ++j) {
                if (std::abs(cut_coeff[j]) > kCoeffTol) {
                    cut.indices.push_back(j);
                    cut.values.push_back(cut_coeff[j]);
                    norm_sq += cut_coeff[j] * cut_coeff[j];
                }
            }

            if (cut.indices.empty() || norm_sq < kCoeffTol) continue;

            cut.lower = cut_rhs;
            cut.upper = kInf;

            // Compute violation.
            Real lhs = 0.0;
            for (Index k = 0; k < static_cast<Index>(cut.indices.size()); ++k) {
                lhs += cut.values[k] * primals[cut.indices[k]];
            }
            const Real violation = cut_rhs - lhs;
            if (violation < min_violation_) continue;

            cut.efficacy = violation / std::sqrt(norm_sq);
            if (!std::isfinite(cut.efficacy) || cut.efficacy <= 0.0) continue;

            ++stats.generated;
            if (pool.addCut(std::move(cut))) {
                ++stats.accepted;
                stats.efficacy_sum += violation / std::sqrt(norm_sq);
                ++accepted;
            }
        }
    }

    return accepted;
}

}  // namespace mipx
