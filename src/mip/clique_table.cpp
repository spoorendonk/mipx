#include "mipx/clique_table.h"

#include <algorithm>
#include <cmath>

namespace mipx {

void CliqueTable::build(const LpProblem& problem,
                        const ConflictGraph& graph) {
    num_cols_ = problem.num_cols;
    cliques_.clear();
    // Literal id space: 2 * num_cols (one for each polarity of each column).
    lit_to_cliques_.assign(2 * num_cols_, {});

    if (graph.numBinaries() < 2) return;

    // Phase 1: Extract cliques from constraint rows.
    // For set-packing rows (all binary, all coeffs ~1, rhs ~1), the entire
    // set of binaries forms a clique.
    for (Index i = 0; i < problem.num_rows; ++i) {
        if (problem.row_upper[i] >= kInf) continue;
        Real rhs = problem.row_upper[i];
        if (!std::isfinite(rhs) || rhs < 0.5 || rhs > 1.0 + 1e-9) continue;

        auto row = problem.matrix.row(i);
        bool valid = true;
        Clique clique;

        for (Index k = 0; k < row.size(); ++k) {
            Index j = row.indices[k];
            Real a = row.values[k];
            if (problem.col_type[j] != VarType::Binary) {
                valid = false;
                break;
            }
            if (std::abs(a - 1.0) > 1e-9) {
                valid = false;
                break;
            }
            if (problem.col_lower[j] < -1e-12) {
                valid = false;
                break;
            }
            clique.literals.push_back({j, false});
        }

        if (!valid || clique.literals.size() < 2) continue;

        // Check if this is an equality constraint (set partitioning).
        if (std::abs(problem.row_lower[i] - 1.0) < 1e-9 &&
            std::abs(rhs - 1.0) < 1e-9) {
            clique.is_equality = true;
        }

        addClique(std::move(clique));
    }

    // Phase 2: Extract cliques from SOS1 constraints.
    for (const auto& sos : problem.sos_constraints) {
        if (sos.type != LpProblem::SosConstraint::Type::Sos1) continue;
        if (sos.vars.size() < 2) continue;

        Clique clique;
        clique.from_sos1 = true;
        bool all_binary = true;
        for (Index j : sos.vars) {
            if (j < 0 || j >= problem.num_cols) {
                all_binary = false;
                break;
            }
            if (problem.col_type[j] != VarType::Binary) {
                all_binary = false;
                break;
            }
            clique.literals.push_back({j, false});
        }
        if (all_binary && clique.literals.size() >= 2) {
            addClique(std::move(clique));
        }
    }

    // Phase 3: Greedy maximal clique extension from conflict graph.
    // For each binary variable, try to extend a 1-clique into a maximal clique.
    // Use a simple heuristic: pick unprocessed binary, greedily extend.
    std::vector<bool> processed(graph.numBinaries(), false);

    for (Index b = 0; b < graph.numBinaries(); ++b) {
        if (processed[b]) continue;
        Index col = graph.toOriginalIndex(b);
        if (col < 0) continue;

        Literal seed = {col, false};
        auto nbrs = graph.neighbors(seed);
        if (nbrs.empty()) continue;

        Clique clique;
        clique.literals.push_back(seed);
        extendClique(clique, graph);

        if (clique.literals.size() >= 2) {
            // Mark all variables in this clique as processed.
            for (const auto& lit : clique.literals) {
                Index bi = graph.toBinaryIndex(lit.var);
                if (bi >= 0) processed[bi] = true;
            }
            addClique(std::move(clique));
        }
    }

    // Phase 4: Merge and subsume.
    mergeAndSubsume();
}

Int CliqueTable::extractFromCut(const Cut& cut, const LpProblem& problem,
                                const ConflictGraph& graph) {
    // Check if the cut represents a clique inequality:
    // sum of binary variables <= 1.
    if (cut.upper > 1.0 + 1e-9 || cut.upper < 1.0 - 1e-9) return 0;

    Clique clique;
    for (Index k = 0; k < static_cast<Index>(cut.indices.size()); ++k) {
        Index j = cut.indices[k];
        Real v = cut.values[k];
        if (j < 0 || j >= problem.num_cols) return 0;
        if (problem.col_type[j] != VarType::Binary) return 0;
        if (std::abs(v - 1.0) > 1e-9) return 0;
        clique.literals.push_back({j, false});
    }

    if (clique.literals.size() < 2) return 0;

    // Try to extend this clique using the conflict graph.
    extendClique(clique, graph);

    addClique(std::move(clique));
    return 1;
}

std::span<const Index> CliqueTable::cliquesOf(Literal lit) const {
    Index id = lit.id();
    if (id < 0 || id >= static_cast<Index>(lit_to_cliques_.size())) return {};
    return lit_to_cliques_[id];
}

Int CliqueTable::mergeAndSubsume() {
    if (cliques_.size() < 2) return 0;

    // Sort cliques by size (descending) for efficient subsumption check.
    // A clique C1 subsumes C2 if all literals of C2 appear in C1.
    Int removed = 0;
    std::vector<bool> dead(cliques_.size(), false);

    // For efficiency, build a hash set for each clique's literal ids.
    for (Index i = 0; i < static_cast<Index>(cliques_.size()); ++i) {
        if (dead[i]) continue;
        const auto& ci = cliques_[i];

        for (Index j = i + 1; j < static_cast<Index>(cliques_.size()); ++j) {
            if (dead[j]) continue;
            const auto& cj = cliques_[j];

            // Check if cj is a subset of ci.
            if (cj.literals.size() <= ci.literals.size()) {
                bool subset = true;
                for (const auto& lit : cj.literals) {
                    bool found = false;
                    for (const auto& lit2 : ci.literals) {
                        if (lit.var == lit2.var &&
                            lit.complemented == lit2.complemented) {
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        subset = false;
                        break;
                    }
                }
                if (subset) {
                    dead[j] = true;
                    ++removed;
                    continue;
                }
            }

            // Check if ci is a subset of cj.
            if (ci.literals.size() <= cj.literals.size()) {
                bool subset = true;
                for (const auto& lit : ci.literals) {
                    bool found = false;
                    for (const auto& lit2 : cj.literals) {
                        if (lit.var == lit2.var &&
                            lit.complemented == lit2.complemented) {
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        subset = false;
                        break;
                    }
                }
                if (subset) {
                    dead[i] = true;
                    ++removed;
                    break;  // ci is dead, no need to check further.
                }
            }
        }
    }

    if (removed > 0) {
        // Compact cliques and rebuild lit_to_cliques_.
        std::vector<Clique> new_cliques;
        new_cliques.reserve(cliques_.size() - removed);
        for (Index i = 0; i < static_cast<Index>(cliques_.size()); ++i) {
            if (!dead[i]) {
                new_cliques.push_back(std::move(cliques_[i]));
            }
        }
        cliques_ = std::move(new_cliques);

        // Rebuild lit_to_cliques_.
        for (auto& v : lit_to_cliques_) v.clear();
        for (Index i = 0; i < static_cast<Index>(cliques_.size()); ++i) {
            for (const auto& lit : cliques_[i].literals) {
                Index id = lit.id();
                if (id >= 0 && id < static_cast<Index>(lit_to_cliques_.size())) {
                    lit_to_cliques_[id].push_back(i);
                }
            }
        }
    }

    return removed;
}

Int CliqueTable::separateCliqueCover(const LpProblem& problem,
                                     std::span<const Real> primals,
                                     CutPool& pool, Real min_violation,
                                     Int max_cuts) const {
    (void)problem;
    Int accepted = 0;

    for (Index ci = 0; ci < static_cast<Index>(cliques_.size()) &&
                        accepted < max_cuts;
         ++ci) {
        const auto& clq = cliques_[ci];
        if (clq.literals.size() < 2) continue;

        // Compute the LHS of the clique inequality: sum of literal values.
        Real lhs = 0.0;
        for (const auto& lit : clq.literals) {
            if (lit.var < 0 ||
                lit.var >= static_cast<Index>(primals.size()))
                continue;
            Real val = primals[lit.var];
            if (lit.complemented) val = 1.0 - val;
            lhs += val;
        }

        Real violation = lhs - 1.0;
        if (violation < min_violation) continue;

        // Build the cut: sum of literals <= 1.
        Cut cut;
        cut.family = CutFamily::Clique;
        cut.lower = -kInf;
        cut.upper = 1.0;

        // Sort literals by variable index for deterministic cut.
        auto sorted_lits = clq.literals;
        std::sort(sorted_lits.begin(), sorted_lits.end(),
                  [](const Literal& a, const Literal& b) {
                      return a.var < b.var;
                  });

        // For complemented literals: x_j' = 1 - x_j
        // sum(x_j) + sum(1 - x_k) <= 1
        // => sum(x_j) - sum(x_k) <= 1 - (number of complemented)
        Real rhs_adjust = 0.0;
        for (const auto& lit : sorted_lits) {
            cut.indices.push_back(lit.var);
            if (lit.complemented) {
                cut.values.push_back(-1.0);
                rhs_adjust -= 1.0;
            } else {
                cut.values.push_back(1.0);
            }
        }
        cut.upper = 1.0 + rhs_adjust;

        // Compute efficacy.
        Real norm_sq = 0.0;
        for (Real v : cut.values) norm_sq += v * v;
        Real lhs_cut = 0.0;
        for (Index k = 0; k < static_cast<Index>(cut.indices.size()); ++k) {
            if (cut.indices[k] < static_cast<Index>(primals.size())) {
                lhs_cut += cut.values[k] * primals[cut.indices[k]];
            }
        }
        Real cut_violation = lhs_cut - cut.upper;
        if (cut_violation < min_violation) continue;

        cut.efficacy = cut_violation / std::sqrt(norm_sq);
        if (!std::isfinite(cut.efficacy) || cut.efficacy <= 0.0) continue;

        if (pool.addCut(std::move(cut))) {
            ++accepted;
        }
    }

    return accepted;
}

std::vector<Clique> CliqueTable::findObjectiveCliques(
    const LpProblem& problem, const ConflictGraph& graph) const {
    std::vector<Clique> result;

    // Identify binary variables with positive objective coefficients
    // (for minimization). If they conflict pairwise, they form an
    // objective clique.
    struct ObjBin {
        Index col;
        Real obj_coeff;
    };
    std::vector<ObjBin> pos_obj_bins;
    for (Index j = 0; j < problem.num_cols; ++j) {
        if (problem.col_type[j] != VarType::Binary) continue;
        Real c = (problem.sense == Sense::Minimize) ? problem.obj[j]
                                                    : -problem.obj[j];
        if (c > 1e-9) {
            pos_obj_bins.push_back({j, c});
        }
    }

    if (pos_obj_bins.size() < 2) return result;

    // Sort by objective coefficient descending for greedy clique building.
    std::sort(pos_obj_bins.begin(), pos_obj_bins.end(),
              [](const ObjBin& a, const ObjBin& b) {
                  return a.obj_coeff > b.obj_coeff;
              });

    // Greedy: try to build a maximal clique from high-cost binaries.
    Clique clique;
    clique.literals.push_back({pos_obj_bins[0].col, false});

    for (Index i = 1; i < static_cast<Index>(pos_obj_bins.size()); ++i) {
        Literal candidate = {pos_obj_bins[i].col, false};
        bool conflicts_all = true;
        for (const auto& lit : clique.literals) {
            if (!graph.conflicts(lit, candidate)) {
                conflicts_all = false;
                break;
            }
        }
        if (conflicts_all) {
            clique.literals.push_back(candidate);
        }
    }

    if (clique.literals.size() >= 2) {
        result.push_back(std::move(clique));
    }

    return result;
}

std::vector<Clique> CliqueTable::cliquePartition(
    std::span<const Real> primals, const LpProblem& problem) const {
    std::vector<Clique> partition;

    // Collect fractional binary variables.
    struct FracBin {
        Index col;
        Real frac;
    };
    std::vector<FracBin> frac_bins;
    for (Index j = 0; j < problem.num_cols; ++j) {
        if (problem.col_type[j] != VarType::Binary) continue;
        if (j >= static_cast<Index>(primals.size())) continue;
        Real val = primals[j];
        Real frac = val - std::floor(val);
        if (frac > 1e-6 && frac < 1.0 - 1e-6) {
            frac_bins.push_back({j, frac});
        }
    }

    if (frac_bins.empty()) return partition;

    // Greedy: assign each fractional binary to the first clique that
    // contains it, or start a new singleton "clique".
    std::vector<bool> assigned(problem.num_cols, false);

    for (const auto& clq : cliques_) {
        Clique cover;
        for (const auto& lit : clq.literals) {
            if (lit.var >= 0 && lit.var < problem.num_cols &&
                !assigned[lit.var] && !lit.complemented) {
                // Check if this variable is fractional.
                if (lit.var < static_cast<Index>(primals.size())) {
                    Real val = primals[lit.var];
                    Real frac = val - std::floor(val);
                    if (frac > 1e-6 && frac < 1.0 - 1e-6) {
                        cover.literals.push_back(lit);
                        assigned[lit.var] = true;
                    }
                }
            }
        }
        if (cover.literals.size() >= 2) {
            partition.push_back(std::move(cover));
        }
    }

    return partition;
}

bool CliqueTable::propagate(std::span<const Real> lower,
                            std::span<const Real> upper,
                            std::vector<BoundUpdate>& updates) const {
    updates.clear();

    for (const auto& clq : cliques_) {
        // Find which literal (if any) is fixed to 1.
        Index fixed_idx = -1;
        Index num_free = 0;
        Index last_free_idx = -1;

        for (Index k = 0; k < static_cast<Index>(clq.literals.size()); ++k) {
            const auto& lit = clq.literals[k];
            if (lit.var < 0 ||
                lit.var >= static_cast<Index>(lower.size()))
                continue;

            Real lb = lower[lit.var];
            Real ub = upper[lit.var];

            // A literal is "fixed to 1" if:
            //   - non-complemented and lb >= 1 - eps
            //   - complemented and ub <= eps
            bool fixed_one = false;
            if (!lit.complemented) {
                fixed_one = (lb > 1.0 - 1e-8);
            } else {
                fixed_one = (ub < 1e-8);
            }

            // A literal is "fixed to 0" if:
            //   - non-complemented and ub <= eps
            //   - complemented and lb >= 1 - eps
            bool fixed_zero = false;
            if (!lit.complemented) {
                fixed_zero = (ub < 1e-8);
            } else {
                fixed_zero = (lb > 1.0 - 1e-8);
            }

            if (fixed_one) {
                if (fixed_idx >= 0) {
                    // Two literals fixed to 1 -- infeasible.
                    return false;
                }
                fixed_idx = k;
            } else if (!fixed_zero) {
                ++num_free;
                last_free_idx = k;
            }
        }

        if (fixed_idx >= 0) {
            // One literal is fixed to 1: fix all others to 0.
            for (Index k = 0; k < static_cast<Index>(clq.literals.size());
                 ++k) {
                if (k == fixed_idx) continue;
                const auto& lit = clq.literals[k];
                if (lit.var < 0 ||
                    lit.var >= static_cast<Index>(lower.size()))
                    continue;

                Real cur_lb = lower[lit.var];
                Real cur_ub = upper[lit.var];

                // Fix literal to 0:
                //   - non-complemented: set ub = 0
                //   - complemented: set lb = 1
                if (!lit.complemented) {
                    if (cur_ub > 1e-8) {
                        Real new_ub = 0.0;
                        if (cur_lb > new_ub + 1e-8) return false;
                        updates.push_back({lit.var, cur_lb, new_ub});
                    }
                } else {
                    if (cur_lb < 1.0 - 1e-8) {
                        Real new_lb = 1.0;
                        if (new_lb > cur_ub + 1e-8) return false;
                        updates.push_back({lit.var, new_lb, cur_ub});
                    }
                }
            }
        } else if (num_free == 1 && clq.is_equality) {
            // Equality clique with one free literal: force it to 1.
            const auto& lit = clq.literals[last_free_idx];
            if (lit.var >= 0 &&
                lit.var < static_cast<Index>(lower.size())) {
                Real cur_lb = lower[lit.var];
                Real cur_ub = upper[lit.var];
                if (!lit.complemented) {
                    if (cur_lb < 1.0 - 1e-8) {
                        if (1.0 > cur_ub + 1e-8) return false;
                        updates.push_back({lit.var, 1.0, cur_ub});
                    }
                } else {
                    if (cur_ub > 1e-8) {
                        if (cur_lb > 0.0 + 1e-8) return false;
                        updates.push_back({lit.var, cur_lb, 0.0});
                    }
                }
            }
        } else if (num_free == 0 && clq.is_equality && fixed_idx < 0) {
            // All literals fixed to 0 in an equality clique -- infeasible.
            return false;
        }
    }

    return true;
}

std::vector<CliqueTable::Substitution> CliqueTable::findSubstitutions(
    const LpProblem& problem) const {
    std::vector<Substitution> result;

    // Look for 2-cliques that are equalities: x_i + x_j = 1.
    // This means x_i = 1 - x_j, so we can substitute.
    for (const auto& clq : cliques_) {
        if (!clq.is_equality) continue;
        if (clq.literals.size() != 2) continue;

        const auto& a = clq.literals[0];
        const auto& b = clq.literals[1];

        // For the substitution to be valid, both must be non-complemented
        // binary variables.
        if (a.var < 0 || a.var >= problem.num_cols) continue;
        if (b.var < 0 || b.var >= problem.num_cols) continue;

        // x_a + x_b = 1 => x_a = 1 - x_b (or vice versa).
        // Eliminate the variable with the higher index.
        if (a.var > b.var) {
            bool comp = !(a.complemented ^ b.complemented);
            result.push_back({a.var, b.var, comp});
        } else {
            bool comp = !(a.complemented ^ b.complemented);
            result.push_back({b.var, a.var, comp});
        }
    }

    return result;
}

void CliqueTable::extendClique(Clique& clique,
                               const ConflictGraph& graph) const {
    // Greedy extension: repeatedly try to add a neighbor of the first literal
    // that conflicts with all current clique members.
    if (clique.literals.empty()) return;

    // Collect candidate literals: neighbors of the first literal.
    auto seed_nbrs = graph.neighbors(clique.literals[0]);

    for (const auto& candidate : seed_nbrs) {
        // Check if candidate conflicts with all current clique members.
        bool conflicts_all = true;
        for (const auto& lit : clique.literals) {
            if (candidate.var == lit.var &&
                candidate.complemented == lit.complemented) {
                conflicts_all = false;  // Already in clique.
                break;
            }
            if (!graph.conflicts(lit, candidate)) {
                conflicts_all = false;
                break;
            }
        }
        if (conflicts_all) {
            clique.literals.push_back(candidate);
        }
    }
}

void CliqueTable::addClique(Clique clique) {
    if (clique.literals.size() < 2) return;

    Index idx = static_cast<Index>(cliques_.size());
    for (const auto& lit : clique.literals) {
        Index id = lit.id();
        if (id >= 0 && id < static_cast<Index>(lit_to_cliques_.size())) {
            lit_to_cliques_[id].push_back(idx);
        }
    }
    cliques_.push_back(std::move(clique));
}

}  // namespace mipx
