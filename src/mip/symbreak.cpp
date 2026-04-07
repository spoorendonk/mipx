#include "mipx/symbreak.h"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <format>
#include <functional>
#include <numeric>
#include <unordered_set>

#include "mipx/lp_problem.h"

namespace mipx {

// ---------------------------------------------------------------------------
// SymbreakGenerator
// ---------------------------------------------------------------------------

SymbreakConstraint SymbreakGenerator::generateSymresack(
    const Permutation& generator,
    const std::vector<VarType>& col_type,
    Index num_cols,
    Index constraint_idx) {
    SymbreakConstraint constraint;
    constraint.type = SymbreakType::Symresack;

    // Symresack inequality for binary permutation sigma:
    //   For each pair (j, sigma(j)) with j < sigma(j), add x_j - x_{sigma(j)}.
    //   Only count each 2-cycle once to avoid cancellation.
    //   This enforces x >= sigma(x) lexicographically.

    for (Index j = 0; j < num_cols &&
         j < static_cast<Index>(generator.size()); ++j) {
        Index sigma_j = generator[j];
        if (sigma_j <= j) continue;  // only j < sigma(j) to avoid double-counting
        if (sigma_j >= num_cols) continue;
        if (col_type[j] == VarType::Continuous) continue;
        if (col_type[sigma_j] == VarType::Continuous) continue;

        // Add x_j - x_{sigma(j)} >= 0 contribution.
        constraint.col_indices.push_back(j);
        constraint.coefficients.push_back(1.0);
        constraint.col_indices.push_back(sigma_j);
        constraint.coefficients.push_back(-1.0);
    }

    constraint.lower = 0.0;
    constraint.upper = kInf;
    constraint.name = std::format("symresack_{}", constraint_idx);
    return constraint;
}

SymbreakConstraint SymbreakGenerator::generateLexLeader(
    const Permutation& generator,
    const std::vector<VarType>& col_type,
    Index num_cols,
    Index /*constraint_idx*/) {
    SymbreakConstraint constraint;
    constraint.type = SymbreakType::Lexicographic;

    // Lex-leader: for the first variable j moved by sigma,
    //   x_j >= x_{sigma(j)}
    // This is a simple but effective symmetry-breaking inequality.

    for (Index j = 0; j < num_cols &&
         j < static_cast<Index>(generator.size()); ++j) {
        Index sigma_j = generator[j];
        if (sigma_j == j) continue;
        if (sigma_j < 0 || sigma_j >= num_cols) continue;
        if (col_type[j] == VarType::Continuous) continue;

        // x_j - x_{sigma(j)} >= 0
        constraint.col_indices = {j, sigma_j};
        constraint.coefficients = {1.0, -1.0};
        constraint.lower = 0.0;
        constraint.upper = kInf;
        constraint.name = std::format("lexleader_{}_{}", j, sigma_j);
        break;
    }

    return constraint;
}

std::vector<std::vector<Index>> SymbreakGenerator::detectOrbitope(
    const std::vector<Permutation>& generators,
    const std::vector<VarType>& col_type,
    Index num_cols) {
    // Detect if generators form a symmetric group on a set of columns.
    // An orbitope exists when a set of k variables form a complete orbit
    // under generators that act as the symmetric group S_k on those variables.

    // For now, detect simple column orbits where all members are binary
    // and the generators act transitively.
    auto orbits = computeVariableOrbits(generators, num_cols);

    std::vector<std::vector<Index>> orbitope_columns;
    for (const auto& orbit : orbits) {
        // Check all members are binary.
        bool all_binary = true;
        for (Index j : orbit) {
            if (col_type[j] != VarType::Binary) {
                all_binary = false;
                break;
            }
        }
        if (all_binary && orbit.size() >= 3) {
            orbitope_columns.push_back(orbit);
        }
    }

    return orbitope_columns;
}

std::vector<SymbreakConstraint> SymbreakGenerator::generate(
    const std::vector<Permutation>& generators,
    const std::vector<VarType>& col_type,
    Index num_cols,
    Index max_constraints) const {
    std::vector<SymbreakConstraint> constraints;

    if (generators.empty()) return constraints;

    // Strategy: generate symresack constraints for binary generators,
    // lex-leader for general integer generators.
    Index constraint_idx = 0;

    for (const auto& gen : generators) {
        if (constraint_idx >= max_constraints) break;

        // Check if this generator only moves binary variables.
        bool all_binary = true;
        bool has_moved = false;
        for (Index j = 0; j < num_cols &&
             j < static_cast<Index>(gen.size()); ++j) {
            if (gen[j] != j) {
                has_moved = true;
                if (col_type[j] != VarType::Binary) {
                    all_binary = false;
                    break;
                }
            }
        }
        if (!has_moved) continue;

        if (all_binary) {
            auto c = generateSymresack(gen, col_type, num_cols, constraint_idx);
            if (!c.col_indices.empty()) {
                constraints.push_back(std::move(c));
                ++constraint_idx;
            }
        } else {
            auto c = generateLexLeader(gen, col_type, num_cols, constraint_idx);
            if (!c.col_indices.empty()) {
                constraints.push_back(std::move(c));
                ++constraint_idx;
            }
        }
    }

    // Also try orbitope detection for additional constraints.
    auto orbitopes = detectOrbitope(generators, col_type, num_cols);
    for (const auto& orbit : orbitopes) {
        if (constraint_idx >= max_constraints) break;
        // For each orbitope, add pairwise ordering: x_{orbit[0]} >= x_{orbit[1]}
        // >= ... >= x_{orbit[k-1]}
        for (std::size_t i = 0; i + 1 < orbit.size(); ++i) {
            if (constraint_idx >= max_constraints) break;
            SymbreakConstraint c;
            c.type = SymbreakType::Orbitope;
            c.col_indices = {orbit[i], orbit[i + 1]};
            c.coefficients = {1.0, -1.0};
            c.lower = 0.0;
            c.upper = kInf;
            c.name = std::format("orbitope_{}_{}", orbit[i], orbit[i + 1]);
            constraints.push_back(std::move(c));
            ++constraint_idx;
        }
    }

    return constraints;
}

Index SymbreakGenerator::addConstraints(
    LpProblem& problem,
    const std::vector<SymbreakConstraint>& constraints) {
    if (constraints.empty()) return 0;

    // Ensure row_names is sized correctly.
    if (problem.row_names.size() < static_cast<std::size_t>(problem.num_rows)) {
        problem.row_names.resize(static_cast<std::size_t>(problem.num_rows));
    }

    Index added = 0;
    for (const auto& c : constraints) {
        if (c.col_indices.empty()) continue;

        // Sort entries by column index for CSR format.
        std::vector<std::pair<Index, Real>> entries;
        entries.reserve(c.col_indices.size());
        for (std::size_t i = 0; i < c.col_indices.size(); ++i) {
            entries.emplace_back(c.col_indices[i], c.coefficients[i]);
        }

        // Merge duplicate column entries.
        std::sort(entries.begin(), entries.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });

        std::vector<Index> cols;
        std::vector<Real> vals;
        cols.reserve(entries.size());
        vals.reserve(entries.size());

        for (std::size_t i = 0; i < entries.size(); ++i) {
            if (!cols.empty() && cols.back() == entries[i].first) {
                vals.back() += entries[i].second;
            } else {
                cols.push_back(entries[i].first);
                vals.push_back(entries[i].second);
            }
        }

        // Remove near-zero entries.
        std::vector<Index> final_cols;
        std::vector<Real> final_vals;
        for (std::size_t i = 0; i < cols.size(); ++i) {
            if (std::abs(vals[i]) > 1e-12) {
                final_cols.push_back(cols[i]);
                final_vals.push_back(vals[i]);
            }
        }

        if (final_cols.empty()) continue;

        problem.matrix.addRow(
            std::span<const Index>(final_cols.data(), final_cols.size()),
            std::span<const Real>(final_vals.data(), final_vals.size()));
        problem.row_lower.push_back(c.lower);
        problem.row_upper.push_back(c.upper);
        problem.row_names.push_back(c.name);
        ++added;
    }

    problem.num_rows = problem.matrix.numRows();
    return added;
}

std::vector<SymbreakConstraint> SymbreakGenerator::generateLexFixing(
    const std::vector<std::vector<Index>>& orbits,
    const std::vector<VarType>& col_type,
    Index num_cols) {
    std::vector<SymbreakConstraint> constraints;

    for (const auto& orbit : orbits) {
        if (orbit.size() < 2) continue;
        Index canon = orbit.front();
        if (col_type[canon] == VarType::Continuous) continue;

        for (std::size_t i = 1; i < orbit.size(); ++i) {
            Index j = orbit[i];
            if (j >= num_cols) continue;
            if (col_type[j] == VarType::Continuous) continue;

            SymbreakConstraint c;
            c.type = SymbreakType::Lexicographic;
            c.col_indices = {canon, j};
            c.coefficients = {1.0, -1.0};
            c.lower = 0.0;
            c.upper = kInf;
            c.name = std::format("lexfix_{}_{}", canon, j);
            constraints.push_back(std::move(c));
        }
    }

    return constraints;
}

Index SymbreakGenerator::aggregateSymmetricConstraints(
    LpProblem& problem,
    const std::vector<Permutation>& generators) {
    if (generators.empty() || problem.num_rows == 0) return 0;

    // For each pair of rows, check if any generator maps one to the other.
    // If so, they are symmetric and we can tighten by taking the intersection
    // of their feasible regions (tighter of the two bounds).

    Index aggregated = 0;
    std::vector<bool> row_merged(static_cast<std::size_t>(problem.num_rows), false);

    // Build a signature for each row for quick comparison.
    struct RowSig {
        std::vector<std::pair<Index, Real>> entries;
        Real lower;
        Real upper;
    };

    auto getRowSig = [&](Index row) -> RowSig {
        RowSig sig;
        auto r = problem.matrix.row(row);
        sig.entries.reserve(static_cast<std::size_t>(r.size()));
        for (Index k = 0; k < r.size(); ++k) {
            sig.entries.emplace_back(r.indices[k], r.values[k]);
        }
        std::sort(sig.entries.begin(), sig.entries.end());
        sig.lower = problem.row_lower[row];
        sig.upper = problem.row_upper[row];
        return sig;
    };

    // For each generator, check if it maps row i to row j.
    for (const auto& gen : generators) {
        for (Index i = 0; i < problem.num_rows; ++i) {
            if (row_merged[i]) continue;

            auto sig_i = getRowSig(i);

            // Apply generator to row i: permute column indices.
            RowSig permuted;
            permuted.entries.reserve(sig_i.entries.size());
            for (const auto& [col, val] : sig_i.entries) {
                Index new_col = (col < static_cast<Index>(gen.size()))
                                    ? gen[col] : col;
                permuted.entries.emplace_back(new_col, val);
            }
            std::sort(permuted.entries.begin(), permuted.entries.end());
            permuted.lower = sig_i.lower;
            permuted.upper = sig_i.upper;

            // Check if permuted row matches any other row.
            for (Index j = i + 1; j < problem.num_rows; ++j) {
                if (row_merged[j]) continue;

                auto sig_j = getRowSig(j);
                if (permuted.entries == sig_j.entries &&
                    std::abs(permuted.lower - sig_j.lower) < 1e-12 &&
                    std::abs(permuted.upper - sig_j.upper) < 1e-12) {
                    // Row j is the image of row i under this generator.
                    // Tighten row i bounds to the intersection.
                    problem.row_lower[i] = std::max(problem.row_lower[i],
                                                     problem.row_lower[j]);
                    problem.row_upper[i] = std::min(problem.row_upper[i],
                                                     problem.row_upper[j]);
                    row_merged[j] = true;
                    ++aggregated;
                    break;
                }
            }
        }
    }

    return aggregated;
}

// ---------------------------------------------------------------------------
// IsomorphismPruner
// ---------------------------------------------------------------------------

void IsomorphismPruner::setGenerators(
    const std::vector<Permutation>& generators, Index num_vars) {
    generators_ = generators;
    num_vars_ = num_vars;
    explored_hashes_.clear();
    num_pruned_ = 0;
}

std::size_t IsomorphismPruner::canonicalHash(
    const std::vector<Real>& col_lower,
    const std::vector<Real>& col_upper) const {
    // Compute a canonical hash for the bound configuration.
    // We sort the bound fingerprints under generator action to find
    // a canonical representative.

    // For each variable, create a fingerprint (lower, upper).
    struct Fingerprint {
        std::uint64_t lb_bits;
        std::uint64_t ub_bits;
    };

    std::vector<Fingerprint> fps(static_cast<std::size_t>(num_vars_));
    for (Index j = 0; j < num_vars_; ++j) {
        fps[j] = {std::bit_cast<std::uint64_t>(col_lower[j]),
                   std::bit_cast<std::uint64_t>(col_upper[j])};
    }

    // Try each generator and its inverse, keep the lexicographically
    // smallest fingerprint sequence.
    auto hashFingerprints = [](const std::vector<Fingerprint>& f) -> std::size_t {
        std::size_t h = 0;
        for (const auto& fp : f) {
            h ^= std::hash<std::uint64_t>()(fp.lb_bits) +
                 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            h ^= std::hash<std::uint64_t>()(fp.ub_bits) +
                 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        }
        return h;
    };

    std::size_t best_hash = hashFingerprints(fps);

    for (const auto& gen : generators_) {
        std::vector<Fingerprint> permuted(static_cast<std::size_t>(num_vars_));
        for (Index j = 0; j < num_vars_; ++j) {
            Index target = (j < static_cast<Index>(gen.size())) ? gen[j] : j;
            if (target < num_vars_) {
                permuted[target] = fps[j];
            }
        }
        std::size_t h = hashFingerprints(permuted);
        best_hash = std::min(best_hash, h);
    }

    return best_hash;
}

bool IsomorphismPruner::canPrune(
    const std::vector<Real>& col_lower,
    const std::vector<Real>& col_upper) const {
    if (generators_.empty()) return false;

    std::size_t h = canonicalHash(col_lower, col_upper);

    for (std::size_t eh : explored_hashes_) {
        if (eh == h) {
            ++num_pruned_;
            return true;
        }
    }
    return false;
}

void IsomorphismPruner::recordExplored(
    const std::vector<Real>& col_lower,
    const std::vector<Real>& col_upper) {
    if (generators_.empty()) return;
    explored_hashes_.push_back(canonicalHash(col_lower, col_upper));

    // Limit memory: keep at most a fixed number of configurations.
    static constexpr std::size_t kMaxExplored = 10000;
    if (explored_hashes_.size() > kMaxExplored) {
        explored_hashes_.erase(explored_hashes_.begin(),
                               explored_hashes_.begin() +
                                   static_cast<std::ptrdiff_t>(
                                       explored_hashes_.size() / 2));
    }
}

void IsomorphismPruner::clear() {
    explored_hashes_.clear();
    num_pruned_ = 0;
}

}  // namespace mipx
