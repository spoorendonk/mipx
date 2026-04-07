#include "mipx/conflict_graph.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <queue>

namespace mipx {

void ConflictGraph::build(const LpProblem& problem) {
    // Identify binary variables and build the mapping.
    col_to_bin_.assign(problem.num_cols, -1);
    bin_to_col_.clear();
    bin_to_col_.reserve(problem.num_cols);

    for (Index j = 0; j < problem.num_cols; ++j) {
        if (problem.col_type[j] == VarType::Binary) {
            col_to_bin_[j] = static_cast<Index>(bin_to_col_.size());
            bin_to_col_.push_back(j);
        }
    }
    num_binaries_ = static_cast<Index>(bin_to_col_.size());
    adj_.assign(numLiterals(), {});
    fixed_.assign(num_binaries_, false);
    fixed_value_.assign(num_binaries_, 0.0);
    num_edges_ = 0;

    if (num_binaries_ < 2) return;

    // Scan constraint rows for conflicts.
    for (Index i = 0; i < problem.num_rows; ++i) {
        // Only look at rows with finite upper bound (<= constraints).
        if (problem.row_upper[i] >= kInf) continue;
        const Real rhs = problem.row_upper[i];
        if (!std::isfinite(rhs)) continue;

        auto row = problem.matrix.row(i);

        // Collect binary variables with positive coefficients in this row.
        // Also check that the row is "valid" for conflict detection:
        // all coefficients non-negative, all variable lower bounds non-negative.
        struct BinEntry {
            Index col;
            Real coeff;
        };
        std::vector<BinEntry> binaries;
        binaries.reserve(static_cast<std::size_t>(row.size()));
        bool valid = true;

        for (Index k = 0; k < row.size(); ++k) {
            Index j = row.indices[k];
            Real a = row.values[k];
            if (a < 0.0 || problem.col_lower[j] < -1e-12) {
                valid = false;
                break;
            }
            if (problem.col_type[j] == VarType::Binary && a > 1e-9) {
                binaries.push_back({j, a});
            }
        }
        if (!valid || binaries.size() < 2) continue;

        // For each pair, check if their coefficients sum exceeds rhs.
        // This means they conflict (can't both be 1).
        for (Index p = 0; p < static_cast<Index>(binaries.size()); ++p) {
            for (Index q = p + 1; q < static_cast<Index>(binaries.size());
                 ++q) {
                if (binaries[p].coeff + binaries[q].coeff > rhs + 1e-9) {
                    Literal a = {binaries[p].col, false};
                    Literal b = {binaries[q].col, false};
                    addConflict(a, b);
                }
            }
        }

        // Check for set-packing/partitioning: all binaries, all coeffs ~1,
        // rhs ~1. This gives us a clique of all binaries in the row.
        // (Already handled pairwise above, but noting the pattern.)
    }

    // Also handle equality rows: if row_lower == row_upper == 1 and all
    // binary coefficients are 1, this is a set-partitioning constraint.
    // The pairwise conflict detection above already catches this.
}

void ConflictGraph::addConflict(Literal a, Literal b) {
    Index a_bin = toBinaryIndex(a.var);
    Index b_bin = toBinaryIndex(b.var);
    if (a_bin < 0 || b_bin < 0) return;
    if (a.var == b.var && a.complemented == b.complemented) return;

    Literal a_int = {a.var, a.complemented};
    Literal b_int = {b.var, b.complemented};

    // Check if edge already exists (linear scan for small adj lists).
    Index a_id = a_int.id();
    Index b_id = b_int.id();
    for (const auto& lit : adj_[a_id]) {
        if (lit.var == b_int.var && lit.complemented == b_int.complemented) {
            return;  // Already exists.
        }
    }

    adj_[a_id].push_back(b_int);
    adj_[b_id].push_back(a_int);
    ++num_edges_;
}

std::span<const Literal> ConflictGraph::neighbors(Literal lit) const {
    Index bin = toBinaryIndex(lit.var);
    if (bin < 0) return {};
    Index id = lit.id();
    if (id < 0 || id >= static_cast<Index>(adj_.size())) return {};
    return adj_[id];
}

bool ConflictGraph::conflicts(Literal a, Literal b) const {
    Index a_bin = toBinaryIndex(a.var);
    if (a_bin < 0) return false;
    Index a_id = a.id();
    if (a_id < 0 || a_id >= static_cast<Index>(adj_.size())) return false;
    for (const auto& lit : adj_[a_id]) {
        if (lit.var == b.var && lit.complemented == b.complemented) {
            return true;
        }
    }
    return false;
}

Index ConflictGraph::toBinaryIndex(Index col) const {
    if (col < 0 || col >= static_cast<Index>(col_to_bin_.size())) return -1;
    return col_to_bin_[col];
}

Index ConflictGraph::toOriginalIndex(Index bin_idx) const {
    if (bin_idx < 0 || bin_idx >= num_binaries_) return -1;
    return bin_to_col_[bin_idx];
}

std::vector<Index> ConflictGraph::connectedComponents(
    Index& num_components) const {
    std::vector<Index> comp(num_binaries_, -1);
    num_components = 0;

    for (Index b = 0; b < num_binaries_; ++b) {
        if (comp[b] >= 0) continue;

        // BFS from binary variable b.
        Index comp_id = num_components++;
        std::queue<Index> bfs;
        bfs.push(b);
        comp[b] = comp_id;

        while (!bfs.empty()) {
            Index cur = bfs.front();
            bfs.pop();
            Index col = bin_to_col_[cur];

            // Check both literal polarities for neighbors.
            for (int polarity = 0; polarity < 2; ++polarity) {
                Literal lit = {col, polarity == 1};
                auto nbrs = neighbors(lit);
                for (const auto& nbr : nbrs) {
                    Index nb = toBinaryIndex(nbr.var);
                    if (nb >= 0 && comp[nb] < 0) {
                        comp[nb] = comp_id;
                        bfs.push(nb);
                    }
                }
            }
        }
    }

    return comp;
}

void ConflictGraph::fixVariable(Index col, Real value) {
    Index bin = toBinaryIndex(col);
    if (bin < 0) return;
    fixed_[bin] = true;
    fixed_value_[bin] = value;
}

void ConflictGraph::unfixVariable(Index col) {
    Index bin = toBinaryIndex(col);
    if (bin < 0) return;
    fixed_[bin] = false;
}

bool ConflictGraph::isFixed(Index col) const {
    Index bin = toBinaryIndex(col);
    if (bin < 0) return false;
    return fixed_[bin];
}

Real ConflictGraph::fixedValue(Index col) const {
    Index bin = toBinaryIndex(col);
    if (bin < 0) return 0.0;
    return fixed_value_[bin];
}

}  // namespace mipx
