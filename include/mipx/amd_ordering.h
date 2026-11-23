#pragma once

#include "mipx/core.h"

#include <span>
#include <vector>

namespace mipx {

struct AmdOptions {
    // A node with more than max(16, dense_alpha·√n) neighbours is dense: it
    // is left out of the graph and ordered last (SuiteSparse AMD's rule; the
    // paper has none). Negative: no node is dense.
    double dense_alpha = 10.0;
    // Absorb an element whose remaining variables all lie in the new element.
    bool aggressive = true;
    // Workspace beyond the pattern and n entries, as a fraction of the
    // pattern's size; less means more compactions, never a different order.
    double elbow_room = 0.2;
};

/// What one ordering run did, for tests and diagnostics.
struct AmdInfo {
    Index compactions = 0;  // workspace garbage collections
    Index merges = 0;       // variables merged into a supervariable (step 5)
    Index dense = 0;        // nodes ordered last as dense
};

/// Approximate minimum degree ordering on the quotient graph, after
/// Amestoy, Davis & Duff, "An approximate minimum degree ordering
/// algorithm", SIAM J. Matrix Anal. Appl. 17(4), 1996.
///
/// The pattern is a symmetric matrix's in compressed columns: either
/// triangle or both, duplicates and the diagonal ignored; col_starts has
/// n + 1 entries and every row index lies in [0, n). On return perm[k] is
/// the k-th node to eliminate and iperm its inverse. info, if given, is
/// filled in.
void amdOrder(Index n, std::span<const Index> col_starts, std::span<const Index> row_indices,
              std::vector<Index>& perm, std::vector<Index>& iperm, const AmdOptions& opts = {},
              AmdInfo* info = nullptr);

}  // namespace mipx
