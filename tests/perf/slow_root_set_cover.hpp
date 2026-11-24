#pragma once

#include "mipx/lp_problem.h"

#include <algorithm>
#include <cstddef>
#include <random>
#include <utility>
#include <vector>

namespace mipx::test {

// Each column covers per_col distinct rows at an integer cost in [1, 100].
inline LpProblem buildSlowRootSetCover(Index rows, Index cols, Index per_col) {
    LpProblem lp;
    lp.name = "slow_root_set_cover";
    lp.sense = Sense::Minimize;
    lp.num_cols = cols;
    lp.num_rows = rows;
    std::mt19937 rng(336);
    std::uniform_int_distribution<Index> row_dist(0, rows - 1);
    std::uniform_int_distribution<Index> cost_dist(1, 100);
    std::vector<Triplet> trips;
    trips.reserve(static_cast<std::size_t>(cols) * static_cast<std::size_t>(per_col));
    std::vector<Index> picked;
    for (Index j = 0; j < cols; ++j) {
        lp.obj.push_back(static_cast<Real>(cost_dist(rng)));
        picked.clear();
        while (static_cast<Index>(picked.size()) < per_col) {
            const Index i = row_dist(rng);
            if (std::ranges::find(picked, i) == picked.end()) {
                picked.push_back(i);
                trips.push_back({i, j, 1.0});
            }
        }
    }
    lp.col_lower.assign(static_cast<std::size_t>(cols), 0.0);
    lp.col_upper.assign(static_cast<std::size_t>(cols), 1.0);
    lp.col_type.assign(static_cast<std::size_t>(cols), VarType::Binary);
    lp.row_lower.assign(static_cast<std::size_t>(rows), 1.0);
    lp.row_upper.assign(static_cast<std::size_t>(rows), kInf);
    lp.matrix = SparseMatrix(rows, cols, std::move(trips));
    return lp;
}

}  // namespace mipx::test
