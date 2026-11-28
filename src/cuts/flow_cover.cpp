#include "mipx/separators.h"

#include <algorithm>
#include <cmath>

namespace mipx {

namespace {

struct FlowLink {
    Index activation;
    Real capacity;
};

struct Candidate {
    Index row;
    Real sign;
    Real rhs;
    Index flow;
    Index activation;
    Real efficacy;
};

bool isNonnegativeContinuous(const LpProblem& problem, Index col) {
    return problem.col_type[col] == VarType::Continuous && problem.col_lower[col] >= 0.0;
}

}  // namespace

Int SeparatorManager::separateFlowCover(const LpProblem& problem, std::span<const Real> primals,
                                        CutPool& pool, CutFamilyStats& stats) {
    const Index global_rows =
        global_row_count_ < 0 ? problem.num_rows : std::min(problem.num_rows, global_row_count_);
    const auto is_local = [&](Index row) { return std::ranges::binary_search(local_rows_, row); };
    std::vector<std::vector<FlowLink>> links(problem.num_cols);
    bool has_links = false;
    for (Index i = 0; i < global_rows && !pastDeadline(); ++i) {
        const auto row = problem.matrix.row(i);
        if (row.indices.size() != 2 || is_local(i)) {
            continue;
        }
        const int flow_pos = isNonnegativeContinuous(problem, row.indices[0]) ? 0 : 1;
        const Index flow = row.indices[flow_pos];
        const Index activation = row.indices[1 - flow_pos];
        if (!isNonnegativeContinuous(problem, flow) || !isBinaryColumn(problem, activation) ||
            !(problem.col_lower[activation] >= 0.0 && problem.col_upper[activation] <= 1.0)) {
            continue;
        }
        const Real flow_coef = row.values[flow_pos];
        const Real activation_coef = row.values[1 - flow_pos];
        const Real bound = flow_coef > 0.0 ? problem.row_upper[i] : problem.row_lower[i];
        if (bound != 0.0 || !std::isfinite(flow_coef) || flow_coef == 0.0) {
            continue;
        }
        const Real capacity = -activation_coef / flow_coef;
        if (std::isfinite(capacity) && capacity > 0.0) {
            links[flow].push_back({activation, capacity});
            has_links = true;
        }
    }
    if (!has_links) {
        return 0;
    }

    std::vector<Candidate> candidates;
    for (Index i = 0; i < global_rows && !pastDeadline(); ++i) {
        if (is_local(i)) {
            continue;
        }
        const auto row = problem.matrix.row(i);
        for (const Real sign : {1.0, -1.0}) {
            const Real rhs = sign > 0.0 ? problem.row_upper[i] : -problem.row_lower[i];
            if (!(rhs > 0.0 && std::isfinite(rhs))) {
                continue;
            }
            ++stats.attempted;
            Real negative_activity = 0.0;
            Real fixed_norm_sq = rhs * rhs;
            bool valid = true;
            for (std::size_t k = 0; k < row.indices.size(); ++k) {
                if (pastDeadline()) {
                    return 0;
                }
                const Real coef = sign * row.values[k];
                if (coef == 0.0) {
                    continue;
                }
                const Index col = row.indices[k];
                if (!std::isfinite(coef) || !isNonnegativeContinuous(problem, col)) {
                    valid = false;
                    break;
                }
                if (coef < 0.0) {
                    negative_activity += coef * primals[col];
                    fixed_norm_sq += coef * coef;
                }
            }
            if (!valid) {
                continue;
            }

            Candidate best{i, sign, rhs, -1, -1, 0.0};
            Real best_violation = 0.0;
            for (std::size_t k = 0; k < row.indices.size(); ++k) {
                if (pastDeadline()) {
                    return 0;
                }
                const Index flow = row.indices[k];
                const Real coef = sign * row.values[k];
                if (coef <= 0.0) {
                    continue;
                }
                for (const auto& [activation, capacity] : links[flow]) {
                    if (pastDeadline()) {
                        return 0;
                    }
                    const Real violation =
                        coef * primals[flow] + negative_activity - rhs * primals[activation];
                    if (coef * capacity > rhs && violation > best_violation) {
                        best_violation = violation;
                        best = {i,          sign,
                                rhs,        flow,
                                activation, violation / std::sqrt(fixed_norm_sq + coef * coef)};
                    }
                }
            }
            if (best.flow >= 0 && std::isfinite(best.efficacy) && best.efficacy > 0.0) {
                candidates.push_back(best);
            }
        }
    }

    std::ranges::stable_sort(candidates, std::ranges::greater{}, &Candidate::efficacy);
    Int accepted = 0;
    for (const auto& [i, sign, rhs, flow, activation, efficacy] : candidates) {
        if (accepted >= max_cuts_per_family_ || pastDeadline()) {
            break;
        }
        // With x=0 the link forces y=0 and the negative terms are nonpositive.
        // With x=1 dropping the other positive terms leaves the same bound b.
        Cut cut;
        cut.family = CutFamily::FlowCover;
        cut.upper = 0.0;
        const auto row = problem.matrix.row(i);
        for (std::size_t k = 0; k < row.indices.size(); ++k) {
            if (pastDeadline()) {
                return accepted;
            }
            const Real coef = sign * row.values[k];
            if (coef < 0.0 || (row.indices[k] == flow && coef > 0.0)) {
                cut.indices.push_back(row.indices[k]);
                cut.values.push_back(coef);
            }
        }
        const auto position = std::ranges::lower_bound(cut.indices, activation);
        cut.values.insert(cut.values.begin() + (position - cut.indices.begin()), -rhs);
        cut.indices.insert(position, activation);
        accepted += addViolatedCut(std::move(cut), primals, pool, stats);
    }
    return accepted;
}

}  // namespace mipx
