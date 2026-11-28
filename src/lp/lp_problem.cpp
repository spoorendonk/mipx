#include "mipx/lp_problem.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <span>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace mipx {

bool LpProblem::hasIntegers() const {
    return std::ranges::any_of(col_type, [](VarType t) {
        return t == VarType::Integer || t == VarType::Binary || t == VarType::SemiContinuous ||
               t == VarType::SemiInteger;
    });
}

namespace {

Index addColumn(LpProblem& problem, Real obj, Real lower, Real upper, VarType type,
                const std::string& name, Real semi_lower = 0.0) {
    const Index idx = problem.num_cols++;
    problem.obj.push_back(obj);
    problem.col_lower.push_back(lower);
    problem.col_upper.push_back(upper);
    problem.col_type.push_back(type);
    problem.col_names.push_back(name);
    problem.col_semi_lower.push_back(semi_lower);
    return idx;
}

Index addRow(LpProblem& problem, std::vector<Triplet>& trips, std::span<const Index> indices,
             std::span<const Real> values, Real lower, Real upper, const std::string& name) {
    const Index row = problem.num_rows++;
    problem.row_lower.push_back(lower);
    problem.row_upper.push_back(upper);
    problem.row_names.push_back(name);
    for (Index k = 0; k < static_cast<Index>(indices.size()); ++k) {
        if (std::abs(values[k]) <= 1e-16)
            continue;
        trips.push_back({row, indices[k], values[k]});
    }
    return row;
}

std::vector<Triplet> extractTriplets(const LpProblem& problem) {
    std::vector<Triplet> trips;
    trips.reserve(problem.matrix.numNonzeros());
    for (Index i = 0; i < problem.num_rows; ++i) {
        auto row = problem.matrix.row(i);
        for (Index k = 0; k < row.size(); ++k) {
            trips.push_back({i, row.indices[k], row.values[k]});
        }
    }
    return trips;
}

void addSos2IntervalRows(LpProblem& out, std::vector<Triplet>& trips,
                         const LpProblem::SosConstraint& sos, std::span<const Index> order) {
    // The first selector column distinguishes sets even when their names coincide.
    const std::string base =
        (sos.name.empty() ? "sos2" : sos.name) + "_" + std::to_string(out.num_cols);
    std::vector<Index> selectors;
    selectors.reserve(order.size() - 1);
    for (std::size_t p = 0; p + 1 < order.size(); ++p) {
        selectors.push_back(addColumn(out, 0.0, 0.0, 1.0, VarType::Binary,
                                      base + "_interval_" + std::to_string(p)));
    }
    const std::vector<Real> ones(selectors.size(), 1.0);
    addRow(out, trips, selectors, ones, -kInf, 1.0, base + "_select");

    // With one selected interval, only its two endpoints can be nonzero.
    // An empty support is also allowed, including when no interval is selected.
    for (std::size_t p = 0; p < order.size(); ++p) {
        const Index var = sos.vars[order[p]];
        const Real neg_upper = -out.col_upper[var];
        std::array<Index, 3> inds = {var, 0, 0};
        const std::array<Real, 3> vals = {1.0, neg_upper, neg_upper};
        std::size_t count = 1;
        if (p > 0) {
            inds[count++] = selectors[p - 1];
        }
        if (p < selectors.size()) {
            inds[count++] = selectors[p];
        }
        addRow(out, trips, std::span(inds).first(count), std::span(vals).first(count), -kInf, 0.0,
               base + "_" + std::to_string(p) + "_link");
    }
}

}  // namespace

void validateModelFeatures(const LpProblem& problem) {
    if (!problem.col_semi_lower.empty() &&
        static_cast<Index>(problem.col_semi_lower.size()) != problem.num_cols) {
        throw std::runtime_error("col_semi_lower size must match num_cols");
    }

    for (Index j = 0; j < problem.num_cols; ++j) {
        if (j >= static_cast<Index>(problem.col_type.size())) {
            throw std::runtime_error("col_type size must match num_cols");
        }
        const VarType t = problem.col_type[j];
        if (t != VarType::SemiContinuous && t != VarType::SemiInteger)
            continue;

        if (problem.col_lower[j] < -1e-9) {
            throw std::runtime_error("semi variable lower bound must be >= 0");
        }
        if (!std::isfinite(problem.col_upper[j])) {
            throw std::runtime_error("semi variable requires finite upper bound");
        }
        const Real semi_lb = problem.col_semi_lower.empty()
                                 ? std::max<Real>(0.0, problem.col_lower[j])
                                 : problem.col_semi_lower[j];
        if (semi_lb < -1e-9 || semi_lb > problem.col_upper[j] + 1e-9) {
            throw std::runtime_error("invalid semi lower bound");
        }
    }

    for (const auto& ic : problem.indicator_constraints) {
        if (ic.binary_var < 0 || ic.binary_var >= problem.num_cols) {
            throw std::runtime_error("indicator binary variable out of range");
        }
        const VarType bt = problem.col_type[ic.binary_var];
        const bool is_binary_type =
            (bt == VarType::Binary) ||
            (bt == VarType::Integer && problem.col_lower[ic.binary_var] >= -1e-9 &&
             problem.col_upper[ic.binary_var] <= 1.0 + 1e-9);
        if (!is_binary_type) {
            throw std::runtime_error("indicator requires binary trigger variable");
        }
        if (ic.indices.size() != ic.values.size()) {
            throw std::runtime_error("indicator indices/values size mismatch");
        }
        if (!std::isfinite(ic.lower) && !std::isfinite(ic.upper)) {
            throw std::runtime_error("indicator must have finite lower or upper");
        }
        for (Index k = 0; k < static_cast<Index>(ic.indices.size()); ++k) {
            const Index j = ic.indices[k];
            if (j < 0 || j >= problem.num_cols) {
                throw std::runtime_error("indicator references variable out of range");
            }
            if (!std::isfinite(problem.col_lower[j]) || !std::isfinite(problem.col_upper[j])) {
                throw std::runtime_error(
                    "indicator fallback needs finite bounds on referenced variables");
            }
        }
    }

    for (const auto& sos : problem.sos_constraints) {
        if (sos.vars.empty()) {
            throw std::runtime_error("SOS set cannot be empty");
        }
        if (!sos.weights.empty() && sos.weights.size() != sos.vars.size()) {
            throw std::runtime_error("SOS weights size mismatch");
        }
        std::unordered_set<Index> seen;
        for (Index v : sos.vars) {
            if (v < 0 || v >= problem.num_cols) {
                throw std::runtime_error("SOS variable out of range");
            }
            if (!seen.insert(v).second) {
                throw std::runtime_error("SOS variables must be unique");
            }
            if (problem.col_lower[v] < -1e-9) {
                throw std::runtime_error("SOS fallback requires nonnegative variable bounds");
            }
            if (!std::isfinite(problem.col_upper[v])) {
                throw std::runtime_error("SOS fallback requires finite variable upper bounds");
            }
        }
    }
}

bool hasAdvancedModelFeatures(const LpProblem& problem) {
    if (!problem.sos_constraints.empty() || !problem.indicator_constraints.empty()) {
        return true;
    }
    return std::ranges::any_of(problem.col_type, [](VarType t) {
        return t == VarType::SemiContinuous || t == VarType::SemiInteger;
    });
}

bool roundIntegerBounds(LpProblem& problem, Real tol) {
    bool ok = true;
    for (Index j = 0; j < problem.num_cols; ++j) {
        if (problem.col_type[j] != VarType::Integer && problem.col_type[j] != VarType::Binary) {
            continue;
        }
        problem.col_lower[j] = std::ceil(problem.col_lower[j] - tol);
        problem.col_upper[j] = std::floor(problem.col_upper[j] + tol);
        if (problem.col_lower[j] > problem.col_upper[j]) {
            ok = false;
        }
    }
    return ok;
}

LpProblem linearizeModelFeatures(const LpProblem& problem) {
    return linearizeModelFeatures(problem, false);
}

LpProblem linearizeModelFeatures(const LpProblem& problem, bool sos2_adjacent_intervals) {
    return linearizeModelFeatures(problem, sos2_adjacent_intervals
                                               ? Sos2Formulation::AdjacentIntervals
                                               : Sos2Formulation::Pairwise);
}

LpProblem linearizeModelFeatures(const LpProblem& problem, Sos2Formulation sos2_formulation) {
    if (sos2_formulation != Sos2Formulation::Native && !problem.native_sos2.empty()) {
        LpProblem unlowered = problem;
        for (auto& members : unlowered.native_sos2) {
            unlowered.sos_constraints.push_back(
                {LpProblem::SosConstraint::Type::Sos2,
                 std::move(members),
                 {},
                 "native_sos2_" + std::to_string(unlowered.sos_constraints.size())});
        }
        unlowered.native_sos2.clear();
        return linearizeModelFeatures(unlowered, sos2_formulation);
    }
    validateModelFeatures(problem);

    if (!hasAdvancedModelFeatures(problem)) {
        LpProblem out = problem;
        if (out.col_semi_lower.empty()) {
            out.col_semi_lower.assign(static_cast<std::size_t>(out.num_cols), 0.0);
        }
        return out;
    }

    LpProblem out = problem;
    if (out.col_semi_lower.empty()) {
        out.col_semi_lower.assign(static_cast<std::size_t>(out.num_cols), 0.0);
    }
    std::vector<Triplet> trips = extractTriplets(problem);

    const Index orig_cols = problem.num_cols;
    for (Index j = 0; j < orig_cols; ++j) {
        const VarType t = out.col_type[j];
        if (t != VarType::SemiContinuous && t != VarType::SemiInteger)
            continue;

        const Real L = out.col_semi_lower[j] > 0.0 ? out.col_semi_lower[j]
                                                   : std::max<Real>(0.0, out.col_lower[j]);
        const Real U = out.col_upper[j];
        out.col_type[j] = (t == VarType::SemiInteger) ? VarType::Integer : VarType::Continuous;
        out.col_lower[j] = 0.0;
        out.col_semi_lower[j] = 0.0;

        const std::string base =
            (j < static_cast<Index>(out.col_names.size()) && !out.col_names[j].empty())
                ? out.col_names[j]
                : ("x" + std::to_string(j));
        const Index z = addColumn(out, 0.0, 0.0, 1.0, VarType::Binary, base + "_semi_active");

        std::array<Index, 2> inds = {j, z};
        std::array<Real, 2> vals = {1.0, -U};
        addRow(out, trips, inds, vals, -kInf, 0.0, base + "_semi_ub");

        if (L > 1e-12) {
            std::array<Real, 2> vals_lb = {1.0, -L};
            addRow(out, trips, inds, vals_lb, 0.0, kInf, base + "_semi_lb");
        }
    }

    for (const auto& ic : problem.indicator_constraints) {
        Real min_activity = 0.0;
        Real max_activity = 0.0;
        for (Index k = 0; k < static_cast<Index>(ic.indices.size()); ++k) {
            const Index j = ic.indices[k];
            const Real a = ic.values[k];
            if (a >= 0.0) {
                min_activity += a * out.col_lower[j];
                max_activity += a * out.col_upper[j];
            } else {
                min_activity += a * out.col_upper[j];
                max_activity += a * out.col_lower[j];
            }
        }

        if (std::isfinite(ic.upper)) {
            const Real M = std::max<Real>(0.0, max_activity - ic.upper) + 1e-9;
            std::vector<Index> inds = ic.indices;
            std::vector<Real> vals = ic.values;
            inds.push_back(ic.binary_var);
            vals.push_back(ic.active_value ? M : -M);
            const Real ub = ic.active_value ? (ic.upper + M) : ic.upper;
            addRow(out, trips, inds, vals, -kInf, ub,
                   ic.name.empty() ? "indicator_up" : (ic.name + "_up"));
        }
        if (std::isfinite(ic.lower)) {
            const Real M = std::max<Real>(0.0, ic.lower - min_activity) + 1e-9;
            std::vector<Index> inds = ic.indices;
            std::vector<Real> vals = ic.values;
            inds.push_back(ic.binary_var);
            vals.push_back(ic.active_value ? -M : M);
            const Real lb = ic.active_value ? (ic.lower - M) : ic.lower;
            addRow(out, trips, inds, vals, lb, kInf,
                   ic.name.empty() ? "indicator_lo" : (ic.name + "_lo"));
        }
    }

    for (const auto& sos : problem.sos_constraints) {
        const auto formulation = sos.type == LpProblem::SosConstraint::Type::Sos2
                                     ? sos2_formulation
                                     : Sos2Formulation::Pairwise;
        if (formulation != Sos2Formulation::Pairwise && sos.vars.size() <= 2) {
            continue;  // Every support of this set already satisfies SOS2.
        }

        std::vector<Index> order(sos.vars.size());
        std::iota(order.begin(), order.end(), 0);
        if (!sos.weights.empty()) {
            std::sort(order.begin(), order.end(), [&](Index a, Index b) {
                if (sos.weights[a] != sos.weights[b])
                    return sos.weights[a] < sos.weights[b];
                return sos.vars[a] < sos.vars[b];
            });
        }

        if (formulation == Sos2Formulation::Native) {
            auto& members = out.native_sos2.emplace_back();
            members.reserve(order.size());
            for (Index p : order) {
                members.push_back(sos.vars[p]);
            }
            continue;
        }
        if (formulation == Sos2Formulation::AdjacentIntervals) {
            addSos2IntervalRows(out, trips, sos, order);
            continue;
        }

        std::vector<Index> y(order.size(), -1);
        for (Index p = 0; p < static_cast<Index>(order.size()); ++p) {
            const Index var = sos.vars[order[p]];
            const std::string base = sos.name.empty() ? ("sos_" + std::to_string(var))
                                                      : (sos.name + "_" + std::to_string(p));
            y[p] = addColumn(out, 0.0, 0.0, 1.0, VarType::Binary, base + "_active");
            std::array<Index, 2> inds = {var, y[p]};
            std::array<Real, 2> vals = {1.0, -out.col_upper[var]};
            addRow(out, trips, inds, vals, -kInf, 0.0, base + "_link");
        }

        std::vector<Real> ones(y.size(), 1.0);
        addRow(out, trips, y, ones, -kInf,
               sos.type == LpProblem::SosConstraint::Type::Sos1 ? 1.0 : 2.0,
               sos.name.empty() ? "sos_card" : (sos.name + "_card"));

        if (sos.type == LpProblem::SosConstraint::Type::Sos2) {
            for (Index i = 0; i < static_cast<Index>(y.size()); ++i) {
                for (Index j = i + 2; j < static_cast<Index>(y.size()); ++j) {
                    std::array<Index, 2> inds = {y[i], y[j]};
                    std::array<Real, 2> vals = {1.0, 1.0};
                    addRow(out, trips, inds, vals, -kInf, 1.0,
                           sos.name.empty() ? "sos2_adj" : (sos.name + "_adj"));
                }
            }
        }
    }

    out.sos_constraints.clear();
    out.indicator_constraints.clear();
    out.matrix = SparseMatrix(out.num_rows, out.num_cols, std::move(trips));
    return out;
}

std::pair<std::size_t, std::size_t> sos2Support(const LpProblem& problem,
                                                std::span<const Index> members,
                                                std::span<const Real> values, Real tolerance) {
    std::size_t first = members.size();
    std::size_t last = first;
    for (std::size_t p = 0; p < members.size(); ++p) {
        const Index j = members[p];
        // A small nonzero can still matter to the objective or a scaled row.
        if (std::abs(values[j]) > tolerance || problem.col_lower[j] > 0.0 ||
            problem.col_upper[j] < 0.0) {
            first = std::min(first, p);
            last = p;
        }
    }
    return {first, last};
}

Index checkColumnBatch(Index num_rows, std::span<const Index> starts,
                       std::span<const Index> indices, std::span<const Real> values,
                       std::span<const Real> obj, std::span<const Real> lower,
                       std::span<const Real> upper) {
    const auto k = obj.size();
    if (lower.size() != k || upper.size() != k) {
        throw std::invalid_argument("addCols: obj, lower and upper differ in size");
    }
    if (indices.size() != values.size()) {
        throw std::invalid_argument("addCols: indices and values differ in size");
    }
    const auto nnz = static_cast<Index>(indices.size());
    if (starts.size() != k && starts.size() != k + 1) {
        throw std::invalid_argument("addCols: starts needs one entry per column, or one more");
    }
    if (starts.size() == k + 1 && starts[k] != nnz) {
        throw std::invalid_argument("addCols: the last start is not the number of entries");
    }
    Index prev = 0;
    for (std::size_t t = 0; t < k; ++t) {
        if (starts[t] < prev || starts[t] > nnz) {
            throw std::invalid_argument("addCols: starts out of order or out of range");
        }
        prev = starts[t];
    }
    for (const Index i : indices) {
        if (i < 0 || i >= num_rows) {
            throw std::invalid_argument("addCols: row index out of range");
        }
    }
    for (std::size_t t = 0; t < k; ++t) {
        if (std::isnan(obj[t]) || std::isnan(lower[t]) || std::isnan(upper[t]) ||
            std::isinf(obj[t]) || lower[t] == kInf || upper[t] == -kInf) {
            throw std::invalid_argument("addCols: NaN, infinite cost or bound on the wrong side");
        }
    }
    for (const Real v : values) {
        if (!std::isfinite(v)) {
            throw std::invalid_argument("addCols: coefficient not finite");
        }
    }
    return static_cast<Index>(k);
}

void appendColumns(LpProblem& problem, std::span<const Index> starts,
                   std::span<const Index> indices, std::span<const Real> values,
                   std::span<const Real> obj, std::span<const Real> lower,
                   std::span<const Real> upper) {
    const auto k = static_cast<Index>(obj.size());
    const auto nnz = static_cast<Index>(indices.size());
    const Index first = problem.num_cols;
    std::vector<Triplet> trips = extractTriplets(problem);
    trips.reserve(trips.size() + indices.size());
    for (Index t = 0; t < k; ++t) {
        for (Index p = starts[t]; p < columnBatchEnd(starts, t, nnz); ++p) {
            trips.push_back({indices[p], first + t, values[p]});
        }
    }
    // The optional per-column vectors grow only when they are kept for every
    // column, so a problem that carries none stays without them.
    const auto sized = [&](const auto& v) { return static_cast<Index>(v.size()) == first; };
    const bool types = sized(problem.col_type);
    const bool names = sized(problem.col_names);
    const bool semi = sized(problem.col_semi_lower);
    for (Index t = 0; t < k; ++t) {
        problem.obj.push_back(obj[t]);
        problem.col_lower.push_back(lower[t]);
        problem.col_upper.push_back(upper[t]);
        if (types) {
            problem.col_type.push_back(VarType::Continuous);
        }
        if (names) {
            problem.col_names.emplace_back();
        }
        if (semi) {
            problem.col_semi_lower.push_back(0.0);
        }
    }
    problem.num_cols += k;
    problem.matrix = SparseMatrix(problem.num_rows, problem.num_cols, std::move(trips));
}

std::vector<Index> sortedColumnSet(Index num_cols, std::span<const Index> cols) {
    std::vector<Index> sorted(cols.begin(), cols.end());
    std::ranges::sort(sorted);
    sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
    if (!sorted.empty() && (sorted.front() < 0 || sorted.back() >= num_cols)) {
        throw std::invalid_argument("removeCols: column index out of range");
    }
    return sorted;
}

void removeColumns(LpProblem& problem, std::span<const Index> sorted_cols) {
    if (sorted_cols.empty()) {
        return;
    }
    if (!problem.sos_constraints.empty() || !problem.indicator_constraints.empty() ||
        !problem.native_sos2.empty()) {
        throw std::invalid_argument("removeCols: SOS and indicator constraints name columns");
    }
    const Index old_cols = problem.num_cols;
    std::vector<Index> col_map(static_cast<std::size_t>(old_cols), 0);
    Index next = 0;
    std::size_t r = 0;
    for (Index j = 0; j < old_cols; ++j) {
        if (r < sorted_cols.size() && sorted_cols[r] == j) {
            col_map[j] = -1;
            ++r;
        } else {
            col_map[j] = next++;
        }
    }
    std::vector<Triplet> trips;
    trips.reserve(static_cast<std::size_t>(problem.matrix.numNonzeros()));
    for (Index i = 0; i < problem.num_rows; ++i) {
        auto row = problem.matrix.row(i);
        for (Index p = 0; p < row.size(); ++p) {
            const Index j = col_map[row.indices[p]];
            if (j >= 0) {
                trips.push_back({i, j, row.values[p]});
            }
        }
    }
    const auto compact = [&](auto& v) {
        if (static_cast<Index>(v.size()) != old_cols) {
            return;
        }
        for (Index j = 0; j < old_cols; ++j) {
            if (col_map[j] >= 0 && col_map[j] != j) {
                v[col_map[j]] = std::move(v[j]);
            }
        }
        v.resize(static_cast<std::size_t>(next));
    };
    compact(problem.obj);
    compact(problem.col_lower);
    compact(problem.col_upper);
    compact(problem.col_type);
    compact(problem.col_names);
    compact(problem.col_semi_lower);
    problem.num_cols = next;
    problem.matrix = SparseMatrix(problem.num_rows, problem.num_cols, std::move(trips));
}

}  // namespace mipx
