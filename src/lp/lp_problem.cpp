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
        return t == VarType::Integer || t == VarType::Binary ||
               t == VarType::SemiContinuous || t == VarType::SemiInteger;
    });
}

namespace {

Index addColumn(LpProblem& problem,
                Real obj,
                Real lower,
                Real upper,
                VarType type,
                const std::string& name,
                Real semi_lower = 0.0) {
    const Index idx = problem.num_cols++;
    problem.obj.push_back(obj);
    problem.col_lower.push_back(lower);
    problem.col_upper.push_back(upper);
    problem.col_type.push_back(type);
    problem.col_names.push_back(name);
    problem.col_semi_lower.push_back(semi_lower);
    return idx;
}

Index addRow(LpProblem& problem,
             std::vector<Triplet>& trips,
             std::span<const Index> indices,
             std::span<const Real> values,
             Real lower,
             Real upper,
             const std::string& name) {
    const Index row = problem.num_rows++;
    problem.row_lower.push_back(lower);
    problem.row_upper.push_back(upper);
    problem.row_names.push_back(name);
    for (Index k = 0; k < static_cast<Index>(indices.size()); ++k) {
        if (std::abs(values[k]) <= 1e-16) continue;
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
        if (t != VarType::SemiContinuous && t != VarType::SemiInteger) continue;

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
            (bt == VarType::Integer &&
             problem.col_lower[ic.binary_var] >= -1e-9 &&
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
            if (!std::isfinite(problem.col_lower[j]) ||
                !std::isfinite(problem.col_upper[j])) {
                throw std::runtime_error(
                    "indicator fallback needs finite bounds on referenced variables");
            }
        }
    }

    for (const auto& sos : problem.sos_constraints) {
        if (sos.vars.empty()) {
            throw std::runtime_error("SOS set cannot be empty");
        }
        if (!sos.weights.empty() &&
            sos.weights.size() != sos.vars.size()) {
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

LpProblem linearizeModelFeatures(const LpProblem& problem) {
    validateModelFeatures(problem);

    const bool has_semi =
        std::ranges::any_of(problem.col_type, [](VarType t) {
            return t == VarType::SemiContinuous || t == VarType::SemiInteger;
        });
    if (!has_semi &&
        problem.sos_constraints.empty() &&
        problem.indicator_constraints.empty()) {
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
        if (t != VarType::SemiContinuous && t != VarType::SemiInteger) continue;

        const Real L = out.col_semi_lower[j] > 0.0
                           ? out.col_semi_lower[j]
                           : std::max<Real>(0.0, out.col_lower[j]);
        const Real U = out.col_upper[j];
        out.col_type[j] = (t == VarType::SemiInteger)
                              ? VarType::Integer
                              : VarType::Continuous;
        out.col_lower[j] = 0.0;
        out.col_semi_lower[j] = 0.0;

        const std::string base =
            (j < static_cast<Index>(out.col_names.size()) && !out.col_names[j].empty())
                ? out.col_names[j]
                : ("x" + std::to_string(j));
        const Index z = addColumn(out, 0.0, 0.0, 1.0, VarType::Binary,
                                  base + "_semi_active");

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
        std::vector<Index> order(sos.vars.size());
        std::iota(order.begin(), order.end(), 0);
        if (!sos.weights.empty()) {
            std::sort(order.begin(), order.end(), [&](Index a, Index b) {
                if (sos.weights[a] != sos.weights[b]) return sos.weights[a] < sos.weights[b];
                return sos.vars[a] < sos.vars[b];
            });
        }

        std::vector<Index> y(order.size(), -1);
        for (Index p = 0; p < static_cast<Index>(order.size()); ++p) {
            const Index var = sos.vars[order[p]];
            const std::string base = sos.name.empty()
                                         ? ("sos_" + std::to_string(var))
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
                           sos.name.empty() ? "sos2_adj"
                                            : (sos.name + "_adj"));
                }
            }
        }
    }

    out.sos_constraints.clear();
    out.indicator_constraints.clear();
    out.matrix = SparseMatrix(out.num_rows, out.num_cols, std::move(trips));
    return out;
}

}  // namespace mipx
