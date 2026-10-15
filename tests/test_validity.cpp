// Validity audit by integer-point enumeration (#227).
//
// Every presolve reduction and node propagator is run on small random and
// hand-built models whose feasible sets can be listed in full. A reduction
// passes when no feasible point of the original is lost (or, for a dual
// reduction, when postsolve maps the reduced model's points back to feasible
// points of equal objective and the optimum is unchanged); a propagator passes
// when every feasible point inside the node's box stays inside the box it
// tightens to.
//
// A model has integral columns plus at most one continuous column. For each
// integral assignment the continuous column's feasible values form a union of
// intervals, computed exactly from the rows, so mixed models are enumerated
// too: free, half-bounded and semi-continuous columns included.

#include "mipx/clique_table.h"
#include "mipx/conflict_graph.h"
#include "mipx/domain.h"
#include "mipx/dual_simplex.h"
#include "mipx/implication_graph.h"
#include "mipx/mip_solver.h"
#include "mipx/objective_propagator.h"
#include "mipx/presolve.h"
#include "mipx/probing.h"
#include "mipx/reduced_cost_fixer.h"
#include "mipx/symmetry.h"
#include "mipx/variable_bounds.h"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <functional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace mipx;
using Catch::Matchers::WithinAbs;

namespace {

constexpr Real kFeasTol = 1e-6;
constexpr Real kObjTol = 1e-6;

struct Interval {
    Real lo;
    Real hi;
};

bool isIntegral(VarType t) {
    return t == VarType::Integer || t == VarType::Binary || t == VarType::SemiInteger;
}

bool isSemi(VarType t) {
    return t == VarType::SemiContinuous || t == VarType::SemiInteger;
}

// The values column j may take, ignoring integrality: one interval, or
// {0} and [L, U] for a semi-continuous or semi-integer column.
std::vector<Interval> columnDomain(const LpProblem& lp, Index j) {
    if (isSemi(lp.col_type[j])) {
        const Real semi_lower = !lp.col_semi_lower.empty() && lp.col_semi_lower[j] > 0.0
                                    ? lp.col_semi_lower[j]
                                    : std::max<Real>(0.0, lp.col_lower[j]);
        return {{0.0, 0.0}, {semi_lower, lp.col_upper[j]}};
    }
    return {{lp.col_lower[j], lp.col_upper[j]}};
}

std::vector<Real> integerValues(const std::vector<Interval>& domain) {
    std::vector<Real> values;
    for (const auto& iv : domain) {
        for (Real v = std::ceil(iv.lo - 1e-9); v <= iv.hi + 1e-9; v += 1.0) {
            values.push_back(v);
        }
    }
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
    return values;
}

Real objective(const LpProblem& lp, const std::vector<Real>& x) {
    Real value = lp.obj_offset;
    for (Index j = 0; j < lp.num_cols; ++j) {
        value += lp.obj[j] * x[j];
    }
    return value;
}

// Objective in minimization terms, so smaller is always better.
Real minTerms(const LpProblem& lp, Real value) {
    return lp.sense == Sense::Maximize ? -value : value;
}

bool pointFeasible(const LpProblem& lp, const std::vector<Real>& x, Real tol) {
    for (Index j = 0; j < lp.num_cols; ++j) {
        bool in_domain = false;
        for (const auto& iv : columnDomain(lp, j)) {
            in_domain = in_domain || (x[j] >= iv.lo - tol && x[j] <= iv.hi + tol);
        }
        if (!in_domain) {
            return false;
        }
        if (isIntegral(lp.col_type[j]) && std::abs(x[j] - std::round(x[j])) > tol) {
            return false;
        }
    }
    for (Index i = 0; i < lp.num_rows; ++i) {
        auto rv = lp.matrix.row(i);
        Real act = 0.0;
        for (Index k = 0; k < rv.size(); ++k) {
            act += rv.values[k] * x[rv.indices[k]];
        }
        if (act < lp.row_lower[i] - tol || act > lp.row_upper[i] + tol) {
            return false;
        }
    }
    return true;
}

std::string describe(const std::vector<Real>& x) {
    std::ostringstream os;
    for (Real v : x) {
        os << " " << v;
    }
    return os.str();
}

std::string describe(const LpProblem& lp) {
    std::ostringstream os;
    os << (lp.sense == Sense::Maximize ? "max" : "min");
    for (Index j = 0; j < lp.num_cols; ++j) {
        os << " " << lp.obj[j] << "*x" << j;
    }
    os << " + " << lp.obj_offset << "\n";
    for (Index i = 0; i < lp.num_rows; ++i) {
        auto rv = lp.matrix.row(i);
        os << "  " << lp.row_lower[i] << " <=";
        for (Index k = 0; k < rv.size(); ++k) {
            os << " " << rv.values[k] << "*x" << rv.indices[k];
        }
        os << " <= " << lp.row_upper[i] << "\n";
    }
    for (Index j = 0; j < lp.num_cols; ++j) {
        const char* type = "C";
        switch (lp.col_type[j]) {
            case VarType::Integer:
                type = "I";
                break;
            case VarType::Binary:
                type = "B";
                break;
            case VarType::SemiContinuous:
                type = "SC";
                break;
            case VarType::SemiInteger:
                type = "SI";
                break;
            default:
                break;
        }
        os << "  x" << j << " " << type << " [" << lp.col_lower[j] << ", " << lp.col_upper[j]
           << "]\n";
    }
    return os.str();
}

// One integral assignment with the feasible values of the continuous column
// (`cont` < 0 when there is none, and then `cont_set` is empty).
struct Slice {
    std::vector<Real> x;
    Index cont = -1;
    std::vector<Interval> cont_set;
};

// Calls `fn` for every integral assignment that has a feasible completion.
void forEachFeasible(const LpProblem& lp, const std::function<void(const Slice&)>& fn) {
    Index cont = -1;
    std::vector<std::vector<Real>> values(lp.num_cols);
    for (Index j = 0; j < lp.num_cols; ++j) {
        if (isIntegral(lp.col_type[j])) {
            values[j] = integerValues(columnDomain(lp, j));
            if (values[j].empty()) {
                return;
            }
        } else {
            REQUIRE(cont < 0);  // The enumeration handles one continuous column.
            cont = j;
            values[j] = {0.0};
        }
    }

    Slice slice;
    slice.cont = cont;
    slice.x.assign(lp.num_cols, 0.0);
    std::vector<std::size_t> pos(lp.num_cols, 0);
    while (true) {
        for (Index j = 0; j < lp.num_cols; ++j) {
            slice.x[j] = values[j][pos[j]];
        }
        bool feasible = true;
        slice.cont_set = cont >= 0 ? columnDomain(lp, cont) : std::vector<Interval>{};
        for (Index i = 0; i < lp.num_rows && feasible; ++i) {
            auto rv = lp.matrix.row(i);
            Real act = 0.0;
            Real c = 0.0;
            for (Index k = 0; k < rv.size(); ++k) {
                if (rv.indices[k] == cont) {
                    c += rv.values[k];
                } else {
                    act += rv.values[k] * slice.x[rv.indices[k]];
                }
            }
            if (c == 0.0) {
                feasible = act >= lp.row_lower[i] - 1e-9 && act <= lp.row_upper[i] + 1e-9;
                continue;
            }
            Real lo = (lp.row_lower[i] - act) / c;
            Real hi = (lp.row_upper[i] - act) / c;
            if (c < 0.0) {
                std::swap(lo, hi);
            }
            std::vector<Interval> kept;
            for (const auto& iv : slice.cont_set) {
                const Interval cut{std::max(iv.lo, lo), std::min(iv.hi, hi)};
                if (cut.lo <= cut.hi + 1e-9) {
                    kept.push_back({cut.lo, std::max(cut.lo, cut.hi)});
                }
            }
            slice.cont_set = std::move(kept);
            feasible = !slice.cont_set.empty();
        }
        if (feasible) {
            fn(slice);
        }

        Index j = 0;
        while (j < lp.num_cols && pos[j] + 1 == values[j].size()) {
            pos[j] = 0;
            ++j;
        }
        if (j == lp.num_cols) {
            break;
        }
        ++pos[j];
    }
}

// Concrete points of a slice: its endpoints, midpoints and a value near zero
// in each interval, so that checks see interior and boundary values alike.
std::vector<std::vector<Real>> slicePoints(const Slice& s) {
    if (s.cont < 0) {
        return {s.x};
    }
    std::vector<std::vector<Real>> points;
    auto add = [&](Real v) {
        auto x = s.x;
        x[s.cont] = v;
        points.push_back(std::move(x));
    };
    for (const auto& iv : s.cont_set) {
        if (std::isfinite(iv.lo)) {
            add(iv.lo);
        }
        if (std::isfinite(iv.hi)) {
            add(iv.hi);
        }
        if (std::isfinite(iv.lo) && std::isfinite(iv.hi)) {
            add(0.5 * (iv.lo + iv.hi));
        }
        add(std::clamp<Real>(0.0, iv.lo, iv.hi));
    }
    return points;
}

// The best point of a slice; false when the continuous column is unbounded
// in the improving direction.
bool bestOfSlice(const LpProblem& lp, const Slice& s, std::vector<Real>& best) {
    best = s.x;
    if (s.cont < 0) {
        return true;
    }
    const Real c = minTerms(lp, lp.obj[s.cont]);
    Real value = c > 0.0 ? kInf : -kInf;
    for (const auto& iv : s.cont_set) {
        if (c > 0.0) {
            value = std::min(value, iv.lo);
        } else if (c < 0.0) {
            value = std::max(value, iv.hi);
        } else {
            value = std::clamp<Real>(0.0, iv.lo, iv.hi);
        }
    }
    if (!std::isfinite(value)) {
        return false;
    }
    best[s.cont] = value;
    return true;
}

struct Enumeration {
    bool feasible = false;
    bool unbounded = false;
    Real best = kInf;  // Objective, in the model's own sense.
    std::vector<Real> argbest;
};

Enumeration enumerate(const LpProblem& lp) {
    Enumeration e;
    forEachFeasible(lp, [&](const Slice& s) {
        std::vector<Real> x;
        if (!bestOfSlice(lp, s, x)) {
            e.unbounded = true;
            return;
        }
        const Real value = objective(lp, x);
        if (!e.feasible || minTerms(lp, value) < minTerms(lp, e.best)) {
            e.best = value;
            e.argbest = x;
        }
        e.feasible = true;
    });
    return e;
}

// ---- Models -----------------------------------------------------------------

struct ModelSpec {
    int min_int = 2;
    int max_int = 4;
    bool continuous = false;  // One continuous column: bounded, free, half-free or semi.
    bool semi_integer = false;
    bool objective = true;
    bool symmetric = false;   // Duplicate a column, so the model has a symmetry.
    int int_width = 3;        // General integer columns span up to this many units.
    bool equalities = false;  // Half the rows are equalities.
    bool tiny_coefs = false;  // Rows may carry one coefficient of 1e-9 to 1e-6.
};

/// A random model; `anchor_out`, when given, receives the point its rows
/// were drawn around (feasible unless a row's offset excludes it).
LpProblem randomModel(std::mt19937& rng, const ModelSpec& spec,
                      std::vector<Real>* anchor_out = nullptr) {
    auto uniform = [&](int lo, int hi) { return std::uniform_int_distribution<int>(lo, hi)(rng); };
    auto chance = [&](double p) { return std::uniform_real_distribution<double>(0, 1)(rng) < p; };
    const std::vector<Real> int_coefs = {1, 2, 3, 5, 7, -1, -2, -3, -4, -7};
    const std::vector<Real> cont_coefs = {1, 2, 0.5, 1.5, -1, -2, -0.5, -3};

    LpProblem lp;
    lp.name = "validity";
    lp.sense = chance(0.5) ? Sense::Minimize : Sense::Maximize;
    const int n_int = uniform(spec.min_int, spec.max_int);
    auto addCol = [&](Real lb, Real ub, VarType type) {
        lp.col_lower.push_back(lb);
        lp.col_upper.push_back(ub);
        lp.col_type.push_back(type);
        lp.obj.push_back(spec.objective ? uniform(-4, 4) : 0.0);
        lp.col_names.push_back("x" + std::to_string(lp.num_cols));
        ++lp.num_cols;
    };
    for (int j = 0; j < n_int; ++j) {
        const int kind = uniform(0, 9);
        if (kind <= 2) {
            addCol(0.0, 1.0, VarType::Binary);
        } else if (kind == 3) {
            addCol(0.0, 1.0, VarType::Integer);
        } else if (kind == 4 && spec.semi_integer) {
            const Real semi_lower = uniform(1, 2);
            addCol(semi_lower, semi_lower + uniform(1, 2), VarType::SemiInteger);
        } else {
            Real lb = uniform(-2, 1);
            Real ub = lb + uniform(1, spec.int_width);
            // Fractional bounds on integer columns, which every reduction
            // must round in before it reasons from them.
            if (chance(0.2)) {
                lb -= 0.5;
            }
            if (chance(0.2)) {
                ub += 0.3;
            }
            addCol(lb, ub, VarType::Integer);
        }
    }
    if (spec.continuous) {
        switch (uniform(0, 3)) {
            case 0:
                addCol(-2.5, 3.5, VarType::Continuous);
                break;
            case 1:
                addCol(-kInf, kInf, VarType::Continuous);
                break;
            case 2:
                addCol(0.0, kInf, VarType::Continuous);
                break;
            default:
                addCol(1.5, 4.0, VarType::SemiContinuous);
                break;
        }
    }
    Index sym_src = -1;
    if (spec.symmetric) {
        sym_src = uniform(0, n_int - 1);
        addCol(lp.col_lower[sym_src], lp.col_upper[sym_src], lp.col_type[sym_src]);
        lp.obj.back() = lp.obj[sym_src];
    }

    // A random point of the box: rows are drawn around its activity, so most
    // models are feasible but not trivially so.
    std::vector<Real> anchor(lp.num_cols);
    for (Index j = 0; j < lp.num_cols; ++j) {
        const auto domain = columnDomain(lp, j);
        const auto& iv = domain[uniform(0, static_cast<int>(domain.size()) - 1)];
        if (isIntegral(lp.col_type[j])) {
            const auto values = integerValues({iv});
            anchor[j] = values.empty() ? integerValues(domain)[0]
                                       : values[uniform(0, static_cast<int>(values.size()) - 1)];
        } else {
            const Real lo = std::isfinite(iv.lo) ? iv.lo : -3.0;
            const Real hi = std::isfinite(iv.hi) ? iv.hi : 3.0;
            anchor[j] = lo + (hi - lo) * uniform(0, 4) / 4.0;
        }
    }

    const int m = uniform(1, 3);
    std::vector<Triplet> trips;
    for (int i = 0; i < m; ++i) {
        std::vector<std::pair<Index, Real>> row;
        for (Index j = 0; j < lp.num_cols; ++j) {
            if (chance(0.75)) {
                const auto& pool = isIntegral(lp.col_type[j]) ? int_coefs : cont_coefs;
                row.push_back({j, pool[uniform(0, static_cast<int>(pool.size()) - 1)]});
            }
        }
        if (row.empty()) {
            row.push_back({uniform(0, lp.num_cols - 1), 1.0});
        }
        if (sym_src >= 0) {
            // The copy gets the source's coefficient in every row.
            const Index copy = lp.num_cols - 1;
            std::erase_if(row, [&](const auto& t) { return t.first == copy; });
            for (const auto& [j, a] : std::vector(row)) {
                if (j == sym_src) {
                    row.push_back({copy, a});
                }
            }
        }
        if (spec.tiny_coefs && sym_src < 0 && chance(0.5)) {
            // A near-tolerance coefficient, on a column of the row or a new one.
            const Index j = uniform(0, lp.num_cols - 1);
            const Real a = std::pow(10.0, -uniform(6, 9)) * (chance(0.5) ? 1.0 : -1.0);
            std::erase_if(row, [&](const auto& t) { return t.first == j; });
            row.push_back({j, a});
        }
        Real act = 0.0;
        for (const auto& [j, a] : row) {
            act += a * anchor[j];
            trips.push_back({static_cast<Index>(i), j, a});
        }
        switch (spec.equalities && chance(0.5) ? 0 : uniform(0, 4)) {
            case 0:
                lp.row_lower.push_back(act);
                lp.row_upper.push_back(act);
                break;
            case 1:
                lp.row_lower.push_back(act - uniform(0, 2));
                lp.row_upper.push_back(lp.row_lower.back() + uniform(1, 3));
                break;
            case 2:
                lp.row_lower.push_back(act - uniform(-1, 2));
                lp.row_upper.push_back(kInf);
                break;
            default:
                lp.row_lower.push_back(-kInf);
                lp.row_upper.push_back(act + uniform(-1, 2));
                break;
        }
        lp.row_names.push_back("r" + std::to_string(i));
    }
    lp.num_rows = m;
    lp.matrix = SparseMatrix(lp.num_rows, lp.num_cols, std::move(trips));
    if (chance(0.3)) {
        lp.obj_offset = 1.5;
    }
    if (anchor_out != nullptr) {
        *anchor_out = std::move(anchor);
    }
    return lp;
}

LpProblem makeModel(Sense sense, std::vector<Real> obj, std::vector<Real> lb, std::vector<Real> ub,
                    std::vector<VarType> type, std::vector<std::vector<Real>> rows,
                    std::vector<Real> row_lb, std::vector<Real> row_ub) {
    LpProblem lp;
    lp.name = "validity_fixture";
    lp.sense = sense;
    lp.num_cols = static_cast<Index>(obj.size());
    lp.obj = std::move(obj);
    lp.col_lower = std::move(lb);
    lp.col_upper = std::move(ub);
    lp.col_type = std::move(type);
    for (Index j = 0; j < lp.num_cols; ++j) {
        lp.col_names.push_back("x" + std::to_string(j));
    }
    lp.num_rows = static_cast<Index>(rows.size());
    std::vector<Triplet> trips;
    for (Index i = 0; i < lp.num_rows; ++i) {
        for (Index j = 0; j < lp.num_cols; ++j) {
            if (rows[i][j] != 0.0) {
                trips.push_back({i, j, rows[i][j]});
            }
        }
        lp.row_names.push_back("r" + std::to_string(i));
    }
    lp.row_lower = std::move(row_lb);
    lp.row_upper = std::move(row_ub);
    lp.matrix = SparseMatrix(lp.num_rows, lp.num_cols, std::move(trips));
    return lp;
}

// Hand-built models, each aimed at a reduction's edge case.
std::vector<LpProblem> fixtures() {
    constexpr auto I = VarType::Integer;
    constexpr auto B = VarType::Binary;
    constexpr auto C = VarType::Continuous;
    std::vector<LpProblem> models;
    // #226: coefficient tightening against a continuous column.
    models.push_back(
        makeModel(Sense::Maximize, {100, 1}, {0, -100}, {1, 2}, {I, C}, {{7, 1}}, {-kInf}, {8}));
    // Doubleton equality between integers whose ratio is not integral:
    // 2x - 3y = 1 admits only x = 2, 5 over the box.
    models.push_back(
        makeModel(Sense::Minimize, {1, 1}, {0, 0}, {6, 4}, {I, I}, {{2, -3}}, {1}, {1}));
    // Doubleton equality between an integer and a continuous column.
    models.push_back(makeModel(Sense::Maximize, {1, 2}, {0, 0}, {3, 10}, {I, C}, {{3, -2}, {1, 1}},
                               {1, -kInf}, {1, 4.5}));
    // Free continuous column in a single row (implied free singleton column).
    models.push_back(makeModel(Sense::Minimize, {2, -1, 1}, {0, 0, -kInf}, {3, 2, kInf}, {I, I, C},
                               {{1, 1, -1}, {1, -1, 0}}, {0, -kInf}, {kInf, 1}));
    // Set packing with a knapsack: cliques, implications and forcing rows.
    models.push_back(makeModel(Sense::Maximize, {3, 2, 2, 1}, {0, 0, 0, 0}, {1, 1, 1, 1},
                               {B, B, B, B}, {{1, 1, 1, 0}, {3, 0, 2, 4}, {0, 1, 0, -1}},
                               {-kInf, -kInf, 0}, {1, 5, kInf}));
    // Fractional bounds on integers with negative coefficients and a ranged row.
    models.push_back(makeModel(Sense::Minimize, {-1, -2, 1}, {-1.5, -0.7, 0}, {2.5, 1.2, 3},
                               {I, I, I}, {{-2, 3, 1}, {1, 1, -1}}, {-4, -1}, {3, 1}));
    // Parallel rows that differ by scale and range.
    models.push_back(makeModel(Sense::Maximize, {1, 1, 1}, {0, 0, 0}, {3, 3, 3}, {I, I, I},
                               {{1, 2, 0}, {2, 4, 0}, {0, 1, 1}}, {-kInf, 1, 1}, {4, 7, 1}));
    // A forcing row: max activity equals the rhs lower bound.
    models.push_back(makeModel(Sense::Minimize, {1, 1, 1}, {0, 0, -1}, {2, 1, 1}, {I, I, I},
                               {{1, 2, -1}, {1, 1, 1}}, {5, -kInf}, {kInf, 3}));
    // Dual fixing: x2 appears only in >= rows with positive coefficients and
    // has a positive cost; x0 has zero cost.
    models.push_back(makeModel(Sense::Minimize, {0, 1, 2}, {0, 0, 0}, {2, 3, 2}, {I, I, I},
                               {{1, 1, 1}, {-1, 2, 1}}, {2, 1}, {kInf, kInf}));
    // Semi-continuous column bounded by an integer.
    LpProblem semi = makeModel(Sense::Maximize, {1, 1}, {0, 2}, {3, 5},
                               {I, VarType::SemiContinuous}, {{-2, 1}}, {-kInf}, {0.5});
    models.push_back(semi);
    // Semi-integer column next to a binary.
    models.push_back(makeModel(Sense::Minimize, {-1, 3}, {2, 0}, {4, 1}, {VarType::SemiInteger, B},
                               {{1, -4}}, {-kInf}, {0}));
    return models;
}

// MipSolver solves the linearized model, so the components behind it are
// audited on that form: semi columns become a column and a binary.
LpProblem linearized(const LpProblem& lp) {
    return hasAdvancedModelFeatures(lp) ? linearizeModelFeatures(lp) : lp;
}

std::vector<LpProblem> auditModels(int seed, int count, bool allow_continuous) {
    std::mt19937 rng(seed);
    std::vector<LpProblem> models = fixtures();
    for (int t = 0; t < count; ++t) {
        ModelSpec spec;
        spec.continuous = allow_continuous && t % 2 == 1;
        spec.semi_integer = t % 5 == 2;
        spec.symmetric = t % 7 == 3;
        models.push_back(randomModel(rng, spec));
    }
    return models;
}

// ---- Presolve ---------------------------------------------------------------

struct PresolveCase {
    const char* name;
    PresolveOptions options;
};

std::vector<PresolveCase> presolveCases() {
    std::vector<PresolveCase> cases;
    PresolveOptions o;
    cases.push_back({"default", o});
    PresolveOptions core = o;
    core.enable_forcing_rows = false;
    core.enable_dual_fixing = false;
    core.enable_doubleton_aggregation = false;
    core.enable_parallel_rows = false;
    cases.push_back({"always-on reductions only", core});
    PresolveOptions coeff = o;
    coeff.enable_coefficient_tightening = true;
    cases.push_back({"coefficient tightening (opt-in)", coeff});
    PresolveOptions probing = o;
    probing.enable_probing = true;
    cases.push_back({"probing (opt-in)", probing});
    PresolveOptions all = o;
    all.enable_coefficient_tightening = true;
    all.enable_probing = true;
    cases.push_back({"all opt-in", all});
    return cases;
}

// Checks one model under one option set. With a zero objective every
// reduction must keep the projection of each feasible point; with an
// objective, dual reductions may drop points but never the optimum.
void auditPresolve(const LpProblem& lp, const PresolveCase& pc) {
    const Enumeration orig = enumerate(lp);
    if (orig.unbounded) {
        return;
    }
    Presolver presolver;
    presolver.setOptions(pc.options);
    const LpProblem reduced = presolver.presolve(lp);
    INFO("presolve case: " << pc.name << "\nmodel:\n" << describe(lp));
    if (presolver.isInfeasible()) {
        CHECK(!orig.feasible);
        return;
    }
    const auto& mapping = presolver.colMapping();
    INFO("reduced:\n" << describe(reduced));

    const Enumeration red = enumerate(reduced);
    INFO("optimum " << orig.best << ", reduced optimum " << red.best);
    CHECK(red.feasible == orig.feasible);
    CHECK(!red.unbounded);
    if (red.feasible && orig.feasible) {
        CHECK(std::abs(red.best - orig.best) <= kObjTol);
    }

    // Every point the reduced model admits postsolves to a feasible point of
    // the original with the same objective.
    forEachFeasible(reduced, [&](const Slice& s) {
        for (const auto& xr : slicePoints(s)) {
            const auto full = presolver.postsolve(xr);
            if (full.size() != static_cast<std::size_t>(lp.num_cols)) {
                CHECK(false);
                continue;
            }
            INFO("reduced point" << describe(xr) << " -> postsolved" << describe(full)
                                 << ", objective " << objective(reduced, xr) << " -> "
                                 << objective(lp, full));
            CHECK(pointFeasible(lp, full, kFeasTol));
            CHECK(std::abs(objective(lp, full) - objective(reduced, xr)) <= kObjTol);
        }
    });

    const bool primal = std::all_of(lp.obj.begin(), lp.obj.end(), [](Real c) { return c == 0.0; });
    if (!primal) {
        return;
    }
    forEachFeasible(lp, [&](const Slice& s) {
        for (const auto& x : slicePoints(s)) {
            std::vector<Real> xr(reduced.num_cols);
            for (Index jj = 0; jj < reduced.num_cols; ++jj) {
                xr[jj] = x[mapping[jj]];
            }
            INFO("original point" << describe(x));
            CHECK(pointFeasible(reduced, xr, kFeasTol));
        }
    });
}

LpProblem withoutObjective(LpProblem lp) {
    std::fill(lp.obj.begin(), lp.obj.end(), 0.0);
    return lp;
}

// ---- Node box ---------------------------------------------------------------

// A random node: each integral column narrowed to a random sub-range of its
// values, each bounded continuous column to a random sub-interval.
void randomBox(const LpProblem& lp, std::mt19937& rng, std::vector<Real>& lower,
               std::vector<Real>& upper) {
    lower = lp.col_lower;
    upper = lp.col_upper;
    for (Index j = 0; j < lp.num_cols; ++j) {
        if (std::uniform_int_distribution<int>(0, 2)(rng) == 0) {
            continue;  // Some columns stay at their global bounds.
        }
        if (isIntegral(lp.col_type[j])) {
            const auto values = integerValues({{lp.col_lower[j], lp.col_upper[j]}});
            if (values.empty()) {
                continue;
            }
            std::uniform_int_distribution<int> pick(0, static_cast<int>(values.size()) - 1);
            int a = pick(rng);
            int b = pick(rng);
            lower[j] = values[std::min(a, b)];
            upper[j] = values[std::max(a, b)];
        } else if (std::isfinite(lp.col_lower[j]) && std::isfinite(lp.col_upper[j])) {
            std::uniform_real_distribution<Real> frac(0.0, 0.5);
            const Real width = lp.col_upper[j] - lp.col_lower[j];
            lower[j] = lp.col_lower[j] + frac(rng) * width;
            upper[j] = lp.col_upper[j] - frac(rng) * width;
        }
    }
}

LpProblem withBox(LpProblem lp, const std::vector<Real>& lower, const std::vector<Real>& upper) {
    lp.col_lower = lower;
    lp.col_upper = upper;
    return lp;
}

// Every feasible point of `lp` (the node) lies inside [lower, upper].
void checkInside(const LpProblem& lp, const std::vector<Real>& lower,
                 const std::vector<Real>& upper) {
    forEachFeasible(lp, [&](const Slice& s) {
        INFO("feasible point" << describe(s.x));
        for (Index j = 0; j < lp.num_cols; ++j) {
            INFO("column " << j << " bounds [" << lower[j] << ", " << upper[j] << "]");
            if (j == s.cont) {
                for (const auto& iv : s.cont_set) {
                    CHECK(iv.lo >= lower[j] - kFeasTol);
                    CHECK(iv.hi <= upper[j] + kFeasTol);
                }
            } else {
                CHECK(s.x[j] >= lower[j] - kFeasTol);
                CHECK(s.x[j] <= upper[j] + kFeasTol);
            }
        }
    });
}

bool anyFeasible(const LpProblem& lp) {
    bool found = false;
    forEachFeasible(lp, [&](const Slice&) { found = true; });
    return found;
}

// Value of literal `lit` at the integral assignment x.
bool literalTrue(const std::vector<Real>& x, Literal lit) {
    const bool one = x[lit.var] > 0.5;
    return lit.complemented ? !one : one;
}

}  // namespace

// =============================================================================
// Presolve
// =============================================================================

TEST_CASE("Validity: every presolve reduction keeps the feasible set and optimum", "[validity]") {
    const auto models = auditModels(227, 300, true);
    for (const auto& pc : presolveCases()) {
        for (std::size_t t = 0; t < models.size(); ++t) {
            INFO("model " << t);
            const LpProblem lp = linearized(models[t]);
            auditPresolve(lp, pc);
            auditPresolve(withoutObjective(lp), pc);
        }
    }
}

// =============================================================================
// Node propagators
// =============================================================================

TEST_CASE("Validity: bound propagation keeps every feasible point of the node", "[validity]") {
    const auto models = auditModels(1227, 300, true);
    std::mt19937 rng(7);
    Int tightened = 0;
    Int pruned = 0;
    for (std::size_t t = 0; t < models.size(); ++t) {
        const LpProblem lp = linearized(models[t]);
        INFO("model " << t << ":\n" << describe(lp));

        // The global structures the tree attaches, built on the root model.
        ConflictGraph graph;
        graph.build(lp, BinaryColumns::ZeroOne);
        CliqueTable cliques;
        cliques.build(lp, graph);
        ImplicationGraph implications;
        VariableBoundStore vbs;
        ProbingEngine probing;
        const auto probe_stats = probing.probe(lp, implications, vbs);
        if (probe_stats.infeasible) {
            CHECK_FALSE(anyFeasible(lp));
            continue;
        }

        for (int node = 0; node < 6; ++node) {
            std::vector<Real> lower;
            std::vector<Real> upper;
            if (node == 0) {
                lower = lp.col_lower;
                upper = lp.col_upper;
            } else {
                randomBox(lp, rng, lower, upper);
            }
            const LpProblem box = withBox(lp, lower, upper);
            for (int attach = 0; attach < 3; ++attach) {
                INFO("node " << node << " attach " << attach << "\nbox:\n" << describe(box));
                DomainPropagator dp;
                dp.load(lp);
                if (attach == 1) {
                    dp.setCliqueTable(&cliques);
                } else if (attach == 2 && implications.numImplications() > 0) {
                    dp.setImplicationGraph(&implications);
                }
                for (Index j = 0; j < lp.num_cols; ++j) {
                    dp.setBound(j, lower[j], upper[j]);
                }
                if (!dp.propagate()) {
                    ++pruned;
                    CHECK_FALSE(anyFeasible(box));
                    continue;
                }
                std::vector<Real> new_lower(lp.num_cols);
                std::vector<Real> new_upper(lp.num_cols);
                for (Index j = 0; j < lp.num_cols; ++j) {
                    new_lower[j] = dp.getLower(j);
                    new_upper[j] = dp.getUpper(j);
                    tightened += (new_lower[j] > lower[j] || new_upper[j] < upper[j]) ? 1 : 0;
                }
                checkInside(box, new_lower, new_upper);
            }
        }
    }
    // The audit is not vacuous: propagation both tightened and pruned.
    CHECK(tightened > 0);
    CHECK(pruned > 0);
}

TEST_CASE("Validity: probing learns only what every feasible point satisfies", "[validity]") {
    const auto models = auditModels(2227, 300, true);
    Int learned_fixings = 0;
    Int learned_implications = 0;
    Int learned_bounds = 0;
    for (std::size_t t = 0; t < models.size(); ++t) {
        const LpProblem lp = linearized(models[t]);
        INFO("model " << t << ":\n" << describe(lp));
        ImplicationGraph implications;
        VariableBoundStore vbs;
        ProbingEngine probing;
        const auto stats = probing.probe(lp, implications, vbs);
        if (stats.infeasible) {
            CHECK_FALSE(anyFeasible(lp));
            continue;
        }
        learned_fixings += static_cast<Int>(probing.fixings().size());
        learned_implications += implications.numImplications();
        learned_bounds += vbs.numVUBs() + vbs.numVLBs();
        forEachFeasible(lp, [&](const Slice& s) {
            INFO("feasible point" << describe(s.x));
            for (const auto& [var, value] : probing.fixings()) {
                INFO("fixing x" << var << " = " << value);
                CHECK_THAT(s.x[var], WithinAbs(value, kFeasTol));
            }
            for (const auto& eq : probing.equivalences()) {
                INFO("equivalence x" << eq.var_a << (eq.same_sense ? " = x" : " = 1 - x")
                                     << eq.var_b);
                const Real b = eq.same_sense ? s.x[eq.var_b] : 1.0 - s.x[eq.var_b];
                CHECK_THAT(s.x[eq.var_a], WithinAbs(b, kFeasTol));
            }
            for (Index from : implications.binaryVars()) {
                for (bool val : {false, true}) {
                    if ((s.x[from] > 0.5) != val) {
                        continue;
                    }
                    for (const auto& imp : implications.implications(from, val)) {
                        INFO("implication x" << from << "=" << val << " => x" << imp.to_var << "="
                                             << imp.to_val);
                        CHECK((s.x[imp.to_var] > 0.5) == imp.to_val);
                    }
                }
            }
            // Variable bounds, over the whole feasible range of the
            // continuous column: its intervals' ends, infinite ones included.
            auto valueRange = [&](Index j) -> Interval {
                if (j != s.cont) {
                    return {s.x[j], s.x[j]};
                }
                Interval r{kInf, -kInf};
                for (const auto& iv : s.cont_set) {
                    r.lo = std::min(r.lo, iv.lo);
                    r.hi = std::max(r.hi, iv.hi);
                }
                return r;
            };
            for (Index j = 0; j < lp.num_cols; ++j) {
                const Interval range = valueRange(j);
                for (const auto& vub : vbs.vubs(j)) {
                    INFO("vub x" << j << " <= " << vub.coeff << "*x" << vub.binary_var << " + "
                                 << vub.constant << ", x" << j << " up to " << range.hi);
                    CHECK(range.hi <= vub.coeff * s.x[vub.binary_var] + vub.constant + kFeasTol);
                }
                for (const auto& vlb : vbs.vlbs(j)) {
                    INFO("vlb x" << j << " >= " << vlb.coeff << "*x" << vlb.binary_var << " + "
                                 << vlb.constant << ", x" << j << " down to " << range.lo);
                    CHECK(range.lo >= vlb.coeff * s.x[vlb.binary_var] + vlb.constant - kFeasTol);
                }
            }
        });
    }
    CHECK(learned_fixings > 0);
    CHECK(learned_implications > 0);
    CHECK(learned_bounds > 0);
}

TEST_CASE("Validity: conflicts and cliques hold at every feasible point", "[validity]") {
    const auto models = auditModels(3227, 300, true);
    Index num_cliques = 0;
    for (std::size_t t = 0; t < models.size(); ++t) {
        const LpProblem lp = linearized(models[t]);
        INFO("model " << t << ":\n" << describe(lp));
        ConflictGraph graph;
        graph.build(lp, BinaryColumns::ZeroOne);
        CliqueTable cliques;
        cliques.build(lp, graph);
        cliques.mergeAndSubsume();
        num_cliques += cliques.numCliques();
        forEachFeasible(lp, [&](const Slice& s) {
            INFO("feasible point" << describe(s.x));
            for (Index j = 0; j < lp.num_cols; ++j) {
                if (!graph.isBinary(j)) {
                    continue;
                }
                for (bool comp : {false, true}) {
                    const Literal a{j, comp};
                    if (!literalTrue(s.x, a)) {
                        continue;
                    }
                    for (const Literal b : graph.neighbors(a)) {
                        INFO("conflict x" << a.var << (a.complemented ? "'" : "") << " x" << b.var
                                          << (b.complemented ? "'" : ""));
                        CHECK_FALSE(literalTrue(s.x, b));
                    }
                }
            }
            for (Index c = 0; c < cliques.numCliques(); ++c) {
                const auto& clique = cliques.clique(c);
                Index count = 0;
                for (const auto& lit : clique.literals) {
                    count += literalTrue(s.x, lit) ? 1 : 0;
                }
                INFO("clique " << c);
                CHECK(count <= 1);
                if (clique.is_equality) {
                    CHECK(count == 1);
                }
            }
        });
    }
    CHECK(num_cliques > 0);
}

TEST_CASE("Validity: symmetry handling keeps an image of every feasible point", "[validity]") {
    // The solver detects orbits of interchangeable columns, adds the cut
    // x_variable <= x_canonical for each orbital fix, and at every node raises
    // lb(canonical) to lb(variable), which those cuts imply. That is valid
    // when every feasible point has an image under the orbits' permutations
    // that satisfies the cuts (and, since the image must satisfy the rows
    // too, when the detected orbits are real symmetries).
    const auto models = auditModels(6227, 300, true);
    Int symmetric = 0;
    for (std::size_t t = 0; t < models.size(); ++t) {
        const LpProblem lp = linearized(models[t]);
        SymmetryManager symmetry;
        symmetry.detect(lp);
        if (!symmetry.hasSymmetry()) {
            continue;
        }
        ++symmetric;
        LpProblem with_cuts = lp;
        const Index cuts = symmetry.addSymmetryCuts(with_cuts);
        INFO("model " << t << " (" << cuts << " symmetry cuts):\n" << describe(with_cuts));
        std::vector<std::vector<Index>> orbits;
        bool searchable = true;
        for (const auto& orbit : symmetry.orbits()) {
            searchable = searchable && orbit.size() <= 6;
            if (orbit.size() > 1) {
                orbits.push_back(orbit);
            }
        }
        if (!searchable) {
            continue;  // Too many permutations to try; the cuts stay unjudged.
        }
        forEachFeasible(lp, [&](const Slice& s) {
            for (const auto& x : slicePoints(s)) {
                // Try every combination of one permutation per orbit.
                std::vector<std::vector<Index>> perm = orbits;
                for (auto& p : perm) {
                    std::sort(p.begin(), p.end());
                }
                bool found = false;
                while (!found) {
                    auto y = x;
                    for (std::size_t o = 0; o < orbits.size(); ++o) {
                        for (std::size_t k = 0; k < orbits[o].size(); ++k) {
                            y[orbits[o][k]] = x[perm[o][k]];
                        }
                    }
                    // The image must be as good, or pruning x's twin loses it.
                    found = pointFeasible(with_cuts, y, kFeasTol) &&
                            std::abs(objective(lp, y) - objective(lp, x)) <= kObjTol;
                    std::size_t o = 0;
                    while (o < perm.size() &&
                           !std::next_permutation(perm[o].begin(), perm[o].end())) {
                        ++o;
                    }
                    if (o == perm.size()) {
                        break;
                    }
                }
                INFO("feasible point" << describe(x));
                CHECK(found);
            }
        });
    }
    CHECK(symmetric > 0);
}

TEST_CASE("Validity: reduced-cost fixing keeps every point better than the incumbent",
          "[validity]") {
    const auto models = auditModels(4227, 300, true);
    std::mt19937 rng(11);
    Int tightened_total = 0;
    for (std::size_t t = 0; t < models.size(); ++t) {
        LpProblem lp = linearized(models[t]);
        // The fixer works on minimization models with no offset, as the
        // solver hands it its internal form.
        if (lp.sense == Sense::Maximize) {
            for (auto& c : lp.obj) {
                c = -c;
            }
            lp.sense = Sense::Minimize;
        }
        lp.obj_offset = 0.0;
        INFO("model " << t << ":\n" << describe(lp));

        for (int node = 0; node < 4; ++node) {
            std::vector<Real> lower;
            std::vector<Real> upper;
            if (node == 0) {
                lower = lp.col_lower;
                upper = lp.col_upper;
            } else {
                randomBox(lp, rng, lower, upper);
            }
            const LpProblem box = withBox(lp, lower, upper);
            const Enumeration e = enumerate(box);
            if (!e.feasible || e.unbounded) {
                continue;
            }
            DualSimplexSolver lp_solver;
            lp_solver.setVerbose(false);
            lp_solver.load(box);
            if (lp_solver.solve().status != Status::Optimal) {
                continue;
            }
            const auto rc = lp_solver.getReducedCosts();
            const auto primals = lp_solver.getPrimalValues();
            const Real lp_obj = lp_solver.getObjective();

            // Incumbents at the optimum and above it.
            std::vector<Real> objectives;
            forEachFeasible(box, [&](const Slice& s) {
                std::vector<Real> x;
                if (bestOfSlice(box, s, x)) {
                    objectives.push_back(objective(box, x));
                }
            });
            std::sort(objectives.begin(), objectives.end());
            const std::vector<Real> incumbents = {
                objectives.front(), objectives[objectives.size() / 2], objectives.back()};
            for (Real incumbent : incumbents) {
                for (bool global : {true, false}) {
                    INFO("node " << node << " incumbent " << incumbent << " lp " << lp_obj
                                 << (global ? " global" : " local") << "\nbox:\n"
                                 << describe(box));
                    ReducedCostFixer fixer;
                    fixer.load(lp);
                    std::vector<Real> new_lower = lower;
                    std::vector<Real> new_upper = upper;
                    std::vector<Index> tightened;
                    const bool ok = global
                                        ? fixer.applyGlobalFixing(rc, primals, lp_obj, incumbent,
                                                                  new_lower, new_upper, tightened)
                                        : fixer.applyLocalFixing(rc, primals, lp_obj, incumbent,
                                                                 new_lower, new_upper, tightened);
                    tightened_total += static_cast<Int>(tightened.size());
                    // Every point strictly better than the incumbent survives.
                    forEachFeasible(box, [&](const Slice& s) {
                        std::vector<Real> x;
                        REQUIRE(bestOfSlice(box, s, x));
                        if (objective(box, x) >= incumbent - kObjTol) {
                            return;
                        }
                        INFO("improving point" << describe(x) << " obj " << objective(box, x));
                        REQUIRE(ok);
                        for (Index j = 0; j < lp.num_cols; ++j) {
                            CHECK(x[j] >= new_lower[j] - kFeasTol);
                            CHECK(x[j] <= new_upper[j] + kFeasTol);
                        }
                    });
                }
            }
        }
    }
    CHECK(tightened_total > 0);
}

// Objective propagation (#117) may drop every point whose objective is not
// strictly better than the incumbent, but nothing better. For the continuous
// column that is a sub-interval of each feasible interval, checked at its ends.
TEST_CASE("Validity: objective propagation keeps every point better than the incumbent",
          "[validity][objprop]") {
    const auto models = auditModels(6227, 300, true);
    std::mt19937 rng(13);
    Int tightened_total = 0;
    Int pruned_total = 0;
    for (std::size_t t = 0; t < models.size(); ++t) {
        LpProblem lp = linearized(models[t]);
        // The solver's internal form: a minimization. The offset is kept.
        if (lp.sense == Sense::Maximize) {
            for (auto& c : lp.obj) {
                c = -c;
            }
            lp.obj_offset = -lp.obj_offset;
            lp.sense = Sense::Minimize;
        }
        INFO("model " << t << ":\n" << describe(lp));
        ObjectivePropagator obj_prop;
        obj_prop.load(lp);

        for (int node = 0; node < 4; ++node) {
            std::vector<Real> lower;
            std::vector<Real> upper;
            if (node == 0) {
                lower = lp.col_lower;
                upper = lp.col_upper;
            } else {
                randomBox(lp, rng, lower, upper);
            }
            const LpProblem box = withBox(lp, lower, upper);
            const Enumeration e = enumerate(box);
            if (!e.feasible || e.unbounded) {
                continue;
            }
            std::vector<Real> objectives;
            forEachFeasible(box, [&](const Slice& s) {
                std::vector<Real> x;
                if (bestOfSlice(box, s, x)) {
                    objectives.push_back(objective(box, x));
                }
            });
            std::sort(objectives.begin(), objectives.end());
            const std::vector<Real> incumbents = {objectives.front(),
                                                  objectives[objectives.size() / 2],
                                                  objectives.back(), objectives.back() + 1.0};
            for (Real incumbent : incumbents) {
                // Node propagation, and at the root the global cutoff bounds.
                for (bool global : {false, true}) {
                    if (global && node != 0) {
                        continue;
                    }
                    INFO("node " << node << " incumbent " << incumbent
                                 << (global ? " global" : " node") << "\nbox:\n"
                                 << describe(box));
                    std::vector<Real> new_lower = lower;
                    std::vector<Real> new_upper = upper;
                    bool ok = true;
                    if (global) {
                        ObjectiveCutoffBounds bounds;
                        bounds.load(lp, &obj_prop, nullptr, nullptr);
                        bounds.refresh(incumbent);
                        std::vector<Index> tightened;
                        ok = bounds.enforce(new_lower, new_upper, tightened);
                        tightened_total += static_cast<Int>(tightened.size());
                    } else {
                        DomainPropagator dp;
                        dp.load(lp);
                        dp.setObjectiveCutoff(&obj_prop, incumbent);
                        for (Index j = 0; j < lp.num_cols; ++j) {
                            dp.setBound(j, lower[j], upper[j]);
                        }
                        ok = dp.propagate();
                        tightened_total += dp.numObjectiveTightenings();
                        if (ok) {
                            for (Index j = 0; j < lp.num_cols; ++j) {
                                new_lower[j] = dp.getLower(j);
                                new_upper[j] = dp.getUpper(j);
                            }
                        }
                    }
                    pruned_total += ok ? 0 : 1;
                    const Real limit = incumbent - kObjTol;
                    forEachFeasible(box, [&](const Slice& s) {
                        std::vector<Real> best;
                        REQUIRE(bestOfSlice(box, s, best));
                        if (objective(box, best) >= limit) {
                            return;
                        }
                        INFO("improving slice" << describe(best) << " obj "
                                               << objective(box, best));
                        REQUIRE(ok);
                        for (Index j = 0; j < lp.num_cols; ++j) {
                            if (j == s.cont) {
                                continue;
                            }
                            CHECK(s.x[j] >= new_lower[j] - kFeasTol);
                            CHECK(s.x[j] <= new_upper[j] + kFeasTol);
                        }
                        if (s.cont < 0) {
                            return;
                        }
                        auto at_zero = s.x;
                        at_zero[s.cont] = 0.0;
                        const Real base = objective(box, at_zero);
                        const Real c = box.obj[s.cont];
                        for (const auto& iv : s.cont_set) {
                            Real lo = iv.lo;
                            Real hi = iv.hi;
                            if (c > 0.0) {
                                hi = std::min(hi, (limit - base) / c);
                            } else if (c < 0.0) {
                                lo = std::max(lo, (limit - base) / c);
                            }
                            if (lo > hi) {
                                continue;
                            }
                            INFO("improving interval [" << lo << ", " << hi << "]");
                            if (std::isfinite(lo)) {
                                CHECK(lo >= new_lower[s.cont] - kFeasTol);
                            } else {
                                CHECK(new_lower[s.cont] == -kInf);
                            }
                            if (std::isfinite(hi)) {
                                CHECK(hi <= new_upper[s.cont] + kFeasTol);
                            } else {
                                CHECK(new_upper[s.cont] == kInf);
                            }
                        }
                    });
                }
            }
        }
    }
    // The audit is not vacuous: the objective row both tightened and pruned.
    CHECK(tightened_total > 0);
    CHECK(pruned_total > 0);
}

// =============================================================================
// The solver end to end
// =============================================================================

// Propagators that only run inside the search (tree presolve, orbital fixing
// and symmetry cuts, conflict analysis, root probing and clique propagation)
// are audited through the solver: whatever it switches on, the optimum it
// reports must be the enumerated one.
TEST_CASE("Validity: the solver finds the enumerated optimum under every propagator",
          "[validity]") {
    struct SolverCase {
        const char* name;
        std::function<void(MipSolver&)> configure;
    };
    const std::vector<SolverCase> cases = {
        {"default", [](MipSolver&) {}},
        {"no root cuts", [](MipSolver& s) { s.setCutsEnabled(false); }},
        {"tree presolve at every node",
         [](MipSolver& s) {
             s.setTreePresolveAutoTuning(false);
             s.setTreePresolveControls(100, 0, 1);
         }},
        {"symmetry off", [](MipSolver& s) { s.setSymmetryEnabled(false); }},
        {"no tree pseudocosts", [](MipSolver& s) { s.setTreePseudocostsEnabled(false); }},
        {"objective propagation (opt-in)",
         [](MipSolver& s) { s.setObjectivePropagationEnabled(true); }},
        {"presolve off", [](MipSolver& s) { s.setPresolve(false); }},
        {"probing and cliques (opt-in)",
         [](MipSolver& s) {
             s.setProbingEnabled(true);
             s.setCliqueTableEnabled(true);
         }},
        {"all opt-in",
         [](MipSolver& s) {
             s.setProbingEnabled(true);
             s.setCliqueTableEnabled(true);
             s.setRestartsEnabled(true);
             s.setCutStrengthening(true);
             s.setTreeCutsEnabled(true);
             s.setObjectivePropagationEnabled(true);
             PresolveOptions o = s.getRootPresolveOptions();
             o.enable_coefficient_tightening = true;
             o.enable_probing = true;
             s.setRootPresolveOptions(o);
         }},
    };
    const auto models = auditModels(5227, 200, true);
    Int tree_presolve_runs = 0;
    Int symmetry_orbits = 0;
    for (const auto& sc : cases) {
        for (std::size_t t = 0; t < models.size(); ++t) {
            const LpProblem& lp = models[t];
            const Enumeration e = enumerate(linearized(lp));
            if (e.unbounded) {
                continue;
            }
            INFO("solver case: " << sc.name << "\nmodel " << t << ":\n" << describe(lp));
            MipSolver solver;
            solver.setVerbose(false);
            solver.setNumThreads(1);
            solver.setGapTolerance(0.0);
            sc.configure(solver);
            solver.load(lp);
            const auto result = solver.solve();
            tree_presolve_runs += solver.getTreePresolveStats().runs;
            symmetry_orbits += solver.getSymmetryStats().orbits;
            if (!e.feasible) {
                CHECK(result.status == Status::Infeasible);
                continue;
            }
            REQUIRE(result.status == Status::Optimal);
            REQUIRE(result.solution.size() >= static_cast<std::size_t>(lp.num_cols));
            std::vector<Real> x(result.solution.begin(), result.solution.begin() + lp.num_cols);
            INFO("solution" << describe(x) << " reported objective " << result.objective);
            CHECK(pointFeasible(lp, x, kFeasTol));
            CHECK_THAT(objective(lp, x), WithinAbs(e.best, kObjTol));
            CHECK_THAT(result.objective, WithinAbs(e.best, kObjTol));
        }
    }
    // The search-time propagators did run.
    CHECK(tree_presolve_runs > 0);
    CHECK(symmetry_orbits > 0);
}

TEST_CASE("Validity: end-to-end solves under the opt-in families, every incumbent feasible",
          "[validity]") {
    // Part 2 of the audit (#235): equality-heavy models over wider general
    // integers, some rows with a coefficient of 1e-9 to 1e-6, solved with
    // each opt-in family and all of them at once. Every incumbent the search
    // accepts, heuristic or not, must be feasible on the model as given.
    //
    // A near-tolerance coefficient lets the solver accept a point that
    // violates a row by less than its feasibility tolerance, which the
    // enumeration (at 1e-9) rejects. So the reported solution must be
    // feasible at kFeasTol and at least as good as the enumerated optimum, and
    // exactly as good when it is feasible at the enumeration's tolerance.
    struct SolverCase {
        const char* name;
        std::function<void(MipSolver&)> configure;
    };
    auto coefficientTightening = [](MipSolver& s) {
        PresolveOptions o = s.getRootPresolveOptions();
        o.enable_coefficient_tightening = true;
        s.setRootPresolveOptions(o);
    };
    auto multiRowCuts = [](MipSolver& s) {
        s.setCutFamilyEnabled(CutFamily::IntersectionCut, true);
        s.setCutFamilyEnabled(CutFamily::MultiRow, true);
    };
    const std::vector<SolverCase> cases = {
        {"default", [](MipSolver&) {}},
        {"no cuts, no presolve",
         [](MipSolver& s) {
             s.setCutsEnabled(false);
             s.setPresolve(false);
         }},
        {"intersection and multi-row cuts (opt-in)", multiRowCuts},
        {"cut strengthening (opt-in)", [](MipSolver& s) { s.setCutStrengthening(true); }},
        {"clique table and probing (opt-in)",
         [](MipSolver& s) {
             s.setCliqueTableEnabled(true);
             s.setProbingEnabled(true);
         }},
        {"coefficient tightening (opt-in)", coefficientTightening},
        {"objective propagation (opt-in)",
         [](MipSolver& s) { s.setObjectivePropagationEnabled(true); }},
        {"tree presolve and tree cuts at every node",
         [](MipSolver& s) {
             s.setTreePresolveAutoTuning(false);
             s.setTreePresolveControls(100, 0, 1);
             s.setTreeCutsEnabled(true);
         }},
        {"all opt-in",
         [&](MipSolver& s) {
             multiRowCuts(s);
             coefficientTightening(s);
             s.setCutStrengthening(true);
             s.setCliqueTableEnabled(true);
             s.setProbingEnabled(true);
             s.setRestartsEnabled(true);
             s.setTreeCutsEnabled(true);
             s.setObjectivePropagationEnabled(true);
         }},
    };

    std::mt19937 rng(235);
    std::vector<LpProblem> models;
    std::vector<std::vector<Real>> anchors;
    for (int t = 0; t < 120; ++t) {
        ModelSpec spec;
        spec.min_int = 3;
        spec.max_int = 5;
        spec.int_width = 5;
        spec.equalities = true;
        spec.continuous = t % 3 == 1;
        spec.tiny_coefs = t % 2 == 0;
        spec.symmetric = t % 7 == 3;
        anchors.emplace_back();
        models.push_back(randomModel(rng, spec, &anchors.back()));
    }

    Int judged = 0;
    Int incumbents = 0;
    for (const auto& sc : cases) {
        for (std::size_t t = 0; t < models.size(); ++t) {
            const LpProblem& lp = models[t];
            const Enumeration e = enumerate(linearized(lp));
            if (e.unbounded) {
                continue;
            }
            INFO("solver case: " << sc.name << "\nmodel " << t << ":\n"
                                 << describe(lp) << "enumerated optimum " << e.best << " at"
                                 << describe(e.argbest));
            MipSolver solver;
            solver.setVerbose(false);
            solver.setNumThreads(1);
            solver.setGapTolerance(0.0);
            sc.configure(solver);
            std::vector<std::pair<std::vector<Real>, std::string>> seen;
            solver.setIncumbentObserver([&](const std::vector<Real>& x, const std::string& source) {
                seen.emplace_back(x, source);
            });
            solver.load(lp);
            const auto result = solver.solve();
            for (const auto& [x, source] : seen) {
                INFO("incumbent from " << source << describe(x));
                REQUIRE(x.size() >= static_cast<std::size_t>(lp.num_cols));
                CHECK(pointFeasible(lp, std::vector<Real>(x.begin(), x.begin() + lp.num_cols),
                                    kFeasTol));
                ++incumbents;
            }
            if (!e.feasible) {
                // Infeasible at 1e-9 may still hold a point within kFeasTol.
                // The search keeps every point within the row slack (kTol
                // times the row's magnitude, at least kTol), so when the
                // anchor is one the solver must find a solution.
                if (pointFeasible(lp, anchors[t], DomainPropagator::kTol)) {
                    INFO("anchor" << describe(anchors[t]) << " is feasible within kTol");
                    CHECK(result.status == Status::Optimal);
                }
                if (result.status == Status::Optimal) {
                    std::vector<Real> x(result.solution.begin(),
                                        result.solution.begin() + lp.num_cols);
                    CHECK(pointFeasible(lp, x, kFeasTol));
                } else {
                    CHECK(result.status == Status::Infeasible);
                }
                continue;
            }
            ++judged;
            CHECK(result.status == Status::Optimal);
            if (result.status != Status::Optimal) {
                continue;
            }
            REQUIRE(result.solution.size() >= static_cast<std::size_t>(lp.num_cols));
            std::vector<Real> x(result.solution.begin(), result.solution.begin() + lp.num_cols);
            INFO("solution" << describe(x) << " reported objective " << result.objective
                            << ", enumerated " << e.best);
            CHECK(pointFeasible(lp, x, kFeasTol));
            const Real sign = lp.sense == Sense::Minimize ? 1.0 : -1.0;
            CHECK_THAT(result.objective, WithinAbs(objective(lp, x), kObjTol));
            if (pointFeasible(lp, x, 1e-9)) {
                CHECK_THAT(objective(lp, x), WithinAbs(e.best, kObjTol));
            } else {
                CHECK(sign * objective(lp, x) <= sign * e.best + kObjTol);
            }
        }
    }
    CHECK(judged > 500);
    CHECK(incumbents > 0);
}

// =============================================================================
// Regressions: the reproducers of bugs the audits above found, each kept
// with its issue number.
// =============================================================================

TEST_CASE("Regression #228: presolve keeps fractional bounds of integer columns", "[validity]") {
    constexpr auto I = VarType::Integer;
    // min x0 over x0 integer in [0.5, 3]: the optimum is x0 = 1.
    const LpProblem lower_side =
        makeModel(Sense::Minimize, {1, 0}, {0.5, 0}, {3, 2}, {I, I}, {{1, 1}}, {-kInf}, {4});
    // max -3 x0 over x0 integer in [-0.5, 2]: the optimum is x0 = 0.
    const LpProblem upper_side =
        makeModel(Sense::Maximize, {-3, 0}, {-0.5, 0}, {2, 2}, {I, I}, {{1, 1}}, {-kInf}, {3});
    for (const auto& lp : {lower_side, upper_side}) {
        auditPresolve(lp, presolveCases()[0]);
        MipSolver solver;
        solver.setVerbose(false);
        solver.load(lp);
        const auto result = solver.solve();
        REQUIRE(result.status == Status::Optimal);
        CHECK(pointFeasible(lp, result.solution, kFeasTol));
        CHECK_THAT(objective(lp, result.solution), WithinAbs(enumerate(lp).best, kObjTol));
    }
    // x0 integer in [0.3, 0.7] has no integer value.
    const LpProblem crossed =
        makeModel(Sense::Minimize, {1, 0}, {0.3, 0}, {0.7, 2}, {I, I}, {{1, 1}}, {-kInf}, {4});
    auditPresolve(crossed, presolveCases()[0]);
    for (bool presolve : {true, false}) {
        INFO("presolve " << presolve);
        MipSolver solver;
        solver.setVerbose(false);
        solver.setPresolve(presolve);
        solver.load(crossed);
        CHECK(solver.solve().status == Status::Infeasible);
    }
}

TEST_CASE("Regression #229: the objective offset takes the wrong sign when maximizing",
          "[validity]") {
    // max -x0 + 1.5 s.t. x0 + x1 >= 1 over [0, 1]^2: the optimum is 1.5.
    const LpProblem lp = [] {
        LpProblem m = makeModel(Sense::Maximize, {-1, 0}, {0, 0}, {1, 1},
                                {VarType::Continuous, VarType::Continuous}, {{1, 1}}, {1}, {kInf});
        m.obj_offset = 1.5;
        return m;
    }();
    DualSimplexSolver lp_solver;
    lp_solver.setVerbose(false);
    lp_solver.load(lp);
    REQUIRE(lp_solver.solve().status == Status::Optimal);
    CHECK_THAT(lp_solver.getObjective(), WithinAbs(1.5, kObjTol));

    MipSolver solver;
    solver.setVerbose(false);
    solver.setPresolve(false);
    solver.load(lp);
    CHECK_THAT(solver.solve().objective, WithinAbs(1.5, kObjTol));
}

TEST_CASE("Regression #230: the tree search loses the optimum of a maximization", "[validity]") {
    // max -4x0 - 3x1 - 2x2 - 2x3 s.t. -4x0 - 7x1 - 4x2 <= 15, all integer.
    // The optimum is 9 at (-1, -1, -1, 0); the solver stops at 8 after one
    // node. The same model as a minimization solves correctly, and root cuts
    // usually close the gap before the tree is entered, which hides this.
    constexpr auto I = VarType::Integer;
    const LpProblem lp = makeModel(Sense::Maximize, {-4, -3, -2, -2}, {-1, -2, -2, 0}, {1, 0, 1, 1},
                                   {I, I, I, I}, {{-4, -7, -4, 0}}, {-kInf}, {15});
    REQUIRE_THAT(enumerate(lp).best, WithinAbs(9.0, kObjTol));
    MipSolver solver;
    solver.setVerbose(false);
    solver.setNumThreads(1);
    solver.setCutsEnabled(false);
    solver.setPresolve(false);
    solver.load(lp);
    const auto result = solver.solve();
    REQUIRE(result.status == Status::Optimal);
    CHECK_THAT(objective(lp, result.solution), WithinAbs(9.0, kObjTol));

    // With default settings: max 3x0 - 2x1 + 3x2 - x3 s.t. -x0 - 4x1 + 2x2 <= -3,
    // 7x1 + 7x2 - 4x3 <= 9. The optimum is 4 at (2, 1, 0, 0); the solver
    // returns 3.
    const LpProblem defaults =
        makeModel(Sense::Maximize, {3, -2, 3, -1}, {0, 0, -2, 0}, {2, 1, 1, 1}, {I, I, I, I},
                  {{-1, -4, 2, 0}, {0, 7, 7, -4}}, {-kInf, -kInf}, {-3, 9});
    REQUIRE_THAT(enumerate(defaults).best, WithinAbs(4.0, kObjTol));
    MipSolver default_solver;
    default_solver.setVerbose(false);
    default_solver.setNumThreads(1);
    default_solver.load(defaults);
    const auto default_result = default_solver.solve();
    REQUIRE(default_result.status == Status::Optimal);
    CHECK_THAT(objective(defaults, default_result.solution), WithinAbs(4.0, kObjTol));
}

TEST_CASE("Regression #231: presolve treats semi-continuous columns as integer", "[validity]") {
    // min -2x0 + 2x1 + 4x2 s.t. -x0 <= 1, -4x0 + 2x2 = -4, x2 semi-continuous
    // in {0} or [1.5, 4]. The optimum is -6 at (1, -2, 0). Read as an interval
    // the semi column loses its 0, so a direct presolve call refuses the model
    // and presolves its linearization instead.
    constexpr auto I = VarType::Integer;
    const LpProblem lp =
        makeModel(Sense::Minimize, {-2, 2, 4}, {1, -2, 1.5}, {2, -1, 4},
                  {I, I, VarType::SemiContinuous}, {{-1, 0, 0}, {-4, 0, 2}}, {-kInf, -4}, {1, -4});
    REQUIRE_THAT(enumerate(lp).best, WithinAbs(-6.0, kObjTol));
    Presolver presolver;
    CHECK_THROWS_AS(presolver.presolve(lp), std::invalid_argument);
    auditPresolve(linearized(lp), presolveCases()[0]);
}

// A general-integer knapsack whose column k also sits in three equality rows
// over integer columns x0..x2, with coefficients near 1000 and |x| up to
// 20000. Only k = kv and x = xs (drawn here) satisfy the rows, so once a node
// fixes k the propagator pins x through rows with terms near 1e7. The models
// are those of a seeded generator at the given indices; each comes with its
// optimum, too large to enumerate but, with k and x fixed and the equality
// rows dropped, a plain knapsack.
namespace {

struct ChainKnapsack {
    int index;
    LpProblem lp;
    Real optimum;
};

std::vector<ChainKnapsack> chainKnapsacks(const std::vector<int>& picks) {
    std::vector<ChainKnapsack> models;
    std::mt19937 rng(1);
    std::uniform_int_distribution<int> val(-20000, 20000);
    std::uniform_real_distribution<double> big(500.0, 1500.0);
    std::uniform_real_distribution<double> small(0.5, 3.0);
    constexpr auto I = VarType::Integer;
    for (int t = 0; t <= picks.back(); ++t) {
        const Real xs[3] = {Real(val(rng)), Real(val(rng)), Real(val(rng))};
        const Index kc = t % 10;
        const Real kv = Real(t % 5);
        std::vector<std::vector<Real>> rows(4, std::vector<Real>(13, 0.0));
        std::vector<Real> rhs(3, 0.0);
        for (Index i = 0; i < 3; ++i) {
            rows[i][kc] = big(rng);
            rhs[i] = rows[i][kc] * kv;
            for (Index j = 0; j < 3; ++j) {
                rows[i][10 + j] = (j == i) ? small(rng) : big(rng);
                rhs[i] += rows[i][10 + j] * xs[j];
            }
        }
        if (std::find(picks.begin(), picks.end(), t) == picks.end()) {
            continue;
        }
        rows[3] = {5, 5, 4, 9, 7, 5, 3, 5, 3, 1, 0, 0, 0};
        std::vector<Real> obj = {-15, -14, -17, -17, -2, -17, -14, -8, -14, -13, 0, 0, 0};
        if (t % 2 == 1) {
            obj[kc] = -100.0;
        }
        std::vector<Real> lower(10, 0.0);
        std::vector<Real> upper(10, 5.0);
        lower.insert(lower.end(), 3, -20000.0);
        upper.insert(upper.end(), 3, 20000.0);
        LpProblem lp =
            makeModel(Sense::Minimize, obj, lower, upper, std::vector<VarType>(13, I), rows,
                      {rhs[0], rhs[1], rhs[2], -kInf}, {rhs[0], rhs[1], rhs[2], 82});

        LpProblem fixed = lp;
        fixed.col_lower[kc] = fixed.col_upper[kc] = kv;
        for (Index j = 0; j < 3; ++j) {
            fixed.col_lower[10 + j] = fixed.col_upper[10 + j] = xs[j];
            fixed.row_lower[j] = -kInf;
            fixed.row_upper[j] = kInf;
        }
        MipSolver knapsack;
        knapsack.setVerbose(false);
        knapsack.setNumThreads(1);
        knapsack.setGapTolerance(0.0);
        knapsack.load(fixed);
        const auto truth = knapsack.solve();
        REQUIRE(truth.status == Status::Optimal);
        models.push_back({t, std::move(lp), truth.objective});
    }
    return models;
}

/// Solves each model with cuts and presolve off, tree presolve forced at every
/// node and then off (which leaves the reduced-cost re-propagation).
void checkChainKnapsacks(const std::vector<int>& picks) {
    for (const auto& model : chainKnapsacks(picks)) {
        INFO("model t=" << model.index);
        for (const bool tree_presolve : {true, false}) {
            INFO("tree presolve at every node: " << tree_presolve);
            MipSolver solver;
            solver.setVerbose(false);
            solver.setNumThreads(1);
            solver.setGapTolerance(0.0);
            solver.setCutsEnabled(false);
            solver.setPresolve(false);
            solver.setTreePresolveEnabled(tree_presolve);
            solver.setTreePresolveAutoTuning(false);
            solver.setTreePresolveControls(1000, 1, 1);
            solver.load(model.lp);
            const auto result = solver.solve();
            REQUIRE(result.status == Status::Optimal);
            CHECK_THAT(result.objective, WithinAbs(model.optimum, kObjTol));
        }
    }
}

}  // namespace

TEST_CASE("Regression #234: tree presolve reads a tolerance crossing as infeasible", "[validity]") {
    // The propagator pinned x crossed by a few 1e-12: within its tolerance,
    // but the node code pruned at 1e-12. Tree presolve and the reduced-cost
    // re-propagation each lost the optimum (-455 or -540 for -461 or -544).
    checkChainKnapsacks({211, 2707, 4791});
}

TEST_CASE("Regression #235: integer rounding turns a near-integer bound into a crossing",
          "[validity]") {
    // A derived bound on x missed its integer by 1.9e-7, from the other
    // columns' bounds and the row sums at terms near 1e7. Rounding at 1e-8
    // floored it a whole unit below the lower bound, and the propagator
    // pruned the node holding the optimum (-455 for -461).
    checkChainKnapsacks({751, 5621});
}

TEST_CASE("Regression #235: presolve rounds a bound divided by a tiny coefficient", "[validity]") {
    // An equality row with integer coefficients and one of 1e-7, its rhs off
    // an integer by that term at the optimum. Presolve fixed the other
    // columns, divided the leftover rhs by 1e-7, got a bound 3.5e-8 above an
    // integer, rounded it a whole unit up and reported the model infeasible.
    constexpr auto I = VarType::Integer;
    constexpr auto B = VarType::Binary;
    const std::vector<std::pair<LpProblem, Real>> models = {
        // min -x0 - 2x1 - 3x2 s.t. -7x0 + 1e-7 x2 = -28 + 1e-7: -9 at (4, 1, 1),
        // via a singleton row.
        {makeModel(Sense::Minimize, {-1, -2, -3}, {1, 0, 0}, {5, 1, 1}, {I, I, B}, {{-7, 0, 1e-7}},
                   {-28 + 1e-7}, {-28 + 1e-7}),
         -9.0},
        // max -3x0 + 2x1 + 2x2 s.t. 7x0 - 3x1 + 1e-7 x2 = 18 + 2e-7: -3 at
        // (3, 1, 2), via activity bound tightening.
        {makeModel(Sense::Maximize, {-3, 2, 2}, {-1, 0, 1}, {4, 1, 5}, {I, B, I}, {{7, -3, 1e-7}},
                   {18 + 2e-7}, {18 + 2e-7}),
         -3.0},
    };
    for (const auto& [lp, optimum] : models) {
        INFO(describe(lp));
        REQUIRE_THAT(enumerate(lp).best, WithinAbs(optimum, kObjTol));
        MipSolver solver;
        solver.setVerbose(false);
        solver.setNumThreads(1);
        solver.load(lp);
        const auto result = solver.solve();
        REQUIRE(result.status == Status::Optimal);
        CHECK_THAT(result.objective, WithinAbs(optimum, kObjTol));
    }
}

namespace {

/// Equality rows over integers near 2e4 with fractional coefficients near
/// 1e4, so terms reach 1e9. Each rhs is the anchor's activity, rounded once
/// to double: the anchor sits on its rows up to that rounding.
struct LargeTermModel {
    LpProblem lp;
    std::vector<Real> anchor;
};

std::vector<LargeTermModel> largeTermModels(int count) {
    constexpr auto I = VarType::Integer;
    constexpr Index n = 4;
    constexpr Index m = 2;
    std::mt19937 rng(2361);
    std::uniform_real_distribution<double> coef(1e3, 1e5);
    std::uniform_int_distribution<int> val(-20000, 20000);
    std::uniform_int_distribution<int> cost(-3, 3);
    std::vector<LargeTermModel> models;
    for (int t = 0; t < count; ++t) {
        std::vector<Real> xs(n);
        for (auto& v : xs) {
            v = val(rng);
        }
        std::vector<std::vector<Real>> rows(m, std::vector<Real>(n));
        std::vector<Real> rhs(m);
        for (Index i = 0; i < m; ++i) {
            long double act = 0;
            for (Index j = 0; j < n; ++j) {
                rows[i][j] = coef(rng) * (j % 2 == 1 ? -1 : 1);
                act += static_cast<long double>(rows[i][j]) * xs[j];
            }
            rhs[i] = static_cast<Real>(act);
        }
        std::vector<Real> lower(n);
        std::vector<Real> upper(n);
        std::vector<Real> obj(n);
        for (Index j = 0; j < n; ++j) {
            lower[j] = xs[j] - 2;
            upper[j] = xs[j] + 2;
            obj[j] = cost(rng);
        }
        models.push_back({makeModel(Sense::Minimize, obj, lower, upper, std::vector<VarType>(n, I),
                                    rows, rhs, rhs),
                          xs});
    }
    return models;
}

}  // namespace

TEST_CASE("Regression #236: a row with terms near 1e9 reads its own point as infeasible",
          "[validity]") {
    // The activity sums round by more than an absolute 1e-8 at this size, so
    // the propagator rejected the anchor's own domain and the solver called
    // feasible models infeasible. Rows now hold to 1e-8 relative to their
    // magnitude.
    Int solved = 0;
    for (const auto& [lp, anchor] : largeTermModels(400)) {
        if (!pointFeasible(lp, anchor, kFeasTol)) {
            continue;
        }
        INFO(describe(lp) << "anchor" << describe(anchor));
        DomainPropagator prop;
        prop.load(lp);
        for (Index j = 0; j < lp.num_cols; ++j) {
            prop.setBound(j, anchor[j], anchor[j]);
        }
        CHECK(prop.propagate());
        for (const bool presolve : {false, true}) {
            INFO("presolve: " << presolve);
            MipSolver solver;
            solver.setVerbose(false);
            solver.setNumThreads(1);
            solver.setGapTolerance(0.0);
            solver.setPresolve(presolve);
            solver.setCutsEnabled(presolve);
            solver.load(lp);
            const auto result = solver.solve();
            REQUIRE(result.status == Status::Optimal);
            const std::vector<Real> x(result.solution.begin(),
                                      result.solution.begin() + lp.num_cols);
            CHECK(pointFeasible(lp, x, kFeasTol));
            CHECK(result.objective <= objective(lp, anchor) + kObjTol);
            ++solved;
        }
    }
    CHECK(solved > 300);
}

TEST_CASE("Regression #236: presolve drops an emptied row only within the feasibility tolerance",
          "[validity]") {
    // A fixed column folded into the row leaves its rhs 9 (or 9e-6) short.
    // The emptied row's relative slack at 1e9 is about 10, so the review of
    // the first fix found presolve dropping it and returning x as Optimal.
    // Past kFeasTol the row now stays, and the model is infeasible.
    constexpr auto C = VarType::Continuous;
    for (const Real base : {1e3, 1e9}) {
        const Real x = base + (base > 1e6 ? 9.0 : 9e-6);
        const LpProblem lp = makeModel(Sense::Minimize, {1}, {x}, {x}, {C}, {{1}}, {-kInf}, {base});
        INFO(describe(lp));
        MipSolver solver;
        solver.setVerbose(false);
        solver.setNumThreads(1);
        solver.load(lp);
        CHECK(solver.solve().status == Status::Infeasible);
    }
}

TEST_CASE("Regression #236: propagation through a tiny coefficient keeps the exact point",
          "[validity]") {
    // c x0 + a x1 = c K + t a with x0 fixed at K: x1 = t exactly. The bound on
    // x1 divides the rhs rounding of c K by a, up to 5e-4 for a near 1e-6, and
    // the propagator crossed x1's bounds on 2440 of 3000 such rows. A term
    // below the row's slack now bounds nothing.
    constexpr auto I = VarType::Integer;
    std::mt19937 rng(236);
    std::uniform_real_distribution<double> mantissa(1.0, 10.0);
    std::uniform_int_distribution<int> digit(1, 9);
    std::uniform_int_distribution<int> exponent(7, 10);
    std::uniform_int_distribution<int> target(1, 4);
    for (int trial = 0; trial < 3000; ++trial) {
        const Real k = std::floor(std::pow(10.0, mantissa(rng) / 2.0 + 1.0));
        const Real c = digit(rng);
        const Real a = mantissa(rng) * std::pow(10.0, -exponent(rng));
        const Real t = target(rng);
        const LpProblem lp = makeModel(Sense::Minimize, {0, 1}, {k, 0}, {k, 5}, {I, I}, {{c, a}},
                                       {c * k + t * a}, {c * k + t * a});
        INFO(describe(lp));
        DomainPropagator prop;
        prop.load(lp);
        REQUIRE(prop.propagate());
        CHECK(prop.getLower(1) <= t);
        CHECK(prop.getUpper(1) >= t);
    }
}

TEST_CASE("Regression #240: incumbents stay inside the column bounds with cuts on", "[validity]") {
    // Rounding submitted x3 = -1125 against bounds [1500, 2500] (-1125 is
    // x3's lower bound plus fixed x4's value), and the solve ended Optimal
    // with it, with presolve on and off.
    constexpr auto C = VarType::Continuous;
    constexpr auto I = VarType::Integer;
    LpProblem lp = makeModel(
        Sense::Maximize, {67968.75, -7.296875, -475, 0.0033593749999999999, 0.0013749999999999999},
        {-0.000475, -kInf, -0.029999999999999999, 1500, -2625}, {kInf, kInf, kInf, 2500, -2625},
        {I, I, I, C, I},
        {{262500000, -26250, 0, 1.25, 0},
         {-187.5, 0.02375, 4.75, -3.1250000000000001e-05, 0},
         {0, -0.014999999999999999, 2.25, 1.7500000000000002e-05, 6.2500000000000003e-06}},
        {-100156.25, -0.095468750000000005, -kInf}, {-80156.25, kInf, kInf});
    lp.obj_offset = 9.5;
    for (const bool presolve : {true, false}) {
        INFO("presolve " << presolve);
        MipSolver solver;
        solver.setVerbose(false);
        solver.setNumThreads(1);
        solver.setNodeLimit(300);
        solver.setTimeLimit(2.0);
        solver.setPresolve(presolve);
        std::vector<std::string> bad;
        solver.setIncumbentObserver([&](const std::vector<Real>& x, const std::string& source) {
            if (!pointFeasible(lp, x, kFeasTol * 10)) {
                bad.push_back(source + ": " + describe(x));
            }
        });
        solver.load(lp);
        const auto result = solver.solve();
        for (const auto& b : bad) {
            UNSCOPED_INFO(b);
        }
        CHECK(bad.empty());
        if (result.status == Status::Optimal) {
            INFO(describe(result.solution));
            CHECK(pointFeasible(lp, result.solution, kFeasTol * 10));
        }
    }
}
