#include "mipx/symmetry.h"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <format>
#include <functional>
#include <span>
#include <unordered_map>
#include <vector>

#include "mipx/automorphism.h"
#include "mipx/orbital_fixing.h"
#include "mipx/schreier_sims.h"
#include "mipx/sparse_matrix.h"
#include "mipx/symbreak.h"

namespace mipx {

namespace {

struct ColumnSignature {
    Real obj = 0.0;
    Real lb = 0.0;
    Real ub = 0.0;
    VarType type = VarType::Continuous;
    std::vector<std::pair<Index, Real>> entries;

    bool operator==(const ColumnSignature& other) const {
        if (type != other.type) return false;
        if (std::bit_cast<std::uint64_t>(obj) != std::bit_cast<std::uint64_t>(other.obj)) return false;
        if (std::bit_cast<std::uint64_t>(lb) != std::bit_cast<std::uint64_t>(other.lb)) return false;
        if (std::bit_cast<std::uint64_t>(ub) != std::bit_cast<std::uint64_t>(other.ub)) return false;
        return entries == other.entries;
    }
};

struct ColumnSignatureHash {
    std::size_t operator()(const ColumnSignature& signature) const {
        std::size_t h = std::hash<int>()(static_cast<int>(signature.type));
        auto mix = [&](std::uint64_t value) {
            h ^= std::hash<std::uint64_t>()(value) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        };
        mix(std::bit_cast<std::uint64_t>(signature.obj));
        mix(std::bit_cast<std::uint64_t>(signature.lb));
        mix(std::bit_cast<std::uint64_t>(signature.ub));
        for (const auto& [idx, val] : signature.entries) {
            mix(static_cast<std::uint64_t>(idx));
            mix(std::bit_cast<std::uint64_t>(val));
        }
        return h;
    }
};

}  // namespace

// ---------------------------------------------------------------------------
// Column-signature detection (original fast path)
// ---------------------------------------------------------------------------

void SymmetryManager::detectByColumnSignature(const LpProblem& problem) {
    std::unordered_map<ColumnSignature, std::vector<Index>, ColumnSignatureHash> buckets;

    for (Index j = 0; j < problem.num_cols; ++j) {
        if (problem.col_type[j] == VarType::Continuous) continue;
        detect_work_units_ += 1.0;
        ColumnSignature sig;
        sig.obj = problem.obj[j];
        sig.lb = problem.col_lower[j];
        sig.ub = problem.col_upper[j];
        sig.type = problem.col_type[j];
        auto column = problem.matrix.col(j);
        sig.entries.reserve(static_cast<size_t>(column.size()));
        for (Index k = 0; k < column.size(); ++k) {
            sig.entries.emplace_back(column.indices[k], column.values[k]);
            detect_work_units_ += 1.0;
        }
        buckets[sig].push_back(j);
    }

    for (auto& [sig, vars] : buckets) {
        if (vars.size() < 2) continue;
        std::sort(vars.begin(), vars.end());
        orbits_.push_back(vars);
    }
    std::sort(orbits_.begin(), orbits_.end(),
              [](const std::vector<Index>& a, const std::vector<Index>& b) {
                  return a.front() < b.front();
              });
}

// ---------------------------------------------------------------------------
// Graph automorphism detection (new, complete)
// ---------------------------------------------------------------------------

void SymmetryManager::detectByAutomorphism(const LpProblem& problem) {
    // Build incidence graph.
    ColoredGraph graph = buildIncidenceGraph(problem);
    detect_work_units_ += static_cast<double>(graph.num_vertices);

    // Compute automorphisms.
    automorphism_result_ = computeAutomorphisms(graph, problem.num_cols);
    detect_work_units_ += automorphism_result_.work_units;

    // Use the variable orbits from automorphism computation.
    orbits_ = automorphism_result_.orbits;
    used_full_detection_ = true;

    // Build Schreier-Sims structure if we have generators.
    if (!automorphism_result_.generators.empty()) {
        schreier_sims_.build(automorphism_result_.generators,
                             graph.num_vertices,
                             problem.num_cols);
        detect_work_units_ += schreier_sims_.buildWorkUnits();

        // Set up orbital fixer.
        orbital_fixer_.setSchreierSims(&schreier_sims_);

        // Set up isomorphism pruner.
        isomorphism_pruner_.setGenerators(automorphism_result_.generators,
                                          problem.num_cols);
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

void SymmetryManager::detect(const LpProblem& problem) {
    orbits_.clear();
    canonical_.assign(static_cast<std::size_t>(problem.num_cols), -1);
    orbital_fixes_.clear();
    detect_work_units_ = 0.0;
    cut_work_units_ = 0.0;
    used_full_detection_ = false;
    automorphism_result_ = {};

    detectByColumnSignature(problem);
    buildCanonicalMapping();
}

void SymmetryManager::detectFull(const LpProblem& problem) {
    orbits_.clear();
    canonical_.assign(static_cast<std::size_t>(problem.num_cols), -1);
    orbital_fixes_.clear();
    detect_work_units_ = 0.0;
    cut_work_units_ = 0.0;
    used_full_detection_ = false;
    automorphism_result_ = {};

    // Estimate graph size: num_cols + num_rows + nnz (intermediate vertices).
    Index estimated_vertices = problem.num_cols + problem.num_rows +
                               problem.matrix.numNonzeros();

    if (estimated_vertices <= kMaxGraphVertices) {
        detectByAutomorphism(problem);
    } else {
        // Fall back to column-signature for large problems.
        detectByColumnSignature(problem);
    }

    // If automorphism found fewer orbits than column-signature would,
    // also run column-signature and merge results.
    if (orbits_.empty()) {
        detectByColumnSignature(problem);
    }

    buildCanonicalMapping();
}

void SymmetryManager::buildCanonicalMapping() {
    for (const auto& orbit : orbits_) {
        detect_work_units_ += static_cast<double>(orbit.size());
        Index canon = orbit.front();
        for (Index var : orbit) {
            canonical_[static_cast<std::size_t>(var)] = canon;
        }
        for (Index idx = 1; idx < static_cast<Index>(orbit.size()); ++idx) {
            orbital_fixes_.push_back({orbit[idx], canon});
        }
    }
}

bool SymmetryManager::hasSymmetry() const {
    return !orbits_.empty();
}

Index SymmetryManager::canonical(Index var) const {
    if (var < 0 || var >= static_cast<Index>(canonical_.size())) return var;
    return canonical_[static_cast<std::size_t>(var)] >= 0 ? canonical_[static_cast<std::size_t>(var)] : var;
}

bool SymmetryManager::isCanonical(Index var) const {
    return canonical(var) == var;
}

const std::vector<std::vector<Index>>& SymmetryManager::orbits() const {
    return orbits_;
}

const std::vector<OrbitalFix>& SymmetryManager::orbitalFixes() const {
    return orbital_fixes_;
}

double SymmetryManager::detectWorkUnits() const {
    return detect_work_units_;
}

double SymmetryManager::cutWorkUnits() const {
    return cut_work_units_;
}

Index SymmetryManager::addSymmetryCuts(LpProblem& problem) {
    if (orbital_fixes_.empty()) return 0;
    if (problem.row_names.size() < static_cast<std::size_t>(problem.num_rows)) {
        problem.row_names.resize(static_cast<std::size_t>(problem.num_rows));
    }
    cut_work_units_ = 0.0;
    Index added = 0;
    for (const auto& fix : orbital_fixes_) {
        std::vector<std::pair<Index, Real>> entries = {
            {fix.canonical, -1.0},
            {fix.variable, 1.0},
        };
        std::sort(entries.begin(), entries.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });

        std::vector<Index> cols;
        std::vector<Real> vals;
        cols.reserve(entries.size());
        vals.reserve(entries.size());
        for (const auto& [col, val] : entries) {
            cols.push_back(col);
            vals.push_back(val);
        }

        problem.matrix.addRow(std::span<const Index>(cols.data(), cols.size()),
                              std::span<const Real>(vals.data(), vals.size()));
        problem.row_lower.push_back(-kInf);
        problem.row_upper.push_back(0.0);
        problem.row_names.push_back(std::format("sym_cut_{}_{}", fix.canonical, fix.variable));
        ++added;
        cut_work_units_ += 2.0;
    }
    problem.num_rows = problem.matrix.numRows();
    return added;
}

Index SymmetryManager::addSymbreakConstraints(LpProblem& problem) {
    if (!used_full_detection_ || automorphism_result_.generators.empty()) {
        return 0;
    }

    auto constraints = symbreak_generator_.generate(
        automorphism_result_.generators,
        problem.col_type,
        problem.num_cols);

    Index added = SymbreakGenerator::addConstraints(problem, constraints);
    cut_work_units_ += static_cast<double>(added) * 2.0;
    return added;
}

Index SymmetryManager::aggregateConstraints(LpProblem& problem) {
    if (!used_full_detection_ || automorphism_result_.generators.empty()) {
        return 0;
    }
    return SymbreakGenerator::aggregateSymmetricConstraints(
        problem, automorphism_result_.generators);
}

OrbitalFixingResult SymmetryManager::applyOrbitalFixing(
    std::vector<Real>& col_lower,
    std::vector<Real>& col_upper,
    const std::vector<VarType>& col_type,
    const std::vector<Index>& fixed_vars,
    Index num_cols) const {
    if (used_full_detection_ && schreier_sims_.isBuilt()) {
        return orbital_fixer_.fix(col_lower, col_upper, col_type,
                                  fixed_vars, num_cols);
    }
    // Fall back to canonical-based fixing.
    return OrbitalFixer::fixFromCanonical(col_lower, col_upper,
                                          canonical_, num_cols);
}

bool SymmetryManager::canPruneByIsomorphism(
    const std::vector<Real>& col_lower,
    const std::vector<Real>& col_upper) const {
    return isomorphism_pruner_.canPrune(col_lower, col_upper);
}

void SymmetryManager::recordExplored(
    const std::vector<Real>& col_lower,
    const std::vector<Real>& col_upper) {
    isomorphism_pruner_.recordExplored(col_lower, col_upper);
}

Index SymmetryManager::numIsomorphismPrunes() const {
    return isomorphism_pruner_.numPruned();
}

}  // namespace mipx
