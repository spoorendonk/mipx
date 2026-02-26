#include "mipx/separators.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_set>
#include <vector>

namespace mipx {

namespace {

Real computeLhs(const Cut& cut, std::span<const Real> primals) {
    Real lhs = 0.0;
    for (Index k = 0; k < static_cast<Index>(cut.indices.size()); ++k) {
        const Index j = cut.indices[k];
        if (j < 0 || j >= static_cast<Index>(primals.size())) continue;
        lhs += cut.values[k] * primals[j];
    }
    return lhs;
}

Real computeViolation(const Cut& cut, std::span<const Real> primals) {
    const Real lhs = computeLhs(cut, primals);
    Real violation = 0.0;
    if (cut.lower > -kInf) violation = std::max(violation, cut.lower - lhs);
    if (cut.upper < kInf) violation = std::max(violation, lhs - cut.upper);
    return violation;
}

bool hasFiniteBounds(const Cut& cut) {
    if (!std::isfinite(cut.lower) && cut.lower > -kInf) return false;
    if (!std::isfinite(cut.upper) && cut.upper < kInf) return false;
    return true;
}

bool isNumericallySafeCut(const Cut& cut) {
    if (cut.indices.empty()) return false;
    if (cut.indices.size() != cut.values.size()) return false;
    if (!hasFiniteBounds(cut)) return false;

    Real norm_sq = 0.0;
    Real max_abs = 0.0;
    Real min_abs = std::numeric_limits<Real>::infinity();
    Index prev = -1;

    for (Index k = 0; k < static_cast<Index>(cut.indices.size()); ++k) {
        const Index idx = cut.indices[k];
        const Real val = cut.values[k];
        if (idx < 0 || idx <= prev) return false;
        if (!std::isfinite(val)) return false;
        const Real abs_v = std::abs(val);
        if (abs_v <= 1e-12) return false;
        norm_sq += val * val;
        max_abs = std::max(max_abs, abs_v);
        min_abs = std::min(min_abs, abs_v);
        prev = idx;
    }

    if (norm_sq < 1e-12 || norm_sq > 1e16) return false;
    if (max_abs > 1e6) return false;
    if (min_abs < std::numeric_limits<Real>::infinity() &&
        max_abs / min_abs > 1e8) {
        return false;
    }
    return true;
}

bool addViolatedCut(Cut cut,
                    std::span<const Real> primals,
                    CutPool& pool,
                    CutFamilyStats& stats,
                    Real min_violation) {
    ++stats.generated;
    if (!isNumericallySafeCut(cut)) return false;

    const Real violation = computeViolation(cut, primals);
    if (violation < min_violation) return false;

    Real norm_sq = 0.0;
    for (Real v : cut.values) norm_sq += v * v;
    cut.efficacy = violation / std::sqrt(norm_sq);
    if (!std::isfinite(cut.efficacy) || cut.efficacy <= 0.0) return false;

    const Real efficacy = cut.efficacy;
    if (!pool.addCut(std::move(cut))) return false;
    ++stats.accepted;
    stats.efficacy_sum += efficacy;
    return true;
}

}  // namespace

bool SeparatorManager::isEnabled(CutFamily family) const {
    switch (family) {
        case CutFamily::Gomory: return config_.gomory;
        case CutFamily::Mir: return config_.mir;
        case CutFamily::Cover: return config_.cover;
        case CutFamily::ImpliedBound: return config_.implied_bound;
        case CutFamily::Clique: return config_.clique;
        case CutFamily::ZeroHalf: return config_.zero_half;
        case CutFamily::Mixing: return config_.mixing;
        case CutFamily::Unknown:
        case CutFamily::Count:
        default: return false;
    }
}

Int SeparatorManager::separate(DualSimplexSolver& lp,
                               const LpProblem& problem,
                               std::span<const Real> primals,
                               CutPool& pool,
                               CutSeparationStats& stats) {
    Int total_added = 0;

    auto runFamily = [&](CutFamily family, auto&& fn) {
        if (!isEnabled(family)) return;
        auto t0 = std::chrono::steady_clock::now();
        total_added += fn(stats.at(family));
        stats.at(family).time_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
    };

    runFamily(CutFamily::Gomory, [&](CutFamilyStats& s) {
        return separateGomory(lp, problem, primals, pool, s);
    });
    runFamily(CutFamily::Mir, [&](CutFamilyStats& s) {
        return separateMir(problem, primals, pool, s);
    });
    runFamily(CutFamily::Cover, [&](CutFamilyStats& s) {
        return separateCover(problem, primals, pool, s);
    });
    runFamily(CutFamily::ImpliedBound, [&](CutFamilyStats& s) {
        return separateImpliedBound(problem, primals, pool, s);
    });
    runFamily(CutFamily::Clique, [&](CutFamilyStats& s) {
        return separateClique(problem, primals, pool, s);
    });
    runFamily(CutFamily::ZeroHalf, [&](CutFamilyStats& s) {
        return separateZeroHalf(problem, primals, pool, s);
    });
    runFamily(CutFamily::Mixing, [&](CutFamilyStats& s) {
        return separateMixing(problem, primals, pool, s);
    });

    return total_added;
}

Int SeparatorManager::separateGomory(DualSimplexSolver& lp,
                                     const LpProblem& problem,
                                     std::span<const Real> primals,
                                     CutPool& pool,
                                     CutFamilyStats& stats) {
    gomory_.setMaxCuts(max_cuts_per_family_);
    gomory_.setMinViolation(min_violation_);
    const Int accepted = gomory_.separate(lp, problem, primals, pool);
    stats.attempted += accepted;
    stats.generated += accepted;
    stats.accepted += accepted;
    return accepted;
}

Int SeparatorManager::separateMir(const LpProblem& problem,
                                  std::span<const Real> primals,
                                  CutPool& pool,
                                  CutFamilyStats& stats) {
    Int accepted = 0;
    for (Index i = 0; i < problem.num_rows && accepted < max_cuts_per_family_; ++i) {
        if (problem.row_upper[i] >= kInf || problem.row_lower[i] > -kInf) continue;
        const Real rhs = problem.row_upper[i];
        if (!std::isfinite(rhs)) continue;
        ++stats.attempted;

        auto row = problem.matrix.row(i);
        Cut cut;
        cut.family = CutFamily::Mir;
        cut.lower = -kInf;
        cut.upper = std::floor(rhs + 1e-9);
        bool changed = (cut.upper + 1e-9 < rhs);
        bool valid = true;
        for (Index p = 0; p < row.size(); ++p) {
            const Index j = row.indices[p];
            const Real a = row.values[p];
            if (a < 0.0 || problem.col_lower[j] < -1e-12) {
                valid = false;
                break;
            }
            if (problem.col_type[j] == VarType::Continuous) continue;
            const Real rounded = std::floor(a + 1e-9);
            if (rounded <= 0.0) continue;
            if (rounded + 1e-9 < a) changed = true;
            cut.indices.push_back(j);
            cut.values.push_back(rounded);
        }
        if (!valid || !changed || cut.indices.empty()) continue;
        if (addViolatedCut(std::move(cut), primals, pool, stats, min_violation_)) {
            ++accepted;
        }
    }
    return accepted;
}

Int SeparatorManager::separateCover(const LpProblem& problem,
                                    std::span<const Real> primals,
                                    CutPool& pool,
                                    CutFamilyStats& stats) {
    struct Item {
        Index var;
        Real coeff;
    };

    Int accepted = 0;
    for (Index i = 0; i < problem.num_rows && accepted < max_cuts_per_family_; ++i) {
        if (problem.row_upper[i] >= kInf || problem.row_lower[i] > -kInf) continue;
        const Real rhs = problem.row_upper[i];
        if (!std::isfinite(rhs)) continue;
        ++stats.attempted;

        auto row = problem.matrix.row(i);
        std::vector<Item> items;
        items.reserve(static_cast<std::size_t>(row.size()));
        bool valid = true;
        for (Index p = 0; p < row.size(); ++p) {
            const Index j = row.indices[p];
            const Real a = row.values[p];
            if (a < 0.0 || problem.col_lower[j] < -1e-12) {
                valid = false;
                break;
            }
            if (problem.col_type[j] == VarType::Binary && a > 1e-9) {
                items.push_back({j, a});
            }
        }
        if (!valid || items.size() < 2) continue;

        std::sort(items.begin(), items.end(), [](const Item& a, const Item& b) {
            if (a.coeff != b.coeff) return a.coeff > b.coeff;
            return a.var < b.var;
        });

        std::vector<Item> cover;
        cover.reserve(items.size());
        Real sum = 0.0;
        for (const auto& item : items) {
            cover.push_back(item);
            sum += item.coeff;
            if (sum > rhs + 1e-9) break;
        }
        if (sum <= rhs + 1e-9) continue;

        while (cover.size() > 1) {
            auto min_it = std::min_element(
                cover.begin(), cover.end(),
                [](const Item& a, const Item& b) { return a.coeff < b.coeff; });
            if (sum - min_it->coeff > rhs + 1e-9) {
                sum -= min_it->coeff;
                cover.erase(min_it);
            } else {
                break;
            }
        }

        Cut cut;
        cut.family = CutFamily::Cover;
        cut.lower = -kInf;
        cut.upper = static_cast<Real>(cover.size()) - 1.0;
        for (const auto& item : cover) {
            cut.indices.push_back(item.var);
            cut.values.push_back(1.0);
        }
        std::sort(cut.indices.begin(), cut.indices.end());
        std::vector<Real> sorted_vals;
        sorted_vals.reserve(cut.values.size());
        for (Index idx : cut.indices) {
            for (const auto& item : cover) {
                if (item.var == idx) {
                    sorted_vals.push_back(1.0);
                    break;
                }
            }
        }
        cut.values = std::move(sorted_vals);
        if (addViolatedCut(std::move(cut), primals, pool, stats, min_violation_)) {
            ++accepted;
        }
    }
    return accepted;
}

Int SeparatorManager::separateImpliedBound(const LpProblem& problem,
                                           std::span<const Real> primals,
                                           CutPool& pool,
                                           CutFamilyStats& stats) {
    Int accepted = 0;
    for (Index i = 0; i < problem.num_rows && accepted < max_cuts_per_family_; ++i) {
        if (problem.row_upper[i] >= kInf || problem.row_lower[i] > -kInf) continue;
        const Real rhs = problem.row_upper[i];
        if (!std::isfinite(rhs) || rhs <= 0.0) continue;

        auto row = problem.matrix.row(i);
        bool nonnegative = true;
        for (Index p = 0; p < row.size(); ++p) {
            if (row.values[p] < 0.0 || problem.col_lower[row.indices[p]] < -1e-12) {
                nonnegative = false;
                break;
            }
        }
        if (!nonnegative) continue;

        for (Index px = 0; px < row.size() && accepted < max_cuts_per_family_; ++px) {
            const Index x = row.indices[px];
            const Real ax = row.values[px];
            if (ax <= 1e-9 || problem.col_type[x] == VarType::Binary) continue;

            for (Index py = 0; py < row.size() && accepted < max_cuts_per_family_; ++py) {
                if (px == py) continue;
                const Index y = row.indices[py];
                const Real ay = row.values[py];
                if (problem.col_type[y] != VarType::Binary || ay <= 1e-9) continue;
                ++stats.attempted;

                const Real ub0 = rhs / ax;
                const Real ub1 = (rhs - ay) / ax;
                if (!std::isfinite(ub0) || !std::isfinite(ub1)) continue;
                if (ub1 >= ub0 - 1e-9) continue;

                Cut cut;
                cut.family = CutFamily::ImpliedBound;
                cut.lower = -kInf;
                cut.upper = ub0;
                if (x < y) {
                    cut.indices = {x, y};
                    cut.values = {1.0, ub0 - ub1};
                } else {
                    cut.indices = {y, x};
                    cut.values = {ub0 - ub1, 1.0};
                }
                if (addViolatedCut(std::move(cut), primals, pool, stats, min_violation_)) {
                    ++accepted;
                }
            }
        }
    }
    return accepted;
}

Int SeparatorManager::separateClique(const LpProblem& problem,
                                     std::span<const Real> primals,
                                     CutPool& pool,
                                     CutFamilyStats& stats) {
    Int accepted = 0;
    std::unordered_set<std::uint64_t> seen_pairs;

    for (Index i = 0; i < problem.num_rows && accepted < max_cuts_per_family_; ++i) {
        if (problem.row_upper[i] >= kInf || problem.row_lower[i] > -kInf) continue;
        const Real rhs = problem.row_upper[i];
        if (!std::isfinite(rhs) || rhs <= 0.0) continue;

        auto row = problem.matrix.row(i);
        std::vector<std::pair<Index, Real>> binaries;
        binaries.reserve(static_cast<std::size_t>(row.size()));
        bool nonnegative = true;
        for (Index p = 0; p < row.size(); ++p) {
            const Index j = row.indices[p];
            const Real a = row.values[p];
            if (a < 0.0 || problem.col_lower[j] < -1e-12) {
                nonnegative = false;
                break;
            }
            if (problem.col_type[j] == VarType::Binary && a > 1e-9) {
                binaries.push_back({j, a});
            }
        }
        if (!nonnegative || binaries.size() < 2) continue;

        for (Index p = 0; p < static_cast<Index>(binaries.size()) &&
                         accepted < max_cuts_per_family_; ++p) {
            for (Index q = p + 1; q < static_cast<Index>(binaries.size()) &&
                                accepted < max_cuts_per_family_; ++q) {
                ++stats.attempted;
                const Index a = binaries[p].first;
                const Index b = binaries[q].first;
                const Real ca = binaries[p].second;
                const Real cb = binaries[q].second;
                if (ca + cb <= rhs + 1e-9) continue;

                const Index lo = std::min(a, b);
                const Index hi = std::max(a, b);
                const std::uint64_t key =
                    (static_cast<std::uint64_t>(static_cast<std::uint32_t>(lo)) << 32U) |
                    static_cast<std::uint32_t>(hi);
                if (!seen_pairs.insert(key).second) continue;

                Cut cut;
                cut.family = CutFamily::Clique;
                cut.lower = -kInf;
                cut.upper = 1.0;
                cut.indices = {lo, hi};
                cut.values = {1.0, 1.0};
                if (addViolatedCut(std::move(cut), primals, pool, stats, min_violation_)) {
                    ++accepted;
                }
            }
        }
    }
    return accepted;
}

Int SeparatorManager::separateZeroHalf(const LpProblem& problem,
                                       std::span<const Real> primals,
                                       CutPool& pool,
                                       CutFamilyStats& stats) {
    Int accepted = 0;
    for (Index i = 0; i < problem.num_rows && accepted < max_cuts_per_family_; ++i) {
        if (problem.row_upper[i] >= kInf || problem.row_lower[i] > -kInf) continue;
        const Real rhs = problem.row_upper[i];
        if (!std::isfinite(rhs) || rhs <= 0.0) continue;
        ++stats.attempted;

        auto row = problem.matrix.row(i);
        Cut cut;
        cut.family = CutFamily::ZeroHalf;
        cut.lower = -kInf;
        cut.upper = std::floor(0.5 * rhs + 1e-9);
        bool valid = true;
        bool changed = false;
        for (Index p = 0; p < row.size(); ++p) {
            const Index j = row.indices[p];
            const Real a = row.values[p];
            if (a < 0.0 || problem.col_lower[j] < -1e-12) {
                valid = false;
                break;
            }
            if (problem.col_type[j] == VarType::Continuous) continue;
            const Real rounded = std::floor(0.5 * a + 1e-9);
            if (rounded <= 0.0) continue;
            if (rounded + 1e-9 < a) changed = true;
            cut.indices.push_back(j);
            cut.values.push_back(rounded);
        }
        if (!valid || !changed || cut.indices.empty()) continue;
        if (addViolatedCut(std::move(cut), primals, pool, stats, min_violation_)) {
            ++accepted;
        }
    }
    return accepted;
}

Int SeparatorManager::separateMixing(const LpProblem& problem,
                                     std::span<const Real> primals,
                                     CutPool& pool,
                                     CutFamilyStats& stats) {
    Int accepted = 0;
    constexpr Real kScale = 1.0 / 3.0;
    for (Index i = 0; i < problem.num_rows && accepted < max_cuts_per_family_; ++i) {
        if (problem.row_upper[i] >= kInf || problem.row_lower[i] > -kInf) continue;
        const Real rhs = problem.row_upper[i];
        if (!std::isfinite(rhs) || rhs <= 0.0) continue;
        ++stats.attempted;

        auto row = problem.matrix.row(i);
        Cut cut;
        cut.family = CutFamily::Mixing;
        cut.lower = -kInf;
        cut.upper = std::floor(kScale * rhs + 1e-9);
        bool valid = true;
        bool changed = false;
        for (Index p = 0; p < row.size(); ++p) {
            const Index j = row.indices[p];
            const Real a = row.values[p];
            if (a < 0.0 || problem.col_lower[j] < -1e-12) {
                valid = false;
                break;
            }
            if (problem.col_type[j] == VarType::Continuous) continue;
            const Real rounded = std::floor(kScale * a + 1e-9);
            if (rounded <= 0.0) continue;
            if (rounded + 1e-9 < a) changed = true;
            cut.indices.push_back(j);
            cut.values.push_back(rounded);
        }
        if (!valid || !changed || cut.indices.empty()) continue;
        if (addViolatedCut(std::move(cut), primals, pool, stats, min_violation_)) {
            ++accepted;
        }
    }
    return accepted;
}

}  // namespace mipx
