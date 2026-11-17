#pragma once

// Reading an LP sequence in mcf's "mcf-lpseq 1" format (tinnet-ai/mcf#205;
// specified in mcf's include/mcf/lp/lp_recorder.h, summarised in
// tests/perf/README.md), shared by mipx-lpseq-replay and the opt-in
// mipx-lpseq-replay-highs (#267, #268), so both replay exactly the same ops.
//
// A sequence directory holds manifest.txt, master_0000.mps and ops.txt.
// readOp() parses one op of ops.txt and checks every index against the LP
// it applies to (the caller passes its current dimensions), so a malformed
// sequence is a FormatError, never an out-of-range write.

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <istream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace lpseq {

inline constexpr double kInf = std::numeric_limits<double>::infinity();

struct FormatError : std::runtime_error {
    explicit FormatError(const std::string& what) : std::runtime_error("lpseq: " + what) {}
};

[[noreturn]] inline void fail(const std::string& what) {
    throw FormatError(what);
}

// from_chars reads %.17g back exactly, subnormals and inf/-inf included. A
// literal past double's range (1e400) leaves the value unset, so it is an
// error, not 0.
inline double parseReal(std::string_view s) {
    const char* first = s.data();
    if (!s.empty() && s[0] == '+') {
        ++first;  // from_chars takes no leading '+'
    }
    double v = 0.0;
    const auto [end, ec] = std::from_chars(first, s.data() + s.size(), v);
    if (end != s.data() + s.size() || ec != std::errc() || std::isnan(v)) {
        fail("not a number: " + std::string(s));
    }
    return v;
}

// An index below bound, named `what` in the error.
inline int parseIndex(std::string_view s, int bound, const char* what) {
    int v = 0;
    const auto [end, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
    if (ec != std::errc() || end != s.data() + s.size() || v < 0) {
        fail(std::string(what) + ": not an index: " + std::string(s));
    }
    if (v >= bound) {
        fail(std::string(what) + ": index " + std::string(s) + " out of range (" +
             std::to_string(bound) + ")");
    }
    return v;
}

inline std::string trim(std::string_view s) {
    const auto b = s.find_first_not_of(" \t");
    if (b == std::string_view::npos) {
        return {};
    }
    const auto e = s.find_last_not_of(" \t\r");
    return std::string(s.substr(b, e - b + 1));
}

// key = value lines; the format must be exactly "mcf-lpseq 1", and a replay
// compares objectives of a minimised LP without a constant.
inline void checkManifest(const std::string& dir) {
    std::ifstream in(dir + "/manifest.txt");
    if (!in) {
        fail(dir + "/manifest.txt: cannot open");
    }
    std::string line;
    bool format = false;
    while (std::getline(in, line)) {
        const auto eq = line.find('=');
        if (eq == std::string::npos) {
            continue;
        }
        const std::string_view whole(line);
        const std::string key = trim(whole.substr(0, eq));
        const std::string value = trim(whole.substr(eq + 1));
        if (key == "format") {
            if (value != "mcf-lpseq 1") {
                fail("unsupported format: " + value);
            }
            format = true;
        } else if (key == "sense" && value != "min") {
            fail("unsupported sense: " + value);
        } else if (key == "obj_offset" && parseReal(value) != 0.0) {
            fail("unsupported obj_offset: " + value);
        }
    }
    if (!format) {
        fail(dir + "/manifest.txt: no format line");
    }
}

// The recorder names rows r<i> and columns c<j> by index; an MPS reader that
// drops N rows after the objective (mipx's and HiGHS's do; the recorder
// writes a row with both sides infinite as one) would shift the rows after
// it, so such a master is refused rather than replayed wrong.
template <typename Names>
void checkNames(const Names& names, int count, char prefix, const std::string& path) {
    for (int i = 0; i < count; ++i) {
        const std::string want = prefix + std::to_string(i);
        if (i >= static_cast<int>(names.size()) || names[i] != want) {
            fail(path + ": " + (prefix == 'r' ? "row " : "column ") + std::to_string(i) +
                 " is not " + want + (prefix == 'r' ? " (a free row, written as N?)" : ""));
        }
    }
}

enum class OpKind {
    AddCols,
    DelCols,
    AddRows,
    DelRows,
    ColBounds,
    RowBounds,
    ColCosts,
    Coeffs,
    Solve
};

// One op. Batches are in LpSolver::addCols / addRows form: entries of item t
// at [starts[t], starts[t + 1]), the last ending at indices.size().
struct Op {
    OpKind kind = OpKind::Solve;
    std::vector<int> starts, indices;  // add_*: per item; del_*: the deleted indices
    std::vector<double> values;
    std::vector<double> cost, lower, upper;  // add_cols: all three; add_rows, *_bounds: lower/upper
    std::vector<int> at;                     // *_bounds, col_costs: the index; coeffs: the row
    std::vector<int> col;                    // coeffs: the column
    // solve
    std::string seq, status;
    double recorded = 0.0;
};

namespace detail {

inline void checkBounds(double lo, double up, const char* what) {
    if (lo == kInf || up == -kInf) {
        fail(std::string(what) + ": bound on the wrong side");
    }
}

inline void checkValue(double v, const char* what) {
    if (!std::isfinite(v)) {
        fail(std::string(what) + ": coefficient not finite");
    }
}

// k lines of <a> [<b> [<c>]] <nnz> <index> <value> ..., cols: 3 leading
// numbers (cost, lb, ub), rows: 2 (lb, ub); entry indices below `bound`.
inline void readBatch(std::istream& in, int k, bool cols, int bound, Op& op) {
    const char* what = cols ? "add_cols" : "add_rows";
    std::string line;
    for (int t = 0; t < k; ++t) {
        if (!std::getline(in, line)) {
            fail(std::string(what) + ": batch cut short");
        }
        std::istringstream ls(line);
        std::string a, b, c, n;
        ls >> a >> b;
        if (cols) {
            ls >> c;
            op.cost.push_back(parseReal(a));
            if (std::isinf(op.cost.back())) {
                fail("add_cols: infinite cost");
            }
            op.lower.push_back(parseReal(b));
            op.upper.push_back(parseReal(c));
        } else {
            op.lower.push_back(parseReal(a));
            op.upper.push_back(parseReal(b));
        }
        checkBounds(op.lower.back(), op.upper.back(), what);
        ls >> n;
        const int nnz = parseIndex(n, std::numeric_limits<int>::max(), what);
        op.starts.push_back(static_cast<int>(op.indices.size()));
        for (int p = 0; p < nnz; ++p) {
            std::string idx, val;
            if (!(ls >> idx >> val)) {
                fail(std::string(what) + ": entry list cut short: " + line);
            }
            op.indices.push_back(parseIndex(idx, bound, what));
            op.values.push_back(parseReal(val));
            checkValue(op.values.back(), what);
        }
    }
}

}  // namespace detail

// Read the next op of `in` into `op`, checked against an LP of num_rows
// rows and num_cols columns. False at the end of the file.
inline bool readOp(std::istream& in, int num_rows, int num_cols, Op& op) {
    op = Op{};
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ls(line);
        std::string name;
        if (!(ls >> name)) {
            continue;
        }
        if (name == "solve") {
            std::string obj;
            ls >> op.seq >> op.status >> obj;
            op.kind = OpKind::Solve;
            op.recorded = parseReal(obj);
            return true;
        }
        std::string count;
        ls >> count;
        const int k = parseIndex(count, std::numeric_limits<int>::max(), name.c_str());
        auto field = [&](const char* what) {
            std::string s;
            if (!(ls >> s)) {
                fail(std::string(what) + ": line cut short");
            }
            return s;
        };
        if (name == "add_cols" || name == "add_rows") {
            const bool cols = name == "add_cols";
            op.kind = cols ? OpKind::AddCols : OpKind::AddRows;
            detail::readBatch(in, k, cols, cols ? num_rows : num_cols, op);
        } else if (name == "del_cols" || name == "del_rows") {
            const bool cols = name == "del_cols";
            op.kind = cols ? OpKind::DelCols : OpKind::DelRows;
            for (int t = 0; t < k; ++t) {
                op.indices.push_back(
                    parseIndex(field(name.c_str()), cols ? num_cols : num_rows, name.c_str()));
            }
            std::sort(op.indices.begin(), op.indices.end());
            op.indices.erase(std::unique(op.indices.begin(), op.indices.end()), op.indices.end());
        } else if (name == "col_bounds" || name == "row_bounds") {
            const bool cols = name == "col_bounds";
            op.kind = cols ? OpKind::ColBounds : OpKind::RowBounds;
            for (int t = 0; t < k; ++t) {
                op.at.push_back(
                    parseIndex(field(name.c_str()), cols ? num_cols : num_rows, name.c_str()));
                op.lower.push_back(parseReal(field(name.c_str())));
                op.upper.push_back(parseReal(field(name.c_str())));
                detail::checkBounds(op.lower.back(), op.upper.back(), name.c_str());
            }
        } else if (name == "col_costs") {
            op.kind = OpKind::ColCosts;
            for (int t = 0; t < k; ++t) {
                op.at.push_back(parseIndex(field("col_costs"), num_cols, "col_costs"));
                op.cost.push_back(parseReal(field("col_costs")));
                if (std::isinf(op.cost.back())) {
                    fail("col_costs: infinite cost");
                }
            }
        } else if (name == "coeffs") {
            op.kind = OpKind::Coeffs;
            std::unordered_set<std::int64_t> seen;
            for (int t = 0; t < k; ++t) {
                const int i = parseIndex(field("coeffs"), num_rows, "coeffs row");
                const int j = parseIndex(field("coeffs"), num_cols, "coeffs column");
                if (!seen.insert(static_cast<std::int64_t>(i) * num_cols + j).second) {
                    fail("coeffs: a position twice in one op");
                }
                op.at.push_back(i);
                op.col.push_back(j);
                op.values.push_back(parseReal(field("coeffs")));
                detail::checkValue(op.values.back(), "coeffs");
            }
        } else {
            fail("ops.txt: unknown op " + name);
        }
        return true;
    }
    return false;
}

// Whether a solve's result matches the recorded one: the same status and,
// when optimal, the objective within rel_tol of max(1, |recorded|).
inline bool matches(const Op& solve, const std::string& status, double objective, double rel_tol,
                    double& rel_err) {
    rel_err = std::abs(objective - solve.recorded) / std::max(1.0, std::abs(solve.recorded));
    return status == solve.status && (status != "optimal" || rel_err <= rel_tol);
}

}  // namespace lpseq
