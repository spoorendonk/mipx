#pragma once

// What mipx-presolve-bench and mipx-presolve-bench-highs share (#287): the
// command line both take, the LP both read (mipx's reader, so the two
// presolves see the same model), and the CSV row both write. Their rows begin
// with the same columns, so one script can join them:
//   instance,solver,status,seconds,seconds_median,repeats,
//   rows_before,cols_before,nnz_before,rows_after,cols_after,nnz_after
// and go on with columns of their own.

#include "mipx/io.h"
#include "mipx/lp_problem.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace mipx::presolve_bench {

/// Exit codes of both tools.
inline constexpr int kExitOk = 0;
inline constexpr int kExitCheckFailed = 1;
inline constexpr int kExitUsage = 2;

[[noreturn]] inline void usageError(const std::string& usage, const std::string& what) {
    std::fprintf(stderr, "Error: %s\n%s", what.c_str(), usage.c_str());
    std::exit(kExitUsage);
}

struct CommonOptions {
    std::string path;
    std::string name;  ///< Instance name; from the file name when empty.
    ModelFormat format = ModelFormat::Auto;
    int repeats = 1;
    std::string csv;  ///< Append the row here as well, with a header if new.
};

/// The instance name of `path`: its file name less the model suffixes.
inline std::string instanceName(const std::string& path) {
    std::string name = path.substr(path.find_last_of('/') + 1);
    // In this order: a compression suffix first, and ".emps" before ".mps".
    for (const std::string_view suffix : {".gz", ".bz2", ".emps", ".mps", ".lp"}) {
        if (name.size() > suffix.size() && name.ends_with(suffix)) {
            name.resize(name.size() - suffix.size());
        }
    }
    return name;
}

/// Consume argv[i] (and its value) when it is one of the common options; a
/// first argument not starting with "--" is the model file.
inline bool parseCommon(int argc, char** argv, int& i, CommonOptions& opt,
                        const std::string& usage) {
    const std::string arg = argv[i];
    const auto value = [&]() -> std::string {
        if (i + 1 >= argc) {
            usageError(usage, arg + " needs a value");
        }
        return argv[++i];
    };
    if (arg == "--name") {
        opt.name = value();
    } else if (arg == "--format") {
        const std::string f = value();
        if (f == "auto") {
            opt.format = ModelFormat::Auto;
        } else if (f == "mps") {
            opt.format = ModelFormat::Mps;
        } else if (f == "lp") {
            opt.format = ModelFormat::Lp;
        } else {
            usageError(usage, "--format takes auto, mps or lp");
        }
    } else if (arg == "--repeats") {
        opt.repeats = std::atoi(value().c_str());
        if (opt.repeats < 1) {
            usageError(usage, "--repeats takes a count of at least 1");
        }
    } else if (arg == "--csv") {
        opt.csv = value();
    } else if (!arg.starts_with("--") && opt.path.empty()) {
        opt.path = arg;
    } else {
        return false;
    }
    return true;
}

/// Read the model, or exit with kExitUsage naming why not.
inline LpProblem readOrExit(const CommonOptions& opt) {
    try {
        return readModel(opt.path, opt.format);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Error: %s\n", e.what());
        std::exit(kExitUsage);
    }
}

/// Wall seconds of one call of `f`.
template <typename F>
double secondsOf(F&& f) {
    const auto start = std::chrono::steady_clock::now();
    std::forward<F>(f)();
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

/// Min and median of the repeats' times.
inline std::pair<double, double> minAndMedian(std::vector<double> times) {
    std::sort(times.begin(), times.end());
    const std::size_t n = times.size();
    const double median = n % 2 == 1 ? times[n / 2] : 0.5 * (times[n / 2 - 1] + times[n / 2]);
    return {times.front(), median};
}

/// One CSV row: column names and values, in order.
class Row {
public:
    void add(std::string column, std::string value) {
        columns_.push_back(std::move(column));
        values_.push_back(std::move(value));
    }
    void add(std::string column, long long value) { add(std::move(column), std::to_string(value)); }
    void addInt(std::string column, int value) {
        add(std::move(column), static_cast<long long>(value));
    }
    void addReal(std::string column, double value) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.10g", value);
        add(std::move(column), std::string(buf));
    }

    /// Rows, columns and nonzeros, as rows_<suffix>, cols_<suffix>, nnz_<suffix>.
    void addSizes(const std::string& suffix, long long rows, long long cols, long long nnz) {
        add("rows_" + suffix, rows);
        add("cols_" + suffix, cols);
        add("nnz_" + suffix, nnz);
    }
    void addSizes(const std::string& suffix, const LpProblem& lp) {
        addSizes(suffix, lp.num_rows, lp.num_cols, lp.matrix.numNonzeros());
    }

    /// Header and row on stdout; appended to `csv` too when set, with the
    /// header only when the file is new or empty.
    void emit(const std::string& csv) const {
        const std::string header = join(columns_);
        const std::string row = join(values_);
        std::printf("%s\n%s\n", header.c_str(), row.c_str());
        if (csv.empty()) {
            return;
        }
        bool fresh = true;
        {
            std::ifstream in(csv);
            fresh = !in || in.peek() == std::ifstream::traits_type::eof();
        }
        std::ofstream out(csv, std::ios::app);
        if (!out) {
            std::fprintf(stderr, "Error: cannot write %s\n", csv.c_str());
            std::exit(kExitUsage);
        }
        if (fresh) {
            out << header << '\n';
        }
        out << row << '\n';
    }

private:
    static std::string join(const std::vector<std::string>& cells) {
        std::string s;
        for (std::size_t k = 0; k < cells.size(); ++k) {
            if (k > 0) {
                s += ',';
            }
            s += cells[k];
        }
        return s;
    }

    std::vector<std::string> columns_;
    std::vector<std::string> values_;
};

/// The columns every row begins with.
inline void addCommon(Row& row, const std::string& instance, const std::string& solver,
                      const std::string& status, const std::vector<double>& times,
                      const LpProblem& before) {
    const auto [best, median] = minAndMedian(times);
    row.add("instance", instance);
    row.add("solver", solver);
    row.add("status", status);
    row.addReal("seconds", best);
    row.addReal("seconds_median", median);
    row.addInt("repeats", static_cast<int>(times.size()));
    row.addSizes("before", before);
}

}  // namespace mipx::presolve_bench
