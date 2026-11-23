// End-to-end tests for mipx-presolve-bench (#287): the presolve-only bench
// tool. They spawn the binary, as its output (one CSV row) and exit code are
// what the bench scripts read. Compiled out when the tool is not built or on
// platforms without popen().

#if defined(MIPX_HAS_PRESOLVE_BENCH_BINARY) && !defined(_WIN32)

#include "test_paths.hpp"

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace {

using mipx::test::testDataDir;

struct BenchRun {
    int exit_code = -1;
    std::string output;                      ///< stdout and stderr, interleaved.
    std::map<std::string, std::string> row;  ///< The CSV row by column; empty if none.

    [[nodiscard]] double real(const std::string& column) const {
        return std::strtod(row.at(column).c_str(), nullptr);
    }
    [[nodiscard]] long count(const std::string& column) const {
        return std::strtol(row.at(column).c_str(), nullptr, 10);
    }
};

std::vector<std::string> splitCsv(const std::string& line) {
    std::vector<std::string> cells;
    std::stringstream in(line);
    std::string cell;
    while (std::getline(in, cell, ',')) {
        cells.push_back(cell);
    }
    if (!line.empty() && line.back() == ',') {
        cells.emplace_back();
    }
    return cells;
}

/// Run mipx-presolve-bench with `args` (paths single-quoted by the caller)
/// and read the row after its "instance," header line.
BenchRun runBench(const std::string& args) {
    const std::string command = "'" + mipx::test::presolveBenchBinary() + "' " + args + " 2>&1";
    BenchRun run;
    FILE* pipe = popen(command.c_str(), "r");
    REQUIRE(pipe != nullptr);
    std::array<char, 8192> buffer{};
    while (std::fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
        run.output += buffer.data();
    }
    const int status = pclose(pipe);
    REQUIRE(status != -1);
    REQUIRE(WIFEXITED(status));
    run.exit_code = WEXITSTATUS(status);

    std::istringstream lines(run.output);
    std::string line;
    while (std::getline(lines, line)) {
        if (line.starts_with("instance,")) {
            std::string values;
            REQUIRE(std::getline(lines, values));
            const auto header = splitCsv(line);
            const auto cells = splitCsv(values);
            REQUIRE(header.size() == cells.size());
            for (std::size_t k = 0; k < header.size(); ++k) {
                run.row[header[k]] = cells[k];
            }
            break;
        }
    }
    return run;
}

std::string dataFile(const std::string& name) {
    return "'" + testDataDir() + "/" + name + "'";
}

/// A file under the system temp directory, removed when the test ends.
struct TempFile {
    std::filesystem::path path;
    explicit TempFile(const std::string& name, const std::string& contents = "")
        : path(std::filesystem::temp_directory_path() /
               ("mipx_presolve_bench_" + std::to_string(::getpid()) + "_" + name)) {
        std::filesystem::remove(path);
        if (!contents.empty()) {
            std::ofstream(path) << contents;
        }
    }
    ~TempFile() { std::filesystem::remove(path); }
    TempFile(const TempFile&) = delete;
    TempFile& operator=(const TempFile&) = delete;
    [[nodiscard]] std::string quoted() const { return "'" + path.string() + "'"; }
};

}  // namespace

TEST_CASE("Presolve bench: reports sizes and the reductions that fire", "[presolve][bench]") {
    const auto run = runBench(dataFile("presolve_bench.lp") + " --check-postsolve --repeats 3");
    INFO(run.output);
    REQUIRE(run.exit_code == 0);
    REQUIRE_FALSE(run.row.empty());
    CHECK(run.row.at("instance") == "presolve_bench");
    CHECK(run.row.at("solver") == "mipx");
    CHECK(run.row.at("status") == "reduced");
    CHECK(run.count("repeats") == 3);
    CHECK(run.real("seconds") > 0.0);
    CHECK(run.real("seconds") <= run.real("seconds_median"));

    CHECK(run.count("rows_before") == 8);
    CHECK(run.count("cols_before") == 9);
    CHECK(run.count("nnz_before") == 17);
    CHECK(run.count("rows_after") < 8);
    CHECK(run.count("cols_after") < 9);
    CHECK(run.count("nnz_after") < 17);
    CHECK(run.count("rows_removed") == 8 - run.count("rows_after"));
    CHECK(run.count("vars_removed") == 9 - run.count("cols_after"));

    // The reductions the LP is written to trigger (its header names them).
    for (const char* rule :
         {"singleton_row", "forcing_row", "dominated_row", "parallel_row", "empty_col"}) {
        INFO(rule);
        CHECK(run.count(std::string(rule) + "_changes") > 0);
        CHECK(run.real(std::string(rule) + "_seconds") >= 0.0);
    }
    // Off on the LP path, as in mipx-solve.
    CHECK(run.count("coeff_tightening_changes") == 0);
    CHECK(run.count("probing_changes") == 0);

    CHECK(run.row.at("check") == "pass");
    CHECK(run.row.at("reduced_status") == "optimal");
    CHECK(run.row.at("original_status") == "optimal");
    CHECK(run.real("violation") <= 1e-6);
    CHECK(run.real("objective") == -5.0);
    CHECK(run.real("reference") == -5.0);
}

TEST_CASE("Presolve bench: without --check-postsolve there are no check columns",
          "[presolve][bench]") {
    const auto run = runBench(dataFile("presolve_bench.lp"));
    INFO(run.output);
    REQUIRE(run.exit_code == 0);
    REQUIRE_FALSE(run.row.empty());
    CHECK(run.row.count("check") == 0);
    CHECK(run.count("repeats") == 1);
}

TEST_CASE("Presolve bench: the reference comes from --solu", "[presolve][bench]") {
    const TempFile right("right.solu", "=opt= presolve_bench -5\n=opt= other 1\n");
    const auto pass =
        runBench(dataFile("presolve_bench.lp") + " --check-postsolve --solu " + right.quoted());
    INFO(pass.output);
    CHECK(pass.exit_code == 0);
    CHECK(pass.row.at("check") == "pass");
    // Taken from the file, so the original is not solved.
    CHECK(pass.row.at("original_status").empty());

    const TempFile wrong("wrong.solu", "=opt= presolve_bench -4\n");
    const auto fail =
        runBench(dataFile("presolve_bench.lp") + " --check-postsolve --solu " + wrong.quoted());
    INFO(fail.output);
    CHECK(fail.exit_code == 1);
    CHECK(fail.row.at("check") == "fail");
    CHECK(fail.real("reference") == -4.0);
    CHECK(fail.real("rel_err") > 1e-6);
}

TEST_CASE("Presolve bench: a postsolved point that misses a row fails the check",
          "[presolve][bench]") {
    // Issue #257's LP: its presolved answer, postsolved, has missed its one
    // row by 5e-5. Whatever presolve does with it, the verdict must follow
    // the violation it reports.
    const auto run = runBench(dataFile("presolve_misses_row_lp.lp") + " --check-postsolve");
    INFO(run.output);
    REQUIRE_FALSE(run.row.empty());
    const bool misses = run.real("violation") > 1e-6;
    CHECK(run.row.at("check") == (misses ? "fail" : "pass"));
    CHECK(run.exit_code == (misses ? 1 : 0));
}

TEST_CASE("Presolve bench: an infeasible LP must be infeasible as read", "[presolve][bench]") {
    // tiny.mps: x2 + x3 = 7 with x2 binary and x3 <= 5.
    const auto run = runBench(dataFile("tiny.mps") + " --check-postsolve");
    INFO(run.output);
    REQUIRE(run.exit_code == 0);
    CHECK(run.row.at("status") == "infeasible");
    CHECK(run.row.at("check") == "pass");
    CHECK(run.row.at("original_status") == "infeasible");
}

TEST_CASE("Presolve bench: --csv appends rows under one header", "[presolve][bench]") {
    const TempFile csv("rows.csv");
    for (int k = 0; k < 2; ++k) {
        const auto run = runBench(dataFile("presolve_bench.lp") + " --csv " + csv.quoted());
        REQUIRE(run.exit_code == 0);
    }
    std::ifstream in(csv.path);
    std::vector<std::string> lines;
    for (std::string line; std::getline(in, line);) {
        lines.push_back(line);
    }
    REQUIRE(lines.size() == 3);
    CHECK(lines[0].starts_with("instance,solver,status,"));
    CHECK(lines[1].starts_with("presolve_bench,mipx,reduced,"));
    CHECK(lines[2].starts_with("presolve_bench,mipx,reduced,"));

    // A run with other columns (the check's) is refused, not appended.
    const auto other =
        runBench(dataFile("presolve_bench.lp") + " --check-postsolve --csv " + csv.quoted());
    CHECK(other.exit_code == 2);
    std::ifstream again(csv.path);
    std::size_t count = 0;
    for (std::string line; std::getline(again, line);) {
        ++count;
    }
    CHECK(count == 3);
}

TEST_CASE("Presolve bench: the instance name drops model suffixes in any case",
          "[presolve][bench]") {
    std::ifstream in(testDataDir() + "/presolve_bench.lp");
    std::stringstream lp;
    lp << in.rdbuf();
    const TempFile upper("Upper_Case.LP", lp.str());
    const auto run = runBench(upper.quoted());
    INFO(run.output);
    REQUIRE(run.exit_code == 0);
    CHECK(run.row.at("instance").ends_with("Upper_Case"));
}

TEST_CASE("Presolve bench: a missing or malformed file is a usage error", "[presolve][bench]") {
    const auto missing = runBench(dataFile("no_such_model.mps"));
    CHECK(missing.exit_code == 2);
    CHECK(missing.row.empty());

    const auto malformed = runBench(dataFile("malformed.mps"));
    CHECK(malformed.exit_code == 2);
    CHECK(malformed.row.empty());

    const auto no_model = runBench("--repeats 2");
    CHECK(no_model.exit_code == 2);

    const auto solu_alone = runBench(dataFile("presolve_bench.lp") + " --solu x.solu");
    CHECK(solu_alone.exit_code == 2);
}

#endif
