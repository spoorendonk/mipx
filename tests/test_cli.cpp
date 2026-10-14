// End-to-end tests for the mipx-solve command line driver.
//
// These spawn the real binary because the bug they guard against (issue #198)
// is about what the process prints and what it exits with: reading an
// LP-format file as MPS produced a 0x0 model and reported "Optimal 0.0" with
// exit status 0. Only the process boundary shows that.
//
// The whole file is compiled out when the CLI is not built (MIPX_BUILD_CLI=OFF
// leaves no binary to run) or on platforms without popen().

#if defined(MIPX_HAS_SOLVE_BINARY) && !defined(_WIN32)

#include "test_paths.hpp"

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <sys/wait.h>
#include <unistd.h>

namespace {

using mipx::test::testDataDir;

struct CliRun {
    int exit_code = -1;
    std::string output;  ///< stdout and stderr, interleaved.

    [[nodiscard]] bool mentions(std::string_view needle) const {
        return output.find(needle) != std::string::npos;
    }
};

/// Run mipx-solve with `args` and capture its combined output.
///
/// Arguments are single-quoted, so paths with spaces survive; none of them
/// contain a single quote.
CliRun runSolver(const std::string& args) {
    const std::string command = "'" + mipx::test::solveBinary() + "' " + args + " 2>&1";

    CliRun run;
    FILE* pipe = popen(command.c_str(), "r");
    REQUIRE(pipe != nullptr);

    std::array<char, 4096> buffer{};
    while (std::fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
        run.output += buffer.data();
    }

    const int status = pclose(pipe);
    REQUIRE(status != -1);
    REQUIRE(WIFEXITED(status));
    run.exit_code = WEXITSTATUS(status);
    return run;
}

}  // namespace

TEST_CASE("CLI: an .lp file is solved by the LP reader", "[cli][io]") {
    // Issue #198's reproducer: maximize x + y subject to x + y <= 3.
    const auto run = runSolver("'" + testDataDir() + "/lp_format.lp'");

    CHECK(run.exit_code == 0);
    CHECK(run.mentions("Status: Optimal"));
    CHECK(run.mentions("Objective: 3.0000000000e+00"));
    CHECK_FALSE(run.mentions("Objective: 0.0000000000e+00"));
}

TEST_CASE("CLI: an LP-format file forced to MPS is rejected", "[cli][io]") {
    const auto run = runSolver("'" + testDataDir() + "/lp_format.lp' --format mps");

    CHECK(run.exit_code != 0);
    CHECK(run.mentions("lp_format.lp"));
    CHECK(run.mentions("MPS"));
    CHECK_FALSE(run.mentions("Status: Optimal"));
}

TEST_CASE("CLI: a malformed MPS file is rejected", "[cli][io]") {
    const auto run = runSolver("'" + testDataDir() + "/malformed.mps'");

    CHECK(run.exit_code != 0);
    CHECK(run.mentions("malformed.mps"));
    CHECK(run.mentions("MPS"));
    CHECK_FALSE(run.mentions("Status: Optimal"));
}

TEST_CASE("CLI: an LP file with an unhelpful extension is rejected", "[cli][io]") {
    // Read as MPS, this file's BOUNDS section is a valid MPS header and its
    // bound records invent columns, so it solves to a nonsense answer unless
    // the format check looks for a header exclusive to MPS.
    const auto run = runSolver("'" + testDataDir() + "/lp_format_upper.txt'");

    CHECK(run.exit_code != 0);
    CHECK(run.mentions("lp_format_upper.txt"));
    CHECK_FALSE(run.mentions("Status: Optimal"));
}

TEST_CASE("CLI: a compressed LP file gets no --format hint", "[cli][io]") {
    // Neither reader can take a compressed LP file, so suggesting --format
    // would contradict the error's own advice to decompress it first.
    const std::string path = (std::filesystem::temp_directory_path() /
                              ("mipx_cli_compressed_" + std::to_string(::getpid()) + ".lp.gz"))
                                 .string();
    {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        out << "not really gzip";
    }
    const auto run = runSolver("'" + path + "'");
    std::filesystem::remove(path);

    CHECK(run.exit_code != 0);
    CHECK(run.mentions("Decompress"));
    CHECK_FALSE(run.mentions("Hint:"));
}

TEST_CASE("CLI: a format error from a guessed reader keeps its hint", "[cli][io]") {
    const auto run = runSolver("'" + testDataDir() + "/lp_format_upper.txt'");

    CHECK(run.exit_code != 0);
    CHECK(run.mentions("Hint:"));
}

TEST_CASE("CLI: an MPS file forced to LP is rejected", "[cli][io]") {
    const auto run = runSolver("'" + testDataDir() + "/tiny.mps' --format lp");

    CHECK(run.exit_code != 0);
    CHECK(run.mentions("tiny.mps"));
    CHECK_FALSE(run.mentions("Status: Optimal"));
}

TEST_CASE("CLI: a valid MPS file describing an empty model still solves", "[cli][io]") {
    const auto run = runSolver("'" + testDataDir() + "/empty_model.mps'");

    CHECK(run.exit_code == 0);
    CHECK(run.mentions("Status: Optimal"));
}

TEST_CASE("CLI: a valid MPS instance is unaffected", "[cli][io]") {
    // tiny.mps is infeasible as written (x2 is binary and x2 + x3 = 7 with
    // x3 <= 5), which is exactly the point: the file parses, the solver runs,
    // and the CLI reports the solver's verdict with exit status 0. Format
    // dispatch must not turn that into a read error.
    const auto run = runSolver("'" + testDataDir() + "/tiny.mps' --quiet");

    CHECK(run.exit_code == 0);
    CHECK(run.mentions("Status: Infeasible"));
    CHECK_FALSE(run.mentions("Error:"));
}

TEST_CASE("CLI: an LP that ends without an optimum reports no objective", "[cli]") {
    // Issue #206: time-limited LP runs printed the dual simplex's working
    // objective (1.4e14 on greenbeb) as though it were a result. Every
    // non-optimal exit shares the one reporting path; infeasibility reaches it
    // deterministically, where a time limit would not.
    const auto run = runSolver("'" + testDataDir() + "/infeasible_lp.lp' --dual --quiet");

    CHECK(run.exit_code == 0);
    CHECK(run.mentions("Status: Infeasible"));
    CHECK(run.mentions("Objective: none"));
}

TEST_CASE("CLI: a MIP stopped before any incumbent reports no objective", "[cli]") {
    // Issue #224: MipResult::objective defaults to 0, and the MIP summary
    // printed it whether or not a solution existed. The model has no integer
    // point, so with presolve and cuts off one node cannot find or rule one out.
    const auto run = runSolver("'" + testDataDir() +
                               "/no_incumbent_mip.lp' --no-presolve --no-cuts --node-limit 1 "
                               "--threads 1 --quiet");

    CHECK(run.exit_code == 0);
    CHECK(run.mentions("Status: Node limit"));
    CHECK(run.mentions("Objective: none"));
    CHECK_FALSE(run.mentions("Objective: 0"));
}

TEST_CASE("CLI: an infeasible MIP reports no objective", "[cli]") {
    const auto run = runSolver("'" + testDataDir() + "/infeasible_mip.lp' --quiet");

    CHECK(run.exit_code == 0);
    CHECK(run.mentions("Status: Infeasible"));
    CHECK(run.mentions("Objective: none"));
    CHECK_FALSE(run.mentions("Objective: 0"));
}

TEST_CASE("CLI: --format lp forces the LP reader", "[cli][io]") {
    // The override has to work in both directions, not just as a rejection.
    const auto run = runSolver("'" + testDataDir() + "/lp_format.lp' --format lp");

    CHECK(run.exit_code == 0);
    CHECK(run.mentions("Objective: 3.0000000000e+00"));
}

TEST_CASE("CLI: --format rejects an unknown value", "[cli][io]") {
    const auto run = runSolver("'" + testDataDir() + "/tiny.mps' --format xyz");

    CHECK(run.exit_code != 0);
    CHECK(run.mentions("Invalid --format value"));
}

// Issue #197: probing, presolve probing, the clique table and plunging are
// off by default and were unreachable from the command line. cli_features.mps
// is built so that each one fires: a 3-clique of BV columns, an implication
// chain (x4 <= x5, x4 <= x6, x5 + x6 <= 1) that only probing fixes, and a
// knapsack that branches once cuts are off.
namespace {

CliRun runFeatures(const std::string& flags) {
    return runSolver("'" + testDataDir() + "/cli_features.mps' --threads 1 " + flags);
}

}  // namespace

TEST_CASE("CLI: --help lists the feature flags", "[cli]") {
    const auto run = runSolver("--help");

    CHECK(run.exit_code == 0);
    CHECK(runSolver("'" + testDataDir() + "/cli_features.mps' --help").exit_code == 0);
    for (const char* flag :
         {"--probing", "--no-probing", "--probing-rounds", "--probing-probes", "--probing-time",
          "--presolve-probing", "--no-presolve-probing", "--presolve-probing-rounds",
          "--presolve-probing-probes", "--presolve-probing-time", "--clique-table",
          "--no-clique-table", "--plunge-depth", "--no-plunge", "--plunge-bound-quotient"}) {
        INFO(flag);
        CHECK(run.mentions(flag));
    }
}

TEST_CASE("CLI: --probing reaches the solver", "[cli]") {
    const auto off = runFeatures("");
    const auto on = runFeatures("--probing");
    const auto toggled = runFeatures("--probing --no-probing");

    CHECK(on.exit_code == 0);
    CHECK(on.mentions("Probing: "));
    CHECK(on.mentions("1 fixings"));
    CHECK_FALSE(off.mentions("Probing: "));
    CHECK_FALSE(toggled.mentions("Probing: "));
    // The budget knobs are passed through: one round is all it gets, and one
    // probe per round probes far fewer variables than the default budget.
    CHECK(runFeatures("--probing --probing-rounds 1").mentions(" 1 rounds"));
    CHECK(on.mentions("Probing: 31 vars probed"));
    CHECK(runFeatures("--probing --probing-probes 1").mentions("Probing: 2 vars probed"));
}

TEST_CASE("CLI: --presolve-probing reaches presolve", "[cli]") {
    const auto off = runFeatures("");
    const auto on = runFeatures("--presolve-probing");
    const auto toggled = runFeatures("--presolve-probing --no-presolve-probing");

    CHECK(on.exit_code == 0);
    CHECK(on.mentions("probe=1]"));
    // Without probing nothing in presolve fixes x4, so no reduction is logged.
    CHECK_FALSE(off.mentions("probe=1]"));
    CHECK_FALSE(toggled.mentions("probe=1]"));
    // A one-probe, one-round budget does not reach x4, so nothing is fixed.
    CHECK_FALSE(runFeatures("--presolve-probing --presolve-probing-probes 1 "
                            "--presolve-probing-rounds 1")
                    .mentions("probe=1]"));
}

TEST_CASE("CLI: --clique-table reaches the solver", "[cli]") {
    const auto off = runFeatures("");
    const auto on = runFeatures("--clique-table");
    const auto toggled = runFeatures("--clique-table --no-clique-table");

    CHECK(on.exit_code == 0);
    CHECK(on.mentions("Clique table: 11 binaries"));
    CHECK_FALSE(off.mentions("Clique table:"));
    CHECK_FALSE(toggled.mentions("Clique table:"));
}

TEST_CASE("CLI: --plunge-depth reaches the tree search", "[cli]") {
    const auto off = runFeatures("--no-cuts");
    const auto on = runFeatures("--no-cuts --plunge-depth 4");
    const auto toggled = runFeatures("--no-cuts --plunge-depth 4 --no-plunge");

    CHECK(on.exit_code == 0);
    CHECK(on.mentions("Plunge: nodes="));
    CHECK_FALSE(off.mentions("Plunge: nodes="));
    CHECK_FALSE(toggled.mentions("Plunge: nodes="));
    // All of them still solve the model.
    for (const auto* run : {&off, &on, &toggled}) {
        CHECK(run->mentions("Objective: -2.2000000000e+01"));
    }
}

#endif  // MIPX_HAS_SOLVE_BINARY && !_WIN32
