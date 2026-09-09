// End-to-end tests for the mipx-solve command line driver.
//
// These spawn the real binary because the bug they guard against (issue #198)
// is about what the process prints and what it exits with: reading an
// LP-format file as MPS produced a 0x0 model and reported "Optimal 0.0" with
// exit status 0. Only the process boundary shows that.
//
// The whole file is compiled out when the CLI is not built (MIPX_BUILD_CLI=OFF
// leaves no binary to run) or on platforms without popen().

#if defined(MIPX_SOLVE_BINARY) && !defined(_WIN32)

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstdio>
#include <string>
#include <string_view>
#include <sys/wait.h>

namespace {

std::string testDataDir() {
    return std::string(TEST_DATA_DIR);
}

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
    const std::string command = std::string("'") + MIPX_SOLVE_BINARY + "' " + args + " 2>&1";

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

#endif  // MIPX_SOLVE_BINARY && !_WIN32
