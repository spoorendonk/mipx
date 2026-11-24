#pragma once

#include <string>

// Paths the build knows and the tests need: the instance data under the source
// tree and the binaries the tests spawn. They are absolute, so
// they name the checkout; only test_paths.cpp is compiled with them. Any other test file that
// embedded one would compile differently in every worktree, and the compiler cache (keyed on the
// command line and the preprocessed source) would miss it on each push's fresh local-CI worktree
// (#217).

namespace mipx::test {

/// `tests/data` in the source tree.
std::string testDataDir();

#ifdef MIPX_HAS_SOLVE_BINARY
/// The mipx-solve binary built alongside the tests.
std::string solveBinary();
#endif

#ifdef MIPX_HAS_PRESOLVE_BENCH_BINARY
/// The mipx-presolve-bench binary built alongside the tests.
std::string presolveBenchBinary();
#endif

#ifdef MIPX_HAS_PRESOLVE_CHECK_HIGHS_BINARY
/// The opt-in HiGHS postsolve check executable built alongside the tests.
std::string presolveCheckHighsBinary();
#endif

}  // namespace mipx::test
