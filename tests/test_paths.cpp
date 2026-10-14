#include "test_paths.hpp"

namespace mipx::test {

std::string testDataDir() {
    return TEST_DATA_DIR;
}

#ifdef MIPX_HAS_SOLVE_BINARY
std::string solveBinary() {
    return MIPX_SOLVE_BINARY;
}
#endif

}  // namespace mipx::test
