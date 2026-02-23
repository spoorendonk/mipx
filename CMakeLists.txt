cmake_minimum_required(VERSION 3.25)
project(mipx LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(cmake/CompilerFlags.cmake)

# --- Dependencies ---

find_package(ZLIB REQUIRED)

find_package(TBB QUIET)
option(MIPX_USE_TBB "Enable TBB for parallel tree search" ${TBB_FOUND})

# --- Library ---

add_subdirectory(src)

# --- CLI executable ---

add_executable(mipx-solve src/cli/main.cpp)
target_link_libraries(mipx-solve PRIVATE mipx)
mipx_set_compiler_flags(mipx-solve)

# --- Tests ---

include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.7.1
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(Catch2)

include(${Catch2_SOURCE_DIR}/extras/Catch.cmake)

enable_testing()
add_subdirectory(tests)
