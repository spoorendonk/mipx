cmake_minimum_required(VERSION 3.25)
project(mipx LANGUAGES CXX)

# CUDA language is enabled conditionally in src/CMakeLists.txt after the option
# is resolved, so the project declaration stays CXX-only here.

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(cmake/CompilerFlags.cmake)
include(FetchContent)

# --- Dependencies ---

find_package(ZLIB REQUIRED)
find_package(CUDAToolkit QUIET)

find_package(TBB QUIET)
option(MIPX_USE_TBB "Enable TBB for parallel tree search" ${TBB_FOUND})
option(MIPX_USE_CUDA "Enable CUDA acceleration (barrier + PDLP GPU-resident solver)" ${CUDAToolkit_FOUND})
option(MIPX_ENABLE_LP_LIGHT_HEURISTICS
       "Enable LP-light heuristic adapter and optional LP-assisted pre-root arms"
       ON)
option(MIPX_STRICT_WARNINGS
       "Treat warnings as errors for project targets"
       ON)
option(MIPX_BUILD_CLI "Build mipx CLI executable" ON)
option(MIPX_BUILD_TESTS "Build and run C++ test suite" ON)
option(MIPX_BUILD_PYTHON "Build Python nanobind extension module" OFF)

if(MIPX_USE_CUDA AND NOT CUDAToolkit_FOUND)
    message(FATAL_ERROR "MIPX_USE_CUDA=ON but CUDAToolkit was not found")
endif()

# SIMD codegen policy:
# - default is native for maximum local performance
# - avx2/off remain available for standardized distribution builds
set(MIPX_SIMD_ISA "native" CACHE STRING
    "SIMD code generation target (off, avx2, native)")
set_property(CACHE MIPX_SIMD_ISA PROPERTY STRINGS off avx2 native)
option(MIPX_SIMD_REQUIRE_SUPPORTED
       "Fail configure if requested SIMD ISA flags are not supported"
       OFF)

# --- Library ---

add_subdirectory(src)

# --- CLI executable ---

if(MIPX_BUILD_CLI)
    add_executable(mipx-solve src/cli/main.cpp)
    target_link_libraries(mipx-solve PRIVATE mipx)
    mipx_set_compiler_flags(mipx-solve)
endif()

# --- Python extension ---

if(MIPX_BUILD_PYTHON)
    find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module)
    FetchContent_Declare(
        nanobind
        GIT_REPOSITORY https://github.com/wjakob/nanobind.git
        GIT_TAG        v2.4.0
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(nanobind)
    add_subdirectory(python)
endif()

# --- Tests ---

if(MIPX_BUILD_TESTS)
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
endif()
