cmake_minimum_required(VERSION 3.25)

# Default to Release for single-config generators. Several benchmark tests
# carry wall-clock limits (MIPLIB gen allows 20s against ~1.2s optimized), so
# an unoptimized build fails them on timing alone -- a confusing signal that
# looks like a solver regression. Multi-config generators pick per-build and
# are left alone. Set -DCMAKE_BUILD_TYPE=Debug explicitly for ASan/UBSan.
get_property(_mipx_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT _mipx_multi_config AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
  message(STATUS "CMAKE_BUILD_TYPE not set; defaulting to Release")
endif()

# Compiler cache. The pre-push gate does `rm -rf build` and a full rebuild on
# every push, which is exactly the case a compile cache is for: measured here,
# a wiped build directory goes from ~52s to ~2s once the cache is warm.
# Opportunistic -- no-op when neither tool is installed, and never overrides a
# launcher the caller set.
option(MIPX_USE_CCACHE "Use ccache/sccache as a compiler launcher when available" ON)
if(MIPX_USE_CCACHE AND NOT CMAKE_CXX_COMPILER_LAUNCHER)
    find_program(MIPX_COMPILER_CACHE NAMES ccache sccache)
    if(MIPX_COMPILER_CACHE)
        set(CMAKE_C_COMPILER_LAUNCHER "${MIPX_COMPILER_CACHE}")
        set(CMAKE_CXX_COMPILER_LAUNCHER "${MIPX_COMPILER_CACHE}")
        message(STATUS "Compiler cache: ${MIPX_COMPILER_CACHE}")
    endif()
endif()

project(mipx LANGUAGES CXX)

find_package(CUDAToolkit QUIET)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(cmake/CompilerFlags.cmake)
include(FetchContent)

# --- Dependencies ---

find_package(ZLIB REQUIRED)

find_package(BZip2 QUIET)
option(MIPX_USE_BZIP2 "Enable bzip2 decompression for .mps.bz2 files" ${BZIP2_FOUND})

find_package(TBB QUIET)
option(MIPX_USE_TBB "Enable TBB for parallel tree search" ${TBB_FOUND})
set(_mipx_use_gpu_default ${CUDAToolkit_FOUND})
if(DEFINED MIPX_USE_CUDA AND NOT DEFINED MIPX_USE_GPU)
    set(_mipx_use_gpu_default "${MIPX_USE_CUDA}")
endif()
option(MIPX_USE_GPU "Enable GPU acceleration" ${_mipx_use_gpu_default})
if(DEFINED MIPX_USE_CUDA)
    message(WARNING "MIPX_USE_CUDA is deprecated; use MIPX_USE_GPU instead.")
    if(NOT "${MIPX_USE_GPU}" STREQUAL "${MIPX_USE_CUDA}")
        message(FATAL_ERROR
                "MIPX_USE_GPU (${MIPX_USE_GPU}) and deprecated MIPX_USE_CUDA "
                "(${MIPX_USE_CUDA}) disagree")
    endif()
endif()
option(MIPX_ENABLE_LP_LIGHT_HEURISTICS
       "Enable LP-light heuristic adapter and optional LP-assisted pre-root arms"
       ON)
option(MIPX_STRICT_WARNINGS
       "Treat warnings as errors for project targets"
       ON)
option(MIPX_BUILD_CLI "Build mipx CLI executable" ON)
option(MIPX_BUILD_TESTS "Build and run C++ test suite" ON)
# Build the Python bindings by default when a suitable Python development
# environment is present; fall back to OFF on C++-only systems. Matches the
# auto-detect convention used for TBB/GPU/BZip2 above.
find_package(Python 3.12 QUIET COMPONENTS Interpreter Development.Module)
option(MIPX_BUILD_PYTHON "Build Python nanobind extension module" ${Python_FOUND})

set(MIPX_CUDA_BACKEND_ENABLED OFF)
set(MIPX_HAS_CUDSS OFF)
if(MIPX_USE_GPU)
    if(CUDAToolkit_FOUND)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        set(MIPX_CUDA_BACKEND_ENABLED ON)

        find_library(CUDSS_LIBRARY
                     NAMES cudss
                     HINTS
                         /usr/lib/x86_64-linux-gnu
                         /lib/x86_64-linux-gnu
                         ${CUDAToolkit_LIBRARY_DIR}
                         ${CUDAToolkit_LIBRARY_ROOT}
                     PATH_SUFFIXES lib lib64)
        find_path(CUDSS_INCLUDE_DIR
                  NAMES cudss.h
                  HINTS
                      /usr/include
                      /usr/local/include
                      ${CUDAToolkit_INCLUDE_DIRS})
        if(CUDSS_LIBRARY AND CUDSS_INCLUDE_DIR)
            set(MIPX_HAS_CUDSS ON)
            message(STATUS "Found cuDSS: ${CUDSS_LIBRARY}")
        else()
            message(WARNING
                    "MIPX_USE_GPU=ON but cuDSS was not found; GPU barrier backends will be disabled.")
        endif()
    else()
        message(WARNING
                "MIPX_USE_GPU=ON but CUDAToolkit was not found; building CPU-only.")
    endif()
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
