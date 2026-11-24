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
if(NOT SKBUILD)
    include(cmake/ClangTools.cmake)
endif()
include(FetchContent)

# --- Dependencies ---

find_package(ZLIB REQUIRED)

find_package(BZip2 QUIET)
option(MIPX_USE_BZIP2 "Enable bzip2 decompression for .mps.bz2 files" ${BZIP2_FOUND})

find_package(TBB QUIET)
option(MIPX_USE_TBB "Enable TBB for parallel tree search" ${TBB_FOUND})

# BLAS for the dense blocks of the barrier's supernodal Cholesky (#290),
# OpenBLAS preferred, loaded at run time (src/CMakeLists.txt). Optional:
# without it those blocks run on mipx's own loops.
if(NOT DEFINED BLA_VENDOR)
    set(BLA_VENDOR OpenBLAS)
    find_package(BLAS QUIET)
    unset(BLA_VENDOR)
endif()
if(NOT BLAS_FOUND)
    find_package(BLAS QUIET)
endif()
option(MIPX_USE_BLAS "Use BLAS for the barrier's supernodal Cholesky" ${BLAS_FOUND})
# The shared library to dlopen; empty without one. Its resolved file is the
# fallback: the found path is often a -dev package's symlink, absent on a
# host with the runtime package only, while the symlink itself follows a
# library upgrade, so it is tried first.
set(MIPX_BLAS_SHARED_LIBRARY "")
set(MIPX_BLAS_SHARED_LIBRARY_REAL "")
if(MIPX_USE_BLAS)
    foreach(_lib IN LISTS BLAS_LIBRARIES)
        if(_lib MATCHES "\\.(so|dylib)(\\.[0-9]+)*$" AND EXISTS "${_lib}")
            set(MIPX_BLAS_SHARED_LIBRARY "${_lib}")
            file(REAL_PATH "${_lib}" MIPX_BLAS_SHARED_LIBRARY_REAL)
            break()
        endif()
    endforeach()
    if(MIPX_BLAS_SHARED_LIBRARY)
        message(STATUS "mipx: barrier BLAS loaded at run time from ${MIPX_BLAS_SHARED_LIBRARY}")
    else()
        message(STATUS "mipx: no shared BLAS library in BLAS_LIBRARIES; own loops")
    endif()
endif()
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
        # nvcc's own default architecture (75 for CUDA 13) may be one the
        # driver cannot JIT for the GPU at hand, and then no kernel runs at
        # all (#192). Build for the GPUs found here, or for every major
        # architecture when nvidia-smi finds none.
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
            execute_process(COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
                            RESULT_VARIABLE _mipx_smi_result
                            OUTPUT_VARIABLE _mipx_gpu_caps
                            OUTPUT_STRIP_TRAILING_WHITESPACE
                            ERROR_QUIET)
            if(_mipx_smi_result EQUAL 0 AND _mipx_gpu_caps)
                set(_mipx_cuda_archs native)
            else()
                set(_mipx_cuda_archs all-major)
            endif()
            set(CMAKE_CUDA_ARCHITECTURES ${_mipx_cuda_archs} CACHE STRING
                "CUDA architectures (default: native with a GPU, else all-major)")
        endif()
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

    # Replays LP sequences in mcf's "mcf-lpseq 1" format (#267).
    add_executable(mipx-lpseq-replay tests/perf/lpseq_replay.cpp)
    target_link_libraries(mipx-lpseq-replay PRIVATE mipx)
    mipx_set_compiler_flags(mipx-lpseq-replay)

    # Presolve alone, without the solve, per reduction (#287).
    add_executable(mipx-presolve-bench tests/perf/presolve_bench.cpp)
    target_link_libraries(mipx-presolve-bench PRIVATE mipx)
    mipx_set_compiler_flags(mipx-presolve-bench)

    add_executable(mipx-root-heuristic-deadline-bench EXCLUDE_FROM_ALL
                   tests/perf/root_heuristic_deadline_bench.cpp)
    target_link_libraries(mipx-root-heuristic-deadline-bench PRIVATE mipx)
    mipx_set_compiler_flags(mipx-root-heuristic-deadline-bench)
endif()

# --- Bench-only HiGHS comparator (#268) ---
#
# mipx-lpseq-replay-highs replays the same LP sequences on HiGHS through its
# API (warm as well as cold), for comparison benches. Opt-in and OFF by
# default; the mipx library never links HiGHS (CLAUDE.md, "No external
# solver dependencies": bench targets only). HiGHS is the v1.16.0 release
# archive, pinned by hash, built static with HiPO (needs a BLAS, e.g.
# libopenblas-dev).
option(MIPX_BENCH_HIGHS
       "Build mipx-lpseq-replay-highs and mipx-presolve-bench-highs, fetching HiGHS v1.16.0 (bench only)"
       OFF)
if(MIPX_BENCH_HIGHS)
    function(mipx_fetch_highs)
        # HiGHS's own build options, scoped to this function.
        set(BUILD_SHARED_LIBS OFF)
        set(BUILD_TESTING OFF)
        set(BUILD_EXAMPLES OFF)
        set(BUILD_CXX_EXE OFF)
        set(FAST_BUILD ON)
        set(HIPO ON)
        # HiPO's extras statically in libhighs: as a shared extras library
        # they can go unresolved at run time, and HiGHS then falls back to
        # dual simplex while still reporting "hipo".
        set(BUILD_SHARED_EXTRAS_LIB OFF)
        FetchContent_Declare(highs
            URL https://github.com/ERGO-Code/HiGHS/archive/refs/tags/v1.16.0.tar.gz
            URL_HASH SHA256=0fa9da512955fb08c6cca7bfb274ca12e8a0e8a12e44189bd6e0c355ce7701bc
            DOWNLOAD_EXTRACT_TIMESTAMP FALSE
            # Its headers as system ones for consumers, outside mipx's
            # warnings.
            SYSTEM)
        FetchContent_MakeAvailable(highs)
    endfunction()
    mipx_fetch_highs()
    add_executable(mipx-lpseq-replay-highs tests/perf/lpseq_replay_highs.cpp)
    target_link_libraries(mipx-lpseq-replay-highs PRIVATE highs::highs)
    # mipx's warnings and sanitizers on this file and the shared parser.
    mipx_set_compiler_flags(mipx-lpseq-replay-highs)

    # HiGHS's presolve alone, beside mipx-presolve-bench (#287). It reads the
    # model with mipx's reader, so it links mipx as well; HiGHS stays out of
    # the mipx library.
    add_executable(mipx-presolve-bench-highs tests/perf/presolve_bench_highs.cpp)
    target_link_libraries(mipx-presolve-bench-highs PRIVATE mipx highs::highs)
    mipx_set_compiler_flags(mipx-presolve-bench-highs)
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
