cmake_minimum_required(VERSION 3.25)
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
option(MIPX_BUILD_PYTHON "Build Python nanobind extension module" OFF)

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
