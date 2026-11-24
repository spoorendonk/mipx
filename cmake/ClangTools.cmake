# The clang tools version check (#262).
#
# clang-format and clang-tidy are 22.1.8 on every machine and in every repo of
# the group: clang-format's output, and clang-tidy's checks, change between
# major versions, so a tool of another version reports violations in files
# nobody touched, or misses the ones it should find. The pin lives in
# `.clang-format-version`, which the git hooks read and hold both tools to;
# the PyPI wheels in pyproject.toml's `dev` group install exactly that version
# into `.venv`:
#
#     uv sync    # clang-format==22.1.8 and clang-tidy==22.1.8 into .venv/bin
#
# This checkout's `.venv/bin` is searched first, then PATH (the hooks also try
# the main checkout's from a linked worktree; CMake does not). At configure, a
# missing tool, or one of another major version than the pin's, is a warning,
# or an error with MIPX_REQUIRE_CLANG_TOOLS=ON, its default when the CI
# environment variable is set (and not `false` or `0`). The wheel build
# (scikit-build) lints nothing and skips the check.

file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/.clang-format-version" _mipx_clang_pin LIMIT_COUNT 1)
string(REGEX MATCH "^[ \t]*([0-9]+)" _mipx_clang_major "${_mipx_clang_pin}")
set(_mipx_clang_major "${CMAKE_MATCH_1}")

set(_mipx_require_default OFF)
if(DEFINED ENV{CI} AND NOT "$ENV{CI}" MATCHES "^(|0|false|FALSE|False)$")
    set(_mipx_require_default ON)
endif()
option(MIPX_REQUIRE_CLANG_TOOLS
       "Fail configuration when clang-format or clang-tidy is missing or of another major version"
       ${_mipx_require_default})

function(_mipx_clang_tools_complain msg)
    if(MIPX_REQUIRE_CLANG_TOOLS)
        message(FATAL_ERROR "clang tools: ${msg} (MIPX_REQUIRE_CLANG_TOOLS is ON)")
    else()
        message(WARNING "clang tools: ${msg}")
    endif()
endfunction()

if(_mipx_clang_major STREQUAL "")
    _mipx_clang_tools_complain(
        ".clang-format-version does not begin with a version (\"${_mipx_clang_pin}\")")
else()
    foreach(_mipx_tool IN ITEMS clang-format clang-tidy)
        string(MAKE_C_IDENTIFIER "MIPX_${_mipx_tool}_EXE" _mipx_var)
        string(TOUPPER "${_mipx_var}" _mipx_var)
        # Not cached, so a `uv sync` after the first configure is found by the
        # next one; `.venv/bin` alone first, so no CMAKE_PREFIX_PATH outranks
        # it. A -D of the variable skips both searches.
        find_program(${_mipx_var} NAMES ${_mipx_tool}
                     PATHS "${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin" NO_DEFAULT_PATH NO_CACHE)
        find_program(${_mipx_var} NAMES ${_mipx_tool} NO_CACHE)
        if(NOT ${_mipx_var})
            _mipx_clang_tools_complain(
                "${_mipx_tool} not found; `uv sync` installs ${_mipx_clang_pin} into .venv")
            continue()
        endif()
        execute_process(COMMAND "${${_mipx_var}}" --version
                        OUTPUT_VARIABLE _mipx_version_out ERROR_QUIET)
        string(REGEX MATCH "version ([0-9]+)\\.[0-9.]*" _mipx_version "${_mipx_version_out}")
        if(_mipx_version STREQUAL "")
            set(_mipx_version "of an unknown version")
        endif()
        if(NOT CMAKE_MATCH_1 STREQUAL _mipx_clang_major)
            _mipx_clang_tools_complain("${${_mipx_var}} is ${_mipx_tool} ${_mipx_version}, not \
${_mipx_clang_pin} as .clang-format-version pins; `uv sync` installs it into .venv")
        else()
            message(STATUS "clang tools: ${_mipx_tool} ${_mipx_version} (${${_mipx_var}})")
        endif()
    endforeach()
endif()
