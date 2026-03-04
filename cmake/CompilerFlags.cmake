include(CheckCXXCompilerFlag)

function(mipx_check_cxx_flag flag out_var)
    string(MAKE_C_IDENTIFIER "${flag}" flag_id)
    set(cache_var "MIPX_FLAG_SUPPORTED_${flag_id}")
    if(NOT DEFINED ${cache_var})
        check_cxx_compiler_flag("${flag}" ${cache_var})
    endif()
    set(${out_var} ${${cache_var}} PARENT_SCOPE)
endfunction()

function(mipx_set_simd_flags target)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        return()
    endif()

    string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" mipx_arch)
    if(NOT mipx_arch MATCHES "x86_64|amd64|i[3-6]86")
        return()
    endif()

    if(NOT DEFINED MIPX_SIMD_ISA)
        set(MIPX_SIMD_ISA "avx2")
    endif()
    if(NOT DEFINED MIPX_SIMD_REQUIRE_SUPPORTED)
        set(MIPX_SIMD_REQUIRE_SUPPORTED OFF)
    endif()
    string(TOLOWER "${MIPX_SIMD_ISA}" mipx_simd_isa)

    set(simd_flags "")
    set(status_suffix "")

    if(mipx_simd_isa STREQUAL "off")
        set(status_suffix "SIMD ISA: off")
    elseif(mipx_simd_isa STREQUAL "native")
        mipx_check_cxx_flag("-march=native" has_march_native)
        if(has_march_native)
            list(APPEND simd_flags "-march=native")
            mipx_check_cxx_flag("-mtune=native" has_mtune_native)
            if(has_mtune_native)
                list(APPEND simd_flags "-mtune=native")
            endif()
            set(status_suffix "SIMD ISA: native (${simd_flags})")
        elseif(MIPX_SIMD_REQUIRE_SUPPORTED)
            message(FATAL_ERROR "Requested MIPX_SIMD_ISA=native but compiler does not support -march=native")
        else()
            message(WARNING "Requested MIPX_SIMD_ISA=native but unsupported; falling back to scalar")
            set(status_suffix "SIMD ISA: native requested, scalar fallback")
        endif()
    elseif(mipx_simd_isa STREQUAL "avx2")
        mipx_check_cxx_flag("-mavx2" has_mavx2)
        if(has_mavx2)
            list(APPEND simd_flags "-mavx2")
            mipx_check_cxx_flag("-mfma" has_mfma)
            if(has_mfma)
                list(APPEND simd_flags "-mfma")
            endif()
            set(status_suffix "SIMD ISA: avx2 (${simd_flags})")
        elseif(MIPX_SIMD_REQUIRE_SUPPORTED)
            message(FATAL_ERROR "Requested MIPX_SIMD_ISA=avx2 but compiler does not support -mavx2")
        else()
            message(WARNING "Requested MIPX_SIMD_ISA=avx2 but unsupported; falling back to scalar")
            set(status_suffix "SIMD ISA: avx2 requested, scalar fallback")
        endif()
    else()
        message(FATAL_ERROR
            "Unknown MIPX_SIMD_ISA='${MIPX_SIMD_ISA}'. Supported values: off, avx2, native")
    endif()

    if(simd_flags)
        target_compile_options(${target} PRIVATE ${simd_flags})
    endif()

    if(NOT DEFINED MIPX_SIMD_STATUS_PRINTED)
        message(STATUS "mipx ${status_suffix}")
        set(MIPX_SIMD_STATUS_PRINTED TRUE CACHE INTERNAL "SIMD status printed" FORCE)
    endif()
endfunction()

function(mipx_set_compiler_flags target)
    # Use generator expressions so GCC/Clang flags only apply to CXX sources,
    # not CUDA sources compiled by nvcc.
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>>
    )
    if(MIPX_STRICT_WARNINGS)
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:GNU,Clang>:-Werror>>
        )
    endif()
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:GNU,Clang>:-fsanitize=address,undefined -fno-omit-frame-pointer>>
        )
        target_link_options(${target} PRIVATE
            $<$<CXX_COMPILER_ID:GNU,Clang>:-fsanitize=address,undefined>
        )
    endif()
    if(MSVC)
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/W4 /utf-8>
        )
        if(MIPX_STRICT_WARNINGS)
            target_compile_options(${target} PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:/WX>
            )
        endif()
    endif()

    mipx_set_simd_flags(${target})
endfunction()
