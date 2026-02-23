function(mipx_set_compiler_flags target)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target} PRIVATE
            -Wall -Wextra -Wpedantic -Werror
        )
        if(CMAKE_BUILD_TYPE STREQUAL "Debug")
            target_compile_options(${target} PRIVATE
                -fsanitize=address,undefined
                -fno-omit-frame-pointer
            )
            target_link_options(${target} PRIVATE
                -fsanitize=address,undefined
            )
        endif()
    elseif(MSVC)
        target_compile_options(${target} PRIVATE /W4 /WX /utf-8)
    endif()
endfunction()
