add_library(mipx STATIC
    core.cpp
    lp/sparse_matrix.cpp
    lp/lp_problem.cpp
    lp/lp_solver.cpp
    lp/lu.cpp
    io/mps_reader.cpp
    io/mps_writer.cpp
    io/lp_reader.cpp
    io/solu_reader.cpp
)

target_include_directories(mipx PUBLIC ${PROJECT_SOURCE_DIR}/include)

target_link_libraries(mipx PUBLIC ZLIB::ZLIB)

if(MIPX_USE_TBB)
    target_link_libraries(mipx PUBLIC TBB::tbb)
    target_compile_definitions(mipx PUBLIC MIPX_HAS_TBB)
endif()

mipx_set_compiler_flags(mipx)
