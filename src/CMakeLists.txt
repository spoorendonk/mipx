add_library(mipx STATIC
    core.cpp
    lp/sparse_matrix.cpp
    lp/lp_problem.cpp
    lp/lp_solver.cpp
    lp/lu.cpp
    lp/dual_simplex.cpp
    lp/barrier.cpp
    lp/pdlp.cpp
    io/mps_reader.cpp
    io/mps_writer.cpp
    io/lp_reader.cpp
    io/solu_reader.cpp
    mip/domain.cpp
    mip/node_queue.cpp
    mip/branching.cpp
    mip/mip_solver.cpp
    heuristics/rounding.cpp
    heuristics/diving.cpp
    heuristics/rins.cpp
    heuristics/rens.cpp
    heuristics/feaspump.cpp
    heuristics/local_branching.cpp
    heuristics/aux_objective.cpp
    heuristics/zero_objective.cpp
    heuristics/budget.cpp
    heuristics/runtime.cpp
    presolve/presolve.cpp
    cuts/cut_pool.cpp
    cuts/gomory.cpp
    cuts/separators.cpp
)

target_include_directories(mipx PUBLIC ${PROJECT_SOURCE_DIR}/include)

target_link_libraries(mipx PUBLIC ZLIB::ZLIB)

if(MIPX_USE_TBB)
    target_link_libraries(mipx PUBLIC TBB::tbb)
    target_compile_definitions(mipx PUBLIC MIPX_HAS_TBB)
endif()

if(MIPX_USE_CUDA)
    target_link_libraries(mipx PUBLIC CUDA::cudart CUDA::cusparse)
    target_compile_definitions(mipx PUBLIC MIPX_HAS_CUDA)
endif()

mipx_set_compiler_flags(mipx)
