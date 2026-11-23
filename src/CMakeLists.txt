add_library(mipx STATIC
    core.cpp
    lp/sparse_matrix.cpp
    lp/lp_problem.cpp
    lp/lp_solver.cpp
    lp/lu.cpp
    lp/rank_revealing_lu.cpp
    lp/dual_simplex.cpp
    lp/exact_refinement.cpp
    lp/barrier.cpp
    lp/barrier_cpu.cpp
    lp/barrier_presolve.cpp
    lp/amd_ordering.cpp
    lp/sparse_cholesky.cpp
    lp/pdlp.cpp
    io/mps_reader.cpp
    io/mps_writer.cpp
    io/lp_reader.cpp
    io/model_reader.cpp
    io/solu_reader.cpp
    mip/domain.cpp
    mip/implication_graph.cpp
    mip/variable_bounds.cpp
    mip/probing.cpp
    mip/conflict_graph.cpp
    mip/clique_table.cpp
    mip/node_queue.cpp
    mip/automorphism.cpp
    mip/branching.cpp
    mip/reduced_cost_fixer.cpp
    mip/objective_propagator.cpp
    mip/mip_solver.cpp
    mip/orbital_fixing.cpp
    mip/schreier_sims.cpp
    mip/symbreak.cpp
    heuristics/rounding.cpp
    heuristics/diving.cpp
    heuristics/rins.cpp
    heuristics/rens.cpp
    heuristics/feaspump.cpp
    heuristics/local_branching.cpp
    heuristics/aux_objective.cpp
    heuristics/zero_objective.cpp
    heuristics/feasibility_jump.cpp
    heuristics/crossover.cpp
    heuristics/proximity_search.cpp
    heuristics/undercover.cpp
    heuristics/zi_rounding.cpp
    heuristics/clique_rounding.cpp
    heuristics/shifting.cpp
    heuristics/randomized_rounding.cpp
    heuristics/reduced_cost.cpp
    heuristics/propagation_completion.cpp
    heuristics/local_search.cpp
    heuristics/budget.cpp
    heuristics/runtime.cpp
    heuristics/symmetry.cpp
    presolve/dependent_equations.cpp
    presolve/dual_bounds.cpp
    presolve/presolve.cpp
    presolve/presolve_matrix.cpp
    cuts/cut_pool.cpp
    cuts/gomory.cpp
    cuts/tableau_cut.cpp
    cuts/cut_manager.cpp
    cuts/separators.cpp
    cuts/cmir.cpp
    cuts/cut_strengthening.cpp
    cuts/intersection_cut.cpp
    cuts/lifted_covers.cpp
    cuts/mod_k.cpp
)

set_target_properties(mipx PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(mipx PUBLIC ${PROJECT_SOURCE_DIR}/include)

target_link_libraries(mipx PUBLIC ZLIB::ZLIB)

if(MIPX_USE_BZIP2)
    target_link_libraries(mipx PUBLIC BZip2::BZip2)
    target_compile_definitions(mipx PUBLIC MIPX_HAS_BZIP2)
endif()

if(MIPX_USE_TBB)
    target_link_libraries(mipx PUBLIC TBB::tbb)
    target_compile_definitions(mipx PUBLIC MIPX_HAS_TBB)
endif()

# BLAS is not linked: sparse_cholesky.cpp dlopens the shared library found
# here on first use (OpenBLAS reserves its thread pool at load time, which
# would otherwise fail every mipx binary under an address-space limit). The
# path is a system library's, the same in every checkout, so it does not
# defeat ccache across worktrees.
if(MIPX_BLAS_SHARED_LIBRARY)
    set_property(SOURCE lp/sparse_cholesky.cpp APPEND PROPERTY COMPILE_DEFINITIONS
        "MIPX_BLAS_LIBRARY=\"${MIPX_BLAS_SHARED_LIBRARY}\""
        "MIPX_BLAS_LIBRARY_REAL=\"${MIPX_BLAS_SHARED_LIBRARY_REAL}\"")
    target_link_libraries(mipx PRIVATE ${CMAKE_DL_LIBS})
endif()

if(MIPX_CUDA_BACKEND_ENABLED)
    target_sources(mipx PRIVATE
        lp/pdlp_kernels.cu
    )
    target_link_libraries(mipx PUBLIC CUDA::cudart CUDA::cusparse)
    target_compile_definitions(mipx PUBLIC MIPX_HAS_CUDA)

    if(MIPX_HAS_CUDSS)
        target_sources(mipx PRIVATE lp/barrier_gpu.cu lp/barrier_gpu_wrap.cpp)
        target_include_directories(mipx PRIVATE ${CUDSS_INCLUDE_DIR})
        target_link_libraries(mipx PUBLIC ${CUDSS_LIBRARY})
        target_compile_definitions(mipx PUBLIC MIPX_HAS_CUDSS)
        set_source_files_properties(lp/barrier_gpu.cu PROPERTIES
            LANGUAGE CUDA
            COMPILE_FLAGS "--expt-relaxed-constexpr")
    endif()
endif()

if(MIPX_ENABLE_LP_LIGHT_HEURISTICS)
    target_compile_definitions(mipx PUBLIC MIPX_HAS_LP_LIGHT)
endif()

mipx_set_compiler_flags(mipx)
