add_executable(mipx-tests
    test_core.cpp
    test_sparse_work_vector.cpp
    test_sparse_matrix.cpp
    test_lp_problem.cpp
    test_benchmark.cpp
    test_lp_solver.cpp
    test_lu.cpp
    test_dual_simplex.cpp
    test_incremental.cpp
    test_node_queue.cpp
    test_branching.cpp
    test_domain.cpp
    test_implication_graph.cpp
    test_mip_solver.cpp
    test_heuristics.cpp
    test_presolve.cpp
    test_cuts.cpp
    test_cut_manager.cpp
    test_parallel.cpp
    test_symmetry.cpp
    test_barrier.cpp
    test_pdlp.cpp
    test_exact_refinement.cpp
    test_conflict_graph.cpp
    test_clique_table.cpp
    test_reduced_cost_fixer.cpp
    test_cli.cpp
)

target_link_libraries(mipx-tests PRIVATE
    Catch2::Catch2WithMain
    mipx
)

mipx_set_compiler_flags(mipx-tests)

target_compile_definitions(mipx-tests PRIVATE
    TEST_DATA_DIR="${PROJECT_SOURCE_DIR}/tests/data"
)

# The CLI tests in test_cli.cpp spawn the real binary; they compile to nothing
# when there is no binary to spawn (MIPX_BUILD_CLI=OFF).
if(TARGET mipx-solve)
    target_compile_definitions(mipx-tests PRIVATE
        MIPX_SOLVE_BINARY="$<TARGET_FILE:mipx-solve>"
    )
    add_dependencies(mipx-tests mipx-solve)
endif()

# Fast correctness suite -- the default, and what the git hooks run.
catch_discover_tests(mipx-tests
    TEST_SPEC "~[benchmark]~[miplib]~[netlib]"
)

# Instance-driven solver tests, labelled so the hooks can skip them.
#
# These load real Netlib/MIPLIB instances and assert objectives under
# wall-clock limits, so they measure the machine and the build as much as the
# code: an unoptimized build alone is enough to fail them on timing. That makes
# them a poor per-commit gate. They still run on demand:
#
#   ctest --test-dir build -L solver-regression --output-on-failure
#
# They are a placeholder for a proper benchmark bed with recorded times; the
# intent is to fold them into the MIPLIB benchmark suite rather than keep them
# as pass/fail unit tests. See issue #213.
catch_discover_tests(mipx-tests
    TEST_SPEC "[benchmark],[miplib],[netlib]"
    TEST_PREFIX "solver-regression: "
    PROPERTIES LABELS "solver-regression"
)
