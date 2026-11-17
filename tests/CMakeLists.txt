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
    test_column_changes.cpp
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
    test_objective_propagator.cpp
    test_validity.cpp
    test_lp_validity.cpp
    test_cli.cpp
    test_paths.cpp
)

target_link_libraries(mipx-tests PRIVATE
    Catch2::Catch2WithMain
    mipx
)

mipx_set_compiler_flags(mipx-tests)

# Absolute paths go to test_paths.cpp alone. On the whole target they would put
# the checkout's path on every test file's command line, so each fresh
# worktree the pre-push gate builds in would miss the compiler cache for all of
# them (#217); here one small file misses instead.
set_property(SOURCE test_paths.cpp APPEND PROPERTY COMPILE_DEFINITIONS
    TEST_DATA_DIR="${PROJECT_SOURCE_DIR}/tests/data"
)

# The CLI tests in test_cli.cpp spawn the real binary; they compile to nothing
# when there is no binary to spawn (MIPX_BUILD_CLI=OFF).
if(TARGET mipx-solve)
    target_compile_definitions(mipx-tests PRIVATE MIPX_HAS_SOLVE_BINARY)
    set_property(SOURCE test_paths.cpp APPEND PROPERTY COMPILE_DEFINITIONS
        MIPX_SOLVE_BINARY="$<TARGET_FILE:mipx-solve>"
    )
    add_dependencies(mipx-tests mipx-solve)
endif()

# LP sequence replay (#267): column generation's masters on the public
# CommaLab grid1, as mcf's LP recorder wrote them, replayed on every LP solver
# warm and cold; each solve must reach the recorded objective. Seconds.
if(TARGET mipx-lpseq-replay)
    foreach(devex off on)
        set(_args "${PROJECT_SOURCE_DIR}/tests/data/lpseq/grid1"
            --arm dual-warm --arm dual-cold --arm barrier --arm pdlp)
        if(devex STREQUAL "on")
            list(APPEND _args --primal-devex)
        endif()
        add_test(NAME "LP sequence replay: grid1 masters, primal Devex ${devex}"
            COMMAND mipx-lpseq-replay ${_args} --csv "${CMAKE_CURRENT_BINARY_DIR}/lpseq_grid1_${devex}.csv")
    endforeach()
endif()
# The same sequence on HiGHS, in bench builds only (MIPX_BENCH_HIGHS).
if(TARGET mipx-lpseq-replay-highs)
    add_test(NAME "LP sequence replay (HiGHS): grid1 masters"
        COMMAND mipx-lpseq-replay-highs "${PROJECT_SOURCE_DIR}/tests/data/lpseq/grid1"
            --arm simplex-warm --arm primal-warm --arm simplex-cold --arm ipx-cold --arm hipo-cold
            --csv "${CMAKE_CURRENT_BINARY_DIR}/lpseq_grid1_highs.csv")
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

# Instance gate (#208): every listed Netlib LP (per backend) and curated MIPLIB
# MIP is classified against its .solu reference and held to the expected class
# checked in under tests/perf/instance_gate/. A wrong answer always fails.
#
#   ctest --test-dir build -L instance-gate --output-on-failure
#
# Also labelled solver-regression, so the commit and push gates (which run
# -LE solver-regression) leave it out. Without downloaded instances the script
# exits 77, which ctest reports as skipped.
find_package(Python3 COMPONENTS Interpreter QUIET)
if(TARGET mipx-solve AND Python3_Interpreter_FOUND)
    foreach(suite lp mip)
        add_test(NAME "instance-gate: ${suite}"
            COMMAND Python3::Interpreter
                "${PROJECT_SOURCE_DIR}/tests/perf/run_instance_gate.py" ${suite}
                --binary "$<TARGET_FILE:mipx-solve>"
        )
        set_tests_properties("instance-gate: ${suite}" PROPERTIES
            LABELS "solver-regression;instance-gate"
            SKIP_RETURN_CODE 77
        )
    endforeach()
endif()
