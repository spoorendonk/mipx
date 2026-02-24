add_executable(mipx-tests
    test_core.cpp
    test_sparse_matrix.cpp
    test_lp_problem.cpp
    test_benchmark.cpp
    test_lp_solver.cpp
    test_lu.cpp
    test_dual_simplex.cpp
)

target_link_libraries(mipx-tests PRIVATE
    Catch2::Catch2WithMain
    mipx
)

mipx_set_compiler_flags(mipx-tests)

target_compile_definitions(mipx-tests PRIVATE
    TEST_DATA_DIR="${PROJECT_SOURCE_DIR}/tests/data"
)

catch_discover_tests(mipx-tests)
