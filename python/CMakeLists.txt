nanobind_add_module(_core STABLE_ABI src/bindings.cpp)

target_link_libraries(_core PRIVATE mipx)
target_include_directories(_core PRIVATE ${PROJECT_SOURCE_DIR}/include)

install(TARGETS _core
    LIBRARY DESTINATION mipx
    RUNTIME DESTINATION mipx
)
