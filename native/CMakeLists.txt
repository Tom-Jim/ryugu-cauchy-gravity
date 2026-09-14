cmake_minimum_required(VERSION 3.16)
project(ryugu_esa_bridge CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(ESA_BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/esa-build")
set(ESA_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../vendor/polyhedral-gravity-model")

file(GLOB_RECURSE ESA_OBJS
  "${ESA_BUILD_DIR}/src/polyhedralGravity/CMakeFiles/polyhedralGravity_lib.dir/*.o")
if (ESA_OBJS STREQUAL "")
  message(FATAL_ERROR "ESA object files missing; build native/esa-build first")
endif()

add_library(esa_pg_bridge STATIC esa_bridge.cpp ${ESA_OBJS})
set_source_files_properties(${ESA_OBJS} PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)

target_include_directories(esa_pg_bridge PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${ESA_SRC_DIR}/src"
)
target_include_directories(esa_pg_bridge SYSTEM PUBLIC
  "${ESA_BUILD_DIR}/_deps/spdlog-src/include"
  "${ESA_BUILD_DIR}/_deps/thrust-src"
  "${ESA_BUILD_DIR}/_deps/thrust-src/dependencies/cub"
  "${ESA_BUILD_DIR}/_deps/xsimd-src/include"
  "${ESA_BUILD_DIR}/_deps/tetgen-src"
  "${ESA_BUILD_DIR}/_deps/yaml-cpp-src/include"
)
target_compile_definitions(esa_pg_bridge PRIVATE
  FMT_HEADER_ONLY
  SPDLOG_ACTIVE_LEVEL=4
  _VSTD=std
  THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP
  THRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP
  YAML_CPP_STATIC_DEFINE
)

set(TETGEN_A "${ESA_BUILD_DIR}/libtetgen_lib.a")
set(YAML_A "${ESA_BUILD_DIR}/_deps/yaml-cpp-build/libyaml-cpp.a")
set(SPDLOG_A "${ESA_BUILD_DIR}/_deps/spdlog-build/libspdlog.a")

target_link_libraries(esa_pg_bridge PUBLIC "${TETGEN_A}" "${YAML_A}" "${SPDLOG_A}")

add_executable(cube_smoke cube_smoke.cpp)
target_link_libraries(cube_smoke PRIVATE esa_pg_bridge)

add_executable(ryugu_gradient_bake_cpp bake_main.cpp)
target_link_libraries(ryugu_gradient_bake_cpp PRIVATE esa_pg_bridge)
