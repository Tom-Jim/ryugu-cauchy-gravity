cmake_minimum_required(VERSION 3.24)
project(ryugu_esa_bridge LANGUAGES CXX)

include(FetchContent)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(POLYHEDRAL_GRAVITY_PARALLELIZATION CPP CACHE STRING "" FORCE)
set(BUILD_POLYHEDRAL_GRAVITY_PYTHON_INTERFACE OFF CACHE BOOL "" FORCE)
set(BUILD_POLYHEDRAL_GRAVITY_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_POLYHEDRAL_GRAVITY_EXECUTABLE OFF CACHE BOOL "" FORCE)
set(POLYHEDRAL_GRAVITY_LIBRARY ON CACHE BOOL "" FORCE)
set(POLYHEDRAL_GRAVITY_LOGGING_LEVEL WARN CACHE STRING "" FORCE)

FetchContent_Declare(
  polyhedral_gravity
  GIT_REPOSITORY https://github.com/esa/polyhedral-gravity-model.git
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(polyhedral_gravity)

add_library(esa_pg_bridge STATIC common/esa_bridge.cpp)
target_include_directories(esa_pg_bridge PUBLIC common)
target_link_libraries(esa_pg_bridge PUBLIC polyhedralGravity_lib)
