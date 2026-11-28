cmake_minimum_required(VERSION 3.18...3.31)
project(orthel LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

find_package(Threads REQUIRED)

nanobind_add_module(_r2r NB_STATIC cpp/module.cpp)
target_include_directories(_r2r PRIVATE cpp)
target_link_libraries(_r2r PRIVATE Threads::Threads)

install(TARGETS _r2r LIBRARY DESTINATION orthel/transforms)
