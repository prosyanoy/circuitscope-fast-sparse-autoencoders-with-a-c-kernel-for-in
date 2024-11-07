cmake_minimum_required(VERSION 3.18)
project(circuitscope_kernels LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
option(CS_NATIVE "Compile with -march=native (disable for portable binaries)" ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
  OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenMP REQUIRED)

pybind11_add_module(_kernels csrc/kernels.cpp)
target_link_libraries(_kernels PRIVATE OpenMP::OpenMP_CXX)
target_compile_options(_kernels PRIVATE -O3 -ffast-math -fno-finite-math-only)
if(CS_NATIVE)
  target_compile_options(_kernels PRIVATE -march=native)
endif()
# Drop the built module next to the Python package so `import circuitscope` finds it.
set_target_properties(_kernels PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/circuitscope")
