##===- CMakeLists.txt - graph-mlir cmake root -----------------*- cmake -*-===//
##
## Configure the graph-mlir build.
##
##===----------------------------------------------------------------------===//

cmake_minimum_required(VERSION 3.10)

if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

#-------------------------------------------------------------------------------
# Project setup and globals
#-------------------------------------------------------------------------------

project(graph-mlir LANGUAGES CXX C)
  
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

#-------------------------------------------------------------------------------
# Options and settings
#-------------------------------------------------------------------------------

option(LLVM_INCLUDE_TOOLS "Generate build targets for the LLVM tools." ON)
option(LLVM_BUILD_TOOLS "Build the LLVM tools. If OFF, just generate build targets." ON)

#-------------------------------------------------------------------------------
# MLIR/LLVM Configuration
#-------------------------------------------------------------------------------
find_package(MLIR REQUIRED CONFIG)

set(LLVM_MLIR_BINARY_DIR ${MLIR_DIR}/../../../bin)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

#-------------------------------------------------------------------------------
# GraphMLIR configuration
#-------------------------------------------------------------------------------

# GraphMLIR project.
set(GraphMLIR_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(GraphMLIR_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/bin)
set(GraphMLIR_EXAMPLES_DIR ${GraphMLIR_SOURCE_DIR}/examples)
set(GraphMLIR_INCLUDE_DIR ${GraphMLIR_SOURCE_DIR}/include/)

set(GraphMLIR_OPT_ATTR avx2 CACHE STRING "Target Architecture.")
set(GraphMLIR_OPT_TRIPLE x86_64-unknown-linux-gnu CACHE STRING "Target Triple.")

message(STATUS "Configuring Target Architecture: ${GraphMLIR_OPT_ATTR}")
message(STATUS "Configuring Target Triple: ${GraphMLIR_OPT_TRIPLE}")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${GraphMLIR_BINARY_DIR})

set(GraphMLIR_EXAMPLES OFF CACHE BOOL "Build examples")

# Add MLIR and LLVM headers to the include path
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

# Add GraphMLIR files to the include path
include_directories(${GraphMLIR_MAIN_INCLUDE_DIR})
include_directories(${GraphMLIR_INCLUDE_DIR})
include_directories(${GraphMLIR_INCLUDE_DIR}/Dialect)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/Dialect)
include_directories(${GraphMLIR_SOURCE_DIR}/lib)

#-------------------------------------------------------------------------------
# Directory setup
#-------------------------------------------------------------------------------

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)

if(GraphMLIR_EXAMPLES)
  add_subdirectory(examples)
endif()
