cmake_minimum_required(VERSION 3.0.0)

project(PyFastTsne VERSION 0.1.0)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_LIST_DIR}/cmake )

include(UseCython)

include_directories(${FastTsne_SOURCE_DIR}/include)

add_subdirectory(FastTsne)
add_subdirectory(PyFastTsne)

add_dependencies(PyFastTsne FastTsne)

