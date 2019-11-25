cmake_minimum_required(VERSION 3.0.0)
project(tsne VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 11)

# SET(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" "${CMAKE_MODULE_PATH}")

# SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
# SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lm -lc++ -lstdc++")

# IF(MSVC)
#     SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /fp:fast /O2")
# ELSE()
#     SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -fPIC -ffast-math -funroll-loops")
# ENDIF()

# FIND_PACKAGE(OpenMP 2)

# IF(OPENMP_FOUND)
#     SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#     SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
# ELSE()
#     MESSAGE(WARNING "Not using OpenMP. Performance will suffer.")
# ENDIF()


file(GLOB_RECURSE SOURCES "src/*.cpp")
set(PROJECT_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)

include(CTest)
enable_testing()

include_directories(${PROJECT_INCLUDE_DIR})

find_package(OpenMP 2 REQUIRED)

add_library(tsne ${SOURCES})

add_subdirectory(test)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
