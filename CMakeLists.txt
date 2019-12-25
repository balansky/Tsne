cmake_minimum_required(VERSION 3.0.0)

project(Tsne VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_LIST_DIR}/cmake )

include(UseCython)

find_package(Numpy)

file( COPY setup.py DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
      FILES_MATCHING PATTERN "*.py" )


include_directories(${FastTsne_SOURCE_DIR}/include)

add_subdirectory(FastTsne)
add_subdirectory(PyFastTsne)

add_dependencies(PyFastTsne FastTsne)


enable_testing()
find_file( NOSETESTS_EXECUTABLE nosetests )
add_test( nosetests "${NOSETESTS_EXECUTABLE}" -v --with-xunit )

install(CODE "execute_process(COMMAND ${PYTHON_EXECUTABLE} setup.py install WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})")

