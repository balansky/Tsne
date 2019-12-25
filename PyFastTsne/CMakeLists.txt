
set_source_files_properties(
        PyFastTsne.pyx
        PROPERTIES CYTHON_IS_CXX TRUE)

execute_process(
        COMMAND "${PYTHON_EXECUTABLE}" -c
        "from __future__ import print_function\ntry: import cysignals; print(cysignals.__file__, end='')\nexcept:pass\n"
        OUTPUT_VARIABLE __cysignals_path)

get_filename_component(CYSIGNALS_INCLUDE_DIR ${__cysignals_path} DIRECTORY)

include_directories(${FastTsne_SOURCE_DIR}/include ${PYTHON_NUMPY_INCLUDE_DIR} ${CYSIGNALS_INCLUDE_DIR})

cython_add_module(PyFastTsne PyFastTsne.pyx)

target_link_libraries(PyFastTsne FastTsne)

file( COPY tests/test.py DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
        FILES_MATCHING PATTERN "*.py" )
