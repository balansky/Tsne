
set_source_files_properties(
        PyFastTsne.pyx
        PROPERTIES CYTHON_IS_CXX TRUE)


include_directories(${FastTsne_SOURCE_DIR}/include ${PYTHON_NUMPY_INCLUDE_DIR})

cython_add_module(PyFastTsne PyFastTsne.pyx)

target_link_libraries(PyFastTsne FastTsne)

file( COPY tests/test.py DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
        FILES_MATCHING PATTERN "*.py" )
