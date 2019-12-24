
set_source_files_properties(
        fast_tsne.pyx
        PROPERTIES CYTHON_IS_CXX TRUE)


cython_add_module(PyFastTsne fast_tsne.pyx)

target_link_libraries(PyFastTsne FastTsne)
