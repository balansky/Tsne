import numpy
from distutils.core import setup

from Cython.Build import cythonize
from distutils.extension import Extension

ext = Extension("PyFastTsne", ["fast_tsne.pyx", "FastTsne/src/tsne.cpp"],
                include_dirs=["FastTsne/include"],
                language="c++",
                compiler_directives={'language_level': 3}
)

setup(ext_modules=cythonize(ext))
