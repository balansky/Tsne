# distutils: language = c++
from libcpp cimport bool

import numpy as np
cimport numpy as np

from fast_tsne cimport TSNE

cdef class PyTsne:

    cdef TSNE[double] *c_tsne;

    def __cinit__(self, unsigned short x_dim, unsigned short y_dim, bool verbose):
        self.c_tsne = new TSNE[double](x_dim, y_dim, verbose)

    def run(self, double perplexity, double theta, int max_iter, int stop_lying_iter, int mom_switch_iter):
        cdef np.ndarray arr = np.zeros(self.c_tsne.total(), type=np.double)
        self.c_tsne.run(perplexity, theta, max_iter, stop_lying_iter, mom_switch_iter, <double* >arr.data)
        return arr

    def run(self, size_t n, np.ndarray[double, mode="c"] x, double perplexity, double theta, int max_iter, int stop_lying_iter, int mom_switch_iter):
        cdef np.ndarray arr = np.zeros(n, type=np.double)
        self.c_tsne.run(n, <double*>x.data, perplexity, theta, max_iter, stop_lying_iter, mom_switch_iter, <double*>arr.data)
        return arr

    def __dealloc__(self):
        del self.c_tsne
