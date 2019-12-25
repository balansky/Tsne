# distutils: language = c++

from libcpp cimport bool
from cysignals.signals cimport sig_on, sig_off

import numpy as np
cimport numpy as np

from PyFastTsne cimport TSNE


cdef class PyTsne:

    cdef TSNE[double] *c_tsne;

    cdef unsigned short x_dim;
    cdef unsigned short y_dim;
    cdef double learning_rate;
    cdef double perplexity;
    cdef double theta;
    cdef int n_iter;
    cdef int stop_lying_iter;
    cdef int mom_switch_iter;
    cdef bool verbose;

    def __cinit__(self, unsigned short x_dim, unsigned short y_dim = 2,
                  double learning_rate=200, double perplexity=30, double theta=0.5,
                  int n_iter=1000, int stop_lying_iter=250, int mom_switch_iter=250, bool verbose=False):

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.learning_rate = learning_rate
        self.perplexity = perplexity
        self.theta = theta
        self.n_iter = n_iter
        self.stop_lying_iter = stop_lying_iter
        self.mom_switch_iter = mom_switch_iter
        self.verbose = verbose
        self.c_tsne = new TSNE[double](self.x_dim, self.y_dim, self.verbose)


    def fit_transform(self, np.ndarray[double, ndim=2] x not None, np.ndarray[double, ndim=2] y = None):
        cdef size_t n = x.shape[0]
        cdef np.ndarray arr = np.zeros((n, self.y_dim), dtype=np.float64, order='C')

        del self.c_tsne
        self.c_tsne = new TSNE[double](self.x_dim, self.y_dim, self.verbose)

        if(not x.flags["C_CONTIGUOUS"]):
            x = np.ascontiguousarray(x, dtype=np.float64)

        sig_on()
        if(y is None):
            self.c_tsne.run(n, <double*>x.data, self.learning_rate, self.perplexity, self.theta, self.n_iter, self.stop_lying_iter, self.mom_switch_iter, <double*>arr.data)

        else:
            assert x.shape[0] == y.shape[0], "Number of X must equal To Y"
            if(not y.flags["C_CONTIGUOUS"]):
                y = np.ascontiguousarray(y, dtype=np.float64)
            self.c_tsne.insertItems(n, <double*>x.data, <double*>y.data)
            self.c_tsne.run(self.learning_rate, self.perplexity, self.theta, self.n_iter, self.stop_lying_iter, self.mom_switch_iter, False, <double*>arr.data)

        sig_off()
        return arr


    def fit(self, np.ndarray[double, ndim=2] x not None, np.ndarray[double, ndim=2] y = None):
        self.fit_transform(x, y)
        return self


    def partial_fit(self, np.ndarray[double, ndim=2] x not None, np.ndarray[double, ndim=2] y not None, int n_iter=1000, int stop_lying_iter=250, int mom_switch_iter=250):

        assert x.shape[0] == y.shape[0], "Number of X must equal To Y"

        cdef size_t n = x.shape[0]
        if(not x.flags["C_CONTIGUOUS"]):
            x = np.ascontiguousarray(x, dtype=np.float64)
        if(not y.flags["C_CONTIGUOUS"]):
            y = np.ascontiguousarray(y, dtype=np.float64)
        self.c_tsne.run(n, <double*>x.data, self.learning_rate, self.perplexity, self.theta, n_iter, stop_lying_iter, mom_switch_iter, <double*>y.data)
        return self


    def __dealloc__(self):
        del self.c_tsne
