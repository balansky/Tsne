from libcpp cimport bool

cdef extern from "tsne.hpp" namespace "tsne":
    cdef cppclass TSNE[T]:
        TSNE() except +
        TSNE(unsigned short x_dim, unsigned short y_dim, bool verbose) except +
        TSNE(size_t n, unsigned short x_dim, unsigned short y_dim, T *x, T *y, bool verbose);

        unsigned int total();

        void run(T perplexity, T theta, int max_iter, int stop_lying_iter, int mom_switch_iter, T *ret);

        void run(size_t n, T *x, T perplexity, T theta, int max_iter, int stop_lying_iter, int mom_switch_iter, T *ret);
